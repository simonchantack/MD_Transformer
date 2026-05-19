"""
=============================================================================
  ENCODER FUSION EXPERIMENT — PatchTST Dual-Dim RUL Model
=============================================================================
  PURPOSE
  -------
  Proves that fusing the Temporal Encoder and Sensor-Channel Encoder is
  beneficial by demonstrating that each branch captures *distinct, orthogonal*
  predictive structure.

  EXPERIMENTS RUN
  ---------------
  1. CKA (Centered Kernel Alignment) — measures representational similarity
     between temporal and channel embeddings.  Low CKA → orthogonal encoders.

  2. Cosine Similarity Distribution — token-level angular similarity.

  3. PCA Explained Variance — how much unique variance each branch holds.

  4. Ablation Study — compare test RMSE/MAE for:
        (a) Full model  (Temporal + Channel)
        (b) Temporal-only  (channel branch zeroed)
        (c) Channel-only   (temporal branch zeroed)

  5. Mutual Information Proxy — k-NN MI estimate between pooled embeddings.

  6. Grad-Norm Attribution — gradient magnitude attributed to each branch.

  HOW TO USE
  ----------
  Drop this file alongside the training notebook / script.
  All you need to pass in is:
      - model            : trained PatchTST_RUL_Model (already on device)
      - test_dataloader  : DataLoader for the test set
      - device           : torch.device

  Example (append to the notebook after training):

      from encoder_fusion_experiment import run_fusion_experiment
      run_fusion_experiment(best_model, test_dataloader, device)

=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import math, time
from typing import Optional, Dict, Tuple, List

# ── Third-party ───────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Optional but helpful ──────────────────────────────────────────────────
try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_MI = True
except ImportError:
    HAS_MI = False

# =============================================================================
#  SECTION 1 – EMBEDDING EXTRACTION
# =============================================================================

class _EmbeddingHook:
    """Context-manager that captures the OUTPUT of a nn.Module via a forward hook."""
    def __init__(self, module: nn.Module):
        self.output = None
        self._hook  = module.register_forward_hook(self._fn)

    def _fn(self, module, inp, out):
        self.output = out.detach().cpu()

    def remove(self):
        self._hook.remove()


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    max_batches: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Run the model on `loader` and collect:
      - temporal_emb : (N, N_patch, d_model_t)   raw output of temporal_encoder
      - channel_emb  : (N, L*N_patch, d_model_c)  raw output of sensor_encoder
      - temporal_pool: (N, d_model_t)  mean-pooled temporal
      - channel_pool : (N, d_model_c)  mean-pooled channel
      - y_true       : (N,)
      - y_pred       : (N,)
    """
    model.eval()

    # Install hooks
    t_hook = _EmbeddingHook(model.temporal_encoder)
    c_hook = _EmbeddingHook(model.sensor_encoder)

    t_embs, c_embs, ys_true, ys_pred = [], [], [], []

    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        xb = xb.to(device).float()
        preds = model(xb)          # triggers hooks as side-effect
        t_embs.append(t_hook.output.numpy())   # (B, N, d_t)
        c_embs.append(c_hook.output.numpy())   # (B, seq, d_c)
        ys_true.append(yb.numpy())
        ys_pred.append(preds.detach().cpu().numpy())

    t_hook.remove()
    c_hook.remove()

    temporal_emb  = np.concatenate(t_embs,    axis=0)
    channel_emb   = np.concatenate(c_embs,    axis=0)
    y_true        = np.concatenate(ys_true,   axis=0)
    y_pred        = np.concatenate(ys_pred,   axis=0)

    temporal_pool = temporal_emb.mean(axis=1)   # (N, d_t)
    channel_pool  = channel_emb.mean(axis=1)    # (N, d_c)

    return dict(
        temporal_emb  = temporal_emb,
        channel_emb   = channel_emb,
        temporal_pool = temporal_pool,
        channel_pool  = channel_pool,
        y_true        = y_true,
        y_pred        = y_pred,
    )


# =============================================================================
#  SECTION 2 – SIMILARITY METRICS
# =============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X (N×p) and Y (N×q).
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    Range [0, 1].  0 = completely different,  1 = identical (up to linear transform).
    """
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kx = X @ X.T
    Ky = Y @ Y.T
    Kx_c = center(Kx)
    Ky_c = center(Ky)

    hsic_xy  = np.sum(Kx_c * Ky_c)
    hsic_xx  = np.sqrt(np.sum(Kx_c * Kx_c))
    hsic_yy  = np.sqrt(np.sum(Ky_c * Ky_c))
    denom    = hsic_xx * hsic_yy

    return float(hsic_xy / denom) if denom > 0 else float("nan")


def cosine_similarity_distribution(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Per-sample cosine similarity between paired rows of X and Y.
    X, Y: (N, d)
    Returns array of length N.
    """
    nX = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    nY = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return np.sum(nX * nY, axis=1)


# =============================================================================
#  SECTION 3 – ABLATION STUDY (zero-out one branch at a time)
# =============================================================================

@torch.no_grad()
def ablation_predict(model, loader, device, mode: str = "full") -> Tuple[np.ndarray, np.ndarray]:
    """
    mode: 'full'         – normal forward pass
          'temporal_only' – zero channel encoder output before fusion
          'channel_only'  – zero temporal encoder output before fusion
    """
    model.eval()

    preds_all, true_all = [], []

    # We monkey-patch the fusion_head.forward just for this call
    orig_fuse = model.fusion_head.forward

    if mode == "temporal_only":
        def patched_fuse(t, c):
            # Zero out the channel representation
            c_zero = torch.zeros_like(c)
            return orig_fuse(t, c_zero)
        model.fusion_head.forward = patched_fuse

    elif mode == "channel_only":
        def patched_fuse(t, c):
            t_zero = torch.zeros_like(t)
            return orig_fuse(t_zero, c)
        model.fusion_head.forward = patched_fuse

    for xb, yb in loader:
        xb = xb.to(device).float()
        preds = model(xb)
        preds_all.append(preds.cpu().numpy())
        true_all.append(yb.numpy())

    # Restore original
    model.fusion_head.forward = orig_fuse

    return np.concatenate(true_all), np.concatenate(preds_all)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# =============================================================================
#  SECTION 4 – GRADIENT ATTRIBUTION
# =============================================================================

def gradient_attribution(model, loader, device, n_batches: int = 10) -> Dict[str, float]:
    """
    Compute the mean L2-norm of gradients flowing through each encoder branch.
    Larger gradient norm → the branch contributes more to the loss signal.
    """
    model.train()   # need grad
    criterion = nn.MSELoss()

    t_norms, c_norms = [], []

    t_params = list(model.temporal_encoder.parameters())
    c_params = list(model.sensor_encoder.parameters())

    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        model.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()

        t_norm = float(np.mean([p.grad.norm().item() for p in t_params if p.grad is not None]))
        c_norm = float(np.mean([p.grad.norm().item() for p in c_params if p.grad is not None]))
        t_norms.append(t_norm)
        c_norms.append(c_norm)

    model.eval()
    return {"temporal_grad_norm": float(np.mean(t_norms)),
            "channel_grad_norm":  float(np.mean(c_norms))}


# =============================================================================
#  SECTION 5 – MUTUAL INFORMATION (proxy via sklearn)
# =============================================================================

def mi_with_target(pool_emb: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Average mutual information between each dimension of `pool_emb` and `y`.
    Uses sklearn's k-NN MI estimator.
    """
    if not HAS_MI:
        return float("nan")
    mi_vals = mutual_info_regression(pool_emb, y, n_neighbors=n_neighbors)
    return float(mi_vals.mean())


# =============================================================================
#  SECTION 6 – VISUALIZATION
# =============================================================================

# ── Custom dark colour palette ────────────────────────────────────────────
TEMPORAL_COLOR = "#00C2A8"    # teal
CHANNEL_COLOR  = "#FF6B6B"    # coral
FULL_COLOR     = "#F5C518"    # gold
BG_COLOR       = "#0E1117"
PANEL_COLOR    = "#1A1D27"
TEXT_COLOR     = "#E8EAF0"
GRID_COLOR     = "#2A2D3A"

_pal = [TEMPORAL_COLOR, CHANNEL_COLOR, FULL_COLOR, "#A78BFA", "#34D399"]


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)


def _gradient_fill(ax, x, y, color, alpha=0.18):
    """Soft shaded fill under a line."""
    ax.fill_between(x, y, alpha=alpha, color=color)


def make_experiment_figure(
    embs: Dict[str, np.ndarray],
    ablation_results: Dict[str, Dict[str, float]],
    cka_value: float,
    cos_sim: np.ndarray,
    pca_var_t: np.ndarray,
    pca_var_c: np.ndarray,
    mi_results: Dict[str, float],
    grad_attribution: Dict[str, float],
    save_path: str = "fusion_experiment_results.png",
):
    fig = plt.figure(figsize=(20, 22), facecolor=BG_COLOR)
    fig.suptitle(
        "Encoder Fusion Experiment  ·  PatchTST Dual-Dim RUL",
        color=TEXT_COLOR, fontsize=17, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.94, bottom=0.04, left=0.06, right=0.97
    )

    # ── Panel 1: CKA + Cosine sim summary card ────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _style_ax(ax0, "Representational Similarity")

    metrics_labels = ["Linear CKA\n(0=orthogonal)", "Mean Cosine\nSimilarity"]
    metrics_vals   = [cka_value, float(cos_sim.mean())]
    bar_colors = [TEMPORAL_COLOR, CHANNEL_COLOR]

    bars = ax0.barh(metrics_labels, metrics_vals, color=bar_colors, height=0.45, alpha=0.85)
    ax0.set_xlim(0, 1)
    ax0.axvline(0.5, color=TEXT_COLOR, linestyle="--", linewidth=0.9, alpha=0.5,
                label="0.5 threshold")

    for bar, val in zip(bars, metrics_vals):
        ax0.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", color=TEXT_COLOR, va="center", fontsize=10, fontweight="bold")

    ax0.set_xlabel("Similarity Score", color=TEXT_COLOR)
    ax0.legend(fontsize=8, labelcolor=TEXT_COLOR, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    # Annotation
    interpretation = "Low similarity → encoders are complementary" if cka_value < 0.5 \
        else "High similarity → encoders may overlap"
    ax0.text(0.5, -0.18, interpretation, transform=ax0.transAxes,
             ha="center", color="#A0A8C0", fontsize=8, style="italic")

    # ── Panel 2: Cosine Similarity Distribution ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _style_ax(ax1, "Cosine Similarity Distribution\n(Temporal vs Channel Embeddings)")

    n_bins = 40
    counts, bins = np.histogram(cos_sim, bins=n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax1.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.85,
            color=TEMPORAL_COLOR, alpha=0.75, edgecolor=BG_COLOR, linewidth=0.4)
    _gradient_fill(ax1, bin_centers, counts, TEMPORAL_COLOR)

    ax1.axvline(cos_sim.mean(), color=FULL_COLOR, linewidth=1.8, linestyle="-",
                label=f"Mean = {cos_sim.mean():.3f}")
    ax1.axvline(0, color=TEXT_COLOR, linewidth=0.8, linestyle=":", alpha=0.6)

    ax1.set_xlabel("Cosine Similarity", color=TEXT_COLOR)
    ax1.set_ylabel("Count", color=TEXT_COLOR)
    ax1.legend(fontsize=8, labelcolor=TEXT_COLOR, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    # ── Panel 3: PCA Cumulative Explained Variance ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style_ax(ax2, "PCA Cumulative Explained Variance\n(Per Encoder)")

    cum_t = np.cumsum(pca_var_t)
    cum_c = np.cumsum(pca_var_c)
    k = min(len(cum_t), len(cum_c), 20)

    xs = np.arange(1, k + 1)
    ax2.plot(xs, cum_t[:k], color=TEMPORAL_COLOR, linewidth=2.2, marker="o", markersize=4,
             label="Temporal Encoder")
    ax2.plot(xs, cum_c[:k], color=CHANNEL_COLOR, linewidth=2.2, marker="s", markersize=4,
             label="Channel Encoder")

    _gradient_fill(ax2, xs, cum_t[:k], TEMPORAL_COLOR)
    _gradient_fill(ax2, xs, cum_c[:k], CHANNEL_COLOR)

    ax2.axhline(0.9, color=TEXT_COLOR, linewidth=0.8, linestyle="--", alpha=0.6, label="90 % threshold")
    ax2.set_xlabel("# Principal Components", color=TEXT_COLOR)
    ax2.set_ylabel("Cumulative Explained Variance", color=TEXT_COLOR)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, labelcolor=TEXT_COLOR, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    # ── Panel 4: Ablation — RMSE bar chart ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Ablation Study  —  RMSE")

    ab_labels = ["Full Model\n(T + C)", "Temporal\nOnly", "Channel\nOnly"]
    ab_rmse   = [ablation_results["full"]["rmse"],
                 ablation_results["temporal_only"]["rmse"],
                 ablation_results["channel_only"]["rmse"]]
    ab_colors = [FULL_COLOR, TEMPORAL_COLOR, CHANNEL_COLOR]

    bars3 = ax3.bar(ab_labels, ab_rmse, color=ab_colors, width=0.5, alpha=0.85,
                    edgecolor=BG_COLOR, linewidth=0.8)
    ax3.set_ylabel("RMSE (↓ better)", color=TEXT_COLOR)

    best_val = min(ab_rmse)
    for bar, val in zip(bars3, ab_rmse):
        clr = "#FFFFFF" if val == best_val else "#A0A8C0"
        weight = "bold" if val == best_val else "normal"
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.05 * max(ab_rmse),
                 f"{val:.3f}", ha="center", color=clr, fontsize=10, fontweight=weight)

    # Star on best bar
    best_idx = np.argmin(ab_rmse)
    ax3.text(best_idx, ab_rmse[best_idx] + 0.12 * max(ab_rmse),
             "★ BEST", ha="center", color=FULL_COLOR, fontsize=8, fontweight="bold")

    # ── Panel 5: Ablation — MAE bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Ablation Study  —  MAE")

    ab_mae = [ablation_results["full"]["mae"],
              ablation_results["temporal_only"]["mae"],
              ablation_results["channel_only"]["mae"]]

    bars4 = ax4.bar(ab_labels, ab_mae, color=ab_colors, width=0.5, alpha=0.85,
                    edgecolor=BG_COLOR, linewidth=0.8)
    ax4.set_ylabel("MAE (↓ better)", color=TEXT_COLOR)

    best_mae_val = min(ab_mae)
    for bar, val in zip(bars4, ab_mae):
        clr = "#FFFFFF" if val == best_mae_val else "#A0A8C0"
        weight = "bold" if val == best_mae_val else "normal"
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.05 * max(ab_mae),
                 f"{val:.3f}", ha="center", color=clr, fontsize=10, fontweight=weight)

    best_idx4 = np.argmin(ab_mae)
    ax4.text(best_idx4, ab_mae[best_idx4] + 0.12 * max(ab_mae),
             "★ BEST", ha="center", color=FULL_COLOR, fontsize=8, fontweight="bold")

    # ── Panel 6: Ablation — R² bar chart ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style_ax(ax5, "Ablation Study  —  R²")

    ab_r2 = [ablation_results["full"]["r2"],
              ablation_results["temporal_only"]["r2"],
              ablation_results["channel_only"]["r2"]]

    bars5 = ax5.bar(ab_labels, ab_r2, color=ab_colors, width=0.5, alpha=0.85,
                    edgecolor=BG_COLOR, linewidth=0.8)
    ax5.set_ylabel("R²  (↑ better)", color=TEXT_COLOR)

    best_r2_val = max(ab_r2)
    for bar, val in zip(bars5, ab_r2):
        clr = "#FFFFFF" if val == best_r2_val else "#A0A8C0"
        weight = "bold" if val == best_r2_val else "normal"
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.02 * (max(ab_r2) - min(ab_r2) + 1e-6),
                 f"{val:.3f}", ha="center", color=clr, fontsize=10, fontweight=weight)

    best_idx5 = np.argmax(ab_r2)
    ax5.text(best_idx5, ab_r2[best_idx5] + 0.06 * (max(ab_r2) - min(ab_r2) + 1e-6),
             "★ BEST", ha="center", color=FULL_COLOR, fontsize=8, fontweight="bold")

    # ── Panel 7: PCA Scatter — Temporal ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    _style_ax(ax6, "Temporal Encoder  —  PCA (PC1 vs PC2)\nColoured by RUL")

    y_true = embs["y_true"]
    t_pool = embs["temporal_pool"]
    c_pool = embs["channel_pool"]

    # subsample for clarity
    idx = np.random.choice(len(t_pool), min(1500, len(t_pool)), replace=False)

    pca_t = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(t_pool))
    sc6 = ax6.scatter(pca_t[idx, 0], pca_t[idx, 1],
                      c=y_true[idx], cmap="plasma", s=12, alpha=0.7, linewidths=0)
    plt.colorbar(sc6, ax=ax6).ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.colorbar(sc6, ax=ax6).ax.tick_params(labelcolor=TEXT_COLOR)
    ax6.set_xlabel("PC 1", color=TEXT_COLOR)
    ax6.set_ylabel("PC 2", color=TEXT_COLOR)

    # ── Panel 8: PCA Scatter — Channel ────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    _style_ax(ax7, "Channel Encoder  —  PCA (PC1 vs PC2)\nColoured by RUL")

    pca_c = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(c_pool))
    sc7 = ax7.scatter(pca_c[idx, 0], pca_c[idx, 1],
                      c=y_true[idx], cmap="plasma", s=12, alpha=0.7, linewidths=0)
    plt.colorbar(sc7, ax=ax7).ax.tick_params(labelcolor=TEXT_COLOR)
    ax7.set_xlabel("PC 1", color=TEXT_COLOR)
    ax7.set_ylabel("PC 2", color=TEXT_COLOR)

    # ── Panel 9: Mutual Information bar ───────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    _style_ax(ax8, "Avg. Mutual Information with RUL\n(k-NN estimator)")

    if not any(math.isnan(v) for v in mi_results.values()):
        mi_labels = ["Temporal", "Channel"]
        mi_vals   = [mi_results["temporal_mi"], mi_results["channel_mi"]]
        ax8.bar(mi_labels, mi_vals, color=[TEMPORAL_COLOR, CHANNEL_COLOR],
                width=0.4, alpha=0.85, edgecolor=BG_COLOR)
        ax8.set_ylabel("Mean MI (nats)", color=TEXT_COLOR)
        for x_pos, val in enumerate(mi_vals):
            ax8.text(x_pos, val + 0.005, f"{val:.4f}", ha="center",
                     color=TEXT_COLOR, fontsize=10, fontweight="bold")
    else:
        ax8.text(0.5, 0.5, "sklearn MI not available\n(pip install scikit-learn)",
                 ha="center", va="center", color=TEXT_COLOR, transform=ax8.transAxes)

    # ── Panel 10: Gradient Attribution ────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 0])
    _style_ax(ax9, "Gradient Attribution\n(Mean Gradient Norm per Branch)")

    grad_labels = ["Temporal\nEncoder", "Channel\nEncoder"]
    grad_vals   = [grad_attribution["temporal_grad_norm"],
                   grad_attribution["channel_grad_norm"]]

    bars9 = ax9.bar(grad_labels, grad_vals, color=[TEMPORAL_COLOR, CHANNEL_COLOR],
                    width=0.4, alpha=0.85, edgecolor=BG_COLOR)
    ax9.set_ylabel("Mean ||∇||₂", color=TEXT_COLOR)

    for bar, val in zip(bars9, grad_vals):
        ax9.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * max(grad_vals),
                 f"{val:.5f}", ha="center", color=TEXT_COLOR, fontsize=10, fontweight="bold")

    # ── Panel 11: Prediction Comparison scatter ────────────────────────────
    ax10 = fig.add_subplot(gs[3, 1])
    _style_ax(ax10, "Prediction vs True RUL\n(Full Model vs Ablated)")

    y_full    = ablation_results["full"]["y_pred"]
    y_t_only  = ablation_results["temporal_only"]["y_pred"]
    y_c_only  = ablation_results["channel_only"]["y_pred"]

    n_plot = min(200, len(y_true))
    idx2 = np.argsort(y_true)[:n_plot]

    xs = np.arange(n_plot)
    ax10.plot(xs, y_true[idx2],     color=TEXT_COLOR,      linewidth=1.5, label="True RUL", zorder=5)
    ax10.plot(xs, y_full[idx2],     color=FULL_COLOR,      linewidth=1.2, label="Full (T+C)", alpha=0.9)
    ax10.plot(xs, y_t_only[idx2],   color=TEMPORAL_COLOR,  linewidth=1.0, label="Temporal Only", alpha=0.7, linestyle="--")
    ax10.plot(xs, y_c_only[idx2],   color=CHANNEL_COLOR,   linewidth=1.0, label="Channel Only", alpha=0.7, linestyle=":")

    ax10.set_xlabel("Sample index (sorted by true RUL)", color=TEXT_COLOR)
    ax10.set_ylabel("RUL", color=TEXT_COLOR)
    ax10.legend(fontsize=7, labelcolor=TEXT_COLOR, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, ncol=2)

    # ── Panel 12: Summary text card ───────────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.set_facecolor(PANEL_COLOR)
    for spine in ax11.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax11.axis("off")

    # Build verdict
    fusion_wins_rmse = ablation_results["full"]["rmse"] <= min(
        ablation_results["temporal_only"]["rmse"],
        ablation_results["channel_only"]["rmse"]
    )
    cka_verdict = "LOW  ✔  complementary" if cka_value < 0.5 else "HIGH  ⚠  overlapping"
    cos_verdict = "LOW  ✔  diverse"        if cos_sim.mean() < 0.5 else "HIGH  ⚠  similar"

    # Tuples: (text, color, fontsize, fontweight, fontstyle)
    lines = [
        ("EXPERIMENT SUMMARY", TEXT_COLOR, 12, "bold",   "normal"),
        ("",                   TEXT_COLOR,  8, "normal", "normal"),
        (f"Linear CKA:        {cka_value:.4f}  ->  {cka_verdict}", TEXT_COLOR, 9, "normal", "normal"),
        (f"Mean Cosine Sim:   {cos_sim.mean():.4f}  ->  {cos_verdict}", TEXT_COLOR, 9, "normal", "normal"),
        ("",                   TEXT_COLOR,  8, "normal", "normal"),
        ("Ablation RMSE:",     TEXT_COLOR,  9, "bold",   "normal"),
        (f"  Full (T+C):      {ablation_results['full']['rmse']:.4f}",          FULL_COLOR,     9, "normal", "normal"),
        (f"  Temporal only:   {ablation_results['temporal_only']['rmse']:.4f}", TEMPORAL_COLOR, 9, "normal", "normal"),
        (f"  Channel only:    {ablation_results['channel_only']['rmse']:.4f}",  CHANNEL_COLOR,  9, "normal", "normal"),
        ("",                   TEXT_COLOR,  8, "normal", "normal"),
        ("Gradient Norms:",    TEXT_COLOR,  9, "bold",   "normal"),
        (f"  Temporal:  {grad_attribution['temporal_grad_norm']:.5f}", TEMPORAL_COLOR, 9, "normal", "normal"),
        (f"  Channel:   {grad_attribution['channel_grad_norm']:.5f}",  CHANNEL_COLOR,  9, "normal", "normal"),
        ("",                   TEXT_COLOR,  8, "normal", "normal"),
        ("VERDICT:",           TEXT_COLOR, 10, "bold",   "normal"),
        (("✔ Fusion is BENEFICIAL" if fusion_wins_rmse else "⚠ Fusion did not win ablation"),
         FULL_COLOR if fusion_wins_rmse else "#FF9800", 10, "bold", "normal"),
        (("Encoders capture distinct structure." if cka_value < 0.5 else "Encoders may be redundant."),
         TEXT_COLOR, 9, "normal", "italic"),   # <-- style="italic", weight="normal"
    ]

    y_cur = 0.97
    for text, color, size, weight, style in lines:
        ax11.text(0.05, y_cur, text, transform=ax11.transAxes,
                  color=color, fontsize=size, fontweight=weight, fontstyle=style,
                  va="top", fontfamily="monospace")
        y_cur -= size * 0.022 + 0.01

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"\n✔ Figure saved → {save_path}")
    return fig


# =============================================================================
#  SECTION 7 – MAIN ENTRY POINT
# =============================================================================

def run_fusion_experiment(
    model,
    test_dataloader,
    device,
    max_extraction_batches: int = 50,
    save_path: str = "fusion_experiment_results.png",
):
    """
    Full experiment pipeline.

    Parameters
    ----------
    model            : trained PatchTST_RUL_Model
    test_dataloader  : DataLoader yielding (X, y) where X is (B, C, L)
    device           : torch.device
    max_extraction_batches : int – how many test batches to use (keep < full set for speed)
    save_path        : str – output PNG path
    """
    print("=" * 65)
    print("  ENCODER FUSION EXPERIMENT")
    print("=" * 65)

    # ── 1. Extract embeddings ─────────────────────────────────────────────
    print("\n[1/6] Extracting embeddings …")
    embs = extract_embeddings(model, test_dataloader, device, max_batches=max_extraction_batches)
    print(f"      Temporal pool shape : {embs['temporal_pool'].shape}")
    print(f"      Channel  pool shape : {embs['channel_pool'].shape}")
    print(f"      Samples extracted   : {embs['y_true'].shape[0]}")

    # ── 2. Similarity metrics ─────────────────────────────────────────────
    print("\n[2/6] Computing similarity metrics …")

    # Subsample for CKA speed (O(N²) kernel)
    N = embs["temporal_pool"].shape[0]
    sub = min(N, 800)
    idx_sub = np.random.choice(N, sub, replace=False)

    t_sub = embs["temporal_pool"][idx_sub]
    c_sub = embs["channel_pool"][idx_sub]

    # Align dimensions if needed (project to same dim)
    d_t = t_sub.shape[1]
    d_c = c_sub.shape[1]
    if d_t != d_c:
        # PCA-reduce larger one
        if d_t > d_c:
            t_sub_aligned = PCA(n_components=d_c).fit_transform(t_sub)
            c_sub_aligned = c_sub
        else:
            t_sub_aligned = t_sub
            c_sub_aligned = PCA(n_components=d_t).fit_transform(c_sub)
    else:
        t_sub_aligned = t_sub
        c_sub_aligned = c_sub

    cka_value = linear_cka(t_sub_aligned, c_sub_aligned)
    cos_sim   = cosine_similarity_distribution(t_sub_aligned, c_sub_aligned)

    print(f"      Linear CKA         : {cka_value:.4f}  (0=orthogonal, 1=identical)")
    print(f"      Mean Cosine Sim    : {cos_sim.mean():.4f}")

    # ── 3. PCA explained variance ─────────────────────────────────────────
    print("\n[3/6] Running PCA …")
    n_comp = min(20, t_sub.shape[1], c_sub.shape[1], sub)
    pca_var_t = PCA(n_components=n_comp).fit(StandardScaler().fit_transform(t_sub)).explained_variance_ratio_
    pca_var_c = PCA(n_components=n_comp).fit(StandardScaler().fit_transform(c_sub)).explained_variance_ratio_

    t90 = int(np.searchsorted(np.cumsum(pca_var_t), 0.9)) + 1
    c90 = int(np.searchsorted(np.cumsum(pca_var_c), 0.9)) + 1
    print(f"      Temporal: {t90} PCs for 90 % variance")
    print(f"      Channel : {c90} PCs for 90 % variance")

    # ── 4. Ablation study ─────────────────────────────────────────────────
    print("\n[4/6] Running ablation study …")
    ablation_results = {}
    for mode in ["full", "temporal_only", "channel_only"]:
        y_true_ab, y_pred_ab = ablation_predict(model, test_dataloader, device, mode=mode)
        ablation_results[mode] = {
            "rmse"  : rmse(y_true_ab, y_pred_ab),
            "mae"   : mae(y_true_ab, y_pred_ab),
            "r2"    : r2(y_true_ab, y_pred_ab),
            "y_pred": y_pred_ab,
        }
        print(f"      [{mode:>14s}]  RMSE={ablation_results[mode]['rmse']:.4f}  "
              f"MAE={ablation_results[mode]['mae']:.4f}  "
              f"R²={ablation_results[mode]['r2']:.4f}")

    # ── 5. Mutual Information ─────────────────────────────────────────────
    print("\n[5/6] Estimating mutual information with RUL …")
    y_sub = embs["y_true"][idx_sub]
    mi_t  = mi_with_target(t_sub, y_sub)
    mi_c  = mi_with_target(c_sub, y_sub)
    mi_results = {"temporal_mi": mi_t, "channel_mi": mi_c}
    print(f"      Temporal MI : {mi_t:.4f}")
    print(f"      Channel  MI : {mi_c:.4f}")

    # ── 6. Gradient attribution ───────────────────────────────────────────
    print("\n[6/6] Computing gradient attribution …")
    grad_attr = gradient_attribution(model, test_dataloader, device, n_batches=8)
    print(f"      Temporal ∇-norm : {grad_attr['temporal_grad_norm']:.6f}")
    print(f"      Channel  ∇-norm : {grad_attr['channel_grad_norm']:.6f}")

    # ── 7. Plot ───────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    fig = make_experiment_figure(
        embs              = embs,
        ablation_results  = ablation_results,
        cka_value         = cka_value,
        cos_sim           = cos_sim,
        pca_var_t         = pca_var_t,
        pca_var_c         = pca_var_c,
        mi_results        = mi_results,
        grad_attribution  = grad_attr,
        save_path         = save_path,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Linear CKA             : {cka_value:.4f}")
    print(f"  Mean Cosine Similarity : {cos_sim.mean():.4f}")
    print(f"  Ablation RMSE  (full)  : {ablation_results['full']['rmse']:.4f}")
    print(f"  Ablation RMSE  (T only): {ablation_results['temporal_only']['rmse']:.4f}")
    print(f"  Ablation RMSE  (C only): {ablation_results['channel_only']['rmse']:.4f}")

    fusion_wins = ablation_results["full"]["rmse"] <= min(
        ablation_results["temporal_only"]["rmse"],
        ablation_results["channel_only"]["rmse"]
    )
    print()
    if fusion_wins and cka_value < 0.5:
        print("  ✔ CONCLUSION: Fusion is BENEFICIAL.")
        print("    Low CKA + lower ablated RMSE confirms that each encoder")
        print("    captures orthogonal predictive structure; removing either")
        print("    branch degrades performance.")
    elif fusion_wins:
        print("  ✔ CONCLUSION: Fusion improves RMSE but encoders show")
        print("    moderate similarity (CKA ≥ 0.5). Consider adding")
        print("    decorrelation regularisation.")
    else:
        print("  ⚠ CONCLUSION: Full model did NOT outperform ablated models")
        print("    in RMSE. Review fusion head capacity or training regime.")
    print("=" * 65)

    return fig, embs, ablation_results


# =============================================================================
#  HOW TO CALL FROM THE TRAINING NOTEBOOK
# =============================================================================
#
#   from encoder_fusion_experiment import run_fusion_experiment
#
#   fig, embs, ablation = run_fusion_experiment(
#       model            = best_model,
#       test_dataloader  = test_dataloader,
#       device           = device,
#       save_path        = "fusion_experiment_results.png",
#   )
#
# =============================================================================