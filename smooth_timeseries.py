"""
=============================================================================
  TIME-SERIES SMOOTHING UTILITY  —  PatchTST RUL Pre-processing
=============================================================================
  PURPOSE
  -------
  Reduce erratic / noisy behaviour in sensor readings *per engine* before
  the sliding-window functions consume the data.  The smoothed DataFrame is
  a drop-in replacement for X / X_test: same columns, same index, same row
  order, same shape.

  SMOOTHING METHODS AVAILABLE
  ---------------------------
  "ema"       — Exponential Moving Average (lightweight, causal, good default)
  "savgol"    — Savitzky-Golay filter (preserves peaks, scipy required)
  "gaussian"  — Gaussian-weighted rolling window (symmetric, scipy required)
  "median"    — Rolling median (robust to spike outliers)
  "rolling"   — Simple rolling mean (fast, slight lag)

  HOW TO USE
  ----------
  # After normalisation and BEFORE create_training_sequences_sw / create_testing_sequences_sw:

      from smooth_timeseries import smooth_engine_data, compare_smoothing_methods

      # Smooth training data
      X_smooth = smooth_engine_data(X, features, method="ema", span=5)

      # Smooth test data using SAME parameters
      X_test_smooth = smooth_engine_data(X_test, features, method="ema", span=5)

      # Then feed into existing sliding-window functions as usual:
      X_train_sw, y_train_sw = create_training_sequences_sw(X_smooth,      features, window_size)
      X_testf                = create_testing_sequences_sw (X_test_smooth,  features, window_size, num_of_batches)

  DIAGNOSTIC PLOT
  ---------------
      compare_smoothing_methods(X, features, engine_id=1,
                                 sensor="sm_2", save_path="smoothing_comparison.png")

  PARAMETER GUIDANCE
  ------------------
  - span (EMA)            : 3–10.  Larger = smoother but more lag.
  - window_size (rolling/
    median/gaussian)      : 3–15 (odd preferred for savgol/gaussian).
  - poly_order (savgol)   : 2 or 3 (must be < window_size).
  - sigma (gaussian)      : 1–3.  Larger = more smoothing.

  Start with method="ema", span=5 as a safe default.
  If peaks matter (e.g. fault precursor spikes), prefer method="savgol".
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Optional scipy imports — graceful fallback if not installed
try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn(
        "scipy not found.  'savgol' and 'gaussian' methods unavailable. "
        "Install with:  pip install scipy",
        ImportWarning
    )


# =============================================================================
#  CORE SMOOTHING FUNCTION
# =============================================================================

def smooth_engine_data(
    df: pd.DataFrame,
    features,
    method: str = "ema",
    # EMA
    span: int = 5,
    # Rolling mean / median
    window: int = 5,
    # Savitzky-Golay
    poly_order: int = 2,
    # Gaussian
    sigma: float = 2.0,
    # Shared
    engine_col: str = "engine",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Apply per-engine, per-feature smoothing to a normalised sensor DataFrame.

    Parameters
    ----------
    df          : pd.DataFrame
        The normalised X or X_test DataFrame that contains `engine_col`,
        the sensor `features`, and (for training) a `rul` column.
        All non-feature columns are preserved unchanged.
    features    : list / pd.Index
        Sensor column names to smooth (e.g. the `features` variable from
        your notebook).
    method      : str
        One of  "ema" | "rolling" | "median" | "savgol" | "gaussian"
    span        : int
        EMA span (α = 2/(span+1)).  Larger → smoother.
    window      : int
        Window size for rolling / median / savgol / gaussian.
    poly_order  : int
        Polynomial order for Savitzky-Golay (must be < window).
    sigma       : float
        Standard deviation for Gaussian kernel (in samples).
    engine_col  : str
        Column name that holds the engine / unit ID.
    min_periods : int
        Minimum observations in window before producing a value (rolling/median).

    Returns
    -------
    pd.DataFrame
        Same shape, same column order, same index as `df`.
        Only the `features` columns are smoothed; all other columns are
        copied verbatim.
    """
    method = method.lower()
    _validate_method(method)
    _validate_scipy(method)

    feature_list = list(features)

    # Work on a copy so the caller's DataFrame is never mutated
    out = df.copy()

    for eng_id, grp_idx in df.groupby(engine_col).groups.items():
        # Extract the feature block for this engine (preserves original row order)
        block = df.loc[grp_idx, feature_list].copy()

        if method == "ema":
            smoothed = block.ewm(span=span, min_periods=1, adjust=False).mean()

        elif method == "rolling":
            smoothed = block.rolling(window=window, min_periods=min_periods,
                                     center=False).mean()
            # Fill leading NaNs with original values
            smoothed = smoothed.fillna(block)

        elif method == "median":
            smoothed = block.rolling(window=window, min_periods=min_periods,
                                     center=True).median()
            smoothed = smoothed.fillna(block)

        elif method == "savgol":
            wl = _safe_window(window, len(block), poly_order)
            arr = savgol_filter(block.values, window_length=wl,
                                polyorder=poly_order, axis=0)
            smoothed = pd.DataFrame(arr, index=block.index, columns=block.columns)

        elif method == "gaussian":
            arr = gaussian_filter1d(block.values.astype(float),
                                    sigma=sigma, axis=0)
            smoothed = pd.DataFrame(arr, index=block.index, columns=block.columns)

        out.loc[grp_idx, feature_list] = smoothed.values

    return out


# =============================================================================
#  HELPERS
# =============================================================================

def _validate_method(method: str):
    valid = {"ema", "rolling", "median", "savgol", "gaussian"}
    if method not in valid:
        raise ValueError(
            f"Unknown method '{method}'.  Choose from: {sorted(valid)}"
        )

def _validate_scipy(method: str):
    if method in {"savgol", "gaussian"} and not HAS_SCIPY:
        raise ImportError(
            f"method='{method}' requires scipy.  "
            "Install with:  pip install scipy"
        )

def _safe_window(window: int, n_samples: int, poly_order: int) -> int:
    """Ensure Savitzky-Golay window_length is odd and <= n_samples."""
    wl = min(window, n_samples)
    if wl % 2 == 0:
        wl = max(wl - 1, poly_order + 2)
    wl = max(wl, poly_order + 2)
    return wl


# =============================================================================
#  DIAGNOSTIC: COMPARE ALL METHODS ON ONE ENGINE / SENSOR
# =============================================================================

BG      = "#0E1117"
PANEL   = "#1A1D27"
TEXT    = "#E8EAF0"
GRID    = "#2A2D3A"
RAW_C   = "#FFFFFF"
COLORS  = ["#00C2A8", "#FF6B6B", "#F5C518", "#A78BFA", "#34D399"]
METHODS = ["ema", "rolling", "median", "savgol", "gaussian"]
LABELS  = ["EMA (span=5)", "Rolling mean (w=5)", "Rolling median (w=5)",
           "Savitzky-Golay (w=7,p=2)", "Gaussian (σ=2)"]


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=6)


def compare_smoothing_methods(
    df: pd.DataFrame,
    features,
    engine_id=1,
    sensor: str = None,
    engine_col: str = "engine",
    span: int = 5,
    window: int = 5,
    poly_order: int = 2,
    sigma: float = 2.0,
    save_path: str = "smoothing_comparison.png",
):
    """
    Plot the raw signal alongside every smoothing method for a single
    engine / sensor combination.  Saves a PNG and returns the figure.

    Parameters
    ----------
    df         : the normalised training DataFrame (X)
    features   : sensor feature list
    engine_id  : which engine unit to visualise
    sensor     : sensor name (e.g. "sm_2").  Defaults to features[0].
    save_path  : output PNG path
    """
    feature_list = list(features)
    sensor = sensor or feature_list[0]

    if sensor not in feature_list:
        raise ValueError(f"sensor='{sensor}' not in features list.")

    eng_data = df[df[engine_col] == engine_id].reset_index(drop=True)
    if len(eng_data) == 0:
        raise ValueError(f"engine_id={engine_id} not found in DataFrame.")

    raw = eng_data[sensor].values
    x_ax = np.arange(len(raw))

    # Build smoothed versions (single-engine slice wrapped in a temp df)
    temp = eng_data[[engine_col] + feature_list].copy()
    smoothed_signals = {}
    for method, label in zip(METHODS, LABELS):
        try:
            sm = smooth_engine_data(
                temp, feature_list,
                method=method, span=span, window=window,
                poly_order=poly_order, sigma=sigma,
                engine_col=engine_col
            )
            smoothed_signals[label] = sm[sensor].values
        except ImportError:
            smoothed_signals[label] = None   # scipy missing

    # ── Figure layout ─────────────────────────────────────────────────────
    n_methods = len(METHODS)
    fig = plt.figure(figsize=(18, 4 * (n_methods + 1)), facecolor=BG)
    fig.suptitle(
        f"Smoothing Method Comparison  |  Engine {engine_id}  |  Sensor: {sensor}",
        color=TEXT, fontsize=14, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(n_methods + 1, 1, figure=fig,
                           hspace=0.55, top=0.96, bottom=0.04,
                           left=0.06, right=0.97)

    # Panel 0 – raw signal
    ax0 = fig.add_subplot(gs[0])
    _style_ax(ax0, f"Raw  (normalised)  —  {sensor}")
    ax0.plot(x_ax, raw, color=RAW_C, linewidth=1.0, alpha=0.9)
    ax0.set_xlabel("Time cycle (index)", color=TEXT)
    ax0.set_ylabel("Normalised value", color=TEXT)

    # One panel per method
    for idx, (label, color) in enumerate(zip(LABELS, COLORS)):
        ax = fig.add_subplot(gs[idx + 1])
        _style_ax(ax, label)

        # Raw in background
        ax.plot(x_ax, raw, color=RAW_C, linewidth=0.7, alpha=0.35,
                label="Raw", linestyle="--")

        vals = smoothed_signals.get(label)
        if vals is not None:
            ax.plot(x_ax, vals, color=color, linewidth=1.6, label="Smoothed")
            # Residual fill
            ax.fill_between(x_ax, raw, vals,
                            alpha=0.12, color=color, label="Residual")
            # Metrics in corner
            residual = raw - vals
            rmse_val = float(np.sqrt(np.mean(residual ** 2)))
            ax.text(0.98, 0.92,
                    f"RMSE(raw−smooth)={rmse_val:.4f}",
                    transform=ax.transAxes, ha="right", va="top",
                    color=TEXT, fontsize=8,
                    bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.3"))
        else:
            ax.text(0.5, 0.5, "scipy not installed — method unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#FF6B6B", fontsize=9)

        ax.set_xlabel("Time cycle (index)", color=TEXT)
        ax.set_ylabel("Normalised value", color=TEXT)
        ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL,
                  edgecolor=GRID, loc="upper left")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"✔  Comparison figure saved → {save_path}")
    return fig


# =============================================================================
#  SELF-TEST  (runs when you execute this file directly)
# =============================================================================

if __name__ == "__main__":
    # ── Build a tiny synthetic dataset that mimics your X / X_test structure
    np.random.seed(0)

    features_test = pd.Index([f"sm_{i}" for i in [2, 3, 4, 9, 11, 12, 13, 15, 17, 20, 21]])
    n_engines = 3
    n_cycles  = 80

    rows = []
    for eng in range(1, n_engines + 1):
        t = np.arange(1, n_cycles + 1)
        rul = (n_cycles - t).clip(0)
        for cyc, r in zip(t, rul):
            row = {"engine": eng, "time": cyc, "rul": r}
            for feat in features_test:
                row[feat] = (
                    np.sin(cyc / 10) + 0.5 * np.random.randn()
                )
            rows.append(row)

    df_test_data = pd.DataFrame(rows)
    X_demo = df_test_data[["engine"] + list(features_test) + ["rul"]].copy()

    print("=== Self-test: smooth_engine_data ===")
    print(f"Input  shape : {X_demo.shape}")

    for m in ["ema", "rolling", "median"] + (["savgol", "gaussian"] if HAS_SCIPY else []):
        result = smooth_engine_data(X_demo, features_test, method=m)
        assert result.shape == X_demo.shape,  f"Shape mismatch for method={m}"
        assert list(result.columns) == list(X_demo.columns), f"Column mismatch for method={m}"
        print(f"  [{m:>8s}]  output shape: {result.shape}  ✔")

    print("\n=== Self-test: compare_smoothing_methods ===")
    fig = compare_smoothing_methods(
        X_demo, features_test,
        engine_id=1, sensor="sm_2",
        save_path="smoothing_comparison_selftest.png"
    )
    print("Self-test complete.\n")

    # ── Usage reminder ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  COPY-PASTE SNIPPET FOR YOUR NOTEBOOK")
    print("=" * 65)
    print("""
from smooth_timeseries import smooth_engine_data, compare_smoothing_methods

# --- (Optional) visualise methods before committing to one ---
compare_smoothing_methods(X, features, engine_id=1, sensor="sm_2",
                          save_path="smoothing_comparison.png")

# --- Choose method and apply ---
X_smooth      = smooth_engine_data(X,      features, method="ema", span=5)
X_test_smooth = smooth_engine_data(X_test, features, method="ema", span=5)

# --- Drop-in replacement for existing sliding-window calls ---
X_train_sw, y_train_sw = create_training_sequences_sw(X_smooth,      features, window_size)
X_testf                = create_testing_sequences_sw (X_test_smooth,  features, window_size, num_of_batches)
""")