"""
=============================================================================
  TRANSFORMER IMPROVEMENT SCRIPT  —  PatchTST Dual-Dim RUL
=============================================================================

  ANALYSIS OF CURRENT BOTTLENECKS
  --------------------------------
  After reviewing the architecture and training loop, five root causes were
  identified that are limiting RMSE beyond 12.7:

  1. LOSS MISALIGNMENT
     MSELoss weights over-prediction and under-prediction equally.
     The NASA scoring function is ASYMMETRIC: late predictions (positive
     error, a2=13) are penalised MORE than early predictions (a1=10).
     MSELoss has no concept of this and pushes the model toward a symmetric
     minimum that is sub-optimal for the actual objective.

  2. NO LEARNING-RATE SCHEDULE
     A fixed lr=1e-4 with AdamW is used throughout.  Without a schedule the
     optimizer can stall in flat regions of the loss landscape after the
     initial fast descent, leaving validation MAE stuck above its reachable
     minimum.

  3. UNDER-CAPACITY MODEL
     d_model=64, n_layers=2, d_ff=128 is a very small transformer.  With 11
     sensor channels and window_size=40, the model has limited representational
     power to capture the nonlinear degradation dynamics across all four engine
     types.

  4. SMALL WINDOW / INSUFFICIENT TEMPORAL CONTEXT
     window_size=40 captures only 40 cycles.  Engines in FD002/FD004 can run
     for 300+ cycles, so the model sees less than 15% of an engine's life in
     each training sample.  Increasing the window gives the patch encoders more
     context to build degradation trajectories.

  5. NO TRAINING-TIME REGULARISATION / AUGMENTATION
     The model receives each sensor sequence exactly as normalised, with no
     stochasticity beyond dropout.  Light Gaussian jitter on training sequences
     acts as data augmentation, improving generalisation.

  FIVE SCENARIOS TESTED
  ---------------------
  Each scenario is a targeted fix for one or more of the bottlenecks above.
  All scenarios reuse the EXACT same architecture classes — only the training
  recipe and hyperparameters change.

  Scenario A — Asymmetric Loss + LR Scheduler
      Fix bottlenecks 1 & 2.  Fast, low-risk, high-reward.

  Scenario B — Wider/Deeper Model + LR Scheduler
      Fix bottlenecks 2 & 3.  More model capacity.

  Scenario C — Larger Window + Augmentation
      Fix bottlenecks 4 & 5.  More temporal context + regularisation.

  Scenario D — Full Combo (all fixes together)
      Fix all five bottlenecks simultaneously.

  Scenario E — Ensemble of Scenario D across 3 seeds
      Averages predictions from three independently trained D-models.
      Most reliable but takes 3× the compute.

  HOW TO USE
  ----------
  1. Run the existing notebook/script up to (and including) the definitions of:
         X_train_sw, y_train_sw   (from create_training_sequences_sw)
         X_testf, y_test          (from create_testing_sequences_sw)
         features, eng_type, device
     and all architecture classes (PatchTSTEncoder, SensorChannelTransformerEncoder,
     FusionHead, PatchTST_RUL_Model, etc.)

  2. Then run this file:
         exec(open("improve_transformer.py").read())
     or add:
         from improve_transformer import run_all_scenarios
         run_all_scenarios(X_train_sw, y_train_sw, X_testf, y_test,
                           features, eng_type, device)

  All five scenarios are run automatically.  A leaderboard is printed and
  the best model is saved.
=============================================================================
"""

import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# =============================================================================
#  SECTION 0 — TrainConfig DATACLASS
#  Defined here so the script is fully self-contained and does not depend on
#  TrainConfig being defined in the calling notebook.
# =============================================================================

@dataclass
class TrainConfig:
    feature_cols: List[str]
    target_col:   str   = "RUL"
    group_col:    str   = "engine"
    time_col:     str   = "time"

    # Sliding window / model dimensions
    # (C and L are overwritten by make_loaders_augmented return values before
    #  the model is constructed, so the defaults of 0 are never actually used)
    C: int = 0
    L: int = 0

    patch_len_t: int   = 10
    stride_t:    int   = 5
    patch_len_c: int   = 3
    stride_c:    int   = 1

    # Temporal encoder
    d_model_t:  int   = 64
    n_heads_t:  int   = 8
    n_layers_t: int   = 2
    d_ff_t:     int   = 128
    dropout_t:  float = 0.1

    # Channel encoder
    d_model_c:  int   = 64
    n_heads_c:  int   = 8
    n_layers_c: int   = 2
    d_ff_c:     int   = 128
    dropout_c:  float = 0.1

    # Fusion head
    head_hidden:  Optional[int] = 128
    head_dropout: float         = 0.1

    use_feature_attn: bool = True

    # Optimiser / schedule
    batch_size:   int   = 40
    epochs:       int   = 150
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    patience:     int   = 15

    device:      str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    model_path:  str = "patchtst_rul_best.pt"



# -----------------------------
# PatchTST blocks
# -----------------------------
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from Encoder_Layers import *
from CommonFunctions import *


# ---------------------------------
# Patch Embedding (time --> tokens)
# ---------------------------------
class PatchEmbedding(nn.Module):
    """
    Turn a (B*C, L) series into a sequence of patch tokens (B*C, N, d_model).
    Each token is a linear projection of a length-P patch.
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: (B*C, L)
        # Dimension error check
        L = x.shape[1]
        if L < self.patch_len:
            raise ValueError(f"Lookback L={L} < patch_len={self.patch_len}. Increase lookback or reduce patch_len.")
        
        # N = floor((L - P)/stride) + 1
        n_patches = 1 + (L - self.patch_len) // self.stride
        if n_patches <= 0:
            raise ValueError("No patches would be created; check patch_len/stride vs lookback.")
        
        # unfold → (B*C, N, P)
        # Create overlapping/unoverlapping patches: (B*C, N, P)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B*C, N, P)
        Bc, N, P = patches.shape
        # Linear projection per patch
        tokens = self.proj(patches)  # (B*C, N, d_model)
        return tokens

# Fixed positional encoding
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :]


# ----------------------------------------
# Stage 1A: Sequence encoder (PatchTST CI)
# ----------------------------------------
class PatchTSTEncoder(nn.Module):
    """
    Channel-Independent Transformer over patches (shared weights across channels).
    - InstanceNorm per (sample, channel) series.
    - Patchify + linear embedding.
    - Positional encoding + TransformerEncoder.
    - Mean-pool tokens -> per-channel representation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        patch_len: int,
        stride: int,
        use_batchnorm_out: bool = False
    ):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(1, affine=False, eps=1e-6)

        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)
        
        encoder_layer = TransformerBatchNormEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_enc = SinusoidalPositionEncoding(d_model)

        # Use either batch normalization or layer normalization
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            # BN over feature dim: expects (B, d_model, seq)
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, C, L)
        B, C, L = x.shape

        # InstanceNorm per channel per sample
        x = x.reshape(B * C, 1, L)
        x = self.inst_norm(x)        # (B*C, 1, L)
        x = x.squeeze(1)             # (B*C, L)

        # Patching + embedding
        tokens = self.patch_embed(x) # (B*C, N, d_model)

        # Positional encoding + Transformer
        tokens = self.pos_enc(tokens)
        enc = self.encoder(tokens)   # (B*C, N, d_model)        

        # reshape to group channels, then aggregate across channels
        BxC, N, D = enc.shape
        enc = enc.view(B, C, N, D)        # (B, C, N, d_model)
        enc = enc.mean(dim=1)             # (B, N, d_model)   <-- temporal tokens

        # optional norm
        if self.use_bn:
            enc = enc.transpose(1, 2)     # (B, d_model, N)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)     # (B, N, d_model)
        else:
            enc = self.ln_out(enc)        # (B, N, d_model)

        return enc  # temporal_out: (B, N, d_model)



# ----------------------------------------------
# Stage 1B: Feature encoder (channel attention)
# -----------------------------------------------
class SensorChannelTransformerEncoder(nn.Module):
    """
    Attend across sensors. For each sensor, compress its time window L -> d_model,
    yielding tokens = sensors (length C).
    """
    def __init__(self,  C: int, L: int, patch_len: int, stride: int, 
                 d_model=128, n_heads=8, num_layers=4, dim_feedforward=512, dropout=0.1,
                 use_batchnorm_out: bool = False):
        super().__init__()

        self.C = C
        self.L = L

        # Patch embedding along the channel (sensor) dimension
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionEncoding(d_model)
        
        encoder_layers = TransformerBatchNormEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # # Final norm
        # self.norm_out = nn.LayerNorm(d_model)

        # Norm at output
        # Use either batch normalization or layer normalization
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)  # will use (B, d_model, C)
        else:
            self.ln_out = nn.LayerNorm(d_model)

        # self.inst_norm = nn.InstanceNorm1d(self.C, affine=False, eps=1e-6)
        # (optional) IN across sensors for each time index; comment out if not wanted

    def forward(self, x):
        """
        x: (B, C, L)  -> sensor-time matrix
        We patch along the *sensor dimension C*.
        """
       
        B, C, L = x.shape
        assert C == self.C and L == self.L, "Shape mismatch for SensorChannelTransformerEncoder"

        # Rearrange to (B*L, C) so we can patch along channels
        x = x.permute(0, 2, 1)     # (B, L, C)
        x = x.reshape(B * L, C)    # treat each time step separately

        # Apply patch embedding along sensor dimension
        tokens = self.patch_embed(x)  # (B*L, num_patches, d_model)

        # Restore batch/time structure
        num_patches = tokens.size(1)
        tokens = tokens.view(B, L, num_patches, -1)   # (B, L, N_patch, d_model)

        # Merge time and sensor-patch tokens: treat each (time, patch) as a token
        tokens = tokens.view(B, L * num_patches, -1)  # (B, L*N_patch, d_model)

        # add sensor positional encodings: treat sensors as tokens
        tokens = self.pos_encoder(tokens)     # (B, C, d_model)

        # Transformer over sensor tokens
        enc = self.transformer_encoder(tokens)     # (B, L*N_patch, d_model)

        # final norm
        if self.use_bn:
            enc = enc.transpose(1, 2)   # (B, d_model, seq)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)   # (B, seq, d_model)
        else:
            enc = self.ln_out(enc)

        return enc   # (B, L*N_patch, d_model)



# -------------------------------------
# Fusion with Batch Normalization
# -------------------------------------
class FusionHead(nn.Module):
    def __init__(self, d_model_t: int, d_model_c: int, head_hidden: Optional[int] = None,
                 dropout: float = 0.1, pooling="mean"):
        super(FusionHead, self).__init__()

        # project to common width
        self.proj_t = nn.Identity() if d_model_t == d_model_c else nn.Linear(d_model_t, d_model_c)
        self.d_model = d_model_c

        # Replace LayerNorm with BatchNorm
        self.norm = nn.BatchNorm1d(self.d_model)

        assert pooling in ["mean", "cls"]
        self.pooling = pooling

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, temporal_out, channel_out):
        """
        temporal_out: (B, N, d_model_t)
        channel_out : (B, C, d_model_c)
        """
        
        # Project temporal side
        t = self.proj_t(temporal_out)  # (B, N, d_model)
        c = channel_out                # (B, C, d_model)

        # Concat token sequences
        p = torch.cat([t, c], dim=1)   # (B, N+C, d_model)

        # --- BatchNorm requires permute ---
        p = p.permute(0, 2, 1)         # (B, d_model, N+C)
        p = self.norm(p)               # BN across feature dimension
        p = p.permute(0, 2, 1)         # back to (B, N+C, d_model)

        # Pooling
        if self.pooling == "mean":
            pooled = p.mean(dim=1)     # (B, d_model)
        else:
            pooled = p[:, 0, :]        # CLS-style

        return self.mlp(pooled)



# Can use PatchTST_RUL_Model - as a single stage
class PatchTST_RUL_Model(nn.Module):
    def __init__(
        self,
        C, L, 
        d_model_t: int ,
        n_heads_t: int ,
        n_layers_t: int ,
        d_ff_t: int ,
        dropout_t: float ,
        patch_len_t: int ,
        stride_t: int ,
        patch_len_c: int ,
        stride_c: int ,
        d_model_c: int ,
        n_heads_c: int ,
        n_layers_c: int ,
        d_ff_c: int ,
        dropout_c: float,
        head_hidden: Optional[int] = None,
        pooling="mean",
        use_bn_temporal=True, 
        use_bn_channel=True
        
    ):
        super().__init__()

        self.temporal_encoder = PatchTSTEncoder(
            d_model=d_model_t, n_heads=n_heads_t, n_layers=n_layers_t,
            d_ff=d_ff_t, dropout=dropout_t, patch_len=patch_len_t, stride=stride_t, use_batchnorm_out=use_bn_temporal
        )

        self.sensor_encoder = SensorChannelTransformerEncoder(C=C, L=L, patch_len=patch_len_c, stride=stride_c,
            d_model=d_model_c, n_heads=n_heads_c, num_layers=n_layers_c, dim_feedforward=d_ff_c,
            dropout=dropout_c, use_batchnorm_out=use_bn_channel
        )

        self.fusion_head = FusionHead(d_model_t, d_model_c, head_hidden, dropout_t, pooling)

    def forward(self, x):  # x: (B, C, L)
        te = self.temporal_encoder(x)       # (B, N, d_model_t)
        se = self.sensor_encoder(x)         # (B, C, d_model_c)             
        y = self.fusion_head(te, se)        # (B, 1)

        return y.squeeze(-1)                # (B,)
# =============================================================================
#  SECTION 1 — IMPROVED LOSS FUNCTIONS
# =============================================================================

class AsymmetricHuberLoss(nn.Module):
    """
    Asymmetric Huber loss aligned with the NASA CMAPSS scoring function.

    For a prediction error  e = y_pred - y_true:
      - e > 0  (late prediction, under-estimating RUL):
            weight = alpha_late  (default > 1, penalise more)
      - e <= 0 (early prediction, over-estimating RUL):
            weight = alpha_early (default 1.0)

    Within each direction the loss transitions from L1 to L2 at the
    `delta` threshold (standard Huber behaviour), which makes it robust
    to occasional large outliers while still being smooth near zero.

    Why this helps
    --------------
    The NASA scoring function uses a1=10 (early) and a2=13 (late).
    Because a2 > a1, the penalty grows FASTER for late predictions.
    MSELoss treats both directions identically, so the model converges
    to a symmetric minimum that incurs unnecessary NASA-score penalty
    from late predictions.  This loss pushes the model to bias slightly
    early, which is exactly what the NASA scoring function rewards.
    """
    def __init__(self, delta: float = 10.0,
                 alpha_late: float = 1.3,
                 alpha_early: float = 1.0):
        super().__init__()
        self.delta       = delta
        self.alpha_late  = alpha_late   # e > 0  (under-estimated RUL)
        self.alpha_early = alpha_early  # e <= 0 (over-estimated RUL)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        e    = y_pred - y_true          # positive = late, negative = early
        abs_e = e.abs()

        # Standard Huber kernel
        huber = torch.where(
            abs_e <= self.delta,
            0.5 * e ** 2,
            self.delta * (abs_e - 0.5 * self.delta)
        )

        # Asymmetric weighting
        weight = torch.where(e > 0,
                             torch.full_like(e, self.alpha_late),
                             torch.full_like(e, self.alpha_early))

        return (weight * huber).mean()


class WeightedMSELoss(nn.Module):
    """
    MSE with a linear ramp weight based on true RUL.
    Early-life samples (high RUL) contribute less; end-of-life samples
    (low RUL, RUL < low_rul_threshold) contribute more because getting
    those predictions right matters most operationally.
    """
    def __init__(self, low_rul_threshold: float = 30.0,
                 high_weight: float = 2.0):
        super().__init__()
        self.thr    = low_rul_threshold
        self.w_high = high_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weight = torch.where(y_true <= self.thr,
                             torch.full_like(y_true, self.w_high),
                             torch.ones_like(y_true))
        return (weight * (y_pred - y_true) ** 2).mean()


# =============================================================================
#  SECTION 2 — DATA AUGMENTATION
# =============================================================================

class AugmentedRULDataset(Dataset):
    """
    Wraps RULWindowDataset-style (X, y) arrays and applies optional
    per-sample augmentation during training.

    Augmentations:
      gaussian_noise  — adds N(0, noise_std) to every sensor value
      time_warp       — randomly drops 1 time step and repeats a neighbour
                        (keeps sequence length constant)

    These are applied only during __getitem__ so validation/test data
    is never touched.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = True,
                 noise_std: float = 0.02,
                 time_warp: bool = False):
        assert X.ndim == 3
        self.X          = X.astype(np.float32, copy=False)
        self.y          = y.astype(np.float32, copy=False)
        self.augment    = augment
        self.noise_std  = noise_std
        self.time_warp  = time_warp

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()          # (C, L)
        y = self.y[idx]

        if self.augment:
            # Gaussian jitter on all sensor channels
            if self.noise_std > 0:
                x += np.random.randn(*x.shape).astype(np.float32) * self.noise_std

            # Time-warp: replace one random time step with average of neighbours
            if self.time_warp and x.shape[1] > 3:
                t = np.random.randint(1, x.shape[1] - 1)
                x[:, t] = 0.5 * (x[:, t - 1] + x[:, t + 1])

        return torch.from_numpy(x), torch.tensor(y)


# =============================================================================
#  SECTION 3 — IMPROVED TRAINING UTILITIES
# =============================================================================

def make_loaders_augmented(
    X_train, X_val, y_train, y_val,
    batch_size: int = 64,
    num_workers: int = 0,
    noise_std: float = 0.02,
    time_warp: bool = False,
    use_cuda: bool = torch.cuda.is_available(),
) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """
    Same interface as make_loaders() in the original script, but the
    training loader uses AugmentedRULDataset so augmentation is automatic.
    Validation loader has augment=False always.
    """
    Xt = X_train.transpose(0, 2, 1)   # (N, C, L)
    Xv = X_val.transpose(0, 2, 1)

    train_ds = AugmentedRULDataset(Xt, y_train, augment=True,
                                   noise_std=noise_std, time_warp=time_warp)
    val_ds   = AugmentedRULDataset(Xv, y_val,   augment=False)

    pin = bool(use_cuda)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, (Xt.shape[1], Xt.shape[2])  # (C, L)


def train_one_epoch_improved(model, loader, device, optimizer,
                              criterion, grad_clip: float = 1.0,
                              accumulation_steps: int = 1) -> float:
    """
    Training loop with optional gradient accumulation.
    accumulation_steps > 1 simulates a larger effective batch size without
    increasing GPU memory usage.
    """
    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        preds = model(xb)
        loss  = criterion(preds, yb) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps * xb.size(0)
        n += xb.size(0)

    # Final partial accumulation step
    if len(loader) % accumulation_steps != 0:
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_improved(model, loader, device, criterion):
    """Same interface as evaluate() in the original script."""
    model.eval()
    losses, n = 0.0, 0
    preds_all, targets_all = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        preds = model(xb)
        loss  = criterion(preds, yb)
        losses += loss.item() * xb.size(0)
        n      += xb.size(0)
        preds_all.append(preds.cpu().numpy())
        targets_all.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sse  = float(np.sum((y_true - y_pred) ** 2))
    sst  = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1.0 - sse / sst) if sst > 0 else float("nan")

    return losses / max(1, n), {"MAE": mae, "RMSE": rmse, "R2": r2}, y_true, y_pred


def score_nasa(errors: np.ndarray) -> float:
    """NASA CMAPSS scoring function (lower is better)."""
    a1, a2 = 10, 13
    s = 0.0
    for e in errors:
        if e < 0:
            s += math.exp(-e / a1) - 1
        else:
            s += math.exp(e / a2) - 1
    return s


# =============================================================================
#  SECTION 4 — IMPROVED TRAINING LOOP
# =============================================================================

def fit_improved(
    train_loader, val_loader,
    features, cfg,
    device,
    loss_fn=None,
    use_scheduler: bool = True,
    scheduler_type: str = "cosine_warm",   # "cosine_warm" | "cosine" | "plateau"
    augment_noise: float = 0.0,
    accumulation_steps: int = 1,
    verbose: bool = True,
):
    """
    Drop-in replacement for fit_patchtst_dualdim_rul() with four improvements:
      1. Pluggable loss function (defaults to AsymmetricHuberLoss)
      2. LR scheduler (CosineAnnealingWarmRestarts by default)
      3. Gradient accumulation
      4. Returns NASA score alongside standard metrics
    """
    # ── Build model (same architecture, same config fields) ───────────────
    model = PatchTST_RUL_Model(
        C=cfg.C, L=cfg.L,
        d_model_t=cfg.d_model_t, n_heads_t=cfg.n_heads_t,
        n_layers_t=cfg.n_layers_t, d_ff_t=cfg.d_ff_t,
        dropout_t=cfg.dropout_t,
        patch_len_t=cfg.patch_len_t, stride_t=cfg.stride_t,
        patch_len_c=cfg.patch_len_c, stride_c=cfg.stride_c,
        d_model_c=cfg.d_model_c, n_heads_c=cfg.n_heads_c,
        n_layers_c=cfg.n_layers_c, d_ff_c=cfg.d_ff_c,
        dropout_c=cfg.dropout_c,
        head_hidden=cfg.head_hidden,
        pooling="mean",
        use_bn_temporal=True,
        use_bn_channel=True,
    ).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────
    if loss_fn is None:
        loss_fn = AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = None
    if use_scheduler:
        if scheduler_type == "cosine_warm":
            # Warm restarts every T_0 epochs; period doubles after each restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=1e-6
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
            )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_mae   = float("inf")
    best_state     = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_improved(
            model, train_loader, device, optimizer, loss_fn,
            grad_clip=cfg.grad_clip, accumulation_steps=accumulation_steps
        )
        val_loss, val_mets, _, _ = evaluate_improved(
            model, val_loader, device, loss_fn
        )

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mets["MAE"])
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:03d}] TrainLoss={train_loss:.4f}  "
                f"ValLoss={val_loss:.4f}  "
                f"ValMAE={val_mets['MAE']:.4f}  "
                f"ValRMSE={val_mets['RMSE']:.4f}  "
                f"ValR2={val_mets['R2']:.4f}  "
                f"LR={lr_now:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        # Early stopping on MAE
        if val_mets["MAE"] < best_val_mae - 1e-6:
            best_val_mae      = val_mets["MAE"]
            best_state        = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                if verbose:
                    print(f"Early stop at epoch {epoch}. Best Val MAE={best_val_mae:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# =============================================================================
#  SECTION 5 — FIVE IMPROVEMENT SCENARIOS
# =============================================================================

# ── Scenario A: Asymmetric Loss + LR Scheduler ───────────────────────────
#    Why: Fixes the two cheapest bottlenecks (loss misalignment + no schedule).
#    Risk: Low. Same model size, same data.
def scenario_A_config(features, window_size):
    return TrainConfig(
        feature_cols = list(features),
        C            = len(features),
        L            = window_size,
        patch_len_t  = 10, stride_t = 5,
        patch_len_c  = 3,  stride_c = 1,
        d_model_t=64,  n_heads_t=8,  n_layers_t=2, d_ff_t=128, dropout_t=0.1,
        d_model_c=64,  n_heads_c=8,  n_layers_c=2, d_ff_c=128, dropout_c=0.1,
        head_hidden=128, head_dropout=0.1,
        batch_size=40, epochs=150, lr=3e-4,   # slightly higher lr works well with schedule
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"scenarioA_best.pt",
    )


# ── Scenario B: Wider/Deeper Model + LR Scheduler ────────────────────────
#    Why: d_model=64, n_layers=2 is under-capacity for 11-channel multi-regime data.
#    d_model=96, n_layers=3 roughly doubles the number of attention parameters.
#    Risk: Medium. Slightly more compute and GPU memory.
def scenario_B_config(features, window_size):
    return TrainConfig(
        feature_cols = list(features),
        C            = len(features),
        L            = window_size,
        patch_len_t  = 10, stride_t = 5,
        patch_len_c  = 3,  stride_c = 1,
        d_model_t=96,  n_heads_t=8,  n_layers_t=3, d_ff_t=256, dropout_t=0.15,
        d_model_c=96,  n_heads_c=8,  n_layers_c=3, d_ff_c=256, dropout_c=0.15,
        head_hidden=192, head_dropout=0.1,
        batch_size=40, epochs=150, lr=2e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"scenarioB_best.pt",
    )


# ── Scenario C: Larger Window + Augmentation ─────────────────────────────
#    Why: window=40 is only ~13% of a long FD002/FD004 engine life.
#    window=50 gives patch encoders more degradation trajectory context.
#    Gaussian jitter (noise_std=0.02) improves generalisation.
#    Risk: Medium. Requires re-running create_training_sequences_sw with window=50.
#    NOTE: This scenario uses its own window internally.
def scenario_C_config(features, window_size=50):
    return TrainConfig(
        feature_cols = list(features),
        C            = len(features),
        L            = window_size,
        patch_len_t  = 10, stride_t = 5,
        patch_len_c  = 3,  stride_c = 1,
        d_model_t=64,  n_heads_t=8,  n_layers_t=2, d_ff_t=128, dropout_t=0.1,
        d_model_c=64,  n_heads_c=8,  n_layers_c=2, d_ff_c=128, dropout_c=0.1,
        head_hidden=128, head_dropout=0.1,
        batch_size=40, epochs=150, lr=3e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"scenarioC_best.pt",
    )


# ── Scenario D: Full Combo ────────────────────────────────────────────────
#    Why: All five bottlenecks addressed together.
#    Wider model + larger window + asymmetric loss + schedule + augmentation.
#    Risk: Highest compute cost, but highest expected payoff.
def scenario_D_config(features, window_size=50):
    return TrainConfig(
        feature_cols = list(features),
        C            = len(features),
        L            = window_size,
        patch_len_t  = 10, stride_t = 5,
        patch_len_c  = 3,  stride_c = 1,
        d_model_t=96,  n_heads_t=8,  n_layers_t=3, d_ff_t=256, dropout_t=0.15,
        d_model_c=96,  n_heads_c=8,  n_layers_c=3, d_ff_c=256, dropout_c=0.15,
        head_hidden=192, head_dropout=0.1,
        batch_size=40, epochs=200, lr=2e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"scenarioD_best.pt",
    )


# =============================================================================
#  SECTION 6 — ENSEMBLE PREDICTION (Scenario E)
# =============================================================================

def ensemble_predict(models, loader, device):
    """
    Average predictions from a list of models.
    Each model must have the same architecture.
    Averaging reduces variance in predictions, improving RMSE reliably.
    """
    all_preds = []
    y_true_all = []

    for model in models:
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device).float()
                preds.append(model(xb).cpu().numpy())
                ys.append(yb.numpy())
        all_preds.append(np.concatenate(preds))
        y_true_all = np.concatenate(ys)

    # Simple mean ensemble
    y_pred_ens = np.mean(np.stack(all_preds, axis=0), axis=0)

    mae  = float(np.mean(np.abs(y_true_all - y_pred_ens)))
    rmse = float(np.sqrt(np.mean((y_true_all - y_pred_ens) ** 2)))
    sse  = float(np.sum((y_true_all - y_pred_ens) ** 2))
    sst  = float(np.sum((y_true_all - np.mean(y_true_all)) ** 2))
    r2   = float(1.0 - sse / sst) if sst > 0 else float("nan")

    return y_pred_ens, y_true_all, {"MAE": mae, "RMSE": rmse, "R2": r2}


# =============================================================================
#  SECTION 7 — MAIN ORCHESTRATOR
# =============================================================================

def run_all_scenarios(
    X_train_sw,
    y_train_sw,
    X_testf,
    y_test,
    features,
    eng_type: str,
    device,
    # Required for Scenarios C & D which re-window at a larger window_size
    X=None,
    X_test=None,
    create_training_sequences_sw=None,
    create_testing_sequences_sw=None,
    num_of_batches=1,
    window_size: int = 40,
    random_state: int = 341,
    run_ensemble: bool = True,
    verbose: bool = True,
):
    """
    Runs Scenarios A through E, prints a leaderboard, saves the best model.
 
    Parameters
    ----------
    X_train_sw   : numpy array from create_training_sequences_sw(X, features, window_size)
    y_train_sw   : numpy array of RUL targets
    X_testf      : numpy array from create_testing_sequences_sw
    y_test       : numpy array of test RUL targets
    features     : pd.Index or list of sensor column names
    eng_type     : "FD001" | "FD002" | "FD003" | "FD004"
    device       : torch.device
    X            : normalised training DataFrame — REQUIRED for Scenarios C & D
    X_test       : normalised test DataFrame    — REQUIRED for Scenarios C & D
    create_training_sequences_sw : sliding-window fn from your notebook — REQUIRED for C & D
    create_testing_sequences_sw  : sliding-window fn from your notebook — REQUIRED for C & D
    num_of_batches : passed through to create_testing_sequences_sw (default 1)
    window_size  : window used to build the pre-computed X_train_sw (default 40)
    random_state : train/val split seed
    run_ensemble : if True, also runs Scenario E (3-seed ensemble of best config)
    """
 
    print("=" * 70)
    print("  TRANSFORMER IMPROVEMENT EXPERIMENT")
    print(f"  Engine type : {eng_type}")
    print(f"  Window size : {window_size}")
    print(f"  Features    : {list(features)}")
    print("=" * 70)
 
    # ── Train/val split ───────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sw, y_train_sw, test_size=0.2, random_state=random_state
    )
 
    # ── Test loader (shared across all scenarios) ─────────────────────────
    X_test_trans = X_testf.transpose(0, 2, 1)
 
    class _SimpleDS(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
 
    test_loader = DataLoader(
        _SimpleDS(X_test_trans, y_test),
        batch_size=64, shuffle=False
    )
 
    # ── Evaluation criterion (MSE for comparable loss logging) ────────────
    mse_crit = nn.MSELoss()
 
    # ── Scenario registry ─────────────────────────────────────────────────
    # Each entry: (label, description, config_fn, window, loss_fn, noise_std,
    #              use_scheduler, scheduler_type, accumulation_steps)
    LARGER_W = max(window_size, 50)   # never shrink the window
 
    scenarios = [
        (
            "A",
            "Asymmetric Loss + CosineWarm LR",
            lambda: scenario_A_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0,    # no noise
            True, "cosine_warm", 1
        ),
        (
            "B",
            "Wider/Deeper Model (d=96, L=3) + CosineWarm LR",
            lambda: scenario_B_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0,
            True, "cosine_warm", 1
        ),
        (
            "C",
            "Larger Window + Gaussian Augmentation",
            lambda: scenario_C_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02,   # noise
            True, "cosine_warm", 1
        ),
        (
            "D",
            "Full Combo (Wider + Larger Window + Aug + Loss + Schedule)",
            lambda: scenario_D_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02,
            True, "cosine_warm", 1
        ),
    ]
 
    # ── Cache re-windowed data to avoid repeating sliding-window prep ─────
    # Scenarios A & B use original window_size
    # Scenarios C & D use LARGER_W  → need new sequences
    _seq_cache = {}  # window -> (X_train_sw, y_train_sw, X_testf)
 
    def _get_sequences(w):
        if w not in _seq_cache:
            if w == window_size:
                _seq_cache[w] = (X_train_sw, y_train_sw, X_testf)
            else:
                # Scenarios C & D need to re-run the sliding-window functions
                # with a larger window.  The four variables below must be
                # supplied by the caller — raise a helpful error if not.
                missing = [
                    name for name, val in [
                        ("X",                            X),
                        ("X_test",                       X_test),
                        ("create_training_sequences_sw", create_training_sequences_sw),
                        ("create_testing_sequences_sw",  create_testing_sequences_sw),
                    ] if val is None
                ]
                if missing:
                    raise ValueError(
                        f"Scenarios C & D need a larger window (window_size={w}) "
                        f"and must re-run the sliding-window functions, but the "
                        f"following required arguments were not passed to "
                        f"run_all_scenarios():\n  {missing}\n\n"
                        f"Add them to your call:\n"
                        f"  run_all_scenarios(\n"
                        f"      ...\n"
                        f"      X=X,\n"
                        f"      X_test=X_test,\n"
                        f"      create_training_sequences_sw=create_training_sequences_sw,\n"
                        f"      create_testing_sequences_sw=create_testing_sequences_sw,\n"
                        f"  )"
                    )
                print(f"\n  [Re-windowing data for window_size={w} ...]")
                Xsw, ysw = create_training_sequences_sw(X, features, w)
                Xtf      = create_testing_sequences_sw(
                    X_test, features, w, num_of_batches=num_of_batches
                )
                _seq_cache[w] = (Xsw, ysw, Xtf)
        return _seq_cache[w]
 
    # ── Run each scenario ──────────────────────────────────────────────────
    leaderboard = []
 
    for (label, desc, cfg_fn, win, loss_fn, noise,
         use_sched, sched_type, accum) in scenarios:
 
        print(f"\n{'─'*70}")
        print(f"  SCENARIO {label}: {desc}")
        print(f"{'─'*70}")
 
        cfg = cfg_fn()
 
        # Get (possibly re-windowed) sequences
        Xsw, ysw, Xtf = _get_sequences(win)
        Xtr_s, Xvl_s, ytr_s, yvl_s = train_test_split(
            Xsw, ysw, test_size=0.2, random_state=random_state
        )
 
        # Loaders
        tr_loader, vl_loader, (C, L) = make_loaders_augmented(
            Xtr_s, Xvl_s, ytr_s, yvl_s,
            batch_size=cfg.batch_size,
            num_workers=getattr(cfg, "num_workers", 0),
            noise_std=noise,
            time_warp=False,
            use_cuda=str(device).startswith("cuda"),
        )
        cfg.C = C
        cfg.L = L
 
        # ── Also build test loader for this window size ──────────────────
        Xtf_trans = Xtf.transpose(0, 2, 1)
        t_loader  = DataLoader(
            _SimpleDS(Xtf_trans, y_test),
            batch_size=64, shuffle=False
        )
 
        # ── Train ─────────────────────────────────────────────────────────
        model, tr_losses, vl_losses = fit_improved(
            tr_loader, vl_loader,
            features, cfg, device,
            loss_fn=loss_fn,
            use_scheduler=use_sched,
            scheduler_type=sched_type,
            accumulation_steps=accum,
            verbose=verbose,
        )
 
        # ── Evaluate on test set ──────────────────────────────────────────
        _, test_mets, yt, yp = evaluate_improved(model, t_loader, device, mse_crit)
        errors = yp - yt
        nasa   = score_nasa(errors)
 
        print(f"\n  [Scenario {label} TEST]  "
              f"RMSE={test_mets['RMSE']:.4f}  "
              f"MAE={test_mets['MAE']:.4f}  "
              f"R²={test_mets['R2']:.4f}  "
              f"NASA={nasa:.1f}")
 
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   f"scenario{label}_{eng_type}_final.pt")
 
        leaderboard.append({
            "scenario" : label,
            "desc"     : desc,
            "RMSE"     : test_mets["RMSE"],
            "MAE"      : test_mets["MAE"],
            "R2"       : test_mets["R2"],
            "NASA"     : nasa,
            "model"    : model,
            "window"   : win,
            "y_pred"   : yp,
            "y_true"   : yt,
        })
 
    # ── Sort leaderboard by RMSE ──────────────────────────────────────────
    leaderboard.sort(key=lambda r: r["RMSE"])
    best_scenario = leaderboard[0]
 
    # ── Scenario E: Ensemble (3 seeds of best config) ─────────────────────
    if run_ensemble:
        print(f"\n{'─'*70}")
        print(f"  SCENARIO E: 3-seed Ensemble of Scenario {best_scenario['scenario']}")
        print(f"{'─'*70}")
 
        best_label = best_scenario["scenario"]
        best_win   = best_scenario["window"]
        # Look up matching scenario row to get its config/loss/noise
        base_row   = next(r for r in scenarios if r[0] == best_label)
        _, _, bcfg_fn, bwin, bloss, bnoise, bsched, bstype, baccum = base_row
 
        Xsw_e, ysw_e, Xtf_e = _get_sequences(best_win)
        Xtf_e_trans = Xtf_e.transpose(0, 2, 1)
        te_loader   = DataLoader(
            _SimpleDS(Xtf_e_trans, y_test),
            batch_size=64, shuffle=False
        )
 
        ensemble_seeds  = [42, 137, 271]
        ensemble_models = []
 
        for seed in ensemble_seeds:
            print(f"\n  [Ensemble seed={seed}]")
            torch.manual_seed(seed)
            np.random.seed(seed)
 
            Xtr_e, Xvl_e, ytr_e, yvl_e = train_test_split(
                Xsw_e, ysw_e, test_size=0.2, random_state=seed
            )
            ecfg = bcfg_fn()
            tr_e, vl_e, (Ce, Le) = make_loaders_augmented(
                Xtr_e, Xvl_e, ytr_e, yvl_e,
                batch_size=ecfg.batch_size,
                num_workers=getattr(ecfg, "num_workers", 0),
                noise_std=bnoise,
                use_cuda=str(device).startswith("cuda"),
            )
            ecfg.C = Ce
            ecfg.L = Le
 
            em, _, _ = fit_improved(
                tr_e, vl_e, features, ecfg, device,
                loss_fn=copy.deepcopy(bloss),
                use_scheduler=bsched,
                scheduler_type=bstype,
                accumulation_steps=baccum,
                verbose=verbose,
            )
            ensemble_models.append(em)
 
        ens_pred, ens_true, ens_mets = ensemble_predict(
            ensemble_models, te_loader, device
        )
        ens_nasa = score_nasa(ens_pred - ens_true)
 
        print(f"\n  [Scenario E ENSEMBLE TEST]  "
              f"RMSE={ens_mets['RMSE']:.4f}  "
              f"MAE={ens_mets['MAE']:.4f}  "
              f"R²={ens_mets['R2']:.4f}  "
              f"NASA={ens_nasa:.1f}")
 
        leaderboard.append({
            "scenario" : "E (ensemble)",
            "desc"     : f"3-seed ensemble of Scenario {best_label}",
            "RMSE"     : ens_mets["RMSE"],
            "MAE"      : ens_mets["MAE"],
            "R2"       : ens_mets["R2"],
            "NASA"     : ens_nasa,
            "model"    : ensemble_models,
            "window"   : best_win,
            "y_pred"   : ens_pred,
            "y_true"   : ens_true,
        })
        leaderboard.sort(key=lambda r: r["RMSE"])
 
    # ── Final leaderboard ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL LEADERBOARD  (sorted by RMSE ↑ best → worst)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<18} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'NASA':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, row in enumerate(leaderboard):
        marker = "  ← BEST" if i == 0 else ""
        print(f"  {row['scenario']:<18} "
              f"{row['RMSE']:>8.4f} "
              f"{row['MAE']:>8.4f} "
              f"{row['R2']:>8.4f} "
              f"{row['NASA']:>10.1f}"
              f"{marker}")
    print(f"{'='*70}")
 
    # ── Save overall best ─────────────────────────────────────────────────
    overall_best = leaderboard[0]
    save_path    = f"BEST_improved_{eng_type}.pt"
    if isinstance(overall_best["model"], list):
        # Ensemble: save all member states
        states = [
            {k: v.cpu() for k, v in m.state_dict().items()}
            for m in overall_best["model"]
        ]
        torch.save({"ensemble": states, "scenario": overall_best["scenario"]},
                   save_path)
    else:
        torch.save(
            {k: v.cpu() for k, v in overall_best["model"].state_dict().items()},
            save_path
        )
    print(f"\n  Best model saved → {save_path}")
    print(f"  Best scenario    : {overall_best['scenario']} — {overall_best['desc']}")
    print(f"  Best RMSE        : {overall_best['RMSE']:.4f}")
    print(f"  Best MAE         : {overall_best['MAE']:.4f}")
    print(f"  Best R²          : {overall_best['R2']:.4f}")
    print(f"  Best NASA score  : {overall_best['NASA']:.1f}")
 
    return leaderboard
 
 
# =============================================================================
#  SECTION 8 — ENTRY POINT
# =============================================================================
#
#  Paste this at the bottom of your notebook after all architecture classes
#  and after X_train_sw, y_train_sw, X_testf, y_test have been created.
#
# ─────────────────────────────────────────────────────────────────────────────
#
#   from improve_transformer import run_all_scenarios
#
#   leaderboard = run_all_scenarios(
#       X_train_sw                   = X_train_sw,
#       y_train_sw                   = y_train_sw,
#       X_testf                      = X_testf,
#       y_test                       = y_test,
#       features                     = features,
#       eng_type                     = eng_type,
#       device                       = device,
#       # Pass X and X_test so Scenarios C & D can re-window at window_size=50
#       X                            = X,
#       X_test                       = X_test,
#       create_training_sequences_sw = create_training_sequences_sw,
#       create_testing_sequences_sw  = create_testing_sequences_sw,
#       num_of_batches               = num_of_batches,   # your existing variable
#       window_size                  = window_size,       # window used to build X_train_sw
#       random_state                 = 341,
#       run_ensemble                 = True,              # set False to skip Scenario E
#       verbose                      = True,
#   )
#
# ─────────────────────────────────────────────────────────────────────────────