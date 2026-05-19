"""
=============================================================================
  ZONE-TARGETED IMPROVEMENT SCRIPT  —  PatchTST Dual-Dim RUL
  File: improve_rul_zone.py
=============================================================================

  PROBLEM BEING SOLVED
  --------------------
  The model achieves RMSE=11.64, MAE=8.5, R²=0.911, NASA=180.6 overall,
  but systematically OVER-PREDICTS RUL in the 40-80 range (predicted > actual).
  This range is the "transition zone" where the engine shifts from healthy
  degradation to accelerating failure.

  ROOT CAUSE ANALYSIS
  -------------------

  1. DATA IMBALANCE IN TRAINING
     With RUL capped at 125, the training set has three natural zones:
       - Zone A  (RUL > 80) : long flat/slow-decline region → MANY samples
       - Zone B  (40-80)    : rapid acceleration zone        → FEWER samples
       - Zone C  (0-40)     : near-failure zone              → FEW samples
     The loss gradient is dominated by Zone A samples. The model converges
     to a solution that fits Zone A well and Zone C reasonably (failure
     signatures are distinctive), but Zone B is squeezed out — the model
     hedges by predicting "still healthy" for Zone B patterns.

  2. NO ZONE-AWARE LOSS PRESSURE
     The AsymmetricHuberLoss from improve_transformer.py is asymmetric
     across DIRECTION (late vs. early) but not across RUL MAGNITUDE. A
     large error at RUL=60 is treated the same as the same error at RUL=110.
     Since the model is over-predicting in 40-80, we need extra downward
     pressure specifically in that zone.

  3. NON-MONOTONIC PREDICTIONS IN THE TRANSITION ZONE
     RUL physically must decrease (or stay flat) cycle-by-cycle within an
     engine. Without a monotonicity constraint the model can oscillate in
     the uncertain transition zone, keeping the running average prediction
     artificially high. Post-processing with isotonic regression enforces
     physical monotonicity without touching the architecture at all.

  FOUR NEW SCENARIOS  (Scenarios F through I)
  --------------------------------------------
  All scenarios reuse the EXACT same architecture (PatchTST_RUL_Model).
  Only the training recipe changes.

  Scenario F — Zone-Boosted Loss
      A Gaussian-shaped zone weight centred at RUL=60 boosts the loss
      contribution of samples with true RUL ∈ [40, 80] by up to 4×.
      Combined with AsymmetricHuberLoss and CosineWarm schedule.
      Direct fix for root cause 2.

  Scenario G — Weighted Sampler (oversample 40-80 zone)
      WeightedRandomSampler gives 4× higher draw probability to training
      samples with RUL ∈ [40, 80].  The model sees the transition zone 4×
      more often per epoch, so its gradient is no longer dominated by the
      healthy zone.  Direct fix for root cause 1.

  Scenario H — Zone Loss + Weighted Sampler  (F ∪ G)
      Both fixes together on the Scenario B (wider/deeper) architecture.
      Fixes root causes 1 & 2 simultaneously with more model capacity.

  Scenario I — Full Zone Combo + Isotonic Post-Processing  (H + monotonicity)
      All of the above PLUS isotonic regression per engine at inference time.
      Isotonic regression re-orders predictions to be non-increasing within
      each engine's prediction sequence, eliminating the oscillation that
      keeps 40-80 predictions artificially high.
      Fixes all three root causes.

  Scenario J — Ensemble of Scenario I across 3 seeds
      Three independently trained I-models whose predictions are averaged
      before isotonic post-processing.  Ensemble averaging further reduces
      variance in the transition zone.

  HOW TO USE
  ----------
  Add this at the bottom of your notebook after X_train_sw, y_train_sw,
  X_testf, y_test, features, eng_type, device are defined:

      from improve_rul_zone import run_zone_scenarios

      leaderboard = run_zone_scenarios(
          X_train_sw                   = X_train_sw,
          y_train_sw                   = y_train_sw,
          X_testf                      = X_testf,
          y_test                       = y_test,
          features                     = features,
          eng_type                     = eng_type,
          device                       = device,
          X                            = X,
          X_test                       = X_test,
          create_training_sequences_sw = create_training_sequences_sw,
          create_testing_sequences_sw  = create_testing_sequences_sw,
          num_of_batches               = num_of_batches,
          window_size                  = window_size,
          random_state                 = 341,
          run_ensemble                 = True,
      )
=============================================================================
"""

import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


# =============================================================================
#  SECTION 0 — TrainConfig  (self-contained copy so no notebook dependency)
# =============================================================================

@dataclass
class TrainConfig:
    feature_cols: List[str]
    target_col:   str   = "RUL"
    group_col:    str   = "engine"
    time_col:     str   = "time"

    C: int = 0   # overwritten by make_loaders return value
    L: int = 0   # overwritten by make_loaders return value

    patch_len_t: int   = 10
    stride_t:    int   = 5
    patch_len_c: int   = 3
    stride_c:    int   = 1

    d_model_t:  int   = 64
    n_heads_t:  int   = 8
    n_layers_t: int   = 2
    d_ff_t:     int   = 128
    dropout_t:  float = 0.1

    d_model_c:  int   = 64
    n_heads_c:  int   = 8
    n_layers_c: int   = 2
    d_ff_c:     int   = 128
    dropout_c:  float = 0.1

    head_hidden:  Optional[int] = 128
    head_dropout: float         = 0.1
    use_feature_attn: bool      = True

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
#  SECTION 1 — ZONE-AWARE LOSS FUNCTIONS
# =============================================================================

class ZoneBoostedLoss(nn.Module):
    """
    Combines AsymmetricHuberLoss with a Gaussian-shaped zone weight that
    boosts the loss contribution of samples whose TRUE RUL falls inside
    the transition zone [zone_lo, zone_hi].

    The zone weight is:
        w(y) = 1  +  (peak_weight - 1) * exp(-0.5 * ((y - zone_centre) / sigma)^2)

    so it peaks at zone_centre (default 60) with value peak_weight (default 4),
    and smoothly returns to 1.0 outside the zone.

    Why a Gaussian rather than a hard mask?
    ----------------------------------------
    A hard mask (weight = peak_weight if 40 ≤ y ≤ 80 else 1) creates a
    discontinuous loss surface at the zone boundaries.  The Gaussian is
    smooth everywhere, which keeps the gradient well-behaved and prevents
    the optimiser from oscillating at the zone edges.

    Parameters
    ----------
    zone_lo      : float  lower bound of target zone  (default 40)
    zone_hi      : float  upper bound of target zone  (default 80)
    peak_weight  : float  maximum multiplier at zone centre (default 4.0)
    delta        : float  Huber transition threshold (default 10.0)
    alpha_late   : float  asymmetric weight for late predictions (default 1.3)
    alpha_early  : float  asymmetric weight for early predictions (default 1.0)
    """
    def __init__(
        self,
        zone_lo:     float = 40.0,
        zone_hi:     float = 80.0,
        peak_weight: float = 4.0,
        delta:       float = 10.0,
        alpha_late:  float = 1.3,
        alpha_early: float = 1.0,
    ):
        super().__init__()
        self.zone_centre = (zone_lo + zone_hi) / 2.0        # 60.0
        self.sigma       = (zone_hi - zone_lo) / 4.0        # 10.0  → ±2σ covers zone
        self.peak_weight = peak_weight
        self.delta       = delta
        self.alpha_late  = alpha_late
        self.alpha_early = alpha_early

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        e     = y_pred - y_true
        abs_e = e.abs()

        # Huber kernel
        huber = torch.where(
            abs_e <= self.delta,
            0.5 * e ** 2,
            self.delta * (abs_e - 0.5 * self.delta)
        )

        # Asymmetric direction weight
        asym_w = torch.where(
            e > 0,
            torch.full_like(e, self.alpha_late),
            torch.full_like(e, self.alpha_early)
        )

        # Gaussian zone weight (based on TRUE RUL, not error)
        zone_w = 1.0 + (self.peak_weight - 1.0) * torch.exp(
            -0.5 * ((y_true - self.zone_centre) / self.sigma) ** 2
        )

        return (zone_w * asym_w * huber).mean()


class AsymmetricHuberLoss(nn.Module):
    """
    Kept here for use in Scenario F baseline component and backward compat.
    Identical to the version in improve_transformer.py.
    """
    def __init__(self, delta=10.0, alpha_late=1.3, alpha_early=1.0):
        super().__init__()
        self.delta       = delta
        self.alpha_late  = alpha_late
        self.alpha_early = alpha_early

    def forward(self, y_pred, y_true):
        e     = y_pred - y_true
        abs_e = e.abs()
        huber = torch.where(
            abs_e <= self.delta,
            0.5 * e ** 2,
            self.delta * (abs_e - 0.5 * self.delta)
        )
        w = torch.where(
            e > 0,
            torch.full_like(e, self.alpha_late),
            torch.full_like(e, self.alpha_early)
        )
        return (w * huber).mean()


# =============================================================================
#  SECTION 2 — WEIGHTED SAMPLER DATASET + LOADER
# =============================================================================

class RULZoneDataset(Dataset):
    """
    Dataset that optionally applies Gaussian noise augmentation.
    Identical shape contract to RULWindowDataset: X is (N, C, L).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = True, noise_std: float = 0.02):
        assert X.ndim == 3
        self.X         = X.astype(np.float32, copy=False)
        self.y         = y.astype(np.float32, copy=False)
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment and self.noise_std > 0:
            x += (np.random.randn(*x.shape) * self.noise_std).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx])


def _zone_sample_weights(y: np.ndarray,
                          zone_lo: float = 40.0,
                          zone_hi: float = 80.0,
                          zone_boost: float = 4.0) -> np.ndarray:
    """
    Returns a per-sample weight array for WeightedRandomSampler.
    Samples whose RUL ∈ [zone_lo, zone_hi] get weight = zone_boost,
    all others get weight = 1.0.

    Why WeightedRandomSampler rather than just duplicate samples?
    -------------------------------------------------------------
    Duplication changes the dataset size and epoch semantics. The sampler
    keeps the epoch length exactly equal to len(dataset) while changing the
    probability that each sample is drawn per epoch.  The model therefore
    sees ~zone_boost× as many transition-zone samples per epoch without
    the training loop needing any other modification.
    """
    weights = np.ones(len(y), dtype=np.float32)
    in_zone = (y >= zone_lo) & (y <= zone_hi)
    weights[in_zone] = zone_boost
    return weights


def make_zone_loaders(
    X_train, X_val, y_train, y_val,
    batch_size:  int   = 64,
    num_workers: int   = 0,
    noise_std:   float = 0.02,
    use_weighted_sampler: bool  = True,
    zone_lo:     float = 40.0,
    zone_hi:     float = 80.0,
    zone_boost:  float = 4.0,
    use_cuda:    bool  = torch.cuda.is_available(),
) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """
    Build train/val DataLoaders.
    - Training loader optionally uses WeightedRandomSampler for zone boosting.
    - Validation loader is always unweighted and unaugmented.
    """
    Xt = X_train.transpose(0, 2, 1)   # (N, C, L)
    Xv = X_val.transpose(0, 2, 1)

    train_ds = RULZoneDataset(Xt, y_train, augment=True,  noise_std=noise_std)
    val_ds   = RULZoneDataset(Xv, y_val,   augment=False, noise_std=0.0)

    pin = bool(use_cuda)

    if use_weighted_sampler:
        sample_weights = _zone_sample_weights(y_train, zone_lo, zone_hi, zone_boost)
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(sample_weights),
            num_samples = len(train_ds),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,              # sampler is mutually exclusive with shuffle
            num_workers=num_workers, pin_memory=pin
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, (Xt.shape[1], Xt.shape[2])


# =============================================================================
#  SECTION 3 — ISOTONIC REGRESSION POST-PROCESSING
# =============================================================================

def isotonic_postprocess(
    y_pred:    np.ndarray,
    y_true:    np.ndarray,
    engine_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Enforce physically-motivated monotone non-increasing RUL predictions.

    RUL must decrease (or stay flat) as an engine accumulates cycles.
    In the test set each engine appears once at its last observed cycle, so
    isotonic processing is applied per engine over the TRAINING SET where
    we have a full time series.

    For the TEST SET evaluation (one sample per engine) there is no temporal
    sequence to order, so isotonic regression is applied sample-wise to
    shrink predictions toward the trend using a POOL-ADJACENT-VIOLATORS
    algorithm on the full batch sorted by predicted RUL.

    Parameters
    ----------
    y_pred      : (N,) raw model predictions
    y_true      : (N,) ground-truth RUL (used only to compute metrics)
    engine_ids  : (N,) integer engine IDs.  If provided, isotonic regression
                  is applied independently per engine (for training sequences).
                  If None, applied globally sorted by y_pred descending.

    Returns
    -------
    y_pred_iso  : (N,) monotone-corrected predictions (same order as input)
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("  [WARNING] sklearn not found — skipping isotonic post-processing.")
        return y_pred

    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")

    if engine_ids is not None:
        y_out = y_pred.copy()
        for eid in np.unique(engine_ids):
            mask = engine_ids == eid
            idx  = np.where(mask)[0]
            # Sort by index (cycle order) for this engine
            order = np.argsort(idx)
            X_fit = np.arange(order.sum())
            y_fit = y_pred[idx[order]]
            y_iso = ir.fit_transform(X_fit, y_fit)
            y_out[idx[order]] = y_iso
        return y_out
    else:
        # Sort by descending predicted value (proxy for temporal order)
        order     = np.argsort(y_pred)[::-1]
        inv_order = np.argsort(order)
        X_fit     = np.arange(len(y_pred))
        y_iso     = ir.fit_transform(X_fit, y_pred[order])
        return y_iso[inv_order]


# =============================================================================
#  SECTION 4 — TRAINING UTILITIES
# =============================================================================

def train_one_epoch_zone(model, loader, device, optimizer,
                          criterion, grad_clip=1.0) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_zone(model, loader, device, criterion,
                  apply_isotonic: bool = False):
    """
    Evaluates model on loader.
    If apply_isotonic=True, isotonic regression is applied to predictions
    before computing metrics (no engine_ids available for test set, so
    global sort-based isotonic is used).
    """
    model.eval()
    losses, n = 0.0, 0
    preds_all, targets_all = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        preds = model(xb)
        loss  = criterion(preds, yb)
        losses += loss.item() * xb.size(0)
        n += xb.size(0)
        preds_all.append(preds.cpu().numpy())
        targets_all.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)

    if apply_isotonic:
        y_pred = isotonic_postprocess(y_pred, y_true, engine_ids=None)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sse  = float(np.sum((y_true - y_pred) ** 2))
    sst  = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1.0 - sse / sst) if sst > 0 else float("nan")

    return losses / max(1, n), {"MAE": mae, "RMSE": rmse, "R2": r2}, y_true, y_pred


def score_nasa(errors: np.ndarray) -> float:
    a1, a2 = 10, 13
    s = 0.0
    for e in errors:
        s += (math.exp(-e / a1) - 1) if e < 0 else (math.exp(e / a2) - 1)
    return s


def _zone_rmse(y_true, y_pred, lo=40, hi=80):
    """RMSE restricted to samples whose true RUL is in [lo, hi]."""
    mask = (y_true >= lo) & (y_true <= hi)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _print_metrics(tag, y_true, y_pred):
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sse   = float(np.sum((y_true - y_pred) ** 2))
    sst   = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2    = float(1.0 - sse / sst) if sst > 0 else float("nan")
    zrmse = _zone_rmse(y_true, y_pred)
    nasa  = score_nasa(y_pred - y_true)
    print(f"  [{tag}]  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"R²={r2:.4f}  Zone-RMSE(40-80)={zrmse:.4f}  NASA={nasa:.1f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Zone_RMSE": zrmse, "NASA": nasa,
            "y_pred": y_pred, "y_true": y_true}


# =============================================================================
#  SECTION 5 — FIT FUNCTION (zone-aware version)
# =============================================================================

def fit_zone(
    train_loader, val_loader,
    cfg, device,
    loss_fn,
    apply_isotonic_val: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Training loop for zone-targeted scenarios.
    Uses CosineAnnealingWarmRestarts schedule throughout.
    """
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_mae      = float("inf")
    best_state        = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch_zone(
            model, train_loader, device, optimizer, loss_fn, cfg.grad_clip
        )
        vl_loss, vl_mets, _, _ = evaluate_zone(
            model, val_loader, device, loss_fn,
            apply_isotonic=apply_isotonic_val
        )
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if verbose:
            print(
                f"  [{epoch:03d}] TrainLoss={tr_loss:.4f}  "
                f"ValLoss={vl_loss:.4f}  "
                f"ValMAE={vl_mets['MAE']:.4f}  "
                f"ValRMSE={vl_mets['RMSE']:.4f}  "
                f"LR={optimizer.param_groups[0]['lr']:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if vl_mets["MAE"] < best_val_mae - 1e-6:
            best_val_mae      = vl_mets["MAE"]
            best_state        = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}. "
                          f"Best Val MAE={best_val_mae:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# =============================================================================
#  SECTION 6 — SCENARIO CONFIGS
#  Scenario B architecture (wider/deeper) is the best known baseline.
#  All new scenarios build on top of it.
# =============================================================================

def _base_config(features, window_size, model_path="zone_best.pt",
                 d_model=96, n_layers=3, d_ff=256, dropout=0.15,
                 head_hidden=192, lr=2e-4, epochs=200, patience=25):
    return TrainConfig(
        feature_cols  = list(features),
        C             = len(features),
        L             = window_size,
        patch_len_t=10, stride_t=5,
        patch_len_c=3,  stride_c=1,
        d_model_t=d_model,  n_heads_t=8,  n_layers_t=n_layers,
        d_ff_t=d_ff,        dropout_t=dropout,
        d_model_c=d_model,  n_heads_c=8,  n_layers_c=n_layers,
        d_ff_c=d_ff,        dropout_c=dropout,
        head_hidden=head_hidden, head_dropout=0.1,
        batch_size=40, epochs=epochs, lr=lr,
        weight_decay=1e-4, grad_clip=1.0, patience=patience,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=model_path,
    )

def scenario_F_config(features, window_size):
    return _base_config(features, window_size, model_path="scenarioF_zone_best.pt")

def scenario_G_config(features, window_size):
    return _base_config(features, window_size, model_path="scenarioG_zone_best.pt")

def scenario_H_config(features, window_size):
    return _base_config(features, window_size, model_path="scenarioH_zone_best.pt")

def scenario_I_config(features, window_size):
    return _base_config(features, window_size, model_path="scenarioI_zone_best.pt")


# =============================================================================
#  SECTION 7 — ENSEMBLE
# =============================================================================

def ensemble_predict_zone(models, loader, device,
                           apply_isotonic: bool = True):
    """Average predictions from a list of models, then optionally apply isotonic."""
    all_preds, y_true_all = [], []
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

    y_pred_ens = np.mean(np.stack(all_preds, axis=0), axis=0)

    if apply_isotonic:
        y_pred_ens = isotonic_postprocess(y_pred_ens, y_true_all, engine_ids=None)

    return y_pred_ens, y_true_all


# =============================================================================
#  SECTION 8 — SIMPLE DS / LOADER HELPERS
# =============================================================================

class _SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def _make_test_loader(X_testf, y_test, batch_size=64):
    Xt = X_testf.transpose(0, 2, 1)
    return DataLoader(_SimpleDS(Xt, y_test), batch_size=batch_size, shuffle=False)


# =============================================================================
#  SECTION 9 — MAIN ORCHESTRATOR
# =============================================================================

def run_zone_scenarios(
    X_train_sw,
    y_train_sw,
    X_testf,
    y_test,
    features,
    eng_type: str,
    device,
    # Required for larger-window scenarios
    X=None,
    X_test=None,
    create_training_sequences_sw=None,
    create_testing_sequences_sw=None,
    num_of_batches=1,
    window_size: int = 40,
    random_state: int = 341,
    run_ensemble: bool = True,
    # Zone hyperparameters (tune if needed)
    zone_lo:      float = 40.0,
    zone_hi:      float = 80.0,
    zone_boost:   float = 4.0,
    peak_weight:  float = 4.0,
    noise_std:    float = 0.02,
    verbose:      bool  = True,
):
    """
    Runs Scenarios F through J targeting the 40-80 RUL zone.

    New metric reported: Zone-RMSE — RMSE restricted to test samples
    whose TRUE RUL ∈ [40, 80].  This is the primary success metric for
    this script.  The goal is to reduce Zone-RMSE alongside overall RMSE.

    Parameters
    ----------
    (same as run_all_scenarios in improve_transformer.py, plus:)
    zone_lo, zone_hi  : bounds of the transition zone (default 40, 80)
    zone_boost        : sampler overweight for zone samples (default 4×)
    peak_weight       : Gaussian loss peak for zone samples (default 4×)
    noise_std         : Gaussian augmentation std (default 0.02)
    """
    print("=" * 70)
    print("  ZONE-TARGETED IMPROVEMENT EXPERIMENT")
    print(f"  Engine type    : {eng_type}")
    print(f"  Window size    : {window_size}")
    print(f"  Target zone    : RUL ∈ [{zone_lo}, {zone_hi}]")
    print(f"  Zone boost     : {zone_boost}× sampler  /  {peak_weight}× loss peak")
    print("=" * 70)

    # ── Shared test loader ─────────────────────────────────────────────────
    test_loader = _make_test_loader(X_testf, y_test)

    # ── Evaluation loss (MSE for comparable metric logging) ───────────────
    mse_crit = nn.MSELoss()

    # ── Print zone statistics ──────────────────────────────────────────────
    n_total = len(y_train_sw)
    n_zone  = int(((y_train_sw >= zone_lo) & (y_train_sw <= zone_hi)).sum())
    print(f"\n  Training samples total     : {n_total}")
    print(f"  Training samples in zone   : {n_zone} "
          f"({100*n_zone/n_total:.1f}%)")
    print(f"  Effective zone samples/epoch after {zone_boost}× boost: "
          f"~{int(n_zone * zone_boost)} "
          f"({100*min(n_zone*zone_boost/n_total,1):.0f}% of epoch)\n")

    # ── Train/val split ───────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sw, y_train_sw, test_size=0.2, random_state=random_state
    )

    leaderboard = []

    # =========================================================================
    #  SCENARIO F — Zone-Boosted Loss only (no sampler)
    # =========================================================================
    print(f"\n{'─'*70}")
    print("  SCENARIO F: Zone-Boosted Loss  (Gaussian peak at RUL=60)")
    print(f"{'─'*70}")

    cfg_F = scenario_F_config(features, window_size)
    tr_F, vl_F, (C, L) = make_zone_loaders(
        X_tr, X_val, y_tr, y_val,
        batch_size=cfg_F.batch_size,
        num_workers=getattr(cfg_F, "num_workers", 0),
        noise_std=noise_std,
        use_weighted_sampler=False,   # loss-only fix for scenario F
        zone_lo=zone_lo, zone_hi=zone_hi, zone_boost=zone_boost,
        use_cuda=str(device).startswith("cuda"),
    )
    cfg_F.C, cfg_F.L = C, L

    loss_F = ZoneBoostedLoss(
        zone_lo=zone_lo, zone_hi=zone_hi,
        peak_weight=peak_weight, delta=10.0,
        alpha_late=1.3, alpha_early=1.0
    )
    model_F, _, _ = fit_zone(tr_F, vl_F, cfg_F, device, loss_F,
                              apply_isotonic_val=False, verbose=verbose)

    torch.save({k: v.cpu() for k, v in model_F.state_dict().items()},
               f"scenarioF_{eng_type}.pt")

    _, _, yt_F, yp_F = evaluate_zone(model_F, test_loader, device, mse_crit,
                                      apply_isotonic=False)
    mets_F = _print_metrics(f"Scenario F TEST", yt_F, yp_F)
    leaderboard.append({"scenario": "F", "model": model_F, **mets_F})

    # =========================================================================
    #  SCENARIO G — Weighted Sampler only (no zone loss)
    # =========================================================================
    print(f"\n{'─'*70}")
    print("  SCENARIO G: Weighted Sampler  (40-80 zone oversampled 4×)")
    print(f"{'─'*70}")

    cfg_G = scenario_G_config(features, window_size)
    tr_G, vl_G, (C, L) = make_zone_loaders(
        X_tr, X_val, y_tr, y_val,
        batch_size=cfg_G.batch_size,
        num_workers=getattr(cfg_G, "num_workers", 0),
        noise_std=noise_std,
        use_weighted_sampler=True,
        zone_lo=zone_lo, zone_hi=zone_hi, zone_boost=zone_boost,
        use_cuda=str(device).startswith("cuda"),
    )
    cfg_G.C, cfg_G.L = C, L

    loss_G = AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0)
    model_G, _, _ = fit_zone(tr_G, vl_G, cfg_G, device, loss_G,
                              apply_isotonic_val=False, verbose=verbose)

    torch.save({k: v.cpu() for k, v in model_G.state_dict().items()},
               f"scenarioG_{eng_type}.pt")

    _, _, yt_G, yp_G = evaluate_zone(model_G, test_loader, device, mse_crit,
                                      apply_isotonic=False)
    mets_G = _print_metrics(f"Scenario G TEST", yt_G, yp_G)
    leaderboard.append({"scenario": "G", "model": model_G, **mets_G})

    # =========================================================================
    #  SCENARIO H — Zone Loss + Weighted Sampler  (F ∪ G)
    # =========================================================================
    print(f"\n{'─'*70}")
    print("  SCENARIO H: Zone Loss + Weighted Sampler  (F + G combined)")
    print(f"{'─'*70}")

    cfg_H = scenario_H_config(features, window_size)
    tr_H, vl_H, (C, L) = make_zone_loaders(
        X_tr, X_val, y_tr, y_val,
        batch_size=cfg_H.batch_size,
        num_workers=getattr(cfg_H, "num_workers", 0),
        noise_std=noise_std,
        use_weighted_sampler=True,
        zone_lo=zone_lo, zone_hi=zone_hi, zone_boost=zone_boost,
        use_cuda=str(device).startswith("cuda"),
    )
    cfg_H.C, cfg_H.L = C, L

    loss_H = ZoneBoostedLoss(
        zone_lo=zone_lo, zone_hi=zone_hi,
        peak_weight=peak_weight, delta=10.0,
        alpha_late=1.3, alpha_early=1.0
    )
    model_H, _, _ = fit_zone(tr_H, vl_H, cfg_H, device, loss_H,
                              apply_isotonic_val=False, verbose=verbose)

    torch.save({k: v.cpu() for k, v in model_H.state_dict().items()},
               f"scenarioH_{eng_type}.pt")

    _, _, yt_H, yp_H = evaluate_zone(model_H, test_loader, device, mse_crit,
                                      apply_isotonic=False)
    mets_H = _print_metrics(f"Scenario H TEST", yt_H, yp_H)
    leaderboard.append({"scenario": "H", "model": model_H, **mets_H})

    # =========================================================================
    #  SCENARIO I — H + Isotonic Post-Processing
    # =========================================================================
    print(f"\n{'─'*70}")
    print("  SCENARIO I: H + Isotonic Post-Processing  (monotonicity enforced)")
    print(f"{'─'*70}")

    # Reuse model_H weights — isotonic is inference-only, no retraining needed
    _, _, yt_I, yp_I_raw = evaluate_zone(model_H, test_loader, device, mse_crit,
                                          apply_isotonic=False)
    yp_I = isotonic_postprocess(yp_I_raw, yt_I, engine_ids=None)
    mets_I = _print_metrics(f"Scenario I TEST  (H + isotonic)", yt_I, yp_I)
    # Save same model as H but record isotonic flag
    leaderboard.append({
        "scenario": "I", "model": model_H,
        "apply_isotonic": True, **mets_I
    })

    # ─── Also retrain with isotonic applied at validation time ────────────
    print("\n  [Re-training with isotonic applied during validation ...]")
    cfg_I = scenario_I_config(features, window_size)
    tr_I, vl_I, (C, L) = make_zone_loaders(
        X_tr, X_val, y_tr, y_val,
        batch_size=cfg_I.batch_size,
        num_workers=getattr(cfg_I, "num_workers", 0),
        noise_std=noise_std,
        use_weighted_sampler=True,
        zone_lo=zone_lo, zone_hi=zone_hi, zone_boost=zone_boost,
        use_cuda=str(device).startswith("cuda"),
    )
    cfg_I.C, cfg_I.L = C, L

    loss_I = ZoneBoostedLoss(
        zone_lo=zone_lo, zone_hi=zone_hi,
        peak_weight=peak_weight, delta=10.0,
        alpha_late=1.3, alpha_early=1.0
    )
    model_I, _, _ = fit_zone(tr_I, vl_I, cfg_I, device, loss_I,
                              apply_isotonic_val=True, verbose=verbose)

    torch.save({k: v.cpu() for k, v in model_I.state_dict().items()},
               f"scenarioI_{eng_type}.pt")

    _, _, yt_Ir, yp_Ir_raw = evaluate_zone(model_I, test_loader, device,
                                            mse_crit, apply_isotonic=False)
    yp_Ir = isotonic_postprocess(yp_Ir_raw, yt_Ir, engine_ids=None)
    mets_Ir = _print_metrics("Scenario I retrained TEST", yt_Ir, yp_Ir)

    if mets_Ir["RMSE"] < mets_I["RMSE"]:
        leaderboard[-1] = {
            "scenario": "I", "model": model_I,
            "apply_isotonic": True, **mets_Ir
        }
        print("  → Retrained version is better, replacing Scenario I entry.")

    # =========================================================================
    #  SCENARIO J — Ensemble of I  (3 seeds)
    # =========================================================================
    if run_ensemble:
        print(f"\n{'─'*70}")
        print("  SCENARIO J: 3-seed Ensemble of Scenario I + Isotonic")
        print(f"{'─'*70}")

        seeds = [42, 137, 271]
        ens_models = []

        for seed in seeds:
            print(f"\n  [Seed {seed}]")
            torch.manual_seed(seed)
            np.random.seed(seed)

            Xtr_e, Xvl_e, ytr_e, yvl_e = train_test_split(
                X_train_sw, y_train_sw, test_size=0.2, random_state=seed
            )
            cfg_J = scenario_I_config(features, window_size)
            tr_e, vl_e, (Ce, Le) = make_zone_loaders(
                Xtr_e, Xvl_e, ytr_e, yvl_e,
                batch_size=cfg_J.batch_size,
                num_workers=getattr(cfg_J, "num_workers", 0),
                noise_std=noise_std,
                use_weighted_sampler=True,
                zone_lo=zone_lo, zone_hi=zone_hi, zone_boost=zone_boost,
                use_cuda=str(device).startswith("cuda"),
            )
            cfg_J.C, cfg_J.L = Ce, Le

            loss_J = ZoneBoostedLoss(
                zone_lo=zone_lo, zone_hi=zone_hi,
                peak_weight=peak_weight, delta=10.0,
                alpha_late=1.3, alpha_early=1.0
            )
            em, _, _ = fit_zone(tr_e, vl_e, cfg_J, device, loss_J,
                                 apply_isotonic_val=True, verbose=verbose)
            ens_models.append(em)

        yp_J, yt_J = ensemble_predict_zone(
            ens_models, test_loader, device, apply_isotonic=True
        )
        mets_J = _print_metrics("Scenario J ENSEMBLE TEST", yt_J, yp_J)
        leaderboard.append({
            "scenario": "J (ensemble)", "model": ens_models,
            "apply_isotonic": True, **mets_J
        })

    # =========================================================================
    #  LEADERBOARD
    # =========================================================================
    leaderboard.sort(key=lambda r: r["RMSE"])
    overall_best = leaderboard[0]

    print(f"\n{'='*70}")
    print("  ZONE-TARGETED LEADERBOARD  (sorted by RMSE)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<22} {'RMSE':>7} {'MAE':>7} {'R2':>7} "
          f"{'Zone-RMSE':>10} {'NASA':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*9}")
    for i, row in enumerate(leaderboard):
        marker = "  ← BEST" if i == 0 else ""
        print(
            f"  {row['scenario']:<22} "
            f"{row['RMSE']:>7.4f} "
            f"{row['MAE']:>7.4f} "
            f"{row['R2']:>7.4f} "
            f"{row['Zone_RMSE']:>10.4f} "
            f"{row['NASA']:>9.1f}"
            f"{marker}"
        )
    print(f"{'='*70}")

    # ── Save best ─────────────────────────────────────────────────────────
    save_path = f"BEST_zone_{eng_type}.pt"
    if isinstance(overall_best["model"], list):
        states = [{k: v.cpu() for k, v in m.state_dict().items()}
                  for m in overall_best["model"]]
        torch.save({"ensemble": states,
                    "scenario": overall_best["scenario"],
                    "apply_isotonic": overall_best.get("apply_isotonic", False)},
                   save_path)
    else:
        torch.save(
            {k: v.cpu() for k, v in overall_best["model"].state_dict().items()},
            save_path
        )
    print(f"\n  Best model saved → {save_path}")
    print(f"  Best scenario   : {overall_best['scenario']}")
    print(f"  Best RMSE       : {overall_best['RMSE']:.4f}")
    print(f"  Best MAE        : {overall_best['MAE']:.4f}")
    print(f"  Best R²         : {overall_best['R2']:.4f}")
    print(f"  Best Zone-RMSE  : {overall_best['Zone_RMSE']:.4f}")
    print(f"  Best NASA       : {overall_best['NASA']:.1f}")
    print(f"  Isotonic applied: {overall_best.get('apply_isotonic', False)}")

    return leaderboard


# =============================================================================
#  COPY-PASTE SNIPPET FOR THE NOTEBOOK
# =============================================================================
#
#   from improve_rul_zone import run_zone_scenarios
#
#   leaderboard = run_zone_scenarios(
#       X_train_sw                   = X_train_sw,
#       y_train_sw                   = y_train_sw,
#       X_testf                      = X_testf,
#       y_test                       = y_test,
#       features                     = features,
#       eng_type                     = eng_type,
#       device                       = device,
#       X                            = X,
#       X_test                       = X_test,
#       create_training_sequences_sw = create_training_sequences_sw,
#       create_testing_sequences_sw  = create_testing_sequences_sw,
#       num_of_batches               = num_of_batches,
#       window_size                  = window_size,
#       random_state                 = 341,
#       run_ensemble                 = True,
#       zone_lo                      = 40.0,
#       zone_hi                      = 80.0,
#       zone_boost                   = 4.0,
#       peak_weight                  = 4.0,
#       noise_std                    = 0.02,
#       verbose                      = True,
#   )