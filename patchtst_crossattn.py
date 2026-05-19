"""
=============================================================================
  CROSS-ATTENTION FUSION  —  PatchTST Dual-Encoder RUL Transformer
  File: patchtst_crossattn.py
=============================================================================

  WHY CROSS-ATTENTION INSTEAD OF SIMPLE FUSION
  ---------------------------------------------
  The original FusionHead concatenates temporal and channel embeddings,
  applies BatchNorm, mean-pools, and passes through an MLP.  This is fast
  but has a fundamental limitation: every temporal token contributes equally
  to every channel token in the pooled representation.  There is no
  mechanism for the temporal encoder to selectively query the channel
  encoder for *only* the sensor-correlation patterns that are most relevant
  to its current degradation estimate.

  Cross-attention solves this with two directed information flows:

    Flow 1 (T→C):  Temporal tokens act as QUERIES.
                   Channel tokens act as KEYS and VALUES.
                   Each temporal token (a patch of time) asks:
                   "Which sensor-correlation patterns are most relevant
                    to what I've seen in my time window?"
                   This produces context-enriched temporal tokens.

    Flow 2 (C→T):  Channel tokens act as QUERIES.
                   Temporal tokens act as KEYS and VALUES.
                   Each channel token (a sensor-patch interaction) asks:
                   "Which temporal patterns are most informative about
                    the current multi-sensor correlation?"
                   This produces context-enriched channel tokens.

  Both attended outputs are mean-pooled, concatenated, and projected through
  an MLP to produce the final RUL scalar.

  The net effect is that the fusion is *adaptive*: during degradation, the
  temporal encoder can attend more strongly to sensor channels that are
  diverging from healthy baselines, rather than treating all channel
  patterns equally.

  WHAT THIS FILE PROVIDES
  -----------------------
  1.  CrossAttentionBlock    — single-direction cross-attention (Q from one
                               encoder, K/V from the other) with residual,
                               LayerNorm, and an optional FFN.

  2.  CrossAttentionFusionHead — bidirectional cross-attention fusion.
                               Replaces FusionHead in PatchTST_RUL_Model.

  3.  PatchTST_CrossAttn_Model — identical to PatchTST_RUL_Model except it
                               uses CrossAttentionFusionHead.  The temporal
                               encoder and channel encoder are unchanged.

  4.  fit_crossattn          — mirrors fit_improved() from improve_transformer.py
                               but instantiates PatchTST_CrossAttn_Model.

  5.  run_crossattn_scenarios — mirrors run_all_scenarios() from
                               improve_transformer.py, running the same five
                               scenarios (A–E) with the cross-attention model.
                               All losses, schedulers, augmentation, and
                               ensemble logic are reused from
                               improve_transformer.py.

  HOW TO USE
  ----------
  Place this file in the same directory as improve_transformer.py.
  Then add to your notebook after X_train_sw, y_train_sw, X_testf,
  y_test, features, eng_type, device are ready:

      from patchtst_crossattn import run_crossattn_scenarios

      leaderboard = run_crossattn_scenarios(
          X_train_sw                   = X_train_sw,
          y_train_sw                   = y_train_sw,
          X_testf                      = X_testf,
          y_test                       = y_test,
          features                     = features,
          eng_type                     = eng_type,
          device                       = device,
          X                            = X,           # needed for Scenarios C & D
          X_test                       = X_test,
          create_training_sequences_sw = create_training_sequences_sw,
          create_testing_sequences_sw  = create_testing_sequences_sw,
          num_of_batches               = num_of_batches,
          window_size                  = window_size,
          random_state                 = 341,
          run_ensemble                 = True,
          verbose                      = True,
      )

  COMPARING FUSION STRATEGIES
  ---------------------------
  To directly compare FusionHead vs CrossAttentionFusionHead, run both:

      from improve_transformer    import run_all_scenarios
      from patchtst_crossattn     import run_crossattn_scenarios

      lb_orig   = run_all_scenarios(...)      # original concat fusion
      lb_xattn  = run_crossattn_scenarios(...)  # cross-attention fusion

  The leaderboard format is identical so results can be compared directly.
=============================================================================
"""

import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
from Encoder_Layers import *
from CommonFunctions import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Dict

# ── Import everything reusable from improve_transformer.py ────────────────
# This includes: TrainConfig, AsymmetricHuberLoss, WeightedMSELoss,
# AugmentedRULDataset, make_loaders_augmented, train_one_epoch_improved,
# evaluate_improved, score_nasa, scenario_A/B/C/D_config, ensemble_predict.
from improve_transformer import (
    TrainConfig,
    AsymmetricHuberLoss,
    WeightedMSELoss,
    AugmentedRULDataset,
    make_loaders_augmented,
    train_one_epoch_improved,
    evaluate_improved,
    score_nasa,
    scenario_A_config,
    scenario_B_config,
    scenario_C_config,
    scenario_D_config,
    ensemble_predict,
)

# ── Architecture classes must be available in the calling notebook ─────────
# PatchTSTEncoder, SensorChannelTransformerEncoder, and all their
# sub-classes (PatchEmbedding, SinusoidalPositionEncoding,
# TransformerBatchNormEncoderLayer, ProbSparseMultiheadAttention) are
# defined in your notebook.  This file references them by name so they
# must already be importable or defined in the kernel before this module
# is used.  No circular import is created.

# %%

class supDataset(Dataset):
  def __init__(self, data_list, targets):
    self.data_list = data_list
    self.targets = targets

  # Returns len of dataset
  def __len__(self):
    return len(self.data_list)

  # Takes indices of data len, returns a dictionary of tensors
  def __getitem__(self, idx):
    X = self.data_list[idx]
    y = self.targets[idx]
    # return X, y
    # return torch.tensor(X, dtype=torch.float),  torch.tensor(y, dtype=torch.int64)
    return torch.tensor(X, dtype=torch.float), y

 
# %%
# Training function
import joblib
# Create device object to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0
# device = torch.device("cpu")  # If only using CPU
print(device)


# %%
# -----------------------------
# PatchTST blocks
# -----------------------------
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader


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


# =============================================================================
#  SECTION 1 — CROSS-ATTENTION BLOCK
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """
    Single-direction cross-attention layer.

    Given:
      query_seq  : (B, N_q, d_q)   — the sequence that formulates queries
      context_seq: (B, N_k, d_k)   — the sequence that provides keys and values

    The block:
      1. Projects query, key, value to a common d_model dimension.
      2. Applies scaled dot-product multi-head attention.
      3. Adds a residual connection from the query sequence (projected to
         d_model if dimensions differ).
      4. Applies LayerNorm.
      5. Optionally applies a position-wise FFN with another residual + norm.

    Why LayerNorm instead of BatchNorm here?
    ----------------------------------------
    The fusion head operates on sequences of variable effective length
    (N_q and N_k depend on the window size and patch parameters).
    BatchNorm normalises across the batch dimension, which is problematic
    for variable-length sequences at inference with small batch sizes.
    LayerNorm normalises across the feature dimension independently per
    sample, which is always well-defined regardless of batch size.

    Parameters
    ----------
    d_query   : int   dimension of query sequence features
    d_context : int   dimension of context (key/value) sequence features
    d_model   : int   common projection dimension for attention
    n_heads   : int   number of attention heads (must divide d_model)
    d_ff      : int   inner dimension of optional FFN (0 = skip FFN)
    dropout   : float dropout rate
    """
    def __init__(
        self,
        d_query:   int,
        d_context: int,
        d_model:   int,
        n_heads:   int  = 8,
        d_ff:      int  = 256,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model

        # Project query, key, value to d_model
        self.q_proj = nn.Linear(d_query,   d_model)
        self.k_proj = nn.Linear(d_context, d_model)
        self.v_proj = nn.Linear(d_context, d_model)

        # Multi-head attention (standard scaled dot-product)
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # Residual projection: map input query to d_model if shapes differ
        self.res_proj = (
            nn.Linear(d_query, d_model)
            if d_query != d_model
            else nn.Identity()
        )

        # Post-attention norm
        self.norm1 = nn.LayerNorm(d_model)

        # Optional position-wise FFN
        self.use_ffn = (d_ff > 0)
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query_seq:   torch.Tensor,   # (B, N_q, d_query)
        context_seq: torch.Tensor,   # (B, N_k, d_context)
    ) -> torch.Tensor:               # (B, N_q, d_model)
        """
        Compute cross-attention: query asks questions about context.
        Returns attended query tokens, same length N_q as input.
        """
        Q = self.q_proj(query_seq)    # (B, N_q, d_model)
        K = self.k_proj(context_seq)  # (B, N_k, d_model)
        V = self.v_proj(context_seq)  # (B, N_k, d_model)

        # Scaled dot-product cross-attention
        # attn_out: (B, N_q, d_model)
        attn_out, _ = self.attn(Q, K, V)
        attn_out = self.attn_drop(attn_out)

        # Residual + LayerNorm
        res   = self.res_proj(query_seq)   # (B, N_q, d_model)
        out   = self.norm1(res + attn_out) # (B, N_q, d_model)

        # Optional FFN with residual
        if self.use_ffn:
            out = self.norm2(out + self.ffn(out))

        return out


# =============================================================================
#  SECTION 2 — BIDIRECTIONAL CROSS-ATTENTION FUSION HEAD
# =============================================================================

class CrossAttentionFusionHead(nn.Module):
    """
    Bidirectional cross-attention fusion that replaces FusionHead.

    Two complementary attention flows are computed:

      ① T→C  (Temporal queries Context):
             Each temporal patch token attends over ALL channel tokens.
             The temporal encoder asks: "Given my time window, which
             sensor-correlation patterns should I focus on?"
             Output: context-enriched temporal tokens  (B, N_t, d_model)

      ② C→T  (Channel queries Temporal):
             Each channel patch token attends over ALL temporal tokens.
             The channel encoder asks: "Given the sensor interactions I've
             detected, which temporal patterns are most informative?"
             Output: context-enriched channel tokens   (B, N_c, d_model)

    Both attended sequences are mean-pooled → (B, d_model) each.
    They are concatenated → (B, 2 * d_model), then projected through
    an MLP head to produce the final RUL scalar (B,).

    Bidirectional vs unidirectional
    --------------------------------
    Using only T→C (temporal queries channel) enriches temporal tokens but
    leaves the channel representation unchanged.  Using C→T as well ensures
    the channel side also benefits from temporal context.  The two pooled
    vectors are then complementary: the T→C pool captures temporally-grounded
    sensor context, and the C→T pool captures sensor-grounded temporal context.

    Parameters
    ----------
    d_model_t  : int   temporal encoder output dimension
    d_model_c  : int   channel encoder output dimension
    d_model    : int   shared cross-attention projection dimension
                       (defaults to max(d_model_t, d_model_c))
    n_heads    : int   attention heads (must divide d_model)
    d_ff_xattn : int   FFN hidden dim inside each CrossAttentionBlock
    head_hidden: int   MLP head hidden dim (None → d_model)
    dropout    : float dropout for attention and FFN layers
    """
    def __init__(
        self,
        d_model_t:   int,
        d_model_c:   int,
        d_model:     Optional[int] = None,
        n_heads:     int  = 8,
        d_ff_xattn:  int  = 256,
        head_hidden: Optional[int] = None,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Use max of the two encoder dims as the common projection width
        # if the caller did not specify explicitly.
        if d_model is None:
            d_model = max(d_model_t, d_model_c)

        # Make d_model divisible by n_heads (round up)
        if d_model % n_heads != 0:
            d_model = math.ceil(d_model / n_heads) * n_heads

        self.d_model = d_model

        # ── Flow ①: Temporal queries Channel ─────────────────────────────
        self.t_to_c = CrossAttentionBlock(
            d_query   = d_model_t,
            d_context = d_model_c,
            d_model   = d_model,
            n_heads   = n_heads,
            d_ff      = d_ff_xattn,
            dropout   = dropout,
        )

        # ── Flow ②: Channel queries Temporal ─────────────────────────────
        self.c_to_t = CrossAttentionBlock(
            d_query   = d_model_c,
            d_context = d_model_t,
            d_model   = d_model,
            n_heads   = n_heads,
            d_ff      = d_ff_xattn,
            dropout   = dropout,
        )

        # ── MLP regression head ───────────────────────────────────────────
        # Input: concatenation of the two mean-pooled attended vectors
        #        → size is 2 * d_model
        mlp_in   = 2 * d_model
        mlp_hid  = head_hidden if head_hidden is not None else d_model

        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, mlp_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hid, 1),
        )

    def forward(
        self,
        temporal_out: torch.Tensor,   # (B, N_t, d_model_t)
        channel_out:  torch.Tensor,   # (B, N_c, d_model_c)
    ) -> torch.Tensor:                # (B, 1)
        """
        Fuse temporal and channel embeddings via bidirectional cross-attention.
        """
        # Flow ①: each temporal token attends over all channel tokens
        t_enriched = self.t_to_c(temporal_out, channel_out)   # (B, N_t, d_model)

        # Flow ②: each channel token attends over all temporal tokens
        c_enriched = self.c_to_t(channel_out, temporal_out)   # (B, N_c, d_model)

        # Mean-pool over token sequence dimension
        t_pooled = t_enriched.mean(dim=1)   # (B, d_model)
        c_pooled = c_enriched.mean(dim=1)   # (B, d_model)

        # Concatenate and project to RUL scalar
        fused = torch.cat([t_pooled, c_pooled], dim=-1)   # (B, 2*d_model)
        return self.mlp(fused)                             # (B, 1)


# =============================================================================
#  SECTION 3 — CROSS-ATTENTION RUL MODEL
# =============================================================================

class PatchTST_CrossAttn_Model(nn.Module):
    """
    PatchTST Dual-Encoder RUL model with CrossAttentionFusionHead.

    The temporal encoder (PatchTSTEncoder) and sensor-channel encoder
    (SensorChannelTransformerEncoder) are architecturally identical to
    PatchTST_RUL_Model.  Only the fusion module is replaced.

    Extra constructor parameters vs PatchTST_RUL_Model
    ---------------------------------------------------
    d_model_xattn : int   projection dimension inside CrossAttentionBlock.
                          Defaults to max(d_model_t, d_model_c).
    n_heads_xattn : int   attention heads in the cross-attention blocks.
                          Defaults to n_heads_t.
    d_ff_xattn    : int   FFN hidden dim inside CrossAttentionBlock.
                          Defaults to d_ff_t.
    """
    def __init__(
        self,
        C, L,
        d_model_t:     int,
        n_heads_t:     int,
        n_layers_t:    int,
        d_ff_t:        int,
        dropout_t:     float,
        patch_len_t:   int,
        stride_t:      int,
        patch_len_c:   int,
        stride_c:      int,
        d_model_c:     int,
        n_heads_c:     int,
        n_layers_c:    int,
        d_ff_c:        int,
        dropout_c:     float,
        head_hidden:   Optional[int] = None,
        # Cross-attention fusion parameters
        d_model_xattn: Optional[int] = None,
        n_heads_xattn: Optional[int] = None,
        d_ff_xattn:    Optional[int] = None,
        dropout_xattn: float = 0.1,
        use_bn_temporal: bool = True,
        use_bn_channel:  bool = True,
    ):
        super().__init__()

        # ── Temporal encoder (unchanged) ──────────────────────────────────
        self.temporal_encoder = PatchTSTEncoder(
            d_model  = d_model_t,
            n_heads  = n_heads_t,
            n_layers = n_layers_t,
            d_ff     = d_ff_t,
            dropout  = dropout_t,
            patch_len= patch_len_t,
            stride   = stride_t,
            use_batchnorm_out = use_bn_temporal,
        )

        # ── Sensor-channel encoder (unchanged) ───────────────────────────
        self.sensor_encoder = SensorChannelTransformerEncoder(
            C              = C,
            L              = L,
            patch_len      = patch_len_c,
            stride         = stride_c,
            d_model        = d_model_c,
            n_heads        = n_heads_c,
            num_layers     = n_layers_c,
            dim_feedforward= d_ff_c,
            dropout        = dropout_c,
            use_batchnorm_out = use_bn_channel,
        )

        # ── Resolve cross-attention hyperparameters ───────────────────────
        _d_xattn  = d_model_xattn if d_model_xattn is not None \
                    else max(d_model_t, d_model_c)
        _nh_xattn = n_heads_xattn if n_heads_xattn is not None \
                    else n_heads_t
        _ff_xattn = d_ff_xattn    if d_ff_xattn    is not None \
                    else d_ff_t

        # ── Cross-attention fusion head ───────────────────────────────────
        self.fusion_head = CrossAttentionFusionHead(
            d_model_t   = d_model_t,
            d_model_c   = d_model_c,
            d_model     = _d_xattn,
            n_heads     = _nh_xattn,
            d_ff_xattn  = _ff_xattn,
            head_hidden = head_hidden,
            dropout     = dropout_xattn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, C, L)
        te = self.temporal_encoder(x)    # (B, N_t, d_model_t)
        se = self.sensor_encoder(x)      # (B, N_c, d_model_c)
        y  = self.fusion_head(te, se)    # (B, 1)
        return y.squeeze(-1)             # (B,)


# =============================================================================
#  SECTION 4 — FIT FUNCTION (cross-attention variant)
# =============================================================================

def fit_crossattn(
    train_loader,
    val_loader,
    features,
    cfg: TrainConfig,
    device,
    loss_fn=None,
    use_scheduler:     bool  = True,
    scheduler_type:    str   = "cosine_warm",
    accumulation_steps: int  = 1,
    verbose:           bool  = True,
    # Cross-attention specific (None → derived from cfg)
    d_model_xattn: Optional[int]   = None,
    n_heads_xattn: Optional[int]   = None,
    d_ff_xattn:    Optional[int]   = None,
    dropout_xattn: float           = 0.1,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Mirrors fit_improved() from improve_transformer.py but builds a
    PatchTST_CrossAttn_Model instead of PatchTST_RUL_Model.

    All training utilities (loss, scheduler, gradient clipping, early
    stopping) are identical to improve_transformer.py.

    Parameters
    ----------
    d_model_xattn : projection dimension inside the cross-attention blocks.
                    Defaults to max(cfg.d_model_t, cfg.d_model_c).
    n_heads_xattn : number of cross-attention heads.
                    Defaults to cfg.n_heads_t.
    d_ff_xattn    : FFN hidden dimension inside cross-attention blocks.
                    Defaults to cfg.d_ff_t.
    dropout_xattn : dropout rate for cross-attention layers.
    """
    # ── Build cross-attention model ───────────────────────────────────────
    model = PatchTST_CrossAttn_Model(
        C              = cfg.C,
        L              = cfg.L,
        d_model_t      = cfg.d_model_t,
        n_heads_t      = cfg.n_heads_t,
        n_layers_t     = cfg.n_layers_t,
        d_ff_t         = cfg.d_ff_t,
        dropout_t      = cfg.dropout_t,
        patch_len_t    = cfg.patch_len_t,
        stride_t       = cfg.stride_t,
        patch_len_c    = cfg.patch_len_c,
        stride_c       = cfg.stride_c,
        d_model_c      = cfg.d_model_c,
        n_heads_c      = cfg.n_heads_c,
        n_layers_c     = cfg.n_layers_c,
        d_ff_c         = cfg.d_ff_c,
        dropout_c      = cfg.dropout_c,
        head_hidden    = cfg.head_hidden,
        d_model_xattn  = d_model_xattn,
        n_heads_xattn  = n_heads_xattn,
        d_ff_xattn     = d_ff_xattn,
        dropout_xattn  = dropout_xattn,
        use_bn_temporal= True,
        use_bn_channel = True,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  CrossAttn model parameters: {n_params:,}")

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
    best_val_mae      = float("inf")
    best_state        = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr_loss = train_one_epoch_improved(
            model, train_loader, device, optimizer, loss_fn,
            grad_clip=cfg.grad_clip, accumulation_steps=accumulation_steps
        )
        vl_loss, vl_mets, _, _ = evaluate_improved(
            model, val_loader, device, loss_fn
        )

        if scheduler is not None:
            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_mets["MAE"])
            else:
                scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if verbose:
            print(
                f"[{epoch:03d}] TrainLoss={tr_loss:.4f}  "
                f"ValLoss={vl_loss:.4f}  "
                f"ValMAE={vl_mets['MAE']:.4f}  "
                f"ValRMSE={vl_mets['RMSE']:.4f}  "
                f"ValR2={vl_mets['R2']:.4f}  "
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
#  SECTION 5 — MAIN ORCHESTRATOR
# =============================================================================

def run_crossattn_scenarios(
    X_train_sw,
    y_train_sw,
    X_testf,
    y_test,
    features,
    eng_type: str,
    device,
    # Required for Scenarios C & D (re-windowing)
    X=None,
    X_test=None,
    create_training_sequences_sw=None,
    create_testing_sequences_sw=None,
    num_of_batches=1,
    window_size:    int  = 40,
    random_state:   int  = 341,
    run_ensemble:   bool = True,
    verbose:        bool = True,
    # Cross-attention specific overrides (None → derived from config)
    d_model_xattn:  Optional[int]   = None,
    n_heads_xattn:  Optional[int]   = None,
    d_ff_xattn:     Optional[int]   = None,
    dropout_xattn:  float           = 0.1,
):
    """
    Runs Scenarios A through E with the cross-attention fusion model.

    The scenario configs, losses, schedulers, augmentation, and ensemble
    logic are all imported from improve_transformer.py — only the model
    class is changed to PatchTST_CrossAttn_Model.

    Parameters
    ----------
    (same as run_all_scenarios in improve_transformer.py, plus:)
    d_model_xattn  : cross-attention projection dimension (None = auto)
    n_heads_xattn  : cross-attention heads (None = same as n_heads_t)
    d_ff_xattn     : cross-attention FFN hidden dim (None = same as d_ff_t)
    dropout_xattn  : dropout inside cross-attention blocks (default 0.1)
    """

    print("=" * 70)
    print("  CROSS-ATTENTION FUSION EXPERIMENT")
    print(f"  Engine type : {eng_type}")
    print(f"  Window size : {window_size}")
    print(f"  Features    : {list(features)}")
    print("=" * 70)

    # ── Train/val split ───────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sw, y_train_sw, test_size=0.2, random_state=random_state
    )

    # ── Simple dataset for test loader ────────────────────────────────────
    class _SimpleDS(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    mse_crit = nn.MSELoss()

    # ── Sequence cache for re-windowed scenarios ──────────────────────────
    _seq_cache: Dict = {}

    def _get_sequences(w: int):
        if w not in _seq_cache:
            if w == window_size:
                _seq_cache[w] = (X_train_sw, y_train_sw, X_testf)
            else:
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
                        f"Scenarios C & D require re-windowing but the "
                        f"following arguments are missing: {missing}.\n"
                        f"Pass them to run_crossattn_scenarios()."
                    )
                print(f"\n  [Re-windowing for window_size={w} ...]")
                Xsw, ysw = create_training_sequences_sw(X, features, w)
                Xtf      = create_testing_sequences_sw(
                    X_test, features, w, num_of_batches=num_of_batches
                )
                _seq_cache[w] = (Xsw, ysw, Xtf)
        return _seq_cache[w]

    # ── Scenario registry ─────────────────────────────────────────────────
    LARGER_W = max(window_size, 50)

    # Each entry: (label, description, config_fn, window, loss_fn, noise_std,
    #              use_scheduler, scheduler_type, accumulation_steps)
    scenarios = [
        (
            "A",
            "CrossAttn + Asymmetric Loss + CosineWarm LR",
            lambda: scenario_A_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0, True, "cosine_warm", 1
        ),
        (
            "B",
            "CrossAttn + Wider Model (d=96, L=3) + CosineWarm LR",
            lambda: scenario_B_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0, True, "cosine_warm", 1
        ),
        (
            "C",
            "CrossAttn + Larger Window + Gaussian Augmentation",
            lambda: scenario_C_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02, True, "cosine_warm", 1
        ),
        (
            "D",
            "CrossAttn + Full Combo (Wider + Larger Window + Aug)",
            lambda: scenario_D_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02, True, "cosine_warm", 1
        ),
    ]

    leaderboard = []

    for (label, desc, cfg_fn, win, loss_fn, noise,
         use_sched, sched_type, accum) in scenarios:

        print(f"\n{'─'*70}")
        print(f"  SCENARIO {label}: {desc}")
        print(f"{'─'*70}")

        cfg = cfg_fn()

        Xsw, ysw, Xtf = _get_sequences(win)
        Xtr_s, Xvl_s, ytr_s, yvl_s = train_test_split(
            Xsw, ysw, test_size=0.2, random_state=random_state
        )

        tr_loader, vl_loader, (C, L) = make_loaders_augmented(
            Xtr_s, Xvl_s, ytr_s, yvl_s,
            batch_size   = cfg.batch_size,
            num_workers  = getattr(cfg, "num_workers", 0),
            noise_std    = noise,
            time_warp    = False,
            use_cuda     = str(device).startswith("cuda"),
        )
        cfg.C = C
        cfg.L = L

        Xtf_trans = Xtf.transpose(0, 2, 1)
        t_loader  = DataLoader(
            _SimpleDS(Xtf_trans, y_test), batch_size=64, shuffle=False
        )

        # ── Train cross-attention model ────────────────────────────────────
        model, tr_losses, vl_losses = fit_crossattn(
            tr_loader, vl_loader,
            features, cfg, device,
            loss_fn           = loss_fn,
            use_scheduler     = use_sched,
            scheduler_type    = sched_type,
            accumulation_steps= accum,
            verbose           = verbose,
            d_model_xattn     = d_model_xattn,
            n_heads_xattn     = n_heads_xattn,
            d_ff_xattn        = d_ff_xattn,
            dropout_xattn     = dropout_xattn,
        )

        # ── Evaluate ──────────────────────────────────────────────────────
        _, test_mets, yt, yp = evaluate_improved(
            model, t_loader, device, mse_crit
        )
        nasa = score_nasa(yp - yt)

        print(f"\n  [Scenario {label} TEST]  "
              f"RMSE={test_mets['RMSE']:.4f}  "
              f"MAE={test_mets['MAE']:.4f}  "
              f"R²={test_mets['R2']:.4f}  "
              f"NASA={nasa:.1f}")

        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
            f"crossattn_scenario{label}_{eng_type}.pt"
        )

        leaderboard.append({
            "scenario": label,
            "desc"    : desc,
            "RMSE"    : test_mets["RMSE"],
            "MAE"     : test_mets["MAE"],
            "R2"      : test_mets["R2"],
            "NASA"    : nasa,
            "model"   : model,
            "window"  : win,
            "y_pred"  : yp,
            "y_true"  : yt,
        })

    # ── Sort before ensemble so we know which config to repeat ───────────
    leaderboard.sort(key=lambda r: r["RMSE"])
    best_scenario = leaderboard[0]

    # ── Scenario E: 3-seed ensemble of best config ────────────────────────
    if run_ensemble:
        print(f"\n{'─'*70}")
        print(f"  SCENARIO E: 3-seed CrossAttn Ensemble "
              f"(base: Scenario {best_scenario['scenario']})")
        print(f"{'─'*70}")

        best_label = best_scenario["scenario"]
        best_win   = best_scenario["window"]
        base_row   = next(r for r in scenarios if r[0] == best_label)
        _, _, bcfg_fn, bwin, bloss, bnoise, bsched, bstype, baccum = base_row

        Xsw_e, ysw_e, Xtf_e = _get_sequences(best_win)
        Xtf_e_trans = Xtf_e.transpose(0, 2, 1)
        te_loader   = DataLoader(
            _SimpleDS(Xtf_e_trans, y_test), batch_size=64, shuffle=False
        )

        ensemble_models = []
        for seed in [42, 137, 271]:
            print(f"\n  [Ensemble seed={seed}]")
            torch.manual_seed(seed)
            np.random.seed(seed)

            Xtr_e, Xvl_e, ytr_e, yvl_e = train_test_split(
                Xsw_e, ysw_e, test_size=0.2, random_state=seed
            )
            ecfg = bcfg_fn()
            tr_e, vl_e, (Ce, Le) = make_loaders_augmented(
                Xtr_e, Xvl_e, ytr_e, yvl_e,
                batch_size  = ecfg.batch_size,
                num_workers = getattr(ecfg, "num_workers", 0),
                noise_std   = bnoise,
                use_cuda    = str(device).startswith("cuda"),
            )
            ecfg.C = Ce
            ecfg.L = Le

            em, _, _ = fit_crossattn(
                tr_e, vl_e, features, ecfg, device,
                loss_fn           = copy.deepcopy(bloss),
                use_scheduler     = bsched,
                scheduler_type    = bstype,
                accumulation_steps= baccum,
                verbose           = verbose,
                d_model_xattn     = d_model_xattn,
                n_heads_xattn     = n_heads_xattn,
                d_ff_xattn        = d_ff_xattn,
                dropout_xattn     = dropout_xattn,
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
            "scenario": "E (ensemble)",
            "desc"    : f"3-seed CrossAttn ensemble of Scenario {best_label}",
            "RMSE"    : ens_mets["RMSE"],
            "MAE"     : ens_mets["MAE"],
            "R2"      : ens_mets["R2"],
            "NASA"    : ens_nasa,
            "model"   : ensemble_models,
            "window"  : best_win,
            "y_pred"  : ens_pred,
            "y_true"  : ens_true,
        })
        leaderboard.sort(key=lambda r: r["RMSE"])

    # ── Final leaderboard ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CROSS-ATTENTION LEADERBOARD  (sorted by RMSE)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<22} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'NASA':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, row in enumerate(leaderboard):
        marker = "  ← BEST" if i == 0 else ""
        print(f"  {row['scenario']:<22} "
              f"{row['RMSE']:>8.4f} "
              f"{row['MAE']:>8.4f} "
              f"{row['R2']:>8.4f} "
              f"{row['NASA']:>10.1f}"
              f"{marker}")
    print(f"{'='*70}")

    # ── Save best ─────────────────────────────────────────────────────────
    overall_best = leaderboard[0]
    save_path    = f"BEST_crossattn_{eng_type}.pt"
    if isinstance(overall_best["model"], list):
        states = [
            {k: v.cpu() for k, v in m.state_dict().items()}
            for m in overall_best["model"]
        ]
        torch.save(
            {"ensemble": states, "scenario": overall_best["scenario"],
             "fusion": "cross_attention"},
            save_path
        )
    else:
        torch.save(
            {k: v.cpu()
             for k, v in overall_best["model"].state_dict().items()},
            save_path
        )

    print(f"\n  Best model saved → {save_path}")
    print(f"  Best scenario   : {overall_best['scenario']} — {overall_best['desc']}")
    print(f"  Best RMSE       : {overall_best['RMSE']:.4f}")
    print(f"  Best MAE        : {overall_best['MAE']:.4f}")
    print(f"  Best R²         : {overall_best['R2']:.4f}")
    print(f"  Best NASA score : {overall_best['NASA']:.1f}")

    return leaderboard


# =============================================================================
#  COPY-PASTE SNIPPET FOR YOUR NOTEBOOK
# =============================================================================
#
#   from patchtst_crossattn import run_crossattn_scenarios
#
#   # ── Standard call (FD001 / FD003) ────────────────────────────────────
#   leaderboard = run_crossattn_scenarios(
#       X_train_sw   = X_train_sw,
#       y_train_sw   = y_train_sw,
#       X_testf      = X_testf,
#       y_test       = y_test,
#       features     = features,
#       eng_type     = eng_type,
#       device       = device,
#       X                            = X,
#       X_test                       = X_test,
#       create_training_sequences_sw = create_training_sequences_sw,
#       create_testing_sequences_sw  = create_testing_sequences_sw,
#       num_of_batches               = num_of_batches,
#       window_size  = window_size,
#       random_state = 341,
#       run_ensemble = True,
#       verbose      = True,
#   )
#
#   # ── Optional: tune cross-attention dimensions explicitly ─────────────
#   leaderboard = run_crossattn_scenarios(
#       ...,
#       d_model_xattn = 128,   # projection dim inside cross-attn (default: auto)
#       n_heads_xattn = 8,     # cross-attention heads (default: same as n_heads_t)
#       d_ff_xattn    = 256,   # FFN dim inside cross-attn (default: same as d_ff_t)
#       dropout_xattn = 0.1,
#   )
#
#   # ── Compare original FusionHead vs CrossAttentionFusionHead ──────────
#   from improve_transformer    import run_all_scenarios
#   from patchtst_crossattn     import run_crossattn_scenarios
#
#   lb_orig  = run_all_scenarios(...)
#   lb_xattn = run_crossattn_scenarios(...)
#
#   print("Original  best RMSE:", min(r["RMSE"] for r in lb_orig))
#   print("CrossAttn best RMSE:", min(r["RMSE"] for r in lb_xattn))