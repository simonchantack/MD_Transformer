"""
=============================================================================
  BERT-STYLE MULTI-LAYER FEATURE EXTRACTION  —  PatchTST Dual-Encoder RUL
  File: improve_transformer_bert.py
=============================================================================

  THE BERT FEATURE-BASED APPROACH — ADAPTED FOR TIME-SERIES
  -----------------------------------------------------------
  In the original BERT paper (Devlin et al., 2019), one of the most effective
  transfer-learning strategies was the "feature-based" approach:

    1. A deep transformer is pre-trained on a large corpus.
    2. Its parameters are FROZEN — nothing is fine-tuned.
    3. The hidden-state activations from the TOP-K LAYERS are extracted.
    4. Those activations are concatenated into a single rich representation.
    5. A lightweight task-specific head is trained on top of those features.

  The paper found that concatenating the top four hidden layers equalled or
  outperformed fine-tuning on most NLP tasks, with much less compute.

  WHY THIS HELPS FOR RUL PREDICTION
  -----------------------------------
  Different transformer layers encode different levels of abstraction:

    Lower layers  → local patch patterns, sensor noise, raw feature statistics
    Middle layers → short-range degradation trends, inter-sensor correlations
    Upper layers  → long-range degradation trajectories, failure precursors

  Using ONLY the final layer discards the lower-level structural information
  that the lower layers have learned.  Concatenating the top-4 layers gives
  the regression head simultaneous access to ALL abstraction levels, letting
  it weight them adaptively for the RUL task.

  TWO-PHASE TRAINING STRATEGY
  ----------------------------
  Phase 1 — Pre-train (supervised)
    Train the full PatchTST_RUL_Model normally (exactly as in
    improve_transformer.py Scenarios A-D).  This teaches the encoders to
    build useful representations.  The best Phase-1 checkpoint is saved.

  Phase 2 — Feature-based fine-tuning (BERT-style)
    Load the Phase-1 checkpoint.
    FREEZE all encoder parameters (temporal + channel encoders).
    Register forward hooks on each encoder layer to capture intermediate
    hidden states.
    Concatenate the activations from the top-K layers (default K=4).
    Train ONLY a new BERTStyleRegressionHead on top of those frozen features.
    Because the encoders are frozen, Phase 2 is very fast — typically 20-40
    epochs.

  ARCHITECTURE CHANGES
  ---------------------
  1. LayerActivationCapture   — context manager that hooks into every
                                nn.TransformerEncoder layer and stores outputs.

  2. BERTStyleFusionHead      — replaces FusionHead.
                                Receives K concatenated temporal layer pools
                                + K concatenated channel layer pools.
                                Projects through MLP → RUL scalar.

  3. PatchTST_BERT_Model      — assembles the FROZEN pre-trained encoders
                                with BERTStyleFusionHead.
                                Wraps PatchTST_RUL_Model for Phase 1 and
                                PatchTST_BERT_Model for Phase 2.

  SCENARIOS
  ----------
  Same A-E structure as improve_transformer.py.
  Each scenario runs Phase 1 then Phase 2 automatically.
  The leaderboard reports Phase-2 RMSE so results are directly comparable.

  HOW TO USE
  ----------
      from improve_transformer_bert import run_bert_scenarios

      leaderboard = run_bert_scenarios(
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
          top_k_layers                 = 4,    # number of layers to concatenate
      )
=============================================================================
"""

import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

# ── Reuse all training infrastructure from improve_transformer.py ─────────
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
    fit_improved,
)

# =============================================================================
#   FROM MAIN ENCODER STRUCTURE
# =============================================================================

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
from Encoder_Layers import *
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
#  SECTION 1 — LAYER ACTIVATION CAPTURE
# =============================================================================

class LayerActivationCapture:
    """
    Context manager that registers forward hooks on every layer of a
    nn.TransformerEncoder and stores each layer's output tensor.

    Usage
    -----
        with LayerActivationCapture(model.temporal_encoder.encoder) as cap:
            out = model.temporal_encoder(x)
        layer_outputs = cap.outputs   # list of (B*C, N, d_model) tensors

    Why a context manager?
    ----------------------
    Hooks must be removed after use or they accumulate across calls and leak
    memory.  The context manager guarantees removal even if an exception
    occurs mid-forward-pass.

    Notes
    -----
    • Each element of cap.outputs corresponds to one TransformerEncoder layer,
      in order from layer 0 (closest to input) to layer N-1 (closest to output).
    • Outputs are detached from the computation graph because the feature-based
      phase trains ONLY the head — encoder gradients are not needed.
      Set detach=False if you want end-to-end gradients through the captures.
    """
    def __init__(self, transformer_encoder: nn.TransformerEncoder,
                 detach: bool = True):
        self.encoder = transformer_encoder
        self.detach  = detach
        self.outputs: List[torch.Tensor] = []
        self._hooks:  List = []

    def __enter__(self):
        self.outputs = []
        for layer in self.encoder.layers:
            hook = layer.register_forward_hook(self._capture)
            self._hooks.append(hook)
        return self

    def _capture(self, module, inp, out):
        # TransformerBatchNormEncoderLayer returns (seq, B, d_model)
        # Store a clone so in-place ops later don't corrupt the cache
        t = out.clone()
        if self.detach:
            t = t.detach()
        self.outputs.append(t)

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# =============================================================================
#  SECTION 2 — BERT-STYLE FUSION HEAD
# =============================================================================

class BERTStyleFusionHead(nn.Module):
    """
    Regression head that operates on multi-layer encoder features.

    Inputs
    ------
    temporal_layer_pools : list of K tensors, each (B, d_model_t)
        Mean-pooled representation from each of the top-K temporal encoder layers.

    channel_layer_pools  : list of K tensors, each (B, d_model_c)
        Mean-pooled representation from each of the top-K channel encoder layers.

    Processing pipeline
    -------------------
    1.  Concatenate K temporal pools along the feature axis:
            t_concat  (B, K * d_model_t)
    2.  Concatenate K channel pools along the feature axis:
            c_concat  (B, K * d_model_c)
    3.  Project each to a common width d_proj via linear layers.
    4.  Add them (element-wise) and pass through LayerNorm.
        LayerNorm is used here (not BatchNorm) because the input size
        varies with K and we want per-sample normalisation.
    5.  MLP head: d_proj → head_hidden → 1 → RUL scalar.

    Why concatenate instead of averaging?
    ---------------------------------------
    Averaging loses inter-layer information — the head can no longer
    distinguish which layer a particular feature came from.  Concatenation
    preserves layer identity, letting the MLP learn to weight each layer's
    contribution differently (e.g. rely more on lower layers when the
    degradation signal is still weak).

    Parameters
    ----------
    d_model_t   : temporal encoder feature dimension
    d_model_c   : channel encoder feature dimension
    top_k       : number of layers whose pools are concatenated
    d_proj      : projection dimension after the K-layer concat (default: d_model_t)
    head_hidden : MLP hidden dimension (default: d_proj)
    dropout     : dropout in MLP
    """
    def __init__(
        self,
        d_model_t:   int,
        d_model_c:   int,
        top_k:       int = 4,
        d_proj:      Optional[int] = None,
        head_hidden: Optional[int] = None,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k

        d_proj = d_proj if d_proj is not None else d_model_t
        hid    = head_hidden if head_hidden is not None else d_proj

        # Linear projections from concatenated K-layer feature to d_proj
        self.proj_t = nn.Linear(top_k * d_model_t, d_proj)
        self.proj_c = nn.Linear(top_k * d_model_c, d_proj)

        # Per-sample normalisation (safe for any K and batch size)
        self.norm = nn.LayerNorm(d_proj)

        # MLP regression head
        self.mlp = nn.Sequential(
            nn.Linear(d_proj, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, 1),
        )

    def forward(
        self,
        temporal_layer_pools: List[torch.Tensor],   # K × (B, d_model_t)
        channel_layer_pools:  List[torch.Tensor],   # K × (B, d_model_c)
    ) -> torch.Tensor:

        # Use only the top-K layers (last K in the list)
        t_pools = temporal_layer_pools[-self.top_k:]
        c_pools = channel_layer_pools[-self.top_k:]

        # Concatenate along feature dimension
        t_cat = torch.cat(t_pools, dim=-1)   # (B, K * d_model_t)
        c_cat = torch.cat(c_pools, dim=-1)   # (B, K * d_model_c)

        # Project to common dimension
        t_proj = self.proj_t(t_cat)          # (B, d_proj)
        c_proj = self.proj_c(c_cat)          # (B, d_proj)

        # Fuse and normalise
        fused = self.norm(t_proj + c_proj)   # (B, d_proj)

        return self.mlp(fused)               # (B, 1)


# =============================================================================
#  SECTION 3 — BERT-STYLE FULL MODEL
# =============================================================================

class PatchTST_BERT_Model(nn.Module):
    """
    PatchTST Dual-Encoder model with BERT-style multi-layer feature extraction.

    The temporal and channel encoders are UNCHANGED from PatchTST_RUL_Model.
    The FusionHead is replaced by BERTStyleFusionHead, which receives pooled
    activations from each of the top-K encoder layers rather than only the
    final layer output.

    Training modes
    --------------
    freeze_encoders=False (Phase 1 / end-to-end)
        All parameters are trained.  The model behaves like PatchTST_RUL_Model
        except the head receives multi-layer features.  This is the default
        for Phase 1 training.

    freeze_encoders=True  (Phase 2 / feature-based)
        Encoder parameters are frozen (requires_grad=False).
        Only the BERTStyleFusionHead parameters are trained.
        Hooks capture intermediate layer activations during the forward pass.
        This is the BERT feature-based approach.

    Parameters
    ----------
    (standard PatchTST params, plus:)
    top_k_layers    : int   number of encoder layers whose outputs to concatenate
    d_proj          : int   projection dim inside BERTStyleFusionHead
    freeze_encoders : bool  if True, freeze encoder parameters
    """
    def __init__(
        self,
        C, L,
        d_model_t:   int,
        n_heads_t:   int,
        n_layers_t:  int,
        d_ff_t:      int,
        dropout_t:   float,
        patch_len_t: int,
        stride_t:    int,
        patch_len_c: int,
        stride_c:    int,
        d_model_c:   int,
        n_heads_c:   int,
        n_layers_c:  int,
        d_ff_c:      int,
        dropout_c:   float,
        head_hidden: Optional[int]  = None,
        use_bn_temporal: bool = True,
        use_bn_channel:  bool = True,
        top_k_layers:    int  = 4,
        d_proj:          Optional[int] = None,
        freeze_encoders: bool = False,
    ):
        super().__init__()
        self.top_k = top_k_layers

        # ── Temporal encoder (PatchTSTEncoder from notebook) ──────────────
        self.temporal_encoder = PatchTSTEncoder(
            d_model          = d_model_t,
            n_heads          = n_heads_t,
            n_layers         = n_layers_t,
            d_ff             = d_ff_t,
            dropout          = dropout_t,
            patch_len        = patch_len_t,
            stride           = stride_t,
            use_batchnorm_out= use_bn_temporal,
        )

        # ── Channel encoder (SensorChannelTransformerEncoder from notebook)
        self.sensor_encoder = SensorChannelTransformerEncoder(
            C               = C,
            L               = L,
            patch_len       = patch_len_c,
            stride          = stride_c,
            d_model         = d_model_c,
            n_heads         = n_heads_c,
            num_layers      = n_layers_c,
            dim_feedforward = d_ff_c,
            dropout         = dropout_c,
            use_batchnorm_out= use_bn_channel,
        )

        # ── BERT-style fusion head ─────────────────────────────────────────
        self.fusion_head = BERTStyleFusionHead(
            d_model_t   = d_model_t,
            d_model_c   = d_model_c,
            top_k       = top_k_layers,
            d_proj      = d_proj,
            head_hidden = head_hidden,
            dropout     = dropout_t,
        )

        # Clamp top_k to available layers so the model is always valid
        self.top_k = min(top_k_layers,
                         min(n_layers_t, n_layers_c))

        if freeze_encoders:
            self.freeze_encoders()

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────

    def freeze_encoders(self):
        """Freeze all temporal and channel encoder parameters."""
        for p in self.temporal_encoder.parameters():
            p.requires_grad = False
        for p in self.sensor_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze all temporal and channel encoder parameters."""
        for p in self.temporal_encoder.parameters():
            p.requires_grad = True
        for p in self.sensor_encoder.parameters():
            p.requires_grad = True

    # ── Per-layer pool extraction ─────────────────────────────────────────

    def _extract_temporal_pools(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run temporal encoder, capture all layer outputs, return:
          - final output  (B, N, d_model_t)  — same as PatchTSTEncoder.forward
          - list of N mean-pooled tensors (B, d_model_t), one per layer
        """
        B, C, L = x.shape

        # InstanceNorm per channel (same as PatchTSTEncoder.forward)
        xf = x.reshape(B * C, 1, L)
        xf = self.temporal_encoder.inst_norm(xf)
        xf = xf.squeeze(1)                                    # (B*C, L)

        # Patchify + embed
        tokens = self.temporal_encoder.patch_embed(xf)        # (B*C, N, d_t)

        # Positional encoding
        tokens = self.temporal_encoder.pos_enc(tokens)        # (B*C, N, d_t)

        # Run encoder layer-by-layer, capturing each output
        # TransformerEncoder expects (seq, batch, feat) internally
        src = tokens.permute(1, 0, 2)                         # (N, B*C, d_t)
        layer_pools_t: List[torch.Tensor] = []

        for layer in self.temporal_encoder.encoder.layers:
            src = layer(src)                                   # (N, B*C, d_t)
            # Convert to (B*C, N, d_t), mean-pool over N, reshape to (B, d_t)
            layer_out = src.permute(1, 0, 2)                  # (B*C, N, d_t)
            BxC, N, D = layer_out.shape
            # Reshape to (B, C, N, d_t), mean over C and N
            enc_b = layer_out.view(B, C, N, D)
            pool  = enc_b.mean(dim=(1, 2))                    # (B, d_t)
            layer_pools_t.append(pool)

        # Final output through the same post-processing as PatchTSTEncoder
        enc = src.permute(1, 0, 2)                            # (B*C, N, d_t)
        BxC, N, D = enc.shape
        enc = enc.view(B, C, N, D).mean(dim=1)                # (B, N, d_t)
        if self.temporal_encoder.use_bn:
            enc = enc.transpose(1, 2)
            enc = self.temporal_encoder.bn_out(enc)
            enc = enc.transpose(1, 2)
        else:
            enc = self.temporal_encoder.ln_out(enc)

        return enc, layer_pools_t                              # (B,N,d_t), K×(B,d_t)

    def _extract_channel_pools(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run channel encoder, capture all layer outputs, return:
          - final output  (B, L*N_patch, d_model_c)
          - list of mean-pooled tensors (B, d_model_c), one per layer
        """
        B, C, L = x.shape
        assert C == self.sensor_encoder.C and L == self.sensor_encoder.L

        xp = x.permute(0, 2, 1).reshape(B * L, C)             # (B*L, C)
        tokens = self.sensor_encoder.patch_embed(xp)           # (B*L, N_p, d_c)
        num_patches = tokens.size(1)

        tokens = tokens.view(B, L, num_patches, -1)
        tokens = tokens.view(B, L * num_patches, -1)           # (B, L*N_p, d_c)
        tokens = self.sensor_encoder.pos_encoder(tokens)

        # Run layer-by-layer
        src = tokens.permute(1, 0, 2)                          # (L*N_p, B, d_c)
        layer_pools_c: List[torch.Tensor] = []

        for layer in self.sensor_encoder.transformer_encoder.layers:
            src = layer(src)                                   # (L*N_p, B, d_c)
            layer_out = src.permute(1, 0, 2)                  # (B, L*N_p, d_c)
            pool = layer_out.mean(dim=1)                      # (B, d_c)
            layer_pools_c.append(pool)

        # Final output with same post-processing as SensorChannelTransformerEncoder
        enc = src.permute(1, 0, 2)                            # (B, L*N_p, d_c)
        if self.sensor_encoder.use_bn:
            enc = enc.transpose(1, 2)
            enc = self.sensor_encoder.bn_out(enc)
            enc = enc.transpose(1, 2)
        else:
            enc = self.sensor_encoder.ln_out(enc)

        return enc, layer_pools_c                              # (B,seq,d_c), K×(B,d_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using multi-layer feature extraction.
        All paths — frozen or not — go through _extract_*_pools so the
        gradient flow is consistent.
        """
        _, layer_pools_t = self._extract_temporal_pools(x)
        _, layer_pools_c = self._extract_channel_pools(x)
        y = self.fusion_head(layer_pools_t, layer_pools_c)    # (B, 1)
        return y.squeeze(-1)                                   # (B,)


# =============================================================================
#  SECTION 4 — PHASE-1 TRAINING (standard, reuses improve_transformer.py)
# =============================================================================

def _phase1_config_to_bert_model(cfg: TrainConfig, top_k: int,
                                  device) -> "PatchTST_BERT_Model":
    """Build PatchTST_BERT_Model from a TrainConfig (Phase 1, unfrozen)."""
    return PatchTST_BERT_Model(
        C            = cfg.C,
        L            = cfg.L,
        d_model_t    = cfg.d_model_t,
        n_heads_t    = cfg.n_heads_t,
        n_layers_t   = cfg.n_layers_t,
        d_ff_t       = cfg.d_ff_t,
        dropout_t    = cfg.dropout_t,
        patch_len_t  = cfg.patch_len_t,
        stride_t     = cfg.stride_t,
        patch_len_c  = cfg.patch_len_c,
        stride_c     = cfg.stride_c,
        d_model_c    = cfg.d_model_c,
        n_heads_c    = cfg.n_heads_c,
        n_layers_c   = cfg.n_layers_c,
        d_ff_c       = cfg.d_ff_c,
        dropout_c    = cfg.dropout_c,
        head_hidden  = cfg.head_hidden,
        use_bn_temporal = True,
        use_bn_channel  = True,
        top_k_layers    = top_k,
        freeze_encoders = False,   # Phase 1: train everything
    ).to(device)


def fit_bert_phase1(
    train_loader, val_loader,
    features, cfg: TrainConfig,
    device,
    top_k:              int   = 4,
    loss_fn             = None,
    use_scheduler:      bool  = True,
    scheduler_type:     str   = "cosine_warm",
    accumulation_steps: int   = 1,
    verbose:            bool  = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Phase 1: end-to-end training of PatchTST_BERT_Model (encoders unfrozen).
    The multi-layer head is trained jointly with the encoders.
    This is analogous to pre-training in BERT.
    """
    model = _phase1_config_to_bert_model(cfg, top_k, device)

    if verbose:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Phase 1] BERT model params: {n:,}")

    if loss_fn is None:
        loss_fn = AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

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

    best_val_mae      = float("inf")
    best_state        = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        t0      = time.time()
        tr_loss = train_one_epoch_improved(
            model, train_loader, device, optimizer, loss_fn,
            grad_clip=cfg.grad_clip, accumulation_steps=accumulation_steps,
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
                f"  P1[{epoch:03d}] Loss={tr_loss:.4f}  "
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
                    print(f"  P1 Early stop epoch {epoch}. "
                          f"Best Val MAE={best_val_mae:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# =============================================================================
#  SECTION 5 — PHASE-2 TRAINING (BERT feature-based, frozen encoders)
# =============================================================================

def fit_bert_phase2(
    phase1_model:       nn.Module,
    train_loader,
    val_loader,
    cfg:                TrainConfig,
    device,
    top_k:              int   = 4,
    phase2_epochs:      int   = 60,
    phase2_lr:          float = 5e-4,
    phase2_patience:    int   = 15,
    loss_fn             = None,
    verbose:            bool  = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Phase 2: BERT-style feature-based fine-tuning.

    Steps
    -----
    1.  Copy Phase-1 encoder weights into a new PatchTST_BERT_Model.
    2.  Freeze all encoder parameters (requires_grad=False).
    3.  Reinitialise the BERTStyleFusionHead with fresh weights.
    4.  Train only the head on top of the frozen multi-layer features.

    Why reinitialise the head?
    --------------------------
    The Phase-1 head was jointly optimised with the encoders at a given lr.
    For Phase 2, the encoders are fixed and we want the head to start from
    scratch so it can optimally exploit all K layers, not just the final one.
    A fresh head + higher lr is standard practice in BERT fine-tuning.

    Parameters
    ----------
    phase1_model   : trained Phase-1 PatchTST_BERT_Model
    phase2_epochs  : max epochs for Phase-2 head training (default 60)
    phase2_lr      : learning rate for Phase-2 (default 5e-4, higher than P1)
    phase2_patience: early stopping patience for Phase-2
    """
    if loss_fn is None:
        loss_fn = AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0)

    # Build a fresh Phase-2 model with the SAME encoder architecture
    model2 = PatchTST_BERT_Model(
        C            = cfg.C,
        L            = cfg.L,
        d_model_t    = cfg.d_model_t,
        n_heads_t    = cfg.n_heads_t,
        n_layers_t   = cfg.n_layers_t,
        d_ff_t       = cfg.d_ff_t,
        dropout_t    = cfg.dropout_t,
        patch_len_t  = cfg.patch_len_t,
        stride_t     = cfg.stride_t,
        patch_len_c  = cfg.patch_len_c,
        stride_c     = cfg.stride_c,
        d_model_c    = cfg.d_model_c,
        n_heads_c    = cfg.n_heads_c,
        n_layers_c   = cfg.n_layers_c,
        d_ff_c       = cfg.d_ff_c,
        dropout_c    = cfg.dropout_c,
        head_hidden  = cfg.head_hidden,
        use_bn_temporal = True,
        use_bn_channel  = True,
        top_k_layers    = top_k,
        freeze_encoders = True,    # ← BERT feature-based: freeze encoders
    ).to(device)

    # Copy Phase-1 encoder weights exactly
    p1_sd  = phase1_model.state_dict()
    p2_sd  = model2.state_dict()
    # Copy only encoder keys; skip fusion_head (fresh in Phase 2)
    for k in p2_sd:
        if k.startswith("temporal_encoder.") or k.startswith("sensor_encoder."):
            if k in p1_sd:
                p2_sd[k] = p1_sd[k].clone()
    model2.load_state_dict(p2_sd, strict=True)

    trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model2.parameters())
    if verbose:
        print(f"  [Phase 2] Encoders FROZEN.  "
              f"Trainable: {trainable:,} / {total:,} params "
              f"({100*trainable/total:.1f}%)")

    # Only optimise the head
    head_params = [p for p in model2.parameters() if p.requires_grad]
    optimizer2  = torch.optim.AdamW(
        head_params, lr=phase2_lr, weight_decay=cfg.weight_decay
    )
    # Cosine schedule over Phase-2 epochs
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=phase2_epochs, eta_min=1e-7
    )

    best_val_mae2      = float("inf")
    best_state2        = None
    epochs_no_improve2 = 0
    train_losses2, val_losses2 = [], []

    for epoch in range(1, phase2_epochs + 1):
        t0       = time.time()
        tr_loss2 = train_one_epoch_improved(
            model2, train_loader, device, optimizer2, loss_fn, grad_clip=1.0
        )
        vl_loss2, vl_mets2, _, _ = evaluate_improved(
            model2, val_loader, device, loss_fn
        )
        scheduler2.step()
        train_losses2.append(tr_loss2)
        val_losses2.append(vl_loss2)

        if verbose:
            print(
                f"  P2[{epoch:02d}] Loss={tr_loss2:.4f}  "
                f"ValMAE={vl_mets2['MAE']:.4f}  "
                f"ValRMSE={vl_mets2['RMSE']:.4f}  "
                f"LR={optimizer2.param_groups[0]['lr']:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if vl_mets2["MAE"] < best_val_mae2 - 1e-6:
            best_val_mae2      = vl_mets2["MAE"]
            best_state2        = {k: v.cpu() for k, v in model2.state_dict().items()}
            epochs_no_improve2 = 0
        else:
            epochs_no_improve2 += 1
            if epochs_no_improve2 >= phase2_patience:
                if verbose:
                    print(f"  P2 Early stop epoch {epoch}. "
                          f"Best Val MAE={best_val_mae2:.4f}")
                break

    if best_state2:
        model2.load_state_dict(best_state2)

    return model2, train_losses2, val_losses2


# =============================================================================
#  SECTION 6 — COMBINED FIT (Phase 1 → Phase 2)
# =============================================================================

def fit_bert_twophase(
    train_loader, val_loader,
    features, cfg: TrainConfig,
    device,
    top_k:              int   = 4,
    loss_fn             = None,
    use_scheduler:      bool  = True,
    scheduler_type:     str   = "cosine_warm",
    accumulation_steps: int   = 1,
    phase2_epochs:      int   = 60,
    phase2_lr:          float = 5e-4,
    phase2_patience:    int   = 15,
    verbose:            bool  = True,
) -> Tuple[nn.Module, nn.Module]:
    """
    Convenience wrapper that runs Phase 1 then Phase 2 in sequence.

    Returns
    -------
    model_p1 : trained Phase-1 model (all params trained)
    model_p2 : trained Phase-2 model (frozen encoders, fresh head)

    The Phase-2 model is the final deliverable for evaluation.
    """
    print(f"\n  {'─'*30}  PHASE 1: End-to-End Training  {'─'*30}")
    model_p1, _, _ = fit_bert_phase1(
        train_loader, val_loader, features, cfg, device,
        top_k=top_k, loss_fn=loss_fn,
        use_scheduler=use_scheduler, scheduler_type=scheduler_type,
        accumulation_steps=accumulation_steps, verbose=verbose,
    )

    print(f"\n  {'─'*27}  PHASE 2: BERT Feature-Based Fine-Tuning  {'─'*27}")
    model_p2, _, _ = fit_bert_phase2(
        model_p1, train_loader, val_loader, cfg, device,
        top_k=top_k, phase2_epochs=phase2_epochs,
        phase2_lr=phase2_lr, phase2_patience=phase2_patience,
        loss_fn=loss_fn, verbose=verbose,
    )

    return model_p1, model_p2


# =============================================================================
#  SECTION 7 — SCENARIO CONFIGS
#  (identical hyperparameters to improve_transformer.py; only model changes)
# =============================================================================

def bert_scenario_A_config(features, window_size):
    return TrainConfig(
        feature_cols=list(features), C=len(features), L=window_size,
        patch_len_t=10, stride_t=5, patch_len_c=3, stride_c=1,
        d_model_t=64,  n_heads_t=8, n_layers_t=4, d_ff_t=128,  dropout_t=0.1,
        d_model_c=64,  n_heads_c=8, n_layers_c=4, d_ff_c=128,  dropout_c=0.1,
        head_hidden=128, head_dropout=0.1,
        batch_size=40, epochs=150, lr=3e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"bert_scenarioA_best.pt",
    )
    # Note: n_layers=4 (not 2) so we have enough layers for top_k=4 extraction


def bert_scenario_B_config(features, window_size):
    return TrainConfig(
        feature_cols=list(features), C=len(features), L=window_size,
        patch_len_t=10, stride_t=5, patch_len_c=3, stride_c=1,
        d_model_t=96,  n_heads_t=8, n_layers_t=4, d_ff_t=256,  dropout_t=0.15,
        d_model_c=96,  n_heads_c=8, n_layers_c=4, d_ff_c=256,  dropout_c=0.15,
        head_hidden=192, head_dropout=0.1,
        batch_size=40, epochs=150, lr=2e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"bert_scenarioB_best.pt",
    )


def bert_scenario_C_config(features, window_size):
    return TrainConfig(
        feature_cols=list(features), C=len(features), L=window_size,
        patch_len_t=10, stride_t=5, patch_len_c=3, stride_c=1,
        d_model_t=64,  n_heads_t=8, n_layers_t=4, d_ff_t=128,  dropout_t=0.1,
        d_model_c=64,  n_heads_c=8, n_layers_c=4, d_ff_c=128,  dropout_c=0.1,
        head_hidden=128, head_dropout=0.1,
        batch_size=40, epochs=150, lr=3e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"bert_scenarioC_best.pt",
    )


def bert_scenario_D_config(features, window_size):
    return TrainConfig(
        feature_cols=list(features), C=len(features), L=window_size,
        patch_len_t=10, stride_t=5, patch_len_c=3, stride_c=1,
        d_model_t=96,  n_heads_t=8, n_layers_t=4, d_ff_t=256,  dropout_t=0.15,
        d_model_c=96,  n_heads_c=8, n_layers_c=4, d_ff_c=256,  dropout_c=0.15,
        head_hidden=192, head_dropout=0.1,
        batch_size=40, epochs=200, lr=2e-4,
        weight_decay=1e-4, grad_clip=1.0, patience=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=f"bert_scenarioD_best.pt",
    )


# =============================================================================
#  SECTION 8 — MAIN ORCHESTRATOR
# =============================================================================

def run_bert_scenarios(
    X_train_sw,
    y_train_sw,
    X_testf,
    y_test,
    features,
    eng_type:       str,
    device,
    X=None,
    X_test=None,
    create_training_sequences_sw=None,
    create_testing_sequences_sw=None,
    num_of_batches: int  = 1,
    window_size:    int  = 40,
    random_state:   int  = 341,
    run_ensemble:   bool = True,
    verbose:        bool = True,
    top_k_layers:   int  = 4,
    phase2_epochs:  int  = 60,
    phase2_lr:      float= 5e-4,
    phase2_patience:int  = 15,
):
    """
    Runs BERT-style scenarios A through E.
    Each scenario executes Phase 1 (end-to-end) then Phase 2 (feature-based).
    Leaderboard reports Phase-2 RMSE for direct comparison.

    Parameters
    ----------
    top_k_layers   : number of encoder layers to concatenate (default 4)
    phase2_epochs  : max epochs for Phase-2 head training (default 60)
    phase2_lr      : Phase-2 learning rate (default 5e-4)
    phase2_patience: Phase-2 early stopping patience (default 15)
    """
    print("=" * 70)
    print("  BERT-STYLE MULTI-LAYER FEATURE EXTRACTION EXPERIMENT")
    print(f"  Engine type   : {eng_type}")
    print(f"  Window size   : {window_size}")
    print(f"  Top-K layers  : {top_k_layers}")
    print(f"  Features      : {list(features)}")
    print("=" * 70)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sw, y_train_sw, test_size=0.2, random_state=random_state
    )

    class _SimpleDS(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    mse_crit = nn.MSELoss()
    LARGER_W = max(window_size, 50)
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
                        f"Scenarios C & D require re-windowing but these "
                        f"arguments are missing: {missing}. "
                        f"Pass them to run_bert_scenarios()."
                    )
                print(f"\n  [Re-windowing for window_size={w} ...]")
                Xsw, ysw = create_training_sequences_sw(X, features, w)
                Xtf      = create_testing_sequences_sw(
                    X_test, features, w, num_of_batches=num_of_batches
                )
                _seq_cache[w] = (Xsw, ysw, Xtf)
        return _seq_cache[w]

    base_loss = AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0)

    scenarios = [
        ("A", "BERT + Asymmetric Loss + CosineWarm",
         bert_scenario_A_config, window_size, 0.0),
        ("B", "BERT + Wider Model (d=96, L=4) + CosineWarm",
         bert_scenario_B_config, window_size, 0.0),
        ("C", "BERT + Larger Window + Gaussian Augmentation",
         bert_scenario_C_config, LARGER_W,   0.02),
        ("D", "BERT + Full Combo (Wider + Larger Window + Aug)",
         bert_scenario_D_config, LARGER_W,   0.02),
    ]

    leaderboard = []

    for (label, desc, cfg_fn, win, noise) in scenarios:
        print(f"\n{'─'*70}")
        print(f"  SCENARIO {label}: {desc}")
        print(f"{'─'*70}")

        cfg = cfg_fn(features, win)

        Xsw, ysw, Xtf = _get_sequences(win)
        Xtr_s, Xvl_s, ytr_s, yvl_s = train_test_split(
            Xsw, ysw, test_size=0.2, random_state=random_state
        )

        tr_loader, vl_loader, (C, L) = make_loaders_augmented(
            Xtr_s, Xvl_s, ytr_s, yvl_s,
            batch_size  = cfg.batch_size,
            num_workers = getattr(cfg, "num_workers", 0),
            noise_std   = noise,
            time_warp   = False,
            use_cuda    = str(device).startswith("cuda"),
        )
        cfg.C = C
        cfg.L = L

        Xtf_trans = Xtf.transpose(0, 2, 1)
        t_loader  = DataLoader(
            _SimpleDS(Xtf_trans, y_test), batch_size=64, shuffle=False
        )

        # ── Two-phase training ─────────────────────────────────────────────
        model_p1, model_p2 = fit_bert_twophase(
            tr_loader, vl_loader, features, cfg, device,
            top_k          = top_k_layers,
            loss_fn        = copy.deepcopy(base_loss),
            use_scheduler  = True,
            scheduler_type = "cosine_warm",
            phase2_epochs  = phase2_epochs,
            phase2_lr      = phase2_lr,
            phase2_patience= phase2_patience,
            verbose        = verbose,
        )

        # ── Evaluate Phase-2 model ─────────────────────────────────────────
        _, test_mets_p2, yt, yp_p2 = evaluate_improved(
            model_p2, t_loader, device, mse_crit
        )
        nasa_p2 = score_nasa(yp_p2 - yt)

        # ── Also evaluate Phase-1 for comparison ──────────────────────────
        _, test_mets_p1, _,  yp_p1 = evaluate_improved(
            model_p1, t_loader, device, mse_crit
        )
        nasa_p1 = score_nasa(yp_p1 - yt)

        print(f"\n  [Scenario {label}]  "
              f"Phase-1 RMSE={test_mets_p1['RMSE']:.4f}  "
              f"NASA={nasa_p1:.1f}  |  "
              f"Phase-2 (BERT) RMSE={test_mets_p2['RMSE']:.4f}  "
              f"NASA={nasa_p2:.1f}")

        torch.save(
            {k: v.cpu() for k, v in model_p2.state_dict().items()},
            f"bert_scenario{label}_{eng_type}.pt"
        )

        leaderboard.append({
            "scenario"   : label,
            "desc"       : desc,
            "RMSE"       : test_mets_p2["RMSE"],
            "MAE"        : test_mets_p2["MAE"],
            "R2"         : test_mets_p2["R2"],
            "NASA"       : nasa_p2,
            "RMSE_p1"    : test_mets_p1["RMSE"],
            "NASA_p1"    : nasa_p1,
            "model"      : model_p2,
            "model_p1"   : model_p1,
            "window"     : win,
            "y_pred"     : yp_p2,
            "y_true"     : yt,
        })

    leaderboard.sort(key=lambda r: r["RMSE"])
    best_scenario = leaderboard[0]

    # ── Scenario E: 3-seed ensemble of best Phase-2 config ────────────────
    if run_ensemble:
        print(f"\n{'─'*70}")
        print(f"  SCENARIO E: 3-seed BERT Ensemble "
              f"(base: Scenario {best_scenario['scenario']})")
        print(f"{'─'*70}")

        best_label   = best_scenario["scenario"]
        best_win     = best_scenario["window"]
        best_cfg_fn  = {
            "A": bert_scenario_A_config,
            "B": bert_scenario_B_config,
            "C": bert_scenario_C_config,
            "D": bert_scenario_D_config,
        }[best_label]
        best_noise   = next(s[4] for s in scenarios if s[0] == best_label)

        Xsw_e, ysw_e, Xtf_e = _get_sequences(best_win)
        te_loader   = DataLoader(
            _SimpleDS(Xtf_e.transpose(0, 2, 1), y_test),
            batch_size=64, shuffle=False
        )

        ensemble_models = []
        for seed in [42, 137, 271]:
            print(f"\n  [Ensemble seed={seed}]")
            torch.manual_seed(seed)
            np.random.seed(seed)

            Xtr_e, Xvl_e, ytr_e, yvl_e = train_test_split(
                Xsw_e, ysw_e, test_size=0.2, random_state=seed
            )
            ecfg = best_cfg_fn(features, best_win)
            tr_e, vl_e, (Ce, Le) = make_loaders_augmented(
                Xtr_e, Xvl_e, ytr_e, yvl_e,
                batch_size  = ecfg.batch_size,
                num_workers = getattr(ecfg, "num_workers", 0),
                noise_std   = best_noise,
                use_cuda    = str(device).startswith("cuda"),
            )
            ecfg.C = Ce
            ecfg.L = Le

            _, em_p2 = fit_bert_twophase(
                tr_e, vl_e, features, ecfg, device,
                top_k          = top_k_layers,
                loss_fn        = copy.deepcopy(base_loss),
                use_scheduler  = True,
                scheduler_type = "cosine_warm",
                phase2_epochs  = phase2_epochs,
                phase2_lr      = phase2_lr,
                phase2_patience= phase2_patience,
                verbose        = verbose,
            )
            ensemble_models.append(em_p2)

        ens_pred, ens_true, ens_mets = ensemble_predict(
            ensemble_models, te_loader, device
        )
        ens_nasa = score_nasa(ens_pred - ens_true)

        print(f"\n  [Scenario E BERT ENSEMBLE]  "
              f"RMSE={ens_mets['RMSE']:.4f}  "
              f"MAE={ens_mets['MAE']:.4f}  "
              f"R²={ens_mets['R2']:.4f}  "
              f"NASA={ens_nasa:.1f}")

        leaderboard.append({
            "scenario": "E (ensemble)",
            "desc"    : f"3-seed BERT ensemble of Scenario {best_label}",
            "RMSE"    : ens_mets["RMSE"],
            "MAE"     : ens_mets["MAE"],
            "R2"      : ens_mets["R2"],
            "NASA"    : ens_nasa,
            "RMSE_p1" : float("nan"),
            "NASA_p1" : float("nan"),
            "model"   : ensemble_models,
            "window"  : best_win,
            "y_pred"  : ens_pred,
            "y_true"  : ens_true,
        })
        leaderboard.sort(key=lambda r: r["RMSE"])

    # ── Leaderboard ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BERT LEADERBOARD  (sorted by Phase-2 RMSE)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<22} {'P1 RMSE':>8} {'P2 RMSE':>8} "
          f"{'MAE':>7} {'R2':>7} {'NASA':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")
    for i, row in enumerate(leaderboard):
        marker  = "  ← BEST" if i == 0 else ""
        p1_rmse = f"{row['RMSE_p1']:8.4f}" if not math.isnan(row.get("RMSE_p1", float("nan"))) else "       -"
        print(f"  {row['scenario']:<22} {p1_rmse} "
              f"{row['RMSE']:8.4f} "
              f"{row['MAE']:7.4f} "
              f"{row['R2']:7.4f} "
              f"{row['NASA']:9.1f}"
              f"{marker}")
    print(f"{'='*70}")

    # ── Save best ─────────────────────────────────────────────────────────
    overall_best = leaderboard[0]
    save_path    = f"BEST_bert_{eng_type}.pt"

    if isinstance(overall_best["model"], list):
        states = [
            {k: v.cpu() for k, v in m.state_dict().items()}
            for m in overall_best["model"]
        ]
        torch.save(
            {"ensemble": states, "scenario": overall_best["scenario"],
             "approach": "BERT_feature_based", "top_k_layers": top_k_layers},
            save_path
        )
    else:
        torch.save(
            {k: v.cpu() for k, v in overall_best["model"].state_dict().items()},
            save_path
        )

    print(f"\n  Best model saved → {save_path}")
    print(f"  Best scenario   : {overall_best['scenario']} — {overall_best['desc']}")
    print(f"  Best RMSE (P2)  : {overall_best['RMSE']:.4f}")
    print(f"  Best MAE        : {overall_best['MAE']:.4f}")
    print(f"  Best R²         : {overall_best['R2']:.4f}")
    print(f"  Best NASA       : {overall_best['NASA']:.1f}")

    return leaderboard


# =============================================================================
#  COPY-PASTE SNIPPET FOR THE NOTEBOOK
# =============================================================================
#
#   from improve_transformer_bert import run_bert_scenarios
#
#   leaderboard = run_bert_scenarios(
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
#       top_k_layers                 = 4,    # BERT top-4 layers (tunable)
#       phase2_epochs                = 60,   # epochs for frozen-head training
#       phase2_lr                    = 5e-4, # higher LR for head-only phase
#       phase2_patience              = 15,
#   )
#
#   # ── Compare against original improve_transformer.py ───────────────────
#   from improve_transformer      import run_all_scenarios
#   from improve_transformer_bert import run_bert_scenarios
#
#   lb_orig = run_all_scenarios(...)
#   lb_bert = run_bert_scenarios(...)
#
#   print("Original  best RMSE:", min(r["RMSE"] for r in lb_orig))
#   print("BERT      best RMSE:", min(r["RMSE"] for r in lb_bert))