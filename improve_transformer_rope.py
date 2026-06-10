"""
=============================================================================
  TRANSFORMER IMPROVEMENT WITH ROTARY POSITIONAL EMBEDDINGS (RoPE)
  File: improve_transformer_rope.py
=============================================================================

  WHY RoPE INSTEAD OF SINUSOIDAL POSITION ENCODING
  --------------------------------------------------
  SinusoidalPositionEncoding adds a fixed position vector to each token
  embedding *before* it enters the transformer stack:

      token_i  ←  token_i  +  PE(i)

  This has two practical costs:

    1. EMBEDDING CORRUPTION
       The additive PE permanently mixes positional signal into the learned
       feature representation.  In the deeper layers the model must work
       harder to separate "what sensor reading was this" from "at what
       position in the window did this occur", because both are now blended
       into the same vector.

    2. WEAK RELATIVE POSITION SIGNAL
       Sinusoidal PE encodes absolute position.  For degradation prediction,
       *relative* distance between cycles is more informative than absolute
       position — cycle 200 vs 201 being close together matters more than
       their absolute values.

  Rotary Positional Embedding (RoPE) [Su et al. 2021] fixes both issues:

    • It is applied directly to the QUERY and KEY vectors inside each
      attention head, via a rotation matrix:

          Q_rotated[m] = R(m · Θ) · Q[m]
          K_rotated[n] = R(n · Θ) · K[n]

      where m, n are positions, Θ is a learned or fixed frequency vector,
      and R(θ) is a 2D rotation matrix applied pair-wise to adjacent dims.

    • The dot-product Q_rotated[m] · K_rotated[n]ᵀ naturally encodes the
      RELATIVE angle R((m-n)·Θ) rather than absolute positions.

    • The VALUE vectors and the token embeddings themselves are NEVER
      modified — the positional signal exists purely in the attention
      similarity scores, leaving the feature representations intact.

    • RoPE generalises better to sequence lengths not seen during training,
      which matters when you vary window_size across scenarios.

  HOW IT IS INTEGRATED HERE
  -------------------------
  Three classes are modified from the original architecture.  All other
  logic — training loop, loss, scheduler, scenarios, ensemble — is identical
  to improve_transformer.py.

  1.  RotaryEmbedding         — precomputes cos/sin tables for each position.

  2.  apply_rotary_emb()      — rotates Q or K in-place given the tables.

  3.  RoPESparseMultiheadAttention
                              — replaces ProbSparseMultiheadAttention.
                                Applies RoPE to Q and K after projection,
                                before ProbAttention.

  4.  TransformerRoPEEncoderLayer
                              — identical to TransformerBatchNormEncoderLayer
                                but uses RoPESparseMultiheadAttention.

  5.  PatchTSTEncoderRoPE     — identical to PatchTSTEncoder but:
                                  * uses TransformerRoPEEncoderLayer
                                  * OMITS SinusoidalPositionEncoding
                                    (RoPE renders it redundant)

  6.  SensorChannelTransformerEncoderRoPE
                              — identical to SensorChannelTransformerEncoder
                                but uses TransformerRoPEEncoderLayer and
                                omits SinusoidalPositionEncoding.

  7.  FusionHead              — unchanged.

  8.  PatchTST_RUL_RoPE_Model — assembles the RoPE-enabled encoders with the
                                unchanged FusionHead.

  HOW TO USE
  ----------
      from improve_transformer_rope import run_rope_scenarios

      leaderboard = run_rope_scenarios(
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
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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
)

# ── ProbAttention must be importable (same as your notebook) ──────────────
from attn import ProbAttention

# Training function
import joblib
# Create device object to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0
# device = torch.device("cpu")  # If only using CPU
print(device)
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

# =============================================================================
#  SECTION 1 — ROTARY POSITIONAL EMBEDDING (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Precomputes the cosine and sine tables used by RoPE.

    For a head dimension d_k and a maximum sequence length max_seq_len,
    the frequency for dimension pair i is:

        θ_i = 1 / (10000 ^ (2i / d_k))     i = 0, 1, …, d_k/2 - 1

    For position m, the rotation angle for pair i is  m · θ_i.

    The tables cos_table and sin_table have shape (max_seq_len, d_k/2).
    apply_rotary_emb() uses them to rotate Q or K.

    Parameters
    ----------
    d_k       : int   dimension of each attention head (= d_model // n_heads)
    max_seq_len: int  maximum sequence length the tables cover.
                      Should be ≥ max number of patches in your longest window.
                      Default 2048 is safely large for all window sizes.
    base      : float frequency base (default 10000, same as sinusoidal PE)
    """
    def __init__(self, d_k: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, f"Head dim d_k={d_k} must be even for RoPE."
        self.d_k = d_k

        # θ_i = 1 / base^(2i/d_k)   shape: (d_k/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )
        # register as buffer so it moves with .to(device) and is not a parameter
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute tables up to max_seq_len
        self._build_tables(max_seq_len)

    def _build_tables(self, seq_len: int):
        """Build cos/sin tables of shape (seq_len, d_k/2)."""
        t     = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)   # (seq_len, d_k/2)
        self.register_buffer("cos_table", freqs.cos(), persistent=False)
        self.register_buffer("sin_table", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos and sin tables for positions 0 … seq_len-1.
        If seq_len exceeds the precomputed length the tables are rebuilt.
        """
        if seq_len > self.cos_table.shape[0]:
            self._build_tables(seq_len)
        return self.cos_table[:seq_len], self.sin_table[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Given x of shape (…, d_k), split into two halves and return:
        [-x_second_half, x_first_half]
    This is the standard "rotate_half" trick used in the RoPE paper.
    It is equivalent to rotating each (x_{2i}, x_{2i+1}) pair by 90°,
    which combined with the cos/sin scaling produces the full rotation.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x:   torch.Tensor,    # (B, n_heads, seq_len, d_k)
    cos: torch.Tensor,    # (seq_len, d_k//2)
    sin: torch.Tensor,    # (seq_len, d_k//2)
) -> torch.Tensor:
    """
    Apply RoPE rotation to tensor x.

    The rotation for position m and dimension pair i is:
        x'_{2i}   = x_{2i}   · cos(m·θ_i) − x_{2i+1} · sin(m·θ_i)
        x'_{2i+1} = x_{2i}   · sin(m·θ_i) + x_{2i+1} · cos(m·θ_i)

    This is equivalent to:
        x' = x · cos_expanded + rotate_half(x) · sin_expanded

    where cos_expanded and sin_expanded tile the half-dim tables to full d_k.

    Parameters
    ----------
    x   : (B, n_heads, seq_len, d_k) — the Q or K tensor to rotate
    cos : (seq_len, d_k//2)          — cosine table
    sin : (seq_len, d_k//2)          — sine table

    Returns
    -------
    Rotated tensor of same shape as x.
    """
    d_k      = x.shape[-1]
    seq_len  = x.shape[-2]

    # Tile cos/sin from (seq_len, d_k//2) → (seq_len, d_k) by repeating
    # each value twice: [cos_0, cos_0, cos_1, cos_1, …]
    cos_full = cos[:seq_len].repeat_interleave(2, dim=-1)  # (seq_len, d_k)
    sin_full = sin[:seq_len].repeat_interleave(2, dim=-1)  # (seq_len, d_k)

    # Broadcast to (B, n_heads, seq_len, d_k)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)

    return x * cos_full + rotate_half(x) * sin_full


# =============================================================================
#  SECTION 2 — RoPE-AWARE ATTENTION
# =============================================================================

class RoPESparseMultiheadAttention(nn.Module):
    """
    Replaces ProbSparseMultiheadAttention from your notebook.

    Differences from the original:
      • Applies RoPE to Q and K after projection, before ProbAttention.
      • SinusoidalPositionEncoding is NOT needed alongside this class —
        positional information is encoded entirely within this attention.
      • All other logic (projection, ProbAttention, reshape) is unchanged.

    Parameters
    ----------
    d_model     : int   total model dimension
    nhead       : int   number of attention heads
    dropout     : float attention dropout
    batch_first : bool  expects (B, L, d_model) input when True
    max_seq_len : int   maximum sequence length for RoPE tables (default 2048)
    rope_base   : float RoPE frequency base (default 10000)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 batch_first: bool = True,
                 max_seq_len: int = 2048, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model={d_model} must be divisible by nhead={nhead}"
        assert (d_model // nhead) % 2 == 0, \
            f"Head dim d_k={d_model // nhead} must be even for RoPE"

        self.d_model     = d_model
        self.nhead       = nhead
        self.d_k         = d_model // nhead
        self.batch_first = batch_first

        # Linear projections (same as original)
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ProbAttention (same as original)
        self.attn = ProbAttention(
            mask_flag        = False,
            factor           = 5,
            scale            = None,
            attention_dropout= dropout,
            output_attention = True,
        )

        # RoPE tables
        self.rope = RotaryEmbedding(
            d_k         = self.d_k,
            max_seq_len = max_seq_len,
            base        = rope_base,
        )

    def forward(self, query, key, value, **kwargs):
        """
        query, key, value : (B, L, d_model)   [batch_first=True]
        """
        B, L, _ = query.size()

        # ── Project to multi-head Q, K, V ─────────────────────────────────
        # Shape after view + transpose: (B, nhead, L, d_k)
        Q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_proj(key  ).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # ── Apply RoPE to Q and K — NOT to V ──────────────────────────────
        # RoPE encodes position purely in the similarity scores (Q·Kᵀ).
        # Rotating V would corrupt the attended values with no benefit.
        cos, sin = self.rope(L)
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # ── ProbAttention expects (B, H, L, D) — same as original ─────────
        out, _ = self.attn(Q, K, V)   # (B, H, L, d_k)

        # ── Reshape and project back to (B, L, d_model) ───────────────────
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# =============================================================================
#  SECTION 3 — RoPE ENCODER LAYER
# =============================================================================

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")


class TransformerRoPEEncoderLayer(nn.Module):
    """
    Identical to TransformerBatchNormEncoderLayer from your notebook, with
    one change: self.self_attn is now RoPESparseMultiheadAttention instead
    of ProbSparseMultiheadAttention.

    BatchNorm is preserved (same as original) because it outperformed
    LayerNorm on your CMAPSS benchmarks. Only the positional encoding
    mechanism changes.

    Parameters
    ----------
    d_model        : int   feature dimension
    nhead          : int   number of attention heads
    dim_feedforward: int   FFN hidden dimension
    dropout        : float dropout
    activation     : str   'relu' or 'gelu'
    max_seq_len    : int   RoPE table length (default 2048)
    rope_base      : float RoPE frequency base (default 10000)
    """
    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 activation: str = "relu",
                 max_seq_len: int = 2048, rope_base: float = 10000.0):
        super().__init__()

        # ── RoPE attention (replaces ProbSparseMultiheadAttention) ────────
        self.self_attn = RoPESparseMultiheadAttention(
            d_model     = d_model,
            nhead       = nhead,
            dropout     = dropout,
            batch_first = True,
            max_seq_len = max_seq_len,
            rope_base   = rope_base,
        )

        # ── FFN (unchanged from original) ─────────────────────────────────
        self.linear1  = Linear(d_model, dim_feedforward)
        self.dropout  = Dropout(dropout)
        self.linear2  = Linear(dim_feedforward, d_model)

        # ── BatchNorm (unchanged from original) ───────────────────────────
        self.norm1    = BatchNorm1d(d_model, eps=1e-5)
        self.norm2    = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, is_causal=None,
                src_key_padding_mask=None):
        """
        src shape convention is the same as the original:
          (seq_len, batch_size, d_model)  [batch_first=False for TransformerEncoder]

        The internal self_attn call converts to batch-first internally.
        """
        # TransformerEncoder passes (seq_len, B, d_model); self_attn expects
        # batch_first, so we transpose before the call and back after.
        src_bf = src.permute(1, 0, 2)   # (B, seq_len, d_model)

        src2_bf = self.self_attn(
            src_bf, src_bf, src_bf,
            attn_mask          = src_mask,
            key_padding_mask   = src_key_padding_mask,
        )                                # returns (B, seq_len, d_model)

        # Convert back to (seq_len, B, d_model) for the residual add
        src2 = src2_bf.permute(1, 0, 2)

        # Residual + BatchNorm  (same permute trick as original)
        src = src + self.dropout1(src2)          # (seq, B, d_model)
        src = src.permute(1, 2, 0)               # (B, d_model, seq)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)               # (seq, B, d_model)

        # FFN + Residual + BatchNorm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = src + self.dropout2(src2)          # (seq, B, d_model)
        src  = src.permute(1, 2, 0)               # (B, d_model, seq)
        src  = self.norm2(src)
        src  = src.permute(2, 0, 1)               # (seq, B, d_model)
        return src


# =============================================================================
#  SECTION 4 — RoPE TEMPORAL ENCODER
# =============================================================================

class PatchTSTEncoderRoPE(nn.Module):
    """
    Channel-Independent Transformer over patches — RoPE variant.

    Identical to PatchTSTEncoder EXCEPT:
      • Uses TransformerRoPEEncoderLayer instead of
        TransformerBatchNormEncoderLayer.
      • SinusoidalPositionEncoding is REMOVED.  RoPE inside each attention
        head replaces it entirely.  Adding sinusoidal PE on top of RoPE
        would be redundant and would reintroduce the embedding-corruption
        problem that RoPE is designed to avoid.

    Forward pass shape: x (B, C, L) → (B, N, d_model)
    """
    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        n_layers:  int,
        d_ff:      int,
        dropout:   float,
        patch_len: int,
        stride:    int,
        use_batchnorm_out: bool = False,
        max_seq_len:       int  = 2048,
        rope_base:         float= 10000.0,
    ):
        super().__init__()

        # InstanceNorm per channel per sample (unchanged)
        self.inst_norm = nn.InstanceNorm1d(1, affine=False, eps=1e-6)

        # Patch embedding (unchanged)
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len, stride=stride, d_model=d_model
        )

        # ── RoPE encoder layer — SinusoidalPositionEncoding is NOT added ──
        encoder_layer = TransformerRoPEEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",
            max_seq_len     = max_seq_len,
            rope_base       = rope_base,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # No self.pos_enc — RoPE replaces it

        # Output normalisation (unchanged)
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, N, d_model)"""
        B, C, L = x.shape

        # InstanceNorm per channel per sample
        x = x.reshape(B * C, 1, L)
        x = self.inst_norm(x)            # (B*C, 1, L)
        x = x.squeeze(1)                 # (B*C, L)

        # Patchify + embed
        tokens = self.patch_embed(x)     # (B*C, N, d_model)

        # ── No positional encoding added here — RoPE handles position ─────
        # TransformerEncoder expects (seq, B*C, d_model)
        tokens_t = tokens.permute(1, 0, 2)          # (N, B*C, d_model)
        enc_t    = self.encoder(tokens_t)            # (N, B*C, d_model)
        enc      = enc_t.permute(1, 0, 2)            # (B*C, N, d_model)

        # Channel mean-pool
        BxC, N, D = enc.shape
        enc = enc.view(B, C, N, D).mean(dim=1)       # (B, N, d_model)

        # Output norm
        if self.use_bn:
            enc = enc.transpose(1, 2)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)
        else:
            enc = self.ln_out(enc)

        return enc  # (B, N, d_model)


# =============================================================================
#  SECTION 5 — RoPE SENSOR-CHANNEL ENCODER
# =============================================================================

class SensorChannelTransformerEncoderRoPE(nn.Module):
    """
    Cross-sensor attention encoder — RoPE variant.

    Identical to SensorChannelTransformerEncoder EXCEPT:
      • Uses TransformerRoPEEncoderLayer.
      • SinusoidalPositionEncoding is REMOVED.

    Forward pass shape: x (B, C, L) → (B, L*N_patch, d_model)
    """
    def __init__(
        self,
        C: int, L: int,
        patch_len: int, stride: int,
        d_model:        int   = 128,
        n_heads:        int   = 8,
        num_layers:     int   = 4,
        dim_feedforward:int   = 512,
        dropout:        float = 0.1,
        use_batchnorm_out: bool  = False,
        max_seq_len:    int   = 2048,
        rope_base:      float = 10000.0,
    ):
        super().__init__()
        self.C = C
        self.L = L

        # Patch embedding along the sensor dimension (unchanged)
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len, stride=stride, d_model=d_model
        )

        # ── RoPE encoder — no SinusoidalPositionEncoding ──────────────────
        rope_layer = TransformerRoPEEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            max_seq_len     = max_seq_len,
            rope_base       = rope_base,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            rope_layer, num_layers=num_layers
        )

        # No self.pos_encoder — RoPE replaces it

        # Output normalisation (unchanged)
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, L*N_patch, d_model)"""
        B, C, L = x.shape
        assert C == self.C and L == self.L, \
            f"Shape mismatch: expected (*, {self.C}, {self.L}), got (*, {C}, {L})"

        # Rearrange to (B*L, C)
        x = x.permute(0, 2, 1).reshape(B * L, C)    # (B*L, C)

        # Patch along sensor dimension
        tokens     = self.patch_embed(x)              # (B*L, N_patch, d_model)
        num_patches= tokens.size(1)

        # Restore structure and merge time × patch axes
        tokens = tokens.view(B, L, num_patches, -1)   # (B, L, N, d_model)
        tokens = tokens.view(B, L * num_patches, -1)  # (B, L*N, d_model)

        # ── No positional encoding added — RoPE handles it ────────────────
        # TransformerEncoder expects (seq, B, d_model)
        tokens_t = tokens.permute(1, 0, 2)            # (L*N, B, d_model)
        enc_t    = self.transformer_encoder(tokens_t) # (L*N, B, d_model)
        enc      = enc_t.permute(1, 0, 2)             # (B, L*N, d_model)

        # Output norm (unchanged)
        if self.use_bn:
            enc = enc.transpose(1, 2)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)
        else:
            enc = self.ln_out(enc)

        return enc   # (B, L*N_patch, d_model)


# =============================================================================
#  SECTION 6 — FUSION HEAD (unchanged from original)
# =============================================================================

class FusionHeadRoPE(nn.Module):
    """
    Identical to FusionHead from your notebook.
    Reproduced here so this file is self-contained and does not depend on the
    notebook kernel having FusionHead defined.
    """
    def __init__(self, d_model_t: int, d_model_c: int,
                 head_hidden: Optional[int] = None,
                 dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.proj_t = (nn.Identity() if d_model_t == d_model_c
                       else nn.Linear(d_model_t, d_model_c))
        self.d_model = d_model_c
        self.norm    = nn.BatchNorm1d(self.d_model)
        assert pooling in ("mean", "cls")
        self.pooling = pooling
        hid = head_hidden if head_hidden is not None else d_model_c
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, temporal_out: torch.Tensor,
                channel_out: torch.Tensor) -> torch.Tensor:
        t = self.proj_t(temporal_out)          # (B, N_t, d_model)
        c = channel_out                         # (B, N_c, d_model)
        p = torch.cat([t, c], dim=1)           # (B, N_t+N_c, d_model)
        p = p.permute(0, 2, 1)                 # (B, d_model, seq)
        p = self.norm(p)
        p = p.permute(0, 2, 1)                 # (B, seq, d_model)
        pooled = p.mean(dim=1) if self.pooling == "mean" else p[:, 0, :]
        return self.mlp(pooled)                # (B, 1)


# =============================================================================
#  SECTION 7 — COMPLETE RoPE RUL MODEL
# =============================================================================

class PatchTST_RUL_RoPE_Model(nn.Module):
    """
    PatchTST Dual-Encoder RUL model with Rotary Positional Embeddings.

    Architecture is identical to PatchTST_RUL_Model EXCEPT:
      • PatchTSTEncoder      → PatchTSTEncoderRoPE
      • SensorChannelTransformerEncoder → SensorChannelTransformerEncoderRoPE
      • SinusoidalPositionEncoding is removed from both encoders
      • RoPE is applied to Q and K inside every attention head

    FusionHead is unchanged.

    Extra parameters vs PatchTST_RUL_Model
    ----------------------------------------
    max_seq_len : int   maximum RoPE table length (default 2048)
    rope_base   : float RoPE frequency base (default 10000)
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
        head_hidden: Optional[int] = None,
        use_bn_temporal: bool  = True,
        use_bn_channel:  bool  = True,
        max_seq_len:     int   = 2048,
        rope_base:       float = 10000.0,
    ):
        super().__init__()

        self.temporal_encoder = PatchTSTEncoderRoPE(
            d_model          = d_model_t,
            n_heads          = n_heads_t,
            n_layers         = n_layers_t,
            d_ff             = d_ff_t,
            dropout          = dropout_t,
            patch_len        = patch_len_t,
            stride           = stride_t,
            use_batchnorm_out= use_bn_temporal,
            max_seq_len      = max_seq_len,
            rope_base        = rope_base,
        )

        self.sensor_encoder = SensorChannelTransformerEncoderRoPE(
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
            max_seq_len     = max_seq_len,
            rope_base       = rope_base,
        )

        self.fusion_head = FusionHeadRoPE(
            d_model_t   = d_model_t,
            d_model_c   = d_model_c,
            head_hidden = head_hidden,
            dropout     = dropout_t,
            pooling     = "mean",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        te = self.temporal_encoder(x)    # (B, N_t, d_model_t)
        se = self.sensor_encoder(x)      # (B, N_c, d_model_c)
        y  = self.fusion_head(te, se)    # (B, 1)
        return y.squeeze(-1)             # (B,)


# =============================================================================
#  SECTION 8 — FIT FUNCTION (RoPE variant)
# =============================================================================

def fit_rope(
    train_loader,
    val_loader,
    features,
    cfg: TrainConfig,
    device,
    loss_fn=None,
    use_scheduler:      bool  = True,
    scheduler_type:     str   = "cosine_warm",
    accumulation_steps: int   = 1,
    verbose:            bool  = True,
    max_seq_len:        int   = 2048,
    rope_base:          float = 10000.0,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Mirrors fit_improved() from improve_transformer.py.
    Builds PatchTST_RUL_RoPE_Model instead of PatchTST_RUL_Model.
    All other training logic — loss, scheduler, early stopping — is identical.

    Extra Parameters
    ----------------
    max_seq_len : RoPE table size (default 2048, safely larger than any window)
    rope_base   : RoPE frequency base (default 10000, same as sinusoidal PE)
    """
    # ── Build RoPE model ──────────────────────────────────────────────────
    model = PatchTST_RUL_RoPE_Model(
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
        max_seq_len  = max_seq_len,
        rope_base    = rope_base,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  RoPE model parameters: {n_params:,}")

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
        t0       = time.time()
        tr_loss  = train_one_epoch_improved(
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
#  SECTION 9 — MAIN ORCHESTRATOR
# =============================================================================

def run_rope_scenarios(
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
    num_of_batches: int  = 1,
    window_size:    int  = 40,
    random_state:   int  = 341,
    run_ensemble:   bool = True,
    verbose:        bool = True,
    # RoPE-specific
    max_seq_len:    int   = 2048,
    rope_base:      float = 10000.0,
):
    """
    Runs Scenarios A through E with the RoPE-enabled model.

    Scenario configs, losses, schedulers, augmentation, and ensemble logic
    are all imported from improve_transformer.py.
    Only the model class changes to PatchTST_RUL_RoPE_Model.

    Parameters
    ----------
    (same as run_all_scenarios in improve_transformer.py, plus:)
    max_seq_len : RoPE table size (default 2048)
    rope_base   : RoPE frequency base (default 10000)
    """
    print("=" * 70)
    print("  RoPE TRANSFORMER EXPERIMENT")
    print(f"  Engine type   : {eng_type}")
    print(f"  Window size   : {window_size}")
    print(f"  RoPE base     : {rope_base}")
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

    mse_crit  = nn.MSELoss()
    LARGER_W  = max(window_size, 50)
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
                        f"Pass them to run_rope_scenarios()."
                    )
                print(f"\n  [Re-windowing for window_size={w} ...]")
                Xsw, ysw = create_training_sequences_sw(X, features, w)
                Xtf      = create_testing_sequences_sw(
                    X_test, features, w, num_of_batches=num_of_batches
                )
                _seq_cache[w] = (Xsw, ysw, Xtf)
        return _seq_cache[w]

    # ── Scenario registry (identical to improve_transformer.py) ──────────
    scenarios = [
        (
            "A", "RoPE + Asymmetric Loss + CosineWarm LR",
            lambda: scenario_A_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0, True, "cosine_warm", 1,
        ),
        (
            "B", "RoPE + Wider Model (d=96, L=3) + CosineWarm LR",
            lambda: scenario_B_config(features, window_size),
            window_size,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.0, True, "cosine_warm", 1,
        ),
        (
            "C", "RoPE + Larger Window + Gaussian Augmentation",
            lambda: scenario_C_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02, True, "cosine_warm", 1,
        ),
        (
            "D", "RoPE + Full Combo (Wider + Larger Window + Aug)",
            lambda: scenario_D_config(features, LARGER_W),
            LARGER_W,
            AsymmetricHuberLoss(delta=10.0, alpha_late=1.3, alpha_early=1.0),
            0.02, True, "cosine_warm", 1,
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

        model, tr_losses, vl_losses = fit_rope(
            tr_loader, vl_loader, features, cfg, device,
            loss_fn           = loss_fn,
            use_scheduler     = use_sched,
            scheduler_type    = sched_type,
            accumulation_steps= accum,
            verbose           = verbose,
            max_seq_len       = max_seq_len,
            rope_base         = rope_base,
        )

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
            f"rope_scenario{label}_{eng_type}.pt"
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

    leaderboard.sort(key=lambda r: r["RMSE"])
    best_scenario = leaderboard[0]

    # ── Scenario E: 3-seed ensemble of best config ────────────────────────
    if run_ensemble:
        print(f"\n{'─'*70}")
        print(f"  SCENARIO E: 3-seed RoPE Ensemble "
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
        per_seed_rows   = []          # NEW: capture per-seed test metrics
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

            em, _, _ = fit_rope(
                tr_e, vl_e, features, ecfg, device,
                loss_fn           = copy.deepcopy(bloss),
                use_scheduler     = bsched,
                scheduler_type    = bstype,
                accumulation_steps= baccum,
                verbose           = verbose,
                max_seq_len       = max_seq_len,
                rope_base         = rope_base,
            )
            ensemble_models.append(em)
            # NEW: per-seed test metrics for paired-t-test analysis
            _, _m, _yt, _yp = evaluate_improved(
                em, te_loader, device, nn.MSELoss())
            per_seed_rows.append({
                "variant": "RoPE", "seed": seed, "eng_type": eng_type,
                "RMSE": _m["RMSE"], "MAE": _m["MAE"],
                "R2": _m["R2"],
                "NASA": float(score_nasa(_yp - _yt)),
            })

        # NEW: persist per-seed metrics to CSV (one file per variant per engine)
        import os as _os, pandas as _pd
        _os.makedirs("per_seed_metrics", exist_ok=True)
        _pd.DataFrame(per_seed_rows).to_csv(
            f"per_seed_metrics/RoPE_{eng_type}.csv", index=False)

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
            "desc"    : f"3-seed RoPE ensemble of Scenario {best_label}",
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
    print("  RoPE LEADERBOARD  (sorted by RMSE)")
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
    save_path    = f"BEST_rope_{eng_type}.pt"

    if isinstance(overall_best["model"], list):
        states = [
            {k: v.cpu() for k, v in m.state_dict().items()}
            for m in overall_best["model"]
        ]
        torch.save(
            {"ensemble": states, "scenario": overall_best["scenario"],
             "positional_encoding": "RoPE"},
            save_path
        )
    else:
        torch.save(
            {k: v.cpu() for k, v in overall_best["model"].state_dict().items()},
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
#   from improve_transformer_rope import run_rope_scenarios
#
#   leaderboard = run_rope_scenarios(
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
#       verbose                      = True,
#   )
#
#   # ── Compare sinusoidal PE vs RoPE ────────────────────────────────────
#   from improve_transformer      import run_all_scenarios
#   from improve_transformer_rope import run_rope_scenarios
#
#   lb_sin  = run_all_scenarios(...)     # sinusoidal PE
#   lb_rope = run_rope_scenarios(...)    # RoPE
#
#   print("Sinusoidal PE best RMSE:", min(r["RMSE"] for r in lb_sin))
#   print("RoPE          best RMSE:", min(r["RMSE"] for r in lb_rope))