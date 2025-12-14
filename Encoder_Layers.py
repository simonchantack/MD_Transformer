# Date: Dec-15-2025
# Author: Simon Chan Tack
# File: Encoder_Layers.py
# Code contians classes for different components of the Transformer architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn import ProbAttention
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm, MultiheadAttention, TransformerEncoderLayer


# LearnablePositionalEncoding Class
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, scale_factor=1.0, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.04, 0.04)  # uniform distribution between -0.04 and 0.04

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Fixed positional Encoding
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, max_len, scale_factor=1.0, dropout=0.1):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# tAPE positional Encoding    
class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, max_len=1024, scale_factor=1.0, dropout=0.1):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


# Function to get the positional encoder class based on the encoding type
def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == "tape":
        return tAPE

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

# Function to get the activation function based on the activation type
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# ---------------------------------------------------------------------------------
# # ProbSparseAttention Class - THE ORGINAL VERSION
# # This class implements a probabilistic sparse attention mechanism.
# class ProbSparseMultiheadAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
#         super().__init__()
#         assert d_model % nhead == 0
#         self.batch_first = batch_first
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_k = d_model // nhead

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

#         self.attn = ProbAttention(mask_flag=False, factor=5, scale=None, attention_dropout=dropout, output_attention=False)

#     def forward(self, query, key, value, **kwargs):
#         #  kwargs can include: attn_mask, key_padding_mask, need_weights, etc.
#         B, L, _ = query.size()
#         q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
#         k = self.k_proj(key).view(B, L, self.nhead, self.d_k).transpose(1, 2)
#         v = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

#         # ProbAttention expects (B, H, L, D)
#         out, _ = self.attn(q, k, v)  # ProbAttention doesn't use masks by default

#         # Reshape and project
#         out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
#         return self.out_proj(out)
    
# ---------------------------------------------------------------------------------
# ProbSparseAttention Class - MODIFIED WITH ATTN_MASK
# This class implements a probabilistic sparse attention mechanism.  ****** THIS IS THE ONE TO USE **********
class ProbSparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0
        self.batch_first = batch_first
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ProbAttention(mask_flag=False, factor=5, scale=None, attention_dropout=dropout, output_attention=True)

    def forward(self, query, key, value, **kwargs):
        #  kwargs can include: attn_mask, key_padding_mask, need_weights, etc.
        B, L, _ = query.size()
        q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        k = self.k_proj(key).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # ProbAttention expects (B, H, L, D)
        out, _ = self.attn(q, k, v)  # ProbAttention doesn't use masks by default

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)



# This is to test LN vs BN
# class ProbSparseMultiheadAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
#         super().__init__()
#         assert d_model % nhead == 0
#         self.batch_first = batch_first
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_k = d_model // nhead

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

#         self.attn = ProbAttention(mask_flag=False, factor=5, scale=None, attention_dropout=dropout, output_attention=True)

#     def forward(self, query, key, value, **kwargs):
#         #  kwargs can include: attn_mask, key_padding_mask, need_weights, etc.
#         B, L, _ = query.size()
#         q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
#         k = self.k_proj(key).view(B, L, self.nhead, self.d_k).transpose(1, 2)
#         v = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

#         # ProbAttention expects (B, H, L, D)
#         # out, _ = self.attn(q, k, v)  # ProbAttention doesn't use masks by default
#         attn_output, attn_scores = self.attn(q, k, v)

#         # # Reshape and project
#         # out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

#         # Reshape output
#         out = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
#         return self.out_proj(out), attn_scores


# # ---------------------------------------------------------------------------------
# # ProbSparseMultiheadAttention with Causal Masking
# class ProbSparseMultiheadAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, causal=True):
#         super().__init__()
#         assert d_model % nhead == 0
#         self.batch_first = batch_first
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_k = d_model // nhead
#         self.causal = causal  # NEW: enable causal attention

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

        # self.attn = ProbAttention(
        #     mask_flag=False,   # we’ll override masking manually
        #     factor=5,
        #     scale=None,
        #     attention_dropout=dropout,
        #     output_attention=False
        # )

#     def _generate_causal_mask(self, L, device):
#         """Create a causal mask so that query position i can only attend to <= i."""
#         # Shape: (L, L) with -inf for invalid connections
#         mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
#         return mask  # True where masking should occur

#     def forward(self, query, key, value, **kwargs):
#         B, L, _ = query.size()
#         q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
#         k = self.k_proj(key).view(B, L, self.nhead, self.d_k).transpose(1, 2)
#         v = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

#         # --- Apply causal mask if requested ---
#         attn_mask = None
#         if self.causal:
#             attn_mask = self._generate_causal_mask(L, query.device)  # (L, L)

#         # ProbAttention expects (B, H, L, D) and optional mask
#         out, _ = self.attn(q, k, v, attn_mask=attn_mask)

#         # Reshape and project
#         out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
#         return self.out_proj(out)    


# ---------------------------------------------------------------------------------
# # Probablistic Sparse LMHA - Latent Multi Head Attention  - version 1
# class ProbSparseMultiheadAttention(nn.Module):
#     """
#     ProbSparse Multi-Head Attention with optional Latent bottleneck.
    
#     Modes:
#       - Standard:     out = Attn(Q=X, K=X, V=X)                -> (B, L, D)
#       - Latent mode:  Z = Attn(Q=L,  K=X, V=X)  (compress to M latents)
#                       out = Attn(Q=X, K=Z, V=Z) (expand back to L)
    
#     Returns:
#       (attn_out, None) to stay compatible with code that expects (out, weights).
#     """
#     def __init__(
#         self,
#         d_model,
#         nhead,
#         dropout=0.1,
#         batch_first=True,
#         # --- NEW latent options ---
#         use_latent: bool = False,
#         num_latents: int = 64,
#         share_latents_across_heads: bool = True,
#     ):
#         super().__init__()
#         assert d_model % nhead == 0, "d_model must be divisible by nhead"
#         self.batch_first = batch_first
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_k = d_model // nhead

#         self.use_latent = use_latent
#         self.num_latents = num_latents
#         self.share_latents_across_heads = share_latents_across_heads

#         # Projections
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

#         # ProbSparse attention core (from Informer)
#         self.attn = ProbAttention(
#             mask_flag=False,
#             factor=5,
#             scale=None,
#             attention_dropout=dropout,
#             output_attention=False
#         )

#         # --- NEW: learnable latent tokens ---
#         if self.use_latent:
#             if self.share_latents_across_heads:
#                 # shared across heads: (1, M, D)
#                 self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.d_model) * 0.02)
#             else:
#                 # per-head latents in head dim: (1, H, M, d_k)
#                 self.latents = nn.Parameter(torch.randn(1, self.nhead, self.num_latents, self.d_k) * 0.02)

#     def _split_heads(self, x):
#         # x: (B, L, D) -> (B, H, L, d_k)
#         B, L, D = x.shape
#         x = x.view(B, L, self.nhead, self.d_k).transpose(1, 2)
#         return x

#     def _merge_heads(self, x):
#         # x: (B, H, L, d_k) -> (B, L, D)
#         B, H, L, d_k = x.shape
#         x = x.transpose(1, 2).contiguous().view(B, L, H * d_k)
#         return x

#     def _latent_queries(self, B, device, dtype):
#         """
#         Returns latent queries shaped for attention:
#           - shared latents: project to D, then split to (B, H, M, d_k)
#           - per-head latents: expand to (B, H, M, d_k)
#         """
#         if self.share_latents_across_heads:
#             # (1, M, D) -> (B, M, D) -> project q -> split heads
#             Lq = self.latents.to(device=device, dtype=dtype).expand(B, -1, -1)  # (B, M, D)
#             q_lat = self.q_proj(Lq)  # (B, M, D)
#             q_lat = q_lat.view(B, self.num_latents, self.nhead, self.d_k).transpose(1, 2)  # (B, H, M, d_k)
#         else:
#             # (1, H, M, d_k) -> (B, H, M, d_k)
#             q_lat = self.latents.to(device=device, dtype=dtype).expand(B, -1, -1, -1)  # (B, H, M, d_k)
#         return q_lat

#     def forward(self, query, key, value, **kwargs):
#         """
#         Inputs:
#           query, key, value: (B, L, D) (batch_first=True)
#         Kwargs (optional, forwarded to ProbAttention if supported):
#           - attn_mask: (L, L) or broadcastable to (B, H, L, L)
#           - key_padding_mask: (B, L) boolean; True to mask out padding keys
#         """
#         assert self.batch_first, "This module expects (B, L, D) tensors."

#         B, L, _ = query.size()
#         device = query.device
#         dtype = query.dtype

#         # Project and split heads for the input sequence
#         q = self._split_heads(self.q_proj(query))  # (B, H, L, d_k)
#         k = self._split_heads(self.k_proj(key))    # (B, H, L, d_k)
#         v = self._split_heads(self.v_proj(value))  # (B, H, L, d_k)

#         attn_mask = kwargs.get("attn_mask", None)
#         key_padding_mask = kwargs.get("key_padding_mask", None)

#         if not self.use_latent:
#             # Standard (ProbSparse) self-attention
#             out, _ = self.attn(q, k, v, attn_mask=attn_mask) # , key_padding_mask=key_padding_mask
#             out = self._merge_heads(out)                 # (B, L, D)
#             return self.out_proj(out), None

#         # --- Latent Attention path ---
#         # 1) Latents cross-attend to the full sequence: Z = Attn(L, X, X)
#         q_lat = self._latent_queries(B, device, dtype)   # (B, H, M, d_k)
#         # Cross-attend: queries are latents; keys/values are inputs
#         Z, _ = self.attn(q_lat, k, v, attn_mask=None)  # (B, H, M, d_k)  , key_padding_mask=key_padding_mask

#         # 2) Inputs attend to latent summaries: out = Attn(X, Z, Z)
#         out, _ = self.attn(q, Z, Z, attn_mask=attn_mask)  # (B, H, L, d_k)  , key_padding_mask=None

#         out = self._merge_heads(out)  # (B, L, D)
#         return self.out_proj(out), None


# ----------------------------------------------------------------------------------
# # # Probablistic Sparse LMHA - Latent Multi Head Attention  - version 2 CRASHES
# class ProbSparseMultiheadAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
#         super().__init__()
#         assert d_model % nhead == 0
#         self.batch_first = batch_first
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_k = d_model // nhead

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

#         # Original ProbAttention (does not support key_padding_mask)
#         self.attn = ProbAttention(mask_flag=False, factor=5, scale=None, 
#                                   attention_dropout=dropout, output_attention=False)

#     def forward(self, query, key, value, key_padding_mask=None):
#         """
#         query, key, value: (B, L, d_model)
#         key_padding_mask: (B, L) with True for padded positions
#         """
#         B, L, _ = query.size()

#         # Linear projections
#         q = self.q_proj(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
#         k = self.k_proj(key).view(B, L, self.nhead, self.d_k).transpose(1, 2)
#         v = self.v_proj(value).view(B, L, self.nhead, self.d_k).transpose(1, 2)

#         out_heads = []

#         for h in range(self.nhead):
#             q_h = q[:, h]  # (B, L, d_k)
#             k_h = k[:, h]
#             v_h = v[:, h]

#             # Apply key_padding_mask manually if provided
#             if key_padding_mask is not None:
#                 # Expand mask for attention shape: (B, 1, 1, L)
#                 mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
#                 # Convert True -> -inf, False -> 0
#                 neg_inf = torch.finfo(q_h.dtype).min
#                 v_h = v_h.clone()
#                 k_h = k_h.clone()
#                 # ProbAttention uses QK^T internally, so we will set corresponding K positions to -inf later
#                 # We'll pass a mask tensor directly to ProbAttention
#                 self.attn.mask_flag = True
#                 attn_mask = mask  # ProbAttention expects (B, 1, L, L)
#             else:
#                 attn_mask = None
#                 self.attn.mask_flag = False

#             out_h, _ = self.attn(q_h.unsqueeze(1), k_h.unsqueeze(1), v_h.unsqueeze(1), attn_mask=attn_mask)
#             # Remove extra dimension
#             out_heads.append(out_h.squeeze(1))

#         # Concatenate heads and project
#         out = torch.cat(out_heads, dim=-1)  # (B, L, d_model)
#         return self.out_proj(out)

# # ----------------------------------------------------------------------------------

# USE THIS !!!
# TransformerBatchNormEncoderLayer Class
# This transformer encoder layer block is made up of self-attn and feedforward network.
# It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()

        # MultiheadAttention
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # ProbSparseMultiheadAttention
        self.self_attn = ProbSparseMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, is_causal = None,
                src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class. 
             - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
        - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
        - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
        - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
        """
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]   # src2 shape = (seq, batch, feature) 
                              # because batch false is set to false because we would have batched it before calling self-attention function
        
        
        # Add dropout and add x to output of multihead attention for residual...
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        # Reshape to be able to do batch norm and not layer norm
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # Perform batchnorm
        src = self.norm1(src)
        # restore shape back
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        # apply linear layers 1 and 2 to src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add x/src for residual efect and normalize again
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src 
        
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
#           DECODER STUFF - Not used in final architecture only for exprimenting

# ---------------------------------------------------
# Decoder layer: BatchNorm + ProbSparse + batch_first
# ---------------------------------------------------
class TransformerBatchNormDecoderLayer(nn.Module):
    """
    Decoder layer with:
      - self-attn on tgt (learnable queries)
      - cross-attn on memory (concat of temporal + sensor tokens)
      - FFN
      - BatchNorm (applied across (B, E, S) like your encoder)
    Uses ProbSparseMultiheadAttention for both attentions.
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="gelu"):
        super().__init__()
        # self-attn (queries)
        self.self_attn = ProbSparseMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # cross-attn (queries attend to encoder memory)
        self.cross_attn = ProbSparseMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # BatchNorms (apply by transposing to (B, E, S))
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm3 = nn.BatchNorm1d(d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation must be relu or gelu")

    def _bn(self, x, bn):
        # x: (B, S, E) -> BN over E with BatchNorm1d
        x = x.transpose(1, 2)   # (B, E, S)
        x = bn(x)
        x = x.transpose(1, 2)   # (B, S, E)
        return x

    def forward(
        self,
        tgt,                 # (B, Tq, E)
        memory,              # (B, Sm, E)
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self-attention on queries
        q = k = tgt
        tgt2 = self.self_attn(q, k, tgt)            # (B, Tq, E)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self._bn(tgt, self.norm1)

        # Cross-attention: queries over encoder memory
        tgt2 = self.cross_attn(tgt, memory, memory) # (B, Tq, E)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self._bn(tgt, self.norm2)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self._bn(tgt, self.norm3)
        return tgt

# ----------------------------------------
# Stack of decoder layers + learnable query
# ----------------------------------------
class DualAspectDecoder(nn.Module):
    """
    A small decoder stack that uses a learnable query token (or tokens)
    and cross-attends to the fused encoder memory. Returns (B, d_model).
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_queries: int = 1,
        add_pos_to_query: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))

        # optional learned positional embeddings for queries
        self.add_pos_to_query = add_pos_to_query
        if add_pos_to_query:
            self.query_pos = nn.Parameter(torch.randn(1, num_queries, d_model))

        self.layers = nn.ModuleList([
            TransformerBatchNormDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu"
            ) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, memory):  # memory: (B, S_enc, d_model)
        B = memory.size(0)
        tgt = self.query_embed.expand(B, self.num_queries, -1)  # (B, Tq, d_model)
        if self.add_pos_to_query:
            tgt = tgt + self.query_pos

        for layer in self.layers:
            tgt = layer(tgt, memory)

        tgt = self.norm_out(tgt)             # (B, Tq, d_model)
        # if multiple queries, average; or pick first
        pooled = tgt.mean(dim=1)             # (B, d_model)
        return pooled

# -------------------------------------------------------------------------------------------------------
# Residual Block 1D Module
# This is a residual block for 1D convolutional networks. It consists of two convolutional layers with batch normalization and ReLU activation.
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Project input if dimensions differ
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# To test LN vs BN
import copy
class TransformerEncoderWithAttention(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None):
        attn_maps = []
        output = src
        for layer in self.layers:
            output, attn = layer(output, src_key_padding_mask=src_key_padding_mask)
            attn_maps.append(attn)
        return output, attn_maps




# TSTransformerEncoderClassiregressor Class
# This is the main class that implements the transformer encoder for classification/regression tasks.
# Bring your data in normally (NxTxD), it is in here you permutate it85
# Call this class and specify whether you want pytorch's TransformerEncoder with layer norm or the one we created above with batch norm
# Watch out for your src_mask shape!!!
class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    Transfomations can either be linear, 1D-CNN or no transform
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, transform='linear'):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        # Linear embedding transformation
        self.project_inp = nn.Linear(feat_dim, d_model)  # Linear input transformation for embedding

        # MLP embedding transformation
        self.mlp_proj = nn.Sequential(
                        nn.Linear(feat_dim, feat_dim * 2),
                        nn.ReLU(),
                        nn.Linear(feat_dim * 2, d_model)
                    )  # MLP input transformation for embedding

        # CNN Model - 1D Convnet - Option 1
        self.CNN = nn.Sequential(
            nn.Conv1d(feat_dim, d_model, kernel_size=3, padding="same"),   # feat_dim = 13
            nn.ReLU(),
            nn.Dropout(0.4))
        
        # # CNN Model - 1D Convnet - Option 2
        # # With residual connections and batch normalization
        # self.CNN = nn.Sequential(
        #     ResidualBlock1D(feat_dim, 36, kernel_size=3, dropout=0.1),
        #     ResidualBlock1D(36, 72, kernel_size=3, dropout=0.1),
        #     ResidualBlock1D(72, 144, kernel_size=3, dropout=0.1),
        #     nn.Conv1d(144, d_model, kernel_size=3, padding="same"),
        #     nn.ReLU(),
        #     nn.Dropout(0.4)
        # )


        # LSTM-based embedding
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=d_model,
            num_layers=3,
            dropout=dropout * (1.0 - freeze),
            batch_first=True,
            bidirectional=False
        )

        # Positional encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model=self.d_model, max_len=self.max_len, dropout=dropout*(1.0 - freeze))

        # Transformer Encoder
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)   #         

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks, transform):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        if transform == 'linear':
          # Linear input transformation
          inp = self.project_inp(inp) * math.sqrt(
              self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        elif transform == 'MLP':
            # MLP input transformation            
            inp = self.mlp_proj(inp) * math.sqrt(
                self.d_model)

        elif transform == '1D-CNN':
          # 1D-CNN input transformation
          inp = X.permute(0, 2, 1)
          inp = self.CNN(inp)
          inp = inp.permute(2,0,1)

        elif transform == 'LSTM':
            lstm_out, _ = self.lstm(X)  # (batch_size, seq_len, d_model)
            inp = lstm_out.permute(1, 0, 2)  # (seq_len, batch_size, d_model

        else:
            raise ValueError(f"Unknown transform type: {transform}")


        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)   
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model) This particular permutation is to make sure we are working with NxTxD to compute Y
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings, now this is where you zero it out not actually remove d spaces
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes) (No activation, it is in crossentropy loss)

        return output