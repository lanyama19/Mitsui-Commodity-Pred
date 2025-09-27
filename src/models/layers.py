"""Lightweight Transformer layer primitives (feature gate, Time2Vec, attention blocks)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGate(nn.Module):
    """Learnable feature mask that scales each input channel between 0 and 1."""

    def __init__(self, feature_dim: int, init_value: float = 0.5) -> None:
        super().__init__()
        init = torch.full((feature_dim,), float(init_value)).float()
        self.logits = nn.Parameter(torch.logit(init.clamp(1e-4, 1 - 1e-4)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gating to input tensor and return gated features plus raw gate values."""

        gate = torch.sigmoid(self.logits)
        return x * gate, gate


class Time2Vec(nn.Module):
    """Time2Vec embedding that augments inputs with periodic components."""

    def __init__(self, input_dim: int = 1, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.linear = nn.Linear(input_dim, 1)
        self.freq = nn.Linear(input_dim, k)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Convert scalar timesteps into linear + sinusoidal embeddings."""

        t = t.unsqueeze(-1)
        linear_term = self.linear(t)
        periodic = torch.sin(self.freq(t))
        return torch.cat([linear_term, periodic], dim=-1)


class FeedForward(nn.Module):
    """Two-layer MLP used inside Transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feed-forward transformed tensor."""

        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention with optional FlashAttention backend."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_flash = use_flash
        self.head_dim = d_model // num_heads
        if self.head_dim * num_heads != d_model:
            raise ValueError("d_model must be divisible by num_heads")
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform attention and merge heads back to model dimension."""

        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask[:, None, None, :] if mask is not None else None
        q = q * (1.0 / math.sqrt(self.head_dim))
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn = torch.matmul(attn_weights, v)

        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(attn)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block with residual connections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention then feed-forward network with residual links."""

        attn_input = self.ln1(x)
        x = x + self.attn(attn_input, mask=mask)
        ff_input = self.ln2(x)
        x = x + self.ff(ff_input)
        return x
