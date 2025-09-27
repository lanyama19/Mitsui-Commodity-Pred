from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.layers import FeatureGate, Time2Vec, TransformerEncoderBlock


class LaggedTransformer(nn.Module):
    def __init__(
        self,
        num_targets: int,
        feature_dim: int,
        seq_len: int,
        horizon: int = 1,
        d_model: int = 192,
        num_heads: int = 6,
        depth: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model

        self.gate = FeatureGate(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.time2vec = Time2Vec(k=4)
        self.time_proj = nn.Linear(d_model + 5, d_model)

        self.target_embedding = nn.Embedding(num_targets, d_model)

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_model * ff_multiplier,
                    dropout=dropout,
                    use_flash=use_flash_attention,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, horizon)
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, l, n, f = x.shape
        if n != self.num_targets or f != self.feature_dim:
            raise ValueError("Input shape mismatch with model configuration")

        x_flat = x.view(b * n, l, f)
        gated, gate = self.gate(x_flat)
        z = self.input_proj(gated)
        z = self.input_dropout(z)

        device = x.device
        time_idx = torch.arange(l, device=device, dtype=torch.float32)
        t2v = self.time2vec(time_idx)
        t2v = t2v.unsqueeze(0).unsqueeze(0).expand(b, n, l, -1).reshape(b * n, l, -1)

        z = torch.cat([z, t2v], dim=-1)
        z = self.time_proj(z)

        target_ids = torch.arange(n, device=device)
        target_embed = self.target_embedding(target_ids)
        target_embed = target_embed.unsqueeze(0).unsqueeze(1).expand(b, l, n, -1).reshape(b * n, l, -1)
        z = z + target_embed

        for block in self.encoder:
            z = block(z)

        h = self.final_norm(z)
        final_state = h[:, -1].view(b, n, self.d_model)

        preds = self.regression_head(final_state).squeeze(-1)
        scores = self.score_head(final_state).squeeze(-1)

        gate_vector = torch.sigmoid(self.gate.logits)

        return {
            "preds": preds,  # shape [B, N]
            "scores": scores,
            "gate": gate_vector,
            "hidden": final_state,
        }

    def feature_penalty(self) -> torch.Tensor:
        gate_vector = torch.sigmoid(self.gate.logits)
        return gate_vector.abs().sum()
