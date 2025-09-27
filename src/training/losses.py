from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    gate: torch.Tensor


def huber_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    diff = preds - targets
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=preds.device))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 / delta + linear
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def pairwise_rank_loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_losses = []
    for b in range(scores.shape[0]):
        valid = mask[b] > 0.0
        if valid.sum() < 2:
            continue
        s = scores[b, valid]
        y = targets[b, valid]
        diff_scores = s.unsqueeze(1) - s.unsqueeze(0)
        diff_targets = y.unsqueeze(1) - y.unsqueeze(0)
        weight = diff_targets.abs()
        sign = torch.sign(diff_targets)
        valid_pairs = weight > 0
        if valid_pairs.sum() == 0:
            continue
        softplus = F.softplus(-sign * diff_scores)
        loss = (softplus * weight)[valid_pairs].mean()
        batch_losses.append(loss)
    if not batch_losses:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(batch_losses).mean()


def compute_losses(
    preds: torch.Tensor,
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gate_penalty: torch.Tensor,
    huber_delta: float = 0.2,
    alpha: float = 0.3,
    beta: float = 1.0,
    lambda_g: float = 1e-3,
) -> LossOutput:
    reg = huber_loss(preds, targets, mask, delta=huber_delta)
    rank = pairwise_rank_loss(scores, targets, mask)
    gate = lambda_g * gate_penalty
    total = alpha * reg + beta * rank + gate
    return LossOutput(total=total, regression=reg, ranking=rank, gate=gate)
