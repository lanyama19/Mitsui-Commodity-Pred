from __future__ import annotations

import torch


def spearman_correlation(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    corrs = []
    for b in range(preds.shape[0]):
        valid = mask[b] > 0
        if valid.sum() < 3:
            continue
        p = preds[b, valid]
        t = targets[b, valid]
        p_rank = torch.argsort(torch.argsort(p))
        t_rank = torch.argsort(torch.argsort(t))
        p_rank = p_rank.float()
        t_rank = t_rank.float()
        cov = ((p_rank - p_rank.mean()) * (t_rank - t_rank.mean())).sum()
        denom = torch.sqrt(((p_rank - p_rank.mean()) ** 2).sum() * ((t_rank - t_rank.mean()) ** 2).sum())
        if denom > 0:
            corrs.append(cov / denom)
    if not corrs:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(corrs).mean()
