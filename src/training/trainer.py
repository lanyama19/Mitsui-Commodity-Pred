from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.losses import LossOutput, compute_losses
from src.training.metrics import spearman_correlation


@dataclass
class TrainerConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    huber_delta: float = 0.2
    alpha: float = 0.3
    beta: float = 1.0
    lambda_g: float = 1e-3
    use_amp: bool = True
    device: str = "cuda"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[LossOutput, Dict[str, torch.Tensor]]:
        features = batch["features"].float()
        labels = batch["labels"].float()
        mask = batch["label_mask"].float()

        with autocast(enabled=self.use_amp):
            outputs = self.model(features)
            losses = compute_losses(
                preds=outputs["preds"],
                scores=outputs["scores"],
                targets=labels,
                mask=mask,
                gate_penalty=self.model.feature_penalty(),
                huber_delta=self.config.huber_delta,
                alpha=self.config.alpha,
                beta=self.config.beta,
                lambda_g=self.config.lambda_g,
            )
        return losses, outputs

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, list]:
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_ic": [], "val_ic": []}
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_losses = []
            train_ics = []
            for batch in train_loader:
                batch = self._move_batch(batch)
                self.optimizer.zero_grad(set_to_none=True)
                losses, outputs = self._forward_loss(batch)

                if self.use_amp:
                    self.scaler.scale(losses.total).backward()
                    if self.config.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses.total.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                train_losses.append(losses.total.detach().cpu())
                ic = spearman_correlation(outputs["scores"].detach(), batch["labels"], batch["label_mask"])
                train_ics.append(ic.detach().cpu())

            history["train_loss"].append(torch.stack(train_losses).mean().item())
            history["train_ic"].append(torch.stack(train_ics).mean().item() if train_ics else 0.0)

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                val_ics = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._move_batch(batch)
                        losses, outputs = self._forward_loss(batch)
                        val_losses.append(losses.total.detach().cpu())
                        ic = spearman_correlation(outputs["scores"].detach(), batch["labels"], batch["label_mask"])
                        val_ics.append(ic.detach().cpu())
                history["val_loss"].append(torch.stack(val_losses).mean().item() if val_losses else 0.0)
                history["val_ic"].append(torch.stack(val_ics).mean().item() if val_ics else 0.0)
            else:
                history["val_loss"].append(0.0)
                history["val_ic"].append(0.0)

        return history
