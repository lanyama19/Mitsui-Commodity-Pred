"""Training loop utilities with AMP support, progress reporting, and early stopping."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import LossOutput, compute_losses
from src.training.metrics import spearman_correlation


@dataclass
class TrainerConfig:
    """Configuration for the generic trainer."""

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
    patience: Optional[int] = None
    min_delta: float = 0.0
    metric: str = "val_ic"
    maximize_metric: bool = True


class Trainer:
    """Handles model training, validation, metric logging, and early stopping."""

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
        self.scaler = None
        self.autocast: Callable[[], nullcontext | torch.autocast_mode.autocast] = nullcontext
        if self.use_amp:
            self._init_amp()

    def _init_amp(self) -> None:
        """Initialise AMP helpers with backward-compatible fallbacks."""

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler()
                self.autocast = lambda: torch.amp.autocast("cuda")
                return
            except TypeError:
                pass
        if hasattr(torch.cuda, "amp"):
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
        else:
            self.scaler = None
            self.autocast = nullcontext
            self.use_amp = False

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transfer tensors in the batch to the trainer device."""

        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[LossOutput, Dict[str, torch.Tensor]]:
        """Run forward pass and compute losses inside AMP context if enabled."""

        features = batch["features"].float()
        labels = batch["labels"].float()
        mask = batch["label_mask"].float()

        with self.autocast():
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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[Dict[str, list], Dict[str, torch.Tensor], Dict[str, float]]:
        """Train the model for configured epochs and return history and best state."""

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_ic": [], "val_ic": []}
        best_metric: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        epochs_without_improve = 0

        for epoch in range(1, self.config.epochs + 1):
            print(f"[Trainer] Epoch {epoch}/{self.config.epochs} -- training phase")
            self.model.train()
            train_losses = []
            train_ics = []
            for batch in train_loader:
                batch = self._move_batch(batch)
                self.optimizer.zero_grad(set_to_none=True)
                losses, outputs = self._forward_loss(batch)

                if self.scaler is not None:
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

            epoch_train_loss = torch.stack(train_losses).mean().item()
            epoch_train_ic = torch.stack(train_ics).mean().item() if train_ics else 0.0
            history["train_loss"].append(epoch_train_loss)
            history["train_ic"].append(epoch_train_ic)
            print(f"[Trainer] Epoch {epoch} train_loss={epoch_train_loss:.4f} train_ic={epoch_train_ic:.4f}")

            if val_loader is not None:
                print(f"[Trainer] Epoch {epoch}/{self.config.epochs} -- validation phase")
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
                epoch_val_loss = torch.stack(val_losses).mean().item() if val_losses else 0.0
                epoch_val_ic = torch.stack(val_ics).mean().item() if val_ics else 0.0
                history["val_loss"].append(epoch_val_loss)
                history["val_ic"].append(epoch_val_ic)
                print(f"[Trainer] Epoch {epoch} val_loss={epoch_val_loss:.4f} val_ic={epoch_val_ic:.4f}")
                monitored_metric = epoch_val_ic if self.config.metric == "val_ic" else epoch_val_loss
            else:
                history["val_loss"].append(0.0)
                history["val_ic"].append(0.0)
                monitored_metric = epoch_train_ic if self.config.metric == "val_ic" else epoch_train_loss

            improved = False
            if best_metric is None:
                improved = True
            else:
                if self.config.maximize_metric:
                    improved = monitored_metric > best_metric + self.config.min_delta
                else:
                    improved = monitored_metric < best_metric - self.config.min_delta

            if improved:
                best_metric = monitored_metric
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"[Trainer] New best metric={best_metric:.4f} at epoch {epoch}")
            else:
                epochs_without_improve += 1
                if self.config.patience and epochs_without_improve >= self.config.patience:
                    print(f"[Trainer] Early stopping triggered at epoch {epoch}")
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            best_metric = best_metric if best_metric is not None else monitored_metric

        print("[Trainer] Training finished")
        summary = {
            "best_metric": float(best_metric) if best_metric is not None else float("nan"),
            "best_epoch": int(best_epoch),
        }
        return history, best_state, summary
