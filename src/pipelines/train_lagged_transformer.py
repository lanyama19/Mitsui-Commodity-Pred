"""Training entrypoint for lag-specific Transformer models."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.optim import AdamW

from src import config
from src.datasets.lagged import (
    FeatureNormalizer,
    SequenceConfig,
    create_dataloaders,
    create_datasets,
)
from src.models.multi_asset_transformer import LaggedTransformer
from src.training.trainer import Trainer, TrainerConfig


@dataclass
class ExperimentConfig:
    """High-level knobs describing a single lag training run."""

    lag: int
    seq_len: int = 192
    horizon: int = 1
    train_end: int = 1800
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    huber_delta: float = 0.2
    alpha: float = 0.3
    beta: float = 1.0
    lambda_g: float = 5e-3
    device: str = "cuda"
    patience: int = 5
    min_delta: float = 0.0
    feature_dropout: float = 0.1


def _prepare_run_directory(lag: int) -> Path:
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / "models" / f"lag_{lag}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_lagged_transformer(exp_cfg: ExperimentConfig) -> Dict[str, Any]:
    """Train a Transformer for a specific lag configuration and persist artifacts."""

    print(f"[Pipeline] Building datasets for lag={exp_cfg.lag}")
    seq_cfg = SequenceConfig(lag=exp_cfg.lag, seq_len=exp_cfg.seq_len, horizon=exp_cfg.horizon)
    train_dataset, val_dataset, normalizer, data = create_datasets(seq_cfg, train_end=exp_cfg.train_end)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=exp_cfg.batch_size)

    print(f"[Pipeline] Initialising model (targets={len(data.target_names)}, features={len(data.feature_names)})")
    model = LaggedTransformer(
        num_targets=len(data.target_names),
        feature_dim=len(data.feature_names),
        seq_len=exp_cfg.seq_len,
        horizon=exp_cfg.horizon,
        feature_dropout=exp_cfg.feature_dropout,
    )

    optimizer = AdamW(model.parameters(), lr=exp_cfg.lr, weight_decay=exp_cfg.weight_decay)
    trainer_cfg = TrainerConfig(
        epochs=exp_cfg.epochs,
        lr=exp_cfg.lr,
        weight_decay=exp_cfg.weight_decay,
        grad_clip=exp_cfg.grad_clip,
        huber_delta=exp_cfg.huber_delta,
        alpha=exp_cfg.alpha,
        beta=exp_cfg.beta,
        lambda_g=exp_cfg.lambda_g,
        device=exp_cfg.device,
        patience=exp_cfg.patience,
        min_delta=exp_cfg.min_delta,
    )
    trainer = Trainer(model=model, optimizer=optimizer, config=trainer_cfg)
    print("[Pipeline] Starting training loop")
    history, best_state, summary = trainer.fit(train_loader, val_loader)

    run_dir = _prepare_run_directory(exp_cfg.lag)
    print(f"[Pipeline] Saving artifacts to {run_dir}")

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    torch.save(best_state, run_dir / "best_model.pt")
    torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")

    with open(run_dir / "trainer_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(trainer_cfg), f, indent=2)
    with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(asdict(exp_cfg), f, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if normalizer.mean is not None and normalizer.std is not None:
        np.savez_compressed(run_dir / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)

    metadata = {
        "target_names": data.target_names,
        "feature_names": data.feature_names,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Pipeline] Training complete for lag={exp_cfg.lag}; best_epoch={summary['best_epoch']} best_metric={summary['best_metric']:.4f}")
    return history


if __name__ == "__main__":
    cfg = ExperimentConfig(lag=1)
    train_lagged_transformer(cfg)
