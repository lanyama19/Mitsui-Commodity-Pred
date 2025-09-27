from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.targets import TargetSpec, load_target_specs


@dataclass
class SequenceConfig:
    lag: int
    seq_len: int = 192
    horizon: int = 1
    min_valid_targets: int = 1
    feature_path: Path = config.OUTPUT_DIR / "features" / "all.pkl"
    labels_path: Path = config.TRAIN_LABELS_PATH


@dataclass
class SequenceData:
    features: np.ndarray  # shape: [T, N, F]
    labels: np.ndarray    # shape: [T, N]
    label_mask: np.ndarray  # shape: [T, N]
    date_index: np.ndarray  # shape: [T]
    target_names: List[str]
    feature_names: List[str]


class FeatureNormalizer:
    def __init__(self, eps: float = 1e-6) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = eps

    def fit(self, data: np.ndarray, indices: Sequence[int]) -> None:
        subset = data[indices]
        self.mean = subset.mean(axis=0, keepdims=True)
        self.std = subset.std(axis=0, keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (data - self.mean) / self.std

    def to_dict(self) -> Dict[str, np.ndarray]:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted")
        return {"mean": self.mean, "std": self.std}


class LaggedSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: SequenceData,
        indices: Sequence[int],
        seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])
        start = t - self.seq_len + 1
        features = self.data.features[start : t + 1]
        labels = self.data.labels[t]
        mask = self.data.label_mask[t]
        date_id = int(self.data.date_index[t])
        item = {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(mask),
            "date_id": torch.tensor(date_id, dtype=torch.int32),
        }
        if self.device is not None:
            item = {k: v.to(self.device) for k, v in item.items()}
        return item


def _select_targets_by_lag(specs: Sequence[TargetSpec], lag: int) -> list[TargetSpec]:
    return [spec for spec in specs if spec.lag == lag]


def _load_feature_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    frame = pd.read_pickle(path)
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in feature frame")
    return frame


def _load_labels(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    labels = pd.read_csv(path)
    return labels.set_index("date_id")


def _to_panel(frame: pd.DataFrame, targets: Sequence[str]) -> tuple[np.ndarray, List[str], List[str]]:
    feature_names = list(frame.columns.get_level_values(1).unique())
    reordered_columns = pd.MultiIndex.from_product([targets, feature_names])
    subset = frame.reindex(columns=reordered_columns)
    if subset.isna().any().any():
        raise ValueError("Feature frame contains NaNs after reindexing; check target coverage")
    arr = subset.to_numpy(dtype=np.float32)
    arr = arr.reshape(len(frame.index), len(targets), len(feature_names))
    return arr, list(targets), feature_names


def _screen_features(features: np.ndarray, feature_names: List[str], threshold: float = 1e-6) -> tuple[np.ndarray, List[str]]:
    # features shape [T, N, F]
    variance = features.var(axis=0).mean(axis=0)
    keep_mask = variance > threshold
    if keep_mask.sum() == 0:
        raise ValueError("All features were filtered out; lower the threshold")
    filtered = features[:, :, keep_mask]
    filtered_names = [name for name, keep in zip(feature_names, keep_mask) if keep]
    return filtered, filtered_names


def build_sequence_data(config: SequenceConfig) -> SequenceData:
    specs = load_target_specs()
    lagged_specs = _select_targets_by_lag(specs, config.lag)
    if not lagged_specs:
        raise ValueError(f"No targets found for lag={config.lag}")
    target_names = [spec.name for spec in lagged_specs]

    feature_frame = _load_feature_panel(config.feature_path)
    features, ordered_targets, feature_names = _to_panel(feature_frame, target_names)

    labels = _load_labels(config.labels_path)
    labels = labels[target_names]
    shift = config.lag + config.horizon
    labels = labels.shift(-shift)

    label_mask = ~labels.isna()
    labels = labels.fillna(0.0)

    align_index = labels.index.to_numpy()
    valid_rows = label_mask.any(axis=1)

    features = features[valid_rows]
    labels_arr = labels.to_numpy(dtype=np.float32)[valid_rows]
    mask_arr = label_mask.to_numpy(dtype=np.float32)[valid_rows]
    date_index = align_index[valid_rows]

    features, filtered_feature_names = _screen_features(features, feature_names)

    return SequenceData(
        features=features,
        labels=labels_arr,
        label_mask=mask_arr,
        date_index=date_index,
        target_names=ordered_targets,
        feature_names=filtered_feature_names,
    )


def train_valid_split(
    data: SequenceData,
    seq_len: int,
    train_end: int,
    min_valid_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_index = len(data.date_index)
    valid_mask = data.label_mask.sum(axis=1) >= min_valid_targets
    valid_indices = np.arange(max_index)[valid_mask]
    valid_indices = valid_indices[valid_indices >= seq_len - 1]

    train_idx = valid_indices[data.date_index[valid_indices] <= train_end]
    val_idx = valid_indices[data.date_index[valid_indices] > train_end]
    return train_idx, val_idx


def create_datasets(
    config: SequenceConfig,
    train_end: int,
    normalizer: FeatureNormalizer | None = None,
) -> tuple[LaggedSequenceDataset, LaggedSequenceDataset, FeatureNormalizer, SequenceData]:
    data = build_sequence_data(config)
    train_idx, val_idx = train_valid_split(
        data=data,
        seq_len=config.seq_len,
        train_end=train_end,
        min_valid_targets=config.min_valid_targets,
    )

    if not len(train_idx):
        raise ValueError("Training indices empty; adjust train_end or seq_len")

    if normalizer is None:
        normalizer = FeatureNormalizer()
    normalizer.fit(data.features, train_idx)
    data.features = normalizer.transform(data.features)

    train_dataset = LaggedSequenceDataset(data, train_idx, seq_len=config.seq_len)
    val_dataset = LaggedSequenceDataset(data, val_idx, seq_len=config.seq_len)
    return train_dataset, val_dataset, normalizer, data


def create_dataloaders(
    train_dataset: LaggedSequenceDataset,
    val_dataset: LaggedSequenceDataset,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
