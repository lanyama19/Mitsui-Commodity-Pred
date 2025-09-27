from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src import config

Split = Literal["train", "test", "all"]


def load_price_data(split: Split = "train", columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Return price table for requested split."""
    if split not in {"train", "test", "all"}:
        raise ValueError(f"Unsupported split: {split}")

    if split == "train":
        frame = pd.read_csv(config.TRAIN_PATH, usecols=columns)
    elif split == "test":
        frame = pd.read_csv(config.TEST_PATH, usecols=columns)
    else:
        train = pd.read_csv(config.TRAIN_PATH, usecols=columns)
        test = pd.read_csv(config.TEST_PATH, usecols=columns)
        common_cols = [c for c in train.columns if c in test.columns]
        frame = pd.concat([train[common_cols], test[common_cols]], ignore_index=True)

    return frame


def load_train_labels(columns: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_LABELS_PATH, usecols=columns)


def load_target_pairs() -> pd.DataFrame:
    return pd.read_csv(config.TARGET_PAIRS_PATH)


def load_lagged_test_labels(lag: int) -> pd.DataFrame:
    path = config.LAGGED_TEST_LABELS_DIR / f"test_labels_lag_{lag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
