from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config
from src.data.cleaning import fill_dataframe_with_trend
from src.data.loading import load_price_data

DEFAULT_SKIP_COLUMNS = {"date_id", "is_scored"}


def load_and_clean_prices(
    split: str,
    window: int = 5,
    skip_columns: Iterable[str] | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    frame = load_price_data(split=split)
    skip = set(DEFAULT_SKIP_COLUMNS)
    if skip_columns:
        skip.update(skip_columns)
    cleaned = fill_dataframe_with_trend(frame, window=window, skip_columns=skip)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_pickle(save_path)
    return cleaned


def build_all_cleaned(window: int = 5) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split in ("train", "test", "all"):
        path = config.OUTPUT_DIR / f"clean_{split}.pkl"
        outputs[split] = load_and_clean_prices(split=split, window=window, save_path=path)
    return outputs
