from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_DTYPES = ("float32", "float64", "int32", "int64")


def fill_series_with_trend(series: pd.Series, window: int = 5) -> pd.Series:
    values = series.to_numpy(copy=True, dtype=float)
    filled = np.copy(values)
    diffs: deque[float] = deque(maxlen=window)
    last_valid = np.nan

    for idx in range(len(values)):
        val = values[idx]
        if np.isnan(val):
            if np.isnan(last_valid):
                continue
            slope = float(np.mean(diffs)) if diffs else 0.0
            filled[idx] = last_valid + slope
            last_valid = filled[idx]
        else:
            if np.isnan(last_valid):
                last_valid = val
            else:
                diffs.append(val - last_valid)
                last_valid = val
            filled[idx] = val

    first_valid_idx = np.argmax(~np.isnan(filled)) if np.any(~np.isnan(filled)) else -1
    if first_valid_idx > 0:
        filled[:first_valid_idx] = filled[first_valid_idx]

    if np.any(np.isnan(filled)):
        valid = filled[~np.isnan(filled)]
        fallback = valid[0] if valid.size else 0.0
        filled[np.isnan(filled)] = fallback

    return pd.Series(filled, index=series.index, name=series.name)


def fill_dataframe_with_trend(df: pd.DataFrame, window: int = 5, skip_columns: Iterable[str] | None = None) -> pd.DataFrame:
    skip = set(skip_columns or [])
    cleaned = df.copy()
    for column in cleaned.columns:
        if column in skip:
            continue
        series = cleaned[column]
        if pd.api.types.is_numeric_dtype(series):
            cleaned[column] = fill_series_with_trend(series, window=window)
    return cleaned


def normalize_prices(df: pd.DataFrame, price_columns: Iterable[str], reference: str = "date_id") -> pd.DataFrame:
    cleaned = df.copy()
    for column in price_columns:
        if column == reference:
            continue
        col = cleaned[column]
        mu = col.mean()
        sigma = col.std()
        if sigma == 0 or np.isnan(sigma):
            continue
        cleaned[column] = (col - mu) / sigma
    return cleaned
