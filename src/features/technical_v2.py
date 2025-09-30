from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data.targets import TargetSpec
from src.features import technical as _technical_base


def _nav_from_returns(returns: pd.Series, start: float, mode: str) -> np.ndarray:
    """Build a synthetic net-asset-value series from return differentials."""
    values = returns.to_numpy(dtype=float, copy=True)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0)
    nav = np.empty_like(values)
    nav[0] = start
    if mode == "multiplicative":
        for idx in range(1, values.size):
            nav[idx] = nav[idx - 1] * (1.0 + values[idx])
    elif mode == "additive":
        for idx in range(1, values.size):
            nav[idx] = nav[idx - 1] + values[idx]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return nav


def build_continuous_nav_frame(target_frame: pd.DataFrame, specs: Iterable[TargetSpec]) -> pd.DataFrame:
    """Return synthetic NAV curves for each target based on return differentials."""
    specs = list(specs)
    if not specs:
        return pd.DataFrame(index=target_frame.index)

    returns = target_frame.diff().fillna(0.0)
    nav_data: dict[str, np.ndarray] = {}
    for spec in specs:
        series = returns[spec.name]
        if spec.is_single_asset:
            nav_series = _nav_from_returns(series, start=0.0, mode="additive")
        else:
            nav_series = _nav_from_returns(series, start=1.0, mode="multiplicative")
        nav_data[spec.name] = nav_series

    nav_frame = pd.DataFrame(nav_data, index=target_frame.index)
    return nav_frame


def compute_continuous_technical_features(
    target_frame: pd.DataFrame,
    specs: Iterable[TargetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute TA-Lib indicators on synthetic NAV curves, returning features and NAVs."""
    nav_frame = build_continuous_nav_frame(target_frame, specs)
    if nav_frame.empty:
        return pd.DataFrame(index=target_frame.index), nav_frame

    feature_frames: list[pd.DataFrame] = []
    for column in nav_frame.columns:
        values = nav_frame[column].to_numpy(dtype=float, copy=True)
        technical = _technical_base._technical_for_series(values, nav_frame.index)
        technical.columns = pd.MultiIndex.from_product([[column], technical.columns])
        feature_frames.append(technical)

    technical_features = pd.concat(feature_frames, axis=1)
    return technical_features, nav_frame
