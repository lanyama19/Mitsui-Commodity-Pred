from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.loading import load_target_pairs


@dataclass(frozen=True)
class TargetSpec:
    name: str
    lag: int
    terms: tuple[tuple[str, float], ...]

    @property
    def is_single_asset(self) -> bool:
        return len(self.terms) == 1 and self.terms[0][1] == 1.0


def _parse_pair_expression(expr: str) -> tuple[tuple[str, float], ...]:
    parts: list[str] = []
    operators: list[str] = []
    buf = []
    i = 0
    length = len(expr)
    while i < length:
        if i + 2 < length and expr[i] == " " and expr[i + 2] == " " and expr[i + 1] in "+-":
            parts.append("".join(buf).strip())
            buf = []
            operators.append(expr[i + 1])
            i += 3
            continue
        buf.append(expr[i])
        i += 1
    parts.append("".join(buf).strip())

    terms: list[tuple[str, float]] = []
    for idx, token in enumerate(parts):
        if not token:
            continue
        sign = 1.0
        if idx > 0:
            op = operators[idx - 1]
            if op == "-":
                sign = -1.0
        terms.append((token, sign))
    return tuple(terms)


def load_target_specs() -> list[TargetSpec]:
    pairs = load_target_pairs()
    specs: list[TargetSpec] = []
    for row in pairs.itertuples(index=False):
        terms = _parse_pair_expression(row.pair)
        specs.append(TargetSpec(name=row.target, lag=int(row.lag), terms=terms))
    return specs


def targets_by_lag() -> dict[int, list[TargetSpec]]:
    by_lag: dict[int, list[TargetSpec]] = {}
    for spec in load_target_specs():
        by_lag.setdefault(spec.lag, []).append(spec)
    return by_lag


def target_names() -> list[str]:
    return [spec.name for spec in load_target_specs()]


def build_target_frame(
    prices: pd.DataFrame,
    specs: Sequence[TargetSpec] | None = None,
    log_scale: bool = True,
    eps: float = 1e-8,
    index_col: str | None = "date_id",
) -> pd.DataFrame:
    specs = list(specs or load_target_specs())
    if index_col and index_col in prices.columns:
        price_frame = prices.set_index(index_col)
    else:
        price_frame = prices.copy()
        price_frame.index = prices.index
    target_data = {}
    for spec in specs:
        series = None
        for asset, weight in spec.terms:
            values = price_frame[asset].to_numpy(dtype=float, copy=False)
            if log_scale:
                values = np.log(np.clip(values, eps, None))
            term = weight * values
            series = term if series is None else series + term
        target_data[spec.name] = series
    target_frame = pd.DataFrame(target_data, index=price_frame.index)
    return target_frame


def compute_target_returns(target_frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return target_frame.diff(periods=periods)
