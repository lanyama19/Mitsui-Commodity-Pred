"""Utility script for basic feature EDA filtered by lag.

Usage example:
    python scripts/feature_eda.py --lag 1 --panel all --preview-cols 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.data.targets import load_target_specs

EDA_DIR = config.OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a quick exploratory pass on the generated feature panels, "
            "filtering columns to the targets associated with a specific lag."
        )
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Lag identifier to filter targets (default: 1)",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="all",
        help="Which pickled feature panel to load (matches artifacts/features/<panel>.pkl).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to display in the preview table (default: 10).",
    )
    parser.add_argument(
        "--preview-cols",
        type=int,
        default=12,
        help="How many feature columns to display in the preview table (default: 12).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional filename (within artifacts/eda) to dump the summary table as CSV.",
    )
    parser.add_argument(
        "--zero-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance when counting zero values (default: 1e-12).",
    )
    return parser.parse_args()

def load_feature_panel(panel: str) -> pd.DataFrame:
    path = config.OUTPUT_DIR / "features" / f"{panel}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Feature panel not found: {path}")
    frame = pd.read_pickle(path)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Expected a 2-level MultiIndex on feature columns (target, feature)")
    return frame

def targets_for_lag(lag: int) -> list[str]:
    return [spec.name for spec in load_target_specs() if spec.lag == lag]

def filter_by_lag(frame: pd.DataFrame, target_names: Iterable[str]) -> pd.DataFrame:
    target_names = set(target_names)
    cols = frame.columns
    mask = cols.get_level_values(0).isin(target_names)
    if not mask.any():
        raise ValueError(
            "No columns remaining after lag filter. Check that features were built for the requested lag."
        )
    return frame.loc[:, mask]

def compute_summary(frame: pd.DataFrame, zero_atol: float) -> pd.DataFrame:
    feature_labels = frame.columns.get_level_values(1).unique()
    summary_rows = []
    for feat in feature_labels:
        block = frame.xs(feat, axis=1, level=1, drop_level=False)
        values = block.to_numpy(dtype=float)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if valid_values.size == 0:
            zero_ratio = np.nan
            mean = np.nan
            std = np.nan
        else:
            zero_ratio = np.isclose(valid_values, 0.0, atol=zero_atol).sum() / valid_values.size
            mean = float(np.nanmean(valid_values))
            std = float(np.nanstd(valid_values))
        summary_rows.append(
            {
                "feature": feat,
                "zero_ratio": zero_ratio,
                "mean": mean,
                "std": std,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("feature").reset_index(drop=True)
    return summary

def main() -> None:
    args = parse_args()

    feature_panel = load_feature_panel(args.panel)
    target_names = targets_for_lag(args.lag)
    if not target_names:
        raise ValueError(f"No targets registered for lag={args.lag}")

    lag_frame = filter_by_lag(feature_panel, target_names)
    n_dates = lag_frame.shape[0]
    n_targets = lag_frame.columns.get_level_values(0).nunique()
    n_features = lag_frame.columns.get_level_values(1).nunique()

    print(f"Loaded panel '{args.panel}' with shape {feature_panel.shape}")
    print(
        f"Lag {args.lag}: {n_dates} dates, {n_targets} targets, {n_features} feature columns"
    )

    preview = lag_frame.copy()
    preview.columns = [f"{target}::{feat}" for target, feat in preview.columns]
    preview = preview.iloc[: args.head, : args.preview_cols]
    if preview.empty:
        print("No data available for preview after filtering.")
    else:
        print("\nPreview (truncated):")
        print(preview)

    summary = compute_summary(lag_frame, zero_atol=args.zero_atol)
    print("\nPer-feature summary (zero_ratio, mean, std):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    if args.summary_csv:
        out_path = (EDA_DIR / args.summary_csv).with_suffix('.csv')
        summary.to_csv(out_path, index=False)
        print(f"\nSummary saved to {out_path}")

if __name__ == '__main__':
    main()
