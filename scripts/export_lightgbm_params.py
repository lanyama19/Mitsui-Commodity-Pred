"""Utility to export tuned LightGBM parameters from grid search CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

DEFAULT_IDX_MAP = {
    1: 117,
    2: 15,
    3: 117,
    4: 15,
}

PARAM_COLUMNS = {
    "num_boost_round": "param_num_boost_round",
    "learning_rate": "param_learning_rate",
    "num_leaves": "param_num_leaves",
    "feature_fraction": "param_feature_fraction",
    "bagging_fraction": "param_bagging_fraction",
    "min_data_in_leaf": "param_min_data_in_leaf",
    "lambda_l1": "param_lambda_l1",
    "lambda_l2": "param_lambda_l2",
}


def _load_row(csv_path: Path, index: int) -> pd.Series:
    frame = pd.read_csv(csv_path)
    if "index" in frame.columns:
        frame = frame.set_index("index")
    if index not in frame.index:
        raise KeyError(f"index={index} not present in {csv_path}")
    return frame.loc[index]


def _build_param_entry(row: pd.Series) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for target_key, column in PARAM_COLUMNS.items():
        if column not in row:
            raise KeyError(f"Column '{column}' missing in grid search row")
        value = row[column]
        if pd.isna(value):
            raise ValueError(f"Column '{column}' has NaN value in selected row")
        payload[target_key] = float(value)
    # Preserve integer valued hyperparameters as ints when possible.
    for key in ("num_boost_round", "num_leaves", "min_data_in_leaf"):
        payload[key] = int(payload[key])
    return payload


def export_params(
    version: str,
    indices: Dict[int, int],
    output_path: Path,
    base_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    root = Path(base_dir) if base_dir is not None else Path("artifacts") / "grid_search"
    payload: Dict[str, Dict[str, float]] = {}
    for lag, idx in indices.items():
        csv_name = f"grid_lag{lag}_{version}.csv"
        csv_path = root / csv_name
        row = _load_row(csv_path, idx)
        payload[f"lag{lag}"] = _build_param_entry(row)
        payload[f"lag{lag}"]["train_end"] = int(row.get("train_end", 1800))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return payload


def parse_indices(raw: Iterable[str]) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for item in raw:
        lag_str, idx_str = item.split(":", maxsplit=1)
        lag = int(lag_str.strip())
        idx = int(idx_str.strip())
        result[lag] = idx
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tuned LightGBM params from grid search outputs")
    parser.add_argument("--version", type=str, default="v2", help="Grid search version suffix (e.g. v2)")
    parser.add_argument(
        "--index",
        nargs="*",
        default=[f"{lag}:{idx}" for lag, idx in DEFAULT_IDX_MAP.items()],
        help="Lag:index pairs overriding the default selection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/lightgbm_lag_params.json"),
        help="Where to store the exported parameter JSON",
    )
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=Path("artifacts") / "grid_search",
        help="Directory containing grid CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.index)
    export_params(args.version, indices, args.output, base_dir=args.grid_root)


if __name__ == "__main__":
    main()
