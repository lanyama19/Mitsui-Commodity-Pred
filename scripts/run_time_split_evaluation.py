"""Run LightGBM training on multiple time-based splits and report validation IC."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.train_lightgbm import _compute_daily_ic


@dataclass(frozen=True)
class TimeSplit:
    label: str
    train_end: int
    val_start: int
    val_end: Optional[int]


DEFAULT_SPLITS: List[TimeSplit] = [
    TimeSplit(label="ts_1850_1880", train_end=1850, val_start=1851, val_end=1880),
    TimeSplit(label="ts_1900_1930", train_end=1900, val_start=1901, val_end=1930),
    TimeSplit(label="ts_1930_1960", train_end=1930, val_start=1931, val_end=1960),
]


PYTHON = Path(sys.executable)
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_lightgbm_full.py"
ARTIFACT_ROOT = Path("artifacts/lightgbm_cv")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

PARAM_CONFIG = Path("configs/lightgbm_lag_params.json")
FEATURE_VERSION = "v2"
WEIGHT_SCHEME = "recency_exp"
WEIGHT_DECAY = "1.2"
WEIGHT_FLOOR = "0.5"
WEIGHT_CAP = "10"
BAGGING_FREQ = "5"


def _make_run_prefix(split: TimeSplit) -> str:
    suffix = split.val_end if split.val_end is not None else "end"
    return f"tscv_{split.train_end}_{suffix}"


def _build_run_name(prefix: str, lag: int) -> str:
    name = f"{prefix}_lag{lag}"
    if WEIGHT_SCHEME != "none":
        name = f"{name}_{WEIGHT_SCHEME}"
    if FEATURE_VERSION.lower() != "v1":
        name = f"{name}_{FEATURE_VERSION.lower()}"
    return name


def _load_json(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _compute_val_metrics(run_dir: Path, split: TimeSplit) -> Dict[str, float | int | None]:
    val_pred_path = run_dir / "val_predictions.pkl"
    val_label_path = run_dir / "val_labels.pkl"
    if not val_pred_path.exists() or not val_label_path.exists():
        return {
            "val_ic_mean": None,
            "val_ic_std": None,
            "val_ir": None,
            "n_val_dates": 0,
        }

    preds = pd.read_pickle(val_pred_path).sort_index()
    labels = pd.read_pickle(val_label_path).sort_index()

    val_end = split.val_end if split.val_end is not None else preds.index.max()
    mask = (preds.index >= split.val_start) & (preds.index <= val_end)
    preds = preds.loc[mask]
    labels = labels.loc[mask]

    if preds.empty or labels.empty:
        return {
            "val_ic_mean": None,
            "val_ic_std": None,
            "val_ir": None,
            "n_val_dates": 0,
        }

    daily_ic = _compute_daily_ic(preds, labels)
    if daily_ic.empty:
        return {
            "val_ic_mean": None,
            "val_ic_std": None,
            "val_ir": None,
            "n_val_dates": 0,
        }

    mean_ic = float(daily_ic["ic"].mean())
    std_ic = float(daily_ic["ic"].std(ddof=0)) if len(daily_ic) > 1 else 0.0
    ir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and std_ic > 0 else None
    return {
        "val_ic_mean": mean_ic,
        "val_ic_std": std_ic if std_ic else None,
        "val_ir": float(ir) if ir is not None else None,
        "n_val_dates": int(len(daily_ic)),
    }


def _run_split(split: TimeSplit) -> List[Dict[str, float | int | str | None]]:
    run_prefix = _make_run_prefix(split)
    summary_path = ARTIFACT_ROOT / f"summary_{run_prefix}.json"

    cmd = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--param-config",
        str(PARAM_CONFIG),
    ]
    cmd.extend(["--train-end", str(split.train_end)])
    cmd.extend(["--feature-version", FEATURE_VERSION])
    cmd.extend(["--artifact-root", "lightgbm_cv"])
    cmd.extend(["--run-prefix", run_prefix])
    cmd.extend(["--weight-scheme", WEIGHT_SCHEME])
    cmd.extend(["--weight-decay", WEIGHT_DECAY])
    cmd.extend(["--weight-floor", WEIGHT_FLOOR])
    cmd.extend(["--weight-cap", WEIGHT_CAP])
    cmd.extend(["--lags", "1", "2", "3", "4"])
    cmd.extend(["--bagging-freq", BAGGING_FREQ])
    cmd.extend(["--summary-path", str(summary_path)])

    print(f"[time-split] Running split {split.label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    summary = _load_json(summary_path)
    records: List[Dict[str, float | int | str | None]] = []

    for lag_key, metrics in summary.items():
        lag = int(lag_key)
        run_name = _build_run_name(run_prefix, lag)
        run_dir = ARTIFACT_ROOT / f"lag_{lag}" / run_name
        val_metrics = _compute_val_metrics(run_dir, split)

        record: Dict[str, float | int | str | None] = {
            "split": split.label,
            "lag": lag,
            "train_end": split.train_end,
            "val_start": split.val_start,
            "val_end": split.val_end,
            "train_ic_mean": metrics.get("train_ic_mean"),
            "train_ic_std": metrics.get("train_ic_std"),
            "train_ir": metrics.get("train_ir"),
            "n_train_dates": metrics.get("n_train_dates"),
        }
        record.update(val_metrics)
        records.append(record)
    return records


def main(splits: Iterable[TimeSplit]) -> None:
    all_records: List[Dict[str, float | int | str | None]] = []
    for split in splits:
        split_records = _run_split(split)
        all_records.extend(split_records)

    if not all_records:
        print("[time-split] No records generated.")
        return

    df = pd.DataFrame.from_records(all_records)
    df.sort_values(by=["lag", "train_end"], inplace=True)
    output_path = ARTIFACT_ROOT / "time_split_metrics.csv"
    df.to_csv(output_path, index=False)
    print("[time-split] Metrics saved to", output_path)
    print(df)


if __name__ == "__main__":
    main(DEFAULT_SPLITS)
