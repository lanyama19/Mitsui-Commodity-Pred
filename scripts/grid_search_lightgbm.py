
"""Parallel hyperparameter sweep for LightGBM baselines (CPU/GPU ready)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "joblib is required for parallel execution. Please run `pip install joblib`."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.train_lightgbm import LightGBMConfig, train_lightgbm  # noqa: E402


DEFAULT_GRID: Dict[str, Sequence] = {
    "num_boost_round": [160, 200, 240, 280],
    "learning_rate": [0.03, 0.05, 0.07],
    "num_leaves": [16, 31, 48],
    "feature_fraction": [0.6, 0.7, 0.8],
    "bagging_fraction": [0.6, 0.7, 0.8],
    "min_data_in_leaf": [30, 50, 80],
    "lambda_l1": [0.0, 1e-3, 1e-2],
    "lambda_l2": [1e-3, 1e-2, 1e-1],
    "early_stopping_rounds": [10, 20, 30],
}


def _flatten_grid(grid: Dict[str, Sequence]) -> List[Dict[str, object]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [{k: v for k, v in zip(keys, choice)} for choice in itertools.product(*values)]


def _sample_combos(grid: Dict[str, Sequence], n_samples: int, seed: int) -> List[Dict[str, object]]:
    combos = _flatten_grid(grid)
    rng = random.Random(seed)
    rng.shuffle(combos)
    if n_samples and n_samples < len(combos):
        combos = combos[:n_samples]
    return combos


def _prepare_gpu_env(use_gpu: bool) -> None:
    if not use_gpu:
        return
    cache_root = (Path.cwd() / "artifacts" / "boost_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BOOST_COMPUTE_USE_OFFLINE_CACHE", "0")
    os.environ.setdefault("BOOST_COMPUTE_CACHE_PATH", str(cache_root))


def _evaluate_config(
    idx: int,
    params: Dict[str, object],
    base_cfg: LightGBMConfig,
    verbose: bool,
) -> Dict[str, object]:
    cfg_dict = asdict(base_cfg)
    cfg_dict.update(params)
    cfg = LightGBMConfig(**cfg_dict)
    cfg.run_name = f"grid_{idx:05d}"
    try:
        result = train_lightgbm(cfg, save_artifacts=False, verbose=verbose)
        summary = result["summary"]
        return {
            "index": idx,
            "status": "ok",
            "params": params,
            "train_end": cfg.train_end,
            "train_ic": summary.get("train_ic_mean"),
            "train_ir": summary.get("train_ir"),
            "val_ic": summary.get("val_ic_mean"),
            "val_ir": summary.get("val_ir"),
            "n_train_dates": summary.get("n_train_dates"),
            "n_val_dates": summary.get("n_val_dates"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": idx,
            "status": "error",
            "params": params,
            "train_end": cfg.train_end,
            "error": str(exc),
        }


def _to_dataframe(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        flat = {
            "index": rec.get("index"),
            "status": rec.get("status"),
            "train_end": rec.get("train_end"),
        }
        params = rec.get("params", {})
        for key, value in params.items():
            flat[f"param_{key}"] = value
        for key in [
            "train_ic",
            "train_ir",
            "val_ic",
            "val_ir",
            "n_train_dates",
            "n_val_dates",
            "error",
        ]:
            if key in rec:
                flat[key] = rec[key]
        rows.append(flat)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [
        "train_ic",
        "train_ir",
        "val_ic",
        "val_ir",
        "n_train_dates",
        "n_val_dates",
        "train_end",
        "param_num_boost_round",
        "param_learning_rate",
        "param_num_leaves",
        "param_feature_fraction",
        "param_bagging_fraction",
        "param_min_data_in_leaf",
        "param_lambda_l1",
        "param_lambda_l2",
        "param_early_stopping_rounds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"train_ic", "val_ic"}.issubset(df.columns):
        df["train_val_gap"] = df["train_ic"] - df["val_ic"]
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel LightGBM hyperparameter sweep")
    parser.add_argument("--lag", type=int, default=1, help="Lag bucket to evaluate")
    parser.add_argument("--train-end", type=int, default=1800, help="Fixed train_end across all evaluations")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of parameter combinations to evaluate")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (GPU mode forces 1)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for sampling combinations")
    parser.add_argument("--max-train-gap", type=float, default=0.2, help="Filter threshold for train-val IC gap")
    parser.add_argument("--feature-panel", type=str, default="all_train.pkl", help="Feature panel file name under artifacts/features")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="LightGBM device type")
    parser.add_argument("--gpu-platform-id", type=int, default=None, help="GPU platform id when device=gpu")
    parser.add_argument("--gpu-device-id", type=int, default=None, help="GPU device id when device=gpu")
    parser.add_argument("--max-bin", type=int, default=None, help="Override LightGBM max_bin")
    parser.add_argument("--bin-sample", type=int, default=None, help="Override LightGBM bin_construct_sample_cnt")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV file to store all results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output inside each LightGBM run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_gpu = args.device == "gpu"
    _prepare_gpu_env(use_gpu)

    base_cfg = LightGBMConfig(
        lag=args.lag,
        feature_panel=args.feature_panel,
        train_end=args.train_end,
        device=args.device if args.device != "cpu" else None,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin,
        bin_construct_sample_cnt=args.bin_sample,
    )

    combos = _sample_combos(DEFAULT_GRID, args.n_samples, args.seed)
    backend = "loky"
    if use_gpu:
        if args.n_jobs > 1:
            print("[GridSearch] GPU mode detected; forcing n_jobs=1 to avoid device contention.")
            args.n_jobs = 1
        backend = "threading"
    print(
        f"[GridSearch] Evaluating {len(combos)} configurations on lag={args.lag} "
        f"(train_end={args.train_end}) with {args.n_jobs} workers (backend={backend})"
    )

    evaluator = delayed(_evaluate_config)
    results = Parallel(n_jobs=args.n_jobs, backend=backend)(
        evaluator(idx, params, base_cfg, args.verbose) for idx, params in enumerate(combos)
    )

    df = _to_dataframe(results)
    if df.empty:
        print("[GridSearch] No results collected")
        return

    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        print("[GridSearch] All configurations failed")
    else:
        ok_df = ok_df.sort_values("val_ic", ascending=False)
        filtered = ok_df[ok_df["train_val_gap"] <= args.max_train_gap]
        print()
        print(f"[GridSearch] Top configurations by validation IC (gap <= {args.max_train_gap:.3f}):")
        if filtered.empty:
            print("(no configurations satisfied the gap constraint)")
        else:
            print(filtered.head(10).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[GridSearch] Full results written to {output_path}")


if __name__ == "__main__":
    main()