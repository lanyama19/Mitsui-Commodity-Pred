
"""Convenience launcher for tuned LightGBM baseline (CPU/GPU)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.train_lightgbm import LightGBMConfig, train_lightgbm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tuned LightGBM baseline for a given lag")
    parser.add_argument("--lag", type=int, default=1, help="Lag bucket to train")
    parser.add_argument("--train-end", type=int, default=1800, help="Train/validation split boundary")
    parser.add_argument("--run-name", type=str, default="tuned", help="Directory name for artifacts")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Set LightGBM device")
    parser.add_argument("--gpu-platform-id", type=int, default=None, help="GPU platform id when device=gpu")
    parser.add_argument("--gpu-device-id", type=int, default=None, help="GPU device id when device=gpu")
    parser.add_argument("--max-bin", type=int, default=None, help="Override LightGBM max_bin")
    parser.add_argument("--bin-sample", type=int, default=None, help="Override bin_construct_sample_cnt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LightGBMConfig(
        lag=args.lag,
        train_end=args.train_end,
        horizon=1,
        num_boost_round=200,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.6,
        bagging_fraction=0.6,
        bagging_freq=5,
        min_data_in_leaf=50,
        lambda_l1=1e-3,
        lambda_l2=1e-2,
        early_stopping_rounds=20,
        min_variance=1e-8,
        run_name=f"{args.run_name}_lag{args.lag}",
        device=args.device if args.device != "cpu" else None,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin if args.max_bin is not None else (255 if args.device == "gpu" else None),
        bin_construct_sample_cnt=args.bin_sample if args.bin_sample is not None else (200000 if args.device == "gpu" else None),
    )
    train_lightgbm(cfg)


if __name__ == "__main__":
    main()