"""Quick smoke test for LightGBM GPU training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.train_lightgbm import LightGBMConfig, train_lightgbm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GPU smoke test over a handful of configs")
    parser.add_argument("--lag", type=int, default=1, help="Lag bucket to evaluate")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of configs to try")
    parser.add_argument("--train-end", type=int, default=1800, help="Train/validation boundary")
    parser.add_argument("--gpu-platform-id", type=int, default=0, help="GPU platform id")
    parser.add_argument("--gpu-device-id", type=int, default=0, help="GPU device id")
    return parser.parse_args()


CONFIGS = [
    dict(num_boost_round=200, learning_rate=0.05, num_leaves=31, feature_fraction=0.6, bagging_fraction=0.6, min_data_in_leaf=50, lambda_l1=1e-3, lambda_l2=1e-2, early_stopping_rounds=20),
    dict(num_boost_round=180, learning_rate=0.04, num_leaves=24, feature_fraction=0.65, bagging_fraction=0.7, min_data_in_leaf=60, lambda_l1=1e-3, lambda_l2=5e-3, early_stopping_rounds=15),
    dict(num_boost_round=220, learning_rate=0.06, num_leaves=40, feature_fraction=0.7, bagging_fraction=0.7, min_data_in_leaf=40, lambda_l1=2e-3, lambda_l2=1e-2, early_stopping_rounds=20),
    dict(num_boost_round=240, learning_rate=0.05, num_leaves=48, feature_fraction=0.6, bagging_fraction=0.8, min_data_in_leaf=80, lambda_l1=1e-3, lambda_l2=1e-1, early_stopping_rounds=25),
    dict(num_boost_round=160, learning_rate=0.03, num_leaves=24, feature_fraction=0.8, bagging_fraction=0.6, min_data_in_leaf=30, lambda_l1=0.0, lambda_l2=1e-2, early_stopping_rounds=15),
]


def main() -> None:
    args = parse_args()
    configs = CONFIGS[: args.n_samples]
    for idx, overrides in enumerate(configs, start=1):
        print()
        print(f"[SmokeGPU] Config {idx}/{len(configs)} -> {overrides}")
        cfg = LightGBMConfig(
            lag=args.lag,
            train_end=args.train_end,
            run_name=f"gpu_smoke_{idx}",
            device="gpu",
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            max_bin=255,
            bin_construct_sample_cnt=200000,
            **overrides,
        )
        train_lightgbm(cfg, save_artifacts=False, verbose=True)


if __name__ == "__main__":
    main()