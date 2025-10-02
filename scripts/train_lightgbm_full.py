"""Train LightGBM models per lag on the full training span with recency weighting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.train_lightgbm import LightGBMConfig, train_lightgbm

_VERSION_TO_STORE = {
    "v1": ("features", "lightgbm"),
    "v2": ("features_v2", "lightgbm_v2"),
}


def _resolve_feature_version(version: str) -> tuple[str, str]:
    key = (version or "v1").lower()
    if key in _VERSION_TO_STORE:
        return _VERSION_TO_STORE[key]
    return version, f"lightgbm_{version}"


def _parse_lags(raw: Iterable[str] | None, available: Iterable[int]) -> List[int]:
    if not raw:
        return sorted(set(available))
    lags: List[int] = []
    for item in raw:
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid lag identifier: {item}") from exc
        lags.append(value)
    return sorted(set(lags))


def _load_param_snapshot(path: Path) -> Dict[int, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    param_map: Dict[int, Dict[str, float]] = {}
    for key, params in payload.items():
        if isinstance(key, str) and key.startswith("lag"):
            lag = int(key.replace("lag", ""))
        else:
            lag = int(key)
        param_map[lag] = params
    return param_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM models per lag using full training data")
    parser.add_argument("--param-config", type=Path, default=Path("configs/lightgbm_lag_params.json"), help="JSON mapping lag -> tuned parameters")
    parser.add_argument("--lags", nargs="*", help="Subset of lags to train (defaults to all available in the config)")
    parser.add_argument("--train-end", type=int, default=1960, help="Last date_id to include in training span")
    parser.add_argument("--feature-version", type=str, default="v2", help="Feature store version key (maps to artifacts directory)")
    parser.add_argument("--feature-panel", type=str, default="all_train.pkl", help="Feature panel filename inside the store")
    parser.add_argument("--artifact-root", type=str, default="lightgbm_full", help="Root directory under artifacts for run outputs")
    parser.add_argument("--run-prefix", type=str, default="fulltrain", help="Prefix for per-lag run directories")
    parser.add_argument("--weight-scheme", type=str, default="recency_exp", choices=["none", "recency_exp", "recency_linear"], help="Sample weighting scheme applied to training rows")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Decay rate controlling recency emphasis (larger -> more weight on recent dates)")
    parser.add_argument("--weight-floor", type=float, default=0.1, help="Minimum weight applied after clipping")
    parser.add_argument("--weight-cap", type=float, default=3.0, help="Maximum weight applied after clipping (use <=0 to disable)")
    parser.add_argument("--disable-weight-normalize", action="store_true", help="Skip normalising weights to mean 1")
    parser.add_argument("--bagging-freq", type=int, default=5, help="LightGBM bagging frequency")
    parser.add_argument("--min-variance", type=float, default=1e-9, help="Variance filter threshold")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for LightGBM")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="LightGBM compute device")
    parser.add_argument("--gpu-platform-id", type=int, default=None, help="GPU platform id when device=gpu")
    parser.add_argument("--gpu-device-id", type=int, default=None, help="GPU device id when device=gpu")
    parser.add_argument("--max-bin", type=int, default=None, help="Override LightGBM max_bin")
    parser.add_argument("--bin-sample", type=int, default=None, help="Override LightGBM bin_construct_sample_cnt")
    parser.add_argument("--summary-path", type=Path, default=Path("artifacts/lightgbm_full_train_summary.json"), help="Where to dump aggregated run summaries")
    parser.add_argument("--verbose", action="store_true", help="Emit per-target logs from pipeline")
    return parser.parse_args()


def _build_config(
    lag: int,
    params: Dict[str, float],
    args: argparse.Namespace,
    feature_store: str,
    artifact_root: str,
) -> LightGBMConfig:
    run_name = f"{args.run_prefix}_lag{lag}"
    if args.weight_scheme != "none":
        run_name = f"{run_name}_{args.weight_scheme}"
    if args.feature_version and args.feature_version.lower() != "v1":
        run_name = f"{run_name}_{args.feature_version.lower()}"

    weight_cap = None if args.weight_cap is not None and args.weight_cap <= 0 else args.weight_cap

    return LightGBMConfig(
        lag=lag,
        horizon=args.horizon,
        train_end=args.train_end,
        num_boost_round=int(params.get("num_boost_round", 200)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 31)),
        feature_fraction=float(params.get("feature_fraction", 0.6)),
        bagging_fraction=float(params.get("bagging_fraction", 0.6)),
        bagging_freq=args.bagging_freq,
        min_data_in_leaf=int(params.get("min_data_in_leaf", 50)),
        min_gain_to_split=float(params.get("min_gain_to_split", 0.0)),
        lambda_l1=float(params.get("lambda_l1", 0.0)),
        lambda_l2=float(params.get("lambda_l2", 0.0)),
        early_stopping_rounds=0,
        min_variance=args.min_variance,
        feature_panel=args.feature_panel,
        feature_store=feature_store,
        artifact_root=artifact_root,
        run_name=run_name,
        device=None if args.device == "cpu" else args.device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin,
        bin_construct_sample_cnt=args.bin_sample,
        seed=args.seed,
        weight_scheme=args.weight_scheme,
        weight_decay_rate=args.weight_decay,
        weight_floor=args.weight_floor,
        weight_cap=weight_cap,
        weight_normalize=not args.disable_weight_normalize,
    )


def main() -> None:
    args = parse_args()

    param_map = _load_param_snapshot(args.param_config)
    feature_store, default_artifact_root = _resolve_feature_version(args.feature_version)
    artifact_root = args.artifact_root or default_artifact_root

    lags = _parse_lags(args.lags, param_map.keys())

    results: Dict[int, Dict[str, float]] = {}

    for lag in lags:
        if lag not in param_map:
            raise KeyError(f"Parameters for lag {lag} not found in {args.param_config}")
        cfg = _build_config(lag, param_map[lag], args, feature_store, artifact_root)
        print(f"[train_lightgbm_full] Training lag {lag} with run_name={cfg.run_name}")
        run_result = train_lightgbm(cfg, save_artifacts=True, verbose=args.verbose)
        results[lag] = run_result.get("summary", {})

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[train_lightgbm_full] Saved summary to {args.summary_path}")


if __name__ == "__main__":
    main()
