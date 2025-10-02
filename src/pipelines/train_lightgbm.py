"""LightGBM baseline pipeline for lag-specific models with optional GPU support."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LightGBM is not installed. Please run `pip install lightgbm` in your environment."
    ) from exc

from src import config
from src.data.loading import load_train_labels
from src.data.targets import TargetSpec, load_target_specs


@dataclass
class LightGBMConfig:
    """Configuration container describing one lag-specific LightGBM run."""

    lag: int
    horizon: int = 1
    train_end: int = 1800
    num_boost_round: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 64
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 20
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    early_stopping_rounds: int = 50
    max_depth: int = -1
    subsample_for_bin: int = 200000
    seed: int = 2024
    feature_panel: str = "all_train.pkl"
    feature_store: str = "features"
    artifact_root: str = "lightgbm"
    run_name: str | None = None
    min_variance: float = 1e-9
    device: str | None = None
    gpu_platform_id: int | None = None
    gpu_device_id: int | None = None
    max_bin: int | None = None
    bin_construct_sample_cnt: int | None = None
    weight_scheme: str = "none"
    weight_decay_rate: float = 0.0
    weight_floor: float = 0.1
    weight_cap: float | None = None
    weight_normalize: bool = True


def _prepare_gpu_env(cfg: LightGBMConfig) -> None:
    if cfg.device != "gpu":
        return
    cache_dir = (config.OUTPUT_DIR / "boost_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BOOST_COMPUTE_USE_OFFLINE_CACHE", "0")
    os.environ.setdefault("BOOST_COMPUTE_CACHE_PATH", str(cache_dir))


def _targets_for_lag(specs: Iterable[TargetSpec], lag: int) -> List[str]:
    return [spec.name for spec in specs if spec.lag == lag]


def _load_feature_panel(panel_name: str, store: str = "features") -> pd.DataFrame:
    path = config.OUTPUT_DIR / store / panel_name
    if not path.exists():
        raise FileNotFoundError(f"Feature panel not found: {path}")
    frame = pd.read_pickle(path)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Expected MultiIndex columns (target, feature) in feature panel")
    return frame


def _prepare_run_directory(lag: int, run_name: str | None, artifact_root: str = "lightgbm") -> Path:
    timestamp = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root = config.OUTPUT_DIR / artifact_root
    run_dir = root / f"lag_{lag}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "preprocessing").mkdir(exist_ok=True)
    return run_dir


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size != target.size:
        raise ValueError("Prediction and target vectors must share the same length")
    if pred.size < 3:
        return float("nan")
    p_rank = np.argsort(np.argsort(pred))
    t_rank = np.argsort(np.argsort(target))
    p_centered = p_rank - p_rank.mean()
    t_centered = t_rank - t_rank.mean()
    denom = np.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float((p_centered * t_centered).sum() / denom)


def _compute_daily_ic(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    shared_dates = sorted(set(preds.index).intersection(labels.index))
    records = []
    for date_id in shared_dates:
        pred_row = preds.loc[date_id]
        label_row = labels.loc[date_id]
        mask = (~pred_row.isna()) & (~label_row.isna())
        if mask.sum() < 3:
            continue
        corr = _spearman(pred_row[mask].to_numpy(dtype=float), label_row[mask].to_numpy(dtype=float))
        if np.isfinite(corr):
            records.append({"date_id": int(date_id), "ic": corr})
    return pd.DataFrame.from_records(records)


def _drop_low_variance(features: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    variance = features.var(axis=0, skipna=True)
    keep = variance > threshold
    if not keep.any():
        raise ValueError("All features were filtered out by variance threshold; lower `min_variance`.")
    filtered = features.loc[:, keep]
    return filtered, filtered.columns.tolist()


def _compute_time_distance(index: pd.Index) -> np.ndarray:
    if index.empty:
        return np.empty(0, dtype=float)
    if np.issubdtype(index.dtype, np.number):
        values = index.to_numpy(dtype=float, copy=False)
    else:
        values = np.arange(index.size, dtype=float)
    origin = np.nanmin(values)
    if not np.isfinite(origin):
        return np.empty(0, dtype=float)
    return values - origin


def _compute_sample_weights(index: pd.Index, cfg: LightGBMConfig) -> np.ndarray | None:
    scheme = (cfg.weight_scheme or "none").lower()
    if scheme in {"none", "off"}:
        return None
    distances = _compute_time_distance(index)
    if distances.size == 0:
        return None
    span = float(np.max(distances)) if distances.size else 0.0
    if span > 0:
        rel = distances / span
    else:
        rel = distances
    decay = cfg.weight_decay_rate
    if scheme in {"recency_exp", "exp"}:
        if decay <= 0:
            return None
        weights = np.exp(rel * decay)
    elif scheme in {"recency_linear", "linear"}:
        weights = 1.0 + rel * decay
    else:
        raise ValueError(f"Unsupported weight_scheme: {cfg.weight_scheme}")
    if cfg.weight_cap is not None:
        weights = np.minimum(weights, cfg.weight_cap)
    if cfg.weight_floor is not None:
        weights = np.maximum(weights, cfg.weight_floor)
    if cfg.weight_normalize:
        mean = np.mean(weights)
        if mean > 0:
            weights = weights / mean
    weights = weights.astype(np.float32, copy=False)
    return weights


def _train_target(
    target: str,
    features_panel: pd.DataFrame,
    label_frame: pd.DataFrame,
    cfg: LightGBMConfig,
) -> Tuple[
    Dict[str, float | int | None],
    pd.Series,
    pd.Series,
    pd.Series | None,
    pd.Series | None,
    lgb.Booster,
    Dict[str, Any],
]:
    feature_block = features_panel.xs(target, axis=1, level=0)
    shift = cfg.lag + cfg.horizon
    label_series = label_frame[target].shift(-shift)

    valid_mask = label_series.notna()
    feature_block = feature_block.loc[valid_mask]
    label_series = label_series.loc[valid_mask]

    if feature_block.empty:
        raise ValueError(f"No observations available for target {target} after alignment")

    train_mask = feature_block.index <= cfg.train_end
    val_mask = feature_block.index > cfg.train_end

    if not train_mask.any():
        raise ValueError(f"Training set empty for target {target}; adjust `train_end` or lag")

    train_features = feature_block.loc[train_mask]
    val_features = feature_block.loc[val_mask]
    train_labels = label_series.loc[train_mask]
    val_labels = label_series.loc[val_mask]

    train_features, feature_names = _drop_low_variance(train_features, cfg.min_variance)
    val_features = val_features.reindex(columns=feature_names, fill_value=np.nan)

    train_median = train_features.median(axis=0, skipna=True)
    train_features = train_features.fillna(train_median)
    val_features = val_features.fillna(train_median)

    train_array = train_features.to_numpy(dtype=np.float32, copy=False)
    train_target = train_labels.to_numpy(dtype=np.float32, copy=False)
    train_weights = _compute_sample_weights(train_features.index, cfg)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "lambda_l1": cfg.lambda_l1,
        "lambda_l2": cfg.lambda_l2,
        "max_depth": cfg.max_depth,
        "subsample_for_bin": cfg.subsample_for_bin,
        "verbosity": -1,
        "seed": cfg.seed,
    }
    if cfg.device:
        params["device_type"] = cfg.device
    if cfg.gpu_platform_id is not None:
        params["gpu_platform_id"] = cfg.gpu_platform_id
    if cfg.gpu_device_id is not None:
        params["gpu_device_id"] = cfg.gpu_device_id
    if cfg.max_bin is not None:
        params["max_bin"] = cfg.max_bin
    if cfg.bin_construct_sample_cnt is not None:
        params["bin_construct_sample_cnt"] = cfg.bin_construct_sample_cnt
    if cfg.device == "gpu":
        params.setdefault("max_bin", 255)
        params.setdefault("bin_construct_sample_cnt", 200000)
        params.setdefault("force_col_wise", True)

    train_dataset = lgb.Dataset(
        train_array,
        label=train_target,
        weight=train_weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )

    valid_sets = [train_dataset]
    valid_names = ["train"]
    val_array = None
    val_target = None

    use_validation = len(val_labels) > 0
    if use_validation:
        val_array = val_features.to_numpy(dtype=np.float32, copy=False)
        val_target = val_labels.to_numpy(dtype=np.float32, copy=False)
        valid_sets.append(
            lgb.Dataset(
                val_array,
                label=val_target,
                feature_name=list(feature_names),
                free_raw_data=False,
            )
        )
        valid_names.append("valid")

    callbacks: List[Any] = [lgb.log_evaluation(period=0)]
    if use_validation and cfg.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    best_iteration = booster.best_iteration or cfg.num_boost_round

    train_pred = booster.predict(train_array, num_iteration=best_iteration)
    train_pred_series = pd.Series(train_pred, index=train_features.index, name=target)
    train_label_series = train_labels.copy()
    train_rmse = _rmse(train_target, train_pred)

    val_pred_series: pd.Series | None = None
    val_label_series: pd.Series | None = None
    val_rmse: float | None = None
    if use_validation and val_array is not None and val_target is not None:
        val_pred = booster.predict(val_array, num_iteration=best_iteration)
        val_rmse = _rmse(val_target, val_pred)
        val_pred_series = pd.Series(val_pred, index=val_features.index, name=target)
        val_label_series = val_labels.copy()

    metrics: Dict[str, float | int | None] = {
        "train_samples": int(train_array.shape[0]),
        "val_samples": int(val_array.shape[0]) if val_array is not None else 0,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "best_iteration": int(best_iteration),
    }
    if train_weights is not None and train_weights.size > 0:
        metrics["train_weight_mean"] = float(np.mean(train_weights))
        metrics["train_weight_min"] = float(np.min(train_weights))
        metrics["train_weight_max"] = float(np.max(train_weights))

    preprocessing: Dict[str, Any] = {
        "feature_names": feature_names,
        "median": {name: float(train_median[name]) for name in feature_names},
    }

    return (
        metrics,
        train_pred_series,
        train_label_series,
        val_pred_series,
        val_label_series,
        booster,
        preprocessing,
    )


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def train_lightgbm(
    cfg: LightGBMConfig,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    _prepare_gpu_env(cfg)

    specs = load_target_specs()
    target_names = _targets_for_lag(specs, cfg.lag)
    if not target_names:
        raise ValueError(f"No targets registered for lag={cfg.lag}")

    feature_panel = _load_feature_panel(cfg.feature_panel, cfg.feature_store)
    label_frame = load_train_labels()
    if "date_id" not in label_frame.columns:
        raise KeyError("Expected `date_id` column in training labels")
    label_frame = label_frame.set_index("date_id")

    common_index = feature_panel.index.intersection(label_frame.index)
    if common_index.empty:
        raise ValueError("No overlapping dates between features and labels")
    feature_panel = feature_panel.loc[common_index]
    label_frame = label_frame.loc[common_index, target_names]

    run_dir: Path | None = None
    if save_artifacts:
        run_dir = _prepare_run_directory(cfg.lag, cfg.run_name, cfg.artifact_root)
        if verbose:
            print(f"[LightGBM] Saving run artifacts to {run_dir}")
    else:
        if verbose:
            print(f"[LightGBM] Running lag {cfg.lag} in metrics-only mode (artifacts not persisted)")

    per_target_metrics: Dict[str, Dict[str, float | int | None]] = {}
    train_pred_map: Dict[str, pd.Series] = {}
    train_label_map: Dict[str, pd.Series] = {}
    val_pred_map: Dict[str, pd.Series] = {}
    val_label_map: Dict[str, pd.Series] = {}

    for idx, target in enumerate(target_names, start=1):
        if verbose:
            print(f"[LightGBM] Training target {target} ({idx}/{len(target_names)})")
        (
            metrics,
            train_preds,
            train_labels,
            val_preds,
            val_labels,
            booster,
            preprocessing,
        ) = _train_target(
            target=target,
            features_panel=feature_panel,
            label_frame=label_frame,
            cfg=cfg,
        )
        per_target_metrics[target] = metrics
        train_pred_map[target] = train_preds
        train_label_map[target] = train_labels

        if val_preds is not None and val_labels is not None:
            val_pred_map[target] = val_preds
            val_label_map[target] = val_labels

        if save_artifacts and run_dir is not None:
            model_path = run_dir / "models" / f"{target}.txt"
            booster.save_model(str(model_path))
            preproc_path = run_dir / "preprocessing" / f"{target}.json"
            with open(preproc_path, "w", encoding="utf-8") as f:
                json.dump(preprocessing, f, indent=2)

    summary: Dict[str, float | int | None] = {
        "train_ic_mean": None,
        "train_ic_std": None,
        "train_ir": None,
        "val_ic_mean": None,
        "val_ic_std": None,
        "val_ir": None,
        "n_train_dates": 0,
        "n_val_dates": 0,
    }

    if train_pred_map and train_label_map:
        train_pred_df = pd.DataFrame(train_pred_map).sort_index()
        train_label_df = pd.DataFrame(train_label_map).sort_index()
        train_daily_ic = _compute_daily_ic(train_pred_df, train_label_df)
        if not train_daily_ic.empty:
            summary["train_ic_mean"] = float(train_daily_ic["ic"].mean())
            summary["train_ic_std"] = float(train_daily_ic["ic"].std(ddof=0))
            if summary["train_ic_std"] and summary["train_ic_std"] > 0:
                summary["train_ir"] = summary["train_ic_mean"] / summary["train_ic_std"]
            summary["n_train_dates"] = int(len(train_daily_ic))
            if save_artifacts and run_dir is not None:
                train_daily_ic.to_csv(run_dir / "train_daily_ic.csv", index=False)
        if save_artifacts and run_dir is not None:
            train_pred_df.to_pickle(run_dir / "train_predictions.pkl")
            train_label_df.to_pickle(run_dir / "train_labels.pkl")

    if val_pred_map and val_label_map:
        val_pred_df = pd.DataFrame(val_pred_map).sort_index()
        val_label_df = pd.DataFrame(val_label_map).sort_index()
        daily_ic = _compute_daily_ic(val_pred_df, val_label_df)
        if not daily_ic.empty:
            summary["val_ic_mean"] = float(daily_ic["ic"].mean())
            summary["val_ic_std"] = float(daily_ic["ic"].std(ddof=0))
            if summary["val_ic_std"] and summary["val_ic_std"] > 0:
                summary["val_ir"] = summary["val_ic_mean"] / summary["val_ic_std"]
            summary["n_val_dates"] = int(len(daily_ic))
            if save_artifacts and run_dir is not None:
                daily_ic.to_csv(run_dir / "daily_ic.csv", index=False)
        if save_artifacts and run_dir is not None:
            val_pred_df.to_pickle(run_dir / "val_predictions.pkl")
            val_label_df.to_pickle(run_dir / "val_labels.pkl")

    if save_artifacts and run_dir is not None:
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            payload = {
                "summary": summary,
                "per_target": per_target_metrics,
            }
            json.dump(payload, f, indent=2)

    if verbose:
        print(
            "[LightGBM] Completed lag {lag}: train_ic={train_ic} train_ir={train_ir} val_ic={val_ic} val_ir={val_ir} n_train_dates={n_train} n_val_dates={n_val}".format(
                lag=cfg.lag,
                train_ic=_format_float(summary["train_ic_mean"]),
                train_ir=_format_float(summary["train_ir"]),
                val_ic=_format_float(summary["val_ic_mean"]),
                val_ir=_format_float(summary["val_ir"]),
                n_train=summary["n_train_dates"],
                n_val=summary["n_val_dates"],
            )
        )

    return {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "summary": summary,
        "per_target": per_target_metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM baseline for a specific lag")
    parser.add_argument("--lag", type=int, required=True, help="Lag bucket to train")
    parser.add_argument("--train-end", type=int, default=1800, help="Last date_id assigned to the training split")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days")
    parser.add_argument("--num-boost-round", type=int, default=500, help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for boosting")
    parser.add_argument("--num-leaves", type=int, default=64, help="Maximum number of leaves per tree")
    parser.add_argument("--feature-fraction", type=float, default=0.8, help="Fraction of features used per iteration")
    parser.add_argument("--bagging-fraction", type=float, default=0.8, help="Row subsampling fraction")
    parser.add_argument("--bagging-freq", type=int, default=5, help="Row subsampling frequency")
    parser.add_argument("--lambda-l1", type=float, default=0.0, help="L1 regularisation strength")
    parser.add_argument("--lambda-l2", type=float, default=0.0, help="L2 regularisation strength")
    parser.add_argument("--feature-panel", type=str, default="all_train.pkl", help="Feature pickle filename under artifacts/features")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run directory name")
    parser.add_argument("--min-variance", type=float, default=1e-9, help="Threshold for dropping near-constant features")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Device type for LightGBM")
    parser.add_argument("--gpu-platform-id", type=int, default=None, help="GPU platform id when device=gpu")
    parser.add_argument("--gpu-device-id", type=int, default=None, help="GPU device id when device=gpu")
    parser.add_argument("--max-bin", type=int, default=None, help="Override LightGBM max_bin")
    parser.add_argument("--bin-sample", type=int, default=None, help="Override bin_construct_sample_cnt")
    parser.add_argument("--verbose", action="store_true", help="Print per-target progress")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = LightGBMConfig(
        lag=args.lag,
        train_end=args.train_end,
        horizon=args.horizon,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        min_data_in_leaf=args.min_data_in_leaf,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        feature_panel=args.feature_panel,
        run_name=args.run_name,
        min_variance=args.min_variance,
        device=None if args.device == "cpu" else args.device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        max_bin=args.max_bin,
        bin_construct_sample_cnt=args.bin_sample,
    )
    train_lightgbm(cfg, save_artifacts=True, verbose=args.verbose)


if __name__ == "__main__":
    main()
