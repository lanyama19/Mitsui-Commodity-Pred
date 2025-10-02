"""Kaggle inference entry for Mitsui LightGBM pipeline (no recency weighting)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

BUNDLE_ROOT = Path("/kaggle/input/mitsui-lightgbm-training-lag1lag4")
if not BUNDLE_ROOT.exists():
    # fallback for local testing
    BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "dist" / "kaggle_bundle"
    if str(BUNDLE_ROOT) not in sys.path:
        sys.path.append(str(BUNDLE_ROOT))
else:
    sys.path.append(str(BUNDLE_ROOT))

# Ensure project src is importable
SRC_ROOT = BUNDLE_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.features.pipeline_v2 import build_feature_pipeline

CONFIG_PATH = BUNDLE_ROOT / "configs" / "lightgbm_lag_params.json"
ARTIFACT_ROOT = BUNDLE_ROOT / "artifacts" / "lightgbm_full"
NUM_TARGETS = 424


def _latest_run_dir(lag: int) -> Path:
    runs = sorted((ARTIFACT_ROOT / f"lag_{lag}").iterdir(), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No trained runs found for lag {lag}")
    return runs[-1]


def _load_models() -> Dict[int, Dict[str, lgb.Booster]]:
    boosters: Dict[int, Dict[str, lgb.Booster]] = {}
    for lag in range(1, 5):
        run_dir = _latest_run_dir(lag)
        model_dir = run_dir / "models"
        boosters[lag] = {}
        for model_path in sorted(model_dir.glob("target_*.txt")):
            boosters[lag][model_path.stem] = lgb.Booster(model_file=str(model_path))
    return boosters


def _load_preprocessors() -> Dict[int, Dict[str, List[str] | Dict[str, float]]]:
    preprocessors: Dict[int, Dict[str, List[str] | Dict[str, float]]] = {}
    for lag in range(1, 5):
        run_dir = _latest_run_dir(lag)
        prep_dir = run_dir / "preprocessing"
        sample_file = next(prep_dir.glob("target_*.json"))
        payload = json.loads(sample_file.read_text())
        preprocessors[lag] = {
            "feature_names": payload["feature_names"],
            "median": payload["median"],
        }
    return preprocessors


BOOSTERS = _load_models()
PREPROCESSORS = _load_preprocessors()
TARGET_NAMES = sorted(BOOSTERS[1].keys())


class HistoryStore:
    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []

    def append(self, frame: pl.DataFrame) -> None:
        self.frames.append(frame.to_pandas())

    def combined(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        combined = pd.concat(self.frames, ignore_index=True)
        combined = combined.sort_values("date_id").reset_index(drop=True)
        return combined


HISTORY = HistoryStore()


def _prepare_features(lag: int, history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        raise ValueError("History DataFrame is empty; cannot compute features")
    pipeline = build_feature_pipeline(lag=lag, horizon=1, feature_version="v2")
    features_full = pipeline.transform(history_df)
    latest_date = features_full.index.max()
    features_last = features_full.loc[[latest_date]]

    names = PREPROCESSORS[lag]["feature_names"]
    medians = PREPROCESSORS[lag]["median"]
    features_last = features_last.reindex(columns=names)
    features_last = features_last.fillna(medians)
    return features_last


def _predict_lag(lag: int, features: pd.DataFrame) -> np.ndarray:
    models = BOOSTERS[lag]
    feature_array = features.to_numpy(dtype=np.float32, copy=False)
    preds = np.zeros((feature_array.shape[0], NUM_TARGETS), dtype=np.float32)
    for idx, target in enumerate(TARGET_NAMES):
        booster = models[target]
        preds[:, idx] = booster.predict(feature_array, num_iteration=booster.best_iteration)
    return preds


def predict(test: pl.DataFrame,
            label_lags_1_batch: pl.DataFrame,
            label_lags_2_batch: pl.DataFrame,
            label_lags_3_batch: pl.DataFrame,
            label_lags_4_batch: pl.DataFrame) -> pd.DataFrame:
    HISTORY.append(test)
    history_df = HISTORY.combined()

    blended = np.zeros((1, NUM_TARGETS), dtype=np.float32)
    for lag in range(1, 5):
        feats = _prepare_features(lag, history_df)
        blended += _predict_lag(lag, feats)
    blended /= 4.0
    return pd.DataFrame(blended, columns=[f"target_{i}" for i in range(NUM_TARGETS)])


if __name__ == "__main__":
    import kaggle_evaluation.mitsui_inference_server

    server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
    if BUNDLE_ROOT.is_dir():
        server.serve()
    else:
        # Local smoke test: requires competition data in the bundle directory
        server.run_local_gateway((str(BUNDLE_ROOT),))
