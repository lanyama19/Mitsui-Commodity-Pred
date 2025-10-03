"""Utilities to compute LightGBM features on-the-fly for inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.data.targets import TargetSpec, build_target_frame, load_target_specs
from src.features.gp import GPArtifacts
from src.features.pipeline import _ensure_unique_columns, _ensure_unique_index
from src.features.pipeline_v2 import (
    FeatureArtifacts,
    build_base_features_v2,
    load_feature_artifacts,
)
from src.features.pca import PCAArtifacts


@dataclass
class OnlineFeatureContext:
    specs: list[TargetSpec]
    artifacts: FeatureArtifacts


def load_online_context() -> OnlineFeatureContext:
    specs = load_target_specs()
    artifacts = load_feature_artifacts()
    return OnlineFeatureContext(specs=specs, artifacts=artifacts)


def _prepare_price_subset(price_history: pd.DataFrame) -> pd.DataFrame:
    subset = price_history.copy()
    if 'date_id' not in subset.columns:
        subset = subset.reset_index().rename(columns={'index': 'date_id'})
    subset = subset.drop_duplicates(subset='date_id', keep='last')
    subset = subset.sort_values('date_id').ffill().bfill()
    return subset


def _transform_gp_features(base_features: pd.DataFrame, artifacts: GPArtifacts) -> pd.DataFrame:
    stacked = base_features.stack(level=0, future_stack=True)
    stacked.index.set_names(['date_id', 'target'], inplace=True)
    stacked = stacked.sort_index()
    expected = artifacts.feature_names
    if expected:
        missing = [name for name in expected if name not in stacked.columns]
        if missing:
            raise KeyError(f"Missing GP feature inputs: {missing[:5]}")
        stacked = stacked[expected]
    transformed = artifacts.model.transform(stacked.to_numpy(dtype=float))
    gp_columns = [f"gp_{i+1}" for i in range(transformed.shape[1])]
    gp_df = pd.DataFrame(transformed, index=stacked.index, columns=gp_columns)
    gp_df = gp_df.unstack(level='target').swaplevel(axis=1)
    gp_df = gp_df.sort_index(axis=1)
    gp_df = _ensure_unique_columns(gp_df, 'gp_online')
    gp_df = _ensure_unique_index(gp_df, 'gp_online')
    return gp_df


def _transform_pca_features(target_frame: pd.DataFrame, artifacts: PCAArtifacts) -> pd.DataFrame:
    component_names = list(artifacts.loadings.columns)
    frames = []
    for target in target_frame.columns:
        row = artifacts.loadings.loc[target]
        repeated = np.repeat(row.values.reshape(1, -1), len(target_frame.index), axis=0)
        df = pd.DataFrame(repeated, index=target_frame.index, columns=component_names)
        df.columns = pd.MultiIndex.from_product([[target], component_names])
        frames.append(df)
    pca_df = pd.concat(frames, axis=1)
    pca_df = _ensure_unique_columns(pca_df, 'pca_online')
    pca_df = _ensure_unique_index(pca_df, 'pca_online')
    return pca_df


def compute_feature_panel(
    price_history: pd.DataFrame,
    context: OnlineFeatureContext,
) -> pd.DataFrame:
    price_subset = _prepare_price_subset(price_history)
    specs = context.specs
    artifacts = context.artifacts

    target_frame = build_target_frame(price_subset, specs=specs, index_col='date_id')
    target_frame = _ensure_unique_index(target_frame, 'target_prices_online')
    log_prices = np.log(price_subset.set_index('date_id').clip(lower=1e-8))
    base_features, _, _ = build_base_features_v2(target_frame, log_prices, specs)
    base_features = _ensure_unique_columns(base_features, 'base_online')
    base_features = _ensure_unique_index(base_features, 'base_online')

    gp_features = _transform_gp_features(base_features, artifacts.gp)
    pca_features = _transform_pca_features(target_frame, artifacts.pca)

    combined = pd.concat([base_features, pca_features, gp_features], axis=1)
    combined = combined.sort_index(axis=1)
    return combined


def compute_latest_features(price_history: pd.DataFrame, context: OnlineFeatureContext) -> pd.DataFrame:
    panel = compute_feature_panel(price_history, context)
    latest_date = panel.index.max()
    return panel.loc[[latest_date]]
