from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src import config
from src.data.loading import load_train_labels
from src.data.pipeline import load_and_clean_prices
from src.data.targets import TargetSpec, build_target_frame, load_target_specs
from src.features.factors import compute_factor_features
from src.features.gp import GPArtifacts, generate_gp_features
from src.features.pca import PCAArtifacts, compute_pca_features
from src.features.stats import (
    compute_beta_features,
    compute_correlation_features,
    compute_fourier_features,
    compute_mean_reversion_features,
    compute_moment_features,
    compute_risk_features,
)
from src.features.technical import compute_technical_features


@dataclass
class FeatureArtifacts:
    pca: PCAArtifacts
    gp: GPArtifacts


FEATURE_DIR = config.OUTPUT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def build_target_price_frame() -> Tuple[pd.DataFrame, pd.DataFrame, list[TargetSpec]]:
    """Return target price panel, component log-price panel, and target specs."""

    clean_all = load_and_clean_prices("all")
    specs = load_target_specs()
    asset_names = sorted({asset for spec in specs for asset, _ in spec.terms})

    required_columns = ["date_id", *asset_names]
    missing = [col for col in asset_names if col not in clean_all.columns]
    if missing:
        raise KeyError(f"Missing asset columns in price data: {missing}")

    price_subset = clean_all[required_columns].copy()
    price_subset = price_subset.ffill().bfill()

    target_frame = build_target_frame(price_subset, specs=specs, index_col="date_id")
    log_prices = np.log(price_subset.set_index("date_id").clip(lower=1e-8))
    return target_frame, log_prices, specs


def build_base_features(
    target_frame: pd.DataFrame,
    log_price_frame: pd.DataFrame,
    specs: list[TargetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    technical = compute_technical_features(target_frame)
    factors = compute_factor_features(target_frame)
    correlations = compute_correlation_features(target_frame)
    betas = compute_beta_features(target_frame)
    moments = compute_moment_features(target_frame)
    risk = compute_risk_features(target_frame)
    fourier = compute_fourier_features(target_frame)
    mean_reversion = compute_mean_reversion_features(target_frame, log_price_frame, specs)

    base = pd.concat(
        [
            technical,
            factors,
            correlations,
            betas,
            moments,
            risk,
            fourier,
            mean_reversion,
        ],
        axis=1,
    )
    base = base.sort_index(axis=1)
    technical = technical.sort_index(axis=1)
    return base, technical


def _build_train_mask(target_frame: pd.DataFrame, train_labels: pd.DataFrame) -> pd.Series:
    train_dates = pd.Index(train_labels["date_id"].unique())
    mask = target_frame.index.isin(train_dates)
    return pd.Series(mask, index=target_frame.index, name="is_train")


def build_full_feature_set(
    sample_size: int = 1000,
    n_gp_components: int = 15,
    n_pca_components: int = 5,
) -> tuple[Dict[str, pd.DataFrame], FeatureArtifacts]:
    print('[Features] Building target and component price frames')
    target_frame, log_price_frame, specs = build_target_price_frame()
    print('[Features] Constructing base feature set')
    base_features, technical_features = build_base_features(target_frame, log_price_frame, specs)

    train_labels = load_train_labels()
    train_mask = _build_train_mask(target_frame, train_labels)

    print('[Features] Computing PCA loadings')
    pca_features, pca_artifacts = compute_pca_features(
        target_frame=target_frame,
        train_mask=train_mask,
        n_components=n_pca_components,
    )

    print('[Features] Fitting GP symbolic transformer')
    gp_features, gp_artifacts = generate_gp_features(
        base_features=technical_features,
        train_mask=train_mask,
        train_labels=train_labels,
        n_components=n_gp_components,
        sample_size=sample_size,
    )

    print('[Features] Combining all feature panels')
    all_features = pd.concat([base_features, pca_features, gp_features], axis=1)
    all_features = all_features.sort_index(axis=1)

    train_mask_bool = train_mask.astype(bool)
    outputs: Dict[str, pd.DataFrame] = {
        "target_prices": target_frame,
        "component_log_prices": log_price_frame,
        "base": base_features,
        "technical": technical_features,
        "pca": pca_features,
        "gp": gp_features,
        "all": all_features,
        "train_mask": train_mask,
        "target_prices_train": target_frame.loc[train_mask_bool],
        "target_prices_test": target_frame.loc[~train_mask_bool],
        "component_log_prices_train": log_price_frame.loc[train_mask_bool],
        "component_log_prices_test": log_price_frame.loc[~train_mask_bool],
        "base_train": base_features.loc[train_mask_bool],
        "base_test": base_features.loc[~train_mask_bool],
        "technical_train": technical_features.loc[train_mask_bool],
        "technical_test": technical_features.loc[~train_mask_bool],
        "pca_train": pca_features.loc[train_mask_bool],
        "pca_test": pca_features.loc[~train_mask_bool],
        "gp_train": gp_features.loc[train_mask_bool],
        "gp_test": gp_features.loc[~train_mask_bool],
        "all_train": all_features.loc[train_mask_bool],
        "all_test": all_features.loc[~train_mask_bool],
    }

    artifacts = FeatureArtifacts(pca=pca_artifacts, gp=gp_artifacts)
    return outputs, artifacts


def save_features(feature_frames: Dict[str, pd.DataFrame]) -> None:
    for name, frame in feature_frames.items():
        path = FEATURE_DIR / f"{name}.pkl"
        frame.to_pickle(path)


def build_and_save_features() -> FeatureArtifacts:
    print('[Features] Starting full feature rebuild')
    feature_frames, artifacts = build_full_feature_set()
    print('[Features] Saving feature artifacts to disk')
    save_features(feature_frames)
    print('[Features] Feature generation complete')
    return artifacts
