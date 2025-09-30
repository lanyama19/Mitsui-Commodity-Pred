from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src import config
from src.data.loading import load_train_labels
from src.data.targets import TargetSpec
from src.features.pipeline import (
    FeatureArtifacts,
    _build_train_mask,
    _ensure_unique_columns,
    _ensure_unique_index,
    build_target_price_frame,
)
from src.features.factors import compute_factor_features
from src.features.gp import generate_gp_features
from src.features.pca import compute_pca_features
from src.features.stats import (
    compute_beta_features,
    compute_correlation_features,
    compute_fourier_features,
    compute_mean_reversion_features,
    compute_moment_features,
    compute_risk_features,
)
from src.features.technical_v2 import compute_continuous_technical_features


FEATURE_V2_DIR = config.OUTPUT_DIR / "features_v2"
FEATURE_V2_DIR.mkdir(parents=True, exist_ok=True)



def build_base_features_v2(
    target_frame: pd.DataFrame,
    log_price_frame: pd.DataFrame,
    specs: list[TargetSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nav_technical, nav_frame = compute_continuous_technical_features(target_frame, specs)
    nav_technical = _ensure_unique_columns(nav_technical, "technical_v2")
    nav_technical = _ensure_unique_index(nav_technical, "technical_v2")
    nav_frame = _ensure_unique_index(nav_frame, "target_nav")

    factors = compute_factor_features(target_frame)
    correlations = compute_correlation_features(target_frame)
    betas = compute_beta_features(target_frame)
    moments = compute_moment_features(target_frame)
    risk = compute_risk_features(target_frame)
    fourier = compute_fourier_features(target_frame)
    mean_reversion = compute_mean_reversion_features(target_frame, log_price_frame, specs)

    base = pd.concat(
        [
            nav_technical,
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
    base = _ensure_unique_columns(base, "base_v2")
    base = _ensure_unique_index(base, "base_v2")

    return base.sort_index(axis=1), nav_technical.sort_index(axis=1), nav_frame



def build_full_feature_set_v2(
    sample_size: int = 1000,
    n_gp_components: int = 15,
    n_pca_components: int = 5,
) -> tuple[Dict[str, pd.DataFrame], FeatureArtifacts]:
    print('[Features-v2] Building target and component price frames')
    target_frame, log_price_frame, specs = build_target_price_frame()

    specs = list(specs)

    print('[Features-v2] Constructing base feature set with NAV-derived technicals')
    base_features, technical_features, nav_frame = build_base_features_v2(
        target_frame, log_price_frame, specs
    )

    train_labels = load_train_labels()
    train_mask = _build_train_mask(target_frame, train_labels)

    print('[Features-v2] Computing PCA loadings')
    pca_features, pca_artifacts = compute_pca_features(
        target_frame=target_frame,
        train_mask=train_mask,
        n_components=n_pca_components,
    )
    pca_features = _ensure_unique_columns(pca_features, "pca_v2")
    pca_features = _ensure_unique_index(pca_features, "pca_v2")

    print('[Features-v2] Fitting GP symbolic transformer using NAV technicals')
    gp_features, gp_artifacts = generate_gp_features(
        base_features=technical_features,
        train_mask=train_mask,
        train_labels=train_labels,
        n_components=n_gp_components,
        sample_size=sample_size,
    )
    gp_features = _ensure_unique_columns(gp_features, "gp_v2")
    gp_features = _ensure_unique_index(gp_features, "gp_v2")

    print('[Features-v2] Combining all feature panels')
    all_features = pd.concat([base_features, pca_features, gp_features], axis=1)
    all_features = _ensure_unique_columns(all_features, "all_v2")
    all_features = _ensure_unique_index(all_features, "all_v2")
    all_features = all_features.sort_index(axis=1)

    train_mask_bool = train_mask.astype(bool)

    outputs: Dict[str, pd.DataFrame] = {
        "target_prices": target_frame,
        "component_log_prices": log_price_frame,
        "target_nav": nav_frame,
        "base": base_features,
        "technical": technical_features,
        "pca": pca_features,
        "gp": gp_features,
        "all": all_features,
        "train_mask": train_mask,
        "target_prices_train": target_frame.loc[train_mask_bool],
        "target_prices_test": target_frame.loc[~train_mask_bool],
        "target_nav_train": nav_frame.loc[train_mask_bool],
        "target_nav_test": nav_frame.loc[~train_mask_bool],
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



def save_features_v2(feature_frames: Dict[str, pd.DataFrame]) -> None:
    FEATURE_V2_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in feature_frames.items():
        path = FEATURE_V2_DIR / f"{name}.pkl"
        frame.to_pickle(path)



def build_and_save_features_v2() -> FeatureArtifacts:
    print('[Features-v2] Starting NAV-based feature rebuild')
    feature_frames, artifacts = build_full_feature_set_v2()
    print('[Features-v2] Saving NAV feature artifacts to disk')
    save_features_v2(feature_frames)
    print('[Features-v2] Feature generation v2 complete')
    return artifacts







