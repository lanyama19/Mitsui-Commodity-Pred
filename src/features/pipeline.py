from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from src import config
from src.data.loading import load_train_labels
from src.data.pipeline import load_and_clean_prices
from src.data.targets import build_target_frame, load_target_specs
from src.features.factors import compute_factor_features
from src.features.gp import GPArtifacts, generate_gp_features
from src.features.pca import PCAArtifacts, compute_pca_features
from src.features.technical import compute_technical_features


@dataclass
class FeatureArtifacts:
    pca: PCAArtifacts
    gp: GPArtifacts


FEATURE_DIR = config.OUTPUT_DIR / "features"
FEATURE_DIR.mkdir(exist_ok=True)


def build_target_price_frame() -> pd.DataFrame:
    clean_train = load_and_clean_prices("train")
    specs = load_target_specs()
    target_frame = build_target_frame(clean_train, specs=specs, index_col="date_id")
    return target_frame


def build_base_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    technical = compute_technical_features(target_frame)
    factors = compute_factor_features(target_frame)
    base = pd.concat([technical, factors], axis=1)
    base = base.sort_index(axis=1)
    return base


def build_full_feature_set(
    sample_size: int = 1000,
    n_gp_components: int = 15,
    n_pca_components: int = 5,
) -> tuple[Dict[str, pd.DataFrame], FeatureArtifacts]:
    target_frame = build_target_price_frame()
    base_features = build_base_features(target_frame)

    train_mask = pd.Series(True, index=target_frame.index, name="is_train")
    train_labels = load_train_labels()

    pca_features, pca_artifacts = compute_pca_features(
        target_frame=target_frame,
        train_mask=train_mask,
        n_components=n_pca_components,
    )

    gp_features, gp_artifacts = generate_gp_features(
        base_features=base_features,
        train_mask=train_mask,
        train_labels=train_labels,
        n_components=n_gp_components,
        sample_size=sample_size,
    )

    all_features = pd.concat([base_features, pca_features, gp_features], axis=1)
    all_features = all_features.sort_index(axis=1)

    outputs = {
        "target_prices": target_frame,
        "base": base_features,
        "pca": pca_features,
        "gp": gp_features,
        "all": all_features,
    }

    artifacts = FeatureArtifacts(pca=pca_artifacts, gp=gp_artifacts)
    return outputs, artifacts


def save_features(feature_frames: Dict[str, pd.DataFrame]) -> None:
    for name, frame in feature_frames.items():
        path = FEATURE_DIR / f"{name}.pkl"
        frame.to_pickle(path)


def build_and_save_features() -> FeatureArtifacts:
    feature_frames, artifacts = build_full_feature_set()
    save_features(feature_frames)
    return artifacts
