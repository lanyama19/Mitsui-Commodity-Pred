from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAArtifacts:
    scaler: StandardScaler
    pca: PCA
    loadings: pd.DataFrame


def compute_pca_features(
    target_frame: pd.DataFrame,
    train_mask: pd.Series,
    n_components: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, PCAArtifacts]:
    returns = target_frame.diff().dropna()
    aligned_mask = train_mask.loc[returns.index]
    train_returns = returns.loc[aligned_mask]

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_train = scaler.fit_transform(train_returns)

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(scaled_train)

    component_names = [f"pca_loading_{i+1}" for i in range(n_components)]
    loadings = pd.DataFrame(pca.components_.T, index=returns.columns, columns=component_names)

    repeated_frames = []
    for target in target_frame.columns:
        row = loadings.loc[target]
        repeated = np.repeat(row.values.reshape(1, -1), len(target_frame.index), axis=0)
        df = pd.DataFrame(repeated, index=target_frame.index, columns=component_names)
        df.columns = pd.MultiIndex.from_product([[target], df.columns])
        repeated_frames.append(df)

    features = pd.concat(repeated_frames, axis=1)
    artifacts = PCAArtifacts(scaler=scaler, pca=pca, loadings=loadings)
    return features, artifacts
