from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.utils.validation import check_X_y


@dataclass
class GPArtifacts:
    model: SymbolicTransformer
    feature_names: list[str]


def _ensure_validate_data(model: SymbolicTransformer) -> None:
    if hasattr(model, "_validate_data") and hasattr(model, "_more_tags"):
        return

    def _validate_data(self, X, y, y_numeric=True):
        X, y = check_X_y(X, y, y_numeric=y_numeric)
        self.n_features_in_ = X.shape[1]
        return X, y

    model._validate_data = MethodType(_validate_data, model)

    def _more_tags(self):
        return {"X_types": ["numeric"], "requires_y": True}

    model._more_tags = MethodType(_more_tags, model)


def generate_gp_features(
    base_features: pd.DataFrame,
    train_mask: pd.Series,
    train_labels: pd.DataFrame,
    n_components: int = 15,
    sample_size: int = 1000,
    random_state: int = 123,
) -> tuple[pd.DataFrame, GPArtifacts]:
    stacked_features = base_features.stack(level=0, future_stack=True)
    stacked_features.index.set_names(["date_id", "target"], inplace=True)
    stacked_features = stacked_features.sort_index()

    labels = train_labels.set_index("date_id")
    target_series = labels.stack(future_stack=True).rename("label")
    target_series.index.set_names(["date_id", "target"], inplace=True)
    target_series = target_series.sort_index()

    train_idx = stacked_features.index.intersection(target_series.index)
    train_dates = pd.Index(train_idx.get_level_values("date_id"))
    mask_series = train_mask.groupby(level=0).max() if train_mask.index.has_duplicates else train_mask
    train_filter = train_dates.map(mask_series).fillna(False).to_numpy(dtype=bool)
    train_idx = train_idx[train_filter]

    X_train = stacked_features.loc[train_idx]
    y_train = target_series.loc[train_idx]
    valid_mask = ~y_train.isna()
    X_train = X_train.loc[valid_mask]
    y_train = y_train.loc[valid_mask].to_numpy()

    if sample_size and len(X_train) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_positions = rng.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train.to_numpy(dtype=float)[sample_positions]
        y_sample = y_train[sample_positions]
    else:
        X_sample = X_train.to_numpy(dtype=float)
        y_sample = y_train

    feature_names = list(stacked_features.columns)
    gp = SymbolicTransformer(
        generations=3,
        population_size=200,
        hall_of_fame=40,
        n_components=n_components,
        verbose=0,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "sin", "cos"),
        parsimony_coefficient=0.0005,
        feature_names=feature_names,
        n_jobs=1,
        random_state=random_state,
    )

    _ensure_validate_data(gp)
    gp.fit(X_sample, y_sample)

    transformed = gp.transform(stacked_features.to_numpy(dtype=float))
    gp_columns = [f"gp_{i+1}" for i in range(n_components)]
    gp_df = pd.DataFrame(transformed, index=stacked_features.index, columns=gp_columns)
    gp_df = gp_df.unstack(level="target")
    gp_df = gp_df.swaplevel(axis=1).sort_index(axis=1)

    artifacts = GPArtifacts(model=gp, feature_names=feature_names)
    return gp_df, artifacts
