from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factor_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    returns = target_frame.diff()
    feature_frames = []
    for column in target_frame.columns:
        price = target_frame[column]
        ret = returns[column]
        feats = pd.DataFrame(index=target_frame.index)
        feats["ret_1"] = ret
        feats["ret_5"] = target_frame[column].diff(5)
        feats["ret_10"] = target_frame[column].diff(10)
        feats["ret_20"] = target_frame[column].diff(20)

        for window in (5, 20, 60):
            feats[f"vol_{window}"] = ret.rolling(window, min_periods=1).std()

        for window in (20, 60):
            rolling_mean = ret.rolling(window, min_periods=1).mean()
            rolling_std = ret.rolling(window, min_periods=1).std()
            sharpe = np.divide(rolling_mean, rolling_std, out=np.zeros_like(rolling_mean), where=rolling_std != 0)
            feats[f"sharpe_{window}"] = sharpe
            zscore = (price - price.rolling(window, min_periods=1).mean()) / price.rolling(window, min_periods=1).std()
            feats[f"zscore_{window}"] = zscore

        feats["momentum_20"] = ret.rolling(20, min_periods=1).sum()
        feats["momentum_60"] = ret.rolling(60, min_periods=1).sum()
        feats["reversal_5"] = -ret.shift(1).rolling(5, min_periods=1).mean()
        feats["rolling_min_20"] = price.rolling(20, min_periods=1).min()

        feats = feats.ffill().bfill()
        feats.columns = pd.MultiIndex.from_product([[column], feats.columns])
        feature_frames.append(feats)
    return pd.concat(feature_frames, axis=1)
