from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import talib as ta

TECH_WINDOWS = (5, 10, 20, 60)
RSI_PERIODS = (14, 28)
MOM_PERIODS = (5, 10, 20)
ROC_PERIODS = (5, 10, 20)


def _fill_initial_nan(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.ffill().bfill()


def _technical_for_series(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    feats: Dict[str, np.ndarray] = {}

    for window in TECH_WINDOWS:
        feats[f"sma_{window}"] = ta.SMA(values, timeperiod=window)
        feats[f"ema_{window}"] = ta.EMA(values, timeperiod=window)

    for period in RSI_PERIODS:
        feats[f"rsi_{period}"] = ta.RSI(values, timeperiod=period)

    for period in MOM_PERIODS:
        feats[f"mom_{period}"] = ta.MOM(values, timeperiod=period)

    for period in ROC_PERIODS:
        feats[f"roc_{period}"] = ta.ROC(values, timeperiod=period)

    macd, macd_signal, macd_hist = ta.MACD(values, fastperiod=12, slowperiod=26, signalperiod=9)
    feats["macd"] = macd
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_hist

    feats["apo"] = ta.APO(values)
    feats["ppo"] = ta.PPO(values)
    feats["trix_15"] = ta.TRIX(values, timeperiod=15)
    feats["tema_10"] = ta.TEMA(values, timeperiod=10)
    feats["kama_10"] = ta.KAMA(values, timeperiod=10)
    feats["dema_20"] = ta.DEMA(values, timeperiod=20)

    upper, middle, lower = ta.BBANDS(values, timeperiod=20)
    feats["bb_upper_20"] = upper
    feats["bb_middle_20"] = middle
    feats["bb_lower_20"] = lower

    feats["linreg_14"] = ta.LINEARREG(values, timeperiod=14)
    feats["linreg_slope_14"] = ta.LINEARREG_SLOPE(values, timeperiod=14)

    frame = pd.DataFrame(feats, index=index)
    frame = _fill_initial_nan(frame)
    return frame


def compute_technical_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    feature_frames = []
    for column in target_frame.columns:
        series = target_frame[column].to_numpy(dtype=float)
        feats = _technical_for_series(series, target_frame.index)
        feats.columns = pd.MultiIndex.from_product([[column], feats.columns])
        feature_frames.append(feats)
    return pd.concat(feature_frames, axis=1)
