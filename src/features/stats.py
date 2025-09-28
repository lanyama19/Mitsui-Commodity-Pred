"""Statistical feature generators (risk, spectral, mean-reversion)."""
from __future__ import annotations

from typing import Dict, Iterable

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view

from src.data.targets import TargetSpec

# Correlation settings
CORRELATION_WINDOWS = (30, 60, 90)

# Market beta settings
BETA_WINDOWS = (22, 66)

# Higher-moment settings
MOMENT_WINDOWS = (30, 60, 90)

# Risk settings
VAR_LEVELS = (0.01, 0.05)
VAR_WINDOW = 90
HILL_WINDOW = 90
HILL_MIN_K = 5
HILL_FRACTION = 0.1

# Fourier settings
FOURIER_WINDOWS = (7, 30, 90, 360)

# Mean-reversion settings
HURST_WINDOWS = (30, 90)
OU_WINDOWS = (30, 90)
COINT_WINDOWS = (30, 90)
EPS = 1e-8


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _wrap_with_feature_name(frame: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    wrapped = frame.copy()
    wrapped.columns = pd.MultiIndex.from_arrays([
        wrapped.columns,
        [feature_name] * len(wrapped.columns),
    ])
    return wrapped


def _equal_weight_market_return(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)


# ---------------------------------------------------------------------------
# Correlation & beta features
# ---------------------------------------------------------------------------

def compute_correlation_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    returns = target_frame.diff()
    market_return = _equal_weight_market_return(returns)

    features = []
    for window in CORRELATION_WINDOWS:
        corr = returns.rolling(window, min_periods=window).corr(market_return)
        corr = corr.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        features.append(_wrap_with_feature_name(corr, f"corr_market_{window}"))
    combined = pd.concat(features, axis=1)
    return combined.sort_index(axis=1)


def compute_beta_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    returns = target_frame.diff()
    market_return = _equal_weight_market_return(returns)

    features = []
    for window in BETA_WINDOWS:
        cov = returns.rolling(window, min_periods=window).cov(market_return)
        var = market_return.rolling(window, min_periods=window).var()
        beta = cov.divide(var, axis=0)
        beta = beta.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        features.append(_wrap_with_feature_name(beta, f"beta_{window}"))
    combined = pd.concat(features, axis=1)
    return combined.sort_index(axis=1)


# ---------------------------------------------------------------------------
# Higher-moment features
# ---------------------------------------------------------------------------

def compute_moment_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    returns = target_frame.diff()
    frames = []
    for window in MOMENT_WINDOWS:
        skew = returns.rolling(window, min_periods=window).skew().fillna(0.0)
        kurt = returns.rolling(window, min_periods=window).kurt().fillna(0.0)
        frames.append(_wrap_with_feature_name(skew, f"skew_{window}"))
        frames.append(_wrap_with_feature_name(kurt, f"kurt_{window}"))
    combined = pd.concat(frames, axis=1)
    return combined.sort_index(axis=1)


# ---------------------------------------------------------------------------
# Risk features (VaR & Hill tail index)
# ---------------------------------------------------------------------------

def _rolling_var(returns: pd.DataFrame, level: float) -> pd.DataFrame:
    quantile = returns.rolling(VAR_WINDOW, min_periods=VAR_WINDOW).quantile(level)
    var = (-quantile).fillna(0.0)
    return var


def _hill_estimator_window(arr: np.ndarray, k: int) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size <= k:
        return 0.0
    losses = np.clip(-arr, a_min=0.0, a_max=None)
    losses = np.sort(losses)
    tail = losses[-k:]
    tail = tail[tail > 0]
    if tail.size < k:
        return 0.0
    x_min = tail.min()
    if x_min <= 0:
        return 0.0
    logs = np.log(tail) - np.log(x_min)
    denom = logs.sum()
    if denom <= 0:
        return 0.0
    return float(k / denom)


def _rolling_hill(returns: pd.DataFrame) -> pd.DataFrame:
    k_default = max(HILL_MIN_K, int(HILL_WINDOW * HILL_FRACTION))

    def hill_func(arr: np.ndarray) -> float:
        return _hill_estimator_window(arr, k_default)

    hill = returns.rolling(HILL_WINDOW, min_periods=HILL_WINDOW).apply(hill_func, raw=True)
    return hill.fillna(0.0)


def compute_risk_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    returns = target_frame.diff()
    features = []
    for level in VAR_LEVELS:
        var = _rolling_var(returns, level)
        features.append(_wrap_with_feature_name(var, f"var_{int(level * 100)}"))
    hill = _rolling_hill(returns)
    features.append(_wrap_with_feature_name(hill, "hill_alpha"))
    combined = pd.concat(features, axis=1)
    return combined.sort_index(axis=1)


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------

def _first_fft_component(values: np.ndarray) -> tuple[float, float]:
    if np.isnan(values).any():
        return 0.0, 0.0
    demeaned = values - values.mean()
    if np.allclose(demeaned, 0.0):
        return 0.0, 0.0
    spectrum = np.fft.rfft(demeaned)
    if spectrum.size < 2:
        return 0.0, 0.0
    comp = spectrum[1]
    amp = float(np.abs(comp) / demeaned.size)
    phase = float(np.angle(comp))
    return amp, phase


def _apply_fft(series: pd.Series, window: int, metric: str) -> pd.Series:
    def func(arr: np.ndarray) -> float:
        amp, phase = _first_fft_component(arr)
        return amp if metric == "amp" else phase

    return series.rolling(window, min_periods=window).apply(func, raw=True).fillna(0.0)


def compute_fourier_features(target_frame: pd.DataFrame) -> pd.DataFrame:
    feature_blocks = []
    for column in target_frame.columns:
        series = target_frame[column]
        feats = {}
        for window in FOURIER_WINDOWS:
            feats[f"fourier_amp_{window}"] = _apply_fft(series, window, "amp")
            feats[f"fourier_phase_{window}"] = _apply_fft(series, window, "phase")
        frame = pd.DataFrame(feats, index=target_frame.index)
        frame.columns = pd.MultiIndex.from_product([[column], frame.columns])
        feature_blocks.append(frame)
    combined = pd.concat(feature_blocks, axis=1)
    return combined.sort_index(axis=1)


# ---------------------------------------------------------------------------
# Mean-reversion features (Hurst, OU parameters, cointegration residuals)
# ---------------------------------------------------------------------------

def _compute_hurst_features(values: np.ndarray, windows: tuple[int, ...]) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    n = len(values)
    for window in windows:
        output = np.full(n, 0.5, dtype=np.float64)
        if n >= window:
            windows_view = sliding_window_view(values, window)
            demeaned = windows_view - windows_view.mean(axis=1, keepdims=True)
            std = np.sqrt((demeaned ** 2).mean(axis=1))
            cumulative = np.cumsum(demeaned, axis=1)
            ranges = cumulative.max(axis=1) - cumulative.min(axis=1)
            valid = (std > EPS) & (ranges > EPS)
            hurst_vals = np.full(windows_view.shape[0], 0.5, dtype=np.float64)
            hurst_vals[valid] = np.log(ranges[valid] / std[valid]) / np.log(window)
            output[window - 1 :] = hurst_vals
        results[f"hurst_{window}"] = output
    return results


def _compute_ou_features(values: np.ndarray, windows: tuple[int, ...]) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    n = len(values)
    for window in windows:
        speed = np.zeros(n, dtype=np.float64)
        long_mean = np.zeros(n, dtype=np.float64)
        if n >= window:
            windows_view = sliding_window_view(values, window)
            x_prev = windows_view[:, :-1]
            x_next = windows_view[:, 1:]
            mean_prev = x_prev.mean(axis=1)
            mean_next = x_next.mean(axis=1)
            demean_prev = x_prev - mean_prev[:, None]
            demean_next = x_next - mean_next[:, None]
            var_prev = (demean_prev ** 2).mean(axis=1)
            cov = (demean_prev * demean_next).mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                b = np.divide(cov, var_prev, out=np.zeros_like(cov), where=var_prev > EPS)
            a = mean_next - b * mean_prev
            valid = (b > 0) & (b < 1)
            tmp_speed = np.zeros_like(b)
            tmp_mean = np.zeros_like(b)
            tmp_speed[valid] = -np.log(b[valid])
            denom = 1 - b
            mean_mask = (np.abs(denom) > EPS) & np.isfinite(a)
            tmp_mean[mean_mask] = np.divide(a[mean_mask], denom[mean_mask])
            speed[window-1:] = tmp_speed
            long_mean[window-1:] = tmp_mean
        results[f"ou_speed_{window}"] = speed
        results[f"ou_mean_{window}"] = long_mean
    return results


def _compute_cointegration_features(y: np.ndarray, X: np.ndarray, windows: tuple[int, ...]) -> Dict[str, np.ndarray]:
    n = len(y)
    features: Dict[str, np.ndarray] = {}
    for window in windows:
        residuals = np.zeros(n, dtype=np.float64)
        if n >= window:
            for idx in range(window - 1, n):
                y_win = y[idx - window + 1 : idx + 1]
                X_win = X[idx - window + 1 : idx + 1]
                XtX = X_win.T @ X_win
                XtY = X_win.T @ y_win
                try:
                    beta = np.linalg.solve(XtX, XtY)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(X_win, y_win, rcond=None)[0]
                intercept = y_win.mean() - X_win.mean(axis=0) @ beta
                residuals[idx] = y[idx] - (X[idx] @ beta + intercept)
        features[f"coint_resid_{window}"] = residuals
    return features


def _build_cointegration_matrices(
    spec: TargetSpec,
    target_values: np.ndarray,
    log_price_frame: pd.DataFrame,
    market_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.is_single_asset:
        asset = spec.terms[0][0]
        if asset in log_price_frame.columns:
            y = log_price_frame[asset].to_numpy(dtype=np.float64)
        else:
            y = target_values
        X = market_index[:, None]
    else:
        assets = [term[0] for term in spec.terms]
        base_assets = [asset for asset in assets if asset in log_price_frame.columns]
        if not base_assets:
            y = target_values
            X = market_index[:, None]
        else:
            y = log_price_frame[base_assets[0]].to_numpy(dtype=np.float64)
            remaining = base_assets[1:]
            if remaining:
                X = log_price_frame[remaining].to_numpy(dtype=np.float64)
            else:
                X = market_index[:, None]
    if X.ndim == 1:
        X = X[:, None]
    return y, X


def _compute_target_mean_reversion(
    column: str,
    spec: TargetSpec,
    target_values: np.ndarray,
    log_price_frame: pd.DataFrame,
    market_index: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    features.update(_compute_hurst_features(target_values, HURST_WINDOWS))
    features.update(_compute_ou_features(target_values, OU_WINDOWS))
    y, X = _build_cointegration_matrices(spec, target_values, log_price_frame, market_index)
    features.update(_compute_cointegration_features(y, X, COINT_WINDOWS))
    df = pd.DataFrame(features, index=index)
    df.columns = pd.MultiIndex.from_product([[column], df.columns])
    return df


def compute_mean_reversion_features(
    target_frame: pd.DataFrame,
    log_price_frame: pd.DataFrame,
    specs: Iterable[TargetSpec],
    n_jobs: int | None = None,
) -> pd.DataFrame:
    print('[Features] Mean-reversion: preparing tasks')
    spec_map: Dict[str, TargetSpec] = {spec.name: spec for spec in specs}
    returns = log_price_frame.diff()
    market_return = returns.mean(axis=1).fillna(0.0).to_numpy(dtype=np.float64)
    market_index = np.cumsum(market_return)
    index = target_frame.index

    if n_jobs is None:
        cpu_count = os.cpu_count() or 1
        n_jobs = max(cpu_count - 2, 1)
        if cpu_count > 4:
            n_jobs = min(n_jobs, 4)

    tasks = []
    for column in target_frame.columns:
        spec = spec_map[column]
        target_values = target_frame[column].to_numpy(dtype=np.float64)
        tasks.append((column, spec, target_values))

    def executor(args: tuple[str, TargetSpec, np.ndarray]) -> pd.DataFrame:
        column, spec, target_values = args
        return _compute_target_mean_reversion(
            column,
            spec,
            target_values,
            log_price_frame,
            market_index,
            index,
        )

    if n_jobs == 1:
        print('[Features] Mean-reversion: processing sequentially')
        results = [executor(task) for task in tasks]
    else:
        print(f'[Features] Mean-reversion: running in parallel with {n_jobs} workers')
        results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(executor)(task) for task in tasks)

    combined = pd.concat(results, axis=1)
    print('[Features] Mean-reversion: aggregation complete')
    return combined.sort_index(axis=1).fillna(0.0)
