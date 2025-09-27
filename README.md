# Mitsui-Commodity-Pred

This repository targets cross-asset spread forecasting for commodities, FX, and related instruments. The current focus is on data preparation and feature engineering; the Transformer model will be added in the next phase.

## Data Preprocessing
- **Missing data handling**: `fill_series_with_trend` (see `src/data/cleaning.py`) extrapolates missing observations using the average increment over the latest `window` valid points, guarding against look-ahead leaks.
- **Pipeline entry**: `build_all_cleaned(window=5)` materialises cleaned `train`, `test`, and `all` price tables, stored as `artifacts/clean_*.pkl`.
- **Target construction**: `src/data/targets.py` parses `target_pairs.csv` into `TargetSpec` objects and combines log prices to produce a 424-dimensional target spread matrix.

## Feature Generation
The orchestration lives in `src/features/pipeline.py` and produces several reusable blocks:
- **Technical indicators** (`technical.py`): ~45 TA-Lib indicators per target (moving averages, MACD, Bollinger Bands, RSI, etc.).
- **Classical factors** (`factors.py`): return lags, volatility terms, Sharpe-style ratios, momentum, reversal, and range statistics (~15 features per target).
- **PCA latent factors** (`pca.py`): top 5 principal-component loadings derived from standardised target returns, broadcast to every timestamp.
- **Genetic programming features** (`gp.py`): 15 symbolic transformations learned via `gplearn` with compatibility patches for modern scikit-learn.

Running `build_and_save_features()` writes the following under `artifacts/features/`:
- `target_prices.pkl`: log-price spreads for all targets.
- `base.pkl`: technical + factor features.
- `pca.pkl`: PCA loading features.
- `gp.pkl`: GP-derived features.
- `all.pkl`: concatenation of the three feature groups.

## Quick Start
```powershell
# Activate environment
d:\py\genAI\venv\Scripts\Activate.ps1

# Install dependencies (first run only)
pip install -r requirements.txt

# Generate cleaned price panels
python - <<'PY'
from src.data.pipeline import build_all_cleaned
build_all_cleaned(window=5)
PY

# Generate feature matrices
python - <<'PY'
from src.features.pipeline import build_and_save_features
build_and_save_features()
PY
```

## Code Structure
```
src/
├── config.py              # Paths and shared configuration
├── data/
│   ├── cleaning.py        # Missing-value extrapolation and helpers
│   ├── loading.py         # CSV loaders
│   ├── pipeline.py        # Cleaning pipeline entrypoints
│   └── targets.py         # Target specification parsing & construction
└── features/
    ├── technical.py       # TA-Lib feature block
    ├── factors.py         # Financial factor block
    ├── pca.py             # PCA latent factors + artifacts
    ├── gp.py              # Genetic programming features
    └── pipeline.py        # Feature assembly & persistence
```

---
Upcoming work: build sequence datasets, implement the multi-asset Transformer, and add evaluation/inference scripts.