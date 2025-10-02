# LightGBM Full-Span Training (date_id ≤ 1960)

- **Parameter source**: `artifacts/grid_search/grid_lag1_v2.csv`, `grid_lag2_v2.csv`, `grid_lag3_v2.csv`, `grid_lag4_v2.csv` rows 117/15 exported via `scripts/export_lightgbm_params.py` into `configs/lightgbm_lag_params.json`.
- **Feature store**: `artifacts/features_v2/all_train.pkl`.
- **Weighting scheme**: `recency_exp` with decay=3.0, floor=0.5, cap=10.0 (normalised to mean 1).
- **Training command**:

```bash
python scripts/train_lightgbm_full.py \
  --param-config configs/lightgbm_lag_params.json \
  --train-end 1960 \
  --feature-version v2 \
  --artifact-root lightgbm_full \
  --run-prefix tunedfull \
  --weight-scheme recency_exp \
  --weight-decay 3.0 \
  --weight-floor 0.5 \
  --weight-cap 10 \
  --lags 1 2 3 4 \
  --bagging-freq 5 \
  --summary-path artifacts/lightgbm_full_summary_1960.json
```

- **Training metrics** (from `artifacts/lightgbm_full_summary_1960.json`):
  - lag1: train_ic_mean=0.5074, train_ir=1.98
  - lag2: train_ic_mean=0.6071, train_ir=2.47
  - lag3: train_ic_mean=0.6006, train_ir=2.47
  - lag4: train_ic_mean=0.6797, train_ir=3.01

- **Recency weight ranges** (from `artifacts/lightgbm_full/recency_weight_stats.json`):
  - lag1: min∈[0.1876, 0.1884], max∈[1.8766, 1.8838]
  - lag2: min∈[0.1874, 0.1885], max∈[1.8742, 1.8846]
  - lag3: min∈[0.1874, 0.1884], max∈[1.8736, 1.8844]
  - lag4: min∈[0.1870, 0.1885], max∈[1.8699, 1.8852]

Validation window is empty because the training span covers all available dates. Overfitting mitigation relies on the tuned λ₁/λ₂ regularisation, feature/bagging fractions, and the recency weighting scheme.
