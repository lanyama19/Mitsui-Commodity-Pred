# Multi-Asset Transformer Architecture

This document details the planned architecture for the commodity spread forecasting Transformer, expanding on `multi_asset_transformer.md` with implementation-oriented notes.

## 1. Inputs & Feature Handling
- **Feature tensor**: shape `[batch, seq_len, F]`, where `F` is the union of base, PCA, and GP features after optional pre-screening.
- **Feature gate**: learnable parameter `g ∈ ℝ^F`, constrained to `[0,1]` via sigmoid. Applied elementwise before the input projection: `X_gated = X ⊙ g`.
- **Input projection**: linear layer `W_in: F → d_model` shared across targets, followed by dropout.

## 2. Temporal Embedding
- **Time2Vec** embedding (k = 4 harmonics) computed on relative time index; concatenated with projected features and re-projected to `d_model`.
- Optional positional encoding (sinusoidal) can be added if Time2Vec alone underperforms.

## 3. Encoder Stack
- **Depth**: 3–4 Transformer encoder blocks (Pre-LN).
- **Hidden size**: `d_model = 192` baseline (compatible with seq_len 192); adjustable in {160, 192, 224}.
- **Attention heads**: 6 heads, head dim = 32.
- **Feed-forward**: 2-layer MLP with expansion factor 4, GELU activation, dropout 0.1.
- **Attention kernel**: use `scaled_dot_product_attention` with `enable_flash=True` to leverage FlashAttention when available; gracefully fall back otherwise.
- **Regularisation**: dropout on attention probabilities and residuals, stochastic depth optional (p≤0.1).

## 4. Output Heads
- **Lag-specific branches**: four heads (lag 1–4). Each head processes the final hidden state sequence via:
  1. Temporal pooling (last token or attention pooling) to form context vector.
  2. Regression MLP predicting horizon `H = 20` returns per target.
  3. Pairwise scoring module producing logits for ranking loss (implemented via differences of per-target scores).
- **Normalization**: predictions are z-scored per date before loss to focus on relative ordering.

## 5. Loss Components
- **Regression loss**: Huber (δ = 0.2) over z-scored predictions vs. targets.
- **Rank loss**: logistic pairwise loss weighted by |y_i − y_j| within each lag/date mini-batch.
- **Feature gate penalty**: `λ · ‖g‖₁` with warm-up schedule (λ starts 0, ramp to 1e-3).
- Total loss: `α·L_reg + β·L_rank + λ·‖g‖₁`, with `(α, β) ≈ (0.3, 1.0)`.

## 6. Training Strategy
- **Optimiser**: AdamW (`lr=1e-4`, `betas=(0.9,0.999)`, weight decay 1e-4).
- **Scheduler**: cosine decay with warmup (5%) or OneCycle.
- **Mixed precision**: PyTorch AMP with `GradScaler` for stability.
- **Gradient clipping**: 1.0.
- **Batching**: group samples by date & lag to compute pairwise losses efficiently (target count ≤ ~100 per batch).
- **Checkpointing**: save best IC and last; store feature gate vector per checkpoint.

## 7. Evaluation Metrics
- **Spearman IC / ICIR** per lag & horizon.
- **Top-k hit rate** (k ∈ {5, 10}).
- **Quantile spread** (Q1–Q5 average difference).
- **Gate sparsity**: proportion of features with g < threshold.

## 8. Sequence Length Discussion
- Baseline `seq_len = 192` (~9 months of trading days) balances temporal context with GPU cost (~8 GB for batch size 8 under AMP). We will benchmark `seq_len ∈ {128, 192, 256}` during validation; if memory allows and performance improves, 256 can be adopted, otherwise 192 becomes default.

## 9. Implementation Notes
- Modular design (`src/models/` package) with configurable attention backend, enabling toggling FlashAttention.
- Support TorchScript/ONNX export after inference pipeline stabilises.
- Logging hooks to monitor per-head metrics and feature gate values.
