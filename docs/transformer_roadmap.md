# Transformer Development Roadmap

## Overview
This document tracks the remaining work required to deliver the multi-asset Transformer for spread forecasting. The focus is on building leak-free datasets, implementing the model in PyTorch with CUDA acceleration, and ensuring feature selection plus training efficiency.

## 1. Dataset Assembly & Pre-Processing
- ✅ Cleaned price panels and target constructions saved under `artifacts/`.
- ☐ Build rolling windows with configurable history (default **L = 192** trade days, adjustable). 192 captures ~9 months of daily data and keeps GPU memory moderate; we will compare L ∈ {128, 192, 256} during tuning.
- ☐ For each lag group (1–4), generate horizon-specific label tensors aligned to `date_id + lag`.
- ☐ Implement `Dataset`/`DataLoader` objects that emit `(features[t-L+1:t], targets[t+1:t+H])` sequences without forward leakage.
- ☐ Store train/validation/test splits with metadata (date ranges, lag mapping).

## 2. Feature Management & Selection
- ☐ Pre-screen features using variance check and simple IC/Spearman filters; drop constant/degenerate columns before Tensor export.
- ☐ Standardise features per training fold and persist scaling stats for inference.
- ☐ Integrate **Feature Gate** inside the model: learnable weights g ∈ [0,1] applied prior to projection, L1-penalised.
- ☐ Post-training, analyse gate magnitudes to derive a pruned feature list; optionally rerun fine-tuning with reduced inputs.

## 3. Model Implementation
- ☐ Input projection (shared across targets) + gated features.
- ☐ Time embeddings via Time2Vec and optional positional encodings.
- ☐ Transformer encoder stack (3–4 blocks, d_model 160–224, n_heads 4–6) with Dropout and LayerNorm.
- ☐ Separate output heads per lag horizon (4 heads) producing regression forecasts and pairwise score logits.

## 4. Losses & Metrics
- ☐ Regression head: Huber (δ = 0.2) or MSE hybrid.
- ☐ Rank head: pairwise logistic loss weighted by |Δy|.
- ☐ Feature gate regulariser: λ · ‖g‖₁ with warm-start schedule.
- ☐ Validation metrics: Spearman IC/ICIR, top-k precision, Q1-Q5 spread, gate sparsity statistics.

## 5. Training Loop & Efficiency
- ☐ PyTorch training loop with
  - CUDA + **AMP (automatic mixed precision)**.
  - Gradient clipping & AdamW (lr ≈ 1e-4).
  - Cosine / OneCycle LR scheduling.
- ☐ Attention kernel strategy
  - Prefer `torch.nn.functional.scaled_dot_product_attention` with `enable_flash=True` for FlashAttention compatibility.
  - Fallback to vanilla attention if hardware/driver unsupported.
- ☐ Optional gradient checkpointing & fused LayerNorm for larger configs.

## 6. Evaluation & Inference
- ☐ Rolling validation routine reporting metrics per lag and horizon.
- ☐ Inference script that stitches latest history with test rows, respecting lag-specific label shifts.
- ☐ Export submission-ready CSV once Kaggle evaluation pipeline is hooked up.

## 7. Reproducibility & Tooling
- ☐ Configuration system (YAML/JSON + dataclass loader) storing hyperparameters, paths, and random seeds.
- ☐ Checkpointing (best IC and last) and experiment logging (CSV/JSON + optional TensorBoard).
- ☐ Script to convert training pipeline into notebook-friendly format for exploratory use.

## Next Milestones
1. Implement dataset builders with history length = 192 baseline.
2. Integrate model components and feature gate; stand up AMP + FlashAttention-enabled training loop.
3. Produce first validation results, tune sequence length and hyperparameters, then finalise inference tooling.