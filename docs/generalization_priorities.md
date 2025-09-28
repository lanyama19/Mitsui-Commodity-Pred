# Transformer Generalisation Roadmap

## Generalisation Levers
- **Feature Robustness**: apply rolling median detrending, MAD-based scaling, and regime segmentation to curb outliers and distribution shifts.
- **Multi-Scale Signals**: enrich inputs with wavelet/DFT energy across 5-30 day bands and Hilbert phase/instantaneous frequency features to expose cyclical structure beyond raw returns.
- **Cross-Asset Context**: compute rolling cross-asset correlation, covariance, beta, and curve-shape metrics (term-structure slope, basis) to capture inter-market linkages.
- **Symbolic/Nonlinear Augmentation**: evolve genetic-programming feature templates with stability penalties (L1/L0) and regime-aware operators (conditional smoothing, rank) to expand expressive but controlled signals.
- **Self-Supervised Pretraining**: pretrain the transformer with masked time-series/denoising objectives on combined train+test spans before supervised fine-tuning, leveraging broader dynamics.
- **Architecture Tweaks**: integrate multi-scale or probabilistic attention (Informer/Fedformer style) and share a backbone across lags with lag-specific heads to encourage pattern reuse; add local conv/Hyena blocks for short-term precision.
- **Regularisation & Training Tricks**: mixout, weight decay, stochastic depth, random window cropping, label smoothing, and checkpoint ensembling to mitigate overfit while keeping training stable.
- **Evaluation Discipline**: expand walk-forward validation with per-regime IC/IR reporting to ensure improvements transfer out of sample.

## MVP-Aligned Priority Queue
1. **Walk-Forward IC/IR Reporting** ? instrumentation-only change; unlocks fast feedback on generalisation without touching model.
2. **Rolling Robust Scaling (MAD/median)** ? lightweight preprocessing tweak improving stability of existing features.
3. **Cross-Asset Summary Features** ? reuse current pipelines to add correlation/beta/basis metrics with minimal architectural edits.
4. **Feature Dropout + Mixout Regularisation** ? small training-config updates to restrain overfitting.
5. **Lag-Shared Transformer Backbone** ? adjust output heads while preserving the core implementation, promoting data efficiency.
6. **Multi-Scale Frequency Features** ? extend feature builder with wavelet/DFT descriptors, moderate code addition.
7. **Self-Supervised Pretraining Phase** ? introduce masked prediction pretraining before supervised fine-tune; requires extra training loop but reuses model.
8. **ProbSparse / Multi-Scale Attention** ? deeper architectural change adopting Informer/Fedformer modules once prior steps plateau.
9. **Enhanced GP Feature Search** ? larger search space with regularised symbolic operators after base model is stable.
