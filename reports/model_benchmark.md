# Model Benchmark Overview

This report summarizes the first time-safe baseline benchmark for total corners and Over/Under markets.

## Regression winner
- poisson_regression (MAE 2.574)

## Classification winners
- over_8_5: logistic_regression (Brier 0.253)
- over_9_5: negative_binomial_probability (Brier 0.240)
- over_10_5: negative_binomial_probability (Brier 0.210)
- over_11_5: poisson_probability (Brier 0.159)

## Notes
- Chronological split: train 2023/24-2024/25, validate 2025/26.
- No current-match or post-match fields were used as features.
