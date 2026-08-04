# Confidence Calibration Report

This report summarizes how the calibrated confidence policy improves accepted predictions relative to the full validation set.

## Selected policy
- Accept threshold: 70.0
- Watch threshold: 50.0
- Accept coverage: 0.005
- Watch coverage: 0.008
- Abstain coverage: 0.987

## Validation outcome
- Full MAE: 2.574
- Accepted MAE: 0.252
- Full Brier: 0.289
- Accepted Brier: 0.191

## Interpretation
- The confidence engine now uses validation residuals and classification error to re-rank pre-match confidence rather than relying on the raw heuristic score alone.
- The selected thresholds create a narrower acceptance band, making the engine abstain or watch more often when validation evidence is weak.
