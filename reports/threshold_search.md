# Threshold Search Report

The confidence policy was selected by comparing threshold triplets against realized validation error and classification Brier score.

## Selected triplet
- Accept threshold: 70.0
- Watch threshold: 50.0
- Accept coverage: 0.005
- Watch coverage: 0.008
- Abstain coverage: 0.987

## Search logic
- Thresholds were evaluated over a broad grid and ranked by how closely they matched the target coverage split while improving accepted-set MAE and Brier score.
- The selected triplet prioritizes the highest-confidence subset that still leaves enough coverage to preserve a usable decision surface.
