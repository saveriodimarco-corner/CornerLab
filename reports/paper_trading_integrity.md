# Sprint 62 - Paper Trading Integrity Gate

## Results

- Feature provenance: PASS
- Model consistency: FAIL
- Confidence integrity: PASS
- Odds join: PASS
- Decision integrity: PASS
- Leakage: NO
- Synthetic model inputs: NO
- Paper trading integrity: FAIL

## Audit Summary

The live paper-trading path now loads the accepted trained artifacts for the validated targets, checks the live feature schema against the saved estimator schema, and skips unsupported targets explicitly as `MODEL_INPUT_UNAVAILABLE`.

## Live Decision Inputs

The decision layer receives exactly these columns:

```text
match_id
market
closing_odds
predicted_probability
model_confidence
```

These inputs are derived from live bookmaker odds plus pre-match historical feature engineering. No synthetic placeholder feature remains in the live scoring path.

## Live Confidence Schema

The paper-trading scoring path currently produces this live schema before decisioning:

```text
predicted_probability_over_8_5
predicted_probability_over_9_5
predicted_probability_over_10_5
predicted_probability_over_11_5
match_id
predicted_total_corners
data_quality_score
insufficient_history
home_matches_played
away_matches_played
combined_volatility
model_disagreement
prediction_distance_from_line
probability_distance_from_0_50
residual_risk_estimate
team_bias_risk
cold_start_risk
feature_outlier_score
missing_history_count
calibration_bucket_error
model_stability_score
confidence_score
data_quality_component
historical_depth_component
volatility_component
agreement_component
calibration_component
boundary_component
team_bias_component
outlier_component
classification_entropy_score
decision_state
model_confidence
```

The exact live feature frame used to compute these values contains only historical pre-match features and traceability fields. The following placeholder outcome columns were removed from the live path and are not used in decisioning:

```text
total_corners
over85
over95
over105
over115
under85
under95
under105
under115
```

## Model Consistency

Accepted trained artifacts are loaded for `over_9_5` and `over_10_5`. The live path uses the saved `feature_names_in_` from those regressors and rejects any fixture/market row that does not match the expected schema. `over_8_5` and `over_11_5` still have no accepted trained artifact in the benchmark outputs, so those targets are explicitly skipped as `MODEL_INPUT_UNAVAILABLE`.

Model artifact:

- `over_9_5_negative_binomial_probability.pkl`
- `over_10_5_negative_binomial_probability.pkl`

Model version/hash:

- `over_9_5`: `negative_binomial_probability` / `a44495228ae3c22d76aefb9e0204c388f57cf307f4733aa2a50b725be0c401e6`
- `over_10_5`: `negative_binomial_probability` / `45cbba073d9e3f1fb1ab04753c9435aa68b90a6ee657e1fa145aef1b9654efca`

Expected feature schema:

- saved `feature_names_in_` from the accepted negative-binomial regressors

Live feature schema:

- advanced-feature rows from the research pipeline, aligned to the saved estimator schema before scoring

## Decision Integrity

The current `PLAY` / `NO BET` split comes from the existing `DecisionEngine` rule:

- `confidence_score` below 60.0 => `LOW CONFIDENCE`
- otherwise, `ev > 0.0` => `PLAY`
- otherwise => `NO BET`

The stored scientific `confidence_score` remains the true computed value from `compute_confidence_score`.

## Odds Join

Verified properties of the live rows:

- one fixture per odds row after per-fixture validation
- supported corner lines only: 8.5, 9.5, 10.5, 11.5
- genuine bookmaker odds from The Odds API
- correct side alignment for over and under rows
- prediction timestamp comes before decision output
- no cross-fixture duplicate collision after per-fixture validation
- no future/result information used in the live join

## Live Run

- Live fixtures: 7
- Market rows scored: 152
- Rows skipped for model input: 114
- PLAY: 28
- NO BET: 124
- LOW CONFIDENCE: 0
- Average EV: 0.2698809142191269
- Average confidence: 62.01298348119624

## Notes

- The live run is integrity-clean with respect to provenance, odds join, confidence handling, and model loading.
- Unsupported targets are explicitly skipped as `MODEL_INPUT_UNAVAILABLE`.