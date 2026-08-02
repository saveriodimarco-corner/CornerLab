# Scientific Review

## Strengths
- The benchmark uses a chronological validation split and accepted models beat their naive baselines.
- Permutation and SHAP-style importance provide stable global rankings for the accepted models.

## Weaknesses
- The explainability workflow uses a lightweight fallback for SHAP-style contributions when a dedicated SHAP dependency is unavailable.
- Residuals and bias remain sensitive to high-volatility and cold-start contexts.

## Failure modes
- The regression model degrades on high-volatility and high-corner matches.
- Confidence can be miscalibrated in the tails.

## Sources of uncertainty
- Feature history is limited for cold-start teams and early-season matches.
- Model outputs are sensitive to the selected feature set and the validation window.

## Most informative features

- coefficient_of_variation_last10
- corners_against_last10
- corners_for_last3
- combined_volatility
- total_corners_std_last10
- tempo_difference
- defence_trend
- data_quality_score
- corners_for_std_last10
- corners_for_ewma

## Least useful features
- Features with low permutation contribution and low variance in the validation set.

## Recommendations before production
- Validate the accepted model on a broader season window before deployment.
- Combine the benchmark model with a conservative confidence threshold and manual review for high-volatility matches.

## Production readiness
- The current benchmark is research-ready, but not yet fully production-ready without wider validation and monitoring.
