# Feature Selection Report

## Target summaries
- actual_total_corners: 38 selected features, signal score 29.045
- over_8_5: 38 selected features, signal score 29.309
- over_9_5: 38 selected features, signal score 29.965
- over_10_5: 38 selected features, signal score 28.960
- over_11_5: 38 selected features, signal score 27.030

## Top 20 features by Signal Score

- rest_days_difference: signal score 40.329, train corr -0.046, validation corr -0.065
- home_rest_days: signal score 38.556, train corr -0.029, validation corr -0.082
- corners_for_last3: signal score 37.647, train corr -0.029, validation corr -0.060
- attack_trend: signal score 36.146, train corr -0.049, validation corr -0.093
- combined_volatility: signal score 35.800, train corr -0.033, validation corr -0.046
- attack_difference: signal score 35.157, train corr 0.000, validation corr 0.021
- total_corners_std_last5: signal score 34.996, train corr 0.018, validation corr 0.132
- corners_for_last10: signal score 34.630, train corr 0.017, validation corr 0.007
- away_rest_days: signal score 34.095, train corr -0.001, validation corr -0.070
- away_corners_for_last5: signal score 34.021, train corr -0.012, validation corr -0.074
- total_corners_std_last10: signal score 33.698, train corr 0.010, validation corr 0.039
- combined_trend: signal score 33.352, train corr -0.022, validation corr -0.076
- corners_for_last5: signal score 32.591, train corr -0.011, validation corr -0.050
- defence_trend: signal score 32.562, train corr 0.069, validation corr 0.032
- corners_for_std_last5: signal score 32.122, train corr -0.023, validation corr -0.012
- defence_difference: signal score 31.924, train corr -0.060, validation corr -0.014
- corners_against_std_last5: signal score 31.667, train corr 0.045, validation corr 0.021
- away_corners_against_last5: signal score 31.037, train corr 0.125, validation corr 0.025
- corners_against_last5: signal score 29.593, train corr 0.046, validation corr 0.004
- away_matches_played: signal score 28.712, train corr 0.012, validation corr -0.027

## Recommended feature set for regression
rest_days_difference, home_rest_days, corners_for_last3, attack_trend, combined_volatility, attack_difference, total_corners_std_last5, corners_for_last10, away_rest_days, away_corners_for_last5, total_corners_std_last10, combined_trend, corners_for_last5, defence_trend, corners_for_std_last5, defence_difference, corners_against_std_last5, away_corners_against_last5, corners_against_last5, away_matches_played, total_corners_last5, corners_against_last10, total_corners_ewma, expected_total_corners_baseline, total_corners_last3, corners_against_ewma, away_total_corners_last5, total_corners_last10, expected_away_corners_baseline, corners_against_last3, tempo_trend, data_quality_score, corners_for_ewma, corners_for_std_last10, insufficient_history, tempo_difference, expected_home_corners_baseline, coefficient_of_variation_last10

## Explicitly rejected features
- home_corners_for_last5: highly correlated with a selected feature
- home_corners_for_last5: low signal or redundant with another selected feature
- home_corners_against_last5: highly correlated with a selected feature
- home_corners_against_last5: low signal or redundant with another selected feature
- season_match_number: highly correlated with a selected feature
- season_match_number: low signal or redundant with another selected feature
- home_matches_played: highly correlated with a selected feature
- home_matches_played: low signal or redundant with another selected feature
- home_total_corners_last5: highly correlated with a selected feature
- home_total_corners_last5: low signal or redundant with another selected feature
