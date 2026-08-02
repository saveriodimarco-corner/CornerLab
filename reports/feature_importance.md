# Feature Importance Report

This section summarizes permutation-based and SHAP-style importance for every accepted model.

## actual_total_corners - poisson_regression
Top 30 features:
- coefficient_of_variation_last10: permutation=0.077, shap=0.620
- corners_against_last10: permutation=0.032, shap=0.017
- corners_for_last3: permutation=0.023, shap=0.017
- combined_volatility: permutation=0.021, shap=0.034
- total_corners_std_last10: permutation=0.012, shap=0.069
- tempo_difference: permutation=0.012, shap=0.007
- defence_trend: permutation=0.009, shap=0.019
- data_quality_score: permutation=0.008, shap=0.090
- corners_for_std_last10: permutation=0.005, shap=0.028
- corners_for_ewma: permutation=0.005, shap=0.010
- defence_difference: permutation=0.005, shap=0.007
- total_corners_last3: permutation=0.003, shap=0.004
- total_corners_std_last5: permutation=0.003, shap=0.004
- total_corners_last10: permutation=0.002, shap=0.005
- corners_for_last5: permutation=0.001, shap=0.003
- expected_away_corners_baseline: permutation=0.001, shap=0.001
- corners_against_last5: permutation=0.000, shap=0.002
- home_rest_days: permutation=0.000, shap=0.001
- corners_for_last10: permutation=0.000, shap=0.012
- corners_against_std_last5: permutation=0.000, shap=0.005
- away_matches_played: permutation=0.000, shap=0.000
- tempo_trend: permutation=0.000, shap=0.003
- corners_against_ewma: permutation=0.000, shap=0.000
- total_corners_last5: permutation=-0.000, shap=0.001
- attack_difference: permutation=-0.000, shap=0.000
- insufficient_history: permutation=-0.000, shap=0.085
- combined_trend: permutation=-0.001, shap=0.007
- away_corners_for_last5: permutation=-0.001, shap=0.003
- attack_trend: permutation=-0.001, shap=0.017
- expected_home_corners_baseline: permutation=-0.002, shap=0.003

Cumulative importance (permutation):
- coefficient_of_variation_last10: cumulative=0.499
- corners_against_last10: cumulative=0.705
- corners_for_last3: cumulative=0.853
- combined_volatility: cumulative=0.988
- total_corners_std_last10: cumulative=1.069
- tempo_difference: cumulative=1.146
- defence_trend: cumulative=1.202
- data_quality_score: cumulative=1.256
- corners_for_std_last10: cumulative=1.291
- corners_for_ewma: cumulative=1.323
- defence_difference: cumulative=1.355
- total_corners_last3: cumulative=1.377
- total_corners_std_last5: cumulative=1.398
- total_corners_last10: cumulative=1.413
- corners_for_last5: cumulative=1.419
- expected_away_corners_baseline: cumulative=1.423
- corners_against_last5: cumulative=1.426
- home_rest_days: cumulative=1.429
- corners_for_last10: cumulative=1.432
- corners_against_std_last5: cumulative=1.433
- away_matches_played: cumulative=1.435
- tempo_trend: cumulative=1.436
- corners_against_ewma: cumulative=1.437
- total_corners_last5: cumulative=1.437
- attack_difference: cumulative=1.436
- insufficient_history: cumulative=1.435
- combined_trend: cumulative=1.431
- away_corners_for_last5: cumulative=1.424
- attack_trend: cumulative=1.415
- expected_home_corners_baseline: cumulative=1.402

Features contributing less than 1%:
- corners_for_last5
- expected_away_corners_baseline
- corners_against_last5
- home_rest_days
- corners_for_last10
- corners_against_std_last5
- away_matches_played
- tempo_trend
- corners_against_ewma
- total_corners_last5
- attack_difference
- insufficient_history
- combined_trend
- away_corners_for_last5
- attack_trend
- expected_home_corners_baseline

Unstable rankings:

## over_8_5 - logistic_regression
Top 30 features:
- corners_for_ewma: permutation=0.003, shap=0.102
- total_corners_std_last5: permutation=0.003, shap=0.083
- corners_against_ewma: permutation=0.001, shap=0.041
- total_corners_last3: permutation=0.001, shap=0.053
- corners_against_last5: permutation=0.001, shap=0.053
- corners_for_last3: permutation=0.001, shap=0.093
- defence_difference: permutation=0.001, shap=0.024
- coefficient_of_variation_last10: permutation=0.001, shap=0.766
- corners_for_last5: permutation=0.000, shap=0.044
- combined_volatility: permutation=0.000, shap=0.061
- corners_for_std_last5: permutation=0.000, shap=0.019
- total_corners_last5: permutation=0.000, shap=0.009
- attack_difference: permutation=0.000, shap=0.009
- rest_days_difference: permutation=0.000, shap=0.001
- total_corners_last10: permutation=0.000, shap=0.006
- attack_trend: permutation=0.000, shap=0.039
- combined_trend: permutation=0.000, shap=0.050
- corners_against_std_last5: permutation=0.000, shap=0.002
- corners_for_last10: permutation=0.000, shap=0.005
- data_quality_score: permutation=0.000, shap=0.237
- corners_against_last10: permutation=0.000, shap=0.001
- expected_total_corners_baseline: permutation=-0.000, shap=0.018
- expected_away_corners_baseline: permutation=-0.000, shap=0.000
- away_corners_for_last5: permutation=-0.000, shap=0.055
- home_rest_days: permutation=-0.000, shap=0.001
- away_corners_against_last5: permutation=-0.000, shap=0.078
- tempo_trend: permutation=-0.000, shap=0.015
- insufficient_history: permutation=-0.000, shap=0.113
- corners_against_last3: permutation=-0.000, shap=0.038
- corners_for_std_last10: permutation=-0.000, shap=0.041

Cumulative importance (permutation):
- corners_for_ewma: cumulative=0.493
- total_corners_std_last5: cumulative=0.937
- corners_against_ewma: cumulative=1.147
- total_corners_last3: cumulative=1.313
- corners_against_last5: cumulative=1.471
- corners_for_last3: cumulative=1.608
- defence_difference: cumulative=1.710
- coefficient_of_variation_last10: cumulative=1.802
- corners_for_last5: cumulative=1.863
- combined_volatility: cumulative=1.908
- corners_for_std_last5: cumulative=1.951
- total_corners_last5: cumulative=1.993
- attack_difference: cumulative=2.019
- rest_days_difference: cumulative=2.043
- total_corners_last10: cumulative=2.054
- attack_trend: cumulative=2.061
- combined_trend: cumulative=2.066
- corners_against_std_last5: cumulative=2.068
- corners_for_last10: cumulative=2.070
- data_quality_score: cumulative=2.072
- corners_against_last10: cumulative=2.072
- expected_total_corners_baseline: cumulative=2.072
- expected_away_corners_baseline: cumulative=2.071
- away_corners_for_last5: cumulative=2.069
- home_rest_days: cumulative=2.062
- away_corners_against_last5: cumulative=2.046
- tempo_trend: cumulative=2.023
- insufficient_history: cumulative=2.000
- corners_against_last3: cumulative=1.971
- corners_for_std_last10: cumulative=1.942

Features contributing less than 1%:
- attack_trend
- combined_trend
- corners_against_std_last5
- corners_for_last10
- data_quality_score
- corners_against_last10
- expected_total_corners_baseline
- expected_away_corners_baseline
- away_corners_for_last5
- home_rest_days
- away_corners_against_last5
- tempo_trend
- insufficient_history
- corners_against_last3
- corners_for_std_last10

Unstable rankings:

## over_9_5 - negative_binomial_probability
Top 30 features:
- total_corners_std_last10: permutation=0.003, shap=1.008
- coefficient_of_variation_last10: permutation=0.001, shap=1.005
- corners_for_std_last10: permutation=0.001, shap=1.021
- total_corners_last3: permutation=0.001, shap=1.062
- combined_trend: permutation=0.000, shap=1.049
- away_rest_days: permutation=0.000, shap=1.003
- combined_volatility: permutation=0.000, shap=1.076
- insufficient_history: permutation=0.000, shap=1.001
- home_matches_played: permutation=0.000, shap=1.036
- corners_against_std_last5: permutation=0.000, shap=1.003
- away_total_corners_last5: permutation=0.000, shap=1.035
- tempo_trend: permutation=0.000, shap=1.027
- data_quality_score: permutation=0.000, shap=1.002
- attack_trend: permutation=0.000, shap=1.075
- corners_against_last10: permutation=0.000, shap=1.043
- corners_against_ewma: permutation=0.000, shap=1.036
- rest_days_difference: permutation=0.000, shap=1.015
- corners_for_last10: permutation=0.000, shap=1.018
- corners_for_last5: permutation=0.000, shap=1.031
- corners_for_ewma: permutation=0.000, shap=1.012
- away_corners_for_last5: permutation=0.000, shap=1.068
- total_corners_last10: permutation=-0.000, shap=1.031
- corners_against_last3: permutation=-0.000, shap=1.038
- expected_away_corners_baseline: permutation=-0.000, shap=1.060
- tempo_difference: permutation=-0.000, shap=1.002
- total_corners_last5: permutation=-0.000, shap=1.042
- total_corners_ewma: permutation=-0.000, shap=1.045
- corners_for_std_last5: permutation=-0.000, shap=1.039
- defence_difference: permutation=-0.000, shap=1.030
- away_corners_against_last5: permutation=-0.000, shap=1.029

Cumulative importance (permutation):
- total_corners_std_last10: cumulative=0.601
- coefficient_of_variation_last10: cumulative=0.834
- corners_for_std_last10: cumulative=1.014
- total_corners_last3: cumulative=1.187
- combined_trend: cumulative=1.296
- away_rest_days: cumulative=1.370
- combined_volatility: cumulative=1.443
- insufficient_history: cumulative=1.487
- home_matches_played: cumulative=1.524
- corners_against_std_last5: cumulative=1.551
- away_total_corners_last5: cumulative=1.576
- tempo_trend: cumulative=1.595
- data_quality_score: cumulative=1.611
- attack_trend: cumulative=1.628
- corners_against_last10: cumulative=1.643
- corners_against_ewma: cumulative=1.657
- rest_days_difference: cumulative=1.670
- corners_for_last10: cumulative=1.680
- corners_for_last5: cumulative=1.690
- corners_for_ewma: cumulative=1.698
- away_corners_for_last5: cumulative=1.698
- total_corners_last10: cumulative=1.695
- corners_against_last3: cumulative=1.688
- expected_away_corners_baseline: cumulative=1.678
- tempo_difference: cumulative=1.667
- total_corners_last5: cumulative=1.655
- total_corners_ewma: cumulative=1.636
- corners_for_std_last5: cumulative=1.616
- defence_difference: cumulative=1.596
- away_corners_against_last5: cumulative=1.573

Features contributing less than 1%:
- corners_for_last5
- corners_for_ewma
- away_corners_for_last5
- total_corners_last10
- corners_against_last3
- expected_away_corners_baseline
- tempo_difference
- total_corners_last5
- total_corners_ewma
- corners_for_std_last5
- defence_difference
- away_corners_against_last5

Unstable rankings:

## over_10_5 - negative_binomial_probability
Top 30 features:
- corners_for_std_last5: permutation=0.002, shap=1.025
- corners_for_std_last10: permutation=0.002, shap=1.016
- combined_volatility: permutation=0.001, shap=1.059
- total_corners_last3: permutation=0.001, shap=1.041
- corners_for_last3: permutation=0.001, shap=1.021
- total_corners_std_last10: permutation=0.000, shap=1.032
- corners_for_ewma: permutation=0.000, shap=1.009
- total_corners_ewma: permutation=0.000, shap=1.032
- attack_trend: permutation=0.000, shap=1.057
- insufficient_history: permutation=0.000, shap=1.023
- combined_trend: permutation=0.000, shap=1.050
- season_match_number: permutation=0.000, shap=1.009
- defence_trend: permutation=0.000, shap=1.052
- corners_for_last10: permutation=0.000, shap=1.068
- corners_against_std_last5: permutation=0.000, shap=1.025
- corners_against_last10: permutation=0.000, shap=1.068
- corners_against_ewma: permutation=0.000, shap=1.042
- corners_against_last3: permutation=-0.000, shap=1.025
- attack_difference: permutation=-0.000, shap=1.048
- tempo_trend: permutation=-0.000, shap=1.001
- defence_difference: permutation=-0.000, shap=1.003
- expected_away_corners_baseline: permutation=-0.000, shap=1.053
- away_corners_for_last5: permutation=-0.000, shap=1.045
- tempo_difference: permutation=-0.000, shap=1.037
- away_rest_days: permutation=-0.000, shap=1.030
- corners_for_last5: permutation=-0.000, shap=1.020
- coefficient_of_variation_last10: permutation=-0.000, shap=1.031
- data_quality_score: permutation=-0.000, shap=1.003
- total_corners_last5: permutation=-0.000, shap=1.008
- total_corners_last10: permutation=-0.000, shap=1.010

Cumulative importance (permutation):
- corners_for_std_last5: cumulative=0.316
- corners_for_std_last10: cumulative=0.623
- combined_volatility: cumulative=0.878
- total_corners_last3: cumulative=1.003
- corners_for_last3: cumulative=1.123
- total_corners_std_last10: cumulative=1.211
- corners_for_ewma: cumulative=1.278
- total_corners_ewma: cumulative=1.338
- attack_trend: cumulative=1.373
- insufficient_history: cumulative=1.400
- combined_trend: cumulative=1.425
- season_match_number: cumulative=1.444
- defence_trend: cumulative=1.457
- corners_for_last10: cumulative=1.468
- corners_against_std_last5: cumulative=1.480
- corners_against_last10: cumulative=1.485
- corners_against_ewma: cumulative=1.490
- corners_against_last3: cumulative=1.489
- attack_difference: cumulative=1.489
- tempo_trend: cumulative=1.488
- defence_difference: cumulative=1.487
- expected_away_corners_baseline: cumulative=1.485
- away_corners_for_last5: cumulative=1.482
- tempo_difference: cumulative=1.474
- away_rest_days: cumulative=1.462
- corners_for_last5: cumulative=1.451
- coefficient_of_variation_last10: cumulative=1.438
- data_quality_score: cumulative=1.424
- total_corners_last5: cumulative=1.400
- total_corners_last10: cumulative=1.376

Features contributing less than 1%:
- corners_against_last10
- corners_against_ewma
- corners_against_last3
- attack_difference
- tempo_trend
- defence_difference
- expected_away_corners_baseline
- away_corners_for_last5
- tempo_difference
- away_rest_days
- corners_for_last5
- coefficient_of_variation_last10
- data_quality_score
- total_corners_last5
- total_corners_last10

Unstable rankings:

## over_11_5 - poisson_probability
Top 30 features:
- coefficient_of_variation_last10: permutation=0.001, shap=1.130
- corners_for_std_last10: permutation=0.001, shap=0.233
- corners_for_std_last5: permutation=0.001, shap=0.175
- corners_against_last10: permutation=0.000, shap=0.056
- data_quality_score: permutation=0.000, shap=0.831
- corners_for_last10: permutation=0.000, shap=0.057
- tempo_difference: permutation=0.000, shap=0.039
- home_rest_days: permutation=0.000, shap=0.005
- insufficient_history: permutation=0.000, shap=0.627
- corners_against_last5: permutation=0.000, shap=0.014
- total_corners_last5: permutation=0.000, shap=0.013
- combined_volatility: permutation=0.000, shap=0.155
- total_corners_std_last10: permutation=0.000, shap=0.283
- corners_for_last3: permutation=0.000, shap=0.012
- tempo_trend: permutation=0.000, shap=0.013
- attack_difference: permutation=0.000, shap=0.019
- total_corners_std_last5: permutation=0.000, shap=0.062
- corners_for_last5: permutation=0.000, shap=0.002
- corners_for_ewma: permutation=0.000, shap=0.001
- total_corners_last10: permutation=-0.000, shap=0.001
- season_match_number: permutation=-0.000, shap=0.000
- expected_away_corners_baseline: permutation=-0.000, shap=0.003
- expected_home_corners_baseline: permutation=-0.000, shap=0.004
- expected_total_corners_baseline: permutation=-0.000, shap=0.007
- away_corners_against_last5: permutation=-0.000, shap=0.005
- defence_trend: permutation=-0.000, shap=0.041
- away_corners_for_last5: permutation=-0.000, shap=0.021
- total_corners_last3: permutation=-0.000, shap=0.010
- combined_trend: permutation=-0.000, shap=0.076
- corners_against_last3: permutation=-0.000, shap=0.022

Cumulative importance (permutation):
- coefficient_of_variation_last10: cumulative=0.377
- corners_for_std_last10: cumulative=0.742
- corners_for_std_last5: cumulative=1.102
- corners_against_last10: cumulative=1.358
- data_quality_score: cumulative=1.607
- corners_for_last10: cumulative=1.727
- tempo_difference: cumulative=1.826
- home_rest_days: cumulative=1.908
- insufficient_history: cumulative=1.989
- corners_against_last5: cumulative=2.062
- total_corners_last5: cumulative=2.120
- combined_volatility: cumulative=2.178
- total_corners_std_last10: cumulative=2.230
- corners_for_last3: cumulative=2.256
- tempo_trend: cumulative=2.277
- attack_difference: cumulative=2.294
- total_corners_std_last5: cumulative=2.297
- corners_for_last5: cumulative=2.298
- corners_for_ewma: cumulative=2.298
- total_corners_last10: cumulative=2.297
- season_match_number: cumulative=2.290
- expected_away_corners_baseline: cumulative=2.281
- expected_home_corners_baseline: cumulative=2.271
- expected_total_corners_baseline: cumulative=2.258
- away_corners_against_last5: cumulative=2.245
- defence_trend: cumulative=2.226
- away_corners_for_last5: cumulative=2.206
- total_corners_last3: cumulative=2.166
- combined_trend: cumulative=2.116
- corners_against_last3: cumulative=2.032

Features contributing less than 1%:
- total_corners_std_last5
- corners_for_last5
- corners_for_ewma
- total_corners_last10
- season_match_number
- expected_away_corners_baseline
- expected_home_corners_baseline
- expected_total_corners_baseline
- away_corners_against_last5
- defence_trend
- away_corners_for_last5
- total_corners_last3
- combined_trend
- corners_against_last3

Unstable rankings:

