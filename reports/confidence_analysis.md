# Confidence Analysis

This report compares confidence and error for accepted models.

- actual_total_corners (poisson_regression): confidence_mean=0.373, error_mean=2.574, confidence_error_correlation=-0.840, ece=0.287
- over_8_5 (logistic_regression): confidence_mean=0.574, error_mean=0.471, confidence_error_correlation=-0.015, ece=0.493
- over_9_5 (negative_binomial_probability): confidence_mean=0.647, error_mean=0.597, confidence_error_correlation=-0.066, ece=0.527
- over_10_5 (negative_binomial_probability): confidence_mean=0.623, error_mean=0.708, confidence_error_correlation=0.000, ece=0.551
- over_11_5 (poisson_probability): confidence_mean=0.557, error_mean=0.805, confidence_error_correlation=0.038, ece=0.535
