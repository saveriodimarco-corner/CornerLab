from __future__ import annotations

from math import sqrt
from statistics import mean, pvariance, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple


FEATURE_NAMES = (
    "corner_creation_rate",
    "corner_concession_rate",
    "recent_corner_form",
    "corner_diff_pressure",
)


class FeatureEvaluator:
    def evaluate(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "row_count": 0,
                "missing_ratio": {},
                "variance": {},
                "minimum": {},
                "maximum": {},
                "mean": {},
                "standard_deviation": {},
                "constant_features": [],
                "correlation_matrix": {},
            }

        grouped_values: Dict[str, List[Tuple[int, Optional[float]]]] = {name: [] for name in FEATURE_NAMES}
        for row in rows:
            feature_name = row.get("feature_name")
            if feature_name not in grouped_values:
                continue
            fixture_id = row.get("fixture_id", len(grouped_values[feature_name]))
            grouped_values[feature_name].append((int(fixture_id), row.get("feature_value")))

        missing_ratio: Dict[str, float] = {}
        variance: Dict[str, float] = {}
        minimum: Dict[str, float] = {}
        maximum: Dict[str, float] = {}
        mean_values: Dict[str, float] = {}
        std_values: Dict[str, float] = {}
        constant_features: List[str] = []

        for feature_name in FEATURE_NAMES:
            values = [value for _, value in grouped_values[feature_name] if value is not None]
            missing_count = len(grouped_values[feature_name]) - len(values)
            missing_ratio[feature_name] = missing_count / len(grouped_values[feature_name]) if grouped_values[feature_name] else 0.0

            if not values:
                variance[feature_name] = float("nan")
                minimum[feature_name] = float("nan")
                maximum[feature_name] = float("nan")
                mean_values[feature_name] = float("nan")
                std_values[feature_name] = float("nan")
                continue

            numeric_values = [float(value) for value in values]
            variance_value = pvariance(numeric_values) if len(numeric_values) > 1 else 0.0
            std_value = pstdev(numeric_values) if len(numeric_values) > 1 else 0.0
            variance[feature_name] = float(variance_value)
            minimum[feature_name] = float(min(numeric_values))
            maximum[feature_name] = float(max(numeric_values))
            mean_values[feature_name] = float(mean(numeric_values))
            std_values[feature_name] = float(std_value)
            if variance_value == 0.0:
                constant_features.append(feature_name)

        correlation_matrix: Dict[str, Dict[str, float]] = {}
        fixture_lookup: Dict[str, Dict[int, float]] = {}
        for feature_name in FEATURE_NAMES:
            fixture_lookup[feature_name] = {
                fixture_id: float(value)
                for fixture_id, value in grouped_values[feature_name]
                if value is not None
            }

        for feature_name in FEATURE_NAMES:
            correlation_row: Dict[str, float] = {}
            for other_name in FEATURE_NAMES:
                if feature_name == other_name:
                    correlation_row[other_name] = 1.0
                    continue

                fixture_ids = sorted(set(fixture_lookup[feature_name]) & set(fixture_lookup[other_name]))
                if len(fixture_ids) < 2:
                    correlation_row[other_name] = 0.0
                    continue

                x_values = [fixture_lookup[feature_name][fixture_id] for fixture_id in fixture_ids]
                y_values = [fixture_lookup[other_name][fixture_id] for fixture_id in fixture_ids]
                correlation_row[other_name] = self._pearson_correlation(x_values, y_values)
            correlation_matrix[feature_name] = correlation_row

        return {
            "row_count": len(rows),
            "missing_ratio": missing_ratio,
            "variance": variance,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean_values,
            "standard_deviation": std_values,
            "constant_features": constant_features,
            "correlation_matrix": correlation_matrix,
        }

    @staticmethod
    def _pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_denom = sqrt(sum((x - x_mean) ** 2 for x in x_values))
        y_denom = sqrt(sum((y - y_mean) ** 2 for y in y_values))
        if x_denom == 0.0 or y_denom == 0.0:
            return 0.0
        return numerator / (x_denom * y_denom)
