from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureSelectionContractError(ValueError):
    """Raised when feature selection is invoked without a precomputed signal_score contract."""


class FeatureSelectionEngine:
    def __init__(self) -> None:
        self.correlation_threshold = 0.85
        self.max_missing_ratio = 0.25

    def evaluate(self, feature_dataframe: pd.DataFrame, feature_registry: Optional[Any] = None) -> List[Dict[str, Any]]:
        if feature_dataframe.empty:
            return []

        dataframe = feature_dataframe.copy()
        if "signal_score" not in dataframe.columns:
            raise FeatureSelectionContractError(
                "FeatureSelectionEngine requires signal_score from FeatureImportanceEngine; missing signal_score is a contract violation"
            )

        numeric_columns = [
            column
            for column in dataframe.columns
            if column != "signal_score" and pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        if not numeric_columns:
            return []

        dataframe = dataframe[numeric_columns].copy()
        for column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

        results: List[Dict[str, Any]] = []
        for feature_name in dataframe.columns:
            series = dataframe[feature_name].astype(float)
            valid_series = series.dropna()
            missing_rows = int(series.isna().sum())
            valid_rows = int(len(valid_series))
            missing_ratio = float(missing_rows / len(dataframe)) if len(dataframe) else 0.0
            unique_values = int(valid_series.nunique(dropna=True))
            variance = float(valid_series.var(ddof=0)) if len(valid_series) > 1 else 0.0
            standard_deviation = float(valid_series.std(ddof=0)) if len(valid_series) > 1 else 0.0
            coefficient_of_variation = (
                float(standard_deviation / abs(valid_series.mean()))
                if valid_series.mean() != 0 and len(valid_series) > 0 and standard_deviation > 0
                else 0.0
            )
            constant_feature = "YES" if unique_values <= 1 or variance == 0.0 else "NO"

            max_absolute_correlation = 0.0
            correlated_feature = ""
            if len(valid_series) > 1:
                for other_name in dataframe.columns:
                    if other_name == feature_name:
                        continue
                    other_series = dataframe[other_name].astype(float).dropna()
                    if other_series.empty:
                        continue
                    combined = pd.concat([series, dataframe[other_name].astype(float)], axis=1).dropna()
                    if len(combined) < 2:
                        continue
                    correlation_value = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
                    abs_value = abs(correlation_value)
                    if abs_value > max_absolute_correlation:
                        max_absolute_correlation = abs_value
                        correlated_feature = other_name

            if missing_ratio > self.max_missing_ratio or variance == 0.0 or constant_feature == "YES":
                selection_status = "DROP"
            elif max_absolute_correlation >= self.correlation_threshold:
                tier = self._resolve_tier(feature_name, feature_registry)
                if tier in {"FUNDAMENTAL", "CONTEXT", "MARKET"}:
                    selection_status = "REVIEW"
                else:
                    selection_status = "DROP"
            else:
                selection_status = "KEEP"

            results.append(
                {
                    "feature_id": feature_name,
                    "feature_name": feature_name,
                    "valid_rows": valid_rows,
                    "missing_rows": missing_rows,
                    "missing_ratio": missing_ratio,
                    "unique_values": unique_values,
                    "variance": variance,
                    "standard_deviation": standard_deviation,
                    "coefficient_of_variation": coefficient_of_variation,
                    "constant_feature": constant_feature,
                    "max_absolute_correlation": max_absolute_correlation,
                    "correlated_feature": correlated_feature,
                    "selection_status": selection_status,
                }
            )

        return results

    def _resolve_tier(self, feature_name: str, feature_registry: Optional[Any]) -> str:
        if feature_registry is None:
            return "FUNDAMENTAL"

        for feature in getattr(feature_registry, "as_dicts", lambda: [])():
            if feature.get("name") == feature_name:
                return str(feature.get("tier", "FUNDAMENTAL")).upper()

        try:
            registry_feature = feature_registry.get(feature_name)
        except Exception:
            return "FUNDAMENTAL"

        return str(getattr(registry_feature, "tier", "FUNDAMENTAL")).upper()
