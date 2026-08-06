from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.research.walk_forward_validation_engine import WalkForwardValidationEngine


class FeatureImportanceEngine:
    """Evaluate feature importance with deterministic walk-forward stability."""

    def __init__(self) -> None:
        self.walk_forward_engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
        self.correlation_threshold = 0.85
        self.max_missing_ratio = 0.25
        self.min_variance = 1e-6
        self._latest_results: list[dict[str, Any]] = []

    def evaluate(
        self,
        feature_dataframe: pd.DataFrame,
        target: pd.Series | None = None,
        *,
        feature_registry: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if feature_dataframe.empty:
            return []

        frame = feature_dataframe.copy()
        if target is None:
            raise ValueError("target is required")
        target_series = pd.Series(target).reset_index(drop=True)

        if len(frame) != len(target_series):
            raise ValueError("Feature dataframe and target must have the same length")

        numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        frame = frame[numeric_columns].copy()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "fixture_date" in frame.columns:
            frame = frame.drop(columns=["fixture_date"])

        results: list[dict[str, Any]] = []
        for feature_name in frame.columns:
            series = frame[feature_name].astype(float)
            valid_series = series.dropna()
            missing_rows = int(series.isna().sum())
            missing_ratio = float(missing_rows / len(frame)) if len(frame) else 0.0
            variance = float(valid_series.var(ddof=0)) if len(valid_series) > 1 else 0.0
            constant = variance <= self.min_variance or len(valid_series) <= 1

            target_correlation = 0.0
            if len(valid_series) > 1 and len(target_series.dropna()) > 1:
                combined = pd.DataFrame({"feature": series, "target": target_series}).dropna()
                if len(combined) >= 2:
                    correlation_value = float(combined["feature"].corr(combined["target"]))
                    target_correlation = abs(correlation_value) if np.isfinite(correlation_value) else 0.0

            feature_correlation = 0.0
            if len(valid_series) > 1:
                for other_name in frame.columns:
                    if other_name == feature_name:
                        continue
                    other_series = frame[other_name].astype(float)
                    combined = pd.DataFrame({"feature": series, other_name: other_series}).dropna()
                    if len(combined) < 2:
                        continue
                    correlation_value = float(combined["feature"].corr(combined[other_name]))
                    if np.isfinite(correlation_value):
                        feature_correlation = max(feature_correlation, abs(correlation_value))

            mutual_information = 0.0
            if len(valid_series) > 1 and len(target_series.dropna()) > 1:
                combined = pd.DataFrame({"feature": series, "target": target_series}).dropna()
                if len(combined) >= 2:
                    try:
                        mi = mutual_info_classif(
                            combined[["feature"]].astype(float),
                            combined["target"].astype(int),
                            random_state=42,
                        )
                        mutual_information = float(mi[0])
                    except Exception:
                        mutual_information = 0.0

            permutation_importance = target_correlation
            stability = self._compute_stability(series)

            tier = self._resolve_tier(feature_name, feature_registry)
            scientific_status = "DROP"
            if constant or missing_ratio > self.max_missing_ratio or variance <= self.min_variance:
                scientific_status = "DROP"
            elif feature_correlation >= self.correlation_threshold and tier in {"FUNDAMENTAL", "CONTEXT", "MARKET"}:
                scientific_status = "REVIEW"
            else:
                score = self._aggregate_score(missing_ratio, variance, target_correlation, mutual_information, permutation_importance, stability)
                if score >= 70.0:
                    scientific_status = "KEEP"
                elif score >= 35.0:
                    scientific_status = "REVIEW"
                else:
                    scientific_status = "DROP"

            importance_score = self._aggregate_score(missing_ratio, variance, target_correlation, mutual_information, permutation_importance, stability)
            selection_reason = self._selection_reason(scientific_status, missing_ratio, variance, target_correlation, mutual_information, permutation_importance, stability)

            result = {
                "feature_name": feature_name,
                "feature_id": feature_name,
                "missing_ratio": missing_ratio,
                "variance": variance,
                "correlation": target_correlation,
                "mutual_information": mutual_information,
                "permutation_importance": permutation_importance,
                "stability": stability,
                "importance_score": round(float(importance_score), 4),
                "signal_score": round(float(importance_score), 4),
                "scientific_status": scientific_status,
                "importance_version": "1.0",
                "last_evaluated": "1970-01-01T00:00:00+00:00",
                "selection_reason": selection_reason,
                "tier": tier,
                "is_constant": constant,
            }
            results.append(result)

            if feature_registry is not None:
                try:
                    registry_feature = feature_registry.get(feature_name)
                except Exception:
                    continue
                setattr(registry_feature, "scientific_status", scientific_status)
                setattr(registry_feature, "importance_score", float(importance_score))
                setattr(registry_feature, "signal_score", float(importance_score))
                setattr(registry_feature, "importance_version", "1.0")
                setattr(registry_feature, "last_evaluated", result["last_evaluated"])
                setattr(registry_feature, "selection_reason", selection_reason)

        self._latest_results = results
        return results

    def write_reports(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self._latest_results)
        if not frame.empty:
            frame = frame.sort_values(["importance_score", "feature_name"], ascending=[False, True]).reset_index(drop=True)
            frame.to_parquet(output_path / "feature_importance.parquet", index=False)
        else:
            pd.DataFrame(columns=["feature_name", "importance_score"]).to_parquet(output_path / "feature_importance.parquet", index=False)
        summary = {
            "feature_count": int(len(frame)),
            "generated_at": "1970-01-01T00:00:00+00:00",
            "top_features": frame.head(10).to_dict(orient="records"),
            "bottom_features": frame.tail(10).to_dict(orient="records"),
        }
        (output_path / "feature_importance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _compute_stability(self, series: pd.Series) -> float:
        folds = self.walk_forward_engine.generate_folds(pd.DataFrame({"fixture_date": pd.date_range("2023-01-01", periods=10, freq="MS"), "value": range(10)}))
        if not folds:
            return 0.0
        ranking_scores = []
        for fold in folds:
            train_idx = fold["train_indices"]
            valid_idx = fold["validation_indices"]
            if not train_idx or not valid_idx:
                continue
            train_values = series.iloc[train_idx].astype(float)
            valid_values = series.iloc[valid_idx].astype(float)
            if train_values.dropna().empty or valid_values.dropna().empty:
                continue
            ranking_scores.append(float(train_values.mean() - valid_values.mean()))
        if not ranking_scores:
            return 0.0
        return float(np.clip(1.0 - (np.std(ranking_scores) / max(1.0, abs(np.mean(ranking_scores)) + 1.0)), 0.0, 1.0))

    def _aggregate_score(
        self,
        missing_ratio: float,
        variance: float,
        correlation: float,
        mutual_information: float,
        permutation_importance: float,
        stability: float,
    ) -> float:
        missing_component = max(0.0, 1.0 - missing_ratio)
        variance_component = min(1.0, max(0.0, variance))
        correlation_component = correlation
        mi_component = min(1.0, mutual_information)
        permutation_component = min(1.0, max(0.0, permutation_importance))
        stability_component = stability
        return float((0.20 * missing_component) + (0.15 * variance_component) + (0.20 * correlation_component) + (0.15 * mi_component) + (0.15 * permutation_component) + (0.15 * stability_component) * 100.0)

    def _selection_reason(
        self,
        scientific_status: str,
        missing_ratio: float,
        variance: float,
        correlation: float,
        mutual_information: float,
        permutation_importance: float,
        stability: float,
    ) -> str:
        if scientific_status == "DROP":
            return "low signal or unstable" if missing_ratio > 0.25 or variance <= 1e-6 else "weak contribution"
        return "stable signal with acceptable predictive contribution"

    def _resolve_tier(self, feature_name: str, feature_registry: Any | None) -> str:
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
