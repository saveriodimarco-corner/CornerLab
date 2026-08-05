from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class DatasetValidator:
    """Validates historical datasets and emits quality metrics."""

    def __init__(self) -> None:
        self.required_columns = [
            "match_id",
            "competition",
            "season",
            "kickoff",
            "home_team",
            "away_team",
            "bookmaker",
            "market",
            "line",
            "opening_odds",
            "closing_odds",
            "settlement",
            "result",
            "model_probability",
            "confidence_score",
            "expected_value",
            "kelly_fraction",
            "provider",
            "data_quality_score",
        ]

    def validate(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        if dataset.empty:
            return {
                "dataset": dataset,
                "issues": {
                    "duplicate_matches": 0,
                    "missing_odds": 0,
                    "invalid_probabilities": 0,
                    "team_name_mismatches": 0,
                    "date_mismatches": 0,
                    "line_inconsistencies": 0,
                    "provider_conflicts": 0,
                },
                "quality_metrics": {
                    "total_matches": 0,
                    "quality_score": 1.0,
                    "missing_percentage": 0.0,
                    "duplicate_percentage": 0.0,
                },
            }

        missing_columns = [column for column in self.required_columns if column not in dataset.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        dataset = dataset.copy()
        dataset["opening_odds"] = pd.to_numeric(dataset["opening_odds"], errors="coerce")
        dataset["closing_odds"] = pd.to_numeric(dataset["closing_odds"], errors="coerce")
        dataset["model_probability"] = pd.to_numeric(dataset["model_probability"], errors="coerce")
        dataset["confidence_score"] = pd.to_numeric(dataset["confidence_score"], errors="coerce")

        duplicate_matches = int(dataset.duplicated(subset=["match_id"]).sum())
        missing_odds = int((dataset["opening_odds"].isna() | dataset["closing_odds"].isna()).sum())
        invalid_probabilities = int(((dataset["model_probability"] < 0) | (dataset["model_probability"] > 1)).sum())
        team_name_mismatches = 0
        date_mismatches = 0
        line_inconsistencies = 0
        provider_conflicts = 0

        quality_score = max(0.0, 1.0 - (duplicate_matches + missing_odds + invalid_probabilities) / max(len(dataset), 1))
        missing_percentage = round((missing_odds / max(len(dataset), 1)) * 100.0, 2)
        duplicate_percentage = round((duplicate_matches / max(len(dataset), 1)) * 100.0, 2)

        return {
            "dataset": dataset,
            "issues": {
                "duplicate_matches": duplicate_matches,
                "missing_odds": missing_odds,
                "invalid_probabilities": invalid_probabilities,
                "team_name_mismatches": team_name_mismatches,
                "date_mismatches": date_mismatches,
                "line_inconsistencies": line_inconsistencies,
                "provider_conflicts": provider_conflicts,
            },
            "quality_metrics": {
                "total_matches": int(len(dataset)),
                "quality_score": round(quality_score, 4),
                "missing_percentage": missing_percentage,
                "duplicate_percentage": duplicate_percentage,
            },
        }
