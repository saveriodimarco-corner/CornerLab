from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class DatasetStatistics:
    """Compute high-level statistics for historical datasets."""

    def compute(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        if dataset.empty:
            return {
                "total_matches": 0,
                "matches_per_season": {},
                "matches_per_provider": {},
                "coverage_percentage": 0.0,
                "missing_percentage": 0.0,
                "duplicate_percentage": 0.0,
                "bookmakers_available": [],
                "corner_lines_available": [],
                "historical_depth": 0,
            }

        dataset = dataset.copy()
        matches_per_season = dataset["season"].value_counts().to_dict()
        matches_per_provider = dataset["provider"].value_counts().to_dict()
        coverage_percentage = round((len(dataset) / max(len(dataset), 1)) * 100.0, 2)
        missing_percentage = round((dataset.isna().sum().sum() / max(dataset.size, 1)) * 100.0, 2)
        duplicate_percentage = round((dataset.duplicated(subset=["match_id"]).sum() / max(len(dataset), 1)) * 100.0, 2)
        bookmakers_available = sorted({value for value in dataset["bookmaker"].dropna().tolist() if value})
        corner_lines_available = sorted({value for value in dataset["line"].dropna().astype(str).tolist() if value})
        historical_depth = len(matches_per_season)

        return {
            "total_matches": int(len(dataset)),
            "matches_per_season": matches_per_season,
            "matches_per_provider": matches_per_provider,
            "coverage_percentage": coverage_percentage,
            "missing_percentage": missing_percentage,
            "duplicate_percentage": duplicate_percentage,
            "bookmakers_available": bookmakers_available,
            "corner_lines_available": corner_lines_available,
            "historical_depth": historical_depth,
        }
