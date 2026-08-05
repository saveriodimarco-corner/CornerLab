from __future__ import annotations

from typing import List

import pandas as pd


class DatasetMerger:
    """Merges multiple historical datasets into one unified schema."""

    def __init__(self) -> None:
        self._required_columns = [
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

    def merge(self, *datasets: pd.DataFrame) -> pd.DataFrame:
        merged_frames: List[pd.DataFrame] = []
        for dataset in datasets:
            frame = dataset.copy()
            for column in self._required_columns:
                if column not in frame.columns:
                    frame[column] = None
            merged_frames.append(frame[self._required_columns])
        if not merged_frames:
            return pd.DataFrame(columns=self._required_columns)
        merged = pd.concat(merged_frames, ignore_index=True)
        return merged.reindex(columns=self._required_columns)

    def required_columns(self) -> List[str]:
        return list(self._required_columns)
