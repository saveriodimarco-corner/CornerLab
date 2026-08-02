from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class TeamRatingEngine:
    """Calculate team corner-based rating metrics from match data."""

    REQUIRED_COLUMNS = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_corners",
        "away_corners",
    }

    def __init__(self) -> None:
        """Initialize the engine."""
        self._logger = None

    def load_matches(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load football match data from a CSV or Parquet file."""
        path = Path(source)
        if path.suffix.lower() == ".csv":
            data = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            data = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

        self._validate_columns(data)
        return data

    def calculate_ratings(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-team corner ratings from a match dataframe."""
        self._validate_columns(matches)

        working = matches.copy()
        working["home_corners"] = pd.to_numeric(working["home_corners"], errors="coerce")
        working["away_corners"] = pd.to_numeric(working["away_corners"], errors="coerce")

        if working[["home_corners", "away_corners"]].isna().any().any():
            raise ValueError("Corner values must be numeric.")

        home_attack = working.groupby("home_team")["home_corners"].mean().rename("home_attack_rating")
        away_attack = working.groupby("away_team")["away_corners"].mean().rename("away_attack_rating")

        home_defence = working.groupby("home_team")["away_corners"].mean().rename("home_defence_rating")
        away_defence = working.groupby("away_team")["home_corners"].mean().rename("away_defence_rating")

        totals = pd.concat([home_attack, away_attack, home_defence, away_defence], axis=1).fillna(0.0)
        totals.index.names = ["team"]
        totals = totals.reset_index()

        totals["overall_attack"] = (totals["home_attack_rating"] + totals["away_attack_rating"]) / 2.0
        totals["overall_defence"] = (totals["home_defence_rating"] + totals["away_defence_rating"]) / 2.0
        totals["tempo_index"] = totals["overall_attack"] + totals["overall_defence"]
        totals["corner_difference"] = totals["home_attack_rating"] - totals["home_defence_rating"]
        totals["corner_balance"] = totals["away_attack_rating"] - totals["away_defence_rating"]

        home_corners = working.groupby("home_team")["home_corners"].std(ddof=0).rename("home_std")
        away_corners = working.groupby("away_team")["away_corners"].std(ddof=0).rename("away_std")
        std_frame = pd.concat([home_corners, away_corners], axis=1).fillna(0.0)
        std_frame.index.names = ["team"]
        std_frame = std_frame.reset_index()
        totals = totals.merge(std_frame, on="team", how="left")
        totals["standard_deviation"] = (totals["home_std"] + totals["away_std"]) / 2.0
        totals["consistency_index"] = 1.0 / (1.0 + totals["standard_deviation"])

        result = totals[[
            "team",
            "home_attack_rating",
            "away_attack_rating",
            "home_defence_rating",
            "away_defence_rating",
            "overall_attack",
            "overall_defence",
            "tempo_index",
            "corner_difference",
            "corner_balance",
            "standard_deviation",
            "consistency_index",
        ]].copy()
        return result.sort_values("team").reset_index(drop=True)

    def build(self, source: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
        """Load match data, calculate ratings, and write a parquet file."""
        matches = self.load_matches(source)
        ratings = self.calculate_ratings(matches)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        ratings.to_parquet(output, index=False)
        return ratings

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Ensure the incoming dataframe contains all required columns."""
        missing = self.REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
