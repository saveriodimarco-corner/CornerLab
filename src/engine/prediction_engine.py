from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import CONFIG


class PredictionEngine:
    """Deterministic corner prediction engine based on weighted ratings and Poisson probabilities."""

    def __init__(self) -> None:
        """Initialize the prediction engine."""
        self._thresholds = list(CONFIG.DEFAULT_THRESHOLDS)

    def load_inputs(self, ratings_path: Union[str, Path], features_path: Union[str, Path]) -> pd.DataFrame:
        """Load rating and feature parquet files and merge them for prediction."""
        ratings = pd.read_parquet(ratings_path)
        features = pd.read_parquet(features_path)
        if ratings.empty or features.empty:
            raise ValueError("Ratings and features inputs must not be empty")

        if "team" in ratings.columns:
            ratings = ratings.rename(columns={"team": "team_name"})

        merged = features.copy()
        if {"home_team", "away_team"}.issubset(merged.columns):
            home_map = ratings.set_index("team_name") if "team_name" in ratings.columns else ratings.copy()
            away_map = ratings.copy()
            if "team_name" in home_map.index.names:
                home_map = home_map.reset_index()
            if "team_name" in away_map.columns:
                away_map = away_map.rename(columns={"team_name": "team_name_away"})
            if "team_name" in home_map.columns:
                home_map = home_map.rename(columns={"team_name": "team_name_home"})

            if "team_name_home" in home_map.columns and "team_name_away" in away_map.columns:
                merged = merged.merge(home_map, left_on="home_team", right_on="team_name_home", how="left", suffixes=("", "_home"))
                merged = merged.merge(away_map, left_on="away_team", right_on="team_name_away", how="left", suffixes=("", "_away"))

        return merged

    def predict(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Generate probabilities for each match using deterministic Poisson-based scoring."""
        if merged.empty:
            raise ValueError("Input data must not be empty")

        predictions: List[Dict[str, float]] = []
        for _, row in merged.iterrows():
            home_rate = min(float(CONFIG.POISSON_LIMIT), max(0.1, float(row.get("expected_home_corner", 0.0))))
            away_rate = min(float(CONFIG.POISSON_LIMIT), max(0.1, float(row.get("expected_away_corner", 0.0))))
            total_rate = home_rate + away_rate

            home_probs = self._predict_distribution(home_rate, row.get("home_corners", 0.0))
            away_probs = self._predict_distribution(away_rate, row.get("away_corners", 0.0))
            total_probs = self._predict_total_distribution(total_rate, float(row.get("home_corners", 0.0)) + float(row.get("away_corners", 0.0)))

            probs = {
                "expected_home_corners": home_rate,
                "expected_away_corners": away_rate,
                "expected_total_corners": total_rate,
            }
            for threshold in self._thresholds:
                probs[f"over_{int(threshold)}"] = 1.0 - self._poisson_cdf(threshold, total_rate)
                probs[f"under_{int(threshold)}"] = self._poisson_cdf(threshold, total_rate)
            probs["actual_total_corners"] = float(row.get("home_corners", 0.0)) + float(row.get("away_corners", 0.0))
            predictions.append(probs)

        prediction_frame = pd.DataFrame(predictions)
        return prediction_frame

    def build(self, ratings_path: Union[str, Path], features_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
        """Load the ratings/features inputs, calculate predictions, and persist them as parquet."""
        merged = self.load_inputs(ratings_path, features_path)
        predictions = self.predict(merged)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        return predictions

    def _predict_distribution(self, rate: float, observed: float) -> float:
        """Return a simple Poisson probability based on the observed value."""
        return float(poisson.pmf(int(round(observed)), rate))

    def _predict_total_distribution(self, rate: float, observed: float) -> float:
        """Return a Poisson probability for the total corners."""
        return float(poisson.pmf(int(round(observed)), rate))

    def _poisson_cdf(self, threshold: float, rate: float) -> float:
        """Return the Poisson cumulative probability up to the threshold."""
        return float(poisson.cdf(int(threshold), rate))
