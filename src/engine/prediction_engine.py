from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import CONFIG
from src.exceptions import InsufficientDataError, InvalidFeatureDataError


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
            raise InsufficientDataError("Input data must not be empty")

        required_fields = ["expected_home_corner", "expected_away_corner", "expected_total_corner"]
        for field in required_fields:
            if field not in merged.columns:
                raise InvalidFeatureDataError(f"Missing required expected-corner field: {field}")

        predictions: List[Dict[str, float]] = []
        for idx, row in merged.iterrows():
            try:
                home_rate = float(row["expected_home_corner"])
                away_rate = float(row["expected_away_corner"])
                total_rate = float(row["expected_total_corner"])
            except (TypeError, ValueError) as exc:
                raise InvalidFeatureDataError("Expected-corner fields must be numeric.") from exc

            if not np.isfinite([home_rate, away_rate, total_rate]).all() or min(home_rate, away_rate, total_rate) < 0.0:
                raise InvalidFeatureDataError("Expected-corner fields must be finite and non-negative.")

            home_probs = self._predict_distribution(home_rate, row.get("home_corners", 0.0))
            away_probs = self._predict_distribution(away_rate, row.get("away_corners", 0.0))
            total_probs = self._predict_total_distribution(total_rate, float(row.get("home_corners", 0.0)) + float(row.get("away_corners", 0.0)))

            default_probability = 1.0 - self._poisson_cdf(8.5, total_rate)
            raw_probability = row.get("predicted_probability", default_probability)
            if pd.isna(raw_probability):
                raw_probability = default_probability
            raw_closing_odds = row.get("closing_odds", 2.0)
            if pd.isna(raw_closing_odds):
                raw_closing_odds = 2.0
            raw_confidence = row.get("model_confidence", row.get("confidence", 0.75))
            if pd.isna(raw_confidence):
                raw_confidence = 0.75

            probs = {
                "expected_home_corner": home_rate,
                "expected_away_corner": away_rate,
                "expected_total_corner": total_rate,
                "match_id": int(row.get("match_id", idx + 1)),
                "market": str(row.get("market", "OVER 8.5")),
                "closing_odds": float(raw_closing_odds),
                "predicted_probability": float(np.clip(float(raw_probability), 0.0, 1.0)),
                "model_confidence": float(np.clip(float(raw_confidence), 0.0, 1.0)),
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
