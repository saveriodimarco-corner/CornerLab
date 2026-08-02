from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from src.config import CONFIG
from src.utils.cache import CacheManager
from src.utils.data_loader import DataLoader
from src.utils.validator import DataValidator


class TeamRatingEngine:
    """Calculate weighted corner-based team rating metrics from match data."""

    REQUIRED_COLUMNS = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_corners",
        "away_corners",
    }

    def __init__(self, alpha: float = 0.30, max_iterations: int = 20, convergence_threshold: float = 1e-6) -> None:
        """Initialize the engine with EWMA and convergence settings."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.alpha = alpha if alpha != 0.30 else CONFIG.EWMA_ALPHA
        self.max_iterations = max_iterations if max_iterations != 20 else CONFIG.MAX_OSA_ITERATIONS
        self.convergence_threshold = convergence_threshold
        self.loader = DataLoader(DataValidator())
        self.cache = CacheManager()

    def load_matches(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load football match data from a CSV or Parquet file."""
        return self.loader.load(source)

    def calculate_ratings(
        self,
        matches: pd.DataFrame,
        alpha: float | None = None,
        max_iterations: int | None = None,
        convergence_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Calculate weighted per-team corner ratings from a match dataframe."""
        self._validate_columns(matches)

        alpha = self.alpha if alpha is None else alpha
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        convergence_threshold = self.convergence_threshold if convergence_threshold is None else convergence_threshold

        working = matches.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working["home_corners"] = pd.to_numeric(working["home_corners"], errors="coerce")
        working["away_corners"] = pd.to_numeric(working["away_corners"], errors="coerce")

        if working["date"].isna().any():
            raise ValueError("The date column must contain valid dates.")
        if working[["home_corners", "away_corners"]].isna().any().any():
            raise ValueError("Corner values must be numeric.")

        working = working.sort_values("date").reset_index(drop=True)

        teams = sorted(set(working["home_team"]).union(working["away_team"]))
        states: Dict[str, Dict[str, float]] = {
            team: {
                "home_attack_rating": 0.0,
                "away_attack_rating": 0.0,
                "home_defence_rating": 0.0,
                "away_defence_rating": 0.0,
                "home_corner_advantage": 0.0,
                "away_corner_penalty": 0.0,
                "opponent_strength_adjustment": 1.0,
            }
            for team in teams
        }

        raw_histories: Dict[str, Dict[str, List[float]]] = {
            team: {
                "home_attack": [],
                "away_attack": [],
                "home_defence": [],
                "away_defence": [],
                "home_corner_advantage": [],
                "away_corner_penalty": [],
                "match_edges": [],
            }
            for team in teams
        }

        previous_state = {team: dict(values) for team, values in states.items()}
        for _ in range(max_iterations):
            current_state = {team: dict(values) for team, values in previous_state.items()}
            per_team_histories = {team: {key: [] for key in raw_histories[team]} for team in teams}
            for _, match in working.iterrows():
                home_team = str(match["home_team"])
                away_team = str(match["away_team"])
                home_corners = float(match["home_corners"])
                away_corners = float(match["away_corners"])

                home_strength = previous_state[home_team]["opponent_strength_adjustment"]
                away_strength = previous_state[away_team]["opponent_strength_adjustment"]
                home_weight = max(0.5, 1.0 + (away_strength - 1.0))
                away_weight = max(0.5, 1.0 + (home_strength - 1.0))

                home_attack_obs = home_corners * home_weight
                away_attack_obs = away_corners * away_weight
                home_defence_obs = max(0.0, 10.0 - away_corners) * home_weight
                away_defence_obs = max(0.0, 10.0 - home_corners) * away_weight
                home_advantage_obs = (home_corners - away_corners) * home_weight
                away_penalty_obs = (away_corners - home_corners) * away_weight
                match_edge = abs(home_corners - away_corners)

                def update_metric(team: str, metric: str, observation: float, sample_key: str, history_key: str) -> None:
                    current_value = current_state[team][metric]
                    new_value = (1.0 - alpha) * current_value + alpha * observation
                    current_state[team][metric] = new_value
                    current_state[team][f"{metric}_sample_size"] = current_state[team].get(f"{metric}_sample_size", 0.0) + 1.0
                    current_state[team][f"{metric}_rating_std"] = current_state[team].get(f"{metric}_rating_std", 0.0)
                    per_team_histories[team][history_key].append(observation)
                    current_state[team][f"{metric}_rating_std"] = self._weighted_std(per_team_histories[team][history_key])
                    current_state[team][f"{metric}_confidence"] = self._confidence(
                        current_state[team].get(f"{metric}_sample_size", 0.0),
                        current_state[team][f"{metric}_rating_std"],
                    )

                update_metric(home_team, "home_attack_rating", home_attack_obs, "home_attack_sample_size", "home_attack")
                update_metric(away_team, "away_attack_rating", away_attack_obs, "away_attack_sample_size", "away_attack")
                update_metric(home_team, "home_defence_rating", home_defence_obs, "home_defence_sample_size", "home_defence")
                update_metric(away_team, "away_defence_rating", away_defence_obs, "away_defence_sample_size", "away_defence")
                update_metric(home_team, "home_corner_advantage", home_advantage_obs, "home_corner_advantage_sample_size", "home_corner_advantage")
                update_metric(away_team, "away_corner_penalty", away_penalty_obs, "away_corner_penalty_sample_size", "away_corner_penalty")
                per_team_histories[home_team]["match_edges"].append(match_edge)
                per_team_histories[away_team]["match_edges"].append(match_edge)

            for team in teams:
                current_state[team]["overall_attack"] = (current_state[team]["home_attack_rating"] + current_state[team]["away_attack_rating"]) / 2.0
                current_state[team]["overall_defence"] = (current_state[team]["home_defence_rating"] + current_state[team]["away_defence_rating"]) / 2.0
                current_state[team]["tempo_index"] = current_state[team]["overall_attack"] + current_state[team]["overall_defence"]
                current_state[team]["corner_difference"] = current_state[team]["home_attack_rating"] - current_state[team]["home_defence_rating"]
                current_state[team]["corner_balance"] = current_state[team]["away_attack_rating"] - current_state[team]["away_defence_rating"]
                current_state[team]["sample_size"] = max(
                    current_state[team].get("home_attack_rating_sample_size", 0.0),
                    current_state[team].get("away_attack_rating_sample_size", 0.0),
                )
                current_state[team]["rating_std"] = self._weighted_std(per_team_histories[team]["match_edges"])
                current_state[team]["confidence"] = self._confidence(current_state[team]["sample_size"], current_state[team]["rating_std"])
                current_state[team]["standard_deviation"] = current_state[team]["rating_std"]
                current_state[team]["consistency_index"] = 1.0 / (1.0 + current_state[team]["standard_deviation"])

            max_delta = max(abs(current_state[team][key] - previous_state[team][key]) for team in teams for key in ["home_attack_rating", "away_attack_rating", "home_defence_rating", "away_defence_rating", "home_corner_advantage", "away_corner_penalty", "opponent_strength_adjustment"])
            previous_state = current_state
            if max_delta <= convergence_threshold:
                break

        ratings = pd.DataFrame(
            [
                {
                    "team": team,
                    "home_attack_rating": current_state[team]["home_attack_rating"],
                    "away_attack_rating": current_state[team]["away_attack_rating"],
                    "home_defence_rating": current_state[team]["home_defence_rating"],
                    "away_defence_rating": current_state[team]["away_defence_rating"],
                    "home_corner_advantage": current_state[team]["home_corner_advantage"],
                    "away_corner_penalty": current_state[team]["away_corner_penalty"],
                    "overall_attack": current_state[team]["overall_attack"],
                    "overall_defence": current_state[team]["overall_defence"],
                    "tempo_index": current_state[team]["tempo_index"],
                    "corner_difference": current_state[team]["corner_difference"],
                    "corner_balance": current_state[team]["corner_balance"],
                    "opponent_strength_adjustment": current_state[team]["opponent_strength_adjustment"],
                    "sample_size": current_state[team]["sample_size"],
                    "rating_std": current_state[team]["rating_std"],
                    "confidence": current_state[team]["confidence"],
                    "standard_deviation": current_state[team]["standard_deviation"],
                    "consistency_index": current_state[team]["consistency_index"],
                    "home_attack_sample_size": current_state[team].get("home_attack_rating_sample_size", 0.0),
                    "home_attack_rating_std": current_state[team].get("home_attack_rating_rating_std", 0.0),
                    "home_attack_confidence": current_state[team].get("home_attack_rating_confidence", 0.0),
                    "away_attack_sample_size": current_state[team].get("away_attack_rating_sample_size", 0.0),
                    "away_attack_rating_std": current_state[team].get("away_attack_rating_rating_std", 0.0),
                    "away_attack_confidence": current_state[team].get("away_attack_rating_confidence", 0.0),
                    "home_defence_sample_size": current_state[team].get("home_defence_rating_sample_size", 0.0),
                    "home_defence_rating_std": current_state[team].get("home_defence_rating_rating_std", 0.0),
                    "home_defence_confidence": current_state[team].get("home_defence_rating_confidence", 0.0),
                    "away_defence_sample_size": current_state[team].get("away_defence_rating_sample_size", 0.0),
                    "away_defence_rating_std": current_state[team].get("away_defence_rating_rating_std", 0.0),
                    "away_defence_confidence": current_state[team].get("away_defence_rating_confidence", 0.0),
                }
                for team in teams
            ]
        )

        ratings = self._normalize_ratings(ratings)
        return ratings.sort_values("team").reset_index(drop=True)

    def build(self, source: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
        """Load match data, calculate ratings, and write a parquet file."""
        cached = self.cache.get(source, output_path)
        if cached is not None:
            return cached

        matches = self.load_matches(source)
        ratings = self.calculate_ratings(matches)
        output = self.loader.ensure_output_dir(output_path)
        ratings.to_parquet(output, index=False)
        self.cache.set(ratings, output)
        return ratings

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Ensure the incoming dataframe contains all required columns."""
        missing = self.REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _weighted_std(self, values: List[float]) -> float:
        """Return the standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        return float(pd.Series(values).std(ddof=0))

    def _confidence(self, sample_size: float, rating_std: float) -> float:
        """Return a confidence score based on sample size and variability."""
        if sample_size <= 0:
            return 0.0
        return min(1.0, sample_size / (sample_size + 5.0)) * (1.0 / (1.0 + rating_std))

    def _normalize_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Scale rating-bearing columns from 0 to 100."""
        normalized = ratings.copy()
        rating_columns = [
            "home_attack_rating",
            "away_attack_rating",
            "home_defence_rating",
            "away_defence_rating",
            "home_corner_advantage",
            "away_corner_penalty",
            "overall_attack",
            "overall_defence",
            "tempo_index",
            "corner_difference",
            "corner_balance",
            "opponent_strength_adjustment",
            "sample_size",
            "rating_std",
            "confidence",
            "standard_deviation",
            "consistency_index",
        ]
        for column in rating_columns:
            if column in normalized.columns:
                series = normalized[column].astype(float)
                if series.max() == series.min():
                    normalized[column] = 50.0
                else:
                    normalized[column] = 100.0 * (series - series.min()) / (series.max() - series.min())
        for column in [
            "home_attack_sample_size",
            "away_attack_sample_size",
            "home_defence_sample_size",
            "away_defence_sample_size",
            "home_attack_rating_std",
            "away_attack_rating_std",
            "home_defence_rating_std",
            "away_defence_rating_std",
            "home_attack_confidence",
            "away_attack_confidence",
            "home_defence_confidence",
            "away_defence_confidence",
        ]:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype(float)
        return normalized
