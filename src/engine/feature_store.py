from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.config import CONFIG
from src.engine.team_rating import TeamRatingEngine
from src.utils.cache import CacheManager
from src.utils.data_loader import DataLoader
from src.utils.validator import DataValidator


class FeatureStore:
    """Generate match-level feature rows for CornerLab training and analysis workflows."""

    REQUIRED_COLUMNS = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_corners",
        "away_corners",
    }

    def __init__(self) -> None:
        """Initialize the feature store with the rating engine dependency."""
        self.rating_engine = TeamRatingEngine()
        self.loader = DataLoader(DataValidator())
        self.cache = CacheManager()

    def load_matches(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load match data from a CSV or Parquet file when a file path is supplied."""
        return self.loader.load(source)

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Create one feature row per match from a match dataframe."""
        working = self._validate_matches(matches).copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        if working["date"].isna().any():
            raise ValueError("The date column must contain valid dates.")

        working = working.sort_values(["date", "season"]).reset_index(drop=True)
        feature_rows: List[Dict[str, Any]] = []

        for index, match in working.iterrows():
            prior_matches = working.iloc[:index].copy()
            feature_rows.append(self._create_feature_row(match, prior_matches))

        features = pd.DataFrame(feature_rows)
        if features.empty:
            return features

        features = self._ensure_no_missing(features)
        return features.reset_index(drop=True)

    def build(self, source: Union[str, Path, pd.DataFrame], output_path: Union[str, Path]) -> pd.DataFrame:
        """Transform input data into features and persist the result as parquet."""
        if isinstance(source, pd.DataFrame):
            matches = source
        else:
            matches = self.load_matches(source)

        cached = self.cache.get(source, output_path) if not isinstance(source, pd.DataFrame) else None
        if cached is not None:
            return cached

        features = self.transform(matches)
        output = self.loader.ensure_output_dir(output_path)
        features.to_parquet(output, index=False)
        self.cache.set(features, output)
        return features

    def _create_feature_row(self, match: pd.Series, prior_matches: pd.DataFrame) -> Dict[str, Any]:
        """Construct one normalized feature row for a single match."""
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_corners = float(match["home_corners"])
        away_corners = float(match["away_corners"])

        prior_ratings = self.rating_engine.calculate_ratings(prior_matches) if not prior_matches.empty else pd.DataFrame(columns=["team", "home_attack_rating", "away_attack_rating", "home_defence_rating", "away_defence_rating", "overall_attack", "overall_defence", "tempo_index", "home_corner_advantage", "away_corner_penalty", "opponent_strength_adjustment", "sample_size", "rating_std", "confidence", "standard_deviation", "consistency_index"])
        prior_ratings = prior_ratings.set_index("team") if not prior_ratings.empty else pd.DataFrame(columns=[])

        home_strength = self._get_rating(prior_ratings, home_team, "home_attack_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_strength = self._get_rating(prior_ratings, away_team, "away_attack_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_defence = self._get_rating(prior_ratings, home_team, "home_defence_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_defence = self._get_rating(prior_ratings, away_team, "away_defence_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_tempo = self._get_rating(prior_ratings, home_team, "tempo_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_tempo = self._get_rating(prior_ratings, away_team, "tempo_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_consistency = self._get_rating(prior_ratings, home_team, "consistency_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_consistency = self._get_rating(prior_ratings, away_team, "consistency_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_advantage = self._get_rating(prior_ratings, home_team, "home_corner_advantage", default=0.0)
        away_penalty = self._get_rating(prior_ratings, away_team, "away_corner_penalty", default=0.0)

        home_form_for = self._rolling_team_stat(prior_matches, home_team, "home", "for")
        home_form_against = self._rolling_team_stat(prior_matches, home_team, "home", "against")
        away_form_for = self._rolling_team_stat(prior_matches, away_team, "away", "for")
        away_form_against = self._rolling_team_stat(prior_matches, away_team, "away", "against")

        home_last5_for = self._rolling_window(prior_matches, home_team, window=5, metric="for")
        home_last5_against = self._rolling_window(prior_matches, home_team, window=5, metric="against")
        away_last5_for = self._rolling_window(prior_matches, away_team, window=5, metric="for")
        away_last5_against = self._rolling_window(prior_matches, away_team, window=5, metric="against")
        home_last10_for = self._rolling_window(prior_matches, home_team, window=10, metric="for")
        away_last10_for = self._rolling_window(prior_matches, away_team, window=10, metric="for")

        home_std = self._team_volatility(prior_matches, home_team)
        away_std = self._team_volatility(prior_matches, away_team)
        combined_std = (home_std + away_std) / 2.0

        expected_total_corner = (home_strength + away_strength + home_defence + away_defence) / 4.0
        expected_home_corner = (home_strength + away_defence) / 2.0
        expected_away_corner = (away_strength + home_defence) / 2.0
        rating_difference = home_strength - away_strength
        tempo_difference = home_tempo - away_tempo

        total_corners = home_corners + away_corners
        features = {
            "home_last5_corner_for": home_last5_for,
            "home_last5_corner_against": home_last5_against,
            "away_last5_corner_for": away_last5_for,
            "away_last5_corner_against": away_last5_against,
            "home_last10_corner_for": home_last10_for,
            "away_last10_corner_for": away_last10_for,
            "home_attack_rating": home_strength,
            "away_attack_rating": away_strength,
            "home_defence_rating": home_defence,
            "away_defence_rating": away_defence,
            "home_tempo": home_tempo,
            "away_tempo": away_tempo,
            "home_consistency": home_consistency,
            "away_consistency": away_consistency,
            "expected_total_corner": expected_total_corner,
            "expected_home_corner": expected_home_corner,
            "expected_away_corner": expected_away_corner,
            "rating_difference": rating_difference,
            "tempo_difference": tempo_difference,
            "home_advantage": (home_advantage + away_penalty) * CONFIG.HOME_ADVANTAGE_FACTOR,
            "home_std": home_std,
            "away_std": away_std,
            "combined_std": combined_std,
            "over85": int(total_corners > 85),
            "over95": int(total_corners > 95),
            "over105": int(total_corners > 105),
            "over115": int(total_corners > 115),
            "under85": int(total_corners < 85),
            "under95": int(total_corners < 95),
            "under105": int(total_corners < 105),
            "under115": int(total_corners < 115),
        }
        return features

    def _validate_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Validate the match dataframe schema and value types."""
        if not isinstance(matches, pd.DataFrame):
            raise TypeError("matches must be a pandas DataFrame")

        missing = self.REQUIRED_COLUMNS.difference(matches.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if matches.empty:
            raise ValueError("Input data must contain at least one match")

        required_numeric = ["home_corners", "away_corners"]
        for column in required_numeric:
            if matches[column].isna().any():
                raise ValueError(f"Column {column} contains missing values")

        return matches.copy()

    def _ensure_no_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure the feature frame contains no missing values and only numeric values."""
        if features.isna().any().any():
            features = features.fillna(0.0)
        if features.isna().any().any():
            raise ValueError("Feature generation produced missing values")
        return features.astype(float, errors="ignore")

    def _get_rating(self, ratings: pd.DataFrame, team: str, metric: str, default: float) -> float:
        """Retrieve a rating for a team from the prior-rating dataframe."""
        if ratings.empty or team not in ratings.index:
            return float(default)
        value = ratings.loc[team, metric]
        return float(value)

    def _rolling_team_stat(self, prior_matches: pd.DataFrame, team: str, role: str, direction: str) -> float:
        """Compute a prior match corner average for a team based on its home/away role."""
        if prior_matches.empty:
            return 0.0

        if role == "home":
            subset = prior_matches[prior_matches["home_team"] == team]
            if direction == "for":
                values = subset["home_corners"]
            else:
                values = subset["away_corners"]
        else:
            subset = prior_matches[prior_matches["away_team"] == team]
            if direction == "for":
                values = subset["away_corners"]
            else:
                values = subset["home_corners"]

        if values.empty:
            return 0.0
        return float(values.tail(5).mean())

    def _rolling_window(self, prior_matches: pd.DataFrame, team: str, window: int, metric: str) -> float:
        """Compute the recent rolling average corner output for a team."""
        if prior_matches.empty:
            return 0.0

        series: List[float] = []
        for _, match in prior_matches.iterrows():
            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            home_corners = float(match["home_corners"])
            away_corners = float(match["away_corners"])
            if home_team == team:
                value = home_corners if metric == "for" else away_corners
            elif away_team == team:
                value = away_corners if metric == "for" else home_corners
            else:
                continue
            series.append(value)

        if not series:
            return 0.0
        return float(pd.Series(series).tail(window).mean())

    def _team_volatility(self, prior_matches: pd.DataFrame, team: str) -> float:
        """Estimate rolling variability from a team's recent corner outcomes."""
        if prior_matches.empty:
            return 0.0
        values: List[float] = []
        for _, match in prior_matches.iterrows():
            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            if home_team == team:
                values.extend([float(match["home_corners"]), float(match["away_corners"])])
            elif away_team == team:
                values.extend([float(match["away_corners"]), float(match["home_corners"])])
        if len(values) < 2:
            return 0.0
        return float(pd.Series(values).std(ddof=0))
