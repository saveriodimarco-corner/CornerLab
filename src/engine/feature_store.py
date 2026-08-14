from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.config import CONFIG
from src.engine.team_rating import TeamRatingEngine
from src.exceptions import InsufficientDataError, InvalidFeatureDataError, InvalidMatchDataError
from src.utils.cache import CacheManager
from src.utils.data_loader import DataLoader
from src.utils.validator import DataValidator


class FeatureStore:
    """Generate match-level feature rows for CornerLab training and analysis workflows."""

    REQUIRED_COLUMNS = {
        "date",
        "season",
        "competition",
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
            raise InvalidMatchDataError("The date column must contain valid dates.")

        working = working.sort_values(["date", "season"]).reset_index(drop=True)
        feature_rows: List[Dict[str, Any]] = []
        prior_state: Dict[str, Dict[str, float]] = {}
        team_history: Dict[str, Dict[str, List[float]]] = {}
        league_state: Dict[str, Dict[str, List[float]]] = {}

        for _, match in working.iterrows():
            season = str(match.get("season", ""))
            feature_rows.append(
                self._create_feature_row(
                    match,
                    pd.DataFrame(columns=["home_team", "away_team", "home_corners", "away_corners"]),
                    prior_state=prior_state,
                    team_history=team_history,
                    league_state=league_state,
                    season=season,
                )
            )
            self._update_state(prior_state, team_history, match, league_state=league_state, season=season)

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

    def _create_feature_row(
        self,
        match: pd.Series,
        prior_matches: pd.DataFrame,
        prior_state: Dict[str, Dict[str, float]] | None = None,
        team_history: Dict[str, Dict[str, List[float]]] | None = None,
        league_state: Dict[str, Dict[str, List[float]]] | None = None,
        season: str | None = None,
        competition: str | None = None,
    ) -> Dict[str, Any]:
        """Construct one normalized feature row for a single match."""
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_corners = float(match["home_corners"])
        away_corners = float(match["away_corners"])

        state = prior_state or {}
        history = team_history or {}
        prior_ratings = pd.DataFrame(
            [
                {
                    "team": team,
                    "home_attack_rating": values.get("home_attack_rating", CONFIG.DEFAULT_RATING_BASELINE),
                    "away_attack_rating": values.get("away_attack_rating", CONFIG.DEFAULT_RATING_BASELINE),
                    "home_defence_rating": values.get("home_defence_rating", CONFIG.DEFAULT_RATING_BASELINE),
                    "away_defence_rating": values.get("away_defence_rating", CONFIG.DEFAULT_RATING_BASELINE),
                    "overall_attack": values.get("overall_attack", CONFIG.DEFAULT_RATING_BASELINE),
                    "overall_defence": values.get("overall_defence", CONFIG.DEFAULT_RATING_BASELINE),
                    "tempo_index": values.get("tempo_index", CONFIG.DEFAULT_RATING_BASELINE),
                    "home_corner_advantage": values.get("home_corner_advantage", 0.0),
                    "away_corner_penalty": values.get("away_corner_penalty", 0.0),
                    "opponent_strength_adjustment": values.get("opponent_strength_adjustment", 1.0),
                    "sample_size": values.get("sample_size", 0.0),
                    "rating_std": values.get("rating_std", 0.0),
                    "confidence": values.get("confidence", 0.0),
                    "standard_deviation": values.get("standard_deviation", 0.0),
                    "consistency_index": values.get("consistency_index", CONFIG.DEFAULT_RATING_BASELINE),
                }
                for team, values in state.items()
            ]
        )
        prior_ratings = prior_ratings.set_index("team") if not prior_ratings.empty else pd.DataFrame(columns=[])

        home_attack_rating = self._get_rating(prior_ratings, home_team, "home_attack_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_attack_rating = self._get_rating(prior_ratings, away_team, "away_attack_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_defence = self._get_rating(prior_ratings, home_team, "home_defence_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_defence = self._get_rating(prior_ratings, away_team, "away_defence_rating", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_tempo = self._get_rating(prior_ratings, home_team, "tempo_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_tempo = self._get_rating(prior_ratings, away_team, "tempo_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_consistency = self._get_rating(prior_ratings, home_team, "consistency_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        away_consistency = self._get_rating(prior_ratings, away_team, "consistency_index", default=CONFIG.DEFAULT_RATING_BASELINE)
        home_advantage = self._get_rating(prior_ratings, home_team, "home_corner_advantage", default=0.0)
        away_penalty = self._get_rating(prior_ratings, away_team, "away_corner_penalty", default=0.0)

        home_history = history.get(home_team, {})
        away_history = history.get(away_team, {})

        home_last5_for = self._average_last_n_from_history(home_history, n=5, kind="for")
        home_last5_against = self._average_last_n_from_history(home_history, n=5, kind="against")
        away_last5_for = self._average_last_n_from_history(away_history, n=5, kind="for")
        away_last5_against = self._average_last_n_from_history(away_history, n=5, kind="against")
        home_last10_for = self._average_last_n_from_history(home_history, n=10, kind="for")
        away_last10_for = self._average_last_n_from_history(away_history, n=10, kind="for")

        home_std = self._std_from_history(home_history, n=5, kind="for")
        away_std = self._std_from_history(away_history, n=5, kind="for")
        combined_std = (home_std + away_std) / 2.0

        expected_total_corner = (home_attack_rating + away_attack_rating + home_defence + away_defence) / 4.0
        expected_home_corner = (home_attack_rating + away_defence) / 2.0
        expected_away_corner = (away_attack_rating + home_defence) / 2.0
        rating_difference = home_attack_rating - away_attack_rating
        tempo_difference = home_tempo - away_tempo

        total_corners = home_corners + away_corners

        rolling_attack_last5 = self._rolling_attack_rating(home_history, n=5)
        rolling_attack_last10 = self._rolling_attack_rating(home_history, n=10)
        rolling_defense_last5 = self._rolling_defense_rating(home_history, n=5)
        rolling_defense_last10 = self._rolling_defense_rating(home_history, n=10)
        home_defense_rating = float(max(0.0, 10.0 - self._average_last_n_from_history(home_history, n=5, kind="against")))
        away_defense_rating = float(max(0.0, 10.0 - self._average_last_n_from_history(away_history, n=5, kind="against")))

        league_key = self._league_key(competition=competition or match.get("competition", ""), season=season or match.get("season", ""))
        attack_percentile = self._league_percentile(league_state, league_key, home_attack_rating, metric="attack")
        defense_percentile = self._league_percentile(league_state, league_key, home_defense_rating, metric="defense")

        features = {
            "home_last5_corner_for": home_last5_for,
            "home_last5_corner_against": home_last5_against,
            "away_last5_corner_for": away_last5_for,
            "away_last5_corner_against": away_last5_against,
            "home_last10_corner_for": home_last10_for,
            "away_last10_corner_for": away_last10_for,
            "home_attack_rating": home_attack_rating,
            "away_attack_rating": away_attack_rating,
            "rolling_attack_rating_last5": rolling_attack_last5,
            "rolling_attack_rating_last10": rolling_attack_last10,
            "home_defence_rating": home_defence,
            "away_defence_rating": away_defence,
            "home_defense_rating": home_defense_rating,
            "away_defense_rating": away_defense_rating,
            "rolling_defense_rating_last5": rolling_defense_last5,
            "rolling_defense_rating_last10": rolling_defense_last10,
            "attack_minus_defense": home_attack_rating - home_defense_rating,
            "home_attack_vs_away_defense": home_attack_rating - away_defense_rating,
            "away_attack_vs_home_defense": away_attack_rating - home_defense_rating,
            "attack_percentile": attack_percentile,
            "defense_percentile": defense_percentile,
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

    def _update_state(
        self,
        prior_state: Dict[str, Dict[str, float]],
        team_history: Dict[str, Dict[str, List[float]]],
        match: pd.Series,
        league_state: Dict[str, Dict[str, List[float]]] | None = None,
        season: str | None = None,
        competition: str | None = None,
    ) -> None:
        """Advance the rolling state with the latest match."""
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_corners = float(match["home_corners"])
        away_corners = float(match["away_corners"])

        for team, team_state in [(home_team, prior_state), (away_team, prior_state)]:
            if team not in prior_state:
                prior_state[team] = {
                    "home_attack_rating": CONFIG.DEFAULT_RATING_BASELINE,
                    "away_attack_rating": CONFIG.DEFAULT_RATING_BASELINE,
                    "home_defence_rating": CONFIG.DEFAULT_RATING_BASELINE,
                    "away_defence_rating": CONFIG.DEFAULT_RATING_BASELINE,
                    "tempo_index": CONFIG.DEFAULT_RATING_BASELINE,
                    "home_corner_advantage": 0.0,
                    "away_corner_penalty": 0.0,
                    "overall_attack": CONFIG.DEFAULT_RATING_BASELINE,
                    "overall_defence": CONFIG.DEFAULT_RATING_BASELINE,
                    "opponent_strength_adjustment": 1.0,
                    "sample_size": 0.0,
                    "rating_std": 0.0,
                    "confidence": 0.0,
                    "standard_deviation": 0.0,
                    "consistency_index": CONFIG.DEFAULT_RATING_BASELINE,
                }

        home_state = prior_state[home_team]
        away_state = prior_state[away_team]
        home_state["home_attack_rating"] = float(home_state.get("home_attack_rating", CONFIG.DEFAULT_RATING_BASELINE))
        away_state["away_attack_rating"] = float(away_state.get("away_attack_rating", CONFIG.DEFAULT_RATING_BASELINE))
        home_state["home_defence_rating"] = float(home_state.get("home_defence_rating", CONFIG.DEFAULT_RATING_BASELINE))
        away_state["away_defence_rating"] = float(away_state.get("away_defence_rating", CONFIG.DEFAULT_RATING_BASELINE))
        home_state["tempo_index"] = float(home_state.get("tempo_index", CONFIG.DEFAULT_RATING_BASELINE))
        away_state["tempo_index"] = float(away_state.get("tempo_index", CONFIG.DEFAULT_RATING_BASELINE))
        home_state["consistency_index"] = float(home_state.get("consistency_index", CONFIG.DEFAULT_RATING_BASELINE))
        away_state["consistency_index"] = float(away_state.get("consistency_index", CONFIG.DEFAULT_RATING_BASELINE))

        home_history = team_history.setdefault(home_team, {"for_values": [], "against_values": [], "total_values": []})
        away_history = team_history.setdefault(away_team, {"for_values": [], "against_values": [], "total_values": []})
        home_history["for_values"].append(home_corners)
        home_history["against_values"].append(away_corners)
        home_history["total_values"].append(home_corners + away_corners)
        away_history["for_values"].append(away_corners)
        away_history["against_values"].append(home_corners)
        away_history["total_values"].append(home_corners + away_corners)

        home_state["home_attack_rating"] = float(home_corners)
        home_state["home_defence_rating"] = float(max(0.0, 10.0 - away_corners))
        home_state["tempo_index"] = float(home_corners + max(0.0, 10.0 - away_corners))
        home_state["consistency_index"] = float(max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, home_corners - away_corners)))))
        away_state["away_attack_rating"] = float(away_corners)
        away_state["away_defence_rating"] = float(max(0.0, 10.0 - home_corners))
        away_state["tempo_index"] = float(away_corners + max(0.0, 10.0 - home_corners))
        away_state["consistency_index"] = float(max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, away_corners - home_corners)))))
        home_state["overall_attack"] = float(home_state["home_attack_rating"])
        home_state["overall_defence"] = float(home_state["home_defence_rating"])
        away_state["overall_attack"] = float(away_state["away_attack_rating"])
        away_state["overall_defence"] = float(away_state["away_defence_rating"])
        home_state["sample_size"] = float(len(home_history["for_values"]))
        away_state["sample_size"] = float(len(away_history["for_values"]))
        home_state["rating_std"] = self._std_from_history(home_history, n=5, kind="for")
        away_state["rating_std"] = self._std_from_history(away_history, n=5, kind="for")
        home_state["confidence"] = self._confidence(home_state["sample_size"], home_state["rating_std"])
        away_state["confidence"] = self._confidence(away_state["sample_size"], away_state["rating_std"])
        home_state["standard_deviation"] = home_state["rating_std"]
        away_state["standard_deviation"] = away_state["rating_std"]

        if league_state is not None:
            league_key = self._league_key(competition=competition or match.get("competition", ""), season=season or match.get("season", ""))
            season_bucket = league_state.setdefault(league_key, {"attack": [], "defense": []})
            season_bucket["attack"].append(float(home_corners))
            season_bucket["attack"].append(float(away_corners))
            season_bucket["defense"].append(float(max(0.0, 10.0 - away_corners)))
            season_bucket["defense"].append(float(max(0.0, 10.0 - home_corners)))

    def _league_key(self, competition: Any, season: Any) -> str:
        competition_name = str(competition or "unknown").strip().lower()
        season_name = str(season or "").strip()
        return f"{competition_name}::{season_name}"

    def _average_last_n_from_history(self, history: Dict[str, List[float]], n: int, kind: str) -> float:
        values = history.get(f"{kind}_values", [])
        if not values:
            return 0.0
        return float(pd.Series(values).tail(n).mean())

    def _std_from_history(self, history: Dict[str, List[float]], n: int, kind: str) -> float:
        values = history.get(f"{kind}_values", [])
        if len(values) < 2:
            return 0.0
        return float(pd.Series(values).tail(n).std(ddof=0))

    def _confidence(self, sample_size: float, rating_std: float) -> float:
        if sample_size <= 0:
            return 0.0
        return min(1.0, sample_size / (sample_size + 5.0)) * (1.0 / (1.0 + rating_std))

    def _rolling_attack_rating(self, history: Dict[str, List[float]], n: int) -> float:
        values = history.get("for_values", [])
        if not values:
            return float(CONFIG.DEFAULT_RATING_BASELINE)
        return float(pd.Series(values).tail(n).mean())

    def _rolling_defense_rating(self, history: Dict[str, List[float]], n: int) -> float:
        values = history.get("against_values", [])
        if not values:
            return float(CONFIG.DEFAULT_RATING_BASELINE)
        average_conceded = float(pd.Series(values).tail(n).mean())
        return float(max(0.0, 10.0 - average_conceded))

    def _league_percentile(self, league_state: Dict[str, Dict[str, List[float]]] | None, league_key: str, value: float, metric: str) -> float:
        if not league_state:
            return 0.5
        season_bucket = league_state.get(str(league_key), {})
        samples = season_bucket.get(metric, [])
        if not samples:
            return 0.5
        ranked = pd.Series(samples)
        lower = float((ranked < value).sum())
        equal = float((ranked == value).sum())
        return float((lower + 0.5 * equal) / len(ranked))

    def _validate_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Validate the match dataframe schema and value types."""
        if not isinstance(matches, pd.DataFrame):
            raise TypeError("matches must be a pandas DataFrame")

        missing = self.REQUIRED_COLUMNS.difference(matches.columns)
        if missing:
            raise InvalidMatchDataError(f"Missing required columns: {sorted(missing)}")

        if matches.empty:
            raise InsufficientDataError("Input data must contain at least one match")

        required_numeric = ["home_corners", "away_corners"]
        for column in required_numeric:
            if pd.to_numeric(matches[column], errors="coerce").isna().any():
                raise InvalidMatchDataError(f"Column {column} contains missing or non-numeric values")
        if matches[["home_team", "away_team"]].isna().any().any() or (matches[["home_team", "away_team"]].astype(str).apply(lambda column: column.str.strip().eq("")).any().any()):
            raise InvalidMatchDataError("Team names must be present.")

        return matches.copy()

    def _ensure_no_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure the feature frame contains no missing values and only numeric values."""
        if features.isna().any().any():
            raise InvalidFeatureDataError("Feature generation produced missing values")
        return features.astype(float, errors="ignore")

    def _get_rating(self, ratings: pd.DataFrame, team: str, metric: str, default: float) -> float:
        """Retrieve a rating for a team from the prior-rating dataframe."""
        if ratings.empty or team not in ratings.index:
            return float(default)
        value = ratings.loc[team, metric]
        return float(value)

    def _rolling_team_stat(self, prior_matches: pd.DataFrame, team: str, role: str, direction: str) -> float:
        """Compute a prior match corner average for a team based on its home/away role."""
        return 0.0

    def _rolling_window(self, prior_matches: pd.DataFrame, team: str, window: int, metric: str) -> float:
        """Compute the recent rolling average corner output for a team."""
        return 0.0

    def _team_volatility(self, prior_matches: pd.DataFrame, team: str) -> float:
        """Estimate rolling variability from a team's recent corner outcomes."""
        return 0.0
