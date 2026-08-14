from pathlib import Path

import pandas as pd
import pytest

from src.engine.team_rating import TeamRatingEngine
from src.exceptions import InvalidMatchDataError


@pytest.fixture
def match_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-08-10", "season": "2024/25", "home_team": "Team A", "away_team": "Team B", "home_corners": 7, "away_corners": 4},
            {"date": "2024-08-17", "season": "2024/25", "home_team": "Team B", "away_team": "Team C", "home_corners": 5, "away_corners": 6},
            {"date": "2024-08-24", "season": "2024/25", "home_team": "Team C", "away_team": "Team A", "home_corners": 3, "away_corners": 8},
            {"date": "2024-08-31", "season": "2024/25", "home_team": "Team A", "away_team": "Team C", "home_corners": 9, "away_corners": 2},
        ]
    )


def test_missing_columns_raise_explicit_error():
    engine = TeamRatingEngine()
    bad_frame = pd.DataFrame({"date": ["2024-08-10"], "season": ["2024/25"], "home_team": ["A"], "away_team": ["B"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        engine.calculate_ratings(bad_frame)


def test_malformed_corner_input_raises_domain_error(match_data: pd.DataFrame):
    engine = TeamRatingEngine()
    match_data.loc[0, "home_corners"] = "not-a-number"

    with pytest.raises(InvalidMatchDataError, match="Corner values must be numeric"):
        engine.calculate_ratings(match_data)


def test_calculate_ratings_returns_expected_columns(match_data: pd.DataFrame):
    engine = TeamRatingEngine()
    ratings = engine.calculate_ratings(match_data)

    expected_columns = {
        "team",
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
        "home_attack_sample_size",
        "home_attack_rating_std",
        "home_attack_confidence",
        "away_attack_sample_size",
        "away_attack_rating_std",
        "away_attack_confidence",
        "home_defence_sample_size",
        "home_defence_rating_std",
        "home_defence_confidence",
        "away_defence_sample_size",
        "away_defence_rating_std",
        "away_defence_confidence",
    }

    assert expected_columns.issubset(set(ratings.columns))
    assert ratings["team"].nunique() == 3
    assert ratings.loc[ratings["team"] == "Team A", "home_attack_rating"].iloc[0] >= 0
    assert ratings.loc[ratings["team"] == "Team A", "confidence"].iloc[0] >= 0


def test_build_pipeline_loads_csv_and_writes_parquet(tmp_path: Path, match_data: pd.DataFrame):
    csv_path = tmp_path / "matches.csv"
    output_path = tmp_path / "ratings.parquet"
    match_data.to_csv(csv_path, index=False)

    engine = TeamRatingEngine()
    ratings = engine.build(csv_path, output_path)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert reloaded.equals(ratings)
