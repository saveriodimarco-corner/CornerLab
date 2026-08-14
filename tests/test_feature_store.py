from pathlib import Path

import pandas as pd
import pytest

from src.engine.feature_store import FeatureStore
from src.exceptions import InvalidMatchDataError


@pytest.fixture
def match_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-08-10", "season": "2024/25", "competition": "Serie A", "home_team": "Team A", "away_team": "Team B", "home_corners": 7, "away_corners": 4},
            {"date": "2024-08-17", "season": "2024/25", "competition": "Serie A", "home_team": "Team B", "away_team": "Team C", "home_corners": 5, "away_corners": 6},
            {"date": "2024-08-24", "season": "2024/25", "competition": "Serie A", "home_team": "Team C", "away_team": "Team A", "home_corners": 3, "away_corners": 8},
            {"date": "2024-08-31", "season": "2024/25", "competition": "Serie A", "home_team": "Team A", "away_team": "Team C", "home_corners": 9, "away_corners": 2},
        ]
    )


def test_missing_columns_raise_explicit_error():
    store = FeatureStore()
    bad_frame = pd.DataFrame({"date": ["2024-08-10"], "season": ["2024/25"], "home_team": ["A"], "away_team": ["B"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        store.transform(bad_frame)


def test_malformed_match_value_raises_domain_error():
    store = FeatureStore()
    bad_frame = pd.DataFrame(
        [
            {"date": "2024-08-10", "season": "2024/25", "competition": "Serie A", "home_team": "A", "away_team": "B", "home_corners": "bad", "away_corners": 4},
        ]
    )

    with pytest.raises(InvalidMatchDataError, match="non-numeric"):
        store.transform(bad_frame)


def test_cold_start_feature_fallback_remains_available():
    store = FeatureStore()
    match = pd.Series(
        {"date": "2024-08-10", "season": "2024/25", "competition": "Serie A", "home_team": "A", "away_team": "B", "home_corners": 0.0, "away_corners": 0.0}
    )

    feature = store._create_feature_row(match, pd.DataFrame(), prior_state={}, team_history={}, league_state={}, season="2024/25", competition="Serie A")

    assert feature["home_last5_corner_for"] == 0.0
    assert feature["attack_percentile"] == 0.5


def test_transform_returns_one_row_per_match_and_expected_features(match_data: pd.DataFrame):
    store = FeatureStore()
    features = store.transform(match_data)

    expected_columns = {
        "home_last5_corner_for",
        "home_last5_corner_against",
        "away_last5_corner_for",
        "away_last5_corner_against",
        "home_last10_corner_for",
        "away_last10_corner_for",
        "home_attack_rating",
        "away_attack_rating",
        "home_defence_rating",
        "away_defence_rating",
        "home_tempo",
        "away_tempo",
        "home_consistency",
        "away_consistency",
        "expected_total_corner",
        "expected_home_corner",
        "expected_away_corner",
        "rating_difference",
        "tempo_difference",
        "home_advantage",
        "home_std",
        "away_std",
        "combined_std",
        "rolling_attack_rating_last5",
        "rolling_attack_rating_last10",
        "rolling_defense_rating_last5",
        "rolling_defense_rating_last10",
        "attack_minus_defense",
        "home_attack_vs_away_defense",
        "away_attack_vs_home_defense",
        "attack_percentile",
        "defense_percentile",
        "over85",
        "over95",
        "over105",
        "over115",
        "under85",
        "under95",
        "under105",
        "under115",
    }

    assert len(features) == len(match_data)
    assert expected_columns.issubset(set(features.columns))
    assert not features.isna().any().any()


def test_create_feature_row_accepts_stateful_history_inputs():
    store = FeatureStore()
    match = pd.Series({
        "date": "2024-08-10",
        "season": "2024/25",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_corners": 7,
        "away_corners": 4,
    })
    prior_state = {
        "Team A": {
            "home_attack_rating": 6.0,
            "away_attack_rating": 5.0,
            "home_defence_rating": 4.0,
            "away_defence_rating": 5.0,
            "tempo_index": 10.0,
            "home_corner_advantage": 0.0,
            "away_corner_penalty": 0.0,
            "consistency_index": 0.8,
        },
        "Team B": {
            "home_attack_rating": 5.0,
            "away_attack_rating": 4.0,
            "home_defence_rating": 5.0,
            "away_defence_rating": 4.0,
            "tempo_index": 9.0,
            "home_corner_advantage": 0.0,
            "away_corner_penalty": 0.0,
            "consistency_index": 0.7,
        },
    }
    team_history = {
        "Team A": {"for_values": [5.0, 6.0], "against_values": [3.0, 4.0], "total_values": [8.0, 10.0]},
        "Team B": {"for_values": [4.0, 4.5], "against_values": [5.0, 4.0], "total_values": [9.0, 8.5]},
    }

    feature = store._create_feature_row(match, pd.DataFrame(columns=["home_team", "away_team", "home_corners", "away_corners"]), prior_state=prior_state, team_history=team_history)

    assert feature["expected_home_corner"] > 0
    assert feature["expected_away_corner"] > 0
    assert feature["expected_total_corner"] > 0
    assert feature["home_last5_corner_for"] == pytest.approx(5.5)


def test_build_writes_parquet(tmp_path: Path, match_data: pd.DataFrame):
    output_path = tmp_path / "features.parquet"
    store = FeatureStore()
    features = store.build(match_data, output_path)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert reloaded.equals(features)
