from pathlib import Path

import pandas as pd
import pytest

from src.engine.feature_store import FeatureStore


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
    store = FeatureStore()
    bad_frame = pd.DataFrame({"date": ["2024-08-10"], "season": ["2024/25"], "home_team": ["A"], "away_team": ["B"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        store.transform(bad_frame)


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


def test_build_writes_parquet(tmp_path: Path, match_data: pd.DataFrame):
    output_path = tmp_path / "features.parquet"
    store = FeatureStore()
    features = store.build(match_data, output_path)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert reloaded.equals(features)
