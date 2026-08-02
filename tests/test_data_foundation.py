from pathlib import Path

import pandas as pd
import pytest

from src.config import CONFIG
from src.engine.feature_store import FeatureStore
from src.engine.team_rating import TeamRatingEngine
from src.utils.cache import CacheManager
from src.utils.data_loader import DataLoader
from src.utils.validator import DataValidator, ValidationError


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


def test_validator_rejects_invalid_data():
    validator = DataValidator()
    bad_frame = pd.DataFrame({"date": ["bad-date"], "season": ["2024/25"], "home_team": ["A"], "away_team": ["B"], "home_corners": [1], "away_corners": [-1]})

    result = validator.validate(bad_frame)
    assert not result.is_valid
    assert any("Invalid dates" in error for error in result.errors) or any("negative" in error for error in result.errors)

    with pytest.raises(ValidationError):
        validator.ensure_valid(bad_frame)


def test_loader_reads_csv_and_parquet(tmp_path: Path, match_data: pd.DataFrame):
    csv_path = tmp_path / "fixtures.csv"
    parquet_path = tmp_path / "fixtures.parquet"
    match_data.to_csv(csv_path, index=False)
    match_data.to_parquet(parquet_path, index=False)

    loader = DataLoader()
    csv_result = loader.load(csv_path)
    parquet_result = loader.load(parquet_path)

    assert list(csv_result.columns) == list(match_data.columns)
    assert list(parquet_result.columns) == list(match_data.columns)


def test_cache_reuses_existing_output(tmp_path: Path, match_data: pd.DataFrame):
    output_path = tmp_path / "cached.parquet"
    cache = CacheManager(tmp_path)
    cache.set(match_data, output_path)

    reloaded = cache.get(match_data.to_csv(index=False), output_path)
    assert reloaded is None


def test_config_contains_expected_settings():
    assert CONFIG.EWMA_ALPHA > 0
    assert CONFIG.MAX_OSA_ITERATIONS > 0
    assert len(CONFIG.DEFAULT_THRESHOLDS) == 4


def test_engine_uses_shared_configuration(match_data: pd.DataFrame):
    ratings_engine = TeamRatingEngine()
    features_engine = FeatureStore()

    ratings = ratings_engine.build(match_data, Path("/tmp/ratings.parquet"))
    features = features_engine.build(match_data, Path("/tmp/features.parquet"))

    assert not ratings.empty
    assert not features.empty
