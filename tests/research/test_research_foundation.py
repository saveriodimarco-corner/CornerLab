from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from src.research.foundation import (
    DatasetSplitter,
    FeatureRegistry,
    LeakageError,
    LeakageValidator,
    ResearchFeature,
    ensure_research_schema,
)


class SampleFeature(ResearchFeature):
    def __init__(self) -> None:
        super().__init__(
            name="sample_feature",
            category="context",
            version="1.0",
            description="sample",
            available_before_kickoff=True,
            lookback_matches=3,
            dependencies=("team_rating",),
        )

    def compute(self) -> float:
        return 1.0


def test_research_feature_exposes_metadata() -> None:
    feature = SampleFeature()

    assert feature.name == "sample_feature"
    assert feature.category == "context"
    assert feature.version == "1.0"
    assert feature.description == "sample"
    assert feature.available_before_kickoff is True
    assert feature.lookback_matches == 3
    assert feature.dependencies == ("team_rating",)


def test_feature_registry_requires_explicit_registration() -> None:
    registry = FeatureRegistry()
    feature = SampleFeature()

    registry.register(feature)

    assert registry.get("sample_feature") is feature
    assert registry.list() == ["sample_feature"]

    with pytest.raises(KeyError):
        registry.get("missing_feature")


def test_leakage_validator_blocks_non_pre_kickoff_features() -> None:
    feature = SampleFeature()
    validator = LeakageValidator()

    validator.validate(feature, kickoff="2025-01-01T00:00:00", source_timestamps=["2024-12-31T23:59:00"])

    feature.available_before_kickoff = False

    with pytest.raises(LeakageError):
        validator.validate(feature, kickoff="2025-01-01T00:00:00", source_timestamps=["2024-12-31T23:59:00"])


def test_dataset_splitter_preserves_chronological_order() -> None:
    splitter = DatasetSplitter()
    rows = ["r0", "r1", "r2", "r3", "r4"]

    result = splitter.split(rows, train_size=2, validation_size=1, test_size=1, out_of_sample_size=1)

    assert result[DatasetSplitter.TRAIN] == ["r0", "r1"]
    assert result[DatasetSplitter.VALIDATION] == ["r2"]
    assert result[DatasetSplitter.TEST] == ["r3"]
    assert result[DatasetSplitter.OUT_OF_SAMPLE] == ["r4"]


def test_ensure_research_schema_creates_feature_scores_table(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"

    ensure_research_schema(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}

    assert "research_feature_scores" in table_names
