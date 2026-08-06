from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class LeakageError(RuntimeError):
    """Raised when a feature is used after its data would have been available."""


class ResearchFeature(ABC):
    def __init__(
        self,
        *,
        name: str,
        category: str,
        version: str,
        description: str,
        available_before_kickoff: bool,
        lookback_matches: int,
        dependencies: Sequence[str],
    ) -> None:
        self.name = name
        self.category = category
        self.version = version
        self.description = description
        self.available_before_kickoff = available_before_kickoff
        self.lookback_matches = lookback_matches
        self.dependencies = tuple(dependencies)

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: Dict[str, ResearchFeature] = {}

    def register(self, feature: ResearchFeature) -> None:
        self._features[feature.name] = feature

    def get(self, name: str) -> ResearchFeature:
        return self._features[name]

    def list(self) -> List[str]:
        return list(self._features.keys())


class LeakageValidator:
    def validate(
        self,
        feature: ResearchFeature,
        *,
        kickoff: str,
        source_timestamps: Sequence[str],
    ) -> None:
        if feature.available_before_kickoff:
            return
        raise LeakageError("Feature is not available before kickoff")


class DatasetSplitter:
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"

    def split(
        self,
        rows: Sequence[Any],
        *,
        train_size: int,
        validation_size: int,
        test_size: int,
        out_of_sample_size: int,
    ) -> Dict[str, List[Any]]:
        if train_size < 0 or validation_size < 0 or test_size < 0 or out_of_sample_size < 0:
            raise ValueError("Split sizes must be non-negative")

        total = train_size + validation_size + test_size + out_of_sample_size
        if total > len(rows):
            raise ValueError("Split sizes exceed available rows")

        start = 0
        splits: Dict[str, List[Any]] = {}
        for name, size in [
            (self.TRAIN, train_size),
            (self.VALIDATION, validation_size),
            (self.TEST, test_size),
            (self.OUT_OF_SAMPLE, out_of_sample_size),
        ]:
            if size:
                splits[name] = list(rows[start:start + size])
                start += size
            else:
                splits[name] = []
        return splits


def ensure_research_schema(db_path: Path | str | None = None) -> Path:
    path = Path(db_path) if db_path is not None else Path("data/research/research.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_feature_scores (
                feature_name TEXT PRIMARY KEY,
                importance REAL,
                stability REAL,
                drift REAL,
                correlation REAL,
                mutual_information REAL,
                last_validation TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()
    return path
