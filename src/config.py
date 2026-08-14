from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Config:
    """Central configuration for CornerLab engine modules."""

    EWMA_ALPHA: float = 0.30
    MAX_OSA_ITERATIONS: int = 20
    HOME_ADVANTAGE_FACTOR: float = 1.0
    POISSON_LIMIT: int = 20
    DEFAULT_THRESHOLDS: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5)
    DEFAULT_RATING_BASELINE: float = 50.0
    DATABASE_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "cornerlab.db"
    DATA_PATHS: Dict[str, Path] = field(default_factory=lambda: {
        "raw": Path(__file__).resolve().parent.parent / "data" / "raw",
        "processed": Path(__file__).resolve().parent.parent / "data" / "processed",
        "features": Path(__file__).resolve().parent.parent / "data" / "features",
        "predictions": Path(__file__).resolve().parent.parent / "data" / "predictions",
    })
    LOG_LEVEL: str = "INFO"
    FEATURE_OUTPUT_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "features" / "features.parquet"
    RATING_OUTPUT_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "team_ratings.parquet"
    PREDICTION_OUTPUT_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "predictions" / "predictions.parquet"


CONFIG = Config()
