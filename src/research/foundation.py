from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Sequence


VALID_FEATURE_STATUSES = (
    "PLANNED",
    "IMPLEMENTED",
    "VALIDATED",
    "SELECTED",
    "REJECTED",
    "DEPRECATED",
)


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
        feature_id: str | None = None,
        tier: str = "CORE",
        status: str = "PLANNED",
        priority: int = 1,
        predictive_hypothesis: str = "",
        owner: str = "research",
        required_inputs: Sequence[str] = (),
        output_type: str = "float",
        unit: str = "n/a",
        minimum_history: int = 0,
        missing_policy: str = "zero",
        leakage_risk: float = 0.0,
        computational_cost: str = "low",
        implementation_sprint: str = "23",
        validation_status: str = "PLANNED",
        selection_status: str = "KEEP",
    ) -> None:
        self.name = name
        self.category = category
        self.version = version
        self.description = description
        self.available_before_kickoff = available_before_kickoff
        self.lookback_matches = lookback_matches
        self.dependencies = tuple(dependencies)
        self.feature_id = feature_id or f"GEN-{name}"
        self.tier = tier
        self.status = status
        self.priority = priority
        self.predictive_hypothesis = predictive_hypothesis
        self.owner = owner
        self.required_inputs = tuple(required_inputs)
        self.output_type = output_type
        self.unit = unit
        self.minimum_history = minimum_history
        self.missing_policy = missing_policy
        self.leakage_risk = float(leakage_risk)
        self.computational_cost = computational_cost
        self.implementation_sprint = implementation_sprint
        self.validation_status = validation_status
        self.selection_status = selection_status

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError


class MetadataFeature(ResearchFeature):
    def compute(self) -> Any:
        return None


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: Dict[str, ResearchFeature] = {}

    def register(self, feature: ResearchFeature) -> None:
        if feature.name in self._features:
            raise ValueError(f"Feature '{feature.name}' is already registered")

        self._validate_feature(feature)
        self._features[feature.name] = feature
        return feature

    def _validate_feature(self, feature: ResearchFeature) -> None:
        if not getattr(feature, "feature_id", None):
            raise ValueError("Feature requires a feature_id")

        for existing in self._features.values():
            if existing.feature_id == feature.feature_id:
                raise ValueError(f"Duplicate feature_id '{feature.feature_id}'")

        if feature.status not in VALID_FEATURE_STATUSES:
            raise ValueError(f"Invalid status '{feature.status}'")

        if feature.validation_status not in VALID_FEATURE_STATUSES:
            raise ValueError(f"Invalid validation_status '{feature.validation_status}'")

        if not feature.category.strip():
            raise ValueError("Feature requires a category")

        if not feature.description.strip():
            raise ValueError("Feature requires a description")

        if not feature.version.strip():
            raise ValueError("Feature requires a version")

        if not feature.owner.strip():
            raise ValueError("Feature requires an owner")

        if feature.status in {"IMPLEMENTED", "VALIDATED", "SELECTED"} and not feature.predictive_hypothesis.strip():
            raise ValueError("Implemented features require a predictive hypothesis")

        if not 0.0 <= float(feature.leakage_risk) <= 1.0:
            raise ValueError("leakage_risk must be between 0.0 and 1.0")

        if not isinstance(feature.priority, int) or feature.priority <= 0:
            raise ValueError("priority must be a positive integer")

    def get(self, name: str) -> ResearchFeature:
        return self._features[name]

    def list(self) -> List[str]:
        return list(self._features.keys())

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                "feature_id": feature.feature_id,
                "name": feature.name,
                "category": feature.category,
                "tier": feature.tier,
                "status": feature.status,
                "priority": feature.priority,
                "description": feature.description,
                "predictive_hypothesis": feature.predictive_hypothesis,
                "version": feature.version,
                "owner": feature.owner,
                "dependencies": feature.dependencies,
                "required_inputs": feature.required_inputs,
                "output_type": feature.output_type,
                "unit": feature.unit,
                "lookback_matches": feature.lookback_matches,
                "minimum_history": feature.minimum_history,
                "missing_policy": feature.missing_policy,
                "available_before_kickoff": feature.available_before_kickoff,
                "leakage_risk": feature.leakage_risk,
                "computational_cost": feature.computational_cost,
                "implementation_sprint": feature.implementation_sprint,
                "validation_status": feature.validation_status,
                "selection_status": feature.selection_status,
            }
            for feature in self._features.values()
        ]


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


def build_corner_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    features = [
        MetadataFeature(
            name="corner_creation_rate",
            category="corner",
            version="1.0",
            description="Average corners created by the home team in recent matches.",
            available_before_kickoff=True,
            lookback_matches=3,
            dependencies=("team_history",),
            feature_id="CF-001",
            tier="CORE",
            status="VALIDATED",
            priority=1,
            predictive_hypothesis="Teams that create more corners recently are likelier to sustain that pattern.",
            owner="research",
            required_inputs=("recent_home_corners_for",),
            output_type="float",
            unit="corners per match",
            minimum_history=3,
            missing_policy="zero",
            leakage_risk=0.0,
            computational_cost="low",
            implementation_sprint="23",
            validation_status="VALIDATED",
        ),
        MetadataFeature(
            name="corner_concession_rate",
            category="corner",
            version="1.0",
            description="Average corners conceded by the home team in recent matches.",
            available_before_kickoff=True,
            lookback_matches=3,
            dependencies=("team_history",),
            feature_id="CF-002",
            tier="CORE",
            status="VALIDATED",
            priority=1,
            predictive_hypothesis="Teams that concede more corners recently are likelier to continue conceding them.",
            owner="research",
            required_inputs=("recent_home_corners_against",),
            output_type="float",
            unit="corners per match",
            minimum_history=3,
            missing_policy="zero",
            leakage_risk=0.0,
            computational_cost="low",
            implementation_sprint="23",
            validation_status="VALIDATED",
        ),
        MetadataFeature(
            name="recent_corner_form",
            category="corner",
            version="1.0",
            description="Recent corner creation balance between home and away teams.",
            available_before_kickoff=True,
            lookback_matches=3,
            dependencies=("team_history",),
            feature_id="CF-003",
            tier="CORE",
            status="VALIDATED",
            priority=2,
            predictive_hypothesis="Recent corner momentum can separate teams before kickoff.",
            owner="research",
            required_inputs=("recent_home_corners_for", "recent_away_corners_for"),
            output_type="float",
            unit="corners",
            minimum_history=3,
            missing_policy="zero",
            leakage_risk=0.0,
            computational_cost="low",
            implementation_sprint="23",
            validation_status="VALIDATED",
        ),
        MetadataFeature(
            name="corner_diff_pressure",
            category="corner",
            version="1.0",
            description="Difference in recent corner pressure between the home and away teams.",
            available_before_kickoff=True,
            lookback_matches=3,
            dependencies=("team_history",),
            feature_id="CF-004",
            tier="CORE",
            status="VALIDATED",
            priority=2,
            predictive_hypothesis="Sharp differences in recent corner pressure can foreshadow outcomes.",
            owner="research",
            required_inputs=("recent_home_corner_pressure", "recent_away_corner_pressure"),
            output_type="float",
            unit="corners per match",
            minimum_history=3,
            missing_policy="zero",
            leakage_risk=0.0,
            computational_cost="low",
            implementation_sprint="23",
            validation_status="VALIDATED",
        ),
    ]
    for feature in features:
        registry.register(feature)
    return registry


CORNER_FEATURE_REGISTRY = build_corner_feature_registry()


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
