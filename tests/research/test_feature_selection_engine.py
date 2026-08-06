from __future__ import annotations

import pandas as pd
import pytest

from src.research.feature_selection_engine import FeatureSelectionContractError, FeatureSelectionEngine
from src.research.foundation import FeatureRegistry, ResearchFeature


def _with_signal_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    return dataframe.assign(signal_score=1.0)


def test_constant_feature_is_dropped() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"constant_feature": [1, 1, 1, 1]})

    results = engine.evaluate(_with_signal_score(dataframe))

    assert len(results) == 1
    assert results[0]["feature_name"] == "constant_feature"
    assert results[0]["selection_status"] == "DROP"
    assert results[0]["constant_feature"] == "YES"
    assert results[0]["variance"] == 0.0


def test_missing_feature_is_dropped() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"missing_feature": [1.0, 2.0, None, None, None]})

    results = engine.evaluate(_with_signal_score(dataframe))

    assert len(results) == 1
    assert results[0]["selection_status"] == "DROP"
    assert results[0]["missing_ratio"] > 0.25


def test_duplicated_feature_is_reviewed() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [0.0, 1.0, 2.0, 3.0],
        }
    )

    results = engine.evaluate(_with_signal_score(dataframe))

    by_name = {item["feature_name"]: item for item in results}

    assert by_name["feature_a"]["selection_status"] == "REVIEW"
    assert by_name["feature_a"]["correlated_feature"] == "feature_b"
    assert by_name["feature_b"]["selection_status"] == "REVIEW"


def test_correlated_feature_is_reviewed() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [0.0, 1.1, 2.2, 3.3],
        }
    )

    results = engine.evaluate(_with_signal_score(dataframe))

    by_name = {item["feature_name"]: item for item in results}

    assert by_name["feature_a"]["selection_status"] == "REVIEW"
    assert by_name["feature_b"]["selection_status"] == "REVIEW"


def test_healthy_feature_is_kept() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame(
        {
            "healthy_feature": [0.0, 1.0, 2.0, 3.0, 4.0],
            "other_feature": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    results = engine.evaluate(_with_signal_score(dataframe))

    by_name = {item["feature_name"]: item for item in results}

    assert by_name["healthy_feature"]["selection_status"] == "KEEP"
    assert by_name["healthy_feature"]["constant_feature"] == "NO"
    assert by_name["healthy_feature"]["missing_ratio"] == 0.0


def test_empty_dataframe_returns_no_results() -> None:
    engine = FeatureSelectionEngine()

    result = engine.evaluate(pd.DataFrame())

    assert result == []


def test_missing_signal_score_raises_contract_error() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0, 3.0]})

    with pytest.raises(FeatureSelectionContractError, match="signal_score"):
        engine.evaluate(dataframe)


def test_fundamental_correlated_feature_is_reviewed() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0, 3.0], "feature_b": [0.0, 1.1, 2.2, 3.3]})
    registry = FeatureRegistry()
    registry.register(
        ResearchFeature(
            name="feature_a",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="F-001",
            tier="FUNDAMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )
    registry.register(
        ResearchFeature(
            name="feature_b",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="F-002",
            tier="FUNDAMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )

    results = engine.evaluate(_with_signal_score(dataframe), feature_registry=registry)
    by_name = {item["feature_name"]: item for item in results}

    assert by_name["feature_a"]["selection_status"] == "REVIEW"
    assert by_name["feature_b"]["selection_status"] == "REVIEW"


def test_experimental_correlated_feature_is_dropped() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0, 3.0], "feature_b": [0.0, 1.1, 2.2, 3.3]})
    registry = FeatureRegistry()
    registry.register(
        ResearchFeature(
            name="feature_a",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="E-001",
            tier="EXPERIMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )
    registry.register(
        ResearchFeature(
            name="feature_b",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="E-002",
            tier="EXPERIMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )

    results = engine.evaluate(_with_signal_score(dataframe), feature_registry=registry)
    by_name = {item["feature_name"]: item for item in results}

    assert by_name["feature_a"]["selection_status"] == "DROP"
    assert by_name["feature_b"]["selection_status"] == "DROP"


def test_fundamental_constant_feature_is_dropped() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"constant_feature": [1.0, 1.0, 1.0, 1.0]})
    registry = FeatureRegistry()
    registry.register(
        ResearchFeature(
            name="constant_feature",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="C-001",
            tier="FUNDAMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )

    results = engine.evaluate(_with_signal_score(dataframe), feature_registry=registry)

    assert results[0]["selection_status"] == "DROP"
    assert results[0]["constant_feature"] == "YES"


def test_fundamental_high_missing_ratio_is_dropped() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"missing_feature": [1.0, 2.0, None, None, None]})
    registry = FeatureRegistry()
    registry.register(
        ResearchFeature(
            name="missing_feature",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="M-001",
            tier="FUNDAMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )

    results = engine.evaluate(_with_signal_score(dataframe), feature_registry=registry)

    assert results[0]["selection_status"] == "DROP"
    assert results[0]["missing_ratio"] > 0.25


def test_healthy_fundamental_feature_is_kept() -> None:
    engine = FeatureSelectionEngine()
    dataframe = pd.DataFrame({"healthy_feature": [0.0, 1.0, 2.0, 3.0, 4.0]})
    registry = FeatureRegistry()
    registry.register(
        ResearchFeature(
            name="healthy_feature",
            category="corner",
            version="1.0",
            description="test",
            available_before_kickoff=True,
            lookback_matches=1,
            dependencies=(),
            feature_id="H-001",
            tier="FUNDAMENTAL",
            status="VALIDATED",
            predictive_hypothesis="test",
            validation_status="VALIDATED",
            selection_status="KEEP",
        )
    )

    results = engine.evaluate(_with_signal_score(dataframe), feature_registry=registry)

    assert results[0]["selection_status"] == "KEEP"
    assert results[0]["constant_feature"] == "NO"
