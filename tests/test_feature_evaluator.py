import pytest

from src.research.feature_evaluator import FeatureEvaluator


@pytest.fixture
def evaluator() -> FeatureEvaluator:
    return FeatureEvaluator()


def test_evaluator_reports_expected_stats_and_missing_ratio(evaluator: FeatureEvaluator) -> None:
    rows = [
        {"fixture_id": 1, "feature_name": "corner_creation_rate", "feature_value": 3.0},
        {"fixture_id": 2, "feature_name": "corner_creation_rate", "feature_value": 4.0},
        {"fixture_id": 3, "feature_name": "corner_creation_rate", "feature_value": None},
        {"fixture_id": 1, "feature_name": "corner_concession_rate", "feature_value": 1.0},
        {"fixture_id": 2, "feature_name": "corner_concession_rate", "feature_value": 2.0},
        {"fixture_id": 3, "feature_name": "corner_concession_rate", "feature_value": 3.0},
        {"fixture_id": 1, "feature_name": "recent_corner_form", "feature_value": 1.0},
        {"fixture_id": 2, "feature_name": "recent_corner_form", "feature_value": 2.0},
        {"fixture_id": 3, "feature_name": "recent_corner_form", "feature_value": 3.0},
        {"fixture_id": 1, "feature_name": "corner_diff_pressure", "feature_value": 0.5},
        {"fixture_id": 2, "feature_name": "corner_diff_pressure", "feature_value": 1.5},
        {"fixture_id": 3, "feature_name": "corner_diff_pressure", "feature_value": 2.5},
    ]

    result = evaluator.evaluate(rows)

    assert result["row_count"] == 12
    assert result["missing_ratio"]["corner_creation_rate"] == pytest.approx(1 / 3)
    assert result["variance"]["corner_concession_rate"] == pytest.approx(2.0 / 3)
    assert result["minimum"]["recent_corner_form"] == 1.0
    assert result["maximum"]["recent_corner_form"] == 3.0
    assert result["mean"]["recent_corner_form"] == pytest.approx(2.0)
    assert result["standard_deviation"]["recent_corner_form"] == pytest.approx(0.8164965809)


def test_constant_feature_is_detected_and_empty_dataset_is_handled(evaluator: FeatureEvaluator) -> None:
    constant_rows = [
        {"fixture_id": 1, "feature_name": "corner_creation_rate", "feature_value": 2.0},
        {"fixture_id": 2, "feature_name": "corner_creation_rate", "feature_value": 2.0},
        {"fixture_id": 3, "feature_name": "corner_creation_rate", "feature_value": 2.0},
    ]

    constant_result = evaluator.evaluate(constant_rows)
    empty_result = evaluator.evaluate([])

    assert "corner_creation_rate" in constant_result["constant_features"]
    assert empty_result["row_count"] == 0
    assert empty_result["correlation_matrix"] == {}
    assert empty_result["missing_ratio"] == {}


def test_correlation_matrix_shape_is_correct(evaluator: FeatureEvaluator) -> None:
    rows = [
        {"fixture_id": 1, "feature_name": "corner_creation_rate", "feature_value": 1.0},
        {"fixture_id": 2, "feature_name": "corner_creation_rate", "feature_value": 2.0},
        {"fixture_id": 3, "feature_name": "corner_creation_rate", "feature_value": 3.0},
        {"fixture_id": 1, "feature_name": "corner_concession_rate", "feature_value": 3.0},
        {"fixture_id": 2, "feature_name": "corner_concession_rate", "feature_value": 2.0},
        {"fixture_id": 3, "feature_name": "corner_concession_rate", "feature_value": 1.0},
        {"fixture_id": 1, "feature_name": "recent_corner_form", "feature_value": 2.0},
        {"fixture_id": 2, "feature_name": "recent_corner_form", "feature_value": 4.0},
        {"fixture_id": 3, "feature_name": "recent_corner_form", "feature_value": 6.0},
        {"fixture_id": 1, "feature_name": "corner_diff_pressure", "feature_value": 0.0},
        {"fixture_id": 2, "feature_name": "corner_diff_pressure", "feature_value": 0.0},
        {"fixture_id": 3, "feature_name": "corner_diff_pressure", "feature_value": 0.0},
    ]

    result = evaluator.evaluate(rows)
    assert set(result["correlation_matrix"].keys()) == {
        "corner_creation_rate",
        "corner_concession_rate",
        "recent_corner_form",
        "corner_diff_pressure",
    }
    assert result["correlation_matrix"]["corner_creation_rate"]["corner_concession_rate"] == pytest.approx(-1.0)
    assert result["correlation_matrix"]["corner_diff_pressure"]["corner_creation_rate"] == pytest.approx(0.0)
