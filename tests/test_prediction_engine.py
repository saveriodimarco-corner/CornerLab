from pathlib import Path

import pandas as pd
import pytest

from src.engine.backtest import Backtest
from src.engine.prediction_engine import PredictionEngine
from src.exceptions import InvalidFeatureDataError


@pytest.fixture
def merged_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "home_team": "Team A",
                "away_team": "Team B",
                "home_corners": 7,
                "away_corners": 4,
                "expected_home_corner": 6.2,
                "expected_away_corner": 4.6,
                "expected_total_corner": 10.8,
            },
            {
                "home_team": "Team C",
                "away_team": "Team A",
                "home_corners": 3,
                "away_corners": 8,
                "expected_home_corner": 4.1,
                "expected_away_corner": 5.3,
                "expected_total_corner": 9.4,
            },
        ]
    )


def test_predict_returns_probabilities_and_expected_values(merged_inputs: pd.DataFrame):
    engine = PredictionEngine()
    predictions = engine.predict(merged_inputs)

    expected_columns = {"expected_home_corner", "expected_away_corner", "expected_total_corner"}
    for threshold in [8.5, 9.5, 10.5, 11.5]:
        expected_columns.add(f"over_{int(threshold)}")
        expected_columns.add(f"under_{int(threshold)}")

    assert expected_columns.issubset(set(predictions.columns))
    assert {"match_id", "market", "closing_odds", "predicted_probability", "model_confidence"}.issubset(set(predictions.columns))
    assert len(predictions) == len(merged_inputs)
    assert predictions["over_8"].between(0.0, 1.0).all()


def test_invalid_required_feature_raises_domain_error(merged_inputs: pd.DataFrame):
    merged_inputs.loc[0, "expected_total_corner"] = "bad"

    with pytest.raises(InvalidFeatureDataError, match="must be numeric"):
        PredictionEngine().predict(merged_inputs)


def test_backtest_summarizes_metrics(merged_inputs: pd.DataFrame):
    engine = PredictionEngine()
    predictions = engine.predict(merged_inputs)
    evaluator = Backtest()
    scored = evaluator.evaluate(predictions)
    summary = evaluator.summarize(predictions)

    assert not scored.empty
    assert {"accuracy", "brier_score", "log_loss", "calibration_error"}.issubset(set(summary.columns))
    assert summary.loc[0, "accuracy"] >= 0.0


def test_build_writes_predictions_parquet(tmp_path: Path, merged_inputs: pd.DataFrame):
    ratings_path = tmp_path / "ratings.parquet"
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "predictions.parquet"

    pd.DataFrame([{"team": "Team A", "home_attack_rating": 0.0, "away_attack_rating": 0.0, "home_defence_rating": 0.0, "away_defence_rating": 0.0, "home_corner_advantage": 0.0, "away_corner_penalty": 0.0, "overall_attack": 0.0, "overall_defence": 0.0, "tempo_index": 0.0, "corner_difference": 0.0, "corner_balance": 0.0, "opponent_strength_adjustment": 0.0, "confidence": 0.0, "standard_deviation": 0.0, "consistency_index": 0.0}]).to_parquet(ratings_path, index=False)
    pd.DataFrame([{"home_team": "Team A", "away_team": "Team B", "home_corners": 7, "away_corners": 4, "expected_home_corner": 6.2, "expected_away_corner": 4.6, "expected_total_corner": 10.8}]).to_parquet(features_path, index=False)

    engine = PredictionEngine()
    predictions = engine.build(ratings_path, features_path, output_path)
    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert reloaded.equals(predictions)
