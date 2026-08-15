from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.decision_engine import MAX_STAKE_FRACTION, build_decision_report, run_decision_engine


def test_decision_engine_builds_metrics_and_writes_reports(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "match_id": 1,
                "market": "Over 8.5",
                "closing_odds": 2.50,
                "predicted_probability": 0.60,
                "model_confidence": 0.80,
            },
            {
                "match_id": 2,
                "market": "Over 9.5",
                "closing_odds": 1.80,
                "predicted_probability": 0.55,
                "model_confidence": 0.40,
            },
            {
                "match_id": 3,
                "market": "Over 10.5",
                "closing_odds": 2.10,
                "predicted_probability": 0.30,
                "model_confidence": 0.90,
            },
        ]
    )

    result = run_decision_engine(predictions=predictions, output_dir=tmp_path, bankroll=100.0)
    report = result["report"]

    assert report is not None
    assert not report.empty
    assert set(["ev", "kelly_fraction", "half_kelly", "recommended_stake", "fair_odds", "market_edge", "confidence_score", "decision"]).issubset(report.columns)

    play_row = report.loc[report["match_id"] == 1].iloc[0]
    assert play_row["decision"] == "PLAY"
    assert play_row["ev"] > 0
    assert play_row["kelly_fraction"] > 0
    assert play_row["half_kelly"] > 0
    assert play_row["recommended_stake"] > 0

    no_bet_row = report.loc[report["match_id"] == 3].iloc[0]
    assert no_bet_row["decision"] == "NO BET"

    expected_files = [
        tmp_path / "data" / "research" / "decision_report.parquet",
        tmp_path / "data" / "research" / "decision_report.csv",
        tmp_path / "data" / "research" / "decision_report.json",
    ]
    for path in expected_files:
        assert path.exists()


@pytest.mark.parametrize(
    "bankroll,half_kelly,expected_stake",
    [
        (100.0, 0.02, 2.00),
        (100.0, 0.05, 5.00),
        (100.0, 0.08, 5.00),
        (80.0, 0.08, 4.00),
        (120.0, 0.08, 6.00),
        (100.0, 0.0, 0.0),
        (100.0, -0.05, 0.0),
        (94.20, 0.08, 4.71),
        (118.40, 0.08, 5.92),
    ],
)
def test_stake_cap_uses_current_bankroll_and_never_exceeds_5_percent(bankroll: float, half_kelly: float, expected_stake: float) -> None:
    kelly_fraction = max(0.0, half_kelly * 2.0)
    # Reverse-engineer a probability/odds pair that yields the target kelly_fraction at fixed odds=2.0.
    odds = 2.0
    predicted_probability = min(0.999, max(0.001, (kelly_fraction * (odds - 1.0) + (1.0 - 0.0)) / odds)) if kelly_fraction > 0 else 0.4
    predictions = pd.DataFrame([{"match_id": 1, "market": "Over 9.5", "closing_odds": odds, "predicted_probability": predicted_probability, "model_confidence": 0.90}])

    report = build_decision_report(predictions, bankroll=bankroll)
    row = report.iloc[0]

    assert row["half_kelly"] == pytest.approx(half_kelly, abs=1e-9) if half_kelly >= 0 else row["half_kelly"] == 0.0
    assert row["stake_cap_fraction"] == MAX_STAKE_FRACTION
    assert row["recommended_stake"] == pytest.approx(expected_stake, abs=1e-6)
    assert row["recommended_stake"] >= 0.0


def test_stake_cap_does_not_alter_ev_confidence_or_market_support() -> None:
    predictions = pd.DataFrame(
        [
            {"match_id": 1, "market": "Over 9.5", "closing_odds": 2.50, "predicted_probability": 0.60, "model_confidence": 0.80},
            {"match_id": 2, "market": "Over 10.5", "closing_odds": 1.80, "predicted_probability": 0.55, "model_confidence": 0.40},
        ]
    )

    uncapped_ev = predictions["predicted_probability"] * predictions["closing_odds"] - 1.0
    report = build_decision_report(predictions, bankroll=100.0)

    assert np.allclose(report["ev"].to_numpy(), uncapped_ev.to_numpy())
    assert report.loc[report["match_id"] == 1, "decision"].iloc[0] == "PLAY"
    assert report.loc[report["match_id"] == 2, "decision"].iloc[0] == "LOW CONFIDENCE"
    assert set(report["market"].unique()).issubset({"OVER 9.5", "OVER 10.5"})

    low_confidence_row = report.loc[report["match_id"] == 2].iloc[0]
    assert low_confidence_row["decision"] == "LOW CONFIDENCE"
    assert low_confidence_row["confidence_score"] < 60.0
