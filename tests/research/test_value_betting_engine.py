from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.value_decision import run_value_betting_engine


def _load_predictions() -> pd.DataFrame:
    return pd.read_parquet(Path.cwd() / "data" / "research" / "confidence_predictions.parquet")


def _valid_external_odds() -> pd.DataFrame:
    predictions = _load_predictions().head(1).copy()
    return pd.DataFrame(
        [
            {
                "match_id": int(predictions.iloc[0]["match_id"]),
                "fixture_date": predictions.iloc[0]["date"],
                "home_team": predictions.iloc[0]["home_team"],
                "away_team": predictions.iloc[0]["away_team"],
                "bookmaker": "test_bookmaker",
                "market": "TOTAL_CORNERS_OVER",
                "line": "8.5",
                "side": "OVER",
                "opening_odds": 1.90,
                "closing_odds": 2.50,
                "odds_timestamp": "2025-08-23T12:00:00",
                "source": "manual",
                "source_fixture_id": "fixture-1",
                "is_closing": True,
                "currency": "EUR",
                "import_timestamp": "2025-08-23T12:00:00",
            }
        ]
    )


def test_value_betting_engine_writes_reports_and_decisions(tmp_path: Path) -> None:
    result = run_value_betting_engine(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
        external_odds=_valid_external_odds(),
        bankroll_start=100.0,
    )

    assert not result["decisions"].empty
    assert {"decision", "ev", "kelly_fraction_full", "stake", "bankroll"}.issubset(result["decisions"].columns)
    assert (tmp_path / "reports" / "value_bets.csv").exists()
    assert (tmp_path / "reports" / "value_summary.md").exists()
    assert (tmp_path / "reports" / "bankroll_curve.csv").exists()


def test_value_betting_engine_uses_positive_ev_and_kelly(tmp_path: Path) -> None:
    result = run_value_betting_engine(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
        external_odds=_valid_external_odds(),
        bankroll_start=100.0,
    )

    decisions = result["decisions"]
    assert decisions["decision"].isin(["BET", "WATCH", "NO BET"]).all()
    assert decisions["ev"].ge(0.0).any()
    assert decisions["kelly_fraction_full"].ge(0.0).any()


def test_bankroll_tracker_exposes_runtime_metrics(tmp_path: Path) -> None:
    result = run_value_betting_engine(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
        external_odds=_valid_external_odds(),
        bankroll_start=100.0,
    )

    summary = result["summary"]
    assert summary["bankroll_start"] == 100.0
    assert summary["final_bankroll"] >= 0.0
    assert summary["max_drawdown"] >= 0.0
