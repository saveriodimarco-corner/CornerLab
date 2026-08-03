from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.betting_decision import run_betting_decision_layer
from src.data.odds_validator import validate_odds_dataframe


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
                "opening_odds": 2.10,
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


def test_betting_decision_requires_external_odds_and_returns_no_data_when_missing(tmp_path: Path) -> None:
    result = run_betting_decision_layer(base_dir=Path.cwd(), output_dir=tmp_path)
    decisions = result["decisions"]

    assert not decisions.empty
    assert decisions["recommendation"].eq("NO DATA").all()
    assert decisions["odds_available"].eq(False).all()
    assert decisions["implied_probability"].isna().all()
    assert decisions["ev"].isna().all()
    assert decisions["edge"].isna().all()


def test_external_closing_odds_are_used_for_ev_and_implied_probability(tmp_path: Path) -> None:
    external_odds = _valid_external_odds()
    result = run_betting_decision_layer(base_dir=Path.cwd(), output_dir=tmp_path, external_odds=external_odds)
    row = result["decisions"].iloc[0]

    assert row["odds_available"] is True
    assert row["odds_validated"] is True
    assert row["is_real_market_odds"] is True
    assert row["implied_probability"] == 1.0 / 2.50
    assert row["ev"] == row["model_probability"] * 2.50 - 1.0
    assert row["edge"] == row["model_probability"] - (1.0 / 2.50)


def test_betting_decision_outputs_are_deterministic(tmp_path: Path) -> None:
    external_odds = _valid_external_odds()
    first = run_betting_decision_layer(base_dir=Path.cwd(), output_dir=tmp_path, external_odds=external_odds)
    second = run_betting_decision_layer(base_dir=Path.cwd(), output_dir=tmp_path, external_odds=external_odds)

    assert first["decisions"].equals(second["decisions"])
    assert first["summary"] == second["summary"]


def test_validator_rejects_goal_odds_and_unmatched_fixtures() -> None:
    predictions = _load_predictions().head(1).copy()
    bad_rows = pd.DataFrame(
        [
            {
                "match_id": int(predictions.iloc[0]["match_id"]),
                "fixture_date": predictions.iloc[0]["date"],
                "home_team": predictions.iloc[0]["home_team"],
                "away_team": predictions.iloc[0]["away_team"],
                "bookmaker": "x",
                "market": "TOTAL_GOALS_OVER",
                "line": "2.5",
                "side": "OVER",
                "opening_odds": 2.10,
                "closing_odds": 2.20,
                "odds_timestamp": "2025-08-23T12:00:00",
                "source": "manual",
                "source_fixture_id": "fixture-1",
                "is_closing": True,
                "currency": "EUR",
                "import_timestamp": "2025-08-23T12:00:00",
            },
            {
                "match_id": 999999,
                "fixture_date": predictions.iloc[0]["date"],
                "home_team": "Nope",
                "away_team": "AlsoNope",
                "bookmaker": "x",
                "market": "TOTAL_CORNERS_OVER",
                "line": "8.5",
                "side": "OVER",
                "opening_odds": 2.10,
                "closing_odds": 2.20,
                "odds_timestamp": "2025-08-23T12:00:00",
                "source": "manual",
                "source_fixture_id": "fixture-2",
                "is_closing": True,
                "currency": "EUR",
                "import_timestamp": "2025-08-23T12:00:00",
            },
        ]
    )
    validated, errors = validate_odds_dataframe(bad_rows, fixtures=predictions)

    assert validated.empty
    assert len(errors) >= 2
