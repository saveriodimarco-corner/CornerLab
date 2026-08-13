from __future__ import annotations

import os

import pandas as pd

from src.ui.app import _apply_filters, _prepare_dashboard_table, _verify_login


def test_verify_login_uses_environment_password(monkeypatch) -> None:
    monkeypatch.setenv("CORNERLAB_APP_PASSWORD", "secret")
    assert _verify_login("secret") is True
    assert _verify_login("wrong") is False


def test_prepare_dashboard_table_computes_market_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Inter",
                "away_team": "Roma",
                "kickoff_utc": "2026-08-24T18:45:00Z",
                "line": "9.5",
                "side": "OVER",
                "bookmaker": "book-a",
                "closing_odds": 2.0,
                "predicted_probability": 0.6,
                "fair_odds": 1.6667,
                "ev": 0.2,
                "decision_confidence_score": 71.0,
                "recommended_stake": 5.0,
                "decision": "PLAY",
                "decision_reason": "POSITIVE_EV",
            }
        ]
    )

    table = _prepare_dashboard_table(frame)
    assert "market implied probability" in table.columns
    assert "edge" in table.columns
    assert float(table.iloc[0]["market implied probability"]) == 0.5


def test_apply_filters_supports_side_and_line_filters() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "side": "OVER", "line": "9.5"},
            {"decision": "NO BET", "side": "UNDER", "line": "10.5"},
        ]
    )

    play_only = _apply_filters(frame, "PLAY ONLY", "ALL", "ALL")
    assert len(play_only) == 1

    under_only = _apply_filters(frame, "ALL", "UNDER", "ALL")
    assert len(under_only) == 1

    line_only = _apply_filters(frame, "ALL", "ALL", "9.5")
    assert len(line_only) == 1
