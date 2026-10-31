from __future__ import annotations

import json
from pathlib import Path

from src.operations import daily_summary


def test_recovers_previous_unsent_matchday_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_date = "2026-10-29"
    fixtures = [
        {
            "fixture_id": 88,
            "competition": "Serie A",
            "kickoff_utc": "2026-10-29T19:45:00Z",
            "home_team": "Fiorentina",
            "away_team": "Atalanta BC",
            "status": "FT",
        }
    ]

    monkeypatch.setattr(
        daily_summary,
        "_serie_a_fixtures_for_rome_date",
        lambda base_dir, date: (
            fixtures if date == target_date else []
        ),
    )
    monkeypatch.setattr(
        daily_summary,
        "_daily_real_bets",
        lambda base_dir, fixture_ids: (
            [
                {
                    "status": "SETTLED_WIN",
                    "actual_stake": 10.0,
                    "profit_loss": 8.0,
                }
            ]
            if fixture_ids == ["88"]
            else []
        ),
    )
    monkeypatch.setattr(
        daily_summary,
        "_resolve_daily_fixture_statuses",
        lambda base_dir, rows: None,
    )

    class FakeRepository:
        def __init__(self, config) -> None:
            pass

        def get_result(self, fixture_id):
            return {
                "fixture_id": fixture_id,
                "total_corners": 10,
            }

    monkeypatch.setattr(
        daily_summary,
        "CollectorRepository",
        FakeRepository,
    )

    formatted_dates: list[str] = []
    monkeypatch.setattr(
        daily_summary,
        "format_daily_summary",
        lambda base_dir, date, rows: (
            formatted_dates.append(date)
            or f"summary:{date}"
        ),
    )

    messages: list[str] = []
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: (
            messages.append(message) or True
        ),
    )

    first = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-11-02T12:00:00Z",
    )
    second = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-11-02T13:00:00Z",
    )

    state = json.loads(
        (
            tmp_path
            / "data"
            / "operations"
            / "daily_summary_state.json"
        ).read_text(encoding="utf-8")
    )

    assert first == {
        "sent": True,
        "reason": "sent",
        "date": target_date,
    }
    assert second["sent"] is False
    assert second["reason"] == "no_pending_summary"
    assert formatted_dates == [target_date]
    assert messages == [f"summary:{target_date}"]
    assert state["sent_dates"] == [target_date]


def test_selects_oldest_pending_summary_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixtures_by_date = {
        "2026-10-29": [
            {
                "fixture_id": 88,
                "kickoff_utc": "2026-10-29T19:45:00Z",
                "status": "FT",
            }
        ],
        "2026-10-31": [
            {
                "fixture_id": 91,
                "kickoff_utc": "2026-10-31T19:45:00Z",
                "status": "FT",
            }
        ],
    }

    monkeypatch.setattr(
        daily_summary,
        "_serie_a_fixtures_for_rome_date",
        lambda base_dir, date: fixtures_by_date.get(
            date,
            [],
        ),
    )
    monkeypatch.setattr(
        daily_summary,
        "_daily_real_bets",
        lambda base_dir, fixture_ids: [
            {
                "status": "SETTLED_LOSS",
                "actual_stake": 10.0,
                "profit_loss": -10.0,
            }
        ],
    )
    monkeypatch.setattr(
        daily_summary,
        "_resolve_daily_fixture_statuses",
        lambda base_dir, rows: None,
    )

    class FakeRepository:
        def __init__(self, config) -> None:
            pass

        def get_result(self, fixture_id):
            return {"fixture_id": fixture_id}

    monkeypatch.setattr(
        daily_summary,
        "CollectorRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        daily_summary,
        "format_daily_summary",
        lambda base_dir, date, rows: date,
    )
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: True,
    )

    result = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-11-02T12:00:00Z",
    )

    assert result["sent"] is True
    assert result["date"] == "2026-10-29"
