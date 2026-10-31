from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.operations import (
    premier_league_forward_notifier as notifier,
)


def _report_rows() -> list[dict]:
    probabilities = [
        0.75,
        0.74,
        0.73,
        0.72,
        0.71,
        0.70,
    ]
    rows = []

    for index, probability in enumerate(
        probabilities,
        start=1,
    ):
        rows.append(
            {
                "prediction_id": (
                    f"prediction-{index}"
                ),
                "prediction_timestamp": (
                    "2026-10-31T10:00:00Z"
                ),
                "selection_status": (
                    "OBSERVATION_ONLY"
                ),
                "fixture_id": index,
                "provider_fixture_id": (
                    f"event-{index}"
                ),
                "competition": (
                    "Premier League"
                ),
                "kickoff_utc": (
                    "2026-10-31T15:00:00Z"
                ),
                "home_team": (
                    f"Home {index}"
                ),
                "away_team": (
                    f"Away {index}"
                ),
                "line": "11.5",
                "side": "UNDER",
                "observed_odds": 1.50,
                "predicted_probability": (
                    probability
                ),
                "expected_value": (
                    probability * 1.50 - 1.0
                ),
                "data_quality_score": 1.0,
                "insufficient_history": False,
            }
        )

    rows.append(
        {
            **rows[0],
            "prediction_id": "low-quality",
            "provider_fixture_id": (
                "low-quality-event"
            ),
            "data_quality_score": 0.8,
        }
    )

    return rows


def test_preview_and_send_are_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    report = {
        "mode": "forward_validation_only",
        "competition": "Premier League",
        "eligible_lines": ["8.5", "11.5"],
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "production_database_writes": False,
        "rows": _report_rows(),
    }
    (
        reports_dir
        / "premier_league_forward_current.json"
    ).write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )

    now = datetime(
        2026,
        10,
        31,
        10,
        0,
        tzinfo=timezone.utc,
    )
    captured: list[dict] = []

    def sender(
        base_dir,
        row,
        request_sender=None,
    ):
        captured.append(row)
        return f"bet-{len(captured)}"

    preview = (
        notifier.notify_forward_price_checks(
            tmp_path,
            now=now,
            dry_run=True,
        )
    )

    assert preview["candidates"] == 5
    assert preview[
        "pending_notifications"
    ] == 5
    assert preview[
        "sent_notifications"
    ] == 0
    assert not notifier._state_path(
        tmp_path
    ).exists()

    first = notifier.notify_forward_price_checks(
        tmp_path,
        now=now,
        confirmation_sender=sender,
    )
    second = (
        notifier.notify_forward_price_checks(
            tmp_path,
            now=now,
            confirmation_sender=sender,
        )
    )

    assert first["sent_notifications"] == 5
    assert second["sent_notifications"] == 0
    assert len(captured) == 5
    assert {
        row["competition"]
        for row in captured
    } == {"Premier League"}
    assert {
        row["quality_tier"]
        for row in captured
    } == {"PREMIER FORWARD"}
    assert all(
        str(row["fixture_id"]).startswith(
            "epl:"
        )
        for row in captured
    )

    state = json.loads(
        notifier._state_path(
            tmp_path
        ).read_text(encoding="utf-8")
    )

    assert len(
        state["sent_prediction_ids"]
    ) == 5
    assert first[
        "telegram_simulation_only"
    ] is True
    assert first[
        "real_betting_enabled"
    ] is False
    assert first[
        "bankroll_real_enabled"
    ] is False
    assert not (
        tmp_path
        / "data"
        / "operations"
        / "real_bets.sqlite"
    ).exists()
