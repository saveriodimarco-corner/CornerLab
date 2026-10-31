from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.operations import premier_league_non_play_notifier as notifier


def fixture() -> dict:
    return {
        "provider_fixture_id": "event-1",
        "kickoff_utc": "2026-11-01T14:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    }


def candidate() -> dict:
    return {
        **fixture(),
        "line": "11.5",
        "side": "UNDER",
    }


def configure(
    monkeypatch,
    *,
    candidates,
    statuses,
) -> None:
    monkeypatch.setattr(
        notifier,
        "_future_matchdays",
        lambda base_dir, now: {
            "2026-11-01": [fixture()]
        },
    )
    monkeypatch.setattr(
        notifier,
        "_read_report",
        lambda base_dir: {"rows": []},
    )
    monkeypatch.setattr(
        notifier,
        "select_forward_price_checks",
        lambda report, now, maximum: candidates,
    )
    monkeypatch.setattr(
        notifier,
        "_simulation_statuses",
        lambda base_dir: statuses,
    )


def test_sends_non_play_once_when_no_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure(
        monkeypatch,
        candidates=[],
        statuses={},
    )
    messages: list[str] = []
    now = datetime(
        2026,
        11,
        1,
        9,
        0,
        tzinfo=timezone.utc,
    )

    first = notifier.notify_premier_non_play(
        tmp_path,
        now=now,
        message_sender=lambda message: (
            messages.append(message) or True
        ),
    )
    second = notifier.notify_premier_non_play(
        tmp_path,
        now=now,
        message_sender=lambda message: (
            messages.append(message) or True
        ),
    )

    assert first["sent_dates"] == [
        "2026-11-01"
    ]
    assert second["sent_dates"] == []
    assert len(messages) == 1
    assert "NON GIOCARE" in messages[0]
    assert "nessuna scommessa reale" in messages[0]
    assert not (
        tmp_path
        / "data"
        / "operations"
        / "real_bets.sqlite"
    ).exists()


def test_waits_for_reply_then_sends_when_all_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    key = (
        "epl:event-1",
        "11.5",
        "UNDER",
    )
    configure(
        monkeypatch,
        candidates=[candidate()],
        statuses={key: ["SUGGESTED"]},
    )
    messages: list[str] = []
    now = datetime(
        2026,
        11,
        1,
        9,
        0,
        tzinfo=timezone.utc,
    )

    waiting = notifier.notify_premier_non_play(
        tmp_path,
        now=now,
        message_sender=lambda message: (
            messages.append(message) or True
        ),
    )
    assert waiting["sent_dates"] == []

    monkeypatch.setattr(
        notifier,
        "_simulation_statuses",
        lambda base_dir: {
            key: ["SKIPPED"]
        },
    )
    sent = notifier.notify_premier_non_play(
        tmp_path,
        now=now,
        message_sender=lambda message: (
            messages.append(message) or True
        ),
    )

    assert sent["sent_dates"] == [
        "2026-11-01"
    ]
    assert len(messages) == 1
    assert "tutti i controlli quota" in messages[0]


def test_never_sends_non_play_if_a_bet_was_placed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    key = (
        "epl:event-1",
        "11.5",
        "UNDER",
    )
    configure(
        monkeypatch,
        candidates=[candidate()],
        statuses={key: ["BET_PLACED"]},
    )

    result = notifier.notify_premier_non_play(
        tmp_path,
        now=datetime(
            2026,
            11,
            1,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        message_sender=lambda message: True,
    )

    assert result["sent_dates"] == []
    assert result["real_betting_enabled"] is False
