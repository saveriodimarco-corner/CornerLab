from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.operations.premier_league_forward_notifier import (
    _read_report,
    select_forward_price_checks,
)
from src.operations.telegram_notifier import send_message


COMPETITION = "Premier League"
STATE_NAME = "forward_non_play_state.json"
ROME_TIMEZONE = "Europe/Rome"
TERMINAL_PLAYED_STATUSES = {
    "BET_PLACED",
    "SETTLED_WIN",
    "SETTLED_LOSS",
    "SETTLED_VOID",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path(base_dir: Path) -> Path:
    return (
        base_dir
        / "data"
        / "premier_league_shadow"
        / STATE_NAME
    )


def _read_state(base_dir: Path) -> dict[str, Any]:
    path = _state_path(base_dir)
    if not path.exists():
        return {
            "sent_dates": [],
            "last_sent_at": None,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    sent_dates = payload.get("sent_dates", [])
    if not isinstance(sent_dates, list):
        raise ValueError("Stato NON GIOCARE Premier non valido")

    return {
        "sent_dates": [str(item) for item in sent_dates],
        "last_sent_at": payload.get("last_sent_at"),
    }


def _write_state(
    base_dir: Path,
    state: dict[str, Any],
) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _future_matchdays(
    base_dir: Path,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    database = (
        base_dir
        / "data"
        / "premier_league_shadow"
        / "collector.sqlite"
    )
    if not database.exists():
        return {}

    try:
        with sqlite3.connect(database) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    provider_fixture_id,
                    kickoff_utc,
                    home_team,
                    away_team
                FROM collector_fixtures
                WHERE competition = ?
                ORDER BY datetime(kickoff_utc)
                """,
                (COMPETITION,),
            ).fetchall()
    except sqlite3.Error:
        return {}

    now_timestamp = pd.Timestamp(now)
    if now_timestamp.tzinfo is None:
        now_timestamp = now_timestamp.tz_localize("UTC")
    else:
        now_timestamp = now_timestamp.tz_convert("UTC")

    today = now_timestamp.tz_convert(
        ROME_TIMEZONE
    ).date()
    tomorrow = today + timedelta(days=1)
    result: dict[str, list[dict[str, Any]]] = {}

    for raw in rows:
        kickoff = pd.to_datetime(
            raw["kickoff_utc"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(kickoff) or kickoff <= now_timestamp:
            continue

        match_date = kickoff.tz_convert(
            ROME_TIMEZONE
        ).date()
        if match_date not in {today, tomorrow}:
            continue

        key = match_date.isoformat()
        result.setdefault(key, []).append(
            {
                "provider_fixture_id": str(
                    raw["provider_fixture_id"]
                ),
                "kickoff_utc": kickoff.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "home_team": str(raw["home_team"]),
                "away_team": str(raw["away_team"]),
            }
        )

    return result


def _line_key(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value).strip()


def _candidate_key(
    row: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        f"epl:{row.get('provider_fixture_id')}",
        _line_key(row.get("line")),
        str(row.get("side", "")).upper(),
    )


def _simulation_statuses(
    base_dir: Path,
) -> dict[tuple[str, str, str], list[str]]:
    database = (
        base_dir
        / "data"
        / "telegram_simulation"
        / "data"
        / "operations"
        / "real_bets.sqlite"
    )
    if not database.exists():
        return {}

    try:
        with sqlite3.connect(database) as conn:
            rows = conn.execute(
                """
                SELECT fixture_id, line, side, status
                FROM real_bets
                WHERE competition = ?
                """,
                (COMPETITION,),
            ).fetchall()
    except sqlite3.Error:
        return {}

    statuses: dict[
        tuple[str, str, str],
        list[str],
    ] = {}
    for fixture_id, line, side, status in rows:
        key = (
            str(fixture_id),
            _line_key(line),
            str(side).upper(),
        )
        statuses.setdefault(key, []).append(
            str(status).upper()
        )
    return statuses


def _candidate_matchday(
    row: dict[str, Any],
) -> str | None:
    kickoff = pd.to_datetime(
        row.get("kickoff_utc"),
        utc=True,
        errors="coerce",
    )
    if pd.isna(kickoff):
        return None
    return kickoff.tz_convert(
        ROME_TIMEZONE
    ).date().isoformat()


def _format_non_play_message(
    match_date: str,
    fixtures: list[dict[str, Any]],
    reason: str,
) -> str:
    date_value = datetime.fromisoformat(
        match_date
    ).strftime("%d/%m/%Y")
    lines = [
        "\U0001F6AB CORNERLAB PREMIER - NON GIOCARE",
        "",
        f"Giornata: {date_value}",
        "",
    ]

    for fixture in fixtures:
        kickoff = pd.to_datetime(
            fixture["kickoff_utc"],
            utc=True,
        ).tz_convert(ROME_TIMEZONE)
        lines.append(
            f"{kickoff.strftime('%H:%M')} - "
            f"{fixture['home_team']} vs "
            f"{fixture['away_team']}"
        )

    lines.extend(
        [
            "",
            f"Motivo: {reason}",
            "Esito: NON GIOCARE",
            "",
            (
                "Modalita: validazione forward; "
                "nessuna scommessa reale."
            ),
        ]
    )
    return "\n".join(lines)


def notify_premier_non_play(
    base_dir: Path | str,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    message_sender: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    matchdays = _future_matchdays(base_dir, now)
    report = _read_report(base_dir)
    candidates = select_forward_price_checks(
        report,
        now=now,
        maximum=10000,
    )
    candidates_by_date: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    for row in candidates:
        match_date = _candidate_matchday(row)
        if match_date is not None:
            candidates_by_date.setdefault(
                match_date,
                [],
            ).append(row)

    statuses = _simulation_statuses(base_dir)
    state = _read_state(base_dir)
    sent_dates = set(state["sent_dates"])
    pending: list[dict[str, Any]] = []

    for match_date, fixtures in sorted(
        matchdays.items()
    ):
        if match_date in sent_dates:
            continue

        kickoffs = [
            pd.to_datetime(
                fixture["kickoff_utc"],
                utc=True,
            )
            for fixture in fixtures
        ]
        earliest = min(kickoffs)
        if pd.Timestamp(now) < (
            earliest - pd.Timedelta(hours=6)
        ):
            continue

        date_candidates = candidates_by_date.get(
            match_date,
            [],
        )
        reason: str | None = None

        if not date_candidates:
            reason = (
                "nessuna giocata soddisfa i "
                "requisiti minimi CornerLab"
            )
        else:
            observed_statuses: list[str] = []
            for row in date_candidates:
                observed_statuses.extend(
                    statuses.get(
                        _candidate_key(row),
                        [],
                    )
                )

            if any(
                status in TERMINAL_PLAYED_STATUSES
                for status in observed_statuses
            ):
                continue

            if (
                observed_statuses
                and all(
                    status == "SKIPPED"
                    for status in observed_statuses
                )
            ):
                reason = (
                    "tutti i controlli quota disponibili "
                    "sono stati segnati come non giocati"
                )
            else:
                continue

        pending.append(
            {
                "date": match_date,
                "fixtures": fixtures,
                "reason": reason,
                "message": _format_non_play_message(
                    match_date,
                    fixtures,
                    reason,
                ),
            }
        )

    if dry_run:
        return {
            "mode": "non_play_preview",
            "pending_dates": [
                item["date"]
                for item in pending
            ],
            "sent_dates": [],
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    sender = message_sender or send_message
    sent_now: list[str] = []

    for item in pending:
        if sender(item["message"]):
            sent_now.append(item["date"])

    if sent_now:
        state["sent_dates"] = list(
            dict.fromkeys(
                state["sent_dates"] + sent_now
            )
        )
        state["last_sent_at"] = now.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        _write_state(base_dir, state)

    return {
        "mode": "telegram_non_play",
        "pending_dates": [
            item["date"]
            for item in pending
        ],
        "sent_dates": sent_now,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }
