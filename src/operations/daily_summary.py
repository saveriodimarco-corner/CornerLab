from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.result_resolver import ResultResolver, TERMINAL_FIXTURE_STATUSES
from src.operations.real_bet_ledger import get_bankroll_snapshot, list_open_bets
from src.operations.telegram_notifier import send_message


ROME = ZoneInfo("Europe/Rome")
SETTLED_STATUSES = {"SETTLED_WIN", "SETTLED_LOSS", "SETTLED_VOID"}


def _state_path(base_dir: Path) -> Path:
    return base_dir / "data" / "operations" / "daily_summary_state.json"


def _read_state(base_dir: Path) -> dict[str, Any]:
    path = _state_path(base_dir)
    if not path.exists():
        return {"sent_dates": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"sent_dates": []}
    if not isinstance(payload.get("sent_dates"), list):
        payload["sent_dates"] = []
    return payload


def _write_state(base_dir: Path, payload: dict[str, Any]) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serie_a_fixtures_for_rome_date(
    base_dir: Path,
    target_date: str,
) -> list[dict[str, Any]]:
    db_path = base_dir / "data" / "collector.sqlite"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM collector_fixtures
            WHERE competition = 'Serie A'
            ORDER BY kickoff_utc, fixture_id
            """
        ).fetchall()
    finally:
        conn.close()

    selected: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        kickoff = _parse_utc(item.get("kickoff_utc"))
        if kickoff is None:
            continue
        if kickoff.astimezone(ROME).date().isoformat() == target_date:
            selected.append(item)
    return selected


def _resolve_daily_fixture_statuses(
    base_dir: Path,
    fixtures: list[dict[str, Any]],
) -> None:
    config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    repo = CollectorRepository(config)
    resolver = ResultResolver(config, repo)

    for fixture in fixtures:
        status = str(fixture.get("status") or "").upper()
        if status in TERMINAL_FIXTURE_STATUSES and repo.get_result(
            fixture["fixture_id"]
        ) is not None:
            continue
        resolver.resolve_fixture(fixture["fixture_id"])


def _daily_real_bets(
    base_dir: Path,
    fixture_ids: list[str],
) -> list[dict[str, Any]]:
    db_path = base_dir / "data" / "operations" / "real_bets.sqlite"
    if not db_path.exists() or not fixture_ids:
        return []

    placeholders = ",".join("?" for _ in fixture_ids)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM real_bets
            WHERE fixture_id IN ({placeholders})
              AND status IN ('SETTLED_WIN', 'SETTLED_LOSS', 'SETTLED_VOID')
            ORDER BY confirmed_timestamp, created_at
            """,
            tuple(fixture_ids),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _general_real_stats(base_dir: Path) -> dict[str, Any]:
    db_path = base_dir / "data" / "operations" / "real_bets.sqlite"
    if not db_path.exists():
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "voids": 0,
            "stake": 0.0,
            "pnl": 0.0,
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'SETTLED_WIN' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN status = 'SETTLED_LOSS' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN status = 'SETTLED_VOID' THEN 1 ELSE 0 END) AS voids,
                COALESCE(SUM(actual_stake), 0.0) AS stake,
                COALESCE(SUM(profit_loss), 0.0) AS pnl
            FROM real_bets
            WHERE status IN ('SETTLED_WIN', 'SETTLED_LOSS', 'SETTLED_VOID')
            """
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def _aggregate_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for bet in bets if bet.get("status") == "SETTLED_WIN")
    losses = sum(1 for bet in bets if bet.get("status") == "SETTLED_LOSS")
    voids = sum(1 for bet in bets if bet.get("status") == "SETTLED_VOID")
    stake = sum(float(bet.get("actual_stake") or 0.0) for bet in bets)
    pnl = sum(float(bet.get("profit_loss") or 0.0) for bet in bets)
    roi = pnl / stake if stake > 0 else 0.0
    decided = wins + losses
    win_rate = wins / decided if decided > 0 else 0.0

    return {
        "total": len(bets),
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "stake": stake,
        "pnl": pnl,
        "roi": roi,
        "win_rate": win_rate,
    }


def _fixture_result_lines(
    base_dir: Path,
    fixtures: list[dict[str, Any]],
) -> list[str]:
    repo = CollectorRepository(
        CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    )
    lines: list[str] = []

    for fixture in fixtures:
        result = repo.get_result(fixture["fixture_id"])
        if result is None:
            continue
        lines.append(
            "⚽ "
            f"{fixture.get('home_team', '?')} "
            f"{int(result.get('home_score') or 0)}-"
            f"{int(result.get('away_score') or 0)} "
            f"{fixture.get('away_team', '?')} "
            f"• corner {int(result.get('total_corners') or 0)}"
        )
    return lines


def format_daily_summary(
    base_dir: Path,
    target_date: str,
    fixtures: list[dict[str, Any]],
) -> str:
    fixture_ids = [str(fixture["fixture_id"]) for fixture in fixtures]
    daily_bets = _daily_real_bets(base_dir, fixture_ids)
    daily = _aggregate_bets(daily_bets)

    general_raw = _general_real_stats(base_dir)
    general_stake = float(general_raw.get("stake") or 0.0)
    general_pnl = float(general_raw.get("pnl") or 0.0)
    general_wins = int(general_raw.get("wins") or 0)
    general_losses = int(general_raw.get("losses") or 0)
    general_decided = general_wins + general_losses
    general_roi = general_pnl / general_stake if general_stake > 0 else 0.0
    general_win_rate = (
        general_wins / general_decided if general_decided > 0 else 0.0
    )

    bankroll = get_bankroll_snapshot(base_dir)
    open_bets = list_open_bets(base_dir)

    lines = [
        "📊 CORNERLAB — RIEPILOGO GIORNALIERO",
        f"📅 {target_date}",
        "",
        "⚽ PARTITE DELLA GIORNATA",
    ]

    result_lines = _fixture_result_lines(base_dir, fixtures)
    lines.extend(result_lines or ["Nessun risultato disponibile."])

    lines += [
        "",
        "🎯 GIOCATE REALI DELLA GIORNATA",
        f"Chiuse: {daily['total']}",
        (
            f"WIN {daily['wins']} • LOSS {daily['losses']} "
            f"• VOID {daily['voids']}"
        ),
        f"Stake: €{daily['stake']:.2f}",
        f"P/L: €{daily['pnl']:+.2f}",
        f"ROI: {daily['roi']:.1%}",
        "",
        "📈 STATISTICHE GENERALI",
        f"Giocate chiuse: {int(general_raw.get('total') or 0)}",
        (
            f"WIN {general_wins} • LOSS {general_losses} "
            f"• VOID {int(general_raw.get('voids') or 0)}"
        ),
        f"Win rate: {general_win_rate:.1%}",
        f"Stake complessivo: €{general_stake:.2f}",
        f"P/L realizzato: €{general_pnl:+.2f}",
        f"ROI complessivo: {general_roi:.1%}",
        "",
        "💼 BANKROLL",
        f"Disponibile: €{bankroll['available_bankroll']:.2f}",
        f"Esposto: €{bankroll['open_exposure']:.2f}",
        f"Totale: €{bankroll['total_bankroll']:.2f}",
        f"Giocate ancora aperte: {len(open_bets)}",
    ]

    return "\n".join(lines)


def maybe_send_daily_summary(
    base_dir: Path | str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)

    now = _parse_utc(completed_at) if completed_at else datetime.now(timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)

    target_date = now.astimezone(ROME).date().isoformat()

    state = _read_state(base_dir)
    if target_date in set(str(item) for item in state.get("sent_dates", [])):
        return {
            "sent": False,
            "reason": "already_sent",
            "date": target_date,
        }

    fixtures = _serie_a_fixtures_for_rome_date(base_dir, target_date)
    if not fixtures:
        return {
            "sent": False,
            "reason": "no_fixtures",
            "date": target_date,
        }

    kickoffs = [
        kickoff
        for kickoff in (_parse_utc(item.get("kickoff_utc")) for item in fixtures)
        if kickoff is not None
    ]
    if not kickoffs:
        return {
            "sent": False,
            "reason": "kickoff_missing",
            "date": target_date,
        }

    # Do not start provider checks before a normal Serie A match could
    # reasonably have finished. The hourly settlement will retry afterwards.
    earliest_check = max(kickoffs) + timedelta(minutes=105)
    if now < earliest_check:
        return {
            "sent": False,
            "reason": "last_match_not_due",
            "date": target_date,
        }

    try:
        _resolve_daily_fixture_statuses(base_dir, fixtures)
    except Exception as exc:
        return {
            "sent": False,
            "reason": "fixture_resolution_failed",
            "date": target_date,
            "error": str(exc),
        }

    fixtures = _serie_a_fixtures_for_rome_date(base_dir, target_date)
    non_terminal = [
        fixture
        for fixture in fixtures
        if str(fixture.get("status") or "").upper()
        not in TERMINAL_FIXTURE_STATUSES
    ]
    if non_terminal:
        return {
            "sent": False,
            "reason": "fixtures_not_terminal",
            "date": target_date,
            "pending": [str(item["fixture_id"]) for item in non_terminal],
        }

    repo = CollectorRepository(
        CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    )
    missing_results = [
        str(fixture["fixture_id"])
        for fixture in fixtures
        if repo.get_result(fixture["fixture_id"]) is None
    ]
    if missing_results:
        return {
            "sent": False,
            "reason": "results_incomplete",
            "date": target_date,
            "pending": missing_results,
        }

    message = format_daily_summary(base_dir, target_date, fixtures)

    if not send_message(message):
        return {
            "sent": False,
            "reason": "telegram_send_failed",
            "date": target_date,
        }

    sent_dates = [str(item) for item in state.get("sent_dates", [])]
    sent_dates.append(target_date)
    state["sent_dates"] = sorted(set(sent_dates))
    state["last_sent_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(base_dir, state)

    return {
        "sent": True,
        "reason": "sent",
        "date": target_date,
    }
