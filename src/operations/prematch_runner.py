from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from health_check import run_health_check
from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.fixture_collector import FixtureCollector
from src.collector.live_provider_adapter import LiveProviderAdapter
from src.collector.odds_collector import OddsCollector
from src.research.observation_freeze import build_production_baseline_manifest, resolve_current_bankroll, settle_paper_trades
from src.research.paper_trading import run_paper_trading


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ROME_TZ = ZoneInfo("Europe/Rome")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _latest_odds_snapshot(db_path: Path, fixture_id: int) -> datetime | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT MAX(snapshot_timestamp)
            FROM collector_odds_snapshots
            WHERE fixture_id = ?
              AND provider = 'the-odds-api'
            """,
            (fixture_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row or not row[0]:
        return None
    return _parse_timestamp(str(row[0]))


def _refresh_decision(
    kickoff_value: str | None,
    latest_snapshot: datetime | None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    kickoff = _parse_timestamp(kickoff_value)

    if kickoff is None:
        return False, "invalid_kickoff"

    if kickoff <= now:
        return False, "already_started"

    now_rome = now.astimezone(ROME_TZ)
    kickoff_rome = kickoff.astimezone(ROME_TZ)
    day_delta = (kickoff_rome.date() - now_rome.date()).days

    # Production horizon: today + tomorrow only.
    if day_delta not in {0, 1}:
        return False, "outside_today_tomorrow"

    # A fixture entering the operational horizon must get at least
    # one snapshot, regardless of distance from kickoff.
    if latest_snapshot is None:
        return True, "first_snapshot"

    hours_to_kickoff = (kickoff - now).total_seconds() / 3600.0
    age_hours = (now - latest_snapshot).total_seconds() / 3600.0

    if hours_to_kickoff > 24:
        ttl_hours = 12.0
    elif hours_to_kickoff > 6:
        ttl_hours = 6.0
    elif hours_to_kickoff > 2:
        ttl_hours = 2.0
    else:
        ttl_hours = 1.0

    if age_hours >= ttl_hours:
        return True, f"snapshot_due_{ttl_hours:g}h"

    return False, f"snapshot_fresh_{ttl_hours:g}h"


def _header_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def run_prematch(base_dir: Path | str | None = None, output_dir: Path | str | None = None, bankroll: float = 100.0) -> dict[str, Any]:
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir) if output_dir is not None else base_dir

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    health = run_health_check(base_dir=base_dir, output_dir=output_dir)
    baseline_manifest = build_production_baseline_manifest(base_dir=base_dir, output_dir=output_dir)
    # Use the current settled bankroll for staking; falls back to `bankroll` only when no settled history exists yet.
    current_bankroll = resolve_current_bankroll(base_dir=output_dir, default_bankroll=bankroll)

    config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    repo = CollectorRepository(config)
    fixture_collector = FixtureCollector(config, repo)
    odds_collector = OddsCollector(config, repo)
    live_adapter = LiveProviderAdapter(config)

    all_fixtures = live_adapter.fetch_fixtures()

    # Production betting scope is Serie A only.
    fixtures = [
        fixture
        for fixture in all_fixtures
        if str(fixture.get("competition") or "").strip().lower() == "serie a"
    ]

    fixture_writes = 0
    odds_writes = 0
    odds_downloaded = 0
    matched_fixture_count = 0
    odds_refresh_attempts = 0
    odds_refresh_skipped = 0
    skip_reasons: dict[str, int] = {}

    quota_used: int | None = None
    quota_remaining: int | None = None
    quota_check_error: str | None = None

    # /sports is a zero-credit endpoint and gives us the current
    # provider usage headers before any paid odds request.
    try:
        live_adapter.the_odds_api.list_sports()
        usage = dict(live_adapter.the_odds_api._usage or {})
        quota_used = _header_int(usage.get("x-requests-used"))
        quota_remaining = _header_int(usage.get("x-requests-remaining"))

        if quota_used is not None and quota_remaining is not None:
            repo.record_provider_usage(
                "the-odds-api",
                requests_used=quota_used,
                requests_remaining=quota_remaining,
                rate_limited=1 if quota_remaining <= 0 else 0,
            )
    except Exception as exc:
        quota_check_error = str(exc)

    for fixture in fixtures:
        saved = fixture_collector.collect_from_provider(fixture)
        fixture_writes += 1 if saved else 0

        if not saved:
            skip_reasons["fixture_not_saved"] = skip_reasons.get("fixture_not_saved", 0) + 1
            odds_refresh_skipped += 1
            continue

        fixture_id = int(saved.get("fixture_id", 0))
        provider_fixture_id = str(fixture.get("provider_fixture_id") or "")

        if not fixture_id or not provider_fixture_id:
            skip_reasons["missing_fixture_id"] = skip_reasons.get("missing_fixture_id", 0) + 1
            odds_refresh_skipped += 1
            continue

        latest_snapshot = _latest_odds_snapshot(config.db_path, fixture_id)
        should_refresh, refresh_reason = _refresh_decision(
            fixture.get("kickoff_utc"),
            latest_snapshot,
        )

        if not should_refresh:
            skip_reasons[refresh_reason] = skip_reasons.get(refresh_reason, 0) + 1
            odds_refresh_skipped += 1
            continue

        # Hard quota guard. Paid odds calls are allowed only when the
        # free quota check returned a known positive balance.
        if quota_remaining is None:
            skip_reasons["quota_unknown"] = skip_reasons.get("quota_unknown", 0) + 1
            odds_refresh_skipped += 1
            continue

        if quota_remaining <= 0:
            skip_reasons["quota_exhausted"] = skip_reasons.get("quota_exhausted", 0) + 1
            odds_refresh_skipped += 1
            continue

        odds_refresh_attempts += 1
        odds_rows = live_adapter.fetch_odds(provider_fixture_id)

        # Capture the provider's latest quota state after the request.
        usage = dict(live_adapter.the_odds_api._usage or {})
        current_used = _header_int(usage.get("x-requests-used"))
        current_remaining = _header_int(usage.get("x-requests-remaining"))

        if current_used is not None:
            quota_used = current_used
        if current_remaining is not None:
            quota_remaining = current_remaining

        resolution = live_adapter.last_odds_resolution.get(provider_fixture_id, {})
        if resolution.get("match_status") == "MATCHED":
            matched_fixture_count += 1

        odds_downloaded += len(odds_rows)

        kickoff = _parse_timestamp(fixture.get("kickoff_utc"))
        now = datetime.now(timezone.utc)
        minutes_to_kickoff = (
            max(0, int((kickoff - now).total_seconds() / 60))
            if kickoff is not None
            else 0
        )

        for row in odds_rows:
            payload = {
                "fixture_id": fixture_id,
                "bookmaker": row.get("bookmaker", "unknown"),
                "market": row.get("market", "UNKNOWN"),
                "line": row.get("line", ""),
                "side": row.get("side", ""),
                "decimal_odds": row.get("odd"),
                "snapshot_timestamp": config.now_utc(),
                "minutes_to_kickoff": minutes_to_kickoff,
                "provider": "the-odds-api",
                "provider_event_id": str(row.get("source_fixture_id") or provider_fixture_id),
                "raw_response_hash": "live_the_odds_api",
                "import_timestamp": config.now_utc(),
            }
            stored = odds_collector.collect_odds(payload)
            if stored is not None:
                odds_writes += 1

    # Persist the final provider state as well as the initial free quota
    # check. The repository reader returns the latest record, so operational
    # monitoring always sees the balance after this prematch run.
    if quota_used is not None and quota_remaining is not None:
        repo.record_provider_usage(
            "the-odds-api",
            requests_used=quota_used,
            requests_remaining=quota_remaining,
            rate_limited=1 if quota_remaining <= 0 else 0,
        )

    paper_trading_result = run_paper_trading(base_dir=base_dir, output_dir=output_dir, bankroll=current_bankroll)
    settlement_result = settle_paper_trades(base_dir=base_dir, output_dir=output_dir, bankroll_start=current_bankroll)
    completed_at = _utc_now()

    result: dict[str, Any] = {
        "run_type": "prematch",
        "started_at": started_at,
        "completed_at": completed_at,
        "health_ok": bool(health.get("ok", False)),
        "collector": {
            "fixtures_fetched": int(len(fixtures)),
            "fixtures_matched_to_odds_events": int(matched_fixture_count),
            "fixture_writes": int(fixture_writes),
            "odds_downloaded": int(odds_downloaded),
            "odds_writes": int(odds_writes),
            "provider": "the-odds-api",
            "provider_status": (
                "quota_exhausted"
                if quota_remaining is not None and quota_remaining <= 0
                else ("ok" if odds_downloaded > 0 else "warning")
            ),
            "odds_refresh_attempts": int(odds_refresh_attempts),
            "odds_refresh_skipped": int(odds_refresh_skipped),
            "odds_refresh_skip_reasons": skip_reasons,
            "quota_used": quota_used,
            "quota_remaining": quota_remaining,
            "quota_check_error": quota_check_error,
            "production_scope": "serie_a_today_tomorrow",
        },
        "paper_trading": paper_trading_result["summary"],
        "settlement": settlement_result.get("summary", {}),
        "performance": settlement_result.get("summary", {}),
        "checkpoint_reports": settlement_result.get("checkpoints", {}),
        "production_baseline": baseline_manifest,
        "output_paths": {
            "report_csv": str(paper_trading_result["output_paths"]["csv"]),
            "report_parquet": str(paper_trading_result["output_paths"]["parquet"]),
            "summary": str(paper_trading_result["output_paths"]["summary"]),
            "run_history": str(paper_trading_result["output_paths"]["history"]),
            "production_baseline": str(output_dir / "reports" / "production_baseline_serie_a.json"),
            "settled_report_csv": str(output_dir / "reports" / "paper_trading_settled.csv"),
            "settled_report_parquet": str(output_dir / "data" / "paper_trading" / "paper_trading_settled.parquet"),
            "performance_report": str(output_dir / "reports" / "paper_trading_performance.json"),
        },
    }

    (reports_dir / "prematch_latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
