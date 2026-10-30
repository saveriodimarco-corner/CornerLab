from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.fixture_collector import FixtureCollector
from src.collector.live_provider_adapter import LiveProviderAdapter
from src.collector.odds_collector import OddsCollector
from src.operations.prematch_runner import (
    _header_int,
    _latest_odds_snapshot,
    _parse_timestamp,
    _refresh_decision,
    _utc_now,
)


COMPETITION = "Premier League"
SPORT_KEY = "soccer_epl"
SHADOW_DB_RELATIVE_PATH = Path(
    "data/premier_league_shadow/collector.sqlite"
)
SHADOW_REPORT_NAME = "premier_league_shadow_latest.json"


def _fixture_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        provider_fixture_id = str(event.get("id") or "").strip()
        kickoff_utc = str(event.get("commence_time") or "").strip()
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()

        if not all(
            [provider_fixture_id, kickoff_utc, home_team, away_team]
        ):
            continue

        rows.append(
            {
                "provider_fixture_id": provider_fixture_id,
                "competition": COMPETITION,
                "season": str(
                    datetime.now(timezone.utc).year
                ),
                "kickoff_utc": kickoff_utc,
                "home_team": home_team,
                "away_team": away_team,
                "status": "NS",
                "provider": "the-odds-api",
            }
        )

    return rows


def _deduplicate_odds(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    selected: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    duplicates_removed = 0
    conflicting_duplicates = 0

    for raw_row in rows:
        row = dict(raw_row)

        try:
            decimal_odds = float(row.get("odd"))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
            continue

        row["odd"] = decimal_odds
        key = (
            str(row.get("bookmaker") or "unknown"),
            str(row.get("market") or "UNKNOWN"),
            str(row.get("line") or ""),
            str(row.get("side") or "").upper(),
        )

        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue

        duplicates_removed += 1
        current_odds = float(current["odd"])

        if current_odds != decimal_odds:
            conflicting_duplicates += 1

        if decimal_odds > current_odds:
            selected[key] = row

    deduplicated = [
        selected[key]
        for key in sorted(selected)
    ]
    return (
        deduplicated,
        duplicates_removed,
        conflicting_duplicates,
    )


def run_premier_league_shadow(
    base_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    *,
    max_paid_requests: int | None = 1,
) -> dict[str, Any]:
    if (
        max_paid_requests is not None
        and max_paid_requests < 0
    ):
        raise ValueError(
            "max_paid_requests must be non-negative"
        )
    base_dir = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parents[2]
    )
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else base_dir
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / SHADOW_DB_RELATIVE_PATH
    config = CollectorConfig(
        db_path=db_path,
        reports_dir=reports_dir,
        data_dir=output_dir / "data",
    )
    repo = CollectorRepository(config)
    fixture_collector = FixtureCollector(config, repo)
    odds_collector = OddsCollector(config, repo)
    live_adapter = LiveProviderAdapter(config)

    started_at = _utc_now()
    fixture_writes = 0
    odds_downloaded = 0
    odds_writes = 0
    odds_refresh_attempts = 0
    odds_refresh_skipped = 0
    matched_fixture_count = 0
    duplicates_removed = 0
    conflicting_duplicates = 0
    skip_reasons: dict[str, int] = {}

    quota_used: int | None = None
    quota_remaining: int | None = None
    quota_check_error: str | None = None

    try:
        live_adapter.the_odds_api.list_sports()
        usage = live_adapter.the_odds_api.get_usage()
        quota_used = _header_int(
            usage.get("x-requests-used")
        )
        quota_remaining = _header_int(
            usage.get("x-requests-remaining")
        )
    except Exception as exc:
        quota_check_error = str(exc)

    if quota_used is not None and quota_remaining is not None:
        repo.record_provider_usage(
            "the-odds-api",
            requests_used=quota_used,
            requests_remaining=quota_remaining,
            rate_limited=1 if quota_remaining <= 0 else 0,
        )

    events = live_adapter.the_odds_api.list_events(
        sport=SPORT_KEY
    )
    fixtures = _fixture_rows(events)

    for fixture in fixtures:
        saved = fixture_collector.collect_from_provider(
            fixture
        )
        if saved:
            fixture_writes += 1

        fixture_id = int(saved.get("fixture_id", 0))
        provider_fixture_id = str(
            fixture.get("provider_fixture_id") or ""
        )

        if not fixture_id or not provider_fixture_id:
            reason = "missing_fixture_id"
            skip_reasons[reason] = (
                skip_reasons.get(reason, 0) + 1
            )
            odds_refresh_skipped += 1
            continue

        latest_snapshot = _latest_odds_snapshot(
            config.db_path,
            fixture_id,
        )
        should_refresh, refresh_reason = _refresh_decision(
            fixture.get("kickoff_utc"),
            latest_snapshot,
        )

        if not should_refresh:
            skip_reasons[refresh_reason] = (
                skip_reasons.get(refresh_reason, 0) + 1
            )
            odds_refresh_skipped += 1
            continue

        if (
            max_paid_requests is not None
            and odds_refresh_attempts
            >= max_paid_requests
        ):
            reason = "paid_request_limit"
            skip_reasons[reason] = (
                skip_reasons.get(reason, 0) + 1
            )
            odds_refresh_skipped += 1
            continue

        if quota_remaining is None:
            reason = "quota_unknown"
            skip_reasons[reason] = (
                skip_reasons.get(reason, 0) + 1
            )
            odds_refresh_skipped += 1
            continue

        if quota_remaining <= 0:
            reason = "quota_exhausted"
            skip_reasons[reason] = (
                skip_reasons.get(reason, 0) + 1
            )
            odds_refresh_skipped += 1
            continue

        odds_refresh_attempts += 1
        raw_odds_rows = live_adapter.fetch_odds(
            provider_fixture_id
        )

        usage = live_adapter.the_odds_api.get_usage()
        current_used = _header_int(
            usage.get("x-requests-used")
        )
        current_remaining = _header_int(
            usage.get("x-requests-remaining")
        )

        if current_used is not None:
            quota_used = current_used
        if current_remaining is not None:
            quota_remaining = current_remaining

        resolution = live_adapter.last_odds_resolution.get(
            provider_fixture_id,
            {},
        )
        if resolution.get("match_status") == "MATCHED":
            matched_fixture_count += 1

        odds_downloaded += len(raw_odds_rows)
        (
            odds_rows,
            removed_count,
            conflict_count,
        ) = _deduplicate_odds(raw_odds_rows)
        duplicates_removed += removed_count
        conflicting_duplicates += conflict_count

        kickoff = _parse_timestamp(
            fixture.get("kickoff_utc")
        )
        now = datetime.now(timezone.utc)
        minutes_to_kickoff = (
            max(
                0,
                int(
                    (kickoff - now).total_seconds() / 60
                ),
            )
            if kickoff is not None
            else 0
        )
        snapshot_timestamp = config.now_utc()

        for row in odds_rows:
            payload = {
                "fixture_id": fixture_id,
                "bookmaker": row.get(
                    "bookmaker",
                    "unknown",
                ),
                "market": row.get(
                    "market",
                    "UNKNOWN",
                ),
                "line": row.get("line", ""),
                "side": row.get("side", ""),
                "decimal_odds": row.get("odd"),
                "snapshot_timestamp": snapshot_timestamp,
                "minutes_to_kickoff": (
                    minutes_to_kickoff
                ),
                "provider": "the-odds-api",
                "provider_event_id": str(
                    row.get("source_fixture_id")
                    or provider_fixture_id
                ),
                "raw_response_hash": (
                    "premier_league_shadow"
                ),
                "import_timestamp": snapshot_timestamp,
            }
            stored = odds_collector.collect_odds(payload)
            if stored is not None:
                odds_writes += 1

    if quota_used is not None and quota_remaining is not None:
        repo.record_provider_usage(
            "the-odds-api",
            requests_used=quota_used,
            requests_remaining=quota_remaining,
            rate_limited=1 if quota_remaining <= 0 else 0,
        )

    completed_at = _utc_now()
    result: dict[str, Any] = {
        "run_type": "premier_league_shadow",
        "mode": "collection_only",
        "competition": COMPETITION,
        "sport_key": SPORT_KEY,
        "started_at": started_at,
        "completed_at": completed_at,
        "fixtures_fetched": len(fixtures),
        "fixture_writes": fixture_writes,
        "fixtures_matched_to_odds_events": (
            matched_fixture_count
        ),
        "odds_refresh_attempts": odds_refresh_attempts,
        "max_paid_requests": max_paid_requests,
        "odds_refresh_skipped": odds_refresh_skipped,
        "odds_refresh_skip_reasons": skip_reasons,
        "odds_downloaded": odds_downloaded,
        "odds_writes": odds_writes,
        "duplicates_removed": duplicates_removed,
        "conflicting_duplicates": (
            conflicting_duplicates
        ),
        "quota_used": quota_used,
        "quota_remaining": quota_remaining,
        "quota_check_error": quota_check_error,
        "provider": "the-odds-api",
        "provider_status": (
            "quota_exhausted"
            if (
                quota_remaining is not None
                and quota_remaining <= 0
            )
            else (
                "ok"
                if odds_downloaded > 0
                else "warning"
            )
        ),
        "database_path": str(db_path),
        "paper_trading_enabled": False,
        "telegram_enabled": False,
        "real_betting_enabled": False,
    }

    report_path = reports_dir / SHADOW_REPORT_NAME
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)
    return result
