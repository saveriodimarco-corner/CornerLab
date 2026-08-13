from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from health_check import run_health_check
from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.fixture_collector import FixtureCollector
from src.collector.live_provider_adapter import LiveProviderAdapter
from src.collector.odds_collector import OddsCollector
from src.research.observation_freeze import build_production_baseline_manifest, settle_paper_trades
from src.research.paper_trading import run_paper_trading


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_prematch(base_dir: Path | str | None = None, output_dir: Path | str | None = None, bankroll: float = 100.0) -> dict[str, Any]:
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir) if output_dir is not None else base_dir

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    health = run_health_check(base_dir=base_dir, output_dir=output_dir)
    baseline_manifest = build_production_baseline_manifest(base_dir=base_dir, output_dir=output_dir)

    config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    repo = CollectorRepository(config)
    fixture_collector = FixtureCollector(config, repo)
    odds_collector = OddsCollector(config, repo)
    live_adapter = LiveProviderAdapter(config)

    fixtures = live_adapter.fetch_fixtures()
    fixture_writes = 0
    odds_writes = 0
    odds_downloaded = 0
    matched_fixture_count = 0

    for fixture in fixtures:
        saved = fixture_collector.collect_from_provider(fixture)
        fixture_writes += 1 if saved else 0
        provider_fixture_id = str(fixture.get("provider_fixture_id") or "")
        if not provider_fixture_id:
            continue

        odds_rows = live_adapter.fetch_odds(provider_fixture_id)
        resolution = live_adapter.last_odds_resolution.get(provider_fixture_id, {})
        if resolution.get("match_status") == "MATCHED":
            matched_fixture_count += 1

        odds_downloaded += len(odds_rows)
        for row in odds_rows:
            payload = {
                "fixture_id": int(saved.get("fixture_id", 0)),
                "bookmaker": row.get("bookmaker", "unknown"),
                "market": row.get("market", "UNKNOWN"),
                "line": row.get("line", ""),
                "side": row.get("side", ""),
                "decimal_odds": row.get("odd"),
                "snapshot_timestamp": config.now_utc(),
                "minutes_to_kickoff": 60,
                "provider": "the-odds-api",
                "provider_event_id": str(row.get("source_fixture_id") or provider_fixture_id),
                "raw_response_hash": "live_the_odds_api",
                "import_timestamp": config.now_utc(),
            }
            stored = odds_collector.collect_odds(payload)
            if stored is not None:
                odds_writes += 1

    paper_trading_result = run_paper_trading(base_dir=base_dir, output_dir=output_dir, bankroll=bankroll)
    settlement_result = settle_paper_trades(base_dir=base_dir, output_dir=output_dir, bankroll_start=bankroll)
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
            "provider_status": "ok" if odds_downloaded > 0 else "warning",
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
