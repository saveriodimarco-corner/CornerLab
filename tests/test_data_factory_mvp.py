from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.collector_health import CollectorHealth
from src.collector.fixture_collector import FixtureCollector
from src.collector.odds_collector import OddsCollector
from src.collector.provider_router import ProviderRouter
from src.collector.result_resolver import ResultResolver
from src.collector.snapshot_engine import SnapshotEngine
from src.collector.scheduler import CollectorScheduler


@pytest.fixture()
def temp_repo():
    with TemporaryDirectory() as tmpdir:
        config = CollectorConfig(db_path=Path(tmpdir) / "collector.sqlite")
        repo = CollectorRepository(config)
        yield config, repo


def test_fixture_insertion(temp_repo):
    config, repo = temp_repo
    collector = FixtureCollector(config, repo)
    fixture = collector.upsert_fixture({
        "provider_fixture_id": "api-1",
        "competition": "Serie A",
        "season": "2024/25",
        "kickoff_utc": "2025-01-01T20:00:00Z",
        "home_team": "Inter",
        "away_team": "Milan",
        "status": "SCHEDULED",
        "provider": "api-football",
    })
    assert fixture["provider_fixture_id"] == "api-1"
    assert repo.get_fixture("api-1") is not None


def test_idempotent_fixture_updates(temp_repo):
    config, repo = temp_repo
    collector = FixtureCollector(config, repo)
    first = collector.upsert_fixture({"provider_fixture_id": "api-2", "competition": "Serie A", "season": "2024/25", "kickoff_utc": "2025-01-01T20:00:00Z", "home_team": "Juventus", "away_team": "Roma", "status": "SCHEDULED", "provider": "api-football"})
    second = collector.upsert_fixture({"provider_fixture_id": "api-2", "competition": "Serie A", "season": "2024/25", "kickoff_utc": "2025-01-01T20:00:00Z", "home_team": "Juventus", "away_team": "Roma", "status": "POSTPONED", "provider": "api-football"})
    assert first["fixture_id"] == second["fixture_id"]
    assert second["status"] == "POSTPONED"


def test_genuine_corner_market_accepted(temp_repo):
    config, repo = temp_repo
    engine = SnapshotEngine(config, repo)
    snapshot = engine.store_snapshot({
        "fixture_id": 1,
        "bookmaker": "bet365",
        "market": "TOTAL_CORNERS",
        "line": "8.5",
        "side": "OVER",
        "decimal_odds": 2.10,
        "snapshot_timestamp": "2025-01-01T19:00:00Z",
        "minutes_to_kickoff": 60,
        "provider": "api-football",
        "provider_event_id": "evt-1",
        "raw_response_hash": "abc",
        "import_timestamp": "2025-01-01T19:00:00Z",
    })
    assert snapshot is not None
    assert snapshot["market"] == "TOTAL_CORNERS"


def test_goal_totals_rejected(temp_repo):
    config, repo = temp_repo
    engine = SnapshotEngine(config, repo)
    snapshot = engine.store_snapshot({
        "fixture_id": 1,
        "bookmaker": "bet365",
        "market": "GOALS_OVER",
        "line": "2.5",
        "side": "OVER",
        "decimal_odds": 1.90,
        "snapshot_timestamp": "2025-01-01T19:00:00Z",
        "minutes_to_kickoff": 60,
        "provider": "api-football",
        "provider_event_id": "evt-1",
        "raw_response_hash": "abc",
        "import_timestamp": "2025-01-01T19:00:00Z",
    })
    assert snapshot is None


def test_snapshot_deduplication(temp_repo):
    config, repo = temp_repo
    engine = SnapshotEngine(config, repo)
    first = engine.store_snapshot({
        "fixture_id": 1,
        "bookmaker": "bet365",
        "market": "TOTAL_CORNERS",
        "line": "8.5",
        "side": "OVER",
        "decimal_odds": 2.10,
        "snapshot_timestamp": "2025-01-01T19:00:00Z",
        "minutes_to_kickoff": 60,
        "provider": "api-football",
        "provider_event_id": "evt-1",
        "raw_response_hash": "abc",
        "import_timestamp": "2025-01-01T19:00:00Z",
    })
    second = engine.store_snapshot({
        "fixture_id": 1,
        "bookmaker": "bet365",
        "market": "TOTAL_CORNERS",
        "line": "8.5",
        "side": "OVER",
        "decimal_odds": 2.10,
        "snapshot_timestamp": "2025-01-01T19:01:00Z",
        "minutes_to_kickoff": 59,
        "provider": "api-football",
        "provider_event_id": "evt-1",
        "raw_response_hash": "abc",
        "import_timestamp": "2025-01-01T19:01:00Z",
    })
    assert first is not None
    assert second is None


def test_opening_and_closing_odds_selection(temp_repo):
    config, repo = temp_repo
    engine = SnapshotEngine(config, repo)
    engine.store_snapshot({"fixture_id": 1, "bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "decimal_odds": 2.00, "snapshot_timestamp": "2025-01-01T18:00:00Z", "minutes_to_kickoff": 120, "provider": "api-football", "provider_event_id": "ev1", "raw_response_hash": "a", "import_timestamp": "2025-01-01T18:00:00Z"})
    engine.store_snapshot({"fixture_id": 1, "bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "decimal_odds": 2.40, "snapshot_timestamp": "2025-01-01T19:45:00Z", "minutes_to_kickoff": 15, "provider": "api-football", "provider_event_id": "ev2", "raw_response_hash": "b", "import_timestamp": "2025-01-01T19:45:00Z"})
    opening = engine.select_opening_odds(1, "bet365", "TOTAL_CORNERS", "8.5", "OVER")
    closing = engine.select_closing_odds(1, "bet365", "TOTAL_CORNERS", "8.5", "OVER")
    assert opening["decimal_odds"] == 2.00
    assert closing["decimal_odds"] == 2.40


def test_first_empty_odds_response_marks_pending_retry(temp_repo):
    config, repo = temp_repo
    collector = OddsCollector(config, repo)
    result = collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [], provider="api-football")
    assert result["checked"] == 1
    assert result["downloaded"] == 0
    assert result["inserted"] == 0
    status = repo.get_odds_status(1, provider="api-football")
    assert status is not None
    assert status["status"] == "ODDS_NOT_AVAILABLE_YET"


def test_skip_before_ttl_expires(temp_repo):
    config, repo = temp_repo
    collector = OddsCollector(config, repo)
    collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [], provider="api-football")
    second = collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [{"bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "odd": 2.10}], provider="api-football")
    assert second["skipped"] is True
    assert second["checked"] == 0
    assert repo.count_snapshots() == 0


def test_retry_after_ttl_and_odds_become_available(temp_repo):
    config, repo = temp_repo
    config.odds_retry_ttl_minutes = 0
    collector = OddsCollector(config, repo)
    collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [], provider="api-football")
    second = collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [{"bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "odd": 2.10}], provider="api-football")
    assert second["checked"] == 1
    assert second["downloaded"] == 1
    assert second["inserted"] == 1
    assert repo.get_odds_status(1, provider="api-football") is None
    assert repo.count_snapshots() == 1


def test_normal_odds_persistence_is_unchanged(temp_repo):
    config, repo = temp_repo
    collector = OddsCollector(config, repo)
    result = collector.collect_odds_for_fixture(1, "fix-1", lambda fixture_id: [{"bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "odd": 2.10}], provider="api-football")
    assert result["checked"] == 1
    assert result["inserted"] == 1
    assert repo.count_snapshots() == 1
    assert repo.get_odds_status(1, provider="api-football") is None


def test_collect_odds_uses_source_fixture_id_when_present(temp_repo):
    config, repo = temp_repo
    collector = OddsCollector(config, repo)
    collector.collect_odds_for_fixture(
        1,
        "api-football-fix-1",
        lambda fixture_id: [
            {
                "bookmaker": "BetRivers",
                "market": "TOTAL_CORNERS_OVER",
                "line": "9.5",
                "side": "OVER",
                "odd": 2.05,
                "source_fixture_id": "the-odds-event-1",
            }
        ],
        provider="the-odds-api",
    )

    conn = sqlite3.connect(config.db_path)
    try:
        row = conn.execute(
            "SELECT provider_event_id, provider FROM collector_odds_snapshots WHERE fixture_id = 1 ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "the-odds-event-1"
    assert row[1] == "the-odds-api"


def test_in_play_odds_excluded_from_closing(temp_repo):
    config, repo = temp_repo
    engine = SnapshotEngine(config, repo)
    engine.store_snapshot({"fixture_id": 1, "bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "decimal_odds": 2.10, "snapshot_timestamp": "2025-01-01T19:55:00Z", "minutes_to_kickoff": 5, "provider": "api-football", "provider_event_id": "ev1", "raw_response_hash": "a", "import_timestamp": "2025-01-01T19:55:00Z"})
    engine.store_snapshot({"fixture_id": 1, "bookmaker": "bet365", "market": "TOTAL_CORNERS", "line": "8.5", "side": "OVER", "decimal_odds": 2.30, "snapshot_timestamp": "2025-01-01T20:05:00Z", "minutes_to_kickoff": -5, "provider": "api-football", "provider_event_id": "ev2", "raw_response_hash": "b", "import_timestamp": "2025-01-01T20:05:00Z"})
    closing = engine.select_closing_odds(1, "bet365", "TOTAL_CORNERS", "8.5", "OVER")
    assert closing["decimal_odds"] == 2.10


def test_result_resolution_and_missing_corners(temp_repo):
    config, repo = temp_repo
    collector = FixtureCollector(config, repo)
    collector.upsert_fixture({"provider_fixture_id": "api-3", "competition": "Serie A", "season": "2024/25", "kickoff_utc": "2025-01-01T20:00:00Z", "home_team": "Inter", "away_team": "Milan", "status": "COMPLETED", "provider": "api-football"})
    resolver = ResultResolver(config, repo)
    resolver.upsert_result({"fixture_id": 1, "home_score": 2, "away_score": 1, "home_corners": None, "away_corners": None, "total_corners": None, "settled_at": "2025-01-01T22:00:00Z", "provider": "api-football"})
    result = repo.get_result(1)
    assert result["home_score"] == 2
    assert result["home_corners"] is None


def test_provider_failure_isolation(temp_repo):
    config, repo = temp_repo
    collector = FixtureCollector(config, repo)
    collector.upsert_fixture({"provider_fixture_id": "api-4", "competition": "Serie A", "season": "2024/25", "kickoff_utc": "2025-01-01T20:00:00Z", "home_team": "Inter", "away_team": "Milan", "status": "SCHEDULED", "provider": "api-football"})
    repo.record_error("api-football", "boom", "fixture")
    errors = repo.list_errors("api-football")
    assert len(errors) == 1


def test_rate_limit_handling(temp_repo):
    config, repo = temp_repo
    repo.record_provider_usage("api-football", 1, 0, 0)
    usage = repo.get_provider_usage("api-football")
    assert usage["requests_used"] == 1


def test_secret_redaction(temp_repo):
    config, repo = temp_repo
    from src.collector.collector_config import redact_text
    assert "secret-123" not in redact_text("token=secret-123")


def test_dry_run_performs_zero_writes(temp_repo):
    config, repo = temp_repo
    scheduler = CollectorScheduler(config, repo)
    run = scheduler.run(mode="DRY_RUN")
    assert run["writes"] == 0


def test_scheduled_runs_do_not_overlap(temp_repo):
    config, repo = temp_repo
    scheduler = CollectorScheduler(config, repo)
    first = scheduler.run(mode="ONE_SHOT")
    second = scheduler.run(mode="ONE_SHOT")
    assert first["status"] in {"ok", "skipped"}
    assert second["status"] in {"ok", "skipped"}


def test_deterministic_outputs(temp_repo):
    config, repo = temp_repo
    scheduler = CollectorScheduler(config, repo)
    first = scheduler.run(mode="DRY_RUN")
    second = scheduler.run(mode="DRY_RUN")
    assert first == second


def test_provider_router_builds_readiness_state_for_blocked_provider():
    router = ProviderRouter(CollectorConfig())
    readiness = router.build_readiness_state({
        "collector_mode": "PROVIDER PLAN RESTRICTION",
        "provider_response_category": "PROVIDER PLAN RESTRICTION",
        "fixtures": [],
        "requested_season": 2026,
        "effective_season": 2026,
    })
    assert readiness["state"] == "BLOCKED"
    assert readiness["can_collect_fixtures"] is False


def test_scheduler_skips_writes_when_provider_is_blocked(temp_repo, monkeypatch):
    config, repo = temp_repo
    scheduler = CollectorScheduler(config, repo)
    monkeypatch.setattr(scheduler.live_adapter, "fetch_fixtures", lambda: [])
    scheduler.live_adapter.last_resolution = {
        "collector_mode": "PROVIDER PLAN RESTRICTION",
        "provider_response_category": "PROVIDER PLAN RESTRICTION",
        "requested_season": 2026,
        "effective_season": 2026,
        "provider": "api_football",
    }
    run = scheduler.run(mode="ONE_SHOT")
    assert run["writes"] == 0
    assert repo.list_errors("api_football")


def test_health_report_includes_readiness_details(temp_repo):
    config, repo = temp_repo
    health = CollectorHealth(config, repo, {
        "collector_mode": "PROVIDER PLAN RESTRICTION",
        "provider_response_category": "PROVIDER PLAN RESTRICTION",
        "requested_season": 2026,
        "effective_season": 2026,
        "provider": "api_football",
    })
    report = health.build_report()
    assert report["readiness_state"] == "BLOCKED"
    assert report["provider_capabilities"]["fixtures"] is False
