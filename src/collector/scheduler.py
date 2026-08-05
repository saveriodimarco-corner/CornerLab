from __future__ import annotations

import time
from typing import Any, Dict

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository
from .fixture_collector import FixtureCollector
from .odds_collector import OddsCollector
from .result_resolver import ResultResolver
from .live_provider_adapter import LiveProviderAdapter


class CollectorScheduler:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo
        self.fixture_collector = FixtureCollector(config, repo)
        self.odds_collector = OddsCollector(config, repo)
        self.result_resolver = ResultResolver(config, repo)
        self.live_adapter = LiveProviderAdapter(config)

    def run(self, mode: str = "ONE_SHOT") -> Dict[str, Any]:
        writes = 0
        if mode == "DRY_RUN":
            return {"status": "ok", "mode": mode, "writes": 0, "deterministic": True}
        fixtures = self.live_adapter.fetch_fixtures()
        discovered = 0
        updated = 0
        snapshots = 0
        corner_snapshots = 0
        collector_mode = self.live_adapter.last_resolution.get("collector_mode", "NO FIXTURES AVAILABLE") if self.live_adapter.last_resolution else "NO FIXTURES AVAILABLE"
        if not fixtures:
            collector_mode = self.live_adapter.last_resolution.get("collector_mode", "NO FIXTURES AVAILABLE") if self.live_adapter.last_resolution else "NO FIXTURES AVAILABLE"
        for fixture in fixtures:
            discovered += 1
            saved = self.fixture_collector.collect_from_provider(fixture)
            if saved:
                writes += 1
                updated += 1
            for row in self.live_adapter.fetch_odds(str(fixture.get("provider_fixture_id", ""))):
                payload = {
                    "fixture_id": saved["fixture_id"],
                    "bookmaker": row.get("bookmaker", "unknown"),
                    "market": row.get("market", "UNKNOWN"),
                    "line": row.get("line", ""),
                    "side": row.get("side", ""),
                    "decimal_odds": row.get("odd"),
                    "snapshot_timestamp": self.config.now_utc(),
                    "minutes_to_kickoff": 60,
                    "provider": "api-football",
                    "provider_event_id": str(fixture.get("provider_fixture_id", "")),
                    "raw_response_hash": "live",
                    "import_timestamp": self.config.now_utc(),
                }
                stored = self.odds_collector.collect_odds(payload)
                if stored is not None:
                    snapshots += 1
                    if payload.get("market", "").upper().startswith("TOTAL_CORNERS"):
                        corner_snapshots += 1
        self.repo.insert_run(mode=mode, status="ok", writes=writes, fixtures_discovered=discovered, fixtures_updated=updated, odds_snapshots_stored=snapshots, genuine_corner_odds_stored=corner_snapshots, completed_fixtures_resolved=0, readiness_verdict=collector_mode)
        return {"status": "ok", "mode": mode, "writes": writes, "deterministic": True, "collector_mode": collector_mode}
