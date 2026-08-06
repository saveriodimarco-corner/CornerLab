from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository
from .provider_router import ProviderRouter


class CollectorHealth:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository, last_resolution: Dict[str, Any] | None = None):
        self.config = config
        self.repo = repo
        self.last_resolution = last_resolution or {}
        self.provider_router = ProviderRouter(config)

    def build_report(self) -> Dict[str, Any]:
        resolution = self.last_resolution or {}
        readiness = self.provider_router.build_readiness_state(resolution)
        return {
            "fixtures_discovered": self.repo.count_fixtures(),
            "fixtures_stored": self.repo.count_fixtures(),
            "odds_checked": 0,
            "odds_skipped_by_ttl": 0,
            "odds_pending_retry": self.repo.count_pending_odds_retries(),
            "odds_downloaded": 0,
            "odds_inserted": 0,
            "odds_snapshots_stored": self.repo.count_snapshots(),
            "completed_fixtures_resolved": self.repo.count_results(),
            "provider": resolution.get("provider") or "api-football",
            "requested_season": resolution.get("requested_season"),
            "effective_season": resolution.get("effective_season"),
            "api_error_category": resolution.get("api_error_category") or resolution.get("provider_response_category"),
            "redacted_api_error_message": resolution.get("redacted_api_error_message") or resolution.get("api_error_message"),
            "recommended_action": resolution.get("recommended_action") or "Review provider availability and retry",
            "readiness_verdict": resolution.get("collector_mode") or "LIVE_COLLECTION READY",
            "readiness_state": readiness["state"],
            "readiness_reason": readiness["reason"],
            "provider_capabilities": readiness["capabilities"],
        }
