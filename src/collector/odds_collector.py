from __future__ import annotations

from typing import Any, Dict, Optional

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository


class OddsCollector:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo

    def collect_odds(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if payload.get("market") in {"GOALS_OVER", "GOALS_UNDER", "GOALS_TOTAL"}:
            return None
        return self.repo.store_snapshot(payload)
