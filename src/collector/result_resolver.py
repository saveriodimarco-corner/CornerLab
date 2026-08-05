from __future__ import annotations

from typing import Any, Dict

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository


class ResultResolver:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo

    def upsert_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.repo.upsert_result(payload)
