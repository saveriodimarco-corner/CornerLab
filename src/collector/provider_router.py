from __future__ import annotations

from typing import Any, Dict

from .collector_config import CollectorConfig


class ProviderRouter:
    def __init__(self, config: CollectorConfig):
        self.config = config

    def route(self, provider: str) -> str:
        return provider
