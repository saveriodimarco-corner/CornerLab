from __future__ import annotations

from typing import Any, Dict

from .collector_config import CollectorConfig


class ProviderRouter:
    def __init__(self, config: CollectorConfig):
        self.config = config

    def route(self, provider: str) -> str:
        aliases = {
            "api-football": "api_football",
            "api_football": "api_football",
            "the-odds-api": "the_odds_api",
            "the_odds_api": "the_odds_api",
        }
        return aliases.get(str(provider or "").strip(), str(provider or ""))

    def _is_blocked(self, resolution: Dict[str, Any] | None) -> bool:
        if not resolution:
            return False
        collector_mode = str(resolution.get("collector_mode") or resolution.get("provider_response_category") or "").upper()
        api_error_category = str(resolution.get("api_error_category") or "").upper()
        blocked_modes = {"PROVIDER PLAN RESTRICTION", "PROVIDER AUTHENTICATION ERROR", "PROVIDER REQUEST ERROR"}
        return collector_mode in blocked_modes or api_error_category in blocked_modes

    def build_capabilities(self, resolution: Dict[str, Any] | None = None) -> Dict[str, bool]:
        resolution = resolution or {}
        if self._is_blocked(resolution):
            return {"fixtures": False, "results": False, "odds": False}
        fixtures = resolution.get("fixtures") or []
        has_fixtures = bool(fixtures)
        return {
            "fixtures": has_fixtures,
            "results": has_fixtures,
            "odds": has_fixtures,
        }

    def build_readiness_state(self, resolution: Dict[str, Any] | None = None) -> Dict[str, Any]:
        resolution = resolution or {}
        capabilities = self.build_capabilities(resolution)
        collector_mode = str(resolution.get("collector_mode") or resolution.get("provider_response_category") or "").upper()
        if self._is_blocked(resolution):
            return {
                "state": "BLOCKED",
                "reason": "provider returned a blocking error state",
                "can_collect_fixtures": False,
                "can_collect_results": False,
                "can_collect_odds": False,
                "capabilities": capabilities,
            }
        if capabilities["fixtures"]:
            return {
                "state": "READY",
                "reason": "fixtures available for live collection",
                "can_collect_fixtures": True,
                "can_collect_results": True,
                "can_collect_odds": True,
                "capabilities": capabilities,
            }
        if collector_mode == "HISTORICAL VALIDATION MODE":
            return {
                "state": "DEGRADED",
                "reason": "historical fallback fixtures are being used",
                "can_collect_fixtures": True,
                "can_collect_results": False,
                "can_collect_odds": False,
                "capabilities": capabilities,
            }
        return {
            "state": "NO_FIXTURES",
            "reason": "no fixtures are available from the provider",
            "can_collect_fixtures": False,
            "can_collect_results": False,
            "can_collect_odds": False,
            "capabilities": capabilities,
        }
