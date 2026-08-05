from __future__ import annotations

from typing import Dict, List


class ProviderRegistry:
    """Lightweight provider registry for historical dataset imports."""

    def __init__(self) -> None:
        self._providers: Dict[str, str] = {}
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        self.register_provider("football_data", "Football-Data")
        self.register_provider("api_football", "API-Football")
        self.register_provider("the_odds_api", "The Odds API")
        self.register_provider("manual_csv", "Manual CSV Import")

    def register_provider(self, provider_name: str, display_name: str) -> None:
        self._providers[provider_name] = display_name

    def list_providers(self) -> List[str]:
        return sorted(self._providers.keys())

    def get_display_name(self, provider_name: str) -> str:
        return self._providers.get(provider_name, provider_name)
