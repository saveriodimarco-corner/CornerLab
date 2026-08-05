from __future__ import annotations

from typing import Any, Dict, List

from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider
from src.collector.collector_config import CollectorConfig


class LiveProviderAdapter:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.api_football = ApiFootballOddsProvider(api_key=config.api_football_key)
        self.last_resolution: Dict[str, Any] | None = None

    def fetch_fixtures(self) -> List[Dict[str, Any]]:
        league = self.api_football.resolve_serie_a_league()
        resolution = self.api_football.resolve_live_fixtures(league.get("league_id"), league.get("season"), next_count=5)
        self.last_resolution = resolution
        fixtures = resolution.get("fixtures", [])
        rows: List[Dict[str, Any]] = []
        for fixture in fixtures:
            fixture_payload = fixture.get("fixture", {})
            teams = fixture_payload.get("teams", {})
            rows.append({
                "provider_fixture_id": str(fixture_payload.get("id") or ""),
                "competition": "Serie A",
                "season": str(resolution.get("effective_season") or league.get("season") or ""),
                "kickoff_utc": fixture_payload.get("date"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "status": fixture_payload.get("status", {}).get("short"),
                "provider": "api-football",
            })
        return rows

    def fetch_odds(self, fixture_id: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in self.api_football.fetch_fixture_odds(fixture_id):
            payload = entry.get("response", []) if isinstance(entry, dict) else []
            if not isinstance(payload, list):
                continue
            for row in payload:
                if isinstance(row, dict):
                    rows.append(row)
        return rows
