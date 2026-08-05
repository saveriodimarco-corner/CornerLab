from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.collector.collector_config import CollectorConfig
from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider


class LiveProviderAdapter:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.api_football = ApiFootballOddsProvider(api_key=config.api_football_key)
        self.last_resolution: Dict[str, Any] | None = None

    def fetch_fixtures(self) -> List[Dict[str, Any]]:
        league = self.api_football.resolve_serie_a_league()
        league_id = league.get("league_id")
        season = league.get("season")
        if league_id is None or season is None:
            self.last_resolution = {
                "collector_mode": "NO FIXTURES AVAILABLE",
                "provider": "api_football",
                "fixtures": [],
            }
            return []

        today = datetime.now(timezone.utc).date()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            fixtures = self.api_football.list_upcoming_fixtures_with_date_range(league_id, season, from_date=from_date, to_date=to_date)
        except Exception:
            fixtures = []

        self.last_resolution = {
            "collector_mode": "LIVE_COLLECTION READY" if fixtures else "NO FIXTURES AVAILABLE",
            "provider": "api_football",
            "fixtures": fixtures,
            "requested_season": season,
            "effective_season": season,
        }

        rows: List[Dict[str, Any]] = []
        for fixture in fixtures:
            fixture_payload = fixture.get("fixture", {})
            teams = fixture_payload.get("teams", {})
            rows.append({
                "provider_fixture_id": str(fixture_payload.get("id") or ""),
                "competition": "Serie A",
                "season": str(season or ""),
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
