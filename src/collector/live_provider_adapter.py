from __future__ import annotations

from typing import Any, Dict, List

from src.collector.collector_config import CollectorConfig
from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider


class LiveProviderAdapter:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.api_football = ApiFootballOddsProvider(api_key=config.api_football_key)
        self.last_resolution: Dict[str, Any] | None = None

    def fetch_fixtures(self) -> List[Dict[str, Any]]:
        league_id = 135
        season = 2026
        try:
            payload = self.api_football._perform_request("/fixtures", params={"league": league_id, "season": season, "next": 7})
        except Exception:
            payload = {}

        fixtures = payload.get("response", []) if isinstance(payload, dict) else []
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
            if not isinstance(entry, dict):
                continue
            bookmaker_payload = entry.get("bookmaker")
            bookmaker_name = bookmaker_payload.get("name") if isinstance(bookmaker_payload, dict) else None
            bets = entry.get("bets", [])
            if not isinstance(bets, list):
                continue
            for bet in bets:
                if not isinstance(bet, dict):
                    continue
                market_name = str(bet.get("name") or "UNKNOWN")
                values = bet.get("values", []) or []
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    odd = value.get("odd")
                    raw_value = value.get("value")
                    if odd is None or raw_value is None:
                        continue
                    line = str(raw_value)
                    side = ""
                    lowered_value = line.lower()
                    lowered_market = market_name.lower()
                    if "over" in lowered_value or "over" in lowered_market:
                        side = "OVER"
                    elif "under" in lowered_value or "under" in lowered_market:
                        side = "UNDER"
                    rows.append({
                        "bookmaker": bookmaker_name or "unknown",
                        "market": market_name,
                        "market_id": bet.get("id"),
                        "line": line,
                        "side": side,
                        "odd": odd,
                    })
        return rows
