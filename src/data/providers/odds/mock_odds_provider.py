from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.data.providers.odds.base_odds_provider import BaseOddsProvider


class MockOddsProvider(BaseOddsProvider):
    name = "mock"

    def __init__(self, payload: list[dict[str, Any]] | None = None) -> None:
        self.payload = payload or [{
            "id": "mock-event-1",
            "sport_key": "soccer_italy_serie_a",
            "home_team": "Juventus",
            "away_team": "Inter",
            "commence_time": "2025-08-23T20:00:00Z",
            "bookmakers": [{
                "key": "bet365",
                "title": "Bet365",
                "markets": [{
                    "key": "alternate_totals_corners",
                    "outcomes": [
                        {"name": "Over", "price": 1.91, "point": 8.5},
                        {"name": "Under", "price": 1.95, "point": 8.5},
                        {"name": "Over", "price": 2.05, "point": 11.5},
                        {"name": "Under", "price": 1.80, "point": 11.5},
                    ],
                }],
            }],
        }]

    def list_sports(self) -> list[dict[str, Any]]:
        return [{"key": "soccer_italy_serie_a", "group": "Soccer", "title": "Italian Serie A"}]

    def list_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        return list(self.payload)

    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds(self.payload)

    def fetch_historical_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds(self.payload)

    def normalize_odds(self, payload: Any) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(columns=[
                "match_id", "fixture_date", "home_team", "away_team", "bookmaker", "market", "line", "side",
                "opening_odds", "closing_odds", "odds_timestamp", "source", "source_fixture_id", "is_closing",
                "currency", "import_timestamp",
            ])
        rows: list[dict[str, Any]] = []
        for event in payload:
            event_id = str(event.get("id", ""))
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "alternate_totals_corners":
                        continue
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        if point not in {8.5, 9.5, 10.5, 11.5}:
                            continue
                        side = "OVER" if str(outcome.get("name", "")).upper() == "OVER" else "UNDER"
                        rows.append({
                            "match_id": -1,
                            "fixture_date": event.get("commence_time", ""),
                            "home_team": event.get("home_team", ""),
                            "away_team": event.get("away_team", ""),
                            "bookmaker": bookmaker.get("title") or bookmaker.get("key") or "",
                            "market": "TOTAL_CORNERS_OVER" if side == "OVER" else "TOTAL_CORNERS_UNDER",
                            "line": str(point),
                            "side": side,
                            "opening_odds": float(outcome.get("price", 1.0)),
                            "closing_odds": float(outcome.get("price", 1.0)),
                            "odds_timestamp": event.get("commence_time", ""),
                            "source": "THE_ODDS_API",
                            "source_fixture_id": event_id,
                            "is_closing": True,
                            "currency": "EUR",
                            "import_timestamp": event.get("commence_time", ""),
                        })
        return pd.DataFrame(rows)

    def get_usage(self) -> dict[str, Any]:
        return {"x-requests-remaining": 999, "x-requests-used": 0, "x-requests-last": 0}
