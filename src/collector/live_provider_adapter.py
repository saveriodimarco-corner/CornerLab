from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, Dict, List

import pandas as pd

from src.collector.collector_config import CollectorConfig
from src.data.odds_matcher import OddsMatcher
from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider
from src.data.providers.odds.the_odds_api import TheOddsApiProvider


class LiveProviderAdapter:
    SUPPORTED_TOTAL_CORNERS_LINES = {"8.5", "9.5", "10.5", "11.5"}
    THE_ODDS_SPORT_KEY = "soccer_italy_serie_a"
    TEAM_ALIASES = {
        "inter": ["inter milan", "internazionale"],
        "atalanta": ["atalanta bc"],
    }

    logger = logging.getLogger(__name__)

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.api_football = ApiFootballOddsProvider(api_key=config.api_football_key)
        self.the_odds_api = TheOddsApiProvider(
            api_key=config.the_odds_api_key,
            sport_key=self.THE_ODDS_SPORT_KEY,
            regions=config.the_odds_regions,
            markets=config.the_odds_market,
            odds_format="decimal",
            date_format="iso",
        )
        self.matcher = OddsMatcher(team_aliases=self.TEAM_ALIASES)
        self.last_resolution: Dict[str, Any] | None = None
        self.last_odds_resolution: Dict[str, Dict[str, Any]] = {}
        self._cached_events: List[Dict[str, Any]] | None = None

    def _extract_line_token(self, value: str) -> str | None:
        matches = re.findall(r"\d+(?:\.\d+)?", value.replace(",", "."))
        if not matches:
            return None
        try:
            line_value = float(matches[-1])
        except ValueError:
            return None
        normalized = f"{line_value:.1f}"
        if normalized not in self.SUPPORTED_TOTAL_CORNERS_LINES:
            return None
        return normalized

    def _normalize_total_corners_row(self, market_name: str, raw_value: str, odd: Any, bookmaker_name: str) -> Dict[str, Any] | None:
        lowered_market = market_name.lower()
        lowered_value = raw_value.lower()
        if "corner" not in lowered_market:
            return None

        side = ""
        if "over" in lowered_value or "over" in lowered_market:
            side = "OVER"
        elif "under" in lowered_value or "under" in lowered_market:
            side = "UNDER"
        if not side:
            return None

        line = self._extract_line_token(raw_value) or self._extract_line_token(market_name)
        if not line:
            return None

        try:
            decimal_odds = float(odd)
        except (TypeError, ValueError):
            return None

        return {
            "bookmaker": bookmaker_name or "unknown",
            "market": "TOTAL_CORNERS_OVER" if side == "OVER" else "TOTAL_CORNERS_UNDER",
            "line": line,
            "side": side,
            "odd": decimal_odds,
            "provider_market_name": market_name,
        }

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
            teams = fixture.get("teams", {}) if isinstance(fixture, dict) else {}
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
        fixture = self._load_collector_fixture(fixture_id)
        if not fixture:
            self.last_odds_resolution[fixture_id] = {"match_status": "UNMATCHED", "reason": "fixture_not_found"}
            self.logger.warning("MATCH_UNRESOLVED fixture_id=%s reason=fixture_not_found", fixture_id)
            return []

        event = self._match_event_for_fixture(fixture)
        if event is None:
            self.last_odds_resolution[fixture_id] = {"match_status": "UNMATCHED", "reason": "event_not_matched"}
            self.logger.warning("MATCH_UNRESOLVED fixture_id=%s home=%s away=%s kickoff=%s", fixture_id, fixture.get("home_team"), fixture.get("away_team"), fixture.get("kickoff_utc"))
            return []

        event_id = str(event.get("id") or "")
        if not event_id:
            self.last_odds_resolution[fixture_id] = {"match_status": "UNMATCHED", "reason": "event_id_missing"}
            self.logger.warning("MATCH_UNRESOLVED fixture_id=%s reason=event_id_missing", fixture_id)
            return []

        odds_df = self.the_odds_api.fetch_event_odds(event_id=event_id, sport=self.THE_ODDS_SPORT_KEY)
        rows: List[Dict[str, Any]] = []
        if odds_df is not None and not odds_df.empty:
            for _, row in odds_df.iterrows():
                line = str(row.get("line") or "")
                side = str(row.get("side") or "").upper()
                if line not in self.SUPPORTED_TOTAL_CORNERS_LINES:
                    continue
                if side not in {"OVER", "UNDER"}:
                    continue
                odd = row.get("closing_odds")
                try:
                    decimal_odds = float(odd)
                except (TypeError, ValueError):
                    continue
                if decimal_odds <= 1.0:
                    continue
                rows.append(
                    {
                        "bookmaker": str(row.get("bookmaker") or "unknown"),
                        "market": str(row.get("market") or "UNKNOWN"),
                        "line": line,
                        "side": side,
                        "odd": decimal_odds,
                        "source_fixture_id": event_id,
                    }
                )

        self.last_odds_resolution[fixture_id] = {
            "match_status": "MATCHED",
            "reason": "event_matched",
            "event_id": event_id,
            "downloaded": len(rows),
        }
        return rows

    def _load_collector_fixture(self, provider_fixture_id: str) -> Dict[str, Any] | None:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT fixture_id, provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status
                FROM collector_fixtures
                WHERE provider_fixture_id = ?
                """,
                (provider_fixture_id,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def _list_the_odds_events(self) -> List[Dict[str, Any]]:
        if self._cached_events is None:
            self._cached_events = self.the_odds_api.list_events(sport=self.THE_ODDS_SPORT_KEY)
        return self._cached_events

    def _match_event_for_fixture(self, fixture: Dict[str, Any]) -> Dict[str, Any] | None:
        events = self._list_the_odds_events()
        if not events:
            return None

        event_rows = []
        event_index: Dict[int, Dict[str, Any]] = {}
        for idx, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue
            event_rows.append(
                {
                    "match_id": idx,
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "date": event.get("commence_time"),
                    "competition": "Serie A",
                    "season": fixture.get("season"),
                }
            )
            event_index[idx] = event

        if not event_rows:
            return None

        match = self.matcher.match_event_to_fixture(
            event={
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "commence_time": fixture.get("kickoff_utc"),
            },
            fixtures=pd.DataFrame(event_rows),
            tolerance_minutes=45,
            competition="Serie A",
            season=str(fixture.get("season") or ""),
        )
        if match.get("match_status") != "MATCHED":
            return None
        return event_index.get(int(match.get("match_id") or 0))
