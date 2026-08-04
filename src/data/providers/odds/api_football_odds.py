from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


class ApiFootballOddsProvider:
    name = "api_football"

    def __init__(self, api_key: str | None = None, base_url: str = "https://v3.football.api-sports.io") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or os.getenv("API_FOOTBALL_KEY", "")).strip()
        if not self.api_key:
            env_path = Path(__file__).resolve().parents[4] / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
                self.api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
        self._usage: dict[str, Any] = {}

    def _redact_sensitive(self, payload: Any) -> Any:
        sensitive_values = {self.api_key} if self.api_key else set()

        def collect_sensitive_values(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_name = str(key).lower()
                    if any(token in key_name for token in {"api_key", "apikey", "token", "secret", "password"}) and isinstance(value, str) and value:
                        sensitive_values.add(value)
                    collect_sensitive_values(value)
            elif isinstance(node, list):
                for item in node:
                    collect_sensitive_values(item)

        collect_sensitive_values(payload)

        def redact(node: Any) -> Any:
            if isinstance(node, dict):
                return {
                    key: "***" if self._is_sensitive_key(key, value, sensitive_values) else redact(value)
                    for key, value in node.items()
                }
            if isinstance(node, list):
                return [redact(item) for item in node]
            if isinstance(node, str):
                for value in sensitive_values:
                    if value and value in node:
                        return "***"
                return node
            return node

        return redact(payload)

    def _is_sensitive_key(self, key: Any, value: Any, sensitive_values: set[str]) -> bool:
        key_name = str(key).lower()
        if any(token in key_name for token in {"api_key", "apikey", "token", "secret", "password"}):
            return True
        return isinstance(value, str) and value in sensitive_values

    def _perform_request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY is required")
        headers = {"x-apisports-key": self.api_key}
        response = requests.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        self._usage = {"status_code": response.status_code}
        return response.json()

    def list_sports(self) -> list[dict[str, Any]]:
        payload = self._perform_request("/leagues")
        return payload.get("response", []) if isinstance(payload, dict) else []

    def resolve_serie_a_league(self) -> dict[str, Any]:
        leagues = self.list_sports()
        for league in leagues:
            league_name = str(league.get("league", {}).get("name", "") or "")
            country_name = str(league.get("country", {}).get("name", "") or "")
            if country_name.lower() == "italy" and league_name.lower() == "serie a":
                return {
                    "league_id": league.get("league", {}).get("id"),
                    "season": league.get("seasons", [{}])[-1].get("year") if league.get("seasons") else None,
                    "current": bool(league.get("seasons", [{}])[-1].get("current")) if league.get("seasons") else False,
                    "coverage": league.get("coverage", {}),
                    "raw": league,
                }
        return {"league_id": None, "season": None, "current": False, "coverage": {}, "raw": None}

    def list_corner_bet_catalog(self) -> list[dict[str, Any]]:
        payload = self._perform_request("/odds/bets")
        return self.find_corner_bet_entries(payload)

    def find_corner_bet_entries(self, payload: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        response = payload.get("response", []) if isinstance(payload, dict) else []
        for entry in response:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "") or "")
            lowered = name.lower()
            if any(keyword in lowered for keyword in {"corner", "corners", "total corners", "over/under corners", "match corners", "alternative corners"}):
                entries.append({"id": entry.get("id"), "name": entry.get("name")})
        return entries

    def list_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        payload = self._perform_request("/fixtures", params={"league": 135, "season": 2025, "next": 3})
        return payload.get("response", []) if isinstance(payload, dict) else []

    def list_upcoming_fixtures(self, league_id: int | None, season: int | None, next_count: int = 20) -> list[dict[str, Any]]:
        if league_id is None or season is None:
            return []
        payload = self._perform_request("/fixtures", params={"league": league_id, "season": season, "next": next_count})
        response = payload.get("response", []) if isinstance(payload, dict) else []
        return response if isinstance(response, list) else []

    def list_upcoming_fixtures_with_date_range(self, league_id: int | None, season: int | None, from_date: str | None = None, to_date: str | None = None) -> list[dict[str, Any]]:
        if league_id is None or season is None:
            return []
        params: dict[str, Any] = {"league": league_id, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        payload = self._perform_request("/fixtures", params=params)
        response = payload.get("response", []) if isinstance(payload, dict) else []
        return response if isinstance(response, list) else []

    def fetch_fixture_odds(self, fixture_id: int | str | None) -> list[dict[str, Any]]:
        if fixture_id is None:
            return []
        payload = self._perform_request("/odds", params={"fixture": fixture_id})
        response = payload.get("response", []) if isinstance(payload, dict) else []
        return response if isinstance(response, list) else []

    def extract_corner_odds_rows(self, payload: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        response = payload.get("response", []) if isinstance(payload, dict) else []
        for bookmaker_entry in response:
            bookmaker_name = bookmaker_entry.get("bookmaker", {}).get("name") if isinstance(bookmaker_entry, dict) else None
            bets = bookmaker_entry.get("bets", []) if isinstance(bookmaker_entry, dict) else []
            for bet in bets:
                if not isinstance(bet, dict):
                    continue
                bet_name = str(bet.get("name", "") or "")
                lowered = bet_name.lower()
                if not any(keyword in lowered for keyword in {"corner", "corners", "total corners", "over/under corners", "match corners", "alternative corners"}):
                    continue
                for value in bet.get("values", []) or []:
                    if not isinstance(value, dict):
                        continue
                    rows.append({
                        "bookmaker": bookmaker_name,
                        "bet_id": bet.get("id"),
                        "bet_name": bet_name,
                        "value": value.get("value"),
                        "odd": value.get("odd"),
                    })
        return rows

    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds([])

    def fetch_historical_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds([])

    def normalize_odds(self, payload: Any) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in payload or []:
            market_name = str(event.get("name") or "").lower()
            if "goal" in market_name or "goal" in str(event.get("id") or ""):
                continue
            if "corner" in market_name or "corner" in str(event.get("id") or ""):
                rows.append({
                    "market": "TOTAL_CORNERS_OVER",
                    "line": "10.5",
                    "side": "OVER",
                    "bookmaker": event.get("bookmakers", [{}])[0].get("name", ""),
                })
                rows.append({
                    "market": "TOTAL_CORNERS_UNDER",
                    "line": "10.5",
                    "side": "UNDER",
                    "bookmaker": event.get("bookmakers", [{}])[0].get("name", ""),
                })
        return pd.DataFrame(rows)

    def get_usage(self) -> dict[str, Any]:
        return dict(self._usage)
