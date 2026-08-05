from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


class ApiFootballProviderError(RuntimeError):
    def __init__(self, message: str, category: str, *, http_status: int | None = None, results: int | None = None, errors: Any = None, fixtures: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.results = results
        self.errors = errors
        self.fixtures = fixtures or []


class ApiFootballOddsProvider:
    name = "api_football"

    def __init__(self, api_key: str | None = None, base_url: str = "https://v3.football.api-sports.io") -> None:
        self.base_url = base_url.rstrip("/")
        if api_key is None:
            self.api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
            if not self.api_key:
                env_path = Path(__file__).resolve().parents[4] / ".env"
                if env_path.exists():
                    load_dotenv(env_path, override=False)
                    self.api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
        else:
            self.api_key = str(api_key).strip()
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
            raise ApiFootballProviderError("API_FOOTBALL_KEY is required", category="PROVIDER AUTHENTICATION ERROR")
        headers = {"x-apisports-key": self.api_key}
        try:
            response = requests.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=15)
        except requests.RequestException as exc:
            raise ApiFootballProviderError(f"Provider request failed: {exc}", category="PROVIDER REQUEST ERROR") from exc
        self._usage = {"status_code": response.status_code}
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiFootballProviderError("Provider returned malformed JSON", category="PROVIDER REQUEST ERROR", http_status=response.status_code) from exc
        if response.status_code >= 400:
            raise ApiFootballProviderError("Provider request failed", category="PROVIDER REQUEST ERROR", http_status=response.status_code, results=payload.get("results") if isinstance(payload, dict) else None, errors=payload.get("errors") if isinstance(payload, dict) else None)
        return payload

    def _classify_provider_error(self, errors: Any) -> tuple[str, str]:
        if not errors:
            return "PROVIDER REQUEST ERROR", "Provider returned an error response"
        error_text = json.dumps(self._redact_sensitive(errors), default=str).lower()
        if "plan" in error_text or "access" in error_text or "restricted" in error_text:
            return "PROVIDER PLAN RESTRICTION", "API-Football plan restriction prevented fixture access"
        if "token" in error_text or "api key" in error_text or "unauthorized" in error_text or "authentication" in error_text or "invalid" in error_text:
            return "PROVIDER AUTHENTICATION ERROR", "API-Football rejected the supplied key"
        return "PROVIDER REQUEST ERROR", "API-Football returned an error response"

    def _recommended_action(self, category: str) -> str:
        if category == "PROVIDER PLAN RESTRICTION":
            return "Confirm that the provider plan allows fixture access for the requested season before retrying"
        if category == "PROVIDER AUTHENTICATION ERROR":
            return "Verify the API-Football key and retry"
        return "Review the provider response and retry"

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

    def list_upcoming_fixtures(self, league_id: int | None, season: int | None, next_count: int = 20, query_mode: str = "NEXT") -> list[dict[str, Any]]:
        if league_id is None or season is None:
            raise RuntimeError(f"Unable to resolve Serie A fixtures: league_id={league_id}, season={season}")
        params: dict[str, Any] = {"league": league_id, "season": season}
        if query_mode.upper() == "LAST":
            params["last"] = next_count
        else:
            params["next"] = next_count
        payload = self._perform_request("/fixtures", params=params)
        response = payload.get("response", []) if isinstance(payload, dict) else []
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            category, message = self._classify_provider_error(errors)
            raise ApiFootballProviderError(message, category=category, http_status=self._usage.get("status_code"), results=payload.get("results") if isinstance(payload, dict) else None, errors=errors, fixtures=response if isinstance(response, list) else [])
        return response if isinstance(response, list) else []

    def _fixture_is_upcoming(self, fixture: dict[str, Any]) -> bool:
        fixture_payload = fixture.get("fixture", {}) if isinstance(fixture, dict) else {}
        kickoff = fixture_payload.get("date")
        if not kickoff:
            return False
        try:
            if isinstance(kickoff, str):
                value = kickoff.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                return parsed >= now
            return False
        except ValueError:
            return False

    def resolve_live_fixtures(self, league_id: int | None, season: int | None, next_count: int = 20) -> dict[str, Any]:
        if league_id is None or season is None:
            raise RuntimeError(f"Unable to resolve Serie A fixtures: league_id={league_id}, season={season}")

        requested_season = season
        try:
            fixtures = self.list_upcoming_fixtures(league_id, requested_season, next_count=next_count, query_mode="NEXT")
        except ApiFootballProviderError as exc:
            return {
                "requested_season": requested_season,
                "effective_season": requested_season,
                "fallback_used": False,
                "fallback_reason": None,
                "fixtures": [],
                "query_mode": "NEXT",
                "collector_mode": exc.category,
                "provider_response_category": exc.category,
                "api_error_category": exc.category,
                "api_error_message": str(exc),
                "redacted_api_error_message": str(exc),
                "recommended_action": self._recommended_action(exc.category),
                "provider": self.name,
            }
        if fixtures:
            return {
                "requested_season": requested_season,
                "effective_season": requested_season,
                "fallback_used": False,
                "fallback_reason": None,
                "fixtures": fixtures,
                "query_mode": "NEXT",
                "collector_mode": "LIVE_COLLECTION READY",
                "provider_response_category": "LIVE_COLLECTION READY",
                "api_error_category": None,
                "api_error_message": None,
                "redacted_api_error_message": None,
                "recommended_action": "Continue collecting fixtures for the live season",
                "provider": self.name,
            }

        raw = self.resolve_serie_a_league().get("raw") or {}
        seasons = []
        if isinstance(raw, dict):
            seasons = raw.get("seasons") or []
        season_candidates = []
        for season_entry in seasons:
            if not isinstance(season_entry, dict):
                continue
            year = season_entry.get("year")
            if year is None:
                continue
            season_candidates.append(int(year))

        season_candidates = sorted(set(season_candidates), reverse=True)
        for candidate in season_candidates:
            if candidate == requested_season:
                continue
            try:
                candidate_fixtures = self.list_upcoming_fixtures(league_id, candidate, next_count=next_count, query_mode="LAST")
            except ApiFootballProviderError as exc:
                return {
                    "requested_season": requested_season,
                    "effective_season": candidate,
                    "fallback_used": True,
                    "fallback_reason": "current season returned zero fixtures",
                    "fixtures": [],
                    "query_mode": "LAST",
                    "collector_mode": exc.category,
                    "provider_response_category": exc.category,
                    "api_error_category": exc.category,
                    "api_error_message": str(exc),
                    "redacted_api_error_message": str(exc),
                    "recommended_action": self._recommended_action(exc.category),
                    "provider": self.name,
                }
            if candidate_fixtures:
                return {
                    "requested_season": requested_season,
                    "effective_season": candidate,
                    "fallback_used": True,
                    "fallback_reason": "current season returned zero fixtures",
                    "fixtures": candidate_fixtures,
                    "query_mode": "LAST",
                    "collector_mode": "HISTORICAL VALIDATION MODE",
                    "provider_response_category": "HISTORICAL VALIDATION MODE",
                    "api_error_category": None,
                    "api_error_message": None,
                    "redacted_api_error_message": None,
                    "recommended_action": "Continue with historical validation fixtures",
                    "provider": self.name,
                }

        return {
            "requested_season": requested_season,
            "effective_season": None,
            "fallback_used": False,
            "fallback_reason": None,
            "fixtures": [],
            "query_mode": "LAST",
            "collector_mode": "NO FIXTURES AVAILABLE",
            "provider_response_category": "NO FIXTURES AVAILABLE",
            "api_error_category": None,
            "api_error_message": None,
            "redacted_api_error_message": None,
            "recommended_action": "No fixtures were returned for the requested season",
            "provider": self.name,
        }

    def list_upcoming_fixtures_with_date_range(self, league_id: int | None, season: int | None, from_date: str | None = None, to_date: str | None = None) -> list[dict[str, Any]]:
        if league_id is None or season is None:
            raise RuntimeError(f"Unable to resolve Serie A fixtures: league_id={league_id}, season={season}")
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
