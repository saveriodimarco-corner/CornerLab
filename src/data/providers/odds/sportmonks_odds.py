from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


class SportmonksOddsProvider:
    name = "sportmonks"

    def __init__(self, api_token: str | None = None, base_url: str = "https://api.sportmonks.com/v3") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = (api_token or os.getenv("SPORTMONKS_API_TOKEN", "")).strip()
        if not self.api_token:
            env_path = Path(__file__).resolve().parents[4] / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
                self.api_token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
        self._usage: dict[str, Any] = {}

    def _redact_sensitive(self, payload: Any) -> Any:
        sensitive_values = {self.api_token} if self.api_token else set()

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
        if not self.api_token:
            raise ValueError("SPORTMONKS_API_TOKEN is required")
        response = requests.get(f"{self.base_url}{path}", params={**(params or {}), "api_token": self.api_token}, timeout=15)
        response.raise_for_status()
        self._usage = {"status_code": response.status_code}
        return response.json()

    def list_sports(self) -> list[dict[str, Any]]:
        return []

    def list_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        return []

    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds([])

    def fetch_historical_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.normalize_odds([])

    def normalize_odds(self, payload: Any) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in payload or []:
            market_name = str(event.get("market_name") or "").lower()
            if "corner" in market_name and "goal" not in market_name:
                for odd in event.get("bookmakers", [{}])[0].get("odds", []):
                    rows.append({"market": "TOTAL_CORNERS_OVER", "line": str(odd.get("line", "")), "side": str(odd.get("name", "")).upper(), "bookmaker": event.get("bookmakers", [{}])[0].get("name", "")})
        return pd.DataFrame(rows)

    def get_usage(self) -> dict[str, Any]:
        return dict(self._usage)
