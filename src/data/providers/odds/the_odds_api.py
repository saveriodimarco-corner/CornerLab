from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.providers.odds.base_odds_provider import BaseOddsProvider


class TheOddsApiProvider(BaseOddsProvider):
    name = "the_odds_api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.the-odds-api.com",
        sport_key: str | None = None,
        regions: str | None = "eu",
        bookmakers: str | None = None,
        markets: str | None = "alternate_totals_corners",
        odds_format: str = "decimal",
        date_format: str = "iso",
        timeout: int = 15,
        retries: int = 2,
        rate_limit_backoff: float = 0.5,
        cache_dir: str | Path | None = None,
        enable_disk_cache: bool = False,
    ) -> None:
        self.api_key = (api_key or os.getenv("THE_ODDS_API_KEY", "")).strip()
        if not self.api_key:
            env_path = Path(__file__).resolve().parents[4] / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
                self.api_key = (api_key or os.getenv("THE_ODDS_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.sport_key = sport_key or ""
        self.regions = regions or "eu"
        self.bookmakers = bookmakers or ""
        self.markets = markets or "alternate_totals_corners"
        self.odds_format = odds_format
        self.date_format = date_format
        self.timeout = timeout
        self.retries = retries
        self.rate_limit_backoff = rate_limit_backoff
        self.cache_dir = Path(cache_dir or Path("data/cache/the_odds_api"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_disk_cache = enable_disk_cache
        self._usage: dict[str, Any] = {}
        self._request_count = 0
        self._memory_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}

    def _build_url(self, path: str, **kwargs: Any) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        prefix = "/v4"
        if path.startswith(prefix):
            url_path = path
        else:
            url_path = f"{prefix}{path}"
        sport = kwargs.get("sport")
        if sport:
            url_path = url_path.replace("{sport}", str(sport))
            if "serie-a" in url_path.lower():
                url_path = url_path.replace("serie-a", str(sport))
        return f"{self.base_url}{url_path}"

    def _redact_sensitive(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            redacted = {}
            for key, value in payload.items():
                if key == "api_key":
                    redacted[key] = "***"
                else:
                    redacted[key] = self._redact_sensitive(value)
            return redacted
        if isinstance(payload, list):
            return [self._redact_sensitive(item) for item in payload]
        if isinstance(payload, str):
            if self.api_key:
                return payload.replace(self.api_key, "***")
            return payload
        return payload

    def _cache_path(self, endpoint: str, params: dict[str, Any] | None = None) -> Path:
        payload = json.dumps({"endpoint": endpoint, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _cache_key(self, endpoint: str, params: dict[str, Any] | None = None) -> tuple[str, tuple[tuple[str, Any], ...]]:
        ordered_params = tuple(sorted((str(key), params.get(key)) for key in (params or {}).keys()))
        return (endpoint, ordered_params)

    def _persist_cache(self, endpoint: str, params: dict[str, Any] | None, payload: Any, headers: dict[str, Any] | None = None) -> None:
        cache_path = self._cache_path(endpoint, params)
        record = {
            "endpoint": endpoint,
            "requested_timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": params.get("event_id") if params else None,
            "response_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
            "headers": headers or {},
            "payload": payload,
        }
        cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _load_cache(self, endpoint: str, params: dict[str, Any] | None = None) -> Any | None:
        key = self._cache_key(endpoint, params)
        if key in self._memory_cache:
            return self._memory_cache[key]
        if not self.enable_disk_cache:
            return None
        cache_path = self._cache_path(endpoint, params)
        if not cache_path.exists():
            return None
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return record.get("payload")

    def _perform_request(self, url: str, headers: dict[str, Any], params: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json(), {
            "x-requests-remaining": response.headers.get("x-requests-remaining"),
            "x-requests-used": response.headers.get("x-requests-used"),
            "x-requests-last": response.headers.get("x-requests-last"),
        }

    def _prepare_request_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_params = dict(params or {})
        if not self.api_key:
            raise ValueError("THE_ODDS_API_KEY is required")
        for key in list(request_params.keys()):
            if request_params[key] in {None, ""}:
                request_params.pop(key)
        request_params["apiKey"] = self.api_key
        return request_params

    def _get_json(self, endpoint: str, params: dict[str, Any] | None = None, bypass_cache: bool = False) -> Any:
        if params is None:
            params = {}
        if not bypass_cache:
            cache_payload = self._load_cache(endpoint, params)
            if cache_payload is not None:
                return cache_payload
        if self.enable_disk_cache and not bypass_cache:
            cache_path = self._cache_path(endpoint, params)
            if cache_path.exists():
                try:
                    record = json.loads(cache_path.read_text(encoding="utf-8"))
                    if record.get("payload") is not None:
                        return record["payload"]
                except Exception:
                    pass
        if not self.api_key:
            raise ValueError("THE_ODDS_API_KEY is required")
        self._request_count += 1
        url = self._build_url(endpoint)
        headers: dict[str, Any] = {}
        request_params = self._prepare_request_params(params)
        try:
            result = self._perform_request(url, headers=headers, params=request_params)
            if isinstance(result, tuple) and len(result) == 2:
                payload, usage = result
            else:
                payload = result
                usage = {}
        except requests.HTTPError as exc:
            raise RuntimeError(f"The Odds API request failed: {self._redact_sensitive(str(exc))}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"The Odds API request failed: {self._redact_sensitive(str(exc))}") from exc
        self._usage = usage
        self._memory_cache[self._cache_key(endpoint, params)] = payload
        if self.enable_disk_cache:
            self._persist_cache(endpoint, params, payload, headers=self._usage)
        return payload

    def list_sports(self) -> list[dict[str, Any]]:
        payload = self._get_json("/v4/sports")
        if isinstance(payload, list):
            return payload
        return []

    def list_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        sport_key = sport or self.sport_key
        if not sport_key:
            raise ValueError("Serie A sport key is required")
        payload = self._get_json(f"/v4/sports/{sport_key}/events")
        if isinstance(payload, list):
            return payload
        return []

    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        sport_key = sport or self.sport_key
        if not sport_key:
            raise ValueError("Serie A sport key is required")
        if not event_id:
            raise ValueError("event_id is required")
        params = {
            "regions": self.regions,
            "bookmakers": self.bookmakers,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
        }
        payload = self._get_json(f"/v4/sports/{sport_key}/events/{event_id}/odds", params=params)
        return self.normalize_odds(payload, fixtures=fixtures)

    def fetch_historical_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.fetch_event_odds(event_id=event_id, sport=sport, fixtures=fixtures)

    def normalize_odds(self, payload: Any, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(columns=[
                "match_id", "fixture_date", "home_team", "away_team", "bookmaker", "market", "line", "side",
                "opening_odds", "closing_odds", "odds_timestamp", "source", "source_fixture_id", "is_closing",
                "currency", "import_timestamp",
            ])
        rows: list[dict[str, Any]] = []
        if not isinstance(payload, list):
            payload = [payload]
        for event in payload:
            event_id = str(event.get("id") or event.get("event_id") or "")
            for bookmaker in event.get("bookmakers", []):
                bookmaker_key = str(bookmaker.get("key") or bookmaker.get("title") or "")
                bookmaker_title = str(bookmaker.get("title") or bookmaker_key or "")
                for market in bookmaker.get("markets", []):
                    market_key = str(market.get("key") or "")
                    if market_key != "alternate_totals_corners":
                        continue
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        if point not in {8.5, 9.5, 10.5, 11.5}:
                            continue
                        if not isinstance(outcome.get("price"), (int, float)):
                            continue
                        price = float(outcome.get("price", 1.0))
                        if price <= 1.0:
                            continue
                        side = str(outcome.get("name") or "").upper()
                        if side not in {"OVER", "UNDER"}:
                            continue
                        rows.append({
                            "match_id": None,
                            "fixture_date": event.get("commence_time") or event.get("date") or "",
                            "home_team": event.get("home_team") or "",
                            "away_team": event.get("away_team") or "",
                            "bookmaker": bookmaker_title or bookmaker_key,
                            "market": "TOTAL_CORNERS_OVER" if side == "OVER" else "TOTAL_CORNERS_UNDER",
                            "line": str(point),
                            "side": side,
                            "opening_odds": price,
                            "closing_odds": price,
                            "odds_timestamp": event.get("commence_time") or event.get("date") or "",
                            "source": "THE_ODDS_API",
                            "source_fixture_id": event_id,
                            "is_closing": True,
                            "currency": "EUR",
                            "import_timestamp": event.get("commence_time") or event.get("date") or "1970-01-01T00:00:00+00:00",
                        })
        normalized = pd.DataFrame(rows)
        if normalized.empty:
            return normalized
        normalized = normalized.sort_values(["bookmaker", "market", "line", "side", "odds_timestamp"], kind="mergesort").reset_index(drop=True)
        return normalized

    def get_usage(self) -> dict[str, Any]:
        return dict(self._usage)
