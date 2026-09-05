from __future__ import annotations

from typing import Any, Dict

from src.data.providers.odds.api_football_odds import (
    ApiFootballOddsProvider,
    ApiFootballProviderError,
)

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository


TERMINAL_FIXTURE_STATUSES = {"FT", "AET", "PEN"}


class ResultResolver:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo
        self.api_football = ApiFootballOddsProvider(api_key=config.api_football_key)

    def upsert_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.repo.upsert_result(payload)

    @staticmethod
    def _corner_count(statistics: list[dict[str, Any]]) -> int | None:
        for item in statistics:
            if str(item.get("type") or "").strip().lower() != "corner kicks":
                continue
            value = item.get("value")
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    def resolve_fixture(self, fixture_id: int | str) -> Dict[str, Any]:
        fixture = self.repo.get_fixture_by_id(fixture_id)
        if fixture is None:
            return {
                "ok": False,
                "reason": "fixture_not_found",
                "fixture_id": fixture_id,
            }

        provider_fixture_id = fixture.get("provider_fixture_id")
        if not provider_fixture_id:
            return {
                "ok": False,
                "reason": "provider_fixture_id_missing",
                "fixture_id": fixture_id,
            }

        try:
            fixture_payload = self.api_football._perform_request(
                "/fixtures",
                params={"id": provider_fixture_id},
            )
        except ApiFootballProviderError as exc:
            return {
                "ok": False,
                "reason": "provider_fixture_request_failed",
                "fixture_id": fixture_id,
                "category": exc.category,
            }

        fixture_rows = (
            fixture_payload.get("response", [])
            if isinstance(fixture_payload, dict)
            else []
        )
        if len(fixture_rows) != 1:
            return {
                "ok": False,
                "reason": "provider_fixture_unavailable",
                "fixture_id": fixture_id,
            }

        provider_row = fixture_rows[0]
        fixture_info = provider_row.get("fixture", {}) or {}
        status = str((fixture_info.get("status", {}) or {}).get("short") or "").upper()

        self.repo.upsert_fixture({
            **fixture,
            "status": status or fixture.get("status"),
        })

        if status not in TERMINAL_FIXTURE_STATUSES:
            return {
                "ok": False,
                "reason": "fixture_not_terminal",
                "fixture_id": fixture_id,
                "status": status,
            }

        try:
            statistics_payload = self.api_football._perform_request(
                "/fixtures/statistics",
                params={"fixture": provider_fixture_id},
            )
        except ApiFootballProviderError as exc:
            return {
                "ok": False,
                "reason": "provider_statistics_request_failed",
                "fixture_id": fixture_id,
                "category": exc.category,
            }

        statistics_rows = (
            statistics_payload.get("response", [])
            if isinstance(statistics_payload, dict)
            else []
        )

        home_team = str(fixture.get("home_team") or "").strip().casefold()
        away_team = str(fixture.get("away_team") or "").strip().casefold()
        home_corners: int | None = None
        away_corners: int | None = None

        for team_row in statistics_rows:
            team_name = str(
                ((team_row.get("team", {}) or {}).get("name")) or ""
            ).strip().casefold()
            corners = self._corner_count(team_row.get("statistics", []) or [])
            if team_name == home_team:
                home_corners = corners
            elif team_name == away_team:
                away_corners = corners

        if home_corners is None or away_corners is None:
            return {
                "ok": False,
                "reason": "corner_statistics_incomplete",
                "fixture_id": fixture_id,
                "status": status,
            }

        goals = provider_row.get("goals", {}) or {}
        home_score = goals.get("home")
        away_score = goals.get("away")

        result = {
            "fixture_id": int(fixture["fixture_id"]),
            "home_score": home_score,
            "away_score": away_score,
            "home_corners": home_corners,
            "away_corners": away_corners,
            "total_corners": home_corners + away_corners,
            "settled_at": self.config.now_utc(),
            "provider": "api-football",
        }

        self.repo.upsert_result(result)

        return {
            "ok": True,
            "reason": "resolved",
            "fixture_id": int(fixture["fixture_id"]),
            "status": status,
            "result": result,
        }
