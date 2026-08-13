from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository


class OddsCollector:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo

    def collect_odds(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if payload.get("market") in {"GOALS_OVER", "GOALS_UNDER", "GOALS_TOTAL"}:
            return None
        return self.repo.store_snapshot(payload)

    def collect_odds_for_fixture(self, fixture_id: int, provider_fixture_id: str, fetch_odds_rows, provider: str = "api-football") -> Dict[str, Any]:
        current_time = datetime.now(timezone.utc)
        existing_status = self.repo.get_odds_status(fixture_id, provider=provider)

        if existing_status is not None:
            status = existing_status.get("status")
            next_retry_after = existing_status.get("next_retry_after")
            if status == "ODDS_NOT_AVAILABLE_YET" and next_retry_after:
                try:
                    retry_at = datetime.fromisoformat(next_retry_after.replace("Z", "+00:00"))
                except ValueError:
                    retry_at = current_time
                if current_time < retry_at:
                    return {"fixture_id": fixture_id, "provider_fixture_id": provider_fixture_id, "checked": 0, "downloaded": 0, "inserted": 0, "skipped": True, "status": status}

        odds_rows = list(fetch_odds_rows(provider_fixture_id))
        checked = 1 if odds_rows is not None else 0
        downloaded = len(odds_rows)
        inserted = 0
        if not odds_rows:
            self.repo.upsert_odds_status(
                fixture_id,
                provider,
                "ODDS_NOT_AVAILABLE_YET",
                checked_at=self.config.now_utc(),
                next_retry_after=(current_time + timedelta(minutes=self.config.odds_retry_ttl_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return {"fixture_id": fixture_id, "provider_fixture_id": provider_fixture_id, "checked": checked, "downloaded": downloaded, "inserted": inserted, "skipped": False, "status": "ODDS_NOT_AVAILABLE_YET"}

        self.repo.delete_odds_status(fixture_id, provider=provider)
        for odds_row in odds_rows:
            payload = {
                "fixture_id": fixture_id,
                "bookmaker": odds_row.get("bookmaker", "unknown"),
                "market": odds_row.get("market", "UNKNOWN"),
                "line": odds_row.get("line", ""),
                "side": odds_row.get("side", ""),
                "decimal_odds": odds_row.get("odd"),
                "snapshot_timestamp": self.config.now_utc(),
                "minutes_to_kickoff": 60,
                "provider": provider,
                "provider_event_id": odds_row.get("source_fixture_id") or provider_fixture_id,
                "raw_response_hash": f"live_{str(provider).replace('-', '_')}",
                "import_timestamp": self.config.now_utc(),
            }
            stored = self.collect_odds(payload)
            if stored is not None:
                inserted += 1
        return {"fixture_id": fixture_id, "provider_fixture_id": provider_fixture_id, "checked": checked, "downloaded": downloaded, "inserted": inserted, "skipped": False, "status": "ODDS_AVAILABLE"}
