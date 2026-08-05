from __future__ import annotations

from typing import Any, Dict, Optional

from .collector_config import CollectorConfig
from .collector_repository import CollectorRepository


class SnapshotEngine:
    def __init__(self, config: CollectorConfig, repo: CollectorRepository):
        self.config = config
        self.repo = repo

    def store_snapshot(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.repo.store_snapshot(payload)

    def select_opening_odds(self, fixture_id: int, bookmaker: str, market: str, line: str, side: str) -> Optional[Dict[str, Any]]:
        import sqlite3
        db = sqlite3.connect(self.repo.db_path)
        db.row_factory = sqlite3.Row
        try:
            row = db.execute(
                "SELECT * FROM collector_odds_snapshots WHERE fixture_id = ? AND bookmaker = ? AND market = ? AND line = ? AND side = ? ORDER BY snapshot_timestamp ASC, snapshot_id ASC LIMIT 1",
                (fixture_id, bookmaker, market, line, side),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            db.close()

    def select_closing_odds(self, fixture_id: int, bookmaker: str, market: str, line: str, side: str) -> Optional[Dict[str, Any]]:
        import sqlite3
        db = sqlite3.connect(self.repo.db_path)
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(
                "SELECT * FROM collector_odds_snapshots WHERE fixture_id = ? AND bookmaker = ? AND market = ? AND line = ? AND side = ? ORDER BY snapshot_timestamp DESC, snapshot_id DESC",
                (fixture_id, bookmaker, market, line, side),
            ).fetchall()
            for row in rows:
                row_dict = dict(row)
                minutes = int(row_dict.get("minutes_to_kickoff") or 0)
                if minutes > 0:
                    return row_dict
            return None
        finally:
            db.close()
