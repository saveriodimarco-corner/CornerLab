from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.path.join("data", "raw", "acquisition.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connect()
        self._initialize_schema()

    def _connect(self) -> None:
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                row_hash TEXT NOT NULL,
                date TEXT NOT NULL,
                season TEXT NOT NULL,
                competition TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                row_hash TEXT NOT NULL,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_corners INTEGER NOT NULL,
                away_corners INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                row_hash TEXT NOT NULL,
                team_name TEXT NOT NULL,
                competition TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS competitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                row_hash TEXT NOT NULL,
                competition_name TEXT NOT NULL,
                season TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                row_hash TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def import_rows(self, rows: List[Dict[str, Any]], provider_name: str) -> int:
        import_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        inserted = 0
        for row in rows:
            row_hash = row.get("row_hash") or self._hash_row(row)
            self.connection.execute(
                "INSERT INTO matches (source, import_date, row_hash, date, season, competition, home_team, away_team) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (provider_name, import_date, row_hash, row.get("date"), row.get("season"), row.get("competition"), row.get("home_team"), row.get("away_team")),
            )
            self.connection.execute(
                "INSERT INTO match_stats (source, import_date, row_hash, date, home_team, away_team, home_corners, away_corners) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (provider_name, import_date, row_hash, row.get("date"), row.get("home_team"), row.get("away_team"), row.get("home_corners"), row.get("away_corners")),
            )
            self.connection.execute(
                "INSERT INTO teams (source, import_date, row_hash, team_name, competition) VALUES (?, ?, ?, ?, ?)",
                (provider_name, import_date, row_hash, row.get("home_team"), row.get("competition")),
            )
            self.connection.execute(
                "INSERT INTO competitions (source, import_date, row_hash, competition_name, season) VALUES (?, ?, ?, ?, ?)",
                (provider_name, import_date, row_hash, row.get("competition"), row.get("season")),
            )
            self.connection.execute(
                "INSERT INTO sources (source, import_date, row_hash, provider_name, status) VALUES (?, ?, ?, ?, ?)",
                (provider_name, import_date, row_hash, provider_name, "imported"),
            )
            inserted += 1
        self.connection.commit()
        return inserted

    def _hash_row(self, row: Dict[str, Any]) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def close(self) -> None:
        self.connection.close()
