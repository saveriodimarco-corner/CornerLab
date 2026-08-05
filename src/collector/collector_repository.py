from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .collector_config import CollectorConfig, hash_payload


class CollectorRepository:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.db_path = config.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS collector_fixtures (
                    fixture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_fixture_id TEXT UNIQUE,
                    competition TEXT,
                    season TEXT,
                    kickoff_utc TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    status TEXT,
                    provider TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_teams (
                    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    team_name TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_bookmakers (
                    bookmaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bookmaker TEXT UNIQUE,
                    provider TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_markets (
                    market_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_name TEXT UNIQUE,
                    provider TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_odds_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER,
                    bookmaker TEXT,
                    market TEXT,
                    line TEXT,
                    side TEXT,
                    decimal_odds REAL,
                    snapshot_timestamp TEXT,
                    minutes_to_kickoff INTEGER,
                    provider TEXT,
                    provider_event_id TEXT,
                    raw_response_hash TEXT,
                    import_timestamp TEXT,
                    UNIQUE(fixture_id, bookmaker, market, line, side, snapshot_timestamp)
                );

                CREATE TABLE IF NOT EXISTS collector_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER UNIQUE,
                    home_score INTEGER,
                    away_score INTEGER,
                    home_corners INTEGER,
                    away_corners INTEGER,
                    total_corners INTEGER,
                    settled_at TEXT,
                    provider TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT,
                    writes INTEGER,
                    fixtures_discovered INTEGER,
                    fixtures_updated INTEGER,
                    odds_snapshots_stored INTEGER,
                    genuine_corner_odds_stored INTEGER,
                    completed_fixtures_resolved INTEGER,
                    readiness_verdict TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_errors (
                    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    message TEXT,
                    scope TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS collector_provider_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    requests_used INTEGER,
                    requests_remaining INTEGER,
                    rate_limited INTEGER,
                    created_at TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_fixture(self, provider_fixture_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM collector_fixtures WHERE provider_fixture_id = ?", (provider_fixture_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def upsert_fixture(self, fixture: Dict[str, Any]) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            now = self.config.now_utc()
            existing = self.get_fixture(fixture["provider_fixture_id"])
            if existing is None:
                cur = conn.execute(
                    """
                    INSERT INTO collector_fixtures (provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status, provider, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fixture["provider_fixture_id"],
                        fixture.get("competition"),
                        fixture.get("season"),
                        fixture.get("kickoff_utc"),
                        fixture.get("home_team"),
                        fixture.get("away_team"),
                        fixture.get("status"),
                        fixture.get("provider"),
                        now,
                        now,
                    ),
                )
                fixture_id = cur.lastrowid
                conn.commit()
                return {**fixture, "fixture_id": fixture_id, "created_at": now, "updated_at": now}
            updated = dict(existing)
            conn.execute(
                """
                UPDATE collector_fixtures
                SET competition = ?, season = ?, kickoff_utc = ?, home_team = ?, away_team = ?, status = ?, provider = ?, updated_at = ?
                WHERE provider_fixture_id = ?
                """,
                (
                    fixture.get("competition", existing.get("competition")),
                    fixture.get("season", existing.get("season")),
                    fixture.get("kickoff_utc", existing.get("kickoff_utc")),
                    fixture.get("home_team", existing.get("home_team")),
                    fixture.get("away_team", existing.get("away_team")),
                    fixture.get("status", existing.get("status")),
                    fixture.get("provider", existing.get("provider")),
                    now,
                    fixture["provider_fixture_id"],
                ),
            )
            conn.commit()
            updated.update({
                "competition": fixture.get("competition", existing.get("competition")),
                "season": fixture.get("season", existing.get("season")),
                "kickoff_utc": fixture.get("kickoff_utc", existing.get("kickoff_utc")),
                "home_team": fixture.get("home_team", existing.get("home_team")),
                "away_team": fixture.get("away_team", existing.get("away_team")),
                "status": fixture.get("status", existing.get("status")),
                "provider": fixture.get("provider", existing.get("provider")),
                "updated_at": now,
            })
            return {**updated, "fixture_id": existing["fixture_id"]}
        finally:
            conn.close()

    def store_snapshot(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if payload.get("market") in {"GOALS_OVER", "GOALS_UNDER", "GOALS_TOTAL"}:
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            payload_ts = payload.get("snapshot_timestamp")
            payload_odds = payload.get("decimal_odds")
            existing_rows = conn.execute(
                "SELECT * FROM collector_odds_snapshots WHERE fixture_id = ? AND bookmaker = ? AND market = ? AND line = ? AND side = ?",
                (
                    payload["fixture_id"],
                    payload.get("bookmaker"),
                    payload.get("market"),
                    payload.get("line"),
                    payload.get("side"),
                ),
            ).fetchall()
            for row in existing_rows:
                try:
                    existing_ts = datetime.fromisoformat(row["snapshot_timestamp"].replace("Z", "+00:00"))
                    incoming_ts = datetime.fromisoformat(payload_ts.replace("Z", "+00:00"))
                    if abs((incoming_ts - existing_ts).total_seconds()) <= 15 * 60 and float(row["decimal_odds"]) == float(payload_odds):
                        return None
                except Exception:
                    continue
            cur = conn.execute(
                """
                INSERT INTO collector_odds_snapshots (fixture_id, bookmaker, market, line, side, decimal_odds, snapshot_timestamp, minutes_to_kickoff, provider, provider_event_id, raw_response_hash, import_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["fixture_id"],
                    payload.get("bookmaker"),
                    payload.get("market"),
                    payload.get("line"),
                    payload.get("side"),
                    payload.get("decimal_odds"),
                    payload.get("snapshot_timestamp"),
                    payload.get("minutes_to_kickoff"),
                    payload.get("provider"),
                    payload.get("provider_event_id"),
                    payload.get("raw_response_hash"),
                    payload.get("import_timestamp"),
                ),
            )
            conn.commit()
            return {**payload, "snapshot_id": cur.lastrowid}
        finally:
            conn.close()

    def upsert_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO collector_results (fixture_id, home_score, away_score, home_corners, away_corners, total_corners, settled_at, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fixture_id) DO UPDATE SET home_score = excluded.home_score, away_score = excluded.away_score, home_corners = excluded.home_corners, away_corners = excluded.away_corners, total_corners = excluded.total_corners, settled_at = excluded.settled_at, provider = excluded.provider
                """,
                (
                    result["fixture_id"],
                    result.get("home_score"),
                    result.get("away_score"),
                    result.get("home_corners"),
                    result.get("away_corners"),
                    result.get("total_corners"),
                    result.get("settled_at"),
                    result.get("provider"),
                ),
            )
            conn.commit()
            return result
        finally:
            conn.close()

    def get_result(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM collector_results WHERE fixture_id = ?", (fixture_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def record_error(self, provider: str, message: str, scope: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO collector_errors (provider, message, scope, created_at) VALUES (?, ?, ?, ?)",
                (provider, message, scope, self.config.now_utc()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_errors(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if provider:
                rows = conn.execute("SELECT * FROM collector_errors WHERE provider = ? ORDER BY error_id DESC", (provider,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM collector_errors ORDER BY error_id DESC").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def record_provider_usage(self, provider: str, requests_used: int, requests_remaining: int, rate_limited: int) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO collector_provider_usage (provider, requests_used, requests_remaining, rate_limited, created_at) VALUES (?, ?, ?, ?, ?)",
                (provider, requests_used, requests_remaining, rate_limited, self.config.now_utc()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_provider_usage(self, provider: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT provider, SUM(requests_used) AS requests_used, MAX(requests_remaining) AS requests_remaining, SUM(rate_limited) AS rate_limited FROM collector_provider_usage WHERE provider = ? GROUP BY provider", (provider,)).fetchone()
            return dict(row) if row is not None else {"provider": provider, "requests_used": 0, "requests_remaining": 0, "rate_limited": 0}
        finally:
            conn.close()

    def insert_run(self, mode: str, status: str, writes: int, fixtures_discovered: int, fixtures_updated: int, odds_snapshots_stored: int, genuine_corner_odds_stored: int, completed_fixtures_resolved: int, readiness_verdict: str) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO collector_runs (mode, started_at, completed_at, status, writes, fixtures_discovered, fixtures_updated, odds_snapshots_stored, genuine_corner_odds_stored, completed_fixtures_resolved, readiness_verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    self.config.now_utc(),
                    self.config.now_utc(),
                    status,
                    writes,
                    fixtures_discovered,
                    fixtures_updated,
                    odds_snapshots_stored,
                    genuine_corner_odds_stored,
                    completed_fixtures_resolved,
                    readiness_verdict,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def count_fixtures(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM collector_fixtures").fetchone()[0])
        finally:
            conn.close()

    def count_snapshots(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM collector_odds_snapshots").fetchone()[0])
        finally:
            conn.close()

    def count_results(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM collector_results").fetchone()[0])
        finally:
            conn.close()
