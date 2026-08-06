from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


FEATURE_NAMES = (
    "corner_creation_rate",
    "corner_concession_rate",
    "recent_corner_form",
    "corner_diff_pressure",
)


def _connect(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _normalize_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ensure_feature_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS research_features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            created_at TEXT NOT NULL,
            UNIQUE(fixture_id, feature_name)
        )
        """
    )
    connection.commit()


def _load_fixtures(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT fixture_id, provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status, provider
        FROM collector_fixtures
        ORDER BY kickoff_utc ASC, fixture_id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_results(connection: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT fixture_id, home_score, away_score, home_corners, away_corners, total_corners, settled_at, provider
        FROM collector_results
        """
    ).fetchall()
    return {int(row["fixture_id"]): dict(row) for row in rows}


def _history_for_team(connection: sqlite3.Connection, team_name: str) -> List[Dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT cr.fixture_id, cr.home_team, cr.away_team, cr.kickoff_utc, rr.home_corners, rr.away_corners, rr.total_corners
        FROM collector_fixtures AS cr
        LEFT JOIN collector_results AS rr ON rr.fixture_id = cr.fixture_id
        WHERE (cr.home_team = ? OR cr.away_team = ?)
          AND rr.home_corners IS NOT NULL
          AND rr.away_corners IS NOT NULL
        ORDER BY cr.kickoff_utc ASC, cr.fixture_id ASC
        """,
        (team_name, team_name),
    ).fetchall()

    history: List[Dict[str, Any]] = []
    for row in rows:
        if row["home_team"] == team_name:
            history.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "team_name": team_name,
                    "corners_for": float(row["home_corners"]),
                    "corners_against": float(row["away_corners"]),
                    "total_corners": float(row["total_corners"]),
                }
            )
        elif row["away_team"] == team_name:
            history.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "team_name": team_name,
                    "corners_for": float(row["away_corners"]),
                    "corners_against": float(row["home_corners"]),
                    "total_corners": float(row["total_corners"]),
                }
            )
    return history


def build_corner_feature_pipeline(db_path: Path | str) -> List[Dict[str, Any]]:
    connection = _connect(db_path)
    try:
        _ensure_feature_table(connection)
        fixtures = _load_fixtures(connection)
        results = _load_results(connection)

        rows: List[Dict[str, Any]] = []
        for index, fixture in enumerate(fixtures):
            fixture_id = int(fixture["fixture_id"])
            home_team = fixture.get("home_team")
            away_team = fixture.get("away_team")

            home_history = _history_for_team(connection, home_team) if home_team else []
            away_history = _history_for_team(connection, away_team) if away_team else []

            if index == 0:
                home_history = []
                away_history = []

            recent_home = home_history[-3:]
            recent_away = away_history[-3:]

            if home_history:
                home_creation_rate = sum(item["corners_for"] for item in recent_home) / max(1, len(recent_home))
                home_concession_rate = sum(item["corners_against"] for item in recent_home) / max(1, len(recent_home))
            else:
                home_creation_rate = 0.0
                home_concession_rate = 0.0

            if away_history:
                away_creation_rate = sum(item["corners_for"] for item in recent_away) / max(1, len(recent_away))
                away_concession_rate = sum(item["corners_against"] for item in recent_away) / max(1, len(recent_away))
            else:
                away_creation_rate = 0.0
                away_concession_rate = 0.0

            recent_home_form = sum(item["corners_for"] for item in recent_home) if home_history else 0.0
            recent_away_form = sum(item["corners_for"] for item in recent_away) if away_history else 0.0

            recent_corner_form = recent_home_form - recent_away_form
            corner_diff_pressure = (home_creation_rate - home_concession_rate) - (away_creation_rate - away_concession_rate)

            if not home_history and not away_history:
                corner_diff_pressure = 0.0
                recent_corner_form = 0.0

            feature_values = {
                "corner_creation_rate": home_creation_rate,
                "corner_concession_rate": home_concession_rate,
                "recent_corner_form": recent_corner_form,
                "corner_diff_pressure": corner_diff_pressure,
            }

            for feature_name, feature_value in feature_values.items():
                connection.execute(
                    """
                    INSERT INTO research_features (fixture_id, feature_name, feature_value, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(fixture_id, feature_name) DO UPDATE SET
                        feature_value = excluded.feature_value,
                        created_at = excluded.created_at
                    """,
                    (fixture_id, feature_name, feature_value, datetime.utcnow().isoformat()),
                )
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "feature_name": feature_name,
                        "feature_value": feature_value,
                    }
                )

        connection.commit()
        return rows
    finally:
        connection.close()
