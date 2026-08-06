from __future__ import annotations

import sqlite3
from pathlib import Path

from src.research.corner_feature_engine import build_corner_feature_pipeline


def test_build_corner_feature_pipeline_persists_feature_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "collector.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE collector_fixtures (
                fixture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_fixture_id TEXT,
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

            CREATE TABLE collector_results (
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
            """
        )
        connection.executemany(
            "INSERT INTO collector_fixtures (fixture_id, provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status, provider, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "f1", "Serie A", "2024", "2024-01-01T20:00:00Z", "Team A", "Team B", "FT", "test", "now", "now"),
                (2, "f2", "Serie A", "2024", "2024-01-08T20:00:00Z", "Team B", "Team A", "FT", "test", "now", "now"),
                (3, "f3", "Serie A", "2024", "2024-01-15T20:00:00Z", "Team A", "Team C", "FT", "test", "now", "now"),
            ],
        )
        connection.executemany(
            "INSERT INTO collector_results (fixture_id, home_score, away_score, home_corners, away_corners, total_corners, settled_at, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, 5, 2, 7, "2024-01-01T21:00:00Z", "test"),
                (2, 0, 1, 4, 3, 7, "2024-01-08T21:00:00Z", "test"),
                (3, 2, 1, 6, 4, 10, "2024-01-15T21:00:00Z", "test"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    rows = build_corner_feature_pipeline(db_path=db_path)

    assert len(rows) == 12
    assert {row["feature_name"] for row in rows} == {
        "corner_creation_rate",
        "corner_concession_rate",
        "recent_corner_form",
        "corner_diff_pressure",
    }

    connection = sqlite3.connect(db_path)
    try:
        persisted = connection.execute(
            "SELECT feature_name, fixture_id, feature_value FROM research_features ORDER BY fixture_id, feature_name"
        ).fetchall()
    finally:
        connection.close()

    assert len(persisted) == 12
    assert any(value == 0.0 for _, _, value in persisted if value is not None)
