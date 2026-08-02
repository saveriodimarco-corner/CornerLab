from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.database import DataStore
from src.data.normalizer import NormalizationError, normalize_row
from src.data.providers import ApiFootballProvider, CSVProvider, FootballDataProvider
from src.data.quality import build_quality_report


def test_normalizer_handles_valid_rows() -> None:
    row = {
        "date": "2024-08-10",
        "season": "2024/25",
        "competition": "premier league",
        "home_team": "team a",
        "away_team": "team b",
        "home_corners": "7",
        "away_corners": "4",
    }
    normalized = normalize_row(row)
    assert normalized["home_team"] == "Team A"
    assert normalized["away_corners"] == 4


def test_normalizer_rejects_malformed_rows() -> None:
    with pytest.raises(NormalizationError):
        normalize_row({"date": "bad-date", "home_team": "", "away_team": "Team B", "home_corners": "-1"})


def test_quality_report_is_generated() -> None:
    rows = [
        {"date": "2024-08-10", "season": "2024/25", "competition": "Premier League", "home_team": "Team A", "away_team": "Team B", "home_corners": 4, "away_corners": 3},
        {"date": "2024-08-10", "season": "2024/25", "competition": "Premier League", "home_team": "Team A", "away_team": "Team B", "home_corners": 4, "away_corners": 3},
        {"date": "2024-08-11", "season": "2024/25", "competition": "Premier League", "home_team": "Team A", "away_team": "Team A", "home_corners": None, "away_corners": 2},
    ]
    report = build_quality_report(rows)
    assert "# Data Quality Report" in report
    assert "Duplicate matches" in report
    assert "Missing statistics" in report


def test_provider_interfaces_return_expected_schema() -> None:
    providers = [FootballDataProvider(), ApiFootballProvider(), CSVProvider(os.path.join("data", "raw", "matches.csv"))]
    for provider in providers:
        assert hasattr(provider, "fetch_matches")
        assert hasattr(provider, "fetch_match_statistics")
        assert hasattr(provider, "fetch_teams")


def test_sqlite_import_persists_rows(tmp_path) -> None:
    db_path = tmp_path / "acquisition.db"
    store = DataStore(str(db_path))
    rows = [{
        "date": "2024-08-10",
        "season": "2024/25",
        "competition": "Premier League",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_corners": 4,
        "away_corners": 3,
        "row_hash": "abc",
    }]
    inserted = store.import_rows(rows, "test_provider")
    assert inserted == 1

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    connection.close()
    assert count == 1
    store.close()
