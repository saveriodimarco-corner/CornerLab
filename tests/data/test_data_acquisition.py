from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.database import DataStore
from src.data.normalizer import NormalizationError, normalize_row, normalize_team_name
from src.data.providers import ApiFootballProvider, CSVProvider, FootballDataProvider
from src.data.quality import build_quality_report, validate_quality_rows


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
    assert "ROWS CHECKED" in report


def _historical_row(**overrides):
    row = {
        "fixture_id": 1,
        "date": "2024-08-10",
        "season": "2024/25",
        "competition": "Serie A",
        "home_team": "Inter",
        "away_team": "Roma",
        "home_goals": 1,
        "away_goals": 0,
        "home_corners": 5,
        "away_corners": 4,
        "total_corners": 9,
    }
    row.update(overrides)
    return row


def test_quality_validator_detects_duplicates_without_cross_competition_collision() -> None:
    duplicate = validate_quality_rows([_historical_row(), _historical_row(fixture_id=2)])
    cross_competition = validate_quality_rows([_historical_row(), _historical_row(competition="Premier League")])

    assert duplicate["duplicates"]
    assert not cross_competition["duplicates"]


def test_team_aliases_are_centralized_and_unknown_names_are_not_remapped() -> None:
    assert normalize_team_name("Inter Milan") == "Inter"
    assert normalize_team_name("Internazionale") == "Inter"
    assert normalize_team_name("Atalanta BC") == "Atalanta"
    assert normalize_team_name("Unknown Athletic") == "Unknown Athletic"


def test_quality_validator_rejects_impossible_values_and_total_mismatches() -> None:
    negative = validate_quality_rows([_historical_row(home_corners=-1, total_corners=3)])
    malformed = validate_quality_rows([_historical_row(home_corners="bad")])
    mismatch = validate_quality_rows([_historical_row(total_corners=99)])

    assert negative["impossible_value_errors"]
    assert malformed["impossible_value_errors"]
    assert mismatch["total_corner_mismatches"]


def test_quality_validator_reports_outliers_and_chronology_errors_without_mutation() -> None:
    rows = [_historical_row(fixture_id=index, date=f"2024-08-{10 + index:02d}", home_corners=total // 2, away_corners=total - total // 2, total_corners=total) for index, total in enumerate([8, 9, 10, 11, 50], start=1)]
    outliers = validate_quality_rows(rows)
    chronology = validate_quality_rows([_historical_row(date="2024-01-01")])

    assert outliers["outlier_warnings"]
    assert chronology["chronology_errors"]
    assert rows[-1]["total_corners"] == 50


def test_quality_validator_detects_duplicate_provider_identity_and_isolates_competitions() -> None:
    base = {"fixture_id": 11, "provider_fixture_id": "provider-11", "provider": "api-football", "date": "2026-08-10", "competition": "Serie A", "home_team": "Inter", "away_team": "Roma"}
    duplicate = validate_quality_rows([base, {**base, "fixture_id": 12}], source="live")
    cross_competition = validate_quality_rows([base, {**base, "competition": "Premier League"}], source="live")

    assert duplicate["duplicates"]
    assert not cross_competition["duplicates"]


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
