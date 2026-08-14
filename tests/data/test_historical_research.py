from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_research_foundation import main as build_research_foundation
from src.data.historical_foundation import build_premier_league_historical_database, build_serie_b_historical_database


def test_historical_research_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    build_research_foundation()

    db_path = tmp_path / "data" / "raw" / "serie_a_historical.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    season_counts = dict(conn.execute("SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season").fetchall())
    self_fixture_count = conn.execute("SELECT COUNT(*) FROM matches WHERE home_team = away_team").fetchone()[0]
    duplicate_hash_count = conn.execute("SELECT COUNT(*) FROM (SELECT row_hash FROM matches GROUP BY row_hash HAVING COUNT(*) > 1)").fetchone()[0]
    conn.close()

    assert match_count == 1140
    assert season_counts == {"2023/24": 380, "2024/25": 380, "2025/26": 380}
    assert self_fixture_count == 0
    assert duplicate_hash_count == 0

    parquet_path = tmp_path / "data" / "research" / "research_dataset.parquet"
    assert parquet_path.exists()

    dataset = pd.read_parquet(parquet_path)
    assert not dataset.empty
    assert {"date", "season", "home_team", "away_team", "home_corners", "away_corners"}.issubset(dataset.columns)

    provenance_report = tmp_path / "reports" / "data_provenance.md"
    assert provenance_report.exists()


def test_historical_research_cache_reuses_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    build_research_foundation()

    metadata_path = tmp_path / "data" / "research" / "cache" / "historical_foundation" / "metadata.json"
    assert metadata_path.exists()

    first_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert first_metadata["cache_status"] == "rebuilt"

    build_research_foundation()
    second_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert second_metadata["cache_status"] == "cache_hit"
    assert second_metadata["fingerprint"] == first_metadata["fingerprint"]


def test_serie_b_historical_database_builds_separate_artifacts(tmp_path) -> None:
    football_data_dir = tmp_path / "data" / "raw" / "football_data"
    football_data_dir.mkdir(parents=True, exist_ok=True)
    header = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HC,AC\n"

    for season_code, year in [("2324", 2023), ("2425", 2024), ("2526", 2025)]:
        rows = [header]
        for idx in range(300):
            month = (idx % 9) + 1
            day = (idx % 27) + 1
            match_year = year if month >= 7 else year + 1
            rows.append(f"I2,{day:02d}/{month:02d}/{match_year},19:30,Home{idx},Away{idx},1,0,H,{idx % 8},{(idx + 2) % 8}\n")
        (football_data_dir / f"I2_{season_code}.csv").write_text("".join(rows), encoding="utf-8")

    csv_path, parquet_path, db_path = build_serie_b_historical_database(tmp_path)

    assert csv_path.exists()
    assert parquet_path.exists()
    assert db_path.exists()

    frame = pd.read_parquet(parquet_path)
    assert len(frame) == 900
    assert set(frame["competition"].unique()) == {"Serie B"}


def test_premier_league_historical_database_builds_separate_artifacts(tmp_path) -> None:
    football_data_dir = tmp_path / "data" / "raw" / "football_data"
    football_data_dir.mkdir(parents=True, exist_ok=True)
    header = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HC,AC\n"

    for season_code, year in [("2324", 2023), ("2425", 2024), ("2526", 2025)]:
        rows = [header]
        for idx in range(300):
            month = (idx % 9) + 1
            day = (idx % 27) + 1
            match_year = year if month >= 7 else year + 1
            rows.append(f"E0,{day:02d}/{month:02d}/{match_year},20:00,Home{idx},Away{idx},1,0,H,{idx % 10},{(idx + 3) % 10}\n")
        (football_data_dir / f"E0_{season_code}.csv").write_text("".join(rows), encoding="utf-8")

    csv_path, parquet_path, db_path = build_premier_league_historical_database(tmp_path)

    assert csv_path.exists()
    assert parquet_path.exists()
    assert db_path.exists()

    frame = pd.read_parquet(parquet_path)
    assert len(frame) == 900
    assert set(frame["competition"].unique()) == {"Premier League"}
