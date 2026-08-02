from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_research_foundation import main as build_research_foundation


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
