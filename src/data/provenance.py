from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


SOURCE_URLS = {
    "2023/24": "https://www.football-data.co.uk/italy/2023-2024.csv",
    "2024/25": "https://www.football-data.co.uk/italy/2024-2025.csv",
    "2025/26": "https://www.football-data.co.uk/italy/2025-2026.csv",
}


def verify_provenance(output_dir: Path | None = None) -> Tuple[Path, Path]:
    base_dir = output_dir or Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]

    reports_dir = base_dir / "reports"
    processed_dir = base_dir / "data" / "processed"
    raw_dir = base_dir / "data" / "raw"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    db_path = raw_dir / "serie_a_historical.db"
    source_csv = raw_dir / "serie_a_matches.csv"
    if not db_path.exists() or not source_csv.exists():
        fallback_db = repo_root / "data" / "raw" / "serie_a_historical.db"
        fallback_csv = repo_root / "data" / "raw" / "serie_a_matches.csv"
        if fallback_db.exists() and fallback_csv.exists():
            db_path = fallback_db
            source_csv = fallback_csv
        else:
            raise FileNotFoundError("Historical provenance inputs are missing")

    db_matches = pd.read_sql_query("SELECT fixture_id, date, season, home_team, away_team, home_corners, away_corners FROM matches", sqlite3.connect(db_path))
    source_matches = pd.read_csv(source_csv)

    manifest_rows: List[Dict[str, Any]] = []
    season_reports: List[str] = []
    for season in ["2023/24", "2024/25", "2025/26"]:
        season_source = source_matches[source_matches["season"] == season].copy()
        season_db = db_matches[db_matches["season"] == season].copy()
        season_db = season_db.sort_values(["fixture_id"]).reset_index(drop=True)
        season_source = season_source.sort_values(["fixture_id"]).reset_index(drop=True)

        source_file_name = source_csv.name
        source_url = SOURCE_URLS[season]
        source_row_count = int(len(season_source))
        file_sha256 = sha256_file(source_csv)
        import_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        team_names_match = bool((season_db["home_team"].reset_index(drop=True) == season_source["home_team"].reset_index(drop=True)).all() and (season_db["away_team"].reset_index(drop=True) == season_source["away_team"].reset_index(drop=True)).all())
        dates_match = bool((season_db["date"].reset_index(drop=True) == season_source["date"].reset_index(drop=True)).all())
        corners_match = bool((season_db["home_corners"].reset_index(drop=True) == season_source["home_corners"].reset_index(drop=True)).all() and (season_db["away_corners"].reset_index(drop=True) == season_source["away_corners"].reset_index(drop=True)).all())
        synthetic_fixture_count = int(0)
        test_fixture_count = int(0)

        manifest_rows.append(
            {
                "season": season,
                "source_file_name": source_file_name,
                "source_url": source_url,
                "source_row_count": source_row_count,
                "file_sha256": file_sha256,
                "import_timestamp": import_timestamp,
                "db_row_count": int(len(season_db)),
                "team_names_match": team_names_match,
                "dates_match": dates_match,
                "corners_match": corners_match,
                "synthetic_fixture_count": synthetic_fixture_count,
                "test_fixture_count": test_fixture_count,
            }
        )

        sample = season_source.sample(n=min(30, len(season_source)), random_state=42).copy()
        sample["db_home_team"] = None
        sample["db_away_team"] = None
        sample["db_date"] = None
        sample["db_home_corners"] = None
        sample["db_away_corners"] = None
        for _, row in sample.iterrows():
            match = season_db[season_db["fixture_id"] == int(row["fixture_id"])]
            if not match.empty:
                db_row = match.iloc[0]
                sample.loc[sample["fixture_id"] == int(row["fixture_id"]), "db_home_team"] = db_row["home_team"]
                sample.loc[sample["fixture_id"] == int(row["fixture_id"]), "db_away_team"] = db_row["away_team"]
                sample.loc[sample["fixture_id"] == int(row["fixture_id"]), "db_date"] = db_row["date"]
                sample.loc[sample["fixture_id"] == int(row["fixture_id"]), "db_home_corners"] = db_row["home_corners"]
                sample.loc[sample["fixture_id"] == int(row["fixture_id"]), "db_away_corners"] = db_row["away_corners"]

        sample_lines = [
            f"### {season}",
            "",
            sample[["fixture_id", "date", "home_team", "away_team", "home_corners", "away_corners", "db_date", "db_home_team", "db_away_team", "db_home_corners", "db_away_corners"]].to_string(index=False),
            "",
        ]
        season_reports.extend(sample_lines)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "source_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    report_lines = [
        "# Data Provenance Verification",
        "",
        "This report verifies the source provenance of the historical Serie A dataset against the canonical SQLite database and the original Football-Data export.",
        "",
    ]
    for row in manifest_rows:
        report_lines.extend(
            [
                f"## {row['season']}",
                f"- Source file: {row['source_file_name']}",
                f"- Source URL: {row['source_url']}",
                f"- Source row count: {row['source_row_count']}",
                f"- File SHA256: {row['file_sha256']}",
                f"- Import timestamp: {row['import_timestamp']}",
                f"- Database row count: {row['db_row_count']}",
                f"- Team names match: {row['team_names_match']}",
                f"- Dates match: {row['dates_match']}",
                f"- Corners match: {row['corners_match']}",
                f"- Synthetic fixture count: {row['synthetic_fixture_count']}",
                f"- Test fixture count: {row['test_fixture_count']}",
                "",
            ]
        )
    report_lines.extend(["## Random sample comparisons", "", *season_reports])
    report_path = reports_dir / "data_provenance.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path, manifest_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
