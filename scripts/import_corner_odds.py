from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.data.providers.manual_corner_odds import ManualCornerOddsProvider


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS corner_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            fixture_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            line TEXT NOT NULL,
            side TEXT NOT NULL,
            opening_odds REAL NOT NULL,
            closing_odds REAL NOT NULL,
            odds_timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            source_fixture_id TEXT NOT NULL,
            is_closing INTEGER NOT NULL,
            currency TEXT NOT NULL,
            import_timestamp TEXT NOT NULL
        )
        """
    )
    connection.commit()


def import_corner_odds(csv_path: str | Path, db_path: str | Path, fixtures: pd.DataFrame | None = None) -> tuple[int, list[str]]:
    provider = ManualCornerOddsProvider(csv_path=csv_path)
    validated, errors = provider.load(fixtures=fixtures)
    if errors:
        return 0, errors

    connection = sqlite3.connect(db_path)
    try:
        ensure_schema(connection)
        connection.executemany(
            """
            INSERT INTO corner_odds (
                match_id, fixture_date, home_team, away_team, bookmaker, market, line, side,
                opening_odds, closing_odds, odds_timestamp, source, source_fixture_id, is_closing,
                currency, import_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(row["match_id"]),
                    str(row["fixture_date"]),
                    str(row["home_team"]),
                    str(row["away_team"]),
                    str(row["bookmaker"]),
                    str(row["market"]),
                    str(row["line"]),
                    str(row["side"]),
                    float(row["opening_odds"]),
                    float(row["closing_odds"]),
                    str(row["odds_timestamp"]),
                    str(row["source"]),
                    str(row["source_fixture_id"]),
                    int(str(row["is_closing"]).strip().lower() in {"1", "true", "yes", "y"}),
                    str(row["currency"]),
                    str(row["import_timestamp"]),
                )
                for _, row in validated.iterrows()
            ],
        )
        connection.commit()
        return len(validated), []
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import manual corner odds")
    parser.add_argument("--csv", default="data/templates/corner_odds_import_template.csv")
    parser.add_argument("--db", default="data/cornerlab.db")
    args = parser.parse_args()

    fixtures = pd.read_parquet("data/research/confidence_predictions.parquet")
    imported, errors = import_corner_odds(args.csv, args.db, fixtures=fixtures)
    if errors:
        print("IMPORT FAILED")
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Imported {imported} odds rows")


if __name__ == "__main__":
    main()
