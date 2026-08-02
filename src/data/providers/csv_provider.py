from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

from src.data.providers.base import BaseProvider


class CSVProvider(BaseProvider):
    """Loads match data from CSV files in the repository data folder."""

    name = "csv"

    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path or os.path.join("data", "raw", "matches.csv")

    def fetch_matches(self) -> List[Dict[str, Any]]:
        return self._read_csv()

    def fetch_match_statistics(self) -> List[Dict[str, Any]]:
        rows = self._read_csv()
        stats = []
        for row in rows:
            stats.append(
                {
                    "date": row.get("date"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_corners": row.get("home_corners"),
                    "away_corners": row.get("away_corners"),
                }
            )
        return stats

    def fetch_teams(self) -> List[Dict[str, Any]]:
        rows = self._read_csv()
        teams = []
        seen = set()
        for row in rows:
            for team_key in ("home_team", "away_team"):
                team_name = row.get(team_key)
                if team_name and team_name not in seen:
                    seen.add(team_name)
                    teams.append({"team_name": team_name, "competition": row.get("competition")})
        return teams

    def _read_csv(self) -> List[Dict[str, Any]]:
        path = Path(self.csv_path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
