from __future__ import annotations

from typing import Any, Dict, List

from src.data.providers.base import BaseProvider


class FootballDataProvider(BaseProvider):
    """A deterministic stub provider that exposes the expected internal schema."""

    name = "football_data"

    def fetch_matches(self) -> List[Dict[str, Any]]:
        return [
            {
                "date": "2024-08-10",
                "season": "2024/25",
                "competition": "Premier League",
                "home_team": "Team A",
                "away_team": "Team B",
            }
        ]

    def fetch_match_statistics(self) -> List[Dict[str, Any]]:
        return [
            {
                "date": "2024-08-10",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_corners": 7,
                "away_corners": 4,
            }
        ]

    def fetch_teams(self) -> List[Dict[str, Any]]:
        return [
            {"team_name": "Team A", "competition": "Premier League"},
            {"team_name": "Team B", "competition": "Premier League"},
        ]
