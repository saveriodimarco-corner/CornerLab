from __future__ import annotations

from typing import Any, Dict, List

from src.data.providers.base import BaseProvider


class ApiFootballProvider(BaseProvider):
    """A deterministic stub provider that exposes the expected internal schema."""

    name = "api_football"

    def fetch_matches(self) -> List[Dict[str, Any]]:
        return [
            {
                "date": "2024-08-17",
                "season": "2024/25",
                "competition": "Premier League",
                "home_team": "Team B",
                "away_team": "Team C",
            }
        ]

    def fetch_match_statistics(self) -> List[Dict[str, Any]]:
        return [
            {
                "date": "2024-08-17",
                "home_team": "Team B",
                "away_team": "Team C",
                "home_corners": 5,
                "away_corners": 6,
            }
        ]

    def fetch_teams(self) -> List[Dict[str, Any]]:
        return [
            {"team_name": "Team B", "competition": "Premier League"},
            {"team_name": "Team C", "competition": "Premier League"},
        ]
