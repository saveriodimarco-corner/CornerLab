from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SUPPORTED_MARKETS = {"TOTAL_CORNERS_OVER", "TOTAL_CORNERS_UNDER"}
SUPPORTED_LINES = {"8.5", "9.5", "10.5", "11.5"}
SUPPORTED_SIDES = {"OVER", "UNDER"}


@dataclass(frozen=True)
class OddsRow:
    match_id: int
    fixture_date: str
    home_team: str
    away_team: str
    bookmaker: str
    market: str
    line: str
    side: str
    opening_odds: float
    closing_odds: float
    odds_timestamp: str
    source: str
    source_fixture_id: str
    is_closing: bool
    currency: str
    import_timestamp: str
