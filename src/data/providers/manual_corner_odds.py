from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.odds_validator import validate_odds_dataframe


class ManualCornerOddsProvider:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path or Path("data/templates/corner_odds_import_template.csv"))

    def load(self, fixtures: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=[
                "match_id",
                "fixture_date",
                "home_team",
                "away_team",
                "bookmaker",
                "market",
                "line",
                "side",
                "opening_odds",
                "closing_odds",
                "odds_timestamp",
                "source",
                "source_fixture_id",
                "is_closing",
                "currency",
                "import_timestamp",
            ]), []

        raw = pd.read_csv(self.csv_path)
        return validate_odds_dataframe(raw, fixtures=fixtures)
