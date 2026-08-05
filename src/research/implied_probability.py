from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def decimal_odds_to_implied_probability(odds: Any) -> float:
    """Convert decimal odds to the bookmaker-implied probability."""
    value = float(odds)
    if pd.isna(value):
        raise ValueError("Odds must not be missing")
    if value <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return float(1.0 / value)


def implied_probability_from_row(row: Any) -> float:
    odds = row.get("closing_odds") if hasattr(row, "get") else getattr(row, "closing_odds", None)
    return decimal_odds_to_implied_probability(odds)
