from __future__ import annotations

from typing import Any

import pandas as pd


def kelly_fraction(model_probability: Any, odds: Any, fraction: float = 1.0) -> float:
    """Return the Kelly fraction for decimal odds and a model probability."""
    probability = float(model_probability)
    decimal_odds = float(odds)
    if pd.isna(probability) or pd.isna(decimal_odds):
        return 0.0
    if not 0.0 <= probability <= 1.0:
        return 0.0
    if decimal_odds <= 1.0:
        return 0.0

    edge = probability * (decimal_odds - 1.0) - (1.0 - probability)
    fraction_value = max(0.0, edge / max(decimal_odds - 1.0, 1e-9))
    return float(max(0.0, fraction_value * max(0.0, fraction)))


def full_kelly(model_probability: Any, odds: Any) -> float:
    return kelly_fraction(model_probability, odds, fraction=1.0)


def half_kelly(model_probability: Any, odds: Any) -> float:
    return kelly_fraction(model_probability, odds, fraction=0.5)


def quarter_kelly(model_probability: Any, odds: Any) -> float:
    return kelly_fraction(model_probability, odds, fraction=0.25)
