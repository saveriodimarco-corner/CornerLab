from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expected_value(model_probability: Any, odds: Any) -> float:
    """Compute the expected value of a binary wager using decimal odds."""
    probability = float(model_probability)
    decimal_odds = float(odds)
    if pd.isna(probability) or pd.isna(decimal_odds):
        raise ValueError("Model probability and odds must not be missing")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Model probability must lie in the [0, 1] range")
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return float(probability * decimal_odds - 1.0)
