from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.premier_league_market_validation import (
    _online_predictions,
    _summarize_bets,
)


def test_online_correction_does_not_leak_within_day():
    frame = pd.DataFrame(
        {
            "date": [
                "2026-08-21",
                "2026-08-21",
                "2026-08-22",
            ],
            "actual_total_corners": [
                5.0,
                15.0,
                8.0,
            ],
        }
    )
    raw = np.asarray([10.0, 10.0, 10.0])

    corrected, trace = _online_predictions(
        frame,
        raw,
        window=20,
        shrinkage=1.0,
        minimum_history=2,
    )

    assert corrected[0] == 10.0
    assert corrected[1] == 10.0
    assert corrected[2] == 10.0
    assert trace[0]["prior_residuals"] == 0
    assert trace[1]["prior_residuals"] == 2


def test_bet_summary_uses_flat_one_unit_stakes():
    bets = pd.DataFrame(
        {
            "won": [True, False, True],
            "reference_odds": [2.0, 1.8, 1.5],
            "predicted_probability": [
                0.6,
                0.7,
                0.8,
            ],
            "expected_value": [0.2, 0.26, 0.2],
        }
    )

    result = _summarize_bets(
        bets,
        bootstrap_samples=1000,
    )

    assert result["bets"] == 3
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["profit_units"] == 0.5
    assert result["roi"] == (0.5 / 3.0)
