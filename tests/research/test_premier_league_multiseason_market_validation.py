from __future__ import annotations

from src.research.premier_league_multiseason_market_validation import (
    training_seasons,
)


def test_training_periods_are_strictly_prior():
    prior = training_seasons("2025/26")
    current = training_seasons("2026/27")

    assert "2025/26" not in prior
    assert prior[-1] == "2024/25"

    assert "2026/27" not in current
    assert current[-1] == "2025/26"
