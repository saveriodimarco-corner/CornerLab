from __future__ import annotations

from src.research.premier_league_multiseason_market_validation import (
    training_seasons,
)


def test_training_periods_are_strictly_prior():
    confirmation = training_seasons("2024/25")
    prior = training_seasons("2025/26")
    current = training_seasons("2026/27")

    assert "2024/25" not in confirmation
    assert confirmation[-1] == "2023/24"

    assert "2025/26" not in prior
    assert prior[-1] == "2024/25"

    assert "2026/27" not in current
    assert current[-1] == "2025/26"


def test_evaluation_coverage_contract_is_locked():
    from src.research import (
        premier_league_multiseason_market_validation
        as module,
    )

    assert module.EVALUATION_SEASONS == (
        "2024/25",
        "2025/26",
        "2026/27",
    )
    assert (
        module.EXPECTED_EVALUATION_MATCHES
        == 840
    )
    assert module.COVERAGE_GATE == (
        "all_840_matches_have_odds"
    )
