from __future__ import annotations

import json
from pathlib import Path

from src.research.premier_league_readiness import (
    assess_premier_league_readiness,
)


def test_readiness_blocks_failed_market_model(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)

    market = {
        "market_validation_passed": False,
        "matches": 460,
        "model_brier": 0.222,
        "market_brier": 0.219,
        "qualified_bet_summary": {
            "bets": 20,
            "roi": -0.14,
        },
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }
    settlement = {
        "settled_predictions_total": 40,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    (
        reports
        / (
            "premier_league_multiseason_"
            "market_validation.json"
        )
    ).write_text(
        json.dumps(market),
        encoding="utf-8",
    )
    (
        reports
        / (
            "premier_league_forward_"
            "settlement_latest.json"
        )
    ).write_text(
        json.dumps(settlement),
        encoding="utf-8",
    )

    result = assess_premier_league_readiness(
        tmp_path
    )

    assert result["promotion_eligible"] is False
    assert result["decision"] == "NOT_READY"
    assert result["real_betting_enabled"] is False
    assert result["automatic_promotion"] is False
    assert (
        "historical_model_beats_market"
        in result["failed_gates"]
    )
