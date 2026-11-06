from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.research.decision_engine import (
    MIN_EV,
    MIN_MARKET_EDGE,
    MIN_ODDS,
    MIN_PREDICTED_PROBABILITY,
    minimum_acceptable_odds,
)


MINIMUM_FORWARD_QUALIFIED_BETS = 20
REPORT_NAME = "premier_league_readiness.json"


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_ledger(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            rows.append(value)

    return rows


def _qualified_forward_bets(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []

    for row in rows:
        if str(
            row.get("result_status") or ""
        ) not in {
            "SETTLED_WIN",
            "SETTLED_LOSS",
        }:
            continue

        probability = float(
            row.get("predicted_probability")
            or 0.0
        )
        odds = float(
            row.get("observed_odds") or 0.0
        )

        if odds <= 1.0:
            continue

        edge = probability - 1.0 / odds
        expected_value = (
            probability * odds - 1.0
        )
        minimum_odds = minimum_acceptable_odds(
            probability
        )
        quality_ok = (
            float(
                row.get("data_quality_score")
                or 0.0
            )
            >= 1.0
            and not bool(
                row.get("insufficient_history")
            )
        )

        if not (
            quality_ok
            and probability
            >= MIN_PREDICTED_PROBABILITY
            and odds >= MIN_ODDS
            and edge >= MIN_MARKET_EDGE
            and expected_value >= MIN_EV
            and odds >= minimum_odds
        ):
            continue

        candidate = dict(row)
        candidate["checked_ev"] = (
            expected_value
        )
        candidates.append(candidate)

    best: dict[str, dict[str, Any]] = {}

    for row in candidates:
        key = str(
            row.get("fixture_id")
            or row.get("provider_fixture_id")
        )
        current = best.get(key)

        if (
            current is None
            or row["checked_ev"]
            > current["checked_ev"]
        ):
            best[key] = row

    return list(best.values())


def assess_premier_league_readiness(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    reports = base_dir / "reports"

    market_path = (
        reports
        / (
            "premier_league_multiseason_"
            "market_validation.json"
        )
    )
    ledger_path = (
        base_dir
        / "data/premier_league_shadow/"
        / "forward_predictions.jsonl"
    )
    settlement_path = (
        reports
        / (
            "premier_league_forward_"
            "settlement_latest.json"
        )
    )

    market = json.loads(
        market_path.read_text(encoding="utf-8")
    )
    settlement = (
        json.loads(
            settlement_path.read_text(
                encoding="utf-8"
            )
        )
        if settlement_path.exists()
        else {}
    )
    ledger = _load_ledger(ledger_path)
    forward_bets = _qualified_forward_bets(
        ledger
    )

    wins = sum(
        str(row.get("result_status"))
        == "SETTLED_WIN"
        for row in forward_bets
    )
    profit = sum(
        (
            float(row["observed_odds"]) - 1.0
            if str(row.get("result_status"))
            == "SETTLED_WIN"
            else -1.0
        )
        for row in forward_bets
    )
    forward_roi = (
        profit / len(forward_bets)
        if forward_bets
        else None
    )

    historical_summary = market.get(
        "qualified_bet_summary",
        {},
    )
    gates = {
        "historical_market_validation_passed": (
            market.get(
                "market_validation_passed"
            )
            is True
        ),
        "historical_model_beats_market": (
            float(market["model_brier"])
            < float(market["market_brier"])
        ),
        "historical_roi_positive": (
            historical_summary.get("roi")
            is not None
            and float(
                historical_summary["roi"]
            )
            > 0.0
        ),
        "forward_minimum_qualified_bets": (
            len(forward_bets)
            >= MINIMUM_FORWARD_QUALIFIED_BETS
        ),
        "forward_roi_positive": (
            forward_roi is not None
            and forward_roi > 0.0
        ),
        "forward_settlement_operational": (
            int(
                settlement.get(
                    "settled_predictions_total",
                    0,
                )
            )
            > 0
        ),
        "safety_contract_intact": (
            market.get(
                "real_betting_enabled"
            )
            is False
            and market.get(
                "bankroll_real_enabled"
            )
            is False
            and settlement.get(
                "real_betting_enabled"
            )
            is False
            and settlement.get(
                "bankroll_real_enabled"
            )
            is False
        ),
    }
    promotion_eligible = all(gates.values())

    result = {
        "run_type": (
            "premier_league_readiness"
        ),
        "promotion_eligible": (
            promotion_eligible
        ),
        "decision": (
            "READY_FOR_SEPARATE_APPROVAL"
            if promotion_eligible
            else "NOT_READY"
        ),
        "failed_gates": [
            name
            for name, passed in gates.items()
            if not passed
        ],
        "gates": gates,
        "historical_matches": int(
            market.get("matches", 0)
        ),
        "historical_model_brier": float(
            market["model_brier"]
        ),
        "historical_market_brier": float(
            market["market_brier"]
        ),
        "historical_qualified_bets": int(
            historical_summary.get("bets", 0)
        ),
        "historical_roi": (
            historical_summary.get("roi")
        ),
        "forward_observations": len(ledger),
        "forward_settled_observations": int(
            settlement.get(
                "settled_predictions_total",
                0,
            )
        ),
        "forward_qualified_bets": len(
            forward_bets
        ),
        "forward_wins": int(wins),
        "forward_profit_units": float(profit),
        "forward_roi": forward_roi,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "automatic_promotion": False,
    }

    output_path = reports / REPORT_NAME
    _atomic_json(output_path, result)
    result["report_path"] = str(output_path)
    return result
