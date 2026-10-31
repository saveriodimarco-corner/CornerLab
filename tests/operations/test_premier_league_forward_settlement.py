from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.operations import real_bet_ledger
from src.operations.premier_league_forward_settlement import (
    run_forward_settlement,
)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def resolve_fixture(self, fixture):
        self.calls += 1
        assert (
            fixture["competition"]
            == "Premier League"
        )
        return {
            "ok": True,
            "reason": "resolved",
            "result": {
                "home_score": 2,
                "away_score": 1,
                "home_corners": 6,
                "away_corners": 4,
                "total_corners": 10,
                "provider": (
                    "football-data-csv"
                ),
            },
        }


def test_settles_forward_and_only_simulation_bankroll(
    tmp_path: Path,
) -> None:
    shadow_dir = (
        tmp_path
        / "data"
        / "premier_league_shadow"
    )
    shadow_dir.mkdir(parents=True)
    ledger_path = (
        shadow_dir
        / "forward_predictions.jsonl"
    )

    common = {
        "prediction_timestamp": (
            "2026-10-31T10:00:00Z"
        ),
        "mode": "forward_validation_only",
        "selection_status": (
            "OBSERVATION_ONLY"
        ),
        "provider_fixture_id": "event-1",
        "competition": "Premier League",
        "season": "2026/27",
        "kickoff_utc": (
            "2026-10-31T12:30:00Z"
        ),
        "home_team": "Chelsea",
        "away_team": "Manchester United",
        "line": "11.5",
        "observed_odds": 1.80,
        "result_status": "PENDING",
        "actual_total_corners": None,
        "settled_at": None,
    }
    predictions = [
        {
            **common,
            "prediction_id": "under",
            "side": "UNDER",
            "predicted_probability": 0.75,
        },
        {
            **common,
            "prediction_id": "over",
            "side": "OVER",
            "predicted_probability": 0.25,
        },
    ]
    ledger_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in predictions
        ),
        encoding="utf-8",
    )

    simulation_base = (
        tmp_path
        / "data"
        / "telegram_simulation"
    )
    suggestion = {
        "fixture_id": "epl:event-1",
        "competition": "Premier League",
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": "11.5",
        "bookmaker": "bet365.it",
        "decision_timestamp": (
            "2026-10-31T10:00:00Z"
        ),
        "home_team": "Chelsea",
        "away_team": "Manchester United",
        "recommended_stake": 5.0,
        "odds_at_decision": 1.80,
        "predicted_probability": 0.75,
        "EV": 0.35,
        "quality_tier": "PREMIER FORWARD",
    }
    suggestion_id = (
        real_bet_ledger.suggestion_key(
            suggestion
        )
    )
    real_bet_ledger.record_suggestion(
        simulation_base,
        suggestion,
    )
    confirmed = (
        real_bet_ledger.confirm_bet(
            simulation_base,
            suggestion_id,
            actual_stake=5.0,
            actual_odds=1.80,
        )
    )
    assert confirmed["ok"] is True

    provider = FakeProvider()
    now = datetime(
        2026,
        10,
        31,
        16,
        0,
        tzinfo=timezone.utc,
    )
    first = run_forward_settlement(
        tmp_path,
        provider=provider,
        now=now,
    )
    second = run_forward_settlement(
        tmp_path,
        provider=provider,
        now=now,
    )

    settled = [
        json.loads(line)
        for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert first[
        "settled_predictions_now"
    ] == 2
    assert first[
        "simulation_settled_now"
    ] == 1
    assert second[
        "settled_predictions_now"
    ] == 0
    assert second[
        "simulation_settled_now"
    ] == 0
    assert provider.calls == 1

    by_side = {
        row["side"]: row
        for row in settled
    }
    assert by_side["UNDER"][
        "result_status"
    ] == "SETTLED_WIN"
    assert by_side["OVER"][
        "result_status"
    ] == "SETTLED_LOSS"
    assert by_side["UNDER"][
        "unit_profit_loss"
    ] == 0.8
    assert by_side["OVER"][
        "unit_profit_loss"
    ] == -1.0

    simulation_snapshot = (
        real_bet_ledger.get_bankroll_snapshot(
            simulation_base
        )
    )
    assert simulation_snapshot[
        "total_bankroll"
    ] == 104.0
    assert simulation_snapshot[
        "open_exposure"
    ] == 0.0

    assert not (
        tmp_path
        / "data"
        / "operations"
        / "real_bets.sqlite"
    ).exists()
