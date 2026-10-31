from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.research import (
    premier_league_forward_scoring as scoring,
)
from src.research.premier_league_forward_model import (
    FEATURES,
)


class FakeModel:
    def predict(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        assert list(frame.columns) == list(FEATURES)
        return np.full(len(frame), 10.0)


def test_forward_scoring_is_isolated_and_idempotent(
    tmp_path: Path,
) -> None:
    cache_dir = (
        tmp_path
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    cache_dir.mkdir(parents=True)

    historical = pd.DataFrame(
        [
            {
                "date": (
                    pd.Timestamp("2026-09-01")
                    + pd.Timedelta(days=index)
                ).strftime("%Y-%m-%d"),
                "season": "2026/27",
                "total_corners": 8 + index % 5,
            }
            for index in range(100)
        ]
    )
    historical.to_parquet(
        cache_dir / "matches_extended.parquet",
        index=False,
    )

    shadow_dir = (
        tmp_path
        / "data"
        / "premier_league_shadow"
    )
    shadow_dir.mkdir(parents=True)
    (
        shadow_dir / "collector.sqlite"
    ).touch()

    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "match_id": 1,
                "provider_fixture_id": "event-1",
                "competition": "Premier League",
                "season": "2026",
                "kickoff_utc": (
                    "2027-01-10T15:00:00Z"
                ),
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "status": "NS",
                "provider": "the-odds-api",
            }
        ]
    )

    odds = pd.DataFrame(
        [
            {
                "match_id": 1,
                "line": "8.5",
                "side": "OVER",
                "closing_odds": 1.90,
                "bookmaker": "Book A",
                "odds_timestamp": (
                    "2027-01-10T10:00:00Z"
                ),
            },
            {
                "match_id": 1,
                "line": "8.5",
                "side": "OVER",
                "closing_odds": 2.00,
                "bookmaker": "Book B",
                "odds_timestamp": (
                    "2027-01-10T10:00:00Z"
                ),
            },
            {
                "match_id": 1,
                "line": "11.5",
                "side": "UNDER",
                "closing_odds": 1.85,
                "bookmaker": "Book C",
                "odds_timestamp": (
                    "2027-01-10T10:00:00Z"
                ),
            },
            {
                "match_id": 1,
                "line": "9.5",
                "side": "OVER",
                "closing_odds": 2.10,
                "bookmaker": "Book D",
                "odds_timestamp": (
                    "2027-01-10T10:00:00Z"
                ),
            },
        ]
    )

    feature_values = {
        feature: 10.0
        for feature in FEATURES
    }
    features = pd.DataFrame(
        [
            {
                **feature_values,
                "match_id": 1,
                "data_quality_score": 1.0,
                "insufficient_history": False,
            }
        ]
    )

    def model_loader(base_dir):
        return FakeModel(), {
            "model_name": "test-model",
            "artifact_hash": "model-hash",
            "calibration_cutoff_date": (
                "2026-10-25"
            ),
        }

    def live_loader(base_dir, database_path=None):
        assert database_path == (
            shadow_dir / "collector.sqlite"
        )
        return fixtures.copy(), odds.copy()

    def feature_builder(history, live_fixtures):
        assert set(
            live_fixtures["season"]
        ) == {"2026/27"}
        return features.copy(), features.copy()

    now = datetime(
        2027,
        1,
        10,
        10,
        0,
        tzinfo=timezone.utc,
    )

    first = scoring.run_forward_scoring(
        tmp_path,
        now=now,
        model_loader=model_loader,
        live_loader=live_loader,
        feature_builder=feature_builder,
    )
    second = scoring.run_forward_scoring(
        tmp_path,
        now=now,
        model_loader=model_loader,
        live_loader=live_loader,
        feature_builder=feature_builder,
    )

    assert first["observations"] == 2
    assert first["new_predictions"] == 2
    assert second["observations"] == 2
    assert second["new_predictions"] == 0
    assert first["real_betting_enabled"] is False
    assert first["bankroll_real_enabled"] is False
    assert (
        first["telegram_simulation_enabled"]
        is False
    )
    assert (
        first["production_database_writes"]
        is False
    )

    ledger_path = Path(first["ledger_path"])
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(ledger_rows) == 2
    assert {
        row["line"]
        for row in ledger_rows
    } == {"8.5", "11.5"}
    assert {
        row["selection_status"]
        for row in ledger_rows
    } == {"OBSERVATION_ONLY"}

    report = json.loads(
        Path(first["report_json_path"]).read_text(
            encoding="utf-8"
        )
    )

    assert report["rows"][0][
        "real_betting_enabled"
    ] is False
    assert not (
        tmp_path
        / "data"
        / "collector.sqlite"
    ).exists()
    assert not (
        tmp_path
        / "data"
        / "operations"
        / "real_bets.sqlite"
    ).exists()
