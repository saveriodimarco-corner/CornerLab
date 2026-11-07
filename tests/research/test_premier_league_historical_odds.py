from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.research.premier_league_historical_odds import (
    _team_key,
    plan_historical_odds_backfill,
    run_historical_odds_backfill,
)


class FakeProvider:
    def __init__(
        self,
        remaining: int = 20000,
    ) -> None:
        self.remaining = remaining
        self._usage = {}
        self.event_calls = 0
        self.odds_calls = 0

    def list_sports(self):
        self._usage = {
            "x-requests-last": "0",
            "x-requests-remaining": str(
                self.remaining
            ),
        }
        return []

    def get_usage(self):
        return dict(self._usage)

    def _get_json(
        self,
        endpoint,
        params=None,
        bypass_cache=False,
    ):
        assert bypass_cache is True

        if endpoint.endswith("/events"):
            self.event_calls += 1
            self.remaining -= 1
            self._usage = {
                "x-requests-last": "1",
                "x-requests-remaining": str(
                    self.remaining
                ),
            }
            return {
                "timestamp": "2026-10-25T09:30:00Z",
                "data": [
                    {
                        "id": "event-1",
                        "commence_time": (
                            "2026-10-25T14:00:00Z"
                        ),
                        "home_team": (
                            "Manchester United"
                        ),
                        "away_team": "Bournemouth",
                    }
                ],
            }

        self.odds_calls += 1
        self.remaining -= 10
        self._usage = {
            "x-requests-last": "10",
            "x-requests-remaining": str(
                self.remaining
            ),
        }
        return {
            "timestamp": "2026-10-25T12:30:00Z",
            "data": {
                "id": "event-1",
                "commence_time": (
                    "2026-10-25T14:00:00Z"
                ),
                "home_team": "Manchester United",
                "away_team": "Bournemouth",
                "bookmakers": [
                    {
                        "key": "book",
                        "title": "Book",
                        "markets": [
                            {
                                "key": (
                                    "alternate_totals_corners"
                                ),
                                "last_update": (
                                    "2026-10-25T12:25:00Z"
                                ),
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "point": 8.5,
                                        "price": 1.91,
                                    },
                                    {
                                        "name": "Under",
                                        "point": 8.5,
                                        "price": 1.95,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
        }


def fixture():
    return {
        "date": "2026-10-25",
        "season": "2026/27",
        "competition": "Premier League",
        "fixture_id": 3879,
        "home_team": "Man United",
        "away_team": "Bournemouth",
        "kickoff_utc": "2026-10-25T14:00:00+00:00",
        "home_corners": 5,
        "away_corners": 4,
        "actual_total_corners": 9,
    }


def test_backfill_is_idempotent_and_isolated(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()

    first = run_historical_odds_backfill(
        tmp_path,
        provider=provider,
        fixtures=[fixture()],
        max_credit_budget=1000,
        credit_reserve=15000,
        now=datetime(
            2026,
            11,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert first["status"] == "complete"
    assert first["credits_spent_this_run"] == 11
    assert first["quota_remaining"] == 19989
    assert first["stored_odds_rows"] == 2
    assert first["real_betting_enabled"] is False
    assert first["production_database_writes"] is False
    assert provider.event_calls == 1
    assert provider.odds_calls == 1

    odds = pd.read_parquet(first["odds_path"])
    assert set(odds["side"]) == {"OVER", "UNDER"}
    assert set(odds["won"]) == {True, False}
    assert set(odds["actual_total_corners"]) == {9}

    second_provider = FakeProvider(
        remaining=19989
    )
    second = run_historical_odds_backfill(
        tmp_path,
        provider=second_provider,
        fixtures=[fixture()],
        max_credit_budget=1000,
        credit_reserve=15000,
    )

    assert second["credits_spent_this_run"] == 0
    assert second["stored_odds_rows"] == 2
    assert second_provider.event_calls == 0
    assert second_provider.odds_calls == 0


def test_guard_stops_before_spending(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()

    result = run_historical_odds_backfill(
        tmp_path,
        provider=provider,
        fixtures=[fixture()],
        max_credit_budget=10,
        credit_reserve=15000,
    )

    assert result["status"] == "stopped"
    assert result["stop_reason"] == "credit_guard"
    assert result["credits_spent_this_run"] == 0
    assert provider.event_calls == 0
    assert provider.odds_calls == 0


def test_historical_team_aliases():
    assert _team_key("West Ham United") == (
        _team_key("West Ham")
    )
    assert _team_key(
        "Wolverhampton Wanderers"
    ) == _team_key("Wolves")
    assert _team_key("Leicester City") == (
        _team_key("Leicester")
    )
