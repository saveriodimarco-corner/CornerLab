from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.operations import premier_league_shadow


class _FakeOddsApi:
    def __init__(self) -> None:
        self._usage = {
            "x-requests-used": "20",
            "x-requests-remaining": "80",
        }

    def list_sports(self):
        return [{"key": "soccer_epl", "active": True}]

    def list_events(self, sport=None):
        assert sport == "soccer_epl"
        return [
            {
                "id": "epl-event-1",
                "commence_time": "2026-10-31T12:30:00Z",
                "home_team": "Chelsea",
                "away_team": "Manchester United",
            }
        ]

    def get_usage(self):
        return dict(self._usage)


class _FakeAdapter:
    def __init__(self, config) -> None:
        self.the_odds_api = _FakeOddsApi()
        self.last_odds_resolution = {}

    def fetch_odds(self, fixture_id):
        assert fixture_id == "epl-event-1"
        self.the_odds_api._usage = {
            "x-requests-used": "21",
            "x-requests-remaining": "79",
        }
        self.last_odds_resolution[fixture_id] = {
            "match_status": "MATCHED"
        }
        return [
            {
                "bookmaker": "Book A",
                "market": "TOTAL_CORNERS_OVER",
                "line": "9.5",
                "side": "OVER",
                "odd": 2.00,
                "source_fixture_id": fixture_id,
            },
            {
                "bookmaker": "Book A",
                "market": "TOTAL_CORNERS_OVER",
                "line": "9.5",
                "side": "OVER",
                "odd": 2.10,
                "source_fixture_id": fixture_id,
            },
            {
                "bookmaker": "Book A",
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "odd": 1.80,
                "source_fixture_id": fixture_id,
            },
        ]


class _NoQuotaOddsApi(_FakeOddsApi):
    def __init__(self) -> None:
        self._usage = {}

    def get_usage(self):
        return {}


class _NoQuotaAdapter(_FakeAdapter):
    def __init__(self, config) -> None:
        self.the_odds_api = _NoQuotaOddsApi()
        self.last_odds_resolution = {}

    def fetch_odds(self, fixture_id):
        raise AssertionError(
            "paid odds request must not run with unknown quota"
        )


def _snapshot_rows(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT bookmaker, market, line, side, decimal_odds
                FROM collector_odds_snapshots
                ORDER BY market, side
                """
            )
        ]
    finally:
        conn.close()


def test_shadow_run_is_isolated_and_deduplicates_odds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        premier_league_shadow,
        "LiveProviderAdapter",
        _FakeAdapter,
    )
    monkeypatch.setattr(
        premier_league_shadow,
        "_refresh_decision",
        lambda *_: (True, "first_snapshot"),
    )

    result = premier_league_shadow.run_premier_league_shadow(
        base_dir=tmp_path,
        output_dir=tmp_path,
    )

    expected_db = (
        tmp_path
        / "data"
        / "premier_league_shadow"
        / "collector.sqlite"
    )
    production_db = tmp_path / "data" / "collector.sqlite"

    assert result["run_type"] == "premier_league_shadow"
    assert result["mode"] == "collection_only"
    assert result["fixtures_fetched"] == 1
    assert result["odds_refresh_attempts"] == 1
    assert result["odds_downloaded"] == 3
    assert result["odds_writes"] == 2
    assert result["duplicates_removed"] == 1
    assert result["conflicting_duplicates"] == 1
    assert result["quota_remaining"] == 79
    assert result["paper_trading_enabled"] is False
    assert result["telegram_enabled"] is False
    assert result["real_betting_enabled"] is False
    assert expected_db.exists()
    assert not production_db.exists()

    rows = _snapshot_rows(expected_db)
    assert len(rows) == 2

    over_row = next(
        row for row in rows if row["side"] == "OVER"
    )
    assert over_row["decimal_odds"] == 2.10

    report_path = (
        tmp_path
        / "reports"
        / "premier_league_shadow_latest.json"
    )
    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    assert report["competition"] == "Premier League"
    assert report["telegram_enabled"] is False


def test_shadow_run_skips_paid_calls_when_quota_is_unknown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        premier_league_shadow,
        "LiveProviderAdapter",
        _NoQuotaAdapter,
    )
    monkeypatch.setattr(
        premier_league_shadow,
        "_refresh_decision",
        lambda *_: (True, "first_snapshot"),
    )

    result = premier_league_shadow.run_premier_league_shadow(
        base_dir=tmp_path,
        output_dir=tmp_path,
    )

    assert result["fixtures_fetched"] == 1
    assert result["odds_refresh_attempts"] == 0
    assert result["odds_writes"] == 0
    assert result["odds_refresh_skipped"] == 1
    assert result["odds_refresh_skip_reasons"] == {
        "quota_unknown": 1
    }

class _TwoEventOddsApi(_FakeOddsApi):
    def list_events(self, sport=None):
        assert sport == "soccer_epl"
        return [
            {
                "id": "epl-event-1",
                "commence_time": "2026-10-31T12:30:00Z",
                "home_team": "Chelsea",
                "away_team": "Manchester United",
            },
            {
                "id": "epl-event-2",
                "commence_time": "2026-10-31T15:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Liverpool",
            },
        ]


class _TwoEventAdapter(_FakeAdapter):
    fetch_calls = []

    def __init__(self, config) -> None:
        self.the_odds_api = _TwoEventOddsApi()
        self.last_odds_resolution = {}

    def fetch_odds(self, fixture_id):
        self.fetch_calls.append(fixture_id)
        self.the_odds_api._usage = {
            "x-requests-used": "21",
            "x-requests-remaining": "79",
        }
        self.last_odds_resolution[fixture_id] = {
            "match_status": "MATCHED"
        }
        return [
            {
                "bookmaker": "Book A",
                "market": "TOTAL_CORNERS_OVER",
                "line": "9.5",
                "side": "OVER",
                "odd": 2.00,
                "source_fixture_id": fixture_id,
            }
        ]


def test_default_paid_request_limit_is_enforced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _TwoEventAdapter.fetch_calls = []

    monkeypatch.setattr(
        premier_league_shadow,
        "LiveProviderAdapter",
        _TwoEventAdapter,
    )
    monkeypatch.setattr(
        premier_league_shadow,
        "_refresh_decision",
        lambda *_: (True, "first_snapshot"),
    )

    result = premier_league_shadow.run_premier_league_shadow(
        base_dir=tmp_path,
        output_dir=tmp_path,
    )

    assert result["fixtures_fetched"] == 2
    assert result["max_paid_requests"] == 1
    assert result["odds_refresh_attempts"] == 1
    assert result["odds_refresh_skipped"] == 1
    assert result["odds_refresh_skip_reasons"] == {
        "paid_request_limit": 1
    }
    assert _TwoEventAdapter.fetch_calls == [
        "epl-event-1"
    ]



def test_premier_league_shadow_uses_season_label():
    assert (
        premier_league_shadow._season_label(
            "2026-10-31T12:30:00Z"
        )
        == "2026/27"
    )
    assert (
        premier_league_shadow._season_label(
            "2027-02-01T20:00:00Z"
        )
        == "2026/27"
    )

    rows = premier_league_shadow._fixture_rows(
        [
            {
                "id": "event-1",
                "commence_time": (
                    "2026-10-31T12:30:00Z"
                ),
                "home_team": "Chelsea",
                "away_team": (
                    "Manchester United"
                ),
            }
        ]
    )

    assert rows[0]["season"] == "2026/27"
