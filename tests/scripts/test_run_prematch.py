from __future__ import annotations

from pathlib import Path

from src.operations import prematch_runner


class _FakeFixtureCollector:
    def __init__(self, config, repo):
        self._next_fixture_id = 100

    def collect_from_provider(self, payload):
        self._next_fixture_id += 1
        return {"fixture_id": self._next_fixture_id}


class _FakeOddsCollector:
    def __init__(self, config, repo):
        self.rows = []

    def collect_odds(self, payload):
        self.rows.append(payload)
        return payload


class _FakeAdapter:
    def __init__(self, config):
        self.last_odds_resolution = {}

    def fetch_fixtures(self):
        return [
            {
                "provider_fixture_id": "fix-1",
                "competition": "Serie A",
                "season": "2026",
                "kickoff_utc": "2026-08-25T18:45:00Z",
                "home_team": "Inter",
                "away_team": "Roma",
                "status": "NS",
                "provider": "api-football",
            }
        ]

    def fetch_odds(self, fixture_id):
        self.last_odds_resolution[fixture_id] = {"match_status": "MATCHED"}
        return [
            {
                "bookmaker": "book-a",
                "market": "TOTAL_CORNERS_OVER",
                "line": "9.5",
                "side": "OVER",
                "odd": 2.0,
                "source_fixture_id": "evt-1",
            }
        ]


class _FakeRepo:
    def __init__(self, config):
        self.config = config


def test_run_prematch_orchestrates_and_persists_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prematch_runner, "run_health_check", lambda **_: {"ok": True})
    monkeypatch.setattr(prematch_runner, "CollectorRepository", _FakeRepo)
    monkeypatch.setattr(prematch_runner, "FixtureCollector", _FakeFixtureCollector)
    monkeypatch.setattr(prematch_runner, "OddsCollector", _FakeOddsCollector)
    monkeypatch.setattr(prematch_runner, "LiveProviderAdapter", _FakeAdapter)
    monkeypatch.setattr(
        prematch_runner,
        "run_paper_trading",
        lambda **_: {
            "summary": {"run_id": "prematch-test", "fixtures": 1, "total_odds_rows": 1},
            "output_paths": {
                "csv": tmp_path / "reports" / "paper_trading_current.csv",
                "parquet": tmp_path / "data" / "paper_trading" / "paper_trades_current.parquet",
                "summary": tmp_path / "reports" / "paper_trading_summary.md",
                "history": tmp_path / "data" / "paper_trading" / "run_history.jsonl",
            },
        },
    )

    result = prematch_runner.run_prematch(base_dir=tmp_path, output_dir=tmp_path, bankroll=100.0)

    assert result["health_ok"] is True
    assert result["collector"]["fixtures_fetched"] == 1
    assert result["collector"]["odds_downloaded"] == 1
    assert result["paper_trading"]["run_id"] == "prematch-test"
    assert (tmp_path / "reports" / "prematch_latest.json").exists()
