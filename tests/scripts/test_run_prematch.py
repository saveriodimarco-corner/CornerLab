from __future__ import annotations

from pathlib import Path

import pandas as pd

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
            },
            {
                "provider_fixture_id": "fix-2",
                "competition": "Premier League",
                "season": "2026",
                "kickoff_utc": "2026-08-26T18:45:00Z",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
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
                "source_fixture_id": f"evt-{fixture_id}",
            }
        ]


class _FakeRepo:
    def __init__(self, config):
        self.config = config


def _fake_manifest(**_):
    return {"git_commit": "abc123", "supported_targets": ["over_9_5", "under_9_5", "over_10_5", "under_10_5"]}


def _fake_settlement(**_):
    return {"summary": {"total_bets": 0, "profit_loss": 0.0, "roi": 0.0, "yield": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "final_bankroll": 100.0, "bankroll_start": 100.0}, "checkpoints": {}}


def test_run_prematch_orchestrates_and_persists_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prematch_runner, "run_health_check", lambda **_: {"ok": True})
    monkeypatch.setattr(prematch_runner, "CollectorRepository", _FakeRepo)
    monkeypatch.setattr(prematch_runner, "FixtureCollector", _FakeFixtureCollector)
    monkeypatch.setattr(prematch_runner, "OddsCollector", _FakeOddsCollector)
    monkeypatch.setattr(prematch_runner, "LiveProviderAdapter", _FakeAdapter)
    monkeypatch.setattr(prematch_runner, "build_production_baseline_manifest", _fake_manifest)
    monkeypatch.setattr(prematch_runner, "settle_paper_trades", _fake_settlement)
    monkeypatch.setattr(
        prematch_runner,
        "run_paper_trading",
        lambda **_: {
            "summary": {"run_id": "prematch-test", "fixtures": 2, "total_odds_rows": 2},
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
    assert result["collector"]["fixtures_fetched"] == 2
    assert result["collector"]["odds_downloaded"] == 2
    assert result["paper_trading"]["run_id"] == "prematch-test"
    assert result["production_baseline"]["git_commit"] == "abc123"
    assert (tmp_path / "reports" / "prematch_latest.json").exists()


def test_run_prematch_uses_dynamic_settled_bankroll_not_fixed_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prematch_runner, "run_health_check", lambda **_: {"ok": True})
    monkeypatch.setattr(prematch_runner, "CollectorRepository", _FakeRepo)
    monkeypatch.setattr(prematch_runner, "FixtureCollector", _FakeFixtureCollector)
    monkeypatch.setattr(prematch_runner, "OddsCollector", _FakeOddsCollector)
    monkeypatch.setattr(prematch_runner, "LiveProviderAdapter", _FakeAdapter)
    monkeypatch.setattr(prematch_runner, "build_production_baseline_manifest", _fake_manifest)
    monkeypatch.setattr(prematch_runner, "settle_paper_trades", _fake_settlement)

    settled_path = tmp_path / "reports" / "paper_trading_settled.csv"
    settled_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fixture_id": 1, "line": "9.5", "bet_result": "WIN", "bankroll_after": 118.40, "settled_timestamp": "2026-08-01T12:00:00Z"}]).to_csv(settled_path, index=False)

    captured_bankrolls = []
    monkeypatch.setattr(
        prematch_runner,
        "run_paper_trading",
        lambda **kwargs: captured_bankrolls.append(kwargs["bankroll"]) or {
            "summary": {"run_id": "prematch-test", "fixtures": 2, "total_odds_rows": 2},
            "output_paths": {
                "csv": tmp_path / "reports" / "paper_trading_current.csv",
                "parquet": tmp_path / "data" / "paper_trading" / "paper_trades_current.parquet",
                "summary": tmp_path / "reports" / "paper_trading_summary.md",
                "history": tmp_path / "data" / "paper_trading" / "run_history.jsonl",
            },
        },
    )

    prematch_runner.run_prematch(base_dir=tmp_path, output_dir=tmp_path, bankroll=100.0)

    assert captured_bankrolls == [118.40]
