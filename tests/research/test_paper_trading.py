from __future__ import annotations

from pathlib import Path
import json
import pickle

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.engine.feature_store import FeatureStore

from src.research.paper_trading import (
    _align_feature_schema,
    _build_odds_input,
    _latest_odds_observations,
    _aggregate_market_opportunities,
    _load_authoritative_models,
    _model_registry_key,
    _resolve_market_probability,
    build_live_research_features,
    feature_row_to_model_input,
    build_live_fixture_features,
    run_paper_trading,
)
from src.research.observation_freeze import build_production_baseline_manifest, resolve_current_bankroll, settle_paper_trades, write_model_observation_artifacts, write_performance_dashboard_artifacts
from src.exceptions import BankrollUnavailableError
from src.research.paper_bet_ledger import record_bet
from src.research.paper_bet_ledger import record_bet


class _DeterministicModel:
    def predict(self, frame: pd.DataFrame):
        return [0.7] * len(frame)




def test_aggregate_market_opportunities_uses_median_and_preserves_distinct_lines() -> None:
    rows = pd.DataFrame(
        [
            {
                "row_order": 0,
                "match_id": 33,
                "fixture_id": 33,
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "bookmaker": "book-b",
                "closing_odds": 1.90,
                "opening_odds": 1.85,
                "predicted_probability": 0.72,
                "model_confidence": 0.72,
                "scoring_status": "SCORED",
                "snapshot_timestamp": "2026-09-05T15:00:00Z",
            },
            {
                "row_order": 1,
                "match_id": 33,
                "fixture_id": 33,
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "bookmaker": "book-a",
                "closing_odds": 2.00,
                "opening_odds": 1.95,
                "predicted_probability": 0.72,
                "model_confidence": 0.72,
                "scoring_status": "SCORED",
                "snapshot_timestamp": "2026-09-05T15:00:00Z",
            },
            {
                "row_order": 2,
                "match_id": 33,
                "fixture_id": 33,
                "market": "TOTAL_CORNERS_UNDER",
                "line": "10.5",
                "side": "UNDER",
                "bookmaker": "book-a",
                "closing_odds": 1.60,
                "opening_odds": 1.55,
                "predicted_probability": 0.80,
                "model_confidence": 0.72,
                "scoring_status": "SCORED",
                "snapshot_timestamp": "2026-09-05T15:00:00Z",
            },
        ]
    )

    aggregated = _aggregate_market_opportunities(rows)

    assert len(aggregated) == 2

    under_95 = aggregated.loc[
        (aggregated["fixture_id"] == 33)
        & (aggregated["line"].astype(str) == "9.5")
        & (aggregated["side"] == "UNDER")
    ].iloc[0]

    assert under_95["closing_odds"] == pytest.approx(1.95)
    assert under_95["reference_price_method"] == "MEDIAN"
    assert int(under_95["reference_bookmaker_count"]) == 2
    assert set(str(under_95["reference_bookmakers"]).split(",")) == {
        "book-a",
        "book-b",
    }

    under_105 = aggregated.loc[
        (aggregated["fixture_id"] == 33)
        & (aggregated["line"].astype(str) == "10.5")
        & (aggregated["side"] == "UNDER")
    ]

    assert len(under_105) == 1
    assert float(under_105.iloc[0]["closing_odds"]) == pytest.approx(1.60)

    reversed_result = _aggregate_market_opportunities(
        rows.iloc[::-1].reset_index(drop=True)
    )

    comparable = [
        "fixture_id",
        "market",
        "line",
        "side",
        "closing_odds",
        "reference_price_method",
        "reference_bookmaker_count",
        "reference_bookmakers",
    ]

    left = aggregated[comparable].sort_values(
        ["fixture_id", "market", "line", "side"]
    ).reset_index(drop=True)

    right = reversed_result[comparable].sort_values(
        ["fixture_id", "market", "line", "side"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)

def test_latest_odds_observations_keeps_only_latest_snapshot_per_bookmaker_market_line_side() -> None:
    odds = pd.DataFrame(
        [
            {
                "fixture_id": 33,
                "bookmaker": "BetMGM",
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "snapshot_timestamp": "2026-09-05T13:00:00Z",
                "import_timestamp": "2026-09-05T13:00:01Z",
                "decimal_odds": 1.60,
            },
            {
                "fixture_id": 33,
                "bookmaker": "BetMGM",
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "snapshot_timestamp": "2026-09-05T15:00:00Z",
                "import_timestamp": "2026-09-05T15:00:01Z",
                "decimal_odds": 1.69,
            },
            {
                "fixture_id": 33,
                "bookmaker": "BetRivers",
                "market": "TOTAL_CORNERS_UNDER",
                "line": "9.5",
                "side": "UNDER",
                "snapshot_timestamp": "2026-09-05T14:00:00Z",
                "import_timestamp": "2026-09-05T14:00:01Z",
                "decimal_odds": 1.62,
            },
            {
                "fixture_id": 33,
                "bookmaker": "BetMGM",
                "market": "TOTAL_CORNERS_UNDER",
                "line": "10.5",
                "side": "UNDER",
                "snapshot_timestamp": "2026-09-05T14:30:00Z",
                "import_timestamp": "2026-09-05T14:30:01Z",
                "decimal_odds": 1.50,
            },
        ]
    )

    latest = _latest_odds_observations(odds)

    assert len(latest) == 3

    betmgm_95 = latest[
        (latest["bookmaker"] == "BetMGM")
        & (latest["line"].astype(str) == "9.5")
    ].iloc[0]

    assert betmgm_95["snapshot_timestamp"] == "2026-09-05T15:00:00Z"
    assert float(betmgm_95["decimal_odds"]) == 1.69

    assert set(latest["bookmaker"]) == {"BetMGM", "BetRivers"}
    assert set(latest["line"].astype(str)) == {"9.5", "10.5"}

def test_build_live_fixture_features_uses_historical_state() -> None:
    historical_matches = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "date": "2024-08-01",
                "season": "2024/25",
                "home_team": "Inter",
                "away_team": "Juventus",
                "home_corners": 6,
                "away_corners": 4,
                "total_corners": 10,
            },
            {
                "fixture_id": 2,
                "date": "2024-08-08",
                "season": "2024/25",
                "home_team": "Juventus",
                "away_team": "Inter",
                "home_corners": 5,
                "away_corners": 5,
                "total_corners": 10,
            },
            {
                "fixture_id": 3,
                "date": "2024-08-15",
                "season": "2024/25",
                "home_team": "Inter",
                "away_team": "Napoli",
                "home_corners": 7,
                "away_corners": 3,
                "total_corners": 10,
            },
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 10,
                "provider_fixture_id": "live-10",
                "competition": "Serie A",
                "season": "2026",
                "kickoff_utc": "2026-08-22T16:30:00+00:00",
                "home_team": "Inter",
                "away_team": "Juventus",
                "status": "NS",
                "provider": "api-football",
            }
        ]
    )

    feature_frame, confidence_frame = build_live_fixture_features(historical_matches, fixtures)

    assert len(feature_frame) == 1
    assert len(confidence_frame) == 1
    assert float(feature_frame.iloc[0]["expected_total_corner"]) > 0.0
    assert confidence_frame.iloc[0]["home_matches_played"] >= 0
    assert confidence_frame.iloc[0]["combined_volatility"] >= 0


def test_run_paper_trading_writes_current_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 900001,
                "match_id": 900001,
                "provider_fixture_id": "evt-deterministic-1",
                "competition": "Serie A",
                "season": "2026/27",
                "kickoff_utc": "2026-08-25T18:45:00Z",
                "date": "2026-08-25",
                "home_team": "Inter",
                "away_team": "Roma",
                "status": "NS",
                "provider": "api-football",
            }
        ]
    )

    odds_rows = []

    for line in ["8.5", "9.5", "10.5", "11.5"]:
        for side, market, price in [
            ("OVER", "TOTAL_CORNERS_OVER", 1.95),
            ("UNDER", "TOTAL_CORNERS_UNDER", 1.95),
        ]:
            odds_rows.append(
                {
                    "match_id": 900001,
                    "fixture_date": "2026-08-25",
                    "home_team": "Inter",
                    "away_team": "Roma",
                    "bookmaker": "TESTBOOK",
                    "market": market,
                    "line": line,
                    "side": side,
                    "opening_odds": price,
                    "closing_odds": price,
                    "odds_timestamp": "2026-08-25T10:00:00Z",
                    "source": "the-odds-api",
                    "source_fixture_id": "evt-deterministic-1",
                    "is_closing": True,
                    "currency": "EUR",
                    "import_timestamp": "2026-08-25T10:00:00Z",
                }
            )

    # Same economic opportunity as TESTBOOK UNDER 9.5, but from a second
    # external bookmaker. It must contribute to the reference market price,
    # not become a second paper bet.
    odds_rows.append(
        {
            "match_id": 900001,
            "fixture_date": "2026-08-25",
            "home_team": "Inter",
            "away_team": "Roma",
            "bookmaker": "SECOND_BOOK",
            "market": "TOTAL_CORNERS_UNDER",
            "line": "9.5",
            "side": "UNDER",
            "opening_odds": 2.05,
            "closing_odds": 2.05,
            "odds_timestamp": "2026-08-25T10:00:00Z",
            "source": "the-odds-api",
            "source_fixture_id": "evt-deterministic-1",
            "is_closing": True,
            "currency": "EUR",
            "import_timestamp": "2026-08-25T10:00:00Z",
        }
    )

    odds = pd.DataFrame(odds_rows)

    monkeypatch.setattr(
        "src.research.paper_trading._load_live_fixtures_and_odds",
        lambda _base_dir: (fixtures.copy(), odds.copy()),
    )

    result = run_paper_trading(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
        bankroll=100.0,
    )

    report = result["report"]
    assert not report.empty

    # 4 lines x OVER/UNDER = 8 economic opportunities.
    # The second bookmaker above must not create a ninth decision row.
    assert len(report) == 8

    under_95 = report.loc[
        (report["fixture_id"].astype(int) == 900001)
        & (report["market"].astype(str) == "TOTAL_CORNERS_UNDER")
        & (report["line"].astype(str) == "9.5")
        & (report["side"].astype(str) == "UNDER")
    ]

    assert len(under_95) == 1
    assert float(under_95.iloc[0]["closing_odds"]) == pytest.approx(2.00)
    assert under_95.iloc[0]["reference_price_method"] == "MEDIAN"
    assert int(under_95.iloc[0]["reference_bookmaker_count"]) == 2
    assert set(
        str(under_95.iloc[0]["reference_bookmakers"]).split(",")
    ) == {"TESTBOOK", "SECOND_BOOK"}
    assert set(report["decision"].unique()).issubset({"PLAY", "LOW CONFIDENCE", "NO BET", "MODEL_UNAVAILABLE"})
    assert set(report.loc[report["decision"] == "MODEL_UNAVAILABLE", "decision_reason"].unique()).issubset({"NO_ACCEPTED_MODEL", "MODEL_INPUT_FAILED", "UNSUPPORTED_MARKET"})

    from src.research.paper_bet_ledger import list_bets
    play_count = int((report["decision"] == "PLAY").sum())
    assert play_count > 0
    assert len(list_bets(tmp_path)) == play_count

    second_result = run_paper_trading(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
        bankroll=100.0,
    )
    second_report = second_result["report"]

    assert int((second_report["decision"] == "PLAY").sum()) == 0
    assert int(
        (second_report["decision_reason"] == "ALREADY_PAPER_TRADED").sum()
    ) == play_count
    assert len(list_bets(tmp_path)) == play_count

    from src.research.paper_bet_ledger import list_bets
    assert len(list_bets(tmp_path)) == int((report["decision"] == "PLAY").sum())
    # Serie A production contract:
    # the authoritative total-corners Poisson model scores all four
    # operational lines. Unsupported rows may belong to competitions
    # without an accepted production count model (e.g. Premier League).
    serie_a = report.loc[
        report["competition"].astype(str).eq("Serie A")
    ].copy()

    for target_name in [
        "over_8_5",
        "over_9_5",
        "over_10_5",
        "over_11_5",
    ]:
        target_rows = serie_a.loc[
            serie_a["target_name"].astype(str).eq(target_name)
        ]

        assert not target_rows.empty
        assert target_rows["market_support_status"].eq("SUPPORTED").all()
        assert target_rows["scoring_status"].eq("SCORED").all()
        assert target_rows["model_version"].eq("poisson_regression").all()

    unsupported_operational = report.loc[
        report["target_name"].isin(
            ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        )
        & report["market_support_status"].eq("UNSUPPORTED")
    ]

    assert not unsupported_operational[
        "competition"
    ].astype(str).eq("Serie A").any()
    assert (report["market"] == "TOTAL_CORNERS_UNDER").any()
    assert "run_id" in report.columns
    assert "decision_timestamp" in report.columns
    assert "odds_at_decision" in report.columns
    assert "quality_tier" in report.columns
    assert "confidence" in report.columns
    assert "market_implied_probability" in report.columns
    assert "edge" in report.columns
    assert "decision_state" in report.columns
    assert "provider_event_id" in report.columns

    assert (tmp_path / "data" / "paper_trading" / "paper_trades_current.parquet").exists()
    assert (tmp_path / "reports" / "paper_trading_current.csv").exists()
    assert (tmp_path / "reports" / "paper_trading_summary.md").exists()
    assert (tmp_path / "data" / "paper_trading" / "run_history.jsonl").exists()
    assert (tmp_path / "data" / "paper_trading" / "runs").exists()


def test_production_manifest_and_settlement_outputs_are_written(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assert record_bet(tmp_path, {
        "run_id": "r1",
        "decision": "PLAY",
        "market_support_status": "SUPPORTED",
        "competition": "Serie A",
        "fixture_id": 1,
        "provider_event_id": "evt-1",
        "home_team": "Inter",
        "away_team": "Roma",
        "kickoff_utc": "2026-08-25T18:45:00Z",
        "market": "TOTAL_CORNERS_OVER",
        "line": "9.5",
        "side": "OVER",
        "bookmaker": "book-a",
        "odds_at_decision": 2.0,
        "closing_odds": 1.9,
        "predicted_probability": 0.62,
        "fair_odds": 1.61,
        "market_implied_probability": 0.5,
        "edge": 0.12,
        "ev": 0.24,
        "decision_confidence_score": 72.0,
        "quality_tier": "TOP",
        "recommended_stake": 2.0,
        "model_artifact": "artifact.pkl",
        "model_hash": "hash123",
        "feature_schema_hash": "schema123",
        "target_name": "over_9_5",
        "stake": 2.0,
        "decision_timestamp": "2026-08-25T10:00:00Z",
    }) is True
    db_path = tmp_path / "data" / "collector.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE collector_results (fixture_id INTEGER UNIQUE, home_score INTEGER, away_score INTEGER, home_corners INTEGER, away_corners INTEGER, total_corners INTEGER, settled_at TEXT, provider TEXT)")
        conn.execute("INSERT INTO collector_results VALUES (1, 1, 0, 6, 5, 11, '2026-08-26T10:00:00Z', 'api-football')")
        conn.execute("CREATE TABLE collector_fixtures (fixture_id INTEGER PRIMARY KEY, provider_fixture_id TEXT, competition TEXT, season TEXT, kickoff_utc TEXT, home_team TEXT, away_team TEXT, status TEXT, provider TEXT, created_at TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO collector_fixtures VALUES (1, 'evt-1', 'Serie A', '2026/27', '2026-08-25T18:45:00Z', 'Inter', 'Roma', 'FT', 'api-football', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')")
        conn.commit()
    finally:
        conn.close()

    manifest = build_production_baseline_manifest(
        base_dir=Path.cwd(),
        output_dir=tmp_path,
    )
    settlement = settle_paper_trades(base_dir=tmp_path, output_dir=tmp_path, bankroll_start=100.0)

    assert (tmp_path / "reports" / "production_baseline_serie_a.json").exists()
    assert manifest["supported_targets"] == [
        "over_8_5",
        "under_8_5",
        "over_9_5",
        "under_9_5",
        "over_10_5",
        "under_10_5",
        "over_11_5",
        "under_11_5",
    ]

    serie_a_registry = manifest[
        "supported_market_registry"
    ]["Serie A"]["model_registry"]

    for target_name in manifest["supported_targets"]:
        model_info = serie_a_registry[target_name]

        assert model_info["model_name"] == "poisson_regression"
        assert model_info["derived_from"] == "actual_total_corners"

    assert (
        manifest["model_artifacts"][0]["target_name"]
        == "actual_total_corners"
    )
    assert (tmp_path / "reports" / "paper_trading_performance.json").exists()
    assert settlement["summary"]["total_bets"] == 1
    assert settlement["summary"]["profit_loss"] > 0



def test_paper_ledger_rejects_duplicate_economic_opportunity(tmp_path: Path) -> None:
    base = {
        "fixture_id": 1,
        "market": "TOTAL_CORNERS_OVER",
        "side": "OVER",
        "line": "9.5",
        "recommended_stake": 2.0,
        "bookmaker": "book-a",
    }

    assert record_bet(tmp_path, base) is True

    duplicate = dict(base, bookmaker="book-b")
    assert record_bet(tmp_path, duplicate) is False

    distinct_line = dict(base, line="10.5", bookmaker="book-a")
    assert record_bet(tmp_path, distinct_line) is True


def test_performance_dashboard_artifacts_are_deterministic_and_observational(tmp_path: Path) -> None:
    payload = {
        "summary": {"bankroll_start": 100.0, "final_bankroll": 113.0, "total_bets": 4, "wins": 3, "losses": 1, "hit_rate": 0.75, "profit_loss": 13.0, "roi": 0.13, "yield": 13.0 / 40.0, "max_drawdown": 0.1},
        "settled_rows": [
            {"settled_timestamp": "2026-08-01T12:00:00Z", "target_name": "over_9_5", "side": "OVER", "quality_tier": "TOP", "bet_result": "WIN", "stake": 10.0, "profit_loss": 10.0, "odds_at_decision": 2.0, "EV": 0.1, "confidence": 70.0, "CLV": 0.01},
            {"settled_timestamp": "2026-08-02T12:00:00Z", "target_name": "over_9_5", "side": "OVER", "quality_tier": "BUONA", "bet_result": "WIN", "stake": 10.0, "profit_loss": 8.0, "odds_at_decision": 1.8, "EV": 0.08, "confidence": 68.0, "CLV": 0.02},
            {"settled_timestamp": "2026-08-10T12:00:00Z", "target_name": "under_10_5", "side": "UNDER", "quality_tier": "MARGINALE", "bet_result": "LOSS", "stake": 10.0, "profit_loss": -10.0, "odds_at_decision": 2.0, "EV": -0.1, "confidence": 61.0, "CLV": -0.01},
            {"settled_timestamp": "2026-08-11T12:00:00Z", "target_name": "under_10_5", "side": "UNDER", "quality_tier": "TOP", "bet_result": "WIN", "stake": 10.0, "profit_loss": 5.0, "odds_at_decision": 1.5, "EV": 0.05, "confidence": 72.0, "CLV": 0.0},
            {"settled_timestamp": "2026-08-12T12:00:00Z", "target_name": "over_10_5", "side": "OVER", "quality_tier": "TOP", "bet_result": "PENDING", "stake": 10.0, "profit_loss": 0.0, "odds_at_decision": 2.0, "EV": 0.1, "confidence": 75.0, "CLV": None},
        ],
    }

    paths = write_performance_dashboard_artifacts(payload, reports_dir=tmp_path, now=pd.Timestamp("2026-08-14T12:00:00Z"))
    dashboard = json.loads(paths["json"].read_text(encoding="utf-8"))
    rows = pd.read_csv(paths["csv"])

    assert all(path.exists() for path in paths.values())
    assert dashboard["settled_bets"] == 4
    assert dashboard["pending_bets"] == 1
    assert dashboard["periods"]["all"]["roi"] == 0.13
    assert dashboard["periods"]["all"]["profit_loss"] == 13.0
    assert dashboard["periods"]["all"]["win_rate"] == 0.75
    assert dashboard["periods"]["all"]["yield"] == 13.0 / 40.0
    assert dashboard["periods"]["all"]["final_bankroll"] == 113.0
    assert dashboard["periods"]["all"]["max_drawdown"] == 10.0 / 118.0
    assert dashboard["periods"]["all"]["longest_winning_streak"] == 2
    assert dashboard["periods"]["all"]["longest_losing_streak"] == 1
    assert len(dashboard["weekly_report"]) == 3
    assert len(dashboard["monthly_report"]) == 1
    assert dashboard["market_breakdown"]["over_9_5"]["total_bets"] == 2
    assert dashboard["side_breakdown"]["OVER"]["total_bets"] == 2
    assert dashboard["quality_breakdown"]["TOP"]["total_bets"] == 2
    assert len(rows) == 4

    empty_paths = write_performance_dashboard_artifacts({"summary": {"bankroll_start": 100.0}, "settled_rows": []}, reports_dir=tmp_path / "empty", now=pd.Timestamp("2026-08-14T12:00:00Z"))
    empty_dashboard = json.loads(empty_paths["json"].read_text(encoding="utf-8"))
    assert empty_dashboard["settled_bets"] == 0


def test_model_observation_artifacts_describe_settled_records_without_mutation(tmp_path: Path) -> None:
    settled_rows = [
        {"fixture_id": 1, "target_name": "over_9_5", "side": "OVER", "quality_tier": "TOP", "bet_result": "WIN", "stake": 10.0, "profit_loss": 10.0, "odds_at_decision": 2.0, "predicted_probability": 0.72, "EV": 0.12, "settled_timestamp": "2026-08-01T12:00:00Z", "model_hash": "frozen"},
        {"fixture_id": 2, "target_name": "under_9_5", "side": "UNDER", "quality_tier": "BUONA", "bet_result": "LOSS", "stake": 10.0, "profit_loss": -10.0, "odds_at_decision": 2.0, "predicted_probability": 0.62, "EV": 0.08, "settled_timestamp": "2026-08-08T12:00:00Z", "model_hash": "frozen"},
        {"fixture_id": 3, "target_name": "over_10_5", "side": "OVER", "quality_tier": "MARGINALE", "bet_result": "WIN", "stake": 20.0, "profit_loss": 20.0, "odds_at_decision": 2.0, "predicted_probability": 0.76, "EV": 0.15, "settled_timestamp": "2026-08-15T12:00:00Z", "model_hash": "frozen"},
        {"fixture_id": 4, "target_name": "under_10_5", "side": "UNDER", "quality_tier": "TOP", "bet_result": "PENDING", "stake": 20.0, "profit_loss": 0.0, "odds_at_decision": 2.0, "predicted_probability": 0.68, "EV": 0.10, "settled_timestamp": "2026-08-16T12:00:00Z", "model_hash": "frozen"},
    ]
    original = [dict(row) for row in settled_rows]

    paths = write_model_observation_artifacts({"summary": {"bankroll_start": 100.0}, "settled_rows": settled_rows}, reports_dir=tmp_path)
    observation = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert all(path.exists() for path in paths.values())
    assert observation["settled_scored_bets"] == 3
    assert observation["economic"]["profit_loss"] == 20.0
    assert observation["economic"]["roi"] == 0.2
    assert observation["brier_score"] == pytest.approx(((0.72 - 1.0) ** 2 + 0.62**2 + (0.76 - 1.0) ** 2) / 3)
    assert observation["by_market"]["over_9_5"]["bets"] == 1
    assert observation["by_side"]["OVER"]["bets"] == 2
    assert observation["by_quality"]["TOP"]["bets"] == 1
    assert next(bucket for bucket in observation["calibration"] if bucket["bucket"] == "0.70-0.75")["count"] == 1
    assert observation["sample_warning"] == "INSUFFICIENT SAMPLE"
    assert settled_rows == original


def test_under_probability_is_complement_of_over_probability() -> None:
    assert _resolve_market_probability("TOTAL_CORNERS_OVER", "OVER", 0.62) == 0.62
    assert _resolve_market_probability("TOTAL_CORNERS_UNDER", "UNDER", 0.62) == 0.38


def _write_settled_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _settled_row(fixture_id: int, settled_timestamp: str, bankroll_after: float, bet_result: str = "WIN", line: str = "9.5") -> dict:
    return {"fixture_id": fixture_id, "line": line, "bet_result": bet_result, "bankroll_after": bankroll_after, "settled_timestamp": settled_timestamp}


def test_resolve_current_bankroll_defaults_to_100_when_no_settled_bets(tmp_path: Path) -> None:
    assert resolve_current_bankroll(tmp_path, default_bankroll=100.0) == 100.0

    _write_settled_ledger(tmp_path / "reports" / "paper_trading_settled.csv", [_settled_row(1, "2026-08-01T12:00:00Z", 105.0, bet_result="PENDING")])
    assert resolve_current_bankroll(tmp_path, default_bankroll=100.0) == 100.0


def test_resolve_current_bankroll_uses_latest_settled_chronology(tmp_path: Path) -> None:
    _write_settled_ledger(
        tmp_path / "reports" / "paper_trading_settled.csv",
        [
            _settled_row(2, "2026-08-02T12:00:00Z", 94.20),
            _settled_row(1, "2026-08-01T12:00:00Z", 100.0),
        ],
    )
    assert resolve_current_bankroll(tmp_path, default_bankroll=100.0) == 94.20


def test_resolve_current_bankroll_reflects_growth_to_118_40(tmp_path: Path) -> None:
    _write_settled_ledger(
        tmp_path / "reports" / "paper_trading_settled.csv",
        [
            _settled_row(1, "2026-08-01T12:00:00Z", 105.0),
            _settled_row(2, "2026-08-02T12:00:00Z", 118.40),
        ],
    )
    assert resolve_current_bankroll(tmp_path, default_bankroll=100.0) == 118.40


def test_resolve_current_bankroll_fails_closed_on_corrupt_data(tmp_path: Path) -> None:
    _write_settled_ledger(tmp_path / "reports" / "paper_trading_settled.csv", [_settled_row(1, "2026-08-01T12:00:00Z", -5.0)])
    with pytest.raises(BankrollUnavailableError):
        resolve_current_bankroll(tmp_path, default_bankroll=100.0)

    _write_settled_ledger(tmp_path / "reports" / "paper_trading_settled.csv", [_settled_row(1, "2026-08-01T12:00:00Z", float("nan"))])
    with pytest.raises(BankrollUnavailableError):
        resolve_current_bankroll(tmp_path, default_bankroll=100.0)


def test_invalid_fixture_model_input_is_non_play_without_blocking_valid_fixture() -> None:
    feature_frame = pd.DataFrame(
        [
            {"match_id": 1, "fixture_id": 1, "competition": "Serie A", "expected_total_corner": 10.0},
            {"match_id": 2, "fixture_id": 2, "competition": "Serie A", "expected_total_corner": "bad"},
        ]
    )
    confidence_frame = pd.DataFrame(
        [
            {"match_id": 1, "model_confidence": 0.8, "confidence_score": 80.0},
            {"match_id": 2, "model_confidence": 0.8, "confidence_score": 80.0},
        ]
    )
    validated_odds = pd.DataFrame(
        [
            {"match_id": 1, "bookmaker": "book", "market": "TOTAL_CORNERS_OVER", "line": "9.5", "side": "OVER", "closing_odds": 2.0},
            {"match_id": 2, "bookmaker": "book", "market": "TOTAL_CORNERS_OVER", "line": "9.5", "side": "OVER", "closing_odds": 2.0},
        ]
    )
    model_bundle = {
        "serie_a/over_9_5": {
            "schema": ["expected_total_corner"],
            "model": _DeterministicModel(),
            "artifact_path": "model.pkl",
            "artifact_hash": "hash",
            "model_version": "test",
        }
    }

    rows = _build_odds_input(feature_frame, confidence_frame, validated_odds, model_bundle)

    assert len(rows) == 2
    assert rows.loc[rows["match_id"] == 1, "scoring_status"].iloc[0] == "SCORED"
    invalid = rows.loc[rows["match_id"] == 2].iloc[0]
    assert invalid["decision"] == "MODEL_UNAVAILABLE"
    assert invalid["decision_reason"] == "MODEL_INPUT_FAILED"


def test_invalid_live_fixture_is_recorded_as_non_play_without_stopping_scoring() -> None:
    fixtures = pd.DataFrame(
        [
            {"fixture_id": 99, "provider_fixture_id": "fixture-99", "kickoff_utc": "bad-time", "competition": "Serie A", "season": "2026/27", "home_team": "Inter", "away_team": "Roma"},
        ]
    )
    feature_frame, confidence_frame = build_live_research_features(pd.DataFrame(), fixtures)
    validated_odds = pd.DataFrame(
        [
            {"match_id": 99, "bookmaker": "book", "market": "TOTAL_CORNERS_OVER", "line": "9.5", "side": "OVER", "closing_odds": 2.0},
        ]
    )
    model_bundle = {"serie_a/over_9_5": {"schema": ["expected_total_corner"], "model": _DeterministicModel(), "artifact_path": "model.pkl", "artifact_hash": "hash", "model_version": "test"}}

    rows = _build_odds_input(feature_frame, confidence_frame, validated_odds, model_bundle, invalid_fixtures=feature_frame.attrs["invalid_fixtures"])

    assert len(rows) == 1
    assert rows.iloc[0]["decision"] == "MODEL_UNAVAILABLE"
    assert rows.iloc[0]["decision"] != "PLAY"
    assert rows.iloc[0]["decision_reason"] == "MODEL_INPUT_FAILED"


def test_model_registry_resolves_by_competition_and_target(tmp_path: Path) -> None:
    research_dir = tmp_path / "data" / "research"
    models_dir = tmp_path / "models" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model = LogisticRegression().fit(pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]}), [0, 1])
    (research_dir / "best_models.json").write_text(json.dumps({"over_9_5": {"accepted": True, "model_name": "negative_binomial_probability"}}), encoding="utf-8")
    (research_dir / "best_models_serie_b.json").write_text(json.dumps({"over_9_5": {"accepted": True, "model_name": "negative_binomial_probability"}}), encoding="utf-8")
    (models_dir / "over_9_5_negative_binomial_probability.pkl").write_bytes(pickle.dumps(model))
    (models_dir / "serie_b_over_9_5_negative_binomial_probability.pkl").write_bytes(pickle.dumps(model))

    bundles = _load_authoritative_models(tmp_path)

    assert _model_registry_key("serie_a", "over_9_5") in bundles
    assert _model_registry_key("serie_b", "over_9_5") in bundles


def test_feature_store_league_state_is_isolated_by_competition() -> None:
    store = FeatureStore()
    prior_state = {}
    team_history = {}
    league_state = {}

    serie_a_match = pd.Series(
        {
            "date": "2026-01-01",
            "season": "2025/26",
            "competition": "Serie A",
            "home_team": "Inter",
            "away_team": "Roma",
            "home_corners": 9.0,
            "away_corners": 1.0,
        }
    )
    serie_b_match = pd.Series(
        {
            "date": "2026-01-02",
            "season": "2025/26",
            "competition": "Serie B",
            "home_team": "Palermo",
            "away_team": "Bari",
            "home_corners": 1.0,
            "away_corners": 1.0,
        }
    )

    store._update_state(prior_state, team_history, serie_a_match, league_state=league_state, season="2025/26", competition="Serie A")  # type: ignore[attr-defined]
    percentile_before = store._league_percentile(league_state, "serie_a::2025/26", 9.0, metric="attack")  # type: ignore[attr-defined]
    store._update_state(prior_state, team_history, serie_b_match, league_state=league_state, season="2025/26", competition="Serie B")  # type: ignore[attr-defined]
    percentile_after = store._league_percentile(league_state, "serie_a::2025/26", 9.0, metric="attack")  # type: ignore[attr-defined]

    assert percentile_before == percentile_after
    assert "serie b::2025/26" in league_state


def test_live_research_features_preserve_competition_identity_and_serie_a_schema() -> None:
    historical_matches = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "date": "2024-08-01",
                "season": "2024/25",
                "competition": "Serie A",
                "home_team": "Inter",
                "away_team": "Roma",
                "home_corners": 7,
                "away_corners": 4,
                "total_corners": 11,
            },
            {
                "fixture_id": 2,
                "date": "2024-08-02",
                "season": "2024/25",
                "competition": "Serie A",
                "home_team": "Milan",
                "away_team": "Lazio",
                "home_corners": 6,
                "away_corners": 5,
                "total_corners": 11,
            },
            {
                "fixture_id": 1,
                "date": "2024-08-01",
                "season": "2024/25",
                "competition": "Premier League",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_corners": 5,
                "away_corners": 3,
                "total_corners": 8,
            },
            {
                "fixture_id": 2,
                "date": "2024-08-02",
                "season": "2024/25",
                "competition": "Premier League",
                "home_team": "Liverpool",
                "away_team": "Everton",
                "home_corners": 8,
                "away_corners": 2,
                "total_corners": 10,
            },
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "provider_fixture_id": "sa-live-1",
                "competition": "Serie A",
                "season": "2026/27",
                "kickoff_utc": "2026-08-22T16:30:00+00:00",
                "home_team": "Inter",
                "away_team": "Roma",
                "status": "NS",
                "provider": "api-football",
            },
            {
                "fixture_id": 2,
                "provider_fixture_id": "epl-live-2",
                "competition": "Premier League",
                "season": "2026/27",
                "kickoff_utc": "2026-08-22T18:30:00+00:00",
                "home_team": "Liverpool",
                "away_team": "Everton",
                "status": "NS",
                "provider": "api-football",
            },
        ]
    )

    feature_frame, confidence_frame = build_live_research_features(historical_matches, fixtures)

    assert not feature_frame.empty
    assert len(feature_frame) == 2
    assert len(confidence_frame) == 2
    assert set(feature_frame["competition"].astype(str).unique()) == {"Serie A", "Premier League"}

    serie_a_row = feature_frame.loc[
        (feature_frame["competition"].astype(str) == "Serie A")
        & (feature_frame["home_team"].astype(str) == "Inter")
        & (feature_frame["away_team"].astype(str) == "Roma")
    ]
    premier_row = feature_frame.loc[
        (feature_frame["competition"].astype(str) == "Premier League")
        & (feature_frame["home_team"].astype(str) == "Liverpool")
        & (feature_frame["away_team"].astype(str) == "Everton")
    ]
    assert len(serie_a_row) == 1
    assert len(premier_row) == 1

    bundles = _load_authoritative_models(Path.cwd())
    for target_name in ["over_9_5", "over_10_5"]:
        registry_key = _model_registry_key("serie_a", target_name)
        assert registry_key in bundles
        model_input = pd.DataFrame([feature_row_to_model_input(serie_a_row.iloc[0], target_name)])
        schema_ok, _ = _align_feature_schema(model_input, bundles[registry_key]["schema"])
        assert schema_ok


def test_settlement_uses_one_canonical_settled_chronology_for_bankroll(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "run_id": "r1",
            "decision": "PLAY",
            "market_support_status": "SUPPORTED",
            "competition": "Serie A",
            "fixture_id": 1,
            "provider_event_id": "evt-1",
            "home_team": "Inter",
            "away_team": "Roma",
            "kickoff_utc": "2026-08-25T18:45:00Z",
            "market": "TOTAL_CORNERS_OVER",
            "line": "9.5",
            "side": "OVER",
            "bookmaker": "MARKET_MEDIAN",
            "odds_at_decision": 2.0,
            "closing_odds": 2.0,
            "predicted_probability": 0.62,
            "fair_odds": 1.61,
            "market_implied_probability": 0.50,
            "edge": 0.12,
            "ev": 0.24,
            "decision_confidence_score": 72.0,
            "quality_tier": "TOP",
            "recommended_stake": 10.0,
            "model_artifact": "artifact.pkl",
            "model_hash": "hash123",
            "feature_schema_hash": "schema123",
            "target_name": "over_9_5",
            "stake": 10.0,
            # Decided first, settled second.
            "decision_timestamp": "2026-08-25T10:00:00Z",
        },
        {
            "run_id": "r1",
            "decision": "PLAY",
            "market_support_status": "SUPPORTED",
            "competition": "Serie A",
            "fixture_id": 2,
            "provider_event_id": "evt-2",
            "home_team": "Milan",
            "away_team": "Napoli",
            "kickoff_utc": "2026-08-25T20:45:00Z",
            "market": "TOTAL_CORNERS_OVER",
            "line": "10.5",
            "side": "OVER",
            "bookmaker": "MARKET_MEDIAN",
            "odds_at_decision": 2.0,
            "closing_odds": 2.0,
            "predicted_probability": 0.60,
            "fair_odds": 1.67,
            "market_implied_probability": 0.50,
            "edge": 0.10,
            "ev": 0.20,
            "decision_confidence_score": 70.0,
            "quality_tier": "TOP",
            "recommended_stake": 20.0,
            "model_artifact": "artifact.pkl",
            "model_hash": "hash123",
            "feature_schema_hash": "schema123",
            "target_name": "over_10_5",
            "stake": 20.0,
            # Decided second, settled first.
            "decision_timestamp": "2026-08-25T11:00:00Z",
        },
    ]

    for row in rows:
        assert record_bet(tmp_path, row) is True

    db_path = tmp_path / "data" / "collector.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE collector_results ("
            "fixture_id INTEGER UNIQUE, "
            "home_score INTEGER, away_score INTEGER, "
            "home_corners INTEGER, away_corners INTEGER, "
            "total_corners INTEGER, settled_at TEXT, provider TEXT)"
        )

        # Fixture 2 settles FIRST and loses OVER 10.5: 100 -> 80.
        conn.execute(
            "INSERT INTO collector_results VALUES "
            "(2, 1, 0, 4, 4, 8, "
            "'2026-08-25T22:00:00Z', 'api-football')"
        )

        # Fixture 1 settles SECOND and wins OVER 9.5 at 2.00: 80 -> 90.
        conn.execute(
            "INSERT INTO collector_results VALUES "
            "(1, 2, 1, 6, 5, 11, "
            "'2026-08-26T10:00:00Z', 'api-football')"
        )
        conn.commit()
    finally:
        conn.close()

    settlement = settle_paper_trades(
        base_dir=tmp_path,
        output_dir=tmp_path,
        bankroll_start=100.0,
    )

    settled = pd.read_csv(
        reports_dir / "paper_trading_settled.csv"
    ).sort_values(
        ["settled_timestamp", "fixture_id", "line"],
        kind="mergesort",
    ).reset_index(drop=True)

    # One canonical chronology: fixture 2 settles first, fixture 1 second.
    assert settled["fixture_id"].tolist() == [2, 1]
    assert settled["bankroll_before"].tolist() == pytest.approx([100.0, 80.0])
    assert settled["bankroll_after"].tolist() == pytest.approx([80.0, 90.0])

    curve = settlement["summary"]["bankroll_curve"]
    assert [row["fixture_id"] for row in curve] == [2, 1]
    assert [row["bankroll_after"] for row in curve] == pytest.approx([80.0, 90.0])

    assert settlement["summary"]["final_bankroll"] == pytest.approx(90.0)
    assert resolve_current_bankroll(
        tmp_path,
        default_bankroll=100.0,
    ) == pytest.approx(90.0)

    # The persisted ledger and payload must describe the same bankroll.
    assert settled.iloc[-1]["bankroll_after"] == pytest.approx(
        settlement["summary"]["final_bankroll"]
    )


def test_settlement_payload_uses_canonical_bankroll_columns_without_replaying_bets() -> None:
    from src.research.observation_freeze import _build_settlement_payload

    settled = pd.DataFrame(
        [
            {
                "settled_timestamp": "2026-08-25T22:00:00Z",
                "fixture_id": 2,
                "line": "10.5",
                "side": "OVER",
                "target_name": "over_10_5",
                "quality_tier": "TOP",
                "bet_result": "LOSS",
                "stake": 20.0,
                "odds_at_decision": 2.0,
                "profit_loss": -20.0,
                "bankroll_before": 100.0,
                "bankroll_after": 80.0,
                "EV": 0.20,
                "confidence": 70.0,
                "CLV": 0.01,
                "predicted_probability": 0.60,
            },
            {
                "settled_timestamp": "2026-08-26T10:00:00Z",
                "fixture_id": 1,
                "line": "9.5",
                "side": "OVER",
                "target_name": "over_9_5",
                "quality_tier": "TOP",
                "bet_result": "WIN",
                "stake": 10.0,
                # Deliberately inconsistent with canonical bankroll_after.
                # Replaying this price would produce 100, not 90.
                "odds_at_decision": 3.0,
                "profit_loss": 10.0,
                "bankroll_before": 80.0,
                "bankroll_after": 90.0,
                "EV": 0.24,
                "confidence": 72.0,
                "CLV": 0.02,
                "predicted_probability": 0.62,
            },
        ]
    )

    payload = _build_settlement_payload(
        settled=settled,
        bankroll_start=100.0,
    )

    assert payload["summary"]["final_bankroll"] == pytest.approx(90.0)
    assert [
        item["bankroll_after"]
        for item in payload["bankroll_curve"]
    ] == pytest.approx([80.0, 90.0])


def test_settlement_is_idempotent_across_repeated_runs_with_existing_ledger(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_row = {
        "run_id": "r1",
        "decision": "PLAY",
        "market_support_status": "SUPPORTED",
        "competition": "Serie A",
        "fixture_id": 1,
        "provider_event_id": "evt-1",
        "home_team": "Inter",
        "away_team": "Roma",
        "kickoff_utc": "2026-08-25T18:45:00Z",
        "market": "TOTAL_CORNERS_OVER",
        "line": "9.5",
        "side": "OVER",
        "bookmaker": "MARKET_MEDIAN",
        "odds_at_decision": 2.0,
        "closing_odds": 2.0,
        "predicted_probability": 0.62,
        "fair_odds": 1.61,
        "market_implied_probability": 0.50,
        "edge": 0.12,
        "ev": 0.24,
        "decision_confidence_score": 72.0,
        "quality_tier": "TOP",
        "recommended_stake": 10.0,
        "model_artifact": "artifact.pkl",
        "model_hash": "hash123",
        "feature_schema_hash": "schema123",
        "target_name": "over_9_5",
        "stake": 10.0,
        "decision_timestamp": "2026-08-25T10:00:00Z",
    }
    assert record_bet(tmp_path, report_row) is True

    db_path = tmp_path / "data" / "collector.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE collector_results ("
            "fixture_id INTEGER UNIQUE, home_score INTEGER, away_score INTEGER, "
            "home_corners INTEGER, away_corners INTEGER, total_corners INTEGER, "
            "settled_at TEXT, provider TEXT)"
        )
        conn.execute(
            "INSERT INTO collector_results VALUES "
            "(1, 1, 0, 6, 5, 11, '2026-08-26T10:00:00Z', 'api-football')"
        )

    first = settle_paper_trades(
        base_dir=tmp_path, output_dir=tmp_path, bankroll_start=100.0
    )
    second = settle_paper_trades(
        base_dir=tmp_path, output_dir=tmp_path, bankroll_start=100.0
    )

    settled = pd.read_csv(reports_dir / "paper_trading_settled.csv")
    assert len(settled) == 1
    assert second["summary"]["total_bets"] == 1
    assert second["summary"]["final_bankroll"] == pytest.approx(
        first["summary"]["final_bankroll"]
    )


def test_settlement_appends_new_opportunity_after_existing_ledger(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    existing = {
        "run_id": "old",
        "decision_timestamp": "2026-08-25T10:00:00Z",
        "fixture_id": 1,
        "provider_event_id": "evt-1",
        "competition": "Serie A",
        "home_team": "Inter",
        "away_team": "Roma",
        "kickoff": "2026-08-25T18:45:00Z",
        "line": "9.5",
        "side": "OVER",
        "bookmaker": "MARKET_MEDIAN",
        "odds_at_decision": 2.0,
        "closing_odds": 2.0,
        "predicted_probability": 0.62,
        "fair_odds": 1.61,
        "market_implied_probability": 0.50,
        "implied_probability_at_decision": 0.50,
        "closing_implied_probability": 0.50,
        "CLV": 0.0,
        "edge": 0.12,
        "EV": 0.24,
        "confidence": 72.0,
        "quality_tier": "TOP",
        "recommended_stake": 10.0,
        "model_artifact": "artifact.pkl",
        "model_hash": "hash123",
        "feature_schema_hash": "schema123",
        "home_corners": 6,
        "away_corners": 5,
        "total_corners": 11,
        "bet_result": "WIN",
        "stake": 10.0,
        "profit_loss": 10.0,
        "bankroll_before": 100.0,
        "bankroll_after": 110.0,
        "settled_timestamp": "2026-08-26T10:00:00Z",
        "target_name": "over_9_5",
    }
    pd.DataFrame([existing]).to_csv(
        reports_dir / "paper_trading_settled.csv", index=False
    )

    new_report = {
        "run_id": "new",
        "decision": "PLAY",
        "market_support_status": "SUPPORTED",
        "competition": "Serie A",
        "fixture_id": 2,
        "provider_event_id": "evt-2",
        "home_team": "Milan",
        "away_team": "Napoli",
        "kickoff_utc": "2026-08-27T18:45:00Z",
        "market": "TOTAL_CORNERS_UNDER",
        "line": "10.5",
        "side": "UNDER",
        "bookmaker": "MARKET_MEDIAN",
        "odds_at_decision": 1.8,
        "closing_odds": 1.8,
        "predicted_probability": 0.65,
        "fair_odds": 1.54,
        "market_implied_probability": 1.0 / 1.8,
        "edge": 0.65 - (1.0 / 1.8),
        "ev": 0.65 * 1.8 - 1.0,
        "decision_confidence_score": 70.0,
        "quality_tier": "TOP",
        "recommended_stake": 5.0,
        "model_artifact": "artifact.pkl",
        "model_hash": "hash123",
        "feature_schema_hash": "schema123",
        "target_name": "under_10_5",
        "stake": 5.0,
        "decision_timestamp": "2026-08-27T10:00:00Z",
    }
    assert record_bet(tmp_path, new_report) is True

    db_path = tmp_path / "data" / "collector.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE collector_results ("
            "fixture_id INTEGER UNIQUE, home_score INTEGER, away_score INTEGER, "
            "home_corners INTEGER, away_corners INTEGER, total_corners INTEGER, "
            "settled_at TEXT, provider TEXT)"
        )
        conn.execute(
            "INSERT INTO collector_results VALUES "
            "(2, 1, 0, 4, 4, 8, '2026-08-28T10:00:00Z', 'api-football')"
        )

    result = settle_paper_trades(
        base_dir=tmp_path, output_dir=tmp_path, bankroll_start=100.0
    )

    settled = pd.read_csv(
        reports_dir / "paper_trading_settled.csv"
    ).sort_values(["settled_timestamp", "fixture_id"]).reset_index(drop=True)

    assert settled["fixture_id"].tolist() == [1, 2]
    assert settled["bankroll_before"].tolist() == pytest.approx([100.0, 110.0])
    assert settled["bankroll_after"].tolist() == pytest.approx([110.0, 114.0])
    assert result["summary"]["total_bets"] == 2
    assert result["summary"]["final_bankroll"] == pytest.approx(114.0)


def test_settlement_ignores_production_style_replays_already_in_ledger(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    settled_rows = [
        {"fixture_id": 17, "side": "UNDER", "line": 10.5, "target_name": "under_10_5", "stake": 5.0, "bet_result": "LOSS", "profit_loss": -5.0, "bankroll_before": 100.0, "bankroll_after": 95.0, "settled_timestamp": "2026-09-05T14:09:54Z"},
        {"fixture_id": 33, "side": "UNDER", "line": 9.5, "target_name": "under_9_5", "stake": 5.0, "bet_result": "LOSS", "profit_loss": -5.0, "bankroll_before": 95.0, "bankroll_after": 90.0, "settled_timestamp": "2026-09-05T18:00:06Z"},
        {"fixture_id": 34, "side": "UNDER", "line": 9.5, "target_name": "under_9_5", "stake": 5.0, "bet_result": "LOSS", "profit_loss": -5.0, "bankroll_before": 90.0, "bankroll_after": 85.0, "settled_timestamp": "2026-09-05T21:00:07Z"},
        {"fixture_id": 36, "side": "OVER", "line": 8.5, "target_name": "over_8_5", "stake": 5.0, "bet_result": "WIN", "profit_loss": 5.5, "bankroll_before": 85.0, "bankroll_after": 90.5, "settled_timestamp": "2026-09-06T15:00:09Z"},
        {"fixture_id": 35, "side": "UNDER", "line": 10.5, "target_name": "under_10_5", "stake": 5.0, "bet_result": "WIN", "profit_loss": 2.675, "bankroll_before": 90.5, "bankroll_after": 93.175, "settled_timestamp": "2026-09-06T16:00:07Z"},
    ]
    # Match the canonical production settlement schema used by the
    # performance payload, while keeping the regression focused on identity,
    # stake and bankroll preservation.
    for row in settled_rows:
        row.update({
            "run_id": "historical",
            "decision_timestamp": "2026-09-01T10:00:00Z",
            "provider_event_id": f"evt-{row['fixture_id']}",
            "competition": "Serie A",
            "home_team": "Home",
            "away_team": "Away",
            "kickoff": "2026-09-05T12:00:00Z",
            "bookmaker": "MARKET_MEDIAN",
            "odds_at_decision": 2.0,
            "closing_odds": 2.0,
            "predicted_probability": 0.60,
            "fair_odds": 1.67,
            "market_implied_probability": 0.50,
            "implied_probability_at_decision": 0.50,
            "closing_implied_probability": 0.50,
            "CLV": 0.0,
            "edge": 0.10,
            "EV": 0.20,
            "confidence": 70.0,
            "quality_tier": "TOP",
            "recommended_stake": row["stake"],
            "model_artifact": "artifact.pkl",
            "model_hash": "hash123",
            "feature_schema_hash": "schema123",
            "home_corners": 5,
            "away_corners": 5,
            "total_corners": 10,
        })

    pd.DataFrame(settled_rows).to_csv(
        reports_dir / "paper_trading_settled.csv", index=False
    )

    current_rows = []
    for fixture_id, market, side, line, target in [
        (33, "TOTAL_CORNERS_UNDER", "UNDER", 9.5, "under_9_5"),
        (34, "TOTAL_CORNERS_UNDER", "UNDER", 9.5, "under_9_5"),
        (35, "TOTAL_CORNERS_UNDER", "UNDER", 10.5, "under_10_5"),
        (36, "TOTAL_CORNERS_OVER", "OVER", 8.5, "over_8_5"),
    ]:
        current_rows.append({
            "run_id": "production-replay",
            "decision": "PLAY",
            "market_support_status": "SUPPORTED",
            "competition": "Serie A",
            "fixture_id": fixture_id,
            "market": market,
            "side": side,
            "line": line,
            "target_name": target,
            "recommended_stake": 4.65875,
            "stake": 4.65875,
        })

    pd.DataFrame(current_rows).to_csv(
        reports_dir / "paper_trading_current.csv", index=False
    )

    result = settle_paper_trades(
        base_dir=tmp_path,
        output_dir=tmp_path,
        bankroll_start=100.0,
    )

    settled = pd.read_csv(
        reports_dir / "paper_trading_settled.csv"
    ).sort_values(
        ["settled_timestamp", "fixture_id", "line"],
        kind="mergesort",
    ).reset_index(drop=True)

    assert len(settled) == 5
    assert settled["fixture_id"].tolist() == [17, 33, 34, 36, 35]
    assert settled["stake"].tolist() == pytest.approx([5.0] * 5)
    assert settled["profit_loss"].sum() == pytest.approx(-6.825)
    assert settled.iloc[-1]["bankroll_after"] == pytest.approx(93.175)

    assert result["summary"]["bankroll_start"] == pytest.approx(100.0)
    assert result["summary"]["total_bets"] == 5
    assert result["summary"]["profit_loss"] == pytest.approx(-6.825)
    assert result["summary"]["final_bankroll"] == pytest.approx(93.175)

    assert resolve_current_bankroll(
        base_dir=tmp_path,
        default_bankroll=100.0,
    ) == pytest.approx(93.175)

def test_paper_bet_ledger_records_each_economic_opportunity_once(tmp_path: Path) -> None:
    from src.research.paper_bet_ledger import list_bets, record_bet

    row = {
        "fixture_id": 33,
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": 9.5,
        "recommended_stake": 5.0,
    }

    assert record_bet(tmp_path, row) is True
    assert record_bet(tmp_path, row) is False

    bets = list_bets(tmp_path)
    assert len(bets) == 1
    assert bets[0]["fixture_id"] == "33"

def test_paper_bet_ledger_detects_existing_opportunity(tmp_path: Path) -> None:
    from src.research.paper_bet_ledger import contains_bet, record_bet

    row = {
        "fixture_id": 33,
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": 9.5,
        "recommended_stake": 5.0,
    }

    assert contains_bet(tmp_path, row) is False
    assert record_bet(tmp_path, row) is True
    assert contains_bet(tmp_path, row) is True

def test_paper_bet_ledger_detects_existing_opportunity(tmp_path: Path) -> None:
    from src.research.paper_bet_ledger import contains_bet, record_bet

    row = {
        "fixture_id": 33,
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": 9.5,
        "recommended_stake": 5.0,
    }

    assert contains_bet(tmp_path, row) is False
    assert record_bet(tmp_path, row) is True
    assert contains_bet(tmp_path, row) is True

def test_suppress_replayed_paper_bets_changes_existing_play_to_no_bet(tmp_path: Path) -> None:
    from src.research.paper_bet_ledger import record_bet
    from src.research.paper_trading import _suppress_replayed_paper_bets

    row = {
        "fixture_id": 33,
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": 9.5,
        "decision": "PLAY",
        "decision_reason": "VALUE_THRESHOLD_MET",
        "recommended_stake": 5.0,
    }
    record_bet(tmp_path, row)

    report = pd.DataFrame([row])
    filtered = _suppress_replayed_paper_bets(report, tmp_path)

    assert filtered.iloc[0]["decision"] == "NO BET"
    assert filtered.iloc[0]["decision_reason"] == "ALREADY_PAPER_TRADED"
    assert float(filtered.iloc[0]["recommended_stake"]) == 0.0

def test_record_new_paper_bets_records_only_play_rows(tmp_path: Path) -> None:
    from src.research.paper_bet_ledger import list_bets
    from src.research.paper_trading import _record_new_paper_bets

    report = pd.DataFrame([
        {
            "fixture_id": 33, "market": "TOTAL_CORNERS_UNDER",
            "side": "UNDER", "line": 9.5,
            "decision": "PLAY", "recommended_stake": 5.0,
        },
        {
            "fixture_id": 34, "market": "TOTAL_CORNERS_UNDER",
            "side": "UNDER", "line": 9.5,
            "decision": "NO BET", "recommended_stake": 0.0,
        },
    ])

    assert _record_new_paper_bets(report, tmp_path) == 1
    bets = list_bets(tmp_path)
    assert len(bets) == 1
    assert bets[0]["fixture_id"] == "33"
