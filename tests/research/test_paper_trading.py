from __future__ import annotations

from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.engine.feature_store import FeatureStore

from src.research.paper_trading import (
    _align_feature_schema,
    _load_authoritative_models,
    _model_registry_key,
    _resolve_market_probability,
    build_live_research_features,
    feature_row_to_model_input,
    build_live_fixture_features,
    run_paper_trading,
)
from src.research.observation_freeze import build_production_baseline_manifest, settle_paper_trades


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


def test_run_paper_trading_writes_current_artifacts(tmp_path: Path) -> None:
    result = run_paper_trading(base_dir=Path.cwd(), output_dir=tmp_path, bankroll=100.0)

    report = result["report"]
    assert not report.empty
    assert set(report["decision"].unique()).issubset({"PLAY", "LOW CONFIDENCE", "NO BET", "MODEL_UNAVAILABLE"})
    assert (report["decision"] == "MODEL_UNAVAILABLE").any()
    assert set(report.loc[report["decision"] == "MODEL_UNAVAILABLE", "decision_reason"].unique()).issubset({"NO_ACCEPTED_MODEL", "MODEL_INPUT_FAILED", "UNSUPPORTED_MARKET"})
    unsupported_targets = report.loc[report["market_support_status"] == "UNSUPPORTED", "target_name"].dropna().unique().tolist()
    assert "over_8_5" in unsupported_targets
    assert "over_11_5" in unsupported_targets
    assert (report.loc[report["target_name"].isin(["over_8_5", "over_11_5"]), "decision"] == "MODEL_UNAVAILABLE").all()
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
    (reports_dir / "paper_trading_current.csv").write_text(
        "run_id,decision,market_support_status,competition,fixture_id,provider_event_id,home_team,away_team,kickoff_utc,line,side,bookmaker,odds_at_decision,closing_odds,predicted_probability,fair_odds,market_implied_probability,edge,ev,decision_confidence_score,quality_tier,recommended_stake,model_artifact,model_hash,feature_schema_hash,target_name,stake,home_corners_result,away_corners_result,total_corners_result,decision_timestamp\n"
        "r1,PLAY,SUPPORTED,Serie A,1,evt-1,Inter,Roma,2026-08-25T18:45:00Z,9.5,OVER,book-a,2.0,1.9,0.62,1.61,0.5,0.12,0.24,72.0,TOP,2.0,artifact.pkl,hash123,schema123,over_9_5,2.0,6,5,11,2026-08-25T10:00:00Z\n",
        encoding="utf-8",
    )
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

    manifest = build_production_baseline_manifest(base_dir=tmp_path, output_dir=tmp_path)
    settlement = settle_paper_trades(base_dir=tmp_path, output_dir=tmp_path, bankroll_start=100.0)

    assert (tmp_path / "reports" / "production_baseline_serie_a.json").exists()
    assert manifest["supported_targets"] == ["over_9_5", "under_9_5", "over_10_5", "under_10_5"]
    assert (tmp_path / "reports" / "paper_trading_performance.json").exists()
    assert settlement["summary"]["total_bets"] == 1
    assert settlement["summary"]["profit_loss"] > 0


def test_under_probability_is_complement_of_over_probability() -> None:
    assert _resolve_market_probability("TOTAL_CORNERS_OVER", "OVER", 0.62) == 0.62
    assert _resolve_market_probability("TOTAL_CORNERS_UNDER", "UNDER", 0.62) == 0.38


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
