from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.paper_trading import build_live_fixture_features, run_paper_trading


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
    assert "decision_state" in report.columns
    assert "provider_event_id" in report.columns

    assert (tmp_path / "data" / "paper_trading" / "paper_trades_current.parquet").exists()
    assert (tmp_path / "reports" / "paper_trading_current.csv").exists()
    assert (tmp_path / "reports" / "paper_trading_summary.md").exists()
