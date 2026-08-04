from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.confidence_engine import (
    apply_abstention_rules,
    run_confidence_engine,
)


def test_confidence_engine_builds_validation_predictions_and_outputs(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)

    table = result["confidence_table"]
    assert not table.empty
    assert table["match_id"].is_unique
    assert set(table["season"].unique()) == {"2025/26"}
    assert {"predicted_total_corners", "absolute_error", "signed_error", "confidence_score", "decision_state"}.issubset(table.columns)
    assert table["confidence_score"].between(0, 100).all()
    assert table["predicted_probability_over_8_5"].between(0, 1).all()
    assert table["predicted_probability_over_9_5"].between(0, 1).all()
    assert table["predicted_probability_over_10_5"].between(0, 1).all()
    assert table["predicted_probability_over_11_5"].between(0, 1).all()
    assert result["leakage_ok"] is True
    assert result["pre_match_feature_columns"]
    assert not set(result["pre_match_feature_columns"]).intersection({"actual_total_corners", "over_8_5", "home_corners", "away_corners"})

    policy = result["policy"]
    assert policy["accept_threshold"] >= policy["watch_threshold"]
    assert policy["accept_threshold"] >= 0
    assert policy["watch_threshold"] >= 0

    outputs = [
        tmp_path / "data" / "research" / "confidence_predictions.parquet",
        tmp_path / "data" / "research" / "confidence_policy.json",
        tmp_path / "reports" / "confidence_engine.md",
        tmp_path / "reports" / "selective_performance.md",
        tmp_path / "reports" / "risk_coverage_analysis.md",
        tmp_path / "reports" / "abstention_analysis.md",
    ]
    for path in outputs:
        assert path.exists(), path


def test_confidence_scores_and_decisions_are_deterministic_and_bounded(tmp_path: Path) -> None:
    first = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    second = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)

    assert first["confidence_table"].equals(second["confidence_table"])
    assert first["policy"] == second["policy"]

    row = pd.DataFrame(
        [{
            "data_quality_score": 0.1,
            "insufficient_history": True,
            "home_matches_played": 1,
            "away_matches_played": 1,
            "combined_volatility": 3.0,
            "model_disagreement": 0.9,
            "prediction_distance_from_line": 0.1,
            "probability_distance_from_0_50": 0.01,
            "residual_risk_estimate": 0.8,
            "team_bias_risk": 0.9,
            "cold_start_risk": 1.0,
            "feature_outlier_score": 0.95,
            "missing_history_count": 4,
            "calibration_bucket_error": 0.8,
            "model_stability_score": 0.2,
            "predicted_total_corners": 8.0,
            "predicted_probability_over_8_5": 0.5,
        }]
    )
    decision = apply_abstention_rules(row, policy={"accept_threshold": 75.0, "watch_threshold": 55.0, "feature_outlier_threshold": 0.8, "model_disagreement_threshold": 0.7, "data_quality_min": 0.25, "team_bias_risk_threshold": 0.8, "probability_distance_threshold": 0.05, "prediction_distance_threshold": 0.2})
    assert decision.iloc[0]["decision_state"] == "ABSTAIN"


def test_confidence_policy_meets_coverage_and_selective_metrics_are_valid(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    policy = result["policy"]
    assert 0.0 <= policy["accept_coverage"] <= 1.0
    assert 0.0 <= policy["watch_coverage"] <= 1.0
    assert 0.0 <= policy["abstain_coverage"] <= 1.0
    assert np.isclose(policy["accept_coverage"] + policy["watch_coverage"] + policy["abstain_coverage"], 1.0, atol=1e-6)
    selective = result["selective_performance"]
    assert selective["regression"]
    assert selective["classification"]
    for label, metrics in selective["regression"].items():
        assert metrics["coverage"] >= 0.0
        assert metrics["coverage"] <= 1.0
        assert metrics["mae"] >= 0.0
    for label, metrics in selective["classification"]["over_8_5"].items():
        assert metrics["coverage"] >= 0.0
        assert metrics["coverage"] <= 1.0
        assert metrics["brier_score"] >= 0.0


def test_confidence_buckets_and_readiness_are_consistent(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    confidence_buckets = result["confidence_buckets"]
    assert not confidence_buckets.empty
    assert confidence_buckets["bucket_probability"].between(0, 1).all()
    readiness = result["readiness_summary"]
    assert readiness["policy_result"] in {"READY FOR BETTING-LAYER RESEARCH", "REQUIRES CONFIDENCE RECALIBRATION", "NO VALID CONFIDENCE POLICY"}
    assert readiness["leakage_result"] in {"NO LEAKAGE", "LEAKAGE DETECTED"}


def test_confidence_policy_selects_a_subset_that_outperforms_full_predictions(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    table = result["confidence_table"]
    policy = result["policy"]

    accepted = table.loc[table["decision_state"] == "ACCEPT"].copy()
    assert not accepted.empty
    assert policy["accept_threshold"] > policy["watch_threshold"]

    full_abs_error = np.abs(table["signed_error"].astype(float))
    accepted_abs_error = np.abs(accepted["signed_error"].astype(float))
    full_brier = np.mean([
        np.mean((table[f"predicted_probability_{target}"] - table[f"actual_outcome_{target}"]) ** 2)
        for target in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
    ])
    accepted_brier = np.mean([
        np.mean((accepted[f"predicted_probability_{target}"] - accepted[f"actual_outcome_{target}"]) ** 2)
        for target in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
    ])

    assert accepted_abs_error.mean() < full_abs_error.mean()
    assert accepted_brier < full_brier


def test_confidence_policy_coverage_splits_the_validation_set(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    policy = result["policy"]

    coverage_total = policy["accept_coverage"] + policy["watch_coverage"] + policy["abstain_coverage"]
    assert np.isclose(coverage_total, 1.0, atol=1e-6)
    assert 0.0 <= policy["accept_coverage"] <= 1.0
    assert 0.0 <= policy["watch_coverage"] <= 1.0
    assert 0.0 <= policy["abstain_coverage"] <= 1.0


def test_confidence_reports_use_actual_decision_state_coverage(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    table = result["confidence_table"]
    policy = result["policy"]

    accept_coverage = float((table["decision_state"].astype(str) == "ACCEPT").mean())
    watch_coverage = float((table["decision_state"].astype(str) == "WATCH").mean())
    abstain_coverage = float((table["decision_state"].astype(str) == "ABSTAIN").mean())

    assert policy["accept_coverage"] == accept_coverage
    assert policy["watch_coverage"] == watch_coverage
    assert policy["abstain_coverage"] == abstain_coverage

    calibration_report = (tmp_path / "reports" / "confidence_calibration.md").read_text(encoding="utf-8")
    threshold_report = (tmp_path / "reports" / "threshold_search.md").read_text(encoding="utf-8")

    assert f"- Accept coverage: {accept_coverage:.3f}" in calibration_report
    assert f"- Watch coverage: {watch_coverage:.3f}" in calibration_report
    assert f"- Abstain coverage: {abstain_coverage:.3f}" in calibration_report
    assert f"- Accept coverage: {accept_coverage:.3f}" in threshold_report
    assert f"- Watch coverage: {watch_coverage:.3f}" in threshold_report
    assert f"- Abstain coverage: {abstain_coverage:.3f}" in threshold_report


def test_policy_search_generates_reports_and_targets_accept_coverage(tmp_path: Path) -> None:
    result = run_confidence_engine(base_dir=Path.cwd(), output_dir=tmp_path)
    policy = result["policy"]

    assert (tmp_path / "reports" / "policy_search.md").exists()
    assert (tmp_path / "reports" / "policy_tradeoff.md").exists()
    assert (tmp_path / "reports" / "final_policy.md").exists()
    assert 0.15 <= policy["accept_coverage"] <= 0.45
