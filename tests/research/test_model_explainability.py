from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.model_benchmark import run_model_benchmark
from src.research.model_explainability import run_model_explainability


def test_model_explainability_reports_and_plots_are_generated(tmp_path: Path) -> None:
    benchmark_results = run_model_benchmark(base_dir=Path.cwd(), output_dir=tmp_path)
    assert benchmark_results["best_models"]

    first_run = run_model_explainability(base_dir=Path.cwd(), output_dir=tmp_path)
    second_run = run_model_explainability(base_dir=Path.cwd(), output_dir=tmp_path)

    report_files = [
        tmp_path / "reports" / "feature_importance.md",
        tmp_path / "reports" / "local_explanations.md",
        tmp_path / "reports" / "error_analysis.md",
        tmp_path / "reports" / "team_bias.md",
        tmp_path / "reports" / "feature_interactions.md",
        tmp_path / "reports" / "model_scientific_review.md",
        tmp_path / "reports" / "confidence_analysis.md",
    ]
    for report_path in report_files:
        assert report_path.exists(), report_path

    error_plots_dir = tmp_path / "reports" / "error_plots"
    assert error_plots_dir.exists()
    assert len(list(error_plots_dir.glob("*.png"))) >= 6

    assert first_run["no_leakage"] is True
    assert first_run["deterministic"] is True
    assert first_run["validation_row_count"] > 0

    assert first_run["importance_summaries"]
    for summary in first_run["importance_summaries"]:
        importance = summary["feature_importance"]
        assert importance
        cumulative = sum(item["permutation_normalized"] for item in importance)
        assert cumulative == cumulative
        assert abs(cumulative - 1.0) < 1e-6 or abs(cumulative - 1.0) < 1e-3

        permutation_order = [item["feature"] for item in importance]
        shap_order = [item["feature"] for item in sorted(importance, key=lambda item: item["shap_like_importance"], reverse=True)]
        overlap = set(permutation_order[:5]).intersection(shap_order[:5])
        assert overlap, summary["target_name"]

    assert first_run["local_explanations"]
    for explanation in first_run["local_explanations"]:
        assert 0.0 <= explanation["confidence"] <= 1.0
        assert explanation["prediction"] == explanation["prediction"]

    assert first_run["confidence_entries"]
    for entry in first_run["confidence_entries"]:
        assert 0.0 <= entry["confidence_mean"] <= 1.0
        assert entry["error_mean"] >= 0.0
        assert entry["ece"] >= 0.0

    assert first_run["importance_summaries"] == second_run["importance_summaries"]
    assert first_run["local_explanations"] == second_run["local_explanations"]
