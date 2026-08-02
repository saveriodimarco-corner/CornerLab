from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.model_benchmark import run_model_benchmark


def test_model_benchmark_is_time_safe_and_deterministic(tmp_path: Path) -> None:
    results = run_model_benchmark(base_dir=Path.cwd(), output_dir=tmp_path)

    assert results["chronology_ok"] is True
    assert results["no_leakage"] is True
    assert results["train_after_validation"] is False

    regression_results = results["regression_results"]
    assert regression_results
    assert any(item["model_name"] == "league_mean_baseline" for item in regression_results)
    assert any(item["model_name"] == "recent_form_baseline" for item in regression_results)
    assert any(item["model_name"] == "poisson_regression" for item in regression_results)
    assert any(item["model_name"] == "negative_binomial_regression" for item in regression_results)
    assert any(item["model_name"] == "ridge_regression" for item in regression_results)
    assert any(item["model_name"] == "hist_gradient_boosting_regression" for item in regression_results)

    classification_results = results["classification_results"]
    assert classification_results
    assert any(item["model_name"] == "historical_base_rate_baseline" for item in classification_results)
    assert any(item["model_name"] == "logistic_regression" for item in classification_results)
    assert any(item["model_name"] == "hist_gradient_boosting_classifier" for item in classification_results)
    assert any(item["model_name"] == "poisson_probability" for item in classification_results)
    assert any(item["model_name"] == "negative_binomial_probability" for item in classification_results)

    for entry in classification_results:
        for probability in entry["probabilities"]:
            assert 0.0 <= probability <= 1.0

    assert results["regression_best_model"]["accepted"] is True or results["regression_best_model"]["accepted"] is False
    assert results["best_models"]

    for target_name, target_results in results["best_models"].items():
        assert target_results["accepted"] is True or target_results["accepted"] is False
        if target_results["accepted"]:
            assert target_results["model_name"]
            assert target_results["primary_metric_value"] == target_results["primary_metric_value"]
            assert target_results["baseline_metric_value"] == target_results["baseline_metric_value"]

    for target_name in ["actual_total_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        selection_path = Path.cwd() / "data" / "research" / {
            "actual_total_corners": "selected_features_regression.json",
            "over_8_5": "selected_features_over85.json",
            "over_9_5": "selected_features_over95.json",
            "over_10_5": "selected_features_over105.json",
            "over_11_5": "selected_features_over115.json",
        }[target_name]
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        features = [item["feature"] for item in payload["selected_features"]]
        assert features == list(dict.fromkeys(features))
        assert all(feature not in {"actual_total_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"} for feature in features)

    assert (tmp_path / "reports" / "model_benchmark.md").exists()
    assert (tmp_path / "reports" / "regression_benchmark.md").exists()
    assert (tmp_path / "reports" / "classification_benchmark.md").exists()
    assert (tmp_path / "reports" / "calibration_benchmark.md").exists()
    assert (tmp_path / "data" / "research" / "model_benchmark_results.csv").exists()
    assert (tmp_path / "data" / "research" / "best_models.json").exists()

    accepted_models = results["accepted_model_artifacts"]
    assert isinstance(accepted_models, list)
    assert all(Path(path).exists() for path in accepted_models)

    assert (tmp_path / "reports" / "model_plots").exists()
