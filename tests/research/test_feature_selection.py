from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.feature_selection import run_feature_selection


def test_feature_selection_is_time_safe_and_deterministic(tmp_path: Path) -> None:
    first_run = run_feature_selection(base_dir=Path.cwd(), output_dir=tmp_path)
    second_run = run_feature_selection(base_dir=Path.cwd(), output_dir=tmp_path)

    regression = first_run["actual_total_corners"]
    assert regression["selected_feature_count"] > 0
    assert regression["selected_features"]
    assert regression["selected_features"] == first_run["actual_total_corners"]["selected_features"]

    dataset = pd.read_parquet(Path.cwd() / "data" / "research" / "advanced_features.parquet")
    for target_name in ["actual_total_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        selected = [item["feature"] for item in first_run[target_name]["selected_features"]]
        assert selected == list(dict.fromkeys(selected))
        assert all(feature in dataset.columns for feature in selected)

    for target_name, target_value in {"actual_total_corners": "regression", "over_8_5": "classification"}.items():
        assert first_run[target_name]["target_type"] == target_value

    for season in ["2023/24", "2024/25", "2025/26"]:
        season_rows = dataset[dataset["season"] == season]
        if season in {"2023/24", "2024/25"}:
            assert not season_rows.empty
        else:
            assert not season_rows.empty

    assert (tmp_path / "data" / "research" / "selected_features_regression.json").exists()
    assert (tmp_path / "data" / "research" / "selected_features_over85.json").exists()
    assert (tmp_path / "data" / "research" / "selected_features_over95.json").exists()
    assert (tmp_path / "data" / "research" / "selected_features_over105.json").exists()
    assert (tmp_path / "data" / "research" / "selected_features_over115.json").exists()
    assert (tmp_path / "reports" / "feature_selection_report.md").exists()
    assert (tmp_path / "reports" / "feature_stability_report.md").exists()
    assert (tmp_path / "reports" / "feature_collinearity_report.md").exists()
