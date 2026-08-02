from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.advanced_features import build_advanced_feature_dataset


def test_advanced_features_are_generated_without_leakage(tmp_path: Path) -> None:
    dataset = build_advanced_feature_dataset(base_dir=Path.cwd(), output_dir=tmp_path)

    assert dataset.shape[0] == 1140
    assert dataset.shape[1] > 40
    assert dataset["match_id"].is_unique
    assert dataset["match_id"].duplicated().sum() == 0
    assert dataset["season"].nunique() == 3
    assert dataset["date"].is_monotonic_increasing

    expected_columns = {
        "match_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "actual_home_corners",
        "actual_away_corners",
        "actual_total_corners",
        "over_8_5",
        "over_9_5",
        "over_10_5",
        "over_11_5",
        "data_quality_score",
        "insufficient_history",
        "corners_for_last3",
        "corners_for_last5",
        "corners_for_last10",
        "corners_against_last3",
        "corners_against_last5",
        "corners_against_last10",
        "total_corners_last3",
        "total_corners_last5",
        "total_corners_last10",
        "corners_for_ewma",
        "corners_against_ewma",
        "total_corners_ewma",
        "home_corners_for_last5",
        "home_corners_against_last5",
        "home_total_corners_last5",
        "away_corners_for_last5",
        "away_corners_against_last5",
        "away_total_corners_last5",
        "corners_for_std_last5",
        "corners_for_std_last10",
        "corners_against_std_last5",
        "total_corners_std_last5",
        "total_corners_std_last10",
        "coefficient_of_variation_last10",
        "attack_trend",
        "defence_trend",
        "tempo_trend",
        "expected_home_corners_baseline",
        "expected_away_corners_baseline",
        "expected_total_corners_baseline",
        "attack_difference",
        "defence_difference",
        "tempo_difference",
        "combined_volatility",
        "combined_trend",
        "home_rest_days",
        "away_rest_days",
        "rest_days_difference",
        "home_matches_played",
        "away_matches_played",
        "season_match_number",
    }
    missing = expected_columns.difference(dataset.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

    assert dataset["actual_home_corners"].equals(dataset["home_corners"])
    assert dataset["actual_away_corners"].equals(dataset["away_corners"])
    assert dataset["actual_total_corners"].equals(dataset["total_corners"])
    assert ((dataset["over_8_5"] == (dataset["total_corners"] > 8.5).astype(int)).all())
    assert ((dataset["over_9_5"] == (dataset["total_corners"] > 9.5).astype(int)).all())
    assert ((dataset["over_10_5"] == (dataset["total_corners"] > 10.5).astype(int)).all())
    assert ((dataset["over_11_5"] == (dataset["total_corners"] > 11.5).astype(int)).all())

    numeric_columns = [col for col in dataset.columns if col not in {"match_id", "season", "date", "home_team", "away_team"}]
    assert np.isfinite(dataset[numeric_columns]).all().all()

    assert dataset["insufficient_history"].sum() > 0
    assert dataset["data_quality_score"].between(0.0, 1.0).all()

    home_row = dataset.iloc[0]
    assert home_row["corners_for_last3"] == 0.0
    assert home_row["home_corners_for_last5"] == 0.0
    assert home_row["home_rest_days"] == 0
    assert home_row["season_match_number"] == 1

    def expected_last3_for(row_idx: int, team: str) -> float:
        prior = dataset.iloc[:row_idx]
        values = []
        for _, match in prior.iterrows():
            if match["home_team"] == team:
                values.append(match["home_corners"])
            elif match["away_team"] == team:
                values.append(match["away_corners"])
        if not values:
            return 0.0
        return float(pd.Series(values).tail(3).mean())

    for idx, row in dataset.iloc[1:].iterrows():
        expected = expected_last3_for(int(idx), row["home_team"])
        assert row["corners_for_last3"] == expected

    assert (tmp_path / "data" / "research" / "advanced_features.parquet").exists()
    assert (tmp_path / "docs" / "ADVANCED_FEATURE_DICTIONARY.md").exists()
    assert (tmp_path / "reports" / "advanced_feature_validation.md").exists()

    repeated = build_advanced_feature_dataset(base_dir=Path.cwd(), output_dir=tmp_path)
    assert repeated.equals(dataset)
