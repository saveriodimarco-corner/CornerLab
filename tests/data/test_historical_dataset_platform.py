from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.historical import (
    DatasetMerger,
    DatasetValidator,
    HistoricalDatasetManager,
    ProviderRegistry,
)


def test_provider_registry_registers_builtin_and_custom_providers() -> None:
    registry = ProviderRegistry()
    providers = registry.list_providers()
    assert {"football_data", "api_football", "the_odds_api", "manual_csv"}.issubset(set(providers))

    registry.register_provider("test_provider", "Test Provider")
    assert "test_provider" in registry.list_providers()


def test_dataset_validator_detects_duplicates_and_quality_issues(tmp_path: Path) -> None:
    validator = DatasetValidator()
    records = [
        {
            "match_id": "m1",
            "competition": "Serie A",
            "season": "2024/25",
            "kickoff": "2024-08-17T20:00:00",
            "home_team": "Juventus",
            "away_team": "Inter",
            "bookmaker": "Bet365",
            "market": "TOTAL_CORNERS_OVER",
            "line": 8.5,
            "opening_odds": 1.90,
            "closing_odds": 2.10,
            "settlement": "PENDING",
            "result": None,
            "model_probability": 0.61,
            "confidence_score": 0.74,
            "expected_value": 0.12,
            "kelly_fraction": 0.02,
            "provider": "football_data",
            "data_quality_score": 1.0,
        },
        {
            "match_id": "m1",
            "competition": "Serie A",
            "season": "2024/25",
            "kickoff": "2024-08-17T20:00:00",
            "home_team": "Juventus",
            "away_team": "Inter",
            "bookmaker": "Bet365",
            "market": "TOTAL_CORNERS_OVER",
            "line": 8.5,
            "opening_odds": 1.90,
            "closing_odds": 2.10,
            "settlement": "PENDING",
            "result": None,
            "model_probability": 0.61,
            "confidence_score": 0.74,
            "expected_value": 0.12,
            "kelly_fraction": 0.02,
            "provider": "football_data",
            "data_quality_score": 1.0,
        },
        {
            "match_id": "m2",
            "competition": "Serie A",
            "season": "2024/25",
            "kickoff": "2024-08-18T20:00:00",
            "home_team": "Milan",
            "away_team": "Napoli",
            "bookmaker": "Pinnacle",
            "market": "TOTAL_CORNERS_OVER",
            "line": 8.5,
            "opening_odds": None,
            "closing_odds": 1.95,
            "settlement": "PENDING",
            "result": None,
            "model_probability": 1.2,
            "confidence_score": 0.91,
            "expected_value": 0.05,
            "kelly_fraction": 0.01,
            "provider": "api_football",
            "data_quality_score": 1.0,
        },
    ]
    df = pd.DataFrame(records)
    validated = validator.validate(df)

    assert validated["issues"]["duplicate_matches"] == 1
    assert validated["issues"]["missing_odds"] == 1
    assert validated["issues"]["invalid_probabilities"] == 1
    assert validated["quality_metrics"]["quality_score"] < 1.0
    assert validated["quality_metrics"]["missing_percentage"] > 0.0


def test_dataset_merger_combines_datasets_and_preserves_schema(tmp_path: Path) -> None:
    merger = DatasetMerger()
    df_a = pd.DataFrame(
        [
            {
                "match_id": "a1",
                "competition": "Serie A",
                "season": "2024/25",
                "kickoff": "2024-08-17T20:00:00",
                "home_team": "Juventus",
                "away_team": "Inter",
                "bookmaker": "Bet365",
                "market": "TOTAL_CORNERS_OVER",
                "line": 8.5,
                "opening_odds": 1.90,
                "closing_odds": 2.10,
                "settlement": "PENDING",
                "result": None,
                "model_probability": 0.61,
                "confidence_score": 0.74,
                "expected_value": 0.12,
                "kelly_fraction": 0.02,
                "provider": "football_data",
                "data_quality_score": 0.95,
            }
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "match_id": "b1",
                "competition": "Serie A",
                "season": "2024/25",
                "kickoff": "2024-08-18T20:00:00",
                "home_team": "Milan",
                "away_team": "Napoli",
                "bookmaker": "Pinnacle",
                "market": "TOTAL_CORNERS_OVER",
                "line": 8.5,
                "opening_odds": 1.95,
                "closing_odds": 2.05,
                "settlement": "PENDING",
                "result": None,
                "model_probability": 0.58,
                "confidence_score": 0.69,
                "expected_value": 0.08,
                "kelly_fraction": 0.01,
                "provider": "api_football",
                "data_quality_score": 0.91,
            }
        ]
    )

    merged = merger.merge(df_a, df_b)
    assert len(merged) == 2
    assert list(merged.columns) == merger.required_columns()


def test_historical_dataset_manager_writes_reports_and_templates(tmp_path: Path) -> None:
    manager = HistoricalDatasetManager(base_dir=tmp_path)
    registry = manager.registry
    registry.register_provider("manual_csv", "Manual CSV Import")

    records = [
        {
            "match_id": "m1",
            "competition": "Serie A",
            "season": "2024/25",
            "kickoff": "2024-08-17T20:00:00",
            "home_team": "Juventus",
            "away_team": "Inter",
            "bookmaker": "Bet365",
            "market": "TOTAL_CORNERS_OVER",
            "line": 8.5,
            "opening_odds": 1.90,
            "closing_odds": 2.10,
            "settlement": "PENDING",
            "result": None,
            "model_probability": 0.61,
            "confidence_score": 0.74,
            "expected_value": 0.12,
            "kelly_fraction": 0.02,
            "provider": "manual_csv",
            "data_quality_score": 0.97,
        }
    ]
    dataset = pd.DataFrame(records)
    prepared = manager.prepare_dataset(dataset, provider_name="manual_csv")
    report_dir = manager.generate_reports(prepared)

    assert list(prepared.columns) == manager.schema_columns()
    assert (report_dir / "historical_dataset_summary.md").exists()
    assert (report_dir / "provider_coverage.md").exists()
    assert (report_dir / "dataset_quality.md").exists()
    assert (report_dir / "missing_data.md").exists()
    assert (report_dir / "coverage_heatmap.csv").exists()

    template_path = tmp_path / "data" / "historical" / "templates" / "manual_csv_template.csv"
    assert template_path.exists()

    summary_payload = json.loads((report_dir / "historical_dataset_summary.md").read_text(encoding="utf-8"))
    assert summary_payload["total_matches"] == 1
