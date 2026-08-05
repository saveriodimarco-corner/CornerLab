from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .dataset_merger import DatasetMerger
from .dataset_statistics import DatasetStatistics
from .dataset_validator import DatasetValidator
from .provider_registry import ProviderRegistry


class HistoricalDatasetManager:
    """Data-only manager for historical datasets and reports."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.registry = ProviderRegistry()
        self.validator = DatasetValidator()
        self.merger = DatasetMerger()
        self.statistics = DatasetStatistics()
        self.reports_dir = self.base_dir / "reports"
        self.templates_dir = self.base_dir / "data" / "historical" / "templates"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def schema_columns(self) -> List[str]:
        return list(self.validator.required_columns)

    def prepare_dataset(self, dataset: pd.DataFrame, provider_name: str) -> pd.DataFrame:
        frame = dataset.copy()
        for column in self.schema_columns():
            if column not in frame.columns:
                frame[column] = None
        frame = frame[self.schema_columns()]
        frame["provider"] = provider_name
        frame["data_quality_score"] = frame["data_quality_score"].fillna(1.0)
        return frame

    def generate_reports(self, dataset: pd.DataFrame) -> Path:
        validation = self.validator.validate(dataset)
        stats = self.statistics.compute(dataset)
        report_dir = self.reports_dir

        summary_payload = {
            "total_matches": stats["total_matches"],
            "matches_per_season": stats["matches_per_season"],
            "matches_per_provider": stats["matches_per_provider"],
            "coverage_percentage": stats["coverage_percentage"],
            "missing_percentage": stats["missing_percentage"],
            "duplicate_percentage": stats["duplicate_percentage"],
            "bookmakers_available": stats["bookmakers_available"],
            "corner_lines_available": stats["corner_lines_available"],
            "historical_depth": stats["historical_depth"],
        }
        (report_dir / "historical_dataset_summary.md").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
        (report_dir / "provider_coverage.md").write_text(
            f"# Provider coverage\n\n{json.dumps(stats['matches_per_provider'], indent=2)}\n",
            encoding="utf-8",
        )
        (report_dir / "dataset_quality.md").write_text(
            f"# Dataset quality\n\n{json.dumps(validation['quality_metrics'], indent=2)}\n",
            encoding="utf-8",
        )
        (report_dir / "missing_data.md").write_text(
            f"# Missing data\n\n{json.dumps(validation['issues'], indent=2)}\n",
            encoding="utf-8",
        )
        (report_dir / "coverage_heatmap.csv").write_text(
            "provider,season,matches\n" + ",\n".join(
                f"{provider},{season},{count}" for provider, season, count in [(provider, season, count) for provider, count in stats['matches_per_provider'].items() for season in ["all"]]
            ) + "\n",
            encoding="utf-8",
        )

        self.create_import_template("manual_csv")
        return report_dir

    def create_import_template(self, provider_name: str) -> Path:
        template_path = self.templates_dir / f"{provider_name}_template.csv"
        template_df = pd.DataFrame(columns=self.schema_columns())
        template_df.to_csv(template_path, index=False)
        return template_path
