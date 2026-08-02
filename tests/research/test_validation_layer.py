from __future__ import annotations

from pathlib import Path

from src.research.validation import generate_validation_reports


def test_validation_reports_are_generated(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    output_paths = generate_validation_reports(tmp_path)

    expected_files = [
        "reports/validation_report.md",
        "reports/poisson_validation.md",
        "reports/rating_analysis.md",
        "reports/feature_correlation.md",
        "reports/season_summary.md",
    ]

    for relative_path in expected_files:
        assert (tmp_path / relative_path).exists(), f"Missing {relative_path}"

    assert (tmp_path / "reports/plots").exists()
    assert len(output_paths) >= len(expected_files)
