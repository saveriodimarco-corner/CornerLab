from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.premier_league_locked_market_blend import (
    OBSERVATION_SEASON,
    VALIDATION_SEASON,
    _paired_probabilities,
    analyze_locked_market_blend,
    run_locked_market_blend,
    select_locked_model_weight,
)


def _candidate_rows() -> pd.DataFrame:
    rows = []
    probabilities = {
        "2024/25": [0.90, 0.10],
        "2025/26": [0.10, 0.90],
        "2026/27": [0.70, 0.30],
    }

    for season, values in probabilities.items():
        for index, probability in enumerate(values):
            fixture_key = f"{season}-{index}"
            actual = 12.0 if index == 0 else 7.0

            for side, model_probability in (
                ("OVER", probability),
                ("UNDER", 1.0 - probability),
            ):
                rows.append(
                    {
                        "season": season,
                        "fixture_key": fixture_key,
                        "match_id": fixture_key,
                        "date": f"{season[:4]}-09-01",
                        "line": 8.5,
                        "side": side,
                        "reference_odds": 1.90,
                        "predicted_probability": (
                            model_probability
                        ),
                        "actual_total_corners": actual,
                        "quality_ok": True,
                    }
                )

    return pd.DataFrame(rows)


def test_weight_is_selected_only_from_development():
    paired = _paired_probabilities(
        _candidate_rows()
    )
    selected, grid = select_locked_model_weight(
        paired
    )

    assert selected == 1.0
    assert len(grid) == 21


def test_analysis_keeps_future_seasons_separate():
    result, candidates, bets = (
        analyze_locked_market_blend(
            _candidate_rows()
        )
    )

    assert result["selected_model_weight"] == 1.0
    assert VALIDATION_SEASON in result[
        "season_results"
    ]
    assert OBSERVATION_SEASON in result[
        "season_results"
    ]
    assert result[
        "real_betting_enabled"
    ] is False
    assert result["automatic_promotion"] is False
    assert len(candidates) == 12
    assert isinstance(bets, pd.DataFrame)


def test_runner_writes_isolated_report(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    _candidate_rows().to_csv(
        reports
        / "premier_league_multiseason_candidates.csv",
        index=False,
    )

    result = run_locked_market_blend(tmp_path)

    assert result["model_artifact_modified"] is False
    assert (
        reports
        / "premier_league_locked_market_blend.json"
    ).exists()
    assert (
        reports
        / (
            "premier_league_locked_market_blend_"
            "candidates.csv"
        )
    ).exists()
