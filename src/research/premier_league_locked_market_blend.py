from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.decision_engine import (
    MIN_EV,
    MIN_MARKET_EDGE,
    MIN_ODDS,
    MIN_PREDICTED_PROBABILITY,
    minimum_acceptable_odds,
)
from src.research.premier_league_market_validation import (
    _summarize_bets,
)


DEVELOPMENT_SEASON = "2024/25"
VALIDATION_SEASON = "2025/26"
OBSERVATION_SEASON = "2026/27"
MODEL_WEIGHTS = tuple(
    round(value / 20.0, 2)
    for value in range(21)
)
REPORT_NAME = (
    "premier_league_locked_market_blend.json"
)
CANDIDATES_NAME = (
    "premier_league_locked_market_blend_candidates.csv"
)
BETS_NAME = (
    "premier_league_locked_market_blend_bets.csv"
)
SOURCE_CANDIDATES_NAME = (
    "premier_league_multiseason_candidates.csv"
)


def _atomic_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _paired_probabilities(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "season",
        "fixture_key",
        "line",
        "side",
        "reference_odds",
        "predicted_probability",
        "actual_total_corners",
    }
    missing = required.difference(candidates.columns)

    if missing:
        raise ValueError(
            "Missing candidate columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []

    for (
        season,
        fixture_key,
        line,
    ), group in candidates.groupby(
        ["season", "fixture_key", "line"],
        sort=False,
    ):
        over = group.loc[group["side"] == "OVER"]
        under = group.loc[group["side"] == "UNDER"]

        if len(over) != 1 or len(under) != 1:
            continue

        over_row = over.iloc[0]
        under_row = under.iloc[0]
        inverse_over = 1.0 / float(
            over_row["reference_odds"]
        )
        inverse_under = 1.0 / float(
            under_row["reference_odds"]
        )
        market_over = inverse_over / (
            inverse_over + inverse_under
        )

        rows.append(
            {
                "season": str(season),
                "fixture_key": str(fixture_key),
                "line": float(line),
                "model_over": float(
                    over_row[
                        "predicted_probability"
                    ]
                ),
                "market_over": float(market_over),
                "actual_over": float(
                    float(
                        over_row[
                            "actual_total_corners"
                        ]
                    )
                    > float(line)
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No paired OVER/UNDER candidate rows"
        )

    return result


def _brier(
    probability: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
) -> float:
    predicted = np.asarray(
        probability,
        dtype=float,
    )
    observed = np.asarray(actual, dtype=float)
    return float(
        np.mean((predicted - observed) ** 2)
    )


def _weight_grid(
    development: pd.DataFrame,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for weight in MODEL_WEIGHTS:
        blended = (
            weight * development["model_over"]
            + (1.0 - weight)
            * development["market_over"]
        )
        rows.append(
            {
                "model_weight": weight,
                "development_brier": _brier(
                    blended,
                    development["actual_over"],
                ),
            }
        )

    return rows


def select_locked_model_weight(
    paired: pd.DataFrame,
) -> tuple[float, list[dict[str, float]]]:
    development = paired.loc[
        paired["season"].astype(str)
        == DEVELOPMENT_SEASON
    ]

    if development.empty:
        raise ValueError(
            "Development season is missing"
        )

    grid = _weight_grid(development)
    selected = min(
        grid,
        key=lambda row: (
            row["development_brier"],
            row["model_weight"],
        ),
    )
    return float(selected["model_weight"]), grid


def _season_calibration(
    paired: pd.DataFrame,
    model_weight: float,
) -> dict[str, Any]:
    blended = (
        model_weight * paired["model_over"]
        + (1.0 - model_weight)
        * paired["market_over"]
    )
    return {
        "binary_observations": int(len(paired)),
        "model_brier": _brier(
            paired["model_over"],
            paired["actual_over"],
        ),
        "market_brier": _brier(
            paired["market_over"],
            paired["actual_over"],
        ),
        "locked_blend_brier": _brier(
            blended,
            paired["actual_over"],
        ),
    }


def _blended_candidates(
    candidates: pd.DataFrame,
    paired: pd.DataFrame,
    model_weight: float,
) -> pd.DataFrame:
    market = paired[
        [
            "season",
            "fixture_key",
            "line",
            "market_over",
        ]
    ].copy()
    result = candidates.copy()
    result["season"] = result[
        "season"
    ].astype(str)
    result = result.merge(
        market,
        on=["season", "fixture_key", "line"],
        how="inner",
        validate="many_to_one",
    )
    result[
        "original_model_probability"
    ] = result["predicted_probability"].astype(float)
    result["market_probability"] = np.where(
        result["side"] == "OVER",
        result["market_over"],
        1.0 - result["market_over"],
    )
    result["predicted_probability"] = (
        model_weight
        * result["original_model_probability"]
        + (1.0 - model_weight)
        * result["market_probability"]
    )
    result["implied_probability"] = (
        1.0 / result["reference_odds"]
    )
    result["market_edge"] = (
        result["predicted_probability"]
        - result["implied_probability"]
    )
    result["expected_value"] = (
        result["predicted_probability"]
        * result["reference_odds"]
        - 1.0
    )
    result["minimum_acceptable_odds"] = [
        minimum_acceptable_odds(value)
        for value in result[
            "predicted_probability"
        ]
    ]
    quality = (
        result["quality_ok"].fillna(False).astype(bool)
        if "quality_ok" in result
        else pd.Series(True, index=result.index)
    )
    result["passed"] = (
        quality
        & (
            result["predicted_probability"]
            >= MIN_PREDICTED_PROBABILITY
        )
        & (result["reference_odds"] >= MIN_ODDS)
        & (result["market_edge"] >= MIN_MARKET_EDGE)
        & (result["expected_value"] >= MIN_EV)
        & (
            result["reference_odds"]
            >= result["minimum_acceptable_odds"]
        )
    )

    if "won" not in result:
        result["won"] = np.where(
            result["side"] == "OVER",
            result["actual_total_corners"]
            > result["line"],
            result["actual_total_corners"]
            < result["line"],
        )

    return result


def _best_fixture_bets(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    return (
        candidates.loc[candidates["passed"]]
        .sort_values(
            [
                "fixture_key",
                "expected_value",
                "predicted_probability",
            ],
            ascending=[True, False, False],
            kind="stable",
        )
        .drop_duplicates(
            subset=["fixture_key"],
            keep="first",
        )
        .sort_values(
            ["season", "date", "fixture_key"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def analyze_locked_market_blend(
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paired = _paired_probabilities(candidates)
    model_weight, grid = select_locked_model_weight(
        paired
    )
    blended_candidates = _blended_candidates(
        candidates,
        paired,
        model_weight,
    )
    bets = _best_fixture_bets(blended_candidates)
    season_results: dict[str, Any] = {}

    for season in (
        DEVELOPMENT_SEASON,
        VALIDATION_SEASON,
        OBSERVATION_SEASON,
    ):
        season_paired = paired.loc[
            paired["season"] == season
        ]
        season_bets = bets.loc[
            bets["season"] == season
        ]

        if season_paired.empty:
            raise ValueError(
                f"Missing evaluation season: {season}"
            )

        season_results[season] = {
            **_season_calibration(
                season_paired,
                model_weight,
            ),
            "qualified_bet_summary": (
                _summarize_bets(season_bets)
            ),
        }

    validation = season_results[
        VALIDATION_SEASON
    ]
    validation_bets = validation[
        "qualified_bet_summary"
    ]
    minimum_validation_sample = (
        validation_bets["bets"] >= 10
    )
    gates = {
        "weight_selected_on_development_only": True,
        "validation_blend_brier_better_than_market": (
            validation["locked_blend_brier"]
            < validation["market_brier"]
        ),
        "validation_blend_brier_not_worse_than_model": (
            validation["locked_blend_brier"]
            <= validation["model_brier"]
        ),
        "validation_at_least_10_qualified_bets": (
            minimum_validation_sample
        ),
        "validation_roi_positive": (
            minimum_validation_sample
            and validation_bets["roi"] is not None
            and validation_bets["roi"] > 0.0
        ),
        "validation_roi_lower_95_percent_above_zero": (
            minimum_validation_sample
            and validation_bets["roi_ci_95"][0]
            is not None
            and validation_bets["roi_ci_95"][0]
            > 0.0
        ),
    }
    result = {
        "run_type": (
            "premier_league_locked_market_blend"
        ),
        "design": {
            "development_season": DEVELOPMENT_SEASON,
            "validation_season": VALIDATION_SEASON,
            "observation_season": OBSERVATION_SEASON,
            "selection_metric": (
                "minimum development Brier score"
            ),
            "tie_break": (
                "lowest model weight"
            ),
            "future_data_used_for_weight_selection": False,
        },
        "model_weights_evaluated": list(
            MODEL_WEIGHTS
        ),
        "selected_model_weight": model_weight,
        "selected_market_weight": 1.0 - model_weight,
        "development_weight_grid": grid,
        "season_results": season_results,
        "gates": gates,
        "experiment_passed": all(gates.values()),
        "production_database_writes": False,
        "model_artifact_modified": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "automatic_promotion": False,
    }
    return result, blended_candidates, bets


def run_locked_market_blend(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    reports = base_dir / "reports"
    source_path = reports / SOURCE_CANDIDATES_NAME

    if not source_path.exists():
        raise FileNotFoundError(
            "Run multiseason market validation first"
        )

    candidates = pd.read_csv(source_path)
    result, blended_candidates, bets = (
        analyze_locked_market_blend(candidates)
    )
    _atomic_text(
        reports / REPORT_NAME,
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_text(
        reports / CANDIDATES_NAME,
        blended_candidates.to_csv(index=False),
    )
    _atomic_text(
        reports / BETS_NAME,
        bets.to_csv(index=False),
    )
    result["report_path"] = str(
        reports / REPORT_NAME
    )
    return result
