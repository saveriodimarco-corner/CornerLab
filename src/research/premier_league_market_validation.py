from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.research.decision_engine import (
    MIN_EV,
    MIN_MARKET_EDGE,
    MIN_ODDS,
    MIN_PREDICTED_PROBABILITY,
    minimum_acceptable_odds,
)
from src.research.premier_league_forward_model import (
    ELIGIBLE_LINES,
    FEATURES,
    add_safe_time_features,
    load_forward_model,
)
from src.research.premier_league_historical_odds import (
    _fixture_key,
)


SEASON = "2026/27"
REPORT_NAME = "premier_league_market_validation.json"
CANDIDATES_NAME = (
    "premier_league_market_validation_candidates.csv"
)
BETS_NAME = "premier_league_market_validation_bets.csv"


def _atomic_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _online_predictions(
    frame: pd.DataFrame,
    raw_predictions: np.ndarray,
    *,
    window: int,
    shrinkage: float,
    minimum_history: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ordered = frame.reset_index(drop=True).copy()
    raw = np.asarray(
        raw_predictions,
        dtype=float,
    )
    corrected = raw.copy()
    residual_history: list[float] = []
    trace: list[dict[str, Any]] = []

    for date_value, group in ordered.groupby(
        "date",
        sort=True,
    ):
        indices = group.index.to_numpy()

        if len(residual_history) >= minimum_history:
            offset = float(
                np.mean(
                    residual_history[-window:]
                )
            )
        else:
            offset = 0.0

        correction = shrinkage * offset
        corrected[indices] = np.clip(
            raw[indices] - correction,
            0.1,
            30.0,
        )

        actual = group[
            "actual_total_corners"
        ].astype(float).to_numpy()
        residuals = raw[indices] - actual
        previous_count = len(residual_history)
        residual_history.extend(
            residuals.tolist()
        )

        trace.append(
            {
                "date": str(date_value),
                "prior_residuals": previous_count,
                "raw_bias_offset": offset,
                "applied_correction": correction,
                "matches": int(len(indices)),
            }
        )

    return corrected, trace


def _probability(
    expected_total: float,
    line: float,
    side: str,
) -> float:
    threshold = int(np.floor(float(line)))
    under = float(
        poisson.cdf(
            threshold,
            max(0.1, float(expected_total)),
        )
    )
    return under if side == "UNDER" else 1.0 - under


def _summarize_bets(
    bets: pd.DataFrame,
    *,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    if bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": None,
            "profit_units": 0.0,
            "roi": None,
            "roi_ci_95": [None, None],
            "average_odds": None,
            "average_probability": None,
            "average_expected_value": None,
            "maximum_drawdown_units": 0.0,
        }

    profits = np.where(
        bets["won"].astype(bool).to_numpy(),
        bets["reference_odds"].astype(float).to_numpy()
        - 1.0,
        -1.0,
    )
    cumulative = np.cumsum(profits)
    peaks = np.maximum.accumulate(
        np.concatenate(([0.0], cumulative))
    )[1:]
    drawdowns = peaks - cumulative

    rng = np.random.default_rng(20261101)
    indices = rng.integers(
        0,
        len(profits),
        size=(bootstrap_samples, len(profits)),
    )
    bootstrap_roi = profits[indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_roi,
        [0.025, 0.975],
    )

    wins = int(bets["won"].astype(bool).sum())
    count = int(len(bets))

    return {
        "bets": count,
        "wins": wins,
        "losses": count - wins,
        "hit_rate": float(wins / count),
        "profit_units": float(profits.sum()),
        "roi": float(profits.mean()),
        "roi_ci_95": [
            float(lower),
            float(upper),
        ],
        "average_odds": float(
            bets["reference_odds"].mean()
        ),
        "average_probability": float(
            bets["predicted_probability"].mean()
        ),
        "average_expected_value": float(
            bets["expected_value"].mean()
        ),
        "maximum_drawdown_units": float(
            drawdowns.max(initial=0.0)
        ),
    }


def run_market_validation(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    cache = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    reports = base_dir / "reports"

    features_path = (
        cache / "advanced_features_extended.parquet"
    )
    odds_path = cache / "historical_corner_odds.parquet"
    calibration_path = (
        reports
        / "premier_league_online_calibration_experiment.json"
    )

    features = pd.read_parquet(features_path)
    features = add_safe_time_features(features)
    current = (
        features.loc[
            features["season"].astype(str) == SEASON
        ]
        .sort_values(
            ["date", "match_id"],
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )
    odds = pd.read_parquet(odds_path)

    model, manifest = load_forward_model(base_dir)
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    settings = calibration["settings"]

    if tuple(calibration["features"]) != FEATURES:
        raise ValueError(
            "Feature della calibrazione non coerenti"
        )

    raw_predictions = np.asarray(
        model.base_model.predict(
            current[list(FEATURES)]
        ),
        dtype=float,
    )
    predictions, trace = _online_predictions(
        current,
        raw_predictions,
        window=int(settings["window"]),
        shrinkage=float(settings["shrinkage"]),
        minimum_history=int(
            settings["minimum_history"]
        ),
    )

    current["expected_total_corners"] = predictions
    current["fixture_key"] = [
        _fixture_key(
            {
                "date": pd.to_datetime(
                    row["date"]
                ).date().isoformat(),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
        )
        for row in current.to_dict("records")
    ]

    prediction_columns = [
        "fixture_key",
        "match_id",
        "date",
        "home_team",
        "away_team",
        "actual_total_corners",
        "expected_total_corners",
    ]

    if "data_quality_score" in current.columns:
        prediction_columns.append(
            "data_quality_score"
        )
    if "insufficient_history" in current.columns:
        prediction_columns.append(
            "insufficient_history"
        )

    prediction_frame = current[
        prediction_columns
    ].copy()

    reference = (
        odds.loc[
            odds["line"].astype(float).isin(
                [float(value) for value in ELIGIBLE_LINES]
            )
        ]
        .groupby(
            ["fixture_key", "line", "side"],
            as_index=False,
        )
        .agg(
            reference_odds=(
                "decimal_odds",
                "median",
            ),
            best_odds=("decimal_odds", "max"),
            bookmaker_count=(
                "bookmaker_key",
                "nunique",
            ),
        )
    )

    candidates = reference.merge(
        prediction_frame,
        on="fixture_key",
        how="inner",
        validate="many_to_one",
    )
    candidates["predicted_probability"] = [
        _probability(mu, line, side)
        for mu, line, side in zip(
            candidates["expected_total_corners"],
            candidates["line"],
            candidates["side"],
        )
    ]
    candidates["implied_probability"] = (
        1.0 / candidates["reference_odds"]
    )
    candidates["market_edge"] = (
        candidates["predicted_probability"]
        - candidates["implied_probability"]
    )
    candidates["expected_value"] = (
        candidates["predicted_probability"]
        * candidates["reference_odds"]
        - 1.0
    )
    candidates["minimum_acceptable_odds"] = [
        minimum_acceptable_odds(value)
        for value in candidates[
            "predicted_probability"
        ]
    ]

    quality = pd.Series(
        True,
        index=candidates.index,
    )
    if "data_quality_score" in candidates.columns:
        quality &= (
            pd.to_numeric(
                candidates["data_quality_score"],
                errors="coerce",
            ).fillna(0.0)
            >= 1.0
        )
    if "insufficient_history" in candidates.columns:
        quality &= ~candidates[
            "insufficient_history"
        ].fillna(True).astype(bool)

    candidates["quality_ok"] = quality
    candidates["passed"] = (
        candidates["quality_ok"]
        & (
            candidates["predicted_probability"]
            >= MIN_PREDICTED_PROBABILITY
        )
        & (candidates["reference_odds"] >= MIN_ODDS)
        & (
            candidates["market_edge"]
            >= MIN_MARKET_EDGE
        )
        & (candidates["expected_value"] >= MIN_EV)
        & (
            candidates["reference_odds"]
            >= candidates["minimum_acceptable_odds"]
        )
    )
    candidates["won"] = np.where(
        candidates["side"] == "OVER",
        candidates["actual_total_corners"]
        > candidates["line"],
        candidates["actual_total_corners"]
        < candidates["line"],
    )

    bets = (
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
            ["date", "match_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    calibration_rows: list[dict[str, Any]] = []

    for (
        fixture_key,
        line,
    ), group in candidates.groupby(
        ["fixture_key", "line"],
        sort=False,
    ):
        over = group.loc[group["side"] == "OVER"]
        under = group.loc[group["side"] == "UNDER"]

        if over.empty or under.empty:
            continue

        over_row = over.iloc[0]
        under_row = under.iloc[0]
        inverse_over = (
            1.0 / float(over_row["reference_odds"])
        )
        inverse_under = (
            1.0 / float(under_row["reference_odds"])
        )
        market_over = inverse_over / (
            inverse_over + inverse_under
        )
        model_over = float(
            over_row["predicted_probability"]
        )
        actual_over = float(
            float(over_row["actual_total_corners"])
            > float(line)
        )

        calibration_rows.append(
            {
                "fixture_key": fixture_key,
                "line": float(line),
                "actual_over": actual_over,
                "model_over": model_over,
                "market_over": market_over,
            }
        )

    calibration_frame = pd.DataFrame(
        calibration_rows
    )
    model_brier = float(
        np.mean(
            (
                calibration_frame["model_over"]
                - calibration_frame["actual_over"]
            )
            ** 2
        )
    )
    market_brier = float(
        np.mean(
            (
                calibration_frame["market_over"]
                - calibration_frame["actual_over"]
            )
            ** 2
        )
    )

    actual = current[
        "actual_total_corners"
    ].astype(float).to_numpy()
    bias = float(
        np.mean(predictions - actual)
    )
    mae = float(
        np.mean(np.abs(predictions - actual))
    )
    bet_summary = _summarize_bets(bets)

    gates = {
        "all_completed_matches_have_odds": (
            odds["fixture_key"].nunique()
            == len(current)
        ),
        "walk_forward_without_same_day_leakage": True,
        "absolute_bias_at_most_0_5": (
            abs(bias) <= 0.5
        ),
        "model_brier_better_than_market": (
            model_brier < market_brier
        ),
        "at_least_30_qualified_bets": (
            bet_summary["bets"] >= 30
        ),
        "positive_roi": (
            bet_summary["roi"] is not None
            and bet_summary["roi"] > 0.0
        ),
        "roi_at_least_3_percent": (
            bet_summary["roi"] is not None
            and bet_summary["roi"] >= 0.03
        ),
        "roi_lower_95_percent_above_zero": (
            bet_summary["roi_ci_95"][0] is not None
            and bet_summary["roi_ci_95"][0] > 0.0
        ),
    }

    result = {
        "run_type": "premier_league_market_validation",
        "season": SEASON,
        "matches": int(len(current)),
        "matches_with_historical_odds": int(
            odds["fixture_key"].nunique()
        ),
        "historical_odds_rows": int(len(odds)),
        "reference_price": (
            "median bookmaker price at kickoff minus 90 minutes"
        ),
        "eligible_lines": list(ELIGIBLE_LINES),
        "online_calibration_settings": settings,
        "model_hash": manifest.get("artifact_hash"),
        "mae": mae,
        "bias": bias,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "qualified_bet_summary": bet_summary,
        "gates": gates,
        "market_validation_passed": all(
            gates.values()
        ),
        "calibration_trace": trace,
        "production_database_writes": False,
        "model_artifact_modified": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    reports.mkdir(parents=True, exist_ok=True)
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
        candidates.to_csv(index=False),
    )
    _atomic_text(
        reports / BETS_NAME,
        bets.to_csv(index=False),
    )

    return result
