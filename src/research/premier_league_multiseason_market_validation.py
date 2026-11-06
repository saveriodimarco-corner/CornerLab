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
from src.research.premier_league_forward_model import (
    ELIGIBLE_LINES,
    FEATURES,
    add_safe_time_features,
    build_base_model,
)
from src.research.premier_league_historical_odds import (
    _fixture_key,
)
from src.research.premier_league_market_validation import (
    _online_predictions,
    _probability,
    _summarize_bets,
)


EVALUATION_SEASONS = ("2025/26", "2026/27")
SEASON_ORDER = (
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
    "2026/27",
)
REPORT_NAME = (
    "premier_league_multiseason_market_validation.json"
)
CANDIDATES_NAME = (
    "premier_league_multiseason_candidates.csv"
)
BETS_NAME = (
    "premier_league_multiseason_bets.csv"
)


def training_seasons(
    evaluation_season: str,
) -> tuple[str, ...]:
    position = SEASON_ORDER.index(
        evaluation_season
    )
    return SEASON_ORDER[:position]


def _atomic_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _calibration_metrics(
    candidates: pd.DataFrame,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []

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
        actual_over = float(
            float(over_row["actual_total_corners"])
            > float(line)
        )

        rows.append(
            {
                "model": float(
                    over_row["predicted_probability"]
                ),
                "market": market_over,
                "actual": actual_over,
            }
        )

    frame = pd.DataFrame(rows)

    return {
        "model_brier": float(
            np.mean(
                (frame["model"] - frame["actual"])
                ** 2
            )
        ),
        "market_brier": float(
            np.mean(
                (frame["market"] - frame["actual"])
                ** 2
            )
        ),
    }


def run_multiseason_market_validation(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    cache = (
        base_dir
        / "data/provider_cache/"
        / "premier_league_validation"
    )
    reports = base_dir / "reports"

    features = pd.read_parquet(
        cache / "advanced_features_extended.parquet"
    )
    features = add_safe_time_features(features)
    odds = pd.read_parquet(
        cache / "historical_corner_odds.parquet"
    )
    odds = odds.loc[
        odds["season"].astype(str).isin(
            EVALUATION_SEASONS
        )
    ].copy()

    calibration = json.loads(
        (
            reports
            / "premier_league_online_calibration_experiment.json"
        ).read_text(encoding="utf-8")
    )
    settings = calibration["settings"]

    if tuple(calibration["features"]) != FEATURES:
        raise ValueError(
            "Schema feature calibrazione non coerente"
        )

    prediction_frames: list[pd.DataFrame] = []
    traces: dict[str, list[dict[str, Any]]] = {}

    for season in EVALUATION_SEASONS:
        training = features.loc[
            features["season"].astype(str).isin(
                training_seasons(season)
            )
        ].copy()
        evaluation = (
            features.loc[
                features["season"].astype(str)
                == season
            ]
            .sort_values(
                ["date", "match_id"],
                kind="stable",
            )
            .reset_index(drop=True)
            .copy()
        )

        model = build_base_model()
        model.fit(
            training[list(FEATURES)],
            training[
                "actual_total_corners"
            ].astype(float),
        )
        raw = np.asarray(
            model.predict(
                evaluation[list(FEATURES)]
            ),
            dtype=float,
        )
        corrected, trace = _online_predictions(
            evaluation,
            raw,
            window=int(settings["window"]),
            shrinkage=float(
                settings["shrinkage"]
            ),
            minimum_history=int(
                settings["minimum_history"]
            ),
        )
        traces[season] = trace
        evaluation[
            "model_expected_total"
        ] = corrected
        evaluation["fixture_key"] = [
            _fixture_key(
                {
                    "date": pd.to_datetime(
                        row["date"]
                    ).date().isoformat(),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                }
            )
            for row in evaluation.to_dict(
                "records"
            )
        ]

        columns = [
            "fixture_key",
            "match_id",
            "season",
            "date",
            "home_team",
            "away_team",
            "actual_total_corners",
            "model_expected_total",
        ]

        if (
            "data_quality_score"
            in evaluation.columns
        ):
            columns.append("data_quality_score")
        if (
            "insufficient_history"
            in evaluation.columns
        ):
            columns.append(
                "insufficient_history"
            )

        prediction_frames.append(
            evaluation[columns].copy()
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    reference = (
        odds.loc[
            odds["line"].astype(float).isin(
                [
                    float(value)
                    for value in ELIGIBLE_LINES
                ]
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
        predictions,
        on="fixture_key",
        how="inner",
        validate="many_to_one",
    )
    candidates["predicted_probability"] = [
        _probability(mu, line, side)
        for mu, line, side in zip(
            candidates["model_expected_total"],
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
    if "data_quality_score" in candidates:
        quality &= (
            pd.to_numeric(
                candidates["data_quality_score"],
                errors="coerce",
            ).fillna(0.0)
            >= 1.0
        )
    if "insufficient_history" in candidates:
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
            ["season", "date", "match_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    combined_calibration = _calibration_metrics(
        candidates
    )
    combined_bets = _summarize_bets(bets)
    season_results: dict[str, Any] = {}

    for season in EVALUATION_SEASONS:
        season_predictions = predictions.loc[
            predictions["season"].astype(str)
            == season
        ]
        season_candidates = candidates.loc[
            candidates["season"].astype(str)
            == season
        ]
        season_bets = bets.loc[
            bets["season"].astype(str) == season
        ]
        actual = season_predictions[
            "actual_total_corners"
        ].astype(float).to_numpy()
        expected = season_predictions[
            "model_expected_total"
        ].astype(float).to_numpy()
        calibration_metrics = (
            _calibration_metrics(
                season_candidates
            )
        )

        season_results[season] = {
            "matches": int(
                len(season_predictions)
            ),
            "mae": float(
                np.mean(
                    np.abs(expected - actual)
                )
            ),
            "bias": float(
                np.mean(expected - actual)
            ),
            **calibration_metrics,
            "qualified_bet_summary": (
                _summarize_bets(
                    season_bets
                )
            ),
        }

    gates = {
        "all_460_matches_have_odds": (
            odds["fixture_key"].nunique()
            == len(predictions)
            == 460
        ),
        "strict_walk_forward_training": True,
        "bias_at_most_0_5_each_season": all(
            abs(
                season_results[season]["bias"]
            )
            <= 0.5
            for season in EVALUATION_SEASONS
        ),
        "combined_model_brier_better_than_market": (
            combined_calibration["model_brier"]
            < combined_calibration["market_brier"]
        ),
        "model_not_materially_worse_each_season": all(
            season_results[season]["model_brier"]
            <= (
                season_results[season][
                    "market_brier"
                ]
                + 0.005
            )
            for season in EVALUATION_SEASONS
        ),
        "at_least_30_qualified_bets": (
            combined_bets["bets"] >= 30
        ),
        "positive_roi": (
            combined_bets["roi"] is not None
            and combined_bets["roi"] > 0.0
        ),
        "roi_at_least_3_percent": (
            combined_bets["roi"] is not None
            and combined_bets["roi"] >= 0.03
        ),
        "roi_lower_95_percent_above_zero": (
            combined_bets["roi_ci_95"][0]
            is not None
            and combined_bets["roi_ci_95"][0]
            > 0.0
        ),
        "positive_roi_each_season": all(
            (
                season_results[season][
                    "qualified_bet_summary"
                ]["roi"]
                is not None
                and season_results[season][
                    "qualified_bet_summary"
                ]["roi"]
                > 0.0
            )
            for season in EVALUATION_SEASONS
        ),
    }

    result = {
        "run_type": (
            "premier_league_multiseason_"
            "market_validation"
        ),
        "evaluation_seasons": list(
            EVALUATION_SEASONS
        ),
        "matches": int(len(predictions)),
        "matches_with_odds": int(
            odds["fixture_key"].nunique()
        ),
        "historical_odds_rows": int(len(odds)),
        "reference_price": (
            "median bookmaker price at kickoff minus 90 minutes"
        ),
        "model": "poisson_alpha_10",
        "eligible_lines": list(ELIGIBLE_LINES),
        "online_calibration_settings": settings,
        **combined_calibration,
        "qualified_bet_summary": combined_bets,
        "season_results": season_results,
        "gates": gates,
        "market_validation_passed": all(
            gates.values()
        ),
        "calibration_trace": traces,
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
