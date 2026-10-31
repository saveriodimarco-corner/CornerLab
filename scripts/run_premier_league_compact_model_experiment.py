from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPO_ROOT
    / "data"
    / "provider_cache"
    / "premier_league_validation"
    / "advanced_features_extended.parquet"
)

TARGET = "actual_total_corners"
LINES = (8.5, 9.5, 10.5, 11.5)


def add_safe_time_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.sort_values(
        ["season", "date", "match_id"],
        kind="stable",
    ).reset_index(drop=True).copy()

    target = result[TARGET].astype(float)
    prior = target.shift(1)

    for window in (20, 50, 100, 200):
        result[
            f"league_mean_last_{window}"
        ] = prior.rolling(
            window,
            min_periods=min(20, window),
        ).mean()

    result["league_std_last_100"] = (
        prior.rolling(
            100,
            min_periods=20,
        ).std()
    )

    result["league_expanding_mean"] = (
        prior.expanding(min_periods=20).mean()
    )

    result["season_expanding_mean"] = (
        result.groupby("season")[TARGET]
        .transform(
            lambda values: (
                values.astype(float)
                .shift(1)
                .expanding(min_periods=5)
                .mean()
            )
        )
    )

    return result


def probability_metrics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> dict:
    rows = {}
    model_scores = []
    baseline_scores = []

    train_target = train[TARGET].astype(float)
    validation_target = validation[
        TARGET
    ].astype(float).to_numpy()

    for line in LINES:
        observed = (
            validation_target > line
        ).astype(float)

        model_probability = poisson.sf(
            int(math.floor(line)),
            predictions,
        )

        baseline_probability = float(
            (train_target > line).mean()
        )

        model_brier = float(
            np.mean(
                (model_probability - observed) ** 2
            )
        )
        baseline_brier = float(
            np.mean(
                (
                    baseline_probability - observed
                )
                ** 2
            )
        )

        rows[str(line)] = {
            "model_brier": model_brier,
            "baseline_brier": baseline_brier,
            "improvement": (
                baseline_brier - model_brier
            ),
        }
        model_scores.append(model_brier)
        baseline_scores.append(baseline_brier)

    return {
        "lines": rows,
        "mean_model_brier": float(
            np.mean(model_scores)
        ),
        "mean_baseline_brier": float(
            np.mean(baseline_scores)
        ),
    }


def evaluate(
    estimator,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[dict, object]:
    model = clone(estimator)
    model.fit(
        train[features],
        train[TARGET].astype(float),
    )

    predictions = np.asarray(
        model.predict(validation[features]),
        dtype=float,
    )
    predictions = np.clip(
        predictions,
        0.1,
        30.0,
    )

    actual = validation[
        TARGET
    ].astype(float).to_numpy()

    baseline_prediction = float(
        train[TARGET].astype(float).mean()
    )
    baseline = np.full(
        len(validation),
        baseline_prediction,
    )

    mae = float(np.mean(np.abs(predictions - actual)))
    baseline_mae = float(
        np.mean(np.abs(baseline - actual))
    )
    rmse = float(
        np.sqrt(
            np.mean(
                (predictions - actual) ** 2
            )
        )
    )
    bias = float(
        np.mean(predictions - actual)
    )

    probabilities = probability_metrics(
        train,
        validation,
        predictions,
    )

    return (
        {
            "mae": mae,
            "baseline_mae": baseline_mae,
            "mae_improvement_fraction": (
                (baseline_mae - mae) / baseline_mae
            ),
            "rmse": rmse,
            "bias": bias,
            **probabilities,
        },
        model,
    )


def build_candidates():
    scaled = lambda estimator: Pipeline(
        [
            ("imputer", SimpleImputer(
                strategy="median"
            )),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )

    unscaled = lambda estimator: Pipeline(
        [
            ("imputer", SimpleImputer(
                strategy="median"
            )),
            ("model", estimator),
        ]
    )

    candidates = []

    for alpha in (0.1, 1.0, 10.0, 100.0):
        candidates.append(
            (
                f"ridge_alpha_{alpha:g}",
                scaled(Ridge(alpha=alpha)),
            )
        )

    for alpha in (
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    ):
        candidates.append(
            (
                f"poisson_alpha_{alpha:g}",
                scaled(
                    PoissonRegressor(
                        alpha=alpha,
                        max_iter=10000,
                    )
                ),
            )
        )

    for epsilon in (1.1, 1.35, 2.0):
        candidates.append(
            (
                f"huber_epsilon_{epsilon:g}",
                scaled(
                    HuberRegressor(
                        epsilon=epsilon,
                        alpha=1.0,
                        max_iter=3000,
                    )
                ),
            )
        )

    candidates.append(
        (
            "linear_regression",
            scaled(LinearRegression()),
        )
    )

    for leaves, regularization in (
        (7, 1.0),
        (7, 10.0),
        (15, 1.0),
        (15, 10.0),
    ):
        candidates.append(
            (
                (
                    "hist_gradient_"
                    f"leaves_{leaves}_"
                    f"l2_{regularization:g}"
                ),
                unscaled(
                    HistGradientBoostingRegressor(
                        max_leaf_nodes=leaves,
                        min_samples_leaf=30,
                        l2_regularization=(
                            regularization
                        ),
                        random_state=42,
                    )
                ),
            )
        )

    return candidates


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset mancante: {DATASET_PATH}"
        )

    frame = pd.read_parquet(DATASET_PATH)
    frame = add_safe_time_features(frame)

    feature_sets = {
        "league_only": [
            "league_mean_last_20",
            "league_mean_last_50",
            "league_mean_last_100",
            "league_mean_last_200",
            "league_std_last_100",
            "league_expanding_mean",
            "season_expanding_mean",
        ],
        "compact_form": [
            "league_mean_last_20",
            "league_mean_last_50",
            "league_mean_last_100",
            "season_expanding_mean",
            "expected_total_corners_baseline",
            "expected_total_corners",
            "total_corners_last5",
            "total_corners_last10",
            "away_total_corners_last5",
            "corners_for_last5",
            "away_corners_for_last5",
            "corners_against_last5",
            "away_corners_against_last5",
        ],
        "compact_strength": [
            "league_mean_last_20",
            "league_mean_last_50",
            "league_mean_last_100",
            "season_expanding_mean",
            "expected_total_corners_baseline",
            "expected_total_corners",
            "home_attack_rating",
            "away_attack_rating",
            "home_defense_rating",
            "away_defense_rating",
            "home_offensive_elo",
            "away_offensive_elo",
            "home_defensive_elo",
            "away_defensive_elo",
            "data_quality_score",
        ],
    }

    for key, features in feature_sets.items():
        missing = [
            feature
            for feature in features
            if feature not in frame.columns
        ]
        if missing:
            raise SystemExit(
                f"Feature mancanti per {key}: {missing}"
            )

    training_seasons = [
        "2016/17",
        "2017/18",
        "2018/19",
        "2019/20",
        "2020/21",
        "2021/22",
        "2022/23",
    ]

    selection_train = frame[
        frame["season"].isin(training_seasons)
    ].copy()
    selection_period = frame[
        frame["season"] == "2023/24"
    ].copy()

    confirmation_train = frame[
        frame["season"].isin(
            training_seasons + ["2023/24"]
        )
    ].copy()
    confirmation_period = frame[
        frame["season"] == "2024/25"
    ].copy()

    test_train = frame[
        frame["season"].isin(
            training_seasons
            + ["2023/24", "2024/25"]
        )
    ].copy()
    test_period = frame[
        frame["season"] == "2025/26"
    ].copy()

    current_train = frame[
        frame["season"].isin(
            training_seasons
            + [
                "2023/24",
                "2024/25",
                "2025/26",
            ]
        )
    ].copy()
    current_period = frame[
        frame["season"] == "2026/27"
    ].copy()

    tuning_rows = []

    warnings.filterwarnings("ignore")

    for feature_set_name, features in (
        feature_sets.items()
    ):
        for model_name, estimator in (
            build_candidates()
        ):
            metrics, _ = evaluate(
                estimator,
                features,
                selection_train,
                selection_period,
            )

            selection_score = (
                metrics["mae"]
                / metrics["baseline_mae"]
                + metrics["mean_model_brier"]
                / metrics["mean_baseline_brier"]
            )

            tuning_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "selection_score": float(
                        selection_score
                    ),
                    **metrics,
                }
            )

    tuning_rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["mae"],
        )
    )

    selected = tuning_rows[0]
    selected_features = feature_sets[
        selected["feature_set"]
    ]

    estimator_lookup = {
        name: estimator
        for name, estimator in build_candidates()
    }
    selected_estimator = estimator_lookup[
        selected["model_name"]
    ]

    confirmation, _ = evaluate(
        selected_estimator,
        selected_features,
        confirmation_train,
        confirmation_period,
    )

    test_metrics, _ = evaluate(
        selected_estimator,
        selected_features,
        test_train,
        test_period,
    )

    current_test, _ = evaluate(
        selected_estimator,
        selected_features,
        current_train,
        current_period,
    )

    gates = {
        "selection_mae_improvement": (
            selected[
                "mae_improvement_fraction"
            ]
            >= 0.005
        ),
        "selection_brier_improvement": (
            selected["mean_model_brier"]
            < selected["mean_baseline_brier"]
        ),
        "selection_bias": (
            abs(selected["bias"]) <= 0.5
        ),
        "confirmation_mae_improvement": (
            confirmation[
                "mae_improvement_fraction"
            ]
            >= 0.005
        ),
        "confirmation_brier_improvement": (
            confirmation["mean_model_brier"]
            < confirmation[
                "mean_baseline_brier"
            ]
        ),
        "confirmation_bias": (
            abs(confirmation["bias"]) <= 0.5
        ),
        "test_mae_improvement": (
            test_metrics[
                "mae_improvement_fraction"
            ]
            >= 0.005
        ),
        "test_brier_improvement": (
            test_metrics["mean_model_brier"]
            < test_metrics[
                "mean_baseline_brier"
            ]
        ),
        "test_bias": (
            abs(test_metrics["bias"]) <= 0.5
        ),
        "current_mae_improvement": (
            current_test[
                "mae_improvement_fraction"
            ]
            >= 0.005
        ),
        "current_brier_improvement": (
            current_test["mean_model_brier"]
            < current_test[
                "mean_baseline_brier"
            ]
        ),
        "current_bias": (
            abs(current_test["bias"]) <= 0.5
        ),
        "no_selection_line_materially_worse": all(
            item["model_brier"]
            <= item["baseline_brier"] + 0.01
            for item in selected["lines"].values()
        ),
        "no_confirmation_line_materially_worse": all(
            item["model_brier"]
            <= item["baseline_brier"] + 0.01
            for item in confirmation[
                "lines"
            ].values()
        ),
        "no_test_line_materially_worse": all(
            item["model_brier"]
            <= item["baseline_brier"] + 0.01
            for item in test_metrics[
                "lines"
            ].values()
        ),
        "no_current_line_materially_worse": all(
            item["model_brier"]
            <= item["baseline_brier"] + 0.01
            for item in current_test[
                "lines"
            ].values()
        ),
    }

    result = {
        "run_type": (
            "premier_league_compact_model_experiment"
        ),
        "training_period": "2016/17-2022/23",
        "selection_period": "2023/24",
        "confirmation_period": "2024/25",
        "test_period": "2025/26",
        "current_test_period": "2026/27",
        "selected_feature_set": selected[
            "feature_set"
        ],
        "selected_model": selected["model_name"],
        "selected_features": selected_features,
        "selection_metrics": selected,
        "confirmation_metrics": confirmation,
        "test_metrics": test_metrics,
        "current_test_metrics": current_test,
        "gates": gates,
        "experiment_passed": all(gates.values()),
        "top_tuning_candidates": tuning_rows[:10],
    }

    output_path = (
        REPO_ROOT
        / "reports"
        / (
            "premier_league_compact_"
            "model_experiment.json"
        )
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "selected_feature_set": result[
                    "selected_feature_set"
                ],
                "selected_model": result[
                    "selected_model"
                ],
                "selection_mae": selected["mae"],
                "selection_baseline_mae": selected[
                    "baseline_mae"
                ],
                "confirmation_metrics": (
                    confirmation
                ),
                "test_metrics": test_metrics,
                "current_test_metrics": (
                    current_test
                ),
                "gates": gates,
                "experiment_passed": result[
                    "experiment_passed"
                ],
                "report_path": str(output_path),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
