from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_premier_league_compact_model_experiment as compact


REPORT_PATH = (
    REPO_ROOT
    / "reports"
    / "premier_league_compact_model_experiment.json"
)


def raw_predictions(
    estimator,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> np.ndarray:
    model = clone(estimator)
    model.fit(
        train[features],
        train[compact.TARGET].astype(float),
    )
    return np.clip(
        np.asarray(
            model.predict(validation[features]),
            dtype=float,
        ),
        0.1,
        30.0,
    )


def online_bias_correction(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    *,
    window: int,
    shrinkage: float,
    minimum_history: int,
) -> tuple[np.ndarray, list[dict]]:
    frame = validation.reset_index(drop=True).copy()
    frame["_raw_prediction"] = predictions

    corrected = np.asarray(
        predictions,
        dtype=float,
    ).copy()

    residual_history: list[float] = []
    trace: list[dict] = []

    for date_value, group in frame.groupby(
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

        corrected[indices] = np.clip(
            predictions[indices]
            - shrinkage * offset,
            0.1,
            30.0,
        )

        actual = group[
            compact.TARGET
        ].astype(float).to_numpy()

        raw_residuals = (
            predictions[indices] - actual
        )
        residual_history.extend(
            raw_residuals.tolist()
        )

        trace.append(
            {
                "date": str(date_value),
                "prior_residuals": int(
                    len(residual_history)
                    - len(indices)
                ),
                "offset": offset,
                "applied_correction": (
                    shrinkage * offset
                ),
            }
        )

    return corrected, trace


def score_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> dict:
    actual = validation[
        compact.TARGET
    ].astype(float).to_numpy()

    baseline_value = float(
        train[compact.TARGET]
        .astype(float)
        .mean()
    )
    baseline = np.full(
        len(validation),
        baseline_value,
    )

    mae = float(
        np.mean(np.abs(predictions - actual))
    )
    baseline_mae = float(
        np.mean(np.abs(baseline - actual))
    )

    probability = compact.probability_metrics(
        train,
        validation,
        predictions,
    )

    return {
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement_fraction": (
            (baseline_mae - mae) / baseline_mae
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    (predictions - actual) ** 2
                )
            )
        ),
        "bias": float(
            np.mean(predictions - actual)
        ),
        **probability,
    }


def evaluate_calibrated(
    estimator,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    window: int,
    shrinkage: float,
    minimum_history: int,
) -> tuple[dict, list[dict]]:
    predictions = raw_predictions(
        estimator,
        features,
        train,
        validation,
    )
    corrected, trace = online_bias_correction(
        validation,
        predictions,
        window=window,
        shrinkage=shrinkage,
        minimum_history=minimum_history,
    )
    return (
        score_predictions(
            train,
            validation,
            corrected,
        ),
        trace,
    )


def period_gate(metrics: dict) -> dict:
    return {
        "mae_improvement": (
            metrics[
                "mae_improvement_fraction"
            ]
            >= 0.005
        ),
        "brier_improvement": (
            metrics["mean_model_brier"]
            < metrics["mean_baseline_brier"]
        ),
        "bias": abs(metrics["bias"]) <= 0.5,
        "no_line_materially_worse": all(
            row["model_brier"]
            <= row["baseline_brier"] + 0.01
            for row in metrics["lines"].values()
        ),
    }


def main() -> None:
    payload = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    frame = pd.read_parquet(
        compact.DATASET_PATH
    )
    frame = compact.add_safe_time_features(
        frame
    )

    features = list(
        payload["selected_features"]
    )
    model_name = str(
        payload["selected_model"]
    )

    estimators = {
        name: estimator
        for name, estimator in (
            compact.build_candidates()
        )
    }
    estimator = estimators[model_name]

    old_seasons = [
        "2016/17",
        "2017/18",
        "2018/19",
        "2019/20",
        "2020/21",
        "2021/22",
        "2022/23",
    ]

    selection_train = frame[
        frame["season"].isin(old_seasons)
    ].copy()
    selection_period = frame[
        frame["season"] == "2023/24"
    ].copy()

    confirmation_train = frame[
        frame["season"].isin(
            old_seasons + ["2023/24"]
        )
    ].copy()
    confirmation_period = frame[
        frame["season"] == "2024/25"
    ].copy()

    test_train = frame[
        frame["season"].isin(
            old_seasons
            + ["2023/24", "2024/25"]
        )
    ].copy()
    test_period = frame[
        frame["season"] == "2025/26"
    ].copy()

    current_train = frame[
        frame["season"].isin(
            old_seasons
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

    warnings.filterwarnings("ignore")
    candidates = []

    for window in (10, 20, 40, 80):
        for shrinkage in (
            0.25,
            0.5,
            0.75,
            1.0,
        ):
            for minimum_history in (
                5,
                10,
                20,
            ):
                metrics, _ = evaluate_calibrated(
                    estimator,
                    features,
                    selection_train,
                    selection_period,
                    window=window,
                    shrinkage=shrinkage,
                    minimum_history=(
                        minimum_history
                    ),
                )

                score = (
                    metrics["mae"]
                    / metrics["baseline_mae"]
                    + metrics[
                        "mean_model_brier"
                    ]
                    / metrics[
                        "mean_baseline_brier"
                    ]
                )

                candidates.append(
                    {
                        "window": window,
                        "shrinkage": shrinkage,
                        "minimum_history": (
                            minimum_history
                        ),
                        "selection_score": float(
                            score
                        ),
                        **metrics,
                    }
                )

    candidates.sort(
        key=lambda row: (
            row["selection_score"],
            row["mae"],
        )
    )
    selected = candidates[0]

    settings = {
        "window": int(selected["window"]),
        "shrinkage": float(
            selected["shrinkage"]
        ),
        "minimum_history": int(
            selected["minimum_history"]
        ),
    }

    confirmation, _ = evaluate_calibrated(
        estimator,
        features,
        confirmation_train,
        confirmation_period,
        **settings,
    )
    test_metrics, _ = evaluate_calibrated(
        estimator,
        features,
        test_train,
        test_period,
        **settings,
    )
    current, current_trace = evaluate_calibrated(
        estimator,
        features,
        current_train,
        current_period,
        **settings,
    )

    period_gates = {
        "selection": period_gate(selected),
        "confirmation": period_gate(
            confirmation
        ),
        "test": period_gate(test_metrics),
        "current": period_gate(current),
    }

    all_gates = {
        f"{period}_{name}": value
        for period, gates in period_gates.items()
        for name, value in gates.items()
    }

    result = {
        "run_type": (
            "premier_league_online_"
            "calibration_experiment"
        ),
        "base_model": model_name,
        "features": features,
        "settings": settings,
        "selection_metrics": selected,
        "confirmation_metrics": confirmation,
        "test_metrics": test_metrics,
        "current_metrics": current,
        "period_gates": period_gates,
        "experiment_passed": all(
            all_gates.values()
        ),
        "current_calibration_trace": (
            current_trace
        ),
        "top_settings": candidates[:10],
    }

    output_path = (
        REPO_ROOT
        / "reports"
        / (
            "premier_league_online_"
            "calibration_experiment.json"
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
                "base_model": model_name,
                "settings": settings,
                "selection_metrics": selected,
                "confirmation_metrics": (
                    confirmation
                ),
                "test_metrics": test_metrics,
                "current_metrics": current,
                "period_gates": period_gates,
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
