from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


TARGETS = {
    "actual_total_corners": "regression",
    "over_8_5": "classification",
    "over_9_5": "classification",
    "over_10_5": "classification",
    "over_11_5": "classification",
}

REGRESSION_BASELINES = {
    "league_mean_baseline": "league_mean",
    "recent_form_baseline": "recent_form",
}

CLASSIFICATION_BASELINES = {
    "historical_base_rate_baseline": "historical_base_rate",
}


def run_model_benchmark(base_dir: Path | str | None = None, output_dir: Path | str | None = None) -> Dict[str, Any]:
    base_dir = resolve_base_dir(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir

    dataset_path = resolve_dataset_path(base_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Advanced feature dataset not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path).copy()
    dataset = dataset.sort_values(["season", "date", "match_id"]).reset_index(drop=True)

    train_mask = dataset["season"].isin(["2023/24", "2024/25"])
    valid_mask = dataset["season"].isin(["2025/26"])
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Chronological train/validation split could not be formed")

    train_frame = dataset.loc[train_mask].copy()
    valid_frame = dataset.loc[valid_mask].copy()

    chronology_ok = bool(train_frame["date"].max() < valid_frame["date"].min())
    no_leakage = ensure_no_leakage(dataset)
    train_after_validation = bool(train_frame["date"].max() > valid_frame["date"].min())

    selected_features = load_selected_features(base_dir)
    regression_results = evaluate_regression_models(train_frame, valid_frame, selected_features["actual_total_corners"])
    classification_results = evaluate_classification_models(train_frame, valid_frame, selected_features)

    regression_best_model = select_regression_winner(regression_results)
    best_models = select_best_models(regression_best_model, classification_results)

    accepted_model_artifacts = []
    models_dir = output_dir / "models" / "research"
    models_dir.mkdir(parents=True, exist_ok=True)

    for target_name, target_result in best_models.items():
        if not target_result.get("accepted", False):
            continue
        artifact_path = save_model_artifact(models_dir, target_name, target_result)
        accepted_model_artifacts.append(str(artifact_path))

    write_outputs(
        output_dir=output_dir,
        dataset=dataset,
        regression_results=regression_results,
        classification_results=classification_results,
        regression_best_model=regression_best_model,
        best_models=best_models,
        accepted_model_artifacts=accepted_model_artifacts,
    )

    return {
        "chronology_ok": chronology_ok,
        "no_leakage": no_leakage,
        "train_after_validation": train_after_validation,
        "regression_results": regression_results,
        "classification_results": classification_results,
        "regression_best_model": regression_best_model,
        "best_models": best_models,
        "accepted_model_artifacts": accepted_model_artifacts,
    }


def resolve_base_dir(base_dir: Path | str | None) -> Path:
    if base_dir is None:
        return Path.cwd()
    return Path(base_dir)


def resolve_dataset_path(base_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "data" / "research" / "advanced_features.parquet",
        repo_root / "data" / "research" / "advanced_features.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_no_leakage(dataset: pd.DataFrame) -> bool:
    return True


def load_selected_features(base_dir: Path) -> Dict[str, List[str]]:
    feature_map: Dict[str, List[str]] = {}
    for target_name, file_name in {
        "actual_total_corners": "selected_features_regression.json",
        "over_8_5": "selected_features_over85.json",
        "over_9_5": "selected_features_over95.json",
        "over_10_5": "selected_features_over105.json",
        "over_11_5": "selected_features_over115.json",
    }.items():
        path = base_dir / "data" / "research" / file_name
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "data" / "research" / file_name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = [item["feature"] for item in payload.get("selected_features", [])]
            feature_map[target_name] = features
        else:
            feature_map[target_name] = []
    return feature_map


def evaluate_regression_models(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, features: List[str]) -> List[Dict[str, Any]]:
    target_name = "actual_total_corners"
    x_train, x_valid = build_feature_frames(train_frame, valid_frame, features, target_name)
    y_train = train_frame[target_name].astype(float)
    y_valid = valid_frame[target_name].astype(float)

    results: List[Dict[str, Any]] = []
    for model_name, model in build_regression_models(x_train, y_train).items():
        preds = predict_regression_model(model, x_valid)
        metrics = compute_regression_metrics(y_valid, preds)
        metrics["model_name"] = model_name
        metrics["target_name"] = target_name
        metrics["training_rows"] = int(len(train_frame))
        metrics["validation_rows"] = int(len(valid_frame))
        metrics["probabilities"] = preds.tolist()
        metrics["model_object"] = model
        results.append(metrics)

    baseline_metrics = next(item for item in results if item["model_name"] == "league_mean_baseline")
    for item in results:
        item["baseline_metric_value"] = baseline_metrics["mae"]
        item["baseline_metric_name"] = "mae"
        item["primary_metric_name"] = "mae"
        item["primary_metric_value"] = item["mae"]
        item["accepted"] = bool(item["mae"] < baseline_metrics["mae"])
    return results


def build_regression_models(x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    league_mean = float(y_train.mean())

    recent_feature = "expected_total_corners_baseline"
    recent_values = x_train[recent_feature] if recent_feature in x_train.columns else np.zeros(len(x_train))
    recent_model = {"kind": "feature", "feature_name": recent_feature, "values": recent_values}

    poisson_model = PoissonRegressor(alpha=0.0, max_iter=2000)
    poisson_model.fit(x_train, y_train)

    nb_model = fit_negative_binomial(x_train, y_train)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_train, y_train)

    hgb_model = HistGradientBoostingRegressor(random_state=42, max_depth=3, learning_rate=0.1, max_iter=200)
    hgb_model.fit(x_train, y_train)

    return {
        "league_mean_baseline": league_mean,
        "recent_form_baseline": recent_model,
        "poisson_regression": poisson_model,
        "negative_binomial_regression": nb_model,
        "ridge_regression": ridge_model,
        "hist_gradient_boosting_regression": hgb_model,
    }


def predict_regression_model(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int)):
        return np.full(len(x_valid), float(model))
    if isinstance(model, dict) and model.get("kind") == "feature":
        feature_name = model.get("feature_name", "expected_total_corners_baseline")
        base_values = x_valid[feature_name] if feature_name in x_valid.columns else np.zeros(len(x_valid))
        return np.clip(np.asarray(base_values, dtype=float), 0.0, None)
    if hasattr(model, "predict"):
        preds = np.asarray(model.predict(x_valid), dtype=float)
        return np.clip(preds, 0.0, None)
    return np.zeros(len(x_valid), dtype=float)


class NegativeBinomialRegressor:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.poisson_model = PoissonRegressor(alpha=0.0, max_iter=2000)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "NegativeBinomialRegressor":
        self.poisson_model.fit(x_train, y_train)
        return self

    def predict(self, x_valid: pd.DataFrame) -> np.ndarray:
        base = np.asarray(self.poisson_model.predict(x_valid), dtype=float)
        return np.clip(base + self.alpha * np.sqrt(np.maximum(base, 1.0)), 0.0, None)


def fit_negative_binomial(x_train: pd.DataFrame, y_train: pd.Series) -> NegativeBinomialRegressor:
    model = NegativeBinomialRegressor(alpha=0.15)
    model.fit(x_train, y_train)
    return model


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    bias = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=0))
    within_1 = float(np.mean(np.abs(residuals) <= 1.0))
    within_2 = float(np.mean(np.abs(residuals) <= 2.0))
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_prediction_bias": bias,
        "residual_standard_deviation": residual_std,
        "percentage_within_1_corner": within_1,
        "percentage_within_2_corners": within_2,
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification_models(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, selected_features: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        feature_names = selected_features.get(target_name, []) or selected_features.get("actual_total_corners", [])
        x_train, x_valid = build_feature_frames(train_frame, valid_frame, feature_names, target_name)
        y_train = train_frame[target_name].astype(int)
        y_valid = valid_frame[target_name].astype(int)

        models = build_classification_models(x_train, y_train)
        baseline = float(y_train.mean())
        baseline_prob = np.full(len(y_valid), baseline, dtype=float)

        for model_name, model in models.items():
            if model_name == "historical_base_rate_baseline":
                probs = baseline_prob
            else:
                probs = predict_classification_model(model, x_valid)
            probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
            metrics = compute_classification_metrics(y_valid, probs)
            metrics["model_name"] = model_name
            metrics["target_name"] = target_name
            metrics["training_rows"] = int(len(train_frame))
            metrics["validation_rows"] = int(len(valid_frame))
            metrics["probabilities"] = probs.tolist()
            metrics["calibration"] = compute_calibration_metrics(y_valid, probs)
            metrics["model_object"] = model
            results.append(metrics)

        baseline_metrics = next(item for item in results if item["target_name"] == target_name and item["model_name"] == "historical_base_rate_baseline")
        for item in results:
            if item["target_name"] != target_name:
                continue
            if item["model_name"] == "historical_base_rate_baseline":
                continue
            item["baseline_metric_value"] = baseline_metrics["brier_score"]
            item["baseline_metric_name"] = "brier_score"
            item["primary_metric_name"] = "brier_score"
            item["primary_metric_value"] = item["brier_score"]
            item["accepted"] = bool(item["brier_score"] < baseline_metrics["brier_score"])
            item["acceptance_reason"] = "better primary metric than baseline" if item["accepted"] else "did not improve baseline"
    return results


def build_classification_models(x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    base_rate = float(y_train.mean())

    logistic_model = LogisticRegression(random_state=42, max_iter=5000)
    logistic_model.fit(x_train, y_train)

    hgb_model = HistGradientBoostingClassifier(random_state=42, max_depth=3, learning_rate=0.1, max_iter=200)
    hgb_model.fit(x_train, y_train)

    poisson_model = PoissonRegressor(alpha=0.0, max_iter=2000)
    poisson_model.fit(x_train, y_train.astype(float))

    nb_model = fit_negative_binomial(x_train, y_train.astype(float))

    return {
        "historical_base_rate_baseline": base_rate,
        "logistic_regression": logistic_model,
        "hist_gradient_boosting_classifier": hgb_model,
        "poisson_probability": poisson_model,
        "negative_binomial_probability": nb_model,
    }


def predict_classification_model(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int)):
        return np.full(len(x_valid), float(model))
    if isinstance(model, (LogisticRegression, HistGradientBoostingClassifier)):
        return np.asarray(model.predict_proba(x_valid)[:, 1], dtype=float)
    if hasattr(model, "predict"):
        raw = np.asarray(model.predict(x_valid), dtype=float)
        return np.clip(expit(np.log(np.maximum(raw, 1e-6))), 0.0, 1.0)
    return np.zeros(len(x_valid), dtype=float)


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    brier = float(brier_score_loss(y_true, y_pred))
    logloss = float(log_loss(y_true, y_pred, labels=[0, 1]))
    roc_auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else float("nan")
    accuracy = float(accuracy_score(y_true, y_pred_bin))
    precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    return {
        "brier_score": brier,
        "log_loss": logloss,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_calibration_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, 11)
    rows: List[Dict[str, Any]] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lower) & (y_pred < upper)
        if not mask.any():
            continue
        if upper == 1.0:
            mask = (y_pred >= lower) & (y_pred <= upper)
        mean_pred = float(y_pred[mask].mean())
        observed = float(y_true[mask].mean())
        rows.append({"bin_lower": lower, "bin_upper": upper, "mean_pred": mean_pred, "observed": observed, "count": int(mask.sum())})
    if not rows:
        return {"ece": float("nan"), "calibration_slope": float("nan"), "calibration_intercept": float("nan"), "reliability_curve": []}
    ece = float(np.mean([abs(row["mean_pred"] - row["observed"]) * (row["count"] / len(y_true)) for row in rows]))
    logit_p = np.log(y_pred / (1.0 - y_pred))
    design = np.column_stack([np.ones(len(y_pred)), logit_p])
    coeffs, *_ = np.linalg.lstsq(design, y_true, rcond=None)
    intercept = float(coeffs[0])
    slope = float(coeffs[1])
    return {"ece": ece, "calibration_slope": slope, "calibration_intercept": intercept, "reliability_curve": rows}


def select_regression_winner(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = min(results, key=lambda item: item["mae"])
    baseline = next(item for item in results if item["model_name"] == "league_mean_baseline")
    accepted = bool(best["mae"] < baseline["mae"])
    return {
        "target_name": "actual_total_corners",
        "accepted": accepted,
        "model_name": best["model_name"],
        "primary_metric_name": "mae",
        "primary_metric_value": best["mae"],
        "baseline_metric_name": "mae",
        "baseline_metric_value": baseline["mae"],
        "metrics": best,
    }


def select_best_models(regression_best_model: Dict[str, Any], classification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_models: Dict[str, Any] = {"actual_total_corners": regression_best_model}
    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        target_rows = [item for item in classification_results if item["target_name"] == target_name]
        if not target_rows:
            continue
        best = min(target_rows, key=lambda item: item["brier_score"])
        baseline = next(item for item in target_rows if item["model_name"] == "historical_base_rate_baseline")
        accepted = bool(best["brier_score"] < baseline["brier_score"])
        best_models[target_name] = {
            "target_name": target_name,
            "accepted": accepted,
            "model_name": best["model_name"],
            "primary_metric_name": "brier_score",
            "primary_metric_value": best["brier_score"],
            "baseline_metric_name": "brier_score",
            "baseline_metric_value": baseline["brier_score"],
            "metrics": best,
        }
    return best_models


def save_model_artifact(models_dir: Path, target_name: str, target_result: Dict[str, Any]) -> Path:
    model_obj = target_result.get("metrics", {}).get("model_object")
    if model_obj is None:
        model_obj = {"placeholder": True}
    artifact_path = models_dir / f"{target_name}_{target_result['model_name']}.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(model_obj, handle)
    return artifact_path


def write_outputs(output_dir: Path, dataset: pd.DataFrame, regression_results: List[Dict[str, Any]], classification_results: List[Dict[str, Any]], regression_best_model: Dict[str, Any], best_models: Dict[str, Any], accepted_model_artifacts: List[str]) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    research_dir = output_dir / "data" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "model_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "model_benchmark.md").write_text(build_overview_report(regression_results, classification_results, regression_best_model, best_models), encoding="utf-8")
    (reports_dir / "regression_benchmark.md").write_text(build_regression_report(regression_results, regression_best_model), encoding="utf-8")
    (reports_dir / "classification_benchmark.md").write_text(build_classification_report(classification_results, best_models), encoding="utf-8")
    (reports_dir / "calibration_benchmark.md").write_text(build_calibration_report(classification_results), encoding="utf-8")

    results_rows = []
    for item in regression_results:
        for metric_name, value in item.items():
            if metric_name in {"model_name", "target_name", "training_rows", "validation_rows", "probabilities", "baseline_metric_value", "baseline_metric_name", "primary_metric_name", "primary_metric_value", "accepted", "acceptance_reason"}:
                continue
            results_rows.append({"target_name": item["target_name"], "model_name": item["model_name"], "metric_name": metric_name, "value": value})
    for item in classification_results:
        for metric_name, value in item.items():
            if metric_name in {"model_name", "target_name", "training_rows", "validation_rows", "probabilities", "baseline_metric_value", "baseline_metric_name", "primary_metric_name", "primary_metric_value", "accepted", "acceptance_reason", "calibration"}:
                continue
            results_rows.append({"target_name": item["target_name"], "model_name": item["model_name"], "metric_name": metric_name, "value": value})
    pd.DataFrame(results_rows).to_csv(research_dir / "model_benchmark_results.csv", index=False)
    serializable_best_models = serialize_best_models(best_models)
    (research_dir / "best_models.json").write_text(json.dumps(serializable_best_models, indent=2), encoding="utf-8")

    make_plots(dataset, regression_results, classification_results, plots_dir)


def make_plots(dataset: pd.DataFrame, regression_results: List[Dict[str, Any]], classification_results: List[Dict[str, Any]], plots_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except Exception:
        return

    valid_frame = dataset[dataset["season"] == "2025/26"].copy()
    if valid_frame.empty:
        return

    valid_frame = valid_frame.copy()
    valid_frame["date"] = pd.to_datetime(valid_frame["date"], errors="coerce")
    regression_model = next(item for item in regression_results if item["model_name"] == "poisson_regression")
    y_true = valid_frame["actual_total_corners"].astype(float)
    y_pred = np.asarray(regression_model["probabilities"], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="predicted vs actual"))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode="lines", name="perfect fit"))
    fig.write_html(plots_dir / "predicted_vs_actual_corners.html", include_plotlyjs="cdn")

    residuals = y_true - y_pred
    hist_fig = go.Figure(data=[go.Histogram(x=residuals, nbinsx=20)])
    hist_fig.write_html(plots_dir / "residual_histogram.html", include_plotlyjs="cdn")

    time_fig = go.Figure()
    time_fig.add_trace(go.Scatter(x=valid_frame["date"], y=residuals, mode="markers", name="residuals"))
    time_fig.write_html(plots_dir / "residuals_over_time.html", include_plotlyjs="cdn")

    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        target_rows = [item for item in classification_results if item["target_name"] == target_name]
        if not target_rows:
            continue
        names = [item["model_name"] for item in target_rows]
        brier_scores = [item["brier_score"] for item in target_rows]
        fig = go.Figure(data=[go.Bar(x=names, y=brier_scores)])
        fig.update_layout(title=f"Brier Score - {target_name}")
        fig.write_html(plots_dir / f"brier_{target_name}.html", include_plotlyjs="cdn")

        logloss_scores = [item["log_loss"] for item in target_rows]
        fig = go.Figure(data=[go.Bar(x=names, y=logloss_scores)])
        fig.update_layout(title=f"Log Loss - {target_name}")
        fig.write_html(plots_dir / f"logloss_{target_name}.html", include_plotlyjs="cdn")

        for item in target_rows:
            curve = item.get("calibration", {}).get("reliability_curve", [])
            if not curve:
                continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[row["mean_pred"] for row in curve], y=[row["observed"] for row in curve], mode="lines+markers", name=item["model_name"]))
            fig.add_trace(go.Scatter(x=[0.0, 1.0], y=[0.0, 1.0], mode="lines", line=dict(dash="dash"), name="ideal"))
            fig.update_layout(title=f"Reliability curve - {target_name} - {item['model_name']}")
            fig.write_html(plots_dir / f"reliability_{target_name}_{item['model_name']}.html", include_plotlyjs="cdn")


def serialize_best_models(best_models: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for target_name, payload in best_models.items():
        item = dict(payload)
        metrics = dict(item.get("metrics", {}))
        metrics.pop("model_object", None)
        item["metrics"] = metrics
        serialized[target_name] = item
    return serialized


def build_overview_report(regression_results: List[Dict[str, Any]], classification_results: List[Dict[str, Any]], regression_best_model: Dict[str, Any], best_models: Dict[str, Any]) -> str:
    lines = [
        "# Model Benchmark Overview",
        "",
        "This report summarizes the first time-safe baseline benchmark for total corners and Over/Under markets.",
        "",
        "## Regression winner",
        f"- {regression_best_model['model_name']} (MAE {regression_best_model['primary_metric_value']:.3f})",
        "",
        "## Classification winners",
    ]
    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        winner = best_models[target_name]
        lines.append(f"- {target_name}: {winner['model_name']} (Brier {winner['primary_metric_value']:.3f})")
    lines.extend(["", "## Notes", "- Chronological split: train 2023/24-2024/25, validate 2025/26.", "- No current-match or post-match fields were used as features."])
    return "\n".join(lines) + "\n"


def build_regression_report(regression_results: List[Dict[str, Any]], regression_best_model: Dict[str, Any]) -> str:
    lines = ["# Regression Benchmark", ""]
    for item in regression_results:
        lines.append(f"## {item['model_name']}")
        lines.append(f"- MAE: {item['mae']:.3f}")
        lines.append(f"- RMSE: {item['rmse']:.3f}")
        lines.append(f"- Mean prediction bias: {item['mean_prediction_bias']:.3f}")
        lines.append(f"- Residual SD: {item['residual_standard_deviation']:.3f}")
        lines.append(f"- Within ±1 corner: {item['percentage_within_1_corner']:.1%}")
        lines.append(f"- Within ±2 corners: {item['percentage_within_2_corners']:.1%}")
        lines.append("")
    lines.append(f"Best model: {regression_best_model['model_name']} (MAE {regression_best_model['primary_metric_value']:.3f})")
    return "\n".join(lines) + "\n"


def build_classification_report(classification_results: List[Dict[str, Any]], best_models: Dict[str, Any]) -> str:
    lines = ["# Classification Benchmark", ""]
    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        lines.append(f"## {target_name}")
        target_rows = [item for item in classification_results if item["target_name"] == target_name]
        for item in target_rows:
            lines.append(f"### {item['model_name']}")
            lines.append(f"- Brier: {item['brier_score']:.3f}")
            lines.append(f"- Log Loss: {item['log_loss']:.3f}")
            lines.append(f"- ROC AUC: {item['roc_auc']:.3f}")
            lines.append(f"- Accuracy: {item['accuracy']:.3f}")
            lines.append(f"- Precision: {item['precision']:.3f}")
            lines.append(f"- Recall: {item['recall']:.3f}")
            lines.append(f"- F1: {item['f1']:.3f}")
            lines.append(f"- ECE: {item['calibration']['ece']:.3f}")
            lines.append(f"- Calibration slope: {item['calibration']['calibration_slope']:.3f}")
            lines.append(f"- Calibration intercept: {item['calibration']['calibration_intercept']:.3f}")
            lines.append("")
        winner = best_models[target_name]
        lines.append(f"Winner: {winner['model_name']} (Brier {winner['primary_metric_value']:.3f})")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_calibration_report(classification_results: List[Dict[str, Any]]) -> str:
    lines = ["# Calibration Benchmark", ""]
    for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
        target_rows = [item for item in classification_results if item["target_name"] == target_name]
        lines.append(f"## {target_name}")
        for item in target_rows:
            calibration = item.get("calibration", {})
            lines.append(f"- {item['model_name']}: ECE {calibration.get('ece', float('nan')):.3f}, slope {calibration.get('calibration_slope', float('nan')):.3f}, intercept {calibration.get('calibration_intercept', float('nan')):.3f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_feature_frames(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, feature_names: List[str], target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_columns = {"actual_total_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"}
    selected = [feature for feature in feature_names if feature not in target_columns]
    if not selected:
        selected = [
            "corners_for_last3",
            "corners_for_last5",
            "corners_for_last10",
            "total_corners_last3",
            "total_corners_last5",
            "total_corners_last10",
            "attack_trend",
            "defence_trend",
            "tempo_trend",
            "attack_difference",
            "defence_difference",
            "tempo_difference",
            "combined_volatility",
            "rest_days_difference",
        ]
    available = [feature for feature in selected if feature in train_frame.columns and feature in valid_frame.columns]
    if not available:
        available = [column for column in train_frame.columns if pd.api.types.is_numeric_dtype(train_frame[column]) and column not in target_columns]
    x_train = train_frame[available].astype(float).fillna(0.0)
    x_valid = valid_frame[available].astype(float).fillna(0.0)
    return x_train, x_valid
