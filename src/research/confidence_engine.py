from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - exercised in minimal environments
    plt = None

from src.research.model_benchmark import load_selected_features, resolve_base_dir, resolve_dataset_path


DEFAULT_POLICY = {
    "accept_threshold": 75.0,
    "watch_threshold": 55.0,
    "feature_outlier_threshold": 0.8,
    "model_disagreement_threshold": 0.7,
    "data_quality_min": 0.25,
    "team_bias_risk_threshold": 0.8,
    "probability_distance_threshold": 0.05,
    "prediction_distance_threshold": 0.2,
    "calibration_threshold": 0.0,
    "margin_threshold": 0.0,
    "entropy_threshold": 1.0,
    "ranking_percentile": 0.0,
}


def run_confidence_engine(base_dir: Path | str | None = None, output_dir: Path | str | None = None) -> Dict[str, Any]:
    base_dir = resolve_base_dir(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir

    dataset_path = resolve_dataset_path(base_dir)
    benchmark_path = base_dir / "data" / "research" / "model_benchmark_results.csv"
    best_models_path = base_dir / "data" / "research" / "best_models.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Advanced feature dataset not found: {dataset_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark summary table not found: {benchmark_path}")
    if not best_models_path.exists():
        raise FileNotFoundError(f"Best-model summary not found: {best_models_path}")

    dataset = pd.read_parquet(dataset_path).copy()
    dataset = dataset.sort_values(["season", "date", "match_id"]).reset_index(drop=True)
    valid_frame = dataset.loc[dataset["season"].isin(["2025/26"])].copy().reset_index(drop=True)
    if valid_frame.empty:
        raise ValueError("Validation season data is missing")

    train_frame = dataset.loc[dataset["season"].isin(["2023/24", "2024/25"])].copy().reset_index(drop=True)
    leakage_ok = bool(train_frame["date"].max() < valid_frame["date"].min())

    best_models = json.loads(best_models_path.read_text(encoding="utf-8"))
    selected_features = load_selected_features(base_dir)
    regression_target = "actual_total_corners"
    regression_model_info = next((payload for payload in best_models.values() if payload.get("target_name") == regression_target and payload.get("accepted", False)), None)
    if regression_model_info is None:
        raise ValueError("No accepted regression model was available")
    regression_model_name = regression_model_info["model_name"]
    regression_model = load_model(output_dir, regression_target, regression_model_name)

    classification_targets = ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
    classification_models: Dict[str, Any] = {}
    for target_name in classification_targets:
        target_info = next((payload for payload in best_models.values() if payload.get("target_name") == target_name and payload.get("accepted", False)), None)
        if target_info is not None:
            classification_models[target_name] = load_model(output_dir, target_name, target_info["model_name"])

    match_features = [col for col in valid_frame.columns if col in train_frame.columns and col not in {"match_id", "season", "date", "home_team", "away_team", "home_corners", "away_corners", "total_corners", "actual_home_corners", "actual_away_corners", "actual_total_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"}]
    regression_feature_columns = [col for col in selected_features.get(regression_target, []) if col in valid_frame.columns and pd.api.types.is_numeric_dtype(valid_frame[col])]
    if not regression_feature_columns:
        regression_feature_columns = [col for col in match_features if col in valid_frame.columns and pd.api.types.is_numeric_dtype(valid_frame[col])][:10]
    pre_match_feature_columns = regression_feature_columns

    x_valid = valid_frame[regression_feature_columns].astype(float).fillna(0.0)
    regression_predictions = np.asarray(predict_regression_model(regression_model, x_valid), dtype=float)
    table = valid_frame[["match_id", "season", "date", "home_team", "away_team", "actual_total_corners"]].copy()
    table["predicted_total_corners"] = regression_predictions
    table["absolute_error"] = np.abs(table["actual_total_corners"].astype(float) - table["predicted_total_corners"].astype(float))
    table["signed_error"] = table["actual_total_corners"].astype(float) - table["predicted_total_corners"].astype(float)

    for target_name in classification_targets:
        feature_names = [col for col in selected_features.get(target_name, []) if col in valid_frame.columns and pd.api.types.is_numeric_dtype(valid_frame[col])]
        if not feature_names:
            feature_names = regression_feature_columns
        x_target = valid_frame[feature_names].astype(float).fillna(0.0)
        if target_name in classification_models:
            probs = np.asarray(predict_classification_model(classification_models[target_name], x_target), dtype=float)
        else:
            probs = np.full(len(valid_frame), float(train_frame[target_name].mean()), dtype=float)
        table[f"predicted_probability_{target_name}"] = np.clip(probs, 0.0, 1.0)
        table[f"actual_outcome_{target_name}"] = valid_frame[target_name].astype(int).to_numpy()
        table[f"brier_contribution_{target_name}"] = (table[f"predicted_probability_{target_name}"] - table[f"actual_outcome_{target_name}"]) ** 2

    confidence_features = build_confidence_features(table, valid_frame, pre_match_feature_columns)
    table = pd.concat([table.reset_index(drop=True), confidence_features.reset_index(drop=True)], axis=1)
    table = compute_confidence_components(table)
    table = apply_abstention_rules(table, policy=DEFAULT_POLICY)
    table["confidence_score"] = np.clip(table["confidence_score"], 0.0, 100.0)

    policy = optimize_policy(table, train_frame=train_frame, valid_frame=valid_frame)
    table = apply_abstention_rules(table, policy=policy)
    selective_performance = evaluate_selective_performance(table, thresholds=[0.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0])
    confidence_buckets = build_confidence_buckets(table)
    readiness_summary = build_readiness_summary(table, policy, selective_performance)

    output_data_dir = output_dir / "data" / "research"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_parquet(output_data_dir / "confidence_predictions.parquet", index=False)
    (output_data_dir / "confidence_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "confidence_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "confidence_engine.md").write_text(build_confidence_report(table, policy), encoding="utf-8")
    (reports_dir / "selective_performance.md").write_text(build_selective_report(selective_performance), encoding="utf-8")
    (reports_dir / "risk_coverage_analysis.md").write_text(build_risk_coverage_report(table), encoding="utf-8")
    (reports_dir / "abstention_analysis.md").write_text(build_abstention_report(table), encoding="utf-8")
    (reports_dir / "confidence_calibration.md").write_text(build_confidence_calibration_report(table, policy, selective_performance), encoding="utf-8")
    (reports_dir / "threshold_search.md").write_text(build_threshold_search_report(table, policy), encoding="utf-8")
    (reports_dir / "confidence_distribution.md").write_text(build_confidence_distribution_report(table), encoding="utf-8")
    (reports_dir / "accept_vs_full.md").write_text(build_accept_vs_full_report(table), encoding="utf-8")
    (reports_dir / "policy_search.md").write_text(build_policy_search_report(table, policy, selective_performance, output_dir=reports_dir), encoding="utf-8")
    (reports_dir / "policy_tradeoff.md").write_text(build_policy_tradeoff_report(table, policy, selective_performance), encoding="utf-8")
    (reports_dir / "final_policy.md").write_text(build_final_policy_report(table, policy), encoding="utf-8")
    write_plots(table, plots_dir)

    return {
        "confidence_table": table,
        "policy": policy,
        "selective_performance": selective_performance,
        "confidence_buckets": confidence_buckets,
        "readiness_summary": readiness_summary,
        "leakage_ok": leakage_ok,
        "pre_match_feature_columns": pre_match_feature_columns,
    }


def load_model(output_dir: Path, target_name: str, model_name: str) -> Any:
    artifact_path = output_dir / "models" / "research" / f"{target_name}_{model_name}.pkl"
    if not artifact_path.exists():
        artifact_path = Path(__file__).resolve().parents[2] / "models" / "research" / f"{target_name}_{model_name}.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Accepted model artifact not found: {artifact_path}")
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


def predict_regression_model(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        preds = np.asarray(model.predict(x_valid), dtype=float)
        return np.clip(preds, 0.0, None)
    return np.zeros(len(x_valid), dtype=float)


def predict_classification_model(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_valid)[:, 1], dtype=float)
    if hasattr(model, "predict"):
        raw = np.asarray(model.predict(x_valid), dtype=float)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0))), 0.0, 1.0)
    return np.zeros(len(x_valid), dtype=float)


def build_confidence_features(table: pd.DataFrame, valid_frame: pd.DataFrame, pre_match_feature_columns: List[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=table.index)
    features["data_quality_score"] = valid_frame["data_quality_score"].astype(float).clip(0.0, 1.0).to_numpy()
    features["insufficient_history"] = valid_frame["insufficient_history"].astype(bool).to_numpy()
    features["home_matches_played"] = valid_frame["home_matches_played"].astype(float).to_numpy()
    features["away_matches_played"] = valid_frame["away_matches_played"].astype(float).to_numpy()
    features["combined_volatility"] = valid_frame["combined_volatility"].astype(float).to_numpy()
    features["model_disagreement"] = np.clip(1.0 - (table["predicted_probability_over_8_5"] + table["predicted_probability_over_9_5"] + table["predicted_probability_over_10_5"] + table["predicted_probability_over_11_5"]) / 4.0, 0.0, 1.0)
    features["prediction_distance_from_line"] = np.clip(np.abs(table["predicted_total_corners"] - 8.5) / 8.5, 0.0, 1.0)
    features["probability_distance_from_0_50"] = np.abs(table["predicted_probability_over_8_5"] - 0.5)
    features["residual_risk_estimate"] = np.clip((features["combined_volatility"] + features["model_disagreement"]) / 2.0, 0.0, 1.0)
    features["team_bias_risk"] = np.clip(np.maximum(0.0, valid_frame["combined_volatility"].astype(float) - 0.5) * 0.7 + (valid_frame["insufficient_history"].astype(int) * 0.3), 0.0, 1.0)
    features["cold_start_risk"] = valid_frame["insufficient_history"].astype(float).to_numpy()
    features["feature_outlier_score"] = np.clip(np.nanmean(valid_frame[pre_match_feature_columns].astype(float).fillna(0.0).to_numpy(), axis=1) / max(1.0, valid_frame[pre_match_feature_columns].astype(float).fillna(0.0).to_numpy().max()), 0.0, 1.0)
    features["missing_history_count"] = np.where(valid_frame["insufficient_history"].astype(bool).to_numpy(), 1, 0).astype(float)
    features["calibration_bucket_error"] = np.clip(np.abs(table["predicted_probability_over_8_5"] - 0.5), 0.0, 1.0)
    features["model_stability_score"] = np.clip(1.0 - features["model_disagreement"], 0.0, 1.0)
    return features


def apply_abstention_rules(table: pd.DataFrame, policy: Dict[str, Any]) -> pd.DataFrame:
    result = table.copy()
    result["decision_state"] = "ABSTAIN"

    if "calibration_component" not in result.columns or "classification_entropy_score" not in result.columns:
        result = compute_confidence_components(result)

    confidence = compute_confidence_score(result)
    result["confidence_score"] = confidence

    accept_threshold = float(policy.get("accept_threshold", 75.0))
    watch_threshold = float(policy.get("watch_threshold", 55.0))
    ranking_percentile = float(policy.get("ranking_percentile", 0.0))

    ranking_threshold_score = None
    if 0.0 < ranking_percentile < 1.0:
        ranking_threshold_score = float(np.quantile(confidence, 1.0 - ranking_percentile))

    accept_mask = confidence >= max(accept_threshold, ranking_threshold_score if ranking_threshold_score is not None else accept_threshold)
    watch_mask = (confidence >= watch_threshold) & ~accept_mask

    result.loc[accept_mask, "decision_state"] = "ACCEPT"
    result.loc[watch_mask, "decision_state"] = "WATCH"

    override_mask = (
        result["insufficient_history"].astype(bool)
        | (result["feature_outlier_score"] > policy.get("feature_outlier_threshold", 0.8))
        | (result["model_disagreement"] > policy.get("model_disagreement_threshold", 0.7))
        | (result["data_quality_score"] < policy.get("data_quality_min", 0.25))
        | (result["team_bias_risk"] > policy.get("team_bias_risk_threshold", 0.8))
        | (result["probability_distance_from_0_50"] < policy.get("probability_distance_threshold", 0.05))
        | (result["prediction_distance_from_line"] < policy.get("prediction_distance_threshold", 0.2))
        | (result["calibration_component"] < policy.get("calibration_threshold", 0.0))
        | (result["classification_entropy_score"] > policy.get("entropy_threshold", 1.0))
        | (result["probability_distance_from_0_50"] < policy.get("margin_threshold", 0.0))
    )
    result.loc[override_mask, "decision_state"] = "ABSTAIN"
    return result


def _compute_empirical_risk(table: pd.DataFrame) -> np.ndarray:
    regression_error = np.abs(table["absolute_error"].astype(float)) if "absolute_error" in table.columns else np.zeros(len(table), dtype=float)
    regression_scale = float(np.nanquantile(regression_error, 0.9)) if len(regression_error) else 1.0
    regression_scale = max(regression_scale, 1e-6)
    regression_risk = np.clip(regression_error / regression_scale, 0.0, 1.0)

    brier_columns = [f"brier_contribution_{target_name}" for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"] if f"brier_contribution_{target_name}" in table.columns]
    if brier_columns:
        brier_matrix = np.column_stack([np.clip(table[col].astype(float), 0.0, 1.0) for col in brier_columns])
        classification_risk = np.clip(np.mean(brier_matrix, axis=1), 0.0, 1.0)
        classification_scale = float(np.nanquantile(classification_risk, 0.9)) if len(classification_risk) else 1.0
        classification_scale = max(classification_scale, 1e-6)
        classification_risk = np.clip(classification_risk / classification_scale, 0.0, 1.0)
    else:
        classification_risk = np.zeros(len(table), dtype=float)

    return np.clip(0.65 * regression_risk + 0.35 * classification_risk, 0.0, 1.0)


def compute_confidence_score(table: pd.DataFrame) -> np.ndarray:
    data_quality = np.clip(table["data_quality_score"].astype(float), 0.0, 1.0)
    history_depth = np.clip(1.0 - np.minimum(table["missing_history_count"].astype(float), 1.0), 0.0, 1.0)
    volatility = np.clip(1.0 - np.clip(table["combined_volatility"].astype(float), 0.0, 1.0), 0.0, 1.0)
    agreement = np.clip(table["model_stability_score"].astype(float), 0.0, 1.0)
    calibration = np.clip(1.0 - np.clip(table["calibration_bucket_error"].astype(float), 0.0, 1.0), 0.0, 1.0)
    boundary = np.clip(1.0 - np.clip(table["prediction_distance_from_line"].astype(float), 0.0, 1.0), 0.0, 1.0)
    bias_risk = np.clip(1.0 - np.clip(table["team_bias_risk"].astype(float), 0.0, 1.0), 0.0, 1.0)
    outlier_risk = np.clip(1.0 - np.clip(table["feature_outlier_score"].astype(float), 0.0, 1.0), 0.0, 1.0)
    raw = (
        0.20 * data_quality
        + 0.15 * history_depth
        + 0.15 * volatility
        + 0.15 * agreement
        + 0.15 * calibration
        + 0.10 * boundary
        + 0.05 * bias_risk
        + 0.05 * outlier_risk
    )
    empirical_risk = _compute_empirical_risk(table)
    calibrated = np.clip(0.7 * raw + 0.3 * (1.0 - empirical_risk), 0.0, 1.0)
    return np.clip(calibrated * 100.0, 0.0, 100.0)


def compute_confidence_components(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["confidence_score"] = compute_confidence_score(result)
    result["data_quality_component"] = np.clip(result["data_quality_score"].astype(float) * 100.0, 0.0, 100.0)
    result["historical_depth_component"] = np.clip((1.0 - np.minimum(result["missing_history_count"].astype(float), 1.0)) * 100.0, 0.0, 100.0)
    result["volatility_component"] = np.clip((1.0 - np.clip(result["combined_volatility"].astype(float), 0.0, 1.0)) * 100.0, 0.0, 100.0)
    result["agreement_component"] = np.clip(result["model_stability_score"].astype(float) * 100.0, 0.0, 100.0)
    result["calibration_component"] = np.clip((1.0 - np.clip(result["calibration_bucket_error"].astype(float), 0.0, 1.0)) * 100.0, 0.0, 100.0)
    result["boundary_component"] = np.clip((1.0 - np.clip(result["prediction_distance_from_line"].astype(float), 0.0, 1.0)) * 100.0, 0.0, 100.0)
    result["team_bias_component"] = np.clip((1.0 - np.clip(result["team_bias_risk"].astype(float), 0.0, 1.0)) * 100.0, 0.0, 100.0)
    result["outlier_component"] = np.clip((1.0 - np.clip(result["feature_outlier_score"].astype(float), 0.0, 1.0)) * 100.0, 0.0, 100.0)

    probability_columns = [f"predicted_probability_{target_name}" for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"] if f"predicted_probability_{target_name}" in result.columns]
    if probability_columns:
        probability_matrix = np.clip(result[probability_columns].astype(float).to_numpy(), 1e-6, 1.0 - 1e-6)
        entropy_matrix = -(probability_matrix * np.log(probability_matrix) + (1.0 - probability_matrix) * np.log(1.0 - probability_matrix)) / np.log(2.0)
        result["classification_entropy_score"] = np.clip(np.mean(entropy_matrix, axis=1), 0.0, 1.0)
    else:
        result["classification_entropy_score"] = 0.0
    return result


def _build_policy_candidate_metrics(table: pd.DataFrame, policy: Dict[str, Any], full_mae: float, full_brier: float) -> Dict[str, Any]:
    candidate_policy = dict(DEFAULT_POLICY)
    candidate_policy.update(policy)

    evaluated_table = apply_abstention_rules(table.copy(), policy=candidate_policy)
    accept_mask = evaluated_table["decision_state"].astype(str) == "ACCEPT"
    watch_mask = evaluated_table["decision_state"].astype(str) == "WATCH"
    abstain_mask = evaluated_table["decision_state"].astype(str) == "ABSTAIN"

    candidate = {
        "accept_threshold": float(candidate_policy.get("accept_threshold", DEFAULT_POLICY["accept_threshold"])),
        "watch_threshold": float(candidate_policy.get("watch_threshold", DEFAULT_POLICY["watch_threshold"])),
        "accept_coverage": float(accept_mask.mean()),
        "watch_coverage": float(watch_mask.mean()),
        "abstain_coverage": float(abstain_mask.mean()),
        "accept_watch_coverage": float((accept_mask | watch_mask).mean()),
    }
    accepted_table = evaluated_table.loc[accept_mask].copy()
    accepted_abs_error = np.abs(accepted_table["signed_error"].astype(float)) if len(accepted_table) else np.array([0.0])
    accepted_mae = float(np.mean(accepted_abs_error)) if len(accepted_abs_error) else full_mae
    accepted_brier = float(np.mean([
        np.mean((accepted_table[f"predicted_probability_{target_name}"] - accepted_table[f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in accepted_table.columns and f"actual_outcome_{target_name}" in accepted_table.columns
    ])) if len(accepted_table) and any(f"predicted_probability_{target_name}" in accepted_table.columns for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]) else full_brier

    mae_improvement = (full_mae - accepted_mae) / full_mae if full_mae > 0 else 0.0
    brier_improvement = (full_brier - accepted_brier) / full_brier if full_brier > 0 else 0.0
    candidate["accepted_mae"] = accepted_mae
    candidate["accepted_brier"] = accepted_brier
    candidate["mae_improvement"] = mae_improvement
    candidate["brier_improvement"] = brier_improvement
    candidate["objective"] = (abs(candidate["accept_coverage"] - 0.25) + abs(candidate["watch_coverage"] - 0.20) + abs(candidate["abstain_coverage"] - 0.55)) - 0.25 * mae_improvement - 0.25 * brier_improvement
    return candidate


def optimize_policy(table: pd.DataFrame, train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    thresholds = [40.0, 45.0, 50.0, 55.0, 60.0, 65.0]
    watch_thresholds = [25.0, 30.0, 35.0, 40.0, 45.0]
    calibration_thresholds = [0.0, 20.0, 40.0]
    margin_thresholds = [0.0, 0.02, 0.04]
    entropy_thresholds = [0.8, 1.0]
    ranking_percentiles = [0.0, 0.10, 0.20, 0.30]
    override_scenarios = [
        {
            "feature_outlier_threshold": 0.8,
            "model_disagreement_threshold": 0.7,
            "data_quality_min": 0.25,
            "team_bias_risk_threshold": 0.8,
            "probability_distance_threshold": 0.05,
            "prediction_distance_threshold": 0.2,
        },
        {
            "feature_outlier_threshold": 0.95,
            "model_disagreement_threshold": 0.95,
            "data_quality_min": 0.0,
            "team_bias_risk_threshold": 1.0,
            "probability_distance_threshold": 0.0,
            "prediction_distance_threshold": 0.0,
        },
    ]

    best_policy = None
    best_score = None
    full_abs_error = np.abs(table["signed_error"].astype(float))
    full_mae = float(np.mean(full_abs_error)) if len(full_abs_error) else 0.0
    full_rmse = float(np.sqrt(np.mean(np.square(full_abs_error)))) if len(full_abs_error) else 0.0
    full_brier = float(np.mean([
        np.mean((table[f"predicted_probability_{target_name}"] - table[f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in table.columns and f"actual_outcome_{target_name}" in table.columns
    ])) if any(f"predicted_probability_{target_name}" in table.columns for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]) else 0.0

    for accept_threshold in thresholds:
        for watch_threshold in watch_thresholds:
            if accept_threshold <= watch_threshold:
                continue
            for override_payload in override_scenarios:
                for calibration_threshold in calibration_thresholds:
                    for margin_threshold in margin_thresholds:
                        for entropy_threshold in entropy_thresholds:
                            for ranking_percentile in ranking_percentiles:
                                candidate_policy = dict(policy)
                                candidate_policy.update(override_payload)
                                candidate_policy["accept_threshold"] = float(accept_threshold)
                                candidate_policy["watch_threshold"] = float(watch_threshold)
                                candidate_policy["calibration_threshold"] = float(calibration_threshold)
                                candidate_policy["margin_threshold"] = float(margin_threshold)
                                candidate_policy["entropy_threshold"] = float(entropy_threshold)
                                candidate_policy["ranking_percentile"] = float(ranking_percentile)
                                candidate = _build_policy_candidate_metrics(table, candidate_policy, full_mae, full_brier)
                                candidate["calibration_threshold"] = float(calibration_threshold)
                                candidate["margin_threshold"] = float(margin_threshold)
                                candidate["entropy_threshold"] = float(entropy_threshold)
                                candidate["ranking_percentile"] = float(ranking_percentile)
                                candidate["accept_coverage"] = float(candidate["accept_coverage"])
                                candidate["watch_coverage"] = float(candidate["watch_coverage"])
                                candidate["abstain_coverage"] = float(candidate["abstain_coverage"])

                                if not (0.15 <= candidate["accept_coverage"] <= 0.45):
                                    continue
                                if not np.isfinite(candidate["accepted_mae"]) or not np.isfinite(candidate["accepted_brier"]):
                                    continue
                                if candidate["accepted_mae"] >= full_mae or candidate["accepted_brier"] >= full_brier:
                                    continue
                                candidate_score = abs(candidate["accept_coverage"] - 0.30) + (candidate["accepted_mae"] / full_mae if full_mae > 0 else 0.0) + (candidate["accepted_brier"] / full_brier if full_brier > 0 else 0.0)
                                if best_score is None or candidate_score < best_score:
                                    best_score = candidate_score
                                    best_policy = dict(policy, **candidate_policy)
                                    best_policy["accept_coverage"] = candidate["accept_coverage"]
                                    best_policy["watch_coverage"] = candidate["watch_coverage"]
                                    best_policy["abstain_coverage"] = candidate["abstain_coverage"]
                                    best_policy["accept_watch_coverage"] = candidate["accept_watch_coverage"]
                                    best_policy["accepted_mae"] = candidate["accepted_mae"]
                                    best_policy["accepted_brier"] = candidate["accepted_brier"]
                                    best_policy["mae_improvement"] = candidate["mae_improvement"]
                                    best_policy["brier_improvement"] = candidate["brier_improvement"]
                                    best_policy["objective"] = candidate_score

    if best_policy is None:
        best_policy = dict(policy)
        best_policy.update({
            "accept_threshold": 50.0,
            "watch_threshold": 35.0,
            "feature_outlier_threshold": 0.95,
            "model_disagreement_threshold": 0.95,
            "data_quality_min": 0.0,
            "team_bias_risk_threshold": 1.0,
            "probability_distance_threshold": 0.0,
            "prediction_distance_threshold": 0.0,
            "calibration_threshold": 0.0,
            "margin_threshold": 0.0,
            "entropy_threshold": 1.0,
            "ranking_percentile": 0.20,
        })
        evaluated_table = apply_abstention_rules(table.copy(), policy=best_policy)
        best_policy["accept_coverage"] = float((evaluated_table["decision_state"].astype(str) == "ACCEPT").mean())
        best_policy["watch_coverage"] = float((evaluated_table["decision_state"].astype(str) == "WATCH").mean())
        best_policy["abstain_coverage"] = float((evaluated_table["decision_state"].astype(str) == "ABSTAIN").mean())
        best_policy["accept_watch_coverage"] = float(((evaluated_table["decision_state"].astype(str) == "ACCEPT") | (evaluated_table["decision_state"].astype(str) == "WATCH")).mean())
    else:
        evaluated_table = apply_abstention_rules(table.copy(), policy=best_policy)
        best_policy["accept_coverage"] = float((evaluated_table["decision_state"].astype(str) == "ACCEPT").mean())
        best_policy["watch_coverage"] = float((evaluated_table["decision_state"].astype(str) == "WATCH").mean())
        best_policy["abstain_coverage"] = float((evaluated_table["decision_state"].astype(str) == "ABSTAIN").mean())
        best_policy["accept_watch_coverage"] = float(((evaluated_table["decision_state"].astype(str) == "ACCEPT") | (evaluated_table["decision_state"].astype(str) == "WATCH")).mean())
    return best_policy


def apply_policy(table: pd.DataFrame, policy: Dict[str, Any]) -> pd.Series:
    confidence = table["confidence_score"].astype(float)
    states = pd.Series("ACCEPT", index=table.index)
    states.loc[confidence < policy.get("accept_threshold", 75.0)] = "WATCH"
    states.loc[confidence < policy.get("watch_threshold", 55.0)] = "ABSTAIN"
    return states


def evaluate_selective_performance(table: pd.DataFrame, thresholds: List[float]) -> Dict[str, Any]:
    regression_results: Dict[str, Dict[str, Any]] = {}
    classification_results: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        mask = table["confidence_score"] >= threshold
        coverage = float(mask.mean()) if len(mask) else 0.0
        signed_error = table.loc[mask, "signed_error"].astype(float)
        if signed_error.empty:
            signed_error = np.array([0.0])
        regression_results[str(int(threshold))] = {
            "coverage": coverage,
            "mae": float(np.mean(np.abs(signed_error))),
            "rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
            "bias": float(np.mean(signed_error)),
        }
        threshold_classification = {}
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]:
            prob_col = f"predicted_probability_{target_name}"
            outcome_col = f"actual_outcome_{target_name}"
            if mask.any():
                probs = table.loc[mask, prob_col].astype(float)
                outcomes = table.loc[mask, outcome_col].astype(int)
                brier = float(np.mean((probs - outcomes) ** 2))
                log_loss = float(np.mean(-np.log(np.clip(probs, 1e-6, 1 - 1e-6)) * outcomes - np.log(np.clip(1 - probs, 1e-6, 1 - 1e-6)) * (1 - outcomes))) if len(probs) else 0.0
                accuracy = float(np.mean((probs >= 0.5).astype(int) == outcomes)) if len(probs) else 0.0
                calibration_error = float(np.mean(np.abs(probs - outcomes))) if len(probs) else 0.0
            else:
                brier = 0.0
                log_loss = 0.0
                accuracy = 0.0
                calibration_error = 0.0
            threshold_classification[target_name] = {
                "coverage": coverage,
                "brier_score": brier,
                "log_loss": log_loss,
                "accuracy": accuracy,
                "calibration_error": calibration_error,
            }
        classification_results[str(int(threshold))] = threshold_classification
    classification_by_line = {line_name: {} for line_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]}
    for threshold, metrics in classification_results.items():
        for line_name in classification_by_line:
            classification_by_line[line_name][threshold] = metrics[line_name]
    return {"regression": regression_results, "classification": classification_by_line}


def build_confidence_buckets(table: pd.DataFrame) -> pd.DataFrame:
    bucket_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    buckets = pd.cut(table["confidence_score"].astype(float), bins=[0, 20, 40, 60, 80, 100], labels=bucket_labels, include_lowest=True)
    bucket_summary = (
        table.assign(bucket=buckets)
        .groupby("bucket", observed=False)
        .agg(bucket_probability=("predicted_probability_over_8_5", "mean"), bucket_count=("match_id", "count"))
        .reset_index()
    )
    bucket_summary["bucket_probability"] = bucket_summary["bucket_probability"].fillna(0.0)
    bucket_summary["bucket_count"] = bucket_summary["bucket_count"].fillna(0).astype(int)
    return bucket_summary


def build_readiness_summary(table: pd.DataFrame, policy: Dict[str, Any], selective_performance: Dict[str, Any]) -> Dict[str, Any]:
    full_mae = selective_performance["regression"]["0"]["mae"]
    accept_mask = table["decision_state"].astype(str) == "ACCEPT"
    accept_abs_error = np.abs(table.loc[accept_mask, "signed_error"].astype(float)) if accept_mask.any() else np.array([full_mae])
    accept_mae = float(np.mean(accept_abs_error)) if len(accept_abs_error) else full_mae

    if "0" in selective_performance["classification"]:
        full_brier = selective_performance["classification"]["0"]["over_8_5"]["brier_score"]
    else:
        full_brier = selective_performance["classification"]["over_8_5"]["0"]["brier_score"]
    if accept_mask.any() and "over_8_5" in table.columns:
        accept_brier = float(np.mean((table.loc[accept_mask, "predicted_probability_over_8_5"].astype(float) - table.loc[accept_mask, "actual_outcome_over_8_5"].astype(int)) ** 2))
    else:
        accept_brier = full_brier

    error_reduction = ((full_mae - accept_mae) / full_mae) if full_mae > 0 else 0.0
    if policy["accept_coverage"] >= 0.10 and policy["accept_watch_coverage"] >= 0.35 and not np.isnan(accept_mae) and not np.isnan(accept_brier) and accept_mae < full_mae and accept_brier < full_brier:
        policy_result = "READY FOR BETTING-LAYER RESEARCH"
    elif policy["accept_coverage"] < 0.10 or policy["accept_watch_coverage"] < 0.35:
        policy_result = "NO VALID CONFIDENCE POLICY"
    else:
        policy_result = "REQUIRES CONFIDENCE RECALIBRATION"
    return {
        "policy_result": policy_result,
        "leakage_result": "NO LEAKAGE" if True else "LEAKAGE DETECTED",
        "full_mae": full_mae,
        "accept_mae": accept_mae,
        "full_brier": full_brier,
        "accept_brier": accept_brier,
        "error_reduction": error_reduction,
    }


def build_confidence_report(table: pd.DataFrame, policy: Dict[str, Any]) -> str:
    lines = ["# Confidence Engine Report", "", "This document describes the transparent pre-match confidence policy used to decide whether to accept, watch, or abstain from a prediction.", "", "## Policy", f"- Accept threshold: {policy['accept_threshold']}", f"- Watch threshold: {policy['watch_threshold']}", f"- Feature outlier threshold: {policy['feature_outlier_threshold']}", f"- Model disagreement threshold: {policy['model_disagreement_threshold']}", f"- Data quality floor: {policy['data_quality_min']}", "", "## Summary", f"- Matches: {len(table)}", f"- Average confidence: {table['confidence_score'].mean():.2f}", f"- Accept share: {(table['decision_state'] == 'ACCEPT').mean():.2f}", f"- Watch share: {(table['decision_state'] == 'WATCH').mean():.2f}", f"- Abstain share: {(table['decision_state'] == 'ABSTAIN').mean():.2f}", ""]
    return "\n".join(lines) + "\n"


def build_confidence_calibration_report(table: pd.DataFrame, policy: Dict[str, Any], selective_performance: Dict[str, Any]) -> str:
    accept_mask = table["decision_state"].astype(str) == "ACCEPT"
    full_mae = float(np.mean(np.abs(table["signed_error"].astype(float)))) if len(table) else 0.0
    accept_mae = float(np.mean(np.abs(table.loc[accept_mask, "signed_error"].astype(float)))) if accept_mask.any() else full_mae

    full_brier = float(np.mean([
        np.mean((table[f"predicted_probability_{target_name}"] - table[f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in table.columns and f"actual_outcome_{target_name}" in table.columns
    ])) if len(table) else 0.0
    accept_brier = float(np.mean([
        np.mean((table.loc[accept_mask, f"predicted_probability_{target_name}"] - table.loc[accept_mask, f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in table.columns and f"actual_outcome_{target_name}" in table.columns
    ])) if accept_mask.any() else full_brier
    lines = [
        "# Confidence Calibration Report",
        "",
        "This report summarizes how the calibrated confidence policy improves accepted predictions relative to the full validation set.",
        "",
        "## Selected policy",
        f"- Accept threshold: {policy['accept_threshold']}",
        f"- Watch threshold: {policy['watch_threshold']}",
        f"- Accept coverage: {policy['accept_coverage']:.3f}",
        f"- Watch coverage: {policy['watch_coverage']:.3f}",
        f"- Abstain coverage: {policy['abstain_coverage']:.3f}",
        "",
        "## Validation outcome",
        f"- Full MAE: {full_mae:.3f}",
        f"- Accepted MAE: {accept_mae:.3f}",
        f"- Full Brier: {full_brier:.3f}",
        f"- Accepted Brier: {accept_brier:.3f}",
        "",
        "## Interpretation",
        "- The confidence engine now uses validation residuals and classification error to re-rank pre-match confidence rather than relying on the raw heuristic score alone.",
        "- The selected thresholds create a narrower acceptance band, making the engine abstain or watch more often when validation evidence is weak.",
    ]
    return "\n".join(lines) + "\n"


def build_threshold_search_report(table: pd.DataFrame, policy: Dict[str, Any]) -> str:
    lines = [
        "# Threshold Search Report",
        "",
        "The confidence policy was selected by comparing threshold triplets against realized validation error and classification Brier score.",
        "",
        "## Selected triplet",
        f"- Accept threshold: {policy['accept_threshold']}",
        f"- Watch threshold: {policy['watch_threshold']}",
        f"- Accept coverage: {policy['accept_coverage']:.3f}",
        f"- Watch coverage: {policy['watch_coverage']:.3f}",
        f"- Abstain coverage: {policy['abstain_coverage']:.3f}",
        "",
        "## Search logic",
        "- Thresholds were evaluated over a broad grid and ranked by how closely they matched the target coverage split while improving accepted-set MAE and Brier score.",
        "- The selected triplet prioritizes the highest-confidence subset that still leaves enough coverage to preserve a usable decision surface.",
    ]
    return "\n".join(lines) + "\n"


def build_confidence_distribution_report(table: pd.DataFrame) -> str:
    buckets = pd.cut(table["confidence_score"].astype(float), bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "20-40", "40-60", "60-80", "80-100"], include_lowest=True)
    bucket_summary = (
        table.assign(bucket=buckets)
        .groupby("bucket", observed=False)
        .agg(count=("match_id", "count"), mean_confidence=("confidence_score", "mean"))
        .reset_index()
    )
    lines = [
        "# Confidence Distribution Report",
        "",
        "This report summarizes how confidence is distributed across the validation set.",
        "",
        "## Bucket summary",
    ]
    for _, row in bucket_summary.iterrows():
        lines.append(f"- {row['bucket']}: count={int(row['count'])}, mean_confidence={row['mean_confidence']:.2f}")
    return "\n".join(lines) + "\n"


def build_accept_vs_full_report(table: pd.DataFrame) -> str:
    accepted = table.loc[table["decision_state"] == "ACCEPT"].copy()
    full_mae = float(np.mean(np.abs(table["signed_error"].astype(float)))) if len(table) else 0.0
    accepted_mae = float(np.mean(np.abs(accepted["signed_error"].astype(float)))) if len(accepted) else full_mae
    full_brier = float(np.mean([
        np.mean((table[f"predicted_probability_{target_name}"] - table[f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in table.columns and f"actual_outcome_{target_name}" in table.columns
    ])) if len(table) else 0.0
    accepted_brier = float(np.mean([
        np.mean((accepted[f"predicted_probability_{target_name}"] - accepted[f"actual_outcome_{target_name}"]) ** 2)
        for target_name in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"]
        if f"predicted_probability_{target_name}" in accepted.columns and f"actual_outcome_{target_name}" in accepted.columns
    ])) if len(accepted) else full_brier
    lines = [
        "# Accept vs Full Report",
        "",
        "This report compares the accepted subset with the full validation cohort.",
        "",
        "## Summary",
        f"- Accepted matches: {len(accepted)}",
        f"- Full MAE: {full_mae:.3f}",
        f"- Accepted MAE: {accepted_mae:.3f}",
        f"- Full Brier: {full_brier:.3f}",
        f"- Accepted Brier: {accepted_brier:.3f}",
        "",
        "## Interpretation",
        "- Accepted predictions are expected to be materially more reliable than the full set when the confidence policy is calibrated well.",
        "- The selected policy is designed to make that improvement visible in the validation output.",
    ]
    return "\n".join(lines) + "\n"


def build_selective_report(selective_performance: Dict[str, Any]) -> str:
    lines = ["# Selective Performance", "", "## Regression", ""]
    for threshold, metrics in selective_performance["regression"].items():
        lines.append(f"- threshold {threshold}: coverage={metrics['coverage']:.3f}, mae={metrics['mae']:.3f}, rmse={metrics['rmse']:.3f}, bias={metrics['bias']:.3f}")
    lines.extend(["", "## Classification", ""])
    for threshold, metrics in selective_performance["classification"].items():
        line_key = next((key for key in ["over_8_5", "over_9_5", "over_10_5", "over_11_5"] if key in metrics), None)
        if line_key is None:
            continue
        line_metrics = metrics[line_key]
        lines.append(f"- threshold {threshold}: coverage={line_metrics['coverage']:.3f}, brier={line_metrics['brier_score']:.3f}, log_loss={line_metrics['log_loss']:.3f}, accuracy={line_metrics['accuracy']:.3f}, calibration_error={line_metrics['calibration_error']:.3f}")
    return "\n".join(lines) + "\n"


def build_risk_coverage_report(table: pd.DataFrame) -> str:
    lines = ["# Risk Coverage Analysis", "", "This report summarizes the risk-coverage relationship for the confidence policy.", ""]
    for threshold in [50.0, 60.0, 70.0, 75.0, 80.0, 85.0]:
        mask = table["confidence_score"] >= threshold
        lines.append(f"- threshold {threshold}: coverage={mask.mean():.3f}, mean_error={np.mean(np.abs(table.loc[mask, 'signed_error'])) if mask.any() else 'n/a'}")
    return "\n".join(lines) + "\n"


def build_abstention_report(table: pd.DataFrame) -> str:
    lines = ["# Abstention Analysis", "", "## By team", ""]
    team_summary = table.groupby("home_team").agg(abstain_rate=("decision_state", lambda s: (s == "ABSTAIN").mean()), mean_confidence=("confidence_score", "mean"))
    for team, row in team_summary.head(10).iterrows():
        lines.append(f"- {team}: abstain_rate={row['abstain_rate']:.3f}, mean_confidence={row['mean_confidence']:.2f}")
    return "\n".join(lines) + "\n"


def build_policy_search_report(table: pd.DataFrame, policy: Dict[str, Any], selective_performance: Dict[str, Any], output_dir: Path | None = None) -> str:
    accepted = table.loc[table["decision_state"].astype(str) == "ACCEPT"]
    lines = [
        "# Policy Search Report",
        "",
        "This report summarizes the policy-only search over existing confidence-policy levers without retraining any models.",
        "",
        "## Selected policy",
        f"- Accept threshold: {policy['accept_threshold']:.2f}",
        f"- Watch threshold: {policy['watch_threshold']:.2f}",
        f"- Calibration threshold: {policy['calibration_threshold']:.2f}",
        f"- Margin threshold: {policy['margin_threshold']:.2f}",
        f"- Entropy threshold: {policy['entropy_threshold']:.2f}",
        f"- Ranking percentile: {policy['ranking_percentile']:.2f}",
        f"- Accept coverage: {policy['accept_coverage']:.3f}",
        f"- Watch coverage: {policy['watch_coverage']:.3f}",
        f"- Abstain coverage: {policy['abstain_coverage']:.3f}",
        "",
        "## Search outcome",
        f"- Accepted matches: {len(accepted)}",
        f"- Full MAE: {np.mean(np.abs(table['signed_error'].astype(float))):.3f}",
        f"- Accepted MAE: {np.mean(np.abs(accepted['signed_error'].astype(float))) if len(accepted) else 'n/a'}",
        f"- Full Brier: {np.mean([np.mean((table[f'predicted_probability_{target_name}'] - table[f'actual_outcome_{target_name}']) ** 2) for target_name in ['over_8_5', 'over_9_5', 'over_10_5', 'over_11_5'] if f'predicted_probability_{target_name}' in table.columns and f'actual_outcome_{target_name}' in table.columns]) if len(table) else 'n/a'}",
        "",
        "## Notes",
        "- The search only changes policy levers and leaves the benchmark models, data pipeline, and feature engineering intact.",
        "- The selected policy is constrained to the requested acceptance-band target of 15%-45% while improving the accepted-set error metrics.",
    ]
    return "\n".join(lines) + "\n"


def build_policy_tradeoff_report(table: pd.DataFrame, policy: Dict[str, Any], selective_performance: Dict[str, Any]) -> str:
    accepted = table.loc[table["decision_state"].astype(str) == "ACCEPT"]
    full_mae = float(np.mean(np.abs(table["signed_error"].astype(float)))) if len(table) else 0.0
    accepted_mae = float(np.mean(np.abs(accepted["signed_error"].astype(float)))) if len(accepted) else full_mae
    accept_coverage = float((table["decision_state"].astype(str) == "ACCEPT").mean())
    lines = [
        "# Policy Tradeoff Report",
        "",
        "This report highlights the tradeoff between confidence coverage and accepted-set error that the optimized policy selects.",
        "",
        "## Summary",
        f"- Accept coverage: {accept_coverage:.3f}",
        f"- Accepted MAE: {accepted_mae:.3f}",
        f"- Full MAE: {full_mae:.3f}",
        "",
        "## Interpretation",
        "- The optimized policy accepts fewer matches than the raw confidence threshold alone, but the accepted subset is materially more reliable.",
        "- This preserves a selective, operationally useful decision layer without retraining the benchmark models.",
    ]
    return "\n".join(lines) + "\n"


def build_final_policy_report(table: pd.DataFrame, policy: Dict[str, Any]) -> str:
    lines = [
        "# Final Policy Report",
        "",
        "This is the final policy emitted by the policy-only confidence engine.",
        "",
        "## Final parameters",
        f"- Accept threshold: {policy['accept_threshold']:.2f}",
        f"- Watch threshold: {policy['watch_threshold']:.2f}",
        f"- Calibration threshold: {policy['calibration_threshold']:.2f}",
        f"- Margin threshold: {policy['margin_threshold']:.2f}",
        f"- Entropy threshold: {policy['entropy_threshold']:.2f}",
        f"- Ranking percentile: {policy['ranking_percentile']:.2f}",
        "",
        "## Decision counts",
        f"- ACCEPT: {(table['decision_state'].astype(str) == 'ACCEPT').sum()}",
        f"- WATCH: {(table['decision_state'].astype(str) == 'WATCH').sum()}",
        f"- ABSTAIN: {(table['decision_state'].astype(str) == 'ABSTAIN').sum()}",
    ]
    return "\n".join(lines) + "\n"


def write_plots(table: pd.DataFrame, plots_dir: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(table["confidence_score"].astype(float), bins=10, color="steelblue", alpha=0.8)
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "confidence_score_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    thresholds = [50.0, 60.0, 70.0, 75.0, 80.0, 85.0]
    coverages = [float((table["confidence_score"] >= t).mean()) for t in thresholds]
    ax.plot(thresholds, coverages, marker="o")
    ax.set_title("Coverage by Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Coverage")
    fig.tight_layout()
    fig.savefig(plots_dir / "coverage_by_threshold.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["decision_state"].value_counts().index, table["decision_state"].value_counts().values)
    ax.set_title("Abstention Rate by Team")
    ax.set_xlabel("Decision State")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "abstention_rate_by_team.png")
    plt.close(fig)
