from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


TARGETS = {
    "actual_total_corners": "regression",
    "over_8_5": "classification",
    "over_9_5": "classification",
    "over_10_5": "classification",
    "over_11_5": "classification",
}

EXCLUDED_COLUMNS = {
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "actual_home_corners",
    "actual_away_corners",
    "actual_total_corners",
    "over_8_5",
    "over_9_5",
    "over_10_5",
    "over_11_5",
    "home_corners",
    "away_corners",
    "total_corners",
}


def run_feature_selection(base_dir: Path | str | None = None, output_dir: Path | str | None = None) -> Dict[str, Dict[str, Any]]:
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

    feature_candidates = [
        column
        for column in dataset.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(dataset[column])
    ]

    feature_metrics: Dict[str, List[Dict[str, Any]]] = {}
    for target_name, target_type in TARGETS.items():
        metrics = compute_feature_metrics(dataset, train_frame, valid_frame, feature_candidates, target_name, target_type)
        feature_metrics[target_name] = metrics

    results: Dict[str, Dict[str, Any]] = {}
    for target_name, target_type in TARGETS.items():
        metrics = feature_metrics[target_name]
        selection = select_features(train_frame, valid_frame, metrics, target_name, target_type)
        results[target_name] = selection

    write_outputs(results, dataset, base_dir, output_dir)
    return results


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


def compute_feature_metrics(dataset: pd.DataFrame, train_frame: pd.DataFrame, valid_frame: pd.DataFrame, feature_candidates: List[str], target_name: str, target_type: str) -> List[Dict[str, Any]]:
    train_target = train_frame[target_name].astype(float)
    valid_target = valid_frame[target_name].astype(float)

    metrics: List[Dict[str, Any]] = []
    for feature in feature_candidates:
        full_series = dataset[feature].astype(float)
        train_series = train_frame[feature].astype(float)
        valid_series = valid_frame[feature].astype(float)

        missing_rate = float(full_series.isna().mean())
        variance = float(train_series.var(ddof=0))
        zero_variance = variance <= 1e-12 or train_series.nunique() <= 1
        near_zero_variance = variance <= 1e-3

        train_corr = pearson_correlation(train_series, train_target)
        valid_corr = pearson_correlation(valid_series, valid_target)
        train_spearman = spearman_correlation(train_series, train_target)
        valid_spearman = spearman_correlation(valid_series, valid_target)

        train_mi = compute_mutual_information(train_series, train_target, target_type)
        valid_mi = compute_mutual_information(valid_series, valid_target, target_type)

        metrics.append(
            {
                "feature": feature,
                "missing_rate": missing_rate,
                "variance": variance,
                "train_corr": train_corr,
                "validation_corr": valid_corr,
                "train_spearman": train_spearman,
                "validation_spearman": valid_spearman,
                "train_mi": train_mi,
                "validation_mi": valid_mi,
                "zero_variance": zero_variance,
                "near_zero_variance": near_zero_variance,
                "target": target_name,
                "target_type": target_type,
            }
        )

    return metrics


def select_features(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, metrics: List[Dict[str, Any]], target_name: str, target_type: str) -> Dict[str, Any]:
    candidate_metrics = [metric for metric in metrics if not metric["zero_variance"] and not metric["near_zero_variance"]]
    if not candidate_metrics:
        return build_empty_selection(target_name, target_type)

    train_features = train_frame[[metric["feature"] for metric in candidate_metrics]].astype(float)
    train_target = train_frame[target_name].astype(float)
    valid_target = valid_frame[target_name].astype(float)

    feature_rank_map = compute_rank_stability(candidate_metrics, train_frame, valid_frame, target_name)
    for metric in candidate_metrics:
        metric["rank_stability"] = feature_rank_map[metric["feature"]]
        metric["sign_stability"] = 1.0 if metric["train_corr"] * metric["validation_corr"] >= 0 else 0.0
        metric["stability_score"] = 0.5 * metric["sign_stability"] + 0.5 * metric["rank_stability"]
        metric["variance_quality"] = min(1.0, np.sqrt(max(metric["variance"], 0.0)) / (np.sqrt(max(metric["variance"], 0.0)) + 1.0))
        metric["missing_quality"] = max(0.0, 1.0 - metric["missing_rate"])
        metric["signal_score"] = compute_signal_score(metric)

    candidate_metrics.sort(key=lambda item: item["signal_score"], reverse=True)

    selected_features: List[Dict[str, Any]] = []
    excluded_features: List[Dict[str, Any]] = []
    selected_names: List[str] = []
    for metric in candidate_metrics:
        feature = metric["feature"]
        if feature in selected_names:
            continue
        if selected_names:
            corr_with_selected = train_features[[feature] + selected_names].corr().iloc[0, 1:]
            if (corr_with_selected.abs() > 0.95).any():
                excluded_features.append(build_exclusion_entry(metric, "highly correlated with a selected feature"))
                continue
        selected_names.append(feature)
        selected_features.append(build_selection_entry(metric))

    for metric in candidate_metrics:
        feature = metric["feature"]
        if feature not in selected_names:
            excluded_features.append(build_exclusion_entry(metric, "low signal or redundant with another selected feature"))

    for metric in metrics:
        if metric["feature"] in {item["feature"] for item in selected_features}:
            continue
        if metric["zero_variance"] or metric["near_zero_variance"]:
            excluded_features.append(build_exclusion_entry(metric, "zero variance" if metric["zero_variance"] else "near-zero variance"))
        elif metric["feature"] not in {item["feature"] for item in excluded_features}:
            excluded_features.append(build_exclusion_entry(metric, "excluded during redundancy control"))

    selected_features.sort(key=lambda item: item["signal_score"], reverse=True)
    excluded_features.sort(key=lambda item: item["signal_score"], reverse=True)

    return {
        "target": target_name,
        "target_type": target_type,
        "train_seasons": ["2023/24", "2024/25"],
        "validation_seasons": ["2025/26"],
        "selected_features": selected_features,
        "excluded_features": excluded_features,
        "exclusion_reason": "Features were removed for zero variance, near-zero variance, or high collinearity with stronger signal features.",
        "selected_feature_count": len(selected_features),
        "train_metric": float(np.nanmean([item["train_metric"] for item in selected_features])) if selected_features else 0.0,
        "validation_metric": float(np.nanmean([item["validation_metric"] for item in selected_features])) if selected_features else 0.0,
        "signal_score": float(np.nanmean([item["signal_score"] for item in selected_features])) if selected_features else 0.0,
    }


def build_empty_selection(target_name: str, target_type: str) -> Dict[str, Any]:
    return {
        "target": target_name,
        "target_type": target_type,
        "train_seasons": ["2023/24", "2024/25"],
        "validation_seasons": ["2025/26"],
        "selected_features": [],
        "excluded_features": [],
        "exclusion_reason": "No stable features were available",
        "selected_feature_count": 0,
        "train_metric": 0.0,
        "validation_metric": 0.0,
        "signal_score": 0.0,
    }


def build_selection_entry(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature": metric["feature"],
        "signal_score": round(float(metric["signal_score"]), 3),
        "train_metric": round(float(metric["train_corr"]), 3),
        "validation_metric": round(float(metric["validation_corr"]), 3),
        "train_mi": round(float(metric["train_mi"]), 3),
        "validation_mi": round(float(metric["validation_mi"]), 3),
        "missing_rate": round(float(metric["missing_rate"]), 3),
        "variance": round(float(metric["variance"]), 3),
    }


def build_exclusion_entry(metric: Dict[str, Any], reason: str) -> Dict[str, Any]:
    entry = build_selection_entry(metric)
    entry["exclusion_reason"] = reason
    return entry


def compute_signal_score(metric: Dict[str, Any]) -> float:
    abs_validation_corr = abs(float(metric["validation_corr"]))
    mi_component = min(1.0, float(metric["validation_mi"]) / (float(metric["validation_mi"]) + 1.0))
    stability_component = float(metric["stability_score"])
    variance_component = float(metric["variance_quality"])
    missing_component = float(metric["missing_quality"])
    score = 100.0 * (
        0.35 * abs_validation_corr
        + 0.25 * mi_component
        + 0.20 * stability_component
        + 0.10 * variance_component
        + 0.10 * missing_component
    )
    return round(float(np.clip(score, 0.0, 100.0)), 3)


def compute_rank_stability(candidate_metrics: List[Dict[str, Any]], train_frame: pd.DataFrame, valid_frame: pd.DataFrame, target_name: str) -> Dict[str, float]:
    train_ranks = {item["feature"]: rank for rank, item in enumerate(sorted(candidate_metrics, key=lambda item: abs(item["train_corr"]), reverse=True), start=1)}
    valid_ranks = {item["feature"]: rank for rank, item in enumerate(sorted(candidate_metrics, key=lambda item: abs(item["validation_corr"]), reverse=True), start=1)}
    max_rank = max(len(candidate_metrics), 1)
    return {feature: max(0.0, 1.0 - abs(train_ranks[feature] - valid_ranks[feature]) / max_rank) for feature in train_ranks}


def pearson_correlation(feature: pd.Series, target: pd.Series) -> float:
    if feature.std(ddof=0) == 0 or target.std(ddof=0) == 0:
        return 0.0
    return float(feature.corr(target))


def spearman_correlation(feature: pd.Series, target: pd.Series) -> float:
    if feature.nunique() <= 1 or target.nunique() <= 1:
        return 0.0
    return float(feature.corr(target, method="spearman"))


def compute_mutual_information(feature: pd.Series, target: pd.Series, target_type: str) -> float:
    feature_values = feature.astype(float).to_numpy().reshape(-1, 1)
    target_values = target.astype(float).to_numpy()
    if target_type == "regression":
        return float(mutual_info_regression(feature_values, target_values, random_state=42)[0])
    return float(mutual_info_classif(feature_values, target_values, random_state=42)[0])


def write_outputs(results: Dict[str, Dict[str, Any]], dataset: pd.DataFrame, base_dir: Path, output_dir: Path) -> None:
    research_dir = output_dir / "data" / "research"
    reports_dir = output_dir / "reports"
    research_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for target_name in TARGETS:
        output_path = research_dir / f"selected_features_{target_name.split('_')[-1]}.json"
        if target_name == "actual_total_corners":
            output_path = research_dir / "selected_features_regression.json"
        elif target_name == "over_8_5":
            output_path = research_dir / "selected_features_over85.json"
        elif target_name == "over_9_5":
            output_path = research_dir / "selected_features_over95.json"
        elif target_name == "over_10_5":
            output_path = research_dir / "selected_features_over105.json"
        elif target_name == "over_11_5":
            output_path = research_dir / "selected_features_over115.json"
        output_path.write_text(json.dumps(results[target_name], indent=2), encoding="utf-8")

    report_path = reports_dir / "feature_selection_report.md"
    report_path.write_text(build_feature_selection_report(results, dataset), encoding="utf-8")

    stability_path = reports_dir / "feature_stability_report.md"
    stability_path.write_text(build_stability_report(results), encoding="utf-8")

    collinearity_path = reports_dir / "feature_collinearity_report.md"
    collinearity_path.write_text(build_collinearity_report(results, dataset), encoding="utf-8")


def build_feature_selection_report(results: Dict[str, Dict[str, Any]], dataset: pd.DataFrame) -> str:
    regression = results["actual_total_corners"]
    top_features = regression["selected_features"][:20]
    lines = [
        "# Feature Selection Report",
        "",
        "## Target summaries",
    ]
    for target_name in TARGETS:
        item = results[target_name]
        lines.append(f"- {target_name}: {item['selected_feature_count']} selected features, signal score {item['signal_score']:.3f}")
    lines.extend([
        "",
        "## Top 20 features by Signal Score",
        "",
    ])
    for feature in top_features:
        lines.append(f"- {feature['feature']}: signal score {feature['signal_score']:.3f}, train corr {feature['train_metric']:.3f}, validation corr {feature['validation_metric']:.3f}")
    lines.extend([
        "",
        "## Recommended feature set for regression",
        ", ".join(feature["feature"] for feature in regression["selected_features"]),
        "",
        "## Explicitly rejected features",
    ])
    for feature in regression["excluded_features"]:
        lines.append(f"- {feature['feature']}: {feature['exclusion_reason']}")
    return "\n".join(lines) + "\n"


def build_stability_report(results: Dict[str, Dict[str, Any]]) -> str:
    unstable_features = []
    for target_name in TARGETS:
        for feature in results[target_name]["selected_features"]:
            unstable_features.append((feature["feature"], feature["train_metric"], feature["validation_metric"], target_name))
    unstable_features = [item for item in unstable_features if item[1] * item[2] < 0]
    lines = [
        "# Feature Stability Report",
        "",
        "## Unstable features",
    ]
    for feature, train_metric, validation_metric, target_name in unstable_features:
        lines.append(f"- {feature} ({target_name}): train {train_metric:.3f}, validation {validation_metric:.3f}")
    lines.extend(["", "## Sign reversals"])
    if not unstable_features:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def build_collinearity_report(results: Dict[str, Dict[str, Any]], dataset: pd.DataFrame) -> str:
    selected_features = list(dict.fromkeys(feature["feature"] for target_name in TARGETS for feature in results[target_name]["selected_features"]))
    feature_frame = dataset[selected_features].astype(float)
    feature_frame = feature_frame.dropna(axis=1)
    corr = feature_frame.corr().abs()
    pairs = []
    for left in corr.columns:
        for right in corr.columns:
            if left >= right:
                continue
            value = corr.loc[left, right]
            if isinstance(value, pd.Series):
                continue
            value = float(value)
            if value > 0.90:
                pairs.append((left, right, value))
    lines = [
        "# Feature Collinearity Report",
        "",
        "## Highly collinear pairs",
    ]
    if not pairs:
        lines.append("- None")
    else:
        for left, right, value in sorted(pairs, key=lambda item: abs(item[2]), reverse=True)[:20]:
            lines.append(f"- {left} / {right}: {value:.3f}")
    return "\n".join(lines) + "\n"
