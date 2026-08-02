from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


RESEARCH_DATASET_PATH = Path("data/research/research_dataset.parquet")


def _format_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def resolve_dataset_path(base_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / RESEARCH_DATASET_PATH,
        repo_root / RESEARCH_DATASET_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def generate_validation_reports(base_dir: Path | None = None) -> List[Path]:
    base_dir = base_dir or Path.cwd()
    reports_dir = base_dir / "reports"
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(base_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Research dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise ValueError("Research dataset is empty")

    output_paths: List[Path] = []

    validation_report = build_validation_report(df, reports_dir)
    validation_path = reports_dir / "validation_report.md"
    validation_path.write_text(validation_report, encoding="utf-8")
    output_paths.append(validation_path)

    poisson_report = build_poisson_report(df, plots_dir)
    poisson_path = reports_dir / "poisson_validation.md"
    poisson_path.write_text(poisson_report, encoding="utf-8")
    output_paths.append(poisson_path)

    rating_report = build_rating_report(df, plots_dir)
    rating_path = reports_dir / "rating_analysis.md"
    rating_path.write_text(rating_report, encoding="utf-8")
    output_paths.append(rating_path)

    corr_report = build_feature_correlation_report(df, plots_dir)
    corr_path = reports_dir / "feature_correlation.md"
    corr_path.write_text(corr_report, encoding="utf-8")
    output_paths.append(corr_path)

    season_summary = build_season_summary(df, plots_dir)
    season_path = reports_dir / "season_summary.md"
    season_path.write_text(season_summary, encoding="utf-8")
    output_paths.append(season_path)

    return output_paths


def build_validation_report(df: pd.DataFrame, reports_dir: Path) -> str:
    total = df["total_corners"].astype(float)
    season_stats = df.groupby("season")["total_corners"].agg(["mean", "std", "median"]).reset_index()
    home_stats = df.groupby("season")["home_corners"].agg(["mean", "std"]).reset_index()
    away_stats = df.groupby("season")["away_corners"].agg(["mean", "std"]).reset_index()

    percentiles = np.percentile(total, [0, 10, 25, 50, 75, 90, 95, 99])
    percentile_table = "\n".join([f"- {p:.0f}%: {v:.2f}" for p, v in zip([0, 10, 25, 50, 75, 90, 95, 99], percentiles)])

    lines = [
        "# Validation Report",
        "",
        "## Distribution overview",
        f"- Total matches: {len(df)}",
        f"- Mean total corners: {total.mean():.2f}",
        f"- Median total corners: {total.median():.2f}",
        f"- Std total corners: {total.std():.2f}",
        "",
        "### Percentiles",
        percentile_table,
        "",
        "## Season distributions",
        _format_table(season_stats),
        "",
        "## Home vs away averages",
        _format_table(pd.concat([home_stats.add_prefix("home_"), away_stats.add_prefix("away_")], axis=1)),
        "",
    ]
    return "\n".join(lines)


def build_poisson_report(df: pd.DataFrame, plots_dir: Path) -> str:
    total = df["total_corners"].astype(float)
    mean_total = float(total.mean())
    poisson_dist = stats.poisson(mean_total)
    observed = total.value_counts().sort_index().reindex(range(int(total.max()) + 1), fill_value=0)
    expected = pd.Series([poisson_dist.pmf(i) * len(df) for i in observed.index], index=observed.index)

    chi_sq = ((observed - expected) ** 2 / expected).sum()
    ks_statistic = stats.kstest(total, lambda x: poisson_dist.cdf(x)).statistic

    residuals = observed - expected
    residual_df = pd.DataFrame({"value": observed.index, "observed": observed.values, "expected": expected.values, "residual": residuals.values})

    qq_path = plots_dir / "poisson_qq.html"
    residual_path = plots_dir / "poisson_residuals.html"
    fig_qq = go.Figure(data=[go.Scatter(x=stats.norm.ppf(np.linspace(0.001, 0.999, len(total))), y=np.sort(total), mode="markers")])
    fig_qq.update_layout(title="Poisson QQ Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Observed Total Corners")
    fig_qq.write_html(qq_path)

    fig_res = px.scatter(residual_df, x="value", y="residual", title="Poisson Residuals")
    fig_res.write_html(residual_path)

    lines = [
        "# Poisson Validation",
        "",
        f"- Observed mean total corners: {mean_total:.2f}",
        f"- KS statistic: {ks_statistic:.4f}",
        f"- Chi-square goodness of fit: {chi_sq:.4f}",
        "",
        "## Expected vs observed",
        _format_table(pd.DataFrame({"value": observed.index, "observed": observed.values, "expected": expected.values}).head(20)),
        "",
        "## Plots",
        f"- QQ plot: [poisson_qq.html]({qq_path.name})",
        f"- Residual plot: [poisson_residuals.html]({residual_path.name})",
        "",
    ]
    return "\n".join(lines)


def build_rating_report(df: pd.DataFrame, plots_dir: Path) -> str:
    rating_columns = [
        "home_attack_rating",
        "away_attack_rating",
        "home_defence_rating",
        "away_defence_rating",
        "home_consistency",
        "away_consistency",
    ]
    summary = df.groupby("season")[[*rating_columns, "total_corners"]].mean().reset_index()
    summary = summary.round(3)

    profile_path = plots_dir / "team_profiles.html"
    fig_profiles = px.box(df, x="season", y="home_attack_rating", title="Home attack rating by season")
    fig_profiles.write_html(profile_path)

    line_path = plots_dir / "rating_evolution.html"
    fig_line = px.line(summary, x="season", y=["home_attack_rating", "away_attack_rating"], title="Rating evolution")
    fig_line.write_html(line_path)

    lines = [
        "# Rating Analysis",
        "",
        "## Team profiles",
        _format_table(summary),
        "",
        "## Rating evolution",
        f"- Profile plot: [team_profiles.html]({profile_path.name})",
        f"- Evolution plot: [rating_evolution.html]({line_path.name})",
        "",
    ]
    return "\n".join(lines)


def build_feature_correlation_report(df: pd.DataFrame, plots_dir: Path) -> str:
    numeric_cols = [
        col for col in df.columns
        if col not in {"date", "season", "competition", "home_team", "away_team", "source", "source_file_name", "source_url", "import_date", "fixture_id", "row_hash"}
    ]
    feature_frame = df[numeric_cols].copy()
    target = feature_frame["total_corners"]
    feature_frame = feature_frame.drop(columns=["total_corners"])

    pearson = feature_frame.corrwith(target).sort_values(ascending=False)
    spearman = feature_frame.apply(lambda col: col.corr(target, method="spearman")).sort_values(ascending=False)

    corr_df = pd.DataFrame({"feature": pearson.index, "pearson": pearson.values, "spearman": spearman.reindex(pearson.index).values})
    corr_df = corr_df.round(3)

    heatmap_path = plots_dir / "feature_correlation_heatmap.html"
    heatmap = feature_frame.corr(numeric_only=True)
    fig_heat = go.Figure(data=go.Heatmap(z=heatmap.values, x=heatmap.columns, y=heatmap.columns, hoverongaps=False))
    fig_heat.update_layout(title="Feature Correlation Heatmap")
    fig_heat.write_html(heatmap_path)

    lines = [
        "# Feature Correlation",
        "",
        _format_table(corr_df),
        "",
        "## Heatmap",
        f"- Heatmap: [feature_correlation_heatmap.html]({heatmap_path.name})",
        "",
    ]
    return "\n".join(lines)


def build_season_summary(df: pd.DataFrame, plots_dir: Path) -> str:
    season_summary = df.groupby("season").agg(
        matches=("fixture_id", "count"),
        mean_total_corners=("total_corners", "mean"),
        mean_home_corners=("home_corners", "mean"),
        mean_away_corners=("away_corners", "mean"),
        home_win_rate=("home_goals", lambda s: (s > df.loc[s.index, "away_goals"]).mean()),
    )
    season_summary = season_summary.reset_index()
    season_summary = season_summary.round(3)

    dist_path = plots_dir / "season_total_corners.html"
    fig_dist = px.box(df, x="season", y="total_corners", title="Season total corners distribution")
    fig_dist.write_html(dist_path)

    lines = [
        "# Season Summary",
        "",
        _format_table(season_summary),
        "",
        "## Plot",
        f"- Season distribution: [season_total_corners.html]({dist_path.name})",
        "",
    ]
    return "\n".join(lines)
