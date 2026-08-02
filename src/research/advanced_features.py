from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_EWMA_ALPHA = 0.3

FEATURE_DOCUMENTATION = {
    "match_id": "Canonical fixture identifier copied from the historical match dataset.",
    "season": "Competition season for the match.",
    "date": "Match date in ISO format.",
    "home_team": "Home team name.",
    "away_team": "Away team name.",
    "actual_home_corners": "Observed home-team corner count for the match.",
    "actual_away_corners": "Observed away-team corner count for the match.",
    "actual_total_corners": "Observed total corners for the match.",
    "over_8_5": "Indicator that the total corner count exceeded 8.5.",
    "over_9_5": "Indicator that the total corner count exceeded 9.5.",
    "over_10_5": "Indicator that the total corner count exceeded 10.5.",
    "over_11_5": "Indicator that the total corner count exceeded 11.5.",
    "corners_for_last3": "Three-match rolling average of prior corners scored by the home team.",
    "corners_for_last5": "Five-match rolling average of prior corners scored by the home team.",
    "corners_for_last10": "Ten-match rolling average of prior corners scored by the home team.",
    "corners_against_last3": "Three-match rolling average of prior corners conceded by the home team.",
    "corners_against_last5": "Five-match rolling average of prior corners conceded by the home team.",
    "corners_against_last10": "Ten-match rolling average of prior corners conceded by the home team.",
    "total_corners_last3": "Three-match rolling average of prior total corners involving the home team.",
    "total_corners_last5": "Five-match rolling average of prior total corners involving the home team.",
    "total_corners_last10": "Ten-match rolling average of prior total corners involving the home team.",
    "corners_for_ewma": "EWMA of prior corners scored by the home team.",
    "corners_against_ewma": "EWMA of prior corners conceded by the home team.",
    "total_corners_ewma": "EWMA of prior total corners involving the home team.",
    "home_corners_for_last5": "Five-match rolling average of prior corners scored by the home team.",
    "home_corners_against_last5": "Five-match rolling average of prior corners conceded by the home team.",
    "home_total_corners_last5": "Five-match rolling average of prior total corners involving the home team.",
    "away_corners_for_last5": "Five-match rolling average of prior corners scored by the away team.",
    "away_corners_against_last5": "Five-match rolling average of prior corners conceded by the away team.",
    "away_total_corners_last5": "Five-match rolling average of prior total corners involving the away team.",
    "corners_for_std_last5": "Five-match standard deviation of prior corners scored by the home team.",
    "corners_for_std_last10": "Ten-match standard deviation of prior corners scored by the home team.",
    "corners_against_std_last5": "Five-match standard deviation of prior corners conceded by the home team.",
    "total_corners_std_last5": "Five-match standard deviation of prior total corners involving the home team.",
    "total_corners_std_last10": "Ten-match standard deviation of prior total corners involving the home team.",
    "coefficient_of_variation_last10": "Coefficient of variation based on prior total corners for the home team.",
    "attack_trend": "Difference between the home team’s recent-5 and recent-10 corners-for averages.",
    "defence_trend": "Difference between the home team’s recent-5 and recent-10 corners-against averages.",
    "tempo_trend": "Difference between the home team’s recent-5 and recent-10 total-corners averages.",
    "expected_home_corners_baseline": "Baseline expectation for the home team’s corners.",
    "expected_away_corners_baseline": "Baseline expectation for the away team’s corners.",
    "expected_total_corners_baseline": "Baseline expectation for total corners in the matchup.",
    "attack_difference": "Difference between the home and away teams’ recent corners-for averages.",
    "defence_difference": "Difference between the home and away teams’ recent corners-against averages.",
    "tempo_difference": "Difference between the home and away teams’ recent total-corners averages.",
    "combined_volatility": "Average volatility of the home and away teams.",
    "combined_trend": "Difference between the home and away teams’ trend signals.",
    "home_rest_days": "Days since the home team’s previous match.",
    "away_rest_days": "Days since the away team’s previous match.",
    "rest_days_difference": "Difference between home and away rest days.",
    "home_matches_played": "Number of matches played by the home team before the current match in the season.",
    "away_matches_played": "Number of matches played by the away team before the current match in the season.",
    "season_match_number": "Ordinal match number within the season.",
    "data_quality_score": "Score between 0 and 1 reflecting the amount of prior history available.",
    "insufficient_history": "Flag indicating that the row has limited history and should be treated as cold-start.",
}


def build_advanced_feature_dataset(base_dir: Path | str | None = None, output_dir: Path | str | None = None, ewma_alpha: float = DEFAULT_EWMA_ALPHA) -> pd.DataFrame:
    base_dir = resolve_base_dir(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir

    matches_path = resolve_matches_path(base_dir)
    if not matches_path.exists():
        raise FileNotFoundError(f"Historical match data not found: {matches_path}")

    matches = pd.read_parquet(matches_path)
    if matches.empty:
        raise ValueError("Historical match data is empty")

    working = matches.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if working["date"].isna().any():
        raise ValueError("Match dates must be valid")

    working = working.sort_values(["season", "date", "fixture_id"]).reset_index(drop=True)
    working["match_id"] = working["fixture_id"].astype(int)

    rows: List[Dict[str, Any]] = []
    team_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    season_match_counter: Dict[str, int] = defaultdict(int)

    for idx, row in working.iterrows():
        season = str(row["season"])
        season_match_counter[season] += 1
        season_match_number = season_match_counter[season]

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_corners = float(row["home_corners"])
        away_corners = float(row["away_corners"])
        total_corners = float(row["total_corners"])

        home_history = team_history[home_team]
        away_history = team_history[away_team]

        home_recent_for = _rolling_stats(home_history, kind="for")
        home_recent_against = _rolling_stats(home_history, kind="against")
        home_recent_total = _rolling_stats(home_history, kind="total")
        away_recent_for = _rolling_stats(away_history, kind="for")
        away_recent_against = _rolling_stats(away_history, kind="against")
        away_recent_total = _rolling_stats(away_history, kind="total")

        home_last3_for = _average_last_n(home_history, n=3, kind="for")
        home_last5_for = _average_last_n(home_history, n=5, kind="for")
        home_last10_for = _average_last_n(home_history, n=10, kind="for")
        home_last3_against = _average_last_n(home_history, n=3, kind="against")
        home_last5_against = _average_last_n(home_history, n=5, kind="against")
        home_last10_against = _average_last_n(home_history, n=10, kind="against")
        home_last3_total = _average_last_n(home_history, n=3, kind="total")
        home_last5_total = _average_last_n(home_history, n=5, kind="total")
        home_last10_total = _average_last_n(home_history, n=10, kind="total")

        away_last3_for = _average_last_n(away_history, n=3, kind="for")
        away_last5_for = _average_last_n(away_history, n=5, kind="for")
        away_last10_for = _average_last_n(away_history, n=10, kind="for")
        away_last3_against = _average_last_n(away_history, n=3, kind="against")
        away_last5_against = _average_last_n(away_history, n=5, kind="against")
        away_last10_against = _average_last_n(away_history, n=10, kind="against")
        away_last3_total = _average_last_n(away_history, n=3, kind="total")
        away_last5_total = _average_last_n(away_history, n=5, kind="total")
        away_last10_total = _average_last_n(away_history, n=10, kind="total")

        home_for_ewma = _ewma(home_history, kind="for", alpha=ewma_alpha)
        home_against_ewma = _ewma(home_history, kind="against", alpha=ewma_alpha)
        home_total_ewma = _ewma(home_history, kind="total", alpha=ewma_alpha)
        away_for_ewma = _ewma(away_history, kind="for", alpha=ewma_alpha)
        away_against_ewma = _ewma(away_history, kind="against", alpha=ewma_alpha)
        away_total_ewma = _ewma(away_history, kind="total", alpha=ewma_alpha)

        home_for_std_5 = _std_last_n(home_history, n=5, kind="for")
        home_for_std_10 = _std_last_n(home_history, n=10, kind="for")
        home_against_std_5 = _std_last_n(home_history, n=5, kind="against")
        home_total_std_5 = _std_last_n(home_history, n=5, kind="total")
        home_total_std_10 = _std_last_n(home_history, n=10, kind="total")
        home_cv_10 = _coefficient_of_variation(home_history, n=10, kind="total")

        away_for_std_5 = _std_last_n(away_history, n=5, kind="for")
        away_against_std_5 = _std_last_n(away_history, n=5, kind="against")
        away_total_std_5 = _std_last_n(away_history, n=5, kind="total")

        home_attack_trend = home_last5_for - home_last10_for
        home_defence_trend = home_last5_against - home_last10_against
        home_tempo_trend = home_last5_total - home_last10_total

        away_attack_trend = away_last5_for - away_last10_for
        away_defence_trend = away_last5_against - away_last10_against
        away_tempo_trend = away_last5_total - away_last10_total

        expected_home_corners_baseline = 0.5 * (home_last5_for + away_last5_against)
        expected_away_corners_baseline = 0.5 * (away_last5_for + home_last5_against)
        expected_total_corners_baseline = 0.5 * (home_last5_total + away_last5_total)

        attack_difference = home_last5_for - away_last5_for
        defence_difference = home_last5_against - away_last5_against
        tempo_difference = home_last5_total - away_last5_total
        combined_volatility = 0.5 * (home_for_std_5 + away_for_std_5)
        combined_trend = home_attack_trend - away_attack_trend

        home_rest_days = _rest_days(home_history, row["date"])
        away_rest_days = _rest_days(away_history, row["date"])
        rest_days_difference = home_rest_days - away_rest_days

        home_matches_played = len([entry for entry in home_history if str(entry["season"]) == season])
        away_matches_played = len([entry for entry in away_history if str(entry["season"]) == season])

        data_quality_score = min(1.0, max(0.0, len(home_history) / 10.0))
        insufficient_history = len(home_history) < 5

        rows.append(
            {
                "match_id": int(row["fixture_id"]),
                "season": season,
                "date": row["date"].strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
                "actual_home_corners": home_corners,
                "actual_away_corners": away_corners,
                "actual_total_corners": total_corners,
                "over_8_5": int(total_corners > 8.5),
                "over_9_5": int(total_corners > 9.5),
                "over_10_5": int(total_corners > 10.5),
                "over_11_5": int(total_corners > 11.5),
                "corners_for_last3": home_last3_for,
                "corners_for_last5": home_last5_for,
                "corners_for_last10": home_last10_for,
                "corners_against_last3": home_last3_against,
                "corners_against_last5": home_last5_against,
                "corners_against_last10": home_last10_against,
                "total_corners_last3": home_last3_total,
                "total_corners_last5": home_last5_total,
                "total_corners_last10": home_last10_total,
                "corners_for_ewma": home_for_ewma,
                "corners_against_ewma": home_against_ewma,
                "total_corners_ewma": home_total_ewma,
                "home_corners_for_last5": home_last5_for,
                "home_corners_against_last5": home_last5_against,
                "home_total_corners_last5": home_last5_total,
                "away_corners_for_last5": away_last5_for,
                "away_corners_against_last5": away_last5_against,
                "away_total_corners_last5": away_last5_total,
                "corners_for_std_last5": home_for_std_5,
                "corners_for_std_last10": home_for_std_10,
                "corners_against_std_last5": home_against_std_5,
                "total_corners_std_last5": home_total_std_5,
                "total_corners_std_last10": home_total_std_10,
                "coefficient_of_variation_last10": home_cv_10,
                "attack_trend": home_attack_trend,
                "defence_trend": home_defence_trend,
                "tempo_trend": home_tempo_trend,
                "expected_home_corners_baseline": expected_home_corners_baseline,
                "expected_away_corners_baseline": expected_away_corners_baseline,
                "expected_total_corners_baseline": expected_total_corners_baseline,
                "attack_difference": attack_difference,
                "defence_difference": defence_difference,
                "tempo_difference": tempo_difference,
                "combined_volatility": combined_volatility,
                "combined_trend": combined_trend,
                "home_rest_days": int(home_rest_days),
                "away_rest_days": int(away_rest_days),
                "rest_days_difference": int(rest_days_difference),
                "home_matches_played": int(home_matches_played),
                "away_matches_played": int(away_matches_played),
                "season_match_number": int(season_match_number),
                "data_quality_score": float(data_quality_score),
                "insufficient_history": bool(insufficient_history),
            }
        )

        team_history[home_team].append(
            {
                "date": row["date"],
                "season": season,
                "for_value": home_corners,
                "against_value": away_corners,
                "total_value": total_corners,
            }
        )
        team_history[away_team].append(
            {
                "date": row["date"],
                "season": season,
                "for_value": away_corners,
                "against_value": home_corners,
                "total_value": total_corners,
            }
        )

    dataset = pd.DataFrame(rows)
    dataset = _clean_feature_frame(dataset)

    output_path = output_dir / "data" / "research" / "advanced_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    docs_path = output_dir / "docs" / "ADVANCED_FEATURE_DICTIONARY.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(build_feature_dictionary(), encoding="utf-8")

    report_path = output_dir / "reports" / "advanced_feature_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_validation_report(dataset), encoding="utf-8")

    return dataset


def resolve_base_dir(base_dir: Path | str | None) -> Path:
    if base_dir is None:
        return Path.cwd()
    return Path(base_dir)


def resolve_matches_path(base_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "data" / "processed" / "serie_a_matches.parquet",
        repo_root / "data" / "processed" / "serie_a_matches.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _rolling_stats(history: List[Dict[str, Any]], kind: str) -> float:
    if not history:
        return 0.0
    values = [entry["for_value"] if kind == "for" else entry["against_value"] if kind == "against" else entry["total_value"] for entry in history]
    if not values:
        return 0.0
    return float(pd.Series(values).tail(5).mean())


def _average_last_n(history: List[Dict[str, Any]], n: int, kind: str) -> float:
    if not history:
        return 0.0
    values = [entry["for_value"] if kind == "for" else entry["against_value"] if kind == "against" else entry["total_value"] for entry in history]
    if not values:
        return 0.0
    return float(pd.Series(values).tail(n).mean())


def _std_last_n(history: List[Dict[str, Any]], n: int, kind: str) -> float:
    if not history:
        return 0.0
    values = [entry["for_value"] if kind == "for" else entry["against_value"] if kind == "against" else entry["total_value"] for entry in history]
    if len(values) < 2:
        return 0.0
    series = pd.Series(values).tail(n)
    if len(series) < 2:
        return 0.0
    return float(series.std(ddof=0))


def _coefficient_of_variation(history: List[Dict[str, Any]], n: int, kind: str) -> float:
    if not history:
        return 0.0
    values = [entry["for_value"] if kind == "for" else entry["against_value"] if kind == "against" else entry["total_value"] for entry in history]
    series = pd.Series(values).tail(n)
    if series.empty:
        return 0.0
    mean = float(series.mean())
    if mean == 0:
        return 0.0
    return float(series.std(ddof=0) / mean)


def _ewma(history: List[Dict[str, Any]], kind: str, alpha: float) -> float:
    if not history:
        return 0.0
    values = [entry["for_value"] if kind == "for" else entry["against_value"] if kind == "against" else entry["total_value"] for entry in history]
    if not values:
        return 0.0
    ewma = 0.0
    for value in values:
        ewma = alpha * float(value) + (1 - alpha) * ewma
    return float(ewma)


def _rest_days(history: List[Dict[str, Any]], current_date: pd.Timestamp) -> int:
    if not history:
        return 0
    last_date = history[-1]["date"]
    return int((current_date - last_date).days)


def _clean_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_frame = dataset.copy()
    feature_frame["date"] = feature_frame["date"].astype(str)
    feature_frame["match_id"] = feature_frame["match_id"].astype(int)
    feature_frame["season_match_number"] = feature_frame["season_match_number"].astype(int)
    feature_frame["home_rest_days"] = feature_frame["home_rest_days"].astype(int)
    feature_frame["away_rest_days"] = feature_frame["away_rest_days"].astype(int)
    feature_frame["rest_days_difference"] = feature_frame["rest_days_difference"].astype(int)
    feature_frame["home_matches_played"] = feature_frame["home_matches_played"].astype(int)
    feature_frame["away_matches_played"] = feature_frame["away_matches_played"].astype(int)
    feature_frame["insufficient_history"] = feature_frame["insufficient_history"].astype(bool)

    for column in feature_frame.columns:
        if column in {"match_id", "season", "date", "home_team", "away_team"}:
            continue
        if column in {"insufficient_history"}:
            continue
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    for column in feature_frame.columns:
        if column in {"match_id", "season", "date", "home_team", "away_team", "insufficient_history"}:
            continue
        feature_frame[column] = feature_frame[column].fillna(0.0)
        feature_frame[column] = feature_frame[column].replace([np.inf, -np.inf], 0.0)

    feature_frame["data_quality_score"] = feature_frame["data_quality_score"].clip(0.0, 1.0)
    feature_frame["insufficient_history"] = feature_frame["insufficient_history"].astype(bool)
    return feature_frame


def build_feature_dictionary() -> str:
    lines = ["# Advanced Feature Dictionary", "", "This document lists the leakage-safe advanced features generated for each match.", ""]
    for feature_name, description in FEATURE_DOCUMENTATION.items():
        lines.append(f"- **{feature_name}**: {description}")
    return "\n".join(lines) + "\n"


def build_validation_report(dataset: pd.DataFrame) -> str:
    numeric_columns = [
        column
        for column in dataset.columns
        if column not in {"match_id", "season", "date", "home_team", "away_team", "insufficient_history"}
    ]
    numeric_frame = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    missing_value_counts = dataset.isna().sum().to_dict()
    infinite_value_counts = {column: int(np.isinf(numeric_frame[column]).sum()) for column in numeric_frame.columns}
    cold_start_rows = dataset.groupby("season")["insufficient_history"].sum().to_dict()

    pearson = numeric_frame.corrwith(numeric_frame["actual_total_corners"]).drop("actual_total_corners").sort_values(ascending=False)
    spearman = numeric_frame.corrwith(numeric_frame["actual_total_corners"], method="spearman").drop("actual_total_corners").sort_values(ascending=False)

    feature_matrix = numeric_frame.drop(columns=["actual_total_corners", "actual_home_corners", "actual_away_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"])
    corr_matrix = feature_matrix.corr(method="pearson")
    corr_pairs = []
    for left in corr_matrix.columns:
        for right in corr_matrix.columns:
            if left >= right:
                continue
            value = float(corr_matrix.loc[left, right])
            if abs(value) > 0.90:
                corr_pairs.append((left, right, value))
    corr_pairs = sorted(corr_pairs, key=lambda item: abs(item[2]), reverse=True)[:15]

    lines = [
        "# Advanced Feature Validation",
        "",
        f"- Rows: {len(dataset)}",
        f"- Generated features: {len([column for column in dataset.columns if column not in {'match_id', 'season', 'date', 'home_team', 'away_team', 'home_corners', 'away_corners', 'total_corners', 'actual_home_corners', 'actual_away_corners', 'actual_total_corners', 'over_8_5', 'over_9_5', 'over_10_5', 'over_11_5'}])}",
        "- Missing value counts:",
        *[f"  - {column}: {count}" for column, count in missing_value_counts.items() if count > 0],
        "- Infinite value counts:",
        *[f"  - {column}: {count}" for column, count in infinite_value_counts.items() if count > 0],
        "- Cold-start rows by season:",
        *[f"  - {season}: {count}" for season, count in cold_start_rows.items()],
        "- Leakage checks:",
        "  - Rolling calculations are based on prior rows only.",
        "  - Current-match values do not contribute to their own features.",
        "  - No future match values enter any rolling statistic.",
        "- Descriptive statistics:",
        numeric_frame.describe().to_string(),
        "- Pearson correlation with total corners:",
        pearson.to_string(),
        "- Spearman correlation with total corners:",
        spearman.to_string(),
        "- Top 15 positive correlations:",
        pearson[pearson > 0].head(15).to_string(),
        "- Top 15 negative correlations:",
        pearson[pearson < 0].tail(15).to_string(),
        "- Highly collinear feature pairs above 0.90:",
        "\n".join([f"  - {left} / {right}: {value:.3f}" for left, right, value in corr_pairs]),
        "",
    ]
    return "\n".join(lines) + "\n"
