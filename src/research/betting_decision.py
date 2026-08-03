from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.odds_validator import validate_odds_dataframe


DEFAULT_THRESHOLDS = {
    "minimum_ev": 0.05,
    "minimum_confidence": 60.0,
    "minimum_probability": 0.60,
    "minimum_odds": 1.50,
}


def run_betting_decision_layer(
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
    external_odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    output_dir = Path(output_dir or base_dir)

    input_path = base_dir / "data" / "research" / "confidence_predictions.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing confidence artifact: {input_path}")

    predictions = pd.read_parquet(input_path)
    if predictions.empty:
        raise ValueError("Confidence artifact is empty")

    resolved_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    validated_odds = pd.DataFrame()
    validation_errors: list[str] = []
    if external_odds is not None and not external_odds.empty:
        validated_odds, validation_errors = validate_odds_dataframe(external_odds, fixtures=predictions)

    decisions = build_betting_decisions(predictions, resolved_thresholds, validated_odds)

    output_paths = write_reports_and_artifacts(decisions, base_dir=base_dir, output_dir=output_dir, validation_errors=validation_errors)

    summary = build_summary(decisions)
    return {
        "decisions": decisions,
        "summary": summary,
        "output_paths": output_paths,
        "thresholds": resolved_thresholds,
        "validation_errors": validation_errors,
    }


def build_betting_decisions(predictions: pd.DataFrame, thresholds: dict[str, float], validated_odds: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()

    if "decision_state" not in frame.columns:
        frame["decision_state"] = "WATCH"
    if "confidence_score" not in frame.columns:
        frame["confidence_score"] = 0.0

    frame = frame.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    frame["market_name"] = ""
    frame["odds"] = np.nan
    frame["opening_odds"] = np.nan
    frame["closing_odds"] = np.nan
    frame["implied_probability"] = np.nan
    frame["raw_edge"] = np.nan
    frame["ev"] = np.nan
    frame["edge"] = np.nan
    frame["confidence"] = np.clip(frame["confidence_score"] * 100.0, 0.0, 100.0)
    frame["odds_available"] = False
    frame["odds_validated"] = False
    frame["is_real_market_odds"] = False
    frame["roi_eligible"] = False
    frame["recommendation"] = "NO DATA"

    if not validated_odds.empty:
        for idx, row in frame.iterrows():
            match_row = validated_odds[validated_odds["match_id"].astype(str).str.strip() == str(row["match_id"])]
            if match_row.empty:
                continue
            odds_row = match_row.iloc[0]
            frame.loc[idx, "market_name"] = f"{odds_row['market']} {odds_row['line']}"
            frame.loc[idx, "odds"] = float(odds_row["closing_odds"])
            frame.loc[idx, "opening_odds"] = float(odds_row["opening_odds"])
            frame.loc[idx, "closing_odds"] = float(odds_row["closing_odds"])
            frame.loc[idx, "odds_available"] = bool(True)
            frame.loc[idx, "odds_validated"] = bool(True)
            frame.loc[idx, "is_real_market_odds"] = bool(True)

            model_prob = _extract_model_probability(row, odds_row)
            frame.loc[idx, "model_probability"] = model_prob
            implied_probability = 1.0 / float(odds_row["closing_odds"])
            raw_edge = model_prob - implied_probability
            ev = model_prob * float(odds_row["closing_odds"]) - 1.0
            frame.loc[idx, "implied_probability"] = implied_probability
            frame.loc[idx, "raw_edge"] = raw_edge
            frame.loc[idx, "ev"] = ev
            frame.loc[idx, "edge"] = raw_edge

            outcome_exists = _outcome_exists(row, odds_row)
            fixture_verified = _fixture_verified(row, odds_row)
            frame.loc[idx, "roi_eligible"] = bool(frame.loc[idx, "odds_available"] and outcome_exists and pd.notna(frame.loc[idx, "closing_odds"]) and fixture_verified)

            if (
                frame.loc[idx, "confidence"] >= thresholds["minimum_confidence"]
                and frame.loc[idx, "ev"] >= thresholds["minimum_ev"]
                and model_prob >= thresholds["minimum_probability"]
                and float(odds_row["closing_odds"]) >= thresholds["minimum_odds"]
                and str(row["decision_state"]).upper() in {"ACCEPT", "WATCH"}
            ):
                frame.loc[idx, "recommendation"] = "BET"
            elif (
                frame.loc[idx, "confidence"] >= thresholds["minimum_confidence"] * 0.8
                and frame.loc[idx, "ev"] >= thresholds["minimum_ev"] * 0.5
                and model_prob >= thresholds["minimum_probability"] * 0.9
            ):
                frame.loc[idx, "recommendation"] = "WATCH"
            else:
                frame.loc[idx, "recommendation"] = "NO BET"
    else:
        frame["model_probability"] = np.nan
        frame["recommendation"] = "NO DATA"

    decisions = frame[[
        "match_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "market_name",
        "odds",
        "opening_odds",
        "closing_odds",
        "implied_probability",
        "raw_edge",
        "model_probability",
        "ev",
        "edge",
        "confidence",
        "decision_state",
        "odds_available",
        "odds_validated",
        "is_real_market_odds",
        "roi_eligible",
        "recommendation",
    ]].copy()
    for column in ["odds_available", "odds_validated", "is_real_market_odds", "roi_eligible"]:
        decisions[column] = decisions[column].apply(lambda value: bool(value)).astype(object)
    return decisions


def _extract_model_probability(row: pd.Series, odds_row: pd.Series) -> float:
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_OVER" and str(odds_row.get("line", "")).strip() == "8.5":
        return float(np.clip(float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_OVER" and str(odds_row.get("line", "")).strip() == "9.5":
        return float(np.clip(float(row.get("predicted_probability_over_9_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_OVER" and str(odds_row.get("line", "")).strip() == "10.5":
        return float(np.clip(float(row.get("predicted_probability_over_10_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_OVER" and str(odds_row.get("line", "")).strip() == "11.5":
        return float(np.clip(float(row.get("predicted_probability_over_11_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_UNDER" and str(odds_row.get("line", "")).strip() == "8.5":
        return float(np.clip(1.0 - float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_UNDER" and str(odds_row.get("line", "")).strip() == "9.5":
        return float(np.clip(1.0 - float(row.get("predicted_probability_over_9_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_UNDER" and str(odds_row.get("line", "")).strip() == "10.5":
        return float(np.clip(1.0 - float(row.get("predicted_probability_over_10_5", 0.0)), 0.0, 1.0))
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_UNDER" and str(odds_row.get("line", "")).strip() == "11.5":
        return float(np.clip(1.0 - float(row.get("predicted_probability_over_11_5", 0.0)), 0.0, 1.0))
    return float(np.clip(float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))


def _outcome_exists(row: pd.Series, odds_row: pd.Series) -> bool:
    line = str(odds_row.get("line", "")).strip()
    if str(odds_row.get("market", "")).upper() == "TOTAL_CORNERS_UNDER":
        return False
    column_name = {
        "8.5": "actual_outcome_over_8_5",
        "9.5": "actual_outcome_over_9_5",
        "10.5": "actual_outcome_over_10_5",
        "11.5": "actual_outcome_over_11_5",
    }.get(line)
    if not column_name:
        return False
    value = row.get(column_name)
    return pd.notna(value)


def _fixture_verified(row: pd.Series, odds_row: pd.Series) -> bool:
    if pd.isna(row.get("match_id")):
        return False
    return str(row.get("match_id")) == str(odds_row.get("match_id"))


def write_reports_and_artifacts(decisions: pd.DataFrame, base_dir: Path, output_dir: Path, validation_errors: list[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data" / "research"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "betting_decisions.parquet"
    decisions.to_parquet(parquet_path, index=False)

    summary_md = build_summary_markdown(decisions)
    ev_dist_md = build_ev_distribution_markdown(decisions)
    betting_summary_md = build_betting_summary_markdown(decisions)
    readiness_md = build_odds_ingestion_readiness_markdown(decisions, validation_errors)
    limitations_md = build_betting_layer_limitations_markdown(decisions)

    (reports_dir / "betting_decision.md").write_text(summary_md, encoding="utf-8")
    (reports_dir / "ev_distribution.md").write_text(ev_dist_md, encoding="utf-8")
    (reports_dir / "betting_summary.md").write_text(betting_summary_md, encoding="utf-8")
    (reports_dir / "odds_ingestion_readiness.md").write_text(readiness_md, encoding="utf-8")
    (reports_dir / "betting_layer_limitations.md").write_text(limitations_md, encoding="utf-8")

    policy_path = reports_dir / "betting_policy.json"
    policy_path.write_text(json.dumps({"thresholds": DEFAULT_THRESHOLDS}, indent=2), encoding="utf-8")

    return {
        "parquet": parquet_path,
        "report": reports_dir / "betting_decision.md",
        "ev_distribution": reports_dir / "ev_distribution.md",
        "summary": reports_dir / "betting_summary.md",
        "policy": policy_path,
        "odds_readiness": reports_dir / "odds_ingestion_readiness.md",
        "limitations": reports_dir / "betting_layer_limitations.md",
    }


def build_summary(decisions: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(decisions)),
        "bet_count": int((decisions["recommendation"] == "BET").sum()),
        "watch_count": int((decisions["recommendation"] == "WATCH").sum()),
        "no_bet_count": int((decisions["recommendation"] == "NO BET").sum()),
        "no_data_count": int((decisions["recommendation"] == "NO DATA").sum()),
        "avg_ev": float(decisions["ev"].mean()) if decisions["ev"].notna().any() else float("nan"),
        "avg_confidence": float(decisions["confidence"].mean()),
        "roi_eligible_rows": int(decisions["roi_eligible"].sum()) if "roi_eligible" in decisions else 0,
    }


def build_summary_markdown(decisions: pd.DataFrame) -> str:
    summary = build_summary(decisions)
    lines = [
        "# Betting Decision Layer",
        "",
        "This report summarizes the research-only betting decision layer built from validated external odds only.",
        "",
        f"- Rows analyzed: {summary['rows']}",
        f"- BET recommendations: {summary['bet_count']}",
        f"- WATCH recommendations: {summary['watch_count']}",
        f"- NO BET recommendations: {summary['no_bet_count']}",
        f"- NO DATA recommendations: {summary['no_data_count']}",
        f"- ROI-eligible rows: {summary['roi_eligible_rows']}",
        f"- Average EV: {summary['avg_ev']:.3f}",
        f"- Average confidence: {summary['avg_confidence']:.1f}",
        "",
        "The layer uses explicit external odds and never generates synthetic odds internally.",
    ]
    return "\n".join(lines) + "\n"


def build_ev_distribution_markdown(decisions: pd.DataFrame) -> str:
    ev_values = decisions["ev"].dropna()
    quantiles = ev_values.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict() if not ev_values.empty else {0.0: np.nan, 0.25: np.nan, 0.5: np.nan, 0.75: np.nan, 1.0: np.nan}
    lines = [
        "# EV Distribution",
        "",
        "This report shows the distribution of expected value across externally validated odds opportunities.",
        "",
        f"- Minimum EV: {quantiles[0.0]:.3f}",
        f"- Q1 EV: {quantiles[0.25]:.3f}",
        f"- Median EV: {quantiles[0.5]:.3f}",
        f"- Q3 EV: {quantiles[0.75]:.3f}",
        f"- Maximum EV: {quantiles[1.0]:.3f}",
    ]
    return "\n".join(lines) + "\n"


def build_betting_summary_markdown(decisions: pd.DataFrame) -> str:
    lines = [
        "# Betting Summary",
        "",
        "## Recommended Opportunities",
        "",
    ]
    if decisions.empty:
        lines.append("No opportunities available.")
        return "\n".join(lines) + "\n"

    bets = decisions.loc[decisions["recommendation"] == "BET"].copy()
    if bets.empty:
        lines.append("No BET recommendations were generated.")
    else:
        top = bets.nlargest(5, "ev")
        for _, row in top.iterrows():
            lines.append(
                f"- {row['home_team']} vs {row['away_team']}: EV {row['ev']:.3f}, confidence {row['confidence']:.1f}, odds {row['closing_odds']:.2f}"
            )
    return "\n".join(lines) + "\n"


def build_odds_ingestion_readiness_markdown(decisions: pd.DataFrame, validation_errors: list[str]) -> str:
    lines = [
        "# Odds Ingestion Readiness",
        "",
        "This report tracks whether the betting layer can rely on validated external corner odds.",
        "",
        f"- External odds rows available: {int(decisions['odds_available'].sum())}",
        f"- Validated odds rows: {int(decisions['odds_validated'].sum())}",
        f"- ROI-eligible rows: {int(decisions['roi_eligible'].sum())}",
        f"- Validation errors: {len(validation_errors)}",
    ]
    if validation_errors:
        lines.append("")
        lines.append("## Validation issues")
        for error in validation_errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def build_betting_layer_limitations_markdown(decisions: pd.DataFrame) -> str:
    return "\n".join([
        "# Betting Layer Limitations",
        "",
        "No historical ROI can be claimed until real corner odds are imported.",
        "",
        f"The current layer only produces EV and edge values for rows with validated external corner odds. Rows without real odds remain marked as NO DATA and are not ROI-eligible.",
        "",
        f"ROI-eligible rows: {int(decisions['roi_eligible'].sum())}",
    ]) + "\n"


if __name__ == "__main__":
    run_betting_decision_layer()
