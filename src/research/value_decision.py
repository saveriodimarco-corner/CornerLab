from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.odds_validator import validate_odds_dataframe
from src.research.bankroll_tracker import BankrollTracker
from src.research.expected_value import expected_value
from src.research.implied_probability import decimal_odds_to_implied_probability
from src.research.kelly import full_kelly, half_kelly, quarter_kelly

DEFAULT_VALUE_POLICY = {
    "confidence_threshold": 60.0,
    "minimum_ev": 0.05,
    "minimum_probability": 0.60,
    "minimum_odds": 1.50,
}


def run_value_betting_engine(
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    external_odds: pd.DataFrame | None = None,
    bankroll_start: float = 100.0,
    policy: dict[str, float] | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    output_dir = Path(output_dir or base_dir)

    input_path = base_dir / "data" / "research" / "confidence_predictions.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing confidence artifact: {input_path}")

    predictions = pd.read_parquet(input_path)
    if predictions.empty:
        raise ValueError("Confidence artifact is empty")

    resolved_policy = {**DEFAULT_VALUE_POLICY, **(policy or {})}
    validated_odds = pd.DataFrame()
    validation_errors: list[str] = []
    if external_odds is not None and not external_odds.empty:
        validated_odds, validation_errors = validate_odds_dataframe(external_odds, fixtures=predictions)

    decisions = build_value_betting_decisions(predictions, validated_odds, resolved_policy, bankroll_start=bankroll_start)
    output_paths = write_value_betting_reports(decisions, output_dir=output_dir)
    summary = build_value_summary(decisions, bankroll_start=bankroll_start)
    return {
        "decisions": decisions,
        "summary": summary,
        "output_paths": output_paths,
        "policy": resolved_policy,
        "validation_errors": validation_errors,
    }


def build_value_betting_decisions(
    predictions: pd.DataFrame,
    validated_odds: pd.DataFrame,
    policy: dict[str, float] | None = None,
    bankroll_start: float = 100.0,
) -> pd.DataFrame:
    policy = policy or DEFAULT_VALUE_POLICY
    work = predictions.copy()
    work = work.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    records: list[dict[str, Any]] = []
    tracker = BankrollTracker(bankroll_start=bankroll_start, bankroll=bankroll_start)

    for _, row in work.iterrows():
        match_rows = validated_odds.loc[validated_odds["match_id"].astype(str).str.strip() == str(row["match_id"])] if not validated_odds.empty else pd.DataFrame()
        if match_rows.empty:
            records.append(_build_no_bet_record(row, policy, tracker, bankroll_start))
            continue

        for _, odds_row in match_rows.iterrows():
            record = _build_bet_record(row, odds_row, policy, tracker, bankroll_start)
            records.append(record)

    decisions = pd.DataFrame(records)
    if decisions.empty:
        return pd.DataFrame(columns=[
            "match_id",
            "season",
            "date",
            "home_team",
            "away_team",
            "market",
            "line",
            "odds",
            "implied_probability",
            "model_probability",
            "ev",
            "confidence",
            "kelly_fraction_full",
            "kelly_fraction_half",
            "kelly_fraction_quarter",
            "decision",
            "stake",
            "bankroll",
            "cumulative_profit",
            "roi",
            "yield",
            "max_drawdown",
        ])

    decisions = decisions.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    return decisions


def _build_no_bet_record(row: pd.Series, policy: dict[str, float], tracker: BankrollTracker, bankroll_start: float) -> dict[str, Any]:
    confidence = float(np.clip(float(row.get("confidence_score", 0.0)), 0.0, 100.0))
    return {
        "match_id": row.get("match_id"),
        "season": row.get("season"),
        "date": row.get("date"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "market": "",
        "line": "",
        "odds": np.nan,
        "implied_probability": np.nan,
        "model_probability": np.nan,
        "ev": np.nan,
        "confidence": confidence,
        "kelly_fraction_full": 0.0,
        "kelly_fraction_half": 0.0,
        "kelly_fraction_quarter": 0.0,
        "decision": "NO BET",
        "stake": 0.0,
        "bankroll": tracker.bankroll,
        "cumulative_profit": tracker.cumulative_profit,
        "roi": tracker.cumulative_profit / bankroll_start if bankroll_start else 0.0,
        "yield": tracker.cumulative_profit / tracker.total_staked if tracker.total_staked else 0.0,
        "max_drawdown": tracker.max_drawdown,
    }


def _build_bet_record(row: pd.Series, odds_row: pd.Series, policy: dict[str, float], tracker: BankrollTracker, bankroll_start: float) -> dict[str, Any]:
    confidence = float(np.clip(float(row.get("confidence_score", 0.0)), 0.0, 100.0))
    model_probability = _extract_model_probability(row, odds_row)
    odds = float(odds_row["closing_odds"])
    implied_probability = decimal_odds_to_implied_probability(odds)
    ev = expected_value(model_probability, odds)
    kelly_full = full_kelly(model_probability, odds)
    kelly_half = half_kelly(model_probability, odds)
    kelly_quarter = quarter_kelly(model_probability, odds)

    decision = "NO BET"
    if confidence >= float(policy.get("confidence_threshold", 60.0)) and ev >= float(policy.get("minimum_ev", 0.05)) and kelly_full > 0.0:
        decision = "BET"
    elif ev > 0.0 and confidence < float(policy.get("confidence_threshold", 60.0)):
        decision = "WATCH"

    stake = 0.0
    if decision == "BET":
        stake = max(0.0, tracker.bankroll * kelly_full)

    outcome = _result_outcome(row, odds_row)
    bankroll_metrics = tracker.update(stake=stake, outcome=outcome, odds=odds)
    return {
        "match_id": row.get("match_id"),
        "season": row.get("season"),
        "date": row.get("date"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "market": str(odds_row.get("market", "")),
        "line": str(odds_row.get("line", "")),
        "odds": odds,
        "implied_probability": implied_probability,
        "model_probability": model_probability,
        "ev": ev,
        "confidence": confidence,
        "kelly_fraction_full": kelly_full,
        "kelly_fraction_half": kelly_half,
        "kelly_fraction_quarter": kelly_quarter,
        "decision": decision,
        "stake": stake,
        "bankroll": bankroll_metrics["bankroll"],
        "cumulative_profit": bankroll_metrics["cumulative_profit"],
        "roi": bankroll_metrics["roi"],
        "yield": bankroll_metrics["yield"],
        "max_drawdown": bankroll_metrics["max_drawdown"],
    }


def _extract_model_probability(row: pd.Series, odds_row: pd.Series) -> float:
    line = str(odds_row.get("line", "")).strip()
    market = str(odds_row.get("market", "")).upper()
    if market == "TOTAL_CORNERS_OVER":
        if line == "8.5":
            return float(np.clip(float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))
        if line == "9.5":
            return float(np.clip(float(row.get("predicted_probability_over_9_5", 0.0)), 0.0, 1.0))
        if line == "10.5":
            return float(np.clip(float(row.get("predicted_probability_over_10_5", 0.0)), 0.0, 1.0))
        if line == "11.5":
            return float(np.clip(float(row.get("predicted_probability_over_11_5", 0.0)), 0.0, 1.0))
    if market == "TOTAL_CORNERS_UNDER":
        if line == "8.5":
            return float(np.clip(1.0 - float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))
        if line == "9.5":
            return float(np.clip(1.0 - float(row.get("predicted_probability_over_9_5", 0.0)), 0.0, 1.0))
        if line == "10.5":
            return float(np.clip(1.0 - float(row.get("predicted_probability_over_10_5", 0.0)), 0.0, 1.0))
        if line == "11.5":
            return float(np.clip(1.0 - float(row.get("predicted_probability_over_11_5", 0.0)), 0.0, 1.0))
    return float(np.clip(float(row.get("predicted_probability_over_8_5", 0.0)), 0.0, 1.0))


def _result_outcome(row: pd.Series, odds_row: pd.Series) -> int:
    line = str(odds_row.get("line", "")).strip()
    market = str(odds_row.get("market", "")).upper()
    if market == "TOTAL_CORNERS_OVER":
        column = {
            "8.5": "actual_outcome_over_8_5",
            "9.5": "actual_outcome_over_9_5",
            "10.5": "actual_outcome_over_10_5",
            "11.5": "actual_outcome_over_11_5",
        }.get(line)
        if column and row.get(column) is not None:
            return int(bool(row.get(column)))
    if market == "TOTAL_CORNERS_UNDER":
        column = {
            "8.5": "actual_outcome_over_8_5",
            "9.5": "actual_outcome_over_9_5",
            "10.5": "actual_outcome_over_10_5",
            "11.5": "actual_outcome_over_11_5",
        }.get(line)
        if column and row.get(column) is not None:
            return int(bool(1 - int(row.get(column))))
    return 0


def write_value_betting_reports(decisions: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir if output_dir.name.lower() == "reports" else output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir

    bets_path = reports_dir / "value_bets.csv"
    summary_path = reports_dir / "value_summary.md"
    bankroll_path = reports_dir / "bankroll_curve.csv"

    if not decisions.empty:
        decisions.to_csv(bets_path, index=False)
        bankroll_curve = decisions[["match_id", "date", "decision", "stake", "bankroll", "cumulative_profit", "roi", "yield", "max_drawdown"]].copy()
        bankroll_curve.to_csv(bankroll_path, index=False)
    else:
        pd.DataFrame(columns=["match_id", "date", "decision", "stake", "bankroll", "cumulative_profit", "roi", "yield", "max_drawdown"]).to_csv(bankroll_path, index=False)
        decisions.to_csv(bets_path, index=False)

    summary_path.write_text(build_value_summary_markdown(decisions), encoding="utf-8")
    return {
        "bets": bets_path,
        "summary": summary_path,
        "bankroll": bankroll_path,
    }


def build_value_summary(decisions: pd.DataFrame, bankroll_start: float) -> dict[str, Any]:
    if decisions.empty:
        return {
            "rows": 0,
            "bet_count": 0,
            "watch_count": 0,
            "no_bet_count": 0,
            "bankroll_start": float(bankroll_start),
            "final_bankroll": float(bankroll_start),
            "max_drawdown": 0.0,
            "cumulative_profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
        }

    return {
        "rows": int(len(decisions)),
        "bet_count": int((decisions["decision"] == "BET").sum()),
        "watch_count": int((decisions["decision"] == "WATCH").sum()),
        "no_bet_count": int((decisions["decision"] == "NO BET").sum()),
        "bankroll_start": float(bankroll_start),
        "final_bankroll": float(decisions["bankroll"].iloc[-1]) if "bankroll" in decisions else float(bankroll_start),
        "max_drawdown": float(decisions["max_drawdown"].max()) if "max_drawdown" in decisions else 0.0,
        "cumulative_profit": float(decisions["cumulative_profit"].iloc[-1]) if "cumulative_profit" in decisions else 0.0,
        "roi": float(decisions["roi"].iloc[-1]) if "roi" in decisions else 0.0,
        "yield": float(decisions["yield"].iloc[-1]) if "yield" in decisions else 0.0,
    }


def build_value_summary_markdown(decisions: pd.DataFrame) -> str:
    summary = build_value_summary(decisions, bankroll_start=100.0)
    lines = [
        "# Value Betting Summary",
        "",
        "This report summarizes the decision-layer value betting workflow using historical closing odds and the existing model probabilities.",
        "",
        f"- Rows analyzed: {summary['rows']}",
        f"- BET recommendations: {summary['bet_count']}",
        f"- WATCH recommendations: {summary['watch_count']}",
        f"- NO BET recommendations: {summary['no_bet_count']}",
        f"- Final bankroll: {summary['final_bankroll']:.2f}",
        f"- Cumulative profit: {summary['cumulative_profit']:.2f}",
        f"- Max drawdown: {summary['max_drawdown']:.2%}",
    ]
    return "\n".join(lines) + "\n"
