from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.bankroll_tracker import BankrollTracker
from src.research.expected_value import expected_value
from src.research.implied_probability import decimal_odds_to_implied_probability
from src.research.kelly import full_kelly, half_kelly, quarter_kelly


def run_policy_grid_search(
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    bankroll_start: float = 100.0,
) -> dict[str, Any]:
    base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    output_dir = Path(output_dir or base_dir)

    input_path = base_dir / "data" / "research" / "confidence_predictions.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing confidence artifact: {input_path}")

    predictions = pd.read_parquet(input_path)
    if predictions.empty:
        raise ValueError("Confidence artifact is empty")

    odds_path = base_dir / "data" / "cornerlab.db"
    if not odds_path.exists():
        raise FileNotFoundError(f"Missing odds database: {odds_path}")

    odds = _load_odds_from_sqlite(odds_path)
    policies = _evaluate_policies(predictions, odds, bankroll_start=bankroll_start)
    output_paths = _write_policy_reports(policies, output_dir=output_dir)
    recommended_policy = policies.sort_values(["roi", "yield", "max_drawdown", "bets"], ascending=[False, False, True, False]).iloc[0].to_dict()
    return {
        "policies": policies,
        "recommended_policy": recommended_policy,
        "output_paths": output_paths,
    }


def _load_odds_from_sqlite(db_path: Path) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = pd.read_sql_query(
            "SELECT match_id, fixture_date, home_team, away_team, bookmaker, market, line, side, opening_odds, closing_odds, odds_timestamp, source, source_fixture_id, is_closing, currency, import_timestamp FROM corner_odds",
            conn,
        )
    finally:
        conn.close()
    return rows


def _evaluate_policies(predictions: pd.DataFrame, odds: pd.DataFrame, bankroll_start: float = 100.0) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions = predictions.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    confidence_thresholds = [60, 65, 70, 75, 80]
    ev_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
    kelly_modes = [
        ("Quarter", quarter_kelly),
        ("Half", half_kelly),
        ("Full", full_kelly),
    ]

    records: list[dict[str, Any]] = []
    for confidence_threshold in confidence_thresholds:
        for ev_threshold in ev_thresholds:
            for kelly_label, kelly_func in kelly_modes:
                tracker = BankrollTracker(bankroll_start=bankroll_start, bankroll=bankroll_start)
                policy_rows: list[dict[str, Any]] = []
                for _, row in predictions.iterrows():
                    match_odds = odds.loc[odds["match_id"].astype(str).str.strip() == str(row["match_id"])] if not odds.empty else pd.DataFrame()
                    if match_odds.empty:
                        continue

                    odds_row = match_odds.iloc[0]
                    model_probability = _extract_model_probability(row, odds_row)
                    decimal_odds = float(odds_row["closing_odds"])
                    implied_probability = decimal_odds_to_implied_probability(decimal_odds)
                    ev = expected_value(model_probability, decimal_odds)
                    confidence = float(np.clip(float(row.get("confidence_score", 0.0)), 0.0, 100.0))

                    decision = "NO BET"
                    stake = 0.0
                    if confidence >= confidence_threshold and ev >= ev_threshold and kelly_func(model_probability, decimal_odds) > 0.0:
                        decision = "BET"
                        stake = max(0.0, tracker.bankroll * kelly_func(model_probability, decimal_odds))
                    elif ev > 0.0 and confidence < confidence_threshold:
                        decision = "WATCH"

                    outcome = _result_outcome(row, odds_row)
                    bankroll_metrics = tracker.update(stake=stake, outcome=outcome, odds=decimal_odds)
                    policy_rows.append(
                        {
                            "match_id": row.get("match_id"),
                            "decision": decision,
                            "stake": stake,
                            "bankroll": bankroll_metrics["bankroll"],
                            "cumulative_profit": bankroll_metrics["cumulative_profit"],
                            "roi": bankroll_metrics["roi"],
                            "yield": bankroll_metrics["yield"],
                            "max_drawdown": bankroll_metrics["max_drawdown"],
                            "ev": ev,
                            "confidence": confidence,
                            "model_probability": model_probability,
                            "implied_probability": implied_probability,
                            "outcome": outcome,
                        }
                    )

                policy_df = pd.DataFrame(policy_rows)
                if policy_df.empty:
                    records.append(_empty_policy_record(confidence_threshold, ev_threshold, kelly_label, bankroll_start))
                    continue

                bet_mask = policy_df["decision"] == "BET"
                watch_mask = policy_df["decision"] == "WATCH"
                no_bet_mask = policy_df["decision"] == "NO BET"
                wins = int(((bet_mask) & (policy_df["outcome"] == 1)).sum())
                win_rate = wins / max(int(bet_mask.sum()), 1)
                records.append(
                    {
                        "confidence_threshold": confidence_threshold,
                        "ev_threshold": ev_threshold,
                        "kelly_fraction": kelly_label,
                        "bets": int(bet_mask.sum()),
                        "watches": int(watch_mask.sum()),
                        "no_bets": int(no_bet_mask.sum()),
                        "win_rate": float(win_rate),
                        "roi": float(policy_df["roi"].iloc[-1]) if "roi" in policy_df else 0.0,
                        "yield": float(policy_df["yield"].iloc[-1]) if "yield" in policy_df else 0.0,
                        "average_ev": float(policy_df["ev"].mean()) if "ev" in policy_df else 0.0,
                        "average_confidence": float(policy_df["confidence"].mean()) if "confidence" in policy_df else 0.0,
                        "profit": float(policy_df["cumulative_profit"].iloc[-1]) if "cumulative_profit" in policy_df else 0.0,
                        "max_drawdown": float(policy_df["max_drawdown"].max()) if "max_drawdown" in policy_df else 0.0,
                        "average_stake": float(policy_df.loc[bet_mask, "stake"].mean()) if bet_mask.any() else 0.0,
                        "final_bankroll": float(policy_df["bankroll"].iloc[-1]) if "bankroll" in policy_df else float(bankroll_start),
                    }
                )

    policies = pd.DataFrame(records)
    if policies.empty:
        return pd.DataFrame(columns=[
            "confidence_threshold",
            "ev_threshold",
            "kelly_fraction",
            "bets",
            "watches",
            "no_bets",
            "win_rate",
            "roi",
            "yield",
            "average_ev",
            "average_confidence",
            "profit",
            "max_drawdown",
            "average_stake",
            "final_bankroll",
        ])

    policies = policies.sort_values(["roi", "yield", "max_drawdown", "bets"], ascending=[False, False, True, False]).reset_index(drop=True)
    policies.insert(0, "rank", range(1, len(policies) + 1))
    return policies


def _empty_policy_record(confidence_threshold: float, ev_threshold: float, kelly_label: str, bankroll_start: float) -> dict[str, Any]:
    return {
        "confidence_threshold": confidence_threshold,
        "ev_threshold": ev_threshold,
        "kelly_fraction": kelly_label,
        "bets": 0,
        "watches": 0,
        "no_bets": 0,
        "win_rate": 0.0,
        "roi": 0.0,
        "yield": 0.0,
        "average_ev": 0.0,
        "average_confidence": 0.0,
        "profit": 0.0,
        "max_drawdown": 0.0,
        "average_stake": 0.0,
        "final_bankroll": float(bankroll_start),
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


def _write_policy_reports(policies: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    grid_path = reports_dir / "policy_grid_search.csv"
    markdown_path = reports_dir / "policy_optimization.md"
    heatmap_path = reports_dir / "policy_heatmap.csv"

    policies.to_csv(grid_path, index=False)
    heatmap_frame = policies[["confidence_threshold", "ev_threshold", "kelly_fraction", "roi", "yield", "profit"]].copy()
    heatmap_frame.to_csv(heatmap_path, index=False)
    markdown_path.write_text(_build_markdown_report(policies), encoding="utf-8")
    return {
        "grid": grid_path,
        "markdown": markdown_path,
        "heatmap": heatmap_path,
    }


def _build_markdown_report(policies: pd.DataFrame) -> str:
    lines = [
        "# Policy Optimization Report",
        "",
        "This report evaluates decision policies over the existing historical validation set without altering the underlying prediction logic.",
        "",
        "## Ranked Policies",
        "",
    ]
    for _, row in policies.iterrows():
        lines.append(
            f"{int(row['rank'])}. Confidence {int(row['confidence_threshold'])} / EV {row['ev_threshold']:.2%} / Kelly {row['kelly_fraction']} -> Bets {int(row['bets'])}, Win Rate {row['win_rate']:.2%}, ROI {row['roi']:.2%}, Yield {row['yield']:.2%}, Profit {row['profit']:.2f}, Max Drawdown {row['max_drawdown']:.2%}, Final Bankroll {row['final_bankroll']:.2f}"
        )
    lines.extend(["", "## Recommended Policy", ""])
    top = policies.iloc[0]
    lines.append(
        f"Recommended production policy: confidence threshold {int(top['confidence_threshold'])}, EV threshold {top['ev_threshold']:.2%}, Kelly {top['kelly_fraction']}."
    )
    lines.append(
        f"Rationale: it achieves the best ranked ROI/yield profile while keeping drawdown controlled and using a modest number of bets."
    )
    return "\n".join(lines) + "\n"
