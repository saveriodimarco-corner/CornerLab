from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_MARKETS = {"OVER 8.5", "OVER 9.5", "OVER 10.5", "OVER 11.5"}

# Production decision policy.
MIN_PREDICTED_PROBABILITY = 0.70
MIN_MARKET_EDGE = 0.05
MIN_EV = 0.08
MIN_CONFIDENCE_SCORE = 60.0
MIN_ODDS = 1.50

# Production staking-risk cap: never stake more than this fraction of current bankroll.
MAX_STAKE_FRACTION = 0.05



def recommended_stake_for_odds(
    predicted_probability: float,
    odds: float,
    bankroll: float,
) -> float:
    """Return Half-Kelly stake capped at the production bankroll fraction."""
    p = float(predicted_probability)
    o = float(odds)
    b = float(bankroll)

    if (
        not np.isfinite(p)
        or not np.isfinite(o)
        or not np.isfinite(b)
        or p <= 0.0
        or p > 1.0
        or o <= 1.0
        or b <= 0.0
    ):
        return 0.0

    kelly_fraction = max(
        0.0,
        (p * (o - 1.0) - (1.0 - p)) / (o - 1.0),
    )
    half_kelly = kelly_fraction * 0.5
    stake_fraction = min(half_kelly, MAX_STAKE_FRACTION)

    return float(max(0.0, b * stake_fraction))


def minimum_acceptable_odds(predicted_probability: float) -> float:
    """Return the minimum decimal odds required to pass all production price gates."""
    p = float(predicted_probability)
    if not np.isfinite(p) or p <= MIN_MARKET_EDGE or p <= 0.0 or p > 1.0:
        return float("nan")

    min_for_odds = MIN_ODDS
    min_for_edge = 1.0 / (p - MIN_MARKET_EDGE)
    min_for_ev = (1.0 + MIN_EV) / p

    return float(max(min_for_odds, min_for_edge, min_for_ev))


def run_decision_engine(
    predictions: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    bankroll: float = 100.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir or Path.cwd())
    if predictions is None:
        raise ValueError("predictions must be provided")

    frame = predictions.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["match_id", "market", "closing_odds", "predicted_probability", "model_confidence"])

    report = build_decision_report(frame, bankroll=bankroll)
    write_outputs(report, output_dir=output_dir)
    return {"report": report, "output_dir": output_dir}


def build_decision_report(predictions: pd.DataFrame, bankroll: float = 100.0) -> pd.DataFrame:
    frame = predictions.copy()
    required_columns = ["match_id", "market", "closing_odds", "predicted_probability", "model_confidence"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    frame = frame.reset_index(drop=True)
    frame["market"] = frame["market"].astype(str).str.strip().str.upper()
    frame["closing_odds"] = pd.to_numeric(frame["closing_odds"], errors="coerce")
    frame["predicted_probability"] = pd.to_numeric(frame["predicted_probability"], errors="coerce")
    frame["model_confidence"] = pd.to_numeric(frame["model_confidence"], errors="coerce")
    frame["predicted_probability"] = frame["predicted_probability"].clip(0.0, 1.0)
    frame["model_confidence"] = frame["model_confidence"].clip(0.0, 1.0)

    report = pd.DataFrame({
        "match_id": frame["match_id"],
        "market": frame["market"],
        "closing_odds": frame["closing_odds"],
        "predicted_probability": frame["predicted_probability"],
        "model_confidence": frame["model_confidence"],
    })

    report["fair_odds"] = np.where(
        frame["predicted_probability"].notna() & frame["closing_odds"].notna(),
        1.0 / np.clip(frame["predicted_probability"], 1e-9, 1.0),
        np.nan,
    )
    report["market_edge"] = np.where(
        frame["predicted_probability"].notna() & frame["closing_odds"].notna(),
        frame["predicted_probability"] - (1.0 / frame["closing_odds"]),
        np.nan,
    )
    report["ev"] = np.where(
        frame["predicted_probability"].notna() & frame["closing_odds"].notna(),
        frame["predicted_probability"] * frame["closing_odds"] - 1.0,
        np.nan,
    )
    report["kelly_fraction"] = np.where(
        frame["predicted_probability"].notna() & frame["closing_odds"].notna(),
        np.maximum(0.0, (frame["predicted_probability"] * (frame["closing_odds"] - 1.0) - (1.0 - frame["predicted_probability"])) / np.maximum(frame["closing_odds"] - 1.0, 1e-9)),
        0.0,
    )
    report["half_kelly"] = report["kelly_fraction"] * 0.5
    report["stake_cap_fraction"] = MAX_STAKE_FRACTION
    report["stake_fraction_used"] = np.minimum(report["half_kelly"], MAX_STAKE_FRACTION)
    report["recommended_stake"] = np.where(
        report["stake_fraction_used"].notna(),
        np.maximum(0.0, bankroll * report["stake_fraction_used"]),
        0.0,
    )
    report["confidence_score"] = frame["model_confidence"] * 100.0

    low_confidence_mask = (
        report["confidence_score"] < MIN_CONFIDENCE_SCORE
    )

    play_mask = (
        ~low_confidence_mask
        & report["closing_odds"].notna()
        & (report["closing_odds"] >= MIN_ODDS)
        & report["predicted_probability"].notna()
        & (report["predicted_probability"] >= MIN_PREDICTED_PROBABILITY)
        & report["market_edge"].notna()
        & (report["market_edge"] >= MIN_MARKET_EDGE)
        & report["ev"].notna()
        & (report["ev"] >= MIN_EV)
    )

    report["decision"] = np.where(
        low_confidence_mask,
        "LOW CONFIDENCE",
        np.where(play_mask, "PLAY", "NO BET"),
    )

    report["decision_reason"] = np.select(
        [
            low_confidence_mask,
            report["closing_odds"].isna(),
            report["closing_odds"] < MIN_ODDS,
            report["predicted_probability"].isna(),
            report["predicted_probability"] < MIN_PREDICTED_PROBABILITY,
            report["market_edge"].isna(),
            report["market_edge"] < MIN_MARKET_EDGE,
            report["ev"].isna(),
            report["ev"] < MIN_EV,
            play_mask,
        ],
        [
            "CONFIDENCE_BELOW_THRESHOLD",
            "ODDS_UNAVAILABLE",
            "ODDS_BELOW_THRESHOLD",
            "PROBABILITY_UNAVAILABLE",
            "PROBABILITY_BELOW_THRESHOLD",
            "EDGE_UNAVAILABLE",
            "EDGE_BELOW_THRESHOLD",
            "EV_UNAVAILABLE",
            "EV_BELOW_THRESHOLD",
            "PRODUCTION_GATES_PASSED",
        ],
        default="NO_BET",
    )

    report["decision"] = report["decision"].astype(str)
    report["decision_reason"] = report["decision_reason"].astype(str)
    return report


def write_outputs(report: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "research"
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "decision_report.parquet"
    csv_path = data_dir / "decision_report.csv"
    json_path = data_dir / "decision_report.json"

    report.to_parquet(parquet_path, index=False)
    report.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report.to_dict(orient="records"), indent=2), encoding="utf-8")
