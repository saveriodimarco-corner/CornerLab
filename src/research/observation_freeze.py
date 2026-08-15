from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.bankroll_tracker import BankrollTracker
from src.exceptions import BankrollUnavailableError


SUPPORTED_SERIE_A_MARKETS = ["over_9_5", "under_9_5", "over_10_5", "under_10_5"]
UNSUPPORTED_SERIE_A_MARKETS = ["over_8_5", "under_8_5", "over_11_5", "under_11_5"]
CHECKPOINTS = [50, 100, 200]


def resolve_current_bankroll(base_dir: Path | str, default_bankroll: float = 100.0) -> float:
    """Resolve the current staking bankroll from the canonical settled ledger.

    Falls back to default_bankroll only when no settled bets exist yet; fails
    closed (raises) if settled history exists but the bankroll is corrupt.
    """
    settled_path = Path(base_dir) / "reports" / "paper_trading_settled.csv"
    if not settled_path.exists():
        return float(default_bankroll)
    try:
        settled = pd.read_csv(settled_path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return float(default_bankroll)
    if settled.empty or "bet_result" not in settled.columns or "bankroll_after" not in settled.columns:
        return float(default_bankroll)

    bets_only = settled.loc[settled["bet_result"].astype(str).isin(["WIN", "LOSS"])].copy()
    if bets_only.empty:
        return float(default_bankroll)

    bets_only["settled_timestamp"] = pd.to_datetime(bets_only.get("settled_timestamp"), errors="coerce")
    bets_only = bets_only.sort_values(["settled_timestamp", "fixture_id", "line"], kind="mergesort")
    latest_bankroll = pd.to_numeric(bets_only["bankroll_after"], errors="coerce").iloc[-1]
    if not np.isfinite(latest_bankroll) or latest_bankroll <= 0.0:
        raise BankrollUnavailableError(f"Settled history exists but current bankroll is invalid: {latest_bankroll!r}")
    return float(latest_bankroll)


def build_production_baseline_manifest(base_dir: Path | str | None = None, output_dir: Path | str | None = None) -> dict[str, Any]:
    base_dir = Path(base_dir or Path.cwd())
    output_dir = Path(output_dir or base_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = reports_dir / "production_baseline_serie_a.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    git_commit = _resolve_git_commit(base_dir)
    current_report = _load_csv_if_exists(reports_dir / "paper_trading_current.csv")
    paper_schema_hash = _hash_values(list(current_report.columns))
    live_policy = _load_json_if_exists(base_dir / "recommended_live_configuration.json")
    betting_policy = _load_json_if_exists(reports_dir / "betting_policy.json")
    best_models = _load_json_if_exists(base_dir / "data" / "research" / "best_models.json")

    bundles: dict[str, dict[str, Any]] = {}
    try:
        from src.research.paper_trading import _load_authoritative_models  # local import to avoid cycles

        bundles = _load_authoritative_models(base_dir)
    except FileNotFoundError:
        bundles = {}
    model_artifacts: list[dict[str, Any]] = []
    feature_schema_hashes: dict[str, str] = {}
    target_to_model_map: dict[str, dict[str, Any]] = {}
    for target_name in ["over_9_5", "over_10_5"]:
        registry_key = f"serie_a/{target_name}"
        bundle = bundles.get(registry_key)
        if bundle is None:
            continue
        feature_schema_hash = _hash_values(bundle.get("schema", []))
        model_info = {
            "target_name": target_name,
            "competition": "Serie A",
            "market_side": "OVER",
            "artifact_path": str(bundle["artifact_path"]),
            "artifact_hash": str(bundle["artifact_hash"]),
            "model_name": str(bundle.get("model_version")),
            "feature_schema_hash": feature_schema_hash,
        }
        model_artifacts.append(model_info)
        feature_schema_hashes[target_name] = feature_schema_hash
        target_to_model_map[target_name] = model_info
        target_to_model_map[target_name.replace("over_", "under_")] = {
            **model_info,
            "target_name": target_name.replace("over_", "under_"),
            "market_side": "UNDER",
        }

    manifest = {
		"release_version": "cornerlab-serie-a-v1.1",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "model_artifacts": model_artifacts,
        "feature_schema_hashes": feature_schema_hashes,
        "supported_targets": SUPPORTED_SERIE_A_MARKETS,
        "unsupported_targets": UNSUPPORTED_SERIE_A_MARKETS,
        "supported_market_registry": {
            "Serie A": {
                "status": "OPERATIVO",
                "supported_targets": SUPPORTED_SERIE_A_MARKETS,
                "unsupported_targets": UNSUPPORTED_SERIE_A_MARKETS,
                "model_registry": target_to_model_map,
            },
            "Premier League": {
                "status": "IN PREPARAZIONE",
                "supported_targets": [],
                "unsupported_targets": ["over_8_5", "under_8_5", "over_9_5", "under_9_5", "over_10_5", "under_10_5", "over_11_5", "under_11_5"],
            },
        },
        "decision_thresholds": {
            "minimum_probability": float(live_policy.get("decision_thresholds", {}).get("minimum_probability", 0.6)),
            "minimum_confidence": float(live_policy.get("decision_thresholds", {}).get("minimum_confidence", 60.0)),
            "minimum_ev": float(live_policy.get("decision_thresholds", {}).get("minimum_ev", 0.05)),
            "accept_threshold": float(live_policy.get("confidence_policy", {}).get("accept_threshold", 69.32447766749223)),
        },
        "staking_config": {
            "kelly_cap": float(live_policy.get("decision_thresholds", {}).get("kelly_cap", 0.2)),
            "minimum_odds": float(betting_policy.get("thresholds", {}).get("minimum_odds", 1.5)),
            "bankroll_start": 100.0,
        },
        "betting_policy_thresholds": betting_policy.get("thresholds", {}),
        "current_paper_trading_schema": {
            "columns": list(current_report.columns),
            "schema_hash": paper_schema_hash,
        },
        "current_paper_trading_schema_hash": paper_schema_hash,
        "live_configuration": live_policy,
        "current_best_models": {
            key: {
                "accepted": bool(value.get("accepted")),
                "model_name": value.get("model_name"),
                "primary_metric_name": value.get("primary_metric_name"),
                "primary_metric_value": value.get("primary_metric_value"),
                "baseline_metric_value": value.get("baseline_metric_value"),
            }
            for key, value in best_models.items()
            if key in SUPPORTED_SERIE_A_MARKETS
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def settle_paper_trades(base_dir: Path | str | None = None, output_dir: Path | str | None = None, bankroll_start: float = 100.0) -> dict[str, Any]:
    base_dir = Path(base_dir or Path.cwd())
    output_dir = Path(output_dir or base_dir)
    reports_dir = output_dir / "reports"
    data_dir = output_dir / "data" / "paper_trading"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "paper_trading_current.csv"
    if not report_path.exists():
        payload = _empty_settlement_payload(bankroll_start=bankroll_start)
        _write_settlement_outputs(payload, reports_dir=reports_dir, data_dir=data_dir)
        return payload

    report = pd.read_csv(report_path)
    if report.empty:
        payload = _empty_settlement_payload(bankroll_start=bankroll_start)
        _write_settlement_outputs(payload, reports_dir=reports_dir, data_dir=data_dir)
        return payload

    settled = _build_settled_trades(report=report, base_dir=base_dir, bankroll_start=bankroll_start)
    payload = _build_settlement_payload(settled, bankroll_start=bankroll_start)
    _write_settlement_outputs(payload, reports_dir=reports_dir, data_dir=data_dir)
    _write_checkpoint_reports(payload, reports_dir=reports_dir)
    return payload


def _build_settled_trades(report: pd.DataFrame, base_dir: Path, bankroll_start: float) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()

    report = report.copy()
    report["competition"] = report.get("competition", "Serie A").astype(str)
    report["decision"] = report.get("decision", "").astype(str)
    report["target_name"] = report.get("target_name", "").astype(str)
    report["side"] = report.get("side", "").astype(str)
    report["line"] = report.get("line", "").astype(str)
    report["bookmaker"] = report.get("bookmaker", "").astype(str)
    report["decision_timestamp"] = pd.to_datetime(report.get("decision_timestamp", pd.NaT), errors="coerce")
    report["decision_timestamp"] = report["decision_timestamp"].fillna(pd.to_datetime(report.get("snapshot_timestamp", pd.NaT), errors="coerce"))
    report["decision_timestamp"] = report["decision_timestamp"].fillna(pd.Timestamp.utcnow())

    results = _load_collector_results(base_dir)
    if results.empty:
        return pd.DataFrame()

    supported_mask = report["competition"].eq("Serie A") & report["decision"].eq("PLAY")
    if "market_support_status" in report.columns:
        supported_mask &= report["market_support_status"].astype(str).eq("SUPPORTED")
    play_rows = report.loc[supported_mask].copy()
    if play_rows.empty:
        return pd.DataFrame()

    merged = play_rows.merge(results, on="fixture_id", how="inner", suffixes=("", "_result"))
    if merged.empty:
        return pd.DataFrame()

    settled_rows: list[dict[str, Any]] = []
    tracker = BankrollTracker(bankroll_start=bankroll_start, bankroll=bankroll_start)
    merged = merged.sort_values(["decision_timestamp", "fixture_id", "line", "bookmaker"], kind="mergesort").reset_index(drop=True)
    for _, row in merged.iterrows():
        outcome = _resolve_bet_outcome(row)
        stake = float(row.get("stake", row.get("recommended_stake", 0.0)) or 0.0)
        odds_at_decision = float(row.get("odds_at_decision", row.get("opening_odds", row.get("closing_odds", np.nan))))
        bankroll_before = float(tracker.bankroll)
        if outcome == 1:
            profit_loss = float(stake * (odds_at_decision - 1.0))
        elif outcome == 0:
            profit_loss = float(-stake)
        else:
            profit_loss = 0.0
            stake = 0.0
        bankroll_after = float(bankroll_before + profit_loss)
        tracker.bankroll = bankroll_after
        tracker.cumulative_profit = bankroll_after - tracker.bankroll_start
        tracker.peak_bankroll = max(tracker.peak_bankroll, bankroll_after)
        tracker.max_drawdown = max(tracker.max_drawdown, (tracker.peak_bankroll - bankroll_after) / tracker.peak_bankroll if tracker.peak_bankroll else 0.0)
        tracker.total_staked += float(stake)
        result_label = "WIN" if outcome == 1 else "LOSS" if outcome == 0 else "VOID"
        closing_odds = float(row.get("closing_odds", np.nan))
        implied_probability_at_decision = float(1.0 / odds_at_decision) if np.isfinite(odds_at_decision) and odds_at_decision > 0.0 else np.nan
        closing_implied_probability = float(1.0 / closing_odds) if np.isfinite(closing_odds) and closing_odds > 0.0 else np.nan
        clv = float(closing_implied_probability - implied_probability_at_decision) if np.isfinite(implied_probability_at_decision) and np.isfinite(closing_implied_probability) else np.nan
        settled_rows.append(
            {
                "run_id": row.get("run_id"),
                "decision_timestamp": _format_timestamp(row.get("decision_timestamp")),
                "fixture_id": int(row.get("fixture_id")),
                "provider_event_id": row.get("provider_event_id"),
                "competition": row.get("competition"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "kickoff": row.get("kickoff_utc"),
                "line": row.get("line"),
                "side": row.get("side"),
                "bookmaker": row.get("bookmaker"),
                "odds_at_decision": odds_at_decision,
                "closing_odds": closing_odds,
                "predicted_probability": float(row.get("predicted_probability", np.nan)),
                "fair_odds": float(row.get("fair_odds", np.nan)),
                "market_implied_probability": float(row.get("market_implied_probability", np.nan)),
                "implied_probability_at_decision": implied_probability_at_decision,
                "closing_implied_probability": closing_implied_probability,
                "CLV": clv,
                "edge": float(row.get("edge", np.nan)),
                "EV": float(row.get("ev", np.nan)),
                "confidence": float(row.get("decision_confidence_score", np.nan)),
                "quality_tier": row.get("quality_tier", row.get("Qualità", "-")),
                "recommended_stake": float(row.get("recommended_stake", row.get("stake", np.nan))),
                "model_artifact": row.get("model_artifact"),
                "model_hash": row.get("model_hash"),
                "feature_schema_hash": row.get("feature_schema_hash"),
                "home_corners": int(row.get("home_corners_result", row.get("home_corners", 0)) or 0),
                "away_corners": int(row.get("away_corners_result", row.get("away_corners", 0)) or 0),
                "total_corners": int(row.get("total_corners_result", row.get("total_corners", 0)) or 0),
                "bet_result": result_label,
                "stake": stake,
                "profit_loss": profit_loss,
                "bankroll_before": bankroll_before,
                "bankroll_after": bankroll_after,
                "settled_timestamp": row.get("settled_at"),
                "target_name": row.get("target_name"),
            }
        )

    return pd.DataFrame(settled_rows)


def _load_collector_results(base_dir: Path) -> pd.DataFrame:
    db_path = base_dir / "data" / "collector.sqlite"
    if not db_path.exists():
        return pd.DataFrame(columns=["fixture_id", "home_corners", "away_corners", "total_corners", "settled_at"])
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query("SELECT fixture_id, home_corners, away_corners, total_corners, settled_at FROM collector_results", connection)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["fixture_id"] = pd.to_numeric(frame["fixture_id"], errors="coerce").astype("Int64")
    frame["home_corners_result"] = pd.to_numeric(frame["home_corners"], errors="coerce")
    frame["away_corners_result"] = pd.to_numeric(frame["away_corners"], errors="coerce")
    frame["total_corners_result"] = pd.to_numeric(frame["total_corners"], errors="coerce")
    return frame


def _resolve_bet_outcome(row: pd.Series) -> int:
    total_corners = row.get("total_corners_result")
    line = str(row.get("line", "")).strip()
    side = str(row.get("side", "")).upper()
    try:
        total_value = float(total_corners)
        line_value = float(line)
    except (TypeError, ValueError):
        return -1
    if side == "OVER":
        return 1 if total_value > line_value else 0
    if side == "UNDER":
        return 1 if total_value < line_value else 0
    return -1


def _build_settlement_payload(settled: pd.DataFrame, bankroll_start: float) -> dict[str, Any]:
    if settled.empty:
        return _empty_settlement_payload(bankroll_start=bankroll_start)

    settled = settled.copy()
    settled["settled_timestamp"] = pd.to_datetime(settled["settled_timestamp"], errors="coerce")
    settled = settled.sort_values(["settled_timestamp", "fixture_id", "line"], kind="mergesort").reset_index(drop=True)
    bets_only = settled.loc[settled["bet_result"].isin(["WIN", "LOSS"])].copy()

    bankroll_tracker = BankrollTracker(bankroll_start=bankroll_start, bankroll=bankroll_start)
    bankroll_curve: list[dict[str, Any]] = []
    for _, row in bets_only.iterrows():
        odds = float(row.get("odds_at_decision", np.nan))
        stake = float(row.get("stake", 0.0) or 0.0)
        outcome = 1 if row["bet_result"] == "WIN" else 0
        metrics = bankroll_tracker.update(stake=stake, outcome=outcome, odds=odds)
        bankroll_curve.append(
            {
                "settled_timestamp": _format_timestamp(row.get("settled_timestamp")),
                "fixture_id": int(row.get("fixture_id")),
                "market": f"{row.get('side')} {row.get('line')}",
                "stake": stake,
                "bankroll_after": float(metrics["bankroll"]),
                "cumulative_profit": float(metrics["cumulative_profit"]),
                "roi": float(metrics["roi"]),
                "yield": float(metrics["yield"]),
                "max_drawdown": float(metrics["max_drawdown"]),
            }
        )

    total_bets = int(len(bets_only))
    wins = int((bets_only["bet_result"] == "WIN").sum())
    losses = int((bets_only["bet_result"] == "LOSS").sum())
    total_stake = float(bets_only["stake"].sum()) if total_bets else 0.0
    profit_loss = float(bets_only["profit_loss"].sum()) if total_bets else 0.0
    hit_rate = float(wins / total_bets) if total_bets else 0.0
    roi = float(profit_loss / bankroll_start) if bankroll_start else 0.0
    yield_value = float(profit_loss / total_stake) if total_stake else 0.0
    average_odds = float(bets_only["odds_at_decision"].mean()) if total_bets else float("nan")
    average_ev = float(bets_only["EV"].mean()) if total_bets and "EV" in bets_only.columns else float("nan")
    average_confidence = float(bets_only["confidence"].mean()) if total_bets else float("nan")
    average_clv = float(bets_only["CLV"].mean()) if "CLV" in bets_only.columns and bets_only["CLV"].notna().any() else float("nan")
    max_drawdown = float(max([item["max_drawdown"] for item in bankroll_curve], default=0.0))

    summary = {
        "bankroll_start": float(bankroll_start),
        "final_bankroll": float(bankroll_tracker.bankroll),
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit_rate,
        "total_stake": total_stake,
        "profit_loss": profit_loss,
        "roi": roi,
        "yield": yield_value,
        "average_odds": average_odds,
        "average_ev": average_ev,
        "average_confidence": average_confidence,
        "average_clv": average_clv,
        "max_drawdown": max_drawdown,
        "bankroll_curve": bankroll_curve,
    }

    market_summary = _group_summary(bets_only, group_column="target_name")
    quality_summary = _group_summary(bets_only, group_column="quality_tier")
    calibration = _build_calibration_summary(bets_only)
    checkpoints = _build_checkpoint_summaries(bets_only, bankroll_start=bankroll_start, calibration=calibration)

    return {
        "summary": summary,
        "markets": market_summary,
        "quality": quality_summary,
        "calibration": calibration,
        "checkpoints": checkpoints,
        "settled_rows": settled.to_dict(orient="records"),
        "bankroll_curve": bankroll_curve,
    }


def _group_summary(frame: pd.DataFrame, group_column: str) -> dict[str, Any]:
    if frame.empty or group_column not in frame.columns:
        return {}
    summary: dict[str, Any] = {}
    for key, group in frame.groupby(frame[group_column].astype(str), dropna=False):
        if not key or key == "nan":
            continue
        summary[str(key)] = {
            "total_bets": int(len(group)),
            "wins": int((group["bet_result"] == "WIN").sum()),
            "losses": int((group["bet_result"] == "LOSS").sum()),
            "hit_rate": float((group["bet_result"] == "WIN").mean()) if len(group) else 0.0,
            "total_stake": float(group["stake"].sum()),
            "profit_loss": float(group["profit_loss"].sum()),
            "roi": float(group["profit_loss"].sum() / 100.0),
            "yield": float(group["profit_loss"].sum() / group["stake"].sum()) if float(group["stake"].sum()) else 0.0,
            "average_odds": float(group["odds_at_decision"].mean()) if group["odds_at_decision"].notna().any() else float("nan"),
            "average_ev": float(group["EV"].mean()) if "EV" in group.columns and group["EV"].notna().any() else float("nan"),
            "average_confidence": float(group["confidence"].mean()) if group["confidence"].notna().any() else float("nan"),
            "average_clv": float(group["CLV"].mean()) if "CLV" in group.columns and group["CLV"].notna().any() else float("nan"),
        }
    return summary


def _build_calibration_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "predicted_probability" not in frame.columns:
        return {}
    probabilities = pd.to_numeric(frame["predicted_probability"], errors="coerce")
    outcomes = frame["bet_result"].map({"WIN": 1, "LOSS": 0}).astype(float)
    valid = probabilities.notna() & outcomes.notna()
    if not valid.any():
        return {}
    frame = frame.loc[valid].copy()
    frame["predicted_probability"] = probabilities.loc[valid]
    frame["outcome"] = outcomes.loc[valid]
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"]
    frame["calibration_bucket"] = pd.cut(frame["predicted_probability"], bins=bins, labels=labels, include_lowest=True, right=True)
    calibration: dict[str, Any] = {}
    for bucket in labels:
        bucket_frame = frame.loc[frame["calibration_bucket"] == bucket]
        if bucket_frame.empty:
            calibration[bucket] = {"count": 0, "hit_rate": 0.0, "average_probability": float("nan")}
            continue
        calibration[bucket] = {
            "count": int(len(bucket_frame)),
            "hit_rate": float(bucket_frame["outcome"].mean()),
            "average_probability": float(bucket_frame["predicted_probability"].mean()),
        }
    calibration["brier"] = float(np.mean(np.square(frame["predicted_probability"] - frame["outcome"])))
    return calibration


def _build_checkpoint_summaries(frame: pd.DataFrame, bankroll_start: float, calibration: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    bet_count = int(len(frame))
    for threshold in CHECKPOINTS:
        reached = bet_count >= threshold
        subset = frame.iloc[:threshold].copy() if reached else frame.iloc[:0].copy()
        if subset.empty:
            summaries[str(threshold)] = {
                "reached": False,
                "bet_count": bet_count,
                "roi": 0.0,
                "yield": 0.0,
                "hit_rate": 0.0,
                "profit_loss": 0.0,
                "max_drawdown": 0.0,
                "average_clv": float("nan"),
                "average_ev": float("nan"),
                "average_confidence": float("nan"),
                "average_odds": float("nan"),
                "brier": float("nan"),
                "calibration": calibration,
                "by_market": {},
                "by_quality": {},
            }
            continue
        total_stake = float(subset["stake"].sum()) if "stake" in subset.columns else 0.0
        profit_loss = float(subset["profit_loss"].sum()) if "profit_loss" in subset.columns else 0.0
        summaries[str(threshold)] = {
            "reached": True,
            "bet_count": int(len(subset)),
            "roi": float(profit_loss / bankroll_start) if bankroll_start else 0.0,
            "yield": float(profit_loss / total_stake) if total_stake else 0.0,
            "hit_rate": float((subset["bet_result"] == "WIN").mean()) if len(subset) else 0.0,
            "profit_loss": profit_loss,
            "max_drawdown": float(subset["bankroll_after"].max() - subset["bankroll_after"].min()) if "bankroll_after" in subset.columns and len(subset) else 0.0,
            "average_clv": float(subset["CLV"].mean()) if "CLV" in subset.columns and subset["CLV"].notna().any() else float("nan"),
            "average_ev": float(subset["EV"].mean()) if "EV" in subset.columns and subset["EV"].notna().any() else float("nan"),
            "average_confidence": float(subset["confidence"].mean()) if "confidence" in subset.columns and subset["confidence"].notna().any() else float("nan"),
            "average_odds": float(subset["odds_at_decision"].mean()) if "odds_at_decision" in subset.columns and subset["odds_at_decision"].notna().any() else float("nan"),
            "brier": float(np.mean(np.square(pd.to_numeric(subset.get("predicted_probability", pd.Series(dtype=float)), errors="coerce") - subset["bet_result"].map({"WIN": 1, "LOSS": 0}).astype(float)))) if "predicted_probability" in subset.columns else float("nan"),
            "calibration": calibration,
            "by_market": _group_summary(subset, "target_name"),
            "by_quality": _group_summary(subset, "quality_tier"),
        }
    return summaries


def _write_checkpoint_reports(payload: dict[str, Any], reports_dir: Path) -> None:
    checkpoints_dir = reports_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for threshold, summary in payload.get("checkpoints", {}).items():
        report = {"checkpoint": int(threshold), **summary}
        (checkpoints_dir / f"{threshold}_play_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_settlement_outputs(payload: dict[str, Any], reports_dir: Path, data_dir: Path) -> None:
    settled = pd.DataFrame(payload.get("settled_rows", []))
    settled_path_csv = reports_dir / "paper_trading_settled.csv"
    settled_path_parquet = data_dir / "paper_trading_settled.parquet"
    summary_path = reports_dir / "paper_trading_performance.json"
    summary_md_path = reports_dir / "paper_trading_performance.md"

    if settled.empty:
        settled = pd.DataFrame(columns=[
            "run_id",
            "decision_timestamp",
            "fixture_id",
            "provider_event_id",
            "competition",
            "home_team",
            "away_team",
            "kickoff",
            "line",
            "side",
            "bookmaker",
            "odds_at_decision",
            "closing_odds",
            "predicted_probability",
            "fair_odds",
            "market_implied_probability",
            "implied_probability_at_decision",
            "closing_implied_probability",
            "CLV",
            "edge",
            "EV",
            "confidence",
            "quality_tier",
            "recommended_stake",
            "model_artifact",
            "model_hash",
            "feature_schema_hash",
            "home_corners",
            "away_corners",
            "total_corners",
            "bet_result",
            "stake",
            "profit_loss",
            "bankroll_before",
            "bankroll_after",
            "settled_timestamp",
            "target_name",
        ])

    settled.to_csv(settled_path_csv, index=False)
    settled.to_parquet(settled_path_parquet, index=False)
    summary = payload.get("summary", _empty_settlement_payload(100.0)["summary"])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    summary_md_path.write_text(_build_performance_markdown(payload), encoding="utf-8")
    write_performance_dashboard_artifacts(payload, reports_dir=reports_dir)
    write_model_observation_artifacts(payload, reports_dir=reports_dir)


def write_model_observation_artifacts(payload: dict[str, Any], reports_dir: Path) -> dict[str, Path]:
    """Write descriptive model-performance artifacts from canonical settled trades."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "model_observation_current.json"
    markdown_path = reports_dir / "model_observation_summary.md"
    settled = pd.DataFrame(payload.get("settled_rows", []))
    if settled.empty:
        settled = pd.DataFrame(columns=["target_name", "side", "quality_tier", "bet_result", "stake", "profit_loss", "odds_at_decision", "predicted_probability", "EV", "settled_timestamp"])
    observation = _build_model_observation(settled, bankroll_start=float(payload.get("summary", {}).get("bankroll_start", 100.0)))
    json_path.write_text(json.dumps(observation, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    markdown_path.write_text(_build_model_observation_markdown(observation), encoding="utf-8")
    return {"json": json_path, "summary": markdown_path}


def _build_model_observation(frame: pd.DataFrame, bankroll_start: float) -> dict[str, Any]:
    table = frame.copy()
    table["bet_result"] = table.get("bet_result", pd.Series("", index=table.index)).astype(str).str.upper()
    table["stake"] = pd.to_numeric(table.get("stake", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0)
    table["profit_loss"] = pd.to_numeric(table.get("profit_loss", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0)
    table["predicted_probability"] = pd.to_numeric(table.get("predicted_probability", pd.Series(np.nan, index=table.index)), errors="coerce")
    table["odds_at_decision"] = pd.to_numeric(table.get("odds_at_decision", pd.Series(np.nan, index=table.index)), errors="coerce")
    table["EV"] = pd.to_numeric(table.get("EV", pd.Series(np.nan, index=table.index)), errors="coerce")
    settled_bets = table.loc[table["bet_result"].isin({"WIN", "LOSS"})].copy()
    settled_bets["observed_outcome"] = settled_bets["bet_result"].map({"WIN": 1.0, "LOSS": 0.0})
    supported = {"over_9_5", "under_9_5", "over_10_5", "under_10_5"}
    settled_bets = settled_bets.loc[settled_bets.get("target_name", pd.Series("", index=settled_bets.index)).astype(str).isin(supported)].copy()
    sample_size = int(len(settled_bets))
    sample_label = "INSUFFICIENT SAMPLE" if sample_size < 20 else "VERY EARLY SIGNAL" if sample_size < 50 else "EARLY OBSERVATION" if sample_size < 100 else "MEANINGFUL OBSERVATION"
    total_stake = float(settled_bets["stake"].sum())
    profit_loss = float(settled_bets["profit_loss"].sum())
    brier_rows = settled_bets.loc[settled_bets["predicted_probability"].notna()].copy()
    brier = float(np.mean(np.square(brier_rows["predicted_probability"] - brier_rows["observed_outcome"]))) if not brier_rows.empty else float("nan")
    return {
        "source": "reports/paper_trading_settled.csv",
        "sample_warning": sample_label,
        "settled_scored_bets": sample_size,
        "voids": int((table["bet_result"] == "VOID").sum()),
        "pending": int((table["bet_result"] == "PENDING").sum()),
        "economic": {
            "bets": sample_size,
            "wins": int((settled_bets["bet_result"] == "WIN").sum()),
            "losses": int((settled_bets["bet_result"] == "LOSS").sum()),
            "stake": total_stake,
            "profit_loss": profit_loss,
            "roi": float(profit_loss / bankroll_start) if bankroll_start else 0.0,
            "yield": float(profit_loss / total_stake) if total_stake else 0.0,
            "average_odds": float(settled_bets["odds_at_decision"].mean()) if settled_bets["odds_at_decision"].notna().any() else float("nan"),
            "bankroll": float(bankroll_start + profit_loss),
            "max_drawdown": _observation_drawdown(settled_bets, bankroll_start),
        },
        "brier_score": brier,
        "calibration": _observation_calibration(settled_bets),
        "by_market": _observation_group(settled_bets, "target_name"),
        "by_side": _observation_group(settled_bets, "side"),
        "by_quality": _observation_group(settled_bets, "quality_tier"),
        "by_month": _observation_periods(settled_bets, "M"),
        "by_week": _observation_periods(settled_bets, "W-MON"),
    }


def _observation_drawdown(frame: pd.DataFrame, bankroll_start: float) -> float:
    if frame.empty:
        return 0.0
    cumulative = bankroll_start + frame["profit_loss"].cumsum()
    peak = pd.concat([pd.Series([bankroll_start]), cumulative], ignore_index=True).cummax().iloc[1:]
    return float(((peak - cumulative) / peak.replace(0.0, np.nan)).fillna(0.0).max())


def _observation_group(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return {}
    groups: dict[str, dict[str, Any]] = {}
    for key, group in frame.groupby(frame[column].astype(str), dropna=False):
        if not key or key == "nan":
            continue
        stake = float(group["stake"].sum())
        profit_loss = float(group["profit_loss"].sum())
        probabilities = group["predicted_probability"].dropna()
        groups[str(key)] = {
            "bets": int(len(group)),
            "hit_rate": float((group["bet_result"] == "WIN").mean()),
            "roi": float(profit_loss / 100.0),
            "profit_loss": profit_loss,
            "brier": float(np.mean(np.square(probabilities - group.loc[probabilities.index, "observed_outcome"]))) if not probabilities.empty else float("nan"),
            "average_predicted_probability": float(probabilities.mean()) if not probabilities.empty else float("nan"),
            "average_odds": float(group["odds_at_decision"].mean()) if group["odds_at_decision"].notna().any() else float("nan"),
            "average_ev": float(group["EV"].mean()) if group["EV"].notna().any() else float("nan"),
            "yield": float(profit_loss / stake) if stake else 0.0,
        }
    return groups


def _observation_calibration(frame: pd.DataFrame) -> list[dict[str, Any]]:
    buckets = [(0.50, 0.55, "0.50-0.55"), (0.55, 0.60, "0.55-0.60"), (0.60, 0.65, "0.60-0.65"), (0.65, 0.70, "0.65-0.70"), (0.70, 0.75, "0.70-0.75"), (0.75, float("inf"), "0.75+")]
    rows: list[dict[str, Any]] = []
    for lower, upper, label in buckets:
        probabilities = frame["predicted_probability"]
        mask = probabilities.ge(lower) & (probabilities.lt(upper) if np.isfinite(upper) else probabilities.ge(lower))
        group = frame.loc[mask & probabilities.notna()]
        observed = float(group["observed_outcome"].mean()) if not group.empty else float("nan")
        average = float(group["predicted_probability"].mean()) if not group.empty else float("nan")
        rows.append({"bucket": label, "count": int(len(group)), "average_predicted_probability": average, "observed_win_frequency": observed, "calibration_gap": float(observed - average) if np.isfinite(observed) and np.isfinite(average) else float("nan")})
    return rows


def _observation_periods(frame: pd.DataFrame, frequency: str) -> list[dict[str, Any]]:
    if frame.empty or "settled_timestamp" not in frame.columns:
        return []
    timestamp = pd.to_datetime(frame["settled_timestamp"], errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    for period, group in frame.assign(_period=timestamp.dt.to_period(frequency)).groupby("_period", sort=False):
        if str(period) == "NaT":
            continue
        rows.append({"period": str(period), **_observation_group(group.assign(target_name="all"), "target_name").get("all", {})})
    return rows


def _build_model_observation_markdown(observation: dict[str, Any]) -> str:
    economic = observation["economic"]
    return "\n".join([
        "# Model Observation Summary",
        "",
        f"- Sample: {observation['sample_warning']} ({observation['settled_scored_bets']} settled scored bets)",
        f"- ROI: {float(economic['roi']):.2%}",
        f"- P/L: {float(economic['profit_loss']):.2f}",
        f"- Brier score: {float(observation['brier_score']):.4f}" if np.isfinite(observation["brier_score"]) else "- Brier score: n/a",
    ]) + "\n"


def write_performance_dashboard_artifacts(
    payload: dict[str, Any], reports_dir: Path, now: pd.Timestamp | None = None
) -> dict[str, Path]:
    """Write dashboard-ready observation artifacts from already-settled paper trades."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    dashboard_json_path = reports_dir / "performance_dashboard_current.json"
    dashboard_csv_path = reports_dir / "performance_dashboard_current.csv"
    dashboard_md_path = reports_dir / "performance_dashboard_summary.md"

    rows = pd.DataFrame(payload.get("settled_rows", []))
    if rows.empty:
        rows = pd.DataFrame(columns=["settled_timestamp", "target_name", "side", "quality_tier", "bet_result", "stake", "profit_loss"])
    rows = rows.copy()
    rows["bet_result"] = rows.get("bet_result", pd.Series("", index=rows.index)).astype(str).str.upper()
    rows["settled_timestamp"] = pd.to_datetime(rows.get("settled_timestamp", pd.Series(pd.NaT, index=rows.index)), errors="coerce", utc=True)
    settled = rows.loc[rows["bet_result"].isin({"WIN", "LOSS"})].copy()
    settled = settled.sort_values("settled_timestamp", kind="stable").reset_index(drop=True)
    reference_time = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if reference_time.tzinfo is None:
        reference_time = reference_time.tz_localize("UTC")

    periods = {
        "7d": settled.loc[settled["settled_timestamp"] >= reference_time - pd.Timedelta(days=7)].copy(),
        "current_month": settled.loc[(settled["settled_timestamp"].dt.year == reference_time.year) & (settled["settled_timestamp"].dt.month == reference_time.month)].copy(),
        "season": settled.loc[settled["settled_timestamp"] >= pd.Timestamp(year=reference_time.year if reference_time.month >= 8 else reference_time.year - 1, month=8, day=1, tz="UTC")].copy(),
        "all": settled,
    }
    dashboard = {
        "generated_at": reference_time.isoformat(),
        "summary": payload.get("summary", _empty_settlement_payload(100.0)["summary"]),
        "periods": {name: _dashboard_summary(frame, bankroll_start=float(payload.get("summary", {}).get("bankroll_start", 100.0))) for name, frame in periods.items()},
        "market_breakdown": _group_summary(settled, "target_name"),
        "side_breakdown": _group_summary(settled, "side"),
        "quality_breakdown": _group_summary(settled, "quality_tier"),
        "weekly_report": _dashboard_time_breakdown(settled, "W-MON"),
        "monthly_report": _dashboard_time_breakdown(settled, "M"),
        "settled_bets": int(len(settled)),
        "pending_bets": int((rows["bet_result"] == "PENDING").sum()),
    }
    dashboard_json_path.write_text(json.dumps(dashboard, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    settled.to_csv(dashboard_csv_path, index=False)
    dashboard_md_path.write_text(_build_performance_dashboard_markdown(dashboard), encoding="utf-8")
    return {"json": dashboard_json_path, "csv": dashboard_csv_path, "summary": dashboard_md_path}


def _dashboard_summary(frame: pd.DataFrame, bankroll_start: float) -> dict[str, Any]:
    if frame.empty:
        return {"total_bets": 0, "wins": 0, "losses": 0, "profit_loss": 0.0, "roi": 0.0, "yield": 0.0, "win_rate": 0.0, "final_bankroll": float(bankroll_start), "max_drawdown": 0.0, "longest_winning_streak": 0, "longest_losing_streak": 0}
    frame = frame.copy().sort_values("settled_timestamp", kind="stable")
    frame["stake"] = pd.to_numeric(frame.get("stake", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["profit_loss"] = pd.to_numeric(frame.get("profit_loss", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    bankroll_curve = bankroll_start + frame["profit_loss"].cumsum()
    running_peak = pd.concat([pd.Series([bankroll_start]), bankroll_curve], ignore_index=True).cummax().iloc[1:]
    drawdown = ((running_peak - bankroll_curve) / running_peak.replace(0.0, np.nan)).fillna(0.0)
    results = frame["bet_result"].tolist()
    return {
        "total_bets": int(len(frame)),
        "wins": int((frame["bet_result"] == "WIN").sum()),
        "losses": int((frame["bet_result"] == "LOSS").sum()),
        "profit_loss": float(frame["profit_loss"].sum()),
        "roi": float(frame["profit_loss"].sum() / bankroll_start) if bankroll_start else 0.0,
        "yield": float(frame["profit_loss"].sum() / frame["stake"].sum()) if float(frame["stake"].sum()) else 0.0,
        "win_rate": float((frame["bet_result"] == "WIN").mean()),
        "final_bankroll": float(bankroll_curve.iloc[-1]),
        "max_drawdown": float(drawdown.max()),
        "longest_winning_streak": _longest_streak(results, "WIN"),
        "longest_losing_streak": _longest_streak(results, "LOSS"),
    }


def _longest_streak(results: list[str], outcome: str) -> int:
    longest = 0
    current = 0
    for result in results:
        current = current + 1 if result == outcome else 0
        longest = max(longest, current)
    return longest


def _dashboard_time_breakdown(frame: pd.DataFrame, frequency: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    timestamp = pd.to_datetime(frame["settled_timestamp"], errors="coerce", utc=True)
    records: list[dict[str, Any]] = []
    for period, group in frame.assign(_period=timestamp.dt.to_period(frequency)).groupby("_period", sort=False):
        item = _dashboard_summary(group, bankroll_start=100.0)
        item["period"] = str(period)
        records.append(item)
    return records


def _build_performance_dashboard_markdown(dashboard: dict[str, Any]) -> str:
    summary = dashboard["summary"]
    lines = [
        "# Performance Dashboard",
        "",
        f"- Settled bets: {dashboard['settled_bets']}",
        f"- Pending bets: {dashboard['pending_bets']}",
        f"- ROI: {float(summary.get('roi', 0.0)):.2%}",
        f"- Profit/Loss: {float(summary.get('profit_loss', 0.0)):.2f}",
        f"- Win rate: {float(summary.get('hit_rate', 0.0)):.2%}",
        f"- Max drawdown: {float(summary.get('max_drawdown', 0.0)):.2%}",
        f"- Final bankroll: {float(summary.get('final_bankroll', summary.get('bankroll_start', 100.0))):.2f}",
        "",
        "## Period Views",
    ]
    for name, period_summary in dashboard["periods"].items():
        lines.append(f"- {name}: {int(period_summary['total_bets'])} settled bets, ROI {float(period_summary['roi']):.2%}, P/L {float(period_summary['profit_loss']):.2f}")
    return "\n".join(lines) + "\n"


def _build_performance_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Paper Trading Performance",
        "",
        f"- Scommesse: {int(summary.get('total_bets', 0))}",
        f"- Profitto/Perdita: {float(summary.get('profit_loss', 0.0)):.2f}",
        f"- ROI: {float(summary.get('roi', 0.0)):.2%}",
        f"- Yield: {float(summary.get('yield', 0.0)):.2%}",
        f"- Hit Rate: {float(summary.get('hit_rate', 0.0)):.2%}",
        f"- Drawdown massimo: {float(summary.get('max_drawdown', 0.0)):.2%}",
        f"- Bankroll: {float(summary.get('final_bankroll', summary.get('bankroll_start', 100.0))):.2f}",
    ]
    average_clv = summary.get("average_clv")
    if average_clv is not None and not pd.isna(average_clv):
        lines.append(f"- CLV medio: {float(average_clv):.4f}")
    else:
        lines.append("- CLV medio: n/a")
    return "\n".join(lines) + "\n"


def _empty_settlement_payload(bankroll_start: float) -> dict[str, Any]:
    return {
        "summary": {
            "bankroll_start": float(bankroll_start),
            "final_bankroll": float(bankroll_start),
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": 0.0,
            "total_stake": 0.0,
            "profit_loss": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "average_odds": float("nan"),
            "average_ev": float("nan"),
            "average_confidence": float("nan"),
            "average_clv": float("nan"),
            "max_drawdown": 0.0,
            "bankroll_curve": [],
        },
        "markets": {},
        "quality": {},
        "calibration": {},
        "checkpoints": {
            str(threshold): {
                "reached": False,
                "bet_count": 0,
                "roi": 0.0,
                "yield": 0.0,
                "hit_rate": 0.0,
                "profit_loss": 0.0,
                "max_drawdown": 0.0,
                "average_clv": float("nan"),
                "average_ev": float("nan"),
                "average_confidence": float("nan"),
                "average_odds": float("nan"),
                "brier": float("nan"),
                "calibration": {},
                "by_market": {},
                "by_quality": {},
            }
            for threshold in CHECKPOINTS
        },
        "settled_rows": [],
        "bankroll_curve": [],
    }


def _resolve_git_commit(base_dir: Path) -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_dir, check=True, capture_output=True, text=True)
        return completed.stdout.strip() or None
    except Exception:
        return None


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _hash_values(values: list[Any]) -> str:
    return hashlib.sha256("|".join([str(value) for value in values]).encode("utf-8")).hexdigest()


def _format_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")