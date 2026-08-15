from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.operations.prematch_runner import run_prematch


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
REPORT_PATH = BASE_DIR / "reports" / "paper_trading_current.csv"
HISTORY_PATH = BASE_DIR / "data" / "paper_trading" / "run_history.jsonl"
PREMATCH_STATUS_PATH = BASE_DIR / "reports" / "prematch_latest.json"
PERFORMANCE_PATH = BASE_DIR / "reports" / "paper_trading_performance.json"
OPERATIONS_STATUS_PATH = BASE_DIR / "reports" / "operations_status.json"
OPERATIONS_HISTORY_PATH = BASE_DIR / "data" / "operations" / "job_history.jsonl"

UI_LABELS = {
	"history": "Storico",
	"update_prematch": "AGGIORNA PRE-PARTITA",
	"last_update": "Ultimo aggiornamento",
	"system_health": "Stato sistema",
	"odds_provider": "Provider quote",
	"decision_filter": "Filtro decisione",
	"competition": "Competizione",
	"competition_status": "Stato competizione",
	"side": "Esito",
	"line": "Linea",
	"fixture": "Partita",
	"kickoff": "Calcio d’inizio",
	"bookmaker": "Bookmaker",
	"odds": "Quota",
	"model_probability": "Probabilità modello",
	"fair_odds": "Quota equa",
	"market_implied_probability": "Probabilità implicita",
	"edge": "Vantaggio",
	"confidence": "Affidabilità",
	"recommended_stake": "Puntata consigliata",
	"decision": "Decisione",
	"performance": "Performance",
	"bets": "Scommesse",
	"profit_loss": "Profitto/Perdita",
	"roi": "ROI",
	"yield": "Yield",
	"hit_rate": "Hit Rate",
	"max_drawdown": "Drawdown massimo",
	"bankroll": "Bankroll",
}

NO_PLAY_MESSAGE = "Nessuna giocata consigliata al momento."
FULL_VIEW_LABEL = "Vista completa"

DEFAULT_DECISION_FILTER = "SOLO GIOCA"

DECISION_DISPLAY_MAP = {
	"PLAY": "GIOCA",
	"NO BET": "NON GIOCARE",
	"LOW CONFIDENCE": "BASSA AFFIDABILITÀ",
	"MODEL_UNAVAILABLE": "MODELLO NON DISPONIBILE",
}

DECISION_FILTER_OPTIONS = {
	"SOLO GIOCA": "PLAY",
	"TUTTI": "ALL",
}

SIDE_FILTER_OPTIONS = {
	"TUTTI": "ALL",
	"Over": "OVER",
	"Under": "UNDER",
}

COMPETITION_FILTER_OPTIONS = {
	"Tutte": "ALL",
	"Serie A": "Serie A",
	"Premier League": "Premier League",
}

LINE_FILTER_OPTIONS = {
	"TUTTE": "ALL",
	"8.5": "8.5",
	"9.5": "9.5",
	"10.5": "10.5",
	"11.5": "11.5",
}

QUALITY_FILTER_OPTIONS = {
	"TUTTE": "ALL",
	"TOP": "TOP",
	"BUONA": "BUONA",
	"MARGINALE": "MARGINALE",
}

QUALITY_INFO_TEXT = "Qualità indica la forza relativa del segnale tra le giocate già approvate da CornerLab. TOP non significa giocata certa."

QUALITY_RELATIVE_WEIGHTS = {
	"ev": 0.7,
	"confidence": 0.3,
}

DEFAULT_PLAY_POLICY_THRESHOLDS = {
	"minimum_probability": 0.60,
	"minimum_confidence": 60.0,
	"minimum_ev": 0.05,
	"accept_threshold": 75.0,
}

COMPETITION_STATUS_ORDER = ["Serie A", "Premier League"]
PERFORMANCE_PERIOD_OPTIONS = {
	"ULTIMI 7 GIORNI": "7d",
	"MESE CORRENTE": "current_month",
	"MESE PRECEDENTE": "previous_month",
	"STAGIONE": "season",
	"TOTALE": "all",
}
SUPPORTED_MARKET_LABELS = {
	"over_9_5": "OVER 9.5",
	"under_9_5": "UNDER 9.5",
	"over_10_5": "OVER 10.5",
	"under_10_5": "UNDER 10.5",
}
SETTLED_BETS_PATH = BASE_DIR / "reports" / "paper_trading_settled.csv"
PERFORMANCE_DASHBOARD_PATH = BASE_DIR / "reports" / "performance_dashboard_current.json"
PERFORMANCE_DASHBOARD_CSV_PATH = BASE_DIR / "reports" / "performance_dashboard_current.csv"
MODEL_OBSERVATION_PATH = BASE_DIR / "reports" / "model_observation_current.json"


def _autoload_env(env_path: Path | None = None) -> bool:
	target = env_path or ENV_PATH
	if not target.exists():
		return False
	return bool(load_dotenv(target, override=False))


_autoload_env()


def _safe_float(value: Any, default: float = 0.0) -> float:
	try:
		if value is None or pd.isna(value):
			return default
		return float(value)
	except (TypeError, ValueError):
		return default


def _market_label_for_row(row: dict[str, Any]) -> str:
	if row.get("market"):
		market_value = str(row.get("market", "")).strip()
		if market_value:
			return market_value
	target_name = str(row.get("target_name", "")).strip().lower()
	if target_name in SUPPORTED_MARKET_LABELS:
		return SUPPORTED_MARKET_LABELS[target_name]
	line = str(row.get("line", "")).strip()
	side = str(row.get("side", "")).strip().upper()
	if side and line:
		return f"{side} {line}"
	return "ALTRO"


def _read_settled_bets(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()
	try:
		frame = pd.read_csv(path)
	except (pd.errors.EmptyDataError, OSError, ValueError):
		return pd.DataFrame()
	if frame.empty:
		return frame
	frame = frame.copy()
	for column in ["settled_timestamp", "decision_timestamp", "kickoff"]:
		if column in frame.columns:
			frame[column] = pd.to_datetime(frame[column], errors="coerce")
	frame["bet_result"] = frame.get("bet_result", pd.Series("", index=frame.index)).astype(str).str.upper()
	frame["side"] = frame.get("side", pd.Series("", index=frame.index)).astype(str).str.upper()
	frame["quality_tier"] = frame.get("quality_tier", pd.Series("MARGINALE", index=frame.index)).astype(str).str.upper()
	frame["stake"] = pd.to_numeric(frame.get("stake", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
	frame["profit_loss"] = pd.to_numeric(frame.get("profit_loss", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
	frame["odds_at_decision"] = pd.to_numeric(frame.get("odds_at_decision", pd.Series(np.nan, index=frame.index)), errors="coerce")
	frame["EV"] = pd.to_numeric(frame.get("EV", pd.Series(np.nan, index=frame.index)), errors="coerce")
	frame["confidence"] = pd.to_numeric(frame.get("confidence", pd.Series(np.nan, index=frame.index)), errors="coerce")
	frame["market"] = frame.apply(_market_label_for_row, axis=1)
	if "target_name" in frame.columns:
		frame["target_name"] = frame["target_name"].astype(str).str.lower()
	return frame


def _filter_period(frame: pd.DataFrame, period_key: str, month_filter: str | None = None) -> pd.DataFrame:
	if frame.empty:
		return frame
	filtered = frame.copy()
	filtered = filtered.loc[filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper().isin({"WIN", "LOSS", "PENDING"}) | filtered.get("bet_result", pd.Series("", index=filtered.index)).isna()].copy()
	if "settled_timestamp" not in filtered.columns:
		return filtered
	filtered = filtered.loc[filtered["settled_timestamp"].notna()].copy()
	if filtered.empty:
		return filtered
	if period_key == "7d":
		cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=7)
		filtered = filtered.loc[filtered["settled_timestamp"] >= cutoff].copy()
	elif period_key == "current_month":
		now = pd.Timestamp.utcnow()
		filtered = filtered.loc[(filtered["settled_timestamp"].dt.year == now.year) & (filtered["settled_timestamp"].dt.month == now.month)].copy()
	elif period_key == "previous_month":
		now = pd.Timestamp.utcnow()
		month_start = pd.Timestamp(year=now.year if now.month > 1 else now.year - 1, month=now.month - 1 if now.month > 1 else 12, day=1)
		month_end = month_start + pd.offsets.MonthEnd(1)
		filtered = filtered.loc[(filtered["settled_timestamp"] >= month_start) & (filtered["settled_timestamp"] < month_end)].copy()
	elif period_key == "season":
		now = pd.Timestamp.utcnow()
		season_start = pd.Timestamp(year=now.year if now.month >= 8 else now.year - 1, month=8, day=1)
		filtered = filtered.loc[filtered["settled_timestamp"] >= season_start].copy()
	elif period_key == "all":
		pass
	if month_filter:
		filtered = filtered.loc[filtered["settled_timestamp"].dt.to_period("M").astype(str) == month_filter].copy()
	return filtered.sort_values("settled_timestamp", ascending=False, kind="stable").reset_index(drop=True)


def _calculate_performance_summary(frame: pd.DataFrame, bankroll_start: float = 100.0) -> dict[str, Any]:
	if frame.empty:
		return {
			"total_bets": 0,
			"wins": 0,
			"losses": 0,
			"pending": 0,
			"win_rate": 0.0,
			"stake_total": 0.0,
			"profit_loss": 0.0,
			"roi": 0.0,
			"yield": 0.0,
			"average_odds": float("nan"),
			"bankroll_start": float(bankroll_start),
			"bankroll_end": float(bankroll_start),
			"bankroll_change": 0.0,
			"bankroll_change_pct": 0.0,
			"max_drawdown": 0.0,
			"longest_winning_streak": 0,
			"longest_losing_streak": 0,
			"last_settlement": None,
		}

	settled = frame.copy()
	settled["bet_result"] = settled.get("bet_result", pd.Series("", index=settled.index)).astype(str).str.upper()
	settled["stake"] = pd.to_numeric(settled.get("stake", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)
	settled["profit_loss"] = pd.to_numeric(settled.get("profit_loss", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)
	settled["odds_at_decision"] = pd.to_numeric(settled.get("odds_at_decision", pd.Series(np.nan, index=settled.index)), errors="coerce")
	settled["EV"] = pd.to_numeric(settled.get("EV", pd.Series(np.nan, index=settled.index)), errors="coerce")
	settled["confidence"] = pd.to_numeric(settled.get("confidence", pd.Series(np.nan, index=settled.index)), errors="coerce")
	settled["decision_order"] = settled.get("settled_timestamp", pd.to_datetime(pd.Series([None] * len(settled), index=settled.index))).map(pd.Timestamp)
	settled = settled.loc[settled["bet_result"].isin({"WIN", "LOSS", "PENDING"})].copy()
	settled = settled.sort_values(["settled_timestamp", "decision_order"], ascending=[False, False], kind="stable").reset_index(drop=True)
	settled_bets = settled.loc[settled["bet_result"].isin({"WIN", "LOSS"})].copy()
	pending = int((settled["bet_result"] == "PENDING").sum())
	if settled_bets.empty:
		bankroll_end = float(bankroll_start)
		profit_loss = 0.0
		stake_total = 0.0
		roi = 0.0
		yield_value = 0.0
		win_rate = 0.0
		wins = 0
		losses = 0
		max_drawdown = 0.0
		longest_winning_streak = 0
		longest_losing_streak = 0
	else:
		wins = int((settled_bets["bet_result"] == "WIN").sum())
		losses = int((settled_bets["bet_result"] == "LOSS").sum())
		stake_total = float(settled_bets["stake"].sum())
		profit_loss = float(settled_bets["profit_loss"].sum())
		roi = (profit_loss / bankroll_start * 100.0) if bankroll_start else 0.0
		yield_value = (profit_loss / stake_total * 100.0) if stake_total else 0.0
		win_rate = (wins / len(settled_bets)) if len(settled_bets) else 0.0
		bankroll_end = float(bankroll_start + profit_loss)
		if "bankroll_after" in settled_bets.columns and settled_bets["bankroll_after"].notna().any():
			chronological = settled_bets.copy()
			if "settled_timestamp" in chronological.columns:
				chronological["settled_timestamp"] = pd.to_datetime(chronological["settled_timestamp"], errors="coerce")
				chronological = chronological.loc[chronological["settled_timestamp"].notna()].sort_values("settled_timestamp", ascending=True, kind="stable")
			bankroll_after_values = pd.to_numeric(chronological.get("bankroll_after", pd.Series(dtype=float)), errors="coerce")
			bankroll_after_values = bankroll_after_values.loc[bankroll_after_values.notna()]
			if not bankroll_after_values.empty:
				bankroll_end = float(bankroll_after_values.iloc[-1])
		sequence = settled_bets["bet_result"].astype(str).str.upper().tolist()
		w_streak = 0
		l_streak = 0
		longest_w = 0
		longest_l = 0
		for item in sequence:
			if item == "WIN":
				w_streak += 1
				l_streak = 0
				longest_w = max(longest_w, w_streak)
			elif item == "LOSS":
				l_streak += 1
				w_streak = 0
				longest_l = max(longest_l, l_streak)
		longest_winning_streak = longest_w
		longest_losing_streak = longest_l
		net_bankroll = pd.to_numeric(settled_bets.get("bankroll_after", pd.Series(bankroll_start, index=settled_bets.index)), errors="coerce").fillna(bankroll_start)
		if len(net_bankroll):
			peak = float(net_bankroll.max())
			if peak > 0:
				max_drawdown = max((peak - net_bankroll.min()) / peak, 0.0) * 100.0 if len(net_bankroll) else 0.0
			else:
				max_drawdown = 0.0
		else:
			max_drawdown = 0.0

	bankroll_change = bankroll_end - bankroll_start
	bankroll_change_pct = (bankroll_change / bankroll_start * 100.0) if bankroll_start else 0.0
	last_settlement = None
	if "settled_timestamp" in settled.columns and settled["settled_timestamp"].notna().any():
		last_settlement = pd.to_datetime(settled["settled_timestamp"], errors="coerce").dropna().max()
		if pd.notna(last_settlement):
			last_settlement = last_settlement.isoformat()
	return {
		"total_bets": int(len(settled_bets)),
		"wins": wins,
		"losses": losses,
		"pending": pending,
		"win_rate": win_rate,
		"stake_total": float(stake_total),
		"profit_loss": float(profit_loss),
		"roi": float(roi),
		"yield": float(yield_value),
		"average_odds": float(settled_bets["odds_at_decision"].mean()) if not settled_bets.empty and settled_bets["odds_at_decision"].notna().any() else float("nan"),
		"bankroll_start": float(bankroll_start),
		"bankroll_end": float(bankroll_end),
		"bankroll_change": float(bankroll_change),
		"bankroll_change_pct": float(bankroll_change_pct),
		"max_drawdown": float(max_drawdown),
		"longest_winning_streak": int(longest_winning_streak),
		"longest_losing_streak": int(longest_losing_streak),
		"last_settlement": last_settlement,
	}


def _build_market_breakdown(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
	result: dict[str, dict[str, Any]] = {}
	if frame.empty:
		return result
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	for key, label in SUPPORTED_MARKET_LABELS.items():
		key_rows = filtered.loc[filtered["target_name"].astype(str).str.lower() == key].copy() if "target_name" in filtered.columns else filtered.loc[filtered["market"].astype(str).str.upper() == label.upper()].copy()
		if key_rows.empty:
			continue
		wins = int((key_rows["bet_result"] == "WIN").sum())
		losses = int((key_rows["bet_result"] == "LOSS").sum())
		stake_total = float(key_rows["stake"].sum())
		profit_loss = float(key_rows["profit_loss"].sum())
		result[label] = {
			"bets": int(len(key_rows)),
			"wins": wins,
			"losses": losses,
			"win_rate": float((wins / len(key_rows)) if len(key_rows) else 0.0),
			"stake": stake_total,
			"profit_loss": profit_loss,
			"roi": float((profit_loss / stake_total * 100.0) if stake_total else 0.0),
			"average_odds": float(key_rows["odds_at_decision"].mean()) if key_rows["odds_at_decision"].notna().any() else float("nan"),
			"average_ev": float(key_rows["EV"].mean()) if key_rows["EV"].notna().any() else float("nan"),
			"average_confidence": float(key_rows["confidence"].mean()) if key_rows["confidence"].notna().any() else float("nan"),
		}
	return result


def _build_side_breakdown(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
	if frame.empty:
		return {}
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	bands: dict[str, dict[str, Any]] = {}
	for side_name in ["OVER", "UNDER"]:
		rows = filtered.loc[filtered.get("side", pd.Series("", index=filtered.index)).astype(str).str.upper() == side_name].copy()
		if rows.empty:
			continue
		wins = int((rows["bet_result"] == "WIN").sum())
		losses = int((rows["bet_result"] == "LOSS").sum())
		stake_total = float(rows["stake"].sum())
		profit_loss = float(rows["profit_loss"].sum())
		bands[side_name] = {
			"bets": int(len(rows)),
			"wins": wins,
			"losses": losses,
			"win_rate": float((wins / len(rows)) if len(rows) else 0.0),
			"profit_loss": profit_loss,
			"roi": float((profit_loss / stake_total * 100.0) if stake_total else 0.0),
			"average_odds": float(rows["odds_at_decision"].mean()) if rows["odds_at_decision"].notna().any() else float("nan"),
		}
	return bands


def _build_quality_breakdown(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
	if frame.empty:
		return {}
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	bands: dict[str, dict[str, Any]] = {}
	for label in ["TOP", "BUONA", "MARGINALE"]:
		rows = filtered.loc[filtered.get("quality_tier", pd.Series("", index=filtered.index)).astype(str).str.upper() == label].copy()
		if rows.empty:
			continue
		wins = int((rows["bet_result"] == "WIN").sum())
		stake_total = float(rows["stake"].sum())
		profit_loss = float(rows["profit_loss"].sum())
		bands[label] = {
			"bets": int(len(rows)),
			"wins": wins,
			"losses": int((rows["bet_result"] == "LOSS").sum()),
			"win_rate": float((wins / len(rows)) if len(rows) else 0.0),
			"profit_loss": profit_loss,
			"roi": float((profit_loss / stake_total * 100.0) if stake_total else 0.0),
			"average_odds": float(rows["odds_at_decision"].mean()) if rows["odds_at_decision"].notna().any() else float("nan"),
			"average_ev": float(rows["EV"].mean()) if rows["EV"].notna().any() else float("nan"),
			"average_confidence": float(rows["confidence"].mean()) if rows["confidence"].notna().any() else float("nan"),
		}
	return bands


def _build_monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return pd.DataFrame(columns=["month", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi", "ending_bankroll"])
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	if filtered.empty:
		return pd.DataFrame(columns=["month", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi", "ending_bankroll"])
	filtered["month"] = pd.to_datetime(filtered.get("settled_timestamp", pd.Series(pd.NaT, index=filtered.index)), errors="coerce").dt.to_period("M").astype(str)
	monthly = filtered.groupby("month", dropna=False).agg(
		bets=("bet_result", "size"),
		wins=("bet_result", lambda series: int((series == "WIN").sum())),
		losses=("bet_result", lambda series: int((series == "LOSS").sum())),
		stake=("stake", "sum"),
		profit_loss=("profit_loss", "sum"),
		ending_bankroll=("bankroll_after", lambda series: float(pd.to_numeric(series, errors="coerce").dropna().iloc[-1]) if pd.to_numeric(series, errors="coerce").notna().any() else 0.0),
	).reset_index()
	monthly["win_rate"] = np.where(monthly["bets"] > 0, (monthly["wins"] / monthly["bets"]), 0.0)
	monthly["roi"] = np.where(monthly["stake"] > 0, (monthly["profit_loss"] / monthly["stake"]) * 100.0, 0.0)
	monthly = monthly.sort_values("month", ascending=False, kind="stable").reset_index(drop=True)
	monthly = monthly.rename(columns={"month": "month"})
	return monthly


def _build_weekly_summary(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return pd.DataFrame(columns=["week_start", "week_end", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi"])
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	if filtered.empty:
		return pd.DataFrame(columns=["week_start", "week_end", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi"])
	filtered["settled_timestamp"] = pd.to_datetime(filtered.get("settled_timestamp", pd.Series(pd.NaT, index=filtered.index)), errors="coerce")
	filtered = filtered.loc[filtered["settled_timestamp"].notna()].copy()
	def week_key(value):
		if pd.isna(value):
			return ""
		iso = value.isocalendar()
		return f"{iso[0]}-W{int(iso[1]):02d}"
	filtered["week_key"] = filtered["settled_timestamp"].map(week_key)
	filtered["week_start"] = filtered["settled_timestamp"].map(lambda ts: ts.to_period("W-MON").start_time.strftime("%Y-W%W") if pd.notna(ts) else "")
	filtered["week_end"] = filtered["settled_timestamp"].map(lambda ts: ts.to_period("W-SUN").end_time.strftime("%Y-W%W") if pd.notna(ts) else "")
	weekly = filtered.groupby(["week_start", "week_end"], dropna=False).agg(
		bets=("bet_result", "size"),
		wins=("bet_result", lambda series: int((series == "WIN").sum())),
		losses=("bet_result", lambda series: int((series == "LOSS").sum())),
		stake=("stake", "sum"),
		profit_loss=("profit_loss", "sum"),
	).reset_index()
	weekly["win_rate"] = np.where(weekly["bets"] > 0, (weekly["wins"] / weekly["bets"]), 0.0)
	weekly["roi"] = np.where(weekly["stake"] > 0, (weekly["profit_loss"] / weekly["stake"]) * 100.0, 0.0)
	weekly = weekly.sort_values("week_start", ascending=False, kind="stable").reset_index(drop=True)
	return weekly


def _build_bankroll_curve(frame: pd.DataFrame, bankroll_start: float = 100.0) -> pd.DataFrame:
	if frame.empty:
		return pd.DataFrame(columns=["date", "bankroll"])
	filtered = frame.copy()
	filtered["bet_result"] = filtered.get("bet_result", pd.Series("", index=filtered.index)).astype(str).str.upper()
	filtered = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
	if filtered.empty:
		return pd.DataFrame({"date": [pd.Timestamp.utcnow()], "bankroll": [float(bankroll_start)]})
	filtered["settled_timestamp"] = pd.to_datetime(filtered.get("settled_timestamp", pd.Series(pd.NaT, index=filtered.index)), errors="coerce")
	filtered = filtered.loc[filtered["settled_timestamp"].notna()].sort_values("settled_timestamp", ascending=True, kind="stable")
	curve = []
	current_bankroll = float(bankroll_start)
	for _, row in filtered.iterrows():
		current_bankroll += float(row.get("profit_loss", 0.0) or 0.0)
		curve.append({
			"date": row["settled_timestamp"],
			"bankroll": float(current_bankroll),
		})
	if not curve:
		curve.append({"date": pd.Timestamp.utcnow(), "bankroll": float(bankroll_start)})
	return pd.DataFrame(curve)


def _write_performance_dashboard_snapshot(frame: pd.DataFrame, bankroll_start: float = 100.0) -> None:
	if frame.empty:
		summary = _calculate_performance_summary(frame, bankroll_start=bankroll_start)
		payload = {"summary": summary, "markets": {}, "quality": {}, "side": {}, "monthly": [], "weekly": [], "settled_bets": 0, "pending_bets": 0}
		PERFORMANCE_DASHBOARD_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
		pd.DataFrame().to_csv(PERFORMANCE_DASHBOARD_CSV_PATH, index=False)
		return
	frame = frame.copy()
	frame["bet_result"] = frame.get("bet_result", pd.Series("", index=frame.index)).astype(str).str.upper()
	summary = _calculate_performance_summary(frame, bankroll_start=bankroll_start)
	payload = {
		"summary": summary,
		"markets": _build_market_breakdown(frame),
		"quality": _build_quality_breakdown(frame),
		"side": _build_side_breakdown(frame),
		"monthly": _build_monthly_summary(frame).to_dict(orient="records"),
		"weekly": _build_weekly_summary(frame).to_dict(orient="records"),
		"settled_bets": int(summary["total_bets"]),
		"pending_bets": int(summary["pending"]),
	}
	PERFORMANCE_DASHBOARD_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
	frame = frame.loc[frame["bet_result"].isin({"WIN", "LOSS"})].copy()
	if not frame.empty:
		frame.to_csv(PERFORMANCE_DASHBOARD_CSV_PATH, index=False)
	else:
		pd.DataFrame(columns=["settled_timestamp", "fixture_id", "market", "side", "line", "bet_result", "stake", "profit_loss"]).to_csv(PERFORMANCE_DASHBOARD_CSV_PATH, index=False)


def _render_performance_page() -> None:
	st.header("Performance")
	settled = _read_settled_bets(SETTLED_BETS_PATH)
	period_key = st.selectbox("Periodo", list(PERFORMANCE_PERIOD_OPTIONS.keys()), index=0)
	month_options = []
	if not settled.empty and "settled_timestamp" in settled.columns:
		month_options = sorted(pd.to_datetime(settled["settled_timestamp"], errors="coerce").dropna().dt.to_period("M").astype(str).unique(), reverse=True)
	month_filter = None
	if month_options:
		month_filter = st.selectbox("Mese", ["TUTTI"] + month_options, index=0)
		if month_filter == "TUTTI":
			month_filter = None
	filtered = _filter_period(settled, PERFORMANCE_PERIOD_OPTIONS[period_key], month_filter=month_filter)
	if filtered.empty:
		st.info("Nessuna scommessa ancora chiusa nel periodo selezionato.")
		return
	summary = _calculate_performance_summary(filtered, bankroll_start=100.0)
	total_bets = int(summary["total_bets"])
	pending_bets = int(summary["pending"])
	last_settlement = summary.get("last_settlement") or "-"
	st.caption(f"SETTLED BETS = {total_bets}   •   PENDING BETS = {pending_bets}   •   LAST SETTLEMENT = {last_settlement}")
	_write_performance_dashboard_snapshot(filtered, bankroll_start=100.0)
	kpi_1, kpi_2, kpi_3 = st.columns(3)
	kpi_1.metric("ROI", f"{summary['roi']:.1f}%")
	kpi_2.metric("PROFITTO", f"€{summary['profit_loss']:+.2f}")
	kpi_3.metric("BANKROLL", f"€{summary['bankroll_end']:.2f}")
	kpi_4, kpi_5, kpi_6 = st.columns(3)
	kpi_4.metric("SCOMMESSE", str(total_bets))
	kpi_5.metric("WIN RATE", f"{summary['win_rate'] * 100.0:.1f}%")
	kpi_6.metric("MAX DD", f"{summary['max_drawdown']:.1f}%")

	curve = _build_bankroll_curve(filtered, bankroll_start=100.0)
	if not curve.empty:
		st.subheader("Bankroll")
		st.line_chart(curve.set_index("date")["bankroll"], use_container_width=True)
	market_breakdown = _build_market_breakdown(filtered)
	if market_breakdown:
		st.subheader("Market performance")
		market_df = pd.DataFrame.from_dict(market_breakdown, orient="index").reset_index().rename(columns={"index": "market"})
		st.dataframe(market_df[["market", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi", "average_odds", "average_ev", "average_confidence"]], use_container_width=True, hide_index=True)
	quality_breakdown = _build_quality_breakdown(filtered)
	if quality_breakdown:
		st.subheader("Quality performance")
		quality_df = pd.DataFrame.from_dict(quality_breakdown, orient="index").reset_index().rename(columns={"index": "quality"})
		st.dataframe(quality_df[["quality", "bets", "win_rate", "profit_loss", "roi", "average_odds", "average_ev", "average_confidence"]], use_container_width=True, hide_index=True)
	side_breakdown = _build_side_breakdown(filtered)
	if side_breakdown:
		st.subheader("Side performance")
		side_df = pd.DataFrame.from_dict(side_breakdown, orient="index").reset_index().rename(columns={"index": "side"})
		st.dataframe(side_df[["side", "bets", "win_rate", "profit_loss", "roi", "average_odds"]], use_container_width=True, hide_index=True)
	monthly = _build_monthly_summary(filtered)
	if not monthly.empty:
		st.subheader("Monthly summary")
		st.dataframe(monthly[["month", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi", "ending_bankroll"]], use_container_width=True, hide_index=True)
	weekly = _build_weekly_summary(filtered)
	if not weekly.empty:
		st.subheader("Weekly summary")
		st.dataframe(weekly[["week_start", "week_end", "bets", "wins", "losses", "win_rate", "stake", "profit_loss", "roi"]], use_container_width=True, hide_index=True)
	model_observation = _read_performance_snapshot(MODEL_OBSERVATION_PATH)
	st.subheader("CALIBRAZIONE MODELLO")
	if not model_observation:
		st.caption("Nessuna osservazione regolata disponibile.")
	else:
		st.caption(str(model_observation.get("sample_warning", "INSUFFICIENT SAMPLE")))
		calibration = pd.DataFrame(model_observation.get("calibration", []))
		if calibration.empty:
			st.caption("Nessuna osservazione regolata disponibile.")
		else:
			st.dataframe(calibration[["bucket", "count", "average_predicted_probability", "observed_win_frequency", "calibration_gap"]], use_container_width=True, hide_index=True)
	with st.expander("Vedi scommesse chiuse"):
		filtered_bets = filtered.loc[filtered["bet_result"].isin({"WIN", "LOSS"})].copy()
		if filtered_bets.empty:
			st.caption("Nessuna scommessa ancora chiusa nel periodo selezionato.")
		else:
			visible = filtered_bets[["settled_timestamp", "market", "side", "line", "odds_at_decision", "stake", "bet_result", "profit_loss", "bankroll_after", "EV", "confidence", "quality_tier"]].copy()
			visible = visible.rename(columns={
				"settled_timestamp": "date",
				"market": "market",
				"side": "side",
				"line": "line",
				"odds_at_decision": "odds_at_decision",
				"stake": "stake",
				"bet_result": "result",
				"profit_loss": "profit_loss",
				"bankroll_after": "bankroll_after",
				"EV": "EV",
				"confidence": "confidence",
				"quality_tier": "quality",
			})
			st.dataframe(visible, use_container_width=True, hide_index=True)


def _load_history_rows(history_path: Path) -> list[dict[str, Any]]:
	if not history_path.exists():
		return []
	rows: list[dict[str, Any]] = []
	for line in history_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			rows.append(json.loads(line))
		except json.JSONDecodeError:
			continue
	return rows


def _render_history() -> None:
	st.header(UI_LABELS["history"])
	rows = _load_history_rows(HISTORY_PATH)
	if not rows:
		st.info("Nessuno storico pre-partita disponibile.")
		return

	history_frame = pd.DataFrame(rows)
	history_frame = history_frame.sort_values("run_timestamp", ascending=False, kind="stable")
	columns = [
		"run_timestamp",
		"fixtures_evaluated",
		"odds_rows",
		"play_count",
		"no_bet_count",
		"average_ev",
		"supported_models",
		"warnings",
	]
	visible_columns = [column for column in columns if column in history_frame.columns]
	st.dataframe(history_frame[visible_columns], use_container_width=True, hide_index=True)

	run_choices = history_frame["run_id"].astype(str).tolist()
	selected_run = st.selectbox("Apri esecuzione precedente", run_choices)
	selected_row = history_frame.loc[history_frame["run_id"].astype(str) == selected_run].iloc[0].to_dict()
	run_csv = Path(str(selected_row.get("run_csv", "")))
	if run_csv.exists():
		run_frame = pd.read_csv(run_csv)
		st.subheader(f"Run {selected_run}")
		st.dataframe(_prepare_dashboard_table(run_frame), use_container_width=True, hide_index=True)
	else:
		st.warning("Il file di dettaglio non è disponibile.")


def _verify_login(password: str) -> bool:
	expected_password = os.getenv("CORNERLAB_APP_PASSWORD", "")
	if not expected_password:
		return False
	return password == expected_password


def _read_report(report_path: Path) -> pd.DataFrame:
	if not report_path.exists():
		return pd.DataFrame()
	return pd.read_csv(report_path)


def _read_prematch_status(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return {}


def _read_performance_snapshot(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return {}


def _read_operations_status(path: Path = OPERATIONS_STATUS_PATH) -> dict[str, Any]:
	return _read_performance_snapshot(path)


def _load_operations_history(path: Path = OPERATIONS_HISTORY_PATH, limit: int = 10) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	rows: list[dict[str, Any]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		try:
			rows.append(json.loads(line))
		except json.JSONDecodeError:
			continue
	return list(reversed(rows[-limit:]))


def _italian_system_status(value: Any) -> str:
	return {
		"HEALTHY": "OPERATIVO",
		"DEGRADED": "DEGRADATO",
		"FAILED": "ERRORE",
		"UNKNOWN": "IN ATTESA",
	}.get(str(value), "IN ATTESA")


def _decision_order(value: str) -> int:
	order = {
		"PLAY": 0,
		"LOW CONFIDENCE": 1,
		"NO BET": 2,
		"MODEL_UNAVAILABLE": 3,
	}
	return order.get(str(value), 99)


def _decision_display(value: str) -> str:
	return DECISION_DISPLAY_MAP.get(str(value), str(value))


def _competition_support_states(frame: pd.DataFrame) -> dict[str, str]:
	states = {competition: "IN PREPARAZIONE" for competition in COMPETITION_STATUS_ORDER}
	if frame.empty or "competition" not in frame.columns:
		return states
	for competition in COMPETITION_STATUS_ORDER:
		rows = frame.loc[frame["competition"].astype(str) == competition]
		if rows.empty:
			continue
		if "market_support_status" in rows.columns and (rows["market_support_status"].astype(str) == "SUPPORTED").any():
			states[competition] = "OPERATIVO"
		elif "decision" in rows.columns and (rows["decision"].astype(str) != "MODEL_UNAVAILABLE").any():
			states[competition] = "PARZIALMENTE SUPPORTATO"
		else:
			states[competition] = "IN PREPARAZIONE"
	return states


def _load_play_policy_thresholds() -> dict[str, float]:
	thresholds = DEFAULT_PLAY_POLICY_THRESHOLDS.copy()

	recommended_path = BASE_DIR / "recommended_live_configuration.json"
	if recommended_path.exists():
		try:
			payload = json.loads(recommended_path.read_text(encoding="utf-8"))
			decision = payload.get("decision_thresholds", {})
			confidence_policy = payload.get("confidence_policy", {})
			if "minimum_probability" in decision:
				thresholds["minimum_probability"] = float(decision["minimum_probability"])
			if "minimum_confidence" in decision:
				thresholds["minimum_confidence"] = float(decision["minimum_confidence"])
			if "minimum_ev" in decision:
				thresholds["minimum_ev"] = float(decision["minimum_ev"])
			if "accept_threshold" in confidence_policy:
				thresholds["accept_threshold"] = float(confidence_policy["accept_threshold"])
		except (json.JSONDecodeError, OSError, ValueError, TypeError):
			pass

	policy_path = BASE_DIR / "reports" / "betting_policy.json"
	if policy_path.exists():
		try:
			payload = json.loads(policy_path.read_text(encoding="utf-8"))
			policy_thresholds = payload.get("thresholds", {})
			if "minimum_probability" in policy_thresholds:
				thresholds["minimum_probability"] = float(policy_thresholds["minimum_probability"])
			if "minimum_confidence" in policy_thresholds:
				thresholds["minimum_confidence"] = float(policy_thresholds["minimum_confidence"])
			if "minimum_ev" in policy_thresholds:
				thresholds["minimum_ev"] = float(policy_thresholds["minimum_ev"])
		except (json.JSONDecodeError, OSError, ValueError, TypeError):
			pass

	return thresholds


def _add_play_quality(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return frame
	table = frame.copy()
	table["Qualità"] = "-"

	play_mask = table["decision"].astype(str) == "PLAY"
	play = table.loc[play_mask].copy()
	if play.empty:
		return table

	ev_series = pd.to_numeric(play.get("ev", pd.Series(index=play.index, dtype=float)), errors="coerce")
	cf_series = pd.to_numeric(play.get("decision_confidence_score", pd.Series(index=play.index, dtype=float)), errors="coerce")
	ev_rank = ev_series.rank(method="dense", pct=True)
	cf_rank = cf_series.rank(method="dense", pct=True)
	combined_score = QUALITY_RELATIVE_WEIGHTS["ev"] * ev_rank + QUALITY_RELATIVE_WEIGHTS["confidence"] * cf_rank
	combined_score = combined_score.fillna(0.0)
	play["quality_score"] = combined_score

	top_threshold = float(combined_score.quantile(2.0 / 3.0))
	good_threshold = float(combined_score.quantile(1.0 / 3.0))
	policy = _load_play_policy_thresholds()
	accept_confidence = float(policy.get("accept_threshold", policy["minimum_confidence"]))
	confidence_median = float(cf_series.median()) if cf_series.notna().any() else np.nan

	is_top_relative = combined_score >= top_threshold
	is_good_relative = combined_score >= good_threshold
	is_top_conf_ok = (cf_series >= confidence_median) if np.isfinite(confidence_median) else pd.Series(False, index=play.index)
	is_clearly_strong_small_sample = is_top_relative & is_top_conf_ok & (cf_series >= accept_confidence)

	play["Qualità"] = "MARGINALE"
	play.loc[is_good_relative, "Qualità"] = "BUONA"
	if len(play) < 6:
		play.loc[is_clearly_strong_small_sample, "Qualità"] = "TOP"
	else:
		play.loc[is_top_relative & is_top_conf_ok, "Qualità"] = "TOP"

	table.loc[play_mask, "Qualità"] = play["Qualità"].to_numpy()
	return table


def _to_percentage(value: Any, scale: float = 100.0, signed: bool = False, digits: int = 1) -> str:
	if value is None or pd.isna(value):
		return "-"
	numeric = float(value) * scale
	if signed:
		return f"{numeric:+.{digits}f}%"
	return f"{numeric:.{digits}f}%"


def _to_decimal(value: Any, digits: int = 2) -> str:
	if value is None or pd.isna(value):
		return "-"
	return f"{float(value):.{digits}f}"


def _prepare_play_cards(frame: pd.DataFrame, side_filter: str, line_filter: str, quality_filter: str, competition_filter: str = "Tutte") -> list[dict[str, Any]]:
	if frame.empty:
		return []

	table = _add_play_quality(frame)
	table = table.loc[table["decision"] == "PLAY"].copy()
	competition_internal = COMPETITION_FILTER_OPTIONS.get(competition_filter, "ALL")
	if competition_internal != "ALL" and "competition" in table.columns:
		table = table.loc[table["competition"].astype(str) == competition_internal]
	quality_internal = QUALITY_FILTER_OPTIONS.get(quality_filter, "ALL")
	if quality_internal != "ALL":
		table = table.loc[table["Qualità"] == quality_internal]
	side_internal = SIDE_FILTER_OPTIONS.get(side_filter, "ALL")
	if side_internal in {"OVER", "UNDER"}:
		table = table.loc[table["side"].astype(str).str.upper() == side_internal]
	line_internal = LINE_FILTER_OPTIONS.get(line_filter, "ALL")
	if line_internal != "ALL":
		table = table.loc[table["line"].astype(str) == line_internal]

	if table.empty:
		return []

	quality_order = {"TOP": 0, "BUONA": 1, "MARGINALE": 2}
	table["quality_rank"] = table["Qualità"].map(quality_order).fillna(99)
	table = table.sort_values(["quality_rank", "ev", "decision_confidence_score"], ascending=[True, False, False], kind="stable")
	cards: list[dict[str, Any]] = []
	for _, row in table.iterrows():
		market_implied_probability = np.nan
		if pd.notna(row.get("closing_odds")) and float(row.get("closing_odds")) > 0.0:
			market_implied_probability = 1.0 / float(row.get("closing_odds"))
		edge = np.nan
		if pd.notna(row.get("predicted_probability")) and pd.notna(market_implied_probability):
			edge = float(row.get("predicted_probability")) - float(market_implied_probability)

		cards.append(
			{
				"partita": f"{row.get('home_team', '')} - {row.get('away_team', '')}",
				"competizione": row.get("competition", "-"),
				"kickoff": row.get("kickoff_utc"),
				"esito": str(row.get("side", "")).upper(),
				"linea": str(row.get("line", "")),
				"bookmaker": row.get("bookmaker"),
				"quota": row.get("closing_odds"),
				"probabilita_modello": row.get("predicted_probability"),
				"valore_atteso": row.get("ev"),
				"affidabilita": row.get("decision_confidence_score"),
				"puntata_consigliata": row.get("recommended_stake", row.get("stake")),
				"decisione": "PLAY",
				"qualita": row.get("Qualità", "-"),
				"quota_equa": row.get("fair_odds"),
				"probabilita_implicita": market_implied_probability,
				"vantaggio": edge,
				"model_target": row.get("target_name"),
				"odds_timestamp": row.get("snapshot_timestamp"),
				"half_kelly_teorico": row.get("half_kelly"),
				"cap_massimo": row.get("stake_cap_fraction"),
				"stake_applicato": row.get("stake_fraction_used"),
			}
		)
	return cards


def _render_play_cards(cards: list[dict[str, Any]]) -> None:
	for card in cards:
		if card.get("qualita") == "TOP":
			st.markdown("<div style='padding:0.35rem 0.6rem;background:#1f7a1f;color:white;border-radius:8px;display:inline-block;font-weight:700;'>TOP</div>", unsafe_allow_html=True)
		elif card.get("qualita") == "BUONA":
			st.markdown("<div style='padding:0.3rem 0.55rem;background:#d98e04;color:white;border-radius:8px;display:inline-block;font-weight:600;'>BUONA</div>", unsafe_allow_html=True)
		else:
			st.markdown("<div style='padding:0.25rem 0.5rem;background:#8a8a8a;color:white;border-radius:8px;display:inline-block;font-weight:600;'>MARGINALE</div>", unsafe_allow_html=True)

		st.markdown(f"### {card['partita']}")
		st.caption(f"{UI_LABELS['competition']}: {card.get('competizione') or '-'}")
		st.caption(f"{UI_LABELS['kickoff']}: {card.get('kickoff') or '-'}")
		st.markdown(f"**{card.get('esito', '-') } {card.get('linea', '-')}**")
		st.markdown(f"{UI_LABELS['bookmaker']}: **{card.get('bookmaker') or '-'}**")

		metric_col1, metric_col2, metric_col3 = st.columns(3)
		metric_col1.metric(UI_LABELS["odds"], _to_decimal(card.get("quota")))
		metric_col2.metric(UI_LABELS["model_probability"], _to_percentage(card.get("probabilita_modello"), scale=100.0, signed=False, digits=1))
		metric_col3.metric("EV", _to_percentage(card.get("valore_atteso"), scale=100.0, signed=True, digits=1))

		minor_col1, minor_col2 = st.columns(2)
		minor_col1.metric(UI_LABELS["confidence"], _to_percentage(card.get("affidabilita"), scale=1.0, signed=False, digits=0))
		minor_col2.metric(UI_LABELS["recommended_stake"], _to_percentage(card.get("puntata_consigliata"), scale=1.0, signed=False, digits=1))

		st.success(DECISION_DISPLAY_MAP.get("PLAY", "GIOCA"))
		with st.expander("Dettagli"):
			st.write(f"{UI_LABELS['fair_odds']}: {_to_decimal(card.get('quota_equa'))}")
			st.write(f"{UI_LABELS['market_implied_probability']}: {_to_percentage(card.get('probabilita_implicita'), scale=100.0, signed=False, digits=1)}")
			st.write(f"{UI_LABELS['edge']}: {_to_percentage(card.get('vantaggio'), scale=100.0, signed=True, digits=1)}")
			if card.get("half_kelly_teorico") is not None:
				st.write(f"Half Kelly teorico: {_to_percentage(card.get('half_kelly_teorico'), scale=100.0, signed=False, digits=1)}")
			if card.get("cap_massimo") is not None:
				st.write(f"Cap massimo: {_to_percentage(card.get('cap_massimo'), scale=100.0, signed=False, digits=1)}")
			if card.get("stake_applicato") is not None:
				st.write(f"Stake applicato: {_to_percentage(card.get('stake_applicato'), scale=100.0, signed=False, digits=1)}")
			if card.get("model_target"):
				st.write(f"model target: {card.get('model_target')}")
			if card.get("odds_timestamp"):
				st.write(f"odds timestamp: {card.get('odds_timestamp')}")
		st.divider()


def _prepare_dashboard_table(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return frame
	table = _add_play_quality(frame)
	table["market implied probability"] = np.where(
		table["closing_odds"].notna() & (table["closing_odds"] > 0.0),
		1.0 / table["closing_odds"],
		np.nan,
	)
	table["edge"] = table["predicted_probability"] - table["market implied probability"]
	table["recommended stake"] = table.get("recommended_stake", table.get("stake", np.nan))
	table["fixture"] = table["home_team"].astype(str) + " vs " + table["away_team"].astype(str)
	table["decision_rank"] = table["decision"].map(_decision_order)
	table["decision_display"] = table["decision"].astype(str).map(_decision_display)
	table = table.sort_values(["decision_rank", "ev"], ascending=[True, False], kind="stable")

	desired_columns = [
		"Qualità",
		"competition",
		"fixture",
		"kickoff_utc",
		"line",
		"side",
		"bookmaker",
		"closing_odds",
		"predicted_probability",
		"fair_odds",
		"market implied probability",
		"edge",
		"ev",
		"decision_confidence_score",
		"recommended stake",
		"decision_display",
		"decision_reason",
	]
	present_columns = [column for column in desired_columns if column in table.columns]
	return table[present_columns].rename(
		columns={
			"Qualità": "Qualità",
			"competition": UI_LABELS["competition"],
			"fixture": UI_LABELS["fixture"],
			"kickoff_utc": UI_LABELS["kickoff"],
			"line": UI_LABELS["line"],
			"side": UI_LABELS["side"],
			"bookmaker": UI_LABELS["bookmaker"],
			"closing_odds": UI_LABELS["odds"],
			"predicted_probability": UI_LABELS["model_probability"],
			"fair_odds": UI_LABELS["fair_odds"],
			"market implied probability": UI_LABELS["market_implied_probability"],
			"edge": UI_LABELS["edge"],
			"decision_confidence_score": UI_LABELS["confidence"],
			"recommended stake": UI_LABELS["recommended_stake"],
			"decision_display": UI_LABELS["decision"],
			"ev": "EV",
		}
	)


def _apply_filters(frame: pd.DataFrame, filter_mode: str, side_filter: str, line_filter: str, quality_filter: str, competition_filter: str = "Tutte") -> pd.DataFrame:
	if frame.empty:
		return frame
	table = _add_play_quality(frame)
	decision_internal = DECISION_FILTER_OPTIONS.get(filter_mode, "ALL")
	if decision_internal != "ALL":
		table = table.loc[table["decision"] == decision_internal]
	competition_internal = COMPETITION_FILTER_OPTIONS.get(competition_filter, "ALL")
	if competition_internal != "ALL" and "competition" in table.columns:
		table = table.loc[table["competition"].astype(str) == competition_internal]
	quality_internal = QUALITY_FILTER_OPTIONS.get(quality_filter, "ALL")
	if quality_internal != "ALL":
		table = table.loc[table["Qualità"] == quality_internal]
	side_internal = SIDE_FILTER_OPTIONS.get(side_filter, "ALL")
	if side_internal in {"OVER", "UNDER"}:
		table = table.loc[table["side"].astype(str).str.upper() == side_internal]
	line_internal = LINE_FILTER_OPTIONS.get(line_filter, "ALL")
	if line_internal != "ALL":
		table = table.loc[table["line"].astype(str) == line_internal]
	return table


def _render_dashboard() -> None:
	st.header("Dashboard")
	st.caption("Paper Trading mode")

	status = _read_prematch_status(PREMATCH_STATUS_PATH)
	last_update = status.get("completed_at") or "not run yet"
	health_ok = status.get("health_ok")
	provider_status = status.get("collector", {}).get("provider_status", "unknown")

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("CornerLab", "Operational")
	col2.metric(UI_LABELS["last_update"], str(last_update))
	col3.metric(UI_LABELS["system_health"], "OK" if health_ok else "CHECK")
	col4.metric(UI_LABELS["odds_provider"], str(provider_status).upper())

	operations = _read_operations_status()
	st.subheader("STATO SISTEMA")
	ops_col1, ops_col2 = st.columns(2)
	ops_col1.metric("Sistema", _italian_system_status(operations.get("system_status")))
	ops_col2.metric("Scheduler", str(operations.get("scheduler_status", "IN ATTESA")).replace("SYSTEMD_TIMER_CONFIGURED", "CONFIGURATO"))
	ops_col3, ops_col4 = st.columns(2)
	ops_col3.metric("Ultimo prematch", str(operations.get("last_prematch_success") or "-"))
	ops_col4.metric("Ultimo settlement", str(operations.get("last_settlement_success") or "-"))
	st.caption(f"Ultimo aggiornamento quote: {operations.get('last_successful_odds_refresh') or '-'}")
	st.caption(f"Ultimo errore: {operations.get('last_error_summary') or '-'}")
	with st.expander("Ultime attività sistema"):
		activity_rows = _load_operations_history()
		if not activity_rows:
			st.caption("Nessuna attività automatica registrata.")
		else:
			for activity in activity_rows:
				st.write(f"{activity.get('completed_at', '-')} | {activity.get('job_type', '-')} | {activity.get('status', '-')} | {activity.get('duration_seconds', '-')}s")
				if activity.get("error_summary"):
					st.caption(str(activity["error_summary"]))

	performance_snapshot = _read_performance_snapshot(PERFORMANCE_PATH)
	performance_summary = performance_snapshot.get("summary", {}) if isinstance(performance_snapshot, dict) else {}
	with st.expander(UI_LABELS["performance"], expanded=False):
		performance_col1, performance_col2, performance_col3 = st.columns(3)
		performance_col1.metric(UI_LABELS["bets"], str(int(performance_summary.get("total_bets", 0))))
		performance_col2.metric(UI_LABELS["profit_loss"], _to_decimal(performance_summary.get("profit_loss", 0.0)))
		performance_col3.metric(UI_LABELS["roi"], _to_percentage(performance_summary.get("roi", 0.0), scale=100.0, signed=False, digits=1))
		performance_col4, performance_col5, performance_col6 = st.columns(3)
		performance_col4.metric(UI_LABELS["yield"], _to_percentage(performance_summary.get("yield", 0.0), scale=100.0, signed=False, digits=1))
		performance_col5.metric(UI_LABELS["hit_rate"], _to_percentage(performance_summary.get("hit_rate", 0.0), scale=100.0, signed=False, digits=1))
		performance_col6.metric(UI_LABELS["max_drawdown"], _to_percentage(performance_summary.get("max_drawdown", 0.0), scale=100.0, signed=False, digits=1))
		performance_col7, performance_col8 = st.columns(2)
		performance_col7.metric(UI_LABELS["bankroll"], _to_decimal(performance_summary.get("final_bankroll", performance_summary.get("bankroll_start", 100.0))))
		average_clv = performance_summary.get("average_clv")
		performance_col8.metric("CLV medio", _to_decimal(average_clv, digits=4) if average_clv is not None and not pd.isna(average_clv) else "-")
		if not performance_snapshot:
			st.caption("Nessun paper trade reale è ancora stato regolato.")

	if "run_active" not in st.session_state:
		st.session_state["run_active"] = False

	if st.button(UI_LABELS["update_prematch"], type="primary", disabled=bool(st.session_state["run_active"])):
		st.session_state["run_active"] = True
		try:
			with st.status("Running pre-match pipeline...", expanded=True) as state:
				state.write("Health check")
				state.write("Fixtures and The Odds API refresh")
				state.write("Validated scoring and persistence")
				result = run_prematch(base_dir=BASE_DIR, output_dir=BASE_DIR, bankroll=100.0)
				state.update(label="Pre-match run completed", state="complete")
			st.success(f"Run completed: {result['paper_trading'].get('run_id', 'unknown')}")
		except Exception as exc:
			st.error(f"Pre-match run failed: {exc}")
		finally:
			st.session_state["run_active"] = False

	report = _read_report(REPORT_PATH)
	if report.empty:
		st.info("Nessun report paper-trading disponibile. Esegui AGGIORNA PRE-PARTITA.")
		return

	competition_states = _competition_support_states(report)
	state_col1, state_col2 = st.columns(2)
	state_col1.metric("Serie A", competition_states.get("Serie A", "IN PREPARAZIONE"))
	state_col2.metric("Serie B", competition_states.get("Serie B", "IN PREPARAZIONE"))

	filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
	filter_mode = filter_col1.selectbox(UI_LABELS["decision_filter"], list(DECISION_FILTER_OPTIONS.keys()), index=0)
	competition_filter = filter_col2.selectbox(UI_LABELS["competition"], list(COMPETITION_FILTER_OPTIONS.keys()), index=0)
	side_filter = filter_col3.selectbox(UI_LABELS["side"], list(SIDE_FILTER_OPTIONS.keys()), index=0)
	line_filter = filter_col4.selectbox(UI_LABELS["line"], list(LINE_FILTER_OPTIONS.keys()), index=0)
	quality_filter = filter_col5.selectbox("Qualità", list(QUALITY_FILTER_OPTIONS.keys()), index=0)
	st.caption(QUALITY_INFO_TEXT)

	play_cards = _prepare_play_cards(report, side_filter=side_filter, line_filter=line_filter, quality_filter=quality_filter, competition_filter=competition_filter)
	if play_cards:
		_render_play_cards(play_cards)
	else:
		st.info(NO_PLAY_MESSAGE)

	with st.expander(FULL_VIEW_LABEL):
		filtered = _apply_filters(report, filter_mode, side_filter, line_filter, quality_filter, competition_filter)
		dashboard_table = _prepare_dashboard_table(filtered)
		st.dataframe(dashboard_table, use_container_width=True, hide_index=True)


def _render_history() -> None:
	st.header(UI_LABELS["history"])
	rows = _load_history_rows(HISTORY_PATH)
	if not rows:
		st.info("Nessuno storico pre-partita disponibile.")
		return

	history_frame = pd.DataFrame(rows)
	history_frame = history_frame.sort_values("run_timestamp", ascending=False, kind="stable")
	columns = [
		"run_timestamp",
		"fixtures_evaluated",
		"odds_rows",
		"play_count",
		"no_bet_count",
		"average_ev",
		"supported_models",
		"warnings",
	]
	visible_columns = [column for column in columns if column in history_frame.columns]
	st.dataframe(history_frame[visible_columns], use_container_width=True, hide_index=True)

	run_choices = history_frame["run_id"].astype(str).tolist()
	selected_run = st.selectbox("Apri esecuzione precedente", run_choices)
	selected_row = history_frame.loc[history_frame["run_id"].astype(str) == selected_run].iloc[0].to_dict()
	run_csv = Path(str(selected_row.get("run_csv", "")))
	if run_csv.exists():
		run_frame = pd.read_csv(run_csv)
		st.subheader(f"Run {selected_run}")
		st.dataframe(_prepare_dashboard_table(run_frame), use_container_width=True, hide_index=True)
	else:
		st.warning("Il file di dettaglio non è disponibile.")


def main() -> None:
	st.set_page_config(page_title="CornerLab", layout="centered")
	st.markdown(
		"""
		<style>
			.block-container {max-width: 860px; padding-top: 1.2rem; padding-bottom: 2rem;}
			.stButton button {width: 100%; min-height: 3rem; font-size: 1rem;}
			[data-testid="stDataFrame"] {font-size: 0.92rem;}
		</style>
		""",
		unsafe_allow_html=True,
	)

	if "authenticated" not in st.session_state:
		st.session_state["authenticated"] = False

	if not st.session_state["authenticated"]:
		st.title("CornerLab")
		st.caption("Dashboard operativo mobile")
		st.subheader("Login")
		password = st.text_input("Password", type="password")
		if st.button("Accedi", type="primary"):
			if _verify_login(password):
				st.session_state["authenticated"] = True
				st.rerun()
			else:
				st.error("Credenziali non valide o CORNERLAB_APP_PASSWORD non configurata.")
		return

	st.title("CornerLab")
	page = st.radio("Pagina", ["Dashboard", UI_LABELS["history"], "Performance"], horizontal=True)
	if page == "Dashboard":
		_render_dashboard()
	elif page == UI_LABELS["history"]:
		_render_history()
	else:
		_render_performance_page()


if __name__ == "__main__":
	main()
