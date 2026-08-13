from __future__ import annotations

import json
import os
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


def _autoload_env(env_path: Path | None = None) -> bool:
	target = env_path or ENV_PATH
	if not target.exists():
		return False
	return bool(load_dotenv(target, override=False))


_autoload_env()


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
	page = st.radio("Pagina", ["Dashboard", UI_LABELS["history"]], horizontal=True)
	if page == "Dashboard":
		_render_dashboard()
	else:
		_render_history()


if __name__ == "__main__":
	main()
