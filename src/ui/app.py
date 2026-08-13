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


def _decision_order(value: str) -> int:
	order = {
		"PLAY": 0,
		"LOW CONFIDENCE": 1,
		"NO BET": 2,
		"MODEL_UNAVAILABLE": 3,
	}
	return order.get(str(value), 99)


def _prepare_dashboard_table(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return frame
	table = frame.copy()
	table["market implied probability"] = np.where(
		table["closing_odds"].notna() & (table["closing_odds"] > 0.0),
		1.0 / table["closing_odds"],
		np.nan,
	)
	table["edge"] = table["predicted_probability"] - table["market implied probability"]
	table["recommended stake"] = table.get("recommended_stake", table.get("stake", np.nan))
	table["fixture"] = table["home_team"].astype(str) + " vs " + table["away_team"].astype(str)
	table["decision_rank"] = table["decision"].map(_decision_order)
	table = table.sort_values(["decision_rank", "ev"], ascending=[True, False], kind="stable")

	desired_columns = [
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
		"decision",
		"decision_reason",
	]
	present_columns = [column for column in desired_columns if column in table.columns]
	return table[present_columns].rename(
		columns={
			"kickoff_utc": "kickoff",
			"line": "market line",
			"closing_odds": "odds",
			"predicted_probability": "model probability",
			"decision_confidence_score": "confidence",
		}
	)


def _apply_filters(frame: pd.DataFrame, filter_mode: str, side_filter: str, line_filter: str) -> pd.DataFrame:
	if frame.empty:
		return frame
	table = frame.copy()
	if filter_mode == "PLAY ONLY":
		table = table.loc[table["decision"] == "PLAY"]
	if side_filter in {"OVER", "UNDER"}:
		table = table.loc[table["side"].astype(str).str.upper() == side_filter]
	if line_filter != "ALL":
		table = table.loc[table["line"].astype(str) == line_filter]
	return table


def _render_dashboard() -> None:
	st.header("Dashboard")
	st.caption("PAPER TRADING MODE")

	status = _read_prematch_status(PREMATCH_STATUS_PATH)
	last_update = status.get("completed_at") or "not run yet"
	health_ok = status.get("health_ok")
	provider_status = status.get("collector", {}).get("provider_status", "unknown")

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("CornerLab", "Operational")
	col2.metric("Last update", str(last_update))
	col3.metric("System health", "OK" if health_ok else "CHECK")
	col4.metric("Odds provider", str(provider_status).upper())

	if "run_active" not in st.session_state:
		st.session_state["run_active"] = False

	if st.button("UPDATE PRE-MATCH", type="primary", disabled=bool(st.session_state["run_active"])):
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
		st.info("No paper-trading report available yet. Run UPDATE PRE-MATCH.")
		return

	filter_col1, filter_col2, filter_col3 = st.columns(3)
	filter_mode = filter_col1.selectbox("Decision filter", ["ALL", "PLAY ONLY"], index=0)
	side_filter = filter_col2.selectbox("Side", ["ALL", "OVER", "UNDER"], index=0)
	line_filter = filter_col3.selectbox("Line", ["ALL", "8.5", "9.5", "10.5", "11.5"], index=0)

	filtered = _apply_filters(report, filter_mode, side_filter, line_filter)
	dashboard_table = _prepare_dashboard_table(filtered)
	st.dataframe(dashboard_table, use_container_width=True, hide_index=True)


def _render_history() -> None:
	st.header("History")
	rows = _load_history_rows(HISTORY_PATH)
	if not rows:
		st.info("No pre-match run history available yet.")
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
	selected_run = st.selectbox("Open previous run", run_choices)
	selected_row = history_frame.loc[history_frame["run_id"].astype(str) == selected_run].iloc[0].to_dict()
	run_csv = Path(str(selected_row.get("run_csv", "")))
	if run_csv.exists():
		run_frame = pd.read_csv(run_csv)
		st.subheader(f"Run {selected_run}")
		st.dataframe(_prepare_dashboard_table(run_frame), use_container_width=True, hide_index=True)
	else:
		st.warning("Run detail file is not available.")


def main() -> None:
	st.set_page_config(page_title="CornerLab", layout="centered")
	st.markdown(
		"""
		<style>
			.block-container {max-width: 860px; padding-top: 1.2rem; padding-bottom: 2rem;}
			.stButton button {width: 100%; min-height: 3rem; font-size: 1rem;}
		</style>
		""",
		unsafe_allow_html=True,
	)

	if "authenticated" not in st.session_state:
		st.session_state["authenticated"] = False

	if not st.session_state["authenticated"]:
		st.title("CornerLab")
		st.caption("Mobile operational dashboard")
		st.subheader("Login")
		password = st.text_input("Password", type="password")
		if st.button("Sign in", type="primary"):
			if _verify_login(password):
				st.session_state["authenticated"] = True
				st.rerun()
			else:
				st.error("Invalid credentials or CORNERLAB_APP_PASSWORD is not configured.")
		return

	st.title("CornerLab")
	page = st.radio("Page", ["DASHBOARD", "HISTORY"], horizontal=True)
	if page == "DASHBOARD":
		_render_dashboard()
	else:
		_render_history()


if __name__ == "__main__":
	main()
