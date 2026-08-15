from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.research.decision_engine import MAX_STAKE_FRACTION


SUGGESTED = "SUGGESTED"
BET_PLACED = "BET_PLACED"
SKIPPED = "SKIPPED"
SETTLED_WIN = "SETTLED_WIN"
SETTLED_LOSS = "SETTLED_LOSS"
SETTLED_VOID = "SETTLED_VOID"

OPENING_BALANCE_AMOUNT = 100.0


def _utc_now() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def suggestion_key(row: dict[str, Any]) -> str:
	"""Canonical decision identity shared with the Telegram PLAY notification dedup key."""
	return "|".join(str(row.get(field, "")) for field in ["fixture_id", "market", "side", "line", "bookmaker", "decision_timestamp"])


def _db_path(base_dir: Path | str) -> Path:
	path = Path(base_dir) / "data" / "operations" / "real_bets.sqlite"
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def _connect(base_dir: Path | str) -> sqlite3.Connection:
	conn = sqlite3.connect(_db_path(base_dir))
	conn.row_factory = sqlite3.Row
	conn.executescript(
		"""
		CREATE TABLE IF NOT EXISTS real_bets (
			bet_id TEXT PRIMARY KEY,
			suggestion_id TEXT UNIQUE NOT NULL,
			fixture_id TEXT,
			competition TEXT,
			home_team TEXT,
			away_team TEXT,
			market TEXT,
			side TEXT,
			line TEXT,
			bookmaker TEXT,
			suggested_stake REAL,
			actual_stake REAL,
			suggested_odds REAL,
			actual_odds REAL,
			predicted_probability REAL,
			ev REAL,
			quality_tier TEXT,
			decision_timestamp TEXT,
			confirmed_timestamp TEXT,
			status TEXT NOT NULL,
			settled_timestamp TEXT,
			bet_result TEXT,
			profit_loss REAL,
			created_at TEXT,
			updated_at TEXT
		);

		CREATE TABLE IF NOT EXISTS bankroll_ledger (
			entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
			event_type TEXT NOT NULL,
			amount REAL NOT NULL,
			bet_id TEXT,
			created_at TEXT,
			note TEXT
		);
		"""
	)
	conn.commit()
	return conn


def ensure_opening_balance(conn: sqlite3.Connection, amount: float = OPENING_BALANCE_AMOUNT) -> None:
	count = conn.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0]
	if int(count) == 0:
		conn.execute(
			"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('OPENING_BALANCE', ?, NULL, ?, NULL)",
			(float(amount), _utc_now()),
		)
		conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
	return dict(row) if row is not None else None


def _fetch_bet(conn: sqlite3.Connection, suggestion_id: str) -> dict[str, Any] | None:
	row = conn.execute("SELECT * FROM real_bets WHERE suggestion_id = ?", (suggestion_id,)).fetchone()
	return _row_to_dict(row)


def _snapshot(conn: sqlite3.Connection) -> dict[str, float]:
	available = conn.execute("SELECT COALESCE(SUM(amount), 0.0) FROM bankroll_ledger").fetchone()[0]
	exposure = conn.execute("SELECT COALESCE(SUM(actual_stake), 0.0) FROM real_bets WHERE status = ?", (BET_PLACED,)).fetchone()[0]
	realized = conn.execute(
		"SELECT COALESCE(SUM(profit_loss), 0.0) FROM real_bets WHERE status IN (?, ?, ?)",
		(SETTLED_WIN, SETTLED_LOSS, SETTLED_VOID),
	).fetchone()[0]
	available = float(available or 0.0)
	exposure = float(exposure or 0.0)
	realized = float(realized or 0.0)
	return {
		"available_bankroll": available,
		"open_exposure": exposure,
		"total_bankroll": available + exposure,
		"realized_pnl": realized,
	}


def get_bankroll_snapshot(base_dir: Path | str) -> dict[str, float]:
	conn = _connect(base_dir)
	try:
		ensure_opening_balance(conn)
		return _snapshot(conn)
	finally:
		conn.close()


def get_bet(base_dir: Path | str, suggestion_id: str) -> dict[str, Any] | None:
	conn = _connect(base_dir)
	try:
		return _fetch_bet(conn, suggestion_id)
	finally:
		conn.close()


def get_bet_by_id(base_dir: Path | str, bet_id: str) -> dict[str, Any] | None:
	conn = _connect(base_dir)
	try:
		row = conn.execute("SELECT * FROM real_bets WHERE bet_id = ?", (bet_id,)).fetchone()
		return _row_to_dict(row)
	finally:
		conn.close()


def record_suggestion(base_dir: Path | str, row: dict[str, Any]) -> str:
	"""Record a model PLAY as SUGGESTED; idempotent by canonical decision identity."""
	suggestion_id = suggestion_key(row)
	conn = _connect(base_dir)
	try:
		existing = _fetch_bet(conn, suggestion_id)
		if existing is not None:
			return str(existing["bet_id"])
		bet_id = uuid4().hex
		now = _utc_now()
		conn.execute(
			"""
			INSERT INTO real_bets (
				bet_id, suggestion_id, fixture_id, competition, home_team, away_team,
				market, side, line, bookmaker, suggested_stake, actual_stake,
				suggested_odds, actual_odds, predicted_probability, ev, quality_tier,
				decision_timestamp, confirmed_timestamp, status, settled_timestamp,
				bet_result, profit_loss, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, NULL, ?, NULL, NULL, NULL, ?, ?)
			""",
			(
				bet_id,
				suggestion_id,
				str(row.get("fixture_id", "")),
				row.get("competition"),
				row.get("home_team"),
				row.get("away_team"),
				row.get("market"),
				row.get("side"),
				row.get("line"),
				row.get("bookmaker"),
				float(row.get("recommended_stake", row.get("stake", 0.0)) or 0.0),
				float(row.get("odds_at_decision", row.get("closing_odds", 0.0)) or 0.0),
				float(row.get("predicted_probability", 0.0) or 0.0),
				float(row.get("EV", row.get("ev", 0.0)) or 0.0),
				row.get("quality_tier"),
				row.get("decision_timestamp"),
				SUGGESTED,
				now,
				now,
			),
		)
		conn.commit()
		return bet_id
	finally:
		conn.close()


def confirm_bet(base_dir: Path | str, suggestion_id: str, actual_stake: float | None = None, actual_odds: float | None = None) -> dict[str, Any]:
	"""Confirm a suggestion as a real placed bet; idempotent for duplicate confirmations."""
	conn = _connect(base_dir)
	try:
		row = _fetch_bet(conn, suggestion_id)
		if row is None:
			return {"ok": False, "reason": "unknown_suggestion", "bet": None}
		if row["status"] != SUGGESTED:
			return {"ok": row["status"] == BET_PLACED, "reason": "already_processed", "bet": row}

		ensure_opening_balance(conn)
		snapshot = _snapshot(conn)
		stake = float(actual_stake) if actual_stake is not None else float(row["suggested_stake"] or 0.0)
		odds = float(actual_odds) if actual_odds is not None else float(row["suggested_odds"] or 0.0)

		if stake <= 0.0:
			return {"ok": False, "reason": "invalid_stake", "bet": row}
		if odds <= 1.0:
			return {"ok": False, "reason": "invalid_odds", "bet": row}
		available = snapshot["available_bankroll"]
		if stake > available + 1e-9:
			return {"ok": False, "reason": "insufficient_available_bankroll", "bet": row}
		cap = available * MAX_STAKE_FRACTION
		if stake > cap + 1e-9:
			return {"ok": False, "reason": "exceeds_stake_cap", "bet": row}

		now = _utc_now()
		conn.execute(
			"UPDATE real_bets SET status = ?, actual_stake = ?, actual_odds = ?, confirmed_timestamp = ?, updated_at = ? WHERE suggestion_id = ?",
			(BET_PLACED, stake, odds, now, now, suggestion_id),
		)
		conn.execute(
			"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('BET_PLACED', ?, ?, ?, ?)",
			(-stake, row["bet_id"], now, suggestion_id),
		)
		conn.commit()
		return {"ok": True, "reason": None, "bet": _fetch_bet(conn, suggestion_id)}
	finally:
		conn.close()


def modify_stake(base_dir: Path | str, suggestion_id: str, new_stake: float) -> dict[str, Any]:
	return confirm_bet(base_dir, suggestion_id, actual_stake=new_stake)


def modify_odds(base_dir: Path | str, suggestion_id: str, new_odds: float) -> dict[str, Any]:
	return confirm_bet(base_dir, suggestion_id, actual_odds=new_odds)


def skip_suggestion(base_dir: Path | str, suggestion_id: str) -> dict[str, Any]:
	"""Mark a suggestion as skipped; repeated presses are idempotent and never touch the bankroll."""
	conn = _connect(base_dir)
	try:
		row = _fetch_bet(conn, suggestion_id)
		if row is None:
			return {"ok": False, "reason": "unknown_suggestion", "bet": None}
		if row["status"] != SUGGESTED:
			return {"ok": row["status"] == SKIPPED, "reason": "already_processed", "bet": row}
		now = _utc_now()
		conn.execute("UPDATE real_bets SET status = ?, updated_at = ? WHERE suggestion_id = ?", (SKIPPED, now, suggestion_id))
		conn.commit()
		return {"ok": True, "reason": None, "bet": _fetch_bet(conn, suggestion_id)}
	finally:
		conn.close()


def settle_real_bet(base_dir: Path | str, suggestion_id: str, bet_result: str) -> dict[str, Any]:
	"""Settle a real placed bet using its actual stake/odds only; idempotent and never touches SUGGESTED/SKIPPED rows."""
	bet_result = str(bet_result).upper()
	conn = _connect(base_dir)
	try:
		row = _fetch_bet(conn, suggestion_id)
		if row is None:
			return {"ok": False, "reason": "unknown_suggestion", "bet": None}
		if row["status"] != BET_PLACED:
			return {"ok": str(row["status"]).startswith("SETTLED"), "reason": "not_placed_or_already_settled", "bet": row}
		if bet_result not in {"WIN", "LOSS", "VOID"}:
			return {"ok": False, "reason": "invalid_bet_result", "bet": row}

		stake = float(row["actual_stake"] or 0.0)
		odds = float(row["actual_odds"] or 0.0)
		now = _utc_now()

		if bet_result == "WIN":
			profit_loss = stake * (odds - 1.0)
			status = SETTLED_WIN
			conn.execute(
				"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('BET_WIN_RETURN', ?, ?, ?, ?)",
				(stake * odds, row["bet_id"], now, suggestion_id),
			)
		elif bet_result == "LOSS":
			profit_loss = -stake
			status = SETTLED_LOSS
			# No ledger event: the stake was already deducted at placement and remains consumed.
		else:
			profit_loss = 0.0
			status = SETTLED_VOID
			conn.execute(
				"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('BET_VOID_RETURN', ?, ?, ?, ?)",
				(stake, row["bet_id"], now, suggestion_id),
			)

		conn.execute(
			"UPDATE real_bets SET status = ?, settled_timestamp = ?, bet_result = ?, profit_loss = ?, updated_at = ? WHERE suggestion_id = ?",
			(status, now, bet_result, profit_loss, now, suggestion_id),
		)
		conn.commit()
		return {"ok": True, "reason": None, "bet": _fetch_bet(conn, suggestion_id)}
	finally:
		conn.close()


def record_deposit(base_dir: Path | str, amount: float) -> dict[str, Any]:
	amount = float(amount)
	if amount <= 0.0:
		return {"ok": False, "reason": "invalid_amount", "snapshot": None}
	conn = _connect(base_dir)
	try:
		ensure_opening_balance(conn)
		conn.execute(
			"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('DEPOSIT', ?, NULL, ?, NULL)",
			(amount, _utc_now()),
		)
		conn.commit()
		return {"ok": True, "reason": None, "snapshot": _snapshot(conn)}
	finally:
		conn.close()


def record_withdrawal(base_dir: Path | str, amount: float) -> dict[str, Any]:
	amount = float(amount)
	if amount <= 0.0:
		return {"ok": False, "reason": "invalid_amount", "snapshot": None}
	conn = _connect(base_dir)
	try:
		ensure_opening_balance(conn)
		snapshot = _snapshot(conn)
		if amount > snapshot["available_bankroll"] + 1e-9:
			return {"ok": False, "reason": "insufficient_available_bankroll", "snapshot": snapshot}
		conn.execute(
			"INSERT INTO bankroll_ledger (event_type, amount, bet_id, created_at, note) VALUES ('WITHDRAWAL', ?, NULL, ?, NULL)",
			(-amount, _utc_now()),
		)
		conn.commit()
		return {"ok": True, "reason": None, "snapshot": _snapshot(conn)}
	finally:
		conn.close()
