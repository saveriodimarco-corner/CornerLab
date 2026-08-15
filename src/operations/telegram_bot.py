from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from src.operations import real_bet_ledger
from src.operations.telegram_notifier import send_message


_NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")

# Forbidden by design: this module never triggers prematch/settlement/retraining,
# never changes thresholds/staking policy, and never executes shell commands.


def _authorized(chat_id: Any) -> bool:
	configured = os.getenv("TELEGRAM_CHAT_ID", "").strip()
	return bool(configured) and str(chat_id).strip() == configured


def build_play_keyboard(bet_id: str) -> dict[str, Any]:
	"""Callback data carries only an opaque bet_id, never monetary or model values."""
	return {
		"inline_keyboard": [
			[
				{"text": "✅ Conferma", "callback_data": f"confirm:{bet_id}"},
				{"text": "💶 Modifica stake", "callback_data": f"stake:{bet_id}"},
			],
			[
				{"text": "📈 Modifica quota", "callback_data": f"odds:{bet_id}"},
				{"text": "❌ Non giocata", "callback_data": f"skip:{bet_id}"},
			],
		]
	}


def build_bankroll_keyboard() -> dict[str, Any]:
	return {"inline_keyboard": [[{"text": "➕ Aggiungi fondi", "callback_data": "deposit"}, {"text": "➖ Preleva fondi", "callback_data": "withdraw"}]]}


def format_suggestion_prompt(row: dict[str, Any]) -> str:
	fixture = f"{row.get('home_team', '-')} vs {row.get('away_team', '-')}"
	market = f"{str(row.get('side', '')).upper()} {row.get('line', '')} corner"
	return (
		f"🎯 CORNERLAB — CONFERMA GIOCATA\n\n{fixture}\n{market}\n\n"
		f"Quota: {float(row.get('odds_at_decision', row.get('closing_odds', 0.0))):.2f}\n"
		f"Stake suggerito: €{float(row.get('recommended_stake', row.get('stake', 0.0))):.2f}\n"
		f"Qualità: {row.get('quality_tier', '-')}"
	)


def format_pending_suggestion(bet: dict[str, Any]) -> str:
	"""Render a still-unconfirmed suggestion using its current (possibly edited) values."""
	fixture = f"{bet.get('home_team', '-')} vs {bet.get('away_team', '-')}"
	market = f"{str(bet.get('side', '')).upper()} {bet.get('line', '')} corner"
	odds = bet.get("actual_odds") if bet.get("actual_odds") is not None else bet.get("suggested_odds", 0.0)
	stake = bet.get("actual_stake") if bet.get("actual_stake") is not None else bet.get("suggested_stake", 0.0)
	return (
		f"🎯 CORNERLAB — CONFERMA GIOCATA\n\n{fixture}\n{market}\n\n"
		f"Quota attuale: {float(odds or 0.0):.2f}\n"
		f"Stake attuale: €{float(stake or 0.0):.2f}\n"
		f"Qualità: {bet.get('quality_tier', '-')}"
	)


def format_bet_confirmation(bet: dict[str, Any], snapshot: dict[str, float]) -> str:
	fixture = f"{bet.get('home_team', '-')} vs {bet.get('away_team', '-')}"
	market = f"{str(bet.get('side', '')).upper()} {bet.get('line', '')} corner"
	return (
		f"✅ GIOCATA REGISTRATA\n\n{fixture}\n{market}\n\n"
		f"Stake reale: €{float(bet.get('actual_stake', 0.0)):.2f}\n"
		f"Quota reale: {float(bet.get('actual_odds', 0.0)):.2f}\n\n"
		f"Bankroll totale: €{snapshot['total_bankroll']:.2f}\n"
		f"Esposizione aperta: €{snapshot['open_exposure']:.2f}\n"
		f"Disponibile: €{snapshot['available_bankroll']:.2f}"
	)


def format_bankroll_message(snapshot: dict[str, float]) -> str:
	return (
		"💰 CORNERLAB — BANKROLL\n\n"
		f"Totale: €{snapshot['total_bankroll']:.2f}\n"
		f"Esposizione aperta: €{snapshot['open_exposure']:.2f}\n"
		f"Disponibile: €{snapshot['available_bankroll']:.2f}\n"
		f"P/L realizzato: €{snapshot['realized_pnl']:+.2f}"
	)


def offer_bet_confirmation(base_dir: Path | str, row: dict[str, Any], request_sender: Callable[[str, bytes, float], None] | None = None) -> str:
	"""Record a SUGGESTED bet and send the interactive confirmation keyboard; returns the opaque bet_id."""
	bet_id = real_bet_ledger.record_suggestion(base_dir, row)
	send_message(format_suggestion_prompt(row), request_sender=request_sender, reply_markup=build_play_keyboard(bet_id))
	return bet_id


def _pending_state_path(base_dir: Path | str) -> Path:
	path = Path(base_dir) / "data" / "operations" / "telegram_pending_state.json"
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def _read_pending_state(base_dir: Path | str) -> dict[str, Any]:
	path = _pending_state_path(base_dir)
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}


def _write_pending_state(base_dir: Path | str, state: dict[str, Any]) -> None:
	path = _pending_state_path(base_dir)
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
		json.dump(state, handle)
		temporary = Path(handle.name)
	temporary.replace(path)


def _set_pending(base_dir: Path | str, chat_id: Any, entry: dict[str, Any]) -> None:
	state = _read_pending_state(base_dir)
	state[str(chat_id)] = entry
	_write_pending_state(base_dir, state)


def _pop_pending(base_dir: Path | str, chat_id: Any) -> dict[str, Any] | None:
	state = _read_pending_state(base_dir)
	entry = state.pop(str(chat_id), None)
	_write_pending_state(base_dir, state)
	return entry


def _send_result_confirmation(base_dir: Path | str, result: dict[str, Any], request_sender: Callable[[str, bytes, float], None] | None) -> None:
	bet = result.get("bet")
	if result.get("ok") and bet is not None and bet.get("status") == real_bet_ledger.BET_PLACED:
		snapshot = real_bet_ledger.get_bankroll_snapshot(base_dir)
		send_message(format_bet_confirmation(bet, snapshot), request_sender=request_sender)


def _resend_suggestion(result: dict[str, Any], request_sender: Callable[[str, bytes, float], None] | None) -> None:
	"""An edited suggestion stays SUGGESTED, so re-offer it with the same four buttons."""
	bet = result.get("bet")
	if result.get("ok") and bet is not None and bet.get("status") == real_bet_ledger.SUGGESTED:
		send_message(format_pending_suggestion(bet), request_sender=request_sender, reply_markup=build_play_keyboard(bet["bet_id"]))


def handle_callback(base_dir: Path | str, chat_id: Any, callback_data: str, request_sender: Callable[[str, bytes, float], None] | None = None) -> dict[str, Any]:
	"""Resolve an opaque inline-button callback server-side; unauthorized chats are silently ignored."""
	if not _authorized(chat_id):
		return {"ok": False, "reason": "unauthorized"}

	action, _, token = str(callback_data).partition(":")

	if action == "deposit":
		_set_pending(base_dir, chat_id, {"action": "deposit"})
		send_message("Inserisci l'importo da depositare (>0):", request_sender=request_sender)
		return {"ok": True, "reason": None}
	if action == "withdraw":
		_set_pending(base_dir, chat_id, {"action": "withdraw"})
		send_message("Inserisci l'importo da prelevare (>0):", request_sender=request_sender)
		return {"ok": True, "reason": None}

	bet = real_bet_ledger.get_bet_by_id(base_dir, token)
	if bet is None:
		return {"ok": False, "reason": "unknown_bet"}
	suggestion_id = bet["suggestion_id"]

	if action == "confirm":
		result = real_bet_ledger.confirm_bet(base_dir, suggestion_id)
		_send_result_confirmation(base_dir, result, request_sender)
		return result
	if action == "skip":
		result = real_bet_ledger.skip_suggestion(base_dir, suggestion_id)
		if result.get("ok"):
			send_message("❌ Giocata segnata come non giocata.", request_sender=request_sender)
		return result
	if action == "stake":
		_set_pending(base_dir, chat_id, {"action": "modify_stake", "bet_id": token})
		send_message("Inserisci lo stake reale (es. 4.20):", request_sender=request_sender)
		return {"ok": True, "reason": None}
	if action == "odds":
		_set_pending(base_dir, chat_id, {"action": "modify_odds", "bet_id": token})
		send_message("Inserisci la quota reale (es. 1.92):", request_sender=request_sender)
		return {"ok": True, "reason": None}
	return {"ok": False, "reason": "unknown_action"}


def handle_message(base_dir: Path | str, chat_id: Any, text: str, request_sender: Callable[[str, bytes, float], None] | None = None) -> dict[str, Any]:
	"""Handle the narrow /bankroll command and pending numeric replies only; unauthorized chats are ignored."""
	if not _authorized(chat_id):
		return {"ok": False, "reason": "unauthorized"}

	text = str(text).strip()
	if text == "/bankroll":
		snapshot = real_bet_ledger.get_bankroll_snapshot(base_dir)
		send_message(format_bankroll_message(snapshot), request_sender=request_sender, reply_markup=build_bankroll_keyboard())
		return {"ok": True, "reason": None, "snapshot": snapshot}

	pending = _read_pending_state(base_dir).get(str(chat_id))
	if pending is None:
		# Arbitrary free text with no pending interaction must never trigger anything.
		return {"ok": False, "reason": "no_pending_interaction"}
	if not _NUMBER_PATTERN.match(text):
		return {"ok": False, "reason": "invalid_number"}

	amount = float(text)
	action = pending.get("action")
	_pop_pending(base_dir, chat_id)

	if action == "modify_stake":
		bet = real_bet_ledger.get_bet_by_id(base_dir, pending.get("bet_id"))
		if bet is None:
			return {"ok": False, "reason": "unknown_bet"}
		result = real_bet_ledger.modify_stake(base_dir, bet["suggestion_id"], amount)
		_resend_suggestion(result, request_sender)
		return result
	if action == "modify_odds":
		bet = real_bet_ledger.get_bet_by_id(base_dir, pending.get("bet_id"))
		if bet is None:
			return {"ok": False, "reason": "unknown_bet"}
		result = real_bet_ledger.modify_odds(base_dir, bet["suggestion_id"], amount)
		_resend_suggestion(result, request_sender)
		return result
	if action == "deposit":
		result = real_bet_ledger.record_deposit(base_dir, amount)
		if result.get("ok"):
			send_message(format_bankroll_message(result["snapshot"]), request_sender=request_sender)
		return result
	if action == "withdraw":
		result = real_bet_ledger.record_withdrawal(base_dir, amount)
		if result.get("ok"):
			send_message(format_bankroll_message(result["snapshot"]), request_sender=request_sender)
		return result
	return {"ok": False, "reason": "unknown_pending_action"}
