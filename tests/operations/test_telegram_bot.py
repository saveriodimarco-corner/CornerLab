from __future__ import annotations

from pathlib import Path

import pytest

from src.operations import real_bet_ledger, telegram_bot


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_TELEGRAM_ENABLED", "true")
	monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
	monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")


def _play_row(fixture_id: int = 1) -> dict:
	return {
		"fixture_id": fixture_id,
		"competition": "Serie A",
		"market": "TOTAL_CORNERS_OVER",
		"side": "OVER",
		"line": "9.5",
		"bookmaker": "book",
		"decision_timestamp": "2026-08-15T12:00:00Z",
		"home_team": "Inter",
		"away_team": "Napoli",
		"recommended_stake": 4.20,
		"odds_at_decision": 1.92,
		"predicted_probability": 0.587,
		"EV": 0.121,
		"quality_tier": "TOP",
	}


def test_offer_and_confirm_via_callback_places_bet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	messages = []
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda _, payload, __: messages.append(payload))

	result = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda _, payload, __: messages.append(payload))

	assert result["ok"] is True
	assert result["bet"]["status"] == real_bet_ledger.BET_PLACED
	assert len(messages) == 2


def test_callback_payload_never_contains_monetary_or_model_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	bet_id = real_bet_ledger.record_suggestion(tmp_path, _play_row())
	keyboard = telegram_bot.build_play_keyboard(bet_id)

	callback_values = [button["callback_data"] for row in keyboard["inline_keyboard"] for button in row]

	assert all("4.2" not in value and "1.92" not in value for value in callback_values)
	assert all(value.split(":", 1)[1] == bet_id for value in callback_values)


def _sent_text(payloads: list[bytes]) -> str:
	import urllib.parse

	return "\n".join(urllib.parse.unquote_plus(payload.decode()) for payload in payloads)


def test_modify_stake_via_callback_then_message_persists_actual_stake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	telegram_bot.handle_callback(tmp_path, "999", f"stake:{bet_id}", request_sender=lambda *_: None)
	payloads: list[bytes] = []
	result = telegram_bot.handle_message(tmp_path, "999", "3.50", request_sender=lambda _, payload, __: payloads.append(payload))
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)
	sent = _sent_text(payloads)

	assert result["ok"] is True
	assert result["bet"]["actual_stake"] == 3.50
	assert result["bet"]["status"] == real_bet_ledger.SUGGESTED
	assert snapshot["open_exposure"] == 0.0
	assert "CONFERMA GIOCATA" in sent
	assert "Stake attuale: €3.50" in sent
	assert "GIOCATA REGISTRATA" not in sent
	assert f"confirm:{bet_id}" in sent


def test_modify_odds_via_callback_then_message_persists_actual_odds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	telegram_bot.handle_callback(tmp_path, "999", f"odds:{bet_id}", request_sender=lambda *_: None)
	payloads: list[bytes] = []
	result = telegram_bot.handle_message(tmp_path, "999", "2.05", request_sender=lambda _, payload, __: payloads.append(payload))
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)
	sent = _sent_text(payloads)

	assert result["ok"] is True
	assert result["bet"]["actual_odds"] == 2.05
	assert result["bet"]["status"] == real_bet_ledger.SUGGESTED
	assert snapshot["open_exposure"] == 0.0
	assert "Quota attuale: 2.05" in sent
	assert "GIOCATA REGISTRATA" not in sent
	assert f"skip:{bet_id}" in sent


def test_edited_suggestion_confirms_with_edited_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	telegram_bot.handle_callback(tmp_path, "999", f"stake:{bet_id}", request_sender=lambda *_: None)
	telegram_bot.handle_message(tmp_path, "999", "3.50", request_sender=lambda *_: None)
	telegram_bot.handle_callback(tmp_path, "999", f"odds:{bet_id}", request_sender=lambda *_: None)
	telegram_bot.handle_message(tmp_path, "999", "2.10", request_sender=lambda *_: None)

	payloads: list[bytes] = []
	result = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda _, payload, __: payloads.append(payload))
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["bet"]["status"] == real_bet_ledger.BET_PLACED
	assert result["bet"]["actual_stake"] == 3.50
	assert result["bet"]["actual_odds"] == 2.10
	assert snapshot["open_exposure"] == 3.50
	assert "GIOCATA REGISTRATA" in _sent_text(payloads)


def test_skip_via_callback_has_no_bankroll_impact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	result = telegram_bot.handle_callback(tmp_path, "999", f"skip:{bet_id}", request_sender=lambda *_: None)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["ok"] is True
	assert snapshot["total_bankroll"] == 100.0


def test_duplicate_confirm_callback_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	first = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda *_: None)
	second = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda *_: None)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert first["ok"] is True
	assert second["reason"] == "already_processed"
	assert snapshot["open_exposure"] == 4.20


def test_unauthorized_chat_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=lambda *_: None)

	result = telegram_bot.handle_callback(tmp_path, "111", f"confirm:{bet_id}", request_sender=lambda *_: None)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["reason"] == "unauthorized"
	assert snapshot["total_bankroll"] == 100.0


def test_bankroll_command_returns_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	messages = []

	result = telegram_bot.handle_message(tmp_path, "999", "/bankroll", request_sender=lambda _, payload, __: messages.append(payload))

	assert result["ok"] is True
	assert result["snapshot"]["total_bankroll"] == 100.0
	assert len(messages) == 1


def test_deposit_flow_increases_bankroll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	telegram_bot.handle_callback(tmp_path, "999", "deposit", request_sender=lambda *_: None)
	result = telegram_bot.handle_message(tmp_path, "999", "50", request_sender=lambda *_: None)

	assert result["ok"] is True
	assert result["snapshot"]["total_bankroll"] == 150.0


def test_withdrawal_flow_decreases_bankroll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	telegram_bot.handle_callback(tmp_path, "999", "withdraw", request_sender=lambda *_: None)
	result = telegram_bot.handle_message(tmp_path, "999", "20", request_sender=lambda *_: None)

	assert result["ok"] is True
	assert result["snapshot"]["total_bankroll"] == 80.0


def test_arbitrary_free_text_without_pending_state_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	result = telegram_bot.handle_message(tmp_path, "999", "rm -rf /", request_sender=lambda *_: None)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["reason"] == "no_pending_interaction"
	assert snapshot["total_bankroll"] == 100.0


def test_telegram_send_failure_does_not_corrupt_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)

	def _raise(*_args):
		raise RuntimeError("telegram unavailable")

	bet_id = telegram_bot.offer_bet_confirmation(tmp_path, _play_row(), request_sender=_raise)
	result = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=_raise)

	assert result["ok"] is True
	assert result["bet"]["status"] == real_bet_ledger.BET_PLACED
