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
	telegram_bot.handle_callback(tmp_path, "999", f"odds:{bet_id}", request_sender=lambda *_: None)
	telegram_bot.handle_message(tmp_path, "999", "2.05", request_sender=lambda *_: None)

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


def test_pending_suggestion_never_displays_external_suggested_odds_as_current() -> None:
	bet = {
		"home_team": "Inter",
		"away_team": "Napoli",
		"side": "UNDER",
		"line": "10.5",
		"suggested_odds": 1.92,
		"suggested_stake": 5.0,
		"actual_odds": None,
		"actual_stake": None,
		"quality_tier": "TOP",
	}

	message = telegram_bot.format_pending_suggestion(bet)

	assert "1.92" not in message
	assert "Quota attuale" not in message


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
	telegram_bot.handle_callback(tmp_path, "999", f"odds:{bet_id}", request_sender=lambda *_: None)
	telegram_bot.handle_message(tmp_path, "999", "2.05", request_sender=lambda *_: None)

	first = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda *_: None)
	second = telegram_bot.handle_callback(tmp_path, "999", f"confirm:{bet_id}", request_sender=lambda *_: None)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert first["ok"] is True
	assert second["reason"] == "already_processed"
	assert snapshot["open_exposure"] == 5.0


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

	assert bet_id == ""
	assert result["ok"] is False
	assert result["reason"] == "unknown_bet"


def test_offer_fixture_confirmation_groups_multiple_candidates_in_one_message(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_configure(monkeypatch)

	first = _play_row() | {
		"side": "UNDER",
		"line": "10.5",
		"market": "TOTAL_CORNERS_UNDER",
		"predicted_probability": 0.75,
	}
	second = _play_row() | {
		"side": "UNDER",
		"line": "11.5",
		"market": "TOTAL_CORNERS_UNDER",
		"predicted_probability": 0.82,
	}

	payloads: list[bytes] = []

	bet_ids = telegram_bot.offer_fixture_confirmation(
		tmp_path,
		[first, second],
		request_sender=lambda _, payload, __: payloads.append(payload),
	)

	sent = _sent_text(payloads)

	assert len(bet_ids) == 2
	assert len(payloads) == 1
	assert "Inter vs Napoli" in sent
	assert "UNDER 10.5" in sent
	assert "UNDER 11.5" in sent
	assert f"odds:{bet_ids[0]}" in sent
	assert f"odds:{bet_ids[1]}" in sent
	assert "Conferma" not in sent

def test_offer_block_confirmation_sends_one_message_for_multiple_fixtures(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	rows = [
		{
			"fixture_id": 28,
			"competition": "Serie A",
			"market": "TOTAL_CORNERS_UNDER",
			"side": "UNDER",
			"line": "10.5",
			"home_team": "Lecce",
			"away_team": "Roma",
			"predicted_probability": 0.77,
			"confidence_score": 66.0,
			"kickoff_utc": "2026-08-31T16:30:00Z",
			"target_name": "under_10_5",
		},
		{
			"fixture_id": 29,
			"competition": "Serie A",
			"market": "TOTAL_CORNERS_UNDER",
			"side": "UNDER",
			"line": "11.5",
			"home_team": "Atalanta",
			"away_team": "Bologna",
			"predicted_probability": 0.82,
			"confidence_score": 68.0,
			"kickoff_utc": "2026-08-31T18:45:00Z",
			"target_name": "under_11_5",
		},
	]

	payloads = []

	monkeypatch.setattr(
		telegram_bot,
		"send_message",
		lambda text, request_sender=None, reply_markup=None: payloads.append(
			{"text": text, "reply_markup": reply_markup}
		) or True,
	)

	bet_ids = telegram_bot.offer_block_confirmation(tmp_path, rows)

	assert len(bet_ids) == 2
	assert len(payloads) == 1
	assert "Lecce vs Roma" in payloads[0]["text"]
	assert "Atalanta vs Bologna" in payloads[0]["text"]
	assert "UNDER 10.5" in payloads[0]["text"]
	assert "UNDER 11.5" in payloads[0]["text"]


def test_minimum_odds_display_rounds_up_to_next_cent() -> None:
	from src.operations.telegram_bot import _display_minimum_odds

	assert _display_minimum_odds(1.542857142857) == 1.55
	assert _display_minimum_odds(1.50) == 1.50
	assert _display_minimum_odds(1.549999999999) == 1.55


def test_block_candidate_can_receive_actual_odds_and_return_single_confirmation(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_configure(monkeypatch)

	rows = [
		{
			"fixture_id": 29,
			"competition": "Serie A",
			"market": "TOTAL_CORNERS_UNDER",
			"side": "UNDER",
			"line": "11.5",
			"home_team": "Atalanta",
			"away_team": "Bologna",
			"predicted_probability": 0.82,
			"confidence_score": 68.0,
			"kickoff_utc": "2026-08-31T18:45:00Z",
			"target_name": "under_11_5",
		},
	]

	bet_ids = telegram_bot.offer_block_confirmation(
		tmp_path,
		rows,
		request_sender=lambda *_: None,
	)

	assert len(bet_ids) == 1
	bet_id = bet_ids[0]

	telegram_bot.handle_callback(
		tmp_path,
		"999",
		f"odds:{bet_id}",
		request_sender=lambda *_: None,
	)

	payloads: list[bytes] = []

	result = telegram_bot.handle_message(
		tmp_path,
		"999",
		"1.60",
		request_sender=lambda _, payload, __: payloads.append(payload),
	)

	sent = _sent_text(payloads)

	assert result["ok"] is True
	assert result["bet"]["actual_odds"] == 1.60
	assert result["bet"]["status"] == real_bet_ledger.SUGGESTED
	assert "Atalanta vs Bologna" in sent
	assert "UNDER 11.5" in sent
	assert "Quota attuale: 1.60" in sent
	assert f"confirm:{bet_id}" in sent
	assert "GIOCATA REGISTRATA" not in sent
