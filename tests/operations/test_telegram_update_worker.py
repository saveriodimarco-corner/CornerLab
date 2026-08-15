from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.operations import real_bet_ledger, telegram_update_worker


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_TELEGRAM_ENABLED", "true")
	monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-value")
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


class _Transport:
	"""Deterministic Telegram transport returning queued getUpdates batches."""

	def __init__(self, batches: list[list[dict]]):
		self.batches = list(batches)
		self.calls: list[tuple[str, dict]] = []

	def __call__(self, url: str, payload: bytes, timeout: float):
		method = url.rsplit("/", 1)[-1]
		fields = dict(item.split("=", 1) for item in payload.decode().split("&") if "=" in item)
		self.calls.append((method, fields))
		if method == "getUpdates":
			return {"ok": True, "result": self.batches.pop(0) if self.batches else []}
		return {"ok": True, "result": True}


def _message_update(update_id: int, chat_id: int, text: str) -> dict:
	return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


def _callback_update(update_id: int, chat_id: int, data: str, query_id: str = "cbq-1") -> dict:
	return {"update_id": update_id, "callback_query": {"id": query_id, "data": data, "message": {"chat": {"id": chat_id}}}}


def test_message_update_is_routed_to_handle_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	routed = []
	monkeypatch.setattr(telegram_update_worker.telegram_bot, "handle_message", lambda base_dir, chat_id, text: routed.append((chat_id, text)) or {"ok": True})
	transport = _Transport([[_message_update(10, 999, "/bankroll")]])

	processed = telegram_update_worker.poll_once(tmp_path, transport)

	assert processed == 1
	assert routed == [(999, "/bankroll")]


def test_callback_update_is_routed_to_handle_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = real_bet_ledger.record_suggestion(tmp_path, _play_row())
	transport = _Transport([[_callback_update(10, 999, f"confirm:{bet_id}")]])

	telegram_update_worker.poll_once(tmp_path, transport)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert snapshot["open_exposure"] == 4.20


def test_callback_query_is_acknowledged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = real_bet_ledger.record_suggestion(tmp_path, _play_row())
	transport = _Transport([[_callback_update(10, 999, f"skip:{bet_id}", query_id="cbq-42")]])

	telegram_update_worker.poll_once(tmp_path, transport)

	acks = [fields for method, fields in transport.calls if method == "answerCallbackQuery"]
	assert acks and acks[0]["callback_query_id"] == "cbq-42"


def test_unauthorized_chat_cannot_mutate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = real_bet_ledger.record_suggestion(tmp_path, _play_row())
	transport = _Transport([[_callback_update(10, 111, f"confirm:{bet_id}")]])

	telegram_update_worker.poll_once(tmp_path, transport)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)
	bet = real_bet_ledger.get_bet_by_id(tmp_path, bet_id)

	assert snapshot["open_exposure"] == 0.0
	assert snapshot["total_bankroll"] == 100.0
	assert bet["status"] == real_bet_ledger.SUGGESTED


def test_offset_is_persisted_after_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	transport = _Transport([[_message_update(41, 999, "/bankroll")]])

	telegram_update_worker.poll_once(tmp_path, transport)
	state = json.loads((tmp_path / "data" / "operations" / "telegram_update_state.json").read_text())

	assert state["next_offset"] == 42
	assert telegram_update_worker.read_offset(tmp_path) == 42


def test_restart_resumes_from_saved_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	telegram_update_worker.write_offset(tmp_path, 77)
	transport = _Transport([[]])

	telegram_update_worker.poll_once(tmp_path, transport)

	get_updates = [fields for method, fields in transport.calls if method == "getUpdates"]
	assert get_updates[0]["offset"] == "77"


def test_duplicate_update_is_not_processed_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	bet_id = real_bet_ledger.record_suggestion(tmp_path, _play_row())
	update = _callback_update(10, 999, f"confirm:{bet_id}")
	transport = _Transport([[update], [update]])

	telegram_update_worker.poll_once(tmp_path, transport)
	second_processed = telegram_update_worker.poll_once(tmp_path, transport)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert second_processed == 0
	assert snapshot["open_exposure"] == 4.20
	assert telegram_update_worker.read_offset(tmp_path) == 11


def test_network_error_backs_off_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	attempts = []
	sleeps = []

	def _failing_transport(url: str, payload: bytes, timeout: float):
		attempts.append(url)
		raise OSError("network unreachable")

	telegram_update_worker.run_worker(tmp_path, _failing_transport, max_iterations=3, sleeper=sleeps.append)

	assert len(attempts) == 3
	assert sleeps == [5.0, 10.0, 20.0]


def test_malformed_updates_are_ignored_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	malformed = [
		{"update_id": 1},
		{"update_id": 2, "message": {"chat": {}}},
		{"update_id": 3, "callback_query": {"id": "x", "message": {}}},
		{"update_id": 4, "message": {"chat": {"id": 999}, "text": None}},
	]
	transport = _Transport([malformed])

	processed = telegram_update_worker.poll_once(tmp_path, transport)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert processed == 4
	assert snapshot["total_bankroll"] == 100.0
	assert telegram_update_worker.read_offset(tmp_path) == 5


def test_token_is_never_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
	_configure(monkeypatch)

	def _failing_transport(url: str, payload: bytes, timeout: float):
		raise OSError("network unreachable")

	with caplog.at_level(logging.WARNING):
		telegram_update_worker.run_worker(tmp_path, _failing_transport, max_iterations=2, sleeper=lambda _: None)

	assert caplog.text
	assert "secret-token-value" not in caplog.text


def test_worker_does_nothing_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_TELEGRAM_ENABLED", "false")
	monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-value")
	monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
	calls = []

	iterations = telegram_update_worker.run_worker(tmp_path, lambda *args: calls.append(args), max_iterations=3)

	assert iterations == 0
	assert calls == []
