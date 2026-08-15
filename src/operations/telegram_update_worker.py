from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from src.operations import telegram_bot


LOGGER = logging.getLogger(__name__)

LONG_POLL_TIMEOUT = 30
REQUEST_TIMEOUT = 45.0
BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 60.0

# Forbidden by design: this worker only routes updates to telegram_bot and never
# triggers prematch/settlement/retraining, changes thresholds or staking, nor executes shell commands.


def _enabled() -> bool:
	return os.getenv("CORNERLAB_TELEGRAM_ENABLED", "false").strip().lower() == "true"


def _state_path(base_dir: Path | str) -> Path:
	path = Path(base_dir) / "data" / "operations" / "telegram_update_state.json"
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def read_offset(base_dir: Path | str) -> int:
	path = _state_path(base_dir)
	if not path.exists():
		return 0
	try:
		return int(json.loads(path.read_text(encoding="utf-8")).get("next_offset", 0))
	except (OSError, ValueError, TypeError, json.JSONDecodeError):
		return 0


def write_offset(base_dir: Path | str, next_offset: int) -> None:
	path = _state_path(base_dir)
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
		json.dump({"next_offset": int(next_offset)}, handle)
		temporary = Path(handle.name)
	temporary.replace(path)


def _api_call(method: str, fields: dict[str, Any], request_sender: Callable[[str, bytes, float], Any] | None) -> Any:
	token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
	payload = urllib.parse.urlencode(fields).encode("utf-8")
	url = f"https://api.telegram.org/bot{token}/{method}"
	if request_sender is not None:
		return request_sender(url, payload, REQUEST_TIMEOUT)
	request = urllib.request.Request(url, data=payload, method="POST")
	with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
		return json.loads(response.read().decode("utf-8"))


def fetch_updates(offset: int, request_sender: Callable[[str, bytes, float], Any] | None = None) -> list[dict[str, Any]]:
	response = _api_call("getUpdates", {"offset": offset, "timeout": LONG_POLL_TIMEOUT}, request_sender)
	if not isinstance(response, dict) or not response.get("ok"):
		return []
	result = response.get("result")
	return [item for item in result if isinstance(item, dict)] if isinstance(result, list) else []


def answer_callback_query(callback_query_id: str, request_sender: Callable[[str, bytes, float], Any] | None = None) -> bool:
	"""Acknowledge the callback so Telegram stops showing the loading spinner."""
	try:
		_api_call("answerCallbackQuery", {"callback_query_id": callback_query_id}, request_sender)
		return True
	except Exception:
		LOGGER.warning("Telegram callback acknowledgement failed")
		return False


def process_update(base_dir: Path | str, update: dict[str, Any], request_sender: Callable[[str, bytes, float], Any] | None = None) -> dict[str, Any]:
	"""Route a single update; malformed payloads are ignored without raising."""
	if not isinstance(update, dict):
		return {"ok": False, "reason": "malformed_update"}

	callback_query = update.get("callback_query")
	if isinstance(callback_query, dict):
		chat = (callback_query.get("message") or {}).get("chat") or {}
		chat_id = chat.get("id")
		callback_data = callback_query.get("data")
		query_id = callback_query.get("id")
		if chat_id is None or not isinstance(callback_data, str):
			return {"ok": False, "reason": "malformed_update"}
		result = telegram_bot.handle_callback(base_dir, chat_id, callback_data)
		if query_id is not None:
			answer_callback_query(str(query_id), request_sender)
		return result

	message = update.get("message")
	if isinstance(message, dict):
		chat_id = (message.get("chat") or {}).get("id")
		text = message.get("text")
		if chat_id is None or not isinstance(text, str):
			return {"ok": False, "reason": "malformed_update"}
		return telegram_bot.handle_message(base_dir, chat_id, text)

	return {"ok": False, "reason": "malformed_update"}


def poll_once(base_dir: Path | str, request_sender: Callable[[str, bytes, float], Any] | None = None) -> int:
	"""Fetch and process one batch of updates, persisting the offset after each update."""
	offset = read_offset(base_dir)
	updates = fetch_updates(offset, request_sender)
	processed = 0
	for update in updates:
		update_id = update.get("update_id")
		if not isinstance(update_id, int) or update_id < offset:
			continue
		try:
			process_update(base_dir, update, request_sender)
		except Exception:
			LOGGER.warning("Telegram update processing failed for update_id=%s", update_id)
		write_offset(base_dir, update_id + 1)
		offset = update_id + 1
		processed += 1
	return processed


def run_worker(base_dir: Path | str, request_sender: Callable[[str, bytes, float], Any] | None = None, max_iterations: int | None = None, sleeper: Callable[[float], None] = time.sleep) -> int:
	"""Long-poll Telegram until stopped; transport failures back off instead of tight-looping."""
	if not _enabled() or not os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or not os.getenv("TELEGRAM_CHAT_ID", "").strip():
		LOGGER.warning("Telegram worker disabled or not configured")
		return 0

	iterations = 0
	backoff = BACKOFF_SECONDS
	while max_iterations is None or iterations < max_iterations:
		iterations += 1
		try:
			poll_once(base_dir, request_sender)
			backoff = BACKOFF_SECONDS
		except Exception:
			# Never log the token or raw request payloads.
			LOGGER.warning("Telegram polling failed, retrying in %.0fs", backoff)
			sleeper(backoff)
			backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
	return iterations
