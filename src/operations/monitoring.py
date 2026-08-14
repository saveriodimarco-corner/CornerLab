from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"


def _utc_now() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
	except (OSError, json.JSONDecodeError):
		return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=True)
		temporary = Path(handle.name)
	temporary.replace(path)


def load_job_history(base_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
	path = base_dir / "data" / "operations" / "job_history.jsonl"
	if not path.exists():
		return []
	rows: list[dict[str, Any]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		try:
			rows.append(json.loads(line))
		except json.JSONDecodeError:
			continue
	return rows[-limit:] if limit is not None else rows


def _parse_timestamp(value: Any) -> datetime | None:
	if not value:
		return None
	try:
		return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
	except ValueError:
		return None


def _latest(rows: list[dict[str, Any]], job_type: str, statuses: set[str]) -> dict[str, Any] | None:
	matching = [row for row in rows if row.get("job_type") == job_type and row.get("status") in statuses]
	return matching[-1] if matching else None


def _duration_seconds(row: dict[str, Any] | None) -> float | None:
	if not row:
		return None
	if row.get("duration_seconds") is not None:
		return float(row["duration_seconds"])
	started = _parse_timestamp(row.get("started_at"))
	completed = _parse_timestamp(row.get("completed_at"))
	return round((completed - started).total_seconds(), 3) if started and completed else None


def _is_stale(timestamp: Any, threshold_minutes: int, now: datetime) -> bool:
	parsed = _parse_timestamp(timestamp)
	return parsed is None or (now - parsed).total_seconds() > threshold_minutes * 60


def derive_operations_status(base_dir: Path | str, now: datetime | None = None) -> dict[str, Any]:
	base_dir = Path(base_dir)
	now = now or datetime.now(timezone.utc)
	rows = load_job_history(base_dir)
	prematch_success = _latest(rows, "prematch", {"SUCCESS"})
	prematch_failure = _latest(rows, "prematch", {"FAILED"})
	settlement_success = _latest(rows, "settlement", {"SUCCESS"})
	settlement_failure = _latest(rows, "settlement", {"FAILED"})
	latest_prematch = _latest(rows, "prematch", {"SUCCESS", "FAILED"})
	latest_settlement = _latest(rows, "settlement", {"SUCCESS", "FAILED"})
	latest_failure = max([row for row in [latest_prematch, latest_settlement] if row and row.get("status") == "FAILED"], key=lambda row: row.get("completed_at", ""), default=None)
	prematch_stale_minutes = int(os.getenv("CORNERLAB_PREMATCH_STALE_MINUTES", "120"))
	settlement_stale_minutes = int(os.getenv("CORNERLAB_SETTLEMENT_STALE_MINUTES", "180"))
	if not rows:
		system_status = UNKNOWN
		warning_summary = None
	elif latest_failure:
		system_status = FAILED
		warning_summary = None
	elif _is_stale(prematch_success.get("completed_at") if prematch_success else None, prematch_stale_minutes, now) or _is_stale(settlement_success.get("completed_at") if settlement_success else None, settlement_stale_minutes, now):
		system_status = DEGRADED
		warning_summary = "Operational job freshness threshold exceeded"
	elif any(int(row.get("warning_count", 0)) > 0 for row in rows[-2:]):
		system_status = DEGRADED
		warning_summary = "Recent automation warning"
	else:
		system_status = HEALTHY
		warning_summary = None
	latest_successful_prematch = prematch_success.get("completed_at") if prematch_success else None
	latest_successful_settlement = settlement_success.get("completed_at") if settlement_success else None
	status = {
		"system_status": system_status,
		"last_prematch_success": latest_successful_prematch,
		"last_prematch_failure": prematch_failure.get("completed_at") if prematch_failure else None,
		"last_settlement_success": latest_successful_settlement,
		"last_settlement_failure": settlement_failure.get("completed_at") if settlement_failure else None,
		"prematch_last_duration_seconds": _duration_seconds(prematch_success or prematch_failure),
		"settlement_last_duration_seconds": _duration_seconds(settlement_success or settlement_failure),
		"prematch_last_exit_code": (prematch_failure or prematch_success or {}).get("exit_code"),
		"settlement_last_exit_code": (settlement_failure or settlement_success or {}).get("exit_code"),
		"current_lock_state": "UNLOCKED",
		"last_error_summary": latest_failure.get("error_summary") if latest_failure else None,
		"last_warning_summary": warning_summary,
		"last_successful_odds_refresh": latest_successful_prematch,
		"last_successful_fixture_refresh": latest_successful_prematch,
		"last_successful_performance_refresh": latest_successful_settlement or latest_successful_prematch,
		"scheduler_status": "SYSTEMD_TIMER_CONFIGURED",
		"updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
	}
	status["last_successful_prematch"] = status["last_prematch_success"]
	status["last_failed_prematch"] = status["last_prematch_failure"]
	status["last_successful_settlement"] = status["last_settlement_success"]
	status["last_failed_settlement"] = status["last_settlement_failure"]
	status["current_job_lock_state"] = status["current_lock_state"]
	status["scheduler_expected_status"] = status["scheduler_status"]
	return status


def _send_webhook(url: str, payload: dict[str, Any]) -> None:
	request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
	with urllib.request.urlopen(request, timeout=5):
		pass


def refresh_operations_status(
	base_dir: Path | str,
	lock_state: str = "UNLOCKED",
	alert_sender: Callable[[str, dict[str, Any]], None] | None = None,
	now: datetime | None = None,
) -> dict[str, Any]:
	base_dir = Path(base_dir)
	status = derive_operations_status(base_dir, now=now)
	status["current_lock_state"] = lock_state
	state_path = base_dir / "data" / "operations" / "automation_state.json"
	state = _read_json(state_path)
	previous = state.get("monitoring_last_status")
	current = status["system_status"]
	degraded_count = int(state.get("monitoring_degraded_count", 0)) + 1 if current == DEGRADED else 0
	state["monitoring_degraded_count"] = degraded_count
	state["monitoring_last_status"] = current
	last_alert_status = state.get("monitoring_last_alert_status")
	alert_transition = (current == FAILED and previous != FAILED) or (current == DEGRADED and degraded_count >= 2 and last_alert_status != DEGRADED) or (current == HEALTHY and previous == FAILED)
	webhook_url = os.getenv("CORNERLAB_ALERT_WEBHOOK_URL", "").strip()
	if alert_transition and webhook_url:
		payload = {
			"CornerLab": "operations alert",
			"job_type": "automation",
			"status": current,
			"timestamp": status["updated_at"],
			"error_summary": status["last_error_summary"] or status["last_warning_summary"],
			"last_successful_run": status["last_prematch_success"] or status["last_settlement_success"],
		}
		try:
			(alert_sender or _send_webhook)(webhook_url, payload)
			status["last_alert_status"] = current
			state["monitoring_last_alert_status"] = current
		except Exception:
			status["last_warning_summary"] = "Alert delivery failed"
	_write_json(state_path, state)
	_write_json(base_dir / "reports" / "operations_status.json", status)
	return status