from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from health_check import run_health_check
from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.operations.monitoring import refresh_operations_status
from src.operations.prematch_runner import run_prematch
from src.research.observation_freeze import settle_paper_trades


def _utc_now() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobAlreadyRunningError(RuntimeError):
	"""Raised when an overlapping automation invocation is rejected."""


@contextmanager
def job_lock(base_dir: Path, job_type: str) -> Iterator[None]:
	"""Use an advisory OS lock that is released automatically on process exit."""
	import fcntl

	lock_dir = base_dir / "data" / "operations"
	lock_dir.mkdir(parents=True, exist_ok=True)
	lock_path = lock_dir / f"{job_type}.lock"
	with lock_path.open("a+", encoding="utf-8") as handle:
		try:
			fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		except BlockingIOError as exc:
			raise JobAlreadyRunningError(f"{job_type} is already running") from exc
		handle.seek(0)
		handle.truncate()
		handle.write(str(os.getpid()))
		handle.flush()
		try:
			yield
		finally:
			fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=True)
		temporary_path = Path(handle.name)
	temporary_path.replace(path)


def _append_history(base_dir: Path, record: dict[str, Any]) -> None:
	if "duration_seconds" not in record:
		try:
			started = datetime.fromisoformat(str(record["started_at"]).replace("Z", "+00:00"))
			completed = datetime.fromisoformat(str(record["completed_at"]).replace("Z", "+00:00"))
			record["duration_seconds"] = round((completed - started).total_seconds(), 3)
		except (KeyError, TypeError, ValueError):
			record["duration_seconds"] = None
	path = base_dir / "data" / "operations" / "job_history.jsonl"
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _update_status(base_dir: Path, job_type: str, status: str, completed_at: str, error_summary: str | None, lock_state: str) -> None:
	refresh_operations_status(base_dir, lock_state=lock_state)


def _fingerprint_paths(paths: list[Path]) -> str:
	digest = hashlib.sha256()
	for path in paths:
		digest.update(str(path).encode("utf-8"))
		if path.exists():
			stat = path.stat()
			digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
	return digest.hexdigest()


def _in_window() -> bool:
	start = os.getenv("CORNERLAB_JOB_WINDOW_START", "00:00")
	end = os.getenv("CORNERLAB_JOB_WINDOW_END", "23:59")
	now = datetime.now(timezone.utc).strftime("%H:%M")
	return start <= now <= end


def _run_job(
	job_type: str,
	base_dir: Path,
	idempotency_key: str,
	runner: Callable[[], dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
	job_id = f"{job_type}-{uuid4().hex[:12]}"
	started_at = _utc_now()
	state_path = base_dir / "data" / "operations" / "automation_state.json"
	state = _read_json(state_path)
	try:
		with job_lock(base_dir, job_type):
			if state.get(f"{job_type}_key") == idempotency_key:
				completed_at = _utc_now()
				payload = {"job_id": job_id, "job_type": job_type, "outcome": "SKIPPED_IDEMPOTENT", "started_at": started_at, "completed_at": completed_at}
				_append_history(base_dir, {**payload, "status": "SKIPPED", "exit_code": 0, "fixtures_seen": 0, "rows_inserted": 0, "rows_skipped": 1, "rows_settled": 0, "warning_count": 0, "error_summary": None})
				_update_status(base_dir, job_type, "SUCCESS", completed_at, None, "UNLOCKED")
				return 0, payload
			result = runner()
			completed_at = _utc_now()
			state[f"{job_type}_key"] = idempotency_key
			state[f"{job_type}_last_job_id"] = job_id
			_write_json(state_path, state)
			collector = result.get("collector", {})
			settlement = result.get("settlement", result.get("summary", {}))
			payload = {"job_id": job_id, "job_type": job_type, "outcome": "SUCCESS", "started_at": started_at, "completed_at": completed_at, "result": result}
			_append_history(base_dir, {"job_id": job_id, "job_type": job_type, "started_at": started_at, "completed_at": completed_at, "status": "SUCCESS", "exit_code": 0, "fixtures_seen": int(collector.get("fixtures_fetched", 0)), "rows_inserted": int(collector.get("odds_writes", 0)), "rows_skipped": 0, "rows_settled": int(settlement.get("total_bets", 0)), "warning_count": len(result.get("validation_errors", [])), "error_summary": None, "provider_usage": result.get("provider_usage", {})})
			_update_status(base_dir, job_type, "SUCCESS", completed_at, None, "UNLOCKED")
			return 0, payload
	except JobAlreadyRunningError as exc:
		completed_at = _utc_now()
		payload = {"job_id": job_id, "job_type": job_type, "outcome": "SKIPPED_LOCKED", "started_at": started_at, "completed_at": completed_at, "error": str(exc)}
		_append_history(base_dir, {**payload, "status": "SKIPPED", "exit_code": 0, "fixtures_seen": 0, "rows_inserted": 0, "rows_skipped": 1, "rows_settled": 0, "warning_count": 1, "error_summary": str(exc)})
		_update_status(base_dir, job_type, "SUCCESS", completed_at, None, "LOCKED")
		return 0, payload
	except Exception as exc:
		completed_at = _utc_now()
		payload = {"job_id": job_id, "job_type": job_type, "outcome": "FAILED", "started_at": started_at, "completed_at": completed_at, "error": str(exc)}
		_append_history(base_dir, {**payload, "status": "FAILED", "exit_code": 1, "fixtures_seen": 0, "rows_inserted": 0, "rows_skipped": 0, "rows_settled": 0, "warning_count": 0, "error_summary": str(exc)})
		_update_status(base_dir, job_type, "FAILED", completed_at, str(exc), "UNLOCKED")
		return 1, payload


def run_prematch_job(base_dir: Path | str | None = None) -> tuple[int, dict[str, Any]]:
	base_dir = Path(base_dir or Path.cwd())
	if not _in_window():
		return 0, {"job_type": "prematch", "outcome": "SKIPPED_OUTSIDE_WINDOW", "started_at": _utc_now(), "completed_at": _utc_now()}
	key = f"{datetime.now(timezone.utc):%Y%m%d%H}"
	return _run_job("prematch", base_dir, key, lambda: _prematch_with_quota(base_dir))


def _prematch_with_quota(base_dir: Path) -> dict[str, Any]:
	health = run_health_check(base_dir=base_dir, output_dir=base_dir)
	if not bool(health.get("ok", False)):
		raise RuntimeError("core health check failed before prematch")
	result = run_prematch(base_dir=base_dir, output_dir=base_dir, bankroll=100.0)
	repo = CollectorRepository(CollectorConfig(db_path=base_dir / "data" / "collector.sqlite"))
	result["provider_usage"] = {provider: repo.get_provider_usage(provider) for provider in ["the-odds-api", "api-football"]}
	return result


def run_settlement_job(base_dir: Path | str | None = None) -> tuple[int, dict[str, Any]]:
	base_dir = Path(base_dir or Path.cwd())
	key = _fingerprint_paths([base_dir / "reports" / "paper_trading_current.csv", base_dir / "data" / "collector.sqlite"])
	return _run_job("settlement", base_dir, key, lambda: settle_paper_trades(base_dir=base_dir, output_dir=base_dir, bankroll_start=100.0))