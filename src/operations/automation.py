from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from health_check import run_health_check
from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.result_resolver import ResultResolver
from src.operations.real_bet_ledger import list_open_bets, settle_real_bet
from src.operations.monitoring import refresh_operations_status
from src.operations.prematch_runner import run_prematch
from src.research.observation_freeze import settle_paper_trades
from src.operations.telegram_notifier import format_settlement_completed, send_message
from src.operations.daily_summary import maybe_send_daily_summary
from src.operations import telegram_bot

import pandas as pd


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


def _provider_state_is_usable(
	state: dict[str, Any],
	now: datetime,
	max_age_minutes: int = 90,
) -> bool:
	if not state:
		return False

	try:
		remaining = int(state.get("requests_remaining"))
		rate_limited = int(state.get("rate_limited", 0))
	except (TypeError, ValueError):
		return False

	if remaining <= 0 or rate_limited != 0:
		return False

	created_at_raw = state.get("created_at")
	if not created_at_raw:
		return False

	try:
		created_at = datetime.fromisoformat(
			str(created_at_raw).replace("Z", "+00:00")
		)
	except (TypeError, ValueError):
		return False

	if created_at.tzinfo is None:
		created_at = created_at.replace(tzinfo=timezone.utc)
	else:
		created_at = created_at.astimezone(timezone.utc)

	if now.tzinfo is None:
		now = now.replace(tzinfo=timezone.utc)
	else:
		now = now.astimezone(timezone.utc)

	age_seconds = (now - created_at).total_seconds()

	if age_seconds < 0:
		return False

	return age_seconds <= max_age_minutes * 60


def _six_hour_alert_due(kickoff: datetime, now: datetime) -> bool:
	if kickoff.tzinfo is None:
		kickoff = kickoff.replace(tzinfo=timezone.utc)
	else:
		kickoff = kickoff.astimezone(timezone.utc)

	if now.tzinfo is None:
		now = now.replace(tzinfo=timezone.utc)
	else:
		now = now.astimezone(timezone.utc)

	seconds_to_kickoff = (kickoff - now).total_seconds()

	return (5 * 3600 + 45 * 60) <= seconds_to_kickoff <= 6 * 3600


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
			_notify_success(base_dir, job_type, result, completed_at)
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


def _notify_success(base_dir: Path, job_type: str, result: dict[str, Any], completed_at: str) -> None:
	"""Optional notifications consume persisted/canonical outputs and never affect the job outcome."""
	try:
		if job_type == "prematch":
			collector = result.get("collector", {})
			quota_remaining = collector.get("quota_remaining")
			quota_check_error = collector.get("quota_check_error")

			# Fail closed: never issue actionable betting alerts when provider
			# quota is exhausted OR cannot be verified.
			if quota_remaining is None:
				send_message(
					"🚨 CORNERLAB — STATO QUOTE NON VERIFICABILE\n\n"
					"Il controllo quota di The Odds API non è riuscito.\n"
					"Nessuna PLAY viene inviata finché la disponibilità delle quote non è verificata.\n\n"
					f"Dettaglio: {quota_check_error or 'quota status unknown'}\n"
					f"Ora: {completed_at}"
				)
				return

			if int(quota_remaining) <= 0:
				send_message(
					"🚨 CORNERLAB — PREMATCH NON OPERATIVO PER LE QUOTE\n\n"
					"Analisi eseguita, ma The Odds API ha 0 crediti disponibili.\n"
					"Nessuna PLAY viene inviata finché le quote non possono essere aggiornate.\n\n"
					f"Ora: {completed_at}"
				)
				return

			# Prematch runs never dispatch T-6h betting alerts.
			# Those are handled exclusively by run_alert_check_job().

		elif job_type == "settlement":
			summary = result.get("summary", result.get("settlement", {}))
			if int(summary.get("total_bets", 0)) > 0:
				send_message(format_settlement_completed(summary, completed_at))
			maybe_send_daily_summary(base_dir, completed_at)
	except Exception:
		return


def _offer_bet_confirmations(base_dir: Path, report: "pd.DataFrame") -> int:
    from src.operations.telegram_notifier import (
        _notified_keys,
        _record_notification,
        select_actionable_plays,
    )

    now = datetime.now(timezone.utc)
    rome = ZoneInfo("Europe/Rome")

    # Un blocco per giornata calcistica italiana.
    blocks: dict[str, list[dict[str, Any]]] = {}

    for row_dict in select_actionable_plays(report, now=now):
        kickoff_raw = row_dict.get("kickoff_utc")
        try:
            kickoff = datetime.fromisoformat(
                str(kickoff_raw).replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            continue

        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        else:
            kickoff = kickoff.astimezone(timezone.utc)

        block_date = kickoff.astimezone(rome).date().isoformat()
        blocks.setdefault(block_date, []).append(row_dict)

    notified = _notified_keys(base_dir)
    sent = 0

    for block_date, rows in sorted(blocks.items()):
        kickoffs = []

        for row in rows:
            try:
                kickoff = datetime.fromisoformat(
                    str(row.get("kickoff_utc")).replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                continue

            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            else:
                kickoff = kickoff.astimezone(timezone.utc)

            kickoffs.append(kickoff)

        if not kickoffs:
            continue

        first_kickoff = min(kickoffs)

        # L'intero blocco viene notificato sei ore prima
        # della prima partita della giornata.
        if not _six_hour_alert_due(first_kickoff, now):
            continue

        key = f"block:{block_date}"
        if key in notified:
            continue

        bet_ids = telegram_bot.offer_block_confirmation(base_dir, rows)
        if not bet_ids:
            continue

        _record_notification(
            base_dir,
            key,
            "INTERACTIVE_MODEL_CANDIDATE_BLOCK",
        )
        notified.add(key)
        sent += 1

    return sent

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


def _paper_fixture_ids_needing_results(base_dir: Path) -> list[str]:
	report_path = base_dir / "reports" / "paper_trading_current.csv"
	if not report_path.exists():
		return []

	try:
		report = pd.read_csv(report_path)
	except Exception:
		return []

	if report.empty or "fixture_id" not in report.columns:
		return []

	mask = pd.Series(True, index=report.index)

	if "competition" in report.columns:
		mask &= report["competition"].astype(str).eq("Serie A")

	if "decision" in report.columns:
		mask &= report["decision"].astype(str).eq("PLAY")

	if "market_support_status" in report.columns:
		mask &= report["market_support_status"].astype(str).eq("SUPPORTED")

	fixture_ids = (
		pd.to_numeric(report.loc[mask, "fixture_id"], errors="coerce")
		.dropna()
		.astype(int)
		.astype(str)
		.drop_duplicates()
		.tolist()
	)

	config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
	repo = CollectorRepository(config)

	return [
		fixture_id
		for fixture_id in fixture_ids
		if repo.get_result(fixture_id) is None
	]


def _resolve_open_real_bet_fixtures(base_dir: Path) -> dict[str, Any]:
	config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
	repo = CollectorRepository(config)
	resolver = ResultResolver(config, repo)

	fixture_ids: list[str] = []
	seen: set[str] = set()

	for bet in list_open_bets(base_dir):
		fixture_id = bet.get("fixture_id")
		if fixture_id is None:
			continue

		key = str(fixture_id)
		if key in seen:
			continue

		seen.add(key)
		fixture_ids.append(key)

	for fixture_id in _paper_fixture_ids_needing_results(base_dir):
		key = str(fixture_id)
		if key in seen:
			continue

		seen.add(key)
		fixture_ids.append(key)

	results = [resolver.resolve_fixture(fixture_id) for fixture_id in fixture_ids]

	return {
		"fixtures_checked": len(fixture_ids),
		"fixtures_resolved": sum(1 for item in results if item.get("ok") is True),
		"results": results,
	}


def _corner_bet_result(total_corners: int | float, side: str, line: int | float) -> str | None:
	try:
		total = float(total_corners)
		threshold = float(line)
	except (TypeError, ValueError):
		return None

	direction = str(side or "").strip().upper()

	if direction == "UNDER":
		if total < threshold:
			return "WIN"
		if total > threshold:
			return "LOSS"
		return "VOID"

	if direction == "OVER":
		if total > threshold:
			return "WIN"
		if total < threshold:
			return "LOSS"
		return "VOID"

	return None


def _settle_open_real_bets_from_canonical_results(base_dir: Path) -> dict[str, Any]:
	config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
	repo = CollectorRepository(config)

	settled: list[dict[str, Any]] = []
	failed: list[dict[str, Any]] = []
	pending = 0
	unsupported = 0

	for bet in list_open_bets(base_dir):
		fixture_id = bet.get("fixture_id")
		if fixture_id is None:
			pending += 1
			continue

		result = repo.get_result(fixture_id)
		if result is None or result.get("total_corners") is None:
			pending += 1
			continue

		bet_result = _corner_bet_result(
			result["total_corners"],
			bet.get("side"),
			bet.get("line"),
		)

		if bet_result is None:
			unsupported += 1
			continue

		settlement = settle_real_bet(
			base_dir,
			str(bet["suggestion_id"]),
			bet_result,
		)

		item = {
			"suggestion_id": bet["suggestion_id"],
			"fixture_id": fixture_id,
			"total_corners": result["total_corners"],
			"bet_result": bet_result,
			"settlement": settlement,
		}

		if settlement.get("ok") is True:
			settled.append(item)
		else:
			failed.append(item)

	return {
		"settled": len(settled),
		"failed": len(failed),
		"pending": pending,
		"unsupported": unsupported,
		"results": settled,
		"failures": failed,
	}


def _run_settlement_cycle(base_dir: Path) -> dict[str, Any]:
	result_resolution = _resolve_open_real_bet_fixtures(base_dir)
	real_settlement = _settle_open_real_bets_from_canonical_results(base_dir)
	paper_settlement = settle_paper_trades(
		base_dir=base_dir,
		output_dir=base_dir,
		bankroll_start=100.0,
	)

	summary = paper_settlement.get("summary", paper_settlement)

	return {
		"result_resolution": result_resolution,
		"real_settlement": real_settlement,
		"summary": summary,
		"settlement": summary,
		"paper_settlement": paper_settlement,
	}


def run_settlement_job(base_dir: Path | str | None = None) -> tuple[int, dict[str, Any]]:
	base_dir = Path(base_dir or Path.cwd())
	key = f"{datetime.now(timezone.utc):%Y%m%d%H}"
	return _run_job(
		"settlement",
		base_dir,
		key,
		lambda: _run_settlement_cycle(base_dir),
	)
def run_alert_check_job(base_dir: Path | str | None = None) -> tuple[int, dict[str, Any]]:
	base_dir = Path(base_dir or Path.cwd())
	now = datetime.now(timezone.utc)

	report_path = base_dir / "reports" / "paper_trading_current.csv"
	if not report_path.exists():
		return 0, {
			"job_type": "alert_check",
			"outcome": "SUCCESS",
			"alerts_sent": 0,
			"reason": "report_missing",
		}

	repo = CollectorRepository(
		CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
	)

	provider_state = repo.get_provider_usage("the-odds-api")

	if not _provider_state_is_usable(provider_state, now):
		return 0, {
			"job_type": "alert_check",
			"outcome": "SUCCESS",
			"alerts_sent": 0,
			"reason": "provider_state_unusable",
		}

	try:
		report = pd.read_csv(report_path)
	except Exception as exc:
		return 1, {
			"job_type": "alert_check",
			"outcome": "FAILED",
			"alerts_sent": 0,
			"reason": f"report_read_failed: {exc}",
		}

	alerts_sent = _offer_bet_confirmations(base_dir, report)

	return 0, {
		"job_type": "alert_check",
		"outcome": "SUCCESS",
		"alerts_sent": int(alerts_sent),
	}
