from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.operations import automation
from src.operations import monitoring


def test_job_success_is_idempotent_and_writes_audit_state(tmp_path: Path) -> None:
	calls = []

	def runner():
		calls.append("run")
		return {"collector": {"fixtures_fetched": 2, "odds_writes": 3}, "settlement": {"total_bets": 1}}

	first_code, first = automation._run_job("prematch", tmp_path, "slot-1", runner)
	second_code, second = automation._run_job("prematch", tmp_path, "slot-1", runner)
	history = (tmp_path / "data" / "operations" / "job_history.jsonl").read_text(encoding="utf-8").splitlines()
	status = json.loads((tmp_path / "reports" / "operations_status.json").read_text(encoding="utf-8"))

	assert first_code == 0
	assert first["outcome"] == "SUCCESS"
	assert second_code == 0
	assert second["outcome"] == "SKIPPED_IDEMPOTENT"
	assert calls == ["run"]
	assert len(history) == 2
	assert status["last_successful_prematch"]


def test_job_failure_records_nonzero_without_persisting_success_key(tmp_path: Path) -> None:
	def runner():
		raise RuntimeError("provider unavailable")

	exit_code, payload = automation._run_job("prematch", tmp_path, "slot-1", runner)
	state = automation._read_json(tmp_path / "data" / "operations" / "automation_state.json")
	status = json.loads((tmp_path / "reports" / "operations_status.json").read_text(encoding="utf-8"))

	assert exit_code == 1
	assert payload["outcome"] == "FAILED"
	assert "prematch_key" not in state
	assert status["last_failed_prematch"]


def test_concurrent_lock_skips_duplicate_execution(tmp_path: Path) -> None:
	with automation.job_lock(tmp_path, "settlement"):
		exit_code, payload = automation._run_job("settlement", tmp_path, "state-1", lambda: {})

	assert exit_code == 0
	assert payload["outcome"] == "SKIPPED_LOCKED"


def test_settlement_fingerprint_changes_only_when_canonical_inputs_change(tmp_path: Path) -> None:
	report = tmp_path / "reports" / "paper_trading_current.csv"
	database = tmp_path / "data" / "collector.sqlite"
	report.parent.mkdir(parents=True)
	database.parent.mkdir(parents=True)
	report.write_text("trade\nfirst\n", encoding="utf-8")
	database.write_text("db", encoding="utf-8")
	first = automation._fingerprint_paths([report, database])
	report.write_text("trade\nsecond\n", encoding="utf-8")
	second = automation._fingerprint_paths([report, database])

	assert first != second


def test_prematch_window_skip_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_JOB_WINDOW_START", "23:59")
	monkeypatch.setenv("CORNERLAB_JOB_WINDOW_END", "00:00")

	exit_code, payload = automation.run_prematch_job(tmp_path)

	assert exit_code == 0
	assert payload["outcome"] == "SKIPPED_OUTSIDE_WINDOW"


def test_prematch_wrapper_calls_canonical_runner_once_per_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	calls = []
	monkeypatch.setattr(automation, "_in_window", lambda: True)
	monkeypatch.setattr(automation, "run_health_check", lambda **_: {"ok": True})
	monkeypatch.setattr(automation, "run_prematch", lambda **_: calls.append("run") or {"collector": {"fixtures_fetched": 1, "odds_writes": 1}, "settlement": {"total_bets": 0}})
	monkeypatch.setattr(automation, "CollectorRepository", lambda _: type("Repository", (), {"get_provider_usage": lambda self, provider: {"provider": provider, "requests_remaining": 1}})())

	first_code, first = automation.run_prematch_job(tmp_path)
	second_code, second = automation.run_prematch_job(tmp_path)

	assert first_code == 0
	assert first["outcome"] == "SUCCESS"
	assert second_code == 0
	assert second["outcome"] == "SKIPPED_IDEMPOTENT"
	assert calls == ["run"]


def test_failed_core_health_does_not_run_prematch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	calls = []
	monkeypatch.setattr(automation, "_in_window", lambda: True)
	monkeypatch.setattr(automation, "run_health_check", lambda **_: {"ok": False})
	monkeypatch.setattr(automation, "run_prematch", lambda **_: calls.append("run") or {})

	exit_code, payload = automation.run_prematch_job(tmp_path)

	assert exit_code == 1
	assert payload["outcome"] == "FAILED"
	assert calls == []


def test_settlement_wrapper_keeps_unresolved_trades_pending_and_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	reports = tmp_path / "reports"
	reports.mkdir(parents=True)
	(reports / "paper_trading_current.csv").write_text("canonical input", encoding="utf-8")
	monkeypatch.setattr(automation, "settle_paper_trades", lambda **_: {"summary": {"total_bets": 0, "pending": 1}})

	first_code, first = automation.run_settlement_job(tmp_path)
	second_code, second = automation.run_settlement_job(tmp_path)

	assert first_code == 0
	assert first["outcome"] == "SUCCESS"
	assert first["result"]["summary"]["total_bets"] == 0
	assert first["result"]["summary"]["pending"] == 1
	assert second_code == 0
	assert second["outcome"] == "SKIPPED_IDEMPOTENT"


def _write_history(tmp_path: Path, rows: list[dict]) -> None:
	path = tmp_path / "data" / "operations" / "job_history.jsonl"
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _job(job_type: str, status: str, completed_at: str, error_summary: str | None = None, warning_count: int = 0) -> dict:
	return {"job_type": job_type, "status": status, "started_at": completed_at, "completed_at": completed_at, "exit_code": 0 if status == "SUCCESS" else 1, "warning_count": warning_count, "error_summary": error_summary, "duration_seconds": 1.0}


def test_monitoring_states_cover_unknown_healthy_degraded_failed_and_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	now = monitoring.datetime(2026, 8, 14, 12, 0, tzinfo=monitoring.timezone.utc)
	assert monitoring.derive_operations_status(tmp_path, now=now)["system_status"] == monitoring.UNKNOWN
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-14T11:30:00Z"), _job("settlement", "SUCCESS", "2026-08-14T11:31:00Z")])
	assert monitoring.derive_operations_status(tmp_path, now=now)["system_status"] == monitoring.HEALTHY
	monkeypatch.setenv("CORNERLAB_PREMATCH_STALE_MINUTES", "10")
	assert monitoring.derive_operations_status(tmp_path, now=now)["system_status"] == monitoring.DEGRADED
	_write_history(tmp_path, [_job("prematch", "FAILED", "2026-08-14T11:59:00Z", "provider unavailable"), _job("settlement", "SUCCESS", "2026-08-14T11:59:00Z")])
	assert monitoring.derive_operations_status(tmp_path, now=now)["system_status"] == monitoring.FAILED


def test_alerts_transition_without_spam_and_recover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	now = monitoring.datetime(2026, 8, 14, 12, 0, tzinfo=monitoring.timezone.utc)
	sent = []
	monkeypatch.setenv("CORNERLAB_ALERT_WEBHOOK_URL", "https://alerts.invalid")
	_write_history(tmp_path, [_job("prematch", "FAILED", "2026-08-14T11:59:00Z", "provider unavailable"), _job("settlement", "SUCCESS", "2026-08-14T11:59:00Z")])
	monitoring.refresh_operations_status(tmp_path, alert_sender=lambda _, payload: sent.append(payload), now=now)
	monitoring.refresh_operations_status(tmp_path, alert_sender=lambda _, payload: sent.append(payload), now=now)
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-14T12:00:00Z"), _job("settlement", "SUCCESS", "2026-08-14T12:00:00Z")])
	monitoring.refresh_operations_status(tmp_path, alert_sender=lambda _, payload: sent.append(payload), now=now)

	assert [item["status"] for item in sent] == [monitoring.FAILED, monitoring.HEALTHY]
	assert all("CornerLab" in item and "error_summary" in item for item in sent)


def test_absent_alert_credentials_do_not_block_monitoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CORNERLAB_ALERT_WEBHOOK_URL", raising=False)
	_write_history(tmp_path, [_job("prematch", "FAILED", "2026-08-14T11:59:00Z", "provider unavailable")])

	status = monitoring.refresh_operations_status(tmp_path)

	assert status["system_status"] == monitoring.FAILED


def test_degraded_alert_requires_persistence_and_monitoring_does_not_touch_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	now = monitoring.datetime(2026, 8, 14, 12, 0, tzinfo=monitoring.timezone.utc)
	sent = []
	monkeypatch.setenv("CORNERLAB_ALERT_WEBHOOK_URL", "https://alerts.invalid")
	monkeypatch.setenv("CORNERLAB_PREMATCH_STALE_MINUTES", "10")
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-14T11:00:00Z"), _job("settlement", "SUCCESS", "2026-08-14T11:00:00Z")])
	report = tmp_path / "reports" / "paper_trading_current.csv"
	report.parent.mkdir(parents=True, exist_ok=True)
	report.write_text("decision\nPLAY\n", encoding="utf-8")
	before = report.read_bytes()

	monitoring.refresh_operations_status(tmp_path, alert_sender=lambda _, payload: sent.append(payload), now=now)
	monitoring.refresh_operations_status(tmp_path, alert_sender=lambda _, payload: sent.append(payload), now=now)

	assert [item["status"] for item in sent] == [monitoring.DEGRADED]
	assert report.read_bytes() == before


def test_fresh_prematch_with_no_settlement_history_is_healthy(tmp_path: Path) -> None:
	now = monitoring.datetime(2026, 8, 15, 12, 50, 31, tzinfo=monitoring.timezone.utc)
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-15T12:50:25Z")])

	status = monitoring.derive_operations_status(tmp_path, now=now)

	assert status["system_status"] == monitoring.HEALTHY
	assert status["last_warning_summary"] is None


def test_stale_prematch_without_settlement_history_is_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_PREMATCH_STALE_MINUTES", "60")
	now = monitoring.datetime(2026, 8, 15, 15, 0, 0, tzinfo=monitoring.timezone.utc)
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-15T12:50:25Z")])

	status = monitoring.derive_operations_status(tmp_path, now=now)

	assert status["system_status"] == monitoring.DEGRADED
	assert status["last_warning_summary"] == "Operational job freshness threshold exceeded"


def test_naive_timestamp_is_treated_as_utc_without_false_stale(tmp_path: Path) -> None:
	now = monitoring.datetime(2026, 8, 15, 12, 50, 31, tzinfo=monitoring.timezone.utc)
	_write_history(tmp_path, [{"job_type": "prematch", "status": "SUCCESS", "started_at": "2026-08-15T12:46:34", "completed_at": "2026-08-15T12:50:25", "exit_code": 0, "warning_count": 0, "error_summary": None}])

	status = monitoring.derive_operations_status(tmp_path, now=now)

	assert status["system_status"] == monitoring.HEALTHY


def test_scheduler_not_live_does_not_mark_fresh_job_stale(tmp_path: Path) -> None:
	now = monitoring.datetime(2026, 8, 15, 12, 50, 31, tzinfo=monitoring.timezone.utc)
	_write_history(tmp_path, [_job("prematch", "SUCCESS", "2026-08-15T12:50:25Z")])

	status = monitoring.derive_operations_status(tmp_path, now=now)

	assert status["system_status"] == monitoring.HEALTHY
	assert status["scheduler_status"] == "SYSTEMD_TIMER_CONFIGURED"