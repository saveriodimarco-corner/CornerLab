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


def test_settlement_wrapper_keeps_unresolved_trades_pending_and_is_idempotent_within_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	calls = []
	monkeypatch.setattr(
		automation,
		"settle_paper_trades",
		lambda **_: calls.append("run") or {"summary": {"total_bets": 0, "pending": 1}},
	)

	first_code, first = automation.run_settlement_job(tmp_path)
	second_code, second = automation.run_settlement_job(tmp_path)

	assert first_code == 0
	assert first["outcome"] == "SUCCESS"
	assert first["result"]["summary"]["total_bets"] == 0
	assert first["result"]["summary"]["pending"] == 1
	assert second_code == 0
	assert second["outcome"] == "SKIPPED_IDEMPOTENT"
	assert calls == ["run"]


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

def test_offer_bet_confirmations_groups_candidates_by_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report = pd.DataFrame(
		[
			{
				"fixture_id": 1,
				"competition": "Serie A",
				"market": "TOTAL_CORNERS_UNDER",
				"side": "UNDER",
				"line": "10.5",
				"home_team": "Inter",
				"away_team": "Napoli",
				"predicted_probability": 0.75,
				"confidence_score": 70.0,
				"kickoff_utc": "2026-08-31T18:45:00Z",
				"target_name": "under_10_5",
			},
			{
				"fixture_id": 1,
				"competition": "Serie A",
				"market": "TOTAL_CORNERS_UNDER",
				"side": "UNDER",
				"line": "11.5",
				"home_team": "Inter",
				"away_team": "Napoli",
				"predicted_probability": 0.82,
				"confidence_score": 72.0,
				"kickoff_utc": "2026-08-31T18:45:00Z",
				"target_name": "under_11_5",
			},
		]
	)

	grouped_calls: list[list[dict]] = []

	real_datetime = automation.datetime

	class FixedDateTime(real_datetime):
		@classmethod
		def now(cls, tz=None):
			return real_datetime(
				2026, 8, 31, 12, 45,
				tzinfo=automation.timezone.utc,
			)

	monkeypatch.setattr(automation, "datetime", FixedDateTime)

	monkeypatch.setattr(
		automation,
		"telegram_bot",
		type(
			"FakeTelegram",
			(),
			{
				"offer_block_confirmation": staticmethod(
					lambda base_dir, rows: grouped_calls.append(rows) or ["bet-1", "bet-2"]
				)
			},
		),
	)

	sent = automation._offer_bet_confirmations(tmp_path, report)

	assert sent == 1
	assert len(grouped_calls) == 1
	assert len(grouped_calls[0]) == 2
	assert {row["line"] for row in grouped_calls[0]} == {"10.5", "11.5"}

def test_six_hour_alert_window() -> None:
	kickoff = automation.datetime(
		2026, 8, 31, 18, 45,
		tzinfo=automation.timezone.utc,
	)

	too_early = automation.datetime(
		2026, 8, 31, 12, 44,
		tzinfo=automation.timezone.utc,
	)

	exactly_six_hours = automation.datetime(
		2026, 8, 31, 12, 45,
		tzinfo=automation.timezone.utc,
	)

	inside_window = automation.datetime(
		2026, 8, 31, 12, 55,
		tzinfo=automation.timezone.utc,
	)

	too_late = automation.datetime(
		2026, 8, 31, 13, 1,
		tzinfo=automation.timezone.utc,
	)

	assert automation._six_hour_alert_due(kickoff, too_early) is False
	assert automation._six_hour_alert_due(kickoff, exactly_six_hours) is True
	assert automation._six_hour_alert_due(kickoff, inside_window) is True
	assert automation._six_hour_alert_due(kickoff, too_late) is False

def test_offer_bet_confirmations_respects_six_hour_window(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report = pd.DataFrame(
		[
			{
				"fixture_id": 1,
				"competition": "Serie A",
				"market": "TOTAL_CORNERS_UNDER",
				"side": "UNDER",
				"line": "10.5",
				"home_team": "Inter",
				"away_team": "Napoli",
				"predicted_probability": 0.75,
				"confidence_score": 70.0,
				"kickoff_utc": "2026-08-31T18:45:00Z",
				"target_name": "under_10_5",
			},
		]
	)

	calls = []
	real_datetime = automation.datetime

	monkeypatch.setattr(
		automation,
		"datetime",
		type(
			"FakeDateTime",
			(),
			{
				"now": staticmethod(
					lambda tz=None: real_datetime(
						2026, 8, 31, 11, 0,
						tzinfo=automation.timezone.utc,
					)
				),
				"fromisoformat": staticmethod(real_datetime.fromisoformat),
			},
		),
		raising=False,
	)

	monkeypatch.setattr(
		automation,
		"telegram_bot",
		type(
			"FakeTelegram",
			(),
			{
				"offer_fixture_confirmation": staticmethod(
					lambda base_dir, rows: calls.append(rows) or ["bet-1"]
				)
			},
		),
	)

	sent = automation._offer_bet_confirmations(tmp_path, report)

	assert sent == 0
	assert calls == []

def test_offer_bet_confirmations_sends_one_block_six_hours_before_first_kickoff(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report = pd.DataFrame(
		[
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
	)

	real_datetime = automation.datetime

	class FixedDateTime(real_datetime):
		@classmethod
		def now(cls, tz=None):
			return real_datetime(
				2026, 8, 31, 10, 30,
				tzinfo=automation.timezone.utc,
			)

	monkeypatch.setattr(automation, "datetime", FixedDateTime)

	block_calls = []

	monkeypatch.setattr(
		automation,
		"telegram_bot",
		type(
			"FakeTelegram",
			(),
			{
				"offer_block_confirmation": staticmethod(
					lambda base_dir, rows: block_calls.append(rows) or ["bet-1", "bet-2"]
				),
				"offer_fixture_confirmation": staticmethod(
					lambda base_dir, rows: (_ for _ in ()).throw(
						AssertionError("fixture-level alert must not be used")
					)
				),
			},
		),
	)

	sent = automation._offer_bet_confirmations(tmp_path, report)

	assert sent == 1
	assert len(block_calls) == 1
	assert {row["fixture_id"] for row in block_calls[0]} == {28, 29}

def test_provider_state_is_recent_and_usable() -> None:
	now = automation.datetime(
		2026, 8, 31, 12, 30,
		tzinfo=automation.timezone.utc,
	)

	fresh = {
		"requests_remaining": 100,
		"rate_limited": 0,
		"created_at": "2026-08-31T11:30:00Z",
	}

	stale = {
		"requests_remaining": 100,
		"rate_limited": 0,
		"created_at": "2026-08-31T10:00:00Z",
	}

	exhausted = {
		"requests_remaining": 0,
		"rate_limited": 1,
		"created_at": "2026-08-31T12:00:00Z",
	}

	assert automation._provider_state_is_usable(fresh, now) is True
	assert automation._provider_state_is_usable(stale, now) is False
	assert automation._provider_state_is_usable(exhausted, now) is False
	assert automation._provider_state_is_usable({}, now) is False

def test_run_alert_check_job_uses_persisted_state_without_prematch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report_path = tmp_path / "reports" / "paper_trading_current.csv"
	report_path.parent.mkdir(parents=True, exist_ok=True)

	pd.DataFrame(
		[
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
			}
		]
	).to_csv(report_path, index=False)

	class FakeRepo:
		def __init__(self, config):
			pass

		def get_provider_usage(self, provider):
			return {
				"provider": provider,
				"requests_remaining": 100,
				"rate_limited": 0,
				"created_at": "2026-08-31T12:00:00Z",
			}

	monkeypatch.setattr(automation, "CollectorRepository", FakeRepo)

	monkeypatch.setattr(
		automation,
		"run_prematch",
		lambda *args, **kwargs: (_ for _ in ()).throw(
			AssertionError("alert checker must not call run_prematch")
		),
	)

	calls = []

	monkeypatch.setattr(
		automation,
		"_offer_bet_confirmations",
		lambda base_dir, report: calls.append(len(report)) or 1,
	)

	real_datetime = automation.datetime

	class FixedDateTime(real_datetime):
		@classmethod
		def now(cls, tz=None):
			return real_datetime(
				2026, 8, 31, 12, 45,
				tzinfo=automation.timezone.utc,
			)

	monkeypatch.setattr(automation, "datetime", FixedDateTime)

	code, result = automation.run_alert_check_job(tmp_path)

	assert code == 0
	assert result["outcome"] == "SUCCESS"
	assert result["alerts_sent"] == 1
	assert calls == [1]

def test_run_alert_check_job_suppresses_alerts_with_stale_provider_state(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report_path = tmp_path / "reports" / "paper_trading_current.csv"
	report_path.parent.mkdir(parents=True, exist_ok=True)

	pd.DataFrame(
		[
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
			}
		]
	).to_csv(report_path, index=False)

	class FakeRepo:
		def __init__(self, config):
			pass

		def get_provider_usage(self, provider):
			return {
				"provider": provider,
				"requests_remaining": 100,
				"rate_limited": 0,
				"created_at": "2026-08-31T10:00:00Z",
			}

	monkeypatch.setattr(automation, "CollectorRepository", FakeRepo)

	real_datetime = automation.datetime

	class FixedDateTime(real_datetime):
		@classmethod
		def now(cls, tz=None):
			return real_datetime(
				2026, 8, 31, 12, 45,
				tzinfo=automation.timezone.utc,
			)

	monkeypatch.setattr(automation, "datetime", FixedDateTime)

	monkeypatch.setattr(
		automation,
		"_offer_bet_confirmations",
		lambda *args, **kwargs: (_ for _ in ()).throw(
			AssertionError("must not send alerts with stale provider state")
		),
	)

	code, result = automation.run_alert_check_job(tmp_path)

	assert code == 0
	assert result["outcome"] == "SUCCESS"
	assert result["alerts_sent"] == 0
	assert result["reason"] == "provider_state_unusable"

def test_prematch_success_does_not_dispatch_t6_alerts(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report_path = tmp_path / "reports" / "paper_trading_current.csv"
	report_path.parent.mkdir(parents=True, exist_ok=True)

	pd.DataFrame(
		[
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
			}
		]
	).to_csv(report_path, index=False)

	dispatch_calls = []

	monkeypatch.setattr(
		automation,
		"_offer_bet_confirmations",
		lambda *args, **kwargs: dispatch_calls.append(True) or 0,
	)

	monkeypatch.setattr(
		automation,
		"send_message",
		lambda *args, **kwargs: True,
	)

	result = {
		"collector": {
			"quota_remaining": 100,
			"quota_check_error": None,
		}
	}

	automation._notify_success(
		tmp_path,
		"prematch",
		result,
		"2026-08-31T12:00:00Z",
	)

	assert dispatch_calls == []


def test_offer_bet_confirmations_is_idempotent_for_same_block(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import pandas as pd

	report = pd.DataFrame(
		[
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
	)

	real_datetime = automation.datetime

	class FixedDateTime(real_datetime):
		@classmethod
		def now(cls, tz=None):
			return real_datetime(
				2026, 8, 31, 10, 30,
				tzinfo=automation.timezone.utc,
			)

	monkeypatch.setattr(automation, "datetime", FixedDateTime)

	block_calls = []

	monkeypatch.setattr(
		automation,
		"telegram_bot",
		type(
			"FakeTelegram",
			(),
			{
				"offer_block_confirmation": staticmethod(
					lambda base_dir, rows: block_calls.append(rows) or ["bet-1", "bet-2"]
				),
			},
		),
	)

	first = automation._offer_bet_confirmations(tmp_path, report)
	second = automation._offer_bet_confirmations(tmp_path, report)

	assert first == 1
	assert second == 0
	assert len(block_calls) == 1


def test_resolve_open_real_bet_fixtures_deduplicates_fixture_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        automation,
        "list_open_bets",
        lambda _: [
            {"fixture_id": "31"},
            {"fixture_id": "31"},
            {"fixture_id": "34"},
            {"fixture_id": None},
        ],
    )

    calls = []

    class FakeResolver:
        def __init__(self, config, repo):
            pass

        def resolve_fixture(self, fixture_id):
            calls.append(fixture_id)
            return {
                "ok": fixture_id == "31",
                "fixture_id": fixture_id,
            }

    monkeypatch.setattr(automation, "ResultResolver", FakeResolver)

    result = automation._resolve_open_real_bet_fixtures(tmp_path)

    assert calls == ["31", "34"]
    assert result["fixtures_checked"] == 2
    assert result["fixtures_resolved"] == 1
    assert len(result["results"]) == 2


def test_settlement_cycle_resolves_and_settles_real_bets_before_paper_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        automation,
        "_resolve_open_real_bet_fixtures",
        lambda _: calls.append("resolve") or {
            "fixtures_checked": 1,
            "fixtures_resolved": 1,
            "results": [],
        },
    )

    monkeypatch.setattr(
        automation,
        "_settle_open_real_bets_from_canonical_results",
        lambda _: calls.append("real") or {
            "settled": 1,
            "pending": 0,
            "unsupported": 0,
            "results": [],
        },
    )

    monkeypatch.setattr(
        automation,
        "settle_paper_trades",
        lambda **_: calls.append("paper") or {
            "summary": {
                "total_bets": 2,
                "pending": 0,
            }
        },
    )

    result = automation._run_settlement_cycle(tmp_path)

    assert calls == ["resolve", "real", "paper"]
    assert result["result_resolution"]["fixtures_resolved"] == 1
    assert result["real_settlement"]["settled"] == 1
    assert result["settlement"]["total_bets"] == 2
    assert result["paper_settlement"]["summary"]["pending"] == 0


@pytest.mark.parametrize(
    ("total_corners", "side", "line", "expected"),
    [
        (9, "UNDER", 11.5, "WIN"),
        (12, "UNDER", 11.5, "LOSS"),
        (12, "OVER", 11.5, "WIN"),
        (9, "OVER", 11.5, "LOSS"),
        (10, "UNDER", 10, "VOID"),
        (10, "OVER", 10, "VOID"),
        (9, "UNKNOWN", 9.5, None),
    ],
)
def test_corner_bet_result(total_corners, side, line, expected) -> None:
    assert automation._corner_bet_result(total_corners, side, line) == expected


def test_settle_open_real_bets_uses_only_canonical_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = automation.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = automation.CollectorRepository(config)

    fixture_31 = repo.upsert_fixture(
        {
            "provider_fixture_id": "1550112",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-04T18:45:00Z",
            "home_team": "Genoa",
            "away_team": "Como",
            "status": "FT",
            "provider": "api-football",
        }
    )

    fixture_34 = repo.upsert_fixture(
        {
            "provider_fixture_id": "1550107",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "NS",
            "provider": "api-football",
        }
    )

    repo.upsert_result(
        {
            "fixture_id": fixture_31["fixture_id"],
            "home_score": 1,
            "away_score": 4,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-04T21:00:00Z",
            "provider": "api-football",
        }
    )

    monkeypatch.setattr(
        automation,
        "list_open_bets",
        lambda _: [
            {
                "suggestion_id": "31|TOTAL_CORNERS_UNDER|UNDER|11.5",
                "fixture_id": str(fixture_31["fixture_id"]),
                "side": "UNDER",
                "line": "11.5",
            },
            {
                "suggestion_id": "34|TOTAL_CORNERS_UNDER|UNDER|9.5",
                "fixture_id": str(fixture_34["fixture_id"]),
                "side": "UNDER",
                "line": "9.5",
            },
        ],
    )

    settlements = []

    def fake_settle(base_dir, suggestion_id, bet_result):
        settlements.append((suggestion_id, bet_result))
        return {
            "ok": True,
            "reason": None,
            "bet": {
                "suggestion_id": suggestion_id,
                "bet_result": bet_result,
            },
        }

    monkeypatch.setattr(
        automation,
        "settle_real_bet",
        fake_settle,
    )

    result = automation._settle_open_real_bets_from_canonical_results(tmp_path)

    assert settlements == [
        ("31|TOTAL_CORNERS_UNDER|UNDER|11.5", "WIN"),
    ]
    assert result["settled"] == 1
    assert result["pending"] == 1
    assert result["unsupported"] == 0
    assert result["results"][0]["fixture_id"] == str(fixture_31["fixture_id"])
    assert result["results"][0]["total_corners"] == 9
    assert result["results"][0]["bet_result"] == "WIN"


def test_settle_open_real_bets_settles_multiple_lines_on_same_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = automation.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = automation.CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "4001",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "FT",
            "provider": "api-football",
        }
    )

    repo.upsert_result(
        {
            "fixture_id": fixture["fixture_id"],
            "home_score": 1,
            "away_score": 0,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-05T21:00:00Z",
            "provider": "api-football",
        }
    )

    monkeypatch.setattr(
        automation,
        "list_open_bets",
        lambda _: [
            {
                "suggestion_id": "4001|TOTAL_CORNERS_UNDER|UNDER|10.5",
                "fixture_id": str(fixture["fixture_id"]),
                "side": "UNDER",
                "line": "10.5",
            },
            {
                "suggestion_id": "4001|TOTAL_CORNERS_UNDER|UNDER|11.5",
                "fixture_id": str(fixture["fixture_id"]),
                "side": "UNDER",
                "line": "11.5",
            },
        ],
    )

    calls = []

    monkeypatch.setattr(
        automation,
        "settle_real_bet",
        lambda base_dir, suggestion_id, bet_result: (
            calls.append((suggestion_id, bet_result))
            or {"ok": True, "reason": None}
        ),
    )

    result = automation._settle_open_real_bets_from_canonical_results(tmp_path)

    assert calls == [
        ("4001|TOTAL_CORNERS_UNDER|UNDER|10.5", "WIN"),
        ("4001|TOTAL_CORNERS_UNDER|UNDER|11.5", "WIN"),
    ]

    assert result["settled"] == 2
    assert result["pending"] == 0
    assert result["unsupported"] == 0


def test_automatic_real_settlement_is_idempotent_end_to_end(tmp_path: Path) -> None:
    import sqlite3

    from src.operations import real_bet_ledger

    config = automation.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = automation.CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "5001",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-04T18:45:00Z",
            "home_team": "Genoa",
            "away_team": "Como",
            "status": "FT",
            "provider": "api-football",
        }
    )

    fixture_id = fixture["fixture_id"]

    row = {
        "fixture_id": fixture_id,
        "competition": "Serie A",
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": "11.5",
        "bookmaker": "bet365.it",
        "decision_timestamp": "2026-09-04T12:00:00Z",
        "home_team": "Genoa",
        "away_team": "Como",
        "recommended_stake": 5.0,
        "odds_at_decision": 2.0,
        "predicted_probability": 0.75,
        "EV": 0.50,
        "quality_tier": "TOP",
    }

    suggestion_id = real_bet_ledger.suggestion_key(row)
    real_bet_ledger.record_suggestion(tmp_path, row)

    confirmed = real_bet_ledger.confirm_bet(
        tmp_path,
        suggestion_id,
        actual_stake=5.0,
        actual_odds=2.0,
    )
    assert confirmed["ok"] is True

    repo.upsert_result(
        {
            "fixture_id": fixture_id,
            "home_score": 1,
            "away_score": 4,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-04T21:00:00Z",
            "provider": "api-football",
        }
    )

    first = automation._settle_open_real_bets_from_canonical_results(tmp_path)
    snapshot_after_first = real_bet_ledger.get_bankroll_snapshot(tmp_path)

    second = automation._settle_open_real_bets_from_canonical_results(tmp_path)
    snapshot_after_second = real_bet_ledger.get_bankroll_snapshot(tmp_path)

    bet = real_bet_ledger.get_bet(tmp_path, suggestion_id)

    conn = sqlite3.connect(
        tmp_path / "data" / "operations" / "real_bets.sqlite"
    )
    try:
        win_returns = conn.execute(
            """
            SELECT COUNT(*)
            FROM bankroll_ledger
            WHERE event_type = 'BET_WIN_RETURN'
            """
        ).fetchone()[0]
    finally:
        conn.close()

    assert first["settled"] == 1
    assert first["pending"] == 0

    assert second["settled"] == 0
    assert second["pending"] == 0

    assert bet["status"] == real_bet_ledger.SETTLED_WIN
    assert bet["bet_result"] == "WIN"
    assert bet["profit_loss"] == pytest.approx(5.0)

    assert snapshot_after_first["open_exposure"] == 0.0
    assert snapshot_after_first["total_bankroll"] == pytest.approx(105.0)
    assert snapshot_after_first["realized_pnl"] == pytest.approx(5.0)

    assert snapshot_after_second == snapshot_after_first
    assert win_returns == 1


def test_paper_fixture_ids_needing_results_returns_only_unresolved_supported_plays(
    tmp_path: Path,
) -> None:
    import pandas as pd

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "fixture_id": 31,
                "competition": "Serie A",
                "decision": "PLAY",
                "market_support_status": "SUPPORTED",
            },
            {
                "fixture_id": 31,
                "competition": "Serie A",
                "decision": "PLAY",
                "market_support_status": "SUPPORTED",
            },
            {
                "fixture_id": 32,
                "competition": "Serie A",
                "decision": "PLAY",
                "market_support_status": "SUPPORTED",
            },
            {
                "fixture_id": 33,
                "competition": "Serie A",
                "decision": "NO_BET",
                "market_support_status": "SUPPORTED",
            },
            {
                "fixture_id": 34,
                "competition": "Premier League",
                "decision": "PLAY",
                "market_support_status": "SUPPORTED",
            },
            {
                "fixture_id": 35,
                "competition": "Serie A",
                "decision": "PLAY",
                "market_support_status": "UNSUPPORTED",
            },
        ]
    ).to_csv(reports_dir / "paper_trading_current.csv", index=False)

    config = automation.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = automation.CollectorRepository(config)

    repo.upsert_result(
        {
            "fixture_id": 32,
            "home_score": 1,
            "away_score": 0,
            "home_corners": 6,
            "away_corners": 3,
            "total_corners": 9,
            "settled_at": "2026-09-05T21:00:00Z",
            "provider": "api-football",
        }
    )

    assert automation._paper_fixture_ids_needing_results(tmp_path) == ["31"]


def test_resolve_open_real_bet_fixtures_includes_unresolved_paper_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        automation,
        "list_open_bets",
        lambda _: [
            {"fixture_id": "31"},
            {"fixture_id": "34"},
        ],
    )

    monkeypatch.setattr(
        automation,
        "_paper_fixture_ids_needing_results",
        lambda _: ["34", "35"],
    )

    calls = []

    class FakeResolver:
        def __init__(self, *args, **kwargs):
            pass

        def resolve_fixture(self, fixture_id):
            calls.append(str(fixture_id))
            return {"ok": True, "fixture_id": str(fixture_id)}

    monkeypatch.setattr(automation, "ResultResolver", FakeResolver)

    result = automation._resolve_open_real_bet_fixtures(tmp_path)

    assert calls == ["31", "34", "35"]
    assert result["fixtures_checked"] == 3
    assert result["fixtures_resolved"] == 3


@pytest.mark.parametrize(
    ("total_corners", "side", "line"),
    [
        (9, "UNDER", None),
        (9, "UNDER", ""),
        (9, "UNDER", "bad"),
        (None, "UNDER", "11.5"),
        ("bad", "OVER", "9.5"),
    ],
)
def test_corner_bet_result_fails_closed_on_malformed_values(
    total_corners,
    side,
    line,
) -> None:
    assert automation._corner_bet_result(total_corners, side, line) is None


def test_settle_open_real_bets_does_not_count_failed_settlement_as_settled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.collector.collector_config import CollectorConfig
    from src.collector.collector_repository import CollectorRepository

    config = CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
    )
    repo = CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "1550112",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-04T18:45:00Z",
            "home_team": "Genoa",
            "away_team": "Como",
            "status": "FT",
            "provider": "api-football",
        }
    )

    repo.upsert_result(
        {
            "fixture_id": fixture["fixture_id"],
            "home_score": 1,
            "away_score": 4,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-04T21:00:00Z",
            "provider": "api-football",
        }
    )

    monkeypatch.setattr(
        automation,
        "list_open_bets",
        lambda _: [
            {
                "suggestion_id": "31|TOTAL_CORNERS_UNDER|UNDER|11.5",
                "fixture_id": fixture["fixture_id"],
                "side": "UNDER",
                "line": "11.5",
            }
        ],
    )

    monkeypatch.setattr(
        automation,
        "settle_real_bet",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "simulated_failure",
        },
    )

    result = automation._settle_open_real_bets_from_canonical_results(tmp_path)

    assert result["settled"] == 0
    assert result["failed"] == 1
    assert result["pending"] == 0
    assert result["unsupported"] == 0
    assert result["results"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["settlement"]["reason"] == "simulated_failure"


def test_daily_summary_does_not_send_before_last_match_is_due(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.operations import daily_summary

    config = daily_summary.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = daily_summary.CollectorRepository(config)

    repo.upsert_fixture(
        {
            "provider_fixture_id": "9001",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "NS",
            "provider": "api-football",
        }
    )

    sent = []
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: sent.append(message) or True,
    )

    result = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T19:30:00Z",
    )

    assert result["sent"] is False
    assert result["reason"] == "last_match_not_due"
    assert sent == []


def test_daily_summary_sends_once_after_all_fixtures_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.operations import daily_summary

    config = daily_summary.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = daily_summary.CollectorRepository(config)

    fixture1 = repo.upsert_fixture(
        {
            "provider_fixture_id": "9101",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T16:00:00Z",
            "home_team": "Inter",
            "away_team": "Napoli",
            "status": "FT",
            "provider": "api-football",
        }
    )
    fixture2 = repo.upsert_fixture(
        {
            "provider_fixture_id": "9102",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "FT",
            "provider": "api-football",
        }
    )

    for fixture, home_score, away_score, corners in [
        (fixture1, 2, 1, 10),
        (fixture2, 1, 0, 9),
    ]:
        repo.upsert_result(
            {
                "fixture_id": fixture["fixture_id"],
                "home_score": home_score,
                "away_score": away_score,
                "home_corners": corners // 2,
                "away_corners": corners - (corners // 2),
                "total_corners": corners,
                "settled_at": "2026-09-05T21:00:00Z",
                "provider": "api-football",
            }
        )

    monkeypatch.setattr(
        daily_summary,
        "_resolve_daily_fixture_statuses",
        lambda *args, **kwargs: None,
    )

    sent = []
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: sent.append(message) or True,
    )

    first = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T21:00:00Z",
    )
    second = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T21:30:00Z",
    )

    assert first["sent"] is True
    assert first["reason"] == "sent"
    assert second["sent"] is False
    assert second["reason"] == "already_sent"
    assert len(sent) == 1
    assert "📊 CORNERLAB — RIEPILOGO GIORNALIERO" in sent[0]
    assert "Roma" in sent[0]
    assert "Atalanta" in sent[0]


def test_daily_summary_retries_if_telegram_send_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.operations import daily_summary

    config = daily_summary.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = daily_summary.CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "9201",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "FT",
            "provider": "api-football",
        }
    )

    repo.upsert_result(
        {
            "fixture_id": fixture["fixture_id"],
            "home_score": 1,
            "away_score": 0,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-05T21:00:00Z",
            "provider": "api-football",
        }
    )

    monkeypatch.setattr(
        daily_summary,
        "_resolve_daily_fixture_statuses",
        lambda *args, **kwargs: None,
    )

    responses = iter([False, True])
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: next(responses),
    )

    first = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T21:00:00Z",
    )
    second = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T21:30:00Z",
    )

    assert first["sent"] is False
    assert first["reason"] == "telegram_send_failed"
    assert second["sent"] is True
    assert second["reason"] == "sent"


def test_daily_summary_contains_real_daily_and_general_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.operations import daily_summary, real_bet_ledger

    config = daily_summary.CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = daily_summary.CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "9301",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "FT",
            "provider": "api-football",
        }
    )

    repo.upsert_result(
        {
            "fixture_id": fixture["fixture_id"],
            "home_score": 1,
            "away_score": 0,
            "home_corners": 5,
            "away_corners": 4,
            "total_corners": 9,
            "settled_at": "2026-09-05T21:00:00Z",
            "provider": "api-football",
        }
    )

    row = {
        "fixture_id": fixture["fixture_id"],
        "competition": "Serie A",
        "market": "TOTAL_CORNERS_UNDER",
        "side": "UNDER",
        "line": "11.5",
        "bookmaker": "bet365.it",
        "decision_timestamp": "2026-09-05T12:00:00Z",
        "home_team": "Roma",
        "away_team": "Atalanta",
        "recommended_stake": 5.0,
        "odds_at_decision": 2.0,
        "predicted_probability": 0.75,
        "EV": 0.50,
        "quality_tier": "TOP",
    }

    suggestion_id = real_bet_ledger.suggestion_key(row)
    real_bet_ledger.record_suggestion(tmp_path, row)
    confirmed = real_bet_ledger.confirm_bet(
        tmp_path,
        suggestion_id,
        actual_stake=5.0,
        actual_odds=2.0,
    )
    assert confirmed["ok"] is True

    settled = real_bet_ledger.settle_real_bet(
        tmp_path,
        suggestion_id,
        "WIN",
    )
    assert settled["ok"] is True

    monkeypatch.setattr(
        daily_summary,
        "_resolve_daily_fixture_statuses",
        lambda *args, **kwargs: None,
    )

    sent = []
    monkeypatch.setattr(
        daily_summary,
        "send_message",
        lambda message: sent.append(message) or True,
    )

    result = daily_summary.maybe_send_daily_summary(
        tmp_path,
        completed_at="2026-09-05T21:00:00Z",
    )

    assert result["sent"] is True
    assert len(sent) == 1

    message = sent[0]
    assert "Chiuse: 1" in message
    assert "WIN 1 • LOSS 0 • VOID 0" in message
    assert "Stake: €5.00" in message
    assert "P/L: €+5.00" in message
    assert "ROI: 100.0%" in message
    assert "Giocate chiuse: 1" in message
    assert "Win rate: 100.0%" in message
    assert "P/L realizzato: €+5.00" in message
