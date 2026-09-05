from __future__ import annotations
from datetime import datetime, timezone

from pathlib import Path
from urllib.parse import unquote_plus

import pandas as pd
import pytest

from src.operations import automation, monitoring, telegram_notifier
from src.operations.telegram_notifier import select_actionable_plays


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CORNERLAB_TELEGRAM_ENABLED", "true")
	monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
	monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")


def test_optional_configuration_disabled_or_missing_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CORNERLAB_TELEGRAM_ENABLED", raising=False)
	monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
	monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
	assert telegram_notifier.send_message("test") is False
	monkeypatch.setenv("CORNERLAB_TELEGRAM_ENABLED", "true")
	assert telegram_notifier.send_message("test") is False


def test_success_timeout_and_http_failure_are_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	calls = []
	assert telegram_notifier.send_message("ciao", request_sender=lambda url, payload, timeout: calls.append((url, payload, timeout))) is True
	assert calls[0][2] == 5.0
	assert "test-token" not in telegram_notifier.format_prematch_completed({}, "12:00")
	assert telegram_notifier.send_message("ciao", request_sender=lambda *_: (_ for _ in ()).throw(TimeoutError())) is False
	assert telegram_notifier.send_message("ciao", request_sender=lambda *_: (_ for _ in ()).throw(RuntimeError("http 500"))) is False


def test_play_notifications_use_canonical_records_and_deduplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	report = pd.DataFrame([
		{"fixture_id": 1, "competition": "Serie A", "decision": "PLAY", "target_name": "over_9_5", "market": "TOTAL_CORNERS_OVER", "side": "OVER", "line": "9.5", "bookmaker": "book", "decision_timestamp": "2026-08-15T12:00:00Z", "home_team": "Juventus", "away_team": "Atalanta", "odds_at_decision": 1.91, "predicted_probability": 0.75, "confidence_score": 70.0, "EV": 0.121, "quality_tier": "TOP", "recommended_stake": 10.0, "kickoff": "20:45"},
		{"fixture_id": 2, "competition": "Serie A", "decision": "NO BET", "target_name": "over_10_5", "market": "TOTAL_CORNERS_OVER", "side": "OVER", "line": "10.5", "bookmaker": "book", "decision_timestamp": "2026-08-15T12:00:00Z"},
		{"fixture_id": 3, "competition": "Serie A", "decision": "PLAY", "target_name": "over_8_5", "market": "TOTAL_CORNERS_OVER", "side": "OVER", "line": "8.5", "bookmaker": "book", "decision_timestamp": "2026-08-15T12:00:00Z", "predicted_probability": 0.76, "confidence_score": 71.0},
		{"fixture_id": 4, "competition": "Premier League", "decision": "PLAY", "target_name": "over_9_5", "market": "TOTAL_CORNERS_OVER", "side": "OVER", "line": "9.5", "bookmaker": "book", "decision_timestamp": "2026-08-15T12:00:00Z"},
	])
	messages = []

	first = telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda _, payload, __: messages.append(payload))
	second = telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda _, payload, __: messages.append(payload))

	assert first == 2
	assert second == 0
	assert len(messages) == 2
	assert any(b"Juventus" in message for message in messages)


def test_telegram_failure_never_blocks_prematch_or_settlement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(automation, "send_message", lambda *_: (_ for _ in ()).throw(RuntimeError("telegram down")))
	code, payload = automation._run_job("prematch", tmp_path, "slot", lambda: {"collector": {}, "settlement": {}})
	settlement_code, settlement_payload = automation._run_job("settlement", tmp_path, "settle", lambda: {"summary": {"total_bets": 1, "wins": 1, "losses": 0, "profit_loss": 1.0, "roi": 0.01}})

	assert code == 0 and payload["outcome"] == "SUCCESS"
	assert settlement_code == 0 and settlement_payload["outcome"] == "SUCCESS"


def test_monitoring_telegram_critical_and_recovery_are_transition_bound(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	sent = []
	monkeypatch.setattr(monitoring, "send_message", lambda text: sent.append(text) or False)
	monkeypatch.setattr(monitoring, "derive_operations_status", lambda *_args, **_kwargs: {"system_status": monitoring.FAILED, "updated_at": "2026-08-15T12:00:00Z", "last_error_summary": "quote failure", "last_prematch_success": "2026-08-15T11:00:00Z", "last_settlement_success": None})
	monitoring.refresh_operations_status(tmp_path)
	monitoring.refresh_operations_status(tmp_path)
	monkeypatch.setattr(monitoring, "derive_operations_status", lambda *_args, **_kwargs: {"system_status": monitoring.HEALTHY, "updated_at": "2026-08-15T12:10:00Z", "last_error_summary": None, "last_prematch_success": "2026-08-15T12:10:00Z", "last_settlement_success": None})
	monitoring.refresh_operations_status(tmp_path)

	assert len(sent) == 2
	assert "ERRORE" in sent[0]
	assert "RECOVERY" in sent[1]


def test_message_summaries_are_italian_and_settlement_requires_new_rows() -> None:
	prematch = telegram_notifier.format_prematch_completed({"collector": {"fixtures_fetched": 2, "odds_writes": 4}, "paper_trading": {"play_count": 1}}, "12:00")
	settlement = telegram_notifier.format_settlement_completed({"total_bets": 3, "wins": 2, "losses": 1, "voids": 0, "profit_loss": 14.6, "roi": 0.048}, "12:00")
	assert "PREMATCH COMPLETATO" in prematch
	assert "SETTLEMENT" in settlement
	assert "€+14.60" in settlement


def _play_row(fixture_id: int, side: str, line: str, competition: str = "Serie A", target_name: str | None = None) -> dict:
	return {
		"fixture_id": fixture_id,
		"competition": competition,
		"decision": "PLAY",
		"target_name": target_name or f"{'over' if side == 'OVER' else 'under'}_{line.replace('.', '_')}",
		"market": "TOTAL_CORNERS_OVER" if side == "OVER" else "TOTAL_CORNERS_UNDER",
		"side": side,
		"line": line,
		"bookmaker": "book",
		"decision_timestamp": "2026-08-15T12:00:00Z",
		"home_team": f"Home{fixture_id}",
		"away_team": f"Away{fixture_id}",
		"odds_at_decision": 2.0,
		"predicted_probability": 0.75,
		"confidence_score": 70.0,
		"EV": 0.1,
		"quality_tier": "TOP",
		"recommended_stake": 5.0,
		"kickoff": "20:45",
	}


def test_five_or_fewer_plays_use_one_alert_per_play(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	report = pd.DataFrame([_play_row(index, "OVER", "9.5") for index in range(1, 6)])
	messages = []

	sent = telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda _, payload, __: messages.append(payload))

	assert sent == 5
	assert len(messages) == 5
	assert all("VERIFICA BET365.IT" in unquote_plus(message.decode()) for message in messages)


def test_select_actionable_plays_keeps_multiple_model_candidates_for_same_fixture() -> None:
	rows = [
		_play_row(1, "UNDER", "10.5") | {
			"predicted_probability": 0.75,
			"confidence_score": 70.0,
			"kickoff_utc": "2026-08-31T18:45:00Z",
		},
		_play_row(1, "UNDER", "11.5") | {
			"predicted_probability": 0.82,
			"confidence_score": 72.0,
			"kickoff_utc": "2026-08-31T18:45:00Z",
		},
	]

	report = pd.DataFrame(rows)
	selected = telegram_notifier.select_actionable_plays(
		report,
		now=pd.Timestamp("2026-08-30T19:00:00Z").to_pydatetime(),
	)

	assert len(selected) == 2
	assert {row["line"] for row in selected} == {"10.5", "11.5"}


def test_grouped_candidate_message_uses_bet365_minimum_odds_not_external_price() -> None:
	rows = [
		_play_row(1, "UNDER", "10.5") | {
			"predicted_probability": 0.75,
			"confidence_score": 70.0,
			"odds_at_decision": 2.00,
			"EV": 0.50,
		},
		_play_row(1, "UNDER", "11.5") | {
			"predicted_probability": 0.82,
			"confidence_score": 72.0,
			"odds_at_decision": 1.80,
			"EV": 0.40,
		},
	]

	message = telegram_notifier.format_grouped_play(rows)

	assert "UNDER 10.5" in message
	assert "UNDER 11.5" in message
	assert "Quota minima bet365.it" in message
	assert "Quota:" not in message
	assert "EV:" not in message
	assert "2.00" not in message
	assert "1.80" not in message


def test_notify_new_plays_groups_all_model_candidates_for_same_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)

	rows = [
		_play_row(1, "UNDER", "10.5") | {
			"decision": "PLAY",
			"predicted_probability": 0.75,
			"confidence_score": 70.0,
		},
		_play_row(1, "UNDER", "11.5") | {
			"decision": "NO BET",
			"predicted_probability": 0.82,
			"confidence_score": 72.0,
		},
	]

	report = pd.DataFrame(rows)
	messages = []

	sent = telegram_notifier.notify_new_plays(
		tmp_path,
		report,
		request_sender=lambda _, payload, __: messages.append(payload),
	)

	assert sent == 2
	assert len(messages) == 1

	message = unquote_plus(messages[0].decode())
	assert "UNDER 10.5" in message
	assert "UNDER 11.5" in message
	assert "Quota minima bet365.it" in message


def test_more_than_five_plays_deduplicate_by_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	rows = [_play_row(1, "OVER", "9.5"), _play_row(1, "OVER", "10.5")]
	rows += [_play_row(index, "OVER", "9.5") for index in range(2, 7)]
	report = pd.DataFrame(rows)
	messages = []

	sent = telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda _, payload, __: messages.append(payload))

	assert sent == 7
	assert len(messages) == 6
	fixture_one_message = unquote_plus(next(message for message in messages if b"Home1" in message).decode())
	assert "OVER 9.5" in fixture_one_message
	assert "OVER 10.5" in fixture_one_message
	assert "VERIFICA BET365.IT" in fixture_one_message


def test_grouping_does_not_merge_different_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	rows = [_play_row(index, "OVER", "9.5") for index in range(1, 8)]
	report = pd.DataFrame(rows)
	messages = []

	telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda _, payload, __: messages.append(payload))

	assert len(messages) == 7
	homes = {index for index in range(1, 8) if any(f"Home{index}".encode() in message for message in messages)}
	assert homes == set(range(1, 8))


def test_dedup_remains_fixture_level_across_grouped_and_individual_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	rows = [_play_row(1, "OVER", "9.5"), _play_row(1, "OVER", "10.5")]
	rows += [_play_row(index, "OVER", "9.5") for index in range(2, 7)]
	report = pd.DataFrame(rows)
	telegram_notifier.notify_new_plays(tmp_path, report, request_sender=lambda *_: None)

	extra_row = pd.DataFrame([_play_row(1, "UNDER", "9.5")])
	messages = []
	sent = telegram_notifier.notify_new_plays(tmp_path, pd.concat([report, extra_row], ignore_index=True), request_sender=lambda _, payload, __: messages.append(payload))

	assert sent == 1
	assert len(messages) == 1
	assert "UNDER 9.5" in unquote_plus(messages[0].decode())


def test_rerun_emits_no_duplicate_play_or_prematch_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_configure(monkeypatch)
	report_path = tmp_path / "reports" / "paper_trading_current.csv"
	report_path.parent.mkdir(parents=True)
	pd.DataFrame([_play_row(1, "OVER", "9.5")]).to_csv(report_path, index=False)
	summary_calls = []
	monkeypatch.setattr(automation, "send_message", lambda text: summary_calls.append(text) or True)

	first_code, _ = automation._run_job("prematch", tmp_path, "slot", lambda: {"collector": {"fixtures_fetched": 1, "odds_writes": 1, "quota_remaining": 100}, "settlement": {"total_bets": 0}})
	second_code, second_payload = automation._run_job("prematch", tmp_path, "slot", lambda: {"collector": {"fixtures_fetched": 1, "odds_writes": 1, "quota_remaining": 100}, "settlement": {"total_bets": 0}})

	assert first_code == 0 and second_code == 0
	assert second_payload["outcome"] == "SKIPPED_IDEMPOTENT"
	# I run prematch ordinari restano silenziosi quando non ci sono
	# nuovi alert utili da inviare.
	assert summary_calls == []

def test_select_actionable_plays_deduplicates_same_fixture_side_line() -> None:
	now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)

	row = _play_row(29, "UNDER", "10.5")
	row["competition"] = "Serie A"
	row["target_name"] = "under_10_5"
	row["kickoff_utc"] = "2026-08-31T18:45:00Z"
	row["predicted_probability"] = 0.728944
	row["confidence_score"] = 68.247439

	report = pd.DataFrame([row, row.copy(), row.copy()])

	selected = select_actionable_plays(report, now=now)

	assert len(selected) == 1
	assert selected[0]["fixture_id"] == 29
	assert selected[0]["side"] == "UNDER"
	assert float(selected[0]["line"]) == 10.5

def test_select_actionable_plays_keeps_distinct_lines_same_fixture() -> None:
    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)

    row_105 = _play_row(29, "UNDER", "10.5")
    row_105["competition"] = "Serie A"
    row_105["target_name"] = "under_10_5"
    row_105["kickoff_utc"] = "2026-08-31T18:45:00Z"
    row_105["predicted_probability"] = 0.728944
    row_105["confidence_score"] = 68.247439

    row_115 = _play_row(29, "UNDER", "11.5")
    row_115["competition"] = "Serie A"
    row_115["target_name"] = "under_11_5"
    row_115["kickoff_utc"] = "2026-08-31T18:45:00Z"
    row_115["predicted_probability"] = 0.821587
    row_115["confidence_score"] = 68.247439

    report = pd.DataFrame([
        row_105,
        row_105.copy(),
        row_115,
        row_115.copy(),
    ])

    selected = select_actionable_plays(report, now=now)

    assert len(selected) == 2
    assert {float(row["line"]) for row in selected} == {10.5, 11.5}
