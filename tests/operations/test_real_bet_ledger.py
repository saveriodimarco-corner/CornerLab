from __future__ import annotations

from pathlib import Path

import pytest

from src.operations import real_bet_ledger
from src.research.decision_engine import MAX_STAKE_FRACTION


def _play_row(fixture_id: int = 1, **overrides) -> dict:
	row = {
		"fixture_id": fixture_id,
		"competition": "Serie A",
		"market": "TOTAL_CORNERS_OVER",
		"side": "OVER",
		"line": "9.5",
		"bookmaker": "book",
		"decision_timestamp": "2026-08-15T12:00:00Z",
		"home_team": "Inter",
		"away_team": "Napoli",
		"recommended_stake": 5.0,
		"odds_at_decision": 1.92,
		"predicted_probability": 0.587,
		"EV": 0.121,
		"quality_tier": "TOP",
	}
	row.update(overrides)
	return row


def _count_ledger_events(base_dir: Path, event_type: str) -> int:
	import sqlite3

	conn = sqlite3.connect(base_dir / "data" / "operations" / "real_bets.sqlite")
	try:
		return int(conn.execute("SELECT COUNT(*) FROM bankroll_ledger WHERE event_type = ?", (event_type,)).fetchone()[0])
	finally:
		conn.close()


def test_suggested_play_does_not_affect_bankroll(tmp_path: Path) -> None:
	suggestion_id = real_bet_ledger.suggestion_key(_play_row())
	real_bet_ledger.record_suggestion(tmp_path, _play_row())

	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)
	bet = real_bet_ledger.get_bet(tmp_path, suggestion_id)

	assert snapshot["total_bankroll"] == 100.0
	assert snapshot["open_exposure"] == 0.0
	assert bet["status"] == real_bet_ledger.SUGGESTED


def test_confirm_suggested_stake_places_bet_and_reserves_exposure(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	result = real_bet_ledger.confirm_bet(tmp_path, suggestion_id)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["ok"] is True
	assert result["bet"]["status"] == real_bet_ledger.BET_PLACED
	assert result["bet"]["actual_stake"] == 5.0
	assert result["bet"]["actual_odds"] == 1.92
	assert snapshot["open_exposure"] == 5.0
	assert snapshot["available_bankroll"] == 95.0
	assert snapshot["total_bankroll"] == 100.0


def test_modify_stake_persists_actual_stake_and_keeps_suggested(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	result = real_bet_ledger.modify_stake(tmp_path, suggestion_id, 3.0)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["ok"] is True
	assert result["bet"]["actual_stake"] == 3.0
	assert result["bet"]["status"] == real_bet_ledger.SUGGESTED
	assert snapshot["open_exposure"] == 0.0
	assert snapshot["available_bankroll"] == 100.0


def test_modify_odds_persists_actual_odds_and_keeps_suggested(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	result = real_bet_ledger.modify_odds(tmp_path, suggestion_id, 2.05)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["ok"] is True
	assert result["bet"]["actual_odds"] == 2.05
	assert result["bet"]["suggested_odds"] == pytest.approx(1.92)
	assert result["bet"]["status"] == real_bet_ledger.SUGGESTED
	assert snapshot["open_exposure"] == 0.0
	assert snapshot["available_bankroll"] == 100.0


def test_edit_stake_then_odds_then_confirm_creates_single_placed_bet(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	real_bet_ledger.modify_stake(tmp_path, suggestion_id, 3.50)
	real_bet_ledger.modify_odds(tmp_path, suggestion_id, 2.10)
	result = real_bet_ledger.confirm_bet(tmp_path, suggestion_id)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)
	placed_events = _count_ledger_events(tmp_path, "BET_PLACED")

	assert result["bet"]["status"] == real_bet_ledger.BET_PLACED
	assert result["bet"]["actual_stake"] == 3.50
	assert result["bet"]["actual_odds"] == 2.10
	assert placed_events == 1
	assert snapshot["open_exposure"] == 3.50


def test_skip_after_edits_still_produces_skipped_without_bankroll_impact(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.modify_stake(tmp_path, suggestion_id, 3.50)
	real_bet_ledger.modify_odds(tmp_path, suggestion_id, 2.10)

	result = real_bet_ledger.skip_suggestion(tmp_path, suggestion_id)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["bet"]["status"] == real_bet_ledger.SKIPPED
	assert snapshot["open_exposure"] == 0.0
	assert snapshot["total_bankroll"] == 100.0


def test_skip_never_touches_bankroll_and_is_idempotent(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	first = real_bet_ledger.skip_suggestion(tmp_path, suggestion_id)
	second = real_bet_ledger.skip_suggestion(tmp_path, suggestion_id)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert first["ok"] is True
	assert second["ok"] is True
	assert first["bet"]["status"] == real_bet_ledger.SKIPPED
	assert snapshot["total_bankroll"] == 100.0


def test_duplicate_confirm_does_not_duplicate_bet_or_exposure(tmp_path: Path) -> None:
	row = _play_row()
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)

	first = real_bet_ledger.confirm_bet(tmp_path, suggestion_id)
	second = real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=99.0)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert first["ok"] is True
	assert second["reason"] == "already_processed"
	assert second["bet"]["actual_stake"] == 5.0
	assert snapshot["open_exposure"] == 5.0
	assert _count_ledger_events(tmp_path, "BET_PLACED") == 1


def test_deposit_increases_bankroll(tmp_path: Path) -> None:
	result = real_bet_ledger.record_deposit(tmp_path, 50.0)

	assert result["ok"] is True
	assert result["snapshot"]["total_bankroll"] == 150.0
	assert result["snapshot"]["available_bankroll"] == 150.0


def test_withdrawal_decreases_bankroll(tmp_path: Path) -> None:
	result = real_bet_ledger.record_withdrawal(tmp_path, 20.0)

	assert result["ok"] is True
	assert result["snapshot"]["total_bankroll"] == 80.0


def test_withdrawal_above_available_is_rejected(tmp_path: Path) -> None:
	result = real_bet_ledger.record_withdrawal(tmp_path, 150.0)
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert result["ok"] is False
	assert result["reason"] == "insufficient_available_bankroll"
	assert snapshot["total_bankroll"] == 100.0


def test_open_exposure_reduces_available_bankroll(tmp_path: Path) -> None:
	row = _play_row(fixture_id=1, recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=5.0)

	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert snapshot["total_bankroll"] == 100.0
	assert snapshot["open_exposure"] == 5.0
	assert snapshot["available_bankroll"] == 95.0


def test_stake_cap_applies_to_available_bankroll_not_total(tmp_path: Path) -> None:
	first_row = _play_row(fixture_id=1, recommended_stake=15.0)
	real_bet_ledger.record_suggestion(tmp_path, first_row)
	real_bet_ledger.confirm_bet(tmp_path, real_bet_ledger.suggestion_key(first_row), actual_stake=15.0)

	second_row = _play_row(fixture_id=2, recommended_stake=10.0)
	real_bet_ledger.record_suggestion(tmp_path, second_row)
	rejected = real_bet_ledger.confirm_bet(tmp_path, real_bet_ledger.suggestion_key(second_row), actual_stake=10.0)
	accepted = real_bet_ledger.confirm_bet(tmp_path, real_bet_ledger.suggestion_key(second_row), actual_stake=85.0 * MAX_STAKE_FRACTION)

	assert rejected["ok"] is False
	assert rejected["reason"] == "exceeds_stake_cap"
	assert accepted["ok"] is True
	assert accepted["bet"]["actual_stake"] == pytest.approx(85.0 * MAX_STAKE_FRACTION)


def test_settlement_uses_actual_stake_and_actual_odds_not_suggested(tmp_path: Path) -> None:
	row = _play_row(recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=4.20, actual_odds=2.00)

	result = real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "WIN")

	assert result["bet"]["profit_loss"] == pytest.approx(4.20 * (2.00 - 1.0))


def test_win_settlement_updates_bankroll_correctly(tmp_path: Path) -> None:
	row = _play_row(recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=5.0, actual_odds=2.0)

	real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "WIN")
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert snapshot["open_exposure"] == 0.0
	assert snapshot["total_bankroll"] == 105.0
	assert snapshot["realized_pnl"] == 5.0


def test_loss_settlement_updates_bankroll_correctly(tmp_path: Path) -> None:
	row = _play_row(recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=5.0, actual_odds=2.0)

	real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "LOSS")
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert snapshot["open_exposure"] == 0.0
	assert snapshot["total_bankroll"] == 95.0
	assert snapshot["realized_pnl"] == -5.0


def test_void_settlement_returns_stake_with_no_net_effect(tmp_path: Path) -> None:
	row = _play_row(recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=5.0, actual_odds=2.0)

	real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "VOID")
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert snapshot["open_exposure"] == 0.0
	assert snapshot["total_bankroll"] == 100.0
	assert snapshot["realized_pnl"] == 0.0


def test_settlement_is_idempotent(tmp_path: Path) -> None:
	row = _play_row(recommended_stake=5.0)
	suggestion_id = real_bet_ledger.suggestion_key(row)
	real_bet_ledger.record_suggestion(tmp_path, row)
	real_bet_ledger.confirm_bet(tmp_path, suggestion_id, actual_stake=5.0, actual_odds=2.0)

	real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "WIN")
	second = real_bet_ledger.settle_real_bet(tmp_path, suggestion_id, "WIN")
	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert second["reason"] == "not_placed_or_already_settled"
	assert snapshot["total_bankroll"] == 105.0


def test_suggested_or_skipped_bets_can_never_be_settled(tmp_path: Path) -> None:
	suggested_row = _play_row(fixture_id=10)
	real_bet_ledger.record_suggestion(tmp_path, suggested_row)
	suggested_result = real_bet_ledger.settle_real_bet(tmp_path, real_bet_ledger.suggestion_key(suggested_row), "WIN")

	skipped_row = _play_row(fixture_id=11)
	real_bet_ledger.record_suggestion(tmp_path, skipped_row)
	real_bet_ledger.skip_suggestion(tmp_path, real_bet_ledger.suggestion_key(skipped_row))
	skipped_result = real_bet_ledger.settle_real_bet(tmp_path, real_bet_ledger.suggestion_key(skipped_row), "WIN")

	snapshot = real_bet_ledger.get_bankroll_snapshot(tmp_path)

	assert suggested_result["ok"] is False
	assert skipped_result["ok"] is False
	assert snapshot["total_bankroll"] == 100.0
