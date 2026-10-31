from __future__ import annotations

from pathlib import Path

from scripts import run_premier_league_shadow


def test_pipeline_orders_isolated_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def collect(**kwargs):
        calls.append("collection")
        assert kwargs["base_dir"] == tmp_path
        assert kwargs["output_dir"] == tmp_path
        return {
            "provider_status": "ok",
            "fixtures_fetched": 21,
        }

    def settle(base_dir):
        calls.append("settlement")
        assert base_dir == tmp_path
        return {
            "settled_predictions_now": 0,
            "simulation_settled_now": 0,
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    def sync_history(base_dir):
        calls.append("history_sync")
        assert base_dir == tmp_path
        return {
            "status": "updated",
            "added_matches": 1,
            "model_refit": False,
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    def score(base_dir):
        calls.append("scoring")
        assert base_dir == tmp_path
        return {
            "observations": 32,
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    def notify(
        base_dir,
        *,
        dry_run,
    ):
        calls.append("notifications")
        assert base_dir == tmp_path
        assert dry_run is False
        return {
            "sent_notifications": 0,
            "telegram_simulation_only": True,
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    def notify_non_play(
        base_dir,
        *,
        dry_run,
    ):
        calls.append("non_play")
        assert base_dir == tmp_path
        assert dry_run is False
        return {
            "sent_dates": [],
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
        }

    monkeypatch.setattr(
        run_premier_league_shadow,
        "run_premier_league_shadow",
        collect,
    )
    monkeypatch.setattr(
        run_premier_league_shadow,
        "run_forward_settlement",
        settle,
    )
    monkeypatch.setattr(
        run_premier_league_shadow,
        "sync_current_season_history",
        sync_history,
    )
    monkeypatch.setattr(
        run_premier_league_shadow,
        "run_forward_scoring",
        score,
    )
    monkeypatch.setattr(
        run_premier_league_shadow,
        "notify_forward_price_checks",
        notify,
    )
    monkeypatch.setattr(
        run_premier_league_shadow,
        "notify_premier_non_play",
        notify_non_play,
    )

    result = (
        run_premier_league_shadow.run_pipeline(
            tmp_path
        )
    )

    assert calls == [
        "collection",
        "settlement",
        "history_sync",
        "scoring",
        "notifications",
        "non_play",
    ]
    assert result["mode"] == (
        "forward_validation_only"
    )
    assert result["history_sync"][
        "added_matches"
    ] == 1
    assert result["history_sync"][
        "model_refit"
    ] is False
    assert result[
        "non_play_notifications"
    ]["sent_dates"] == []
    assert (
        result["real_betting_enabled"]
        is False
    )
    assert (
        result["bankroll_real_enabled"]
        is False
    )
