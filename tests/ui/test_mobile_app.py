from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.ui.app import (
    DECISION_DISPLAY_MAP,
    DEFAULT_DECISION_FILTER,
    FULL_VIEW_LABEL,
    NO_PLAY_MESSAGE,
    QUALITY_INFO_TEXT,
    UI_LABELS,
    _add_play_quality,
    _apply_filters,
    _autoload_env,
    _load_history_rows,
    _prepare_play_cards,
    _prepare_dashboard_table,
    _verify_login,
)


def test_verify_login_uses_environment_password(monkeypatch) -> None:
    monkeypatch.setenv("CORNERLAB_APP_PASSWORD", "secret")
    assert _verify_login("secret") is True
    assert _verify_login("wrong") is False


def test_prepare_dashboard_table_computes_market_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Inter",
                "away_team": "Roma",
                "kickoff_utc": "2026-08-24T18:45:00Z",
                "line": "9.5",
                "side": "OVER",
                "bookmaker": "book-a",
                "closing_odds": 2.0,
                "predicted_probability": 0.6,
                "fair_odds": 1.6667,
                "ev": 0.2,
                "decision_confidence_score": 71.0,
                "recommended_stake": 5.0,
                "decision": "PLAY",
                "decision_reason": "POSITIVE_EV",
            }
        ]
    )

    table = _prepare_dashboard_table(frame)
    assert UI_LABELS["market_implied_probability"] in table.columns
    assert UI_LABELS["edge"] in table.columns
    assert UI_LABELS["decision"] in table.columns
    assert float(table.iloc[0][UI_LABELS["market_implied_probability"]]) == 0.5
    assert table.iloc[0][UI_LABELS["decision"]] == "GIOCA"


def test_apply_filters_supports_side_and_line_filters() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "side": "OVER", "line": "9.5"},
            {"decision": "NO BET", "side": "UNDER", "line": "10.5"},
        ]
    )

    play_only = _apply_filters(frame, "SOLO GIOCA", "TUTTI", "TUTTE", "TUTTE")
    assert len(play_only) == 1

    under_only = _apply_filters(frame, "TUTTI", "Under", "TUTTE", "TUTTE")
    assert len(under_only) == 1

    line_only = _apply_filters(frame, "TUTTI", "TUTTI", "9.5", "TUTTE")
    assert len(line_only) == 1


def test_default_decision_filter_is_solo_gioca() -> None:
    assert DEFAULT_DECISION_FILTER == "SOLO GIOCA"


def test_prepare_play_cards_returns_only_play_and_sorted() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Inter",
                "away_team": "Monza",
                "kickoff_utc": "2026-08-24T20:45:00Z",
                "side": "OVER",
                "line": "9.5",
                "bookmaker": "BetRivers",
                "closing_odds": 2.04,
                "predicted_probability": 0.785,
                "ev": 0.184,
                "decision_confidence_score": 67.0,
                "recommended_stake": 2.0,
                "decision": "PLAY",
                "fair_odds": 1.27,
                "target_name": "over_9_5",
                "snapshot_timestamp": "2026-08-13T12:00:00Z",
            },
            {
                "home_team": "Roma",
                "away_team": "Lazio",
                "kickoff_utc": "2026-08-24T18:45:00Z",
                "side": "UNDER",
                "line": "10.5",
                "bookmaker": "Bet365",
                "closing_odds": 2.20,
                "predicted_probability": 0.55,
                "ev": 0.10,
                "decision_confidence_score": 63.0,
                "recommended_stake": 1.2,
                "decision": "PLAY",
                "fair_odds": 1.82,
                "target_name": "over_10_5",
                "snapshot_timestamp": "2026-08-13T12:05:00Z",
            },
            {
                "home_team": "Milan",
                "away_team": "Napoli",
                "kickoff_utc": "2026-08-24T16:30:00Z",
                "side": "OVER",
                "line": "9.5",
                "bookmaker": "Bet365",
                "closing_odds": 1.90,
                "predicted_probability": 0.45,
                "ev": -0.05,
                "decision_confidence_score": 70.0,
                "recommended_stake": 0.0,
                "decision": "NO BET",
            },
        ]
    )

    cards = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")

    assert len(cards) == 2
    assert all(card["decisione"] == "PLAY" for card in cards)
    assert cards[0]["partita"] == "Inter - Monza"
    assert cards[0]["quota"] == 2.04
    assert cards[0]["probabilita_modello"] == 0.785
    assert cards[0]["valore_atteso"] == 0.184
    assert cards[0]["affidabilita"] == 67.0
    assert cards[0]["qualita"] in {"TOP", "BUONA", "MARGINALE"}


def test_no_play_state_helper_path_returns_empty_cards() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "NO BET", "side": "OVER", "line": "9.5"},
            {"decision": "LOW CONFIDENCE", "side": "UNDER", "line": "10.5"},
        ]
    )
    cards = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")
    assert cards == []
    assert NO_PLAY_MESSAGE == "Nessuna giocata consigliata al momento."


def test_every_play_gets_exactly_one_quality_tier() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.20, "decision_confidence_score": 62.0, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "predicted_probability": 0.6, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.40, "decision_confidence_score": 63.0, "side": "OVER", "line": "9.5", "home_team": "C", "away_team": "D", "bookmaker": "x", "closing_odds": 2.0, "predicted_probability": 0.7, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.60, "decision_confidence_score": 64.0, "side": "UNDER", "line": "10.5", "home_team": "E", "away_team": "F", "bookmaker": "y", "closing_odds": 2.2, "predicted_probability": 0.55, "kickoff_utc": "2026-08-24T18:45:00Z"},
        ]
    )
    cards = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")
    assert len(cards) == 3
    assert all(card["qualita"] in {"TOP", "BUONA", "MARGINALE"} for card in cards)


def test_quality_filter_operates_on_display_tier_only() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.20, "decision_confidence_score": 62.0, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "predicted_probability": 0.6, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.60, "decision_confidence_score": 64.0, "side": "UNDER", "line": "10.5", "home_team": "E", "away_team": "F", "bookmaker": "y", "closing_odds": 2.2, "predicted_probability": 0.55, "kickoff_utc": "2026-08-24T18:45:00Z"},
        ]
    )
    enriched = _add_play_quality(frame)
    expected_top = int((enriched["Qualità"] == "TOP").sum())
    top_cards = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TOP")
    assert len(top_cards) == expected_top
    assert all(card["qualita"] == "TOP" for card in top_cards)


def test_weak_slate_can_have_zero_top() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.06, "decision_confidence_score": 60.2, "predicted_probability": 0.61, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.07, "decision_confidence_score": 60.4, "predicted_probability": 0.62, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "x", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
            {"decision": "PLAY", "ev": 0.08, "decision_confidence_score": 60.6, "predicted_probability": 0.63, "side": "OVER", "line": "10.5", "home_team": "E", "away_team": "F", "bookmaker": "y", "closing_odds": 2.2, "kickoff_utc": "2026-08-24T20:45:00Z"},
        ]
    )

    quality = _add_play_quality(frame)
    play_quality = quality.loc[quality["decision"] == "PLAY", "Qualità"]
    assert (play_quality == "TOP").sum() == 0


def test_small_slate_uses_only_buona_marginale_without_clear_strength() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.09, "decision_confidence_score": 61.0, "predicted_probability": 0.62, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.10, "decision_confidence_score": 62.0, "predicted_probability": 0.63, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "y", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
            {"decision": "PLAY", "ev": 0.11, "decision_confidence_score": 63.0, "predicted_probability": 0.64, "side": "OVER", "line": "11.5", "home_team": "E", "away_team": "F", "bookmaker": "z", "closing_odds": 2.2, "kickoff_utc": "2026-08-24T20:45:00Z"},
            {"decision": "PLAY", "ev": 0.12, "decision_confidence_score": 64.0, "predicted_probability": 0.65, "side": "UNDER", "line": "9.5", "home_team": "G", "away_team": "H", "bookmaker": "k", "closing_odds": 2.3, "kickoff_utc": "2026-08-24T21:45:00Z"},
            {"decision": "PLAY", "ev": 0.13, "decision_confidence_score": 65.0, "predicted_probability": 0.66, "side": "OVER", "line": "10.5", "home_team": "I", "away_team": "L", "bookmaker": "m", "closing_odds": 2.4, "kickoff_utc": "2026-08-24T22:45:00Z"},
        ]
    )
    quality = _add_play_quality(frame)
    play_quality = quality.loc[quality["decision"] == "PLAY", "Qualità"]
    assert (play_quality == "TOP").sum() == 0
    assert set(play_quality.unique()).issubset({"BUONA", "MARGINALE"})


def test_highest_ev_alone_does_not_force_top() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.20, "decision_confidence_score": 60.0, "predicted_probability": 0.62, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.15, "decision_confidence_score": 75.0, "predicted_probability": 0.70, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "y", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
        ]
    )

    quality = _add_play_quality(frame)
    row_highest_ev = quality.loc[(quality["home_team"] == "A") & (quality["away_team"] == "B")].iloc[0]
    assert row_highest_ev["Qualità"] != "TOP"


def test_varied_slate_produces_multiple_tiers() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.05, "decision_confidence_score": 60.0, "predicted_probability": 0.60, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.07, "decision_confidence_score": 61.0, "predicted_probability": 0.61, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "y", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
            {"decision": "PLAY", "ev": 0.09, "decision_confidence_score": 62.0, "predicted_probability": 0.62, "side": "OVER", "line": "11.5", "home_team": "E", "away_team": "F", "bookmaker": "z", "closing_odds": 2.2, "kickoff_utc": "2026-08-24T20:45:00Z"},
            {"decision": "PLAY", "ev": 0.12, "decision_confidence_score": 68.0, "predicted_probability": 0.66, "side": "UNDER", "line": "9.5", "home_team": "G", "away_team": "H", "bookmaker": "k", "closing_odds": 2.3, "kickoff_utc": "2026-08-24T21:45:00Z"},
            {"decision": "PLAY", "ev": 0.18, "decision_confidence_score": 72.0, "predicted_probability": 0.69, "side": "OVER", "line": "10.5", "home_team": "I", "away_team": "L", "bookmaker": "m", "closing_odds": 2.4, "kickoff_utc": "2026-08-24T22:45:00Z"},
            {"decision": "PLAY", "ev": 0.22, "decision_confidence_score": 76.0, "predicted_probability": 0.72, "side": "UNDER", "line": "11.5", "home_team": "M", "away_team": "N", "bookmaker": "n", "closing_odds": 2.5, "kickoff_utc": "2026-08-24T23:45:00Z"},
        ]
    )
    quality = _add_play_quality(frame)
    play_quality = quality.loc[quality["decision"] == "PLAY", "Qualità"]
    assert len(set(play_quality.unique())) >= 2


def test_current_report_play_rows_do_not_collapse_to_one_tier() -> None:
    report_path = Path("reports/paper_trading_current.csv")
    frame = pd.read_csv(report_path)
    quality = _add_play_quality(frame)
    play_quality = quality.loc[quality["decision"] == "PLAY", "Qualità"]
    play_count = len(play_quality)
    assert play_count >= 10
    assert len(set(play_quality.unique())) > 1


def test_quality_assignment_is_deterministic_for_identical_input() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.10, "decision_confidence_score": 65.0, "predicted_probability": 0.63, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.16, "decision_confidence_score": 72.0, "predicted_probability": 0.68, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "y", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
            {"decision": "PLAY", "ev": 0.08, "decision_confidence_score": 61.0, "predicted_probability": 0.61, "side": "OVER", "line": "11.5", "home_team": "E", "away_team": "F", "bookmaker": "z", "closing_odds": 2.2, "kickoff_utc": "2026-08-24T20:45:00Z"},
        ]
    )

    first = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")
    second = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")
    assert first == second


def test_backend_values_remain_untouched_by_quality_layer() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.11, "decision_confidence_score": 66.0, "predicted_probability": 0.64, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "kickoff_utc": "2026-08-24T18:45:00Z"},
            {"decision": "PLAY", "ev": 0.09, "decision_confidence_score": 62.0, "predicted_probability": 0.61, "side": "UNDER", "line": "10.5", "home_team": "C", "away_team": "D", "bookmaker": "y", "closing_odds": 2.1, "kickoff_utc": "2026-08-24T19:45:00Z"},
        ]
    )
    before = frame.copy(deep=True)

    _ = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")

    assert frame.equals(before)
    assert "Qualità" not in frame.columns


def test_mobile_quality_info_text_present() -> None:
    assert "TOP non significa giocata certa" in QUALITY_INFO_TEXT


def test_ev_and_confidence_values_are_preserved_in_cards() -> None:
    frame = pd.DataFrame(
        [
            {"decision": "PLAY", "ev": 0.31, "decision_confidence_score": 65.4, "side": "OVER", "line": "9.5", "home_team": "A", "away_team": "B", "bookmaker": "x", "closing_odds": 2.0, "predicted_probability": 0.6, "kickoff_utc": "2026-08-24T18:45:00Z"},
        ]
    )
    cards = _prepare_play_cards(frame, side_filter="TUTTI", line_filter="TUTTE", quality_filter="TUTTE")
    assert len(cards) == 1
    assert cards[0]["valore_atteso"] == 0.31
    assert cards[0]["affidabilita"] == 65.4


def test_desktop_full_view_label_available() -> None:
    assert FULL_VIEW_LABEL == "Vista completa"


def test_decision_display_mapping_is_presentation_only() -> None:
    frame = pd.DataFrame(
        [
            {"home_team": "Inter", "away_team": "Roma", "kickoff_utc": "2026-08-24T18:45:00Z", "line": "9.5", "side": "OVER", "bookmaker": "book-a", "closing_odds": 2.0, "predicted_probability": 0.6, "fair_odds": 1.6, "ev": 0.2, "decision_confidence_score": 70.0, "recommended_stake": 2.0, "decision": "PLAY", "decision_reason": "POSITIVE_EV"},
            {"home_team": "Inter", "away_team": "Roma", "kickoff_utc": "2026-08-24T18:45:00Z", "line": "9.5", "side": "UNDER", "bookmaker": "book-a", "closing_odds": 2.0, "predicted_probability": 0.4, "fair_odds": 2.5, "ev": -0.2, "decision_confidence_score": 70.0, "recommended_stake": 0.0, "decision": "NO BET", "decision_reason": "NON_POSITIVE_EV"},
        ]
    )

    source_decisions = frame["decision"].tolist()
    table = _prepare_dashboard_table(frame)

    assert DECISION_DISPLAY_MAP["PLAY"] in table[UI_LABELS["decision"]].tolist()
    assert DECISION_DISPLAY_MAP["NO BET"] in table[UI_LABELS["decision"]].tolist()
    assert frame["decision"].tolist() == source_decisions


def test_italian_labels_constants_match_required_terms() -> None:
    assert UI_LABELS["history"] == "Storico"
    assert UI_LABELS["update_prematch"] == "AGGIORNA PRE-PARTITA"
    assert UI_LABELS["last_update"] == "Ultimo aggiornamento"
    assert UI_LABELS["system_health"] == "Stato sistema"
    assert UI_LABELS["odds_provider"] == "Provider quote"
    assert UI_LABELS["decision_filter"] == "Filtro decisione"
    assert UI_LABELS["side"] == "Esito"
    assert UI_LABELS["line"] == "Linea"


def test_history_loader_reads_jsonl_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text(
        '{"run_id":"r1","fixtures_evaluated":3}\n'
        '{"run_id":"r2","fixtures_evaluated":5}\n',
        encoding="utf-8",
    )

    rows = _load_history_rows(history_path)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "r1"


def test_autoload_env_reads_password_from_env_file(tmp_path: Path, monkeypatch, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CORNERLAB_APP_PASSWORD=file-secret\n", encoding="utf-8")
    monkeypatch.delenv("CORNERLAB_APP_PASSWORD", raising=False)

    loaded = _autoload_env(env_file)

    assert loaded is True
    assert _verify_login("file-secret") is True
    captured = capsys.readouterr()
    assert "file-secret" not in captured.out
    assert "file-secret" not in captured.err


def test_autoload_env_does_not_override_existing_environment(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CORNERLAB_APP_PASSWORD=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("CORNERLAB_APP_PASSWORD", "env-secret")

    loaded = _autoload_env(env_file)

    assert loaded is True
    assert _verify_login("env-secret") is True
    assert _verify_login("file-secret") is False
