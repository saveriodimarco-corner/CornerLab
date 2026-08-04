from __future__ import annotations

import bz2
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.odds_matcher import OddsMatcher
from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider
from src.data.providers.odds.betfair_historical import BetfairHistoricalAudit, parse_betfair_stream_file
from src.data.providers.odds.provider_qualification import evaluate_provider_qualification, build_provider_comparison_matrix
from src.data.providers.odds.sportmonks_odds import SportmonksOddsProvider


def test_market_name_detection_and_goal_totals_rejection() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")
    raw = [{
        "name": "Over/Under Corners",
        "id": 1001,
        "bookmakers": [{"name": "Bet365", "bets": [{"id": 11, "name": "Corners Over/Under", "values": [{"value": 10.5, "odd": 1.91}]}]}],
    }]
    normalized = provider.normalize_odds(raw)
    assert not normalized.empty
    assert set(normalized["market"].tolist()) == {"TOTAL_CORNERS_OVER", "TOTAL_CORNERS_UNDER"}

    goal_raw = [{"name": "Over/Under Goals", "id": 2001}]
    assert provider.normalize_odds(goal_raw).empty


def test_over_under_normalization_and_fixture_matching() -> None:
    provider = SportmonksOddsProvider(api_token="token")
    raw = [{
        "market_id": 99,
        "market_name": "Total Corners",
        "bookmakers": [{"name": "Unibet", "odds": [{"name": "Over", "line": 9.5, "price": 1.95}, {"name": "Under", "line": 9.5, "price": 1.85}]}],
    }]
    normalized = provider.normalize_odds(raw)
    assert normalized.shape[0] == 2
    assert {row["side"] for _, row in normalized.iterrows()} == {"OVER", "UNDER"}

    matcher = OddsMatcher()
    fixtures = pd.DataFrame([{"match_id": 501, "home_team": "Juventus", "away_team": "Inter", "date": "2025-08-23T20:00:00Z"}])
    result = matcher.match_event_to_fixture(
        event={"home_team": "Juventus", "away_team": "Inter", "commence_time": "2025-08-23T20:05:00Z"},
        fixtures=fixtures,
        tolerance_minutes=30,
    )
    assert result["match_status"] == "MATCHED"


def test_missing_token_fails_locally_and_secrets_are_redacted() -> None:
    provider = ApiFootballOddsProvider(api_key="   ")
    with pytest.raises(ValueError, match="API_FOOTBALL_KEY"):
        provider.list_sports()

    redacted = provider._redact_sensitive({"token": "abc123", "detail": "abc123"})
    assert redacted["token"] == "***"
    assert redacted["detail"] == "***"


def test_verdict_rules() -> None:
    record = {
        "provider_name": "Test",
        "events_checked": 3,
        "events_with_odds": 3,
        "events_with_corner_markets": 3,
        "bookmakers_or_exchange_markets": 2,
        "lines_found": ["8.5", "10.5"],
        "over_available": True,
        "under_available": True,
        "opening_odds_available": True,
        "closing_odds_available": True,
        "historical_depth_available": True,
        "timestamp_history_available": True,
        "settlement_available": True,
        "fixture_mapping_rate": 1.0,
        "estimated_cost": "FREE",
        "limitations": "",
    }
    result = evaluate_provider_qualification(record)
    assert result["qualification_verdict"] == "QUALIFIED_FOR_HISTORICAL_BACKTEST"


def test_betfair_stream_parser_with_minimal_fixture(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text(json.dumps({"marketName": "Total Corners", "eventName": "Juventus vs Inter", "kickoffTime": "2025-08-23T20:00:00Z", "selections": [{"name": "Over 10.5", "lastTradedPrice": 2.0, "bestBack": 1.95, "bestLay": 2.05, "tradedVolume": 1200, "settlementStatus": "UNSETTLED", "timestamp": "2025-08-23T19:55:00Z"}]}) + "\n", encoding="utf-8")

    parsed = parse_betfair_stream_file(sample_path)
    assert parsed[0]["marketName"] == "Total Corners"
    assert parsed[0]["selections"][0]["bestBack"] == 1.95

    audit = BetfairHistoricalAudit()
    audit_result = audit.audit_sample_file(sample_path)
    assert audit_result["market_found_in_sample"] is True
    assert audit_result["market_theoretically_supported"] is True
    assert audit_result["closing_price_reconstructable"] is True


def test_deterministic_comparison_matrix() -> None:
    records = [
        {"provider_name": "Alpha", "qualification_verdict": "QUALIFIED_FOR_CURRENT_ODDS", "events_checked": 3, "events_with_odds": 3, "events_with_corner_markets": 2, "bookmakers_or_exchange_markets": 1, "lines_found": ["8.5", "10.5"], "over_available": True, "under_available": True, "opening_odds_available": True, "closing_odds_available": False, "historical_depth_available": False, "timestamp_history_available": False, "settlement_available": False, "fixture_mapping_rate": 0.75, "estimated_cost": "FREE", "limitations": ""},
        {"provider_name": "Beta", "qualification_verdict": "QUALIFIED_FOR_HISTORICAL_BACKTEST", "events_checked": 2, "events_with_odds": 2, "events_with_corner_markets": 2, "bookmakers_or_exchange_markets": 2, "lines_found": ["8.5", "9.5", "10.5"], "over_available": True, "under_available": True, "opening_odds_available": True, "closing_odds_available": True, "historical_depth_available": True, "timestamp_history_available": True, "settlement_available": True, "fixture_mapping_rate": 1.0, "estimated_cost": "PAID", "limitations": ""},
    ]
    matrix = build_provider_comparison_matrix(records)
    assert matrix[0]["provider_name"] == "Alpha"
    assert matrix[1]["provider_name"] == "Beta"
