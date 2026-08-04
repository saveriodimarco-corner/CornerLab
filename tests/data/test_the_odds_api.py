from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from src.data.odds_matcher import OddsMatcher
from src.data.providers.odds import MockOddsProvider, TheOddsApiProvider


class TestTheOddsApiProvider:
    def test_api_key_is_never_exposed(self) -> None:
        provider = TheOddsApiProvider(api_key="secret-key")
        redacted = provider._redact_sensitive({"api_key": "secret-key", "other": "value"})
        assert "secret-key" not in json.dumps(redacted)
        assert redacted["api_key"] == "***"

    def test_endpoint_construction(self) -> None:
        provider = TheOddsApiProvider(base_url="https://example.test")
        assert provider._build_url("sports") == "https://example.test/v4/sports"
        assert provider._build_url("sports/serie-a/events", sport="soccer_italy_serie_a") == "https://example.test/v4/sports/soccer_italy_serie_a/events"

    def test_normalize_rejects_goals_markets_and_retains_supported_lines(self) -> None:
        provider = TheOddsApiProvider(api_key="test-key")
        raw = [{
            "id": "evt-1",
            "sport_key": "soccer_italy_serie_a",
            "home_team": "Juventus",
            "away_team": "Inter",
            "commence_time": "2025-08-23T20:00:00Z",
            "bookmakers": [{
                "key": "bet365",
                "title": "Bet365",
                "markets": [{
                    "key": "alternate_totals_corners",
                    "outcomes": [
                        {"name": "Over", "price": 1.91, "point": 8.5},
                        {"name": "Under", "price": 1.95, "point": 8.5},
                        {"name": "Over", "price": 2.05, "point": 11.5},
                        {"name": "Under", "price": 1.80, "point": 11.5},
                        {"name": "Over", "price": 2.10, "point": 12.5},
                    ],
                }],
            }],
        }]
        normalized = provider.normalize_odds(raw)
        assert not normalized.empty
        assert set(normalized["line"].tolist()) == {"8.5", "11.5"}
        assert set(normalized["market"].tolist()) == {"TOTAL_CORNERS_OVER", "TOTAL_CORNERS_UNDER"}

    def test_normalize_rejects_goals_totals(self) -> None:
        provider = TheOddsApiProvider(api_key="test-key")
        raw = [{
            "id": "evt-2",
            "sport_key": "soccer_italy_serie_a",
            "home_team": "Roma",
            "away_team": "Lazio",
            "commence_time": "2025-08-23T20:00:00Z",
            "bookmakers": [{
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{
                    "key": "alternate_totals_goals",
                    "outcomes": [{"name": "Over", "price": 2.10, "point": 2.5}],
                }],
            }],
        }]
        normalized = provider.normalize_odds(raw)
        assert normalized.empty

    def test_fixture_matching_uses_normalized_teams_and_tolerance(self) -> None:
        matcher = OddsMatcher()
        fixtures = pd.DataFrame([
            {"match_id": 101, "home_team": "Juventus", "away_team": "Inter", "date": "2025-08-23T20:00:00Z"},
        ])
        result = matcher.match_event_to_fixture(
            event={"home_team": "juventus", "away_team": "inter", "commence_time": "2025-08-23T20:15:00Z"},
            fixtures=fixtures,
            tolerance_minutes=30,
        )
        assert result["match_status"] == "MATCHED"
        assert result["match_id"] == 101

    def test_ambiguous_fixture_matching_is_rejected(self) -> None:
        matcher = OddsMatcher()
        fixtures = pd.DataFrame([
            {"match_id": 101, "home_team": "Juventus", "away_team": "Inter", "date": "2025-08-23T20:00:00Z"},
            {"match_id": 102, "home_team": "Juventus", "away_team": "Inter", "date": "2025-08-23T20:30:00Z"},
        ])
        result = matcher.match_event_to_fixture(
            event={"home_team": "juventus", "away_team": "inter", "commence_time": "2025-08-23T20:15:00Z"},
            fixtures=fixtures,
            tolerance_minutes=30,
        )
        assert result["match_status"] == "AMBIGUOUS"

    def test_cache_prevents_duplicate_calls(self) -> None:
        provider = TheOddsApiProvider(api_key="test-key", cache_dir=Path("/tmp/the_odds_api_cache_test"))
        provider.cache_dir.mkdir(parents=True, exist_ok=True)
        provider._request_count = 0

        response_payload = {"status": "ok"}

        with patch.object(provider, "_perform_request", return_value=response_payload) as mocked:
            first = provider._get_json("/v4/sports", params={"foo": "bar"})
            second = provider._get_json("/v4/sports", params={"foo": "bar"})

        assert first == response_payload
        assert second == response_payload
        assert mocked.call_count == 1

    def test_prepared_request_includes_api_key_and_preserves_endpoint_params(self) -> None:
        provider = TheOddsApiProvider(api_key="test-key", base_url="https://example.test")
        params = {"regions": "eu", "bookmakers": "bet365", "oddsFormat": "decimal"}

        with patch.object(provider, "_perform_request", return_value=({}, {})) as mocked:
            provider._get_json("/v4/sports", params=params)

        request_kwargs = mocked.call_args.kwargs
        assert request_kwargs["params"]["apiKey"] == "test-key"
        assert request_kwargs["params"]["regions"] == "eu"
        assert request_kwargs["params"]["bookmakers"] == "bet365"
        assert request_kwargs["params"]["oddsFormat"] == "decimal"
        assert "Authorization" not in request_kwargs["headers"]

    def test_missing_key_fails_before_any_api_call(self) -> None:
        provider = TheOddsApiProvider(api_key="   ", base_url="https://example.test")

        with patch.object(provider, "_perform_request", side_effect=AssertionError("should not be called")) as mocked:
            with pytest.raises(ValueError, match="THE_ODDS_API_KEY"):
                provider._get_json("/v4/sports")

        assert mocked.call_count == 0

    def test_redaction_applies_to_sensitive_payloads_and_exceptions(self) -> None:
        provider = TheOddsApiProvider(api_key="super-secret", base_url="https://example.test")
        redacted = provider._redact_sensitive({"api_key": "super-secret", "detail": "value"})
        assert redacted["api_key"] == "***"
        assert redacted["detail"] == "value"

        with patch.object(provider, "_perform_request", side_effect=requests.RequestException("apiKey=super-secret")) as mocked:
            with pytest.raises(RuntimeError) as excinfo:
                provider._get_json("/v4/sports")

        assert "super-secret" not in str(excinfo.value)
        assert mocked.call_count == 1

    def test_rate_limit_headers_are_recorded(self) -> None:
        provider = TheOddsApiProvider(api_key="test-key")
        provider._usage = {
            "x-requests-remaining": 100,
            "x-requests-used": 20,
            "x-requests-last": 5,
        }
        usage = provider.get_usage()
        assert usage["x-requests-remaining"] == 100

    def test_mock_provider_matches_fixture_data(self) -> None:
        provider = MockOddsProvider()
        fixtures = pd.DataFrame([{"match_id": 7001, "home_team": "Juventus", "away_team": "Inter", "date": "2025-08-23T20:00:00Z"}])
        rows = provider.fetch_event_odds(fixtures=fixtures)
        assert not rows.empty
        assert rows.iloc[0]["source"] == "THE_ODDS_API"


@pytest.mark.parametrize("response", [
    [{"id": "evt-1", "bookmakers": [{"markets": [{"key": "alternate_totals_corners", "outcomes": [{"name": "Over", "price": 2.0, "point": 10.5}]}]}]}],
])
def test_normalization_is_deterministic(response) -> None:
    provider = TheOddsApiProvider(api_key="test-key")
    first = provider.normalize_odds(response)
    second = provider.normalize_odds(response)
    assert first.equals(second)
