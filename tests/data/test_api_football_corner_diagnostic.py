from __future__ import annotations

import pytest

from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider, ApiFootballProviderError


def test_find_corner_bet_catalog_entries() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")
    payload = {
        "response": [
            {"id": 101, "name": "Over/Under Corners"},
            {"id": 102, "name": "Over/Under Goals"},
            {"id": 103, "name": "Alternative Corners"},
        ]
    }
    entries = provider.find_corner_bet_entries(payload)
    assert [entry["id"] for entry in entries] == [101, 103]
    assert [entry["name"] for entry in entries] == ["Over/Under Corners", "Alternative Corners"]


def test_extract_corner_odds_rows() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")
    payload = {
        "response": [
            {
                "bookmaker": {"name": "Bet365"},
                "bets": [
                    {
                        "id": 501,
                        "name": "Corners Over/Under",
                        "values": [
                            {"value": "8.5", "odd": "1.91"},
                            {"value": "9.5", "odd": "1.95"},
                        ],
                    }
                ],
            }
        ]
    }
    rows = provider.extract_corner_odds_rows(payload)
    assert len(rows) == 2
    assert rows[0]["bookmaker"] == "Bet365"
    assert rows[0]["bet_name"] == "Corners Over/Under"
    assert rows[0]["value"] == "8.5"


def test_list_upcoming_fixtures_raises_when_league_resolution_fails() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    with pytest.raises(RuntimeError, match="Unable to resolve Serie A fixtures"):
        provider.list_upcoming_fixtures(None, None, next_count=5)


def test_current_season_with_fixtures_uses_current_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2026, "current": True}]},
    })
    monkeypatch.setattr(provider, "list_upcoming_fixtures", lambda league_id, season, next_count=20, query_mode="NEXT": [{"fixture": {"id": 1, "date": "2030-01-01T20:00:00Z"}}] if season == 2026 else [])

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["requested_season"] == 2026
    assert result["effective_season"] == 2026
    assert result["fallback_used"] is False
    assert result["query_mode"] == "NEXT"
    assert len(result["fixtures"]) == 1
    assert result["collector_mode"] == "LIVE_COLLECTION READY"


def test_current_season_empty_previous_season_populated_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })
    monkeypatch.setattr(provider, "list_upcoming_fixtures", lambda league_id, season, next_count=20, query_mode="NEXT": [{"fixture": {"id": 2, "date": "2023-01-02T20:00:00Z"}}] if season == 2025 else [])

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["requested_season"] == 2026
    assert result["effective_season"] == 2025
    assert result["fallback_used"] is True
    assert result["query_mode"] == "LAST"
    assert result["fallback_reason"] == "current season returned zero fixtures"
    assert result["collector_mode"] == "HISTORICAL VALIDATION MODE"


def test_all_seasons_empty_returns_explicit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })
    monkeypatch.setattr(provider, "list_upcoming_fixtures", lambda league_id, season, next_count=20, query_mode="NEXT": [])

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["effective_season"] is None
    assert result["fallback_used"] is False
    assert result["query_mode"] == "LAST"
    assert result["collector_mode"] == "NO FIXTURES AVAILABLE"


def test_requested_and_effective_seasons_recorded_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2024, "current": False}, {"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })
    monkeypatch.setattr(provider, "list_upcoming_fixtures", lambda league_id, season, next_count=20, query_mode="NEXT": [{"fixture": {"id": 3, "date": "2023-01-03T20:00:00Z"}}] if season == 2025 else [])

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["requested_season"] == 2026
    assert result["effective_season"] == 2025
    assert result["fallback_used"] is True
    assert result["query_mode"] == "LAST"


def test_empty_response_without_errors_returns_no_fixtures_available(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })
    monkeypatch.setattr(provider, "list_upcoming_fixtures", lambda league_id, season, next_count=20, query_mode="NEXT": [])

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["collector_mode"] == "NO FIXTURES AVAILABLE"
    assert result["provider_response_category"] == "NO FIXTURES AVAILABLE"
    assert result["api_error_category"] is None


def test_plan_error_returns_provider_plan_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })

    def raising_fixture_request(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise ApiFootballProviderError("plan restriction", category="PROVIDER PLAN RESTRICTION")

    monkeypatch.setattr(provider, "list_upcoming_fixtures", raising_fixture_request)

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["collector_mode"] == "PROVIDER PLAN RESTRICTION"
    assert result["provider_response_category"] == "PROVIDER PLAN RESTRICTION"
    assert result["api_error_category"] == "PROVIDER PLAN RESTRICTION"
    assert result["recommended_action"]


def test_authentication_error_returns_provider_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })

    def raising_fixture_request(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise ApiFootballProviderError("invalid key", category="PROVIDER AUTHENTICATION ERROR")

    monkeypatch.setattr(provider, "list_upcoming_fixtures", raising_fixture_request)

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["collector_mode"] == "PROVIDER AUTHENTICATION ERROR"
    assert result["provider_response_category"] == "PROVIDER AUTHENTICATION ERROR"
    assert result["api_error_category"] == "PROVIDER AUTHENTICATION ERROR"


def test_provider_error_is_not_silently_converted_to_zero_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")

    monkeypatch.setattr(provider, "resolve_serie_a_league", lambda: {
        "league_id": 135,
        "season": 2026,
        "current": True,
        "coverage": {},
        "raw": {"league": {"id": 135, "name": "Serie A"}, "country": {"name": "Italy"}, "seasons": [{"year": 2025, "current": False}, {"year": 2026, "current": True}]},
    })

    def raising_fixture_request(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise ApiFootballProviderError("request failed", category="PROVIDER REQUEST ERROR")

    monkeypatch.setattr(provider, "list_upcoming_fixtures", raising_fixture_request)

    result = provider.resolve_live_fixtures(league_id=135, season=2026, next_count=5)

    assert result["collector_mode"] == "PROVIDER REQUEST ERROR"
    assert result["fixtures"] == []
    assert result["provider_response_category"] == "PROVIDER REQUEST ERROR"
