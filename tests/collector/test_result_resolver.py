from __future__ import annotations

from pathlib import Path

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.result_resolver import ResultResolver


def test_resolve_terminal_fixture_stores_canonical_corner_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "collector.sqlite",
        api_football_key="test-key",
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
            "status": "NS",
            "provider": "api-football",
        }
    )

    resolver = ResultResolver(config, repo)

    def fake_request(path, params=None):
        if path == "/fixtures":
            assert params == {"id": "1550112"}
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 1550112,
                            "status": {"short": "FT"},
                        },
                        "goals": {
                            "home": 1,
                            "away": 4,
                        },
                    }
                ]
            }

        if path == "/fixtures/statistics":
            assert params == {"fixture": "1550112"}
            return {
                "response": [
                    {
                        "team": {"name": "Genoa"},
                        "statistics": [
                            {"type": "Corner Kicks", "value": 5},
                        ],
                    },
                    {
                        "team": {"name": "Como"},
                        "statistics": [
                            {"type": "Corner Kicks", "value": 4},
                        ],
                    },
                ]
            }

        raise AssertionError(f"Unexpected request: {path} {params}")

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        fake_request,
    )

    resolved = resolver.resolve_fixture(fixture["fixture_id"])
    stored = repo.get_result(fixture["fixture_id"])
    refreshed_fixture = repo.get_fixture_by_id(fixture["fixture_id"])

    assert resolved["ok"] is True
    assert resolved["reason"] == "resolved"
    assert resolved["result"]["home_corners"] == 5
    assert resolved["result"]["away_corners"] == 4
    assert resolved["result"]["total_corners"] == 9

    assert stored is not None
    assert stored["home_score"] == 1
    assert stored["away_score"] == 4
    assert stored["home_corners"] == 5
    assert stored["away_corners"] == 4
    assert stored["total_corners"] == 9
    assert stored["provider"] == "api-football"

    assert refreshed_fixture is not None
    assert refreshed_fixture["status"] == "FT"


def test_non_terminal_fixture_stays_unresolved_and_does_not_fetch_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "2001",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T18:45:00Z",
            "home_team": "Roma",
            "away_team": "Atalanta",
            "status": "NS",
            "provider": "api-football",
        }
    )

    resolver = ResultResolver(config, repo)
    calls = []

    def fake_request(path, params=None):
        calls.append((path, params))
        if path == "/fixtures":
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 2001,
                            "status": {"short": "2H"},
                        },
                        "goals": {
                            "home": 1,
                            "away": 1,
                        },
                    }
                ]
            }

        raise AssertionError("Statistics must not be requested for non-terminal fixture")

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        fake_request,
    )

    resolved = resolver.resolve_fixture(fixture["fixture_id"])
    stored = repo.get_result(fixture["fixture_id"])
    refreshed_fixture = repo.get_fixture_by_id(fixture["fixture_id"])

    assert resolved["ok"] is False
    assert resolved["reason"] == "fixture_not_terminal"
    assert resolved["status"] == "2H"

    assert stored is None
    assert refreshed_fixture is not None
    assert refreshed_fixture["status"] == "2H"

    assert calls == [
        ("/fixtures", {"id": "2001"}),
    ]


def test_terminal_fixture_with_incomplete_corner_statistics_stays_unresolved(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "collector.sqlite",
        api_football_key="test-key",
    )
    repo = CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": "3001",
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-09-05T15:00:00Z",
            "home_team": "Fiorentina",
            "away_team": "Torino",
            "status": "NS",
            "provider": "api-football",
        }
    )

    resolver = ResultResolver(config, repo)

    def fake_request(path, params=None):
        if path == "/fixtures":
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 3001,
                            "status": {"short": "FT"},
                        },
                        "goals": {
                            "home": 2,
                            "away": 1,
                        },
                    }
                ]
            }

        if path == "/fixtures/statistics":
            return {
                "response": [
                    {
                        "team": {"name": "Fiorentina"},
                        "statistics": [
                            {"type": "Corner Kicks", "value": 6},
                        ],
                    },
                    {
                        "team": {"name": "Torino"},
                        "statistics": [
                            {"type": "Corner Kicks", "value": None},
                        ],
                    },
                ]
            }

        raise AssertionError(f"Unexpected request: {path} {params}")

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        fake_request,
    )

    resolved = resolver.resolve_fixture(fixture["fixture_id"])
    stored = repo.get_result(fixture["fixture_id"])
    refreshed_fixture = repo.get_fixture_by_id(fixture["fixture_id"])

    assert resolved["ok"] is False
    assert resolved["reason"] == "corner_statistics_incomplete"
    assert resolved["status"] == "FT"

    assert stored is None
    assert refreshed_fixture is not None
    assert refreshed_fixture["status"] == "FT"


def test_resolve_fixture_fails_closed_on_provider_fixture_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.providers.odds.api_football_odds import ApiFootballProviderError

    config = CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
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
            "status": "NS",
            "provider": "api-football",
        }
    )

    resolver = ResultResolver(config, repo)

    def fail_request(path, params=None):
        raise ApiFootballProviderError(
            "provider unavailable",
            category="PROVIDER REQUEST ERROR",
        )

    monkeypatch.setattr(resolver.api_football, "_perform_request", fail_request)

    result = resolver.resolve_fixture(fixture["fixture_id"])

    assert result["ok"] is False
    assert result["reason"] == "provider_fixture_request_failed"
    assert result["category"] == "PROVIDER REQUEST ERROR"
    assert repo.get_result(fixture["fixture_id"]) is None


def test_resolve_fixture_fails_closed_on_statistics_provider_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.providers.odds.api_football_odds import ApiFootballProviderError

    config = CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        api_football_key="test-key",
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
            "status": "NS",
            "provider": "api-football",
        }
    )

    resolver = ResultResolver(config, repo)

    calls = []

    def fake_request(path, params=None):
        calls.append(path)

        if path == "/fixtures":
            return {
                "response": [
                    {
                        "fixture": {
                            "status": {
                                "short": "FT",
                            }
                        },
                        "goals": {
                            "home": 1,
                            "away": 4,
                        },
                    }
                ]
            }

        if path == "/fixtures/statistics":
            raise ApiFootballProviderError(
                "statistics unavailable",
                category="PROVIDER REQUEST ERROR",
            )

        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        fake_request,
    )

    result = resolver.resolve_fixture(fixture["fixture_id"])

    assert calls == ["/fixtures", "/fixtures/statistics"]
    assert result["ok"] is False
    assert result["reason"] == "provider_statistics_request_failed"
    assert result["category"] == "PROVIDER REQUEST ERROR"
    assert repo.get_result(fixture["fixture_id"]) is None

    updated_fixture = repo.get_fixture_by_id(fixture["fixture_id"])
    assert updated_fixture["status"] == "FT"

def test_the_odds_api_fixture_uses_football_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        data_dir=tmp_path / "data",
        api_football_key="free-plan-key",
    )
    repo = CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": (
                "da159a6b84dbb548b4d7b94c722caac5"
            ),
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-10-29T19:45:00Z",
            "home_team": "Fiorentina",
            "away_team": "Atalanta BC",
            "status": "NS",
            "provider": "the-odds-api",
        }
    )

    resolver = ResultResolver(config, repo)

    monkeypatch.setattr(
        resolver.football_data,
        "resolve_fixture",
        lambda item: {
            "ok": True,
            "reason": "resolved",
            "status": "FT",
            "source_url": (
                "https://example.test/2627/I1.csv"
            ),
            "result": {
                "home_score": 3,
                "away_score": 1,
                "home_corners": 8,
                "away_corners": 2,
                "total_corners": 10,
                "provider": "football-data-csv",
            },
        },
    )

    def api_football_must_not_run(*args, **kwargs):
        raise AssertionError(
            "API-Football must not receive "
            "The Odds API fixture identifiers"
        )

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        api_football_must_not_run,
    )

    result = resolver.resolve_fixture(
        fixture["fixture_id"]
    )
    stored = repo.get_result(fixture["fixture_id"])
    refreshed = repo.get_fixture_by_id(
        fixture["fixture_id"]
    )

    assert result["ok"] is True
    assert result["status"] == "FT"
    assert result["result"]["total_corners"] == 10
    assert stored is not None
    assert stored["home_corners"] == 8
    assert stored["away_corners"] == 2
    assert stored["provider"] == "football-data-csv"
    assert refreshed is not None
    assert refreshed["status"] == "FT"


def test_the_odds_api_fixture_fails_closed_when_csv_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "data" / "collector.sqlite",
        data_dir=tmp_path / "data",
        api_football_key="free-plan-key",
    )
    repo = CollectorRepository(config)

    fixture = repo.upsert_fixture(
        {
            "provider_fixture_id": (
                "667cf69986b200753742c90cb9c5126c"
            ),
            "competition": "Serie A",
            "season": "2026",
            "kickoff_utc": "2026-10-29T17:30:00Z",
            "home_team": "Frosinone",
            "away_team": "Lecce",
            "status": "NS",
            "provider": "the-odds-api",
        }
    )

    resolver = ResultResolver(config, repo)

    monkeypatch.setattr(
        resolver.football_data,
        "resolve_fixture",
        lambda item: {
            "ok": False,
            "reason": "football_data_match_not_found",
        },
    )

    def api_football_must_not_run(*args, **kwargs):
        raise AssertionError(
            "API-Football fallback must remain disabled"
        )

    monkeypatch.setattr(
        resolver.api_football,
        "_perform_request",
        api_football_must_not_run,
    )

    result = resolver.resolve_fixture(
        fixture["fixture_id"]
    )

    assert result["ok"] is False
    assert (
        result["reason"]
        == "football_data_match_not_found"
    )
    assert repo.get_result(
        fixture["fixture_id"]
    ) is None
