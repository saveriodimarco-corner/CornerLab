from __future__ import annotations

from pathlib import Path

from src.data.providers.football_data_results import (
    FootballDataResultsProvider,
)


CSV_CONTENT = b"""Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC
29/10/2026,Fiorentina,Atalanta,3,1,8,2
29/10/2026,Frosinone,Lecce,0,1,10,3
"""


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.content


def test_resolves_current_result_with_team_alias_and_reuses_cache(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def opener(url: str, timeout: int):
        calls.append((url, timeout))
        return FakeResponse(CSV_CONTENT)

    provider = FootballDataResultsProvider(
        tmp_path,
        timeout_seconds=11,
        opener=opener,
    )

    fixture = {
        "competition": "Serie A",
        "kickoff_utc": "2026-10-29T19:45:00Z",
        "home_team": "Fiorentina",
        "away_team": "Atalanta BC",
    }

    first = provider.resolve_fixture(fixture)
    second = provider.resolve_fixture(fixture)

    assert first["ok"] is True
    assert first["status"] == "FT"
    assert first["result"] == {
        "home_score": 3,
        "away_score": 1,
        "home_corners": 8,
        "away_corners": 2,
        "total_corners": 10,
        "provider": "football-data-csv",
    }
    assert second["ok"] is True
    assert len(calls) == 1
    assert calls[0][1] == 11
    assert calls[0][0].endswith("/2627/I1.csv")


def test_missing_match_fails_closed(
    tmp_path: Path,
) -> None:
    provider = FootballDataResultsProvider(
        tmp_path,
        opener=lambda url, timeout: FakeResponse(
            CSV_CONTENT
        ),
    )

    result = provider.resolve_fixture({
        "competition": "Serie A",
        "kickoff_utc": "2026-10-29T19:45:00Z",
        "home_team": "Juventus",
        "away_team": "Napoli",
    })

    assert result["ok"] is False
    assert (
        result["reason"]
        == "football_data_match_not_found"
    )
