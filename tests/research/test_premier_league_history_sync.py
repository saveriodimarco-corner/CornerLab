from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.research.premier_league_history_sync import (
    sync_current_season_history,
)


CSV_CONTENT = b"""Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC
25/10/2026,Arsenal,Chelsea,2,1,5,3
31/10/2026,Chelsea,Man United,1,1,7,4
01/11/2026,Liverpool,Arsenal,,,,,
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


def existing_row() -> dict:
    return {
        "date": "2026-10-25",
        "season": "2026/27",
        "competition": "Premier League",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "home_corners": 5,
        "away_corners": 3,
        "total_corners": 8,
        "source": "football-data-csv",
        "source_file_name": "E0_2627.csv",
        "source_url": "test",
        "import_date": "2026-10-25 22:00:00",
        "fixture_id": 3880,
        "row_hash": "existing",
    }


def test_sync_adds_completed_match_once_and_keeps_model_frozen(
    tmp_path: Path,
) -> None:
    cache_dir = (
        tmp_path
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    cache_dir.mkdir(parents=True)
    matches_path = cache_dir / "matches_extended.parquet"
    pd.DataFrame([existing_row()]).to_parquet(
        matches_path,
        index=False,
    )

    artifact_path = (
        cache_dir
        / "premier_league_forward_model.pkl"
    )
    artifact_path.write_bytes(b"frozen-model")
    artifact_hash = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()

    def opener(url: str, timeout: int):
        assert url.endswith("/2627/E0.csv")
        assert timeout == 30
        return FakeResponse(CSV_CONTENT)

    first = sync_current_season_history(
        tmp_path,
        opener=opener,
        now=datetime(
            2026,
            10,
            31,
            22,
            0,
            tzinfo=timezone.utc,
        ),
    )
    second = sync_current_season_history(
        tmp_path,
        opener=opener,
        now=datetime(
            2026,
            10,
            31,
            23,
            0,
            tzinfo=timezone.utc,
        ),
    )
    updated = pd.read_parquet(matches_path)

    assert first["added_matches"] == 1
    assert first["corrected_matches"] == 0
    assert second["added_matches"] == 0
    assert len(updated) == 2
    assert sorted(
        updated["fixture_id"].astype(int)
    ) == [3880, 3881]
    assert int(
        updated.loc[
            updated["date"] == "2026-10-31",
            "total_corners",
        ].iloc[0]
    ) == 11
    assert hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest() == artifact_hash
    assert first["model_refit"] is False
    assert first["real_betting_enabled"] is False


def test_download_failure_keeps_dataset_unchanged(
    tmp_path: Path,
) -> None:
    cache_dir = (
        tmp_path
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    cache_dir.mkdir(parents=True)
    matches_path = cache_dir / "matches_extended.parquet"
    pd.DataFrame([existing_row()]).to_parquet(
        matches_path,
        index=False,
    )
    before = matches_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("temporary outage")

    result = sync_current_season_history(
        tmp_path,
        opener=fail,
    )

    assert result["status"] == "download_failed"
    assert matches_path.read_bytes() == before
