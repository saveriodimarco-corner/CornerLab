from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.research.premier_league_prior_season_backfill import (
    load_prior_season_fixtures,
)


class Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.content


def test_loads_prior_fixture_with_london_timezone(
    tmp_path: Path,
) -> None:
    cache = (
        tmp_path
        / "data/provider_cache/"
        / "premier_league_validation"
    )
    cache.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "match_id": 3501,
                "season": "2025/26",
                "date": "2025-08-15",
                "home_team": "Man United",
                "away_team": "Arsenal",
            }
        ]
    ).to_parquet(
        cache / "advanced_features_extended.parquet",
        index=False,
    )

    content = (
        "Date,Time,HomeTeam,AwayTeam,HC,AC\n"
        "15/08/2025,20:00,"
        "Man United,Arsenal,5,4\n"
    ).encode("utf-8")

    rows = load_prior_season_fixtures(
        tmp_path,
        opener=lambda *args, **kwargs: Response(
            content
        ),
    )

    assert len(rows) == 1
    assert rows[0]["season"] == "2025/26"
    assert rows[0]["fixture_id"] == 3501
    assert rows[0]["kickoff_utc"] == (
        "2025-08-15T19:00:00+00:00"
    )
    assert rows[0]["actual_total_corners"] == 9
