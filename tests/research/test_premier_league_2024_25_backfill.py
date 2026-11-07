from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.premier_league_2024_25_backfill import (
    load_2024_25_fixtures,
)
from tests.research.test_premier_league_prior_season_backfill import (
    Response,
)


def test_loads_2024_25_fixture(
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
                "match_id": 3101,
                "season": "2024/25",
                "date": "2024-08-16",
                "home_team": "Man United",
                "away_team": "Fulham",
            }
        ]
    ).to_parquet(
        cache / "advanced_features_extended.parquet",
        index=False,
    )

    content = (
        "Date,Time,HomeTeam,AwayTeam,HC,AC\n"
        "16/08/2024,20:00,"
        "Man United,Fulham,5,3\n"
    ).encode("utf-8")

    rows = load_2024_25_fixtures(
        tmp_path,
        opener=lambda *args, **kwargs: Response(
            content
        ),
    )

    assert len(rows) == 1
    assert rows[0]["season"] == "2024/25"
    assert rows[0]["fixture_id"] == 3101
    assert rows[0]["kickoff_utc"] == (
        "2024-08-16T19:00:00+00:00"
    )
    assert rows[0]["actual_total_corners"] == 8
