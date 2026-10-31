from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.data.normalizer import normalize_team_name
from src.research.advanced_features import (
    build_advanced_feature_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://www.football-data.co.uk/mmz4281"

SEASONS = (
    ("2016/17", "1617"),
    ("2017/18", "1718"),
    ("2018/19", "1819"),
    ("2019/20", "1920"),
    ("2020/21", "2021"),
    ("2021/22", "2122"),
    ("2022/23", "2223"),
    ("2023/24", "2324"),
    ("2024/25", "2425"),
    ("2025/26", "2526"),
    ("2026/27", "2627"),
)


def stable_hash(row: pd.Series) -> str:
    payload = "|".join(
        str(row.get(name))
        for name in (
            "date",
            "season",
            "home_team",
            "away_team",
            "home_corners",
            "away_corners",
        )
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_season(
    season_label: str,
    season_code: str,
) -> tuple[pd.DataFrame, dict]:
    url = f"{BASE_URL}/{season_code}/E0.csv"

    with urlopen(url, timeout=30) as response:
        content = response.read()

    source_hash = hashlib.sha256(
        content
    ).hexdigest()

    raw = pd.read_csv(
        io.BytesIO(content),
        encoding="utf-8-sig",
    )

    required = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HC",
        "AC",
    ]

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]
    if missing:
        raise RuntimeError(
            f"Colonne mancanti {season_label}: "
            f"{missing}"
        )

    complete = raw.dropna(
        subset=required
    ).copy()

    expected = (
        380
        if season_label != "2026/27"
        else None
    )

    if expected is not None and len(complete) != expected:
        raise RuntimeError(
            f"Copertura non valida {season_label}: "
            f"{len(complete)}"
        )

    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(
                complete["Date"],
                dayfirst=True,
                errors="raise",
            ).dt.strftime("%Y-%m-%d"),
            "season": season_label,
            "competition": "Premier League",
            "home_team": complete[
                "HomeTeam"
            ].map(normalize_team_name),
            "away_team": complete[
                "AwayTeam"
            ].map(normalize_team_name),
            "home_goals": complete[
                "FTHG"
            ].astype(float).astype(int),
            "away_goals": complete[
                "FTAG"
            ].astype(float).astype(int),
            "home_corners": complete[
                "HC"
            ].astype(float).astype(int),
            "away_corners": complete[
                "AC"
            ].astype(float).astype(int),
        }
    )

    rows["total_corners"] = (
        rows["home_corners"]
        + rows["away_corners"]
    )
    rows["source"] = "football-data-csv"
    rows["source_file_name"] = (
        f"E0_{season_code}.csv"
    )
    rows["source_url"] = url
    rows["import_date"] = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "season": season_label,
        "url": url,
        "source_hash": source_hash,
        "rows": int(len(raw)),
        "complete_rows": int(len(rows)),
        "mean_corners": float(
            rows["total_corners"].mean()
        ),
    }

    return rows, metadata


def atomic_parquet(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_name(
        destination.name + ".tmp"
    )
    frame.to_parquet(
        temporary,
        index=False,
    )
    os.replace(temporary, destination)


def main() -> None:
    frames = []
    sources = []

    for season_label, season_code in SEASONS:
        frame, metadata = load_season(
            season_label,
            season_code,
        )
        frames.append(frame)
        sources.append(metadata)

    matches = pd.concat(
        frames,
        ignore_index=True,
    ).sort_values(
        ["season", "date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    matches["fixture_id"] = range(
        1,
        len(matches) + 1,
    )
    matches["row_hash"] = matches.apply(
        stable_hash,
        axis=1,
    )

    with tempfile.TemporaryDirectory() as name:
        stage = Path(name)
        processed = (
            stage
            / "data"
            / "processed"
            / "serie_a_matches.parquet"
        )
        processed.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        matches.to_parquet(
            processed,
            index=False,
        )

        advanced = build_advanced_feature_dataset(
            base_dir=stage,
            output_dir=stage,
        )

    cache_dir = (
        REPO_ROOT
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )

    matches_path = (
        cache_dir / "matches_extended.parquet"
    )
    features_path = (
        cache_dir
        / "advanced_features_extended.parquet"
    )

    atomic_parquet(matches, matches_path)
    atomic_parquet(advanced, features_path)

    result = {
        "run_type": (
            "premier_league_extended_dataset"
        ),
        "matches": int(len(matches)),
        "advanced_rows": int(len(advanced)),
        "season_counts": {
            str(key): int(value)
            for key, value in (
                matches.groupby("season")
                .size()
                .to_dict()
                .items()
            )
        },
        "sources": sources,
        "matches_path": str(matches_path),
        "features_path": str(features_path),
        "production_writes": False,
    }

    report_path = (
        REPO_ROOT
        / "reports"
        / "premier_league_extended_dataset.json"
    )
    report_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "matches": result["matches"],
                "advanced_rows": result[
                    "advanced_rows"
                ],
                "season_counts": result[
                    "season_counts"
                ],
                "features_path": result[
                    "features_path"
                ],
                "production_writes": False,
                "report_path": str(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
