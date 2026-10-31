from __future__ import annotations

import csv
import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.data.normalizer import normalize_team_name


COMPETITION = "Premier League"
CURRENT_SEASON = "2026/27"
CURRENT_SEASON_CODE = "2627"
DIVISION = "E0"
SOURCE_URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/2627/E0.csv"
)
REQUIRED_COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HC",
    "AC",
)
RESULT_COLUMNS = (
    "home_goals",
    "away_goals",
    "home_corners",
    "away_corners",
    "total_corners",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fixture_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("season", "")),
        str(row.get("date", "")),
        str(row.get("home_team", "")),
        str(row.get("away_team", "")),
    )


def _stable_hash(row: dict[str, Any]) -> str:
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
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_source(
    content: bytes,
    *,
    imported_at: datetime,
) -> pd.DataFrame:
    reader = csv.DictReader(
        io.StringIO(content.decode("utf-8-sig"))
    )
    missing = [
        name
        for name in REQUIRED_COLUMNS
        if name not in (reader.fieldnames or [])
    ]
    if missing:
        raise ValueError(
            f"Colonne Football-Data mancanti: {missing}"
        )

    rows: list[dict[str, Any]] = []
    imported = imported_at.strftime("%Y-%m-%d %H:%M:%S")

    for raw in reader:
        if not all(
            str(raw.get(name, "")).strip()
            for name in REQUIRED_COLUMNS
        ):
            continue

        row = {
            "date": datetime.strptime(
                str(raw["Date"]).strip(),
                "%d/%m/%Y",
            ).date().isoformat(),
            "season": CURRENT_SEASON,
            "competition": COMPETITION,
            "home_team": normalize_team_name(
                str(raw["HomeTeam"])
            ),
            "away_team": normalize_team_name(
                str(raw["AwayTeam"])
            ),
            "home_goals": int(float(raw["FTHG"])),
            "away_goals": int(float(raw["FTAG"])),
            "home_corners": int(float(raw["HC"])),
            "away_corners": int(float(raw["AC"])),
            "source": "football-data-csv",
            "source_file_name": (
                f"{DIVISION}_{CURRENT_SEASON_CODE}.csv"
            ),
            "source_url": SOURCE_URL,
            "import_date": imported,
        }
        row["total_corners"] = (
            row["home_corners"] + row["away_corners"]
        )
        row["row_hash"] = _stable_hash(row)
        rows.append(row)

    return pd.DataFrame(rows)


def _atomic_bytes(content: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def _atomic_parquet(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def sync_current_season_history(
    base_dir: Path | str,
    *,
    opener: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    opener = opener or urlopen
    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    matches_path = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
        / "matches_extended.parquet"
    )
    cache_path = (
        base_dir
        / "data"
        / "provider_cache"
        / "football_data"
        / f"{CURRENT_SEASON_CODE}_{DIVISION}.csv"
    )

    safety = {
        "model_refit": False,
        "model_artifact_modified": False,
        "production_database_writes": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    if not matches_path.exists():
        return {
            "status": "dataset_missing",
            "added_matches": 0,
            "corrected_matches": 0,
            **safety,
        }

    try:
        with opener(SOURCE_URL, timeout=30) as response:
            content = response.read()
    except (OSError, TimeoutError, URLError) as exc:
        return {
            "status": "download_failed",
            "error": str(exc),
            "added_matches": 0,
            "corrected_matches": 0,
            **safety,
        }

    source = _parse_source(
        content,
        imported_at=now.astimezone(timezone.utc),
    )
    existing = pd.read_parquet(matches_path)
    records = existing.to_dict("records")
    by_key = {
        _fixture_key(row): index
        for index, row in enumerate(records)
    }

    numeric_ids = pd.to_numeric(
        existing.get("fixture_id", pd.Series(dtype=float)),
        errors="coerce",
    )
    next_fixture_id = (
        int(numeric_ids.max()) + 1
        if numeric_ids.notna().any()
        else 1
    )
    added = 0
    corrected = 0

    for incoming in source.to_dict("records"):
        key = _fixture_key(incoming)
        index = by_key.get(key)

        if index is None:
            incoming["fixture_id"] = next_fixture_id
            next_fixture_id += 1
            records.append(incoming)
            by_key[key] = len(records) - 1
            added += 1
            continue

        current = records[index]
        changed = any(
            int(current.get(name)) != int(incoming.get(name))
            for name in RESULT_COLUMNS
        )
        if not changed:
            continue

        fixture_id = current.get("fixture_id")
        current.update(incoming)
        current["fixture_id"] = fixture_id
        corrected += 1

    _atomic_bytes(content, cache_path)

    if added or corrected:
        updated = pd.DataFrame(records).sort_values(
            ["season", "date", "home_team", "away_team"],
            kind="stable",
        ).reset_index(drop=True)
        _atomic_parquet(updated, matches_path)
    else:
        updated = existing

    return {
        "status": "updated",
        "source_completed_matches": int(len(source)),
        "previous_matches": int(len(existing)),
        "added_matches": added,
        "corrected_matches": corrected,
        "total_matches": int(len(updated)),
        "matches_path": str(matches_path),
        "cache_path": str(cache_path),
        **safety,
    }
