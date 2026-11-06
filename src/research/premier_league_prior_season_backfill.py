from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from src.research.premier_league_historical_odds import (
    CACHE_RELATIVE,
    CATALOG_NAME,
    STATE_NAME,
    _fixture_key,
    _read_json,
    _team_key,
    run_historical_odds_backfill,
)


SEASON = "2025/26"
SEASON_CODE = "2526"
DIVISION = "E0"
SOURCE_URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/2526/E0.csv"
)
SOURCE_RELATIVE = Path(
    "data/provider_cache/football_data/2526_E0.csv"
)
MAX_TOTAL_CREDIT_BUDGET = 5000
CREDIT_RESERVE = 15000
REPORT_NAME = (
    "premier_league_historical_odds_2025_26.json"
)
LONDON = ZoneInfo("Europe/London")


def _atomic_bytes(
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_content(
    base_dir: Path,
    *,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    path = base_dir / SOURCE_RELATIVE

    if path.exists():
        return path.read_bytes()

    opener = opener or urlopen

    with opener(SOURCE_URL, timeout=30) as response:
        content = response.read()

    _atomic_bytes(path, content)
    return content


def load_prior_season_fixtures(
    base_dir: Path | str,
    *,
    opener: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    base_dir = Path(base_dir)
    content = _source_content(
        base_dir,
        opener=opener,
    )
    source = list(
        csv.DictReader(
            io.StringIO(
                content.decode("utf-8-sig")
            )
        )
    )

    features_path = (
        base_dir
        / CACHE_RELATIVE
        / "advanced_features_extended.parquet"
    )
    features = pd.read_parquet(features_path)
    prior = features.loc[
        features["season"].astype(str) == SEASON
    ].copy()

    ids: dict[tuple[str, str, str], Any] = {}

    for row in prior.to_dict("records"):
        date_value = pd.to_datetime(
            row["date"]
        ).date().isoformat()
        ids[
            (
                date_value,
                _team_key(row["home_team"]),
                _team_key(row["away_team"]),
            )
        ] = row.get("match_id")

    fixtures: list[dict[str, Any]] = []

    for raw in source:
        required = (
            "Date",
            "Time",
            "HomeTeam",
            "AwayTeam",
            "HC",
            "AC",
        )
        if not all(
            str(raw.get(name, "")).strip()
            for name in required
        ):
            continue

        local_kickoff = datetime.strptime(
            (
                f"{raw['Date'].strip()} "
                f"{raw['Time'].strip()}"
            ),
            "%d/%m/%Y %H:%M",
        ).replace(tzinfo=LONDON)
        kickoff = local_kickoff.astimezone(
            timezone.utc
        )
        date_value = local_kickoff.date().isoformat()
        home_team = str(raw["HomeTeam"]).strip()
        away_team = str(raw["AwayTeam"]).strip()
        home_corners = int(float(raw["HC"]))
        away_corners = int(float(raw["AC"]))
        lookup = (
            date_value,
            _team_key(home_team),
            _team_key(away_team),
        )

        row = {
            "date": date_value,
            "season": SEASON,
            "competition": "Premier League",
            "fixture_id": ids.get(lookup),
            "home_team": home_team,
            "away_team": away_team,
            "kickoff_utc": kickoff.isoformat(),
            "home_corners": home_corners,
            "away_corners": away_corners,
            "actual_total_corners": (
                home_corners + away_corners
            ),
        }
        row["fixture_key"] = _fixture_key(row)
        fixtures.append(row)

    return sorted(
        fixtures,
        key=lambda item: item["kickoff_utc"],
    )


def plan_prior_season_backfill(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    cache = base_dir / CACHE_RELATIVE
    fixtures = load_prior_season_fixtures(
        base_dir
    )
    state = _read_json(
        cache / STATE_NAME,
        {
            "total_credits_spent": 0,
            "fixtures": {},
        },
    )
    catalog = _read_json(
        cache / CATALOG_NAME,
        {"matchdays": {}},
    )
    terminal = {"complete", "no_odds"}

    pending = [
        fixture
        for fixture in fixtures
        if state.get("fixtures", {})
        .get(fixture["fixture_key"], {})
        .get("status")
        not in terminal
    ]
    missing_days = {
        fixture["date"]
        for fixture in pending
        if fixture["date"]
        not in catalog.get("matchdays", {})
    }

    return {
        "mode": "preview",
        "season": SEASON,
        "completed_matches": len(fixtures),
        "already_attempted": (
            len(fixtures) - len(pending)
        ),
        "pending_matches": len(pending),
        "missing_event_catalog_days": len(
            missing_days
        ),
        "estimated_new_credits": (
            len(pending) * 10
            + len(missing_days)
        ),
        "historical_credits_spent": int(
            state.get("total_credits_spent", 0)
        ),
        "maximum_total_credit_budget": (
            MAX_TOTAL_CREDIT_BUDGET
        ),
        "credit_reserve": CREDIT_RESERVE,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }


def run_prior_season_backfill(
    base_dir: Path | str,
    *,
    max_fixtures: int | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    fixtures = load_prior_season_fixtures(
        base_dir
    )
    result = run_historical_odds_backfill(
        base_dir,
        fixtures=fixtures,
        max_credit_budget=(
            MAX_TOTAL_CREDIT_BUDGET
        ),
        credit_reserve=CREDIT_RESERVE,
        max_fixtures=max_fixtures,
    )
    result["requested_season"] = SEASON
    result["real_betting_enabled"] = False
    result["bankroll_real_enabled"] = False

    report_path = (
        base_dir / "reports" / REPORT_NAME
    )
    _atomic_json(report_path, result)
    result["season_report_path"] = str(
        report_path
    )
    return result
