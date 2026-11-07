from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.collector.collector_config import CollectorConfig
from src.data.providers.odds.the_odds_api import (
    TheOddsApiProvider,
)


COMPETITION = "Premier League"
SEASON = "2026/27"
SPORT_KEY = "soccer_epl"
REGIONS = "us"
MARKET = "alternate_totals_corners"
ELIGIBLE_LINES = {8.5, 9.5, 10.5, 11.5}

DEFAULT_MAX_CREDIT_BUDGET = 1000
DEFAULT_CREDIT_RESERVE = 15000
EVENT_CATALOG_COST = 1
EVENT_ODDS_COST = 10

LONDON = ZoneInfo("Europe/London")

CACHE_RELATIVE = Path(
    "data/provider_cache/premier_league_validation"
)
SOURCE_RELATIVE = Path(
    "data/provider_cache/football_data/2627_E0.csv"
)
ODDS_NAME = "historical_corner_odds.parquet"
STATE_NAME = "historical_corner_odds_state.json"
CATALOG_NAME = "historical_event_catalog.json"
REPORT_NAME = "premier_league_historical_odds_backfill.json"

TEAM_ALIASES = {
    "brighton and hove albion": "brighton",
    "coventry city": "coventry",
    "hull city": "hull",
    "ipswich town": "ipswich",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "nottingham forest": "nott m forest",
    "notts forest": "nott m forest",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _team_key(value: Any) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value or "").casefold(),
    )
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return TEAM_ALIASES.get(text, text)


def _fixture_key(row: dict[str, Any]) -> str:
    payload = "|".join(
        (
            str(row["date"]),
            _team_key(row["home_team"]),
            _team_key(row["away_team"]),
        )
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:24]


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(
    path: Path,
    default: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return default
    return value if isinstance(value, dict) else default


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


def _atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_completed_fixtures(
    base_dir: Path,
) -> list[dict[str, Any]]:
    csv_path = base_dir / SOURCE_RELATIVE
    matches_path = (
        base_dir
        / CACHE_RELATIVE
        / "matches_extended.parquet"
    )

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not matches_path.exists():
        raise FileNotFoundError(matches_path)

    matches = pd.read_parquet(matches_path)
    current = matches.loc[
        matches["season"].astype(str) == SEASON
    ].copy()

    fixture_ids: dict[
        tuple[str, str, str],
        Any,
    ] = {}

    for row in current.to_dict("records"):
        date_value = pd.to_datetime(
            row["date"]
        ).date().isoformat()
        key = (
            date_value,
            _team_key(row["home_team"]),
            _team_key(row["away_team"]),
        )
        fixture_ids[key] = row.get("fixture_id")

    source_rows = list(
        csv.DictReader(
            csv_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        )
    )
    fixtures: list[dict[str, Any]] = []

    for raw in source_rows:
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
            "competition": COMPETITION,
            "fixture_id": fixture_ids.get(lookup),
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


def _usage(
    provider: Any,
    *,
    fallback_cost: int,
    previous_remaining: int | None,
) -> tuple[int, int | None]:
    usage = dict(provider.get_usage() or {})
    cost = _integer(
        usage.get("x-requests-last")
    )
    if cost is None:
        cost = fallback_cost

    remaining = _integer(
        usage.get("x-requests-remaining")
    )
    if remaining is None and previous_remaining is not None:
        remaining = max(
            0,
            previous_remaining - cost,
        )

    return max(0, cost), remaining


def _can_spend(
    *,
    total_spent: int,
    expected_cost: int,
    maximum_budget: int,
    remaining: int | None,
    reserve: int,
) -> bool:
    if total_spent + expected_cost > maximum_budget:
        return False
    if remaining is None:
        return False
    return remaining - expected_cost >= reserve


def _event_for_fixture(
    events: list[dict[str, Any]],
    fixture: dict[str, Any],
) -> dict[str, Any] | None:
    kickoff = datetime.fromisoformat(
        fixture["kickoff_utc"]
    )

    for event in events:
        if (
            _team_key(event.get("home_team"))
            != _team_key(fixture["home_team"])
            or _team_key(event.get("away_team"))
            != _team_key(fixture["away_team"])
        ):
            continue

        raw = str(
            event.get("commence_time") or ""
        ).replace("Z", "+00:00")
        try:
            event_kickoff = datetime.fromisoformat(raw)
        except ValueError:
            continue

        if event_kickoff.tzinfo is None:
            event_kickoff = event_kickoff.replace(
                tzinfo=timezone.utc
            )

        if abs(
            (event_kickoff - kickoff).total_seconds()
        ) <= 36 * 3600:
            return event

    return None


def _extract_odds_rows(
    payload: Any,
    *,
    fixture: dict[str, Any],
    event_id: str,
    requested_snapshot: str,
) -> list[dict[str, Any]]:
    returned_snapshot = None
    event = payload

    if isinstance(payload, dict) and "data" in payload:
        returned_snapshot = payload.get("timestamp")
        event = payload.get("data")

    if isinstance(event, list):
        event = next(
            (
                item
                for item in event
                if str(item.get("id")) == event_id
            ),
            event[0] if event else {},
        )

    if not isinstance(event, dict):
        return []

    total = int(fixture["actual_total_corners"])
    rows: list[dict[str, Any]] = []

    for bookmaker in event.get("bookmakers", []):
        bookmaker_name = str(
            bookmaker.get("title")
            or bookmaker.get("key")
            or "unknown"
        )
        bookmaker_key = str(
            bookmaker.get("key")
            or bookmaker_name
        )

        for market in bookmaker.get("markets", []):
            if market.get("key") != MARKET:
                continue

            last_update = (
                market.get("last_update")
                or bookmaker.get("last_update")
                or returned_snapshot
                or requested_snapshot
            )

            for outcome in market.get("outcomes", []):
                try:
                    line = float(outcome.get("point"))
                    decimal_odds = float(
                        outcome.get("price")
                    )
                except (TypeError, ValueError):
                    continue

                side = str(
                    outcome.get("name") or ""
                ).upper()

                if (
                    line not in ELIGIBLE_LINES
                    or side not in {"OVER", "UNDER"}
                    or decimal_odds <= 1.0
                    or decimal_odds > 10.0
                ):
                    continue

                won = (
                    total > line
                    if side == "OVER"
                    else total < line
                )

                rows.append(
                    {
                        "fixture_key": fixture[
                            "fixture_key"
                        ],
                        "fixture_id": fixture.get(
                            "fixture_id"
                        ),
                        "competition": COMPETITION,
                        "season": str(
                            fixture.get("season") or SEASON
                        ),
                        "date": fixture["date"],
                        "kickoff_utc": fixture[
                            "kickoff_utc"
                        ],
                        "home_team": fixture[
                            "home_team"
                        ],
                        "away_team": fixture[
                            "away_team"
                        ],
                        "provider_event_id": event_id,
                        "requested_snapshot": (
                            requested_snapshot
                        ),
                        "returned_snapshot": (
                            returned_snapshot
                        ),
                        "odds_timestamp": last_update,
                        "bookmaker": bookmaker_name,
                        "bookmaker_key": bookmaker_key,
                        "market": MARKET,
                        "line": line,
                        "side": side,
                        "decimal_odds": decimal_odds,
                        "home_corners": fixture[
                            "home_corners"
                        ],
                        "away_corners": fixture[
                            "away_corners"
                        ],
                        "actual_total_corners": total,
                        "won": bool(won),
                        "provider": "the-odds-api",
                    }
                )

    return rows


def plan_historical_odds_backfill(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    cache_dir = base_dir / CACHE_RELATIVE
    state = _read_json(
        cache_dir / STATE_NAME,
        {
            "version": 1,
            "total_credits_spent": 0,
            "fixtures": {},
        },
    )
    catalog = _read_json(
        cache_dir / CATALOG_NAME,
        {
            "version": 1,
            "matchdays": {},
        },
    )
    fixtures = _load_completed_fixtures(base_dir)
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
    estimate = (
        len(missing_days) * EVENT_CATALOG_COST
        + len(pending) * EVENT_ODDS_COST
    )

    return {
        "mode": "preview",
        "completed_matches": len(fixtures),
        "already_attempted": len(fixtures) - len(pending),
        "pending_matches": len(pending),
        "missing_event_catalog_days": len(missing_days),
        "estimated_credits": estimate,
        "historical_credits_spent": int(
            state.get("total_credits_spent", 0)
        ),
        "maximum_credit_budget": (
            DEFAULT_MAX_CREDIT_BUDGET
        ),
        "credit_reserve": DEFAULT_CREDIT_RESERVE,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }


def run_historical_odds_backfill(
    base_dir: Path | str,
    *,
    provider: Any | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    max_credit_budget: int = (
        DEFAULT_MAX_CREDIT_BUDGET
    ),
    credit_reserve: int = DEFAULT_CREDIT_RESERVE,
    max_fixtures: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    now = now or _utc_now()
    cache_dir = base_dir / CACHE_RELATIVE
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    state_path = cache_dir / STATE_NAME
    catalog_path = cache_dir / CATALOG_NAME
    odds_path = cache_dir / ODDS_NAME
    report_path = reports_dir / REPORT_NAME

    state = _read_json(
        state_path,
        {
            "version": 1,
            "total_credits_spent": 0,
            "fixtures": {},
        },
    )
    catalog = _read_json(
        catalog_path,
        {
            "version": 1,
            "matchdays": {},
        },
    )
    state.setdefault("fixtures", {})
    state.setdefault("total_credits_spent", 0)
    catalog.setdefault("matchdays", {})

    fixtures = fixtures or _load_completed_fixtures(
        base_dir
    )
    fixtures = sorted(
        fixtures,
        key=lambda item: item["kickoff_utc"],
    )

    for fixture in fixtures:
        fixture.setdefault(
            "fixture_key",
            _fixture_key(fixture),
        )

    if provider is None:
        config = CollectorConfig()
        provider = TheOddsApiProvider(
            api_key=config.the_odds_api_key,
            sport_key=SPORT_KEY,
            regions=REGIONS,
            markets=MARKET,
            enable_disk_cache=False,
        )

    total_spent = int(
        state.get("total_credits_spent", 0)
    )
    spent_this_run = 0
    remaining: int | None = None
    new_rows: list[dict[str, Any]] = []
    attempted_now = 0
    completed_now = 0
    no_odds_now = 0
    unmatched_now = 0
    stop_reason: str | None = None
    error: str | None = None

    try:
        provider.list_sports()
        cost, remaining = _usage(
            provider,
            fallback_cost=0,
            previous_remaining=None,
        )
        total_spent += cost
        spent_this_run += cost
    except Exception as exc:
        stop_reason = "quota_check_failed"
        error = str(exc)

    terminal = {"complete", "no_odds"}
    pending = [
        fixture
        for fixture in fixtures
        if state["fixtures"]
        .get(fixture["fixture_key"], {})
        .get("status")
        not in terminal
    ]

    if max_fixtures is not None:
        pending = pending[:max(0, max_fixtures)]

    by_day: dict[str, list[dict[str, Any]]] = {}
    for fixture in pending:
        by_day.setdefault(
            fixture["date"],
            [],
        ).append(fixture)

    if stop_reason is None and remaining is None:
        stop_reason = "quota_unknown"

    for matchday, day_fixtures in sorted(
        by_day.items()
    ):
        if stop_reason is not None:
            break

        catalog_record = catalog["matchdays"].get(
            matchday
        )

        if catalog_record is None:
            minimum_cost = (
                EVENT_CATALOG_COST + EVENT_ODDS_COST
            )
            if not _can_spend(
                total_spent=total_spent,
                expected_cost=minimum_cost,
                maximum_budget=max_credit_budget,
                remaining=remaining,
                reserve=credit_reserve,
            ):
                stop_reason = "credit_guard"
                break

            earliest = min(
                datetime.fromisoformat(
                    fixture["kickoff_utc"]
                )
                for fixture in day_fixtures
            )
            discovery_snapshot = (
                earliest - timedelta(hours=3)
            ).isoformat().replace("+00:00", "Z")

            try:
                payload = provider._get_json(
                    (
                        "/v4/historical/sports/"
                        f"{SPORT_KEY}/events"
                    ),
                    params={
                        "date": discovery_snapshot,
                        "dateFormat": "iso",
                    },
                    bypass_cache=True,
                )
            except Exception as exc:
                stop_reason = (
                    "historical_events_request_failed"
                )
                error = str(exc)
                break

            cost, remaining = _usage(
                provider,
                fallback_cost=EVENT_CATALOG_COST,
                previous_remaining=remaining,
            )
            total_spent += cost
            spent_this_run += cost

            events = (
                payload.get("data", [])
                if isinstance(payload, dict)
                else []
            )
            catalog_record = {
                "requested_snapshot": (
                    discovery_snapshot
                ),
                "returned_snapshot": (
                    payload.get("timestamp")
                    if isinstance(payload, dict)
                    else None
                ),
                "events": events,
                "api_cost": cost,
            }
            catalog["matchdays"][
                matchday
            ] = catalog_record
            _atomic_json(catalog_path, catalog)

        events = catalog_record.get("events", [])

        for fixture in day_fixtures:
            if stop_reason is not None:
                break

            fixture_key = fixture["fixture_key"]
            current_status = (
                state["fixtures"]
                .get(fixture_key, {})
                .get("status")
            )
            if current_status in terminal:
                continue

            event = _event_for_fixture(
                events,
                fixture,
            )
            if event is None:
                state["fixtures"][fixture_key] = {
                    "status": "event_not_found",
                    "updated_at": now.isoformat(),
                }
                unmatched_now += 1
                _atomic_json(state_path, state)
                continue

            if not _can_spend(
                total_spent=total_spent,
                expected_cost=EVENT_ODDS_COST,
                maximum_budget=max_credit_budget,
                remaining=remaining,
                reserve=credit_reserve,
            ):
                stop_reason = "credit_guard"
                break

            event_id = str(event.get("id") or "")
            kickoff = datetime.fromisoformat(
                fixture["kickoff_utc"]
            )
            odds_snapshot = (
                kickoff - timedelta(minutes=90)
            ).isoformat().replace("+00:00", "Z")

            try:
                payload = provider._get_json(
                    (
                        "/v4/historical/sports/"
                        f"{SPORT_KEY}/events/"
                        f"{event_id}/odds"
                    ),
                    params={
                        "date": odds_snapshot,
                        "regions": REGIONS,
                        "markets": MARKET,
                        "oddsFormat": "decimal",
                        "dateFormat": "iso",
                    },
                    bypass_cache=True,
                )
            except Exception as exc:
                stop_reason = (
                    "historical_odds_request_failed"
                )
                error = str(exc)
                break

            cost, remaining = _usage(
                provider,
                fallback_cost=EVENT_ODDS_COST,
                previous_remaining=remaining,
            )
            total_spent += cost
            spent_this_run += cost
            attempted_now += 1

            rows = _extract_odds_rows(
                payload,
                fixture=fixture,
                event_id=event_id,
                requested_snapshot=odds_snapshot,
            )
            new_rows.extend(rows)

            status = "complete" if rows else "no_odds"
            state["fixtures"][fixture_key] = {
                "status": status,
                "event_id": event_id,
                "requested_snapshot": odds_snapshot,
                "rows": len(rows),
                "api_cost": cost,
                "updated_at": now.isoformat(),
            }
            completed_now += int(bool(rows))
            no_odds_now += int(not rows)
            state["total_credits_spent"] = total_spent
            state["last_quota_remaining"] = remaining
            _atomic_json(state_path, state)

            if new_rows:
                additions = pd.DataFrame(new_rows)
                if odds_path.exists():
                    existing = pd.read_parquet(
                        odds_path
                    )
                    combined = pd.concat(
                        [existing, additions],
                        ignore_index=True,
                    )
                else:
                    combined = additions

                combined = combined.drop_duplicates(
                    subset=[
                        "fixture_key",
                        "provider_event_id",
                        "requested_snapshot",
                        "bookmaker_key",
                        "line",
                        "side",
                    ],
                    keep="last",
                ).sort_values(
                    [
                        "kickoff_utc",
                        "bookmaker_key",
                        "line",
                        "side",
                    ],
                    kind="stable",
                )
                _atomic_parquet(odds_path, combined)

    state["total_credits_spent"] = total_spent
    state["last_quota_remaining"] = remaining
    state["last_run_at"] = now.isoformat()
    _atomic_json(state_path, state)
    _atomic_json(catalog_path, catalog)

    stored_rows = (
        len(pd.read_parquet(odds_path))
        if odds_path.exists()
        else 0
    )
    statuses: dict[str, int] = {}
    for item in state["fixtures"].values():
        status = str(item.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    result = {
        "mode": "historical_odds_backfill",
        "status": (
            "stopped"
            if stop_reason
            else "complete"
        ),
        "completed_source_matches": len(fixtures),
        "attempted_now": attempted_now,
        "completed_with_odds_now": completed_now,
        "no_odds_now": no_odds_now,
        "event_not_found_now": unmatched_now,
        "stored_odds_rows": stored_rows,
        "fixture_statuses": statuses,
        "credits_spent_this_run": spent_this_run,
        "historical_credits_spent": total_spent,
        "maximum_credit_budget": max_credit_budget,
        "quota_remaining": remaining,
        "credit_reserve": credit_reserve,
        "stop_reason": stop_reason,
        "error": error,
        "odds_path": str(odds_path),
        "state_path": str(state_path),
        "catalog_path": str(catalog_path),
        "production_database_writes": False,
        "model_refit": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }
    _atomic_json(report_path, result)
    result["report_path"] = str(report_path)
    return result
