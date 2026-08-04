from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider


TARGET_LINES = {"8.5", "9.5", "10.5", "11.5"}
CORNER_KEYWORDS = {"corner", "corners", "total corners", "over/under corners", "match corners", "alternative corners"}


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        record = _build_error_record("API CONFIGURATION ERROR", "API_FOOTBALL_KEY missing")
        _write_outputs(record)
        print(_format_summary(record))
        return 0

    provider = ApiFootballOddsProvider(api_key=api_key)
    requests_consumed = 0

    def request_json(path: str, params: dict[str, Any] | None = None) -> Any:
        nonlocal requests_consumed
        requests_consumed += 1
        response = requests.get(f"{provider.base_url}{path}", headers={"x-apisports-key": api_key}, params=params, timeout=20)
        if response.status_code in {401, 403, 422}:
            raise ValueError(f"API configuration error: HTTP {response.status_code}")
        response.raise_for_status()
        return response.json()

    try:
        request_json("/status")
        leagues_payload = request_json("/leagues")
        league_record = None
        for league in leagues_payload.get("response", []):
            league_name = str(league.get("league", {}).get("name", "") or "")
            country_name = str(league.get("country", {}).get("name", "") or "")
            if country_name.lower() != "italy" or league_name.lower() != "serie a":
                continue
            seasons = league.get("seasons", []) or []
            current_season = None
            for season in seasons:
                if season.get("current"):
                    current_season = season
                    break
            if current_season is None and seasons:
                current_season = seasons[-1]
            league_record = {
                "league_id": league.get("league", {}).get("id"),
                "season": current_season.get("year") if current_season else None,
                "current": bool(current_season.get("current")) if current_season else False,
                "coverage": league.get("coverage", {}) or {},
                "raw": league,
            }
            break
        if league_record is None:
            record = _build_error_record("API CONFIGURATION ERROR", "Serie A league could not be resolved")
            _write_outputs(record)
            print(_format_summary(record))
            return 0

        corner_catalogue = provider.find_corner_bet_entries(request_json("/odds/bets"))
        fixture_queries: list[dict[str, Any]] = []
        upcoming_fixtures: list[dict[str, Any]] = []
        target_league_id = league_record["league_id"]
        target_season = league_record["season"]
        if target_league_id and target_season:
            fixture_queries.append({"path": "/fixtures", "params": {"league": target_league_id, "season": target_season, "next": 20}})
            initial_payload = request_json("/fixtures", params={"league": target_league_id, "season": target_season, "next": 20})
            initial_fixtures = initial_payload.get("response", []) if isinstance(initial_payload, dict) else []
            if isinstance(initial_fixtures, list) and initial_fixtures:
                upcoming_fixtures = [fixture for fixture in initial_fixtures if _is_upcoming_fixture(fixture)]
            if not upcoming_fixtures:
                today = date.today()
                for start_offset in [0, 31]:
                    start_date = (today + timedelta(days=start_offset)).strftime("%Y-%m-%d")
                    end_date = (today + timedelta(days=start_offset + 29)).strftime("%Y-%m-%d")
                    fixture_queries.append({"path": "/fixtures", "params": {"league": target_league_id, "season": target_season, "from": start_date, "to": end_date}})
                    payload = request_json("/fixtures", params={"league": target_league_id, "season": target_season, "from": start_date, "to": end_date})
                    fixtures = payload.get("response", []) if isinstance(payload, dict) else []
                    if isinstance(fixtures, list):
                        upcoming_fixtures.extend([fixture for fixture in fixtures if _is_upcoming_fixture(fixture)])
                    if len(upcoming_fixtures) >= 3:
                        break
        upcoming_fixtures = upcoming_fixtures[:3]

        fixtures_checked = 0
        fixtures_with_general_odds = 0
        fixtures_with_corner_odds = 0
        bookmakers_found: set[str] = set()
        corner_lines_found: set[str] = set()
        corner_rows: list[dict[str, Any]] = []
        for fixture in upcoming_fixtures:
            fixture_id = fixture.get("fixture", {}).get("id")
            if not fixture_id:
                continue
            fixtures_checked += 1
            odds_payload = request_json("/odds", params={"fixture": fixture_id})
            odds_response = odds_payload.get("response", []) if isinstance(odds_payload, dict) else []
            if isinstance(odds_response, list) and odds_response:
                fixtures_with_general_odds += 1
            fixture_corner_rows = provider.extract_corner_odds_rows(odds_payload)
            if fixture_corner_rows:
                fixtures_with_corner_odds += 1
            for row in fixture_corner_rows:
                bookmaker = row.get("bookmaker")
                if bookmaker:
                    bookmakers_found.add(str(bookmaker))
                value = row.get("value")
                if value is not None and str(value) in TARGET_LINES:
                    corner_lines_found.add(str(value))
                corner_rows.append({"fixture_id": fixture_id, **row})

        over_present = any(str(row.get("value")) in {"8.5", "9.5", "10.5", "11.5"} and str(row.get("odd")) for row in corner_rows if row.get("bet_name", "").lower().find("over") >= 0)
        under_present = any(str(row.get("value")) in {"8.5", "9.5", "10.5", "11.5"} and str(row.get("odd")) for row in corner_rows if row.get("bet_name", "").lower().find("under") >= 0)
        has_corner_catalogue = bool(corner_catalogue)

        if upcoming_fixtures and fixtures_with_corner_odds and over_present and under_present and len(corner_lines_found) >= 2:
            verdict = "QUALIFIED_FOR_CURRENT_CORNER_ODDS"
        elif not upcoming_fixtures and league_record["league_id"] and league_record["season"] is not None:
            verdict = "NO CURRENT FIXTURES AVAILABLE"
        elif has_corner_catalogue and not fixtures_with_corner_odds:
            verdict = "MARKET CATALOGUE ONLY"
        elif fixtures_checked and fixtures_with_general_odds and not fixtures_with_corner_odds:
            verdict = "INSUFFICIENT_CORNER_COVERAGE"
        else:
            verdict = "INSUFFICIENT_CORNER_COVERAGE"

        record = {
            "provider_name": "API-Football",
            "probe_date": datetime.now(timezone.utc).isoformat(),
            "competition": "Serie A",
            "league_id": league_record["league_id"],
            "resolved_season": league_record["season"],
            "odds_coverage_flag": bool(league_record.get("coverage", {}).get("odds")),
            "corner_bet_catalogue_entries": corner_catalogue,
            "upcoming_fixtures_found": len(upcoming_fixtures),
            "fixtures_checked": fixtures_checked,
            "fixtures_with_general_odds": fixtures_with_general_odds,
            "fixtures_with_corner_odds": fixtures_with_corner_odds,
            "bookmakers_found": sorted(bookmakers_found),
            "corner_lines_found": sorted(corner_lines_found),
            "requests_consumed": requests_consumed,
            "final_verdict": verdict,
            "fixture_queries": fixture_queries,
            "fixture_details": [
                {
                    "fixture_id": fixture.get("fixture", {}).get("id"),
                    "date": fixture.get("fixture", {}).get("date"),
                    "home_team": fixture.get("teams", {}).get("home", {}).get("name"),
                    "away_team": fixture.get("teams", {}).get("away", {}).get("name"),
                }
                for fixture in upcoming_fixtures
            ],
            "corner_rows": corner_rows,
            "limitations": "Live diagnostic only; no historical or betting changes",
        }
        _write_outputs(record)
        print(_format_summary(record))
        return 0
    except Exception as exc:
        record = _build_error_record("API CONFIGURATION ERROR", str(exc))
        _write_outputs(record)
        print(_format_summary(record))
        return 0


def _is_upcoming_fixture(fixture: dict[str, Any]) -> bool:
    fixture_info = fixture.get("fixture", {})
    timestamp = fixture_info.get("timestamp")
    if not timestamp:
        return True
    try:
        return int(timestamp) >= int(datetime.now(timezone.utc).timestamp())
    except (TypeError, ValueError):
        return True


def _build_error_record(verdict: str, limitations: str) -> dict[str, Any]:
    return {
        "provider_name": "API-Football",
        "probe_date": datetime.now(timezone.utc).isoformat(),
        "competition": "Serie A",
        "league_id": None,
        "resolved_season": None,
        "odds_coverage_flag": False,
        "corner_bet_catalogue_entries": [],
        "upcoming_fixtures_found": 0,
        "fixtures_checked": 0,
        "fixtures_with_general_odds": 0,
        "fixtures_with_corner_odds": 0,
        "bookmakers_found": [],
        "corner_lines_found": [],
        "requests_consumed": 0,
        "final_verdict": verdict,
        "fixture_queries": [],
        "fixture_details": [],
        "corner_rows": [],
        "limitations": limitations,
    }


def _format_summary(record: dict[str, Any]) -> str:
    return "\n".join([
        f"league_id={record['league_id']}",
        f"resolved_season={record['resolved_season']}",
        f"odds_coverage_flag={record['odds_coverage_flag']}",
        f"corner_bet_catalogue_entries={record['corner_bet_catalogue_entries']}",
        f"upcoming_fixtures_found={record['upcoming_fixtures_found']}",
        f"fixtures_checked={record['fixtures_checked']}",
        f"fixtures_with_general_odds={record['fixtures_with_general_odds']}",
        f"fixtures_with_corner_odds={record['fixtures_with_corner_odds']}",
        f"bookmakers_found={record['bookmakers_found']}",
        f"corner_lines_found={record['corner_lines_found']}",
        f"requests_consumed={record['requests_consumed']}",
        f"final_verdict={record['final_verdict']}",
    ])


def _write_outputs(record: dict[str, Any]) -> None:
    report_path = Path("reports/api_football_corner_probe.md")
    manifest_path = Path("data/processed/api_football_probe_manifest.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# API-Football corner odds probe",
            "",
            f"- Probe date: {record['probe_date']}",
            f"- Provider: {record['provider_name']}",
            f"- Competition: {record['competition']}",
            f"- League ID: {record['league_id']}",
            f"- Resolved season: {record['resolved_season']}",
            f"- Odds coverage flag: {record['odds_coverage_flag']}",
            f"- Corner bet catalogue entries: {record['corner_bet_catalogue_entries']}",
            f"- Upcoming fixtures found: {record['upcoming_fixtures_found']}",
            f"- Fixtures checked: {record['fixtures_checked']}",
            f"- Fixtures with general odds: {record['fixtures_with_general_odds']}",
            f"- Fixtures with corner odds: {record['fixtures_with_corner_odds']}",
            f"- Bookmakers found: {record['bookmakers_found']}",
            f"- Corner lines found: {record['corner_lines_found']}",
            f"- Requests consumed: {record['requests_consumed']}",
            f"- Fixture queries attempted: {record['fixture_queries']}",
            f"- Fixture details: {record['fixture_details']}",
            f"- Corner rows: {record['corner_rows']}",
            f"- Limitations: {record['limitations']}",
            f"- Final verdict: {record['final_verdict']}",
        ]) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider_name", "probe_date", "competition", "league_id", "resolved_season", "odds_coverage_flag", "corner_bet_catalogue_entries", "upcoming_fixtures_found", "fixtures_checked", "fixtures_with_general_odds", "fixtures_with_corner_odds", "bookmakers_found", "corner_lines_found", "requests_consumed", "final_verdict", "fixture_queries", "fixture_details", "corner_rows", "limitations"])
        writer.writeheader()
        writer.writerow({
            **record,
            "corner_bet_catalogue_entries": str(record["corner_bet_catalogue_entries"]),
            "bookmakers_found": str(record["bookmakers_found"]),
            "corner_lines_found": str(record["corner_lines_found"]),
            "fixture_queries": str(record["fixture_queries"]),
            "fixture_details": str(record["fixture_details"]),
            "corner_rows": str(record["corner_rows"]),
        })


if __name__ == "__main__":
    main()
