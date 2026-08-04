from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.odds.sportmonks_odds import SportmonksOddsProvider

TARGET_MARKET_KEYWORDS = ["total corners", "corners over/under", "match corners", "alternative total corners", "alternative corners"]


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        record = _build_error_record("AUTHENTICATION_REQUIRED", "SPORTMONKS_API_TOKEN missing")
        _write_outputs(record)
        print(_format_summary(record))
        return 0

    provider = SportmonksOddsProvider(api_token=token)
    requests_consumed = 0

    def request_json(path: str, params: dict[str, Any] | None = None) -> Any:
        nonlocal requests_consumed
        requests_consumed += 1
        response = requests.get(f"{provider.base_url}{path}", params={**(params or {}), "api_token": token}, timeout=20)
        if response.status_code in {401, 403, 422}:
            raise ValueError(f"AUTHENTICATION_REQUIRED: HTTP {response.status_code}")
        response.raise_for_status()
        return response.json()

    try:
        league_payload = request_json("/leagues")
        league_record = None
        for league in league_payload.get("data", []) if isinstance(league_payload, dict) else []:
            if not isinstance(league, dict):
                continue
            if str(league.get("country_name", "")).lower() == "italy" and str(league.get("name", "")).lower() == "serie a":
                league_record = {
                    "league_id": league.get("id"),
                    "season": league.get("current_season_id") or league.get("season_id"),
                    "country": league.get("country_name"),
                    "name": league.get("name"),
                    "raw": league,
                }
                break

        if league_record is None:
            record = _build_error_record("AUTHENTICATION_REQUIRED", "Serie A league could not be resolved")
            _write_outputs(record)
            print(_format_summary(record))
            return 0

        catalogue_payload = request_json("/odds/markets")
        catalogue_entries: list[dict[str, Any]] = []
        for market in catalogue_payload.get("data", []) if isinstance(catalogue_payload, dict) else []:
            if not isinstance(market, dict):
                continue
            name = str(market.get("name", "") or "")
            lowered = name.lower()
            if any(keyword in lowered for keyword in TARGET_MARKET_KEYWORDS):
                catalogue_entries.append({"id": market.get("id"), "name": market.get("name")})

        fixtures_payload = request_json("/fixtures/coming-soon", params={"league_id": league_record["league_id"], "include": "localTeam,visitorTeam"})
        fixtures = [fixture for fixture in fixtures_payload.get("data", []) if isinstance(fixture, dict)] if isinstance(fixtures_payload, dict) else []
        selected_fixtures = fixtures[:3]

        fixtures_checked = 0
        fixtures_with_corner_odds = 0
        bookmakers_found: set[str] = set()
        lines_found: set[str] = set()
        corner_rows: list[dict[str, Any]] = []
        historical_retention = "NO_EVIDENCE"
        closing_odds_reconstructable = "NO"

        for fixture in selected_fixtures:
            fixture_id = fixture.get("id")
            if not fixture_id:
                continue
            fixtures_checked += 1
            odds_payload = request_json("/odds/pre-match", params={"fixture_id": fixture_id, "include": "bookmaker"})
            odds_data = odds_payload.get("data", []) if isinstance(odds_payload, dict) else []
            if not odds_data:
                continue
            fixtures_with_corner_odds += 1
            for market in odds_data:
                if not isinstance(market, dict):
                    continue
                market_name = str(market.get("market_name") or "")
                lowered = market_name.lower()
                if "goal" in lowered:
                    continue
                if not any(keyword in lowered for keyword in TARGET_MARKET_KEYWORDS):
                    continue
                bookmaker_name = str(market.get("bookmaker_name") or market.get("bookmaker", {}).get("name") or "")
                if bookmaker_name:
                    bookmakers_found.add(bookmaker_name)
                line_value = market.get("line")
                if line_value is not None:
                    lines_found.add(str(line_value))
                corner_rows.append({
                    "fixture_id": fixture_id,
                    "bookmaker": bookmaker_name,
                    "market_id": market.get("id"),
                    "market_name": market_name,
                    "line": line_value,
                    "over": market.get("over"),
                    "under": market.get("under"),
                    "odds_timestamp": market.get("odds_timestamp") or market.get("updated_at") or market.get("created_at"),
                })

        if fixtures_checked and fixtures_with_corner_odds and corner_rows:
            verdict = "QUALIFIED_FOR_CURRENT_CORNER_ODDS"
        elif catalogue_entries:
            verdict = "MARKET_CATALOGUE_ONLY"
        elif not fixtures_checked:
            verdict = "NO CURRENT FIXTURES AVAILABLE"
        else:
            verdict = "INSUFFICIENT_CORNER_COVERAGE"

        if verdict == "QUALIFIED_FOR_CURRENT_CORNER_ODDS":
            historical_retention = "DIRECT_EVIDENCE_REQUIRED"
            closing_odds_reconstructable = "NO"

        record = {
            "provider_name": "Sportmonks",
            "probe_date": datetime.now(timezone.utc).isoformat(),
            "competition": "Serie A",
            "season": league_record["season"],
            "fixtures_found": len(fixtures),
            "fixtures_checked": fixtures_checked,
            "corner_catalogue_entries": catalogue_entries,
            "fixtures_with_corner_odds": fixtures_with_corner_odds,
            "bookmakers_found": sorted(bookmakers_found),
            "lines_found": sorted(lines_found),
            "historical_retention": historical_retention,
            "closing_odds_reconstructable": closing_odds_reconstructable,
            "requests_consumed": requests_consumed,
            "final_verdict": verdict,
            "fixture_details": [
                {
                    "fixture_id": fixture.get("id"),
                    "date": fixture.get("starting_at"),
                    "home_team": fixture.get("localTeam", {}).get("name"),
                    "away_team": fixture.get("visitorTeam", {}).get("name"),
                }
                for fixture in selected_fixtures
            ],
            "corner_rows": corner_rows,
            "limitations": "Live diagnostic only; no historical or betting changes",
        }
        _write_outputs(record)
        print(_format_summary(record))
        return 0
    except Exception as exc:
        record = _build_error_record("AUTHENTICATION_REQUIRED" if "HTTP 401" in str(exc) or "HTTP 403" in str(exc) or "HTTP 422" in str(exc) else "INSUFFICIENT_CORNER_COVERAGE", str(exc))
        _write_outputs(record)
        print(_format_summary(record))
        return 0


def _build_error_record(verdict: str, limitations: str) -> dict[str, Any]:
    return {
        "provider_name": "Sportmonks",
        "probe_date": datetime.now(timezone.utc).isoformat(),
        "competition": "Serie A",
        "season": None,
        "fixtures_found": 0,
        "fixtures_checked": 0,
        "corner_catalogue_entries": [],
        "fixtures_with_corner_odds": 0,
        "bookmakers_found": [],
        "lines_found": [],
        "historical_retention": "NO_EVIDENCE",
        "closing_odds_reconstructable": "NO",
        "requests_consumed": 0,
        "final_verdict": verdict,
        "fixture_details": [],
        "corner_rows": [],
        "limitations": limitations,
    }


def _format_summary(record: dict[str, Any]) -> str:
    return "\n".join([
        f"competition={record['competition']}",
        f"season={record['season']}",
        f"fixtures_found={record['fixtures_found']}",
        f"fixtures_checked={record['fixtures_checked']}",
        f"corner_catalogue_entries={len(record['corner_catalogue_entries'])}",
        f"fixtures_with_corner_odds={record['fixtures_with_corner_odds']}",
        f"bookmakers_found={record['bookmakers_found']}",
        f"lines_found={record['lines_found']}",
        f"historical_retention={record['historical_retention']}",
        f"closing_odds_reconstructable={record['closing_odds_reconstructable']}",
        f"requests_consumed={record['requests_consumed']}",
        f"final_verdict={record['final_verdict']}",
    ])


def _write_outputs(record: dict[str, Any]) -> None:
    report_path = Path("reports/sportmonks_corner_probe.md")
    manifest_path = Path("data/processed/sportmonks_probe_manifest.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Sportmonks corner odds probe",
            "",
            f"- Probe date: {record['probe_date']}",
            f"- Competition: {record['competition']}",
            f"- Season: {record['season']}",
            f"- Fixtures found: {record['fixtures_found']}",
            f"- Fixtures checked: {record['fixtures_checked']}",
            f"- Corner catalogue entries: {record['corner_catalogue_entries']}",
            f"- Fixtures with corner odds: {record['fixtures_with_corner_odds']}",
            f"- Bookmakers found: {record['bookmakers_found']}",
            f"- Lines found: {record['lines_found']}",
            f"- Historical retention: {record['historical_retention']}",
            f"- Closing odds reconstructable: {record['closing_odds_reconstructable']}",
            f"- Requests consumed: {record['requests_consumed']}",
            f"- Fixture details: {record['fixture_details']}",
            f"- Corner rows: {record['corner_rows']}",
            f"- Limitations: {record['limitations']}",
            f"- Final verdict: {record['final_verdict']}",
        ]) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider_name", "probe_date", "competition", "season", "fixtures_found", "fixtures_checked", "corner_catalogue_entries", "fixtures_with_corner_odds", "bookmakers_found", "lines_found", "historical_retention", "closing_odds_reconstructable", "requests_consumed", "final_verdict", "fixture_details", "corner_rows", "limitations"])
        writer.writeheader()
        writer.writerow({
            **record,
            "corner_catalogue_entries": str(record["corner_catalogue_entries"]),
            "bookmakers_found": str(record["bookmakers_found"]),
            "lines_found": str(record["lines_found"]),
            "fixture_details": str(record["fixture_details"]),
            "corner_rows": str(record["corner_rows"]),
        })


if __name__ == "__main__":
    main()
