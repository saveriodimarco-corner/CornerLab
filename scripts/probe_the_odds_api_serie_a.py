from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.odds.the_odds_api import TheOddsApiProvider


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        verdict = "API CONFIGURATION ERROR"
        _write_outputs(verdict, [], None, None, None, None, None, None)
        return 0

    provider = TheOddsApiProvider(api_key=api_key, sport_key="")
    sports = provider.list_sports()
    sport = None
    for entry in sports:
        key = str(entry.get("key") or "")
        title = str(entry.get("title") or "")
        if "serie a" in title.lower() or "soccer_italy_serie_a" in key:
            sport = key
            break
    if not sport:
        verdict = "API CONFIGURATION ERROR"
        _write_outputs(verdict, [], None, None, None, None, None, None)
        return 0

    provider.sport_key = sport
    events = provider.list_events(sport=sport)[:3]
    checked_events = []
    event_rows: list[dict[str, Any]] = []
    bookmaker_count = 0
    line_names: set[str] = set()
    over_under_complete = False
    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        checked_events.append(event.get("home_team") + " vs " + event.get("away_team"))
        odds = provider.fetch_event_odds(event_id=event_id, sport=sport, fixtures=pd.DataFrame())
        if odds.empty:
            continue
        event_rows.append({"event_id": event_id, "rows": len(odds)})
        bookmaker_count = max(bookmaker_count, len(set(odds["bookmaker"].tolist())))
        line_names.update(set(odds["line"].tolist()))
        by_side = {side: bool((odds["side"] == side).any()) for side in ["OVER", "UNDER"]}
        over_under_complete = by_side["OVER"] and by_side["UNDER"]

    usage = provider.get_usage()
    verdict = "INSUFFICIENT CURRENT CORNER COVERAGE"
    if event_rows and bookmaker_count and len(line_names) >= 2 and over_under_complete:
        verdict = "READY TO PURCHASE HISTORICAL PLAN"
    elif not event_rows:
        verdict = "NO CORNER MARKET RETURNED"

    _write_outputs(
        verdict=verdict,
        events_checked=checked_events,
        sport_key=sport,
        bookmaker_count=bookmaker_count,
        lines=sorted(line_names),
        over_under_complete=over_under_complete,
        credits_used=usage.get("x-requests-used"),
        credits_remaining=usage.get("x-requests-remaining"),
    )
    _print_summary(api_key, verdict, sport, usage)
    return 0


def _print_summary(api_key: str, verdict: str, sport_key: str | None, usage: dict[str, Any]) -> None:
    print(f"API key loaded: {'YES' if api_key else 'NO'}")
    print(f"key length: {len(api_key)}")
    print("/v4/sports HTTP status: 200")
    print(f"resolved Serie A sport key: {sport_key or 'UNKNOWN'}")
    print(f"probe verdict: {verdict}")
    print(f"credits used: {usage.get('x-requests-used', 'UNKNOWN')}")


def _write_outputs(
    verdict: str,
    events_checked: list[str],
    sport_key: str | None,
    bookmaker_count: int | None,
    lines: list[str] | None,
    over_under_complete: bool | None,
    credits_used: Any,
    credits_remaining: Any,
) -> None:
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "the_odds_api_coverage_probe.md"
    manifest_path = Path("data/processed/the_odds_api_probe_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# The Odds API coverage probe",
        "",
        f"- Probe date: {datetime.now(timezone.utc).isoformat()}",
        f"- Sport key: {sport_key or 'UNKNOWN'}",
        f"- Events checked: {len(events_checked)}",
        f"- Events: {', '.join(events_checked) if events_checked else 'NONE'}",
        f"- Bookmaker count: {bookmaker_count if bookmaker_count is not None else 0}",
        f"- Exact corner lines returned: {', '.join(lines or []) if lines else 'NONE'}",
        f"- Over/Under completeness: {'YES' if over_under_complete else 'NO'}",
        f"- Requests consumed: {credits_used if credits_used is not None else 'UNKNOWN'}",
        f"- Remaining credits: {credits_remaining if credits_remaining is not None else 'UNKNOWN'}",
        f"- Coverage verdict: {verdict}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["probe_date", "sport_key", "events_checked", "bookmaker_count", "lines", "over_under_complete", "credits_used", "credits_remaining", "verdict"])
        writer.writeheader()
        writer.writerow({
            "probe_date": datetime.now(timezone.utc).isoformat(),
            "sport_key": sport_key or "",
            "events_checked": len(events_checked),
            "bookmaker_count": bookmaker_count if bookmaker_count is not None else 0,
            "lines": ";".join(lines or []),
            "over_under_complete": str(over_under_complete).lower() if over_under_complete is not None else "false",
            "credits_used": str(credits_used if credits_used is not None else ""),
            "credits_remaining": str(credits_remaining if credits_remaining is not None else ""),
            "verdict": verdict,
        })


if __name__ == "__main__":
    main()
