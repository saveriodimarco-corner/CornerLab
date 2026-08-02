from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.database import DataStore
from src.data.normalizer import NormalizationError, normalize_row
from src.data.providers import ApiFootballProvider, CSVProvider, FootballDataProvider
from src.data.quality import build_quality_report


def main() -> None:
    providers = [
        FootballDataProvider(),
        ApiFootballProvider(),
        CSVProvider(os.path.join("data", "raw", "matches.csv")),
    ]

    all_rows = []
    for provider in providers:
        raw_matches = provider.fetch_matches()
        raw_stats = provider.fetch_match_statistics()
        raw_teams = provider.fetch_teams()
        combined_rows = []
        for match in raw_matches:
            match_row = dict(match)
            stats_row = next((item for item in raw_stats if item.get("date") == match_row.get("date") and item.get("home_team") == match_row.get("home_team") and item.get("away_team") == match_row.get("away_team")), {})
            if stats_row:
                match_row.update(stats_row)
            combined_rows.append(match_row)
        for team in raw_teams:
            combined_rows.append({**team, **{"date": None, "season": None, "competition": team.get("competition")}})

        normalized_rows = []
        for row in combined_rows:
            try:
                normalized = normalize_row(row)
            except NormalizationError:
                continue
            normalized_rows.append(normalized)

        if normalized_rows:
            quality_report = build_quality_report(normalized_rows)
            report_path = Path("reports") / f"quality_report_{provider.name}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(quality_report, encoding="utf-8")
            all_rows.extend(normalized_rows)

            store = DataStore()
            store.import_rows(normalized_rows, provider.name)
            store.close()

    if all_rows:
        combined_report = build_quality_report(all_rows)
        Path("reports/quality_report.md").write_text(combined_report, encoding="utf-8")

    print("Imported data from configured providers")


if __name__ == "__main__":
    main()
