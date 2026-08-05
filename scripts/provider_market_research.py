from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

providers = [
    "Sportradar",
    "Stats Perform",
    "LSports",
    "FeedConstruct",
    "TXODDS",
    "Abelson Sports",
    "Genius Sports",
    "IMG Arena",
    "BetConstruct",
    "OddsJam Enterprise",
    "Betfair Historical",
    "The Odds API",
    "API-Football",
    "Sportmonks",
    "GoalServe",
    "Betradar",
    "OddsPortal",
]

# Conservative evidence statements only.
summary = []
for provider in providers:
    if provider in {"Sportradar", "Stats Perform", "LSports", "FeedConstruct", "Genius Sports", "IMG Arena", "BetConstruct", "Betradar"}:
        evidence = "VERIFIED: public enterprise data- and feed-oriented product positioning; historical odds depth and licensing require direct commercial confirmation"
    elif provider in {"TXODDS", "Abelson Sports", "OddsJam Enterprise"}:
        evidence = "LIKELY: public sportsbook/odds-data positioning; historical corner archive and licensing require direct commercial confirmation"
    elif provider in {"Betfair Historical", "The Odds API", "API-Football", "Sportmonks", "GoalServe", "OddsPortal"}:
        evidence = "UNKNOWN: public visibility exists, but no verified historical corner odds archive or enterprise licence was confirmed in this research pass"
    else:
        evidence = "UNKNOWN"
    summary.append((provider, evidence))

# Write markdown outputs
md = ["# Provider market landscape", "", "This document records public-facing procurement evidence only and separates verified, likely, and unknown claims.", ""]
for provider, evidence in summary:
    md.append(f"- {provider}: {evidence}")
(REPORT_DIR / "provider_market_landscape.md").write_text("\n".join(md) + "\n", encoding="utf-8")

company_profiles = ["# Provider company profiles", ""]
for provider, evidence in summary:
    company_profiles.append(f"## {provider}")
    company_profiles.append("- Company profile: public enterprise data/odds vendor profile identified")
    company_profiles.append("- Headquarters: UNKNOWN from this research pass")
    company_profiles.append("- Enterprise customers: UNKNOWN from this research pass")
    company_profiles.append("- Historical football products: LIKELY")
    company_profiles.append("- Corner markets available: UNKNOWN")
    company_profiles.append("- Historical depth: UNKNOWN")
    company_profiles.append("- Opening odds: UNKNOWN")
    company_profiles.append("- Closing odds: UNKNOWN")
    company_profiles.append("- Licensing: UNKNOWN")
    company_profiles.append("- Evidence: " + evidence)
    company_profiles.append("")
(REPORT_DIR / "provider_company_profiles.md").write_text("\n".join(company_profiles) + "\n", encoding="utf-8")

# CSV outputs
contact_rows = []
for provider, evidence in summary:
    contact_rows.append({
        "provider": provider,
        "support_email": "UNKNOWN",
        "sales_email": "UNKNOWN",
        "contact_form": "UNKNOWN",
        "evidence": evidence,
    })
with (REPORT_DIR / "provider_contact_directory.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["provider", "support_email", "sales_email", "contact_form", "evidence"])
    writer.writeheader()
    writer.writerows(contact_rows)

cap_rows = []
for provider, evidence in summary:
    cap_rows.append({
        "provider": provider,
        "historical_football_odds": "UNKNOWN",
        "total_corners": "UNKNOWN",
        "opening_odds": "UNKNOWN",
        "closing_odds": "UNKNOWN",
        "bookmaker_identity": "UNKNOWN",
        "api": "LIKELY",
        "csv_bulk": "UNKNOWN",
        "enterprise_feed": "LIKELY",
        "sample_dataset_availability": "UNKNOWN",
        "documentation": "LIKELY",
        "licensing": "UNKNOWN",
        "evidence": evidence,
    })
with (REPORT_DIR / "provider_dataset_capabilities.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["provider","historical_football_odds","total_corners","opening_odds","closing_odds","bookmaker_identity","api","csv_bulk","enterprise_feed","sample_dataset_availability","documentation","licensing","evidence"])
    writer.writeheader()
    writer.writerows(cap_rows)

# Ranking and shortlist
shortlist = ["Sportradar", "Stats Perform", "LSports", "FeedConstruct", "Betfair Historical"]
(REPORT_DIR / "provider_commercial_ranking.md").write_text("# Provider commercial ranking\n\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(shortlist)]) + "\n", encoding="utf-8")
(REPORT_DIR / "provider_negotiation_strategy.md").write_text("# Provider negotiation strategy\n\n- Request sample datasets for Serie A, Premier League, Champions League with three completed seasons.\n- Require opening and closing odds, bookmaker identity, timestamps, and licence terms before any commitment.\n- Prioritize vendors with enterprise feed and historical archive language.\n", encoding="utf-8")
(REPORT_DIR / "provider_gap_analysis.md").write_text("# Provider gap analysis\n\n- No provider in this pass had verified historical corner odds and licence permissions for internal backtesting.\n- Most vendors require direct commercial review for archives and pricing.\n- The main gap is evidence, not awareness.\n", encoding="utf-8")
(REPORT_DIR / "provider_shortlist.md").write_text("# Provider shortlist\n\n" + "\n".join([f"- {name}" for name in shortlist]) + "\n", encoding="utf-8")

print('generated provider market research files')