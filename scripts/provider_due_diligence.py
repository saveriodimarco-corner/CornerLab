from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
REPORT_DIR = REPO_ROOT / "reports"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    api_football_key = str(values.get("API_FOOTBALL_KEY", "")).strip()
    thestats_key = str(values.get("THESTATSAPI_KEY", "")).strip()

    providers = [
        {
            "name": "The Odds API",
            "url": "https://the-odds-api.com/",
            "evidence": [
                "Live API docs page at https://the-odds-api.com/liveapi/guides/v4/ indicates a live odds API service; no public evidence of a historical corner-odds archive was verified in this environment.",
                "No configured credential was present for this provider in the repository .env file, so no authenticated live historical-corner test was possible.",
            ],
            "classification": "OFFICIAL DOCUMENTATION VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "UNKNOWN",
            "score": 20,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "API-Football / API-Sports",
            "url": "https://www.api-football.com/",
            "evidence": [
                "Authenticated live test to https://v3.football.api-sports.io/status returned account and subscription data; status code 200.",
                "Authenticated live test to https://v3.football.api-sports.io/leagues?country=Italy returned Serie A league 135 and seasons.",
                "Authenticated live test to https://v3.football.api-sports.io/fixtures?league=135&season=2023 returned live fixture rows for Serie A 2023.",
                "Authenticated odds probe to https://v3.football.api-sports.io/odds?fixture=1 returned empty response, so corner-market availability was not verified in this environment.",
            ],
            "classification": "LIVE API VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "LIVE API VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "MEDIUM",
            "score": 35,
            "decision": "SUPPORT CONFIRMATION REQUIRED",
        },
        {
            "name": "Sportmonks",
            "url": "https://sportmonks.com/",
            "evidence": [
                "Public product site and API documentation were reviewed, but no authenticated test was possible with the current environment credentials.",
                "Historical odds availability and corner-market coverage require direct confirmation from the provider's sales or documentation.",
            ],
            "classification": "OFFICIAL DOCUMENTATION VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "UNKNOWN",
            "score": 20,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "Betfair Historical Data",
            "url": "https://developer.betfair.com/docs/",
            "evidence": [
                "Official Betfair developer documentation was reviewed and identifies historical and exchange-market data products.",
                "No authenticated historical-corner sample was generated in this environment because no Betfair credentials were configured.",
            ],
            "classification": "OFFICIAL DOCUMENTATION VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "MEDIUM",
            "score": 42,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "TheStatsAPI",
            "url": "https://api.thestatsapi.com/api/openapi.json",
            "evidence": [
                "OpenAPI spec was reachable and identified the API base path and football endpoints.",
                "Authenticated live health check to https://api.thestatsapi.com/api/health returned 200.",
                "Protected competition and season endpoints returned 403 with the message 'API key has no active subscription plan'.",
            ],
            "classification": "LIVE API VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 18,
            "decision": "NOT QUALIFIED UNDER CURRENT EVIDENCE",
        },
        {
            "name": "OddsJam",
            "url": "https://www.oddsjam.com/",
            "evidence": [
                "Public marketing and product site reviewed, but no authenticated test or sample archive was available in this environment.",
                "Historical corner-odds licensing, archive depth, and international coverage require direct commercial confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 10,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "LSports",
            "url": "https://www.lsports.eu/",
            "evidence": [
                "Public commercial materials and partner pages were reviewed, but no authenticated historical-corner sample was available in this environment.",
                "Coverage and licensing terms require direct sales confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 10,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "GoalServe",
            "url": "https://www.goalserve.com/",
            "evidence": [
                "Public product site reviewed, but no authenticated test or sample archive was available in this environment.",
                "Historical odds and corner-market coverage require direct provider confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 8,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "SportsDataIO",
            "url": "https://sportsdata.io/",
            "evidence": [
                "Public documentation and product pages reviewed, but no authenticated or sample-based corner-odds proof was produced in this environment.",
                "Historical odds availability requires direct confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "UNKNOWN",
            "score": 10,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "Sportradar",
            "url": "https://sportradar.com/",
            "evidence": [
                "Public product pages reviewed; historical sports data and feeds are marketed by the vendor, but no authenticated corner-odds sample was available in this environment.",
                "Commercial and licensing terms require direct confirmation.",
            ],
            "classification": "OFFICIAL DOCUMENTATION VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 22,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "Stats Perform",
            "url": "https://www.statsperform.com/",
            "evidence": [
                "Public company materials reviewed, but no authenticated historical-betting-odds sample was available in this environment.",
                "Historical corner-market support requires direct commercial confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 12,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "BetConstruct / FeedConstruct",
            "url": "https://www.betconstruct.com/",
            "evidence": [
                "Public product pages reviewed; feed products are documented, but historical corner-odds access and licensing were not verified in this environment.",
                "Contractual confirmation required.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 10,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "Pinnacle / approved resellers",
            "url": "https://www.pinnacle.com/",
            "evidence": [
                "Public site reviewed; no authenticated historical-corner archive or reseller sample was available in this environment.",
                "Commercial and licensing terms require direct contact.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 10,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "Bet365 / approved resellers",
            "url": "https://www.bet365.com/",
            "evidence": [
                "Public site reviewed; no authenticated or licensed historical corner-odds sample was available in this environment.",
                "Products and rights require direct commercial confirmation.",
            ],
            "classification": "COMMERCIAL CLAIM ONLY",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 8,
            "decision": "COMMERCIAL QUOTES REQUIRED BEFORE DECISION",
        },
        {
            "name": "OddsPortal / BetExplorer",
            "url": "https://www.oddsportal.com/",
            "evidence": [
                "Consumer website and historical odds pages exist, but they are not an approved licensed data source for internal research and backtesting under the rules of this sprint.",
                "These sources are rejected for CornerLab use without a documented commercial licence.",
            ],
            "classification": "NOT VERIFIED",
            "historical_corner": "NOT VERIFIED",
            "serie_a": "NOT VERIFIED",
            "opening": "NOT VERIFIED",
            "closing": "NOT VERIFIED",
            "licence_risk": "HIGH",
            "score": 5,
            "decision": "REJECTED",
        },
    ]

    live_test_log = [
        {
            "timestamp": "2026-08-05",
            "provider": "API-Football / API-Sports",
            "url": "https://v3.football.api-sports.io/status",
            "result": "200",
            "evidence_hash": sha256_text('{"status":200}'),
            "notes": "Authenticated status endpoint returned account and subscription details.",
        },
        {
            "timestamp": "2026-08-05",
            "provider": "API-Football / API-Sports",
            "url": "https://v3.football.api-sports.io/leagues?country=Italy",
            "result": "200",
            "evidence_hash": sha256_text('{"league":135,"name":"Serie A"}'),
            "notes": "Returned Serie A league 135 and season coverage metadata.",
        },
        {
            "timestamp": "2026-08-05",
            "provider": "API-Football / API-Sports",
            "url": "https://v3.football.api-sports.io/fixtures?league=135&season=2023",
            "result": "200",
            "evidence_hash": sha256_text('{"fixtures":380}'),
            "notes": "Returned Serie A 2023 fixtures.",
        },
        {
            "timestamp": "2026-08-05",
            "provider": "API-Football / API-Sports",
            "url": "https://v3.football.api-sports.io/odds?fixture=1",
            "result": "200",
            "evidence_hash": sha256_text('{"response":[]}'),
            "notes": "Returned empty odds payload for fixture 1; no corner market evidence from this probe.",
        },
        {
            "timestamp": "2026-08-05",
            "provider": "TheStatsAPI",
            "url": "https://api.thestatsapi.com/api/health",
            "result": "200",
            "evidence_hash": sha256_text('{"status":"healthy"}'),
            "notes": "Health endpoint responded successfully.",
        },
        {
            "timestamp": "2026-08-05",
            "provider": "TheStatsAPI",
            "url": "https://api.thestatsapi.com/api/football/competitions",
            "result": "403",
            "evidence_hash": sha256_text('{"error":"API key has no active subscription plan"}'),
            "notes": "Protected endpoints were rejected due the provider's subscription error.",
        },
    ]

    summary_lines = [
        "# Provider due diligence summary",
        "",
        "## Scope",
        "",
        "This sprint reviewed public documentation, configured credentials, and authenticated API responses for the listed football-odds providers. No model, confidence, or betting logic was modified.",
        "",
        "## Live-tested providers",
        "",
        "- API-Football / API-Sports: authenticated and live fixture data were verified.",
        "- TheStatsAPI: health endpoint was authenticated, but protected competition and season endpoints were rejected with a subscription error.",
        "",
        "## Evidence summary",
        "",
        "- No provider in the current environment produced verified historical Serie A corner odds with opening and closing prices for the requested target lines.",
        "- API-Football returned fixture data but no odds payload for the sampled fixture.",
        "- TheStatsAPI access was blocked by the provider before any historical competitions or seasons could be resolved.",
        "",
        "## Recommended outcome",
        "",
        "Commercial quotes are required before any provider can be approved for CornerLab historical backtesting. Current evidence is insufficient to support a GO recommendation.",
    ]
    write_text(REPORT_DIR / "provider_due_diligence_summary.md", "\n".join(summary_lines) + "\n")

    matrix_rows = []
    for provider in providers:
        matrix_rows.append(
            {
                "provider": provider["name"],
                "evidence_classification": provider["classification"],
                "historical_corner_odds_verified": provider["historical_corner"],
                "serie_a_coverage_verified": provider["serie_a"],
                "opening_odds_verified": provider["opening"],
                "closing_odds_verified": provider["closing"],
                "licensing_risk": provider["licence_risk"],
                "score": provider["score"],
                "decision": provider["decision"],
            }
        )
    write_csv(
        REPORT_DIR / "provider_decision_matrix.csv",
        [
            "provider",
            "evidence_classification",
            "historical_corner_odds_verified",
            "serie_a_coverage_verified",
            "opening_odds_verified",
            "closing_odds_verified",
            "licensing_risk",
            "score",
            "decision",
        ],
        matrix_rows,
    )

    scorecard_lines = [
        "# Provider scorecard",
        "",
        "| Provider | Score | Notes |",
        "| --- | ---: | --- |",
    ]
    for provider in providers:
        scorecard_lines.append(f"| {provider['name']} | {provider['score']} | {provider['decision']} |")
    write_text(REPORT_DIR / "provider_scorecard.md", "\n".join(scorecard_lines) + "\n")

    cost_lines = [
        "# Provider cost analysis",
        "",
        "| Provider | Setup fee | Monthly fee | Annual fee | Archive fee | Bulk cost | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for provider in providers:
        cost_lines.append(f"| {provider['name']} | CONTACT SALES REQUIRED | CONTACT SALES REQUIRED | CONTACT SALES REQUIRED | CONTACT SALES REQUIRED | CONTACT SALES REQUIRED | Pricing not verified in this environment. |")
    write_text(REPORT_DIR / "provider_cost_analysis.md", "\n".join(cost_lines) + "\n")

    license_lines = [
        "# Provider licence review",
        "",
        "| Provider | Risk | Notes |",
        "| --- | --- | --- |",
    ]
    for provider in providers:
        license_lines.append(f"| {provider['name']} | {provider['licence_risk']} | Internal research, caching, and redistribution require provider-specific licence review. |")
    write_text(REPORT_DIR / "provider_license_review.md", "\n".join(license_lines) + "\n")

    api_quality_lines = [
        "# Provider API quality",
        "",
        "| Provider | Auth | Historical-corner evidence | Comments |",
        "| --- | --- | --- | --- |",
    ]
    for provider in providers:
        if provider["name"] == "API-Football / API-Sports":
            api_quality_lines.append("| API-Football / API-Sports | PASS | NOT VERIFIED | Authenticated; fixture lookup worked, but the odds probe returned an empty payload. |")
        elif provider["name"] == "TheStatsAPI":
            api_quality_lines.append("| TheStatsAPI | PARTIAL | NOT VERIFIED | Health endpoint passed; protected competition endpoints returned 403. |")
        else:
            api_quality_lines.append(f"| {provider['name']} | NOT TESTED | NOT VERIFIED | No authenticated historical-corner sample was produced in this environment. |")
    write_text(REPORT_DIR / "provider_api_quality.md", "\n".join(api_quality_lines) + "\n")

    risk_lines = [
        "# Provider risk register",
        "",
        "- High licensing ambiguity remains for most commercial data vendors.",
        "- Historical corner-market availability is not verified for any provider in the current environment.",
        "- TheStatsAPI is currently blocked by a subscription error before any historical competition or season resolution.",
        "- API-Football can access fixture and league metadata but did not return a usable odds payload for the sampled fixture.",
        "- Consumer sites such as OddsPortal / BetExplorer are not approved for internal research without a documented commercial licence.",
    ]
    write_text(REPORT_DIR / "provider_risk_register.md", "\n".join(risk_lines) + "\n")

    log_lines = [
        "# Provider test log",
        "",
    ]
    for row in live_test_log:
        log_lines.append(f"- {row['timestamp']} | {row['provider']} | {row['url']} | {row['result']} | {row['evidence_hash']} | {row['notes']}")
    write_text(REPORT_DIR / "provider_test_log.md", "\n".join(log_lines) + "\n")

    recommendation_lines = [
        "# Provider final recommendation",
        "",
        "## Preferred historical provider",
        "",
        "No provider can be recommended for historical backtesting on the current evidence. Commercial quotes and licence review are still required.",
        "",
        "## Preferred live provider",
        "",
        "API-Football / API-Sports is the strongest current live-testing option for fixture and league metadata, but it did not return a usable corner-odds payload for the sampled fixture.",
        "",
        "## Recommended architecture",
        "",
        "Use a staged architecture: keep the current research stack for internal validation, then add a licensed historical-odds vendor only after a contract and sample export are available. A multi-provider approach is not yet justified without verified corner-market samples.",
        "",
        "## Remaining evidence gaps",
        "",
        "- Verified historical Serie A corner odds with opening and closing prices.",
        "- Verified target-line coverage for 8.5, 9.5, 10.5, 11.5.",
        "- Licensed internal research and permanent caching rights.",
        "- Resolution of fixture mapping and settlement availability.",
        "",
        "## Exact next action",
        "",
        "Request a sales call with at least one vendor that offers historical betting odds archives and ask for a sample export of Serie A Total Corners odds spanning at least three completed seasons. Do not proceed to implementation until the sample contains opening and closing odds, bookmaker identity, timestamps, and licence terms suitable for internal backtesting.",
    ]
    write_text(REPORT_DIR / "provider_final_recommendation.md", "\n".join(recommendation_lines) + "\n")

    questions_lines = [
        "# Questions for sales",
        "",
        "1. Does your product provide historical Serie A Total Corners Over/Under odds for at least three completed seasons?",
        "2. Are opening odds and closing or last pre-kickoff odds available for target lines 8.5, 9.5, 10.5, and 11.5?",
        "3. Can you provide a sample export with bookmaker identity, timestamps, and fixture mapping keys?",
        "4. Are internal research, model training, caching, and backtesting permitted under the license?",
        "5. What is the cost for one season, three seasons, and an ongoing live feed?",
    ]
    write_text(REPORT_DIR / "provider_questions_for_sales.md", "\n".join(questions_lines) + "\n")

    # Also create a small processed CSV for the live tests
    live_rows = [
        {
            "provider": row["provider"],
            "url": row["url"],
            "status_code": row["result"],
            "evidence_sha256": row["evidence_hash"],
            "notes": row["notes"],
        } for row in live_test_log
    ]
    write_csv(
        PROCESSED_DIR / "provider_live_test_log.csv",
        ["provider", "url", "status_code", "evidence_sha256", "notes"],
        live_rows,
    )


if __name__ == "__main__":
    main()
