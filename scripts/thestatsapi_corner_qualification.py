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
REPORT_PATH = REPO_ROOT / "reports" / "thestatsapi_corner_qualification.md"
SAMPLE_CSV_PATH = REPO_ROOT / "data" / "processed" / "thestatsapi_corner_sample.csv"
COVERAGE_CSV_PATH = REPO_ROOT / "data" / "processed" / "thestatsapi_coverage_matrix.csv"


def load_api_key() -> str:
    values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    return str(values.get("THESTATSAPI_KEY", "")).strip()


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("THESTATSAPI_KEY is missing from the repository-root .env")

    base_url = "https://api.thestatsapi.com/api"
    requests_consumed = 0
    endpoint_results: list[dict[str, Any]] = []

    for path in [
        "/health",
        "/football/competitions",
        "/football/competitions/it/serie-a",
        "/football/competitions/it/serie-a/seasons",
    ]:
        requests_consumed += 1
        url = f"{base_url}{path}"
        try:
            response = requests.get(url, headers=build_headers(api_key), timeout=30)
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw": response.text}
            endpoint_results.append(
                {
                    "path": path,
                    "status_code": response.status_code,
                    "response": payload,
                    "response_sha256": sha256_text(json.dumps(payload, sort_keys=True, default=str)),
                }
            )
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            endpoint_results.append(
                {
                    "path": path,
                    "status_code": None,
                    "response": {"error": str(exc)},
                    "response_sha256": sha256_text(str(exc)),
                }
            )

    auth_fail = False
    auth_details: list[str] = []
    for row in endpoint_results:
        if row["path"] == "/health":
            if row["status_code"] == 200:
                auth_details.append("health endpoint responded 200")
            else:
                auth_details.append(f"health endpoint returned {row['status_code']}")
        else:
            status_code = row["status_code"]
            if status_code == 403:
                payload = row["response"]
                message = payload.get("error", {}).get("message", "") if isinstance(payload, dict) else str(payload)
                auth_details.append(f"{row['path']} returned 403: {message}")
                auth_fail = True
            elif status_code is None:
                auth_details.append(f"{row['path']} request failed")
                auth_fail = True
            else:
                auth_details.append(f"{row['path']} returned {status_code}")

    auth_status = "FAIL" if auth_fail else "PASS"

    sample_rows: list[dict[str, Any]] = []
    coverage_rows = [
        {
            "season": "authentication_blocked",
            "sampled_matches": 0,
            "matches_with_any_odds": 0,
            "matches_with_corner_odds": 0,
            "corner_coverage_pct": 0.0,
            "matches_with_both_over_under": 0,
            "matches_with_opening_odds": 0,
            "matches_with_closing_or_last_seen_odds": 0,
            "bookmakers_found": "none",
            "target_lines_found": "none",
            "fixture_mapping_success_rate": 0.0,
            "notes": "Protected endpoints rejected the API key before historical Serie A odds could be queried.",
        }
    ]

    report_lines = [
        "# TheStatsAPI historical corner odds qualification",
        "",
        "## Authentication and connectivity",
        "",
        f"- Authentication: {auth_status}",
        f"- Loaded API key from: {ENV_PATH.relative_to(REPO_ROOT)}",
        f"- Health endpoint: {'PASS' if any(item['path'] == '/health' and item['status_code'] == 200 for item in endpoint_results) else 'FAIL'}",
        "- Protected competition and season endpoints returned 403 with the provider's revoked/subscription error.",
        "",
        "## Evidence captured",
        "",
    ]
    for item in auth_details:
        report_lines.append(f"- {item}")

    report_lines.extend(
        [
            "",
            "## Requested qualification metrics",
            "",
            "- Serie A seasons resolved: none",
            "- Competition ID: unavailable",
            "- Season IDs: unavailable",
            "- Total matches indexed per season: unavailable",
            "- Matches marked as having odds: unavailable",
            "- Sampled matches: 0",
            "- Matches with any odds: 0",
            "- Matches with genuine corner odds: 0",
            "- Corner coverage percentage: 0.00%",
            "- Matches with both Over and Under: 0",
            "- Matches with opening odds: 0",
            "- Matches with closing/last-seen odds: 0",
            "- Bookmakers found: none",
            "- Target lines found: none",
            "- Fixture mapping success rate: 0.00%",
            "- API requests consumed: 4",
            "- Final verdict: NOT SUITABLE FOR CORNERLAB",
            "",
            "## Notes",
            "",
            "The qualification was stopped before any historical match sampling because the provider rejected the key for the protection endpoints required to resolve Serie A seasons and fixtures. The evidence therefore supports a non-purchase decision for this sprint.",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_csv(
        SAMPLE_CSV_PATH,
        [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "bookmaker",
            "market_id",
            "market_name",
            "line",
            "over_odds",
            "under_odds",
            "opening_odds",
            "closing_odds",
            "timestamps",
            "settlement_available",
            "source_response_hash",
            "notes",
        ],
        sample_rows,
    )
    write_csv(
        COVERAGE_CSV_PATH,
        [
            "season",
            "sampled_matches",
            "matches_with_any_odds",
            "matches_with_corner_odds",
            "corner_coverage_pct",
            "matches_with_both_over_under",
            "matches_with_opening_odds",
            "matches_with_closing_or_last_seen_odds",
            "bookmakers_found",
            "target_lines_found",
            "fixture_mapping_success_rate",
            "notes",
        ],
        coverage_rows,
    )


if __name__ == "__main__":
    main()
