from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


VALID_VERDICTS = {
    "QUALIFIED_FOR_CURRENT_ODDS",
    "QUALIFIED_FOR_HISTORICAL_BACKTEST",
    "CURRENT_ONLY",
    "INSUFFICIENT_CORNER_COVERAGE",
    "MANUAL_REVIEW_REQUIRED",
    "AUTHENTICATION_REQUIRED",
    "NOT SUITABLE",
}


def evaluate_provider_qualification(record: dict[str, Any]) -> dict[str, Any]:
    verdict = "INSUFFICIENT_CORNER_COVERAGE"
    if not record.get("events_with_odds", 0):
        verdict = "AUTHENTICATION_REQUIRED" if not record.get("provider_name") else "INSUFFICIENT_CORNER_COVERAGE"
    elif record.get("events_with_corner_markets", 0) and record.get("over_available") and record.get("under_available"):
        if record.get("historical_depth_available") and record.get("timestamp_history_available") and record.get("settlement_available"):
            verdict = "QUALIFIED_FOR_HISTORICAL_BACKTEST"
        elif record.get("fixture_mapping_rate", 0.0) >= 0.5 and record.get("events_with_corner_markets", 0) >= 1:
            verdict = "QUALIFIED_FOR_CURRENT_ODDS"
        else:
            verdict = "CURRENT_ONLY"
    elif record.get("events_with_corner_markets", 0):
        verdict = "MANUAL_REVIEW_REQUIRED"
    if not record.get("lines_found"):
        verdict = "INSUFFICIENT_CORNER_COVERAGE"
    result = dict(record)
    result["probe_date"] = result.get("probe_date") or datetime.now(timezone.utc).isoformat()
    result["qualification_verdict"] = verdict
    return result


def build_provider_comparison_matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_provider_qualification(record) for record in records]
