from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class QualityIssue(Exception):
    """Raised when a row fails quality checks."""


def build_quality_report(rows: List[Dict[str, Any]]) -> str:
    duplicate_matches = detect_duplicate_matches(rows)
    missing_statistics = detect_missing_statistics(rows)
    invalid_totals = detect_invalid_totals(rows)
    team_consistency = detect_team_consistency(rows)
    date_consistency = detect_date_consistency(rows)

    lines = ["# Data Quality Report", "", f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"- Duplicate matches: {len(duplicate_matches)}")
    lines.append(f"- Missing statistics: {len(missing_statistics)}")
    lines.append(f"- Invalid totals: {len(invalid_totals)}")
    lines.append(f"- Team consistency issues: {len(team_consistency)}")
    lines.append(f"- Date consistency issues: {len(date_consistency)}")
    lines.append("")
    if duplicate_matches:
        lines.append("## Duplicate matches")
        for item in duplicate_matches:
            lines.append(f"- {item}")
        lines.append("")
    if missing_statistics:
        lines.append("## Missing statistics")
        for item in missing_statistics:
            lines.append(f"- {item}")
        lines.append("")
    if invalid_totals:
        lines.append("## Invalid totals")
        for item in invalid_totals:
            lines.append(f"- {item}")
        lines.append("")
    if team_consistency:
        lines.append("## Team consistency")
        for item in team_consistency:
            lines.append(f"- {item}")
        lines.append("")
    if date_consistency:
        lines.append("## Date consistency")
        for item in date_consistency:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def detect_duplicate_matches(rows: List[Dict[str, Any]]) -> List[str]:
    seen = {}
    issues = []
    for row in rows:
        key = (row.get("date"), row.get("home_team"), row.get("away_team"), row.get("competition"))
        if key in seen:
            issues.append(f"Duplicate match: {key}")
        else:
            seen[key] = row
    return issues


def detect_missing_statistics(rows: List[Dict[str, Any]]) -> List[str]:
    issues = []
    for row in rows:
        if row.get("home_corners") is None or row.get("away_corners") is None:
            issues.append(f"Missing statistics: {row.get('home_team')} vs {row.get('away_team')}")
    return issues


def detect_invalid_totals(rows: List[Dict[str, Any]]) -> List[str]:
    issues = []
    for row in rows:
        try:
            total = int(row.get("home_corners", 0)) + int(row.get("away_corners", 0))
        except (TypeError, ValueError):
            total = None
        if total is None or total < 0:
            issues.append(f"Invalid totals: {row.get('home_team')} vs {row.get('away_team')}")
    return issues


def detect_team_consistency(rows: List[Dict[str, Any]]) -> List[str]:
    issues = []
    for row in rows:
        home_team = (row.get("home_team") or "").strip()
        away_team = (row.get("away_team") or "").strip()
        if not home_team or not away_team or home_team == away_team:
            issues.append(f"Team consistency issue: {home_team} vs {away_team}")
    return issues


def detect_date_consistency(rows: List[Dict[str, Any]]) -> List[str]:
    issues = []
    for row in rows:
        date_value = row.get("date")
        if not date_value:
            issues.append("Date consistency issue: missing date")
            continue
        try:
            datetime.fromisoformat(str(date_value))
        except ValueError:
            try:
                datetime.strptime(str(date_value), "%Y-%m-%d")
            except ValueError:
                issues.append(f"Date consistency issue: {date_value}")
    return issues


def compute_row_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
