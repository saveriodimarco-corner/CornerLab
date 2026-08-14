from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.normalizer import NormalizationError, normalize_competition_name, normalize_date, normalize_season_label, normalize_team_name


class QualityIssue(Exception):
    """Raised when a row fails quality checks."""


def validate_quality_rows(rows: List[Dict[str, Any]], source: str = "historical") -> Dict[str, List[str]]:
    """Validate canonical historical or live rows without mutating source values."""
    errors: List[str] = []
    warnings: List[str] = []
    duplicates: List[str] = []
    team_issues: List[str] = []
    chronology_errors: List[str] = []
    impossible_values: List[str] = []
    total_mismatches: List[str] = []
    provider_duplicates: List[str] = []
    competition_collisions: List[str] = []
    parsed_totals: List[float] = []
    historical_keys: set[tuple[str, str, str, str]] = set()
    provider_keys: set[tuple[str, str, str]] = set()
    competition_fixture_keys: Dict[tuple[str, str], tuple[str, str]] = {}
    dates: List[datetime] = []

    required = ["competition", "home_team", "away_team"]
    if source == "historical":
        required.extend(["date", "season", "home_corners", "away_corners"])

    for index, row in enumerate(rows):
        identifier = str(row.get("fixture_id") or row.get("provider_fixture_id") or index)
        missing = [field for field in required if row.get(field) is None or (isinstance(row.get(field), str) and not row.get(field).strip())]
        if source == "live" and not (row.get("kickoff_utc") or row.get("date")):
            missing.append("kickoff_utc")
        if missing:
            errors.append(f"{identifier}: missing required columns {missing}")
            continue
        try:
            date_value = normalize_date(row.get("date") or row.get("kickoff_utc"))
            parsed_date = datetime.fromisoformat(date_value)
            competition = normalize_competition_name(row["competition"])
            home_team = normalize_team_name(row["home_team"])
            away_team = normalize_team_name(row["away_team"])
        except NormalizationError as exc:
            errors.append(f"{identifier}: {exc}")
            continue
        if home_team == away_team:
            issue = f"{identifier}: home and away team are identical"
            errors.append(issue)
            team_issues.append(issue)
        dates.append(parsed_date)
        historical_key = (competition.casefold(), date_value, home_team.casefold(), away_team.casefold())
        if historical_key in historical_keys:
            issue = f"{identifier}: duplicate historical fixture {historical_key}"
            errors.append(issue)
            duplicates.append(issue)
        historical_keys.add(historical_key)

        provider = str(row.get("provider") or "").strip()
        provider_fixture_id = str(row.get("provider_fixture_id") or "").strip()
        if source == "live" and (not provider or not provider_fixture_id):
            errors.append(f"{identifier}: provider identity is required")
        if provider and provider_fixture_id:
            provider_key = (competition.casefold(), provider.casefold(), provider_fixture_id)
            if provider_key in provider_keys:
                issue = f"{identifier}: duplicate provider fixture {provider_key}"
                errors.append(issue)
                provider_duplicates.append(issue)
            provider_keys.add(provider_key)
        fixture_id = row.get("fixture_id")
        if fixture_id is not None:
            collision_key = (competition.casefold(), str(fixture_id))
            teams = (home_team.casefold(), away_team.casefold())
            if collision_key in competition_fixture_keys and competition_fixture_keys[collision_key] != teams:
                issue = f"{identifier}: conflicting fixture identity {collision_key}"
                errors.append(issue)
                competition_collisions.append(issue)
            competition_fixture_keys[collision_key] = teams

        if source == "historical":
            try:
                season = normalize_season_label(row["season"])
                start_year = int(season[:4])
                if not datetime(start_year, 7, 1) <= parsed_date <= datetime(start_year + 1, 6, 30):
                    issue = f"{identifier}: scientific chronology failure outside season {season}"
                    errors.append(issue)
                    chronology_errors.append(issue)
                if parsed_date.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc):
                    issue = f"{identifier}: future result row"
                    errors.append(issue)
                    chronology_errors.append(issue)
            except NormalizationError as exc:
                errors.append(f"{identifier}: {exc}")
            numeric_values: Dict[str, float] = {}
            for field in ["home_corners", "away_corners", "total_corners", "home_goals", "away_goals"]:
                if field not in row or row.get(field) is None:
                    continue
                try:
                    value = float(row[field])
                except (TypeError, ValueError):
                    value = float("nan")
                if not math.isfinite(value) or value < 0:
                    issue = f"{identifier}: impossible {field}={row.get(field)!r}"
                    errors.append(issue)
                    impossible_values.append(issue)
                else:
                    numeric_values[field] = value
            if {"home_corners", "away_corners", "total_corners"}.issubset(numeric_values) and numeric_values["total_corners"] != numeric_values["home_corners"] + numeric_values["away_corners"]:
                issue = f"{identifier}: total_corners mismatch"
                errors.append(issue)
                total_mismatches.append(issue)
            if "home_corners" in numeric_values and "away_corners" in numeric_values:
                parsed_totals.append(numeric_values["home_corners"] + numeric_values["away_corners"])

    if len(dates) > 1 and dates != sorted(dates):
        warnings.append("SOURCE_ORDER_WARNING: source rows are not chronological; pipeline sorting is required before scientific use")
    if len(parsed_totals) >= 4:
        ordered = sorted(parsed_totals)
        lower = ordered[int(0.25 * (len(ordered) - 1))]
        upper = ordered[int(0.75 * (len(ordered) - 1))]
        threshold = upper + 3.0 * (upper - lower)
        for index, total in enumerate(parsed_totals):
            if total > threshold:
                warnings.append(f"OUTLIER_WARNING: total_corners={total:g} exceeds historical IQR threshold {threshold:g} at valid row {index}")

    return {
        "errors": errors,
        "warnings": warnings,
        "duplicates": duplicates + provider_duplicates + competition_collisions,
        "team_normalization_issues": team_issues,
        "chronology_errors": chronology_errors,
        "impossible_value_errors": impossible_values,
        "outlier_warnings": [item for item in warnings if item.startswith("OUTLIER_WARNING")],
        "total_corner_mismatches": total_mismatches,
    }


def build_quality_report(rows: List[Dict[str, Any]]) -> str:
    validation = validate_quality_rows(rows)
    duplicate_matches = detect_duplicate_matches(rows)
    missing_statistics = detect_missing_statistics(rows)
    invalid_totals = detect_invalid_totals(rows)
    team_consistency = detect_team_consistency(rows)
    date_consistency = detect_date_consistency(rows)

    lines = ["# Data Quality Report", "", f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.extend([
        f"- ROWS CHECKED = {len(rows)}",
        f"- ERROR COUNT = {len(validation['errors'])}",
        f"- WARNING COUNT = {len(validation['warnings'])}",
        f"- DUPLICATES = {len(validation['duplicates'])}",
        f"- TEAM NORMALIZATION ISSUES = {len(validation['team_normalization_issues'])}",
        f"- CHRONOLOGY ERRORS = {len(validation['chronology_errors'])}",
        f"- IMPOSSIBLE VALUE ERRORS = {len(validation['impossible_value_errors'])}",
        f"- OUTLIER WARNINGS = {len(validation['outlier_warnings'])}",
        f"- TOTAL CORNER MISMATCHES = {len(validation['total_corner_mismatches'])}",
        "",
    ])
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
