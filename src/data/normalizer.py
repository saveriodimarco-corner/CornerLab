from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class NormalizationError(ValueError):
    """Raised when a row cannot be normalized safely."""


def normalize_date(value: Any) -> str:
    if value is None:
        raise NormalizationError("date is required")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise NormalizationError("date is required")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d")
            except ValueError as exc:
                raise NormalizationError("invalid date") from exc
        return parsed.strftime("%Y-%m-%d")
    raise NormalizationError("invalid date")


def normalize_team_name(value: Any) -> str:
    if not isinstance(value, str):
        raise NormalizationError("team name must be a string")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise NormalizationError("team name is required")
    return cleaned.title()


def normalize_competition_name(value: Any) -> str:
    if not isinstance(value, str):
        raise NormalizationError("competition name must be a string")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise NormalizationError("competition name is required")
    return cleaned.title()


def normalize_season_label(value: Any) -> str:
    if not isinstance(value, str):
        raise NormalizationError("season label must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise NormalizationError("season label is required")
    if "/" in cleaned:
        return cleaned
    if len(cleaned) == 4 and cleaned.isdigit():
        return f"{cleaned}/{str(int(cleaned) + 1)[-2:]}"
    raise NormalizationError("invalid season label")


def normalize_corner_value(value: Any, field_name: str) -> int:
    if value is None:
        raise NormalizationError(f"{field_name} is required")
    try:
        converted = int(float(value))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"{field_name} must be numeric") from exc
    if converted < 0:
        raise NormalizationError(f"{field_name} cannot be negative")
    return converted


def normalize_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "date": normalize_date(raw_row.get("date")),
        "season": normalize_season_label(raw_row.get("season") or raw_row.get("season_label") or ""),
        "competition": normalize_competition_name(raw_row.get("competition") or raw_row.get("competition_name") or ""),
        "home_team": normalize_team_name(raw_row.get("home_team")),
        "away_team": normalize_team_name(raw_row.get("away_team")),
        "home_corners": normalize_corner_value(raw_row.get("home_corners"), "home_corners"),
        "away_corners": normalize_corner_value(raw_row.get("away_corners"), "away_corners"),
    }
    return normalized
