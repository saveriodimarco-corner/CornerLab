from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


class ValidationError(ValueError):
    """Raised when input data does not meet the expected schema or quality checks."""


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a validation pass."""

    is_valid: bool
    errors: tuple[str, ...] = ()


class DataValidator:
    """Validate incoming match data before transformation or storage."""

    REQUIRED_COLUMNS = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_corners",
        "away_corners",
    }

    def validate(self, data: Any) -> ValidationResult:
        """Validate a dataframe or object that can be converted to a dataframe."""
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValidationError("Input data cannot be converted to a pandas DataFrame") from exc

        errors: List[str] = []
        if data.empty:
            errors.append("Input data must contain at least one match")

        missing = self.REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

        if not errors:
            self._check_duplicates(data, errors)
            self._check_dates(data, errors)
            self._check_teams(data, errors)
            self._check_corners(data, errors)
            self._check_season_format(data, errors)
            self._check_nulls(data, errors)

        return ValidationResult(is_valid=not errors, errors=tuple(errors))

    def _check_duplicates(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure fixtures are unique by a composite key."""
        key = data[["date", "home_team", "away_team", "season"]].copy()
        duplicates = key.duplicated()
        if duplicates.any():
            errors.append("Duplicate fixtures detected")

    def _check_dates(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure dates are valid and not in the future."""
        parsed = pd.to_datetime(data["date"], errors="coerce")
        if parsed.isna().any():
            errors.append("Invalid dates detected")
        else:
            future_mask = parsed > pd.Timestamp.today().normalize()
            if future_mask.any():
                errors.append("Future dates detected")

    def _check_teams(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure teams are present and non-empty."""
        for column in ["home_team", "away_team"]:
            if data[column].isna().any() or (data[column].astype(str).str.strip() == "").any():
                errors.append(f"Missing teams detected in column {column}")

    def _check_corners(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure corner counts are numeric and non-negative."""
        for column in ["home_corners", "away_corners"]:
            numeric_values = pd.to_numeric(data[column], errors="coerce")
            if numeric_values.isna().any():
                errors.append(f"Column {column} contains non-numeric values")
            elif (numeric_values < 0).any():
                errors.append(f"Column {column} contains negative values")

    def _check_season_format(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure seasons match the expected year/year format."""
        season_matches = data["season"].astype(str).str.fullmatch(r"\d{4}/\d{2}")
        if not season_matches.all() and not data["season"].empty:
            errors.append("Season values must follow the YYYY/YY format")

    def _check_nulls(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Ensure no null values remain across required columns."""
        required = list(self.REQUIRED_COLUMNS)
        if data[required].isna().any().any():
            errors.append("Null values detected in required columns")

    def ensure_valid(self, data: Any) -> pd.DataFrame:
        """Validate and return the dataframe or raise ValidationError."""
        result = self.validate(data)
        if not result.is_valid:
            raise ValidationError("; ".join(result.errors))
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
