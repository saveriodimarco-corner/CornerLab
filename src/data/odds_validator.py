from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.odds_contract import SUPPORTED_LINES, SUPPORTED_MARKETS, SUPPORTED_SIDES


def validate_odds_dataframe(odds: pd.DataFrame, fixtures: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    if odds is None or odds.empty:
        return pd.DataFrame(columns=[
            "match_id",
            "fixture_date",
            "home_team",
            "away_team",
            "bookmaker",
            "market",
            "line",
            "side",
            "opening_odds",
            "closing_odds",
            "odds_timestamp",
            "source",
            "source_fixture_id",
            "is_closing",
            "currency",
            "import_timestamp",
        ]), []

    errors: list[str] = []
    cleaned = odds.copy()

    required_columns = [
        "match_id",
        "fixture_date",
        "home_team",
        "away_team",
        "bookmaker",
        "market",
        "line",
        "side",
        "opening_odds",
        "closing_odds",
        "odds_timestamp",
        "source",
        "source_fixture_id",
        "is_closing",
        "currency",
        "import_timestamp",
    ]
    if not set(required_columns).issubset(cleaned.columns):
        missing = sorted(set(required_columns) - set(cleaned.columns))
        raise ValueError(f"Missing required odds columns: {missing}")

    if cleaned["match_id"].isna().any():
        errors.append("missing match_id")
    cleaned["match_id"] = pd.to_numeric(cleaned["match_id"], errors="coerce")

    for idx, row in cleaned.iterrows():
        if pd.isna(row["match_id"]):
            continue
        if not isinstance(row["source"], str) or not row["source"].strip():
            errors.append(f"missing source at row {idx}")
        market_value = str(row["market"]).strip() if pd.notna(row["market"]) else ""
        if market_value not in SUPPORTED_MARKETS:
            if "GOAL" in market_value.upper():
                errors.append(f"goal-total odds incorrectly labelled as corner odds at row {idx}")
            elif market_value:
                errors.append(f"unsupported market at row {idx}")
        line_value = str(row["line"]).strip() if pd.notna(row["line"]) else ""
        if line_value not in SUPPORTED_LINES:
            errors.append(f"unsupported lines at row {idx}")
        if pd.isna(row["opening_odds"]) or pd.isna(row["closing_odds"]):
            errors.append(f"missing odds at row {idx}")
        else:
            try:
                opening = float(row["opening_odds"])
                closing = float(row["closing_odds"])
            except Exception:
                errors.append(f"malformed odds at row {idx}")
                continue
            if opening <= 1.0 or closing <= 1.0:
                errors.append(f"odds <= 1.0 at row {idx}")

        try:
            datetime.fromisoformat(str(row["fixture_date"]).replace("Z", "+00:00"))
        except Exception:
            errors.append(f"malformed dates at row {idx}")

        try:
            datetime.fromisoformat(str(row["odds_timestamp"]).replace("Z", "+00:00"))
        except Exception:
            errors.append(f"malformed dates at row {idx}")

        if row.get("side") not in SUPPORTED_SIDES:
            errors.append(f"unsupported side at row {idx}")

    if fixtures is not None and not fixtures.empty:
        fixture_keys = set()
        for _, fixture in fixtures.iterrows():
            fixture_keys.add((str(fixture.get("home_team", "")).strip().lower(), str(fixture.get("away_team", "")).strip().lower(), str(fixture.get("date", "")).strip()))
        for _, row in cleaned.iterrows():
            if pd.isna(row["match_id"]):
                continue
            match_id = int(row["match_id"])
            fixture_match = fixtures[fixtures["match_id"].astype(str).str.strip() == str(match_id)]
            if fixture_match.empty:
                errors.append(f"unmatched fixtures for match_id {match_id}")
                continue
            fixture_row = fixture_match.iloc[0]
            key = (str(fixture_row.get("home_team", "")).strip().lower(), str(fixture_row.get("away_team", "")).strip().lower(), str(fixture_row.get("date", "")).strip())
            if (str(row["home_team"]).strip().lower(), str(row["away_team"]).strip().lower(), str(row["fixture_date"]).strip()) != key:
                errors.append(f"unmatched fixtures for match_id {match_id}")

    duplicates = cleaned.duplicated(subset=["bookmaker", "market", "line", "side", "odds_timestamp"], keep=False)
    if duplicates.any():
        errors.append("duplicate bookmaker/market/line/side/timestamp rows")

    if errors:
        return pd.DataFrame(columns=required_columns), errors

    return cleaned, []
