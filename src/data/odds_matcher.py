from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd


class OddsMatcher:
    def __init__(self, team_aliases: dict[str, list[str]] | None = None) -> None:
        self.team_aliases = team_aliases or {}

    def _normalize_team(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_aliases(self, value: Any) -> set[str]:
        aliases = {self._normalize_team(value)}
        for alias in self.team_aliases.get(self._normalize_team(value), []):
            aliases.add(self._normalize_team(alias))
        return aliases

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def match_event_to_fixture(self, event: dict[str, Any], fixtures: pd.DataFrame, tolerance_minutes: int = 30, competition: str | None = None, season: str | None = None) -> dict[str, Any]:
        if fixtures is None or fixtures.empty:
            return {"match_status": "UNMATCHED", "match_id": None, "reason": "no fixtures"}

        home_aliases = self._build_aliases(event.get("home_team"))
        away_aliases = self._build_aliases(event.get("away_team"))

        event_time = self._parse_datetime(event.get("commence_time"))
        candidates: list[tuple[dict[str, Any], int, int]] = []
        for _, fixture in fixtures.iterrows():
            fixture_home = self._normalize_team(fixture.get("home_team"))
            fixture_away = self._normalize_team(fixture.get("away_team"))
            home_match = fixture_home in home_aliases
            away_match = fixture_away in away_aliases
            competition_match = True
            if competition is not None and fixture.get("competition") is not None:
                competition_match = self._normalize_team(fixture.get("competition")) == self._normalize_team(competition)
            season_match = True
            if season is not None and fixture.get("season") is not None:
                season_match = str(fixture.get("season")) == str(season)
            if home_match and away_match and competition_match and season_match:
                fixture_time = self._parse_datetime(fixture.get("date"))
                if event_time is not None and fixture_time is not None:
                    delta_minutes = abs(int((fixture_time - event_time).total_seconds() // 60))
                    if delta_minutes <= tolerance_minutes:
                        candidates.append((fixture.to_dict(), 1, delta_minutes))
                    else:
                        candidates.append((fixture.to_dict(), 0, delta_minutes))
                else:
                    candidates.append((fixture.to_dict(), 1, 0))

        if not candidates:
            return {"match_status": "UNMATCHED", "match_id": None, "reason": "no team match"}

        scored = sorted(candidates, key=lambda item: (-item[1], item[2]))
        best = scored[0][0]
        exact_matches = [candidate for candidate in scored if candidate[1] == 1]
        if len(exact_matches) > 1:
            return {"match_status": "AMBIGUOUS", "match_id": None, "reason": "multiple plausible fixtures"}
        return {"match_status": "MATCHED", "match_id": int(best.get("match_id", 0)), "reason": "matched"}
