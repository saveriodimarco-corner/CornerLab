from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo


ROME = ZoneInfo("Europe/Rome")
CACHE_MAX_AGE = timedelta(hours=84)
BASE_URL = "https://" + "www.football-data.co.uk/mmz4281"

DIVISION_CODES = {
    "Serie A": "I1",
}

TEAM_ALIASES = {
    "as roma": "roma",
    "roma": "roma",
    "ac milan": "milan",
    "milan": "milan",
    "inter milan": "inter",
    "internazionale": "inter",
    "inter": "inter",
    "atalanta bc": "atalanta",
    "atalanta": "atalanta",
    "hellas verona": "verona",
    "verona": "verona",
}


class FootballDataResultsError(RuntimeError):
    pass


class FootballDataResultsProvider:
    def __init__(
        self,
        data_dir: Path | str,
        *,
        timeout_seconds: int = 15,
        opener: Callable[..., Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_dir = (
            Path(data_dir)
            / "provider_cache"
            / "football_data"
        )
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urlopen
        self._now = now or (
            lambda: datetime.now(timezone.utc)
        )

    @staticmethod
    def _parse_kickoff(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _season_code(kickoff: datetime) -> str:
        local = kickoff.astimezone(ROME)
        start_year = (
            local.year
            if local.month >= 7
            else local.year - 1
        )
        end_year = start_year + 1
        return (
            f"{start_year % 100:02d}"
            f"{end_year % 100:02d}"
        )

    @staticmethod
    def _normalize_team(value: Any) -> str:
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value or "").casefold(),
        ).strip()
        return TEAM_ALIASES.get(normalized, normalized)

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _row_date(value: Any) -> datetime.date | None:
        raw = str(value or "").strip()

        for pattern in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(
                    raw,
                    pattern,
                ).date()
            except ValueError:
                continue

        return None

    def _source(
        self,
        competition: str,
        kickoff: datetime,
    ) -> tuple[str, Path] | None:
        division = DIVISION_CODES.get(competition)
        if division is None:
            return None

        season = self._season_code(kickoff)
        url = f"{BASE_URL}/{season}/{division}.csv"
        cache_path = self.cache_dir / f"{season}_{division}.csv"
        return url, cache_path

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False

        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        )
        return self._now() - modified <= CACHE_MAX_AGE

    def _download(self, url: str) -> bytes:
        with self._opener(
            url,
            timeout=self.timeout_seconds,
        ) as response:
            return response.read()

    def _load_content(
        self,
        url: str,
        cache_path: Path,
    ) -> bytes:
        if self._cache_is_fresh(cache_path):
            return cache_path.read_bytes()

        try:
            content = self._download(url)
        except Exception as exc:
            if cache_path.exists():
                return cache_path.read_bytes()
            raise FootballDataResultsError(
                "football_data_download_failed"
            ) from exc

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(cache_path)
        return content

    def resolve_fixture(
        self,
        fixture: dict[str, Any],
    ) -> dict[str, Any]:
        competition = str(
            fixture.get("competition") or ""
        ).strip()
        kickoff = self._parse_kickoff(
            fixture.get("kickoff_utc")
        )

        if kickoff is None:
            return {
                "ok": False,
                "reason": "kickoff_invalid",
            }

        source = self._source(competition, kickoff)
        if source is None:
            return {
                "ok": False,
                "reason": "competition_not_supported",
            }

        url, cache_path = source

        try:
            content = self._load_content(
                url,
                cache_path,
            )
        except FootballDataResultsError:
            return {
                "ok": False,
                "reason": "football_data_unavailable",
            }

        try:
            decoded = content.decode("utf-8-sig")
            rows = list(
                csv.DictReader(io.StringIO(decoded))
            )
        except (UnicodeDecodeError, csv.Error):
            return {
                "ok": False,
                "reason": "football_data_invalid",
            }

        target_date = kickoff.astimezone(ROME).date()
        target_home = self._normalize_team(
            fixture.get("home_team")
        )
        target_away = self._normalize_team(
            fixture.get("away_team")
        )

        matches = [
            row
            for row in rows
            if self._row_date(row.get("Date"))
            == target_date
            and self._normalize_team(
                row.get("HomeTeam")
            )
            == target_home
            and self._normalize_team(
                row.get("AwayTeam")
            )
            == target_away
        ]

        if not matches:
            return {
                "ok": False,
                "reason": "football_data_match_not_found",
                "source_url": url,
            }

        if len(matches) != 1:
            return {
                "ok": False,
                "reason": "football_data_match_ambiguous",
                "source_url": url,
            }

        row = matches[0]
        home_score = self._integer(row.get("FTHG"))
        away_score = self._integer(row.get("FTAG"))
        home_corners = self._integer(row.get("HC"))
        away_corners = self._integer(row.get("AC"))

        if None in (
            home_score,
            away_score,
            home_corners,
            away_corners,
        ):
            return {
                "ok": False,
                "reason": "football_data_result_incomplete",
                "source_url": url,
            }

        return {
            "ok": True,
            "reason": "resolved",
            "status": "FT",
            "source_url": url,
            "result": {
                "home_score": home_score,
                "away_score": away_score,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": (
                    home_corners + away_corners
                ),
                "provider": "football-data-csv",
            },
        }
