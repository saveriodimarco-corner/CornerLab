from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH, override=False)


@dataclass
class CollectorConfig:
    db_path: Path | None = None
    api_football_key: Optional[str] = None
    the_odds_api_key: Optional[str] = None
    scheduler_interval_minutes: int = 10
    closing_odds_cutoff_minutes: int = 5
    request_timeout_seconds: int = 15
    max_retries: int = 2
    rate_limit_threshold: int = 20
    odds_retry_ttl_minutes: int = 30
    reports_dir: Path = field(default_factory=lambda: REPO_ROOT / "reports")
    docs_dir: Path = field(default_factory=lambda: REPO_ROOT / "docs")
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.data_dir / "collector.sqlite"
        self.db_path = Path(self.db_path)
        self.reports_dir = Path(self.reports_dir)
        self.docs_dir = Path(self.docs_dir)
        self.data_dir = Path(self.data_dir)
        self.api_football_key = self.api_football_key or os.getenv("API_FOOTBALL_KEY")
        self.the_odds_api_key = self.the_odds_api_key or os.getenv("THE_ODDS_API_KEY")

    def now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def redact_text(value: str) -> str:
    if not value:
        return ""
    redacted = re.sub(r"([A-Za-z0-9_\-]{3,})", "[REDACTED]", value)
    return redacted


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
