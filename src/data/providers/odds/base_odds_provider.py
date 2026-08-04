from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseOddsProvider(ABC):
    name: str = "base"

    @abstractmethod
    def list_sports(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_historical_event_odds(self, event_id: str | None = None, sport: str | None = None, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def normalize_odds(self, payload: Any) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_usage(self) -> dict[str, Any]:
        raise NotImplementedError
