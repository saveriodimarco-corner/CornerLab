from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Abstract interface for all match-data providers."""

    name: str = "base"

    @abstractmethod
    def fetch_matches(self) -> List[Dict[str, Any]]:
        """Return a list of raw match dictionaries."""

    @abstractmethod
    def fetch_match_statistics(self) -> List[Dict[str, Any]]:
        """Return a list of raw match-statistics dictionaries."""

    @abstractmethod
    def fetch_teams(self) -> List[Dict[str, Any]]:
        """Return a list of raw team dictionaries."""
