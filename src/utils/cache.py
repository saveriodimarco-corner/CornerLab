from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class CacheManager:
    """Reuse persisted parquet artifacts when the source file is unchanged."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the cache manager with an optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, source_path: Union[str, Path, pd.DataFrame], output_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Return cached data if the source file is unchanged and the output exists."""
        output = Path(output_path)
        if not output.exists():
            return None

        if isinstance(source_path, pd.DataFrame):
            return None

        source = Path(source_path)
        if not source.exists():
            return None
        if output.stat().st_mtime < source.stat().st_mtime:
            return None
        return pd.read_parquet(output)

    def set(self, data: pd.DataFrame, output_path: Union[str, Path]) -> Path:
        """Persist data to the cache path."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output, index=False)
        return output
