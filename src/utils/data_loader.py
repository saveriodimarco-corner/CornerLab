from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from sqlalchemy import create_engine

from src.config import CONFIG
from src.utils.validator import DataValidator


class DataLoader:
    """Shared loader for CSV, Parquet, and SQLite-backed match data."""

    def __init__(self, validator: Optional[DataValidator] = None) -> None:
        """Initialize the loader with a validator instance."""
        self.validator = validator or DataValidator()

    def load(self, source: Union[str, Path, pd.DataFrame], *, validate: bool = True) -> pd.DataFrame:
        """Load data from CSV, parquet, SQLite, or a dataframe object."""
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".csv":
                data = pd.read_csv(path)
            elif path.suffix.lower() in {".parquet", ".pq"}:
                data = pd.read_parquet(path)
            elif path.exists() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                data = self._load_sqlite(path)
            else:
                raise ValueError("Unsupported source format")
        else:
            raise TypeError("source must be a path, dataframe, or sqlite path")

        if validate:
            self.validator.ensure_valid(data)
        return data

    def _load_sqlite(self, path: Path) -> pd.DataFrame:
        """Load a dataframe from a sqlite database path if a table exists."""
        engine = create_engine(f"sqlite:///{path}")
        with engine.connect() as connection:
            table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", connection)
        if table_names.empty:
            raise ValueError("No tables found in SQLite database")

        preferred_tables = ["matches", "fixtures", "games", "game_data"]
        for table_name in preferred_tables:
            if table_name in table_names["name"].tolist():
                return pd.read_sql_table(table_name, engine)

        first_table = table_names["name"].iloc[0]
        return pd.read_sql_table(first_table, engine)

    def ensure_output_dir(self, path: Union[str, Path]) -> Path:
        """Ensure an output directory exists for a parquet artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def save_parquet(self, data: pd.DataFrame, output_path: Union[str, Path]) -> Path:
        """Persist a dataframe to parquet after validation."""
        self.validator.ensure_valid(data)
        output = self.ensure_output_dir(output_path)
        data.to_parquet(output, index=False)
        return output
