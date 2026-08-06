from __future__ import annotations

from typing import Any

import pandas as pd


class WalkForwardValidationEngine:
    """Generate deterministic walk-forward folds without temporal leakage."""

    def __init__(self, *, train_length: int = 6, validation_length: int = 1, test_length: int = 1) -> None:
        self.train_length = train_length
        self.validation_length = validation_length
        self.test_length = test_length

    def generate_folds(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        if data.empty:
            return []

        frame = data.copy()
        if "fixture_date" not in frame.columns:
            raise ValueError("Missing fixture_date column")

        frame["fixture_date"] = pd.to_datetime(frame["fixture_date"], errors="coerce")
        if frame["fixture_date"].isna().any():
            raise ValueError("Fixture dates must be valid datetimes")

        if not frame["fixture_date"].is_monotonic_increasing:
            raise ValueError("Dataset must be chronologically sorted")

        if frame["fixture_date"].duplicated().any():
            raise ValueError("Duplicate fixture dates are not allowed")

        total_rows = len(frame)
        required = self.train_length + self.validation_length + self.test_length
        if required <= 0:
            raise ValueError("Window lengths must be positive")
        if total_rows < required:
            return []

        folds: list[dict[str, Any]] = []
        fold_index = 0
        while True:
            train_stop = self.train_length + fold_index
            validation_stop = train_stop + self.validation_length
            test_stop = validation_stop + self.test_length

            if test_stop > total_rows:
                break

            train_slice = frame.iloc[:train_stop]
            validation_slice = frame.iloc[train_stop:validation_stop]
            test_slice = frame.iloc[validation_stop:test_stop]

            if train_slice.empty or validation_slice.empty or test_slice.empty:
                break

            folds.append(
                {
                    "train_start": train_slice["fixture_date"].iloc[0],
                    "train_end": train_slice["fixture_date"].iloc[-1],
                    "validation_start": validation_slice["fixture_date"].iloc[0],
                    "validation_end": validation_slice["fixture_date"].iloc[-1],
                    "test_start": test_slice["fixture_date"].iloc[0],
                    "test_end": test_slice["fixture_date"].iloc[-1],
                    "train_indices": list(range(0, train_stop)),
                    "validation_indices": list(range(train_stop, validation_stop)),
                    "test_indices": list(range(validation_stop, test_stop)),
                }
            )
            fold_index += 1

        return folds
