from __future__ import annotations

import pandas as pd

from src.research.walk_forward_validation_engine import WalkForwardValidationEngine


def _build_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fixture_date": pd.to_datetime(
                [
                    "2023-09-01",
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                ]
            ),
            "value": list(range(10)),
        }
    )


def test_generate_folds_respects_temporal_ordering() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset()

    folds = engine.generate_folds(data)

    assert len(folds) == 6
    assert folds[0]["train_start"] == pd.Timestamp("2023-09-01")
    assert folds[0]["train_end"] == pd.Timestamp("2023-11-01")
    assert folds[0]["validation_start"] == pd.Timestamp("2023-12-01")
    assert folds[0]["validation_end"] == pd.Timestamp("2023-12-01")
    assert folds[0]["test_start"] == pd.Timestamp("2024-01-01")
    assert folds[0]["test_end"] == pd.Timestamp("2024-01-01")
    assert folds[-1]["test_end"] == pd.Timestamp("2024-06-01")


def test_generate_folds_has_no_leakage_between_windows() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset()

    folds = engine.generate_folds(data)

    for fold in folds:
        train_dates = data.iloc[fold["train_indices"]]["fixture_date"]
        validation_dates = data.iloc[fold["validation_indices"]]["fixture_date"]
        test_dates = data.iloc[fold["test_indices"]]["fixture_date"]

        assert train_dates.max() < validation_dates.min()
        assert validation_dates.max() < test_dates.min()


def test_generate_folds_has_no_overlap_between_windows() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset()

    folds = engine.generate_folds(data)

    for fold in folds:
        assert set(fold["train_indices"]).isdisjoint(set(fold["validation_indices"]))
        assert set(fold["train_indices"]).isdisjoint(set(fold["test_indices"]))
        assert set(fold["validation_indices"]).isdisjoint(set(fold["test_indices"]))


def test_generate_folds_is_deterministic() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset()

    first = engine.generate_folds(data)
    second = engine.generate_folds(data.copy())

    assert first == second


def test_unsorted_dataset_is_rejected() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset().iloc[::-1].reset_index(drop=True)

    try:
        engine.generate_folds(data)
    except ValueError as exc:
        assert "chronologically sorted" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for unsorted dataset")


def test_duplicate_dates_are_rejected() -> None:
    engine = WalkForwardValidationEngine(train_length=3, validation_length=1, test_length=1)
    data = _build_dataset().copy()
    data.loc[0, "fixture_date"] = data.loc[1, "fixture_date"]

    try:
        engine.generate_folds(data)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for duplicate fixture dates")
