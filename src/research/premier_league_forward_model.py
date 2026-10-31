from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COMPETITION = "Premier League"
MODEL_NAME = "poisson_alpha_10_online_bias"
ELIGIBLE_LINES = ("8.5", "11.5")
CALIBRATION_WINDOW = 80
CALIBRATION_SHRINKAGE = 0.5
CALIBRATION_MINIMUM_HISTORY = 20

FEATURES = (
    "league_mean_last_20",
    "league_mean_last_50",
    "league_mean_last_100",
    "season_expanding_mean",
    "expected_total_corners_baseline",
    "expected_total_corners",
    "total_corners_last5",
    "total_corners_last10",
    "away_total_corners_last5",
    "corners_for_last5",
    "away_corners_for_last5",
    "corners_against_last5",
    "away_corners_against_last5",
)


class PremierLeagueForwardModel:
    def __init__(
        self,
        base_model: Any,
        bias_correction: float,
        feature_names: tuple[str, ...] = FEATURES,
    ) -> None:
        self.base_model = base_model
        self.bias_correction = float(
            bias_correction
        )
        self.feature_names_in_ = np.asarray(
            feature_names,
            dtype=object,
        )

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        prediction = np.asarray(
            self.base_model.predict(frame),
            dtype=float,
        )
        return np.clip(
            prediction - self.bias_correction,
            0.1,
            30.0,
        )


def add_safe_time_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.sort_values(
        ["season", "date", "match_id"],
        kind="stable",
    ).reset_index(drop=True).copy()

    target = result[
        "actual_total_corners"
    ].astype(float)
    prior = target.shift(1)

    for window in (20, 50, 100):
        result[
            f"league_mean_last_{window}"
        ] = prior.rolling(
            window,
            min_periods=20,
        ).mean()

    result["season_expanding_mean"] = (
        result.groupby("season")[
            "actual_total_corners"
        ]
        .transform(
            lambda values: (
                values.astype(float)
                .shift(1)
                .expanding(min_periods=5)
                .mean()
            )
        )
    )

    return result


def build_base_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                PoissonRegressor(
                    alpha=10.0,
                    max_iter=10000,
                ),
            ),
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()



def load_forward_model(
    base_dir: Path | str,
) -> tuple[
    PremierLeagueForwardModel,
    dict[str, Any],
]:
    base_dir = Path(base_dir)
    cache_dir = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    manifest_path = (
        cache_dir
        / "premier_league_forward_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest non trovato: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    required_contract = {
        "mode": "forward_validation_only",
        "paper_trading_enabled": True,
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    for field, expected in required_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(
                "Contratto di sicurezza non valido: "
                f"{field}"
            )

    if (
        manifest.get("feature_names")
        != list(FEATURES)
    ):
        raise ValueError(
            "Schema feature del modello non valido"
        )

    if (
        manifest.get("eligible_lines")
        != list(ELIGIBLE_LINES)
    ):
        raise ValueError(
            "Linee abilitate non valide"
        )

    artifact_value = str(
        manifest.get("artifact_path") or ""
    ).strip()

    if not artifact_value:
        raise ValueError(
            "Percorso artefatto mancante"
        )

    artifact_path = Path(artifact_value)

    if not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path

    if (
        artifact_path.resolve().parent
        != cache_dir.resolve()
    ):
        raise ValueError(
            "Artefatto fuori dalla directory isolata"
        )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artefatto non trovato: {artifact_path}"
        )

    expected_hash = str(
        manifest.get("artifact_hash") or ""
    )
    actual_hash = _sha256(artifact_path)

    if (
        not expected_hash
        or actual_hash != expected_hash
    ):
        raise ValueError(
            "Hash artefatto Premier League non valido"
        )

    try:
        model = pickle.loads(
            artifact_path.read_bytes()
        )
    except Exception as exc:
        raise ValueError(
            "Artefatto Premier League non leggibile"
        ) from exc

    if not isinstance(
        model,
        PremierLeagueForwardModel,
    ):
        raise ValueError(
            "Tipo modello Premier League non valido"
        )

    if (
        tuple(
            str(item)
            for item in model.feature_names_in_
        )
        != FEATURES
    ):
        raise ValueError(
            "Schema interno del modello non valido"
        )

    return model, manifest


def freeze_forward_model(
    base_dir: Path | str,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    dataset_path = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
        / "advanced_features_extended.parquet"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {dataset_path}"
        )

    dataset = pd.read_parquet(dataset_path)
    dataset = add_safe_time_features(dataset)

    missing = [
        feature
        for feature in FEATURES
        if feature not in dataset.columns
    ]
    if missing:
        raise ValueError(
            f"Feature mancanti: {missing}"
        )

    training = dataset[
        dataset["season"] != "2026/27"
    ].copy()
    calibration = dataset[
        dataset["season"] == "2026/27"
    ].copy()

    if len(calibration) < (
        CALIBRATION_MINIMUM_HISTORY
    ):
        raise ValueError(
            "Cronologia corrente insufficiente"
        )

    base_model = build_base_model()
    base_model.fit(
        training[list(FEATURES)],
        training[
            "actual_total_corners"
        ].astype(float),
    )

    raw_predictions = np.asarray(
        base_model.predict(
            calibration[list(FEATURES)]
        ),
        dtype=float,
    )
    actual = calibration[
        "actual_total_corners"
    ].astype(float).to_numpy()

    residuals = raw_predictions - actual
    bias_correction = (
        CALIBRATION_SHRINKAGE
        * float(
            np.mean(
                residuals[
                    -CALIBRATION_WINDOW:
                ]
            )
        )
    )

    model = PremierLeagueForwardModel(
        base_model=base_model,
        bias_correction=bias_correction,
    )

    cache_dir = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = (
        cache_dir
        / "premier_league_forward_model.pkl"
    )
    temporary_artifact = artifact_path.with_name(
        artifact_path.name + ".tmp"
    )
    temporary_artifact.write_bytes(
        pickle.dumps(model)
    )
    os.replace(
        temporary_artifact,
        artifact_path,
    )

    cutoff = str(
        calibration["date"].max()
    )

    result = {
        "run_type": (
            "premier_league_forward_model_freeze"
        ),
        "competition": COMPETITION,
        "mode": "forward_validation_only",
        "model_name": MODEL_NAME,
        "feature_names": list(FEATURES),
        "eligible_lines": list(
            ELIGIBLE_LINES
        ),
        "training_rows": int(len(training)),
        "calibration_rows": int(
            len(calibration)
        ),
        "calibration_cutoff_date": cutoff,
        "bias_correction": (
            bias_correction
        ),
        "calibration_window": (
            CALIBRATION_WINDOW
        ),
        "calibration_shrinkage": (
            CALIBRATION_SHRINKAGE
        ),
        "artifact_path": str(
            artifact_path
        ),
        "artifact_hash": _sha256(
            artifact_path
        ),
        "frozen_at": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper_trading_enabled": True,
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    manifest_path = (
        cache_dir
        / "premier_league_forward_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result["manifest_path"] = str(
        manifest_path
    )

    return result
