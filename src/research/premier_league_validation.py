from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data.historical_foundation import (
    generate_premier_league_matches,
)
from src.data.normalizer import normalize_team_name
from src.research.advanced_features import (
    build_advanced_feature_dataset,
)
from src.research.feature_selection import (
    run_feature_selection,
)
from src.research.model_benchmark import (
    run_model_benchmark,
)


COMPETITION = "Premier League"
COMPETITION_SLUG = "premier_league"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

COMPLETED_SEASONS = {
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526",
}
CURRENT_SEASON = ("2026/27", "2627")

FORBIDDEN_MODEL_FEATURES = {
    "actual_home_corners",
    "actual_away_corners",
    "actual_total_corners",
    "home_corners",
    "away_corners",
    "total_corners",
    "over_8_5",
    "over_9_5",
    "over_10_5",
    "over_11_5",
}

PROTECTED_RELATIVE_PATHS = (
    Path("data/research/best_models.json"),
    Path("data/research/model_benchmark_results.csv"),
)


def _download_csv(
    season_code: str,
    *,
    timeout_seconds: int = 30,
) -> tuple[bytes, str]:
    url = f"{BASE_URL}/{season_code}/E0.csv"

    with urlopen(url, timeout=timeout_seconds) as response:
        content = response.read()

    if not content:
        raise RuntimeError(
            f"Empty Premier League source: {url}"
        )

    return content, url


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _protected_hashes(
    base_dir: Path,
) -> dict[str, str]:
    paths = [
        base_dir / relative
        for relative in PROTECTED_RELATIVE_PATHS
    ]

    models_dir = base_dir / "models" / "research"
    if models_dir.exists():
        paths.extend(
            path
            for path in models_dir.glob("*.pkl")
            if not path.name.startswith(
                f"{COMPETITION_SLUG}_"
            )
        )

    return {
        str(path.relative_to(base_dir)): _sha256_path(path)
        for path in sorted(set(paths))
        if path.exists()
    }


def _parse_current_season(
    content: bytes,
    source_url: str,
) -> pd.DataFrame:
    text = content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    parsed: list[dict[str, Any]] = []

    season_label, season_code = CURRENT_SEASON

    for raw in rows:
        required = (
            raw.get("Date"),
            raw.get("HomeTeam"),
            raw.get("AwayTeam"),
            raw.get("FTHG"),
            raw.get("FTAG"),
            raw.get("HC"),
            raw.get("AC"),
        )

        if any(
            value is None
            or not str(value).strip()
            for value in required
        ):
            continue

        match_date = pd.to_datetime(
            raw["Date"],
            dayfirst=True,
            errors="coerce",
        )

        if pd.isna(match_date):
            continue

        home_corners = int(float(raw["HC"]))
        away_corners = int(float(raw["AC"]))

        parsed.append(
            {
                "date": match_date.strftime("%Y-%m-%d"),
                "season": season_label,
                "competition": COMPETITION,
                "home_team": normalize_team_name(
                    raw["HomeTeam"]
                ),
                "away_team": normalize_team_name(
                    raw["AwayTeam"]
                ),
                "home_goals": int(float(raw["FTHG"])),
                "away_goals": int(float(raw["FTAG"])),
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": (
                    home_corners + away_corners
                ),
                "source": "football-data-csv",
                "source_file_name": (
                    f"E0_{season_code}.csv"
                ),
                "source_url": source_url,
                "import_date": datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    if not parsed:
        raise RuntimeError(
            "Current Premier League source has no "
            "completed corner results"
        )

    return pd.DataFrame(parsed)


def _stable_row_hash(row: pd.Series) -> str:
    values = [
        row.get("date"),
        row.get("season"),
        row.get("competition"),
        row.get("home_team"),
        row.get("away_team"),
        row.get("home_corners"),
        row.get("away_corners"),
    ]
    payload = "|".join(str(value) for value in values)
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _prepare_matches(
    stage_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    source_dir = (
        stage_dir / "data" / "raw" / "football_data"
    )
    source_dir.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}

    for season_label, season_code in (
        COMPLETED_SEASONS.items()
    ):
        content, url = _download_csv(season_code)
        destination = (
            source_dir / f"E0_{season_code}.csv"
        )
        destination.write_bytes(content)

        frame = pd.read_csv(
            io.BytesIO(content),
            encoding="utf-8-sig",
        )

        completed = frame[
            frame["HC"].notna()
            & frame["AC"].notna()
        ]

        if len(frame) != 380 or len(completed) != 380:
            raise RuntimeError(
                f"Invalid source coverage for "
                f"{season_label}: rows={len(frame)} "
                f"completed={len(completed)}"
            )

        source_hashes[url] = _sha256_bytes(content)

    historical = generate_premier_league_matches(
        stage_dir
    )

    current_label, current_code = CURRENT_SEASON
    current_content, current_url = _download_csv(
        current_code
    )
    current = _parse_current_season(
        current_content,
        current_url,
    )
    source_hashes[current_url] = _sha256_bytes(
        current_content
    )

    matches = pd.concat(
        [historical, current],
        ignore_index=True,
        sort=False,
    )

    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    matches["fixture_id"] = range(
        1,
        len(matches) + 1,
    )
    matches["row_hash"] = matches.apply(
        _stable_row_hash,
        axis=1,
    )

    expected_counts = {
        "2023/24": 380,
        "2024/25": 380,
        "2025/26": 380,
    }
    actual_counts = (
        matches.groupby("season").size().to_dict()
    )

    for season, expected in expected_counts.items():
        if int(actual_counts.get(season, 0)) != expected:
            raise RuntimeError(
                f"Unexpected match count for {season}: "
                f"{actual_counts.get(season, 0)}"
            )

    if int(actual_counts.get(current_label, 0)) <= 0:
        raise RuntimeError(
            "No completed current-season matches"
        )

    return matches, source_hashes


def _selected_features(
    stage_dir: Path,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    for path in sorted(
        (stage_dir / "data" / "research").glob(
            "selected_features_*.json"
        )
    ):
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        result[path.name] = [
            str(item.get("feature"))
            for item in payload.get(
                "selected_features",
                [],
            )
        ]

    return result


def _evaluate_validation(
    dataset: pd.DataFrame,
    benchmark: dict[str, Any],
    selected: dict[str, list[str]],
) -> dict[str, Any]:
    ordered = dataset.sort_values(
        ["season", "date", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    train = ordered[
        ordered["season"].isin(
            ["2023/24", "2024/25"]
        )
    ].copy()
    validation = ordered[
        ordered["season"] == "2025/26"
    ].copy()

    regression = benchmark[
        "regression_best_model"
    ]
    metrics = regression.get("metrics", {})

    predictions = np.asarray(
        metrics.get("probabilities", []),
        dtype=float,
    )

    if len(predictions) != len(validation):
        raise RuntimeError(
            "Validation prediction count mismatch"
        )

    selected_values = {
        feature
        for features in selected.values()
        for feature in features
    }
    leaked_features = sorted(
        selected_values & FORBIDDEN_MODEL_FEATURES
    )

    mae = float(
        regression.get(
            "primary_metric_value",
            math.inf,
        )
    )
    baseline_mae = float(
        regression.get(
            "baseline_metric_value",
            math.inf,
        )
    )

    mae_improvement = (
        (baseline_mae - mae) / baseline_mae
        if math.isfinite(baseline_mae)
        and baseline_mae > 0.0
        else float("-inf")
    )

    observed_totals = validation[
        "actual_total_corners"
    ].to_numpy(dtype=float)

    probability_metrics: dict[str, Any] = {}

    for line in (8.5, 9.5, 10.5, 11.5):
        threshold = int(math.floor(line))
        observed = (
            observed_totals > line
        ).astype(float)

        model_probability = poisson.sf(
            threshold,
            predictions,
        )

        baseline_probability = float(
            (
                train["actual_total_corners"]
                .astype(float)
                > line
            ).mean()
        )

        model_brier = float(
            np.mean(
                (
                    model_probability - observed
                )
                ** 2
            )
        )
        baseline_brier = float(
            np.mean(
                (
                    baseline_probability - observed
                )
                ** 2
            )
        )

        probability_metrics[str(line)] = {
            "model_brier": model_brier,
            "baseline_brier": baseline_brier,
            "improvement": (
                baseline_brier - model_brier
            ),
        }

    mean_model_brier = float(
        np.mean(
            [
                item["model_brier"]
                for item in probability_metrics.values()
            ]
        )
    )
    mean_baseline_brier = float(
        np.mean(
            [
                item["baseline_brier"]
                for item in probability_metrics.values()
            ]
        )
    )

    mean_bias = float(
        metrics.get(
            "mean_prediction_bias",
            math.inf,
        )
    )

    gates = {
        "chronology_ok": bool(
            benchmark.get("chronology_ok")
        ),
        "train_after_validation_false": not bool(
            benchmark.get(
                "train_after_validation",
                True,
            )
        ),
        "validation_rows_380": (
            len(validation) == 380
        ),
        "accepted_non_baseline_model": bool(
            regression.get("accepted")
        )
        and not str(
            regression.get("model_name", "")
        ).endswith("baseline"),
        "mae_better_than_baseline": (
            mae < baseline_mae
        ),
        "mae_improvement_at_least_0_5_percent": (
            mae_improvement >= 0.005
        ),
        "absolute_bias_at_most_0_5": (
            abs(mean_bias) <= 0.5
        ),
        "mean_probability_brier_better": (
            mean_model_brier
            < mean_baseline_brier
        ),
        "no_line_materially_worse": all(
            item["model_brier"]
            <= item["baseline_brier"] + 0.01
            for item in probability_metrics.values()
        ),
        "no_outcome_feature_leakage": (
            not leaked_features
        ),
    }

    return {
        "validation_passed": all(gates.values()),
        "gates": gates,
        "model_name": regression.get("model_name"),
        "validation_rows": len(validation),
        "training_rows": len(train),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement_fraction": (
            mae_improvement
        ),
        "mean_prediction_bias": mean_bias,
        "mean_model_brier": mean_model_brier,
        "mean_baseline_brier": (
            mean_baseline_brier
        ),
        "probability_metrics": (
            probability_metrics
        ),
        "leaked_features": leaked_features,
    }


def _atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_name(
        destination.name + ".tmp"
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _publish_artifacts(
    base_dir: Path,
    stage_dir: Path,
    matches: pd.DataFrame,
    benchmark: dict[str, Any],
) -> list[str]:
    published: list[str] = []

    historical_path = (
        base_dir
        / "data"
        / "raw"
        / "premier_league_matches.csv"
    )
    historical_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_csv = historical_path.with_name(
        historical_path.name + ".tmp"
    )
    matches.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, historical_path)
    published.append(str(historical_path))

    copies = {
        (
            stage_dir
            / "data"
            / "research"
            / "advanced_features.parquet"
        ): (
            base_dir
            / "data"
            / "research"
            / "advanced_features_premier_league.parquet"
        ),
        (
            stage_dir
            / "data"
            / "research"
            / "best_models.json"
        ): (
            base_dir
            / "data"
            / "research"
            / "best_models_premier_league.json"
        ),
        (
            stage_dir
            / "data"
            / "research"
            / "model_benchmark_results.csv"
        ): (
            base_dir
            / "data"
            / "research"
            / "model_benchmark_results_premier_league.csv"
        ),
        (
            stage_dir
            / "reports"
            / "model_benchmark.md"
        ): (
            base_dir
            / "reports"
            / "premier_league_model_benchmark.md"
        ),
    }

    for source, destination in copies.items():
        _atomic_copy(source, destination)
        published.append(str(destination))

    for source in sorted(
        (stage_dir / "data" / "research").glob(
            "selected_features_*.json"
        )
    ):
        suffix = source.name.removeprefix(
            "selected_features_"
        )
        destination = (
            base_dir
            / "data"
            / "research"
            / (
                "selected_features_"
                f"{COMPETITION_SLUG}_{suffix}"
            )
        )
        _atomic_copy(source, destination)
        published.append(str(destination))

    for raw_artifact in benchmark.get(
        "accepted_model_artifacts",
        [],
    ):
        source = Path(raw_artifact)
        destination = (
            base_dir
            / "models"
            / "research"
            / f"{COMPETITION_SLUG}_{source.name}"
        )
        _atomic_copy(source, destination)
        published.append(str(destination))

    return published


def run_premier_league_validation(
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    base_dir = Path(
        base_dir
        or Path(__file__).resolve().parents[2]
    )

    protected_before = _protected_hashes(base_dir)
    started_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    with tempfile.TemporaryDirectory() as name:
        stage_dir = Path(name)
        matches, source_hashes = _prepare_matches(
            stage_dir
        )

        processed_path = (
            stage_dir
            / "data"
            / "processed"
            / "serie_a_matches.parquet"
        )
        processed_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        matches.to_parquet(
            processed_path,
            index=False,
        )

        dataset = build_advanced_feature_dataset(
            base_dir=stage_dir,
            output_dir=stage_dir,
        )

        diagnostic_dataset_path = (
            base_dir
            / "data"
            / "provider_cache"
            / "premier_league_validation"
            / "advanced_features.parquet"
        )
        diagnostic_dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_dataset_path = (
            diagnostic_dataset_path.with_name(
                diagnostic_dataset_path.name
                + ".tmp"
            )
        )
        dataset.to_parquet(
            temporary_dataset_path,
            index=False,
        )
        os.replace(
            temporary_dataset_path,
            diagnostic_dataset_path,
        )

        run_feature_selection(
            base_dir=stage_dir,
            output_dir=stage_dir,
        )

        benchmark = run_model_benchmark(
            base_dir=stage_dir,
            output_dir=stage_dir,
        )

        selected = _selected_features(stage_dir)
        validation = _evaluate_validation(
            dataset,
            benchmark,
            selected,
        )

        regression_diagnostics = [
            {
                "model_name": str(
                    item.get("model_name", "")
                ),
                "mae": float(
                    item.get("mae", float("nan"))
                ),
                "rmse": float(
                    item.get("rmse", float("nan"))
                ),
                "r2": float(
                    item.get("r2", float("nan"))
                ),
                "mean_prediction_bias": float(
                    item.get(
                        "mean_prediction_bias",
                        float("nan"),
                    )
                ),
                "accepted": bool(
                    item.get("accepted", False)
                ),
            }
            for item in benchmark.get(
                "regression_results",
                [],
            )
        ]

        classification_diagnostics = [
            {
                "target_name": str(
                    item.get("target_name", "")
                ),
                "model_name": str(
                    item.get("model_name", "")
                ),
                "brier_score": float(
                    item.get(
                        "brier_score",
                        float("nan"),
                    )
                ),
                "baseline_metric_value": float(
                    item.get(
                        "baseline_metric_value",
                        float("nan"),
                    )
                ),
                "roc_auc": float(
                    item.get(
                        "roc_auc",
                        float("nan"),
                    )
                ),
                "accepted": bool(
                    item.get("accepted", False)
                ),
                "calibration": item.get(
                    "calibration",
                    {},
                ),
            }
            for item in benchmark.get(
                "classification_results",
                [],
            )
        ]

        best_model_diagnostics = {
            target: {
                "accepted": bool(
                    payload.get("accepted", False)
                ),
                "model_name": payload.get(
                    "model_name"
                ),
                "primary_metric_name": payload.get(
                    "primary_metric_name"
                ),
                "primary_metric_value": payload.get(
                    "primary_metric_value"
                ),
                "baseline_metric_value": payload.get(
                    "baseline_metric_value"
                ),
            }
            for target, payload in benchmark.get(
                "best_models",
                {},
            ).items()
        }

        published: list[str] = []
        if validation["validation_passed"]:
            published = _publish_artifacts(
                base_dir,
                stage_dir,
                matches,
                benchmark,
            )

    protected_after = _protected_hashes(base_dir)

    if protected_before != protected_after:
        raise RuntimeError(
            "Serie A protected artifacts changed"
        )

    completed_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = {
        "run_type": (
            "premier_league_model_validation"
        ),
        "competition": COMPETITION,
        "started_at": started_at,
        "completed_at": completed_at,
        "historical_matches": int(len(matches)),
        "diagnostic_dataset_path": str(
            diagnostic_dataset_path
        ),
        "season_counts": {
            str(key): int(value)
            for key, value in (
                matches.groupby("season")
                .size()
                .to_dict()
                .items()
            )
        },
        "source_hashes": source_hashes,
        "serie_a_artifacts_unchanged": True,
        "published_artifacts": published,
        "selected_features": selected,
        "regression_diagnostics": (
            regression_diagnostics
        ),
        "classification_diagnostics": (
            classification_diagnostics
        ),
        "best_model_diagnostics": (
            best_model_diagnostics
        ),
        **validation,
    }

    report_path = (
        base_dir
        / "reports"
        / "premier_league_model_validation.json"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)

    return result
