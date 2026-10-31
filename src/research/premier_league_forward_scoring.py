from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.research.paper_trading import (
    _load_live_fixtures_and_odds,
    build_live_research_features,
)
from src.research.premier_league_forward_model import (
    COMPETITION,
    ELIGIBLE_LINES,
    FEATURES,
    load_forward_model,
)


RUN_TYPE = "premier_league_forward_scoring"
MODE = "forward_validation_only"
REPORT_JSON_NAME = (
    "premier_league_forward_current.json"
)
REPORT_CSV_NAME = (
    "premier_league_forward_current.csv"
)
LEDGER_NAME = "forward_predictions.jsonl"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _season_label(
    kickoff_utc: Any,
) -> str:
    kickoff = pd.to_datetime(
        kickoff_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(kickoff):
        raise ValueError(
            f"Kickoff non valido: {kickoff_utc}"
        )

    year = int(kickoff.year)

    if int(kickoff.month) >= 7:
        start_year = year
        end_year = year + 1
    else:
        start_year = year - 1
        end_year = year

    return (
        f"{start_year}/"
        f"{str(end_year)[-2:]}"
    )


def _canonical_line(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Linea non valida: {value}"
        ) from exc

    if not math.isfinite(numeric):
        raise ValueError(
            f"Linea non finita: {value}"
        )

    return f"{numeric:.1f}"


def _historical_context(
    historical_matches: pd.DataFrame,
    current_season: str,
) -> dict[str, float]:
    required = {
        "date",
        "season",
        "total_corners",
    }
    missing = sorted(
        required - set(historical_matches.columns)
    )

    if missing:
        raise ValueError(
            f"Colonne storiche mancanti: {missing}"
        )

    frame = historical_matches.copy()
    frame["_date"] = pd.to_datetime(
        frame["date"],
        utc=True,
        errors="coerce",
    )
    frame["_total"] = pd.to_numeric(
        frame["total_corners"],
        errors="coerce",
    )
    frame = frame.loc[
        frame["_date"].notna()
        & frame["_total"].notna()
        & np.isfinite(frame["_total"])
    ].sort_values(
        ["_date"],
        kind="stable",
    )

    if len(frame) < 100:
        raise ValueError(
            "Cronologia Premier insufficiente "
            "per lo scoring"
        )

    values = frame["_total"].astype(float)

    season_values = frame.loc[
        frame["season"].astype(str)
        == str(current_season),
        "_total",
    ].astype(float)

    season_mean = (
        float(season_values.mean())
        if not season_values.empty
        else float(values.tail(100).mean())
    )

    return {
        "league_mean_last_20": float(
            values.tail(20).mean()
        ),
        "league_mean_last_50": float(
            values.tail(50).mean()
        ),
        "league_mean_last_100": float(
            values.tail(100).mean()
        ),
        "season_expanding_mean": season_mean,
    }


def _best_eligible_market_rows(
    odds: pd.DataFrame,
    fixture_ids: set[int],
) -> pd.DataFrame:
    if odds.empty or not fixture_ids:
        return pd.DataFrame()

    frame = odds.copy()
    frame["match_id"] = pd.to_numeric(
        frame["match_id"],
        errors="coerce",
    )
    frame = frame.loc[
        frame["match_id"].notna()
    ].copy()
    frame["match_id"] = frame[
        "match_id"
    ].astype(int)

    frame = frame.loc[
        frame["match_id"].isin(fixture_ids)
    ].copy()

    canonical_lines: list[str | None] = []

    for value in frame["line"]:
        try:
            canonical_lines.append(
                _canonical_line(value)
            )
        except ValueError:
            canonical_lines.append(None)

    frame["line"] = canonical_lines
    frame["side"] = (
        frame["side"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    frame["closing_odds"] = pd.to_numeric(
        frame["closing_odds"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["line"].isin(ELIGIBLE_LINES)
        & frame["side"].isin(
            {"OVER", "UNDER"}
        )
        & frame["closing_odds"].notna()
        & np.isfinite(frame["closing_odds"])
        & (frame["closing_odds"] > 1.0)
    ].copy()

    if frame.empty:
        return frame

    if "odds_timestamp" not in frame.columns:
        frame["odds_timestamp"] = ""

    if "bookmaker" not in frame.columns:
        frame["bookmaker"] = ""

    frame = frame.sort_values(
        [
            "match_id",
            "line",
            "side",
            "closing_odds",
            "odds_timestamp",
            "bookmaker",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    )

    return (
        frame.groupby(
            ["match_id", "line", "side"],
            as_index=False,
            sort=True,
        )
        .head(1)
        .reset_index(drop=True)
    )


def _prediction_id(
    model_hash: str,
    provider_fixture_id: Any,
    line: str,
    side: str,
) -> str:
    payload = "|".join(
        [
            str(model_hash),
            str(provider_fixture_id),
            str(line),
            str(side),
        ]
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _read_prediction_ids(
    ledger_path: Path,
) -> set[str]:
    if not ledger_path.exists():
        return set()

    identifiers: set[str] = set()

    for line_number, raw in enumerate(
        ledger_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ledger Premier non valido alla "
                f"riga {line_number}"
            ) from exc

        identifier = str(
            payload.get("prediction_id") or ""
        )

        if not identifier:
            raise ValueError(
                "Prediction ID mancante nel ledger "
                f"alla riga {line_number}"
            )

        identifiers.add(identifier)

    return identifiers


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        content,
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_reports(
    report_json_path: Path,
    report_csv_path: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    _atomic_write_text(
        report_json_path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
    )

    report_frame = pd.DataFrame(rows)
    report_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_csv = report_csv_path.with_name(
        report_csv_path.name + ".tmp"
    )
    report_frame.to_csv(
        temporary_csv,
        index=False,
    )
    os.replace(
        temporary_csv,
        report_csv_path,
    )


def run_forward_scoring(
    base_dir: Path | str,
    *,
    now: datetime | None = None,
    model_loader: Callable[..., Any] | None = None,
    live_loader: Callable[..., Any] | None = None,
    feature_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    now = now or _utc_now()

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    now = now.astimezone(timezone.utc)

    model_loader = (
        model_loader or load_forward_model
    )
    live_loader = (
        live_loader
        or _load_live_fixtures_and_odds
    )
    feature_builder = (
        feature_builder
        or build_live_research_features
    )

    cache_dir = (
        base_dir
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    matches_path = (
        cache_dir
        / "matches_extended.parquet"
    )
    shadow_dir = (
        base_dir
        / "data"
        / "premier_league_shadow"
    )
    shadow_db = shadow_dir / "collector.sqlite"
    ledger_path = shadow_dir / LEDGER_NAME
    report_json_path = (
        base_dir / "reports" / REPORT_JSON_NAME
    )
    report_csv_path = (
        base_dir / "reports" / REPORT_CSV_NAME
    )

    if not matches_path.exists():
        raise FileNotFoundError(
            f"Storico Premier non trovato: "
            f"{matches_path}"
        )

    if not shadow_db.exists():
        raise FileNotFoundError(
            f"Database shadow non trovato: "
            f"{shadow_db}"
        )

    model, manifest = model_loader(base_dir)

    historical = pd.read_parquet(matches_path)
    historical_dates = pd.to_datetime(
        historical["date"],
        utc=True,
        errors="coerce",
    )

    knowledge_date = now.date()
    historical = historical.loc[
        historical_dates.dt.date
        < knowledge_date
    ].copy()

    fixtures, odds = live_loader(
        base_dir,
        database_path=shadow_db,
    )

    fixtures = fixtures.loc[
        fixtures["competition"]
        .astype(str)
        .str.strip()
        .eq(COMPETITION)
    ].copy()

    fixtures["kickoff_timestamp"] = pd.to_datetime(
        fixtures["kickoff_utc"],
        utc=True,
        errors="coerce",
    )
    fixtures = fixtures.loc[
        fixtures["kickoff_timestamp"].notna()
        & (
            fixtures["kickoff_timestamp"]
            > pd.Timestamp(now)
        )
    ].copy()

    fixtures["season"] = fixtures[
        "kickoff_utc"
    ].map(_season_label)

    future_fixture_ids = set(
        pd.to_numeric(
            fixtures["match_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    market_rows = _best_eligible_market_rows(
        odds,
        future_fixture_ids,
    )

    fixture_ids_with_markets = set(
        market_rows["match_id"].astype(int).tolist()
        if not market_rows.empty
        else []
    )
    score_fixtures = fixtures.loc[
        fixtures["match_id"]
        .astype(int)
        .isin(fixture_ids_with_markets)
    ].copy()

    rows: list[dict[str, Any]] = []

    if not score_fixtures.empty:
        feature_frame, _ = feature_builder(
            historical,
            score_fixtures,
        )

        if feature_frame.empty:
            raise ValueError(
                "Nessuna feature Premier generata"
            )

        seasons = sorted(
            score_fixtures["season"]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(seasons) != 1:
            raise ValueError(
                "Scoring su più stagioni non supportato"
            )

        context = _historical_context(
            historical,
            seasons[0],
        )

        for name, value in context.items():
            feature_frame[name] = value

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in feature_frame.columns
        ]

        if missing_features:
            raise ValueError(
                "Feature Premier mancanti: "
                f"{missing_features}"
            )

        model_input = feature_frame[
            list(FEATURES)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        predictions = np.asarray(
            model.predict(model_input),
            dtype=float,
        ).reshape(-1)

        if len(predictions) != len(feature_frame):
            raise ValueError(
                "Numero previsioni Premier non valido"
            )

        if (
            not np.isfinite(predictions).all()
            or (predictions <= 0.0).any()
        ):
            raise ValueError(
                "Previsioni Premier non valide"
            )

        prediction_lookup = {
            int(match_id): float(prediction)
            for match_id, prediction in zip(
                feature_frame[
                    "match_id"
                ].astype(int),
                predictions,
            )
        }
        feature_lookup = (
            feature_frame
            .drop_duplicates(
                subset=["match_id"],
                keep="first",
            )
            .set_index("match_id")
        )
        fixture_lookup = (
            score_fixtures
            .drop_duplicates(
                subset=["match_id"],
                keep="first",
            )
            .set_index("match_id")
        )

        generated_at = now.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        model_hash = str(
            manifest["artifact_hash"]
        )

        for market in market_rows.to_dict(
            "records"
        ):
            match_id = int(market["match_id"])

            if (
                match_id not in prediction_lookup
                or match_id not in fixture_lookup.index
            ):
                continue

            fixture = fixture_lookup.loc[match_id]
            feature = feature_lookup.loc[match_id]
            expected_total = prediction_lookup[
                match_id
            ]
            line = _canonical_line(
                market["line"]
            )
            line_value = float(line)
            over_probability = float(
                poisson.sf(
                    int(math.floor(line_value)),
                    expected_total,
                )
            )
            side = str(
                market["side"]
            ).upper()
            probability = (
                over_probability
                if side == "OVER"
                else 1.0 - over_probability
            )
            observed_odds = float(
                market["closing_odds"]
            )
            expected_value = (
                probability * observed_odds - 1.0
            )
            provider_fixture_id = fixture.get(
                "provider_fixture_id"
            )
            identifier = _prediction_id(
                model_hash,
                provider_fixture_id,
                line,
                side,
            )

            rows.append(
                {
                    "prediction_id": identifier,
                    "prediction_timestamp": (
                        generated_at
                    ),
                    "mode": MODE,
                    "selection_status": (
                        "OBSERVATION_ONLY"
                    ),
                    "real_betting_enabled": False,
                    "telegram_enabled": False,
                    "bankroll_real_enabled": False,
                    "fixture_id": match_id,
                    "provider_fixture_id": str(
                        provider_fixture_id
                    ),
                    "competition": COMPETITION,
                    "season": str(
                        fixture.get("season")
                    ),
                    "kickoff_utc": str(
                        fixture.get("kickoff_utc")
                    ),
                    "home_team": str(
                        fixture.get("home_team")
                    ),
                    "away_team": str(
                        fixture.get("away_team")
                    ),
                    "line": line,
                    "side": side,
                    "bookmaker": str(
                        market.get("bookmaker") or ""
                    ),
                    "observed_odds": (
                        observed_odds
                    ),
                    "odds_timestamp": str(
                        market.get(
                            "odds_timestamp"
                        )
                        or ""
                    ),
                    "predicted_total_corners": (
                        expected_total
                    ),
                    "predicted_probability": (
                        probability
                    ),
                    "fair_odds": (
                        1.0 / probability
                        if probability > 0.0
                        else None
                    ),
                    "expected_value": expected_value,
                    "positive_edge_observation": (
                        expected_value > 0.0
                    ),
                    "data_quality_score": float(
                        feature.get(
                            "data_quality_score",
                            np.nan,
                        )
                    ),
                    "insufficient_history": bool(
                        feature.get(
                            "insufficient_history",
                            False,
                        )
                    ),
                    "model_name": str(
                        manifest.get(
                            "model_name",
                            "",
                        )
                    ),
                    "model_hash": model_hash,
                    "calibration_cutoff_date": str(
                        manifest.get(
                            "calibration_cutoff_date",
                            "",
                        )
                    ),
                    "result_status": "PENDING",
                    "actual_total_corners": None,
                    "settled_at": None,
                }
            )

    existing_ids = _read_prediction_ids(
        ledger_path
    )
    new_rows = [
        row
        for row in rows
        if row["prediction_id"]
        not in existing_ids
    ]

    if new_rows:
        ledger_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with ledger_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            for row in new_rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    generated_at = now.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = {
        "run_type": RUN_TYPE,
        "mode": MODE,
        "generated_at": generated_at,
        "competition": COMPETITION,
        "model_name": str(
            manifest.get("model_name", "")
        ),
        "model_hash": str(
            manifest["artifact_hash"]
        ),
        "eligible_lines": list(
            ELIGIBLE_LINES
        ),
        "future_fixtures": int(
            len(fixtures)
        ),
        "fixtures_with_eligible_odds": int(
            len(score_fixtures)
        ),
        "observations": int(len(rows)),
        "new_predictions": int(
            len(new_rows)
        ),
        "paper_trading_enabled": True,
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "production_database_writes": False,
        "production_report_writes": False,
        "ledger_path": str(ledger_path),
        "report_json_path": str(
            report_json_path
        ),
        "report_csv_path": str(
            report_csv_path
        ),
        "rows": rows,
    }

    _write_reports(
        report_json_path,
        report_csv_path,
        payload,
        rows,
    )

    return {
        key: payload[key]
        for key in (
            "run_type",
            "mode",
            "generated_at",
            "competition",
            "model_name",
            "model_hash",
            "eligible_lines",
            "future_fixtures",
            "fixtures_with_eligible_odds",
            "observations",
            "new_predictions",
            "paper_trading_enabled",
            "telegram_simulation_enabled",
            "real_betting_enabled",
            "bankroll_real_enabled",
            "production_database_writes",
            "production_report_writes",
            "ledger_path",
            "report_json_path",
            "report_csv_path",
        )
    }
