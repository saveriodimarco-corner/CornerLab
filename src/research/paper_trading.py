from __future__ import annotations

import sqlite3
import hashlib
import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data.odds_validator import validate_odds_dataframe
from src.engine.feature_store import FeatureStore
from src.research.advanced_features import build_advanced_feature_dataset
from src.research.confidence_engine import DEFAULT_POLICY, apply_abstention_rules, build_confidence_features, compute_confidence_components
from src.research.decision_engine import build_decision_report


LINE_TO_PROBABILITY_COLUMN = {
    "8.5": "predicted_probability_over_8_5",
    "9.5": "predicted_probability_over_9_5",
    "10.5": "predicted_probability_over_10_5",
    "11.5": "predicted_probability_over_11_5",
}

SUPPORTED_DECISIONS = {"PLAY", "NO BET", "LOW CONFIDENCE"}

MARKET_LINE_TO_TARGET_NAME = {
    "8.5": "over_8_5",
    "9.5": "over_9_5",
    "10.5": "over_10_5",
    "11.5": "over_11_5",
}

VALIDATED_MODEL_TARGETS = {
    "over_9_5": "negative_binomial_probability",
    "over_10_5": "negative_binomial_probability",
}

SYNTHETIC_OUTCOME_COLUMNS = {
    "total_corners",
    "over85",
    "over95",
    "over105",
    "over115",
    "under85",
    "under95",
    "under105",
    "under115",
}


def run_paper_trading(base_dir: str | Path | None = None, output_dir: str | Path | None = None, bankroll: float = 100.0) -> dict[str, Any]:
    base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    output_dir = Path(output_dir or base_dir)

    historical_matches = _load_historical_matches(base_dir)
    fixtures, odds = _load_live_fixtures_and_odds(base_dir)
    validated_frames: list[pd.DataFrame] = []
    validation_errors: list[str] = []
    for match_id, odds_group in odds.groupby("match_id", sort=False):
        fixture_subset = fixtures.loc[fixtures["match_id"].astype(int) == int(match_id)]
        validated_group, group_errors = validate_odds_dataframe(odds_group.copy(), fixtures=fixture_subset)
        validation_errors.extend(group_errors)
        if not validated_group.empty:
            validated_frames.append(validated_group)

    validated_odds = pd.concat(validated_frames, ignore_index=True) if validated_frames else pd.DataFrame()
    if validated_odds.empty:
        raise ValueError("No validated live odds were available for paper trading")

    feature_frame, confidence_frame = build_live_research_features(historical_matches, fixtures)
    model_bundle = _load_authoritative_models(base_dir)
    scored_confidence_frame = _build_confidence_frame(feature_frame, confidence_frame)
    odds_input = _build_odds_input(feature_frame, scored_confidence_frame, validated_odds, model_bundle)

    if odds_input.empty:
        raise ValueError("No paper-trading rows could be built from the live odds")

    scored_rows = odds_input.loc[odds_input["scoring_status"] == "SCORED"].copy()
    unavailable_rows = odds_input.loc[odds_input["scoring_status"] != "SCORED"].copy()

    decision_input = scored_rows[["match_id", "market", "closing_odds", "predicted_probability", "model_confidence"]].copy()
    scored_report = build_decision_report(decision_input, bankroll=bankroll)
    scored_report = scored_report.rename(columns={"confidence_score": "decision_confidence_score"})
    traceability_columns = [
        column
        for column in scored_rows.columns
        if column not in {"match_id", "market", "closing_odds", "predicted_probability", "model_confidence"} and column not in scored_report.columns
    ]
    scored_report = pd.concat([scored_report.reset_index(drop=True), scored_rows[traceability_columns].reset_index(drop=True)], axis=1)

    if not scored_report.empty:
        scored_report["market_implied_probability"] = np.where(
            scored_report["closing_odds"].notna() & (scored_report["closing_odds"] > 0.0),
            1.0 / scored_report["closing_odds"],
            np.nan,
        )
        scored_report["edge"] = scored_report["predicted_probability"] - scored_report["market_implied_probability"]
        scored_report["stake"] = scored_report.get("recommended_stake", np.nan)
        scored_report["decision_reason"] = np.where(
            scored_report["decision"] == "LOW CONFIDENCE",
            "CONFIDENCE_BELOW_THRESHOLD",
            np.where(scored_report["decision"] == "PLAY", "POSITIVE_EV", "NON_POSITIVE_EV"),
        )

    report = pd.concat([scored_report, unavailable_rows], ignore_index=True, sort=False)
    report = report.sort_values("row_order", kind="stable").reset_index(drop=True)

    run_id = f"prematch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    report["run_id"] = run_id

    data_dir = output_dir / "data" / "paper_trading"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "paper_trades_current.parquet"
    csv_path = reports_dir / "paper_trading_current.csv"
    summary_path = reports_dir / "paper_trading_summary.md"
    run_parquet_path = runs_dir / f"{run_id}.parquet"
    run_csv_path = runs_dir / f"{run_id}.csv"

    report.to_parquet(parquet_path, index=False)
    report.to_csv(csv_path, index=False)
    report.to_parquet(run_parquet_path, index=False)
    report.to_csv(run_csv_path, index=False)
    summary_path.write_text(build_summary_markdown(report, validation_errors), encoding="utf-8")

    supported_market_rows = int((report["market_support_status"] == "SUPPORTED").sum()) if "market_support_status" in report.columns else 0
    unsupported_market_rows = int((report["market_support_status"] == "UNSUPPORTED").sum()) if "market_support_status" in report.columns else 0
    model_input_failed_rows = int((report["scoring_status"] == "FAILED").sum()) if "scoring_status" in report.columns else 0
    model_scored_rows = int(report["decision"].isin(SUPPORTED_DECISIONS).sum()) if "decision" in report.columns else 0
    over_under_warnings = _validate_over_under_complement(report)
    scored_mask = report["decision"].isin(SUPPORTED_DECISIONS)

    history_entry = {
        "run_id": run_id,
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fixtures_evaluated": int(report["fixture_id"].nunique()) if "fixture_id" in report.columns else 0,
        "odds_rows": int(len(report)),
        "play_count": int((report["decision"] == "PLAY").sum()),
        "no_bet_count": int((report["decision"] == "NO BET").sum()),
        "low_confidence_count": int((report["decision"] == "LOW CONFIDENCE").sum()),
        "average_ev": float(report.loc[scored_mask, "ev"].mean()) if report.loc[scored_mask, "ev"].notna().any() else None,
        "supported_models": sorted(report.loc[report["market_support_status"] == "SUPPORTED", "target_name"].dropna().astype(str).unique().tolist()),
        "warnings": list(validation_errors) + over_under_warnings,
        "report_csv": str(csv_path),
        "report_parquet": str(parquet_path),
        "run_csv": str(run_csv_path),
        "run_parquet": str(run_parquet_path),
    }
    _append_run_history(data_dir / "run_history.jsonl", history_entry)

    return {
        "report": report,
        "summary": {
            "run_id": run_id,
            "total_odds_rows": int(len(validated_odds)),
            "supported_market_rows": supported_market_rows,
            "unsupported_market_rows": unsupported_market_rows,
            "model_scored_rows": model_scored_rows,
            "model_input_failed_rows": model_input_failed_rows,
            "play_count": int((report["decision"] == "PLAY").sum()),
            "low_confidence_count": int((report["decision"] == "LOW CONFIDENCE").sum()),
            "no_bet_count": int((report["decision"] == "NO BET").sum()),
            "average_ev": float(report.loc[scored_mask, "ev"].mean()) if "ev" in report.columns and report.loc[scored_mask, "ev"].notna().any() else float("nan"),
            "average_confidence": float(report.loc[scored_mask, "decision_confidence_score"].mean()) if "decision_confidence_score" in report.columns and report.loc[scored_mask, "decision_confidence_score"].notna().any() else float("nan"),
            "fixtures": int(report["fixture_id"].nunique()) if "fixture_id" in report.columns else 0,
            "over_under_engine_ok": len(over_under_warnings) == 0,
            "warnings": over_under_warnings,
        },
        "output_paths": {
            "parquet": parquet_path,
            "csv": csv_path,
            "summary": summary_path,
            "run_parquet": run_parquet_path,
            "run_csv": run_csv_path,
            "history": data_dir / "run_history.jsonl",
        },
        "validation_errors": validation_errors,
    }


def _load_authoritative_models(base_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = base_dir / "data" / "research" / "best_models.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Best-model manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundles: dict[str, dict[str, Any]] = {}
    for target_name in MARKET_LINE_TO_TARGET_NAME.values():
        info = manifest.get(target_name)
        if not info or not bool(info.get("accepted")) or str(info.get("model_name", "")).endswith("baseline"):
            continue
        model_name = str(info.get("model_name"))
        artifact_path = base_dir / "models" / "research" / f"{target_name}_{model_name}.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Validated trained artifact is missing: {artifact_path}")
        blob = artifact_path.read_bytes()
        model = pickle.loads(blob)
        bundles[target_name] = {
            "artifact_path": artifact_path,
            "artifact_hash": hashlib.sha256(blob).hexdigest(),
            "model": model,
            "schema": _extract_model_schema(model),
            "model_version": info.get("model_name"),
                "target_name": target_name,
        }
    return bundles


def build_live_research_features(historical_matches: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    live_rows: list[pd.DataFrame] = []
    for _, fixture in fixtures.sort_values(["kickoff_utc", "fixture_id"]).iterrows():
        combined_matches = historical_matches.copy()
        live_match = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(fixture["kickoff_utc"], utc=True, errors="coerce").strftime("%Y-%m-%d"),
                    "season": str(fixture.get("season", "")),
                    "competition": str(fixture.get("competition", "Serie A")),
                    "home_team": fixture.get("home_team"),
                    "away_team": fixture.get("away_team"),
                    "home_goals": 0,
                    "away_goals": 0,
                    "home_corners": 0,
                    "away_corners": 0,
                    "total_corners": 0,
                    "source": "paper_trading_live_fixture",
                    "source_file_name": "paper_trading_live_fixture",
                    "source_url": "paper_trading_live_fixture",
                    "import_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "fixture_id": int(fixture["fixture_id"]),
                    "row_hash": f"live-{int(fixture['fixture_id'])}",
                }
            ]
        )
        combined_matches = pd.concat([combined_matches, live_match], ignore_index=True)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            processed_dir = temp_dir / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            combined_path = processed_dir / "serie_a_matches.parquet"
            combined_matches.to_parquet(combined_path, index=False)
            advanced_features = build_advanced_feature_dataset(base_dir=temp_dir, output_dir=temp_dir)

        live_feature_row = advanced_features.loc[advanced_features["match_id"].astype(int) == int(fixture["fixture_id"])]
        if live_feature_row.empty:
            continue
        live_rows.append(live_feature_row.iloc[[0]].copy())

    if not live_rows:
        return pd.DataFrame(), pd.DataFrame()

    feature_frame = pd.concat(live_rows, ignore_index=True)
    confidence_frame = feature_frame.copy()
    return feature_frame, confidence_frame


def _extract_model_schema(model: Any) -> list[str]:
    if hasattr(model, "poisson_model") and hasattr(model.poisson_model, "feature_names_in_"):
        return list(model.poisson_model.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return []


def build_live_fixture_features(historical_matches: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_store = FeatureStore()
    prior_state: dict[str, dict[str, float]] = {}
    team_history: dict[str, dict[str, list[float]]] = {}
    league_state: dict[str, dict[str, list[float]]] = {}

    history = historical_matches.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values(["date", "season", "fixture_id"]).reset_index(drop=True)
    for _, match in history.iterrows():
        feature_store._create_feature_row(  # type: ignore[attr-defined]
            match,
            pd.DataFrame(columns=["home_team", "away_team", "home_corners", "away_corners"]),
            prior_state=prior_state,
            team_history=team_history,
            league_state=league_state,
            season=str(match.get("season", "")),
        )
        feature_store._update_state(  # type: ignore[attr-defined]
            prior_state,
            team_history,
            match,
            league_state=league_state,
            season=str(match.get("season", "")),
        )

    feature_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    ordered_fixtures = fixtures.copy()
    ordered_fixtures["kickoff_utc"] = pd.to_datetime(ordered_fixtures["kickoff_utc"], utc=True, errors="coerce")
    ordered_fixtures = ordered_fixtures.sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)

    for _, fixture in ordered_fixtures.iterrows():
        kickoff = pd.to_datetime(fixture["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        synthetic_match = pd.Series(
            {
                "fixture_id": int(fixture["fixture_id"]),
                "date": kickoff.strftime("%Y-%m-%d"),
                "season": str(fixture.get("season", "")),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "home_corners": 0.0,
                "away_corners": 0.0,
            }
        )

        feature_row = feature_store._create_feature_row(  # type: ignore[attr-defined]
            synthetic_match,
            pd.DataFrame(columns=["home_team", "away_team", "home_corners", "away_corners"]),
            prior_state=prior_state,
            team_history=team_history,
            league_state=league_state,
            season=str(fixture.get("season", "")),
        )

        home_team = str(fixture.get("home_team", ""))
        away_team = str(fixture.get("away_team", ""))
        home_history = team_history.get(home_team, {})
        away_history = team_history.get(away_team, {})
        home_matches_played = int(len(home_history.get("for_values", [])))
        away_matches_played = int(len(away_history.get("for_values", [])))
        home_std = float(feature_store._std_from_history(home_history, n=5, kind="for"))  # type: ignore[attr-defined]
        away_std = float(feature_store._std_from_history(away_history, n=5, kind="for"))  # type: ignore[attr-defined]

        row = {
            "fixture_id": int(fixture["fixture_id"]),
            "provider_fixture_id": fixture.get("provider_fixture_id"),
            "kickoff_utc": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": str(fixture.get("season", "")),
            "date": kickoff.strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "match_id": int(fixture["fixture_id"]),
            **feature_row,
            "data_quality_score": float(min(1.0, ((home_matches_played + away_matches_played) / 2.0) / 10.0)),
            "insufficient_history": bool(home_matches_played < 5 or away_matches_played < 5),
            "home_matches_played": float(home_matches_played),
            "away_matches_played": float(away_matches_played),
            "combined_volatility": float(0.5 * (home_std + away_std)),
        }
        row.update({key: value for key, value in feature_row.items() if key not in SYNTHETIC_OUTCOME_COLUMNS})
        feature_rows.append(row)
        confidence_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"fixture_id", "provider_fixture_id", "kickoff_utc", "season", "date", "home_team", "away_team", "match_id"}
            }
        )

    feature_frame = pd.DataFrame(feature_rows)
    confidence_frame = pd.DataFrame(confidence_rows)
    return feature_frame, confidence_frame


def _build_confidence_frame(feature_frame: pd.DataFrame, confidence_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        return feature_frame

    probability_frame = pd.DataFrame(index=feature_frame.index)
    for line, column in LINE_TO_PROBABILITY_COLUMN.items():
        probability_frame[column] = 1.0 - poisson.cdf(int(float(line)), feature_frame["expected_total_corner"].astype(float))
    probability_frame["match_id"] = feature_frame["match_id"].astype(int).to_numpy()
    probability_frame["predicted_total_corners"] = feature_frame["expected_total_corner"].astype(float).to_numpy()

    pre_match_columns = [
        column
        for column in confidence_frame.columns
        if column not in {"data_quality_score", "insufficient_history", "home_matches_played", "away_matches_played", "combined_volatility"}
        and pd.api.types.is_numeric_dtype(confidence_frame[column])
    ]
    if not pre_match_columns:
        pre_match_columns = [column for column in confidence_frame.select_dtypes(include="number").columns if column != "match_id"]

    scored = pd.concat([probability_frame, build_confidence_features(probability_frame, confidence_frame, pre_match_columns)], axis=1)
    scored = compute_confidence_components(scored)
    scored = apply_abstention_rules(scored, policy=DEFAULT_POLICY)
    scored["model_confidence"] = scored["confidence_score"].astype(float) / 100.0
    scored["decision_state"] = scored["decision_state"].astype(str)
    return scored


def _build_odds_input(feature_frame: pd.DataFrame, confidence_frame: pd.DataFrame, validated_odds: pd.DataFrame, model_bundle: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixture_lookup = feature_frame.set_index("match_id")

    for row_order, (_, odds_row) in enumerate(validated_odds.iterrows()):
        match_id = int(odds_row["match_id"])
        if match_id not in fixture_lookup.index:
            continue

        fixture_row = fixture_lookup.loc[match_id]
        confidence_rows = confidence_frame.loc[confidence_frame["match_id"].astype(int) == match_id]
        if confidence_rows.empty:
            continue
        confidence_row = confidence_rows.iloc[0]

        line = str(odds_row["line"]).strip()
        if line not in LINE_TO_PROBABILITY_COLUMN:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="UNSUPPORTED_MARKET", row_order=row_order, target_name=None))
            continue

        target_name = _target_name_for_market_line(str(odds_row["market"]), line)
        if target_name is None:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="UNSUPPORTED_MARKET", row_order=row_order, target_name=None))
            continue

        if target_name not in model_bundle:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="NO_ACCEPTED_MODEL", row_order=row_order, target_name=target_name))
            continue

        bundle = model_bundle[target_name]
        live_feature_frame = pd.DataFrame([feature_row_to_model_input(fixture_row, target_name)])
        expected_features = bundle["schema"]
        if not expected_features:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="MODEL_INPUT_FAILED", row_order=row_order, target_name=target_name))
            continue

        schema_match, aligned_frame = _align_feature_schema(live_feature_frame, expected_features)
        if not schema_match:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="MODEL_INPUT_FAILED", row_order=row_order, target_name=target_name))
            continue

        try:
            over_probability = float(np.asarray(_predict_with_loaded_model(bundle["model"], aligned_frame)).reshape(-1)[0])
        except Exception:
            rows.append(_build_unavailable_row(fixture_row, odds_row, confidence_row, reason="MODEL_INPUT_FAILED", row_order=row_order, target_name=target_name))
            continue

        model_probability = _resolve_market_probability(market=str(odds_row.get("market", "")), side=str(odds_row.get("side", "")), over_probability=over_probability)

        rows.append(
            {
                "row_order": row_order,
                "paper_trade_row_id": f"{match_id}:{odds_row['bookmaker']}:{odds_row['market']}:{line}:{odds_row['side']}",
                "match_id": match_id,
                "fixture_id": match_id,
                "provider_fixture_id": fixture_row.get("provider_fixture_id"),
                "kickoff_utc": fixture_row.get("kickoff_utc"),
                "season": fixture_row.get("season"),
                "date": fixture_row.get("date"),
                "home_team": fixture_row.get("home_team"),
                "away_team": fixture_row.get("away_team"),
                "bookmaker": odds_row.get("bookmaker"),
                "market": odds_row.get("market"),
                "line": line,
                "side": odds_row.get("side"),
                "target_name": target_name,
                "market_support_status": "SUPPORTED",
                "scoring_status": "SCORED",
                "source": odds_row.get("source"),
                "source_fixture_id": odds_row.get("source_fixture_id"),
                "provider_event_id": odds_row.get("source_fixture_id"),
                "raw_response_hash": odds_row.get("raw_response_hash"),
                "snapshot_timestamp": odds_row.get("odds_timestamp"),
                "minutes_to_kickoff": odds_row.get("minutes_to_kickoff"),
                "opening_odds": float(odds_row.get("opening_odds", float("nan"))),
                "closing_odds": float(odds_row.get("closing_odds", float("nan"))),
                "predicted_probability": model_probability,
                "model_confidence": float(confidence_row.get("model_confidence", 0.0)),
                "confidence_score": float(confidence_row.get("confidence_score", 0.0)),
                "decision_state": confidence_row.get("decision_state"),
                "predicted_total_corners": float(confidence_row.get("predicted_total_corners", np.nan)),
                "data_quality_score": float(confidence_row.get("data_quality_score", np.nan)),
                "insufficient_history": bool(confidence_row.get("insufficient_history", False)),
                "home_matches_played": float(confidence_row.get("home_matches_played", np.nan)),
                "away_matches_played": float(confidence_row.get("away_matches_played", np.nan)),
                "combined_volatility": float(confidence_row.get("combined_volatility", np.nan)),
                "model_artifact": str(bundle["artifact_path"]),
                "model_hash": str(bundle["artifact_hash"]),
                "feature_schema_hash": hashlib.sha256("|".join(expected_features).encode("utf-8")).hexdigest(),
                "feature_vector_version": target_name,
                "prediction_timestamp": odds_row.get("odds_timestamp"),
                "model_version": bundle["model_version"],
                "decision": None,
                "decision_reason": None,
                "ev": np.nan,
                "kelly_fraction": np.nan,
                "stake": np.nan,
                "fair_odds": np.nan,
            }
        )

    return pd.DataFrame(rows)


def _target_name_for_market_line(market: str, line: str) -> str | None:
    market = str(market).upper()
    line = str(line).strip()
    if market not in {"TOTAL_CORNERS_OVER", "TOTAL_CORNERS_UNDER"}:
        return None
    return MARKET_LINE_TO_TARGET_NAME.get(line)


def _resolve_market_probability(market: str, side: str, over_probability: float) -> float:
    market = str(market).upper()
    side = str(side).upper()
    over_probability = float(np.clip(over_probability, 0.0, 1.0))
    if market == "TOTAL_CORNERS_UNDER" or side == "UNDER":
        return float(np.clip(1.0 - over_probability, 0.0, 1.0))
    return over_probability


def _align_feature_schema(frame: pd.DataFrame, expected_features: list[str]) -> tuple[bool, pd.DataFrame]:
    if not expected_features:
        return False, frame
    missing = [feature for feature in expected_features if feature not in frame.columns]
    if missing:
        return False, frame
    aligned = frame[expected_features].copy()
    return True, aligned


def _predict_with_loaded_model(model: Any, aligned_frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        result = np.asarray(model.predict_proba(aligned_frame)[:, 1], dtype=float)
    elif hasattr(model, "predict"):
        result = np.asarray(model.predict(aligned_frame), dtype=float)
    else:
        raise ValueError("Loaded model cannot produce predictions")
    return np.clip(result, 0.0, 1.0)


def feature_row_to_model_input(feature_row: pd.Series, target_name: str) -> dict[str, Any]:
    target_name = str(target_name)
    model_input = feature_row.to_dict()
    if target_name == "over_9_5":
        return model_input
    if target_name == "over_10_5":
        return model_input
    return model_input


def _build_unavailable_row(
    fixture_row: pd.Series,
    odds_row: pd.Series,
    confidence_row: pd.Series,
    reason: str,
    row_order: int,
    target_name: str | None,
) -> dict[str, Any]:
    market_support_status = "SUPPORTED" if reason == "MODEL_INPUT_FAILED" else "UNSUPPORTED"
    scoring_status = "FAILED" if reason == "MODEL_INPUT_FAILED" else "NOT_APPLICABLE"
    return {
        "row_order": row_order,
        "paper_trade_row_id": f"{int(odds_row['match_id'])}:{odds_row['bookmaker']}:{odds_row['market']}:{str(odds_row['line']).strip()}:{odds_row['side']}",
        "match_id": int(odds_row["match_id"]),
        "fixture_id": int(odds_row["match_id"]),
        "provider_fixture_id": fixture_row.get("provider_fixture_id"),
        "kickoff_utc": fixture_row.get("kickoff_utc"),
        "season": fixture_row.get("season"),
        "date": fixture_row.get("date"),
        "home_team": fixture_row.get("home_team"),
        "away_team": fixture_row.get("away_team"),
        "bookmaker": odds_row.get("bookmaker"),
        "market": odds_row.get("market"),
        "line": str(odds_row.get("line", "")).strip(),
        "side": odds_row.get("side"),
        "target_name": target_name,
        "market_support_status": market_support_status,
        "scoring_status": scoring_status,
        "source": odds_row.get("source"),
        "source_fixture_id": odds_row.get("source_fixture_id"),
        "provider_event_id": odds_row.get("source_fixture_id"),
        "raw_response_hash": odds_row.get("raw_response_hash"),
        "snapshot_timestamp": odds_row.get("odds_timestamp"),
        "minutes_to_kickoff": odds_row.get("minutes_to_kickoff"),
        "opening_odds": float(odds_row.get("opening_odds", float("nan"))),
        "closing_odds": float(odds_row.get("closing_odds", float("nan"))),
        "predicted_probability": float("nan"),
        "model_confidence": float(confidence_row.get("model_confidence", 0.0)),
        "confidence_score": float(confidence_row.get("confidence_score", 0.0)),
        "decision_state": confidence_row.get("decision_state"),
        "predicted_total_corners": float(confidence_row.get("predicted_total_corners", np.nan)),
        "data_quality_score": float(confidence_row.get("data_quality_score", np.nan)),
        "insufficient_history": bool(confidence_row.get("insufficient_history", False)),
        "home_matches_played": float(confidence_row.get("home_matches_played", np.nan)),
        "away_matches_played": float(confidence_row.get("away_matches_played", np.nan)),
        "combined_volatility": float(confidence_row.get("combined_volatility", np.nan)),
        "model_artifact": "MODEL_UNAVAILABLE",
        "model_hash": "MODEL_UNAVAILABLE",
        "feature_schema_hash": "MODEL_UNAVAILABLE",
        "feature_vector_version": "MODEL_UNAVAILABLE",
        "prediction_timestamp": odds_row.get("odds_timestamp"),
        "model_version": "MODEL_UNAVAILABLE",
        "decision": "MODEL_UNAVAILABLE",
        "decision_reason": reason,
        "ev": np.nan,
        "kelly_fraction": np.nan,
        "stake": np.nan,
        "fair_odds": np.nan,
        "decision_confidence_score": np.nan,
    }


def _append_run_history(history_path: Path, entry: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _validate_over_under_complement(report: pd.DataFrame, tolerance: float = 1e-9) -> list[str]:
    warnings: list[str] = []
    if report.empty:
        return warnings

    scored = report.loc[report["decision"].isin(SUPPORTED_DECISIONS)].copy()
    if scored.empty:
        return warnings

    grouped = scored.groupby(["fixture_id", "bookmaker", "line"], dropna=False)
    for group_key, group in grouped:
        over_rows = group.loc[group["side"].astype(str).str.upper() == "OVER"]
        under_rows = group.loc[group["side"].astype(str).str.upper() == "UNDER"]
        if over_rows.empty or under_rows.empty:
            continue
        over_probability = float(over_rows.iloc[0]["predicted_probability"])
        under_probability = float(under_rows.iloc[0]["predicted_probability"])
        if not np.isfinite(over_probability) or not np.isfinite(under_probability):
            continue
        if abs((over_probability + under_probability) - 1.0) > tolerance:
            warnings.append(f"OVER_UNDER_COMPLEMENT_MISMATCH fixture={group_key[0]} bookmaker={group_key[1]} line={group_key[2]}")
    return warnings


def _load_historical_matches(base_dir: Path) -> pd.DataFrame:
    matches_path = base_dir / "data" / "raw" / "serie_a_matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Historical match source not found: {matches_path}")
    matches = pd.read_csv(matches_path)
    if matches.empty:
        raise ValueError("Historical match source is empty")
    return matches


def _load_live_fixtures_and_odds(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    database_path = base_dir / "data" / "collector.sqlite"
    if not database_path.exists():
        raise FileNotFoundError(f"Collector database not found: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        fixtures = pd.read_sql_query(
            """
            SELECT fixture_id, provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status, provider
            FROM collector_fixtures
            ORDER BY kickoff_utc ASC, fixture_id ASC
            """,
            connection,
        )
        odds = pd.read_sql_query(
            """
            SELECT
                o.fixture_id,
                f.provider_fixture_id,
                f.competition,
                f.season,
                f.kickoff_utc,
                f.home_team,
                f.away_team,
                o.bookmaker,
                o.market,
                o.line,
                o.side,
                o.decimal_odds,
                o.snapshot_timestamp,
                o.minutes_to_kickoff,
                o.provider,
                o.provider_event_id,
                o.raw_response_hash,
                o.import_timestamp
            FROM collector_odds_snapshots AS o
            INNER JOIN collector_fixtures AS f ON f.fixture_id = o.fixture_id
            WHERE o.provider = 'the-odds-api'
            ORDER BY o.fixture_id ASC, o.market ASC, o.line ASC, o.side ASC, o.bookmaker ASC
            """,
            connection,
        )

    fixtures = fixtures.copy()
    fixtures["match_id"] = fixtures["fixture_id"].astype(int)
    fixtures["date"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    odds = odds.copy()
    odds["match_id"] = odds["fixture_id"].astype(int)
    odds["fixture_date"] = pd.to_datetime(odds["kickoff_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    odds["opening_odds"] = pd.to_numeric(odds["decimal_odds"], errors="coerce")
    odds["closing_odds"] = pd.to_numeric(odds["decimal_odds"], errors="coerce")
    odds["odds_timestamp"] = odds["snapshot_timestamp"]
    odds["source"] = odds["provider"]
    odds["source_fixture_id"] = odds["provider_event_id"]
    odds["is_closing"] = True
    odds["currency"] = "EUR"
    odds = odds.sort_values(["fixture_id", "bookmaker", "market", "line", "side", "snapshot_timestamp", "import_timestamp"]).drop_duplicates(
        subset=["fixture_id", "bookmaker", "market", "line", "side", "snapshot_timestamp"],
        keep="last",
    )

    odds = odds[[
        "match_id",
        "fixture_date",
        "home_team",
        "away_team",
        "bookmaker",
        "market",
        "line",
        "side",
        "opening_odds",
        "closing_odds",
        "odds_timestamp",
        "source",
        "source_fixture_id",
        "is_closing",
        "currency",
        "import_timestamp",
    ]].copy()

    return fixtures, odds


def build_summary_markdown(report: pd.DataFrame, validation_errors: list[str]) -> str:
    scored_mask = report["decision"].isin(SUPPORTED_DECISIONS) if "decision" in report.columns else pd.Series(dtype=bool)
    supported_market_rows = int((report["market_support_status"] == "SUPPORTED").sum()) if "market_support_status" in report.columns else 0
    unsupported_market_rows = int((report["market_support_status"] == "UNSUPPORTED").sum()) if "market_support_status" in report.columns else 0
    lines = [
        "# Paper Trading Summary",
        "",
        "This report scores the current live corners odds against fixture-level pre-match probabilities derived from historical Serie A corner history.",
        "",
        f"- Total rows: {len(report)}",
        f"- Supported market rows: {supported_market_rows}",
        f"- Unsupported market rows: {unsupported_market_rows}",
        f"- Model scored rows: {int(scored_mask.sum())}",
        f"- Fixtures covered: {report['fixture_id'].nunique() if 'fixture_id' in report.columns else 0}",
        f"- PLAY decisions: {(report['decision'] == 'PLAY').sum()}",
        f"- LOW CONFIDENCE decisions: {(report['decision'] == 'LOW CONFIDENCE').sum()}",
        f"- NO BET decisions: {(report['decision'] == 'NO BET').sum()}",
        f"- MODEL_UNAVAILABLE decisions: {(report['decision'] == 'MODEL_UNAVAILABLE').sum()}",
        f"- Average EV: {report.loc[scored_mask, 'ev'].mean():.3f}" if report.loc[scored_mask, 'ev'].notna().any() else "- Average EV: nan",
        f"- Average decision confidence: {report.loc[scored_mask, 'decision_confidence_score'].mean():.1f}" if report.loc[scored_mask, 'decision_confidence_score'].notna().any() else "- Average decision confidence: nan",
        "",
        "## Validation",
    ]
    if validation_errors:
        lines.extend(f"- {error}" for error in validation_errors)
    else:
        lines.append("- Live odds validation passed without errors.")
    return "\n".join(lines) + "\n"
