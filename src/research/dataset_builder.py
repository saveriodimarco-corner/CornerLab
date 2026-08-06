from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.research.feature_selection_engine import FeatureSelectionEngine
from src.research.foundation import CORNER_FEATURE_REGISTRY, FeatureRegistry


class DatasetBuilder:
    def __init__(
        self,
        *,
        feature_selection_engine: Optional[FeatureSelectionEngine] = None,
        feature_registry: Optional[FeatureRegistry] = None,
        base_dir: Optional[Path | str] = None,
    ) -> None:
        self.feature_selection_engine = feature_selection_engine or FeatureSelectionEngine()
        self.feature_registry = feature_registry or CORNER_FEATURE_REGISTRY
        self.base_dir = Path(base_dir) if base_dir is not None else Path("data/research/datasets")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        feature_frame: pd.DataFrame,
        *,
        feature_set_version: str,
        source_snapshot_id: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Path]:
        if feature_frame.empty:
            raise ValueError("feature_frame must not be empty")

        frame = feature_frame.copy()
        if "fixture_id" not in frame.columns:
            raise ValueError("fixture_id is required")
        if "kickoff_utc" not in frame.columns:
            raise ValueError("kickoff_utc is required")
        if "total_corners" not in frame.columns:
            raise ValueError("total_corners is required")

        frame = frame.sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
        if frame["kickoff_utc"].isna().any():
            raise ValueError("kickoff_utc contains invalid values")

        selected_features = self._selected_feature_names(frame)
        if not selected_features:
            if self.feature_registry is not None and len(self.feature_registry.list()) > 0:
                selected_features = [name for name in self.feature_registry.list() if name in frame.columns]
            if not selected_features:
                raise ValueError("No KEEP features available")

        selected_features = [name for name in selected_features if name in frame.columns]

        excluded_rows_by_reason: Dict[str, int] = {
            "missing_feature_value": 0,
            "missing_target": 0,
            "insufficient_lookback": 0,
            "duplicate_fixture_id": 0,
        }

        results: List[Dict[str, Any]] = []
        valid_row_count = 0
        seen_fixture_ids = set()

        for index, row in frame.iterrows():
            fixture_id = int(row["fixture_id"])
            is_duplicate = fixture_id in seen_fixture_ids
            if is_duplicate:
                excluded_rows_by_reason["duplicate_fixture_id"] += 1
            seen_fixture_ids.add(fixture_id)

            row_payload = {
                "fixture_id": int(row["fixture_id"]),
                "kickoff_utc": row["kickoff_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "season": row.get("season"),
                "home_team_id": row.get("home_team_id"),
                "away_team_id": row.get("away_team_id"),
                "total_corners": int(row["total_corners"]),
                "source_snapshot_id": source_snapshot_id,
                "feature_set_version": feature_set_version,
                **{feature_name: row.get(feature_name) for feature_name in selected_features},
                "target_over_8_5": int(int(row["total_corners"]) > 9.5),
                "target_over_9_5": int(int(row["total_corners"]) > 10.5),
                "target_over_10_5": int(int(row["total_corners"]) > 11.5),
                "target_over_11_5": int(int(row["total_corners"]) > 12.5),
            }
            results.append(row_payload)

            if is_duplicate:
                continue

            feature_missing = any(pd.isna(row.get(feature_name)) for feature_name in selected_features if feature_name in frame.columns)
            if feature_missing:
                excluded_rows_by_reason["missing_feature_value"] += 1
                continue

            if pd.isna(row["total_corners"]):
                excluded_rows_by_reason["missing_target"] += 1
                continue

            if not self._passes_lookback(frame, index, selected_features):
                excluded_rows_by_reason["insufficient_lookback"] += 1
                continue

            valid_row_count += 1

        if not results:
            raise ValueError("No valid rows remained after filtering")

        dataset = pd.DataFrame(results).sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)
        dataset["kickoff_utc"] = pd.to_datetime(dataset["kickoff_utc"], utc=True)

        self._validate_leakage(dataset, selected_features)

        content_hash = self._compute_content_hash(dataset, feature_set_version, source_snapshot_id)
        parquet_path = self.base_dir / f"cornerlab_dataset_{feature_set_version}_{content_hash}.parquet"
        manifest_fields = {
            "content_hash": content_hash,
            "dataset_path": str(parquet_path),
            "selected_features": selected_features,
            "targets": ["target_over_8_5", "target_over_9_5", "target_over_10_5", "target_over_11_5"],
            "excluded_rows_by_reason": excluded_rows_by_reason,
            "feature_set_version": feature_set_version,
            "source_snapshot_id": source_snapshot_id,
        }
        manifest = self._build_manifest(dataset, manifest_fields=manifest_fields, row_count=valid_row_count)
        if parquet_path.exists():
            return dataset, manifest, parquet_path

        dataset.to_parquet(parquet_path, index=False)
        manifest_path = parquet_path.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return dataset, manifest, parquet_path

    def _selected_feature_names(self, frame: pd.DataFrame) -> List[str]:
        if self.feature_registry is None:
            return []

        registry_features = self.feature_registry.as_dicts()
        available_names = [feature["name"] for feature in registry_features if feature["name"] in frame.columns]
        if not available_names:
            return []

        evaluation_frame = frame[available_names].copy()
        evaluation_frame["signal_score"] = 1.0
        evaluation_results = self.feature_selection_engine.evaluate(evaluation_frame)
        results_by_name = {result["feature_name"]: result for result in evaluation_results}

        selected: List[str] = []
        for feature_name in available_names:
            result = results_by_name.get(feature_name)
            if result is None:
                continue
            selection_status = result.get("selection_status", "DROP")
            if selection_status == "KEEP":
                selected.append(feature_name)
                continue
            if selection_status != "REVIEW":
                continue

            correlated_feature = result.get("correlated_feature")
            if not correlated_feature:
                selected.append(feature_name)
                continue

            if selected:
                correlation_value = result.get("max_absolute_correlation", 0.0)
                if correlation_value >= 0.99:
                    continue

            selected.append(feature_name)

        if selected:
            return selected

        for feature_name in available_names:
            registry_feature = None
            for item in self.feature_registry.as_dicts():
                if item["name"] == feature_name:
                    registry_feature = item
                    break
            if registry_feature is None:
                continue
            if registry_feature.get("selection_status") == "KEEP":
                selected.append(feature_name)
        return selected

    def _passes_lookback(self, frame: pd.DataFrame, row_index: int, selected_features: Sequence[str]) -> bool:
        for feature_name in selected_features:
            if feature_name not in frame.columns:
                return False
            feature_values = frame.iloc[:row_index + 1][feature_name]
            if feature_values.isna().sum() > 0:
                return False
        return True

    def _validate_leakage(self, dataset: pd.DataFrame, selected_features: Sequence[str]) -> None:
        for feature_name in selected_features:
            registry_feature = self.feature_registry.get(feature_name) if self.feature_registry is not None else None
            if registry_feature is not None and not getattr(registry_feature, "available_before_kickoff", True):
                raise ValueError(f"leakage violation for {feature_name}")

    def _compute_content_hash(self, dataset: pd.DataFrame, feature_set_version: str, source_snapshot_id: str) -> str:
        payload = json.dumps(
            {
                "feature_set_version": feature_set_version,
                "source_snapshot_id": source_snapshot_id,
                "columns": dataset.columns.tolist(),
                "rows": dataset.to_dict(orient="records"),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _build_manifest(
        self,
        dataset: pd.DataFrame,
        *,
        manifest_fields: Dict[str, Any],
        row_count: int,
    ) -> Dict[str, Any]:
        return {
            "dataset_path": manifest_fields["dataset_path"],
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "row_count": int(row_count),
            "column_count": int(len(dataset.columns)),
            "seasons": sorted(str(value) for value in dataset["season"].dropna().unique().tolist()),
            "selected_features": manifest_fields["selected_features"],
            "targets": manifest_fields["targets"],
            "earliest_kickoff": dataset["kickoff_utc"].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latest_kickoff": dataset["kickoff_utc"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_set_version": manifest_fields["feature_set_version"],
            "source_snapshot_id": manifest_fields["source_snapshot_id"],
            "content_hash": manifest_fields["content_hash"],
            "excluded_rows_by_reason": manifest_fields["excluded_rows_by_reason"],
        }
