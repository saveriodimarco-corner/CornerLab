from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.providers.football_data_results import (
    FootballDataResultsProvider,
)
from src.operations.automation import (
    _corner_bet_result,
)
from src.operations.real_bet_ledger import (
    list_open_bets,
    settle_real_bet,
)


COMPETITION = "Premier League"
SETTLEMENT_DELAY = timedelta(
    hours=2,
    minutes=30,
)
LEDGER_NAME = "forward_predictions.jsonl"
REPORT_NAME = (
    "premier_league_forward_"
    "settlement_latest.json"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_ledger(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    for line_number, raw in enumerate(
        path.read_text(
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
                "Ledger Premier non valido "
                f"alla riga {line_number}"
            ) from exc

        if not payload.get("prediction_id"):
            raise ValueError(
                "Prediction ID mancante "
                f"alla riga {line_number}"
            )

        rows.append(payload)

    return rows


def _atomic_write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _result_from_row(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        total = int(
            row["actual_total_corners"]
        )
        home = int(row["home_corners"])
        away = int(row["away_corners"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "home_score": row.get(
            "home_score"
        ),
        "away_score": row.get(
            "away_score"
        ),
        "home_corners": home,
        "away_corners": away,
        "total_corners": total,
        "provider": row.get(
            "result_provider",
            "football-data-csv",
        ),
    }


def _settle_prediction(
    row: dict[str, Any],
    result: dict[str, Any],
    settled_at: str,
) -> None:
    total = int(result["total_corners"])
    bet_result = _corner_bet_result(
        total,
        row.get("side"),
        row.get("line"),
    )

    if bet_result is None:
        raise ValueError(
            "Mercato forward non supportato"
        )

    if bet_result == "WIN":
        status = "SETTLED_WIN"
        outcome = 1.0
        unit_profit_loss = (
            float(row["observed_odds"])
            - 1.0
        )
    elif bet_result == "LOSS":
        status = "SETTLED_LOSS"
        outcome = 0.0
        unit_profit_loss = -1.0
    else:
        status = "SETTLED_VOID"
        outcome = None
        unit_profit_loss = 0.0

    probability = float(
        row["predicted_probability"]
    )
    brier_score = (
        float(
            (
                probability - outcome
            )
            ** 2
        )
        if outcome is not None
        else None
    )

    row.update(
        {
            "result_status": status,
            "bet_result": bet_result,
            "actual_total_corners": total,
            "home_score": result.get(
                "home_score"
            ),
            "away_score": result.get(
                "away_score"
            ),
            "home_corners": int(
                result["home_corners"]
            ),
            "away_corners": int(
                result["away_corners"]
            ),
            "result_provider": str(
                result.get(
                    "provider",
                    "football-data-csv",
                )
            ),
            "settled_at": settled_at,
            "brier_score": brier_score,
            "unit_profit_loss": (
                unit_profit_loss
            ),
        }
    )


def run_forward_settlement(
    base_dir: Path | str,
    *,
    provider: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    now = now or _utc_now()

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    now = now.astimezone(timezone.utc)
    settled_at = now.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    shadow_dir = (
        base_dir
        / "data"
        / "premier_league_shadow"
    )
    ledger_path = (
        shadow_dir / LEDGER_NAME
    )
    report_path = (
        base_dir
        / "reports"
        / REPORT_NAME
    )
    simulation_base = (
        base_dir
        / "data"
        / "telegram_simulation"
    )

    rows = _read_ledger(ledger_path)
    provider = (
        provider
        or FootballDataResultsProvider(
            base_dir / "data"
        )
    )

    result_by_fixture: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:
        provider_fixture_id = str(
            row.get(
                "provider_fixture_id"
            )
            or ""
        )
        existing = _result_from_row(row)

        if provider_fixture_id and existing:
            result_by_fixture[
                provider_fixture_id
            ] = existing

    pending_groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    not_due = 0

    for row in rows:
        if (
            row.get("result_status")
            != "PENDING"
        ):
            continue

        kickoff = pd.to_datetime(
            row.get("kickoff_utc"),
            utc=True,
            errors="coerce",
        )

        if pd.isna(kickoff):
            continue

        if (
            kickoff.to_pydatetime()
            + SETTLEMENT_DELAY
            > now
        ):
            not_due += 1
            continue

        provider_fixture_id = str(
            row.get(
                "provider_fixture_id"
            )
            or ""
        )

        if provider_fixture_id:
            pending_groups.setdefault(
                provider_fixture_id,
                [],
            ).append(row)

    resolved_fixtures = 0
    unresolved_fixtures: list[
        dict[str, Any]
    ] = []
    settled_predictions = 0

    for (
        provider_fixture_id,
        fixture_rows,
    ) in pending_groups.items():
        result = result_by_fixture.get(
            provider_fixture_id
        )

        if result is None:
            reference = fixture_rows[0]
            resolution = (
                provider.resolve_fixture(
                    {
                        "competition": (
                            COMPETITION
                        ),
                        "kickoff_utc": (
                            reference.get(
                                "kickoff_utc"
                            )
                        ),
                        "home_team": (
                            reference.get(
                                "home_team"
                            )
                        ),
                        "away_team": (
                            reference.get(
                                "away_team"
                            )
                        ),
                    }
                )
            )

            if (
                resolution.get("ok")
                is not True
            ):
                unresolved_fixtures.append(
                    {
                        "provider_fixture_id": (
                            provider_fixture_id
                        ),
                        "fixture": (
                            f"{reference.get('home_team')} "
                            f"vs "
                            f"{reference.get('away_team')}"
                        ),
                        "reason": resolution.get(
                            "reason"
                        ),
                    }
                )
                continue

            result = dict(
                resolution["result"]
            )
            result_by_fixture[
                provider_fixture_id
            ] = result
            resolved_fixtures += 1

        for row in fixture_rows:
            _settle_prediction(
                row,
                result,
                settled_at,
            )
            settled_predictions += 1

    if settled_predictions:
        _atomic_write_jsonl(
            ledger_path,
            rows,
        )

    simulation_settled: list[
        dict[str, Any]
    ] = []
    simulation_failed: list[
        dict[str, Any]
    ] = []
    simulation_pending = 0

    for bet in list_open_bets(
        simulation_base
    ):
        if (
            str(bet.get("competition"))
            != COMPETITION
        ):
            continue

        fixture_id = str(
            bet.get("fixture_id") or ""
        )
        provider_fixture_id = (
            fixture_id[4:]
            if fixture_id.startswith("epl:")
            else ""
        )
        result = result_by_fixture.get(
            provider_fixture_id
        )

        if result is None:
            simulation_pending += 1
            continue

        bet_result = _corner_bet_result(
            result["total_corners"],
            bet.get("side"),
            bet.get("line"),
        )

        if bet_result is None:
            simulation_failed.append(
                {
                    "suggestion_id": bet.get(
                        "suggestion_id"
                    ),
                    "reason": (
                        "unsupported_market"
                    ),
                }
            )
            continue

        settlement = settle_real_bet(
            simulation_base,
            str(bet["suggestion_id"]),
            bet_result,
        )
        item = {
            "suggestion_id": bet[
                "suggestion_id"
            ],
            "provider_fixture_id": (
                provider_fixture_id
            ),
            "total_corners": result[
                "total_corners"
            ],
            "bet_result": bet_result,
            "settlement": settlement,
        }

        if settlement.get("ok") is True:
            simulation_settled.append(item)
        else:
            simulation_failed.append(item)

    settled_rows = [
        row
        for row in rows
        if str(
            row.get("result_status")
        ).startswith("SETTLED_")
    ]
    brier_values = [
        float(row["brier_score"])
        for row in settled_rows
        if row.get("brier_score")
        is not None
        and np.isfinite(
            float(row["brier_score"])
        )
    ]
    unit_profit_values = [
        float(row["unit_profit_loss"])
        for row in settled_rows
        if row.get("unit_profit_loss")
        is not None
        and np.isfinite(
            float(
                row[
                    "unit_profit_loss"
                ]
            )
        )
    ]

    report = {
        "run_type": (
            "premier_league_forward_settlement"
        ),
        "mode": "forward_validation_only",
        "competition": COMPETITION,
        "completed_at": settled_at,
        "ledger_rows": len(rows),
        "settled_predictions_now": (
            settled_predictions
        ),
        "settled_predictions_total": (
            len(settled_rows)
        ),
        "pending_not_due": not_due,
        "resolved_fixtures_now": (
            resolved_fixtures
        ),
        "unresolved_fixtures": (
            unresolved_fixtures
        ),
        "mean_brier_score": (
            float(np.mean(brier_values))
            if brier_values
            else None
        ),
        "unit_profit_loss_total": (
            float(
                np.sum(
                    unit_profit_values
                )
            )
            if unit_profit_values
            else 0.0
        ),
        "simulation_settled_now": len(
            simulation_settled
        ),
        "simulation_pending": (
            simulation_pending
        ),
        "simulation_failed": (
            simulation_failed
        ),
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "production_database_writes": (
            False
        ),
        "ledger_path": str(ledger_path),
        "report_path": str(report_path),
    }

    _atomic_write_json(
        report_path,
        report,
    )
    return report
