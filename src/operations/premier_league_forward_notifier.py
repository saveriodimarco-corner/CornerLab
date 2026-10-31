from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.operations.telegram_bot import (
    offer_simulation_confirmation,
)
from src.research.decision_engine import (
    MIN_PREDICTED_PROBABILITY,
    minimum_acceptable_odds,
)
from src.research.premier_league_forward_model import (
    COMPETITION,
    ELIGIBLE_LINES,
)


MAX_FORWARD_ALERTS = 5
ROME = ZoneInfo("Europe/Rome")
REPORT_NAME = (
    "premier_league_forward_current.json"
)
STATE_NAME = (
    "forward_telegram_state.json"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path(
    base_dir: Path,
) -> Path:
    return (
        base_dir
        / "data"
        / "premier_league_shadow"
        / STATE_NAME
    )


def _read_state(
    base_dir: Path,
) -> dict[str, Any]:
    path = _state_path(base_dir)

    if not path.exists():
        return {
            "sent_prediction_ids": [],
            "last_sent_at": None,
        }

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    sent = payload.get(
        "sent_prediction_ids",
        [],
    )

    if not isinstance(sent, list):
        raise ValueError(
            "Stato Telegram Premier non valido"
        )

    return {
        "sent_prediction_ids": [
            str(item)
            for item in sent
        ],
        "last_sent_at": payload.get(
            "last_sent_at"
        ),
    }


def _write_state(
    base_dir: Path,
    state: dict[str, Any],
) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_report(
    base_dir: Path,
) -> dict[str, Any]:
    path = (
        base_dir
        / "reports"
        / REPORT_NAME
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Report Premier non trovato: {path}"
        )

    report = json.loads(
        path.read_text(encoding="utf-8")
    )

    required_contract = {
        "mode": "forward_validation_only",
        "competition": COMPETITION,
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "production_database_writes": False,
    }

    for field, expected in required_contract.items():
        if report.get(field) != expected:
            raise ValueError(
                "Contratto report Premier non valido: "
                f"{field}"
            )

    if (
        report.get("eligible_lines")
        != list(ELIGIBLE_LINES)
    ):
        raise ValueError(
            "Linee Premier del report non valide"
        )

    rows = report.get("rows")

    if not isinstance(rows, list):
        raise ValueError(
            "Osservazioni Premier mancanti"
        )

    return report


def select_forward_price_checks(
    report: dict[str, Any],
    *,
    now: datetime | None = None,
    maximum: int = MAX_FORWARD_ALERTS,
) -> list[dict[str, Any]]:
    now = now or _utc_now()

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    now_utc = pd.Timestamp(
        now.astimezone(timezone.utc)
    )
    today_rome = (
        now_utc.to_pydatetime()
        .astimezone(ROME)
        .date()
    )
    tomorrow_rome = (
        today_rome + timedelta(days=1)
    )

    candidates: list[dict[str, Any]] = []

    for raw in report.get("rows", []):
        row = dict(raw)

        if (
            row.get("selection_status")
            != "OBSERVATION_ONLY"
        ):
            continue

        if (
            str(row.get("competition"))
            != COMPETITION
        ):
            continue

        if (
            str(row.get("line"))
            not in ELIGIBLE_LINES
        ):
            continue

        if bool(
            row.get("insufficient_history")
        ):
            continue

        try:
            quality = float(
                row.get("data_quality_score")
            )
            probability = float(
                row.get(
                    "predicted_probability"
                )
            )
        except (TypeError, ValueError):
            continue

        if quality < 1.0:
            continue

        if (
            probability
            < MIN_PREDICTED_PROBABILITY
        ):
            continue

        kickoff = pd.to_datetime(
            row.get("kickoff_utc"),
            utc=True,
            errors="coerce",
        )

        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            continue

        kickoff_date = (
            kickoff.to_pydatetime()
            .astimezone(ROME)
            .date()
        )

        if kickoff_date not in {
            today_rome,
            tomorrow_rome,
        }:
            continue

        row[
            "minimum_acceptable_odds"
        ] = minimum_acceptable_odds(
            probability
        )
        candidates.append(row)

    candidates.sort(
        key=lambda item: (
            float(
                item[
                    "predicted_probability"
                ]
            ),
            str(item.get("prediction_id")),
        ),
        reverse=True,
    )

    seen: set[
        tuple[str, str, str]
    ] = set()
    selected: list[dict[str, Any]] = []

    for row in candidates:
        key = (
            str(
                row.get(
                    "provider_fixture_id"
                )
            ),
            str(row.get("line")),
            str(row.get("side")).upper(),
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(row)

        if len(selected) >= maximum:
            break

    selected.sort(
        key=lambda item: pd.to_datetime(
            item.get("kickoff_utc"),
            utc=True,
            errors="coerce",
        )
    )
    return selected


def _telegram_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    side = str(
        row.get("side") or ""
    ).upper()
    provider_fixture_id = str(
        row.get("provider_fixture_id") or ""
    )

    return {
        "fixture_id": (
            f"epl:{provider_fixture_id}"
        ),
        "competition": COMPETITION,
        "market": (
            "TOTAL_CORNERS_OVER"
            if side == "OVER"
            else "TOTAL_CORNERS_UNDER"
        ),
        "side": side,
        "line": str(row.get("line")),
        "bookmaker": "bet365.it",
        "decision_timestamp": str(
            row.get("prediction_timestamp")
        ),
        "home_team": str(
            row.get("home_team")
        ),
        "away_team": str(
            row.get("away_team")
        ),
        "recommended_stake": 0.0,
        "odds_at_decision": float(
            row.get("observed_odds")
        ),
        "predicted_probability": float(
            row.get("predicted_probability")
        ),
        "EV": float(
            row.get("expected_value")
        ),
        "quality_tier": (
            "PREMIER FORWARD"
        ),
    }


def notify_forward_price_checks(
    base_dir: Path | str,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    request_sender: (
        Callable[
            [str, bytes, float],
            None,
        ]
        | None
    ) = None,
    confirmation_sender: (
        Callable[..., str] | None
    ) = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    now = now or _utc_now()

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    now = now.astimezone(timezone.utc)
    report = _read_report(base_dir)
    candidates = select_forward_price_checks(
        report,
        now=now,
    )
    state = _read_state(base_dir)
    sent_ids = set(
        state["sent_prediction_ids"]
    )
    pending = [
        row
        for row in candidates
        if str(row["prediction_id"])
        not in sent_ids
    ]

    preview = [
        {
            "prediction_id": str(
                row["prediction_id"]
            ),
            "kickoff_utc": str(
                row["kickoff_utc"]
            ),
            "fixture": (
                f"{row['home_team']} vs "
                f"{row['away_team']}"
            ),
            "line": str(row["line"]),
            "side": str(row["side"]),
            "predicted_probability": float(
                row[
                    "predicted_probability"
                ]
            ),
            "minimum_acceptable_odds": float(
                row[
                    "minimum_acceptable_odds"
                ]
            ),
        }
        for row in pending
    ]

    if dry_run:
        return {
            "mode": "simulation_preview",
            "candidates": len(candidates),
            "pending_notifications": (
                len(pending)
            ),
            "sent_notifications": 0,
            "telegram_simulation_only": True,
            "real_betting_enabled": False,
            "bankroll_real_enabled": False,
            "preview": preview,
        }

    confirmation_sender = (
        confirmation_sender
        or offer_simulation_confirmation
    )
    sent_now: list[str] = []

    for row in pending:
        bet_id = confirmation_sender(
            base_dir,
            _telegram_row(row),
            request_sender=request_sender,
        )

        if bet_id:
            sent_now.append(
                str(row["prediction_id"])
            )

    if sent_now:
        merged = list(
            dict.fromkeys(
                state[
                    "sent_prediction_ids"
                ]
                + sent_now
            )
        )
        state = {
            "sent_prediction_ids": merged,
            "last_sent_at": now.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        _write_state(base_dir, state)

    return {
        "mode": "telegram_simulation",
        "candidates": len(candidates),
        "pending_notifications": (
            len(pending)
        ),
        "sent_notifications": (
            len(sent_now)
        ),
        "telegram_simulation_only": True,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
        "preview": preview,
    }
