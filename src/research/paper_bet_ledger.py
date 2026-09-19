from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _db_path(base_dir: Path | str) -> Path:
    path = Path(base_dir) / "data" / "paper_trading" / "paper_bets.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def suggestion_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("fixture_id", "")),
            str(row.get("market", "")),
            str(row.get("side", "")).upper(),
            str(row.get("line", "")),
        ]
    )


def _connect(base_dir: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(base_dir))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_bets (
            suggestion_key TEXT PRIMARY KEY,
            fixture_id TEXT NOT NULL,
            market TEXT NOT NULL,
            side TEXT NOT NULL,
            line TEXT NOT NULL,
            recommended_stake REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(paper_bets)").fetchall()
    }
    if "payload_json" not in columns:
        conn.execute("ALTER TABLE paper_bets ADD COLUMN payload_json TEXT")
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE paper_bets ADD COLUMN status TEXT NOT NULL DEFAULT 'OPEN'"
        )
    return conn


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def record_bet(base_dir: Path | str, row: dict[str, Any]) -> bool:
    key = suggestion_key(row)
    with _connect(base_dir) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO paper_bets
            (suggestion_key, fixture_id, market, side, line,
             recommended_stake, created_at, payload_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                str(row.get("fixture_id", "")),
                str(row.get("market", "")),
                str(row.get("side", "")).upper(),
                str(row.get("line", "")),
                float(row.get("recommended_stake", row.get("stake", 0.0)) or 0.0),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(row, default=_json_default, sort_keys=True),
                "OPEN",
            ),
        )
        return cursor.rowcount == 1


def list_bets(base_dir: Path | str) -> list[dict[str, Any]]:
    with _connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_bets ORDER BY created_at, suggestion_key"
        ).fetchall()
    return [dict(row) for row in rows]


def contains_bet(base_dir: Path | str, row: dict[str, Any]) -> bool:
    key = suggestion_key(row)
    with _connect(base_dir) as conn:
        found = conn.execute(
            "SELECT 1 FROM paper_bets WHERE suggestion_key = ? LIMIT 1",
            (key,),
        ).fetchone()
    return found is not None


def set_bet_status(
    base_dir: Path | str,
    key: str,
    status: str,
) -> bool:
    normalized_status = str(status).strip().upper()
    if normalized_status not in {"OPEN", "SETTLED"}:
        raise ValueError(f"Unsupported paper bet status: {status!r}")

    with _connect(base_dir) as conn:
        cursor = conn.execute(
            "UPDATE paper_bets SET status = ? WHERE suggestion_key = ?",
            (normalized_status, key),
        )
        return cursor.rowcount == 1


def list_open_bets(base_dir: Path | str) -> list[dict[str, Any]]:
    with _connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM paper_bets
            WHERE status = 'OPEN'
            ORDER BY created_at, suggestion_key
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_open_bet_payloads(base_dir: Path | str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for bet in list_open_bets(base_dir):
        raw = bet.get("payload_json")
        if not raw:
            raise ValueError(
                "Open paper bet is missing immutable payload: "
                f"{bet['suggestion_key']}"
            )

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                "Invalid immutable payload for paper bet: "
                f"{bet['suggestion_key']}"
            )

        payloads.append(payload)

    return payloads
