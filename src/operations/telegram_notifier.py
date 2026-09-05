from __future__ import annotations

import json
import logging
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.research.decision_engine import (
    MIN_CONFIDENCE_SCORE,
    MIN_PREDICTED_PROBABILITY,
    minimum_acceptable_odds,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_TARGETS = {
    "over_8_5", "under_8_5",
    "over_9_5", "under_9_5",
    "over_10_5", "under_10_5",
    "over_11_5", "under_11_5",
}


def is_model_candidate(row: dict[str, Any]) -> bool:
    """Price-independent candidate gate; bet365 odds are evaluated later."""
    try:
        probability = float(row.get("predicted_probability", 0.0) or 0.0)
        confidence = float(
            row.get(
                "confidence_score",
                row.get("model_confidence", row.get("confidence", 0.0)),
            )
            or 0.0
        )
    except (TypeError, ValueError):
        return False

    return (
        probability >= MIN_PREDICTED_PROBABILITY
        and confidence >= MIN_CONFIDENCE_SCORE
    )


def _enabled() -> bool:
	return os.getenv("CORNERLAB_TELEGRAM_ENABLED", "false").strip().lower() == "true"


def send_message(text: str, request_sender: Callable[[str, bytes, float], None] | None = None, reply_markup: dict[str, Any] | None = None) -> bool:
	"""Send an optional Telegram message; notification failure is always non-fatal."""
	token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
	chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
	if not _enabled() or not token or not chat_id:
		return False
	try:
		fields = {"chat_id": chat_id, "text": text}
		if reply_markup is not None:
			fields["reply_markup"] = json.dumps(reply_markup)
		payload = urllib.parse.urlencode(fields).encode("utf-8")
		url = f"https://api.telegram.org/bot{token}/sendMessage"
		if request_sender:
			request_sender(url, payload, 5.0)
		else:
			request = urllib.request.Request(url, data=payload, method="POST")
			with urllib.request.urlopen(request, timeout=5):
				pass
		return True
	except Exception:
		LOGGER.warning("Telegram notification delivery failed")
		return False


def format_critical_failure(job_type: str, timestamp: str, error_summary: str | None, last_success: str | None) -> str:
	return f"🔴 CORNERLAB — ERRORE\n\nJob: {job_type.upper()}\nStato: FAILED\nOra: {timestamp}\nErrore: {error_summary or '-'}\nUltimo run valido: {last_success or '-'}"


def format_recovery(timestamp: str, last_prematch: str | None) -> str:
	return f"🟢 CORNERLAB — RECOVERY\n\nSistema tornato operativo\nPrematch: OK\nUltimo aggiornamento: {last_prematch or timestamp}"


def format_prematch_completed(result: dict[str, Any], timestamp: str) -> str:
	collector = result.get("collector", {})
	paper = result.get("paper_trading", {})
	return f"✅ CORNERLAB — PREMATCH COMPLETATO\n\nFixture analizzate: {int(collector.get('fixtures_fetched', 0))}\nQuote aggiornate: {int(collector.get('odds_writes', 0))}\nPLAY trovati: {int(paper.get('play_count', 0))}\nSistema: OPERATIVO\nOra: {timestamp}"


def format_settlement_completed(summary: dict[str, Any], timestamp: str) -> str:
	return f"📊 CORNERLAB — SETTLEMENT\n\nGiocate chiuse: {int(summary.get('total_bets', 0))}\nWIN: {int(summary.get('wins', 0))}\nLOSS: {int(summary.get('losses', 0))}\nVOID: {int(summary.get('voids', 0))}\nP/L sessione: €{float(summary.get('profit_loss', 0.0)):+.2f}\nROI stagione: {float(summary.get('roi', 0.0)):.1%}\nOra: {timestamp}"



MAX_INDIVIDUAL_PLAY_ALERTS = 5


def format_grouped_play(rows: list[dict[str, Any]]) -> str:
	first = rows[0]
	fixture = f"{first.get('home_team', '-')} vs {first.get('away_team', '-')}"
	kickoff = first.get("kickoff", first.get("kickoff_utc", "-"))
	lines = ["🔎 CORNERLAB — VERIFICA BET365.IT", "", fixture, f"Kickoff: {kickoff}"]

	for row in rows:
		probability = float(row.get("predicted_probability", 0.0) or 0.0)
		minimum_odds = minimum_acceptable_odds(probability)
		minimum_odds_display = math.ceil(minimum_odds * 100.0 - 1e-12) / 100.0

		lines.append("")
		lines.append(f"{str(row.get('side', '')).upper()} {row.get('line', '')}")
		lines.append(f"Probabilità modello: {probability:.1%}")
		lines.append(f"Quota minima bet365.it: {minimum_odds_display:.2f}")
		lines.append(f"Qualità: {row.get('quality_tier', '-')}")

	lines.append("")
	lines.append("Controlla le quote reali su bet365.it.")
	return "\n".join(lines)



def stable_notification_key(row: dict[str, Any]) -> str:
    """Stable Telegram identity: at most one actionable alert per fixture."""
    return f"fixture:{row.get('fixture_id', '')}"


def select_actionable_plays(
    report: pd.DataFrame,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return model-valid Serie A candidates for remaining today + tomorrow Rome time."""
    if report.empty:
        return []

    now_utc = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    rome = ZoneInfo("Europe/Rome")
    today_rome = now_utc.to_pydatetime().astimezone(rome).date()
    tomorrow_rome = today_rome + timedelta(days=1)

    candidates: list[dict[str, Any]] = []

    for _, row in report.iterrows():
        item = row.to_dict()

        if not is_model_candidate(item):
            continue

        if str(item.get("competition", "")) != "Serie A":
            continue

        if str(item.get("target_name", "")) not in SUPPORTED_TARGETS:
            continue

        kickoff = pd.to_datetime(item.get("kickoff_utc"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        # Never alert a match that has already started.
        if kickoff <= now_utc:
            continue

        kickoff_date_rome = kickoff.to_pydatetime().astimezone(rome).date()
        if kickoff_date_rome not in {today_rome, tomorrow_rome}:
            continue

        candidates.append(item)

    def number(item: dict[str, Any], *names: str) -> float:
        for name in names:
            try:
                value = float(item.get(name))
                if pd.notna(value):
                    return value
            except (TypeError, ValueError):
                pass
        return float("-inf")

    # Candidate ranking must be price-independent.
    # Actual bet365.it odds are evaluated only after the user enters them.
    candidates.sort(
        key=lambda item: (
            number(item, "predicted_probability"),
            number(item, "confidence_score", "model_confidence", "confidence"),
            str(item.get("target_name", "")),
        ),
        reverse=True,
    )

    seen: set[tuple[str, str, str]] = set()
    selected: list[dict[str, Any]] = []

    for item in candidates:
        key = (
            str(item.get("fixture_id", "")),
            str(item.get("side", "")).upper(),
            str(item.get("line", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)

    # Telegram order follows kickoff chronology.
    # Multiple model-valid lines for the same fixture must survive until
    # actual bet365.it odds are known.
    selected.sort(
        key=lambda item: pd.to_datetime(
            item.get("kickoff_utc"), utc=True, errors="coerce"
        )
    )
    return selected


def _history_path(base_dir: Path) -> Path:
	return base_dir / "data" / "operations" / "telegram_notifications.jsonl"


def _notified_keys(base_dir: Path) -> set[str]:
	path = _history_path(base_dir)
	if not path.exists():
		return set()
	keys: set[str] = set()
	for line in path.read_text(encoding="utf-8").splitlines():
		try:
			keys.add(str(json.loads(line).get("notification_key", "")))
		except json.JSONDecodeError:
			continue
	return keys


def _record_notification(base_dir: Path, notification_key: str, event_type: str) -> None:
	path = _history_path(base_dir)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps({"notification_key": notification_key, "event_type": event_type, "sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n")


def notify_new_plays(base_dir: Path | str, report: pd.DataFrame, request_sender: Callable[[str, bytes, float], None] | None = None) -> int:
	base_dir = Path(base_dir)
	if report.empty:
		return 0

	notified = _notified_keys(base_dir)

	grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
	order: list[str] = []
	seen_keys: set[str] = set()

	for _, row in report.iterrows():
		row_dict = row.to_dict()

		if not is_model_candidate(row_dict):
			continue

		if str(row_dict.get("target_name", "")) not in SUPPORTED_TARGETS:
			continue

		if str(row_dict.get("competition", "")) != "Serie A":
			continue

		fixture_id = str(row_dict.get("fixture_id", ""))
		if not fixture_id:
			continue

		key = "|".join(
			[
				f"fixture:{fixture_id}",
				str(row_dict.get("market", "")),
				str(row_dict.get("side", "")),
				str(row_dict.get("line", "")),
			]
		)

		if key in notified or key in seen_keys:
			continue

		seen_keys.add(key)

		if fixture_id not in grouped:
			grouped[fixture_id] = []
			order.append(fixture_id)

		grouped[fixture_id].append((key, row_dict))

	if not grouped:
		return 0

	sent = 0

	for fixture_id in order:
		entries = grouped[fixture_id]
		message = format_grouped_play([row_dict for _, row_dict in entries])

		if send_message(message, request_sender=request_sender):
			for key, _ in entries:
				_record_notification(base_dir, key, "MODEL_CANDIDATE")
				sent += 1

	return sent
