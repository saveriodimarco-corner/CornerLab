from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Any


class BetfairHistoricalAudit:
    def __init__(self) -> None:
        self.market_keywords = ["corners", "total corners", "over/under corners", "match corners", "corner line"]

    def audit_sample_file(self, sample_path: str | Path) -> dict[str, Any]:
        parsed = parse_betfair_stream_file(sample_path)
        market_found = any(self._market_matches(entry.get("marketName", "")) for entry in parsed)
        return {
            "market_theoretically_supported": True,
            "market_found_in_sample": market_found,
            "historical_seasons_available": ["2024/25", "2025/26"],
            "package_and_estimated_purchase_requirement": "Manual catalogue or commercial package required",
            "closing_price_reconstructable": market_found,
            "sample_entries": len(parsed),
        }

    def _market_matches(self, market_name: str) -> bool:
        lowered = market_name.lower()
        return any(keyword in lowered for keyword in self.market_keywords)


def parse_betfair_stream_file(sample_path: str | Path) -> list[dict[str, Any]]:
    path = Path(sample_path)
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    else:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = []
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        selections = data.get("selections", [])
        formatted = {
            "marketName": data.get("marketName"),
            "eventName": data.get("eventName"),
            "kickoffTime": data.get("kickoffTime"),
            "selections": [
                {
                    "name": selection.get("name"),
                    "lastTradedPrice": selection.get("lastTradedPrice"),
                    "bestBack": selection.get("bestBack"),
                    "bestLay": selection.get("bestLay"),
                    "tradedVolume": selection.get("tradedVolume"),
                    "settlementStatus": selection.get("settlementStatus"),
                    "timestamp": selection.get("timestamp"),
                }
                for selection in selections
            ],
        }
        parsed.append(formatted)
    return parsed
