from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.odds.betfair_historical import BetfairHistoricalAudit


def main() -> int:
    audit = BetfairHistoricalAudit()
    sample_path = Path("data/processed/betfair_sample.jsonl")
    if sample_path.exists():
        result = audit.audit_sample_file(sample_path)
        result["sample_available"] = True
    else:
        result = {
            "market_theoretically_supported": True,
            "market_found_in_sample": False,
            "historical_seasons_available": [],
            "package_and_estimated_purchase_requirement": "Manual catalogue or commercial package required",
            "closing_price_reconstructable": False,
            "sample_entries": 0,
            "sample_available": False,
        }
    report_path = Path("reports/betfair_historical_corner_audit.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Betfair historical corner audit",
            "",
            f"- Probe date: {datetime.now(timezone.utc).isoformat()}",
            f"- Market theoretically supported: {'YES' if result['market_theoretically_supported'] else 'NO'}",
            f"- Market found in sample/catalogue: {'YES' if result['market_found_in_sample'] else 'NO'}",
            f"- Historical seasons available: {', '.join(result['historical_seasons_available']) if result['historical_seasons_available'] else 'NONE'}",
            f"- Package and purchase requirement: {result['package_and_estimated_purchase_requirement']}",
            f"- Closing-price reconstructable: {'YES' if result['closing_price_reconstructable'] else 'NO'}",
            f"- Sample available: {'YES' if result['sample_available'] else 'NO'}",
        ]) + "\n",
        encoding="utf-8",
    )
    docs_path = Path("docs/BETFAIR_HISTORICAL_IMPORT.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        "# Betfair historical import notes\n\n"
        "- This audit is metadata-only and does not purchase or ingest data.\n"
        "- Betfair stream-format JSON/BZ2 data can be parsed with the provided audit helper.\n"
        "- Historical closing-price reconstruction requires sample data or a catalogue with timestamped prices.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    main()
