from __future__ import annotations

from scripts.run_data_factory_once import _apply_result_counters


def test_apply_result_counters_tracks_genuine_inserted() -> None:
    result = {"checked": 1, "downloaded": 8, "inserted": 8, "skipped": False}
    updated = _apply_result_counters(result, 0, 0, 0, 0, 0, 0)

    odds_checked, odds_downloaded, odds_inserted, odds_skipped, odds_retry_skipped, genuine_corner_inserted = updated

    assert odds_checked == 1
    assert odds_downloaded == 8
    assert odds_inserted == 8
    assert odds_skipped == 0
    assert odds_retry_skipped == 0
    assert genuine_corner_inserted == 8
