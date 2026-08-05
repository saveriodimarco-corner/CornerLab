from __future__ import annotations

from pathlib import Path

from src.research.policy_optimization import run_policy_grid_search


def test_run_policy_grid_search_writes_reports(tmp_path: Path) -> None:
    result = run_policy_grid_search(base_dir=Path.cwd(), output_dir=tmp_path)

    assert not result["policies"].empty
    assert {
        "confidence_threshold",
        "ev_threshold",
        "kelly_fraction",
        "bets",
        "win_rate",
        "roi",
        "yield",
        "average_ev",
        "average_confidence",
        "profit",
        "max_drawdown",
        "average_stake",
        "final_bankroll",
        "rank",
    }.issubset(result["policies"].columns)
    assert (tmp_path / "reports" / "policy_grid_search.csv").exists()
    assert (tmp_path / "reports" / "policy_optimization.md").exists()
    assert (tmp_path / "reports" / "policy_heatmap.csv").exists()
    assert result["recommended_policy"]["confidence_threshold"] in {60, 65, 70, 75, 80}
