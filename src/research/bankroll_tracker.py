from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class BankrollTracker:
    bankroll_start: float
    bankroll: float
    total_staked: float = 0.0
    cumulative_profit: float = 0.0
    max_drawdown: float = 0.0
    peak_bankroll: float = 0.0

    def __post_init__(self) -> None:
        self.bankroll = float(self.bankroll_start)
        self.peak_bankroll = float(self.bankroll_start)

    def update(self, stake: float, outcome: int, odds: float) -> dict[str, float]:
        if stake <= 0.0:
            return {
                "bankroll": self.bankroll,
                "stake": 0.0,
                "cumulative_profit": self.cumulative_profit,
                "roi": self.cumulative_profit / self.bankroll_start if self.bankroll_start else 0.0,
                "yield": self.cumulative_profit / self.total_staked if self.total_staked else 0.0,
                "max_drawdown": self.max_drawdown,
            }

        self.total_staked += float(stake)
        payout = float(stake * (odds - 1.0)) if int(outcome) == 1 else -float(stake)
        self.bankroll += payout
        self.cumulative_profit = self.bankroll - self.bankroll_start
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.max_drawdown = max(self.max_drawdown, (self.peak_bankroll - self.bankroll) / self.peak_bankroll if self.peak_bankroll else 0.0)

        return {
            "bankroll": self.bankroll,
            "stake": float(stake),
            "cumulative_profit": self.cumulative_profit,
            "roi": self.cumulative_profit / self.bankroll_start if self.bankroll_start else 0.0,
            "yield": self.cumulative_profit / self.total_staked if self.total_staked else 0.0,
            "max_drawdown": self.max_drawdown,
        }
