from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CONFIG


class Backtest:
    """Evaluate deterministic match predictions using common probabilistic scoring metrics."""

    def __init__(self) -> None:
        """Initialize the backtest evaluator."""
        self._thresholds = list(CONFIG.DEFAULT_THRESHOLDS)

    def evaluate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Return one row per prediction with the result and scoring metrics."""
        if predictions.empty:
            raise ValueError("Predictions must not be empty")

        required_columns = {"actual_total_corners"}
        for threshold in self._thresholds:
            required_columns.add(f"over_{int(threshold)}")
            required_columns.add(f"under_{int(threshold)}")

        missing = required_columns.difference(predictions.columns)
        if missing:
            raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

        scored_rows: List[Dict[str, float]] = []
        for _, row in predictions.iterrows():
            actual_total = float(row["actual_total_corners"])
            row_scores: Dict[str, float] = {"actual_total_corners": actual_total}
            for threshold in self._thresholds:
                over_prob = float(row[f"over_{int(threshold)}"])
                under_prob = float(row[f"under_{int(threshold)}"])
                outcome = int(actual_total > threshold)
                row_scores[f"threshold_{int(threshold)}"] = threshold
                row_scores[f"over_prob_{int(threshold)}"] = over_prob
                row_scores[f"under_prob_{int(threshold)}"] = under_prob
                row_scores[f"outcome_{int(threshold)}"] = outcome
                row_scores[f"brier_{int(threshold)}"] = (over_prob - outcome) ** 2
                row_scores[f"log_loss_{int(threshold)}"] = self._log_loss(over_prob, outcome)
            scored_rows.append(row_scores)

        scored = pd.DataFrame(scored_rows)
        return scored

    def summarize(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate metrics over all predictions."""
        scored = self.evaluate(predictions)
        summary = pd.DataFrame(
            [
                {
                    "accuracy": self._accuracy(scored),
                    "brier_score": self._mean(scored, prefix="brier_"),
                    "log_loss": self._mean(scored, prefix="log_loss_"),
                    "calibration_error": self._calibration_error(scored),
                }
            ]
        )
        return summary

    def _accuracy(self, scored: pd.DataFrame) -> float:
        """Compute mean accuracy across threshold outcomes."""
        columns = [col for col in scored.columns if col.startswith("outcome_")]
        if not columns:
            return 0.0
        return float(scored[columns].mean().mean())

    def _mean(self, scored: pd.DataFrame, prefix: str) -> float:
        """Compute the mean of a set of score columns."""
        columns = [col for col in scored.columns if col.startswith(prefix)]
        if not columns:
            return 0.0
        return float(scored[columns].mean().mean())

    def _calibration_error(self, scored: pd.DataFrame) -> float:
        """Compute the average gap between mean probability and observed frequency."""
        errors: List[float] = []
        for threshold in self._thresholds:
            over_col = f"over_prob_{int(threshold)}"
            outcome_col = f"outcome_{int(threshold)}"
            if over_col in scored.columns and outcome_col in scored.columns:
                errors.append(abs(float(scored[over_col].mean()) - float(scored[outcome_col].mean())))
        return float(np.mean(errors)) if errors else 0.0

    def _log_loss(self, probability: float, outcome: int) -> float:
        """Compute binary log loss from a probability and outcome."""
        epsilon = 1e-15
        probability = min(max(probability, epsilon), 1.0 - epsilon)
        return float(-(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability)))
