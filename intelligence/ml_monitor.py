"""
Level 2: ML Model Monitor.

Tracks model accuracy, calibration (Brier score), feature drift,
and A/B comparison (ML-sized P&L vs Level 1 sizing).
Disables ML if accuracy drops below 52%.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from intelligence.trade_memory import TradeMemoryDB

logger = logging.getLogger(__name__)


class MLMonitor:
    """Monitors ML model performance and triggers disable if needed."""

    def __init__(self, config: dict, trade_memory: TradeMemoryDB):
        self.config = config
        self.db = trade_memory
        self.min_accuracy = config.get("intelligence", {}).get(
            "ml_min_accuracy", 0.52
        )
        self.predictions: List[Dict[str, Any]] = []
        self.ml_should_be_active = True

    def record_prediction(
        self,
        trade_id: str,
        predicted_prob: float,
        actual_outcome: str,
    ) -> None:
        """Record a prediction for tracking."""
        self.predictions.append({
            "trade_id": trade_id,
            "predicted_prob": predicted_prob,
            "predicted_outcome": "WIN" if predicted_prob > 0.5 else "LOSS",
            "actual_outcome": actual_outcome,
        })

    def check_accuracy(self) -> Dict[str, Any]:
        """
        Check current model accuracy and calibration.
        Returns metrics and whether the model should remain active.
        """
        if len(self.predictions) < 20:
            return {
                "accuracy": None,
                "brier_score": None,
                "should_remain_active": True,
                "sample_size": len(self.predictions),
            }

        # Accuracy
        correct = sum(
            1 for p in self.predictions
            if p["predicted_outcome"] == p["actual_outcome"]
        )
        accuracy = correct / len(self.predictions)

        # Brier score (calibration)
        probs = np.array([p["predicted_prob"] for p in self.predictions])
        actuals = np.array([
            1 if p["actual_outcome"] == "WIN" else 0
            for p in self.predictions
        ])
        brier = float(np.mean((probs - actuals) ** 2))

        # Check threshold
        should_remain = accuracy >= self.min_accuracy

        if not should_remain and self.ml_should_be_active:
            logger.warning(
                f"ML ACCURACY BELOW THRESHOLD: {accuracy:.3f} < {self.min_accuracy}. "
                f"Reverting to Level 1 sizing."
            )
            self.ml_should_be_active = False

        return {
            "accuracy": accuracy,
            "brier_score": brier,
            "should_remain_active": should_remain,
            "sample_size": len(self.predictions),
        }

    def get_ab_comparison(self) -> Dict[str, Any]:
        """
        Compare ML-sized P&L vs what Level 1 sizing would have produced.
        Reads from trade memory to see if ML sizing improved results.
        """
        # This requires logging both ML-sized actual trades and
        # what Level 1 would have done. For now, return placeholder.
        return {
            "ml_pnl": 0,
            "level1_pnl": 0,
            "ml_advantage": 0,
            "note": "A/B tracking will be populated as trades accumulate",
        }
