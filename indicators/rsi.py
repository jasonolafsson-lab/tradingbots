"""
RSI (Relative Strength Index) calculator.

Uses Wilder's smoothing method. Default period = 7 (short-term, per spec).
"""

from __future__ import annotations

import numpy as np
from typing import List


class RSICalculator:
    """Compute RSI from a series of closing prices."""

    def __init__(self, period: int = 7):
        self.period = period

    def calculate(self, closes: List[float]) -> float:
        """
        Calculate RSI from closing prices.

        Args:
            closes: List of closing prices (oldest first).

        Returns:
            RSI value (0-100). Returns 50 if insufficient data.
        """
        if len(closes) < self.period + 1:
            return 50.0

        prices = np.array(closes, dtype=float)
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder's smoothing (EMA-style)
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        for i in range(self.period, len(gains)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return float(np.clip(rsi, 0.0, 100.0))
