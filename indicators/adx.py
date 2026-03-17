"""
ADX (Average Directional Index) calculator.

Measures trend strength regardless of direction.
ADX > 25 = trending, ADX < 20 = range-bound. Default period = 14.
"""

from __future__ import annotations

import numpy as np
from typing import List


class ADXCalculator:
    """Compute ADX from high, low, close series."""

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
    ) -> float:
        """
        Calculate ADX from price series.

        Args:
            highs: List of high prices (oldest first).
            lows: List of low prices.
            closes: List of closing prices.

        Returns:
            ADX value (0-100). Returns 0 if insufficient data.
        """
        n = len(closes)
        if n < self.period + 1:
            return 0.0

        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        c = np.array(closes, dtype=float)

        # True Range
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Directional Movement
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's smoothing
        atr = self._wilder_smooth(tr, self.period)
        plus_di_raw = self._wilder_smooth(plus_dm, self.period)
        minus_di_raw = self._wilder_smooth(minus_dm, self.period)

        if atr is None or plus_di_raw is None or minus_di_raw is None:
            return 0.0

        # +DI and -DI
        plus_di = 100.0 * plus_di_raw / np.maximum(atr, 1e-10)
        minus_di = 100.0 * minus_di_raw / np.maximum(atr, 1e-10)

        # DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100.0 * di_diff / np.maximum(di_sum, 1e-10)

        # ADX = Wilder's smooth of DX
        if len(dx) < self.period:
            return float(np.mean(dx)) if len(dx) > 0 else 0.0

        adx = np.mean(dx[:self.period])
        for i in range(self.period, len(dx)):
            adx = (adx * (self.period - 1) + dx[i]) / self.period

        return float(np.clip(adx, 0.0, 100.0))

    @staticmethod
    def _wilder_smooth(data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing to a series."""
        if len(data) < period:
            return None

        result = np.zeros(len(data) - period + 1)
        result[0] = np.mean(data[:period])

        for i in range(1, len(result)):
            result[i] = (result[i - 1] * (period - 1) + data[period - 1 + i]) / period

        return result
