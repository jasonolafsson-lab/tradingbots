"""
Volume Profile calculator.

Computes volume ratio: current bar volume vs N-bar rolling average.
Used for confirming breakouts and signal strength.
"""

from __future__ import annotations

import numpy as np
from typing import List


class VolumeProfileCalculator:
    """Compute volume ratio and related metrics."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate(self, volumes: List[float]) -> float:
        """
        Calculate the ratio of the latest bar's volume to the rolling average.

        Args:
            volumes: List of volume values (oldest first).

        Returns:
            Ratio (e.g., 1.5 means current is 50% above average).
            Returns 1.0 if insufficient data.
        """
        if not volumes:
            return 1.0

        if len(volumes) <= self.lookback:
            avg = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        else:
            avg = np.mean(volumes[-self.lookback - 1:-1])

        if avg <= 0:
            return 1.0

        return float(volumes[-1] / avg)

    def is_above_average(
        self,
        volumes: List[float],
        threshold: float = 1.0,
    ) -> bool:
        """Check if latest volume is above the threshold ratio."""
        return self.calculate(volumes) > threshold

    @staticmethod
    def prior_day_vs_20d_avg(
        prior_day_volume: float,
        avg_20d_volume: float,
    ) -> float:
        """
        Compute ratio of prior day volume to 20-day average.
        Used in pre-market scanner for unusual volume detection.
        """
        if avg_20d_volume <= 0:
            return 1.0
        return prior_day_volume / avg_20d_volume
