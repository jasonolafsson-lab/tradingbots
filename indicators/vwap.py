"""
VWAP (Volume Weighted Average Price) calculator with standard deviation bands and slope.

Computed from IBKR streaming bars. Resets each trading day.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict

from data.market_state import Bar


class VWAPCalculator:
    """
    Computes VWAP, upper/lower bands (±2 SD), and VWAP slope.
    All computations use cumulative intraday data.
    """

    def calculate(self, bars: List[Bar]) -> Dict[str, float]:
        """
        Calculate VWAP and bands from a list of intraday bars.

        Returns:
            dict with keys: vwap, upper_band, lower_band, slope
        """
        if not bars or len(bars) < 2:
            return {"vwap": 0.0, "upper_band": 0.0, "lower_band": 0.0, "slope": 0.0}

        # Typical price = (H + L + C) / 3
        typical_prices = np.array([(b.high + b.low + b.close) / 3.0 for b in bars])
        volumes = np.array([b.volume for b in bars])

        # Avoid division by zero
        cum_volume = np.cumsum(volumes)
        if cum_volume[-1] == 0:
            return {"vwap": 0.0, "upper_band": 0.0, "lower_band": 0.0, "slope": 0.0}

        cum_vp = np.cumsum(typical_prices * volumes)
        vwap_series = cum_vp / np.maximum(cum_volume, 1e-10)
        vwap = vwap_series[-1]

        # Standard deviation of price from VWAP
        # Using cumulative variance: sum((tp - vwap)^2 * vol) / sum(vol)
        cum_vp2 = np.cumsum((typical_prices ** 2) * volumes)
        variance = cum_vp2 / np.maximum(cum_volume, 1e-10) - vwap_series ** 2
        # Clamp negative variance (floating point)
        variance = np.maximum(variance, 0.0)
        std_dev = np.sqrt(variance[-1])

        upper_band = vwap + 2.0 * std_dev
        lower_band = vwap - 2.0 * std_dev

        # VWAP slope: linear regression of last 3 VWAP values
        slope = self._compute_slope(vwap_series, lookback=3)

        return {
            "vwap": float(vwap),
            "upper_band": float(upper_band),
            "lower_band": float(lower_band),
            "slope": float(slope),
            "std_dev": float(std_dev),
        }

    @staticmethod
    def _compute_slope(series: np.ndarray, lookback: int = 3) -> float:
        """
        Compute slope via linear regression of last N values.
        Returns slope in price-per-bar units.
        """
        if len(series) < lookback:
            lookback = len(series)
        if lookback < 2:
            return 0.0

        y = series[-lookback:]
        x = np.arange(lookback, dtype=float)

        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)

        if var == 0:
            return 0.0

        return float(cov / var)

    @staticmethod
    def price_vs_vwap_sd(price: float, vwap: float, std_dev: float) -> float:
        """
        Return how many standard deviations the price is from VWAP.
        Positive = above VWAP, Negative = below.
        """
        if std_dev <= 0:
            return 0.0
        return (price - vwap) / std_dev
