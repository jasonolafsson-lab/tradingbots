"""
Trade Filters — Applied after signal generation, before order placement.

Three filters:
1. Kill Zone (time-of-day)    — Only trade during high-probability windows
2. Market Regime              — Align direction with daily EMA trend
3. Volume Surge Confirmation  — Require volume spike on signal bar
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import List, Optional, Dict, Tuple

import numpy as np

from data.market_state import (
    Bar, Signal, TickerState, MarketState, Direction,
)

logger = logging.getLogger("trade_filters")


# ============================================================
# Filter 1: Kill Zone (Time-of-Day)
# ============================================================

class KillZoneFilter:
    """Only allow trades during configured high-probability time windows."""

    def __init__(self, config: dict):
        kz = config.get("kill_zones", {})
        self.enabled = kz.get("enabled", False)

        self.windows: List[Tuple[dtime, dtime]] = []
        for window in kz.get("windows", []):
            start = self._parse_time(window["start"])
            end = self._parse_time(window["end"])
            self.windows.append((start, end))

    @staticmethod
    def _parse_time(t: str) -> dtime:
        parts = t.split(":")
        return dtime(int(parts[0]), int(parts[1]))

    def check(self, current_time: dtime) -> bool:
        """
        Return True if trading is ALLOWED at current_time.
        If filter is disabled, always returns True.
        """
        if not self.enabled:
            return True

        for window_start, window_end in self.windows:
            if window_start <= current_time <= window_end:
                return True

        logger.info(
            f"FILTER BLOCKED [KillZone]: time={current_time.strftime('%H:%M')} "
            f"outside allowed windows"
        )
        return False


# ============================================================
# Filter 2: Market Regime (Daily EMA Trend)
# ============================================================

class MarketRegimeFilter:
    """
    Align trade direction with the daily trend of the underlying.
    Uses 9 EMA and 21 EMA on daily closes (derived from 1-min bars).

    - Bull (9 EMA > 21 EMA): only CALL
    - Bear (9 EMA < 21 EMA): only PUT
    - Choppy (EMAs within threshold): reduce size by 50%
    """

    def __init__(self, config: dict):
        rf = config.get("regime_filter", {})
        self.enabled = rf.get("enabled", False)
        self.fast_period = rf.get("fast_ema", 9)
        self.slow_period = rf.get("slow_ema", 21)
        self.choppy_threshold_pct = rf.get("choppy_threshold_pct", 0.003)
        self.choppy_size_reduction = rf.get("choppy_size_reduction", 0.50)

    def classify(self, daily_closes: List[float]) -> str:
        """
        Classify regime from daily closing prices.
        Returns 'BULL', 'BEAR', or 'CHOPPY'.
        Requires at least slow_period daily closes.
        """
        if len(daily_closes) < self.slow_period:
            return "BULL"  # Default: no filter if insufficient data

        fast_ema = self._ema(daily_closes, self.fast_period)
        slow_ema = self._ema(daily_closes, self.slow_period)

        if slow_ema == 0:
            return "BULL"

        diff_pct = (fast_ema - slow_ema) / slow_ema

        if abs(diff_pct) <= self.choppy_threshold_pct:
            return "CHOPPY"
        elif fast_ema > slow_ema:
            return "BULL"
        else:
            return "BEAR"

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        """Compute EMA of the given values, return final EMA value."""
        if not values:
            return 0.0
        multiplier = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = (v - ema) * multiplier + ema
        return ema

    def check(
        self,
        signal: Signal,
        daily_closes: List[float],
    ) -> Tuple[bool, float]:
        """
        Check if signal direction aligns with daily regime.

        Returns:
            (allowed, size_multiplier)
            - allowed=False means trade is blocked
            - size_multiplier=0.5 means reduce size (choppy regime)
            - size_multiplier=1.0 means full size
        """
        if not self.enabled:
            return True, 1.0

        regime = self.classify(daily_closes)

        if regime == "CHOPPY":
            logger.info(
                f"FILTER [RegimeFilter]: {signal.ticker} regime=CHOPPY "
                f"(EMAs within {self.choppy_threshold_pct:.1%}) — "
                f"reducing size by {self.choppy_size_reduction:.0%}"
            )
            return True, self.choppy_size_reduction

        if regime == "BULL" and signal.direction == Direction.PUT:
            logger.info(
                f"FILTER BLOCKED [RegimeFilter]: {signal.ticker} "
                f"PUT signal blocked — daily regime is BULL (9 EMA > 21 EMA)"
            )
            return False, 0.0

        if regime == "BEAR" and signal.direction == Direction.CALL:
            logger.info(
                f"FILTER BLOCKED [RegimeFilter]: {signal.ticker} "
                f"CALL signal blocked — daily regime is BEAR (9 EMA < 21 EMA)"
            )
            return False, 0.0

        return True, 1.0


# ============================================================
# Filter 3: Volume Surge Confirmation
# ============================================================

class VolumeSurgeFilter:
    """
    Require the signal bar to have volume >= multiplier * 20-bar avg volume.
    Uses 1-min bars for the check.
    """

    def __init__(self, config: dict):
        strat_config = config.get("strategies", {})
        self.multiplier = strat_config.get("volume_surge_multiplier", 1.8)
        # If multiplier is 0 or not set, filter is effectively disabled
        self.enabled = self.multiplier > 0
        self.lookback = 20

    def check(self, bars_1m: List[Bar]) -> bool:
        """
        Check if the latest 1-min bar has a volume surge.
        Returns True if trade is ALLOWED (surge confirmed or filter disabled).
        """
        if not self.enabled:
            return True

        if len(bars_1m) < self.lookback + 1:
            # Not enough bars for a meaningful average — allow trade
            return True

        # Latest bar volume
        current_vol = bars_1m[-1].volume

        # 20-bar average (excluding the current bar)
        avg_vol = np.mean([b.volume for b in bars_1m[-(self.lookback + 1):-1]])

        if avg_vol <= 0:
            return True

        ratio = current_vol / avg_vol

        if ratio >= self.multiplier:
            return True

        logger.info(
            f"FILTER BLOCKED [VolumeSurge]: bar volume={current_vol:.0f}, "
            f"20-bar avg={avg_vol:.0f}, ratio={ratio:.2f}x "
            f"(need {self.multiplier:.1f}x)"
        )
        return False


# ============================================================
# Combined Filter Manager
# ============================================================

class TradeFilterManager:
    """
    Applies all three filters in sequence.
    Call apply() after signal generation but before order placement.
    """

    def __init__(self, config: dict):
        self.kill_zone = KillZoneFilter(config)
        self.regime_filter = MarketRegimeFilter(config)
        self.volume_surge = VolumeSurgeFilter(config)

    def apply(
        self,
        signal: Signal,
        current_time: dtime,
        bars_1m: List[Bar],
        daily_closes: List[float],
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Apply all filters to a signal.

        Args:
            signal: The generated trading signal
            current_time: Current time (ET) as time object
            bars_1m: Ticker's 1-minute bars for the current session
            daily_closes: List of daily closing prices (oldest first),
                          used for regime EMA calculation

        Returns:
            (allowed, size_multiplier, block_reason)
            - allowed: True if trade passes all filters
            - size_multiplier: 1.0 normally, 0.5 if choppy regime
            - block_reason: string reason if blocked, None if allowed
        """
        # Filter 1: Kill Zone
        if not self.kill_zone.check(current_time):
            return False, 0.0, "KILL_ZONE"

        # Filter 2: Market Regime
        regime_ok, size_mult = self.regime_filter.check(signal, daily_closes)
        if not regime_ok:
            return False, 0.0, "REGIME_FILTER"

        # Filter 3: Volume Surge
        if not self.volume_surge.check(bars_1m):
            return False, 0.0, "VOLUME_SURGE"

        return True, size_mult, None
