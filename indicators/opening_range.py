"""
Opening Range tracker.

Records the high and low of the first 15 minutes (9:30-9:45 AM ET).
Used by the Momentum Breakout strategy.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

from data.market_state import TickerState

logger = logging.getLogger(__name__)


class OpeningRangeTracker:
    """
    Tracks the opening range (first N minutes of the session).
    Sets opening_range_high and opening_range_low on each TickerState.
    """

    def __init__(self, config: dict):
        or_config = config.get("opening_range", {})
        self.duration_minutes = or_config.get("duration_minutes", 15)
        # OR is 9:30 to 9:45 by default
        self.or_end_hour = 9
        self.or_end_minute = 30 + self.duration_minutes

    def update(self, ts: TickerState) -> None:
        """
        Update opening range from the latest bars.
        Call this on each new bar during the opening range period.
        """
        if ts.opening_range_set:
            return  # Already locked in

        if not ts.bars_1m:
            return

        # Check if we have enough bars to set the opening range
        # We need bars from 9:30 to 9:45 (15 minutes of 1-min bars)
        or_bars = []
        for bar in ts.bars_1m:
            bar_time = bar.timestamp.time() if hasattr(bar.timestamp, 'time') else None
            if bar_time is None:
                continue
            # Bars within 9:30-9:45
            if (bar_time >= dtime(9, 30) and
                    bar_time < dtime(self.or_end_hour, self.or_end_minute)):
                or_bars.append(bar)

        if len(or_bars) >= self.duration_minutes:
            # We have enough data to set the range
            ts.opening_range_high = max(b.high for b in or_bars)
            ts.opening_range_low = min(b.low for b in or_bars)
            ts.opening_range_set = True
            logger.info(
                f"{ts.ticker} Opening Range set: "
                f"H={ts.opening_range_high:.2f} L={ts.opening_range_low:.2f}"
            )
        elif or_bars:
            # Partial range — update running high/low
            ts.opening_range_high = max(b.high for b in or_bars)
            ts.opening_range_low = min(b.low for b in or_bars)
