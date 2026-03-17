"""
Trade Quality Metrics.

Tracks execution quality: slippage, fill time, bid-ask at entry.
Used for post-market analysis and identifying execution issues.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Tracks and reports execution quality metrics."""

    def __init__(self):
        self.fill_times: List[float] = []
        self.slippages: List[float] = []
        self.bid_ask_spreads: List[float] = []

    def record_fill(
        self,
        signal_price: float,
        fill_price: float,
        fill_time_sec: float,
        bid_ask_spread: float,
    ) -> Dict[str, float]:
        """Record execution quality for a single fill."""
        slippage = fill_price - signal_price
        self.fill_times.append(fill_time_sec)
        self.slippages.append(slippage)
        self.bid_ask_spreads.append(bid_ask_spread)

        return {
            "slippage": slippage,
            "fill_time_sec": fill_time_sec,
            "bid_ask_spread": bid_ask_spread,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of execution quality metrics."""
        if not self.fill_times:
            return {"fills": 0}

        return {
            "fills": len(self.fill_times),
            "avg_fill_time": float(np.mean(self.fill_times)),
            "avg_slippage": float(np.mean(self.slippages)),
            "avg_bid_ask": float(np.mean(self.bid_ask_spreads)),
            "max_slippage": float(np.max(np.abs(self.slippages))),
            "max_fill_time": float(np.max(self.fill_times)),
        }

    def reset(self) -> None:
        """Reset metrics for a new session."""
        self.fill_times.clear()
        self.slippages.clear()
        self.bid_ask_spreads.clear()
