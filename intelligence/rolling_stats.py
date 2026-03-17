"""
Level 1: Rolling Performance Statistics.

Computes rolling metrics per strategy/ticker/time-bucket/day-of-week.
Used by the auto_adjuster to scale position sizes based on what's working.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from intelligence.trade_memory import TradeMemoryDB

logger = logging.getLogger(__name__)


class RollingStats:
    """Computes rolling performance statistics from trade memory."""

    def __init__(self, trade_memory: TradeMemoryDB, window: int = 50):
        self.db = trade_memory
        self.window = window
        self.stats_cache: Dict[str, Dict[str, Any]] = {}

    def refresh(self) -> None:
        """Refresh all rolling statistics. Call daily after market close."""
        self.stats_cache = {}

        # Per strategy
        for strategy in ["MOMENTUM", "REVERSION", "DAY2",
                         "TUESDAY_REVERSAL", "GREEN_SECTOR"]:
            key = f"strategy:{strategy}"
            self.stats_cache[key] = self._compute_stats(strategy=strategy)

        # Per ticker
        for ticker in ["SPY", "QQQ", "NVDA", "TSLA"]:
            key = f"ticker:{ticker}"
            self.stats_cache[key] = self._compute_stats(ticker=ticker)

        # Per strategy/ticker combo
        for strategy in ["MOMENTUM", "REVERSION", "DAY2"]:
            for ticker in ["SPY", "QQQ", "NVDA", "TSLA"]:
                key = f"{strategy}:{ticker}"
                trades = self.db.get_trades(
                    strategy=strategy, ticker=ticker, last_n=self.window
                )
                self.stats_cache[key] = self._compute_from_trades(trades)

        # Overall
        self.stats_cache["overall"] = self._compute_stats()

        logger.info(
            f"Rolling stats refreshed. "
            f"Overall: {self.stats_cache.get('overall', {})}"
        )

    def _compute_stats(
        self,
        strategy: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute stats for a given filter."""
        trades = self.db.get_trades(
            strategy=strategy, ticker=ticker, last_n=self.window
        )
        return self._compute_from_trades(trades)

    @staticmethod
    def _compute_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics from a list of trade records."""
        if not trades:
            return {
                "count": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe": 0,
            }

        outcomes = [t.get("outcome", "LOSS") for t in trades]
        pnls = [t.get("pnl_percent", 0) or 0 for t in trades]
        pnl_dollars = [t.get("pnl_dollars", 0) or 0 for t in trades]

        wins = sum(1 for o in outcomes if o == "WIN")
        losses = sum(1 for o in outcomes if o == "LOSS")
        count = len(trades)

        win_rate = wins / count if count > 0 else 0

        # Profit factor
        gross_profit = sum(p for p in pnl_dollars if p > 0)
        gross_loss = abs(sum(p for p in pnl_dollars if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0
        )

        # Max drawdown
        cumulative = np.cumsum(pnl_dollars)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Sharpe (simplified: mean return / std of returns)
        pnl_arr = np.array(pnls)
        sharpe = (
            float(np.mean(pnl_arr) / np.std(pnl_arr))
            if len(pnl_arr) > 1 and np.std(pnl_arr) > 0
            else 0
        )

        return {
            "count": count,
            "win_rate": win_rate,
            "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "total_pnl": sum(pnl_dollars),
        }

    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get cached stats for a given key."""
        return self.stats_cache.get(key, {})

    def get_win_rate(
        self,
        strategy: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> float:
        """Get win rate for a strategy/ticker combination."""
        if strategy and ticker:
            key = f"{strategy}:{ticker}"
        elif strategy:
            key = f"strategy:{strategy}"
        elif ticker:
            key = f"ticker:{ticker}"
        else:
            key = "overall"

        stats = self.stats_cache.get(key, {})
        return stats.get("win_rate", 0)

    def get_profit_factor(
        self,
        strategy: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> float:
        """Get profit factor for a strategy/ticker combination."""
        if strategy and ticker:
            key = f"{strategy}:{ticker}"
        elif strategy:
            key = f"strategy:{strategy}"
        else:
            key = "overall"

        stats = self.stats_cache.get(key, {})
        return stats.get("profit_factor", 0)
