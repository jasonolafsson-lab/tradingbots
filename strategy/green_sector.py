"""
Green Sector / Sector Relative Strength Strategy.

Triggers when SPY is red but one or more sector ETFs are green.
Prioritizes watchlist tickers in the green sector for call setups.
Uses momentum breakout or VWAP reversion entry logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.market_state import (
    TickerState, MarketState, Signal, Direction, Regime, Bias
)

logger = logging.getLogger(__name__)


class GreenSectorStrategy:
    """Green Sector signal generator."""

    def __init__(self, config: dict):
        self.config = config
        gs = config.get("strategies", {}).get("green_sector", {})
        self.spy_red_threshold = gs.get("spy_red_threshold", -0.003)
        self.sector_names_threshold = gs.get("sector_names_threshold", 2)

    def evaluate(
        self,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Evaluate Green Sector conditions for a ticker.
        Only generates CALL signals (sector showing strength).
        """
        # SPY must be red
        if market_state.spy_session_return >= self.spy_red_threshold:
            return None

        # This ticker's sector must be in the green list
        sector = self._get_ticker_sector(ts.ticker)
        if not sector or sector not in market_state.green_sectors:
            return None

        # Ticker itself should be positive or near positive
        if ts.session_return < -0.005:
            # Ticker is red along with SPY — not showing relative strength
            return None

        # Use momentum or reversion logic for entry refinement
        direction = Direction.CALL  # Green sector = bullish bias

        strength = self._compute_strength(ts, market_state, sector)
        if strength < 30:
            return None

        return Signal(
            ticker=ts.ticker,
            direction=direction,
            strategy="GREEN_SECTOR",
            regime=Regime.GREEN_SECTOR,
            strength_score=strength,
            entry_price_target=ts.last_price,
        )

    def _get_ticker_sector(self, ticker: str) -> Optional[str]:
        """Get the sector name for a watchlist ticker."""
        tickers_config = self.config.get("tickers_config", {})
        mapping = tickers_config.get("sector_mapping", {})
        for sector_name, info in mapping.items():
            if ticker in info.get("tickers", []):
                return sector_name
        return None

    def _compute_strength(
        self,
        ts: TickerState,
        market_state: MarketState,
        sector: str,
    ) -> float:
        """Compute signal strength (0-100)."""
        score = 0.0

        # Sector relative strength magnitude (max 25)
        sector_return = 0.0
        if ts.scanner_result:
            sector_return = ts.scanner_result.sector_rs
        rs = sector_return - market_state.spy_session_return
        if rs > 0.01:
            score += 25
        elif rs > 0.005:
            score += 15
        elif rs > 0:
            score += 8

        # Ticker's own session return (max 20)
        if ts.session_return > 0.005:
            score += 20
        elif ts.session_return > 0:
            score += 10

        # VWAP position (max 15)
        if ts.last_price > ts.vwap:
            score += 15

        # Volume (max 15)
        if ts.volume_ratio > 1.3:
            score += 15
        elif ts.volume_ratio > 1.0:
            score += 8

        # UW flow (max 15)
        if ts.uw_net_premium_direction == Bias.BULLISH:
            score += 15

        # Number of green sectors (breadth) (max 10)
        if len(market_state.green_sectors) >= 3:
            score += 10
        elif len(market_state.green_sectors) >= 2:
            score += 5

        return min(score, 100.0)
