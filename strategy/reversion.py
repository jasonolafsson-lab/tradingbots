"""
VWAP Mean Reversion Strategy.

Triggers when price is extended beyond VWAP ± 2 standard deviations
with RSI confirmation and slowing pressure.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.market_state import (
    TickerState, MarketState, Signal, Direction, Regime, Bias
)

logger = logging.getLogger(__name__)


class ReversionStrategy:
    """VWAP Mean Reversion signal generator."""

    def __init__(self, config: dict):
        self.config = config
        rev = config.get("strategies", {}).get("reversion", {})
        self.vwap_sd_threshold = rev.get("vwap_sd_threshold", 2.0)
        self.rsi_oversold = rev.get("rsi_oversold", 30)
        self.rsi_overbought = rev.get("rsi_overbought", 70)
        self.adx_max = rev.get("adx_max", 35)

    def evaluate(
        self,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Evaluate mean reversion conditions.
        Returns Signal if conditions met, None otherwise.
        """
        if ts.vwap <= 0:
            return None

        direction = self._get_direction(ts)
        if direction is None:
            return None

        # Countertrend guard: no reversion in strong trends
        if ts.adx_14 > self.adx_max:
            logger.debug(f"{ts.ticker} reversion: ADX too high ({ts.adx_14:.1f})")
            return None

        # Don't trade reversion into an active OR breakout
        if ts.opening_range_set:
            if (direction == Direction.CALL and
                    ts.last_price < ts.opening_range_low and
                    ts.adx_14 > 25):
                return None  # Active breakdown, not reversion
            if (direction == Direction.PUT and
                    ts.last_price > ts.opening_range_high and
                    ts.adx_14 > 25):
                return None  # Active breakout, not reversion

        strength = self._compute_strength(ts, direction, market_state)
        if strength < 30:
            return None

        return Signal(
            ticker=ts.ticker,
            direction=direction,
            strategy="REVERSION",
            regime=Regime.REVERSION,
            strength_score=strength,
            entry_price_target=ts.last_price,
        )

    def _get_direction(self, ts: TickerState) -> Optional[Direction]:
        """Determine direction based on VWAP deviation and RSI."""
        vwap_range = ts.vwap_upper_band - ts.vwap
        if vwap_range <= 0:
            return None

        price_vs_vwap_sd = (ts.last_price - ts.vwap) / vwap_range * 2.0

        # Below VWAP - 2 SD with RSI oversold → Call (bounce expected)
        if price_vs_vwap_sd < -self.vwap_sd_threshold and ts.rsi_7 < self.rsi_oversold:
            return Direction.CALL

        # Above VWAP + 2 SD with RSI overbought → Put (fade expected)
        if price_vs_vwap_sd > self.vwap_sd_threshold and ts.rsi_7 > self.rsi_overbought:
            return Direction.PUT

        return None

    def _compute_strength(
        self,
        ts: TickerState,
        direction: Direction,
        market_state: MarketState,
    ) -> float:
        """Compute signal strength score (0-100)."""
        score = 0.0

        vwap_range = ts.vwap_upper_band - ts.vwap
        if vwap_range <= 0:
            return 0.0

        price_vs_vwap_sd = abs((ts.last_price - ts.vwap) / vwap_range * 2.0)

        # VWAP deviation magnitude (max 25)
        dev_bonus = min((price_vs_vwap_sd - 2.0) / 1.0, 1.0) * 25
        score += max(dev_bonus, 0)

        # RSI extremity (max 25)
        if direction == Direction.CALL:
            rsi_bonus = max(0, (self.rsi_oversold - ts.rsi_7) / 15.0) * 25
        else:
            rsi_bonus = max(0, (ts.rsi_7 - self.rsi_overbought) / 15.0) * 25
        score += rsi_bonus

        # Volume slowing (selling/buying pressure fading) (max 15)
        # Lower volume ratio = less selling pressure on the move
        if ts.volume_ratio < 1.0:
            score += 15  # Volume declining = pressure slowing
        elif ts.volume_ratio < 1.3:
            score += 8

        # UW flow divergence (institutional buying on dip) (max 20)
        if direction == Direction.CALL and ts.uw_net_premium_direction == Bias.BULLISH:
            score += 20  # Price down but institutions buying calls
        elif direction == Direction.PUT and ts.uw_net_premium_direction == Bias.BEARISH:
            score += 20
        elif ts.uw_net_premium_direction == Bias.NEUTRAL:
            score += 5

        # Range-bound confirmation (ADX low) (max 15)
        if ts.adx_14 < 20:
            score += 15
        elif ts.adx_14 < 25:
            score += 8

        return min(score, 100.0)
