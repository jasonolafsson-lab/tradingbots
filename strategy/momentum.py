"""
Momentum Breakout Strategy.

Triggers on breakout above Opening Range High (call) or below Opening Range Low (put).
Requires: ADX > 25, volume expansion, VWAP alignment, optional UW flow confirmation.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.market_state import (
    TickerState, MarketState, Signal, Direction, Regime, Bias
)

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """Momentum Breakout signal generator."""

    def __init__(self, config: dict):
        self.config = config
        mom = config.get("strategies", {}).get("momentum", {})
        self.adx_threshold = mom.get("adx_threshold", 25)
        self.volume_threshold = mom.get("volume_ratio_threshold", 1.3)
        self.min_strength = mom.get("min_strength", 30)

    def evaluate(
        self,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Evaluate momentum breakout conditions for a ticker.
        Returns a Signal if conditions are met, None otherwise.
        """
        if not ts.opening_range_set:
            return None

        # Determine direction
        direction = self._get_direction(ts)
        if direction is None:
            return None

        # Compute signal strength (0-100)
        strength = self._compute_strength(ts, direction, market_state)

        # GEX check: no major gamma wall within 0.3% of entry
        if ts.uw_gex_nearest_wall_distance < 0.003:
            logger.debug(
                f"{ts.ticker} momentum: gamma wall too close "
                f"({ts.uw_gex_nearest_wall_distance:.4f})"
            )
            strength -= 20  # Penalize but don't necessarily skip

        if strength < self.min_strength:
            return None  # Not enough confirmations

        return Signal(
            ticker=ts.ticker,
            direction=direction,
            strategy="MOMENTUM",
            regime=Regime.MOMENTUM,
            strength_score=strength,
            entry_price_target=ts.last_price,
        )

    def _get_direction(self, ts: TickerState) -> Optional[Direction]:
        """Determine call or put based on OR breakout direction."""
        if ts.last_price > ts.opening_range_high:
            # Breakout above OR — must be above VWAP with positive slope
            if ts.last_price > ts.vwap and ts.vwap_slope > 0:
                return Direction.CALL

        if ts.last_price < ts.opening_range_low:
            # Breakdown below OR — must be below VWAP with negative slope
            if ts.last_price < ts.vwap and ts.vwap_slope < 0:
                return Direction.PUT

        return None

    def _compute_strength(
        self,
        ts: TickerState,
        direction: Direction,
        market_state: MarketState,
    ) -> float:
        """
        Compute signal strength score (0-100).
        More confirmations = higher score.
        """
        score = 0.0

        # ADX strength (max 20)
        if ts.adx_14 >= self.adx_threshold:
            adx_bonus = min((ts.adx_14 - self.adx_threshold) / 15.0, 1.0) * 20
            score += adx_bonus

        # Volume expansion (max 20)
        if ts.volume_ratio >= self.volume_threshold:
            vol_bonus = min((ts.volume_ratio - 1.0) / 1.0, 1.0) * 20
            score += vol_bonus

        # VWAP slope alignment (max 15)
        if direction == Direction.CALL and ts.vwap_slope > 0:
            score += 15
        elif direction == Direction.PUT and ts.vwap_slope < 0:
            score += 15

        # OR break magnitude (max 15)
        if direction == Direction.CALL:
            break_pct = (ts.last_price - ts.opening_range_high) / ts.opening_range_high
        else:
            break_pct = (ts.opening_range_low - ts.last_price) / ts.opening_range_low
        score += min(break_pct / 0.005, 1.0) * 15

        # UW flow confirmation (max 15)
        if direction == Direction.CALL and ts.uw_net_premium_direction == Bias.BULLISH:
            score += 15
        elif direction == Direction.PUT and ts.uw_net_premium_direction == Bias.BEARISH:
            score += 15
        elif ts.uw_net_premium_direction == Bias.NEUTRAL:
            score += 5  # Neutral is okay

        # Scanner bias alignment (max 15)
        if ts.scanner_result:
            if direction == Direction.CALL and ts.scanner_result.bias == Bias.BULLISH:
                score += 15
            elif direction == Direction.PUT and ts.scanner_result.bias == Bias.BEARISH:
                score += 15

        return min(score, 100.0)
