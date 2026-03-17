"""
Day 2 Continuation Strategy.

Based on the X10 research paper insight that strong Day 1 moves
often see continuation on Day 2, especially with catalyst + volume.

Entry: Buy on pullback to prior-day breakout level or session VWAP.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.market_state import (
    TickerState, MarketState, Signal, Direction, Regime, Bias
)

logger = logging.getLogger(__name__)


class Day2Strategy:
    """Day 2 Continuation signal generator."""

    def __init__(self, config: dict):
        self.config = config
        d2 = config.get("strategies", {}).get("day2", {})
        self.score_threshold = d2.get("score_threshold", 70)
        self.spy_limit = d2.get("spy_selloff_limit", -0.005)

    def evaluate(
        self,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Evaluate Day 2 continuation conditions.
        Requires pre-market scanner to have flagged this ticker.
        """
        if ts.scanner_result is None:
            return None

        if ts.scanner_result.day2_score < self.score_threshold:
            return None

        # SPY sanity check
        if market_state.spy_session_return < self.spy_limit:
            logger.debug(
                f"{ts.ticker} Day2: SPY selling off "
                f"({market_state.spy_session_return:.3%})"
            )
            return None

        # Determine direction from scanner bias
        direction = self._get_direction(ts)
        if direction is None:
            return None

        # Check for pullback to support level
        if not self._is_pullback_to_support(ts, direction):
            return None

        # Check for volume pickup on resumption
        # (We look for volume starting to increase after the pullback)

        # Compute signal
        strength = self._compute_strength(ts, direction, market_state)
        if strength < 30:
            return None

        # Compute gap percentage for gap guard
        gap_pct = 0.0
        if ts.prior_close > 0 and ts.bars_1m:
            first_price = ts.bars_1m[0].open if ts.bars_1m else ts.last_price
            gap_pct = (first_price - ts.prior_close) / ts.prior_close

        return Signal(
            ticker=ts.ticker,
            direction=direction,
            strategy="DAY2",
            regime=Regime.DAY2_CONTINUATION,
            strength_score=strength,
            entry_price_target=ts.last_price,
            day2_score=ts.scanner_result.day2_score,
            gap_pct=gap_pct,
        )

    def _get_direction(self, ts: TickerState) -> Optional[Direction]:
        """Determine direction from scanner bias."""
        if ts.scanner_result is None:
            return None

        if ts.scanner_result.bias == Bias.BULLISH:
            return Direction.CALL
        elif ts.scanner_result.bias == Bias.BEARISH:
            return Direction.PUT

        # Strong close quality can also indicate direction
        if ts.scanner_result.close_quality > 0.80:
            return Direction.CALL
        elif ts.scanner_result.close_quality < 0.20:
            return Direction.PUT

        return None

    def _is_pullback_to_support(
        self, ts: TickerState, direction: Direction
    ) -> bool:
        """
        Check if price has pulled back to a support level and is holding.
        Support levels: prior-day breakout level, session VWAP, prior-day close.
        """
        if ts.last_price <= 0:
            return False

        if direction == Direction.CALL:
            # For bullish Day 2: price should be near support and holding
            support_levels = [
                ts.vwap,
                ts.prior_close,
                ts.prior_high,  # Prior day high as support on continuation
            ]
            for level in support_levels:
                if level <= 0:
                    continue
                # Price within 0.5% of support level and above it
                distance_pct = (ts.last_price - level) / level
                if -0.005 <= distance_pct <= 0.010:
                    return True

        elif direction == Direction.PUT:
            # For bearish Day 2: price near resistance and failing
            resistance_levels = [
                ts.vwap,
                ts.prior_close,
                ts.prior_low,  # Prior day low as resistance on bearish cont.
            ]
            for level in resistance_levels:
                if level <= 0:
                    continue
                distance_pct = (level - ts.last_price) / level
                if -0.005 <= distance_pct <= 0.010:
                    return True

        return False

    def _compute_strength(
        self,
        ts: TickerState,
        direction: Direction,
        market_state: MarketState,
    ) -> float:
        """Compute signal strength (0-100)."""
        score = 0.0

        # Day 2 score from scanner (max 30)
        if ts.scanner_result:
            score += (ts.scanner_result.day2_score / 100.0) * 30

        # Volume character (max 20)
        if ts.scanner_result and ts.scanner_result.volume_vs_avg > 1.5:
            score += 20
        elif ts.scanner_result and ts.scanner_result.volume_vs_avg > 1.2:
            score += 10

        # Close quality (max 15)
        if ts.scanner_result:
            if direction == Direction.CALL and ts.scanner_result.close_quality > 0.85:
                score += 15
            elif direction == Direction.PUT and ts.scanner_result.close_quality < 0.15:
                score += 15

        # VWAP alignment (max 15)
        if direction == Direction.CALL and ts.last_price >= ts.vwap:
            score += 15
        elif direction == Direction.PUT and ts.last_price <= ts.vwap:
            score += 15

        # SPY support (max 10)
        if direction == Direction.CALL and market_state.spy_session_return > 0:
            score += 10
        elif direction == Direction.PUT and market_state.spy_session_return < 0:
            score += 10

        # UW flow (max 10)
        if direction == Direction.CALL and ts.uw_net_premium_direction == Bias.BULLISH:
            score += 10
        elif direction == Direction.PUT and ts.uw_net_premium_direction == Bias.BEARISH:
            score += 10

        return min(score, 100.0)
