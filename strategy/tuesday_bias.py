"""
Tuesday Reversal Bias Modifier.

Not a standalone strategy — it modifies the VWAP Mean Reversion strategy
on eligible Tuesdays (when Monday's SPY close was red > 0.5%).

Effect: Prioritizes call setups over put setups.
Deactivates if SPY breaks below Monday's low.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.market_state import (
    TickerState, MarketState, Signal, Direction, Regime
)

logger = logging.getLogger(__name__)


class TuesdayBiasModifier:
    """Modifies reversion signals on eligible Tuesdays."""

    def __init__(self, config: dict):
        self.config = config
        tue = config.get("strategies", {}).get("tuesday_reversal", {})
        self.monday_red_threshold = tue.get("monday_red_threshold", -0.005)
        self.sector_green_count = tue.get("sector_green_count", 2)

    def is_active(self, market_state: MarketState) -> bool:
        """Check if Tuesday Reversal Bias conditions are met."""
        if market_state.day_of_week != "Tuesday":
            return False

        if market_state.monday_spy_close_return >= self.monday_red_threshold:
            return False

        return True

    def modify(
        self,
        signal: Optional[Signal],
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Modify a reversion signal based on Tuesday bias.

        - Call setups: boost priority (let through)
        - Put setups: deprioritize unless SPY breaks below Monday's low
        """
        if signal is None:
            return None

        if not self.is_active(market_state):
            return signal  # Pass through unchanged

        # Check for deactivation: SPY breaks below Monday's low
        # (If we had Monday's low stored, we'd check here)
        # For now, if SPY is making new lows, don't apply Tuesday bias
        if market_state.spy_session_return < self.monday_red_threshold * 2:
            logger.info("Tuesday bias deactivated: SPY making new lows")
            return signal

        if signal.direction == Direction.CALL:
            # Boost call setups on Tuesday reversal
            signal.strength_score = min(signal.strength_score + 15, 100)
            signal.tuesday_bias = True
            logger.info(
                f"{ts.ticker} Tuesday bias: boosting CALL signal "
                f"(strength now {signal.strength_score:.0f})"
            )

        elif signal.direction == Direction.PUT:
            # Deprioritize put setups
            signal.strength_score = max(signal.strength_score - 20, 0)
            logger.info(
                f"{ts.ticker} Tuesday bias: deprioritizing PUT signal "
                f"(strength now {signal.strength_score:.0f})"
            )
            if signal.strength_score < 30:
                return None  # Below threshold after penalty

        return signal
