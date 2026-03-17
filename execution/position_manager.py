"""
Position Manager.

Tracks the current open position and verifies state against IBKR.
Ensures the bot's internal state matches actual broker state.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.ibkr_client import IBKRClient
from data.market_state import Position

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages position state and IBKR verification."""

    def __init__(self, ibkr: IBKRClient):
        self.ibkr = ibkr

    async def verify_no_position(self) -> bool:
        """
        Verify with IBKR that there are no open option positions.
        Called before any new entry to prevent double-positioning.
        """
        has_pos = await self.ibkr.has_open_position()
        if has_pos:
            logger.warning(
                "IBKR reports open option position when bot thinks none exist. "
                "Blocking new entry."
            )
        return not has_pos

    async def sync_position_price(self, position: Position) -> None:
        """
        Update position's current price from IBKR market data.
        Also update unrealized P&L and excursion tracking.
        """
        if position is None:
            return

        try:
            right = "C" if position.direction.value == "CALL" else "P"
            greeks = await self.ibkr.get_option_greeks(
                position.ticker,
                position.expiry,
                position.strike,
                right,
            )

            bid = greeks.get("bid", 0) or 0
            ask = greeks.get("ask", 0) or 0

            if bid > 0 and ask > 0:
                position.current_price = (bid + ask) / 2.0
            elif bid > 0:
                position.current_price = bid

            # Update P&L
            if position.entry_price > 0:
                position.unrealized_pnl_pct = (
                    (position.current_price - position.entry_price)
                    / position.entry_price
                )

                # Track maximum favorable and adverse excursion
                position.max_favorable = max(
                    position.max_favorable,
                    position.unrealized_pnl_pct,
                )
                position.max_adverse = min(
                    position.max_adverse,
                    position.unrealized_pnl_pct,
                )

                # Trailing stop tracking
                if (position.unrealized_pnl_pct >= 0.25 and
                        not position.trailing_active):
                    position.trailing_active = True
                    position.trailing_peak = position.current_price
                    logger.info(
                        f"Trailing stop activated for {position.ticker} "
                        f"at +{position.unrealized_pnl_pct:.1%}"
                    )

                if position.trailing_active:
                    position.trailing_peak = max(
                        position.trailing_peak,
                        position.current_price,
                    )

        except Exception as e:
            logger.warning(f"Failed to sync position price: {e}")
