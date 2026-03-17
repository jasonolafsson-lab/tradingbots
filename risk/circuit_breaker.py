"""
Circuit Breaker.

Enforces daily-level risk controls:
- Daily P&L loss limit (-3%)
- Consecutive loss cooldown (3 losses → 30 min)
- Cooldown between trades (5 min normal, 10 min after loss)
- Max trades per day (6)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from data.market_state import MarketState, Position

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")


class CircuitBreaker:
    """Enforces session-level risk limits."""

    def __init__(self, config: dict):
        risk = config.get("risk", {})
        self.daily_loss_limit = risk.get("daily_loss_limit_pct", 0.03)
        self.max_consecutive_losses = risk.get("max_consecutive_losses", 3)
        self.cooldown_normal = risk.get("cooldown_after_trade_sec", 300)
        self.cooldown_after_loss = risk.get("cooldown_after_loss_sec", 600)
        self.consecutive_cooldown = risk.get("consecutive_loss_cooldown_sec", 1800)
        self.max_trades = risk.get("max_trades_per_day", 6)

    def is_triggered(self, market_state: MarketState) -> bool:
        """Check if any circuit breaker condition is active."""
        # Daily loss limit
        if market_state.daily_pnl < 0:
            # We need account value to compute %
            # For now, check the flag
            if market_state.circuit_breaker_triggered:
                return True

        # Max trades
        if market_state.trades_today >= self.max_trades:
            logger.info(f"Max trades reached ({self.max_trades})")
            market_state.circuit_breaker_triggered = True
            return True

        return market_state.circuit_breaker_triggered

    def check_daily_pnl(
        self,
        daily_pnl: float,
        account_value: float,
        market_state: MarketState,
    ) -> bool:
        """
        Check if daily P&L exceeds the loss limit.
        Returns True if circuit breaker should be triggered.
        """
        if account_value <= 0:
            return False

        pnl_pct = daily_pnl / account_value

        if pnl_pct <= -self.daily_loss_limit:
            logger.warning(
                f"CIRCUIT BREAKER: Daily loss {pnl_pct:.2%} "
                f"exceeds limit {-self.daily_loss_limit:.2%}. "
                f"Shutting down for the day."
            )
            market_state.circuit_breaker_triggered = True
            return True

        return False

    def record_trade_result(
        self,
        position: Position,
        market_state: MarketState,
    ) -> None:
        """
        Record a completed trade and update circuit breaker state.
        Called after every trade exit.
        """
        market_state.trades_today += 1
        now = datetime.now(ET)
        market_state.last_trade_close_time = now

        # Determine if win or loss
        is_loss = position.unrealized_pnl_pct < 0

        if is_loss:
            market_state.consecutive_losses += 1
            cooldown_sec = self.cooldown_after_loss

            if market_state.consecutive_losses >= self.max_consecutive_losses:
                cooldown_sec = self.consecutive_cooldown
                logger.warning(
                    f"Consecutive losses: {market_state.consecutive_losses}. "
                    f"Cooldown: {cooldown_sec // 60} minutes."
                )
        else:
            market_state.consecutive_losses = 0
            cooldown_sec = self.cooldown_normal

        market_state.cooldown_until = now + timedelta(seconds=cooldown_sec)

        logger.info(
            f"Trade #{market_state.trades_today}: "
            f"{'LOSS' if is_loss else 'WIN'} "
            f"(consecutive losses: {market_state.consecutive_losses}) "
            f"Cooldown until {market_state.cooldown_until.strftime('%H:%M:%S')}"
        )
