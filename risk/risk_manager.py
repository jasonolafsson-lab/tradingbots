"""
Risk Manager.

Monitors open positions and determines when to exit:
- Stop loss (-25% single-leg, -40% spread, with ATR adjustment)
- Take profit (+50% single-leg, +80% spread)
- Trailing stop (activates at +25%, trails at 15%)
- Time stop (30 minutes max hold)
- End-of-day forced close (3:55 PM ET)
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from data.market_state import (
    Position, MarketState, ExitReason, ContractType
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")


class RiskManager:
    """Evaluates exit conditions for open positions."""

    def __init__(self, config: dict):
        self.config = config

        sl = config.get("stop_loss", {})
        self.sl_single = sl.get("single_leg_pct", 0.25)
        self.sl_spread = sl.get("spread_pct", 0.40)
        self.vol_adj_threshold = sl.get("vol_adjustment_threshold", 1.5)
        self.vol_adj_extra = sl.get("vol_adjustment_extra_pct", 0.05)

        tp = config.get("take_profit", {})
        self.tp_single = tp.get("single_leg_pct", 0.50)
        self.tp_spread = tp.get("spread_pct", 0.80)

        trail = config.get("trailing_stop", {})
        self.trail_activation = trail.get("activation_pct", 0.25)
        self.trail_distance = trail.get("trail_distance_pct", 0.15)

        time_stop = config.get("time_stop", {})
        self.max_hold_minutes = time_stop.get("max_hold_minutes", 30)

        sched = config.get("schedule", {})
        force_close_str = sched.get("force_close", "15:55")
        parts = force_close_str.split(":")
        self.force_close_time = dtime(int(parts[0]), int(parts[1]))

    def check_exit(
        self,
        position: Position,
        market_state: MarketState,
    ) -> Optional[ExitReason]:
        """
        Check all exit conditions for an open position.
        Returns ExitReason if position should be closed, None otherwise.
        Priority: circuit breaker > EOD > stop loss > time stop > trailing > TP
        """
        # Circuit breaker
        if market_state.circuit_breaker_triggered:
            return ExitReason.CIRCUIT_BREAKER

        # End of day
        now_et = datetime.now(ET).time()
        if now_et >= self.force_close_time:
            return ExitReason.EOD_CLOSE

        # Stop loss
        if self._is_stop_loss_hit(position, market_state):
            return ExitReason.STOP_LOSS

        # Time stop
        if self._is_time_stop_hit(position):
            return ExitReason.TIME_STOP

        # Trailing stop
        if self._is_trailing_stop_hit(position):
            return ExitReason.TRAILING_STOP

        # Take profit
        if self._is_take_profit_hit(position):
            return ExitReason.TAKE_PROFIT

        return None

    def _is_stop_loss_hit(
        self, position: Position, market_state: MarketState
    ) -> bool:
        """Check if stop loss is hit."""
        if position.entry_price <= 0:
            return False

        # Determine stop percentage
        if position.contract_type == ContractType.DEBIT_SPREAD:
            stop_pct = self.sl_spread
        else:
            stop_pct = self.sl_single

        # ATR volatility adjustment
        ts = market_state.tickers.get(position.ticker)
        if ts and ts.atr_14 > 0 and ts.atr_20_avg > 0:
            if ts.atr_14 > self.vol_adj_threshold * ts.atr_20_avg:
                stop_pct += self.vol_adj_extra
                logger.debug(
                    f"Vol-adjusted stop for {position.ticker}: "
                    f"{stop_pct:.0%} (ATR elevated)"
                )

        # Check if current loss exceeds stop
        if position.unrealized_pnl_pct <= -stop_pct:
            logger.info(
                f"STOP LOSS: {position.ticker} "
                f"loss={position.unrealized_pnl_pct:.1%} "
                f"stop={-stop_pct:.1%}"
            )
            return True

        return False

    def _is_take_profit_hit(self, position: Position) -> bool:
        """Check if take profit target is hit."""
        if position.contract_type == ContractType.DEBIT_SPREAD:
            tp_pct = self.tp_spread
        else:
            tp_pct = self.tp_single

        if position.unrealized_pnl_pct >= tp_pct:
            logger.info(
                f"TAKE PROFIT: {position.ticker} "
                f"gain={position.unrealized_pnl_pct:.1%} "
                f"target={tp_pct:.1%}"
            )
            return True

        return False

    def _is_trailing_stop_hit(self, position: Position) -> bool:
        """Check if trailing stop is hit."""
        if not position.trailing_active:
            return False

        if position.trailing_peak <= 0:
            return False

        # Trailing stop: price has fallen trail_distance from peak
        current_from_peak = (
            (position.current_price - position.trailing_peak)
            / position.trailing_peak
        )

        if current_from_peak <= -self.trail_distance:
            logger.info(
                f"TRAILING STOP: {position.ticker} "
                f"peak=${position.trailing_peak:.2f} "
                f"current=${position.current_price:.2f} "
                f"drop={current_from_peak:.1%}"
            )
            return True

        return False

    def _is_time_stop_hit(self, position: Position) -> bool:
        """Check if position has been held too long."""
        if position.entry_time is None:
            return False

        now = datetime.now(ET)
        entry = position.entry_time
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=ET)

        elapsed = (now - entry).total_seconds() / 60.0

        if elapsed >= self.max_hold_minutes:
            logger.info(
                f"TIME STOP: {position.ticker} "
                f"held {elapsed:.0f} min (max {self.max_hold_minutes})"
            )
            return True

        return False
