"""
Regime Engine.

Classifies the current market state into one of:
  NO_TRADE, DAY2_CONTINUATION, GREEN_SECTOR, MOMENTUM, REVERSION

Priority order (highest first):
  1. Safety checks → NO_TRADE
  2. Day 2 Continuation
  3. Green Sector
  4. Tuesday Reversal Bias (modifier, not standalone)
  5. Momentum Breakout
  6. Mean Reversion
  7. No Trade (default)
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime

from data.market_state import (
    TickerState, MarketState, Regime, Bias
)

logger = logging.getLogger(__name__)


class RegimeEngine:
    """Classifies the market regime for each ticker."""

    def __init__(self, config: dict):
        self.config = config
        strat = config.get("strategies", {})

        # Momentum thresholds
        mom = strat.get("momentum", {})
        self.adx_momentum_threshold = mom.get("adx_threshold", 25)
        self.volume_ratio_threshold = mom.get("volume_ratio_threshold", 1.3)

        # Reversion thresholds
        rev = strat.get("reversion", {})
        self.vwap_sd_threshold = rev.get("vwap_sd_threshold", 2.0)
        self.adx_max_reversion = rev.get("adx_max", 35)

        # Day 2 thresholds
        d2 = strat.get("day2", {})
        self.day2_score_threshold = d2.get("score_threshold", 70)
        self.day2_spy_limit = d2.get("spy_selloff_limit", -0.005)

        # Tuesday reversal thresholds
        tue = strat.get("tuesday_reversal", {})
        self.monday_red_threshold = tue.get("monday_red_threshold", -0.005)
        self.tuesday_sector_green = tue.get("sector_green_count", 2)

        # Green sector thresholds
        gs = strat.get("green_sector", {})
        self.spy_red_threshold = gs.get("spy_red_threshold", -0.003)
        self.sector_names_threshold = gs.get("sector_names_threshold", 2)

        # Strategy enable flags
        self.strategies_enabled = {
            "momentum": mom.get("enabled", True),
            "reversion": rev.get("enabled", True),
            "day2": d2.get("enabled", True),
            "tuesday_reversal": tue.get("enabled", True),
            "green_sector": gs.get("enabled", True),
        }

    def classify(
        self,
        ts: TickerState,
        market_state: MarketState,
    ) -> Regime:
        """
        Classify the regime for a ticker. Returns the highest-priority
        regime that matches current conditions.
        """
        # 1. Safety checks → NO_TRADE
        if self._safety_check_fails(market_state):
            return Regime.NO_TRADE

        # 2. Day 2 Continuation
        if (self.strategies_enabled["day2"] and
                self._is_day2_candidate(ts, market_state)):
            return Regime.DAY2_CONTINUATION

        # 3. Green Sector
        if (self.strategies_enabled["green_sector"] and
                self._is_green_sector(ts, market_state)):
            return Regime.GREEN_SECTOR

        # 4. Tuesday Reversal Bias (modifier flag, not standalone regime)
        if (self.strategies_enabled["tuesday_reversal"] and
                self._is_tuesday_reversal(market_state)):
            ts.tuesday_bias_active = True
        else:
            ts.tuesday_bias_active = False

        # 5. Momentum Breakout
        if (self.strategies_enabled["momentum"] and
                self._is_momentum(ts)):
            return Regime.MOMENTUM

        # 6. Mean Reversion
        if (self.strategies_enabled["reversion"] and
                self._is_reversion(ts)):
            return Regime.REVERSION

        # 7. Default: No Trade
        return Regime.NO_TRADE

    def _safety_check_fails(self, market_state: MarketState) -> bool:
        """Check if any safety condition blocks trading."""
        # Circuit breaker
        if market_state.circuit_breaker_triggered:
            return True

        # Cooldown active
        if market_state.is_in_cooldown():
            return True

        # Max trades per day
        max_trades = self.config.get("risk", {}).get("max_trades_per_day", 6)
        if market_state.trades_today >= max_trades:
            return True

        return False

    def _is_day2_candidate(
        self, ts: TickerState, market_state: MarketState
    ) -> bool:
        """Check if ticker qualifies for Day 2 Continuation regime."""
        if ts.scanner_result is None:
            return False

        # Day 2 score must meet threshold
        if ts.scanner_result.day2_score < self.day2_score_threshold:
            return False

        # SPY must not be actively selling off
        if market_state.spy_session_return < self.day2_spy_limit:
            return False

        return True

    def _is_green_sector(
        self, ts: TickerState, market_state: MarketState
    ) -> bool:
        """Check if Green Sector conditions are met for this ticker."""
        # SPY must be red
        if market_state.spy_session_return >= self.spy_red_threshold:
            return False

        # Need green sectors
        if len(market_state.green_sectors) < 1:
            return False

        # Check if this ticker's sector is green
        tickers_config = self.config.get("tickers_config", {})
        mapping = tickers_config.get("sector_mapping", {})
        for sector_name, info in mapping.items():
            if ts.ticker in info.get("tickers", []):
                if sector_name in market_state.green_sectors:
                    return True

        return False

    def _is_tuesday_reversal(self, market_state: MarketState) -> bool:
        """Check if Tuesday Reversal Bias conditions are met."""
        if market_state.day_of_week != "Tuesday":
            return False

        # Monday's SPY close must have been red > threshold
        if market_state.monday_spy_close_return >= self.monday_red_threshold:
            return False

        return True

    def _is_momentum(self, ts: TickerState) -> bool:
        """Check if momentum breakout conditions are met."""
        # ADX must show trend
        if ts.adx_14 < self.adx_momentum_threshold:
            return False

        # Must have opening range set
        if not ts.opening_range_set:
            return False

        # Price must have broken OR high or low
        if ts.last_price <= 0:
            return False

        broke_high = ts.last_price > (ts.opening_range_high or float("inf"))
        broke_low = ts.last_price < (ts.opening_range_low or 0)

        if not (broke_high or broke_low):
            return False

        # Volume should be expanding
        if ts.volume_ratio < self.volume_ratio_threshold:
            return False

        # VWAP alignment
        if broke_high and ts.last_price < ts.vwap:
            return False  # Breakout above OR but below VWAP — weak
        if broke_low and ts.last_price > ts.vwap:
            return False  # Breakdown below OR but above VWAP — weak

        return True

    def _is_reversion(self, ts: TickerState) -> bool:
        """Check if mean reversion conditions are met."""
        # Must NOT be in strong trend
        if ts.adx_14 > self.adx_max_reversion:
            return False

        # Check if price is extended from VWAP
        if ts.vwap <= 0:
            return False

        vwap_range = ts.vwap_upper_band - ts.vwap
        if vwap_range <= 0:
            return False

        price_vs_vwap_sd = (ts.last_price - ts.vwap) / vwap_range * 2.0

        # Extended below VWAP with RSI oversold
        if price_vs_vwap_sd < -self.vwap_sd_threshold and ts.rsi_7 < 30:
            return True

        # Extended above VWAP with RSI overbought
        if price_vs_vwap_sd > self.vwap_sd_threshold and ts.rsi_7 > 70:
            return True

        return False
