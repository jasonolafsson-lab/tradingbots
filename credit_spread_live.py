"""
0DTE Credit Spread Seller — Live Paper Trading Runner ("Bot 4")

Sells OTM credit spreads on SPY, profiting from theta decay.
Direction chosen by morning brief sentiment:
  BULLISH  -> sell put credit spreads (OTM puts expire worthless)
  BEARISH  -> sell call credit spreads (OTM calls expire worthless)
  NEUTRAL  -> sell both sides (iron condor as 2 separate spreads)

Target: 8-15% monthly returns via high win-rate premium collection.

Usage:
    python credit_spread_live.py                  # Normal mode (paper trading)
    python credit_spread_live.py --dry-run        # Signal generation only, no orders
    python credit_spread_live.py --config alt.yaml

Architecture:
    - Reuses Bot 1's IBKR client and order manager
    - Own signal logic (IV percentile + sentiment + delta selection)
    - Sells spreads (collects credit) — opposite of Bots 1-3
    - Client ID 4 to avoid conflicts with Bot 1 (id=1), Bot 2 (id=2), Bot 3 (id=3)
    - Tracks multiple open positions (up to 2 simultaneous spreads)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml
from zoneinfo import ZoneInfo

from data.ibkr_client import IBKRClient, IBKRConnectionError, IBKRAccountError
from data.market_state import (
    Bar, TickerState, MarketState, Signal,
    Regime, Direction, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from execution.order_manager import OrderManager
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window

logger = logging.getLogger("credit_spread")
ET = ZoneInfo("US/Eastern")

PROJECT_ROOT = Path(__file__).parent


# ============================================================
# Credit Spread Position
# ============================================================

@dataclass
class CreditSpreadPosition:
    """Tracks a single credit spread position."""
    ticker: str
    spread_type: str              # "PUT_CREDIT" or "CALL_CREDIT"
    short_strike: float
    long_strike: float
    expiry: str
    spread_width: float           # dollar width between strikes
    credit_received: float        # per-spread credit (positive number)
    num_contracts: int
    entry_time: datetime

    # IBKR contracts for closing
    short_contract: Any = None
    long_contract: Any = None
    right: str = "P"              # "P" for put spread, "C" for call spread

    # Live tracking
    current_spread_value: float = 0.0   # Current cost to buy back
    unrealized_pnl_pct: float = 0.0     # % of max profit captured
    max_favorable: float = 0.0          # Best P&L seen
    max_adverse: float = 0.0            # Worst P&L seen

    # Market context at entry
    iv_percentile: float = 0.0
    short_delta_at_entry: float = 0.0
    sentiment_at_entry: str = "NEUTRAL"
    signal_strength: float = 0.0

    @property
    def max_profit(self) -> float:
        """Max profit per contract = credit received * 100."""
        return self.credit_received * 100 * self.num_contracts

    @property
    def max_loss(self) -> float:
        """Max loss per contract = (width - credit) * 100."""
        return (self.spread_width - self.credit_received) * 100 * self.num_contracts

    @property
    def pnl_dollars(self) -> float:
        """Current P&L in dollars."""
        return (self.credit_received - self.current_spread_value) * 100 * self.num_contracts


# ============================================================
# Config
# ============================================================

class CreditSpreadConfig:
    """Live trading config for the credit spread seller."""

    def __init__(self, config: dict):
        cs = config.get("credit_spread_bot", {})

        # Account
        self.account_value = cs.get("account_value", 25_000)
        self.risk_per_trade_pct = cs.get("risk_per_trade_pct", 0.03)
        self.max_contracts = cs.get("max_contracts", 10)

        # Schedule
        self.earliest_entry = dtime(10, 0)       # 10:00 AM ET
        self.last_entry = dtime(14, 0)           # 2:00 PM ET
        self.gamma_warning = dtime(14, 30)       # 2:30 PM — tighten stops
        self.force_close = dtime(15, 55)         # 3:55 PM

        # Spread parameters
        self.ticker = cs.get("ticker", "SPY")
        self.spread_width = cs.get("spread_width", 2)
        self.short_leg_delta_min = cs.get("short_leg_delta_min", 0.10)
        self.short_leg_delta_max = cs.get("short_leg_delta_max", 0.20)
        self.min_credit = cs.get("min_credit", 0.30)
        self.iv_percentile_min = cs.get("iv_percentile_min", 30)

        # Exit rules
        self.take_profit_pct = cs.get("take_profit_pct", 0.50)
        self.stop_loss_multiplier = cs.get("stop_loss_multiplier", 2.0)
        self.gamma_warning_multiplier = cs.get("gamma_warning_multiplier", 1.5)

        # Risk
        self.max_trades_per_day = cs.get("max_trades_per_day", 3)
        self.max_open_spreads = cs.get("max_open_spreads", 2)
        self.cooldown_after_trade_sec = cs.get("cooldown_after_trade_sec", 120)
        self.cooldown_after_loss_sec = cs.get("cooldown_after_loss_sec", 300)
        self.daily_loss_limit_pct = cs.get("daily_loss_limit_pct", 0.05)

        # Execution
        self.midpoint_offset = cs.get("midpoint_offset", 0.02)


# ============================================================
# Signal Generation
# ============================================================

def check_credit_spread_signal(
    sentiment: str,
    vix_level: float,
    vix_regime: str,
    iv_percentile: float,
    config: CreditSpreadConfig,
    current_time: dtime,
) -> Optional[Dict]:
    """
    Determine if conditions are right for a credit spread.
    Returns signal dict with spread_type and strength, or None.
    """
    # IV filter: only sell when there's decent premium
    if iv_percentile < config.iv_percentile_min:
        return None

    # Determine spread direction from sentiment
    if sentiment == "BULLISH":
        spread_type = "PUT_CREDIT"   # Sell OTM puts (bullish = puts expire worthless)
    elif sentiment == "BEARISH":
        spread_type = "CALL_CREDIT"  # Sell OTM calls (bearish = calls expire worthless)
    else:
        spread_type = "PUT_CREDIT"   # Default to put spreads (upward bias)

    # Compute strength score (0-100)
    strength = 0.0

    # IV percentile contribution (max 30 pts)
    iv_score = min((iv_percentile - 30) / 40, 1.0) * 30
    strength += max(iv_score, 0)

    # Sentiment clarity (max 20 pts)
    if sentiment in ("BULLISH", "BEARISH"):
        strength += 20
    else:
        strength += 10  # Neutral = less conviction

    # VIX regime bonus (max 15 pts)
    if vix_regime in ("ELEVATED", "HIGH"):
        strength += 15
    elif vix_regime == "NORMAL":
        strength += 8

    # Time of day: earlier = more theta to capture (max 10 pts)
    hour = current_time.hour
    if hour <= 11:
        strength += 10
    elif hour <= 13:
        strength += 7
    else:
        strength += 3

    # VIX level bonus (max 10 pts)
    if vix_level > 20:
        strength += 10
    elif vix_level > 17:
        strength += 6
    elif vix_level > 14:
        strength += 3

    # Minimum strength threshold
    if strength < 35:
        return None

    return {
        "spread_type": spread_type,
        "strength": min(strength, 100.0),
        "iv_percentile": iv_percentile,
    }


# ============================================================
# Position Sizing
# ============================================================

def size_credit_spread(
    credit: float,
    spread_width: float,
    config: CreditSpreadConfig,
    account_value: float,
    size_multiplier: float = 1.0,
) -> int:
    """Conservative position sizing for credit spreads."""
    if credit <= 0 or spread_width <= 0:
        return 0

    # Max loss per contract = (width - credit) * 100
    max_loss_per = (spread_width - credit) * 100
    if max_loss_per <= 0:
        return 0

    # Max risk dollars
    max_risk = account_value * config.risk_per_trade_pct
    contracts = math.floor(max_risk / max_loss_per)

    # Apply morning brief multiplier
    if size_multiplier != 1.0:
        contracts = max(1, int(contracts * size_multiplier))

    contracts = min(contracts, config.max_contracts)
    return max(contracts, 1) if contracts > 0 else 0


# ============================================================
# Live Credit Spread Bot
# ============================================================

class CreditSpreadBot:
    """Live paper trading bot for 0DTE Credit Spreads (Bot 4)."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.cs_config = CreditSpreadConfig(config)
        self.running = False
        self.shutdown_requested = False

        # IBKR connection — client_id 4
        config_copy = dict(config)
        config_copy["ibkr"] = dict(config["ibkr"])
        config_copy["ibkr"]["client_id"] = 4
        self.ibkr = IBKRClient(config_copy)

        # Execution
        self.order_manager = OrderManager(config, self.ibkr)

        # Indicators (for VWAP reference)
        self.vwap_calc = VWAPCalculator()
        self.market_state = MarketState()

        # Position tracking — multiple simultaneous spreads
        self.positions: List[CreditSpreadPosition] = []
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.cooldown_until: Optional[datetime] = None
        self.circuit_breaker: bool = False

        # Morning brief data
        self.size_multiplier: float = 1.0
        self.sentiment: str = "NEUTRAL"
        self.vix_level: float = 18.0
        self.vix_regime: str = "NORMAL"

        # Trade log
        self.trade_log: List[dict] = []

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def _time_et(self) -> dtime:
        return self._now_et().time()

    async def start(self):
        """Main entry point."""
        self._setup_signals()

        logger.info("=" * 60)
        logger.info("  BOT 4 — 0DTE Credit Spread Seller")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Ticker: {self.cs_config.ticker}")
        logger.info(f"Spread width: ${self.cs_config.spread_width}")
        logger.info(f"Short delta: {self.cs_config.short_leg_delta_min}-{self.cs_config.short_leg_delta_max}")
        logger.info(f"Min credit: ${self.cs_config.min_credit}")
        logger.info(f"IV threshold: >{self.cs_config.iv_percentile_min}th percentile")
        logger.info(f"Window: {self.cs_config.earliest_entry} - {self.cs_config.last_entry} ET")
        logger.info(f"TP: {self.cs_config.take_profit_pct:.0%} of max | "
                    f"SL: {self.cs_config.stop_loss_multiplier}x credit")
        logger.info(f"Max trades/day: {self.cs_config.max_trades_per_day} | "
                    f"Max open: {self.cs_config.max_open_spreads}")
        logger.info(f"Risk/trade: {self.cs_config.risk_per_trade_pct:.0%} | "
                    f"Daily limit: -{self.cs_config.daily_loss_limit_pct:.0%}")

        try:
            logger.info("Connecting to IBKR...")
            await self.ibkr.connect()
            logger.info("IBKR connected (client_id=4).")

            self.running = True
            await self._main_loop()

        except IBKRConnectionError as e:
            logger.error(f"IBKR connection failed: {e}")
        except IBKRAccountError as e:
            logger.error(f"IBKR account error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def _main_loop(self):
        """Main loop — sell credit spreads based on conditions."""
        ticker = self.cs_config.ticker

        # Subscribe to streaming data
        await self.ibkr.subscribe_realtime_bars(ticker)
        logger.info(f"Subscribed to {ticker} real-time bars")

        # Load morning brief
        brief = load_daily_brief()
        if brief:
            self.size_multiplier = brief.bot4_spreads.get("size_multiplier", 1.0)
            if not brief.bot4_spreads.get("enabled", True):
                logger.warning(f"Morning brief says Bot 4 DISABLED: "
                             f"{brief.bot4_spreads.get('notes', '')}")
                return

            self.sentiment = brief.market_sentiment
            self.vix_level = brief.vix_level
            self.vix_regime = brief.vix_regime

            logger.info(f"Morning brief: sentiment={self.sentiment}, "
                       f"risk={brief.risk_level}, VIX={self.vix_level:.1f} "
                       f"({self.vix_regime}), size={self.size_multiplier}x")
            if brief.no_trade_windows:
                for w in brief.no_trade_windows:
                    logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")
            if brief.bot4_spreads.get("notes"):
                logger.info(f"  Notes: {brief.bot4_spreads['notes']}")

        # Wait for entry window
        now_t = self._time_et()
        if now_t < self.cs_config.earliest_entry:
            wait_mins = (
                datetime.combine(datetime.today(), self.cs_config.earliest_entry) -
                datetime.combine(datetime.today(), now_t)
            ).total_seconds() / 60
            logger.info(f"Waiting {wait_mins:.0f} min for entry window "
                       f"({self.cs_config.earliest_entry} ET)...")

            while self.running and self._time_et() < self.cs_config.earliest_entry:
                await self._check_kill_switch()
                await asyncio.sleep(5)

        logger.info("Entry window open — scanning for credit spread opportunities...")

        # Main trading loop
        while self.running and self._time_et() < self.cs_config.force_close:
            await self._check_kill_switch()

            if self.circuit_breaker:
                logger.warning("Circuit breaker active — no new trades")
                for pos in list(self.positions):
                    await self._manage_position(pos)
                await asyncio.sleep(10)
                continue

            # Manage all open positions
            for pos in list(self.positions):
                await self._manage_position(pos)

            # Look for new entries
            if (self._time_et() <= self.cs_config.last_entry and
                    len(self.positions) < self.cs_config.max_open_spreads and
                    self.trades_today < self.cs_config.max_trades_per_day and
                    not self._in_cooldown() and
                    not is_in_no_trade_window(self._time_et())):
                await self._scan_for_signal()

            await asyncio.sleep(10)  # 10-second loop

        # EOD
        await self._force_close_eod()

    async def _scan_for_signal(self):
        """Check conditions for a credit spread entry."""
        # Get current IV estimate (use VIX as proxy if UW not available)
        iv_percentile = self._estimate_iv_percentile()

        sig = check_credit_spread_signal(
            sentiment=self.sentiment,
            vix_level=self.vix_level,
            vix_regime=self.vix_regime,
            iv_percentile=iv_percentile,
            config=self.cs_config,
            current_time=self._time_et(),
        )

        if sig is None:
            return

        spread_type = sig["spread_type"]

        # Check if we already have this type of spread open
        for pos in self.positions:
            if pos.spread_type == spread_type:
                return  # Don't double up on same direction

        logger.info(
            f"SIGNAL: {spread_type} spread "
            f"strength={sig['strength']:.0f} IV%={iv_percentile:.0f} "
            f"sentiment={self.sentiment}"
        )

        if not self.dry_run:
            await self._execute_credit_spread(spread_type, sig)
        else:
            logger.info(f"  [DRY RUN] Would sell {spread_type} spread")

    def _estimate_iv_percentile(self) -> float:
        """Estimate IV percentile using VIX as proxy."""
        # VIX-based heuristic:
        #   VIX < 13 -> ~10th percentile
        #   VIX 13-16 -> ~25th percentile
        #   VIX 16-19 -> ~45th percentile
        #   VIX 19-22 -> ~60th percentile
        #   VIX 22-28 -> ~75th percentile
        #   VIX > 28 -> ~90th percentile
        vix = self.vix_level
        if vix <= 0:
            return 40  # Default

        if vix < 13:
            return 10 + (vix - 10) * 5
        elif vix < 16:
            return 25 + (vix - 13) * 6.7
        elif vix < 19:
            return 45 + (vix - 16) * 5
        elif vix < 22:
            return 60 + (vix - 19) * 5
        elif vix < 28:
            return 75 + (vix - 22) * 2.5
        else:
            return min(90 + (vix - 28), 99)

    async def _execute_credit_spread(self, spread_type: str, sig: Dict):
        """Select strikes, size, and sell the credit spread."""
        ticker = self.cs_config.ticker
        ts = self.market_state.get_ticker(ticker)

        try:
            # 1. Select strikes
            spread_info = await self._select_spread_strikes(spread_type, ts)
            if spread_info is None:
                logger.warning(f"No suitable {spread_type} spread found")
                return

            # 2. Size position
            account_value = self.cs_config.account_value
            try:
                if not self.dry_run:
                    live_value = await self.ibkr.get_net_liquidation()
                    if live_value and live_value > 0:
                        account_value = live_value
            except Exception:
                pass

            contracts = size_credit_spread(
                credit=spread_info["credit"],
                spread_width=spread_info["spread_width"],
                config=self.cs_config,
                account_value=account_value,
                size_multiplier=self.size_multiplier,
            )

            if contracts == 0:
                logger.warning("Position size = 0, skipping")
                return

            credit = spread_info["credit"]
            logger.info(
                f"ENTRY: SELL {contracts}x {ticker} "
                f"{spread_info['short_strike']}/{spread_info['long_strike']}"
                f"{spread_info['right']} @ ${credit:.2f} credit "
                f"(max loss: ${(spread_info['spread_width'] - credit) * 100 * contracts:,.0f})"
            )

            # 3. Submit order
            trade = await self.order_manager.submit_credit_spread_entry(
                spread_info=spread_info,
                quantity=contracts,
            )

            # 4. Wait for fill
            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=15)
            if not filled:
                filled = await self.order_manager.reprice(trade, max_attempts=1)
                if not filled:
                    await self.ibkr.cancel_order(trade)
                    logger.warning("Credit spread order not filled, cancelled")
                    return

            # 5. Build position
            fill_price = credit  # Use our target credit
            if trade.orderStatus and trade.orderStatus.avgFillPrice:
                fill_price = abs(trade.orderStatus.avgFillPrice)

            position = CreditSpreadPosition(
                ticker=ticker,
                spread_type=spread_type,
                short_strike=spread_info["short_strike"],
                long_strike=spread_info["long_strike"],
                expiry=spread_info["expiry"],
                spread_width=spread_info["spread_width"],
                credit_received=fill_price,
                num_contracts=contracts,
                entry_time=datetime.now(ET),
                short_contract=spread_info["short_contract"],
                long_contract=spread_info["long_contract"],
                right=spread_info["right"],
                iv_percentile=sig.get("iv_percentile", 0),
                short_delta_at_entry=spread_info.get("short_delta", 0),
                sentiment_at_entry=self.sentiment,
                signal_strength=sig["strength"],
                current_spread_value=fill_price,
            )

            self.positions.append(position)

            logger.info(
                f"FILLED: {spread_type} {ticker} "
                f"{spread_info['short_strike']}/{spread_info['long_strike']}"
                f"{spread_info['right']} x{contracts} "
                f"@ ${fill_price:.2f} credit"
            )

        except Exception as e:
            logger.error(f"Error executing credit spread: {e}", exc_info=True)

    async def _select_spread_strikes(
        self, spread_type: str, ts: TickerState
    ) -> Optional[Dict]:
        """Select short and long strikes for the credit spread."""
        ticker = self.cs_config.ticker

        try:
            chains = await self.ibkr.get_option_chains(ticker)
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None

        if not chains:
            return None

        # Find 0DTE expiry
        today = datetime.now().strftime("%Y%m%d")
        available_expiries = set()
        for chain in chains:
            for exp in chain.get("expirations", []):
                available_expiries.add(exp)

        target_expiry = None
        if today in available_expiries:
            target_expiry = today
        else:
            future = sorted([e for e in available_expiries if e >= today])
            if future:
                target_expiry = future[0]

        if not target_expiry:
            return None

        # Get all strikes
        all_strikes = set()
        for chain in chains:
            all_strikes.update(chain.get("strikes", []))
        strikes = sorted(all_strikes)

        price = ts.last_price
        if price <= 0:
            return None

        # Determine right and strike direction
        if spread_type == "PUT_CREDIT":
            right = "P"
            # Short put below current price, long put even lower
            candidates = [s for s in strikes if s < price]
        else:  # CALL_CREDIT
            right = "C"
            # Short call above current price, long call even higher
            candidates = [s for s in strikes if s > price]

        if not candidates:
            return None

        # Sort: for puts, closest to price first; for calls, closest first
        if spread_type == "PUT_CREDIT":
            candidates = sorted(candidates, reverse=True)  # Highest OTM puts first
        else:
            candidates = sorted(candidates)  # Lowest OTM calls first

        # Find short leg at target delta (0.10 - 0.20)
        best = None
        target_delta = (self.cs_config.short_leg_delta_min +
                       self.cs_config.short_leg_delta_max) / 2  # 0.15

        for strike in candidates[:15]:  # Check up to 15 strikes
            try:
                greeks = await self.ibkr.get_option_greeks(
                    ticker, target_expiry, strike, right
                )
            except Exception:
                continue

            delta = abs(greeks.get("delta", 0) or 0)
            bid = greeks.get("bid", 0) or 0
            ask = greeks.get("ask", 0) or 0

            if bid <= 0 or ask <= 0:
                continue

            # Delta must be in our range
            if delta < self.cs_config.short_leg_delta_min:
                continue
            if delta > self.cs_config.short_leg_delta_max:
                continue

            mid = (bid + ask) / 2.0

            dist = abs(delta - target_delta)
            if best is None or dist < best["_dist"]:
                best = {
                    "short_strike": strike,
                    "short_mid": mid,
                    "short_bid": bid,
                    "short_ask": ask,
                    "short_delta": delta,
                    "short_iv": greeks.get("impliedVol", 0) or 0,
                    "_dist": dist,
                }

        if best is None:
            return None

        # Determine long leg strike
        short_strike = best["short_strike"]
        width = self.cs_config.spread_width

        if spread_type == "PUT_CREDIT":
            long_strike = short_strike - width
        else:
            long_strike = short_strike + width

        # Verify long strike exists
        if long_strike not in all_strikes:
            # Find nearest available strike
            nearest = min(strikes, key=lambda s: abs(s - long_strike))
            if abs(nearest - long_strike) > width * 0.5:
                return None  # Too far from target
            long_strike = nearest
            width = abs(short_strike - long_strike)

        # Get long leg greeks
        try:
            long_greeks = await self.ibkr.get_option_greeks(
                ticker, target_expiry, long_strike, right
            )
        except Exception:
            return None

        long_bid = long_greeks.get("bid", 0) or 0
        long_ask = long_greeks.get("ask", 0) or 0
        long_mid = (long_bid + long_ask) / 2.0

        # Net credit = short premium - long premium
        net_credit = best["short_mid"] - long_mid
        if net_credit < self.cs_config.min_credit:
            logger.debug(f"Credit ${net_credit:.2f} below minimum ${self.cs_config.min_credit}")
            return None

        # Qualify contracts
        try:
            short_contract = self.ibkr.make_option_contract(
                ticker, target_expiry, short_strike, right
            )
            long_contract = self.ibkr.make_option_contract(
                ticker, target_expiry, long_strike, right
            )
            short_contract = await self.ibkr.qualify_contract(short_contract)
            long_contract = await self.ibkr.qualify_contract(long_contract)
        except Exception as e:
            logger.error(f"Contract qualification failed: {e}")
            return None

        return {
            "short_strike": short_strike,
            "long_strike": long_strike,
            "spread_width": width,
            "credit": net_credit,
            "right": right,
            "expiry": target_expiry,
            "short_contract": short_contract,
            "long_contract": long_contract,
            "short_delta": best["short_delta"],
            "short_iv": best["short_iv"],
        }

    async def _manage_position(self, pos: CreditSpreadPosition):
        """Monitor and manage an open credit spread position."""
        now = self._now_et()
        hold_mins = (now - pos.entry_time).total_seconds() / 60.0

        # Update current spread value
        if not self.dry_run:
            try:
                short_greeks = await self.ibkr.get_option_greeks(
                    pos.ticker, pos.expiry, pos.short_strike, pos.right
                )
                long_greeks = await self.ibkr.get_option_greeks(
                    pos.ticker, pos.expiry, pos.long_strike, pos.right
                )

                short_bid = short_greeks.get("bid", 0) or 0
                short_ask = short_greeks.get("ask", 0) or 0
                long_bid = long_greeks.get("bid", 0) or 0
                long_ask = long_greeks.get("ask", 0) or 0

                if short_bid > 0 or short_ask > 0:
                    short_mid = (short_bid + short_ask) / 2.0
                    long_mid = (long_bid + long_ask) / 2.0
                    pos.current_spread_value = max(short_mid - long_mid, 0)

            except Exception as e:
                logger.debug(f"Price update error: {e}")

        # Calculate P&L as percentage of credit received
        if pos.credit_received > 0:
            # Profit = credit - current_value, as % of credit
            profit_pct = (pos.credit_received - pos.current_spread_value) / pos.credit_received
            pos.unrealized_pnl_pct = profit_pct
            pos.max_favorable = max(pos.max_favorable, profit_pct)
            pos.max_adverse = min(pos.max_adverse, profit_pct)

        # Check exits (priority order)
        exit_reason = None
        stop_multiplier = self.cs_config.stop_loss_multiplier

        # 1. EOD force close
        if self._time_et() >= self.cs_config.force_close:
            exit_reason = ExitReason.EOD_CLOSE

        # 2. Stop loss: spread value has increased to 2x credit
        elif pos.current_spread_value >= pos.credit_received * stop_multiplier:
            exit_reason = ExitReason.STOP_LOSS

        # 3. Take profit: spread value has decayed to 50% of credit
        elif pos.current_spread_value <= pos.credit_received * (1 - self.cs_config.take_profit_pct):
            exit_reason = ExitReason.TAKE_PROFIT

        # 4. Gamma warning: after 2:30 PM, tighten stop
        elif (self._time_et() >= self.cs_config.gamma_warning and
              pos.current_spread_value >= pos.credit_received * self.cs_config.gamma_warning_multiplier):
            exit_reason = ExitReason.TIME_STOP
            logger.info("Gamma warning: stop tightened, closing position")

        # 5. Time stop: within 30 min of close and losing
        elif (self._time_et() >= dtime(15, 25) and
              pos.current_spread_value > pos.credit_received):
            exit_reason = ExitReason.TIME_STOP

        if exit_reason is not None:
            await self._close_position(pos, exit_reason)

    async def _close_position(self, pos: CreditSpreadPosition, reason: ExitReason):
        """Close a credit spread position."""
        pnl_dollars = pos.pnl_dollars
        pnl_pct = pos.unrealized_pnl_pct
        hold_mins = (self._now_et() - pos.entry_time).total_seconds() / 60.0

        emoji = "+" if pnl_dollars > 0 else "-"
        logger.info(
            f"EXIT [{emoji}]: {pos.spread_type} {pos.ticker} "
            f"{pos.short_strike}/{pos.long_strike}{pos.right} "
            f"({reason.value}) P&L: ${pnl_dollars:+,.2f} "
            f"({pnl_pct:+.1%} of credit) hold={hold_mins:.1f}min"
        )

        if not self.dry_run:
            try:
                await self.order_manager.close_credit_spread(
                    short_contract=pos.short_contract,
                    long_contract=pos.long_contract,
                    quantity=pos.num_contracts,
                    buy_back_price=pos.current_spread_value,
                    reason=reason,
                )
            except Exception as e:
                logger.error(f"Error closing spread: {e}")
                try:
                    await self.ibkr.close_all_positions()
                except Exception:
                    pass

        self.trade_log.append({
            "ticker": pos.ticker,
            "spread_type": pos.spread_type,
            "short_strike": pos.short_strike,
            "long_strike": pos.long_strike,
            "right": pos.right,
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": self._now_et().isoformat(),
            "credit_received": pos.credit_received,
            "exit_spread_value": pos.current_spread_value,
            "contracts": pos.num_contracts,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "exit_reason": reason.value,
            "hold_minutes": hold_mins,
            "strength": pos.signal_strength,
            "iv_percentile": pos.iv_percentile,
            "sentiment": pos.sentiment_at_entry,
        })

        self.daily_pnl += pnl_dollars
        self.trades_today += 1

        # Cooldown
        if pnl_dollars < 0:
            cooldown = self.cs_config.cooldown_after_loss_sec
        else:
            cooldown = self.cs_config.cooldown_after_trade_sec

        self.cooldown_until = self._now_et() + timedelta(seconds=cooldown)

        # Circuit breaker
        if self.daily_pnl / self.cs_config.account_value <= -self.cs_config.daily_loss_limit_pct:
            self.circuit_breaker = True
            logger.warning(f"CIRCUIT BREAKER: daily P&L = ${self.daily_pnl:+,.2f}")

        # Remove from active positions
        if pos in self.positions:
            self.positions.remove(pos)

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now_et() < self.cooldown_until

    async def _force_close_eod(self):
        """Force close all positions at EOD."""
        for pos in list(self.positions):
            logger.warning(f"EOD: Force closing {pos.spread_type} spread")
            await self._close_position(pos, ExitReason.EOD_CLOSE)

        if not self.dry_run:
            try:
                await self.ibkr.cancel_all_orders()
            except Exception:
                pass

        self._print_daily_summary()

    def _print_daily_summary(self):
        """Print end-of-day summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  BOT 4 CREDIT SPREADS — Daily Summary")
        logger.info("=" * 50)
        logger.info(f"Trades: {len(self.trade_log)}")
        logger.info(f"Daily P&L: ${self.daily_pnl:+,.2f}")

        if self.trade_log:
            wins = [t for t in self.trade_log if t["pnl_dollars"] > 0]
            logger.info(f"Win rate: {len(wins)}/{len(self.trade_log)}")
            for t in self.trade_log:
                logger.info(
                    f"  {t['spread_type']} {t['short_strike']}/{t['long_strike']}"
                    f"{t['right']} P&L: ${t['pnl_dollars']:+,.2f} "
                    f"[{t['exit_reason']}] hold={t['hold_minutes']:.1f}m"
                )
        logger.info("=" * 50)

    def _setup_signals(self):
        """Signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.warning(f"Signal {signum} received — shutting down")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def _check_kill_switch(self):
        """Check for kill switch."""
        if self.shutdown_requested:
            self.running = False
            return

        kill_file = "KILL_CREDIT_SPREAD"
        if os.path.exists(kill_file):
            logger.warning(f"Kill file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)

        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Bot 4...")
        self.running = False

        try:
            if self.positions and not self.dry_run:
                logger.warning("Emergency: closing all open spreads")
                await self.ibkr.close_all_positions()

            if not self.dry_run:
                await self.ibkr.cancel_all_orders()

            await self.ibkr.disconnect()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        logger.info("Bot 4 shutdown complete.")


# ============================================================
# Main
# ============================================================

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/credit_spread_{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bot 4 — 0DTE Credit Spread Seller — Paper Trading"
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal generation only, no orders")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    bot = CreditSpreadBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
