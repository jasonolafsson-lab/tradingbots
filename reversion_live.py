"""
0DTE Mean Reversion Bot — Live Paper Trading Runner ("Bot 3")

Buy-the-dip strategy: buys SPY 0DTE calls when price is oversold
(below Bollinger Bands + RSI extreme + VWAP deviation).

Backtested: +31.3% over 6 months, PF 1.31, 2.6 trades/day.

Usage:
    python reversion_live.py                  # Normal mode (paper trading)
    python reversion_live.py --dry-run        # Signal generation only, no orders
    python reversion_live.py --config alt.yaml

Architecture:
    - Reuses Bot 1's IBKR client and order manager
    - Own signal logic (Bollinger Band + RSI + VWAP mean reversion)
    - Calls-only (buy-the-dip)
    - Client ID 3 to avoid conflicts with Bot 1 (id=1) and Bot 2 (id=2)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import signal
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import yaml
from zoneinfo import ZoneInfo

from data.ibkr_client import IBKRClient, IBKRConnectionError, IBKRAccountError
from data.market_state import (
    Bar, TickerState, MarketState, Signal, Position,
    Regime, Direction, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.volume_profile import VolumeProfileCalculator
from execution.order_manager import OrderManager
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window, get_bot_size_multiplier

logger = logging.getLogger("reversion_live")
ET = ZoneInfo("US/Eastern")

PROJECT_ROOT = Path(__file__).parent


# ============================================================
# Bollinger Band Calculator
# ============================================================

class BollingerBandCalculator:
    """Compute Bollinger Bands from closing prices."""

    def __init__(self, period: int = 20, num_std: float = 1.8):
        self.period = period
        self.num_std = num_std

    def calculate(self, closes: List[float]) -> Dict[str, float]:
        if len(closes) < self.period:
            return {"middle": 0, "upper": 0, "lower": 0, "bandwidth": 0, "pct_b": 0.5}

        window = closes[-self.period:]
        middle = np.mean(window)
        std = np.std(window, ddof=1)

        upper = middle + self.num_std * std
        lower = middle - self.num_std * std
        bandwidth = (upper - lower) / middle if middle > 0 else 0

        price = closes[-1]
        band_range = upper - lower
        pct_b = (price - lower) / band_range if band_range > 0 else 0.5

        return {
            "middle": float(middle),
            "upper": float(upper),
            "lower": float(lower),
            "bandwidth": float(bandwidth),
            "pct_b": float(pct_b),
        }


# ============================================================
# Live Config (v6 LOCKED parameters)
# ============================================================

class ReversionLiveConfig:
    """Live trading config — mirrors backtested v6 parameters."""

    def __init__(self, config: dict):
        rev = config.get("reversion_bot", {})

        # Account
        self.account_value = rev.get("account_value", 25_000)
        self.risk_per_trade_pct = rev.get("risk_per_trade_pct", 0.03)
        self.max_contracts = rev.get("max_contracts", 10)

        # Schedule — trades all day (9:50 AM - 3:30 PM)
        self.earliest_entry = dtime(9, 50)
        self.last_entry = dtime(15, 30)
        self.force_close = dtime(15, 55)

        # Entry Filters (v6 LOCKED)
        self.bb_period = rev.get("bb_period", 20)
        self.bb_std = rev.get("bb_std", 1.8)
        self.rsi_oversold = rev.get("rsi_oversold", 35.0)
        self.adx_max = rev.get("adx_max", 30.0)
        self.vwap_sd_min = rev.get("vwap_sd_min", 1.0)
        self.min_strength = rev.get("min_strength", 30.0)

        # Exit Rules (v6 LOCKED)
        self.stop_loss_pct = rev.get("stop_loss_pct", 0.15)
        self.take_profit_pct = rev.get("take_profit_pct", 0.25)
        self.trail_activation_pct = rev.get("trail_activation_pct", 0.12)
        self.trail_distance_pct = rev.get("trail_distance_pct", 0.08)
        self.breakeven_activation_pct = rev.get("breakeven_activation_pct", 0.08)
        self.max_hold_minutes = rev.get("max_hold_minutes", 15)

        # Risk
        self.max_trades_per_day = rev.get("max_trades_per_day", 8)
        self.cooldown_after_trade_sec = rev.get("cooldown_after_trade_sec", 60)
        self.cooldown_after_loss_sec = rev.get("cooldown_after_loss_sec", 120)
        self.consecutive_loss_cooldown_sec = rev.get("consecutive_loss_cooldown_sec", 300)
        self.daily_loss_limit_pct = rev.get("daily_loss_limit_pct", 0.03)

        # Delta target
        self.delta_target = rev.get("delta", 0.45)


# ============================================================
# Signal Generator — Mean Reversion (Calls Only)
# ============================================================

def check_reversion_signal(
    ts: TickerState,
    bb_data: Dict[str, float],
    config: ReversionLiveConfig,
) -> Optional[Signal]:
    """
    Buy-the-dip signal: SPY oversold + BB lower band + RSI extreme.
    Returns Signal for CALL direction only.
    """
    price = ts.last_price
    if price <= 0 or bb_data["middle"] <= 0 or ts.vwap <= 0:
        return None

    pct_b = bb_data["pct_b"]

    # Must be at or below lower BB
    if pct_b > 0.05:
        return None

    # RSI must be oversold
    if ts.rsi_7 > config.rsi_oversold:
        return None

    # ADX must be below threshold (range-bound market)
    if ts.adx_14 > config.adx_max:
        return None

    # VWAP deviation: must be below VWAP by minimum SD
    vwap_range = ts.vwap_upper_band - ts.vwap
    if vwap_range <= 0:
        return None
    price_vs_vwap_sd = (price - ts.vwap) / vwap_range * 2.0
    if price_vs_vwap_sd > -config.vwap_sd_min:
        return None  # Not far enough below VWAP

    # Compute strength score
    strength = compute_strength(ts, bb_data, config)
    if strength < config.min_strength:
        return None

    return Signal(
        ticker=ts.ticker,
        direction=Direction.CALL,
        strategy="MEAN_REVERSION",
        regime=Regime.REVERSION,
        strength_score=strength,
        entry_price_target=price,
        timestamp=datetime.now(ET),
    )


def compute_strength(
    ts: TickerState,
    bb_data: Dict[str, float],
    config: ReversionLiveConfig,
) -> float:
    """Compute signal strength 0-100."""
    score = 0.0
    pct_b = bb_data["pct_b"]

    # BB extremity (max 25)
    extremity = max(0, (0.05 - pct_b) / 0.15)
    score += min(extremity, 1.0) * 25

    # RSI extremity (max 25)
    rsi_ext = max(0, (config.rsi_oversold - ts.rsi_7) / 15.0)
    score += min(rsi_ext, 1.0) * 25

    # VWAP deviation (max 20)
    vwap_range = ts.vwap_upper_band - ts.vwap
    if vwap_range > 0:
        vwap_sd = abs((ts.last_price - ts.vwap) / vwap_range * 2.0)
        vwap_bonus = min((vwap_sd - 1.0) / 1.0, 1.0) * 20
        score += max(vwap_bonus, 0)

    # Volume exhaustion (max 15)
    if ts.volume_ratio < 0.8:
        score += 15
    elif ts.volume_ratio < 1.0:
        score += 10
    elif ts.volume_ratio < 1.2:
        score += 5

    # ADX confirms range-bound (max 15)
    if ts.adx_14 < 18:
        score += 15
    elif ts.adx_14 < 22:
        score += 10
    elif ts.adx_14 < config.adx_max:
        score += 5

    return min(score, 100.0)


# ============================================================
# Position Sizing
# ============================================================

def size_reversion_position(
    premium: float,
    config: ReversionLiveConfig,
    account_value: float,
) -> int:
    """Conservative position sizing for mean reversion."""
    if premium <= 0:
        return 0

    risk_per_contract = premium * 100 * config.stop_loss_pct
    if risk_per_contract <= 0:
        return 0

    max_risk = account_value * config.risk_per_trade_pct
    contracts = math.floor(max_risk / risk_per_contract)
    contracts = min(contracts, config.max_contracts)

    return max(contracts, 1) if contracts > 0 else 0


# ============================================================
# Live Reversion Bot
# ============================================================

class ReversionBot:
    """Live paper trading bot for 0DTE Mean Reversion (Bot 3)."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.rev_config = ReversionLiveConfig(config)
        self.running = False
        self.shutdown_requested = False

        # IBKR connection — client_id 3 (Bot 1=1, Bot 2=2)
        config_copy = dict(config)
        config_copy["ibkr"] = dict(config["ibkr"])
        config_copy["ibkr"]["client_id"] = 3
        self.ibkr = IBKRClient(config_copy)

        # Execution
        self.order_manager = OrderManager(config, self.ibkr)

        # Indicators
        self.vwap_calc = VWAPCalculator()
        self.rsi_calc = RSICalculator(period=7)
        self.adx_calc = ADXCalculator(period=14)
        self.vol_calc = VolumeProfileCalculator(lookback=20)
        self.bb_calc = BollingerBandCalculator(
            period=self.rev_config.bb_period,
            num_std=self.rev_config.bb_std,
        )

        # Market state
        self.market_state = MarketState()
        self.bb_data: Dict[str, float] = {"middle": 0, "upper": 0, "lower": 0, "bandwidth": 0, "pct_b": 0.5}

        # Position tracking
        self.position: Optional[Position] = None
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None
        self.circuit_breaker: bool = False

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
        logger.info("  BOT 3 — 0DTE Mean Reversion (Buy-the-Dip)")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Ticker: SPY only (calls only)")
        logger.info(f"Window: {self.rev_config.earliest_entry} - {self.rev_config.last_entry} ET")
        logger.info(f"BB: {self.rev_config.bb_period}-period, {self.rev_config.bb_std} SD")
        logger.info(f"RSI oversold: <{self.rev_config.rsi_oversold}")
        logger.info(f"ADX max: {self.rev_config.adx_max}")
        logger.info(f"SL: -{self.rev_config.stop_loss_pct:.0%} | TP: +{self.rev_config.take_profit_pct:.0%}")
        logger.info(f"Trail: +{self.rev_config.trail_activation_pct:.0%} / {self.rev_config.trail_distance_pct:.0%}")
        logger.info(f"BE stop: +{self.rev_config.breakeven_activation_pct:.0%}")
        logger.info(f"Time stop: {self.rev_config.max_hold_minutes} min")
        logger.info(f"Delta: {self.rev_config.delta_target}")
        logger.info(f"Max trades/day: {self.rev_config.max_trades_per_day}")

        try:
            logger.info("Connecting to IBKR...")
            await self.ibkr.connect()
            logger.info("IBKR connected (client_id=3).")

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
        """Main loop — subscribe to SPY, scan for dip-buy signals all day."""

        # Subscribe to SPY streaming data
        await self.ibkr.subscribe_realtime_bars("SPY")
        logger.info("Subscribed to SPY real-time bars")

        # Wait for indicators to warm up
        logger.info("Warming up indicators (need ~60 3-min bars for BB)...")
        warm_start = self._now_et()
        while self.running and (self._now_et() - warm_start).total_seconds() < 300:
            if self._time_et() >= self.rev_config.earliest_entry:
                break
            await self._update_indicators()
            await asyncio.sleep(5)

        # If before earliest entry, wait
        now_t = self._time_et()
        if now_t < self.rev_config.earliest_entry:
            wait_mins = (
                datetime.combine(datetime.today(), self.rev_config.earliest_entry) -
                datetime.combine(datetime.today(), now_t)
            ).total_seconds() / 60
            logger.info(f"Waiting {wait_mins:.0f} min for entry window ({self.rev_config.earliest_entry} ET)...")

            while self.running and self._time_et() < self.rev_config.earliest_entry:
                await self._update_indicators()
                await self._check_kill_switch()
                await asyncio.sleep(5)

        # Load morning brief for today's adjustments
        brief = load_daily_brief()
        if brief:
            self.size_multiplier = brief.bot3_reversion.get("size_multiplier", 1.0)
            if not brief.bot3_reversion.get("enabled", True):
                logger.warning(f"Morning brief says Bot 3 DISABLED: {brief.bot3_reversion.get('notes', '')}")
                return
            logger.info(f"Morning brief loaded: sentiment={brief.market_sentiment}, "
                       f"risk={brief.risk_level}, size={self.size_multiplier}x")
            if brief.no_trade_windows:
                for w in brief.no_trade_windows:
                    logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")
        else:
            self.size_multiplier = 1.0

        logger.info("Entry window open — scanning for dip-buy signals...")

        # Main trading loop: 9:50 AM - 3:55 PM ET
        while self.running and self._time_et() < self.rev_config.force_close:
            await self._check_kill_switch()
            await self._update_indicators()

            if self.circuit_breaker:
                logger.warning("Circuit breaker active — no new trades")
                if self.position:
                    await self._manage_position()
                await asyncio.sleep(10)
                continue

            # Manage existing position
            if self.position:
                await self._manage_position()
            else:
                # Look for new entries
                if (self._time_et() <= self.rev_config.last_entry and
                        self.trades_today < self.rev_config.max_trades_per_day and
                        not self._in_cooldown() and
                        not is_in_no_trade_window(self._time_et())):
                    await self._scan_for_signal()

            await asyncio.sleep(5)  # 5-second loop

        # EOD
        await self._force_close_eod()

    async def _update_indicators(self):
        """Update all indicators from latest SPY bars."""
        ts = self.market_state.get_ticker("SPY")

        if ts.bars_1m and len(ts.bars_1m) >= 2:
            vwap_data = self.vwap_calc.calculate(ts.bars_1m)
            ts.vwap = vwap_data["vwap"]
            ts.vwap_upper_band = vwap_data["upper_band"]
            ts.vwap_lower_band = vwap_data["lower_band"]
            ts.vwap_slope = vwap_data["slope"]

        if ts.bars_3m:
            closes = [b.close for b in ts.bars_3m]
            highs = [b.high for b in ts.bars_3m]
            lows = [b.low for b in ts.bars_3m]
            volumes = [b.volume for b in ts.bars_3m]
            ts.rsi_7 = self.rsi_calc.calculate(closes)
            ts.adx_14 = self.adx_calc.calculate(highs, lows, closes)
            ts.volume_ratio = self.vol_calc.calculate(volumes)

            # Bollinger Bands on 3-min closes
            self.bb_data = self.bb_calc.calculate(closes)

    async def _scan_for_signal(self):
        """Check SPY for dip-buy signal."""
        ts = self.market_state.get_ticker("SPY")

        sig = check_reversion_signal(ts, self.bb_data, self.rev_config)
        if sig is None:
            return

        logger.info(
            f"SIGNAL: SPY CALL (dip-buy) strength={sig.strength_score:.0f} "
            f"RSI={ts.rsi_7:.1f} BB%B={self.bb_data['pct_b']:.3f} "
            f"ADX={ts.adx_14:.1f}"
        )

        if not self.dry_run:
            await self._execute_signal(sig)
        else:
            logger.info(f"  [DRY RUN] Would buy SPY call @ ${ts.last_price:.2f}")

    async def _execute_signal(self, signal: Signal):
        """Execute: select 0DTE SPY call, size, submit order."""
        ts = self.market_state.get_ticker("SPY")

        try:
            # 1. Select 0DTE contract
            contract_info = await self._select_0dte_call(ts)
            if contract_info is None:
                logger.warning("No suitable 0DTE SPY call found")
                return

            # 2. Size position
            account_value = self.rev_config.account_value
            try:
                if not self.dry_run:
                    live_value = await self.ibkr.get_net_liquidation()
                    if live_value and live_value > 0:
                        account_value = live_value
            except Exception:
                pass

            premium = contract_info["mid"]
            contracts = size_reversion_position(
                premium, self.rev_config, account_value
            )

            # Apply morning brief size adjustment
            if hasattr(self, 'size_multiplier') and self.size_multiplier != 1.0:
                original = contracts
                contracts = max(1, int(contracts * self.size_multiplier))
                if contracts != original:
                    logger.info(f"Morning brief size adj: {original} -> {contracts} contracts ({self.size_multiplier}x)")

            if contracts == 0:
                logger.warning("Position size = 0, skipping")
                return

            logger.info(
                f"ENTRY: BUY {contracts}x SPY "
                f"{contract_info['strike']}C "
                f"@ ${premium:.2f} (delta={contract_info['delta']:.2f})"
            )

            # 3. Submit order
            trade = await self.order_manager.submit_entry(
                contract_info=contract_info,
                quantity=contracts,
                signal=signal,
            )

            # 4. Wait for fill
            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=10)
            if not filled:
                filled = await self.order_manager.reprice(trade, max_attempts=1)
                if not filled:
                    await self.ibkr.cancel_order(trade)
                    logger.warning("Order not filled, cancelled")
                    return

            # 5. Build position
            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else premium
            self.position = Position(
                ticker="SPY",
                direction=Direction.CALL,
                contract_type=ContractType.SINGLE_LEG,
                strategy="MEAN_REVERSION",
                regime=Regime.REVERSION.value,
                signal=signal,
                strike=contract_info["strike"],
                expiry=contract_info["expiry"],
                dte=contract_info.get("dte", 0),
                delta_at_entry=contract_info["delta"],
                iv_at_entry=contract_info.get("iv", 0),
                entry_time=datetime.now(ET),
                entry_price=fill_price,
                num_contracts=contracts,
                current_price=fill_price,
                rsi_value=ts.rsi_7,
                adx_value=ts.adx_14,
                volume_ratio=ts.volume_ratio,
                signal_strength_score=signal.strength_score,
            )

            logger.info(
                f"FILLED: SPY CALL x{contracts} @ ${fill_price:.2f}"
            )

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _select_0dte_call(self, ts: TickerState) -> Optional[dict]:
        """Select a 0DTE SPY call targeting our delta."""
        try:
            chains = await self.ibkr.get_option_chains("SPY")
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None

        if not chains:
            return None

        today = datetime.now().strftime("%Y%m%d")
        available_expiries = set()
        for chain in chains:
            for exp in chain.get("expirations", []):
                available_expiries.add(exp)

        # 0DTE first, then 1DTE
        target_expiry = None
        if today in available_expiries:
            target_expiry = today
        else:
            future = sorted([e for e in available_expiries if e >= today])
            if future:
                target_expiry = future[0]

        if not target_expiry:
            return None

        # Get strikes near current price
        all_strikes = set()
        for chain in chains:
            all_strikes.update(chain.get("strikes", []))
        strikes = sorted(all_strikes)

        price = ts.last_price
        nearest = sorted(strikes, key=lambda s: abs(s - price))[:10]

        best = None
        target_delta = self.rev_config.delta_target

        for strike in nearest:
            try:
                greeks = await self.ibkr.get_option_greeks(
                    "SPY", target_expiry, strike, "C"
                )
            except Exception:
                continue

            delta = abs(greeks.get("delta", 0) or 0)
            bid = greeks.get("bid", 0) or 0
            ask = greeks.get("ask", 0) or 0

            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            if mid < 0.10:
                continue

            if delta < 0.25 or delta > 0.55:
                continue

            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > 0.15:
                continue

            dist = abs(delta - target_delta)
            if best is None or dist < best["_dist"]:
                try:
                    ibkr_contract = self.ibkr.make_option_contract(
                        "SPY", target_expiry, strike, "C"
                    )
                    ibkr_contract = await self.ibkr.qualify_contract(ibkr_contract)
                except Exception:
                    continue

                best = {
                    "contract_type": ContractType.SINGLE_LEG,
                    "strike": strike,
                    "expiry": target_expiry,
                    "dte": 0,
                    "right": "C",
                    "delta": delta,
                    "iv": greeks.get("impliedVol", 0) or 0,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "entry_price": mid,
                    "spread_width": None,
                    "ibkr_contract": ibkr_contract,
                    "_dist": dist,
                }

        if best:
            best.pop("_dist", None)
        return best

    async def _manage_position(self):
        """Monitor and manage open position."""
        if self.position is None:
            return

        pos = self.position
        now = self._now_et()
        hold_mins = (now - pos.entry_time).total_seconds() / 60.0

        # Update current price
        if not self.dry_run:
            try:
                greeks = await self.ibkr.get_option_greeks(
                    pos.ticker, pos.expiry, pos.strike, "C"
                )
                bid = greeks.get("bid", 0) or 0
                ask = greeks.get("ask", 0) or 0
                if bid > 0 and ask > 0:
                    pos.current_price = (bid + ask) / 2.0
            except Exception as e:
                logger.debug(f"Price update error: {e}")

        if pos.entry_price > 0:
            pos.unrealized_pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
        pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
        pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

        # Break-even stop activation
        if (not pos.breakeven_stop_active and
                pos.max_favorable >= self.rev_config.breakeven_activation_pct):
            pos.breakeven_stop_active = True
            logger.info(f"BE stop activated at +{pos.max_favorable:.1%}")

        # Trailing stop management
        if not pos.trailing_active and pos.unrealized_pnl_pct >= self.rev_config.trail_activation_pct:
            pos.trailing_active = True
            pos.trailing_peak = pos.current_price
            logger.info(f"Trail activated at +{pos.unrealized_pnl_pct:.1%}")
        elif pos.trailing_active and pos.current_price > pos.trailing_peak:
            pos.trailing_peak = pos.current_price

        # Check exits (priority order)
        exit_reason = None

        # 1. EOD force close
        if self._time_et() >= self.rev_config.force_close:
            exit_reason = ExitReason.EOD_CLOSE

        # 2. Stop loss
        elif pos.unrealized_pnl_pct <= -self.rev_config.stop_loss_pct:
            exit_reason = ExitReason.STOP_LOSS

        # 3. Break-even stop
        elif pos.breakeven_stop_active and pos.unrealized_pnl_pct <= 0.0:
            exit_reason = ExitReason.BREAKEVEN_STOP

        # 4. Time stop
        elif hold_mins >= self.rev_config.max_hold_minutes:
            exit_reason = ExitReason.TIME_STOP

        # 5. Trailing stop
        elif pos.trailing_active and pos.trailing_peak > 0:
            drop = (pos.current_price - pos.trailing_peak) / pos.trailing_peak
            if drop <= -self.rev_config.trail_distance_pct:
                exit_reason = ExitReason.TRAILING_STOP

        # 6. Take profit
        elif pos.unrealized_pnl_pct >= self.rev_config.take_profit_pct:
            exit_reason = ExitReason.TAKE_PROFIT

        if exit_reason is not None:
            await self._close_position(exit_reason)

    async def _close_position(self, reason: ExitReason):
        """Close position and record trade."""
        pos = self.position
        if pos is None:
            return

        pnl_pct = pos.unrealized_pnl_pct
        pnl_dollars = (pos.current_price - pos.entry_price) * 100 * pos.num_contracts
        hold_mins = (self._now_et() - pos.entry_time).total_seconds() / 60.0

        emoji = "+" if pnl_pct > 0 else "-"
        logger.info(
            f"EXIT [{emoji}]: SPY CALL ({reason.value}) "
            f"P&L: {pnl_pct:+.1%} (${pnl_dollars:+,.2f}) "
            f"hold={hold_mins:.1f}min"
        )

        if not self.dry_run:
            try:
                await self.order_manager.close_position(pos, reason)
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                try:
                    await self.ibkr.close_all_positions()
                except Exception:
                    pass

        self.trade_log.append({
            "ticker": "SPY",
            "direction": "CALL",
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": self._now_et().isoformat(),
            "entry_price": pos.entry_price,
            "exit_price": pos.current_price,
            "contracts": pos.num_contracts,
            "pnl_pct": pnl_pct,
            "pnl_dollars": pnl_dollars,
            "exit_reason": reason.value,
            "hold_minutes": hold_mins,
            "strength": pos.signal_strength_score,
            "rsi": pos.rsi_value,
            "adx": pos.adx_value,
        })

        self.daily_pnl += pnl_dollars
        self.trades_today += 1

        # Cooldown
        if pnl_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                cooldown = self.rev_config.consecutive_loss_cooldown_sec
            else:
                cooldown = self.rev_config.cooldown_after_loss_sec
        else:
            self.consecutive_losses = 0
            cooldown = self.rev_config.cooldown_after_trade_sec

        self.cooldown_until = self._now_et() + timedelta(seconds=cooldown)

        # Circuit breaker
        if self.daily_pnl / self.rev_config.account_value <= -self.rev_config.daily_loss_limit_pct:
            self.circuit_breaker = True
            logger.warning(f"CIRCUIT BREAKER: daily P&L = ${self.daily_pnl:+,.2f}")

        self.position = None

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now_et() < self.cooldown_until

    async def _force_close_eod(self):
        """Force close all positions at EOD."""
        if self.position:
            logger.warning("EOD: Force closing open position")
            await self._close_position(ExitReason.EOD_CLOSE)

        if not self.dry_run:
            try:
                await self.ibkr.cancel_all_orders()
            except Exception:
                pass

        self._print_daily_summary()

    def _print_daily_summary(self):
        """Print end-of-day summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  BOT 3 MEAN REVERSION — Daily Summary")
        logger.info("=" * 50)
        logger.info(f"Trades: {len(self.trade_log)}")
        logger.info(f"Daily P&L: ${self.daily_pnl:+,.2f}")

        if self.trade_log:
            wins = [t for t in self.trade_log if t["pnl_dollars"] > 0]
            logger.info(f"Win rate: {len(wins)}/{len(self.trade_log)}")
            for t in self.trade_log:
                logger.info(
                    f"  SPY CALL P&L: ${t['pnl_dollars']:+,.2f} ({t['pnl_pct']:+.1%}) "
                    f"[{t['exit_reason']}] hold={t['hold_minutes']:.1f}m "
                    f"RSI={t['rsi']:.0f}"
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

        kill_file = "KILL_REVERSION"
        if os.path.exists(kill_file):
            logger.warning(f"Kill file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)

        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Bot 3...")
        self.running = False

        try:
            if self.position and not self.dry_run:
                logger.warning("Emergency: closing open position")
                await self.ibkr.close_all_positions()

            if not self.dry_run:
                await self.ibkr.cancel_all_orders()

            await self.ibkr.disconnect()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        logger.info("Bot 3 shutdown complete.")


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
            logging.FileHandler(f"{log_dir}/reversion_{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bot 3 — 0DTE Mean Reversion (Buy-the-Dip) — Paper Trading"
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal generation only, no orders")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    bot = ReversionBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
