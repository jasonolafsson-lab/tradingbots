"""
Opening Range Breakout Bot — Live Paper Trading Runner ("Bot 5")

Trades the explosive first 15-30 minutes after the opening range establishes.
Measures the 9:30-9:45 range, then buys calls on breakout above OR high,
or puts on breakdown below OR low.

Target: 8-12% monthly returns via fast momentum captures.

Usage:
    python orb_live.py                  # Normal mode (paper trading)
    python orb_live.py --dry-run        # Signal generation only, no orders
    python orb_live.py --config alt.yaml

Architecture:
    - Reuses Bot 1's IBKR client, order manager, and indicator modules
    - Self-contained opening range tracking (no dependency on Bot 1)
    - Trades SPY and QQQ (calls + puts based on breakout direction)
    - Client ID 5 to avoid conflicts with Bots 1-4
    - Entry window: 9:45-10:15 AM only (30 min of intense action)
    - Quick exits: 10-20 min holds, tight stops, aggressive TPs
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
from data.finnhub_feed import FinnhubFeed
from data.market_state import (
    Bar, TickerState, MarketState, Signal, Position,
    Regime, Direction, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.volume_profile import VolumeProfileCalculator
from execution.order_manager import OrderManager
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window

logger = logging.getLogger("orb_live")
ET = ZoneInfo("US/Eastern")

PROJECT_ROOT = Path(__file__).parent


# ============================================================
# Opening Range Tracker (self-contained for Bot 5)
# ============================================================

class ORBRangeTracker:
    """Track the opening range (9:30-9:45) for each ticker."""

    def __init__(self, duration_minutes: int = 15):
        self.duration = duration_minutes
        self.or_start = dtime(9, 30)
        self.or_end_dt = datetime.combine(datetime.today(),
                                          dtime(9, 30)) + timedelta(minutes=duration_minutes)
        self.or_end = self.or_end_dt.time()
        self._ranges: Dict[str, Dict] = {}

    def update(self, ticker: str, bars_1m: List[Bar]) -> None:
        """Update opening range from 1-minute bars."""
        if ticker in self._ranges and self._ranges[ticker].get("locked"):
            return

        or_bars = []
        for bar in bars_1m:
            bar_time = bar.timestamp.time() if hasattr(bar.timestamp, 'time') else bar.timestamp
            if isinstance(bar_time, datetime):
                bar_time = bar_time.time()
            if self.or_start <= bar_time < self.or_end:
                or_bars.append(bar)

        if not or_bars:
            return

        high = max(b.high for b in or_bars)
        low = min(b.low for b in or_bars)
        total_vol = sum(b.volume for b in or_bars)
        avg_vol = total_vol / len(or_bars) if or_bars else 0

        self._ranges[ticker] = {
            "high": high,
            "low": low,
            "range_width": high - low,
            "range_pct": (high - low) / low if low > 0 else 0,
            "avg_volume": avg_vol,
            "bar_count": len(or_bars),
            "locked": len(or_bars) >= self.duration,  # Lock after 15 bars
        }

    def get_range(self, ticker: str) -> Optional[Dict]:
        """Get the opening range for a ticker."""
        return self._ranges.get(ticker)

    def is_ready(self, ticker: str) -> bool:
        """Check if the opening range is established."""
        r = self._ranges.get(ticker)
        return r is not None and r.get("locked", False)


# ============================================================
# Config
# ============================================================

class ORBConfig:
    """Live trading config for the ORB bot."""

    def __init__(self, config: dict):
        orb = config.get("orb_bot", {})

        # Account
        self.account_value = orb.get("account_value", 25_000)
        self.risk_per_trade_pct = orb.get("risk_per_trade_pct", 0.05)
        self.max_contracts = orb.get("max_contracts", 15)

        # Schedule
        self.or_start = dtime(9, 30)
        self.or_end = dtime(9, 45)
        self.earliest_entry = dtime(9, 45)       # After OR establishes
        self.last_entry = dtime(10, 15)           # 30-min entry window
        self.force_close = dtime(10, 30)          # Hard close — no holding past 10:30

        # Tickers
        self.tickers = orb.get("tickers", ["SPY", "QQQ"])

        # Opening range parameters
        self.or_duration_minutes = orb.get("or_duration_minutes", 15)
        self.breakout_threshold_pct = orb.get("breakout_threshold_pct", 0.001)
        self.volume_surge_min = orb.get("volume_surge_min", 1.3)
        self.min_strength = orb.get("min_strength", 40)

        # Contract selection
        self.delta_target = orb.get("delta_target", 0.40)

        # Exit rules
        self.stop_loss_pct = orb.get("stop_loss_pct", 0.20)
        self.take_profit_pct = orb.get("take_profit_pct", 0.50)
        self.trail_activation_pct = orb.get("trail_activation_pct", 0.20)
        self.trail_distance_pct = orb.get("trail_distance_pct", 0.10)
        self.max_hold_minutes = orb.get("max_hold_minutes", 20)

        # Risk
        self.max_trades_per_session = orb.get("max_trades_per_session", 3)
        self.cooldown_after_trade_sec = orb.get("cooldown_after_trade_sec", 60)
        self.daily_loss_limit_pct = orb.get("daily_loss_limit_pct", 0.03)


# ============================================================
# Signal Generation
# ============================================================

def check_orb_signal(
    ticker: str,
    ts: TickerState,
    or_data: Dict,
    config: ORBConfig,
    sentiment: str = "NEUTRAL",
) -> Optional[Signal]:
    """
    Opening Range Breakout signal.
    Buys calls on breakout above OR high, puts on breakdown below OR low.
    """
    price = ts.last_price
    if price <= 0 or or_data is None:
        return None

    or_high = or_data["high"]
    or_low = or_data["low"]
    or_range = or_data["range_width"]

    if or_range <= 0:
        return None

    # Check for breakout above OR high
    breakout_above = price > or_high * (1 + config.breakout_threshold_pct)
    # Check for breakdown below OR low
    breakdown_below = price < or_low * (1 - config.breakout_threshold_pct)

    if not breakout_above and not breakdown_below:
        return None

    direction = Direction.CALL if breakout_above else Direction.PUT

    # Volume confirmation
    if ts.volume_ratio < config.volume_surge_min:
        return None

    # VWAP confirmation
    if direction == Direction.CALL and price < ts.vwap:
        return None  # Breakout above but below VWAP — weak
    if direction == Direction.PUT and price > ts.vwap:
        return None  # Breakdown below but above VWAP — weak

    # Compute strength score
    strength = compute_orb_strength(
        price, ts, or_data, direction, config, sentiment
    )

    if strength < config.min_strength:
        return None

    return Signal(
        ticker=ticker,
        direction=direction,
        strategy="ORB_BREAKOUT",
        regime=Regime.MOMENTUM,
        strength_score=strength,
        entry_price_target=price,
        timestamp=datetime.now(ET),
    )


def compute_orb_strength(
    price: float,
    ts: TickerState,
    or_data: Dict,
    direction: Direction,
    config: ORBConfig,
    sentiment: str,
) -> float:
    """Compute ORB signal strength 0-100."""
    score = 0.0
    or_high = or_data["high"]
    or_low = or_data["low"]
    or_range = or_data["range_width"]

    # 1. Breakout magnitude — how far past OR (max 25)
    if direction == Direction.CALL:
        break_dist = (price - or_high) / or_high
    else:
        break_dist = (or_low - price) / or_low

    break_score = min(break_dist / 0.005, 1.0) * 25  # Full score at 0.5% breakout
    score += max(break_score, 0)

    # 2. Volume surge (max 25)
    vol_ratio = ts.volume_ratio
    if vol_ratio >= 2.0:
        score += 25
    elif vol_ratio >= 1.5:
        score += 20
    elif vol_ratio >= config.volume_surge_min:
        score += 15

    # 3. VWAP confirmation + slope (max 15)
    if direction == Direction.CALL:
        if price > ts.vwap and ts.vwap_slope > 0:
            score += 15
        elif price > ts.vwap:
            score += 8
    else:
        if price < ts.vwap and ts.vwap_slope < 0:
            score += 15
        elif price < ts.vwap:
            score += 8

    # 4. OR range width — wider = stronger breakout (max 15)
    range_pct = or_data.get("range_pct", 0)
    if range_pct > 0.005:      # > 0.5% range
        score += 15
    elif range_pct > 0.003:    # > 0.3%
        score += 10
    elif range_pct > 0.001:    # > 0.1%
        score += 5

    # 5. Sentiment alignment (max 10)
    if direction == Direction.CALL and sentiment == "BULLISH":
        score += 10
    elif direction == Direction.PUT and sentiment == "BEARISH":
        score += 10
    elif sentiment == "NEUTRAL":
        score += 5

    # 6. ADX trending confirmation (max 10)
    if ts.adx_14 > 25:
        score += 10
    elif ts.adx_14 > 20:
        score += 7
    elif ts.adx_14 > 15:
        score += 3

    return min(score, 100.0)


# ============================================================
# Position Sizing
# ============================================================

def size_orb_position(
    premium: float,
    config: ORBConfig,
    account_value: float,
    size_multiplier: float = 1.0,
) -> int:
    """Aggressive position sizing for ORB (fast trades, tight stops)."""
    if premium <= 0:
        return 0

    risk_per_contract = premium * 100 * config.stop_loss_pct
    if risk_per_contract <= 0:
        return 0

    max_risk = account_value * config.risk_per_trade_pct
    contracts = math.floor(max_risk / risk_per_contract)

    if size_multiplier != 1.0:
        contracts = max(1, int(contracts * size_multiplier))

    contracts = min(contracts, config.max_contracts)
    return max(contracts, 1) if contracts > 0 else 0


# ============================================================
# Live ORB Bot
# ============================================================

class ORBBot:
    """Live paper trading bot for Opening Range Breakout (Bot 5)."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.orb_config = ORBConfig(config)
        self.running = False
        self.shutdown_requested = False

        # IBKR connection — client_id 5
        config_copy = dict(config)
        config_copy["ibkr"] = dict(config["ibkr"])
        config_copy["ibkr"]["client_id"] = 5
        self.ibkr = IBKRClient(config_copy)

        # Execution
        self.order_manager = OrderManager(config, self.ibkr)

        # Indicators
        self.vwap_calcs: Dict[str, VWAPCalculator] = {}
        self.rsi_calcs: Dict[str, RSICalculator] = {}
        self.adx_calcs: Dict[str, ADXCalculator] = {}
        self.vol_calcs: Dict[str, VolumeProfileCalculator] = {}

        for ticker in self.orb_config.tickers:
            self.vwap_calcs[ticker] = VWAPCalculator()
            self.rsi_calcs[ticker] = RSICalculator(period=7)
            self.adx_calcs[ticker] = ADXCalculator(period=14)
            self.vol_calcs[ticker] = VolumeProfileCalculator(lookback=20)

        # Opening range tracker
        self.or_tracker = ORBRangeTracker(self.orb_config.or_duration_minutes)

        # Market state
        self.market_state = MarketState()

        # Position tracking (one position per ticker)
        self.positions: Dict[str, Position] = {}
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.cooldown_until: Optional[datetime] = None
        self.circuit_breaker: bool = False

        # Morning brief
        self.size_multiplier: float = 1.0
        self.sentiment: str = "NEUTRAL"

        # Trade log
        self.trade_log: List[dict] = []

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def _time_et(self) -> dtime:
        return self._now_et().time()

    async def start(self):
        """Main entry point."""
        self._setup_signals()

        tickers_str = ", ".join(self.orb_config.tickers)
        logger.info("=" * 60)
        logger.info("  BOT 5 — Opening Range Breakout (ORB)")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Tickers: {tickers_str}")
        logger.info(f"Opening range: {self.orb_config.or_start}-{self.orb_config.or_end} ET")
        logger.info(f"Entry window: {self.orb_config.earliest_entry}-{self.orb_config.last_entry} ET")
        logger.info(f"Force close: {self.orb_config.force_close} ET")
        logger.info(f"Breakout threshold: {self.orb_config.breakout_threshold_pct:.2%}")
        logger.info(f"Volume surge: >= {self.orb_config.volume_surge_min}x")
        logger.info(f"SL: -{self.orb_config.stop_loss_pct:.0%} | TP: +{self.orb_config.take_profit_pct:.0%}")
        logger.info(f"Trail: +{self.orb_config.trail_activation_pct:.0%} / "
                    f"{self.orb_config.trail_distance_pct:.0%}")
        logger.info(f"Max hold: {self.orb_config.max_hold_minutes} min")
        logger.info(f"Max trades: {self.orb_config.max_trades_per_session}")
        logger.info(f"Risk/trade: {self.orb_config.risk_per_trade_pct:.0%}")

        try:
            logger.info("Connecting to IBKR...")
            await self.ibkr.connect()
            logger.info("IBKR connected (client_id=5).")

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
        """Main loop — collect OR, then trade breakouts."""

        # Start Finnhub price feed (replaces IBKR real-time bars to avoid Error 420)
        self.finnhub_feed = FinnhubFeed(self.market_state, tickers=self.orb_config.tickers)
        await self.finnhub_feed.start()
        for ticker in self.orb_config.tickers:
            logger.info(f"Finnhub feed started for {ticker}")

        # Load morning brief
        brief = load_daily_brief()
        if brief:
            self.size_multiplier = brief.bot5_orb.get("size_multiplier", 1.0)
            if not brief.bot5_orb.get("enabled", True):
                logger.warning(f"Morning brief says Bot 5 DISABLED: "
                             f"{brief.bot5_orb.get('notes', '')}")
                return

            self.sentiment = brief.market_sentiment
            logger.info(f"Morning brief: sentiment={self.sentiment}, "
                       f"risk={brief.risk_level}, size={self.size_multiplier}x")
            if brief.bot5_orb.get("notes"):
                logger.info(f"  Notes: {brief.bot5_orb['notes']}")
            if brief.no_trade_windows:
                for w in brief.no_trade_windows:
                    logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")
        else:
            self.size_multiplier = 1.0

        # Phase 1: Collect opening range (9:30-9:45)
        logger.info("Collecting opening range (9:30-9:45 ET)...")

        while self.running and self._time_et() < self.orb_config.or_end:
            await self._check_kill_switch()
            await self._update_indicators()

            # Update OR tracker
            for ticker in self.orb_config.tickers:
                ts = self.market_state.get_ticker(ticker)
                self.or_tracker.update(ticker, ts.bars_1m)

            await asyncio.sleep(3)  # Fast polling during OR collection

        # Log established ranges
        all_ready = True
        for ticker in self.orb_config.tickers:
            or_data = self.or_tracker.get_range(ticker)
            if or_data and self.or_tracker.is_ready(ticker):
                logger.info(
                    f"OR {ticker}: high=${or_data['high']:.2f} low=${or_data['low']:.2f} "
                    f"range=${or_data['range_width']:.2f} ({or_data['range_pct']:.2%}) "
                    f"avg_vol={or_data['avg_volume']:.0f}"
                )
            else:
                logger.warning(f"OR {ticker}: NOT READY (bars={or_data['bar_count'] if or_data else 0})")
                all_ready = False

        if not all_ready:
            logger.warning("Not all opening ranges established — continuing with available")

        # Phase 2: Trade breakouts (9:45-10:30)
        logger.info("Entry window open — scanning for OR breakouts...")

        while self.running and self._time_et() < self.orb_config.force_close:
            await self._check_kill_switch()
            await self._update_indicators()

            if self.circuit_breaker:
                logger.warning("Circuit breaker active — no new trades")
                for ticker in list(self.positions.keys()):
                    await self._manage_position(ticker)
                await asyncio.sleep(5)
                continue

            # Manage existing positions
            for ticker in list(self.positions.keys()):
                await self._manage_position(ticker)

            # Look for new entries (only during entry window)
            if (self._time_et() <= self.orb_config.last_entry and
                    self.trades_today < self.orb_config.max_trades_per_session and
                    not self._in_cooldown() and
                    not is_in_no_trade_window(self._time_et())):

                for ticker in self.orb_config.tickers:
                    if ticker in self.positions:
                        continue  # Already have a position in this ticker
                    if not self.or_tracker.is_ready(ticker):
                        continue

                    await self._scan_ticker(ticker)

            await asyncio.sleep(3)  # 3-second loop — fast for breakout trading

        # EOD
        await self._force_close_all()

    async def _update_indicators(self):
        """Update indicators for all tickers."""
        for ticker in self.orb_config.tickers:
            ts = self.market_state.get_ticker(ticker)

            if ts.bars_1m and len(ts.bars_1m) >= 2:
                vwap_data = self.vwap_calcs[ticker].calculate(ts.bars_1m)
                ts.vwap = vwap_data["vwap"]
                ts.vwap_upper_band = vwap_data["upper_band"]
                ts.vwap_lower_band = vwap_data["lower_band"]
                ts.vwap_slope = vwap_data["slope"]

            if ts.bars_3m:
                closes = [b.close for b in ts.bars_3m]
                highs = [b.high for b in ts.bars_3m]
                lows = [b.low for b in ts.bars_3m]
                volumes = [b.volume for b in ts.bars_3m]
                ts.rsi_7 = self.rsi_calcs[ticker].calculate(closes)
                ts.adx_14 = self.adx_calcs[ticker].calculate(highs, lows, closes)
                ts.volume_ratio = self.vol_calcs[ticker].calculate(volumes)

    async def _scan_ticker(self, ticker: str):
        """Check a single ticker for ORB signal."""
        ts = self.market_state.get_ticker(ticker)
        or_data = self.or_tracker.get_range(ticker)

        sig = check_orb_signal(ticker, ts, or_data, self.orb_config, self.sentiment)
        if sig is None:
            return

        logger.info(
            f"SIGNAL: {ticker} {sig.direction.value} (OR breakout) "
            f"strength={sig.strength_score:.0f} "
            f"price=${ts.last_price:.2f} "
            f"OR=[${or_data['high']:.2f}/{or_data['low']:.2f}] "
            f"vol={ts.volume_ratio:.1f}x"
        )

        if not self.dry_run:
            await self._execute_signal(sig, ticker)
        else:
            logger.info(f"  [DRY RUN] Would buy {ticker} {sig.direction.value}")

    async def _execute_signal(self, signal: Signal, ticker: str):
        """Execute: select 0DTE option, size, submit order."""
        ts = self.market_state.get_ticker(ticker)

        try:
            # 1. Select 0DTE contract
            right = "C" if signal.direction == Direction.CALL else "P"
            contract_info = await self._select_0dte_option(ticker, right, ts)
            if contract_info is None:
                logger.warning(f"No suitable 0DTE {ticker} {right} found")
                return

            # 2. Size position
            account_value = self.orb_config.account_value
            try:
                if not self.dry_run:
                    live_value = await self.ibkr.get_net_liquidation()
                    if live_value and live_value > 0:
                        account_value = live_value
            except Exception:
                pass

            premium = contract_info["mid"]
            contracts = size_orb_position(
                premium, self.orb_config, account_value, self.size_multiplier
            )

            if contracts == 0:
                logger.warning("Position size = 0, skipping")
                return

            logger.info(
                f"ENTRY: BUY {contracts}x {ticker} "
                f"{contract_info['strike']}{right} "
                f"@ ${premium:.2f} (delta={contract_info['delta']:.2f})"
            )

            # 3. Submit order
            trade = await self.order_manager.submit_entry(
                contract_info=contract_info,
                quantity=contracts,
                signal=signal,
            )

            # 4. Wait for fill (tight timeout — market is fast)
            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=8)
            if not filled:
                filled = await self.order_manager.reprice(trade, max_attempts=1)
                if not filled:
                    await self.ibkr.cancel_order(trade)
                    logger.warning("Order not filled, cancelled")
                    return

            # 5. Build position
            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else premium
            position = Position(
                ticker=ticker,
                direction=signal.direction,
                contract_type=ContractType.SINGLE_LEG,
                strategy="ORB_BREAKOUT",
                regime=Regime.MOMENTUM.value,
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

            self.positions[ticker] = position
            logger.info(f"FILLED: {ticker} {signal.direction.value} x{contracts} @ ${fill_price:.2f}")

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _select_0dte_option(
        self, ticker: str, right: str, ts: TickerState
    ) -> Optional[dict]:
        """Select a 0DTE option for the ORB trade."""
        try:
            chains = await self.ibkr.get_option_chains(ticker)
        except Exception as e:
            logger.error(f"Option chain error for {ticker}: {e}")
            return None

        if not chains:
            return None

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

        all_strikes = set()
        for chain in chains:
            all_strikes.update(chain.get("strikes", []))
        strikes = sorted(all_strikes)

        price = ts.last_price
        if price <= 0:
            return None

        nearest = sorted(strikes, key=lambda s: abs(s - price))[:10]
        best = None
        target_delta = self.orb_config.delta_target

        for strike in nearest:
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
            mid = (bid + ask) / 2.0
            if mid < 0.15:
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
                        ticker, target_expiry, strike, right
                    )
                    ibkr_contract = await self.ibkr.qualify_contract(ibkr_contract)
                except Exception:
                    continue

                best = {
                    "contract_type": ContractType.SINGLE_LEG,
                    "strike": strike,
                    "expiry": target_expiry,
                    "dte": 0,
                    "right": right,
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

    async def _manage_position(self, ticker: str):
        """Monitor and manage an open position."""
        pos = self.positions.get(ticker)
        if pos is None:
            return

        now = self._now_et()
        hold_mins = (now - pos.entry_time).total_seconds() / 60.0
        right = "C" if pos.direction == Direction.CALL else "P"

        # Update current price
        if not self.dry_run:
            try:
                greeks = await self.ibkr.get_option_greeks(
                    pos.ticker, pos.expiry, pos.strike, right
                )
                bid = greeks.get("bid", 0) or 0
                ask = greeks.get("ask", 0) or 0
                if bid > 0 and ask > 0:
                    pos.current_price = (bid + ask) / 2.0
            except Exception as e:
                logger.debug(f"Price update error for {ticker}: {e}")

        if pos.entry_price > 0:
            pos.unrealized_pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
        pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
        pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

        # Trailing stop management
        if (not pos.trailing_active and
                pos.unrealized_pnl_pct >= self.orb_config.trail_activation_pct):
            pos.trailing_active = True
            pos.trailing_peak = pos.current_price
            logger.info(f"Trail activated for {ticker} at +{pos.unrealized_pnl_pct:.1%}")
        elif pos.trailing_active and pos.current_price > pos.trailing_peak:
            pos.trailing_peak = pos.current_price

        # Check exits (priority order)
        exit_reason = None

        # 1. Force close at window end
        if self._time_et() >= self.orb_config.force_close:
            exit_reason = ExitReason.EOD_CLOSE

        # 2. Stop loss
        elif pos.unrealized_pnl_pct <= -self.orb_config.stop_loss_pct:
            exit_reason = ExitReason.STOP_LOSS

        # 3. Time stop
        elif hold_mins >= self.orb_config.max_hold_minutes:
            exit_reason = ExitReason.TIME_STOP

        # 4. Trailing stop
        elif pos.trailing_active and pos.trailing_peak > 0:
            drop = (pos.current_price - pos.trailing_peak) / pos.trailing_peak
            if drop <= -self.orb_config.trail_distance_pct:
                exit_reason = ExitReason.TRAILING_STOP

        # 5. Take profit
        elif pos.unrealized_pnl_pct >= self.orb_config.take_profit_pct:
            exit_reason = ExitReason.TAKE_PROFIT

        if exit_reason is not None:
            await self._close_position(ticker, exit_reason)

    async def _close_position(self, ticker: str, reason: ExitReason):
        """Close position and record trade."""
        pos = self.positions.get(ticker)
        if pos is None:
            return

        pnl_pct = pos.unrealized_pnl_pct
        pnl_dollars = (pos.current_price - pos.entry_price) * 100 * pos.num_contracts
        hold_mins = (self._now_et() - pos.entry_time).total_seconds() / 60.0

        emoji = "+" if pnl_pct > 0 else "-"
        logger.info(
            f"EXIT [{emoji}]: {ticker} {pos.direction.value} ({reason.value}) "
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
            "ticker": ticker,
            "direction": pos.direction.value,
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
        self.cooldown_until = self._now_et() + timedelta(
            seconds=self.orb_config.cooldown_after_trade_sec
        )

        # Circuit breaker
        if self.daily_pnl / self.orb_config.account_value <= -self.orb_config.daily_loss_limit_pct:
            self.circuit_breaker = True
            logger.warning(f"CIRCUIT BREAKER: daily P&L = ${self.daily_pnl:+,.2f}")

        del self.positions[ticker]

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now_et() < self.cooldown_until

    async def _force_close_all(self):
        """Force close all positions at window end."""
        for ticker in list(self.positions.keys()):
            logger.warning(f"Window close: Force closing {ticker}")
            await self._close_position(ticker, ExitReason.EOD_CLOSE)

        if not self.dry_run:
            try:
                await self.ibkr.cancel_all_orders()
            except Exception:
                pass

        self._print_daily_summary()

    def _print_daily_summary(self):
        """Print session summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  BOT 5 ORB — Session Summary")
        logger.info("=" * 50)
        logger.info(f"Trades: {len(self.trade_log)}")
        logger.info(f"Session P&L: ${self.daily_pnl:+,.2f}")

        if self.trade_log:
            wins = [t for t in self.trade_log if t["pnl_dollars"] > 0]
            logger.info(f"Win rate: {len(wins)}/{len(self.trade_log)}")
            for t in self.trade_log:
                logger.info(
                    f"  {t['ticker']} {t['direction']} "
                    f"P&L: ${t['pnl_dollars']:+,.2f} ({t['pnl_pct']:+.1%}) "
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

        kill_file = "KILL_ORB"
        if os.path.exists(kill_file):
            logger.warning(f"Kill file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)

        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Bot 5...")
        self.running = False

        try:
            if self.positions and not self.dry_run:
                logger.warning("Emergency: closing all open positions")
                await self.ibkr.close_all_positions()

            if not self.dry_run:
                await self.ibkr.cancel_all_orders()

            await self.ibkr.disconnect()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        logger.info("Bot 5 shutdown complete.")


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
            logging.FileHandler(f"{log_dir}/orb_{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bot 5 — Opening Range Breakout (ORB) — Paper Trading"
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal generation only, no orders")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    bot = ORBBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
