"""
Gamma Acceleration Scalper — Live Paper Trading Runner ("Bot 6")

Exploits the gamma explosion in 0DTE options during the last 2 hours
before expiry. As options approach expiry, small underlying price moves
create massive option price swings. Bot identifies rapid price
acceleration moments and rides the gamma wave.

Different from Bot 2 (Power Hour Scalper):
  - Bot 2 trades 3:00-3:45 PM based on momentum (3 consecutive bars)
  - Bot 6 trades 2:00-3:00 PM based on gamma acceleration (fast moves)
  - Bot 6 uses tighter stops and faster exits (5-10 min vs 22 min)
  - Bot 6 targets slightly OTM options for maximum gamma leverage

Target: 10-15% monthly returns via gamma acceleration captures.

Usage:
    python gamma_scalper_live.py                  # Normal mode (paper trading)
    python gamma_scalper_live.py --dry-run        # Signal generation only
    python gamma_scalper_live.py --config alt.yaml

Architecture:
    - Reuses Bot 1's IBKR client and order manager
    - Own signal logic (rapid price move detection + gamma estimation)
    - SPY only (highest 0DTE gamma liquidity)
    - Client ID 6 to avoid conflicts with Bots 1-5
    - Entry window: 2:00-3:00 PM only (before Bot 2's power hour)
    - Very fast exits: 5-10 min holds, tight stops
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import signal
import sys
from collections import deque
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
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window

logger = logging.getLogger("gamma_scalper")
ET = ZoneInfo("US/Eastern")

PROJECT_ROOT = Path(__file__).parent


# ============================================================
# Gamma Move Detector
# ============================================================

class GammaMoveDetector:
    """
    Detects rapid price acceleration suitable for gamma scalping.

    Tracks recent 1-minute closes and identifies moments where SPY
    moves >= threshold in a short lookback window, indicating
    gamma-fueled momentum.
    """

    def __init__(self, min_move_pct: float = 0.001, lookback_bars: int = 2):
        self.min_move_pct = min_move_pct     # 0.10% minimum move
        self.lookback_bars = lookback_bars     # Over 2 bars (2 minutes)
        self._recent_prices: deque = deque(maxlen=60)  # Last 60 1-min closes

    def update(self, price: float) -> None:
        """Add a new price point."""
        self._recent_prices.append({
            "price": price,
            "time": datetime.now(ET),
        })

    def detect_acceleration(self) -> Optional[Dict]:
        """
        Detect if a gamma acceleration move is happening.
        Returns move details or None.
        """
        if len(self._recent_prices) < self.lookback_bars + 1:
            return None

        current = self._recent_prices[-1]["price"]
        lookback = self._recent_prices[-(self.lookback_bars + 1)]["price"]

        if lookback <= 0:
            return None

        move_pct = (current - lookback) / lookback

        if abs(move_pct) < self.min_move_pct:
            return None

        # Determine direction
        direction = Direction.CALL if move_pct > 0 else Direction.PUT

        # Calculate move velocity (move per minute)
        time_diff = (self._recent_prices[-1]["time"] -
                    self._recent_prices[-(self.lookback_bars + 1)]["time"]).total_seconds()
        velocity = abs(move_pct) / (time_diff / 60) if time_diff > 0 else 0

        # Check if move is accelerating (current bar bigger than previous)
        accelerating = False
        if len(self._recent_prices) >= self.lookback_bars + 2:
            prev_move = abs(
                self._recent_prices[-2]["price"] -
                self._recent_prices[-(self.lookback_bars + 2)]["price"]
            ) / self._recent_prices[-(self.lookback_bars + 2)]["price"]
            accelerating = abs(move_pct) > prev_move * 1.1  # 10% bigger than previous

        return {
            "direction": direction,
            "move_pct": move_pct,
            "abs_move_pct": abs(move_pct),
            "velocity": velocity,
            "accelerating": accelerating,
            "current_price": current,
            "lookback_price": lookback,
        }


# ============================================================
# Config
# ============================================================

class GammaConfig:
    """Live trading config for the gamma scalper."""

    def __init__(self, config: dict):
        gc = config.get("gamma_bot", {})

        # Account
        self.account_value = gc.get("account_value", 25_000)
        self.risk_per_trade_pct = gc.get("risk_per_trade_pct", 0.04)
        self.max_contracts = gc.get("max_contracts", 15)

        # Schedule — 2:00-3:00 PM ET (before Bot 2's power hour)
        self.earliest_entry = dtime(14, 0)       # 2:00 PM ET
        self.last_entry = dtime(15, 0)           # 3:00 PM ET
        self.force_close = dtime(15, 10)         # 3:10 PM — close before Bot 2 starts

        # Ticker
        self.ticker = gc.get("ticker", "SPY")

        # Move detection
        self.min_move_pct = gc.get("min_move_pct", 0.001)          # 0.10% move
        self.min_move_lookback_bars = gc.get("min_move_lookback_bars", 2)

        # Signal
        self.min_strength = gc.get("min_strength", 40)
        self.delta_target = gc.get("delta_target", 0.35)

        # Exit rules — very fast
        self.stop_loss_pct = gc.get("stop_loss_pct", 0.15)         # -15% tight stop
        self.take_profit_pct = gc.get("take_profit_pct", 0.40)     # +40% aggressive TP
        self.trail_activation_pct = gc.get("trail_activation_pct", 0.15)
        self.trail_distance_pct = gc.get("trail_distance_pct", 0.08)
        self.max_hold_minutes = gc.get("max_hold_minutes", 10)     # 10 min max

        # Risk
        self.max_trades_per_session = gc.get("max_trades_per_session", 4)
        self.cooldown_after_trade_sec = gc.get("cooldown_after_trade_sec", 90)
        self.cooldown_after_loss_sec = gc.get("cooldown_after_loss_sec", 180)
        self.daily_loss_limit_pct = gc.get("daily_loss_limit_pct", 0.04)


# ============================================================
# Signal Generation
# ============================================================

def check_gamma_signal(
    ticker: str,
    ts: TickerState,
    move_data: Dict,
    config: GammaConfig,
    sentiment: str = "NEUTRAL",
    vix_level: float = 18.0,
) -> Optional[Signal]:
    """
    Gamma acceleration signal.
    Triggers when SPY makes a rapid move, indicating gamma-fueled momentum.
    """
    direction = move_data["direction"]

    # VWAP confirmation — move must align with VWAP
    if direction == Direction.CALL and ts.last_price < ts.vwap:
        return None  # Upward move but below VWAP — weak
    if direction == Direction.PUT and ts.last_price > ts.vwap:
        return None  # Downward move but above VWAP — weak

    # Compute strength
    strength = compute_gamma_strength(ts, move_data, config, sentiment, vix_level)

    if strength < config.min_strength:
        return None

    return Signal(
        ticker=ticker,
        direction=direction,
        strategy="GAMMA_SCALP",
        regime=Regime.MOMENTUM,
        strength_score=strength,
        entry_price_target=ts.last_price,
        timestamp=datetime.now(ET),
    )


def compute_gamma_strength(
    ts: TickerState,
    move_data: Dict,
    config: GammaConfig,
    sentiment: str,
    vix_level: float,
) -> float:
    """Compute gamma signal strength 0-100."""
    score = 0.0

    # 1. Move magnitude — bigger move = more gamma (max 25)
    move_pct = move_data["abs_move_pct"]
    move_score = min(move_pct / 0.003, 1.0) * 25  # Full score at 0.30% move
    score += max(move_score, 0)

    # 2. Move velocity — faster = more explosive (max 20)
    velocity = move_data["velocity"]
    vel_score = min(velocity / 0.002, 1.0) * 20  # Full at 0.20%/min
    score += max(vel_score, 0)

    # 3. Acceleration — move getting bigger (max 15)
    if move_data["accelerating"]:
        score += 15
    else:
        score += 5  # Still decent even without acceleration

    # 4. Volume confirmation (max 15)
    vol_ratio = ts.volume_ratio
    if vol_ratio >= 1.8:
        score += 15
    elif vol_ratio >= 1.3:
        score += 10
    elif vol_ratio >= 1.0:
        score += 5

    # 5. VIX level — higher VIX = bigger gamma (max 10)
    if vix_level > 22:
        score += 10
    elif vix_level > 18:
        score += 7
    elif vix_level > 15:
        score += 4

    # 6. Sentiment alignment (max 10)
    if move_data["direction"] == Direction.CALL and sentiment == "BULLISH":
        score += 10
    elif move_data["direction"] == Direction.PUT and sentiment == "BEARISH":
        score += 10
    elif sentiment == "NEUTRAL":
        score += 5

    # 7. VWAP slope confirmation (max 5)
    if move_data["direction"] == Direction.CALL and ts.vwap_slope > 0:
        score += 5
    elif move_data["direction"] == Direction.PUT and ts.vwap_slope < 0:
        score += 5

    return min(score, 100.0)


# ============================================================
# Position Sizing
# ============================================================

def size_gamma_position(
    premium: float,
    config: GammaConfig,
    account_value: float,
    size_multiplier: float = 1.0,
) -> int:
    """Aggressive sizing for gamma scalps (fast in/out, tight stops)."""
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
# Live Gamma Scalper Bot
# ============================================================

class GammaScalperBot:
    """Live paper trading bot for Gamma Acceleration Scalping (Bot 6)."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.gamma_config = GammaConfig(config)
        self.running = False
        self.shutdown_requested = False

        # IBKR connection — client_id 6
        config_copy = dict(config)
        config_copy["ibkr"] = dict(config["ibkr"])
        config_copy["ibkr"]["client_id"] = 6
        self.ibkr = IBKRClient(config_copy)

        # Execution
        self.order_manager = OrderManager(config, self.ibkr)

        # Indicators
        self.vwap_calc = VWAPCalculator()
        self.rsi_calc = RSICalculator(period=7)
        self.adx_calc = ADXCalculator(period=14)
        self.vol_calc = VolumeProfileCalculator(lookback=20)

        # Gamma move detector
        self.move_detector = GammaMoveDetector(
            min_move_pct=self.gamma_config.min_move_pct,
            lookback_bars=self.gamma_config.min_move_lookback_bars,
        )

        # Market state
        self.market_state = MarketState()

        # Position tracking
        self.position: Optional[Position] = None
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.cooldown_until: Optional[datetime] = None
        self.circuit_breaker: bool = False

        # Morning brief
        self.size_multiplier: float = 1.0
        self.sentiment: str = "NEUTRAL"
        self.vix_level: float = 18.0

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
        logger.info("  BOT 6 — Gamma Acceleration Scalper")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Ticker: {self.gamma_config.ticker}")
        logger.info(f"Window: {self.gamma_config.earliest_entry}-{self.gamma_config.last_entry} ET")
        logger.info(f"Force close: {self.gamma_config.force_close} ET")
        logger.info(f"Min move: {self.gamma_config.min_move_pct:.2%} in "
                    f"{self.gamma_config.min_move_lookback_bars} bars")
        logger.info(f"Delta target: {self.gamma_config.delta_target}")
        logger.info(f"SL: -{self.gamma_config.stop_loss_pct:.0%} | "
                    f"TP: +{self.gamma_config.take_profit_pct:.0%}")
        logger.info(f"Trail: +{self.gamma_config.trail_activation_pct:.0%} / "
                    f"{self.gamma_config.trail_distance_pct:.0%}")
        logger.info(f"Max hold: {self.gamma_config.max_hold_minutes} min")
        logger.info(f"Max trades: {self.gamma_config.max_trades_per_session}")

        try:
            logger.info("Connecting to IBKR...")
            await self.ibkr.connect()
            logger.info("IBKR connected (client_id=6).")

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
        """Main loop — detect gamma acceleration, ride the wave."""
        ticker = self.gamma_config.ticker

        # Subscribe to streaming data
        await self.ibkr.subscribe_realtime_bars(ticker)
        logger.info(f"Subscribed to {ticker} real-time bars")

        # Load morning brief
        brief = load_daily_brief()
        if brief:
            self.size_multiplier = brief.bot6_gamma.get("size_multiplier", 1.0)
            if not brief.bot6_gamma.get("enabled", True):
                logger.warning(f"Morning brief says Bot 6 DISABLED: "
                             f"{brief.bot6_gamma.get('notes', '')}")
                return

            self.sentiment = brief.market_sentiment
            self.vix_level = brief.vix_level

            logger.info(f"Morning brief: sentiment={self.sentiment}, "
                       f"risk={brief.risk_level}, VIX={self.vix_level:.1f}, "
                       f"size={self.size_multiplier}x")
            if brief.bot6_gamma.get("notes"):
                logger.info(f"  Notes: {brief.bot6_gamma['notes']}")
            if brief.no_trade_windows:
                for w in brief.no_trade_windows:
                    logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")
        else:
            self.size_multiplier = 1.0

        # Wait for entry window
        now_t = self._time_et()
        if now_t < self.gamma_config.earliest_entry:
            wait_mins = (
                datetime.combine(datetime.today(), self.gamma_config.earliest_entry) -
                datetime.combine(datetime.today(), now_t)
            ).total_seconds() / 60
            logger.info(f"Waiting {wait_mins:.0f} min for gamma window "
                       f"({self.gamma_config.earliest_entry} ET)...")

            # Warm up indicators while waiting
            while self.running and self._time_et() < self.gamma_config.earliest_entry:
                await self._check_kill_switch()
                await self._update_indicators()
                # Feed move detector with prices during warm-up
                ts = self.market_state.get_ticker(ticker)
                if ts.last_price > 0:
                    self.move_detector.update(ts.last_price)
                await asyncio.sleep(5)

        logger.info("Gamma window open — scanning for acceleration moves...")

        # Main trading loop: 2:00 PM - 3:10 PM
        while self.running and self._time_et() < self.gamma_config.force_close:
            await self._check_kill_switch()
            await self._update_indicators()

            # Feed move detector
            ts = self.market_state.get_ticker(ticker)
            if ts.last_price > 0:
                self.move_detector.update(ts.last_price)

            if self.circuit_breaker:
                logger.warning("Circuit breaker active — no new trades")
                if self.position:
                    await self._manage_position()
                await asyncio.sleep(5)
                continue

            # Manage existing position
            if self.position:
                await self._manage_position()
            else:
                # Look for new entries
                if (self._time_et() <= self.gamma_config.last_entry and
                        self.trades_today < self.gamma_config.max_trades_per_session and
                        not self._in_cooldown() and
                        not is_in_no_trade_window(self._time_et())):
                    await self._scan_for_signal()

            await asyncio.sleep(3)  # 3-second loop — gamma moves are fast

        # EOD
        await self._force_close_eod()

    async def _update_indicators(self):
        """Update indicators from latest bars."""
        ticker = self.gamma_config.ticker
        ts = self.market_state.get_ticker(ticker)

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

    async def _scan_for_signal(self):
        """Check for gamma acceleration signal."""
        ticker = self.gamma_config.ticker
        ts = self.market_state.get_ticker(ticker)

        move_data = self.move_detector.detect_acceleration()
        if move_data is None:
            return

        sig = check_gamma_signal(
            ticker, ts, move_data, self.gamma_config,
            self.sentiment, self.vix_level
        )
        if sig is None:
            return

        logger.info(
            f"SIGNAL: {ticker} {sig.direction.value} (gamma acceleration) "
            f"strength={sig.strength_score:.0f} "
            f"move={move_data['move_pct']:+.3%} "
            f"velocity={move_data['velocity']:.4f}%/min "
            f"{'ACCELERATING' if move_data['accelerating'] else 'steady'}"
        )

        if not self.dry_run:
            await self._execute_signal(sig)
        else:
            logger.info(f"  [DRY RUN] Would buy {ticker} {sig.direction.value}")

    async def _execute_signal(self, signal: Signal):
        """Execute: select 0DTE option, size, submit order."""
        ticker = self.gamma_config.ticker
        ts = self.market_state.get_ticker(ticker)

        try:
            right = "C" if signal.direction == Direction.CALL else "P"
            contract_info = await self._select_0dte_option(ticker, right, ts)
            if contract_info is None:
                logger.warning(f"No suitable 0DTE {ticker} {right} found")
                return

            # Size position
            account_value = self.gamma_config.account_value
            try:
                if not self.dry_run:
                    live_value = await self.ibkr.get_net_liquidation()
                    if live_value and live_value > 0:
                        account_value = live_value
            except Exception:
                pass

            premium = contract_info["mid"]
            contracts = size_gamma_position(
                premium, self.gamma_config, account_value, self.size_multiplier
            )

            if contracts == 0:
                logger.warning("Position size = 0, skipping")
                return

            logger.info(
                f"ENTRY: BUY {contracts}x {ticker} "
                f"{contract_info['strike']}{right} "
                f"@ ${premium:.2f} (delta={contract_info['delta']:.2f})"
            )

            # Submit order (tight fill timeout — speed matters)
            trade = await self.order_manager.submit_entry(
                contract_info=contract_info,
                quantity=contracts,
                signal=signal,
            )

            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=6)
            if not filled:
                filled = await self.order_manager.reprice(trade, max_attempts=1)
                if not filled:
                    await self.ibkr.cancel_order(trade)
                    logger.warning("Order not filled, cancelled")
                    return

            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else premium
            self.position = Position(
                ticker=ticker,
                direction=signal.direction,
                contract_type=ContractType.SINGLE_LEG,
                strategy="GAMMA_SCALP",
                regime=Regime.MOMENTUM.value,
                signal=signal,
                strike=contract_info["strike"],
                expiry=contract_info["expiry"],
                dte=0,
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

            logger.info(f"FILLED: {ticker} {signal.direction.value} x{contracts} @ ${fill_price:.2f}")

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _select_0dte_option(
        self, ticker: str, right: str, ts: TickerState
    ) -> Optional[dict]:
        """Select a 0DTE option — slightly OTM for max gamma leverage."""
        try:
            chains = await self.ibkr.get_option_chains(ticker)
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

        nearest = sorted(strikes, key=lambda s: abs(s - price))[:12]
        best = None
        target_delta = self.gamma_config.delta_target

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
            if mid < 0.10:
                continue

            if delta < 0.20 or delta > 0.50:
                continue

            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > 0.15:
                continue

            # Estimate gamma (higher for ATM, lower for OTM)
            gamma = greeks.get("gamma", 0) or 0

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
                    "gamma": gamma,
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
        """Monitor and manage open position — very fast exits."""
        if self.position is None:
            return

        pos = self.position
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
                logger.debug(f"Price update error: {e}")

        if pos.entry_price > 0:
            pos.unrealized_pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
        pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
        pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

        # Trailing stop management
        if (not pos.trailing_active and
                pos.unrealized_pnl_pct >= self.gamma_config.trail_activation_pct):
            pos.trailing_active = True
            pos.trailing_peak = pos.current_price
            logger.info(f"Trail activated at +{pos.unrealized_pnl_pct:.1%}")
        elif pos.trailing_active and pos.current_price > pos.trailing_peak:
            pos.trailing_peak = pos.current_price

        # Check exits (priority order)
        exit_reason = None

        # 1. Force close at window end
        if self._time_et() >= self.gamma_config.force_close:
            exit_reason = ExitReason.EOD_CLOSE

        # 2. Stop loss — tight
        elif pos.unrealized_pnl_pct <= -self.gamma_config.stop_loss_pct:
            exit_reason = ExitReason.STOP_LOSS

        # 3. Time stop — very short holds
        elif hold_mins >= self.gamma_config.max_hold_minutes:
            exit_reason = ExitReason.TIME_STOP

        # 4. Trailing stop
        elif pos.trailing_active and pos.trailing_peak > 0:
            drop = (pos.current_price - pos.trailing_peak) / pos.trailing_peak
            if drop <= -self.gamma_config.trail_distance_pct:
                exit_reason = ExitReason.TRAILING_STOP

        # 5. Take profit
        elif pos.unrealized_pnl_pct >= self.gamma_config.take_profit_pct:
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
            f"EXIT [{emoji}]: SPY {pos.direction.value} ({reason.value}) "
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
        if pnl_dollars < 0:
            cooldown = self.gamma_config.cooldown_after_loss_sec
        else:
            cooldown = self.gamma_config.cooldown_after_trade_sec

        self.cooldown_until = self._now_et() + timedelta(seconds=cooldown)

        # Circuit breaker
        if self.daily_pnl / self.gamma_config.account_value <= -self.gamma_config.daily_loss_limit_pct:
            self.circuit_breaker = True
            logger.warning(f"CIRCUIT BREAKER: daily P&L = ${self.daily_pnl:+,.2f}")

        self.position = None

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now_et() < self.cooldown_until

    async def _force_close_eod(self):
        """Force close at window end."""
        if self.position:
            logger.warning("Window close: Force closing open position")
            await self._close_position(ExitReason.EOD_CLOSE)

        if not self.dry_run:
            try:
                await self.ibkr.cancel_all_orders()
            except Exception:
                pass

        self._print_daily_summary()

    def _print_daily_summary(self):
        """Print session summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  BOT 6 GAMMA SCALPER — Session Summary")
        logger.info("=" * 50)
        logger.info(f"Trades: {len(self.trade_log)}")
        logger.info(f"Session P&L: ${self.daily_pnl:+,.2f}")

        if self.trade_log:
            wins = [t for t in self.trade_log if t["pnl_dollars"] > 0]
            logger.info(f"Win rate: {len(wins)}/{len(self.trade_log)}")
            for t in self.trade_log:
                logger.info(
                    f"  SPY {t['direction']} "
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

        kill_file = "KILL_GAMMA"
        if os.path.exists(kill_file):
            logger.warning(f"Kill file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)

        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Bot 6...")
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

        logger.info("Bot 6 shutdown complete.")


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
            logging.FileHandler(f"{log_dir}/gamma_scalper_{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bot 6 — Gamma Acceleration Scalper — Paper Trading"
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal generation only, no orders")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    bot = GammaScalperBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
