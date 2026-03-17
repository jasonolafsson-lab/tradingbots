"""
Aggressive 0DTE Scalper — Live Paper Trading Runner ("Bot 2")

Pure Power Hour momentum scalping on 0DTE options.
Trades QQQ + NVDA, 3:00-3:45 PM ET only.

Usage:
    python scalper_live.py                  # Normal mode (paper trading)
    python scalper_live.py --dry-run        # Signal generation only, no orders
    python scalper_live.py --config alt.yaml

Architecture:
    - Reuses Bot 1's IBKR client, contract selector, and order manager
    - Own signal logic (Power Hour Momentum from backtest_scalper.py)
    - Own position sizing (aggressive: 20% risk per trade)
    - Own risk management (trailing stops, time stops)
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
from execution.contract_selector import ContractSelector
from execution.order_manager import OrderManager
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window

logger = logging.getLogger("scalper_live")
ET = ZoneInfo("US/Eastern")

PROJECT_ROOT = Path(__file__).parent


# ============================================================
# Scalper Configuration (matches backtest v12a — the proven config)
# ============================================================

class ScalperLiveConfig:
    """Live trading config — mirrors the backtested parameters."""

    def __init__(self, config: dict):
        scalper = config.get("scalper", {})

        # Account
        self.account_value = scalper.get("account_value", 25_000)
        self.risk_per_trade_pct = scalper.get("risk_per_trade_pct", 0.20)
        self.max_notional_pct = scalper.get("max_notional_pct", 0.40)
        self.max_contracts = scalper.get("max_contracts", 50)

        # Schedule
        self.power_hour_start = dtime(15, 0)    # 3:00 PM ET
        self.power_hour_end = dtime(15, 45)     # 3:45 PM ET
        self.last_entry = dtime(15, 40)         # Last entry at 3:40 PM
        self.force_close = dtime(15, 55)        # Force close at 3:55 PM

        # Power Hour Momentum
        self.ph_momentum_bars = scalper.get("ph_momentum_bars", 3)
        self.ph_min_move_pct = scalper.get("ph_min_move_pct", 0.0015)
        self.ph_volume_surge = scalper.get("ph_volume_surge", 1.3)

        # Exit Rules
        self.stop_loss_pct = scalper.get("stop_loss_pct", 0.25)
        self.take_profit_pct = scalper.get("take_profit_pct", 0.75)
        self.trail_activation_pct = scalper.get("trail_activation_pct", 0.15)
        self.trail_distance_pct = scalper.get("trail_distance_pct", 0.10)
        self.max_hold_minutes = scalper.get("max_hold_minutes", 22)
        self.time_decay_exit_min = scalper.get("time_decay_exit_min", 18)

        # Signal quality
        self.max_strength = scalper.get("max_strength", 85)
        self.put_size_multiplier = scalper.get("put_size_multiplier", 0.65)

        # Trades per day
        self.max_pm_trades = scalper.get("max_pm_trades", 2)
        self.cooldown_after_trade_sec = scalper.get("cooldown_after_trade_sec", 60)
        self.cooldown_after_loss_sec = scalper.get("cooldown_after_loss_sec", 120)
        self.daily_loss_limit_pct = scalper.get("daily_loss_limit_pct", 0.05)

        # Delta target for 0DTE
        self.delta_target = scalper.get("delta", 0.40)

        # Tickers
        self.tickers = scalper.get("tickers", ["QQQ", "NVDA"])


# ============================================================
# Signal Generator (Power Hour Momentum)
# ============================================================

def check_power_hour_momentum(
    ts: TickerState,
    config: ScalperLiveConfig,
) -> Optional[Signal]:
    """
    Power Hour Momentum: 3 consecutive bars moving in the same direction
    with volume surge and VWAP alignment.
    """
    n = config.ph_momentum_bars
    if len(ts.bars_1m) < n + 5:
        return None

    recent = ts.bars_1m[-n:]

    all_green = all(b.close > b.open for b in recent)
    all_red = all(b.close < b.open for b in recent)

    if not all_green and not all_red:
        return None

    # Minimum move magnitude
    move = abs(recent[-1].close - recent[0].open)
    move_pct = move / recent[0].open if recent[0].open > 0 else 0
    if move_pct < config.ph_min_move_pct:
        return None

    # Volume surge vs session average
    if len(ts.bars_1m) > 20:
        session_avg_vol = np.mean([b.volume for b in ts.bars_1m[:-n]])
        recent_avg_vol = np.mean([b.volume for b in recent])
        if session_avg_vol > 0:
            vol_ratio = recent_avg_vol / session_avg_vol
            if vol_ratio < config.ph_volume_surge:
                return None
        else:
            vol_ratio = 1.0
    else:
        vol_ratio = 1.0

    direction = Direction.CALL if all_green else Direction.PUT

    # Must be above VWAP for calls, below for puts
    if ts.vwap > 0:
        if direction == Direction.CALL and ts.last_price < ts.vwap:
            return None
        if direction == Direction.PUT and ts.last_price > ts.vwap:
            return None

    # Strength scoring (same as backtest)
    strength = 50.0
    strength += min(move_pct / 0.005, 1.0) * 15          # Move magnitude (max +15)
    strength += min((vol_ratio - 1.0) / 1.0, 1.0) * 15   # Volume (max +15)
    if ts.adx_14 >= 20:
        strength += min((ts.adx_14 - 20) / 20.0, 1.0) * 10  # ADX (max +10)
    if direction == Direction.CALL and ts.vwap_slope > 0:
        strength += 10                                     # VWAP slope (max +10)
    elif direction == Direction.PUT and ts.vwap_slope < 0:
        strength += 10

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="POWER_HOUR",
        regime=Regime.MOMENTUM,
        strength_score=min(strength, 100),
        entry_price_target=ts.last_price,
        timestamp=datetime.now(ET),
    )


# ============================================================
# Position Sizing
# ============================================================

def size_scalp_position(
    premium: float,
    config: ScalperLiveConfig,
    account_value: float,
    direction: Direction,
) -> int:
    """Aggressive position sizing for 0DTE scalps."""
    if premium <= 0:
        return 0

    max_risk = account_value * config.risk_per_trade_pct
    max_notional = account_value * config.max_notional_pct

    risk_per_contract = premium * 100 * config.stop_loss_pct
    if risk_per_contract <= 0:
        return 0

    contracts = math.floor(max_risk / risk_per_contract)

    # Notional cap
    notional = contracts * premium * 100
    if notional > max_notional:
        contracts = math.floor(max_notional / (premium * 100))

    contracts = min(contracts, config.max_contracts)

    # Reduce PUT size
    if direction == Direction.PUT:
        contracts = max(1, int(contracts * config.put_size_multiplier))

    return max(contracts, 1) if contracts > 0 else 0


# ============================================================
# Live Scalper Bot
# ============================================================

class ScalperBot:
    """Live paper trading bot for the 0DTE Power Hour scalper."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.scalper_config = ScalperLiveConfig(config)
        self.running = False
        self.shutdown_requested = False

        # IBKR connection — use client_id 2 to avoid conflict with Bot 1
        scalper_config_copy = dict(config)
        scalper_config_copy["ibkr"] = dict(config["ibkr"])
        scalper_config_copy["ibkr"]["client_id"] = 2
        self.ibkr = IBKRClient(scalper_config_copy)

        # Execution (reuse Bot 1's infra)
        # ContractSelector needs UW client — we pass None since we don't use UW for scalper
        self.order_manager = OrderManager(config, self.ibkr)

        # Indicators
        self.vwap_calc = VWAPCalculator()
        self.rsi_calc = RSICalculator(period=7)
        self.adx_calc = ADXCalculator(period=14)
        self.vol_calc = VolumeProfileCalculator(lookback=20)

        # Market state
        self.market_state = MarketState()
        self.ticker_states: Dict[str, TickerState] = {}

        # Position tracking
        self.position: Optional[Position] = None
        self.pm_trades: int = 0
        self.daily_pnl: float = 0.0
        self.cooldown_until: Optional[datetime] = None
        self.circuit_breaker: bool = False

        # Trade log
        self.trade_log: List[dict] = []

        # Morning brief
        self.size_multiplier: float = 1.0

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def _time_et(self) -> dtime:
        return self._now_et().time()

    async def start(self):
        """Main entry point."""
        self._setup_signals()

        logger.info("=" * 60)
        logger.info("  SCALPER BOT 2 — 0DTE Power Hour Momentum")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Tickers: {self.scalper_config.tickers}")
        logger.info(f"Risk/Trade: {self.scalper_config.risk_per_trade_pct:.0%}")
        logger.info(f"Window: {self.scalper_config.power_hour_start} - {self.scalper_config.power_hour_end} ET")
        logger.info(f"Stop: -{self.scalper_config.stop_loss_pct:.0%} | TP: +{self.scalper_config.take_profit_pct:.0%}")
        logger.info(f"Trail: +{self.scalper_config.trail_activation_pct:.0%} / {self.scalper_config.trail_distance_pct:.0%}")
        logger.info(f"Max hold: {self.scalper_config.max_hold_minutes} min")

        try:
            # Connect to IBKR (always — even dry-run needs market data)
            logger.info("Connecting to IBKR...")
            await self.ibkr.connect()
            logger.info("IBKR connected.")

            # Load morning brief for today's adjustments
            brief = load_daily_brief()
            if brief:
                self.size_multiplier = brief.bot2_scalper.get("size_multiplier", 1.0)
                if not brief.bot2_scalper.get("enabled", True):
                    logger.warning(f"Morning brief says Bot 2 DISABLED: {brief.bot2_scalper.get('notes', '')}")
                    return
                logger.info(f"Morning brief: sentiment={brief.market_sentiment}, "
                           f"risk={brief.risk_level}, Bot 2 size={self.size_multiplier}x")
                if brief.no_trade_windows:
                    for w in brief.no_trade_windows:
                        logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")

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
        """Main event loop — wait for Power Hour, then scalp."""

        # Subscribe to streaming data (always — dry-run still needs bars for signals)
        for ticker in self.scalper_config.tickers:
            await self.ibkr.subscribe_realtime_bars(ticker)
            self.ticker_states[ticker] = self.market_state.get_ticker(ticker)
            logger.info(f"Subscribed to {ticker} real-time bars")

        # Wait for bars to accumulate (need context for indicators)
        logger.info("Accumulating bar data for indicators...")
        wait_start = self._now_et()
        while self.running and (self._now_et() - wait_start).total_seconds() < 300:
            # Wait up to 5 min for enough bars, but check if already in PH window
            if self._time_et() >= self.scalper_config.power_hour_start:
                break
            await self._update_indicators()
            await asyncio.sleep(5)

        # If before Power Hour, wait
        now_t = self._time_et()
        if now_t < self.scalper_config.power_hour_start:
            wait_mins = (
                datetime.combine(datetime.today(), self.scalper_config.power_hour_start) -
                datetime.combine(datetime.today(), now_t)
            ).total_seconds() / 60
            logger.info(f"Waiting {wait_mins:.0f} min for Power Hour (3:00 PM ET)...")

            while self.running and self._time_et() < self.scalper_config.power_hour_start:
                await self._update_indicators()
                await self._check_kill_switch()
                await asyncio.sleep(5)

        logger.info("🔥 POWER HOUR — scanning for momentum signals...")

        # Main scalping loop: 3:00 - 3:55 PM ET
        while self.running and self._time_et() < self.scalper_config.force_close:
            await self._check_kill_switch()
            await self._update_indicators()

            # Check circuit breaker
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
                # Look for new entries during window (respecting no-trade windows)
                if (self._time_et() <= self.scalper_config.last_entry and
                        self.pm_trades < self.scalper_config.max_pm_trades and
                        not self._in_cooldown() and
                        not is_in_no_trade_window(self._time_et())):
                    await self._scan_for_signals()

            await asyncio.sleep(3)  # 3-second loop for faster reaction than Bot 1

        # Force close EOD
        await self._force_close_eod()

    async def _update_indicators(self):
        """Update VWAP, RSI, ADX, volume ratio from latest bars."""
        for ticker in self.scalper_config.tickers:
            ts = self.market_state.get_ticker(ticker)

            if ts.bars_1m and len(ts.bars_1m) >= 2:
                # VWAP
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

    async def _scan_for_signals(self):
        """Check all tickers for Power Hour momentum signals."""
        signals: List[Signal] = []

        for ticker in self.scalper_config.tickers:
            ts = self.market_state.get_ticker(ticker)

            sig = check_power_hour_momentum(ts, self.scalper_config)
            if sig:
                signals.append(sig)

        if not signals:
            return

        # Take strongest signal
        signals.sort(key=lambda s: s.strength_score, reverse=True)
        best = signals[0]

        # Quality filter
        if best.strength_score < 40 or best.strength_score > self.scalper_config.max_strength:
            logger.debug(
                f"Signal rejected: {best.ticker} strength={best.strength_score:.0f} "
                f"(range: 40-{self.scalper_config.max_strength})"
            )
            return

        logger.info(
            f"📊 SIGNAL: {best.ticker} {best.direction.value} "
            f"via POWER_HOUR (strength={best.strength_score:.0f})"
        )

        if not self.dry_run:
            await self._execute_signal(best)
        else:
            logger.info(f"  [DRY RUN] Would execute {best.ticker} {best.direction.value}")

    async def _execute_signal(self, signal: Signal):
        """Execute a signal: select 0DTE contract, size, submit order."""
        ts = self.market_state.get_ticker(signal.ticker)

        try:
            # 1. Select 0DTE contract via IBKR
            contract_info = await self._select_0dte_contract(signal, ts)
            if contract_info is None:
                logger.warning(f"No suitable 0DTE contract for {signal.ticker}")
                return

            # 2. Size position
            account_value = self.scalper_config.account_value
            try:
                if not self.dry_run:
                    live_value = await self.ibkr.get_net_liquidation()
                    if live_value and live_value > 0:
                        account_value = live_value
            except Exception:
                pass  # Use configured value

            premium = contract_info["mid"]
            contracts = size_scalp_position(
                premium, self.scalper_config, account_value, signal.direction
            )

            # Apply morning brief size adjustment
            if self.size_multiplier != 1.0:
                original = contracts
                contracts = max(1, int(contracts * self.size_multiplier))
                if contracts != original:
                    logger.info(f"Morning brief size adj: {original} -> {contracts} contracts ({self.size_multiplier}x)")

            if contracts == 0:
                logger.warning("Position size = 0, skipping")
                return

            logger.info(
                f"📝 ENTRY: BUY {contracts}x {signal.ticker} "
                f"{contract_info['strike']}{contract_info['right']} "
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
                # Try repricing once
                filled = await self.order_manager.reprice(trade, max_attempts=1)
                if not filled:
                    await self.ibkr.cancel_order(trade)
                    logger.warning(f"Order not filled for {signal.ticker}, cancelled")
                    return

            # 5. Build position
            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else premium
            self.position = Position(
                ticker=signal.ticker,
                direction=signal.direction,
                contract_type=ContractType.SINGLE_LEG,
                strategy="POWER_HOUR",
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
                signal_strength_score=signal.strength_score,
            )

            logger.info(
                f"✅ FILLED: {self.position.ticker} {self.position.direction.value} "
                f"x{contracts} @ ${fill_price:.2f}"
            )

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _select_0dte_contract(
        self, signal: Signal, ts: TickerState
    ) -> Optional[dict]:
        """Select a 0DTE option contract targeting our delta."""
        try:
            chains = await self.ibkr.get_option_chains(signal.ticker)
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None

        if not chains:
            return None

        # Get today's expiry (0DTE)
        today = datetime.now().strftime("%Y%m%d")
        available_expiries = set()
        for chain in chains:
            for exp in chain.get("expirations", []):
                available_expiries.add(exp)

        # Prefer 0DTE, fallback to 1DTE
        target_expiry = None
        if today in available_expiries:
            target_expiry = today
        else:
            # Find nearest future expiry
            future = sorted([e for e in available_expiries if e >= today])
            if future:
                target_expiry = future[0]

        if not target_expiry:
            return None

        right = "C" if signal.direction == Direction.CALL else "P"

        # Get strikes near current price
        all_strikes = set()
        for chain in chains:
            all_strikes.update(chain.get("strikes", []))
        strikes = sorted(all_strikes)

        # Get ~10 strikes nearest to current price
        price = ts.last_price
        nearest = sorted(strikes, key=lambda s: abs(s - price))[:10]

        # Query greeks for each candidate
        best = None
        target_delta = self.scalper_config.delta_target

        for strike in nearest:
            try:
                greeks = await self.ibkr.get_option_greeks(
                    signal.ticker, target_expiry, strike, right
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
                continue  # Too cheap — likely too far OTM

            # Delta filter: 0.25 - 0.55
            if delta < 0.25 or delta > 0.55:
                continue

            # Spread filter: <15% for 0DTE
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > 0.15:
                continue

            # Pick closest to target delta
            dist = abs(delta - target_delta)
            if best is None or dist < best["_dist"]:
                # Qualify contract
                try:
                    ibkr_contract = self.ibkr.make_option_contract(
                        signal.ticker, target_expiry, strike, right
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

    async def _manage_position(self):
        """Monitor and manage an open position."""
        if self.position is None:
            return

        pos = self.position
        now = self._now_et()
        hold_mins = (now - pos.entry_time).total_seconds() / 60.0

        # Update current price from IBKR
        if not self.dry_run:
            try:
                right = "C" if pos.direction == Direction.CALL else "P"
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
        if not pos.trailing_active and pos.unrealized_pnl_pct >= self.scalper_config.trail_activation_pct:
            pos.trailing_active = True
            pos.trailing_peak = pos.current_price
            logger.info(
                f"🔄 Trail activated: {pos.ticker} at +{pos.unrealized_pnl_pct:.1%}"
            )
        elif pos.trailing_active and pos.current_price > pos.trailing_peak:
            pos.trailing_peak = pos.current_price

        # Check exits (priority order)
        exit_reason = None

        # 1. EOD force close
        if self._time_et() >= self.scalper_config.force_close:
            exit_reason = ExitReason.EOD_CLOSE

        # 2. Stop loss
        elif pos.unrealized_pnl_pct <= -self.scalper_config.stop_loss_pct:
            exit_reason = ExitReason.STOP_LOSS

        # 3. Take profit
        elif pos.unrealized_pnl_pct >= self.scalper_config.take_profit_pct:
            exit_reason = ExitReason.TAKE_PROFIT

        # 4. Trailing stop
        elif pos.trailing_active:
            drop = (pos.current_price - pos.trailing_peak) / pos.trailing_peak if pos.trailing_peak > 0 else 0
            if drop <= -self.scalper_config.trail_distance_pct:
                exit_reason = ExitReason.TRAILING_STOP

        # 5. Time stop
        elif hold_mins >= self.scalper_config.max_hold_minutes:
            exit_reason = ExitReason.TIME_STOP

        # 6. Time decay tightening
        elif hold_mins >= self.scalper_config.time_decay_exit_min:
            tightened_stop = self.scalper_config.stop_loss_pct * 0.70
            if pos.unrealized_pnl_pct <= -tightened_stop:
                exit_reason = ExitReason.TIME_STOP

        if exit_reason is not None:
            await self._close_position(exit_reason)

    async def _close_position(self, reason: ExitReason):
        """Close the current position."""
        pos = self.position
        if pos is None:
            return

        pnl_pct = pos.unrealized_pnl_pct
        pnl_dollars = (pos.current_price - pos.entry_price) * 100 * pos.num_contracts
        hold_mins = (self._now_et() - pos.entry_time).total_seconds() / 60.0

        logger.info(
            f"{'🟢' if pnl_pct > 0 else '🔴'} EXIT: {pos.ticker} {pos.direction.value} "
            f"({reason.value}) P&L: {pnl_pct:+.1%} (${pnl_dollars:+,.2f}) "
            f"hold={hold_mins:.1f}min"
        )

        # Close via IBKR
        if not self.dry_run:
            try:
                await self.order_manager.close_position(pos, reason)
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                try:
                    await self.ibkr.close_all_positions()
                except Exception:
                    pass

        # Record trade
        self.trade_log.append({
            "ticker": pos.ticker,
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
        })

        # Update state
        self.daily_pnl += pnl_dollars
        self.pm_trades += 1

        # Cooldown
        if pnl_pct < 0:
            self.cooldown_until = self._now_et() + timedelta(
                seconds=self.scalper_config.cooldown_after_loss_sec
            )
        else:
            self.cooldown_until = self._now_et() + timedelta(
                seconds=self.scalper_config.cooldown_after_trade_sec
            )

        # Circuit breaker
        if self.daily_pnl / self.scalper_config.account_value <= -self.scalper_config.daily_loss_limit_pct:
            self.circuit_breaker = True
            logger.warning(f"⚠️ CIRCUIT BREAKER: daily P&L = ${self.daily_pnl:+,.2f}")

        self.position = None

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now_et() < self.cooldown_until

    async def _force_close_eod(self):
        """Force close all positions at end of day."""
        if self.position:
            logger.warning("EOD: Force closing open position")
            await self._close_position(ExitReason.EOD_CLOSE)

        if not self.dry_run:
            try:
                await self.ibkr.cancel_all_orders()
            except Exception:
                pass

        # Print daily summary
        self._print_daily_summary()

    def _print_daily_summary(self):
        """Print end-of-day trade summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  SCALPER BOT 2 — Daily Summary")
        logger.info("=" * 50)
        logger.info(f"Trades: {len(self.trade_log)}")
        logger.info(f"Daily P&L: ${self.daily_pnl:+,.2f}")

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
        """Set up signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.warning(f"Signal {signum} received — shutting down")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def _check_kill_switch(self):
        """Check for kill switch conditions."""
        if self.shutdown_requested:
            self.running = False
            return

        kill_file = "KILL_SCALPER"
        if os.path.exists(kill_file):
            logger.warning(f"Kill file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)

        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down scalper...")
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

        logger.info("Scalper shutdown complete.")


# ============================================================
# Main
# ============================================================

def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load config YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging():
    """Configure logging for the scalper."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/scalper_{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="0DTE Scalper Bot — Paper Trading")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal generation only, no orders")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    # Add scalper-specific config defaults if not in YAML
    if "scalper" not in config:
        config["scalper"] = {
            "account_value": 25_000,
            "risk_per_trade_pct": 0.20,
            "max_notional_pct": 0.40,
            "max_contracts": 50,
            "ph_momentum_bars": 3,
            "ph_min_move_pct": 0.0015,
            "ph_volume_surge": 1.3,
            "stop_loss_pct": 0.25,
            "take_profit_pct": 0.75,
            "trail_activation_pct": 0.15,
            "trail_distance_pct": 0.10,
            "max_hold_minutes": 22,
            "time_decay_exit_min": 18,
            "max_strength": 85,
            "put_size_multiplier": 0.65,
            "max_pm_trades": 2,
            "cooldown_after_trade_sec": 60,
            "cooldown_after_loss_sec": 120,
            "daily_loss_limit_pct": 0.05,
            "delta": 0.40,
            "tickers": ["QQQ", "NVDA"],
        }

    bot = ScalperBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
