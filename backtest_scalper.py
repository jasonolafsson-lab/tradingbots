"""
Aggressive 0DTE Scalper Backtester — "Bot 2"

Pure momentum scalping on 0DTE options:
  - Strategy 1: Opening Range Breakout (ORB) — 9:45-10:30 AM
  - Strategy 2: VWAP Reclaim/Rejection — all day
  - Strategy 3: Power Hour Momentum — 3:00-3:45 PM

Design philosophy: Trade what's moving. Quick entries, quick exits.
Rarely hold more than 10 minutes. 2-3 trades per day.

Usage:
    python backtest_scalper.py --start 2025-09-15 --end 2026-03-13 --tickers QQQ NVDA --from-cache
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import yaml

# Reuse existing bot infrastructure
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.market_state import (
    Bar, TickerState, MarketState, Signal, Position,
    Regime, Direction, Bias, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.opening_range import OpeningRangeTracker
from indicators.volume_profile import VolumeProfileCalculator

# Reuse existing backtest infrastructure
from backtest import (
    load_config, load_data, build_daily_bars, aggregate_3m_bars,
    estimate_option_premium, TradeRecord, CACHE_DIR,
)

ET = ZoneInfo("US/Eastern")
logger = logging.getLogger("scalper_backtest")


# ============================================================
# Scalper Configuration
# ============================================================

@dataclass
class ScalperConfig:
    """All tunable parameters for the aggressive scalper."""
    # Account
    account_value: float = 25_000
    risk_per_trade_pct: float = 0.20       # 20% of account per trade — max aggression
    max_notional_pct: float = 0.40         # 40% max notional
    max_contracts: int = 50

    # Schedule
    orb_window_start: dtime = dtime(9, 45)
    orb_window_end: dtime = dtime(10, 30)
    morning_momentum_start: dtime = dtime(9, 50)
    morning_momentum_end: dtime = dtime(10, 45)
    midday_start: dtime = dtime(10, 30)
    midday_end: dtime = dtime(14, 30)     # VWAP trades allowed midday too
    power_hour_start: dtime = dtime(15, 0)    # Power Hour — 3:00 PM start (tested: earlier = worse)
    power_hour_end: dtime = dtime(15, 45)
    force_close: dtime = dtime(15, 55)
    last_entry: dtime = dtime(15, 40)

    # ORB Strategy
    orb_enabled: bool = False               # Disabled — loses money in backtests
    orb_breakout_margin_pct: float = 0.0005
    orb_min_volume_ratio: float = 1.2
    orb_min_range_pct: float = 0.001
    orb_max_range_pct: float = 0.020

    # VWAP Strategy — morning CALLs only, higher strength filter
    vwap_enabled: bool = False              # Disabled — adds DD without proportional return
    vwap_calls_only: bool = True            # Only CALL direction (PUTs drag P&L)
    vwap_min_strength: float = 55           # Higher bar than Power Hour (was 40)
    vwap_reclaim_bars: int = 2              # Price must cross VWAP and hold 2 bars
    vwap_min_distance_pct: float = 0.001    # Must have been at least 0.1% away from VWAP
    vwap_volume_confirm: float = 1.0        # No volume filter (RSI does filtering)

    # Momentum Strategy (Morning + Power Hour)
    mom_bars: int = 3                       # 3 consecutive bars in same direction
    mom_min_move_pct: float = 0.0015        # Minimum 0.15% move in 3 bars
    mom_volume_surge: float = 1.3           # Volume surge vs session/20-bar average

    # Power Hour Strategy
    ph_momentum_bars: int = 3               # 3 consecutive bars — the edge lives here
    ph_min_move_pct: float = 0.0015         # 0.15% minimum move
    ph_volume_surge: float = 1.3            # Volume surge filter

    # Exit Rules
    stop_loss_pct: float = 0.25             # -25% of option premium (tight cut)
    take_profit_pct: float = 0.75           # +75% take profit (winners run big)
    trail_activation_pct: float = 0.15      # Activate trail at +15%
    trail_distance_pct: float = 0.10        # Trail 10% from peak (lock in gains fast)
    max_hold_minutes: float = 22            # Extended: 15-20 min = 75% win, 20+ = 100%
    time_decay_exit_min: float = 18         # Delayed: was 12, killing winners too early

    # Dead zone — block entries during 2PM hour (0% win rate in backtests)
    dead_zone_start: dtime = dtime(13, 50)
    dead_zone_end: dtime = dtime(14, 55)

    # Signal quality filter
    max_strength: float = 85                # Reject overheated signals (>85 = 25% win rate)

    # Direction bias
    put_size_multiplier: float = 0.65       # PUTs use 65% of CALL size (weaker win rate)

    # Trades per day
    max_am_trades: int = 3                  # Morning session
    max_pm_trades: int = 2                  # Power hour
    max_total_trades: int = 5

    # Cooldown
    cooldown_after_trade_sec: int = 60      # 1 min cooldown
    cooldown_after_loss_sec: int = 120      # 2 min after loss
    daily_loss_limit_pct: float = 0.05      # -5% daily circuit breaker

    # Delta for 0DTE
    delta: float = 0.40                     # Aggressive delta for 0DTE


# ============================================================
# Signal Generators
# ============================================================

def check_orb_breakout(
    ts: TickerState,
    config: ScalperConfig,
    bar_time: datetime,
) -> Optional[Signal]:
    """
    Opening Range Breakout: Price breaks above OR high or below OR low
    with volume confirmation.
    """
    if not ts.opening_range_set:
        return None

    or_high = ts.opening_range_high
    or_low = ts.opening_range_low
    if or_high is None or or_low is None or or_high <= 0:
        return None

    # Check OR range size
    or_range_pct = (or_high - or_low) / or_low
    if or_range_pct < config.orb_min_range_pct:
        return None  # Too narrow — no conviction
    if or_range_pct > config.orb_max_range_pct:
        return None  # Too wide — choppy open

    price = ts.last_price

    # Breakout margin
    margin = or_high * config.orb_breakout_margin_pct

    direction = None
    if price > or_high + margin:
        # Must be above VWAP for bullish breakout
        if ts.vwap > 0 and price > ts.vwap:
            direction = Direction.CALL
    elif price < or_low - margin:
        # Must be below VWAP for bearish breakdown
        if ts.vwap > 0 and price < ts.vwap:
            direction = Direction.PUT

    if direction is None:
        return None

    # Volume confirmation
    if ts.volume_ratio < config.orb_min_volume_ratio:
        return None

    # Strength scoring
    strength = 50.0

    # Volume bonus (max +15)
    strength += min((ts.volume_ratio - 1.0) / 1.0, 1.0) * 15

    # VWAP alignment bonus (max +15)
    if direction == Direction.CALL and ts.vwap_slope > 0:
        strength += 15
    elif direction == Direction.PUT and ts.vwap_slope < 0:
        strength += 15

    # ADX bonus — trend strength (max +10)
    if ts.adx_14 >= 20:
        strength += min((ts.adx_14 - 20) / 20.0, 1.0) * 10

    # Break magnitude bonus (max +10)
    if direction == Direction.CALL:
        break_pct = (price - or_high) / or_high
    else:
        break_pct = (or_low - price) / or_low
    strength += min(break_pct / 0.005, 1.0) * 10

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="ORB_SCALP",
        regime=Regime.MOMENTUM,
        strength_score=min(strength, 100),
        entry_price_target=price,
        timestamp=bar_time,
    )


def check_vwap_reclaim(
    ts: TickerState,
    config: ScalperConfig,
    bar_time: datetime,
) -> Optional[Signal]:
    """
    VWAP Reclaim/Rejection:
    - Price was below VWAP, crosses above and holds → CALL
    - Price was above VWAP, crosses below and holds → PUT

    Requires volume confirmation on the cross bar.
    """
    if ts.vwap <= 0 or len(ts.bars_1m) < config.vwap_reclaim_bars + 5:
        return None

    bars = ts.bars_1m
    recent = bars[-config.vwap_reclaim_bars:]
    prior = bars[-(config.vwap_reclaim_bars + 3):-config.vwap_reclaim_bars]

    if not prior or not recent:
        return None

    price = ts.last_price
    vwap = ts.vwap

    # Check if price crossed VWAP recently
    prior_avg = np.mean([b.close for b in prior])
    distance_pct = abs(prior_avg - vwap) / vwap

    if distance_pct < config.vwap_min_distance_pct:
        return None  # Wasn't far enough from VWAP for a meaningful cross

    direction = None

    # Bullish reclaim: was below VWAP, now above + RSI was below 45 (pulling back)
    if prior_avg < vwap and all(b.close > vwap for b in recent):
        if ts.rsi_7 <= 60:  # RSI not overbought — room to run
            direction = Direction.CALL

    # Bearish rejection: was above VWAP, now below + RSI was above 55 (extended)
    elif prior_avg > vwap and all(b.close < vwap for b in recent):
        if ts.rsi_7 >= 40:  # RSI not oversold — room to fall
            direction = Direction.PUT

    if direction is None:
        return None

    # Volume confirmation
    if ts.volume_ratio < config.vwap_volume_confirm:
        return None

    # Strength scoring
    strength = 45.0

    # Distance from VWAP (max +15) — bigger distance = stronger signal
    strength += min(distance_pct / 0.003, 1.0) * 15

    # VWAP slope alignment (max +15)
    if direction == Direction.CALL and ts.vwap_slope > 0:
        strength += 15
    elif direction == Direction.PUT and ts.vwap_slope < 0:
        strength += 15

    # Volume bonus (max +10)
    strength += min((ts.volume_ratio - 1.0) / 1.0, 1.0) * 10

    # RSI confirmation (max +15)
    if direction == Direction.CALL and ts.rsi_7 > 50:
        strength += min((ts.rsi_7 - 50) / 20.0, 1.0) * 15
    elif direction == Direction.PUT and ts.rsi_7 < 50:
        strength += min((50 - ts.rsi_7) / 20.0, 1.0) * 15

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="VWAP_SCALP",
        regime=Regime.MOMENTUM,
        strength_score=min(strength, 100),
        entry_price_target=price,
        timestamp=bar_time,
    )


def check_morning_momentum(
    ts: TickerState,
    config: ScalperConfig,
    bar_time: datetime,
) -> Optional[Signal]:
    """
    Morning Momentum: 3+ consecutive bars in same direction after
    opening range is set. Catch the early breakout move.
    Similar to Power Hour but with opening range context.
    """
    if not ts.opening_range_set:
        return None

    n = config.mom_bars
    if len(ts.bars_1m) < n + 5:
        return None

    recent = ts.bars_1m[-n:]

    all_green = all(b.close > b.open for b in recent)
    all_red = all(b.close < b.open for b in recent)

    if not all_green and not all_red:
        return None

    # Minimum move magnitude
    move = abs(recent[-1].close - recent[0].open)
    move_pct = move / recent[0].open
    if move_pct < config.mom_min_move_pct:
        return None

    # Volume surge vs 20-bar avg
    if len(ts.bars_1m) > 20:
        avg_vol = np.mean([b.volume for b in ts.bars_1m[-21:-n]])
        recent_avg_vol = np.mean([b.volume for b in recent])
        if avg_vol > 0:
            vol_ratio = recent_avg_vol / avg_vol
            if vol_ratio < config.mom_volume_surge:
                return None
        else:
            vol_ratio = 1.0
    else:
        return None  # Need at least 20 bars of context

    direction = Direction.CALL if all_green else Direction.PUT

    # Must be above VWAP for calls, below for puts
    if direction == Direction.CALL and ts.vwap > 0 and ts.last_price < ts.vwap:
        return None
    if direction == Direction.PUT and ts.vwap > 0 and ts.last_price > ts.vwap:
        return None

    # Must be breaking out of opening range
    if direction == Direction.CALL and ts.last_price < ts.opening_range_high:
        return None  # Not a breakout — still inside OR
    if direction == Direction.PUT and ts.last_price > ts.opening_range_low:
        return None  # Not a breakdown

    strength = 55.0  # Base score for morning momentum

    # Move magnitude bonus (max +15)
    strength += min(move_pct / 0.005, 1.0) * 15

    # Volume bonus (max +15)
    strength += min((vol_ratio - 1.0) / 1.0, 1.0) * 15

    # ADX bonus (max +10)
    if ts.adx_14 >= 20:
        strength += min((ts.adx_14 - 20) / 20.0, 1.0) * 10

    # VWAP slope alignment (max +5)
    if direction == Direction.CALL and ts.vwap_slope > 0:
        strength += 5
    elif direction == Direction.PUT and ts.vwap_slope < 0:
        strength += 5

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="MORNING_MOM",
        regime=Regime.MOMENTUM,
        strength_score=min(strength, 100),
        entry_price_target=ts.last_price,
        timestamp=bar_time,
    )


def check_power_hour_momentum(
    ts: TickerState,
    config: ScalperConfig,
    bar_time: datetime,
) -> Optional[Signal]:
    """
    Power Hour Momentum: 3+ consecutive bars moving in the same direction
    with volume surge. Ride the late-day push.
    """
    n = config.ph_momentum_bars
    if len(ts.bars_1m) < n + 5:
        return None

    recent = ts.bars_1m[-n:]

    # Check all bars are in the same direction
    all_green = all(b.close > b.open for b in recent)
    all_red = all(b.close < b.open for b in recent)

    if not all_green and not all_red:
        return None

    # Minimum move magnitude
    move = abs(recent[-1].close - recent[0].open)
    move_pct = move / recent[0].open
    if move_pct < config.ph_min_move_pct:
        return None

    # Volume surge
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
    if direction == Direction.CALL and ts.last_price < ts.vwap:
        return None
    if direction == Direction.PUT and ts.last_price > ts.vwap:
        return None

    # Strength scoring
    strength = 50.0

    # Move magnitude (max +15)
    strength += min(move_pct / 0.005, 1.0) * 15

    # Volume (max +15)
    strength += min((vol_ratio - 1.0) / 1.0, 1.0) * 15

    # ADX — trend strength (max +10)
    if ts.adx_14 >= 20:
        strength += min((ts.adx_14 - 20) / 20.0, 1.0) * 10

    # VWAP slope alignment (max +10)
    if direction == Direction.CALL and ts.vwap_slope > 0:
        strength += 10
    elif direction == Direction.PUT and ts.vwap_slope < 0:
        strength += 10

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="POWER_HOUR",
        regime=Regime.MOMENTUM,
        strength_score=min(strength, 100),
        entry_price_target=ts.last_price,
        timestamp=bar_time,
    )


# ============================================================
# Position Sizing for Aggressive Scalping
# ============================================================

def size_scalp_position(
    premium: float,
    config: ScalperConfig,
    account_value: float,
) -> int:
    """
    Aggressive position sizing.
    Risk 5% of account per trade on 0DTE options.
    """
    if premium <= 0:
        return 0

    max_risk = account_value * config.risk_per_trade_pct
    max_notional = account_value * config.max_notional_pct

    # Risk per contract = premium * 100 * stop_loss_pct
    risk_per_contract = premium * 100 * config.stop_loss_pct

    if risk_per_contract <= 0:
        return 0

    contracts = math.floor(max_risk / risk_per_contract)

    # Notional cap
    notional = contracts * premium * 100
    if notional > max_notional:
        contracts = math.floor(max_notional / (premium * 100))

    contracts = min(contracts, config.max_contracts)
    contracts = max(contracts, 1)

    return contracts


# ============================================================
# 0DTE Option Premium Model
# ============================================================

def estimate_0dte_premium(
    ticker: str,
    underlying_price: float,
    delta: float,
) -> float:
    """
    0DTE options are CHEAPER than weekly/monthly.
    Lower premium = higher gamma = bigger % moves.
    """
    # 0DTE premiums are roughly 40-60% of weekly premiums
    if ticker in ("SPY", "QQQ"):
        multiplier = 0.0018  # ~40% cheaper than weekly (was 0.003)
    else:
        multiplier = 0.0035  # ~40% cheaper than weekly (was 0.006)

    premium = underlying_price * multiplier * (delta / 0.35)
    return max(premium, 0.10)


# ============================================================
# 0DTE Gamma-Enhanced P&L Model
# ============================================================

def estimate_0dte_option_price(
    entry_premium: float,
    underlying_entry: float,
    underlying_now: float,
    delta: float,
    direction: Direction,
    hold_minutes: float,
) -> float:
    """
    0DTE options have higher gamma than weekly/monthly options.
    Small underlying moves produce bigger % premium changes.

    Models:
    - Delta-based move (linear)
    - Gamma kicker (acceleration on bigger moves)
    - Theta decay (0DTE bleeds fast in last hours)
    """
    if direction == Direction.CALL:
        underlying_move = underlying_now - underlying_entry
    else:
        underlying_move = underlying_entry - underlying_now

    # Delta component (linear)
    delta_pnl = underlying_move * delta

    # Gamma kicker — 0DTE options accelerate on bigger moves
    # Approximate gamma as 15% of delta per $1 underlying move
    move_magnitude = abs(underlying_now - underlying_entry)
    gamma_estimate = delta * 0.15  # Conservative gamma estimate
    gamma_pnl = 0.5 * gamma_estimate * (move_magnitude ** 2) / underlying_entry

    # Apply gamma — helps both wins AND losses (realistic)
    if underlying_move > 0:
        total_move = delta_pnl + gamma_pnl
    else:
        total_move = delta_pnl - gamma_pnl * 0.3  # Gamma helps less on losses (delta shrinks rapidly)

    # Theta decay — 0DTE bleeds ~0.15% per minute in first half, faster near close
    theta_decay_per_min = entry_premium * 0.0015  # 0.15% per minute baseline
    theta_cost = theta_decay_per_min * hold_minutes

    new_premium = entry_premium + total_move - theta_cost
    return max(new_premium, 0.01)


# ============================================================
# Day Replay Engine (Scalper Version)
# ============================================================

def replay_day_scalper(
    replay_date: date,
    ticker_bars: Dict[str, List[Bar]],
    prior_days: Dict[str, List[dict]],
    config: ScalperConfig,
    account_value: float,
) -> Tuple[List[TradeRecord], float]:
    """Replay a single day with aggressive scalp strategies."""

    # Indicators
    vwap_calc = VWAPCalculator()
    rsi_calc = RSICalculator(period=7)
    adx_calc = ADXCalculator(period=14)
    vol_calc = VolumeProfileCalculator(lookback=20)
    or_config = {"opening_range": {"duration_minutes": 15}}
    or_tracker = OpeningRangeTracker(or_config)

    # Market state
    ticker_states: Dict[str, TickerState] = {}
    for ticker in ticker_bars:
        ts = TickerState(ticker=ticker)
        history = prior_days.get(ticker, [])
        if history:
            last_day = history[-1]
            ts.prior_high = last_day["high"]
            ts.prior_low = last_day["low"]
            ts.prior_close = last_day["close"]
            ts.prior_volume = last_day["volume"]
        ticker_states[ticker] = ts

    # Build bar lookup
    all_times = set()
    bar_lookup: Dict[str, Dict[datetime, Bar]] = {}
    for ticker, bars in ticker_bars.items():
        bar_lookup[ticker] = {b.timestamp: b for b in bars}
        for b in bars:
            all_times.add(b.timestamp)
    sorted_times = sorted(all_times)

    # State
    open_position: Optional[dict] = None
    trades: List[TradeRecord] = []
    current_account = account_value
    am_trades = 0
    pm_trades = 0
    daily_pnl = 0.0
    cooldown_until: Optional[datetime] = None
    circuit_breaker = False

    # Track which tickers have been signaled to avoid repeats
    last_signal_bar: Dict[str, datetime] = {}

    for bar_time in sorted_times:
        # Convert UTC bar time to ET for schedule comparison
        if bar_time.tzinfo is not None:
            bar_time_et = bar_time.astimezone(ET)
        else:
            bar_time_et = bar_time
        bar_t = bar_time_et.time()

        # Update ticker states
        for ticker in ticker_bars:
            ts = ticker_states[ticker]
            bar = bar_lookup[ticker].get(bar_time)
            if bar is None:
                continue

            ts.bars_1m.append(bar)
            ts.last_price = bar.close
            ts.last_bar_time = bar_time

            if ts.prior_close > 0:
                ts.session_return = (ts.last_price - ts.prior_close) / ts.prior_close

            # VWAP
            if len(ts.bars_1m) >= 2:
                vwap_data = vwap_calc.calculate(ts.bars_1m)
                ts.vwap = vwap_data["vwap"]
                ts.vwap_upper_band = vwap_data["upper_band"]
                ts.vwap_lower_band = vwap_data["lower_band"]
                ts.vwap_slope = vwap_data["slope"]

            # RSI, ADX, Volume from 3-min bars
            ts.bars_3m = aggregate_3m_bars(ts.bars_1m)
            if ts.bars_3m:
                closes = [b.close for b in ts.bars_3m]
                highs = [b.high for b in ts.bars_3m]
                lows = [b.low for b in ts.bars_3m]
                volumes = [b.volume for b in ts.bars_3m]
                ts.rsi_7 = rsi_calc.calculate(closes)
                ts.adx_14 = adx_calc.calculate(highs, lows, closes)
                ts.volume_ratio = vol_calc.calculate(volumes)

            # Opening range — OR tracker expects 9:30-9:45 ET times
            # Bars may be in UTC, so we need to check manually
            if not ts.opening_range_set:
                or_bars = []
                for b in ts.bars_1m:
                    bt = b.timestamp.astimezone(ET) if b.timestamp.tzinfo else b.timestamp
                    bt_time = bt.time()
                    if dtime(9, 30) <= bt_time < dtime(9, 45):
                        or_bars.append(b)
                if len(or_bars) >= 15:
                    ts.opening_range_high = max(b.high for b in or_bars)
                    ts.opening_range_low = min(b.low for b in or_bars)
                    ts.opening_range_set = True
                elif or_bars:
                    ts.opening_range_high = max(b.high for b in or_bars)
                    ts.opening_range_low = min(b.low for b in or_bars)

        # --- Manage open position ---
        if open_position is not None:
            pos = open_position["position"]
            delta = open_position["delta"]
            underlying_entry = open_position["underlying_entry"]
            ticker = pos.ticker
            ts = ticker_states.get(ticker)

            if ts:
                underlying_now = ts.last_price
                hold_mins = (bar_time - pos.entry_time).total_seconds() / 60.0

                # 0DTE gamma-enhanced option pricing
                option_price = estimate_0dte_option_price(
                    entry_premium=pos.entry_price,
                    underlying_entry=underlying_entry,
                    underlying_now=underlying_now,
                    delta=delta,
                    direction=pos.direction,
                    hold_minutes=hold_mins,
                )

                pos.current_price = option_price
                pos.unrealized_pnl_pct = (option_price - pos.entry_price) / pos.entry_price
                pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
                pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

                # Trailing stop management
                if not pos.trailing_active and pos.unrealized_pnl_pct >= config.trail_activation_pct:
                    pos.trailing_active = True
                    pos.trailing_peak = option_price
                elif pos.trailing_active and option_price > pos.trailing_peak:
                    pos.trailing_peak = option_price

                # Check exits
                exit_reason = None

                # 1. EOD force close
                if bar_t >= config.force_close:
                    exit_reason = ExitReason.EOD_CLOSE

                # 2. Stop loss
                elif pos.unrealized_pnl_pct <= -config.stop_loss_pct:
                    exit_reason = ExitReason.STOP_LOSS

                # 3. Take profit
                elif pos.unrealized_pnl_pct >= config.take_profit_pct:
                    exit_reason = ExitReason.TAKE_PROFIT

                # 4. Trailing stop
                elif pos.trailing_active:
                    drop = (option_price - pos.trailing_peak) / pos.trailing_peak
                    if drop <= -config.trail_distance_pct:
                        exit_reason = ExitReason.TRAILING_STOP

                # 5. Time stop
                elif hold_mins >= config.max_hold_minutes:
                    exit_reason = ExitReason.TIME_STOP

                # 6. Time decay tightening — after 18 min, tighten stop to -17.5%
                elif hold_mins >= config.time_decay_exit_min:
                    tightened_stop = config.stop_loss_pct * 0.70  # Tighten to 70% of original (was 60%)
                    if pos.unrealized_pnl_pct <= -tightened_stop:
                        exit_reason = ExitReason.TIME_STOP

                if exit_reason is not None:
                    pnl_pct = pos.unrealized_pnl_pct
                    pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts

                    trades.append(TradeRecord(
                        ticker=ticker,
                        direction=pos.direction.value,
                        strategy=pos.strategy,
                        entry_time=pos.entry_time,
                        exit_time=bar_time,
                        entry_price=pos.entry_price,
                        exit_price=option_price,
                        num_contracts=pos.num_contracts,
                        delta=delta,
                        underlying_entry=underlying_entry,
                        underlying_exit=underlying_now,
                        pnl_pct=pnl_pct,
                        pnl_dollars=pnl_dollars,
                        exit_reason=exit_reason.value,
                        strength_score=pos.signal_strength_score,
                        hold_minutes=hold_mins,
                    ))

                    current_account += pnl_dollars
                    daily_pnl += pnl_dollars

                    # Track AM/PM trades
                    if pos.entry_time.time() < dtime(14, 30):
                        am_trades += 1
                    else:
                        pm_trades += 1

                    # Cooldown
                    if pnl_pct < 0:
                        cooldown_until = bar_time + timedelta(seconds=config.cooldown_after_loss_sec)
                    else:
                        cooldown_until = bar_time + timedelta(seconds=config.cooldown_after_trade_sec)

                    # Circuit breaker
                    if daily_pnl / config.account_value <= -config.daily_loss_limit_pct:
                        circuit_breaker = True

                    open_position = None

        # --- Look for new entries ---
        if (open_position is None
                and not circuit_breaker
                and (cooldown_until is None or bar_time >= cooldown_until)):

            total_trades = am_trades + pm_trades
            if total_trades >= config.max_total_trades:
                continue

            # Determine which strategies to check based on time
            signals: List[Signal] = []

            for ticker in ticker_bars:
                ts = ticker_states[ticker]

                # Skip if we just signaled this ticker within last 2 min
                last_sig = last_signal_bar.get(ticker)
                if last_sig and (bar_time - last_sig).total_seconds() < 120:
                    continue

                # Morning session: ORB + Morning Momentum
                if config.orb_enabled and config.orb_window_start <= bar_t <= config.orb_window_end:
                    if am_trades < config.max_am_trades:
                        sig = check_orb_breakout(ts, config, bar_time)
                        if sig:
                            signals.append(sig)

                # Morning Momentum: DISABLED — loses money in backtests
                # if config.morning_momentum_start <= bar_t <= config.morning_momentum_end:
                #     if am_trades < config.max_am_trades:
                #         sig = check_morning_momentum(ts, config, bar_time)
                #         if sig:
                #             signals.append(sig)

                # VWAP reclaim: morning only (9:50-13:50), skip 2PM dead zone
                # Don't let VWAP fire during Power Hour — momentum owns 3:00+
                if config.vwap_enabled and dtime(9, 50) <= bar_t < config.dead_zone_start:
                    if total_trades < config.max_total_trades:
                        sig = check_vwap_reclaim(ts, config, bar_time)
                        if sig:
                            # Apply VWAP-specific filters
                            if config.vwap_calls_only and sig.direction != Direction.CALL:
                                sig = None  # Skip PUT signals for VWAP
                            elif sig.strength_score < config.vwap_min_strength:
                                sig = None  # Require higher conviction
                            if sig:
                                signals.append(sig)

                # Midday momentum: DISABLED — PF 0.96, adds noise
                # if dtime(11, 0) <= bar_t <= dtime(14, 30):
                #     ...

                # Power hour
                if config.power_hour_start <= bar_t <= config.power_hour_end:
                    if pm_trades < config.max_pm_trades:
                        sig = check_power_hour_momentum(ts, config, bar_time)
                        if sig:
                            signals.append(sig)

            # Take the strongest signal
            if signals:
                signals.sort(key=lambda s: s.strength_score, reverse=True)
                best = signals[0]

                if 40 <= best.strength_score <= config.max_strength:
                    premium = estimate_0dte_premium(
                        best.ticker, best.entry_price_target, config.delta
                    )
                    contracts = size_scalp_position(premium, config, current_account)

                    # Reduce PUT size — weaker win rate
                    if best.direction == Direction.PUT:
                        contracts = max(1, int(contracts * config.put_size_multiplier))

                    if contracts > 0:
                        pos = Position(
                            ticker=best.ticker,
                            direction=best.direction,
                            contract_type=ContractType.SINGLE_LEG,
                            strategy=best.strategy,
                            regime=best.regime.value,
                            signal=best,
                            delta_at_entry=config.delta,
                            entry_time=bar_time,
                            entry_price=premium,
                            num_contracts=contracts,
                            current_price=premium,
                            signal_strength_score=best.strength_score,
                        )

                        open_position = {
                            "position": pos,
                            "delta": config.delta,
                            "underlying_entry": best.entry_price_target,
                        }

                        last_signal_bar[best.ticker] = bar_time

    # Force close EOD
    if open_position is not None:
        pos = open_position["position"]
        delta = open_position["delta"]
        underlying_entry = open_position["underlying_entry"]
        ticker = pos.ticker
        ts = ticker_states.get(ticker)

        if ts and sorted_times:
            last_time = sorted_times[-1]
            underlying_now = ts.last_price
            hold_mins = (last_time - pos.entry_time).total_seconds() / 60.0

            option_price = estimate_0dte_option_price(
                pos.entry_price, underlying_entry, underlying_now,
                delta, pos.direction, hold_mins,
            )

            pnl_pct = (option_price - pos.entry_price) / pos.entry_price
            pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts

            trades.append(TradeRecord(
                ticker=ticker,
                direction=pos.direction.value,
                strategy=pos.strategy,
                entry_time=pos.entry_time,
                exit_time=last_time,
                entry_price=pos.entry_price,
                exit_price=option_price,
                num_contracts=pos.num_contracts,
                delta=delta,
                underlying_entry=underlying_entry,
                underlying_exit=underlying_now,
                pnl_pct=pnl_pct,
                pnl_dollars=pnl_dollars,
                exit_reason="EOD_CLOSE",
                strength_score=pos.signal_strength_score,
                hold_minutes=hold_mins,
            ))
            current_account += pnl_dollars

    return trades, current_account


# ============================================================
# Report Engine (reuses same format)
# ============================================================

def generate_scalper_report(
    trades: List[TradeRecord],
    starting_capital: float,
    start_date: date,
    end_date: date,
) -> None:
    """Generate report for scalper backtest."""

    if not trades:
        print("\n" + "=" * 70)
        print("  SCALPER BACKTEST — NO TRADES GENERATED")
        print("=" * 70)
        print("\nNo qualifying scalp signals found in this period.")
        return

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = sum(t.pnl_dollars for t in losses) if losses else 0
    win_rate = len(wins) / len(trades) * 100
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_dollars for t in losses]) if losses else 0
    avg_trade = np.mean([t.pnl_dollars for t in trades])
    avg_hold = np.mean([t.hold_minutes for t in trades])
    largest_win = max([t.pnl_dollars for t in trades])
    largest_loss = min([t.pnl_dollars for t in trades])
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0

    equity = [starting_capital]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollars)

    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    for val in equity:
        peak = max(peak, val)
        dd = val - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100

    # Consecutive streaks
    max_consec_wins = max_consec_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl_dollars > 0:
            current_streak = current_streak + 1 if current_streak > 0 else 1
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1
            max_consec_losses = max(max_consec_losses, abs(current_streak))

    # By strategy
    strategy_stats = {}
    for t in trades:
        s = t.strategy
        if s not in strategy_stats:
            strategy_stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0,
                                 "gross_profit": 0.0, "gross_loss": 0.0}
        strategy_stats[s]["trades"] += 1
        strategy_stats[s]["pnl"] += t.pnl_dollars
        if t.pnl_dollars > 0:
            strategy_stats[s]["wins"] += 1
            strategy_stats[s]["gross_profit"] += t.pnl_dollars
        else:
            strategy_stats[s]["gross_loss"] += t.pnl_dollars

    # By ticker
    ticker_stats = {}
    for t in trades:
        tk = t.ticker
        if tk not in ticker_stats:
            ticker_stats[tk] = {"trades": 0, "wins": 0, "pnl": 0.0}
        ticker_stats[tk]["trades"] += 1
        ticker_stats[tk]["pnl"] += t.pnl_dollars
        if t.pnl_dollars > 0:
            ticker_stats[tk]["wins"] += 1

    # By exit reason
    exit_stats = {}
    for t in trades:
        exit_stats[t.exit_reason] = exit_stats.get(t.exit_reason, 0) + 1

    # Daily P&L
    daily_pnl: Dict[date, float] = {}
    for t in trades:
        d = t.entry_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl_dollars

    best_day = max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
    worst_day = min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
    green_days = sum(1 for v in daily_pnl.values() if v > 0)
    red_days = sum(1 for v in daily_pnl.values() if v <= 0)

    # Monthly
    monthly_stats: Dict[str, dict] = {}
    for t in trades:
        month = t.entry_time.strftime("%Y-%m")
        if month not in monthly_stats:
            monthly_stats[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly_stats[month]["trades"] += 1
        monthly_stats[month]["pnl"] += t.pnl_dollars
        if t.pnl_dollars > 0:
            monthly_stats[month]["wins"] += 1

    # Time-of-day breakdown
    morning_trades = [t for t in trades if t.entry_time.time() < dtime(11, 0)]
    midday_trades = [t for t in trades if dtime(11, 0) <= t.entry_time.time() < dtime(15, 0)]
    power_trades = [t for t in trades if t.entry_time.time() >= dtime(15, 0)]

    # ── Print Report ──
    print("\n" + "=" * 70)
    print(f"  AGGRESSIVE SCALPER BACKTEST — {start_date} to {end_date}")
    print("=" * 70)

    print(f"\n{'Starting Capital:':<30} ${starting_capital:>12,.2f}")
    print(f"{'Ending Capital:':<30} ${equity[-1]:>12,.2f}")
    print(f"{'Net P&L:':<30} ${total_pnl:>12,.2f}")
    print(f"{'Return:':<30} {total_pnl / starting_capital * 100:>11.2f}%")

    months = max(1, (end_date - start_date).days / 30)
    monthly_return = (total_pnl / starting_capital * 100) / months
    print(f"{'Avg Monthly Return:':<30} {monthly_return:>11.2f}%")

    print(f"\n{'Total Trades:':<30} {len(trades):>12}")
    print(f"{'Winning Trades:':<30} {len(wins):>12}")
    print(f"{'Losing Trades:':<30} {len(losses):>12}")
    print(f"{'Win Rate:':<30} {win_rate:>11.1f}%")
    print(f"{'Profit Factor:':<30} {profit_factor:>12.2f}")

    print(f"\n{'Avg Win:':<30} ${avg_win:>12,.2f} ({avg_win_pct:>+.1f}%)")
    print(f"{'Avg Loss:':<30} ${avg_loss:>12,.2f} ({avg_loss_pct:>+.1f}%)")
    print(f"{'Avg Trade:':<30} ${avg_trade:>12,.2f}")
    print(f"{'Largest Win:':<30} ${largest_win:>12,.2f}")
    print(f"{'Largest Loss:':<30} ${largest_loss:>12,.2f}")

    print(f"\n{'Avg Hold Time:':<30} {avg_hold:>10.1f} min")
    print(f"{'Max Consecutive Wins:':<30} {max_consec_wins:>12}")
    print(f"{'Max Consecutive Losses:':<30} {max_consec_losses:>12}")
    print(f"{'Max Drawdown:':<30} ${max_dd:>12,.2f} ({max_dd_pct:.1f}%)")

    if best_day[0]:
        print(f"\n{'Best Day:':<30} {best_day[0]} (${best_day[1]:>+,.2f})")
        print(f"{'Worst Day:':<30} {worst_day[0]} (${worst_day[1]:>+,.2f})")
    print(f"{'Trading Days:':<30} {len(daily_pnl):>12}")
    print(f"{'Green Days:':<30} {green_days:>12}")
    print(f"{'Red Days:':<30} {red_days:>12}")
    print(f"{'Daily Win Rate:':<30} {green_days / max(1, len(daily_pnl)) * 100:>11.1f}%")

    # Trades per day
    if daily_pnl:
        avg_trades_per_day = len(trades) / len(daily_pnl)
        print(f"{'Avg Trades/Trading Day:':<30} {avg_trades_per_day:>12.1f}")

    # Strategy breakdown
    print("\n" + "-" * 70)
    print("  PERFORMANCE BY STRATEGY")
    print("-" * 70)
    print(f"  {'Strategy':<20} {'Trades':>8} {'Win%':>8} {'Net P&L':>12} {'PF':>8}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 12} {'─' * 8}")
    for strat, stats in sorted(strategy_stats.items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        pf = stats["gross_profit"] / abs(stats["gross_loss"]) if stats["gross_loss"] != 0 else float("inf")
        print(f"  {strat:<20} {stats['trades']:>8} {wr:>7.1f}% ${stats['pnl']:>11,.2f} {pf:>8.2f}")

    # Ticker breakdown
    print("\n" + "-" * 70)
    print("  PERFORMANCE BY TICKER")
    print("-" * 70)
    print(f"  {'Ticker':<10} {'Trades':>8} {'Win%':>8} {'Net P&L':>12}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 12}")
    for tk, stats in sorted(ticker_stats.items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"  {tk:<10} {stats['trades']:>8} {wr:>7.1f}% ${stats['pnl']:>11,.2f}")

    # Time of day breakdown
    print("\n" + "-" * 70)
    print("  PERFORMANCE BY TIME OF DAY")
    print("-" * 70)
    for label, group in [("Morning (9:45-11:00)", morning_trades),
                         ("Midday (11:00-3:00)", midday_trades),
                         ("Power Hour (3:00+)", power_trades)]:
        if group:
            g_wins = sum(1 for t in group if t.pnl_dollars > 0)
            g_pnl = sum(t.pnl_dollars for t in group)
            g_wr = g_wins / len(group) * 100
            print(f"  {label:<25} {len(group):>5} trades  {g_wr:>5.1f}% win  ${g_pnl:>+10,.2f}")

    # Exit reasons
    print("\n" + "-" * 70)
    print("  EXIT REASON BREAKDOWN")
    print("-" * 70)
    for reason, count in sorted(exit_stats.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"  {reason:<25} {count:>5} ({pct:>5.1f}%)")

    # Monthly breakdown
    print("\n" + "-" * 70)
    print("  MONTHLY BREAKDOWN")
    print("-" * 70)
    print(f"  {'Month':<12} {'Trades':>8} {'Win%':>8} {'Net P&L':>12} {'Return':>10}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 12} {'─' * 10}")
    for month, stats in sorted(monthly_stats.items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        ret = stats["pnl"] / starting_capital * 100
        print(f"  {month:<12} {stats['trades']:>8} {wr:>7.1f}% ${stats['pnl']:>11,.2f} {ret:>+9.2f}%")

    print("\n" + "=" * 70)

    # Save CSV
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trades_path = reports_dir / f"scalper_trades_{timestamp}.csv"
    with open(trades_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker", "direction", "strategy", "entry_time", "exit_time",
            "entry_price", "exit_price", "contracts", "delta",
            "underlying_entry", "underlying_exit",
            "pnl_pct", "pnl_dollars", "exit_reason", "strength", "hold_min",
        ])
        for t in trades:
            writer.writerow([
                t.ticker, t.direction, t.strategy,
                t.entry_time.isoformat(), t.exit_time.isoformat(),
                f"{t.entry_price:.4f}", f"{t.exit_price:.4f}",
                t.num_contracts, f"{t.delta:.3f}",
                f"{t.underlying_entry:.2f}", f"{t.underlying_exit:.2f}",
                f"{t.pnl_pct:.4f}", f"{t.pnl_dollars:.2f}",
                t.exit_reason, f"{t.strength_score:.0f}", f"{t.hold_minutes:.1f}",
            ])
    print(f"\nTrades CSV saved: {trades_path}")

    # Equity curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(equity, linewidth=1.5, color="#FF6B00")
        ax1.axhline(y=starting_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e >= starting_capital for e in equity],
                        alpha=0.15, color="green")
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e < starting_capital for e in equity],
                        alpha=0.15, color="red")
        ax1.set_title(f"Aggressive Scalper Equity — {start_date} to {end_date}",
                      fontsize=14, fontweight="bold")
        ax1.set_ylabel("Account Value ($)")
        ax1.set_xlabel("Trade #")
        ax1.grid(True, alpha=0.3)

        colors = ["green" if t.pnl_dollars > 0 else "red" for t in trades]
        ax2.bar(range(len(trades)), [t.pnl_dollars for t in trades], color=colors, alpha=0.7)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_title("Individual Trade P&L", fontsize=12)
        ax2.set_ylabel("P&L ($)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = reports_dir / f"scalper_equity_{timestamp}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Equity curve saved: {chart_path}")
    except ImportError:
        pass


# ============================================================
# Main
# ============================================================

async def run_scalper_backtest(
    start_date: date,
    end_date: date,
    tickers: List[str],
    from_cache: bool = False,
) -> None:
    """Run the aggressive scalper backtest."""
    config = ScalperConfig()

    print(f"\nAggressive 0DTE Scalper — Backtester")
    print(f"Period: {start_date} to {end_date}")
    print(f"Tickers: {tickers}")
    print(f"Starting Capital: ${config.account_value:,.0f}")
    print(f"Risk/Trade: {config.risk_per_trade_pct:.0%} (${config.account_value * config.risk_per_trade_pct:,.0f})")
    print(f"Stop: -{config.stop_loss_pct:.0%} | TP: +{config.take_profit_pct:.0%} | Trail: +{config.trail_activation_pct:.0%}/{config.trail_distance_pct:.0%}")
    print(f"Max hold: {config.max_hold_minutes:.0f} min")
    print(f"Data source: {'Cache' if from_cache else 'IBKR TWS'}")
    print()

    # Load data (reuse existing infrastructure)
    all_bars = await load_data(tickers, start_date, end_date, from_cache)

    daily_by_ticker: Dict[str, Dict[date, dict]] = {}
    for ticker, bars in all_bars.items():
        daily_by_ticker[ticker] = build_daily_bars(bars)

    all_dates = set()
    for ticker_daily in daily_by_ticker.values():
        all_dates.update(ticker_daily.keys())
    sorted_dates = sorted(all_dates)

    if not sorted_dates:
        print("No trading data found.")
        return

    print(f"Found {len(sorted_dates)} trading days\n")

    all_trades: List[TradeRecord] = []
    current_account = config.account_value

    for i, replay_date in enumerate(sorted_dates):
        today_bars: Dict[str, List[Bar]] = {}
        for ticker, bars in all_bars.items():
            day_bars = [b for b in bars if b.timestamp.date() == replay_date]
            if day_bars:
                today_bars[ticker] = day_bars

        if not today_bars:
            continue

        prior_days: Dict[str, List[dict]] = {}
        for ticker, daily in daily_by_ticker.items():
            prior = [daily[d] for d in sorted(daily.keys()) if d < replay_date]
            prior_days[ticker] = prior

        day_trades, current_account = replay_day_scalper(
            replay_date, today_bars, prior_days, config, current_account,
        )

        if day_trades:
            all_trades.extend(day_trades)
            day_pnl = sum(t.pnl_dollars for t in day_trades)
            print(f"  {replay_date} ({replay_date.strftime('%a')}): "
                  f"{len(day_trades)} trades, P&L: ${day_pnl:>+,.2f} "
                  f"(Account: ${current_account:,.2f})")
        else:
            if (i + 1) % 10 == 0:
                print(f"  {replay_date}: no signals (day {i + 1}/{len(sorted_dates)})")

    generate_scalper_report(all_trades, config.account_value, start_date, end_date)


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive 0DTE Scalper Backtester"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", default=["QQQ", "NVDA"],
                        help="Tickers to backtest (default: QQQ NVDA)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Load data from CSV cache")

    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    asyncio.run(run_scalper_backtest(start, end, args.tickers, args.from_cache))


if __name__ == "__main__":
    main()
