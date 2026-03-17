"""
Backtester for Bot 3 — 0DTE Mean Reversion Bot.

Fades overextended intraday moves on SPY using:
  - Bollinger Band touches/breaches (20-period, 2.0 SD on 3-min bars)
  - RSI extremes (7-period on 3-min bars)
  - VWAP deviation (price vs VWAP ± bands)
  - ADX filter (only enter when market is range-bound, ADX < threshold)
  - Volume exhaustion (declining volume on the extended move)

Entry logic:
  BUY CALL when price touches/breaks lower BB + RSI < oversold + below VWAP lower band
  BUY PUT  when price touches/breaks upper BB + RSI > overbought + above VWAP upper band

Exit logic (tighter than momentum — mean reversion trades are faster):
  - Take profit: 20% (reversion snaps are quick)
  - Stop loss: 15% (tight — if it keeps trending, we're wrong)
  - Trailing stop: activate at +12%, trail 8%
  - Time stop: 15 min (reversion should happen fast or not at all)
  - Break-even stop: activate at +8%, exit if drops back to 0%

Usage:
    python backtest_reversion.py --start 2025-09-15 --end 2026-03-13 --from-cache
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.market_state import (
    Bar, TickerState, MarketState, Signal, Position,
    Regime, Direction, Bias, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.volume_profile import VolumeProfileCalculator

ET = ZoneInfo("US/Eastern")
logger = logging.getLogger("backtest_reversion")


# ============================================================
# Configuration — Bot 3 Mean Reversion Parameters
# ============================================================

@dataclass
class ReversionConfig:
    """All tunable parameters for the mean reversion bot."""

    # --- Entry Filters (v6 LOCKED) ---
    bb_period: int = 20              # Bollinger Band lookback (3-min bars)
    bb_std: float = 1.8              # BB standard deviations (v6: 1.8)
    rsi_period: int = 7              # RSI period
    rsi_oversold: float = 35.0       # RSI below this → oversold (v6: 35)
    rsi_overbought: float = 65.0     # RSI above this → overbought (v6: 65)
    adx_max: float = 30.0            # Only enter when ADX < this (v6: 30)
    vwap_sd_min: float = 1.0         # Min VWAP SD from mean (v6: 1.0)
    volume_exhaustion: bool = True   # Require declining volume on the move
    min_strength: float = 30.0       # Minimum signal strength (v6: 30)
    calls_only: bool = True          # Calls only — buy-the-dip (v6: True)

    # --- Schedule ---
    earliest_entry: str = "09:50"    # Wait for indicators to warm up
    last_entry: str = "15:30"
    force_close: str = "15:55"

    # --- Risk / Exits ---
    simulated_account_value: float = 25000.0
    max_trade_risk_pct: float = 0.03       # 3% per trade (conservative)
    max_trades_per_day: int = 8
    daily_loss_limit_pct: float = 0.03     # -3% circuit breaker

    stop_loss_pct: float = 0.15            # 15% stop loss (v6)
    take_profit_pct: float = 0.25          # 25% take profit (v6: was 20%)
    trailing_activation_pct: float = 0.12  # Activate trail at +12%
    trailing_distance_pct: float = 0.08    # Trail 8% from peak
    breakeven_activation_pct: float = 0.08 # BE stop activates at +8%
    time_stop_minutes: float = 15          # Max hold 15 min (v6)

    # --- Cooldowns ---
    cooldown_after_trade_sec: int = 60     # 1 min after any trade
    cooldown_after_loss_sec: int = 120     # 2 min after a loss
    consecutive_loss_cooldown_sec: int = 300  # 5 min after 3 losses

    # --- Contract ---
    delta_target: float = 0.45             # Higher delta for reversion (v6: 0.45)
                                           # (closer to ATM = more responsive)


# ============================================================
# Bollinger Band Calculator
# ============================================================

class BollingerBandCalculator:
    """Compute Bollinger Bands from closing prices."""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def calculate(self, closes: List[float]) -> Dict[str, float]:
        """
        Returns: {middle, upper, lower, bandwidth, pct_b}
        pct_b: 0 = at lower band, 1 = at upper band, <0 or >1 = outside bands
        """
        if len(closes) < self.period:
            return {"middle": 0, "upper": 0, "lower": 0, "bandwidth": 0, "pct_b": 0.5}

        window = closes[-self.period:]
        middle = np.mean(window)
        std = np.std(window, ddof=1)  # sample std dev

        upper = middle + self.num_std * std
        lower = middle - self.num_std * std
        bandwidth = (upper - lower) / middle if middle > 0 else 0

        # %B: where price is relative to bands
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
# Mean Reversion Signal Generator
# ============================================================

def evaluate_reversion_signal(
    ts: TickerState,
    bb: Dict[str, float],
    config: ReversionConfig,
) -> Optional[Signal]:
    """
    Evaluate mean reversion conditions.
    Returns Signal if all conditions met, None otherwise.
    """
    price = ts.last_price
    if price <= 0 or bb["middle"] <= 0 or ts.vwap <= 0:
        return None

    # --- Determine direction ---
    direction = None
    pct_b = bb["pct_b"]

    # OVERSOLD: price at or below lower BB + RSI oversold
    if pct_b <= 0.05 and ts.rsi_7 <= config.rsi_oversold:
        direction = Direction.CALL  # Buy call, expecting bounce

    # OVERBOUGHT: price at or above upper BB + RSI overbought
    elif pct_b >= 0.95 and ts.rsi_7 >= config.rsi_overbought:
        direction = Direction.PUT   # Buy put, expecting fade

    if direction is None:
        return None

    # --- Calls-only mode (buy-the-dip) ---
    if config.calls_only and direction != Direction.CALL:
        return None

    # --- ADX filter: only trade in range-bound markets ---
    if ts.adx_14 > config.adx_max:
        return None  # Strong trend — don't fade it

    # --- VWAP deviation confirmation ---
    vwap_range = ts.vwap_upper_band - ts.vwap
    if vwap_range <= 0:
        return None
    price_vs_vwap_sd = (price - ts.vwap) / vwap_range * 2.0

    if direction == Direction.CALL and price_vs_vwap_sd > -config.vwap_sd_min:
        return None  # Not far enough below VWAP
    if direction == Direction.PUT and price_vs_vwap_sd < config.vwap_sd_min:
        return None  # Not far enough above VWAP

    # --- Compute strength score ---
    strength = compute_reversion_strength(ts, bb, direction, config)
    if strength < config.min_strength:
        return None

    return Signal(
        ticker=ts.ticker,
        direction=direction,
        strategy="MEAN_REVERSION",
        regime=Regime.REVERSION,
        strength_score=strength,
        entry_price_target=price,
    )


def compute_reversion_strength(
    ts: TickerState,
    bb: Dict[str, float],
    direction: Direction,
    config: ReversionConfig,
) -> float:
    """Compute signal strength 0-100 based on multiple confirmations."""
    score = 0.0

    # 1. BB extremity — how far outside/at the band (max 25)
    pct_b = bb["pct_b"]
    if direction == Direction.CALL:
        # Lower the pct_b, the more extreme (below lower band = negative pct_b)
        extremity = max(0, (0.05 - pct_b) / 0.15)  # 0 at pct_b=0.05, 1 at pct_b=-0.10
    else:
        extremity = max(0, (pct_b - 0.95) / 0.15)
    score += min(extremity, 1.0) * 25

    # 2. RSI extremity (max 25)
    if direction == Direction.CALL:
        rsi_ext = max(0, (config.rsi_oversold - ts.rsi_7) / 15.0)
    else:
        rsi_ext = max(0, (ts.rsi_7 - config.rsi_overbought) / 15.0)
    score += min(rsi_ext, 1.0) * 25

    # 3. VWAP deviation magnitude (max 20)
    vwap_range = ts.vwap_upper_band - ts.vwap
    if vwap_range > 0:
        vwap_sd = abs((ts.last_price - ts.vwap) / vwap_range * 2.0)
        vwap_bonus = min((vwap_sd - 1.5) / 1.0, 1.0) * 20
        score += max(vwap_bonus, 0)

    # 4. Volume exhaustion — declining volume = selling/buying pressure fading (max 15)
    if config.volume_exhaustion:
        if ts.volume_ratio < 0.8:
            score += 15  # Volume clearly declining — pressure fading
        elif ts.volume_ratio < 1.0:
            score += 10
        elif ts.volume_ratio < 1.2:
            score += 5
        # High volume on extreme move = trend continuation, NOT reversion → no points

    # 5. ADX confirms range-bound (max 15)
    if ts.adx_14 < 18:
        score += 15
    elif ts.adx_14 < 22:
        score += 10
    elif ts.adx_14 < config.adx_max:
        score += 5

    return min(score, 100.0)


# ============================================================
# Data Loading (reuse from backtest.py)
# ============================================================

CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"


def load_bars_from_csv(ticker: str, start: date, end: date) -> Optional[List[Bar]]:
    """Load bars from CSV cache."""
    path = CACHE_DIR / f"{ticker}_{start.isoformat()}_{end.isoformat()}.csv"
    if not path.exists():
        # Try to find any cache file that covers our range
        for f in CACHE_DIR.glob(f"{ticker}_*.csv"):
            bars = _load_csv(f)
            if bars:
                # Filter to date range
                filtered = [b for b in bars if start <= b.timestamp.date() <= end]
                if filtered:
                    logger.info(f"Loaded {len(filtered)} bars for {ticker} from {f.name}")
                    return filtered
        return None

    return _load_csv(path)


def _load_csv(path: Path) -> List[Bar]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ET)
            bars.append(Bar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return bars


def aggregate_3m_bars(bars_1m: List[Bar]) -> List[Bar]:
    """Aggregate 1-min bars into 3-min bars."""
    bars_3m = []
    for i in range(0, len(bars_1m) - 2, 3):
        chunk = bars_1m[i:i + 3]
        if len(chunk) < 3:
            break
        bars_3m.append(Bar(
            timestamp=chunk[0].timestamp,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
        ))
    return bars_3m


def estimate_option_premium(underlying_price: float, delta: float) -> float:
    """Estimate 0DTE SPY option premium."""
    premium = underlying_price * 0.003 * (delta / 0.35)
    return max(premium, 0.10)


# ============================================================
# Backtest Market State
# ============================================================

class BacktestMarketState(MarketState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim_now: Optional[datetime] = None

    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        now = self.sim_now or datetime.now()
        return now < self.cooldown_until


# ============================================================
# Trade Record
# ============================================================

@dataclass
class TradeRecord:
    ticker: str
    direction: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    num_contracts: int
    delta: float
    underlying_entry: float
    underlying_exit: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str
    strength_score: float
    hold_minutes: float
    rsi_at_entry: float = 0.0
    bb_pct_b_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    vwap_sd_at_entry: float = 0.0


# ============================================================
# Day Replay Engine
# ============================================================

def replay_day(
    replay_date: date,
    spy_bars: List[Bar],
    config: ReversionConfig,
    account_value: float,
) -> Tuple[List[TradeRecord], float]:
    """Replay a single trading day for SPY mean reversion."""

    market_state = BacktestMarketState()
    market_state.today = replay_date
    market_state.day_of_week = replay_date.strftime("%A")

    # Indicators
    vwap_calc = VWAPCalculator()
    rsi_calc = RSICalculator(period=config.rsi_period)
    adx_calc = ADXCalculator(period=14)
    vol_calc = VolumeProfileCalculator(lookback=20)
    bb_calc = BollingerBandCalculator(period=config.bb_period, num_std=config.bb_std)

    # Schedule
    earliest_entry = _parse_time(config.earliest_entry)
    last_entry = _parse_time(config.last_entry)
    force_close = _parse_time(config.force_close)

    # Init ticker state
    ts = market_state.get_ticker("SPY")

    # Sort bars
    sorted_bars = sorted(spy_bars, key=lambda b: b.timestamp)

    # State
    open_position: Optional[dict] = None
    trades: List[TradeRecord] = []
    current_account = account_value

    # Current BB data
    bb_data = {"middle": 0, "upper": 0, "lower": 0, "bandwidth": 0, "pct_b": 0.5}

    for bar in sorted_bars:
        # Convert to ET
        bar_time_et = bar.timestamp.astimezone(ET) if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=ET)
        market_state.sim_now = bar_time_et

        # Append 1-min bar (ET)
        bar_et = Bar(
            timestamp=bar_time_et,
            open=bar.open, high=bar.high, low=bar.low,
            close=bar.close, volume=bar.volume,
        )
        ts.bars_1m.append(bar_et)
        ts.last_price = bar.close
        ts.last_bar_time = bar_time_et

        # --- Update Indicators ---
        # VWAP
        if len(ts.bars_1m) >= 2:
            vwap_data = vwap_calc.calculate(ts.bars_1m)
            ts.vwap = vwap_data["vwap"]
            ts.vwap_upper_band = vwap_data["upper_band"]
            ts.vwap_lower_band = vwap_data["lower_band"]
            ts.vwap_slope = vwap_data["slope"]

        # 3-min bars for RSI, ADX, volume, BB
        ts.bars_3m = aggregate_3m_bars(ts.bars_1m)
        if ts.bars_3m:
            closes_3m = [b.close for b in ts.bars_3m]
            highs_3m = [b.high for b in ts.bars_3m]
            lows_3m = [b.low for b in ts.bars_3m]
            volumes_3m = [b.volume for b in ts.bars_3m]

            ts.rsi_7 = rsi_calc.calculate(closes_3m)
            ts.adx_14 = adx_calc.calculate(highs_3m, lows_3m, closes_3m)
            ts.volume_ratio = vol_calc.calculate(volumes_3m)

            # Bollinger Bands on 3-min closes
            bb_data = bb_calc.calculate(closes_3m)

        # --- Check Exits ---
        if open_position is not None:
            pos = open_position["position"]
            delta = open_position["delta"]
            underlying_entry = open_position["underlying_entry"]
            underlying_now = ts.last_price

            # Delta-estimated option price
            if pos.direction == Direction.CALL:
                option_price = pos.entry_price + (underlying_now - underlying_entry) * delta
            else:
                option_price = pos.entry_price + (underlying_entry - underlying_now) * delta

            option_price = max(option_price, 0.01)
            pos.current_price = option_price
            pos.unrealized_pnl_pct = (option_price - pos.entry_price) / pos.entry_price
            pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
            pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

            # Break-even stop activation
            if (not pos.breakeven_stop_active
                    and pos.max_favorable >= config.breakeven_activation_pct):
                pos.breakeven_stop_active = True

            # Trailing stop management
            if not pos.trailing_active and pos.unrealized_pnl_pct >= config.trailing_activation_pct:
                pos.trailing_active = True
                pos.trailing_peak = option_price
            elif pos.trailing_active and option_price > pos.trailing_peak:
                pos.trailing_peak = option_price

            # Check exit conditions
            exit_reason = check_exit(pos, market_state, bar_time_et, config)

            if exit_reason is not None:
                hold_mins = (bar_time_et - pos.entry_time).total_seconds() / 60.0
                pnl_pct = pos.unrealized_pnl_pct
                pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts

                trades.append(TradeRecord(
                    ticker="SPY",
                    direction=pos.direction.value,
                    strategy="MEAN_REVERSION",
                    entry_time=pos.entry_time,
                    exit_time=bar_time_et,
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
                    rsi_at_entry=pos.rsi_value,
                    bb_pct_b_at_entry=open_position.get("bb_pct_b", 0),
                    adx_at_entry=pos.adx_value,
                    vwap_sd_at_entry=open_position.get("vwap_sd", 0),
                ))

                current_account += pnl_dollars
                market_state.daily_pnl += pnl_dollars
                market_state.trades_today += 1

                if pnl_pct < 0:
                    market_state.consecutive_losses += 1
                    if market_state.consecutive_losses >= 3:
                        cooldown = config.consecutive_loss_cooldown_sec
                    else:
                        cooldown = config.cooldown_after_loss_sec
                else:
                    market_state.consecutive_losses = 0
                    cooldown = config.cooldown_after_trade_sec

                market_state.cooldown_until = bar_time_et + timedelta(seconds=cooldown)

                # Circuit breaker
                if market_state.daily_pnl / config.simulated_account_value <= -config.daily_loss_limit_pct:
                    market_state.circuit_breaker_triggered = True

                open_position = None

        # --- Look for new entries ---
        bar_t = bar_time_et.time()
        if (open_position is None
                and bar_t >= earliest_entry
                and bar_t <= last_entry
                and not market_state.circuit_breaker_triggered
                and not market_state.is_in_cooldown()
                and market_state.trades_today < config.max_trades_per_day):

            signal = evaluate_reversion_signal(ts, bb_data, config)

            if signal is not None:
                delta = config.delta_target
                premium = estimate_option_premium(ts.last_price, delta)

                # Size position: risk-based
                stop_risk = premium * config.stop_loss_pct
                max_risk_dollars = config.simulated_account_value * config.max_trade_risk_pct
                num_contracts = max(1, int(max_risk_dollars / (stop_risk * 100)))
                num_contracts = min(num_contracts, 10)  # Cap at 10

                # Compute VWAP SD for logging
                vwap_range = ts.vwap_upper_band - ts.vwap
                vwap_sd = (ts.last_price - ts.vwap) / vwap_range * 2.0 if vwap_range > 0 else 0

                pos = Position(
                    ticker="SPY",
                    direction=signal.direction,
                    contract_type=ContractType.SINGLE_LEG,
                    strategy="MEAN_REVERSION",
                    regime=Regime.REVERSION.value,
                    signal=signal,
                    delta_at_entry=delta,
                    entry_time=bar_time_et,
                    entry_price=premium,
                    num_contracts=num_contracts,
                    current_price=premium,
                    adx_value=ts.adx_14,
                    rsi_value=ts.rsi_7,
                    volume_ratio=ts.volume_ratio,
                    signal_strength_score=signal.strength_score,
                )

                open_position = {
                    "position": pos,
                    "delta": delta,
                    "underlying_entry": ts.last_price,
                    "bb_pct_b": bb_data["pct_b"],
                    "vwap_sd": vwap_sd,
                }

    # --- Force close any remaining position at EOD ---
    if open_position is not None:
        pos = open_position["position"]
        delta = open_position["delta"]
        underlying_entry = open_position["underlying_entry"]
        underlying_now = ts.last_price

        if pos.direction == Direction.CALL:
            option_price = pos.entry_price + (underlying_now - underlying_entry) * delta
        else:
            option_price = pos.entry_price + (underlying_entry - underlying_now) * delta

        option_price = max(option_price, 0.01)
        pnl_pct = (option_price - pos.entry_price) / pos.entry_price
        pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts
        hold_mins = (sorted_bars[-1].timestamp.astimezone(ET) - pos.entry_time).total_seconds() / 60.0

        trades.append(TradeRecord(
            ticker="SPY",
            direction=pos.direction.value,
            strategy="MEAN_REVERSION",
            entry_time=pos.entry_time,
            exit_time=sorted_bars[-1].timestamp.astimezone(ET),
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
            rsi_at_entry=pos.rsi_value,
            bb_pct_b_at_entry=open_position.get("bb_pct_b", 0),
            adx_at_entry=pos.adx_value,
            vwap_sd_at_entry=open_position.get("vwap_sd", 0),
        ))
        current_account += pnl_dollars

    return trades, current_account


# ============================================================
# Exit Logic
# ============================================================

def check_exit(
    position: Position,
    market_state: BacktestMarketState,
    sim_now: datetime,
    config: ReversionConfig,
) -> Optional[ExitReason]:
    """Check all exit conditions."""

    # Circuit breaker
    if market_state.circuit_breaker_triggered:
        return ExitReason.CIRCUIT_BREAKER

    # EOD
    fc = config.force_close.split(":")
    force_close_time = dtime(int(fc[0]), int(fc[1]))
    if sim_now.time() >= force_close_time:
        return ExitReason.EOD_CLOSE

    # Stop loss
    if position.unrealized_pnl_pct <= -config.stop_loss_pct:
        return ExitReason.STOP_LOSS

    # Break-even stop
    if position.breakeven_stop_active and position.unrealized_pnl_pct <= 0.0:
        return ExitReason.BREAKEVEN_STOP

    # Time stop
    if position.entry_time:
        elapsed = (sim_now - position.entry_time).total_seconds() / 60.0
        if elapsed >= config.time_stop_minutes:
            return ExitReason.TIME_STOP

    # Trailing stop
    if position.trailing_active and position.trailing_peak > 0:
        drop = (position.current_price - position.trailing_peak) / position.trailing_peak
        if drop <= -config.trailing_distance_pct:
            return ExitReason.TRAILING_STOP

    # Take profit
    if position.unrealized_pnl_pct >= config.take_profit_pct:
        return ExitReason.TAKE_PROFIT

    return None


# ============================================================
# Report Generation
# ============================================================

def generate_report(
    trades: List[TradeRecord],
    starting_capital: float,
    start_date: date,
    end_date: date,
) -> None:
    """Generate backtest report."""
    if not trades:
        print("\n" + "=" * 70)
        print("  BOT 3 MEAN REVERSION — BACKTEST REPORT — NO TRADES")
        print("=" * 70)
        print("\nNo qualifying reversion signals found in this period.")
        print("Possible causes:")
        print("  - Bollinger Bands not extreme enough (BB std too tight)")
        print("  - RSI thresholds too strict")
        print("  - ADX max too low (filtering out valid setups)")
        print("  - VWAP deviation requirement too high")
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
    largest_win = max(t.pnl_dollars for t in trades)
    largest_loss = min(t.pnl_dollars for t in trades)
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0

    # Trading days & frequency
    trading_days = len(set(t.entry_time.date() for t in trades))
    trades_per_day = len(trades) / trading_days if trading_days > 0 else 0

    # Equity curve
    equity = [starting_capital]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollars)

    # Max drawdown
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

    # Exit reasons
    exit_stats = {}
    exit_pnl = {}
    for t in trades:
        r = t.exit_reason
        exit_stats[r] = exit_stats.get(r, 0) + 1
        exit_pnl[r] = exit_pnl.get(r, 0) + t.pnl_dollars

    # Direction breakdown
    calls = [t for t in trades if t.direction == "CALL"]
    puts = [t for t in trades if t.direction == "PUT"]

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

    # Entry quality analysis
    avg_rsi_entry = np.mean([t.rsi_at_entry for t in trades])
    avg_bb_entry = np.mean([t.bb_pct_b_at_entry for t in trades])
    avg_adx_entry = np.mean([t.adx_at_entry for t in trades])

    # ── Print Report ──
    print("\n" + "=" * 70)
    print(f"  BOT 3 — 0DTE MEAN REVERSION BACKTEST — {start_date} to {end_date}")
    print("=" * 70)

    print(f"\n{'Starting Capital:':<30} ${starting_capital:>12,.2f}")
    print(f"{'Ending Capital:':<30} ${equity[-1]:>12,.2f}")
    print(f"{'Net P&L:':<30} ${total_pnl:>12,.2f}")
    print(f"{'Return:':<30} {total_pnl / starting_capital * 100:>11.2f}%")

    print(f"\n{'Total Trades:':<30} {len(trades):>12}")
    print(f"{'Trades/Day:':<30} {trades_per_day:>12.1f}")
    print(f"{'Trading Days:':<30} {trading_days:>12}")
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
    print(f"{'Green Days:':<30} {green_days:>12}")
    print(f"{'Red Days:':<30} {red_days:>12}")

    # Direction breakdown
    print("\n" + "-" * 70)
    print("  DIRECTION BREAKDOWN")
    print("-" * 70)
    if calls:
        call_wins = [t for t in calls if t.pnl_dollars > 0]
        call_pnl = sum(t.pnl_dollars for t in calls)
        print(f"  {'CALLS':<15} {len(calls):>5} trades, "
              f"{len(call_wins)/len(calls)*100:>5.1f}% win, "
              f"${call_pnl:>+10,.2f}")
    if puts:
        put_wins = [t for t in puts if t.pnl_dollars > 0]
        put_pnl = sum(t.pnl_dollars for t in puts)
        print(f"  {'PUTS':<15} {len(puts):>5} trades, "
              f"{len(put_wins)/len(puts)*100:>5.1f}% win, "
              f"${put_pnl:>+10,.2f}")

    # Exit reasons
    print("\n" + "-" * 70)
    print("  EXIT REASON BREAKDOWN")
    print("-" * 70)
    print(f"  {'Reason':<25} {'Count':>6} {'%':>7} {'P&L':>12}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 7} {'─' * 12}")
    for reason, count in sorted(exit_stats.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        pnl = exit_pnl.get(reason, 0)
        print(f"  {reason:<25} {count:>6} {pct:>6.1f}% ${pnl:>11,.2f}")

    # Entry quality
    print("\n" + "-" * 70)
    print("  ENTRY CONDITIONS (AVERAGES)")
    print("-" * 70)
    print(f"  {'RSI at Entry:':<25} {avg_rsi_entry:>8.1f}")
    print(f"  {'BB %B at Entry:':<25} {avg_bb_entry:>8.3f}")
    print(f"  {'ADX at Entry:':<25} {avg_adx_entry:>8.1f}")

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

    # Save trades CSV
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = reports_dir / f"reversion_backtest_{timestamp}.csv"
    with open(trades_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker", "direction", "strategy", "entry_time", "exit_time",
            "entry_price", "exit_price", "contracts", "delta",
            "underlying_entry", "underlying_exit",
            "pnl_pct", "pnl_dollars", "exit_reason", "strength",
            "hold_min", "rsi_entry", "bb_pct_b", "adx_entry", "vwap_sd",
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
                f"{t.rsi_at_entry:.1f}", f"{t.bb_pct_b_at_entry:.3f}",
                f"{t.adx_at_entry:.1f}", f"{t.vwap_sd_at_entry:.2f}",
            ])
    print(f"\nTrades CSV: {trades_path}")

    # Equity curve chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(equity, linewidth=1.5, color="#9C27B0")  # Purple for reversion
        ax1.axhline(y=starting_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e >= starting_capital for e in equity],
                        alpha=0.15, color="green")
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e < starting_capital for e in equity],
                        alpha=0.15, color="red")
        ax1.set_title(f"Bot 3 Mean Reversion — Equity Curve ({start_date} to {end_date})",
                      fontsize=14, fontweight="bold")
        ax1.set_ylabel("Account Value ($)")
        ax1.grid(True, alpha=0.3)

        colors = ["green" if t.pnl_dollars > 0 else "red" for t in trades]
        ax2.bar(range(len(trades)), [t.pnl_dollars for t in trades], color=colors, alpha=0.7)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_title("Individual Trade P&L", fontsize=12)
        ax2.set_ylabel("P&L ($)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = reports_dir / f"reversion_equity_{timestamp}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Equity curve: {chart_path}")
    except ImportError:
        pass


# ============================================================
# Helpers
# ============================================================

def _parse_time(t: str) -> dtime:
    parts = t.split(":")
    return dtime(int(parts[0]), int(parts[1]))


def build_daily_bars(bars: List[Bar]) -> Dict[date, dict]:
    """Group 1-min bars by date and compute daily OHLCV."""
    daily: Dict[date, list] = {}
    for b in bars:
        d = b.timestamp.date()
        daily.setdefault(d, []).append(b)

    result = {}
    for d, day_bars in daily.items():
        result[d] = {
            "date": d,
            "open": day_bars[0].open,
            "high": max(b.high for b in day_bars),
            "low": min(b.low for b in day_bars),
            "close": day_bars[-1].close,
            "volume": sum(b.volume for b in day_bars),
        }
    return result


# ============================================================
# Main
# ============================================================

async def run_backtest(
    start_date: date,
    end_date: date,
    from_cache: bool = False,
    config: Optional[ReversionConfig] = None,
) -> None:
    """Run the mean reversion backtest."""
    if config is None:
        config = ReversionConfig()

    print(f"\nBot 3 — 0DTE Mean Reversion Backtester")
    print(f"Period: {start_date} to {end_date}")
    print(f"Ticker: SPY only")
    print(f"Starting Capital: ${config.simulated_account_value:,.0f}")
    print(f"\nEntry Parameters:")
    print(f"  BB: {config.bb_period}-period, {config.bb_std} SD")
    print(f"  RSI: {config.rsi_period}-period, oversold<{config.rsi_oversold}, overbought>{config.rsi_overbought}")
    print(f"  ADX max: {config.adx_max} (range-bound only)")
    print(f"  VWAP min SD: {config.vwap_sd_min}")
    print(f"  Min strength: {config.min_strength}")
    print(f"\nExit Parameters:")
    print(f"  SL: {config.stop_loss_pct*100:.0f}% | TP: {config.take_profit_pct*100:.0f}%")
    print(f"  Trail: activate {config.trailing_activation_pct*100:.0f}%, distance {config.trailing_distance_pct*100:.0f}%")
    print(f"  BE stop: activate {config.breakeven_activation_pct*100:.0f}%")
    print(f"  Time stop: {config.time_stop_minutes:.0f} min")
    print(f"  Delta: {config.delta_target}")
    print()

    # Load SPY data
    spy_bars = load_bars_from_csv("SPY", start_date, end_date)
    if not spy_bars:
        print("ERROR: No cached SPY data found. Run Bot 1 backtest first to cache data.")
        return

    # Group by date
    daily_bars: Dict[date, List[Bar]] = {}
    for b in spy_bars:
        d = b.timestamp.date()
        if start_date <= d <= end_date:
            daily_bars.setdefault(d, []).append(b)

    sorted_dates = sorted(daily_bars.keys())
    print(f"Found {len(sorted_dates)} trading days\n")

    # Replay each day
    all_trades: List[TradeRecord] = []
    current_account = config.simulated_account_value

    for i, replay_date in enumerate(sorted_dates):
        day_trades, current_account = replay_day(
            replay_date,
            daily_bars[replay_date],
            config,
            current_account,
        )

        if day_trades:
            all_trades.extend(day_trades)
            day_pnl = sum(t.pnl_dollars for t in day_trades)
            print(f"  {replay_date} ({replay_date.strftime('%a')}): "
                  f"{len(day_trades)} trades, P&L: ${day_pnl:>+,.2f} "
                  f"(Account: ${current_account:,.2f})")
        elif (i + 1) % 20 == 0:
            print(f"  {replay_date}: no signals (day {i+1}/{len(sorted_dates)})")

    generate_report(all_trades, config.simulated_account_value, start_date, end_date)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Bot 3 — 0DTE Mean Reversion on SPY"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--from-cache", action="store_true", help="Use cached CSV data")

    # Tunable parameters via CLI
    parser.add_argument("--bb-period", type=int, default=20)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--rsi-oversold", type=float, default=30.0)
    parser.add_argument("--rsi-overbought", type=float, default=70.0)
    parser.add_argument("--adx-max", type=float, default=25.0)
    parser.add_argument("--vwap-sd-min", type=float, default=1.5)
    parser.add_argument("--min-strength", type=float, default=40.0)
    parser.add_argument("--stop-loss", type=float, default=0.15)
    parser.add_argument("--take-profit", type=float, default=0.20)
    parser.add_argument("--time-stop", type=float, default=15.0)
    parser.add_argument("--delta", type=float, default=0.42)
    parser.add_argument("--calls-only", action="store_true", help="Only buy calls (buy-the-dip)")

    args = parser.parse_args()

    config = ReversionConfig(
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        adx_max=args.adx_max,
        vwap_sd_min=args.vwap_sd_min,
        min_strength=args.min_strength,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        time_stop_minutes=args.time_stop,
        delta_target=args.delta,
        calls_only=args.calls_only,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    asyncio.run(run_backtest(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        args.from_cache,
        config,
    ))


if __name__ == "__main__":
    main()
