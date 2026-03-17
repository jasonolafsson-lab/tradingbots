"""
Backtesting Engine for Options Bot V1.

Replays historical 1-minute bars through the bot's exact strategy logic,
simulates options trades using delta-estimated P&L, and produces a
comprehensive performance report.

Usage:
    python backtest.py --start 2025-09-15 --end 2026-03-01 --tickers SPY QQQ NVDA TSLA
    python backtest.py --start 2025-09-15 --end 2026-03-01 --from-cache
    python backtest.py --start 2026-01-01 --end 2026-03-01 --tickers SPY --from-cache
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

# Add project root to path so we can import bot modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.market_state import (
    Bar, TickerState, MarketState, Signal, Position, ScannerResult,
    Regime, Direction, Bias, ContractType, ExitReason,
)
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.opening_range import OpeningRangeTracker
from indicators.volume_profile import VolumeProfileCalculator
from strategy.regime_engine import RegimeEngine
from strategy.momentum import MomentumStrategy
from strategy.reversion import ReversionStrategy
from strategy.day2 import Day2Strategy
from strategy.tuesday_bias import TuesdayBiasModifier
from strategy.green_sector import GreenSectorStrategy
from risk.sizing import PositionSizer
from risk.circuit_breaker import CircuitBreaker
from filters.trade_filters import TradeFilterManager

ET = ZoneInfo("US/Eastern")

logger = logging.getLogger("backtest")


# ============================================================
# Configuration
# ============================================================

def load_config() -> dict:
    """Load settings.yaml and tickers.yaml."""
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    tickers_path = PROJECT_ROOT / "config" / "tickers.yaml"

    with open(settings_path) as f:
        config = yaml.safe_load(f)

    if tickers_path.exists():
        with open(tickers_path) as f:
            config["tickers_config"] = yaml.safe_load(f)

    return config


# ============================================================
# Data Layer — Fetch & Cache Historical Bars
# ============================================================

CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"


def _cache_path(ticker: str, start: date, end: date) -> Path:
    return CACHE_DIR / f"{ticker}_{start.isoformat()}_{end.isoformat()}.csv"


def save_bars_to_csv(ticker: str, bars: List[Bar], start: date, end: date) -> None:
    """Save bars to CSV cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume])
    logger.info(f"Cached {len(bars)} bars for {ticker} -> {path.name}")


def load_bars_from_csv(ticker: str, start: date, end: date) -> Optional[List[Bar]]:
    """Load bars from CSV cache if available."""
    path = _cache_path(ticker, start, end)
    if not path.exists():
        return None

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

    logger.info(f"Loaded {len(bars)} cached bars for {ticker}")
    return bars


async def fetch_from_ibkr(
    tickers: List[str],
    start: date,
    end: date,
) -> Dict[str, List[Bar]]:
    """
    Fetch historical 1-min bars from IBKR TWS (port 7497).
    Chunks requests into 20-day blocks with 15-second pacing.
    """
    try:
        from ib_async import IB, Stock
    except ImportError:
        logger.error("ib_async not installed. Run: pip install ib_async")
        sys.exit(1)

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=50)
    logger.info("Connected to IBKR TWS for historical data")

    all_bars: Dict[str, List[Bar]] = {}

    for ticker in tickers:
        logger.info(f"Fetching {ticker} data from {start} to {end}...")
        contract = Stock(ticker, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)

        ticker_bars: List[Bar] = []
        chunk_end = end
        chunk_size = timedelta(days=28)  # ~20 trading days

        while chunk_end > start:
            chunk_start = max(start, chunk_end - chunk_size)
            end_dt_str = f"{chunk_end.strftime('%Y%m%d')} 23:59:59"
            duration_days = (chunk_end - chunk_start).days + 1
            duration_str = f"{duration_days} D"

            try:
                ibkr_bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt_str,
                    durationStr=duration_str,
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                )

                for b in ibkr_bars:
                    ts = b.date
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=ET)
                    ticker_bars.append(Bar(
                        timestamp=ts,
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume,
                    ))

                logger.info(f"  {ticker}: fetched {len(ibkr_bars)} bars "
                           f"(chunk ending {chunk_end})")

            except Exception as e:
                logger.warning(f"  {ticker} chunk error: {e}")

            chunk_end = chunk_start - timedelta(days=1)

            # Pacing: 15 second delay between requests
            if chunk_end > start:
                logger.info("  Pacing delay (15s)...")
                await asyncio.sleep(15)

        # Sort by timestamp and deduplicate
        ticker_bars.sort(key=lambda b: b.timestamp)
        seen = set()
        deduped = []
        for b in ticker_bars:
            key = b.timestamp.isoformat()
            if key not in seen:
                seen.add(key)
                deduped.append(b)

        all_bars[ticker] = deduped
        save_bars_to_csv(ticker, deduped, start, end)
        logger.info(f"  {ticker}: total {len(deduped)} bars")

        # Pacing between tickers
        await asyncio.sleep(5)

    ib.disconnect()
    return all_bars


async def load_data(
    tickers: List[str],
    start: date,
    end: date,
    from_cache: bool = False,
) -> Dict[str, List[Bar]]:
    """Load data from cache or fetch from IBKR."""
    all_bars: Dict[str, List[Bar]] = {}
    missing = []

    for ticker in tickers:
        cached = load_bars_from_csv(ticker, start, end)
        if cached:
            all_bars[ticker] = cached
        else:
            missing.append(ticker)

    if missing:
        if from_cache:
            logger.error(f"Cache miss for {missing}. Run without --from-cache first.")
            sys.exit(1)
        fetched = await fetch_from_ibkr(missing, start, end)
        all_bars.update(fetched)

    return all_bars


# ============================================================
# Helper: Aggregate 3-min bars from 1-min bars
# ============================================================

def aggregate_3m_bars(bars_1m: List[Bar]) -> List[Bar]:
    """Aggregate 1-min bars into 3-min bars for RSI/ADX/volume computation."""
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


# ============================================================
# Backtest Market State — Overrides datetime.now()
# ============================================================

class BacktestMarketState(MarketState):
    """MarketState with simulated time for cooldown checks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim_now: Optional[datetime] = None

    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        now = self.sim_now or datetime.now()
        return now < self.cooldown_until


# ============================================================
# Backtest Risk Manager — Uses simulated time
# ============================================================

class BacktestRiskManager:
    """Risk manager that uses simulated time instead of datetime.now()."""

    def __init__(self, config: dict):
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

        be = config.get("breakeven_stop", {})
        self.be_enabled = be.get("enabled", False)
        self.be_activation_pct = be.get("activation_pct", 0.07)

        time_stop = config.get("time_stop", {})
        self.max_hold_minutes = time_stop.get("max_hold_minutes", 30)

        sched = config.get("schedule", {})
        fc = sched.get("force_close", "15:55").split(":")
        self.force_close_time = dtime(int(fc[0]), int(fc[1]))

    def check_exit(
        self,
        position: Position,
        market_state: BacktestMarketState,
        sim_now: datetime,
    ) -> Optional[ExitReason]:
        """Check all exit conditions using simulated time."""
        # Circuit breaker
        if market_state.circuit_breaker_triggered:
            return ExitReason.CIRCUIT_BREAKER

        # EOD
        now_et = sim_now.time()
        if now_et >= self.force_close_time:
            return ExitReason.EOD_CLOSE

        # Stop loss
        if self._check_stop_loss(position, market_state):
            return ExitReason.STOP_LOSS

        # Break-even stop: once trade was up X%, exit if it drops back to breakeven
        if self._check_breakeven_stop(position):
            return ExitReason.BREAKEVEN_STOP

        # Time stop
        if self._check_time_stop(position, sim_now):
            return ExitReason.TIME_STOP

        # Trailing stop
        if self._check_trailing_stop(position):
            return ExitReason.TRAILING_STOP

        # Take profit
        if self._check_take_profit(position):
            return ExitReason.TAKE_PROFIT

        return None

    def _check_stop_loss(self, position: Position, market_state: BacktestMarketState) -> bool:
        if position.entry_price <= 0:
            return False
        stop_pct = self.sl_spread if position.contract_type == ContractType.DEBIT_SPREAD else self.sl_single
        ts = market_state.tickers.get(position.ticker)
        if ts and ts.atr_14 > 0 and ts.atr_20_avg > 0:
            if ts.atr_14 > self.vol_adj_threshold * ts.atr_20_avg:
                stop_pct += self.vol_adj_extra
        return position.unrealized_pnl_pct <= -stop_pct

    def _check_take_profit(self, position: Position) -> bool:
        tp = self.tp_spread if position.contract_type == ContractType.DEBIT_SPREAD else self.tp_single
        return position.unrealized_pnl_pct >= tp

    def _check_trailing_stop(self, position: Position) -> bool:
        if not position.trailing_active or position.trailing_peak <= 0:
            return False
        drop = (position.current_price - position.trailing_peak) / position.trailing_peak
        return drop <= -self.trail_distance

    def _check_breakeven_stop(self, position: Position) -> bool:
        """Exit at breakeven if trade was profitable but drifted back."""
        if not self.be_enabled:
            return False
        if not position.breakeven_stop_active:
            return False
        # If price has dropped back to entry (or below), exit
        return position.unrealized_pnl_pct <= 0.0

    def _check_time_stop(self, position: Position, sim_now: datetime) -> bool:
        if position.entry_time is None:
            return False
        entry = position.entry_time
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=ET)
        elapsed = (sim_now - entry).total_seconds() / 60.0
        return elapsed >= self.max_hold_minutes


# ============================================================
# Simulated Trade Record
# ============================================================

@dataclass
class TradeRecord:
    """Completed simulated trade."""
    ticker: str
    direction: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float        # option premium
    exit_price: float         # option premium at exit
    num_contracts: int
    delta: float
    underlying_entry: float
    underlying_exit: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str
    strength_score: float
    hold_minutes: float


# ============================================================
# Options Premium Estimator
# ============================================================

def estimate_option_premium(
    ticker: str,
    underlying_price: float,
    delta: float,
) -> float:
    """
    Estimate an option premium based on underlying price and delta.
    Simple empirical approximation for intraday 0DTE/short-dated options.
    """
    # ETFs (SPY, QQQ) have lower IV → cheaper premiums
    # Stocks (NVDA, TSLA) have higher IV → more expensive
    if ticker in ("SPY", "QQQ"):
        multiplier = 0.003
    else:
        multiplier = 0.006

    premium = underlying_price * multiplier * (delta / 0.35)
    return max(premium, 0.10)  # Floor at $0.10


# ============================================================
# Scanner Simulation (for Day2 strategy)
# ============================================================

def compute_scanner_result(
    ticker: str,
    prior_day: Optional[dict],
    daily_history: List[dict],
) -> Optional[ScannerResult]:
    """
    Compute a ScannerResult from prior day data.
    Simplified version of PreMarketScanner for backtest.
    """
    if prior_day is None:
        return None

    prior_high = prior_day["high"]
    prior_low = prior_day["low"]
    prior_close = prior_day["close"]
    prior_volume = prior_day["volume"]
    prior_open = prior_day["open"]

    if prior_high == prior_low or prior_close == 0:
        return None

    # Close quality
    close_quality = (prior_close - prior_low) / (prior_high - prior_low)

    # Volume vs 20-day average
    if len(daily_history) >= 20:
        avg_vol = np.mean([d["volume"] for d in daily_history[-20:]])
        volume_vs_avg = prior_volume / avg_vol if avg_vol > 0 else 1.0
    else:
        volume_vs_avg = 1.0

    # Session return
    session_return = (prior_close - prior_open) / prior_open if prior_open > 0 else 0

    # Day2 score: weighted composite
    score = 0.0
    # Strong close (max 30)
    if close_quality > 0.80:
        score += 30
    elif close_quality > 0.60:
        score += 15
    elif close_quality < 0.20:
        score += 30  # Strong bearish close
    elif close_quality < 0.40:
        score += 15

    # High volume (max 30)
    if volume_vs_avg > 2.0:
        score += 30
    elif volume_vs_avg > 1.5:
        score += 20
    elif volume_vs_avg > 1.2:
        score += 10

    # Session magnitude (max 20)
    abs_return = abs(session_return)
    if abs_return > 0.03:
        score += 20
    elif abs_return > 0.02:
        score += 15
    elif abs_return > 0.01:
        score += 10

    # Trend alignment with SMA20 (max 20)
    if len(daily_history) >= 20:
        sma20 = np.mean([d["close"] for d in daily_history[-20:]])
        if session_return > 0 and prior_close > sma20:
            score += 20
        elif session_return < 0 and prior_close < sma20:
            score += 20

    # Bias
    if close_quality > 0.60 and session_return > 0:
        bias = Bias.BULLISH
    elif close_quality < 0.40 and session_return < 0:
        bias = Bias.BEARISH
    else:
        bias = Bias.NEUTRAL

    return ScannerResult(
        ticker=ticker,
        bias=bias,
        day2_score=score,
        close_quality=close_quality,
        volume_vs_avg=volume_vs_avg,
        key_levels={
            "prior_high": prior_high,
            "prior_low": prior_low,
            "prior_close": prior_close,
        },
    )


# ============================================================
# Day Replay Engine
# ============================================================

def replay_day(
    replay_date: date,
    ticker_bars: Dict[str, List[Bar]],
    prior_days: Dict[str, List[dict]],
    config: dict,
    account_value: float,
) -> Tuple[List[TradeRecord], float]:
    """
    Replay a single trading day. Returns (trades, ending_account_value).
    """
    # Initialize components
    market_state = BacktestMarketState()
    market_state.today = replay_date
    market_state.day_of_week = replay_date.strftime("%A")

    # Indicators
    vwap_calc = VWAPCalculator()
    rsi_calc = RSICalculator(period=7)
    adx_calc = ADXCalculator(period=14)
    or_tracker = OpeningRangeTracker(config)
    vol_calc = VolumeProfileCalculator(lookback=20)

    # Strategies
    regime_engine = RegimeEngine(config)
    strategies = {
        "MOMENTUM": MomentumStrategy(config),
        "REVERSION": ReversionStrategy(config),
        "DAY2": Day2Strategy(config),
        "GREEN_SECTOR": GreenSectorStrategy(config),
    }
    tuesday_bias = TuesdayBiasModifier(config)

    # Risk & Sizing
    risk_manager = BacktestRiskManager(config)
    circuit_breaker = CircuitBreaker(config)
    sizer = PositionSizer(config)

    # Trade Filters
    trade_filters = TradeFilterManager(config)

    # Schedule
    sched = config.get("schedule", {})
    earliest_entry = _parse_time(sched.get("earliest_entry", "09:45"))
    last_entry = _parse_time(sched.get("last_entry", "15:30"))
    force_close = _parse_time(sched.get("force_close", "15:55"))

    # Initialize ticker states with prior day data
    for ticker in ticker_bars:
        ts = market_state.get_ticker(ticker)
        history = prior_days.get(ticker, [])
        if history:
            last_day = history[-1]
            ts.prior_high = last_day["high"]
            ts.prior_low = last_day["low"]
            ts.prior_close = last_day["close"]
            ts.prior_volume = last_day["volume"]
            if len(history) >= 20:
                ts.prior_volume_20d_avg = np.mean([d["volume"] for d in history[-20:]])
            ts.close_quality = (last_day["close"] - last_day["low"]) / max(last_day["high"] - last_day["low"], 0.01)

        # Scanner result for Day2
        scanner = compute_scanner_result(ticker, history[-1] if history else None, history)
        ts.scanner_result = scanner

    # Build daily closes per ticker for regime filter (from prior day history)
    daily_closes_by_ticker: Dict[str, List[float]] = {}
    for ticker in ticker_bars:
        history = prior_days.get(ticker, [])
        daily_closes_by_ticker[ticker] = [d["close"] for d in history if d.get("close", 0) > 0]

    # Monday SPY return for Tuesday bias
    if market_state.day_of_week == "Tuesday":
        spy_history = prior_days.get("SPY", [])
        if spy_history:
            last = spy_history[-1]
            if last["open"] > 0:
                market_state.monday_spy_close_return = (last["close"] - last["open"]) / last["open"]

    # Collect all unique bar times across tickers and sort
    all_times = set()
    for ticker, bars in ticker_bars.items():
        for b in bars:
            all_times.add(b.timestamp)
    sorted_times = sorted(all_times)

    # Build lookup: ticker -> {timestamp: bar}
    bar_lookup: Dict[str, Dict[datetime, Bar]] = {}
    for ticker, bars in ticker_bars.items():
        bar_lookup[ticker] = {b.timestamp: b for b in bars}

    # State tracking
    open_position: Optional[dict] = None  # {position, delta, underlying_entry}
    trades: List[TradeRecord] = []
    current_account = account_value

    # Strategy delta midpoints
    strat_deltas = {}
    strat_config = config.get("strategies", {})
    for name, cfg in strat_config.items():
        if isinstance(cfg, dict):
            dmin = cfg.get("delta_min", 0.35)
            dmax = cfg.get("delta_max", 0.45)
            strat_deltas[name.upper()] = (dmin + dmax) / 2.0
    # Map strategy names
    strat_deltas["DAY2"] = strat_deltas.get("DAY2", strat_deltas.get("DAY2_CONTINUATION", 0.40))
    strat_deltas.setdefault("MOMENTUM", 0.35)
    strat_deltas.setdefault("REVERSION", 0.45)
    strat_deltas.setdefault("GREEN_SECTOR", 0.40)
    strat_deltas.setdefault("TUESDAY_REVERSAL", 0.45)

    # Replay bar by bar
    for bar_time in sorted_times:
        # Convert to ET for all time comparisons
        bar_time_et = bar_time.astimezone(ET) if bar_time.tzinfo else bar_time.replace(tzinfo=ET)
        market_state.sim_now = bar_time_et

        # Update each ticker's state
        for ticker in ticker_bars:
            ts = market_state.get_ticker(ticker)
            bar = bar_lookup[ticker].get(bar_time)
            if bar is None:
                continue

            # Append to 1-min bars (convert to ET for OR tracker)
            bar_et = Bar(
                timestamp=bar.timestamp.astimezone(ET) if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=ET),
                open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=bar.volume,
            )
            ts.bars_1m.append(bar_et)
            ts.last_price = bar.close
            ts.last_bar_time = bar_time_et

            # Session return
            if ts.prior_close > 0:
                ts.session_return = (ts.last_price - ts.prior_close) / ts.prior_close

            # Update indicators
            # VWAP (from 1-min bars for today only)
            if len(ts.bars_1m) >= 2:
                vwap_data = vwap_calc.calculate(ts.bars_1m)
                ts.vwap = vwap_data["vwap"]
                ts.vwap_upper_band = vwap_data["upper_band"]
                ts.vwap_lower_band = vwap_data["lower_band"]
                ts.vwap_slope = vwap_data["slope"]

            # Aggregate 3-min bars for RSI/ADX/volume
            ts.bars_3m = aggregate_3m_bars(ts.bars_1m)
            if ts.bars_3m:
                closes = [b.close for b in ts.bars_3m]
                highs = [b.high for b in ts.bars_3m]
                lows = [b.low for b in ts.bars_3m]
                volumes = [b.volume for b in ts.bars_3m]

                ts.rsi_7 = rsi_calc.calculate(closes)
                ts.adx_14 = adx_calc.calculate(highs, lows, closes)
                ts.volume_ratio = vol_calc.calculate(volumes)

            # Opening range
            if not ts.opening_range_set:
                or_tracker.update(ts)

        # Update SPY session return
        spy_ts = market_state.tickers.get("SPY")
        if spy_ts:
            market_state.spy_session_return = spy_ts.session_return

        # --- Check exits on open position ---
        if open_position is not None:
            pos = open_position["position"]
            delta = open_position["delta"]
            underlying_entry = open_position["underlying_entry"]
            ticker = pos.ticker

            ts = market_state.tickers.get(ticker)
            if ts:
                underlying_now = ts.last_price
                # Delta-estimated option price
                if pos.direction == Direction.CALL:
                    option_price = pos.entry_price + (underlying_now - underlying_entry) * delta
                else:
                    option_price = pos.entry_price + (underlying_entry - underlying_now) * delta

                option_price = max(option_price, 0.01)
                pos.current_price = option_price
                pos.unrealized_pnl_pct = (option_price - pos.entry_price) / pos.entry_price

                # Track max favorable/adverse
                pos.max_favorable = max(pos.max_favorable, pos.unrealized_pnl_pct)
                pos.max_adverse = min(pos.max_adverse, pos.unrealized_pnl_pct)

                # Break-even stop activation
                if (risk_manager.be_enabled and not pos.breakeven_stop_active
                        and pos.max_favorable >= risk_manager.be_activation_pct):
                    pos.breakeven_stop_active = True

                # Trailing stop management
                if not pos.trailing_active and pos.unrealized_pnl_pct >= risk_manager.trail_activation:
                    pos.trailing_active = True
                    pos.trailing_peak = option_price
                elif pos.trailing_active and option_price > pos.trailing_peak:
                    pos.trailing_peak = option_price

                # Check exit
                exit_reason = risk_manager.check_exit(pos, market_state, bar_time_et)
                if exit_reason is not None:
                    # Close position
                    hold_mins = (bar_time_et - pos.entry_time).total_seconds() / 60.0
                    pnl_pct = pos.unrealized_pnl_pct
                    pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts

                    trades.append(TradeRecord(
                        ticker=ticker,
                        direction=pos.direction.value,
                        strategy=pos.strategy,
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
                    ))

                    current_account += pnl_dollars
                    market_state.daily_pnl += pnl_dollars

                    # Record in circuit breaker
                    market_state.trades_today += 1
                    if pnl_pct < 0:
                        market_state.consecutive_losses += 1
                        cooldown_sec = config.get("risk", {}).get("cooldown_after_loss_sec", 600)
                        if market_state.consecutive_losses >= 3:
                            cooldown_sec = config.get("risk", {}).get("consecutive_loss_cooldown_sec", 1800)
                    else:
                        market_state.consecutive_losses = 0
                        cooldown_sec = config.get("risk", {}).get("cooldown_after_trade_sec", 300)

                    market_state.cooldown_until = bar_time_et + timedelta(seconds=cooldown_sec)
                    market_state.last_trade_close_time = bar_time_et

                    # Check daily loss limit
                    sim_account = config.get("risk", {}).get("simulated_account_value", account_value)
                    daily_loss_limit = config.get("risk", {}).get("daily_loss_limit_pct", 0.03)
                    if market_state.daily_pnl / sim_account <= -daily_loss_limit:
                        market_state.circuit_breaker_triggered = True

                    open_position = None

        # --- Look for new entries ---
        bar_t = bar_time_et.time()
        if (open_position is None
                and bar_t >= earliest_entry
                and bar_t <= last_entry
                and not market_state.circuit_breaker_triggered
                and not market_state.is_in_cooldown()):

            max_trades = config.get("risk", {}).get("max_trades_per_day", 6)
            if market_state.trades_today < max_trades:
                for ticker in ticker_bars:
                    ts = market_state.get_ticker(ticker)

                    # Regime classification
                    regime = regime_engine.classify(ts, market_state)
                    ts.current_regime = regime

                    if regime == Regime.NO_TRADE:
                        continue

                    # Evaluate strategy
                    signal = None
                    if regime == Regime.MOMENTUM and "MOMENTUM" in strategies:
                        signal = strategies["MOMENTUM"].evaluate(ts, market_state)
                    elif regime == Regime.REVERSION and "REVERSION" in strategies:
                        signal = strategies["REVERSION"].evaluate(ts, market_state)
                        if ts.tuesday_bias_active:
                            signal = tuesday_bias.modify(signal, ts, market_state)
                    elif regime == Regime.DAY2_CONTINUATION and "DAY2" in strategies:
                        signal = strategies["DAY2"].evaluate(ts, market_state)
                    elif regime == Regime.GREEN_SECTOR and "GREEN_SECTOR" in strategies:
                        signal = strategies["GREEN_SECTOR"].evaluate(ts, market_state)

                    if signal is None:
                        continue

                    # === Apply Trade Filters ===
                    filter_ok, size_mult, block_reason = trade_filters.apply(
                        signal=signal,
                        current_time=bar_t,
                        bars_1m=ts.bars_1m,
                        daily_closes=daily_closes_by_ticker.get(ticker, []),
                    )
                    if not filter_ok:
                        logger.debug(
                            f"Signal BLOCKED by {block_reason}: "
                            f"{signal.ticker} {signal.direction.value} "
                            f"via {signal.strategy} at {bar_time}"
                        )
                        continue

                    # Estimate option premium and size
                    strat_key = signal.strategy.upper()
                    delta = strat_deltas.get(strat_key, 0.35)
                    premium = estimate_option_premium(ticker, ts.last_price, delta)

                    # Size position
                    contract_info = {
                        "contract_type": ContractType.SINGLE_LEG,
                        "entry_price": premium,
                        "mid": premium,
                        "delta": delta,
                    }
                    sim_value = config.get("risk", {}).get("simulated_account_value", account_value)
                    size_info = sizer.calculate(
                        signal=signal,
                        contract_info=contract_info,
                        account_value=float(sim_value),
                        market_state=market_state,
                    )

                    # Apply regime filter size reduction (choppy market)
                    if size_mult < 1.0:
                        original = size_info["contracts"]
                        size_info["contracts"] = max(1, int(size_info["contracts"] * size_mult))
                        logger.debug(
                            f"Regime size reduction: {original} -> {size_info['contracts']} contracts"
                        )

                    if size_info["contracts"] == 0:
                        continue

                    # Create position
                    pos = Position(
                        ticker=signal.ticker,
                        direction=signal.direction,
                        contract_type=ContractType.SINGLE_LEG,
                        strategy=signal.strategy,
                        regime=signal.regime.value,
                        signal=signal,
                        delta_at_entry=delta,
                        entry_time=bar_time_et,
                        entry_price=premium,
                        num_contracts=size_info["contracts"],
                        current_price=premium,
                        spy_session_return=market_state.spy_session_return,
                        adx_value=ts.adx_14,
                        rsi_value=ts.rsi_7,
                        volume_ratio=ts.volume_ratio,
                        signal_strength_score=signal.strength_score,
                    )

                    open_position = {
                        "position": pos,
                        "delta": delta,
                        "underlying_entry": ts.last_price,
                    }

                    break  # One entry per bar scan

    # Force close any remaining position at EOD
    if open_position is not None:
        pos = open_position["position"]
        delta = open_position["delta"]
        underlying_entry = open_position["underlying_entry"]
        ticker = pos.ticker
        ts = market_state.tickers.get(ticker)

        if ts and sorted_times:
            last_time = sorted_times[-1]
            last_time_et = last_time.astimezone(ET) if last_time.tzinfo else last_time.replace(tzinfo=ET)
            underlying_now = ts.last_price

            if pos.direction == Direction.CALL:
                option_price = pos.entry_price + (underlying_now - underlying_entry) * delta
            else:
                option_price = pos.entry_price + (underlying_entry - underlying_now) * delta

            option_price = max(option_price, 0.01)
            pnl_pct = (option_price - pos.entry_price) / pos.entry_price
            pnl_dollars = (option_price - pos.entry_price) * 100 * pos.num_contracts
            hold_mins = (last_time_et - pos.entry_time).total_seconds() / 60.0

            trades.append(TradeRecord(
                ticker=ticker,
                direction=pos.direction.value,
                strategy=pos.strategy,
                entry_time=pos.entry_time,
                exit_time=last_time_et,
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


def _parse_time(t: str) -> dtime:
    parts = t.split(":")
    return dtime(int(parts[0]), int(parts[1]))


# ============================================================
# Build Daily OHLCV from 1-min bars
# ============================================================

def build_daily_bars(bars: List[Bar]) -> Dict[date, dict]:
    """Group 1-min bars by date and compute daily OHLCV."""
    daily: Dict[date, list] = {}
    for b in bars:
        d = b.timestamp.date() if hasattr(b.timestamp, 'date') else b.timestamp.date()
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
# Report Engine
# ============================================================

def generate_report(
    trades: List[TradeRecord],
    starting_capital: float,
    start_date: date,
    end_date: date,
) -> None:
    """Generate comprehensive backtest report."""

    if not trades:
        print("\n" + "=" * 60)
        print("  BACKTEST REPORT — NO TRADES GENERATED")
        print("=" * 60)
        print("\nThe bot did not find any qualifying signals in this period.")
        print("This could mean:")
        print("  - Opening range breakouts didn't meet ADX/volume thresholds")
        print("  - RSI never hit extreme levels for reversion")
        print("  - Insufficient bar data for indicator computation")
        print("\nTry a longer date range or different tickers.")
        return

    # Compute metrics
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

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl_dollars > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_consec_losses = max(max_consec_losses, abs(current_streak))

    # By strategy
    strategy_stats = {}
    for t in trades:
        s = t.strategy
        if s not in strategy_stats:
            strategy_stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
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
        r = t.exit_reason
        exit_stats[r] = exit_stats.get(r, 0) + 1

    # Daily P&L
    daily_pnl: Dict[date, float] = {}
    for t in trades:
        d = t.entry_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl_dollars

    best_day = max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
    worst_day = min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)

    # Monthly breakdown
    monthly_stats: Dict[str, dict] = {}
    for t in trades:
        month = t.entry_time.strftime("%Y-%m")
        if month not in monthly_stats:
            monthly_stats[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly_stats[month]["trades"] += 1
        monthly_stats[month]["pnl"] += t.pnl_dollars
        if t.pnl_dollars > 0:
            monthly_stats[month]["wins"] += 1

    # ── Print Report ──
    print("\n" + "=" * 70)
    print(f"  BACKTEST REPORT — {start_date} to {end_date}")
    print("=" * 70)

    print(f"\n{'Starting Capital:':<30} ${starting_capital:>12,.2f}")
    print(f"{'Ending Capital:':<30} ${equity[-1]:>12,.2f}")
    print(f"{'Net P&L:':<30} ${total_pnl:>12,.2f}")
    print(f"{'Return:':<30} {total_pnl / starting_capital * 100:>11.2f}%")

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

    # Save report to file
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"backtest_{timestamp}.txt"

    # Save trades CSV
    trades_path = reports_dir / f"backtest_trades_{timestamp}.csv"
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

    # Plot equity curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Equity curve
        ax1.plot(equity, linewidth=1.5, color="#2196F3")
        ax1.axhline(y=starting_capital, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e >= starting_capital for e in equity],
                        alpha=0.15, color="green")
        ax1.fill_between(range(len(equity)), starting_capital, equity,
                        where=[e < starting_capital for e in equity],
                        alpha=0.15, color="red")
        ax1.set_title(f"Equity Curve — {start_date} to {end_date}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Account Value ($)")
        ax1.set_xlabel("Trade #")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Trade P&L bar chart
        colors = ["green" if t.pnl_dollars > 0 else "red" for t in trades]
        ax2.bar(range(len(trades)), [t.pnl_dollars for t in trades], color=colors, alpha=0.7)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_title("Individual Trade P&L", fontsize=12)
        ax2.set_ylabel("P&L ($)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = reports_dir / f"backtest_equity_{timestamp}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Equity curve saved: {chart_path}")

    except ImportError:
        print("matplotlib not installed — skipping equity curve chart.")
        print("Install with: pip install matplotlib")


# ============================================================
# Main Entry Point
# ============================================================

async def run_backtest(
    start_date: date,
    end_date: date,
    tickers: List[str],
    from_cache: bool = False,
) -> None:
    """Run the complete backtest."""
    config = load_config()
    account_value = float(config.get("risk", {}).get("simulated_account_value", 25000))

    print(f"\nOptions Bot V1 — Backtester")
    print(f"Period: {start_date} to {end_date}")
    print(f"Tickers: {tickers}")
    print(f"Starting Capital: ${account_value:,.0f}")
    print(f"Data source: {'Cache' if from_cache else 'IBKR TWS (port 7497)'}")
    print()

    # Load data
    all_bars = await load_data(tickers, start_date, end_date, from_cache)

    # Build daily OHLCV for prior-day data
    daily_by_ticker: Dict[str, Dict[date, dict]] = {}
    for ticker, bars in all_bars.items():
        daily_by_ticker[ticker] = build_daily_bars(bars)

    # Get all unique trading dates
    all_dates = set()
    for ticker_daily in daily_by_ticker.values():
        all_dates.update(ticker_daily.keys())
    sorted_dates = sorted(all_dates)

    if not sorted_dates:
        print("No trading data found for the specified period.")
        return

    print(f"Found {len(sorted_dates)} trading days\n")

    # Replay each day
    all_trades: List[TradeRecord] = []
    current_account = account_value
    days_with_trades = 0

    for i, replay_date in enumerate(sorted_dates):
        # Get today's 1-min bars for each ticker
        today_bars: Dict[str, List[Bar]] = {}
        for ticker, bars in all_bars.items():
            day_bars = [b for b in bars if b.timestamp.date() == replay_date]
            if day_bars:
                today_bars[ticker] = day_bars

        if not today_bars:
            continue

        # Get prior day history for each ticker
        prior_days: Dict[str, List[dict]] = {}
        for ticker, daily in daily_by_ticker.items():
            prior = [daily[d] for d in sorted(daily.keys()) if d < replay_date]
            prior_days[ticker] = prior

        # Replay the day
        day_trades, current_account = replay_day(
            replay_date, today_bars, prior_days, config, current_account,
        )

        if day_trades:
            days_with_trades += 1
            all_trades.extend(day_trades)
            day_pnl = sum(t.pnl_dollars for t in day_trades)
            print(f"  {replay_date} ({replay_date.strftime('%a')}): "
                  f"{len(day_trades)} trades, P&L: ${day_pnl:>+,.2f} "
                  f"(Account: ${current_account:,.2f})")
        else:
            # Print progress every 10 days
            if (i + 1) % 10 == 0:
                print(f"  {replay_date}: no signals (day {i + 1}/{len(sorted_dates)})")

    # Generate report
    generate_report(all_trades, account_value, start_date, end_date)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Options Bot V1 strategies on historical data"
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["SPY", "QQQ", "NVDA", "TSLA"],
        help="Tickers to backtest (default: SPY QQQ NVDA TSLA)"
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Load data from CSV cache (skip IBKR connection)"
    )

    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    asyncio.run(run_backtest(start, end, args.tickers, args.from_cache))


if __name__ == "__main__":
    main()
