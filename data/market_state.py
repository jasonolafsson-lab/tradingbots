"""
Unified market state object per ticker.

This is the shared data structure that all modules read from and write to.
The data clients (IBKR, UW) populate it; the indicators, scanner, regime engine,
and strategy modules consume it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, List, Dict


class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Regime(str, Enum):
    NO_TRADE = "NO_TRADE"
    MOMENTUM = "MOMENTUM"
    REVERSION = "REVERSION"
    DAY2_CONTINUATION = "DAY2_CONTINUATION"
    GREEN_SECTOR = "GREEN_SECTOR"
    # Tuesday bias is a modifier, not a standalone regime
    TUESDAY_BIAS = "TUESDAY_BIAS"


class Direction(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class ContractType(str, Enum):
    SINGLE_LEG = "SINGLE_LEG"
    DEBIT_SPREAD = "DEBIT_SPREAD"
    CREDIT_SPREAD = "CREDIT_SPREAD"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    EOD_CLOSE = "EOD_CLOSE"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"


@dataclass
class Bar:
    """A single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ScannerResult:
    """Output from the pre-market scanner for one ticker."""
    ticker: str
    bias: Bias = Bias.NEUTRAL
    day2_score: float = 0.0
    sector_rs: float = 0.0       # Sector relative strength vs SPY
    priority_rank: int = 0
    close_quality: float = 0.0   # (close - low) / (high - low)
    volume_vs_avg: float = 0.0   # Prior day volume / 20-day avg
    catalyst_score: float = 0.0  # UW flow-based catalyst metric
    key_levels: Dict[str, float] = field(default_factory=dict)
    # key_levels: prior_high, prior_low, prior_close, sma_20, sma_50


@dataclass
class TickerState:
    """Complete state for a single ticker, updated every bar."""
    ticker: str

    # --- Price Data (from IBKR) ---
    bars_1m: List[Bar] = field(default_factory=list)
    bars_3m: List[Bar] = field(default_factory=list)
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    # --- Computed Indicators ---
    vwap: float = 0.0
    vwap_upper_band: float = 0.0   # VWAP + 2 SD
    vwap_lower_band: float = 0.0   # VWAP - 2 SD
    vwap_slope: float = 0.0        # 3-bar linear regression
    rsi_7: float = 50.0
    adx_14: float = 0.0
    volume_ratio: float = 0.0      # Current bar vol / 20-bar avg
    atr_14: float = 0.0
    atr_20_avg: float = 0.0        # 20-day average ATR (for vol adjustment)

    # --- Opening Range ---
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    opening_range_set: bool = False

    # --- Prior Day Data ---
    prior_high: float = 0.0
    prior_low: float = 0.0
    prior_close: float = 0.0
    prior_volume: float = 0.0
    prior_volume_20d_avg: float = 0.0
    close_quality: float = 0.0     # (close - low) / (high - low)

    # --- Pre-market ---
    premarket_high: float = 0.0
    premarket_low: float = 0.0

    # --- Session Metrics ---
    session_return: float = 0.0    # Current session % return

    # --- VWAP Internals (for cumulative computation) ---
    _cum_volume: float = 0.0
    _cum_vp: float = 0.0          # Cumulative (volume * typical_price)
    _cum_vp2: float = 0.0         # For std dev computation

    # --- UW Intelligence ---
    uw_net_premium_direction: Bias = Bias.NEUTRAL
    uw_gex_levels: Dict[str, float] = field(default_factory=dict)
    uw_gex_nearest_wall_distance: float = 999.0  # % distance to nearest wall
    uw_dark_pool_direction: Bias = Bias.NEUTRAL
    uw_iv_percentile: float = 0.0
    uw_flow_alerts: List[dict] = field(default_factory=list)

    # --- Scanner Output ---
    scanner_result: Optional[ScannerResult] = None

    # --- Regime ---
    current_regime: Regime = Regime.NO_TRADE
    tuesday_bias_active: bool = False

    # --- Timestamps ---
    last_bar_time: Optional[datetime] = None
    last_uw_update: float = 0.0    # Unix timestamp of last UW data refresh

    def is_data_stale(self, stale_seconds: int = 120) -> bool:
        """Check if price data is stale (no new bar within threshold)."""
        if self.last_bar_time is None:
            return True
        elapsed = (datetime.now() - self.last_bar_time).total_seconds()
        return elapsed > stale_seconds


@dataclass
class MarketState:
    """
    Top-level market state container.
    Holds per-ticker states plus market-wide data.
    """
    tickers: Dict[str, TickerState] = field(default_factory=dict)

    # --- Market-Wide ---
    spy_session_return: float = 0.0
    sector_returns: Dict[str, float] = field(default_factory=dict)  # ETF ticker → return %
    green_sectors: List[str] = field(default_factory=list)           # Sectors green while SPY red

    # --- Calendar ---
    today: date = field(default_factory=date.today)
    day_of_week: str = ""        # "Monday", "Tuesday", etc.
    monday_spy_close_return: float = 0.0  # Prior Monday's SPY return (for Tuesday logic)

    # --- Session Control ---
    trading_active: bool = False
    circuit_breaker_triggered: bool = False
    daily_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_close_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    def get_ticker(self, ticker: str) -> TickerState:
        """Get or create a ticker state."""
        if ticker not in self.tickers:
            self.tickers[ticker] = TickerState(ticker=ticker)
        return self.tickers[ticker]

    def is_in_cooldown(self) -> bool:
        """Check if the bot is in a cooldown period."""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until


@dataclass
class Signal:
    """A trading signal generated by a strategy module."""
    ticker: str
    direction: Direction
    strategy: str                # e.g. "MOMENTUM", "REVERSION", "DAY2"
    regime: Regime
    strength_score: float        # 0-100, how many confirmations aligned
    entry_price_target: float    # Underlying price at signal
    timestamp: datetime = field(default_factory=datetime.now)
    tuesday_bias: bool = False   # Was Tuesday bias active?
    day2_score: float = 0.0      # Day 2 candidacy score if applicable
    gap_pct: float = 0.0         # Gap from prior close (for Day 2 gap guard)

    @property
    def is_gap_risk(self) -> bool:
        """Check if Day 2 gap guard should reduce size."""
        return abs(self.gap_pct) > 0.03


@dataclass
class Position:
    """Tracks an open position."""
    ticker: str
    direction: Direction
    contract_type: ContractType
    strategy: str
    regime: str
    signal: Signal

    # Contract details
    strike: float = 0.0
    expiry: str = ""
    dte: int = 0
    delta_at_entry: float = 0.0
    iv_at_entry: float = 0.0
    iv_percentile: float = 0.0
    spread_width: Optional[float] = None

    # Execution
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0     # Premium or debit paid
    num_contracts: int = 0
    entry_slippage: float = 0.0
    bid_ask_at_entry: float = 0.0
    fill_time_sec: float = 0.0

    # Live tracking
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_favorable: float = 0.0   # Highest unrealized gain %
    max_adverse: float = 0.0     # Deepest unrealized loss %

    # Trailing stop state
    trailing_active: bool = False
    trailing_peak: float = 0.0

    # Break-even stop state
    breakeven_stop_active: bool = False

    # Market context at entry (for trade memory)
    spy_session_return: float = 0.0
    underlying_vs_vwap: float = 0.0
    vwap_slope: float = 0.0
    adx_value: float = 0.0
    rsi_value: float = 0.0
    volume_ratio: float = 0.0
    sector_rs: float = 0.0
    uw_net_premium_direction: str = "NEUTRAL"
    gex_nearest_wall_distance: float = 0.0
    day2_score: float = 0.0
    signal_strength_score: float = 0.0

    # IBKR order IDs
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
