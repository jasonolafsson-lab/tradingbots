"""
Intraday Options Trading Bot — V1 Paper Trading
Entry point and main event loop.

Usage:
    python main.py                    # Normal mode
    python main.py --dry-run          # Signal generation only, no orders
    python main.py --config alt.yaml  # Custom config file
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import yaml
from zoneinfo import ZoneInfo

from data.ibkr_client import IBKRClient, IBKRConnectionError, IBKRAccountError
from data.uw_client import UWClient
from data.market_state import MarketState, TickerState, Regime, Signal, Position
from data.finnhub_feed import FinnhubFeed
from scanner.premarket_scanner import PreMarketScanner
from scanner.sector_tracker import SectorTracker
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
from execution.contract_selector import ContractSelector
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from risk.risk_manager import RiskManager
from risk.circuit_breaker import CircuitBreaker
from risk.sizing import PositionSizer
from intelligence.trade_memory import TradeMemoryDB
from intelligence.activation_manager import ActivationManager
from intelligence.morning_brief import load_daily_brief, is_in_no_trade_window
from logging_mod.trade_logger import TradeLogger
from logging_mod.quality_metrics import QualityMetrics
from filters.trade_filters import TradeFilterManager

logger = logging.getLogger("options_bot")

ET = ZoneInfo("US/Eastern")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tickers config
    tickers_path = os.path.join(os.path.dirname(config_path), "tickers.yaml")
    if os.path.exists(tickers_path):
        with open(tickers_path, "r") as f:
            config["tickers_config"] = yaml.safe_load(f)

    # Override API token from environment if set
    env_token = os.environ.get("UW_API_TOKEN")
    if env_token:
        config.setdefault("unusual_whales", {})["api_token"] = env_token

    return config


class OptionsBot:
    """Main bot orchestrator."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.running = False
        self.shutdown_requested = False

        # Core state
        self.market_state = MarketState()
        self.position: Optional[Position] = None

        # Data clients
        self.ibkr = IBKRClient(config)
        self.uw = UWClient(config)

        # Scanner
        self.scanner = PreMarketScanner(config, self.ibkr, self.uw)
        self.sector_tracker = SectorTracker(config, self.ibkr, self.uw)

        # Indicators
        self.vwap_calc = VWAPCalculator()
        self.rsi_calc = RSICalculator(period=7)
        self.adx_calc = ADXCalculator(period=14)
        self.or_tracker = OpeningRangeTracker(config)
        self.vol_calc = VolumeProfileCalculator(lookback=20)

        # Strategies
        self.regime_engine = RegimeEngine(config)
        self.strategies = {
            "MOMENTUM": MomentumStrategy(config),
            "REVERSION": ReversionStrategy(config),
            "DAY2": Day2Strategy(config),
            "GREEN_SECTOR": GreenSectorStrategy(config),
        }
        self.tuesday_bias = TuesdayBiasModifier(config)

        # Execution
        self.contract_selector = ContractSelector(config, self.ibkr, self.uw)
        self.order_manager = OrderManager(config, self.ibkr)
        self.position_manager = PositionManager(self.ibkr)

        # Risk
        self.risk_manager = RiskManager(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.sizer = PositionSizer(config)

        # Intelligence
        self.trade_memory = TradeMemoryDB(config)
        self.activation_manager = ActivationManager(config, self.trade_memory)

        # Logging
        self.trade_logger = TradeLogger(config)
        self.quality_metrics = QualityMetrics()

        # Trade Filters
        self.trade_filters = TradeFilterManager(config)
        # Daily closes per ticker for regime filter (populated from IBKR daily data)
        self.daily_closes: dict = {}  # ticker -> List[float]

        # Morning brief
        self.size_multiplier: float = 1.0

        # Schedule config
        sched = config.get("schedule", {})
        self.premarket_scan_time = self._parse_time(sched.get("premarket_scan_time", "09:00"))
        self.market_open_time = self._parse_time(sched.get("market_open", "09:30"))
        self.earliest_entry_time = self._parse_time(sched.get("earliest_entry", "09:45"))
        self.last_entry_time = self._parse_time(sched.get("last_entry", "15:30"))
        self.force_close_time = self._parse_time(sched.get("force_close", "15:55"))

    @staticmethod
    def _parse_time(t: str) -> dtime:
        parts = t.split(":")
        return dtime(int(parts[0]), int(parts[1]))

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def _time_et(self) -> dtime:
        return self._now_et().time()

    async def start(self) -> None:
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("OPTIONS BOT V1 — Starting")
        logger.info(f"Mode: {'DRY RUN (no orders)' if self.dry_run else 'PAPER TRADING'}")
        logger.info("=" * 60)

        try:
            # 1. Connect to IBKR
            await self.ibkr.connect()
            account_summary = await self.ibkr.get_account_summary()
            net_liq = account_summary.get("NetLiquidation", 0)
            logger.info(f"Account value: ${net_liq:,.2f}")

            # 2. Connect to UW (non-blocking)
            uw_ok = await self.uw.check_health()
            logger.info(f"Unusual Whales API: {'Connected' if uw_ok else 'UNAVAILABLE (continuing without)'}")

            # 3. Initialize trade memory DB
            self.trade_memory.initialize()

            # 3b. Load morning brief for today's adjustments
            brief = load_daily_brief()
            if brief:
                self.size_multiplier = brief.bot1_momentum.get("size_multiplier", 1.0)
                if not brief.bot1_momentum.get("enabled", True):
                    logger.warning(f"Morning brief says Bot 1 DISABLED: {brief.bot1_momentum.get('notes', '')}")
                    return
                logger.info(f"Morning brief: sentiment={brief.market_sentiment}, "
                           f"risk={brief.risk_level}, Bot 1 size={self.size_multiplier}x")
                if brief.no_trade_windows:
                    for w in brief.no_trade_windows:
                        logger.info(f"  No-trade window: {w['start']}-{w['end']} ({w['reason']})")

            # 4. Set up kill switch
            self._setup_kill_switch()

            # 5. Enter main loop
            self.running = True
            await self._main_loop()

        except IBKRConnectionError as e:
            logger.error(f"IBKR connection failed: {e}")
        except IBKRAccountError as e:
            logger.error(f"IBKR account error: {e}")
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt — shutting down")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def _main_loop(self) -> None:
        """Main trading loop."""
        # Initialize market state with day info
        now = self._now_et()
        self.market_state.today = now.date()
        self.market_state.day_of_week = now.strftime("%A")

        # Load ticker watchlist
        tickers_config = self.config.get("tickers_config", {})
        watchlist = tickers_config.get("watchlist", [])
        for t in watchlist:
            self.market_state.get_ticker(t["ticker"])

        logger.info(f"Watchlist: {[t['ticker'] for t in watchlist]}")
        logger.info(f"Day: {self.market_state.day_of_week}")

        # Pre-market scan (if before market open)
        if self._time_et() <= self.market_open_time:
            logger.info("Waiting for pre-market scan time...")
            while self._time_et() < self.premarket_scan_time and self.running:
                await self._check_kill_switch()
                await asyncio.sleep(10)

            if self.running:
                await self._run_premarket_scan()

            # Wait for market open
            while self._time_et() < self.market_open_time and self.running:
                await self._check_kill_switch()
                await asyncio.sleep(5)

        # Start Finnhub price feed (replaces IBKR real-time bars to avoid Error 420)
        feed_tickers = [t["ticker"] for t in watchlist]
        self.finnhub_feed = FinnhubFeed(self.market_state, tickers=feed_tickers)
        await self.finnhub_feed.start()
        logger.info(f"Finnhub feed started for {feed_tickers}")

        # Wait for opening range to establish
        logger.info("Collecting opening range (9:30-9:45)...")
        while self._time_et() < self.earliest_entry_time and self.running:
            await self._update_market_data()
            await self._check_kill_switch()
            await asyncio.sleep(5)

        logger.info("Opening range established. Trading active.")

        # Main signal evaluation loop
        while self.running and self._time_et() < self.force_close_time:
            await self._check_kill_switch()

            # Update all market data and indicators
            await self._update_market_data()

            # Check circuit breakers
            if self.circuit_breaker.is_triggered(self.market_state):
                logger.warning("Circuit breaker triggered — no new trades")
                if self.position:
                    await self._manage_position()
                await asyncio.sleep(30)
                continue

            # Manage existing position
            if self.position:
                await self._manage_position()
            else:
                # Only look for new signals if within entry window, no cooldown, and no event window
                if (self._time_et() < self.last_entry_time and
                        not self.market_state.is_in_cooldown() and
                        not is_in_no_trade_window(self._time_et())):
                    await self._evaluate_signals()

            # Sleep until next evaluation (roughly every 5 seconds)
            await asyncio.sleep(5)

        # End of day: force close
        await self._force_close_eod()

    async def _run_premarket_scan(self) -> None:
        """Run the pre-market scanner."""
        logger.info("Running pre-market scanner...")
        results = await self.scanner.run_scan(self.market_state)
        for result in results:
            ts = self.market_state.get_ticker(result.ticker)
            ts.scanner_result = result
            logger.info(
                f"  {result.ticker}: bias={result.bias.value} "
                f"day2_score={result.day2_score:.0f} "
                f"sector_rs={result.sector_rs:.3f} "
                f"rank={result.priority_rank}"
            )
        self.trade_logger.log_scanner_results(results)

        # Fetch daily closes for regime filter
        await self._load_daily_closes()

    async def _load_daily_closes(self) -> None:
        """Fetch recent daily closing prices for the regime filter."""
        if not self.trade_filters.regime_filter.enabled:
            return

        tickers_config = self.config.get("tickers_config", {})
        watchlist = tickers_config.get("watchlist", [])
        lookback_days = self.trade_filters.regime_filter.slow_period + 5  # Extra buffer

        for t in watchlist:
            ticker = t["ticker"]
            try:
                daily_bars = await self.ibkr.get_historical_bars(
                    ticker,
                    duration=f"{lookback_days} D",
                    bar_size="1 day",
                    what_to_show="TRADES",
                )
                if daily_bars:
                    self.daily_closes[ticker] = [b.close for b in daily_bars]
                    logger.info(
                        f"  Regime filter: loaded {len(daily_bars)} daily closes for {ticker}"
                    )
                else:
                    self.daily_closes[ticker] = []
            except Exception as e:
                logger.warning(f"  Regime filter: failed to load daily data for {ticker}: {e}")
                self.daily_closes[ticker] = []

    async def _update_market_data(self) -> None:
        """Update market state with latest data from all sources."""
        # Update per-ticker indicators
        for ticker, ts in self.market_state.tickers.items():
            # Update indicators from latest bars
            if ts.bars_3m:
                closes = [b.close for b in ts.bars_3m]
                highs = [b.high for b in ts.bars_3m]
                lows = [b.low for b in ts.bars_3m]
                volumes = [b.volume for b in ts.bars_3m]

                ts.rsi_7 = self.rsi_calc.calculate(closes)
                ts.adx_14 = self.adx_calc.calculate(highs, lows, closes)
                ts.volume_ratio = self.vol_calc.calculate(volumes)

            # VWAP (from 1-min bars)
            if ts.bars_1m:
                vwap_data = self.vwap_calc.calculate(ts.bars_1m)
                ts.vwap = vwap_data["vwap"]
                ts.vwap_upper_band = vwap_data["upper_band"]
                ts.vwap_lower_band = vwap_data["lower_band"]
                ts.vwap_slope = vwap_data["slope"]

                if ts.vwap > 0:
                    ts.session_return = (ts.last_price - ts.prior_close) / ts.prior_close if ts.prior_close else 0

            # Opening range
            if not ts.opening_range_set:
                self.or_tracker.update(ts)

        # Update SPY session return for market-wide state
        spy_state = self.market_state.tickers.get("SPY")
        if spy_state:
            self.market_state.spy_session_return = spy_state.session_return

        # Update sector returns periodically
        await self.sector_tracker.update(self.market_state)

    async def _evaluate_signals(self) -> None:
        """Evaluate all tickers for trading signals."""
        for ticker, ts in self.market_state.tickers.items():
            # Run regime engine
            regime = self.regime_engine.classify(ts, self.market_state)
            ts.current_regime = regime

            if regime == Regime.NO_TRADE:
                continue

            # Get signal from appropriate strategy
            signal = None
            strategy_name = regime.value

            if regime == Regime.MOMENTUM and "MOMENTUM" in self.strategies:
                signal = self.strategies["MOMENTUM"].evaluate(ts, self.market_state)
            elif regime == Regime.REVERSION and "REVERSION" in self.strategies:
                signal = self.strategies["REVERSION"].evaluate(ts, self.market_state)
                # Apply Tuesday bias modifier if active
                if ts.tuesday_bias_active:
                    signal = self.tuesday_bias.modify(signal, ts, self.market_state)
            elif regime == Regime.DAY2_CONTINUATION and "DAY2" in self.strategies:
                signal = self.strategies["DAY2"].evaluate(ts, self.market_state)
            elif regime == Regime.GREEN_SECTOR and "GREEN_SECTOR" in self.strategies:
                signal = self.strategies["GREEN_SECTOR"].evaluate(ts, self.market_state)

            if signal is not None:
                logger.info(
                    f"SIGNAL: {signal.ticker} {signal.direction.value} "
                    f"via {signal.strategy} (strength={signal.strength_score:.0f})"
                )

                # === Apply Trade Filters ===
                filter_ok, size_mult, block_reason = self.trade_filters.apply(
                    signal=signal,
                    current_time=self._time_et(),
                    bars_1m=ts.bars_1m,
                    daily_closes=self.daily_closes.get(ticker, []),
                )
                if not filter_ok:
                    logger.info(
                        f"SIGNAL BLOCKED by {block_reason}: "
                        f"{signal.ticker} {signal.direction.value} via {signal.strategy}"
                    )
                    self.trade_logger.log_signal(signal, executed=False, reason=block_reason)
                    continue

                # Store size multiplier for regime filter choppy reduction
                if size_mult < 1.0:
                    signal._regime_size_mult = size_mult
                    logger.info(
                        f"Regime filter: choppy market — will reduce size by "
                        f"{(1 - size_mult):.0%} for {signal.ticker}"
                    )

                if not self.dry_run:
                    await self._execute_signal(signal, ts)
                else:
                    self.trade_logger.log_signal(signal, executed=False, reason="DRY_RUN")

    async def _execute_signal(self, signal: Signal, ts: TickerState) -> None:
        """Execute a trading signal: select contract, size, submit order."""
        try:
            # 1. Select contract
            contract_info = await self.contract_selector.select(signal, ts)
            if contract_info is None:
                self.trade_logger.log_signal(signal, executed=False, reason="NO_CONTRACT")
                return

            # 2. Size position (use simulated account value if configured)
            sim_value = self.config.get("risk", {}).get("simulated_account_value")
            if sim_value:
                account_value = float(sim_value)
            else:
                account_value = await self.ibkr.get_net_liquidation()
            size_info = self.sizer.calculate(
                signal=signal,
                contract_info=contract_info,
                account_value=account_value,
                market_state=self.market_state,
            )
            # Apply regime filter size reduction if choppy
            regime_size_mult = getattr(signal, '_regime_size_mult', 1.0)
            if regime_size_mult < 1.0:
                original = size_info["contracts"]
                size_info["contracts"] = max(1, int(size_info["contracts"] * regime_size_mult))
                logger.info(
                    f"Regime size reduction: {original} -> {size_info['contracts']} contracts"
                )

            # Apply morning brief size adjustment
            if self.size_multiplier != 1.0:
                original = size_info["contracts"]
                size_info["contracts"] = max(1, int(size_info["contracts"] * self.size_multiplier))
                if size_info["contracts"] != original:
                    logger.info(f"Morning brief size adj: {original} -> {size_info['contracts']} contracts ({self.size_multiplier}x)")

            if size_info["contracts"] == 0:
                self.trade_logger.log_signal(signal, executed=False, reason="SIZE_ZERO")
                return

            # 3. Submit order
            trade = await self.order_manager.submit_entry(
                contract_info=contract_info,
                quantity=size_info["contracts"],
                signal=signal,
            )

            # 4. Wait for fill
            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=15)

            if filled:
                # Build position object
                self.position = self._build_position(signal, ts, contract_info, size_info, trade)
                self.trade_logger.log_entry(self.position)
                logger.info(f"FILLED: {self.position.ticker} {self.position.direction.value}")

                # Place bracket orders (stop + TP)
                if not self.dry_run:
                    await self.order_manager.place_bracket(self.position)
            else:
                # Reprice attempts
                repriced = await self.order_manager.reprice(trade, max_attempts=2)
                if not repriced:
                    await self.ibkr.cancel_order(trade)
                    self.trade_logger.log_signal(signal, executed=False, reason="NO_FILL")

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)
            self.trade_logger.log_signal(signal, executed=False, reason=f"ERROR: {e}")

    async def _manage_position(self) -> None:
        """Monitor and manage an open position."""
        if self.position is None:
            return

        exit_reason = self.risk_manager.check_exit(self.position, self.market_state)

        if exit_reason is not None:
            logger.info(f"EXIT: {self.position.ticker} — {exit_reason.value}")
            if not self.dry_run:
                await self.order_manager.close_position(self.position, exit_reason)

            # Record trade
            self.trade_memory.record_trade(self.position, exit_reason)
            self.trade_logger.log_exit(self.position, exit_reason)

            # Update circuit breaker state
            self.circuit_breaker.record_trade_result(
                self.position, self.market_state
            )

            self.position = None

    async def _force_close_eod(self) -> None:
        """Force close all positions at end of day."""
        if self.position and not self.dry_run:
            logger.warning("EOD: Force closing all positions")
            from data.market_state import ExitReason
            await self.order_manager.close_position(self.position, ExitReason.EOD_CLOSE)
            self.trade_memory.record_trade(self.position, ExitReason.EOD_CLOSE)
            self.trade_logger.log_exit(self.position, ExitReason.EOD_CLOSE)
            self.position = None

        # Cancel any remaining orders
        if not self.dry_run:
            await self.ibkr.cancel_all_orders()

        # Post-market tasks
        self.trade_memory.backup()
        self.trade_logger.write_daily_summary(self.market_state)
        logger.info("End of day. All positions closed. Logs written.")

    def _build_position(
        self, signal, ts, contract_info, size_info, trade
    ) -> Position:
        """Build a Position object from trade details."""
        from data.market_state import ContractType
        pos = Position(
            ticker=signal.ticker,
            direction=signal.direction,
            contract_type=contract_info.get("contract_type", ContractType.SINGLE_LEG),
            strategy=signal.strategy,
            regime=signal.regime.value,
            signal=signal,
            strike=contract_info.get("strike", 0),
            expiry=contract_info.get("expiry", ""),
            dte=contract_info.get("dte", 0),
            delta_at_entry=contract_info.get("delta", 0),
            iv_at_entry=contract_info.get("iv", 0),
            iv_percentile=ts.uw_iv_percentile,
            spread_width=contract_info.get("spread_width"),
            entry_time=datetime.now(ET),
            entry_price=size_info.get("entry_price", 0),
            num_contracts=size_info["contracts"],
            # Market context at entry
            spy_session_return=self.market_state.spy_session_return,
            underlying_vs_vwap=(ts.last_price - ts.vwap) / max(ts.vwap_upper_band - ts.vwap, 0.01) if ts.vwap else 0,
            vwap_slope=ts.vwap_slope,
            adx_value=ts.adx_14,
            rsi_value=ts.rsi_7,
            volume_ratio=ts.volume_ratio,
            sector_rs=ts.scanner_result.sector_rs if ts.scanner_result else 0,
            uw_net_premium_direction=ts.uw_net_premium_direction.value,
            gex_nearest_wall_distance=ts.uw_gex_nearest_wall_distance,
            day2_score=signal.day2_score,
            signal_strength_score=signal.strength_score,
        )
        return pos

    def _setup_kill_switch(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.warning(f"Received signal {signum} — requesting shutdown")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def _check_kill_switch(self) -> None:
        """Check for kill switch conditions."""
        # Check signal-based shutdown
        if self.shutdown_requested:
            logger.warning("Kill switch triggered via signal")
            self.running = False
            return

        # Check file-based kill switch
        kill_file = self.config.get("safety", {}).get("kill_switch_file", "KILL")
        if os.path.exists(kill_file):
            logger.warning(f"Kill switch file '{kill_file}' detected")
            self.running = False
            os.remove(kill_file)
            return

        # Check IBKR connection
        if not self.ibkr.is_connected():
            logger.error("IBKR connection lost!")
            self.running = False

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False

        try:
            if self.position and not self.dry_run:
                logger.warning("Emergency: closing open position")
                await self.ibkr.close_all_positions()

            await self.ibkr.cancel_all_orders()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            await self.ibkr.disconnect()
            logger.info("Shutdown complete.")


def setup_logging(config: dict) -> None:
    """Configure logging."""
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    today = datetime.now().strftime("%Y-%m-%d")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/{today}.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Intraday Options Trading Bot V1")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to settings YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals only, do not place orders")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    bot = OptionsBot(config, dry_run=args.dry_run)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
