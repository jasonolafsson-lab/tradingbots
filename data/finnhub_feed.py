"""
Finnhub real-time price feed — shared architecture.

Two modes:
  1. WRITER mode (finnhub_streamer.py): Single process connects to Finnhub WebSocket,
     aggregates trades into bars, writes to data/live_prices.json every second.
  2. READER mode (used by all 6 bots): Reads data/live_prices.json, populates
     TickerState.bars_1m / bars_3m / last_price.

This avoids Finnhub free-tier's 1-connection WebSocket limit.

Architecture:
  finnhub_streamer.py (1 process) → data/live_prices.json ← Bot 1, 2, 3, 4, 5, 6
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from data.market_state import Bar, MarketState, TickerState

logger = logging.getLogger(__name__)

LIVE_PRICES_PATH = (Path(__file__).parent.parent / "data" / "live_prices.json").resolve()


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


class FinnhubFeed:
    """
    READER mode — reads live prices from shared file written by finnhub_streamer.py.

    Polls data/live_prices.json every 1 second and populates MarketState.

    Usage:
        feed = FinnhubFeed(market_state, tickers=["SPY", "QQQ"])
        await feed.start()
        # ... bars_1m, bars_3m, last_price now populate automatically
        await feed.stop()
    """

    MAX_1M_BARS = 390  # Full trading day

    def __init__(
        self,
        market_state: MarketState,
        tickers: List[str],
        prices_path: Optional[Path] = None,
    ):
        self.market_state = market_state
        self.tickers = [t.upper() for t in tickers]
        self.prices_path = prices_path or LIVE_PRICES_PATH
        self._running = False
        self._reader_task: Optional[asyncio.Task] = None
        self._bar_task: Optional[asyncio.Task] = None
        self._last_bar_counts: Dict[str, int] = {}  # track new bars

    async def start(self) -> None:
        """Start reading from shared price file."""
        self._running = True
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._bar_task = asyncio.create_task(self._bar_sync_loop())
        logger.info(f"Finnhub feed (reader) started for {self.tickers}")
        logger.info(f"Reading prices from: {self.prices_path}")

    async def stop(self) -> None:
        """Stop the reader."""
        self._running = False
        for task in [self._reader_task, self._bar_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Finnhub feed stopped")

    async def _reader_loop(self) -> None:
        """Poll the shared price file every second for last_price updates."""
        while self._running:
            try:
                if self.prices_path.exists():
                    data = json.loads(self.prices_path.read_text())
                    for ticker in self.tickers:
                        if ticker in data:
                            td = data[ticker]
                            ts = self.market_state.get_ticker(ticker)
                            price = td.get("last_price", 0.0)
                            if price > 0:
                                ts.last_price = price
                                ts.last_bar_time = datetime.now()
                                ts.data_source = "finnhub"
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Price file read error: {e}")
                await asyncio.sleep(2)

    async def _bar_sync_loop(self) -> None:
        """Sync 1-min and 3-min bars from the shared price file every 60 seconds."""
        # Align to next minute boundary
        await asyncio.sleep(2)
        now = datetime.now()
        seconds_to_next = 60 - now.second
        await asyncio.sleep(seconds_to_next)

        while self._running:
            try:
                if self.prices_path.exists():
                    data = json.loads(self.prices_path.read_text())
                    for ticker in self.tickers:
                        if ticker in data:
                            self._sync_bars(ticker, data[ticker])
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Bar sync error: {e}")
                await asyncio.sleep(60)

    def _sync_bars(self, ticker: str, td: dict) -> None:
        """Sync bars from the shared file into TickerState."""
        ts = self.market_state.get_ticker(ticker)
        file_bars = td.get("bars_1m", [])

        if not file_bars:
            return

        # Convert JSON bars to Bar objects
        new_bars = []
        for b in file_bars:
            try:
                bar = Bar(
                    timestamp=datetime.fromisoformat(b["timestamp"]),
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b.get("volume", 0),
                )
                new_bars.append(bar)
            except (KeyError, ValueError):
                continue

        if not new_bars:
            return

        # Only append new bars (compare count)
        prev_count = self._last_bar_counts.get(ticker, 0)
        if len(new_bars) > prev_count:
            # Append only the new ones
            ts.bars_1m = new_bars[-self.MAX_1M_BARS:]
            ts.bars_3m = aggregate_3m_bars(ts.bars_1m)
            ts.last_price = new_bars[-1].close
            ts.last_bar_time = new_bars[-1].timestamp
            self._last_bar_counts[ticker] = len(new_bars)

            logger.debug(
                f"Synced {ticker}: {len(ts.bars_1m)} 1m bars, "
                f"{len(ts.bars_3m)} 3m bars, last={ts.last_price:.2f}"
            )

    @property
    def is_connected(self) -> bool:
        """Check if price file exists and is fresh."""
        if not self.prices_path.exists():
            return False
        age = datetime.now().timestamp() - self.prices_path.stat().st_mtime
        return age < 10  # Fresh if updated in last 10 seconds

    @property
    def data_source(self) -> str:
        return "finnhub"
