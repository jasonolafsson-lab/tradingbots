"""
Finnhub WebSocket real-time price feed.

Replaces IBKR reqRealTimeBars() for market data streaming.
IBKR is still used for order execution, option chains, account info.

Flow:
  Finnhub WS → raw trades → 1-min bar aggregation → 3-min bar aggregation
  → TickerState.bars_1m / bars_3m / last_price
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import aiohttp

from data.market_state import Bar, MarketState, TickerState

logger = logging.getLogger(__name__)


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
    Real-time price feed from Finnhub WebSocket.

    Subscribes to trade events, aggregates into 1-min and 3-min bars,
    and populates MarketState.TickerState for each ticker.

    Usage:
        feed = FinnhubFeed(market_state, tickers=["SPY", "QQQ"])
        await feed.start()
        # ... bars_1m, bars_3m, last_price now populate automatically
        await feed.stop()
    """

    WS_URL = "wss://ws.finnhub.io"
    POLL_URL = "https://finnhub.io/api/v1/quote"
    RECONNECT_DELAY = 5       # seconds between reconnect attempts
    MAX_RECONNECT_DELAY = 60  # cap exponential backoff
    BAR_INTERVAL = 60         # 1-minute bars
    MAX_1M_BARS = 390         # Full trading day of 1-min bars

    def __init__(
        self,
        market_state: MarketState,
        tickers: List[str],
        api_key: Optional[str] = None,
    ):
        self.market_state = market_state
        self.tickers = [t.upper() for t in tickers]
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")

        # WebSocket state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._bar_task: Optional[asyncio.Task] = None
        self._reconnect_count = 0

        # Trade accumulation for bar building
        # {ticker: {"trades": [(price, volume, timestamp), ...], "current_bar_start": datetime}}
        self._trade_buffers: Dict[str, dict] = {}
        for ticker in self.tickers:
            self._trade_buffers[ticker] = {
                "trades": [],
                "current_bar_start": None,
                "last_price": 0.0,
            }

        # Polling fallback state
        self._using_polling = False
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the Finnhub data feed."""
        if not self.api_key:
            logger.error("No FINNHUB_API_KEY set — cannot start price feed")
            raise RuntimeError("FINNHUB_API_KEY required for Finnhub feed")

        self._running = True
        self._session = aiohttp.ClientSession()

        # Start WebSocket connection
        self._ws_task = asyncio.create_task(self._ws_loop())
        # Start bar builder
        self._bar_task = asyncio.create_task(self._bar_builder_loop())

        logger.info(f"Finnhub feed started for {self.tickers}")

    async def stop(self) -> None:
        """Stop the data feed and clean up."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._bar_task:
            self._bar_task.cancel()
            try:
                await self._bar_task
            except asyncio.CancelledError:
                pass

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        logger.info("Finnhub feed stopped")

    # ----------------------------------------------------------------
    # WebSocket connection
    # ----------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                url = f"{self.WS_URL}?token={self.api_key}"
                async with self._session.ws_connect(url, heartbeat=30) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    self._using_polling = False
                    logger.info("Finnhub WebSocket connected")

                    # Subscribe to all tickers
                    for ticker in self.tickers:
                        await ws.send_json({"type": "subscribe", "symbol": ticker})
                        logger.info(f"Finnhub WS subscribed: {ticker}")

                    # Read messages
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._process_ws_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"Finnhub WS error: {ws.exception()}")
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                            break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Finnhub WS connection error: {e}")

            if not self._running:
                break

            # Reconnect with exponential backoff
            self._reconnect_count += 1
            delay = min(
                self.RECONNECT_DELAY * (2 ** (self._reconnect_count - 1)),
                self.MAX_RECONNECT_DELAY,
            )
            logger.info(f"Finnhub WS reconnecting in {delay}s (attempt {self._reconnect_count})")

            # Switch to polling fallback if too many reconnects
            if self._reconnect_count >= 3 and not self._using_polling:
                logger.warning("Switching to polling fallback after 3 WS failures")
                self._using_polling = True
                self._poll_task = asyncio.create_task(self._poll_loop())

            await asyncio.sleep(delay)

    def _process_ws_message(self, raw: str) -> None:
        """Process a Finnhub WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if data.get("type") != "trade":
            return

        for trade in data.get("data", []):
            symbol = trade.get("s", "")
            price = trade.get("p", 0.0)
            volume = trade.get("v", 0.0)
            ts_ms = trade.get("t", 0)

            if symbol not in self._trade_buffers:
                continue

            ts = datetime.fromtimestamp(ts_ms / 1000.0)
            buf = self._trade_buffers[symbol]
            buf["trades"].append((price, volume, ts))
            buf["last_price"] = price

            # Update last_price immediately
            ticker_state = self.market_state.get_ticker(symbol)
            ticker_state.last_price = price
            ticker_state.last_bar_time = ts
            ticker_state.data_source = "finnhub_ws"

    # ----------------------------------------------------------------
    # Polling fallback
    # ----------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Fallback: poll /quote every 5 seconds if WebSocket fails."""
        logger.info("Finnhub polling fallback started")
        while self._running and self._using_polling:
            try:
                for ticker in self.tickers:
                    await self._poll_quote(ticker)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Finnhub poll error: {e}")
                await asyncio.sleep(10)

    async def _poll_quote(self, ticker: str) -> None:
        """Fetch a single quote from Finnhub REST API."""
        url = f"{self.POLL_URL}?symbol={ticker}&token={self.api_key}"
        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get("c", 0.0)  # current price
                    if price > 0:
                        now = datetime.now()
                        buf = self._trade_buffers[ticker]
                        buf["trades"].append((price, 0, now))
                        buf["last_price"] = price

                        ts = self.market_state.get_ticker(ticker)
                        ts.last_price = price
                        ts.last_bar_time = now
                        ts.data_source = "finnhub_poll"
                elif resp.status == 429:
                    logger.warning("Finnhub rate limit hit, backing off")
                    await asyncio.sleep(30)
        except Exception as e:
            logger.debug(f"Finnhub poll failed for {ticker}: {e}")

    # ----------------------------------------------------------------
    # Bar aggregation
    # ----------------------------------------------------------------

    async def _bar_builder_loop(self) -> None:
        """
        Every minute, aggregate accumulated trades into 1-min bars.
        Then re-aggregate 1-min bars into 3-min bars.
        """
        # Wait for first trade data
        await asyncio.sleep(2)

        # Align to next minute boundary
        now = datetime.now()
        seconds_to_next = 60 - now.second
        await asyncio.sleep(seconds_to_next)

        while self._running:
            try:
                bar_end = datetime.now().replace(second=0, microsecond=0)
                bar_start = bar_end - timedelta(minutes=1)

                for ticker in self.tickers:
                    self._build_1m_bar(ticker, bar_start, bar_end)

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Bar builder error: {e}")
                await asyncio.sleep(60)

    def _build_1m_bar(self, ticker: str, bar_start: datetime, bar_end: datetime) -> None:
        """Build a 1-minute bar from accumulated trades."""
        buf = self._trade_buffers[ticker]
        trades = buf["trades"]

        # Filter trades within this minute
        minute_trades = [
            (p, v, t) for p, v, t in trades
            if bar_start <= t < bar_end and p > 0
        ]

        # Remove old trades (keep last 2 minutes for safety)
        cutoff = bar_start - timedelta(minutes=2)
        buf["trades"] = [(p, v, t) for p, v, t in trades if t >= cutoff]

        if not minute_trades:
            # No trades this minute — create bar from last known price
            last = buf["last_price"]
            if last <= 0:
                return
            bar = Bar(
                timestamp=bar_start,
                open=last,
                high=last,
                low=last,
                close=last,
                volume=0,
            )
        else:
            prices = [p for p, v, t in minute_trades]
            volumes = [v for p, v, t in minute_trades]
            bar = Bar(
                timestamp=bar_start,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(volumes),
            )

        # Append to TickerState
        ts = self.market_state.get_ticker(ticker)
        ts.bars_1m.append(bar)
        ts.last_price = bar.close
        ts.last_bar_time = bar_end

        # Trim to max size
        if len(ts.bars_1m) > self.MAX_1M_BARS:
            ts.bars_1m = ts.bars_1m[-self.MAX_1M_BARS:]

        # Re-aggregate 3-min bars
        ts.bars_3m = aggregate_3m_bars(ts.bars_1m)

        logger.debug(
            f"Finnhub bar: {ticker} {bar_start.strftime('%H:%M')} "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
            f"C={bar.close:.2f} V={bar.volume:.0f} "
            f"(1m={len(ts.bars_1m)}, 3m={len(ts.bars_3m)})"
        )

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected or polling is active."""
        if self._using_polling:
            return True
        return self._ws is not None and not self._ws.closed

    @property
    def data_source(self) -> str:
        """Current data source type."""
        if self._using_polling:
            return "finnhub_poll"
        return "finnhub_ws"
