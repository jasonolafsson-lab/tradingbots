#!/usr/bin/env python3
"""
Finnhub WebSocket Streamer — standalone service.

Single process that connects to Finnhub WebSocket, aggregates trades
into 1-min bars, and writes data/live_prices.json every second.

All 6 bots read from this shared file — avoids Finnhub free-tier
1-connection WebSocket limit.

Usage:
    python finnhub_streamer.py                    # All default tickers
    python finnhub_streamer.py SPY QQQ NVDA       # Custom tickers

Systemd: deploy/services/finnhub-feed.service
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("finnhub_streamer")

# All tickers used across all 6 bots
DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA"]

LIVE_PRICES_PATH = (Path(__file__).parent / "data" / "live_prices.json").resolve()
# Ensure the data directory exists
LIVE_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
WS_URL = "wss://ws.finnhub.io"
POLL_URL = "https://finnhub.io/api/v1/quote"
MAX_1M_BARS = 390  # Full trading day


class FinnhubStreamer:
    """
    Connects to Finnhub WebSocket, builds 1-min bars, writes to shared file.
    """

    def __init__(self, tickers: List[str], api_key: str):
        self.tickers = [t.upper() for t in tickers]
        self.api_key = api_key
        self._running = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._reconnect_count = 0

        # Per-ticker state
        self._trade_buffers: Dict[str, dict] = {}
        self._bars_1m: Dict[str, List[dict]] = {}
        self._last_prices: Dict[str, float] = {}

        for ticker in self.tickers:
            self._trade_buffers[ticker] = []
            self._bars_1m[ticker] = []
            self._last_prices[ticker] = 0.0

    async def run(self) -> None:
        """Main entry point."""
        self._running = True
        self._session = aiohttp.ClientSession()

        # Register shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        logger.info(f"Finnhub streamer starting for {self.tickers}")
        logger.info(f"Writing prices to: {LIVE_PRICES_PATH}")

        # Run WebSocket, bar builder, and file writer concurrently
        await asyncio.gather(
            self._ws_loop(),
            self._bar_builder_loop(),
            self._file_writer_loop(),
        )

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down streamer...")
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    # ----------------------------------------------------------------
    # WebSocket
    # ----------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """WebSocket loop with auto-reconnect and polling fallback."""
        while self._running:
            try:
                url = f"{WS_URL}?token={self.api_key}"
                async with self._session.ws_connect(url, heartbeat=30) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    logger.info("Finnhub WebSocket connected")

                    for ticker in self.tickers:
                        await ws.send_json({"type": "subscribe", "symbol": ticker})
                        logger.info(f"Subscribed: {ticker}")

                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._process_message(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.ERROR,
                                          aiohttp.WSMsgType.CLOSED,
                                          aiohttp.WSMsgType.CLOSING):
                            break

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"WS error: {e}")

            if not self._running:
                return

            self._reconnect_count += 1
            delay = min(5 * (2 ** (self._reconnect_count - 1)), 60)
            logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})")

            # After 3 failures, switch to polling
            if self._reconnect_count >= 3:
                logger.warning("WS failed 3x — switching to polling")
                await self._poll_loop()
                return

            await asyncio.sleep(delay)

    def _process_message(self, raw: str) -> None:
        """Process a Finnhub trade message."""
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
            self._trade_buffers[symbol].append((price, volume, ts))
            self._last_prices[symbol] = price

    async def _poll_loop(self) -> None:
        """Fallback polling mode."""
        logger.info("Polling fallback active")
        while self._running:
            try:
                for ticker in self.tickers:
                    url = f"{POLL_URL}?symbol={ticker}&token={self.api_key}"
                    async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price = data.get("c", 0.0)
                            if price > 0:
                                now = datetime.now()
                                self._trade_buffers[ticker].append((price, 0, now))
                                self._last_prices[ticker] = price
                        elif resp.status == 429:
                            logger.warning("Rate limited, waiting 30s")
                            await asyncio.sleep(30)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                await asyncio.sleep(10)

    # ----------------------------------------------------------------
    # Bar builder
    # ----------------------------------------------------------------

    async def _bar_builder_loop(self) -> None:
        """Build 1-min bars every minute."""
        await asyncio.sleep(3)

        # Align to minute boundary
        now = datetime.now()
        await asyncio.sleep(60 - now.second)

        while self._running:
            try:
                bar_end = datetime.now().replace(second=0, microsecond=0)
                bar_start = bar_end - timedelta(minutes=1)

                for ticker in self.tickers:
                    self._build_bar(ticker, bar_start, bar_end)

                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Bar builder error: {e}")
                await asyncio.sleep(60)

    def _build_bar(self, ticker: str, bar_start: datetime, bar_end: datetime) -> None:
        """Build a 1-min bar from trades."""
        trades = self._trade_buffers[ticker]

        minute_trades = [
            (p, v, t) for p, v, t in trades
            if bar_start <= t < bar_end and p > 0
        ]

        # Cleanup old trades
        cutoff = bar_start - timedelta(minutes=2)
        self._trade_buffers[ticker] = [(p, v, t) for p, v, t in trades if t >= cutoff]

        last = self._last_prices[ticker]
        if not minute_trades:
            if last <= 0:
                return
            bar = {"timestamp": bar_start.isoformat(), "open": last, "high": last,
                   "low": last, "close": last, "volume": 0}
        else:
            prices = [p for p, v, t in minute_trades]
            volumes = [v for p, v, t in minute_trades]
            bar = {
                "timestamp": bar_start.isoformat(),
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": sum(volumes),
            }

        self._bars_1m[ticker].append(bar)

        # Trim
        if len(self._bars_1m[ticker]) > MAX_1M_BARS:
            self._bars_1m[ticker] = self._bars_1m[ticker][-MAX_1M_BARS:]

        logger.info(
            f"Bar: {ticker} {bar_start.strftime('%H:%M')} "
            f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} "
            f"C={bar['close']:.2f} V={bar['volume']:.0f} "
            f"({len(self._bars_1m[ticker])} bars)"
        )

    # ----------------------------------------------------------------
    # File writer
    # ----------------------------------------------------------------

    async def _file_writer_loop(self) -> None:
        """Write current state to data/live_prices.json every second."""
        while self._running:
            try:
                output = {}
                for ticker in self.tickers:
                    output[ticker] = {
                        "last_price": self._last_prices[ticker],
                        "bars_1m": self._bars_1m[ticker],
                        "updated_at": datetime.now().isoformat(),
                    }

                # Atomic write (write to temp, then rename)
                tmp_path = LIVE_PRICES_PATH.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(output))
                tmp_path.rename(LIVE_PRICES_PATH)

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"File write error: {e}")
                await asyncio.sleep(2)


async def main():
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.error("FINNHUB_API_KEY not set")
        sys.exit(1)

    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    streamer = FinnhubStreamer(tickers=tickers, api_key=api_key)
    await streamer.run()


if __name__ == "__main__":
    asyncio.run(main())
