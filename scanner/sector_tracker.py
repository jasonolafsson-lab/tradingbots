"""
Sector ETF Relative Strength Tracker.

Monitors sector ETF returns vs SPY to detect:
- Green Sector conditions (sector green while SPY red)
- Relative strength/weakness for signal prioritization
- Next-day carryover bias
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from data.ibkr_client import IBKRClient
from data.uw_client import UWClient
from data.market_state import MarketState

logger = logging.getLogger(__name__)


class SectorTracker:
    """Tracks sector ETF performance relative to SPY."""

    def __init__(self, config: dict, ibkr: IBKRClient, uw: UWClient):
        self.config = config
        self.ibkr = ibkr
        self.uw = uw
        self._last_update: float = 0
        self._update_interval = config.get("unusual_whales", {}).get(
            "sector_refresh", 300
        )

        # Load sector ETFs from config
        tickers_config = config.get("tickers_config", {})
        self.sector_etfs = [
            s["ticker"] for s in tickers_config.get("sector_etfs", [])
        ]
        self.sector_names = {
            s["ticker"]: s["sector"]
            for s in tickers_config.get("sector_etfs", [])
        }

    async def update(self, market_state: MarketState) -> None:
        """
        Update sector returns and green sector detection.
        Only refreshes every N seconds (configured interval).
        """
        now = time.time()
        if now - self._last_update < self._update_interval:
            return

        self._last_update = now

        # Try UW sector data first
        if self.uw.is_available():
            sector_data = await self.uw.get_sector_data()
            if sector_data:
                self._parse_uw_sector_data(sector_data, market_state)
                self._detect_green_sectors(market_state)
                return

        # Fallback: compute from IBKR price data
        await self._compute_from_ibkr(market_state)
        self._detect_green_sectors(market_state)

    def _parse_uw_sector_data(
        self, data: dict, market_state: MarketState
    ) -> None:
        """Parse sector data from Unusual Whales API."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    ticker = item.get("ticker", item.get("symbol", ""))
                    change = item.get("change_percent", item.get("return", 0))
                    if ticker and change is not None:
                        market_state.sector_returns[ticker] = float(change)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    change = value.get("change_percent", value.get("return", 0))
                    market_state.sector_returns[key] = float(change) if change else 0.0

    async def _compute_from_ibkr(self, market_state: MarketState) -> None:
        """Compute sector returns from IBKR real-time data."""
        for etf_ticker in self.sector_etfs:
            try:
                # Get today's return for this sector ETF
                ts = market_state.tickers.get(etf_ticker)
                if ts and ts.prior_close > 0 and ts.last_price > 0:
                    ret = (ts.last_price - ts.prior_close) / ts.prior_close
                    market_state.sector_returns[etf_ticker] = ret
            except Exception as e:
                logger.debug(f"Could not compute return for {etf_ticker}: {e}")

    def _detect_green_sectors(self, market_state: MarketState) -> None:
        """
        Detect sectors that are green while SPY is red.
        This powers the Green Sector strategy.
        """
        spy_return = market_state.spy_session_return
        green_threshold = self.config.get("strategies", {}).get(
            "green_sector", {}
        ).get("spy_red_threshold", -0.003)

        market_state.green_sectors = []

        if spy_return >= green_threshold:
            # SPY is not red enough for Green Sector detection
            return

        for etf_ticker, ret in market_state.sector_returns.items():
            if ret > 0:
                sector_name = self.sector_names.get(etf_ticker, etf_ticker)
                market_state.green_sectors.append(sector_name)
                logger.info(
                    f"Green Sector detected: {sector_name} ({etf_ticker}) "
                    f"+{ret:.2%} while SPY {spy_return:.2%}"
                )

    def get_sector_for_ticker(self, ticker: str) -> str:
        """Get the sector name for a given watchlist ticker."""
        tickers_config = self.config.get("tickers_config", {})
        mapping = tickers_config.get("sector_mapping", {})
        for sector_name, info in mapping.items():
            if ticker in info.get("tickers", []):
                return sector_name
        return ""

    def get_sector_return(
        self, sector_name: str, market_state: MarketState
    ) -> float:
        """Get the return for a sector by name."""
        tickers_config = self.config.get("tickers_config", {})
        mapping = tickers_config.get("sector_mapping", {})
        info = mapping.get(sector_name, {})
        etf = info.get("etf", "")
        if etf:
            return market_state.sector_returns.get(etf, 0.0)
        return 0.0
