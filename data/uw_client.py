"""
Unusual Whales API client.

Serves as the intelligence layer — never provides execution-critical data.
Graceful degradation: if UW is unreachable, the bot continues on IBKR data alone.

REST endpoints: https://api.unusualwhales.com
WebSocket: Real-time GEX streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, List, Any

import httpx

from data.market_state import Bias

logger = logging.getLogger(__name__)


class UWClient:
    """
    Unusual Whales REST + WebSocket client.
    All methods handle errors gracefully — UW being down never stops the bot.
    """

    def __init__(self, config: dict):
        self.config = config.get("unusual_whales", {})
        self.base_url = self.config.get("base_url", "https://api.unusualwhales.com")
        self.api_token = self.config.get("api_token") or os.environ.get("UW_API_TOKEN", "")
        self.timeout = self.config.get("request_timeout_sec", 10)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay_sec", 2)
        self.available = False
        self._last_request_time: Dict[str, float] = {}

        # Rate limiting
        self._request_timestamps: List[float] = []

        if not self.api_token:
            logger.warning(
                "No Unusual Whales API token configured. "
                "UW intelligence features will be disabled."
            )

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Make an authenticated GET request to the UW API.
        Returns None on any failure (graceful degradation).
        """
        if not self.api_token:
            return None

        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        url,
                        headers=self.headers,
                        params=params,
                    )

                if response.status_code == 200:
                    self.available = True
                    data = response.json()
                    return data.get("data", data)

                elif response.status_code == 429:
                    # Rate limited
                    logger.warning(f"UW API rate limited on {endpoint}. Backing off.")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

                elif response.status_code == 401:
                    logger.error("UW API authentication failed. Check API token.")
                    self.available = False
                    return None

                else:
                    logger.warning(
                        f"UW API {endpoint} returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay)

            except httpx.TimeoutException:
                logger.warning(f"UW API timeout on {endpoint} (attempt {attempt + 1})")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.warning(f"UW API error on {endpoint}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        self.available = False
        return None

    # --- Net Premium Flow ---

    async def get_net_premium_flow(self, ticker: str) -> Optional[dict]:
        """
        Get net premium flow (calls vs puts) for a ticker.
        Returns direction and magnitude of institutional flow.
        """
        data = await self._request(f"/api/stock/{ticker}/net-prem-ticks")
        if data is None:
            return None

        # Parse the flow data to determine direction
        try:
            if isinstance(data, list) and len(data) > 0:
                recent = data[-1] if isinstance(data[-1], dict) else {}
                net_prem = float(recent.get("net_premium", 0))
                return {
                    "net_premium": net_prem,
                    "direction": (
                        Bias.BULLISH if net_prem > 0
                        else Bias.BEARISH if net_prem < 0
                        else Bias.NEUTRAL
                    ),
                    "raw": data[-5:] if len(data) >= 5 else data,
                }
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Error parsing net premium for {ticker}: {e}")

        return None

    # --- Flow Alerts ---

    async def get_flow_alerts(self) -> Optional[List[dict]]:
        """Get unusual options flow alerts across all tickers."""
        data = await self._request("/api/option-trades/flow-alerts")
        if data is None:
            return None
        if isinstance(data, list):
            return data[:20]  # Limit to most recent 20
        return None

    async def get_ticker_flow(self, ticker: str) -> Optional[List[dict]]:
        """Get recent options flow for a specific ticker."""
        data = await self._request(f"/api/stock/{ticker}/flow-recent")
        if data is None:
            return None
        if isinstance(data, list):
            return data[:20]
        return None

    # --- GEX / Gamma Exposure ---

    async def get_gex_data(self, ticker: str) -> Optional[dict]:
        """
        Get gamma exposure levels for a ticker.
        Identifies key gamma walls that act as support/resistance.
        """
        data = await self._request(f"/api/stock/{ticker}/gamma-exposure")
        if data is None:
            # Try alternative endpoint
            data = await self._request(f"/api/stock/{ticker}/greek-exposure")
        return data

    def parse_gex_walls(
        self,
        gex_data: Optional[dict],
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Parse GEX data to find the nearest gamma wall and its distance.
        """
        result = {
            "walls": [],
            "nearest_wall_price": 0.0,
            "nearest_wall_distance_pct": 999.0,
        }

        if gex_data is None or current_price <= 0:
            return result

        try:
            # GEX data structure varies; handle common formats
            levels = []
            if isinstance(gex_data, list):
                levels = gex_data
            elif isinstance(gex_data, dict):
                levels = gex_data.get("levels", gex_data.get("data", []))

            if not levels:
                return result

            walls = []
            for level in levels:
                if isinstance(level, dict):
                    strike = float(level.get("strike", level.get("price", 0)))
                    gex_val = float(level.get("gex", level.get("gamma_exposure", 0)))
                    if strike > 0 and abs(gex_val) > 0:
                        walls.append({
                            "strike": strike,
                            "gex": gex_val,
                            "distance_pct": abs(strike - current_price) / current_price,
                        })

            if walls:
                walls.sort(key=lambda x: x["distance_pct"])
                result["walls"] = walls[:5]
                result["nearest_wall_price"] = walls[0]["strike"]
                result["nearest_wall_distance_pct"] = walls[0]["distance_pct"]

        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing GEX data: {e}")

        return result

    # --- Dark Pool ---

    async def get_dark_pool_data(self, ticker: str) -> Optional[dict]:
        """Get dark pool activity for a ticker."""
        data = await self._request(f"/api/stock/{ticker}/dark-pool")
        return data

    # --- Sector ETF Data ---

    async def get_sector_data(self) -> Optional[dict]:
        """Get sector ETF performance data."""
        data = await self._request("/api/etf/sectors")
        return data

    # --- Volatility / IV ---

    async def get_iv_data(self, ticker: str) -> Optional[dict]:
        """Get IV rank and IV percentile for a ticker."""
        data = await self._request(f"/api/stock/{ticker}/volatility")
        return data

    def parse_iv_percentile(self, iv_data: Optional[dict]) -> float:
        """Extract IV percentile from volatility data."""
        if iv_data is None:
            return 0.0
        try:
            if isinstance(iv_data, dict):
                return float(
                    iv_data.get("iv_percentile",
                    iv_data.get("ivPercentile",
                    iv_data.get("iv_rank", 0)))
                )
            if isinstance(iv_data, list) and len(iv_data) > 0:
                latest = iv_data[-1]
                return float(latest.get("iv_percentile", 0))
        except (ValueError, KeyError):
            pass
        return 0.0

    # --- Historical Candles (for pre-market scanner) ---

    async def get_historical_candles(
        self,
        ticker: str,
        period: str = "1M",
    ) -> Optional[List[dict]]:
        """Get historical candle data for a ticker (pre-market scanner use)."""
        data = await self._request(
            f"/api/stock/{ticker}/candles",
            params={"period": period},
        )
        if data is None:
            return None
        if isinstance(data, list):
            return data
        return None

    # --- Option Chains (for IV/flow context, not for execution) ---

    async def get_option_chain(self, ticker: str) -> Optional[dict]:
        """Get option chain data from UW (for flow analysis, not execution)."""
        data = await self._request(f"/api/stock/{ticker}/option-chains")
        return data

    # --- Health Check ---

    async def check_health(self) -> bool:
        """Check if the UW API is reachable and authenticated."""
        if not self.api_token:
            return False
        data = await self._request("/api/etf/sectors")
        return data is not None

    def is_available(self) -> bool:
        """Check if UW API was reachable on last request."""
        return self.available
