"""
Pre-Market Scanner.

Runs at 9:00 AM ET daily. Evaluates each watchlist ticker and produces:
- Directional bias (BULLISH / BEARISH / NEUTRAL)
- Day 2 candidacy score (0-100)
- Sector relative strength
- Priority ranking

This drives which tickers and strategies get attention during the session.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from data.ibkr_client import IBKRClient
from data.uw_client import UWClient
from data.market_state import (
    MarketState, TickerState, ScannerResult, Bias, Bar
)

logger = logging.getLogger(__name__)


class PreMarketScanner:
    """
    Pre-market scanner that ranks watchlist tickers for the session.
    """

    def __init__(self, config: dict, ibkr: IBKRClient, uw: UWClient):
        self.config = config
        self.ibkr = ibkr
        self.uw = uw

    async def run_scan(self, market_state: MarketState) -> List[ScannerResult]:
        """
        Run the full pre-market scan for all tickers.
        Returns ranked ScannerResult list.
        """
        results = []

        for ticker in market_state.tickers:
            try:
                result = await self._scan_ticker(ticker, market_state)
                results.append(result)
            except Exception as e:
                logger.error(f"Scanner error for {ticker}: {e}")
                results.append(ScannerResult(ticker=ticker))

        # Sort by priority (Day 2 candidates first, then by composite score)
        results.sort(key=lambda r: (-r.day2_score, -r.sector_rs), reverse=False)
        results.sort(key=lambda r: r.day2_score, reverse=True)

        # Assign priority ranks
        for i, result in enumerate(results):
            result.priority_rank = i + 1

        return results

    async def _scan_ticker(
        self, ticker: str, market_state: MarketState
    ) -> ScannerResult:
        """Scan a single ticker and produce a ScannerResult."""
        result = ScannerResult(ticker=ticker)
        ts = market_state.get_ticker(ticker)

        # 1. Get prior-day data from IBKR
        prior = await self.ibkr.get_prior_day_data(ticker)
        ts.prior_high = prior["high"]
        ts.prior_low = prior["low"]
        ts.prior_close = prior["close"]
        ts.prior_volume = prior["volume"]

        # 2. Prior-day close quality: (close - low) / (high - low)
        price_range = ts.prior_high - ts.prior_low
        if price_range > 0:
            result.close_quality = (ts.prior_close - ts.prior_low) / price_range
            ts.close_quality = result.close_quality

        # 3. Volume character — prior day vs 20-day average
        hist_bars = await self.ibkr.get_historical_bars(
            ticker, duration="25 D", bar_size="1 day", use_rth=True
        )
        if len(hist_bars) >= 20:
            avg_vol = np.mean([b.volume for b in hist_bars[-21:-1]])
            ts.prior_volume_20d_avg = avg_vol
            result.volume_vs_avg = ts.prior_volume / avg_vol if avg_vol > 0 else 1.0

        # 4. Catalyst presence (UW flow data)
        catalyst_score = 0.0
        if self.uw.is_available() or self.uw.api_token:
            flow = await self.uw.get_net_premium_flow(ticker)
            if flow:
                # Strong directional flow = catalyst
                net_prem = abs(flow.get("net_premium", 0))
                if net_prem > 1_000_000:
                    catalyst_score += 30
                elif net_prem > 500_000:
                    catalyst_score += 15
                ts.uw_net_premium_direction = flow.get("direction", Bias.NEUTRAL)

            # Check for unusual flow
            ticker_flow = await self.uw.get_ticker_flow(ticker)
            if ticker_flow and len(ticker_flow) > 5:
                catalyst_score += 10  # Elevated activity

            # Dark pool
            dp = await self.uw.get_dark_pool_data(ticker)
            if dp:
                catalyst_score += 5

            # IV percentile
            iv_data = await self.uw.get_iv_data(ticker)
            ts.uw_iv_percentile = self.uw.parse_iv_percentile(iv_data)

        result.catalyst_score = catalyst_score

        # 5. Key levels
        result.key_levels = {
            "prior_high": ts.prior_high,
            "prior_low": ts.prior_low,
            "prior_close": ts.prior_close,
        }

        # Compute moving averages from historical data
        if len(hist_bars) >= 50:
            closes = [b.close for b in hist_bars]
            result.key_levels["sma_20"] = np.mean(closes[-20:])
            result.key_levels["sma_50"] = np.mean(closes[-50:])

        # 6. Day 2 candidacy score (0-100)
        result.day2_score = self._compute_day2_score(result)

        # 7. Determine bias
        result.bias = self._determine_bias(result, ts)

        return result

    def _compute_day2_score(self, result: ScannerResult) -> float:
        """
        Compute Day 2 candidacy score (0-100).
        Combines: close quality + volume + catalyst + flow direction.
        """
        score = 0.0

        # Close quality (max 30 points)
        # Strong close (>0.80) = bullish Day 2 candidate
        # Weak close (<0.20) = bearish Day 2 candidate
        if result.close_quality > 0.80:
            score += 30 * ((result.close_quality - 0.80) / 0.20)
        elif result.close_quality < 0.20:
            score += 30 * ((0.20 - result.close_quality) / 0.20)

        # Volume character (max 25 points)
        if result.volume_vs_avg > 2.0:
            score += 25
        elif result.volume_vs_avg > 1.5:
            score += 20
        elif result.volume_vs_avg > 1.2:
            score += 10

        # Catalyst (max 35 points)
        score += min(result.catalyst_score, 35)

        # Sector context (max 10 points)
        if abs(result.sector_rs) > 0.005:
            score += 10

        return min(score, 100.0)

    @staticmethod
    def _determine_bias(result: ScannerResult, ts: TickerState) -> Bias:
        """Determine directional bias from scanner data."""
        bullish_signals = 0
        bearish_signals = 0

        # Close quality
        if result.close_quality > 0.70:
            bullish_signals += 1
        elif result.close_quality < 0.30:
            bearish_signals += 1

        # Volume + direction
        if ts.uw_net_premium_direction == Bias.BULLISH:
            bullish_signals += 1
        elif ts.uw_net_premium_direction == Bias.BEARISH:
            bearish_signals += 1

        # Above/below key levels
        if ts.prior_close > result.key_levels.get("sma_20", 0):
            bullish_signals += 1
        elif ts.prior_close < result.key_levels.get("sma_20", float("inf")):
            bearish_signals += 1

        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            return Bias.BULLISH
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            return Bias.BEARISH
        return Bias.NEUTRAL
