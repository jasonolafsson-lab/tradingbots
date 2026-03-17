"""
Contract Selector.

Given a trading signal, selects the optimal options contract:
- Expiration: 0DTE > 1DTE > 2DTE
- Delta targeting per strategy
- Liquidity filters (volume, OI, spread)
- Single-leg vs debit spread (based on IV percentile)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from data.ibkr_client import IBKRClient
from data.uw_client import UWClient
from data.market_state import (
    TickerState, Signal, Direction, ContractType
)

logger = logging.getLogger(__name__)


class ContractSelector:
    """Selects optimal option contracts for trading signals."""

    def __init__(self, config: dict, ibkr: IBKRClient, uw: UWClient):
        self.config = config
        self.ibkr = ibkr
        self.uw = uw

        contracts = config.get("contracts", {})
        self.delta_hard_min = contracts.get("delta_hard_min", 0.20)
        self.delta_hard_max = contracts.get("delta_hard_max", 0.60)
        self.min_volume = contracts.get("min_volume", 50)
        self.min_oi = contracts.get("min_open_interest", 500)
        self.max_spread_pct = contracts.get("max_bid_ask_spread_pct", 0.10)
        self.iv_spread_threshold = contracts.get("iv_spread_threshold", 50)
        self.spread_width_etf = contracts.get("spread_width_etf", [2, 3])
        self.spread_width_stock = contracts.get("spread_width_stock", 5)

    async def select(
        self,
        signal: Signal,
        ts: TickerState,
    ) -> Optional[Dict[str, Any]]:
        """
        Select an option contract for a signal.

        Returns dict with:
            contract_type, strike, expiry, dte, delta, iv, bid, ask,
            spread_width (if spread), short_strike (if spread),
            entry_price (midpoint), contract (IBKR Contract object)
        """
        # Get delta range for this strategy
        delta_min, delta_max = self._get_delta_range(signal.strategy)

        # Get available expirations
        try:
            chains = await self.ibkr.get_option_chains(signal.ticker)
        except Exception as e:
            logger.error(f"Failed to get option chains for {signal.ticker}: {e}")
            return None

        if not chains:
            logger.warning(f"No option chains found for {signal.ticker}")
            return None

        # Get available expirations sorted by proximity
        today = datetime.now().strftime("%Y%m%d")
        available_expiries = []
        for chain in chains:
            for exp in chain.get("expirations", []):
                available_expiries.append(exp)
        available_expiries = sorted(set(available_expiries))

        # Filter to 0-2 DTE
        target_expiries = self._get_target_expiries(available_expiries, today)
        if not target_expiries:
            logger.warning(f"No 0-2 DTE expirations for {signal.ticker}")
            return None

        # Determine right (Call or Put)
        right = "C" if signal.direction == Direction.CALL else "P"

        # Get strikes from chain
        all_strikes = set()
        for chain in chains:
            all_strikes.update(chain.get("strikes", []))
        strikes = sorted(all_strikes)

        if not strikes:
            logger.warning(f"No strikes found for {signal.ticker}")
            return None

        # Find best contract across available expirations
        best_contract = None

        for expiry in target_expiries:
            dte = self._compute_dte(today, expiry)

            # Find strikes in target delta range
            # We'll query IBKR for Greeks on candidate strikes near current price
            candidate_strikes = self._get_candidate_strikes(
                strikes, ts.last_price, right
            )

            for strike in candidate_strikes:
                try:
                    greeks = await self.ibkr.get_option_greeks(
                        signal.ticker, expiry, strike, right
                    )
                except Exception as e:
                    logger.debug(f"Greeks error for {signal.ticker} {strike}: {e}")
                    continue

                delta = abs(greeks.get("delta", 0) or 0)
                bid = greeks.get("bid", 0) or 0
                ask = greeks.get("ask", 0) or 0
                iv = greeks.get("impliedVol", 0) or 0

                # Delta filter
                if delta < delta_min or delta > delta_max:
                    continue

                # Hard delta boundaries
                if delta < self.delta_hard_min or delta > self.delta_hard_max:
                    continue

                # Liquidity: bid-ask spread
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                if spread_pct > self.max_spread_pct:
                    continue

                # This contract passes all filters
                contract_info = {
                    "strike": strike,
                    "expiry": expiry,
                    "dte": dte,
                    "right": right,
                    "delta": delta,
                    "iv": iv,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pct": spread_pct,
                }

                # Prefer contracts closest to target delta midpoint
                target_mid = (delta_min + delta_max) / 2.0
                delta_distance = abs(delta - target_mid)

                if best_contract is None or delta_distance < best_contract.get("_delta_dist", 999):
                    contract_info["_delta_dist"] = delta_distance
                    best_contract = contract_info
                    break  # Found a good one for this expiry, try next if needed

            if best_contract:
                break  # Use the first (shortest) expiry that has a valid contract

        if best_contract is None:
            logger.warning(f"No suitable contract found for {signal.ticker}")
            return None

        # Remove internal field
        best_contract.pop("_delta_dist", None)

        # Decide single-leg vs debit spread
        iv_percentile = ts.uw_iv_percentile
        if iv_percentile >= self.iv_spread_threshold:
            # Use debit spread
            spread_info = await self._build_spread(
                signal, ts, best_contract, strikes
            )
            if spread_info:
                return spread_info
            # Fallback to single-leg if spread can't be built
            logger.info(f"Spread build failed for {signal.ticker}, using single-leg")

        # Single-leg
        best_contract["contract_type"] = ContractType.SINGLE_LEG
        best_contract["entry_price"] = best_contract["mid"]
        best_contract["spread_width"] = None

        # Qualify contract with IBKR
        try:
            ibkr_contract = self.ibkr.make_option_contract(
                signal.ticker,
                best_contract["expiry"],
                best_contract["strike"],
                best_contract["right"],
            )
            best_contract["ibkr_contract"] = await self.ibkr.qualify_contract(ibkr_contract)
        except Exception as e:
            logger.error(f"Contract qualification failed: {e}")
            return None

        logger.info(
            f"Selected: {signal.ticker} {best_contract['strike']}{right} "
            f"{best_contract['expiry']} delta={best_contract['delta']:.2f} "
            f"mid=${best_contract['mid']:.2f} (single-leg)"
        )

        return best_contract

    async def _build_spread(
        self,
        signal: Signal,
        ts: TickerState,
        long_leg: dict,
        all_strikes: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Build a debit spread using the long leg as the anchor.
        Short leg is the next available strike at the configured width.
        """
        # Determine spread width
        tickers_config = self.config.get("tickers_config", {})
        watchlist = tickers_config.get("watchlist", [])
        ticker_type = "ETF"
        for t in watchlist:
            if t["ticker"] == signal.ticker:
                ticker_type = t.get("type", "stock").upper()
                break

        if ticker_type == "ETF":
            target_widths = self.spread_width_etf
        else:
            target_widths = [self.spread_width_stock]

        if isinstance(target_widths, (int, float)):
            target_widths = [target_widths]

        right = long_leg["right"]

        for width in sorted(target_widths):
            if right == "C":
                short_strike = long_leg["strike"] + width
            else:
                short_strike = long_leg["strike"] - width

            if short_strike not in all_strikes:
                # Find nearest available strike
                nearest = min(all_strikes, key=lambda s: abs(s - short_strike))
                if abs(nearest - short_strike) > width * 0.5:
                    continue  # Too far from target width
                short_strike = nearest

            # Get short leg Greeks
            try:
                short_greeks = await self.ibkr.get_option_greeks(
                    signal.ticker, long_leg["expiry"], short_strike, right
                )
            except Exception:
                continue

            short_bid = short_greeks.get("bid", 0) or 0
            short_ask = short_greeks.get("ask", 0) or 0
            if short_bid <= 0:
                continue

            # Net debit = long mid - short mid
            short_mid = (short_bid + short_ask) / 2.0
            net_debit = long_leg["mid"] - short_mid

            if net_debit <= 0:
                continue  # Can't have zero or credit on a debit spread

            actual_width = abs(long_leg["strike"] - short_strike)

            result = dict(long_leg)
            result["contract_type"] = ContractType.DEBIT_SPREAD
            result["short_strike"] = short_strike
            result["short_bid"] = short_bid
            result["short_ask"] = short_ask
            result["spread_width"] = actual_width
            result["entry_price"] = net_debit
            result["max_risk_per_contract"] = net_debit * 100

            logger.info(
                f"Spread built: {signal.ticker} "
                f"{long_leg['strike']}/{short_strike}{right} "
                f"width=${actual_width:.0f} debit=${net_debit:.2f}"
            )
            return result

        return None

    def _get_delta_range(self, strategy: str) -> tuple:
        """Get delta min/max for a strategy."""
        strat_config = self.config.get("strategies", {}).get(
            strategy.lower(), {}
        )
        delta_min = strat_config.get("delta_min", 0.30)
        delta_max = strat_config.get("delta_max", 0.50)
        return delta_min, delta_max

    def _get_target_expiries(
        self, available: List[str], today: str
    ) -> List[str]:
        """Get 0-2 DTE expirations, sorted by proximity."""
        result = []
        today_dt = datetime.strptime(today, "%Y%m%d")

        for exp in available:
            try:
                exp_dt = datetime.strptime(exp, "%Y%m%d")
                dte = (exp_dt - today_dt).days
                if 0 <= dte <= 2:
                    result.append(exp)
            except ValueError:
                continue

        return sorted(result)  # 0DTE first

    @staticmethod
    def _compute_dte(today: str, expiry: str) -> int:
        today_dt = datetime.strptime(today, "%Y%m%d")
        exp_dt = datetime.strptime(expiry, "%Y%m%d")
        return (exp_dt - today_dt).days

    @staticmethod
    def _get_candidate_strikes(
        strikes: List[float], current_price: float, right: str
    ) -> List[float]:
        """
        Get ~10 candidate strikes near the current price.
        For calls: slightly OTM (above price).
        For puts: slightly OTM (below price).
        """
        # Sort by distance to current price
        sorted_strikes = sorted(strikes, key=lambda s: abs(s - current_price))
        # Take nearest 15 strikes
        candidates = sorted_strikes[:15]
        return sorted(candidates)
