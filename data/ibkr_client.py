"""
Interactive Brokers client using ib_async.

Handles:
- Connection to IB Gateway (paper only)
- Account verification and allowlist
- Streaming real-time bars (1-min, 3-min)
- Real-time option quotes and Greeks
- Option chain discovery
- Order submission, cancellation, and status
- Position and P&L queries
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Callable, Dict, Any

from ib_async import (
    IB, Stock, Option, Contract, ComboLeg, Order, LimitOrder,
    MarketOrder, Trade, BarData, Ticker, AccountValue,
    RealTimeBarList, util
)

from data.market_state import Bar, TickerState

logger = logging.getLogger(__name__)


class IBKRConnectionError(Exception):
    pass


class IBKRAccountError(Exception):
    pass


class IBKRClient:
    """
    Manages all interactions with Interactive Brokers via IB Gateway.
    Paper-only enforcement is built in.
    """

    BLOCKED_PORTS = {7496}  # Live trading port — HARD BLOCKED

    def __init__(self, config: dict):
        self.config = config
        self.ib = IB()
        self.host = config["ibkr"]["host"]
        self.port = config["ibkr"]["port"]
        self.client_id = config["ibkr"]["client_id"]
        self.allowed_accounts = config["ibkr"]["allowed_accounts"]
        self.timeout = config["ibkr"]["timeout_sec"]
        self.connected = False
        self.account_id: Optional[str] = None

        # Bar subscriptions: ticker -> RealTimeBarList
        self._bar_subscriptions: Dict[str, Any] = {}
        # Market data subscriptions: contract -> Ticker
        self._mkt_data_subs: Dict[str, Any] = {}
        # Callbacks for new bar data
        self._bar_callbacks: List[Callable] = []

        # Safety check: block live port
        if self.port in self.BLOCKED_PORTS:
            raise IBKRConnectionError(
                f"Port {self.port} is BLOCKED. Live trading is not allowed in V1. "
                f"Use port 7497 for paper trading."
            )

    async def connect(self) -> None:
        """Connect to IB Gateway and verify paper account."""
        logger.info(f"Connecting to IBKR at {self.host}:{self.port} "
                     f"(client_id={self.client_id})")
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.config["ibkr"].get("readonly", False),
            )
        except Exception as e:
            raise IBKRConnectionError(
                f"Failed to connect to IB Gateway at {self.host}:{self.port}: {e}"
            ) from e

        self.connected = True
        logger.info("Connected to IB Gateway")

        # Verify account
        await self._verify_account()

    async def _verify_account(self) -> None:
        """Verify we're connected to an allowed paper account."""
        accounts = self.ib.managedAccounts()
        if not accounts:
            raise IBKRAccountError("No managed accounts found")

        self.account_id = accounts[0]
        logger.info(f"Account ID: {self.account_id}")

        # Check allowlist (if configured)
        if self.allowed_accounts:
            if self.account_id not in self.allowed_accounts:
                await self.disconnect()
                raise IBKRAccountError(
                    f"Account {self.account_id} is NOT in the allowed list. "
                    f"Allowed: {self.allowed_accounts}. "
                    f"This is a safety check to prevent trading on wrong accounts."
                )

        # Verify it looks like a paper account (DU prefix)
        if not self.account_id.startswith("DU"):
            logger.warning(
                f"Account {self.account_id} does not have 'DU' prefix. "
                f"Paper accounts typically start with 'DU'. "
                f"Proceeding but please verify this is a paper account."
            )

        logger.info(f"Account {self.account_id} verified")

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")

    # --- Account Queries ---

    async def get_account_summary(self) -> Dict[str, float]:
        """Get key account metrics."""
        summary = {}
        account_values = self.ib.accountValues(self.account_id)
        for av in account_values:
            if av.tag in ("NetLiquidation", "BuyingPower", "TotalCashValue",
                          "UnrealizedPnL", "RealizedPnL"):
                if av.currency == "USD":
                    summary[av.tag] = float(av.value)
        return summary

    async def get_buying_power(self) -> float:
        """Get current buying power."""
        summary = await self.get_account_summary()
        return summary.get("BuyingPower", 0.0)

    async def get_net_liquidation(self) -> float:
        """Get net liquidation value (total account value)."""
        summary = await self.get_account_summary()
        return summary.get("NetLiquidation", 0.0)

    async def get_daily_pnl(self) -> float:
        """Get today's realized + unrealized P&L."""
        pnl = self.ib.pnl(self.account_id)
        if pnl:
            return pnl[0].dailyPnL or 0.0
        return 0.0

    # --- Positions ---

    async def get_positions(self) -> List[dict]:
        """Get all open positions."""
        positions = self.ib.positions(self.account_id)
        result = []
        for pos in positions:
            result.append({
                "contract": pos.contract,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.marketValue if hasattr(pos, 'marketValue') else None,
            })
        return result

    async def has_open_position(self) -> bool:
        """Check if there's any open option position."""
        positions = await self.get_positions()
        return any(
            abs(p["quantity"]) > 0 and
            p["contract"].secType in ("OPT", "BAG")
            for p in positions
        )

    # --- Market Data ---

    def make_stock_contract(self, ticker: str) -> Stock:
        """Create a stock contract for a ticker."""
        return Stock(ticker, "SMART", "USD")

    def make_option_contract(
        self,
        ticker: str,
        expiry: str,      # YYYYMMDD
        strike: float,
        right: str,        # "C" or "P"
    ) -> Option:
        """Create an option contract."""
        return Option(ticker, expiry, strike, right, "SMART", currency="USD")

    async def qualify_contract(self, contract: Contract) -> Contract:
        """Qualify a contract to ensure it's tradable on IBKR."""
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"Contract could not be qualified: {contract}")
        return qualified[0]

    async def get_option_chains(self, ticker: str) -> List[dict]:
        """Get available option expirations and strikes for a ticker."""
        stock = self.make_stock_contract(ticker)
        await self.qualify_contract(stock)
        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId
        )
        result = []
        for chain in chains:
            if chain.exchange == "SMART":
                result.append({
                    "exchange": chain.exchange,
                    "expirations": sorted(chain.expirations),
                    "strikes": sorted(chain.strikes),
                })
        return result

    async def get_option_greeks(
        self,
        ticker: str,
        expiry: str,
        strike: float,
        right: str,
    ) -> dict:
        """Get real-time Greeks for a specific option contract."""
        contract = self.make_option_contract(ticker, expiry, strike, right)
        await self.qualify_contract(contract)

        ticker_data = self.ib.reqMktData(contract, genericTickList="106")
        # Wait briefly for data to arrive
        await asyncio.sleep(2)

        greeks = {}
        if ticker_data.modelGreeks:
            mg = ticker_data.modelGreeks
            greeks = {
                "delta": mg.delta,
                "gamma": mg.gamma,
                "theta": mg.theta,
                "vega": mg.vega,
                "impliedVol": mg.impliedVol,
            }
        greeks["bid"] = ticker_data.bid
        greeks["ask"] = ticker_data.ask
        greeks["last"] = ticker_data.last

        self.ib.cancelMktData(contract)
        return greeks

    async def subscribe_realtime_bars(
        self,
        ticker: str,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Subscribe to real-time 5-second bars for a ticker.
        These are aggregated into 1-min and 3-min bars internally.
        """
        stock = self.make_stock_contract(ticker)
        await self.qualify_contract(stock)

        bars = self.ib.reqRealTimeBars(
            stock,
            barSize=5,          # 5-second bars
            whatToShow="TRADES",
            useRTH=True,
        )

        self._bar_subscriptions[ticker] = bars
        if callback:
            bars.updateEvent += callback
        logger.info(f"Subscribed to real-time bars for {ticker}")

    async def get_historical_bars(
        self,
        ticker: str,
        duration: str = "2 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> List[Bar]:
        """Get historical bar data for a ticker."""
        stock = self.make_stock_contract(ticker)
        await self.qualify_contract(stock)

        bars = await self.ib.reqHistoricalDataAsync(
            stock,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )

        result = []
        for b in bars:
            result.append(Bar(
                timestamp=b.date if isinstance(b.date, datetime) else datetime.now(),
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
            ))
        return result

    async def get_prior_day_data(self, ticker: str) -> dict:
        """Get prior day OHLCV for a ticker."""
        bars = await self.get_historical_bars(
            ticker,
            duration="2 D",
            bar_size="1 day",
            use_rth=True,
        )
        if len(bars) >= 2:
            prior = bars[-2]  # Second to last = prior day
            return {
                "high": prior.high,
                "low": prior.low,
                "close": prior.close,
                "volume": prior.volume,
                "open": prior.open,
            }
        elif len(bars) == 1:
            prior = bars[0]
            return {
                "high": prior.high,
                "low": prior.low,
                "close": prior.close,
                "volume": prior.volume,
                "open": prior.open,
            }
        return {"high": 0, "low": 0, "close": 0, "volume": 0, "open": 0}

    # --- Order Management ---

    async def place_limit_order(
        self,
        contract: Contract,
        action: str,      # "BUY" or "SELL"
        quantity: int,
        limit_price: float,
    ) -> Trade:
        """Place a limit order."""
        order = LimitOrder(action, quantity, limit_price)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            f"Placed {action} limit order: {contract.symbol} "
            f"qty={quantity} @ ${limit_price:.2f}"
        )
        return trade

    async def place_bracket_order(
        self,
        contract: Contract,
        action: str,
        quantity: int,
        limit_price: float,
        stop_price: float,
        tp_price: float,
    ) -> List[Trade]:
        """Place a bracket order (entry + stop + take profit)."""
        bracket = self.ib.bracketOrder(
            action=action,
            quantity=quantity,
            limitPrice=limit_price,
            takeProfitPrice=tp_price,
            stopLossPrice=stop_price,
        )
        trades = []
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            trades.append(trade)
        logger.info(
            f"Placed bracket: {action} {contract.symbol} "
            f"entry=${limit_price:.2f} stop=${stop_price:.2f} tp=${tp_price:.2f}"
        )
        return trades

    async def place_combo_order(
        self,
        legs: List[dict],   # [{"contract": Contract, "action": str, "ratio": int}]
        action: str,
        quantity: int,
        limit_price: float,
    ) -> Trade:
        """Place a combo (spread) order as a single atomic BAG order."""
        combo = Contract()
        combo.symbol = legs[0]["contract"].symbol
        combo.secType = "BAG"
        combo.exchange = "SMART"
        combo.currency = "USD"

        combo_legs = []
        for leg in legs:
            cl = ComboLeg()
            cl.conId = leg["contract"].conId
            cl.ratio = leg["ratio"]
            cl.action = leg["action"]
            cl.exchange = "SMART"
            combo_legs.append(cl)
        combo.comboLegs = combo_legs

        order = LimitOrder(action, quantity, limit_price)
        order.tif = "DAY"
        trade = self.ib.placeOrder(combo, order)

        logger.info(
            f"Placed combo order: {combo.symbol} "
            f"qty={quantity} @ ${limit_price:.2f} (net debit/credit)"
        )
        return trade

    async def cancel_order(self, trade: Trade) -> None:
        """Cancel a pending order."""
        self.ib.cancelOrder(trade.order)
        logger.info(f"Cancelled order: {trade.order.orderId}")

    async def cancel_all_orders(self) -> None:
        """Cancel ALL open orders (emergency)."""
        open_orders = await self.ib.reqAllOpenOrdersAsync()
        for trade in self.ib.openTrades():
            self.ib.cancelOrder(trade.order)
        logger.warning("Cancelled ALL open orders (emergency)")

    async def close_all_positions(self) -> None:
        """Close all open option positions at market (emergency / EOD)."""
        positions = await self.get_positions()
        for pos in positions:
            if abs(pos["quantity"]) > 0 and pos["contract"].secType in ("OPT", "BAG"):
                action = "SELL" if pos["quantity"] > 0 else "BUY"
                qty = abs(pos["quantity"])
                order = MarketOrder(action, qty)
                order.tif = "DAY"
                self.ib.placeOrder(pos["contract"], order)
                logger.warning(
                    f"Emergency close: {action} {qty}x {pos['contract'].symbol}"
                )

    # --- Utilities ---

    async def wait_for_fill(self, trade: Trade, timeout_sec: float = 15) -> bool:
        """Wait for an order to fill, with timeout."""
        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < timeout_sec:
            if trade.isDone():
                return trade.orderStatus.status == "Filled"
            await asyncio.sleep(0.5)
        return False

    def is_connected(self) -> bool:
        """Check if still connected to IB Gateway."""
        return self.ib.isConnected()

    async def sleep(self, seconds: float) -> None:
        """Async sleep that keeps IB event loop running."""
        await self.ib.sleepAsync(seconds)
