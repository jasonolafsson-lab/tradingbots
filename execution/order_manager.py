"""
Order Manager.

Handles order submission, repricing, bracket placement, and position closing.
All orders go through IBKR. Supports single-leg and combo (spread) orders.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any

from data.ibkr_client import IBKRClient
from data.market_state import (
    Position, Signal, Direction, ContractType, ExitReason
)

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages the order lifecycle for the bot."""

    def __init__(self, config: dict, ibkr: IBKRClient):
        self.config = config
        self.ibkr = ibkr

        exec_config = config.get("execution", {})
        self.midpoint_offset = exec_config.get("midpoint_offset", 0.02)
        self.fill_timeout = exec_config.get("fill_timeout_sec", 15)
        self.max_reprices = exec_config.get("max_reprices", 2)
        self.reprice_step = exec_config.get("reprice_step", 0.02)

    async def submit_entry(
        self,
        contract_info: Dict[str, Any],
        quantity: int,
        signal: Signal,
    ):
        """
        Submit an entry order (single-leg or spread).
        Returns the IBKR Trade object.
        """
        contract_type = contract_info.get("contract_type", ContractType.SINGLE_LEG)

        if contract_type == ContractType.DEBIT_SPREAD:
            return await self._submit_spread_entry(contract_info, quantity, signal)
        else:
            return await self._submit_single_entry(contract_info, quantity, signal)

    async def _submit_single_entry(
        self,
        contract_info: Dict[str, Any],
        quantity: int,
        signal: Signal,
    ):
        """Submit a single-leg limit order."""
        ibkr_contract = contract_info["ibkr_contract"]
        mid = contract_info["mid"]

        # Set limit price: mid + offset for buys
        limit_price = round(mid + self.midpoint_offset, 2)

        logger.info(
            f"Submitting entry: BUY {quantity}x "
            f"{signal.ticker} {contract_info['strike']}{contract_info['right']} "
            f"@ ${limit_price:.2f} (mid=${mid:.2f})"
        )

        trade = await self.ibkr.place_limit_order(
            contract=ibkr_contract,
            action="BUY",
            quantity=quantity,
            limit_price=limit_price,
        )
        return trade

    async def _submit_spread_entry(
        self,
        contract_info: Dict[str, Any],
        quantity: int,
        signal: Signal,
    ):
        """Submit a debit spread as an IBKR combo/BAG order."""
        right = contract_info["right"]
        long_strike = contract_info["strike"]
        short_strike = contract_info["short_strike"]
        net_debit = contract_info["entry_price"]

        # Qualify both legs
        long_contract = self.ibkr.make_option_contract(
            signal.ticker, contract_info["expiry"], long_strike, right
        )
        short_contract = self.ibkr.make_option_contract(
            signal.ticker, contract_info["expiry"], short_strike, right
        )

        long_contract = await self.ibkr.qualify_contract(long_contract)
        short_contract = await self.ibkr.qualify_contract(short_contract)

        legs = [
            {"contract": long_contract, "action": "BUY", "ratio": 1},
            {"contract": short_contract, "action": "SELL", "ratio": 1},
        ]

        limit_price = round(net_debit + self.midpoint_offset, 2)

        logger.info(
            f"Submitting spread: {signal.ticker} "
            f"{long_strike}/{short_strike}{right} "
            f"{quantity}x @ ${limit_price:.2f} net debit"
        )

        trade = await self.ibkr.place_combo_order(
            legs=legs,
            action="BUY",
            quantity=quantity,
            limit_price=limit_price,
        )
        return trade

    async def reprice(self, trade, max_attempts: int = None) -> bool:
        """
        Attempt to reprice an unfilled order by making it more aggressive.
        Returns True if eventually filled, False if all attempts exhausted.
        """
        if max_attempts is None:
            max_attempts = self.max_reprices

        for attempt in range(max_attempts):
            # Cancel existing order
            await self.ibkr.cancel_order(trade)
            await asyncio.sleep(1)

            # Adjust price more aggressively
            new_price = round(
                trade.order.lmtPrice + self.reprice_step, 2
            )

            logger.info(
                f"Repricing (attempt {attempt + 1}/{max_attempts}): "
                f"${trade.order.lmtPrice:.2f} → ${new_price:.2f}"
            )

            trade.order.lmtPrice = new_price
            trade = self.ibkr.ib.placeOrder(trade.contract, trade.order)

            filled = await self.ibkr.wait_for_fill(trade, timeout_sec=self.fill_timeout)
            if filled:
                return True

        return False

    async def place_bracket(self, position: Position) -> None:
        """
        Place stop-loss and take-profit orders for an open position.
        These are protective orders that stay active until hit or cancelled.
        """
        # Stop and TP prices are managed by the RiskManager
        # The bracket is placed here once the risk levels are known
        # For now, the risk_manager will handle exits by monitoring price
        # and calling close_position when needed.
        #
        # In a more advanced version, we'd place actual bracket orders with IBKR.
        # For V1, we monitor and close manually for more control.
        logger.info(
            f"Position open: {position.ticker} {position.direction.value} "
            f"entry=${position.entry_price:.2f} x{position.num_contracts}"
        )

    async def close_position(
        self,
        position: Position,
        reason: ExitReason,
    ) -> None:
        """Close an open position."""
        if position.contract_type == ContractType.DEBIT_SPREAD:
            await self._close_spread(position, reason)
        else:
            await self._close_single(position, reason)

    async def _close_single(
        self, position: Position, reason: ExitReason
    ) -> None:
        """Close a single-leg position."""
        contract = self.ibkr.make_option_contract(
            position.ticker,
            position.expiry,
            position.strike,
            "C" if position.direction == Direction.CALL else "P",
        )
        try:
            contract = await self.ibkr.qualify_contract(contract)
        except Exception as e:
            logger.error(f"Failed to qualify exit contract: {e}")
            # Emergency: try market order
            await self.ibkr.close_all_positions()
            return

        # Use aggressive limit near bid for speed
        # In emergency (EOD, circuit breaker), use market
        if reason in (ExitReason.EOD_CLOSE, ExitReason.CIRCUIT_BREAKER,
                      ExitReason.KILL_SWITCH):
            from ib_async import MarketOrder
            order = MarketOrder("SELL", position.num_contracts)
            self.ibkr.ib.placeOrder(contract, order)
        else:
            # Limit order near bid
            limit_price = max(position.current_price - self.midpoint_offset, 0.01)
            await self.ibkr.place_limit_order(
                contract=contract,
                action="SELL",
                quantity=position.num_contracts,
                limit_price=round(limit_price, 2),
            )

        logger.info(
            f"Closed {position.ticker} {position.direction.value} "
            f"({reason.value}) x{position.num_contracts}"
        )

    async def _close_spread(
        self, position: Position, reason: ExitReason
    ) -> None:
        """Close a debit spread position."""
        # For spreads, we close by selling the spread (reversing the combo)
        # Emergency: close both legs individually
        if reason in (ExitReason.EOD_CLOSE, ExitReason.CIRCUIT_BREAKER,
                      ExitReason.KILL_SWITCH):
            await self.ibkr.close_all_positions()
        else:
            # Try to close as a combo
            right = "C" if position.direction == Direction.CALL else "P"
            long_contract = self.ibkr.make_option_contract(
                position.ticker, position.expiry, position.strike, right
            )
            short_contract = self.ibkr.make_option_contract(
                position.ticker, position.expiry, position.spread_width, right
            )
            try:
                long_contract = await self.ibkr.qualify_contract(long_contract)
                short_contract = await self.ibkr.qualify_contract(short_contract)

                legs = [
                    {"contract": long_contract, "action": "SELL", "ratio": 1},
                    {"contract": short_contract, "action": "BUY", "ratio": 1},
                ]

                exit_price = max(position.current_price - self.midpoint_offset, 0.01)
                await self.ibkr.place_combo_order(
                    legs=legs,
                    action="SELL",
                    quantity=position.num_contracts,
                    limit_price=round(exit_price, 2),
                )
            except Exception as e:
                logger.error(f"Spread close failed: {e}. Using emergency close.")
                await self.ibkr.close_all_positions()

        logger.info(
            f"Closed spread {position.ticker} {position.direction.value} "
            f"({reason.value}) x{position.num_contracts}"
        )
