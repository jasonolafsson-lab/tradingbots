"""
Position Sizing Calculator.

Determines how many contracts to trade based on:
- Account value and max risk per trade (2%)
- Option premium or spread debit
- Stop loss percentage
- Gap guard (Day 2 with >3% gap → 50% reduction)
- Intelligence-level adjustments (Level 1+)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Any, Optional

from data.market_state import (
    Signal, MarketState, ContractType
)

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates position size in contracts."""

    def __init__(self, config: dict):
        risk = config.get("risk", {})
        self.max_risk_pct = risk.get("max_trade_risk_pct", 0.02)
        self.max_notional_pct = risk.get("max_notional_pct", 0.05)
        self.max_contracts_single = risk.get("max_contracts_single", 10)
        self.max_contracts_spread = risk.get("max_contracts_spread", 20)

        sl = config.get("stop_loss", {})
        self.sl_single = sl.get("single_leg_pct", 0.25)
        self.sl_spread = sl.get("spread_pct", 0.40)

        gap = config.get("gap_risk", {})
        self.gap_threshold = gap.get("day2_gap_threshold_pct", 0.03)
        self.gap_reduction = gap.get("gap_size_reduction", 0.50)

    def calculate(
        self,
        signal: Signal,
        contract_info: Dict[str, Any],
        account_value: float,
        market_state: MarketState,
        intelligence_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate position size.

        Returns dict with:
            contracts: int — number of contracts to trade
            risk_per_contract: float — dollar risk per contract
            total_risk: float — total dollar risk
            entry_price: float — expected entry price
            sizing_notes: str — explanation of sizing decisions
        """
        if account_value <= 0:
            return {"contracts": 0, "risk_per_contract": 0, "total_risk": 0,
                    "entry_price": 0, "sizing_notes": "No account value"}

        max_risk_dollars = account_value * self.max_risk_pct
        max_notional = account_value * self.max_notional_pct
        contract_type = contract_info.get("contract_type", ContractType.SINGLE_LEG)
        entry_price = contract_info.get("entry_price", contract_info.get("mid", 0))
        notes = []

        if entry_price <= 0:
            return {"contracts": 0, "risk_per_contract": 0, "total_risk": 0,
                    "entry_price": 0, "sizing_notes": "Invalid entry price"}

        if contract_type == ContractType.DEBIT_SPREAD:
            # Spread: max risk = debit * 100 per contract
            risk_per_contract = entry_price * 100
            max_contracts = self.max_contracts_spread
            notes.append("Debit spread sizing")
        else:
            # Single-leg: risk = premium * 100 * stop_loss_pct per contract
            risk_per_contract = entry_price * 100 * self.sl_single
            max_contracts = self.max_contracts_single
            notes.append("Single-leg sizing")

        # Base number of contracts
        if risk_per_contract <= 0:
            return {"contracts": 0, "risk_per_contract": 0, "total_risk": 0,
                    "entry_price": entry_price, "sizing_notes": "Zero risk per contract"}

        contracts = math.floor(max_risk_dollars / risk_per_contract)

        # Notional limit check
        notional = contracts * entry_price * 100
        if notional > max_notional:
            contracts = math.floor(max_notional / (entry_price * 100))
            notes.append(f"Notional capped at {self.max_notional_pct:.0%}")

        # Max contracts cap
        contracts = min(contracts, max_contracts)

        # Gap guard: Day 2 candidates with large gap
        if signal.is_gap_risk:
            contracts = max(1, math.floor(contracts * self.gap_reduction))
            notes.append(f"Gap guard: reduced to {self.gap_reduction:.0%} "
                        f"(gap={signal.gap_pct:.1%})")

        # Intelligence multiplier (Level 1+ adjustments)
        if intelligence_multiplier != 1.0:
            contracts = max(1, math.floor(contracts * intelligence_multiplier))
            notes.append(f"Intelligence multiplier: {intelligence_multiplier:.2f}x")

        # Minimum 1 contract
        contracts = max(contracts, 1) if contracts > 0 else 0

        total_risk = contracts * risk_per_contract

        result = {
            "contracts": contracts,
            "risk_per_contract": risk_per_contract,
            "total_risk": total_risk,
            "entry_price": entry_price,
            "notional": contracts * entry_price * 100,
            "risk_pct_of_account": total_risk / account_value if account_value > 0 else 0,
            "sizing_notes": "; ".join(notes),
        }

        logger.info(
            f"Position size: {contracts} contracts "
            f"(risk=${total_risk:.0f} = {result['risk_pct_of_account']:.1%} of account) "
            f"[{result['sizing_notes']}]"
        )

        return result
