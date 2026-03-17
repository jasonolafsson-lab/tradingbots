"""
Trade Logger.

Writes structured JSON-lines event logs and trade blotter.
One log file per trading day. Captures every decision the bot makes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from data.market_state import (
    Signal, Position, ExitReason, ScannerResult, MarketState
)

logger = logging.getLogger(__name__)


class TradeLogger:
    """Structured event logging for full session reconstruction."""

    def __init__(self, config: dict):
        self.log_dir = config.get("logging", {}).get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(self.log_dir, f"{self.today}.jsonl")

    def _write_event(self, event: dict) -> None:
        """Write a single event to the JSON-lines log."""
        event["timestamp"] = datetime.now().isoformat()
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write event log: {e}")

    def log_system(self, message: str, **kwargs) -> None:
        """Log a system event (startup, shutdown, connectivity)."""
        self._write_event({
            "category": "SYSTEM",
            "message": message,
            **kwargs,
        })

    def log_scanner_results(self, results: List[ScannerResult]) -> None:
        """Log pre-market scanner output."""
        for r in results:
            self._write_event({
                "category": "SCANNER",
                "ticker": r.ticker,
                "bias": r.bias.value,
                "day2_score": r.day2_score,
                "sector_rs": r.sector_rs,
                "priority_rank": r.priority_rank,
                "close_quality": r.close_quality,
                "volume_vs_avg": r.volume_vs_avg,
                "catalyst_score": r.catalyst_score,
                "key_levels": r.key_levels,
            })

    def log_regime_change(
        self, ticker: str, old_regime: str, new_regime: str
    ) -> None:
        """Log a regime classification change."""
        self._write_event({
            "category": "REGIME",
            "ticker": ticker,
            "old_regime": old_regime,
            "new_regime": new_regime,
        })

    def log_signal(
        self,
        signal: Signal,
        executed: bool = True,
        reason: str = "",
    ) -> None:
        """Log a signal generation event."""
        self._write_event({
            "category": "SIGNAL",
            "ticker": signal.ticker,
            "direction": signal.direction.value,
            "strategy": signal.strategy,
            "strength_score": signal.strength_score,
            "entry_price_target": signal.entry_price_target,
            "executed": executed,
            "reject_reason": reason,
            "tuesday_bias": signal.tuesday_bias,
            "day2_score": signal.day2_score,
        })

    def log_entry(self, position: Position) -> None:
        """Log a position entry."""
        self._write_event({
            "category": "ORDER",
            "event": "ENTRY",
            "ticker": position.ticker,
            "direction": position.direction.value,
            "strategy": position.strategy,
            "contract_type": position.contract_type.value,
            "strike": position.strike,
            "expiry": position.expiry,
            "dte": position.dte,
            "delta": position.delta_at_entry,
            "entry_price": position.entry_price,
            "num_contracts": position.num_contracts,
            "spread_width": position.spread_width,
        })

    def log_exit(self, position: Position, reason: ExitReason) -> None:
        """Log a position exit."""
        self._write_event({
            "category": "ORDER",
            "event": "EXIT",
            "ticker": position.ticker,
            "direction": position.direction.value,
            "exit_reason": reason.value,
            "entry_price": position.entry_price,
            "exit_price": position.current_price,
            "pnl_pct": position.unrealized_pnl_pct,
            "max_favorable": position.max_favorable,
            "max_adverse": position.max_adverse,
            "hold_duration_sec": (
                (datetime.now() - position.entry_time).total_seconds()
                if position.entry_time else 0
            ),
        })

    def log_risk_event(self, event_type: str, **kwargs) -> None:
        """Log a risk management event."""
        self._write_event({
            "category": "RISK",
            "event": event_type,
            **kwargs,
        })

    def write_daily_summary(self, market_state: MarketState) -> None:
        """Write end-of-day summary."""
        summary = {
            "category": "DAILY_SUMMARY",
            "date": self.today,
            "trades_today": market_state.trades_today,
            "daily_pnl": market_state.daily_pnl,
            "consecutive_losses": market_state.consecutive_losses,
            "circuit_breaker_triggered": market_state.circuit_breaker_triggered,
            "spy_return": market_state.spy_session_return,
        }
        self._write_event(summary)
        logger.info(f"Daily summary written to {self.log_file}")
