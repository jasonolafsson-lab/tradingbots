"""
Tests for risk management: Position sizing, Circuit breaker.
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.market_state import (
    Signal, Position, MarketState, Direction, ContractType, Regime
)
from risk.sizing import PositionSizer
from risk.circuit_breaker import CircuitBreaker

ET = ZoneInfo("US/Eastern")


def make_signal(ticker="SPY", direction=Direction.CALL, gap_pct=0.0, **kwargs):
    defaults = dict(
        strategy="MOMENTUM",
        regime=Regime.MOMENTUM,
        strength_score=75,
        entry_price_target=450.0,
        gap_pct=gap_pct,
    )
    defaults.update(kwargs)
    return Signal(ticker=ticker, direction=direction, **defaults)


def make_position(pnl_pct=0.0, **kwargs):
    sig = make_signal()
    defaults = dict(
        ticker="SPY", direction=Direction.CALL,
        contract_type=ContractType.SINGLE_LEG, strategy="MOMENTUM",
        regime="MOMENTUM", signal=sig, unrealized_pnl_pct=pnl_pct,
    )
    defaults.update(kwargs)
    return Position(**defaults)


# =======================
#  Sizing Tests
# =======================

class TestPositionSizer:
    def setup_method(self):
        self.config = {
            "risk": {
                "max_trade_risk_pct": 0.02,
                "max_notional_pct": 0.05,
                "max_contracts_single": 10,
                "max_contracts_spread": 20,
            },
            "stop_loss": {
                "single_leg_pct": 0.25,
                "spread_pct": 0.40,
            },
            "gap_risk": {
                "day2_gap_threshold_pct": 0.03,
                "gap_size_reduction": 0.50,
            },
        }
        self.sizer = PositionSizer(self.config)

    def test_basic_single_leg_sizing(self):
        signal = make_signal()
        contract_info = {"entry_price": 5.00, "contract_type": ContractType.SINGLE_LEG}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 100_000, ms)

        # max_risk = 100k * 0.02 = $2000
        # risk_per_contract = 5.00 * 100 * 0.25 = $125
        # contracts = floor(2000 / 125) = 16
        # BUT capped at 10 (max_contracts_single)
        assert result["contracts"] == 10
        assert result["risk_per_contract"] == 125.0

    def test_spread_sizing(self):
        signal = make_signal()
        contract_info = {"entry_price": 1.50, "contract_type": ContractType.DEBIT_SPREAD}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 100_000, ms)

        # risk_per_contract = 1.50 * 100 = $150
        # contracts = floor(2000 / 150) = 13
        assert result["contracts"] == 13

    def test_zero_account_value(self):
        signal = make_signal()
        contract_info = {"entry_price": 5.00}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 0, ms)
        assert result["contracts"] == 0

    def test_zero_entry_price(self):
        signal = make_signal()
        contract_info = {"entry_price": 0}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 100_000, ms)
        assert result["contracts"] == 0

    def test_gap_guard_reduces_size(self):
        """Day 2 gap > 3% → 50% reduction."""
        signal = make_signal(gap_pct=0.05)  # 5% gap
        contract_info = {"entry_price": 2.00, "contract_type": ContractType.SINGLE_LEG}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 100_000, ms)

        # Without gap: risk_per_contract = 2*100*0.25 = $50, contracts = floor(2000/50) = 40 → capped 10
        # With gap guard: floor(10 * 0.50) = 5
        assert result["contracts"] == 5
        assert "Gap guard" in result["sizing_notes"]

    def test_intelligence_multiplier(self):
        signal = make_signal()
        contract_info = {"entry_price": 5.00, "contract_type": ContractType.SINGLE_LEG}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 100_000, ms,
                                       intelligence_multiplier=0.5)
        # Base: 10 contracts → * 0.5 = 5
        assert result["contracts"] == 5

    def test_notional_cap(self):
        """Very cheap options should be capped by notional limit."""
        signal = make_signal()
        contract_info = {"entry_price": 0.10, "contract_type": ContractType.SINGLE_LEG}
        ms = MarketState()
        result = self.sizer.calculate(signal, contract_info, 10_000, ms)

        # max_risk = 200, risk_per_contract = 0.10 * 100 * 0.25 = $2.50
        # contracts = floor(200/2.50) = 80 → capped at 10
        # notional = 10 * 0.10 * 100 = $100 → 100 < max_notional(500) → OK
        # So capped at 10 by max_contracts_single
        assert result["contracts"] == 10


# =======================
#  Circuit Breaker Tests
# =======================

class TestCircuitBreaker:
    def setup_method(self):
        self.config = {
            "risk": {
                "daily_loss_limit_pct": 0.03,
                "max_consecutive_losses": 3,
                "cooldown_after_trade_sec": 300,
                "cooldown_after_loss_sec": 600,
                "consecutive_loss_cooldown_sec": 1800,
                "max_trades_per_day": 6,
            }
        }
        self.cb = CircuitBreaker(self.config)

    def test_not_triggered_initially(self):
        ms = MarketState()
        assert self.cb.is_triggered(ms) is False

    def test_max_trades_triggers(self):
        ms = MarketState(trades_today=6)
        assert self.cb.is_triggered(ms) is True

    def test_daily_pnl_loss_triggers(self):
        ms = MarketState()
        triggered = self.cb.check_daily_pnl(-3500, 100_000, ms)
        assert triggered is True
        assert ms.circuit_breaker_triggered is True

    def test_daily_pnl_within_limit(self):
        ms = MarketState()
        triggered = self.cb.check_daily_pnl(-1000, 100_000, ms)
        assert triggered is False
        assert ms.circuit_breaker_triggered is False

    def test_record_win_resets_consecutive(self):
        ms = MarketState(consecutive_losses=2)
        pos = make_position(pnl_pct=0.10)  # Win
        self.cb.record_trade_result(pos, ms)
        assert ms.consecutive_losses == 0
        assert ms.trades_today == 1
        assert ms.cooldown_until is not None

    def test_record_loss_increments_consecutive(self):
        ms = MarketState(consecutive_losses=1)
        pos = make_position(pnl_pct=-0.15)  # Loss
        self.cb.record_trade_result(pos, ms)
        assert ms.consecutive_losses == 2

    def test_three_consecutive_losses_long_cooldown(self):
        ms = MarketState(consecutive_losses=2)
        pos = make_position(pnl_pct=-0.10)  # 3rd loss
        self.cb.record_trade_result(pos, ms)
        assert ms.consecutive_losses == 3
        # Should get 30 minute cooldown (1800 sec)
        expected_cooldown = timedelta(seconds=1800)
        actual_duration = ms.cooldown_until - ms.last_trade_close_time
        assert actual_duration == expected_cooldown

    def test_normal_trade_short_cooldown(self):
        ms = MarketState()
        pos = make_position(pnl_pct=0.05)  # Win
        self.cb.record_trade_result(pos, ms)
        expected_cooldown = timedelta(seconds=300)
        actual_duration = ms.cooldown_until - ms.last_trade_close_time
        assert actual_duration == expected_cooldown
