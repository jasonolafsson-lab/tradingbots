"""
Tests for the Regime Engine classification logic.
"""

import pytest
from datetime import datetime, timedelta

from data.market_state import (
    TickerState, MarketState, Regime, Bias, ScannerResult, Bar
)
from strategy.regime_engine import RegimeEngine


def make_config(**overrides):
    """Default config suitable for regime engine tests."""
    config = {
        "risk": {"max_trades_per_day": 6},
        "strategies": {
            "momentum": {
                "enabled": True,
                "adx_threshold": 25,
                "volume_ratio_threshold": 1.3,
            },
            "reversion": {
                "enabled": True,
                "vwap_sd_threshold": 2.0,
                "adx_max": 35,
            },
            "day2": {
                "enabled": True,
                "score_threshold": 70,
                "spy_selloff_limit": -0.005,
            },
            "tuesday_reversal": {
                "enabled": True,
                "monday_red_threshold": -0.005,
            },
            "green_sector": {
                "enabled": True,
                "spy_red_threshold": -0.003,
            },
        },
        "tickers_config": {
            "sector_mapping": {
                "technology": {"etf": "XLK", "tickers": ["NVDA", "QQQ"]},
            },
        },
    }
    config.update(overrides)
    return config


class TestRegimeEngine:
    def setup_method(self):
        self.engine = RegimeEngine(make_config())

    def test_default_is_no_trade(self):
        """With minimal state, regime should be NO_TRADE."""
        ts = TickerState(ticker="SPY")
        ms = MarketState()
        result = self.engine.classify(ts, ms)
        assert result == Regime.NO_TRADE

    def test_circuit_breaker_forces_no_trade(self):
        ts = TickerState(ticker="SPY", adx_14=30, opening_range_set=True,
                         opening_range_high=100, last_price=105,
                         volume_ratio=2.0, vwap=101)
        ms = MarketState(circuit_breaker_triggered=True)
        result = self.engine.classify(ts, ms)
        assert result == Regime.NO_TRADE

    def test_cooldown_forces_no_trade(self):
        ts = TickerState(ticker="SPY")
        ms = MarketState(cooldown_until=datetime.now() + timedelta(minutes=10))
        result = self.engine.classify(ts, ms)
        assert result == Regime.NO_TRADE

    def test_max_trades_forces_no_trade(self):
        ts = TickerState(ticker="SPY")
        ms = MarketState(trades_today=6)
        result = self.engine.classify(ts, ms)
        assert result == Regime.NO_TRADE

    def test_day2_regime(self):
        """Day 2 candidate with good score → DAY2_CONTINUATION."""
        ts = TickerState(
            ticker="SPY",
            scanner_result=ScannerResult(ticker="SPY", day2_score=85),
        )
        ms = MarketState(spy_session_return=0.001)  # SPY not selling off
        result = self.engine.classify(ts, ms)
        assert result == Regime.DAY2_CONTINUATION

    def test_day2_blocked_by_spy_selloff(self):
        ts = TickerState(
            ticker="SPY",
            scanner_result=ScannerResult(ticker="SPY", day2_score=85),
        )
        ms = MarketState(spy_session_return=-0.02)  # SPY deep red
        result = self.engine.classify(ts, ms)
        assert result != Regime.DAY2_CONTINUATION

    def test_green_sector_regime(self):
        """SPY red + ticker in green sector → GREEN_SECTOR."""
        ts = TickerState(ticker="NVDA")
        ms = MarketState(
            spy_session_return=-0.01,
            green_sectors=["technology"],
        )
        result = self.engine.classify(ts, ms)
        assert result == Regime.GREEN_SECTOR

    def test_momentum_regime(self):
        """OR breakout + ADX + volume + VWAP alignment → MOMENTUM."""
        ts = TickerState(
            ticker="SPY",
            adx_14=30,
            opening_range_set=True,
            opening_range_high=100,
            opening_range_low=95,
            last_price=102,       # Above OR high
            volume_ratio=1.5,
            vwap=101,             # Price above VWAP (aligned)
        )
        ms = MarketState()
        result = self.engine.classify(ts, ms)
        assert result == Regime.MOMENTUM

    def test_momentum_blocked_weak_adx(self):
        ts = TickerState(
            ticker="SPY",
            adx_14=15,  # Weak
            opening_range_set=True,
            opening_range_high=100,
            last_price=102,
            volume_ratio=1.5,
            vwap=101,
        )
        ms = MarketState()
        result = self.engine.classify(ts, ms)
        assert result != Regime.MOMENTUM

    def test_reversion_regime(self):
        """Price extended from VWAP + RSI extreme + low ADX → REVERSION."""
        ts = TickerState(
            ticker="SPY",
            adx_14=20,
            vwap=100,
            vwap_upper_band=102,
            vwap_lower_band=98,
            last_price=96,  # Below lower band (extended)
            rsi_7=22,       # Oversold
        )
        ms = MarketState()
        result = self.engine.classify(ts, ms)
        assert result == Regime.REVERSION

    def test_reversion_blocked_high_adx(self):
        ts = TickerState(
            ticker="SPY",
            adx_14=40,  # Too trendy for reversion
            vwap=100,
            vwap_upper_band=102,
            vwap_lower_band=98,
            last_price=96,
            rsi_7=22,
        )
        ms = MarketState()
        result = self.engine.classify(ts, ms)
        assert result != Regime.REVERSION

    def test_tuesday_bias_sets_flag(self):
        """Tuesday + Monday red → tuesday_bias_active flag set."""
        ts = TickerState(ticker="SPY")
        ms = MarketState(
            day_of_week="Tuesday",
            monday_spy_close_return=-0.01,  # Monday was red
        )
        self.engine.classify(ts, ms)
        assert ts.tuesday_bias_active is True

    def test_tuesday_bias_not_on_wednesday(self):
        ts = TickerState(ticker="SPY")
        ms = MarketState(
            day_of_week="Wednesday",
            monday_spy_close_return=-0.01,
        )
        self.engine.classify(ts, ms)
        assert ts.tuesday_bias_active is False

    def test_day2_priority_over_momentum(self):
        """Day 2 is higher priority than momentum."""
        ts = TickerState(
            ticker="SPY",
            scanner_result=ScannerResult(ticker="SPY", day2_score=85),
            adx_14=30,
            opening_range_set=True,
            opening_range_high=100,
            last_price=102,
            volume_ratio=1.5,
            vwap=101,
        )
        ms = MarketState(spy_session_return=0.001)
        result = self.engine.classify(ts, ms)
        assert result == Regime.DAY2_CONTINUATION  # Not MOMENTUM

    def test_disabled_strategy_skipped(self):
        config = make_config()
        config["strategies"]["momentum"]["enabled"] = False
        engine = RegimeEngine(config)

        ts = TickerState(
            ticker="SPY",
            adx_14=30, opening_range_set=True, opening_range_high=100,
            last_price=102, volume_ratio=1.5, vwap=101,
        )
        ms = MarketState()
        result = engine.classify(ts, ms)
        assert result != Regime.MOMENTUM
