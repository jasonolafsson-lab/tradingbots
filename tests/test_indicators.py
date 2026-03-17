"""
Tests for indicator calculations: VWAP, RSI, ADX, Opening Range, Volume Profile.
"""

import pytest
from datetime import datetime

from data.market_state import Bar, TickerState
from indicators.vwap import VWAPCalculator
from indicators.rsi import RSICalculator
from indicators.adx import ADXCalculator
from indicators.opening_range import OpeningRangeTracker
from indicators.volume_profile import VolumeProfileCalculator


# --- Helper ---

def make_bar(close, high=None, low=None, open_=None, volume=1000, ts=None):
    """Create a test bar with defaults."""
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    if open_ is None:
        open_ = close
    if ts is None:
        ts = datetime.now()
    return Bar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=volume)


# =======================
#  VWAP Tests
# =======================

class TestVWAP:
    def setup_method(self):
        self.calc = VWAPCalculator()

    def test_empty_bars_returns_zeros(self):
        result = self.calc.calculate([])
        assert result["vwap"] == 0.0
        assert result["slope"] == 0.0

    def test_single_bar_returns_zeros(self):
        """Need at least 2 bars."""
        result = self.calc.calculate([make_bar(100)])
        assert result["vwap"] == 0.0

    def test_basic_vwap_calculation(self):
        bars = [
            make_bar(100, high=101, low=99, volume=1000),
            make_bar(102, high=103, low=101, volume=2000),
            make_bar(104, high=105, low=103, volume=1500),
        ]
        result = self.calc.calculate(bars)
        assert result["vwap"] > 0
        # VWAP should be between min and max typical prices
        assert 99 < result["vwap"] < 105

    def test_upper_band_above_vwap(self):
        bars = [make_bar(100 + i, volume=1000) for i in range(10)]
        result = self.calc.calculate(bars)
        assert result["upper_band"] > result["vwap"]
        assert result["lower_band"] < result["vwap"]

    def test_slope_positive_for_rising(self):
        bars = [make_bar(100 + i * 5, high=101 + i * 5, low=99 + i * 5, volume=1000)
                for i in range(10)]
        result = self.calc.calculate(bars)
        assert result["slope"] > 0

    def test_price_vs_vwap_sd(self):
        sd = VWAPCalculator.price_vs_vwap_sd(price=102, vwap=100, std_dev=1.0)
        assert sd == 2.0

        sd_neg = VWAPCalculator.price_vs_vwap_sd(price=98, vwap=100, std_dev=1.0)
        assert sd_neg == -2.0

        sd_zero = VWAPCalculator.price_vs_vwap_sd(price=100, vwap=100, std_dev=0)
        assert sd_zero == 0.0


# =======================
#  RSI Tests
# =======================

class TestRSI:
    def setup_method(self):
        self.calc = RSICalculator(period=7)

    def test_insufficient_data_returns_50(self):
        result = self.calc.calculate([100, 101, 102])
        assert result == 50.0

    def test_all_up_returns_100(self):
        closes = [100 + i for i in range(20)]
        result = self.calc.calculate(closes)
        assert result == 100.0

    def test_all_down_returns_near_zero(self):
        closes = [200 - i for i in range(20)]
        result = self.calc.calculate(closes)
        assert result < 5.0

    def test_rsi_range(self):
        import random
        random.seed(42)
        closes = [100 + random.uniform(-5, 5) for _ in range(50)]
        result = self.calc.calculate(closes)
        assert 0 <= result <= 100

    def test_oversold_region(self):
        closes = [100]
        for i in range(20):
            closes.append(closes[-1] - 1.5 + (0.2 if i % 5 == 0 else 0))
        result = self.calc.calculate(closes)
        assert result < 40


# =======================
#  ADX Tests
# =======================

class TestADX:
    def setup_method(self):
        self.calc = ADXCalculator(period=14)

    def test_insufficient_data(self):
        result = self.calc.calculate([100]*5, [99]*5, [100]*5)
        assert result == 0.0

    def test_strong_trend(self):
        n = 40
        highs = [100 + i * 2 + 1 for i in range(n)]
        lows = [100 + i * 2 - 1 for i in range(n)]
        closes = [100 + i * 2 for i in range(n)]
        result = self.calc.calculate(highs, lows, closes)
        assert result > 20

    def test_range_bound(self):
        n = 40
        highs = [101 if i % 2 == 0 else 100 for i in range(n)]
        lows = [99 if i % 2 == 0 else 100 for i in range(n)]
        closes = [100 for i in range(n)]
        result = self.calc.calculate(highs, lows, closes)
        assert result < 30

    def test_adx_range(self):
        n = 40
        highs = [100 + i for i in range(n)]
        lows = [99 + i for i in range(n)]
        closes = [99.5 + i for i in range(n)]
        result = self.calc.calculate(highs, lows, closes)
        assert 0 <= result <= 100


# =======================
#  Opening Range Tests
# =======================

class TestOpeningRange:
    def setup_method(self):
        self.tracker = OpeningRangeTracker(config={"opening_range": {"duration_minutes": 15}})

    def test_no_data_not_set(self):
        ts = TickerState(ticker="SPY")
        self.tracker.update(ts)
        assert ts.opening_range_set is False

    def test_partial_range_updates_high_low(self):
        ts = TickerState(ticker="SPY")
        for i in range(5):
            ts.bars_1m.append(Bar(
                timestamp=datetime(2026, 3, 12, 9, 30 + i),
                open=100, high=102 + i, low=98 - i, close=100, volume=1000,
            ))
        self.tracker.update(ts)
        assert ts.opening_range_set is False
        assert ts.opening_range_high == 106
        assert ts.opening_range_low == 94

    def test_full_range_sets_flag(self):
        ts = TickerState(ticker="SPY")
        for i in range(16):
            ts.bars_1m.append(Bar(
                timestamp=datetime(2026, 3, 12, 9, 30 + i),
                open=100, high=105, low=95, close=100, volume=1000,
            ))
        self.tracker.update(ts)
        assert ts.opening_range_set is True
        assert ts.opening_range_high == 105
        assert ts.opening_range_low == 95

    def test_no_update_after_set(self):
        ts = TickerState(ticker="SPY")
        for i in range(16):
            ts.bars_1m.append(Bar(
                timestamp=datetime(2026, 3, 12, 9, 30 + i),
                open=100, high=105, low=95, close=100, volume=1000,
            ))
        self.tracker.update(ts)
        ts.bars_1m.append(Bar(
            timestamp=datetime(2026, 3, 12, 9, 50),
            open=100, high=200, low=50, close=100, volume=1000,
        ))
        self.tracker.update(ts)
        assert ts.opening_range_high == 105
        assert ts.opening_range_low == 95


# =======================
#  Volume Profile Tests
# =======================

class TestVolumeProfile:
    def setup_method(self):
        self.calc = VolumeProfileCalculator(lookback=5)

    def test_empty_returns_one(self):
        result = self.calc.calculate([])
        assert result == 1.0

    def test_single_volume(self):
        result = self.calc.calculate([1000])
        assert isinstance(result, float)

    def test_expanding_volume(self):
        volumes = [100, 100, 100, 100, 500]
        result = self.calc.calculate(volumes)
        assert result > 1.0

    def test_contracting_volume(self):
        volumes = [500, 500, 500, 500, 100]
        result = self.calc.calculate(volumes)
        assert result < 1.0

    def test_is_above_average(self):
        assert self.calc.is_above_average([100, 100, 100, 500], threshold=1.0)
        assert not self.calc.is_above_average([500, 500, 500, 100], threshold=1.0)
