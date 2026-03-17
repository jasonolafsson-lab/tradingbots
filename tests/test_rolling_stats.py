"""
Tests for Level 1 Intelligence: Rolling Statistics.

Tests the static _compute_from_trades method directly, and
integration tests using TradeMemoryDB + RollingStats.refresh().
"""

import os
import pytest
import tempfile
from datetime import datetime

from data.market_state import (
    Signal, Position, Direction, ContractType, Regime, ExitReason
)
from intelligence.trade_memory import TradeMemoryDB
from intelligence.rolling_stats import RollingStats


def make_position(ticker="SPY", strategy="MOMENTUM", pnl_pct=0.10):
    signal = Signal(
        ticker=ticker, direction=Direction.CALL, strategy=strategy,
        regime=Regime.MOMENTUM, strength_score=75, entry_price_target=450.0,
    )
    return Position(
        ticker=ticker, direction=Direction.CALL,
        contract_type=ContractType.SINGLE_LEG, strategy=strategy,
        regime="MOMENTUM", signal=signal,
        entry_time=datetime(2026, 3, 12, 10, 15),
        entry_price=5.00, num_contracts=3,
        unrealized_pnl_pct=pnl_pct,
        max_favorable=max(pnl_pct, 0),
        max_adverse=min(pnl_pct, 0),
        signal_strength_score=75.0,
    )


@pytest.fixture
def db_and_stats():
    tmpdir = tempfile.mkdtemp()
    config = {
        "logging": {
            "db_path": os.path.join(tmpdir, "test_trades.db"),
            "db_backup_dir": os.path.join(tmpdir, "backups"),
        }
    }
    trade_db = TradeMemoryDB(config)
    trade_db.initialize()
    stats = RollingStats(trade_db, window=50)
    yield trade_db, stats
    trade_db.close()


class TestComputeFromTradesStatic:
    """Test _compute_from_trades static method directly (no DB needed)."""

    def test_empty_trades(self):
        result = RollingStats._compute_from_trades([])
        assert result["count"] == 0
        assert result["win_rate"] == 0

    def test_all_wins(self):
        trades = [
            {"outcome": "WIN", "pnl_percent": 0.10, "pnl_dollars": 100},
            {"outcome": "WIN", "pnl_percent": 0.05, "pnl_dollars": 50},
        ]
        result = RollingStats._compute_from_trades(trades)
        assert result["win_rate"] == 1.0
        assert result["count"] == 2

    def test_mixed_results(self):
        trades = [
            {"outcome": "WIN", "pnl_percent": 0.10, "pnl_dollars": 100},
            {"outcome": "LOSS", "pnl_percent": -0.10, "pnl_dollars": -100},
        ]
        result = RollingStats._compute_from_trades(trades)
        assert result["win_rate"] == 0.5

    def test_profit_factor(self):
        trades = [
            {"outcome": "WIN", "pnl_percent": 0.10, "pnl_dollars": 100},
            {"outcome": "WIN", "pnl_percent": 0.20, "pnl_dollars": 200},
            {"outcome": "LOSS", "pnl_percent": -0.10, "pnl_dollars": -100},
        ]
        result = RollingStats._compute_from_trades(trades)
        # PF = 300 / 100 = 3.0
        assert result["profit_factor"] == pytest.approx(3.0)

    def test_max_drawdown(self):
        trades = [
            {"outcome": "WIN", "pnl_percent": 0.10, "pnl_dollars": 100},
            {"outcome": "LOSS", "pnl_percent": -0.20, "pnl_dollars": -200},
            {"outcome": "LOSS", "pnl_percent": -0.10, "pnl_dollars": -100},
            {"outcome": "WIN", "pnl_percent": 0.15, "pnl_dollars": 150},
        ]
        result = RollingStats._compute_from_trades(trades)
        # Cumulative: 100, -100, -200, -50
        # Peak:       100,  100,  100,  100
        # Drawdown:    0,  -200, -300, -150
        # Max drawdown = -300
        assert result["max_drawdown"] == pytest.approx(-300)


class TestRollingStatsIntegration:
    """Integration tests using DB + RollingStats."""

    def test_refresh_populates_cache(self, db_and_stats):
        db, stats = db_and_stats
        db.record_trade(make_position(pnl_pct=0.10), ExitReason.TAKE_PROFIT)
        db.record_trade(make_position(pnl_pct=-0.10), ExitReason.STOP_LOSS)
        stats.refresh()

        overall = stats.get_stats("overall")
        assert overall["count"] == 2
        assert overall["win_rate"] == 0.5

    def test_strategy_stats(self, db_and_stats):
        db, stats = db_and_stats
        db.record_trade(make_position(strategy="MOMENTUM", pnl_pct=0.10), ExitReason.TAKE_PROFIT)
        db.record_trade(make_position(strategy="REVERSION", pnl_pct=-0.10), ExitReason.STOP_LOSS)
        stats.refresh()

        assert stats.get_win_rate(strategy="MOMENTUM") == 1.0
        assert stats.get_win_rate(strategy="REVERSION") == 0.0

    def test_empty_db(self, db_and_stats):
        db, stats = db_and_stats
        stats.refresh()
        assert stats.get_win_rate() == 0
        assert stats.get_profit_factor() == 0
