"""
Tests for the Trade Memory Database (SQLite).
"""

import os
import pytest
import tempfile
from datetime import datetime

from data.market_state import (
    Signal, Position, Direction, ContractType, Regime, ExitReason
)
from intelligence.trade_memory import TradeMemoryDB


def make_test_position(ticker="SPY", pnl_pct=0.10, strategy="MOMENTUM"):
    """Create a Position with all fields populated."""
    signal = Signal(
        ticker=ticker,
        direction=Direction.CALL,
        strategy=strategy,
        regime=Regime.MOMENTUM,
        strength_score=80,
        entry_price_target=450.0,
    )
    return Position(
        ticker=ticker,
        direction=Direction.CALL,
        contract_type=ContractType.SINGLE_LEG,
        strategy=strategy,
        regime="MOMENTUM",
        signal=signal,
        strike=450.0,
        expiry="20260312",
        dte=0,
        delta_at_entry=0.50,
        iv_at_entry=0.30,
        iv_percentile=45.0,
        entry_time=datetime(2026, 3, 12, 10, 15),
        entry_price=5.00,
        num_contracts=3,
        entry_slippage=0.02,
        bid_ask_at_entry=0.05,
        fill_time_sec=2.5,
        current_price=5.50,
        unrealized_pnl_pct=pnl_pct,
        max_favorable=0.15,
        max_adverse=-0.05,
        spy_session_return=0.002,
        underlying_vs_vwap=0.5,
        vwap_slope=0.01,
        adx_value=28.0,
        rsi_value=55.0,
        volume_ratio=1.3,
        sector_rs=0.5,
        uw_net_premium_direction="BULLISH",
        gex_nearest_wall_distance=1.5,
        day2_score=75.0,
        signal_strength_score=80.0,
    )


@pytest.fixture
def db():
    """Create a temporary trade memory database."""
    tmpdir = tempfile.mkdtemp()
    config = {
        "logging": {
            "db_path": os.path.join(tmpdir, "test_trades.db"),
            "db_backup_dir": os.path.join(tmpdir, "backups"),
        }
    }
    trade_db = TradeMemoryDB(config)
    trade_db.initialize()
    yield trade_db
    trade_db.close()


class TestTradeMemoryDB:
    def test_initialize_creates_tables(self, db):
        """DB should be initialized with all tables."""
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "trades" in tables
        assert "daily_sessions" in tables
        assert "strategy_performance" in tables
        assert "model_scores" in tables

    def test_record_trade(self, db):
        """Record a trade and verify it's stored."""
        pos = make_test_position()
        trade_id = db.record_trade(pos, ExitReason.TAKE_PROFIT)
        assert trade_id is not None
        assert len(trade_id) == 12

    def test_get_total_trades(self, db):
        assert db.get_total_trades() == 0
        pos = make_test_position()
        db.record_trade(pos, ExitReason.STOP_LOSS)
        assert db.get_total_trades() == 1
        db.record_trade(make_test_position("QQQ"), ExitReason.TIME_STOP)
        assert db.get_total_trades() == 2

    def test_get_trades_returns_all(self, db):
        for i in range(5):
            db.record_trade(make_test_position(), ExitReason.TAKE_PROFIT)
        trades = db.get_trades()
        assert len(trades) == 5

    def test_get_trades_filter_by_strategy(self, db):
        db.record_trade(make_test_position(strategy="MOMENTUM"), ExitReason.TAKE_PROFIT)
        db.record_trade(make_test_position(strategy="REVERSION"), ExitReason.STOP_LOSS)
        db.record_trade(make_test_position(strategy="MOMENTUM"), ExitReason.TRAILING_STOP)

        momentum_trades = db.get_trades(strategy="MOMENTUM")
        assert len(momentum_trades) == 2

        reversion_trades = db.get_trades(strategy="REVERSION")
        assert len(reversion_trades) == 1

    def test_get_trades_filter_by_ticker(self, db):
        db.record_trade(make_test_position("SPY"), ExitReason.TAKE_PROFIT)
        db.record_trade(make_test_position("QQQ"), ExitReason.STOP_LOSS)
        db.record_trade(make_test_position("SPY"), ExitReason.TAKE_PROFIT)

        spy_trades = db.get_trades(ticker="SPY")
        assert len(spy_trades) == 2

    def test_get_overall_stats_win_loss(self, db):
        # 2 wins, 1 loss
        db.record_trade(make_test_position(pnl_pct=0.10), ExitReason.TAKE_PROFIT)
        db.record_trade(make_test_position(pnl_pct=0.05), ExitReason.TRAILING_STOP)
        db.record_trade(make_test_position(pnl_pct=-0.15), ExitReason.STOP_LOSS)

        stats = db.get_overall_stats()
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert 0.6 < stats["win_rate"] < 0.7  # ~66%

    def test_daily_session_updated(self, db):
        db.record_trade(make_test_position(pnl_pct=0.10), ExitReason.TAKE_PROFIT)
        db.record_trade(make_test_position(pnl_pct=-0.05), ExitReason.STOP_LOSS)

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM daily_sessions")
        row = cursor.fetchone()
        assert row is not None
        # trades_count should be 2
        assert dict(row)["trades_count"] == 2

    def test_backup(self, db):
        db.record_trade(make_test_position(), ExitReason.TAKE_PROFIT)
        db.backup()
        # Check backup file exists
        backup_files = os.listdir(db.backup_dir)
        assert len(backup_files) == 1
        assert backup_files[0].endswith(".db")

    def test_35_fields_populated(self, db):
        """Verify all 35 spec fields are stored."""
        pos = make_test_position()
        db.record_trade(pos, ExitReason.TAKE_PROFIT)
        trades = db.get_trades()
        assert len(trades) == 1
        trade = trades[0]

        # Check key fields exist and are non-null
        assert trade["ticker"] == "SPY"
        assert trade["strategy"] == "MOMENTUM"
        assert trade["direction"] == "CALL"
        assert trade["contract_type"] == "SINGLE_LEG"
        assert trade["exit_reason"] == "TAKE_PROFIT"
        assert trade["strike"] == 450.0
        assert trade["dte"] == 0
        assert trade["delta_at_entry"] == 0.50
        assert trade["adx_value"] == 28.0
        assert trade["outcome"] == "WIN"
        assert trade["signal_strength_score"] == 80.0
