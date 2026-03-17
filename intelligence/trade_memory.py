"""
Trade Memory Database.

SQLite database with 35-field trade records, logged from V1 day one.
This is the foundation for statistical analysis (Level 1),
ML training (Level 2), and LLM review (Level 3).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from data.market_state import Position, ExitReason

logger = logging.getLogger(__name__)


class TradeMemoryDB:
    """SQLite trade memory database with 35-field schema."""

    def __init__(self, config: dict):
        log_config = config.get("logging", {})
        self.db_path = log_config.get("db_path", "data/trades.db")
        self.backup_dir = log_config.get("db_backup_dir", "data/backups")
        self.conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        self._create_tables()
        logger.info(f"Trade memory DB initialized: {self.db_path}")

    def _create_tables(self) -> None:
        """Create all required tables."""
        cursor = self.conn.cursor()

        # Main trades table — 35 fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                -- Trade Identification (7 fields)
                trade_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                strategy TEXT NOT NULL,
                regime TEXT NOT NULL,
                direction TEXT NOT NULL,
                contract_type TEXT NOT NULL,

                -- Timing (6 fields)
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                hold_duration_sec INTEGER,
                day_of_week TEXT NOT NULL,
                minutes_since_open INTEGER,
                exit_reason TEXT,

                -- Contract Details (6 fields)
                strike REAL,
                dte INTEGER,
                delta_at_entry REAL,
                iv_at_entry REAL,
                iv_percentile REAL,
                spread_width REAL,

                -- Market Context at Entry (10 fields)
                spy_session_return REAL,
                underlying_vs_vwap REAL,
                vwap_slope REAL,
                adx_value REAL,
                rsi_value REAL,
                volume_ratio REAL,
                sector_relative_strength REAL,
                uw_net_premium_direction TEXT,
                gex_nearest_wall_distance REAL,
                day2_score REAL,

                -- Execution Quality (3 fields)
                entry_slippage REAL,
                bid_ask_at_entry REAL,
                fill_time_sec REAL,

                -- Outcome (5 fields)
                pnl_dollars REAL,
                pnl_percent REAL,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                signal_strength_score REAL,
                outcome TEXT,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                ml_confidence_score REAL
            )
        """)

        # Daily sessions summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_sessions (
                date TEXT PRIMARY KEY,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                spy_return REAL DEFAULT 0,
                notes TEXT
            )
        """)

        # Strategy performance (rolling stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                ticker TEXT,
                win_rate REAL,
                avg_pnl_pct REAL,
                profit_factor REAL,
                trade_count INTEGER,
                sample_window INTEGER,
                UNIQUE(date, strategy, ticker)
            )
        """)

        # ML model scores (Level 2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT REFERENCES trades(trade_id),
                model_version TEXT,
                predicted_win_prob REAL,
                actual_outcome TEXT,
                feature_importances TEXT,
                scored_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def record_trade(
        self,
        position: Position,
        exit_reason: ExitReason,
    ) -> str:
        """
        Record a completed trade with all 35 fields.
        Returns the trade_id.
        """
        trade_id = str(uuid.uuid4())[:12]

        now = datetime.now()
        entry_time = position.entry_time or now
        hold_duration = int((now - entry_time).total_seconds()) if position.entry_time else 0
        minutes_since_open = int(
            (entry_time.hour * 60 + entry_time.minute) - (9 * 60 + 30)
        ) if entry_time else 0

        # Compute P&L
        pnl_pct = position.unrealized_pnl_pct
        pnl_dollars = pnl_pct * position.entry_price * 100 * position.num_contracts
        outcome = "WIN" if pnl_pct >= 0 else "LOSS"

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                trade_id, date, ticker, strategy, regime, direction, contract_type,
                entry_time, exit_time, hold_duration_sec, day_of_week, minutes_since_open, exit_reason,
                strike, dte, delta_at_entry, iv_at_entry, iv_percentile, spread_width,
                spy_session_return, underlying_vs_vwap, vwap_slope, adx_value, rsi_value,
                volume_ratio, sector_relative_strength, uw_net_premium_direction,
                gex_nearest_wall_distance, day2_score,
                entry_slippage, bid_ask_at_entry, fill_time_sec,
                pnl_dollars, pnl_percent, max_favorable_excursion, max_adverse_excursion,
                signal_strength_score, outcome
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        """, (
            trade_id,
            now.strftime("%Y-%m-%d"),
            position.ticker,
            position.strategy,
            position.regime,
            position.direction.value,
            position.contract_type.value,
            entry_time.isoformat(),
            now.isoformat(),
            hold_duration,
            entry_time.strftime("%A"),
            minutes_since_open,
            exit_reason.value,
            position.strike,
            position.dte,
            position.delta_at_entry,
            position.iv_at_entry,
            position.iv_percentile,
            position.spread_width,
            position.spy_session_return,
            position.underlying_vs_vwap,
            position.vwap_slope,
            position.adx_value,
            position.rsi_value,
            position.volume_ratio,
            position.sector_rs,
            position.uw_net_premium_direction,
            position.gex_nearest_wall_distance,
            position.day2_score,
            position.entry_slippage,
            position.bid_ask_at_entry,
            position.fill_time_sec,
            pnl_dollars,
            pnl_pct,
            position.max_favorable,
            position.max_adverse,
            position.signal_strength_score,
            outcome,
        ))

        self.conn.commit()

        logger.info(
            f"Trade recorded: {trade_id} | {position.ticker} {position.strategy} "
            f"{outcome} {pnl_pct:+.1%} (${pnl_dollars:+.0f})"
        )

        # Update daily session
        self._update_daily_session(now.strftime("%Y-%m-%d"), pnl_dollars, outcome)

        return trade_id

    def _update_daily_session(
        self, date_str: str, pnl: float, outcome: str
    ) -> None:
        """Update the daily session summary."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO daily_sessions (date, trades_count, wins, losses, total_pnl)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?
        """, (
            date_str,
            1 if outcome == "WIN" else 0,
            1 if outcome == "LOSS" else 0,
            pnl,
            1 if outcome == "WIN" else 0,
            1 if outcome == "LOSS" else 0,
            pnl,
        ))
        self.conn.commit()

    def get_total_trades(self) -> int:
        """Get total number of completed trades."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        return cursor.fetchone()[0]

    def get_trades(
        self,
        strategy: Optional[str] = None,
        ticker: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        query += " ORDER BY entry_time DESC"

        if last_n:
            query += f" LIMIT {last_n}"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall trading statistics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_percent) as avg_pnl,
                SUM(pnl_dollars) as total_pnl,
                AVG(CASE WHEN outcome = 'WIN' THEN pnl_dollars END) as avg_win,
                AVG(CASE WHEN outcome = 'LOSS' THEN pnl_dollars END) as avg_loss
            FROM trades
        """)
        row = cursor.fetchone()
        if row is None:
            return {}

        total, wins, losses = row[0], row[1] or 0, row[2] or 0
        avg_win = row[5] or 0
        avg_loss = abs(row[6] or 1)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl_pct": row[3] or 0,
            "total_pnl": row[4] or 0,
            "profit_factor": (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 0,
        }

    def backup(self) -> None:
        """Create a backup of the database file."""
        if not os.path.exists(self.db_path):
            return

        today = datetime.now().strftime("%Y-%m-%d")
        backup_path = os.path.join(self.backup_dir, f"trades_{today}.db")

        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Trade DB backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
