"""
Seed the trade DB with sample data for dashboard verification.
Run once:  python seed_test_data.py
"""

import os
import sys
import json
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "trades.db"
LOG_DIR = PROJECT_ROOT / "logs"

# ── Seed trades directly into SQLite (bypasses needing all Position deps) ──

STRATEGIES = ["MOMENTUM", "REVERSION", "DAY2", "TUESDAY_REVERSAL", "GREEN_SECTOR"]
TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "AMZN"]
DIRECTIONS = ["CALL", "PUT"]
EXIT_REASONS = ["TAKE_PROFIT", "STOP_LOSS", "TIME_EXIT", "TRAIL_STOP", "EOD_EXIT"]
REGIMES = ["MOMENTUM", "REVERSION", "DAY2", "NO_TRADE"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

random.seed(42)

def create_db():
    os.makedirs(DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            strategy TEXT NOT NULL,
            regime TEXT NOT NULL,
            direction TEXT NOT NULL,
            contract_type TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            hold_duration_sec INTEGER,
            day_of_week TEXT NOT NULL,
            minutes_since_open INTEGER,
            exit_reason TEXT,
            strike REAL,
            dte INTEGER,
            delta_at_entry REAL,
            iv_at_entry REAL,
            iv_percentile REAL,
            spread_width REAL,
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
            entry_slippage REAL,
            bid_ask_at_entry REAL,
            fill_time_sec REAL,
            pnl_dollars REAL,
            pnl_percent REAL,
            max_favorable_excursion REAL,
            max_adverse_excursion REAL,
            signal_strength_score REAL,
            outcome TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            ml_confidence_score REAL
        )
    """)

    cur.execute("""
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

    conn.commit()
    return conn


def seed_trades(conn, num_trades=60):
    cur = conn.cursor()

    # Clear existing
    cur.execute("DELETE FROM trades")
    cur.execute("DELETE FROM daily_sessions")
    conn.commit()

    base_date = datetime(2026, 3, 1, 9, 35)
    daily_summary = {}

    for i in range(num_trades):
        # Spread trades across ~10 trading days
        day_offset = i // 6
        intraday_offset = (i % 6) * 35  # minutes apart
        trade_time = base_date + timedelta(days=day_offset, minutes=intraday_offset)

        # Skip weekends
        while trade_time.weekday() >= 5:
            trade_time += timedelta(days=1)

        trade_id = f"SEED{i:04d}TEST"
        ticker = random.choice(TICKERS)
        strategy = random.choice(STRATEGIES)
        direction = random.choice(DIRECTIONS)

        # Bias toward wins slightly (55% win rate)
        is_win = random.random() < 0.55
        if is_win:
            pnl_pct = random.uniform(0.03, 0.40)
            exit_reason = random.choice(["TAKE_PROFIT", "TRAIL_STOP"])
        else:
            pnl_pct = random.uniform(-0.35, -0.02)
            exit_reason = random.choice(["STOP_LOSS", "TIME_EXIT", "EOD_EXIT"])

        entry_price = random.uniform(2.0, 15.0)
        num_contracts = random.choice([1, 2, 3, 5])
        pnl_dollars = pnl_pct * entry_price * 100 * num_contracts
        outcome = "WIN" if pnl_pct >= 0 else "LOSS"

        hold_sec = random.randint(300, 14400)
        exit_time = trade_time + timedelta(seconds=hold_sec)
        mins_since_open = (trade_time.hour * 60 + trade_time.minute) - (9 * 60 + 30)

        date_str = trade_time.strftime("%Y-%m-%d")

        cur.execute("""
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
            trade_id, date_str, ticker, strategy,
            random.choice(REGIMES), direction, "SINGLE_LEG",
            trade_time.isoformat(), exit_time.isoformat(),
            hold_sec, trade_time.strftime("%A"), mins_since_open, exit_reason,
            round(random.uniform(400, 550), 0),  # strike
            random.choice([0, 1, 2, 3, 5]),       # dte
            round(random.uniform(0.20, 0.60), 2),  # delta
            round(random.uniform(0.15, 0.80), 2),  # iv
            round(random.uniform(20, 90), 1),       # iv_percentile
            None,  # spread_width
            round(random.uniform(-0.02, 0.02), 4),  # spy return
            round(random.uniform(-2.0, 2.0), 2),    # vs vwap
            round(random.uniform(-0.5, 0.5), 3),    # vwap slope
            round(random.uniform(15, 45), 1),        # adx
            round(random.uniform(25, 75), 1),        # rsi
            round(random.uniform(0.5, 3.0), 2),     # volume ratio
            round(random.uniform(-0.02, 0.02), 4),  # sector rs
            random.choice(["BULLISH", "BEARISH", None]),
            round(random.uniform(0, 5), 2),          # gex distance
            round(random.uniform(40, 95), 1) if strategy == "DAY2" else None,
            round(random.uniform(0.0, 0.05), 3),    # slippage
            round(random.uniform(0.05, 0.30), 2),   # bid-ask
            round(random.uniform(0.5, 5.0), 1),     # fill time
            round(pnl_dollars, 2),
            round(pnl_pct, 4),
            round(max(pnl_pct, 0) + random.uniform(0, 0.10), 4),  # MFE
            round(min(pnl_pct, 0) - random.uniform(0, 0.10), 4),  # MAE
            round(random.uniform(50, 95), 1),        # signal strength
            outcome,
        ))

        # Accumulate daily summary
        if date_str not in daily_summary:
            daily_summary[date_str] = {"count": 0, "wins": 0, "losses": 0, "pnl": 0}
        daily_summary[date_str]["count"] += 1
        daily_summary[date_str]["wins"] += 1 if outcome == "WIN" else 0
        daily_summary[date_str]["losses"] += 1 if outcome == "LOSS" else 0
        daily_summary[date_str]["pnl"] += pnl_dollars

    # Insert daily sessions
    for d, s in daily_summary.items():
        cur.execute("""
            INSERT OR REPLACE INTO daily_sessions (date, trades_count, wins, losses, total_pnl, spy_return)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (d, s["count"], s["wins"], s["losses"], round(s["pnl"], 2),
              round(random.uniform(-0.015, 0.015), 4)))

    conn.commit()
    print(f"Seeded {num_trades} trades across {len(daily_summary)} trading days")


def seed_logs():
    """Create a sample .jsonl log file for today."""
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"{today}.jsonl"

    events = [
        {"ts": "09:00:01", "category": "SYSTEM", "event": "bot_started",
         "msg": "Options bot started — dry run mode"},
        {"ts": "09:00:05", "category": "SYSTEM", "event": "ibkr_connected",
         "msg": "Connected to IBKR paper account DU1234567"},
        {"ts": "09:15:00", "category": "SIGNAL", "event": "pre_market_scan",
         "msg": "Pre-market scanner found 4 tickers: SPY, QQQ, NVDA, TSLA"},
        {"ts": "09:30:02", "category": "SIGNAL", "event": "regime_classified",
         "msg": "SPY regime: MOMENTUM (OR breakout + ADX 32 + volume 1.8x)"},
        {"ts": "09:32:15", "category": "SIGNAL", "event": "signal_generated",
         "msg": "BUY CALL signal: SPY MOMENTUM strength=82"},
        {"ts": "09:32:18", "category": "ORDER", "event": "order_placed",
         "msg": "Placed order: SPY 450C 0DTE x3 @ $5.20 limit"},
        {"ts": "09:32:20", "category": "ORDER", "event": "order_filled",
         "msg": "Filled: SPY 450C 0DTE x3 @ $5.18 (slippage: -$0.02)"},
        {"ts": "09:45:00", "category": "SIGNAL", "event": "signal_generated",
         "msg": "BUY PUT signal: QQQ REVERSION strength=71"},
        {"ts": "09:45:03", "category": "ORDER", "event": "order_placed",
         "msg": "Placed order: QQQ 380P 1DTE x2 @ $3.40 limit"},
        {"ts": "10:15:30", "category": "RISK", "event": "stop_triggered",
         "msg": "QQQ PUT stopped out at -15% (-$102)"},
        {"ts": "10:45:00", "category": "ORDER", "event": "take_profit",
         "msg": "SPY CALL take-profit hit at +22% (+$342)"},
        {"ts": "11:00:00", "category": "SIGNAL", "event": "signal_generated",
         "msg": "BUY CALL signal: NVDA MOMENTUM strength=78"},
        {"ts": "13:30:00", "category": "RISK", "event": "circuit_breaker_check",
         "msg": "Circuit breaker OK — 3 trades, daily P&L: +$240"},
        {"ts": "15:45:00", "category": "SYSTEM", "event": "eod_cleanup",
         "msg": "End-of-day: closed 1 remaining position (NVDA +8%)"},
        {"ts": "16:00:01", "category": "SYSTEM", "event": "bot_shutdown",
         "msg": "Bot shutdown complete. Day summary: 3 trades, 2W/1L, +$480"},
    ]

    with open(log_path, "w") as f:
        for ev in events:
            ev["date"] = today
            f.write(json.dumps(ev) + "\n")

    print(f"Wrote {len(events)} log events to {log_path}")


if __name__ == "__main__":
    conn = create_db()
    seed_trades(conn, num_trades=60)
    seed_logs()
    conn.close()
    print(f"\nDatabase: {DB_PATH}")
    print(f"Logs:     {LOG_DIR}")
    print("\nRun the dashboard with:")
    print("  source .venv/bin/activate")
    print("  streamlit run dashboard.py")
