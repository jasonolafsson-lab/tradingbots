"""
Options Bot Dashboard — Streamlit App

View performance, trade history, and live status for all 4 bots.

Usage:
    source .venv/bin/activate
    streamlit run dashboard.py
"""

import json
import os
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml


# ──────────────────────────────────────────────
# Config & Constants
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "trades.db"
BRIEF_PATH = PROJECT_ROOT / "data" / "daily_brief.json"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Strategy → Bot mapping
BOT_STRATEGIES = {
    "Bot 1 (Momentum)": ["MOMENTUM", "GREEN_SECTOR", "DAY2", "TUESDAY_REVERSAL", "REVERSION"],
    "Bot 2 (Scalper)": ["POWER_HOUR"],
    "Bot 3 (Mean Reversion)": ["MEAN_REVERSION"],
    "Bot 4 (Credit Spreads)": ["CREDIT_SPREAD"],
    "Bot 5 (ORB)": ["ORB_BREAKOUT"],
    "Bot 6 (Gamma)": ["GAMMA_SCALP"],
}
STRATEGY_TO_BOT = {}
for _bot, _strats in BOT_STRATEGIES.items():
    for _s in _strats:
        STRATEGY_TO_BOT[_s] = _bot


# ──────────────────────────────────────────────
# Data Access
# ──────────────────────────────────────────────

@st.cache_resource
def get_db_connection():
    """Persistent SQLite connection (read-only)."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def query_df(conn, sql, params=None):
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn, params=params or [])
    except Exception:
        return pd.DataFrame()


def load_morning_brief():
    """Load the latest morning brief JSON."""
    if not BRIEF_PATH.exists():
        return None
    try:
        with open(BRIEF_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def compute_stats(df):
    """Compute trading stats from a DataFrame of trades."""
    if df.empty:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0, "profit_factor": 0,
            "avg_win": 0, "avg_loss": 0, "avg_hold_min": 0,
            "best_trade": 0, "worst_trade": 0,
        }
    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]
    gp = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
    avg_hold = df["hold_duration_sec"].mean() / 60 if "hold_duration_sec" in df.columns else 0

    return {
        "total": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(df) if len(df) > 0 else 0,
        "total_pnl": df["pnl_dollars"].sum(),
        "avg_pnl": df["pnl_dollars"].mean(),
        "profit_factor": gp / gl if gl > 0 else 0,
        "avg_win": wins["pnl_dollars"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl_dollars"].mean() if len(losses) > 0 else 0,
        "avg_hold_min": avg_hold,
        "best_trade": df["pnl_dollars"].max(),
        "worst_trade": df["pnl_dollars"].min(),
    }


# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Options Bot Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Options Bot Dashboard")

conn = get_db_connection()

if conn is None:
    st.warning(
        "No trade database found at `data/trades.db`. "
        "The bot hasn't recorded any trades yet.\n\n"
        "**To get started:**\n"
        "1. Follow the IBKR setup guide (`IBKR_SETUP.md`)\n"
        "2. Run the bot: `python main.py --dry-run`\n"
        "3. Come back here to see results."
    )
    st.stop()


# ──────────────────────────────────────────────
# Sidebar — Filters
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    # Date range
    trades_meta = query_df(conn, "SELECT MIN(date) as min_d, MAX(date) as max_d FROM trades")
    if not trades_meta.empty and trades_meta.iloc[0]["min_d"]:
        min_date = datetime.strptime(trades_meta.iloc[0]["min_d"], "%Y-%m-%d").date()
        max_date = datetime.strptime(trades_meta.iloc[0]["max_d"], "%Y-%m-%d").date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = date.today()

    # Bot filter
    bot_options = ["All Bots"] + list(BOT_STRATEGIES.keys())
    selected_bot = st.selectbox("Bot", bot_options)

    # Ticker filter
    tickers = query_df(conn, "SELECT DISTINCT ticker FROM trades ORDER BY ticker")
    ticker_list = ["All"] + tickers["ticker"].tolist() if not tickers.empty else ["All"]
    selected_ticker = st.selectbox("Ticker", ticker_list)

    # Strategy filter
    strategies = query_df(conn, "SELECT DISTINCT strategy FROM trades ORDER BY strategy")
    strategy_list = ["All"] + strategies["strategy"].tolist() if not strategies.empty else ["All"]
    selected_strategy = st.selectbox("Strategy", strategy_list)

    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("Options Bot V1 • Paper Trading")


# ──────────────────────────────────────────────
# Build Filtered Query
# ──────────────────────────────────────────────

def get_filtered_trades():
    sql = "SELECT * FROM trades WHERE date >= ? AND date <= ?"
    params = [str(start_date), str(end_date)]

    if selected_ticker != "All":
        sql += " AND ticker = ?"
        params.append(selected_ticker)
    if selected_strategy != "All":
        sql += " AND strategy = ?"
        params.append(selected_strategy)
    if selected_bot != "All Bots":
        bot_strats = BOT_STRATEGIES.get(selected_bot, [])
        if bot_strats:
            placeholders = ",".join("?" * len(bot_strats))
            sql += f" AND strategy IN ({placeholders})"
            params.extend(bot_strats)

    sql += " ORDER BY entry_time ASC"
    return query_df(conn, sql, params)


trades = get_filtered_trades()

# Add bot column
if not trades.empty:
    trades["bot"] = trades["strategy"].map(STRATEGY_TO_BOT).fillna("Unknown")
    trades["hold_min"] = trades["hold_duration_sec"].fillna(0) / 60.0


# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🤖 Bot Performance",
    "📋 Trade History",
    "🎯 Strategy Analytics",
    "🌅 Morning Brief",
    "📝 Event Log",
])


# ──────────────────────────────────────────────
# Tab 1: Overview
# ──────────────────────────────────────────────

with tab1:
    if trades.empty:
        st.info("No trades found for the selected filters.")
    else:
        stats = compute_stats(trades)

        # KPI Row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Trades", stats["total"])
        c2.metric("Win Rate", f"{stats['win_rate']:.1%}")
        c3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        c4.metric("Total P&L", f"${stats['total_pnl']:+,.0f}")
        c5.metric("Avg P&L/Trade", f"${stats['avg_pnl']:+,.0f}")
        c6.metric("Avg Hold", f"{stats['avg_hold_min']:.0f} min")

        st.divider()

        # Equity Curve + Daily P&L
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Equity Curve")
            equity = trades[["entry_time", "pnl_dollars"]].copy()
            equity["cumulative_pnl"] = equity["pnl_dollars"].cumsum()
            equity["trade_num"] = range(1, len(equity) + 1)

            fig = px.area(
                equity, x="trade_num", y="cumulative_pnl",
                labels={"trade_num": "Trade #", "cumulative_pnl": "Cumulative P&L ($)"},
            )
            fig.update_traces(line_color="#00cc96", fillcolor="rgba(0,204,150,0.2)")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Daily P&L")
            daily = query_df(conn,
                "SELECT date, total_pnl FROM daily_sessions WHERE date >= ? AND date <= ? ORDER BY date",
                [str(start_date), str(end_date)]
            )
            if not daily.empty:
                colors = ["#00cc96" if x >= 0 else "#ef553b" for x in daily["total_pnl"]]
                fig2 = go.Figure(go.Bar(
                    x=daily["date"], y=daily["total_pnl"],
                    marker_color=colors,
                ))
                fig2.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Date", yaxis_title="P&L ($)",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No daily session data yet.")

        # Distribution + Exit Reasons
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("P&L Distribution")
            fig3 = px.histogram(
                trades, x="pnl_percent", nbins=30,
                color="outcome",
                color_discrete_map={"WIN": "#00cc96", "LOSS": "#ef553b"},
                labels={"pnl_percent": "P&L (%)"},
            )
            fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.subheader("Exit Reasons")
            if "exit_reason" in trades.columns:
                exit_counts = trades["exit_reason"].value_counts()
                fig4 = px.pie(
                    values=exit_counts.values,
                    names=exit_counts.index,
                    hole=0.4,
                )
                fig4.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 2: Bot Performance (NEW)
# ──────────────────────────────────────────────

with tab2:
    st.header("Per-Bot Performance")

    if trades.empty:
        st.info("No trades found.")
    else:
        # Six-column bot comparison
        bot_cols = st.columns(6)

        for idx, (bot_name, strat_list) in enumerate(BOT_STRATEGIES.items()):
            with bot_cols[idx % 6]:
                st.subheader(bot_name)
                bot_df = trades[trades["strategy"].isin(strat_list)]

                if bot_df.empty:
                    st.caption("No trades yet")
                    continue

                s = compute_stats(bot_df)

                # KPIs in 2 columns
                m1, m2 = st.columns(2)
                m1.metric("Trades", s["total"])
                m2.metric("Win Rate", f"{s['win_rate']:.1%}")

                m3, m4 = st.columns(2)
                m3.metric("Profit Factor", f"{s['profit_factor']:.2f}")
                m4.metric("Total P&L", f"${s['total_pnl']:+,.0f}")

                m5, m6 = st.columns(2)
                m5.metric("Avg Win", f"${s['avg_win']:+,.0f}")
                m6.metric("Avg Loss", f"${s['avg_loss']:+,.0f}")

                m7, m8 = st.columns(2)
                m7.metric("Best", f"${s['best_trade']:+,.0f}")
                m8.metric("Worst", f"${s['worst_trade']:+,.0f}")

                st.metric("Avg Hold", f"{s['avg_hold_min']:.0f} min")

                # Mini equity curve
                bot_sorted = bot_df.sort_values("entry_time")
                bot_sorted["cum_pnl"] = bot_sorted["pnl_dollars"].cumsum()
                fig_bot = go.Figure()
                fig_bot.add_trace(go.Scatter(
                    x=list(range(1, len(bot_sorted) + 1)),
                    y=bot_sorted["cum_pnl"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(width=2),
                    fillcolor="rgba(0,204,150,0.15)",
                ))
                fig_bot.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=5, b=0),
                    showlegend=False,
                    xaxis_title="Trade #",
                    yaxis_title="P&L ($)",
                )
                st.plotly_chart(fig_bot, use_container_width=True)

        # Comparison table
        st.divider()
        st.subheader("Side-by-Side Comparison")

        comparison = []
        for bot_name, strat_list in BOT_STRATEGIES.items():
            bot_df = trades[trades["strategy"].isin(strat_list)]
            s = compute_stats(bot_df)
            comparison.append({
                "Bot": bot_name,
                "Trades": s["total"],
                "Win Rate": f"{s['win_rate']:.1%}",
                "Profit Factor": f"{s['profit_factor']:.2f}",
                "Total P&L": f"${s['total_pnl']:+,.0f}",
                "Avg P&L": f"${s['avg_pnl']:+,.0f}",
                "Avg Hold (min)": f"{s['avg_hold_min']:.0f}",
                "Best": f"${s['best_trade']:+,.0f}",
                "Worst": f"${s['worst_trade']:+,.0f}",
            })

        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Tab 3: Trade History
# ──────────────────────────────────────────────

with tab3:
    if trades.empty:
        st.info("No trades found for the selected filters.")
    else:
        st.subheader(f"Trade Log — {len(trades)} trades")

        display_cols = [
            "date", "bot", "ticker", "strategy", "direction",
            "entry_time", "exit_reason", "pnl_dollars", "pnl_percent",
            "hold_duration_sec", "signal_strength_score",
            "delta_at_entry", "strike", "dte", "outcome",
        ]
        available_cols = [c for c in display_cols if c in trades.columns]
        display_df = trades[available_cols].copy()

        # Format
        if "pnl_percent" in display_df.columns:
            display_df["pnl_percent"] = display_df["pnl_percent"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        if "pnl_dollars" in display_df.columns:
            display_df["pnl_dollars"] = display_df["pnl_dollars"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "")
        if "hold_duration_sec" in display_df.columns:
            display_df["hold_min"] = display_df["hold_duration_sec"].apply(
                lambda x: f"{x/60:.1f}" if pd.notna(x) else ""
            )
            display_df = display_df.drop(columns=["hold_duration_sec"])
        if "entry_time" in display_df.columns:
            display_df["entry_time"] = display_df["entry_time"].apply(
                lambda x: x[11:19] if isinstance(x, str) and len(x) > 19 else x
            )

        # Rename columns for display
        col_rename = {
            "date": "Date", "bot": "Bot", "ticker": "Ticker", "strategy": "Strategy",
            "direction": "Dir", "entry_time": "Entry", "exit_reason": "Exit",
            "pnl_dollars": "P&L ($)", "pnl_percent": "P&L %", "hold_min": "Hold (min)",
            "signal_strength_score": "Strength", "delta_at_entry": "Delta",
            "strike": "Strike", "dte": "DTE", "outcome": "Result",
        }
        display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

        # Trade detail expander
        st.divider()
        st.subheader("Trade Detail")
        trade_ids = trades["trade_id"].tolist()
        if trade_ids:
            selected_id = st.selectbox(
                "Select trade",
                trade_ids,
                format_func=lambda x: (
                    f"{x} — "
                    f"{trades[trades['trade_id']==x].iloc[0].get('bot', '?')} | "
                    f"{trades[trades['trade_id']==x].iloc[0]['ticker']} "
                    f"{trades[trades['trade_id']==x].iloc[0]['strategy']} "
                    f"{trades[trades['trade_id']==x].iloc[0]['outcome']}"
                ),
            )
            if selected_id:
                detail = trades[trades["trade_id"] == selected_id].iloc[0]
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Identification**")
                    st.markdown(f"Bot: **{detail.get('bot', '')}**")
                    st.markdown(f"Ticker: **{detail.get('ticker', '')}**")
                    st.markdown(f"Strategy: **{detail.get('strategy', '')}**")
                    st.markdown(f"Direction: **{detail.get('direction', '')}**")
                    st.markdown(f"Regime: **{detail.get('regime', '')}**")
                    st.markdown(f"Exit: **{detail.get('exit_reason', '')}**")

                with c2:
                    st.markdown("**Contract**")
                    st.markdown(f"Strike: **{detail.get('strike', '')}**")
                    st.markdown(f"DTE: **{detail.get('dte', '')}**")
                    st.markdown(f"Delta: **{detail.get('delta_at_entry', '')}**")
                    st.markdown(f"IV: **{detail.get('iv_at_entry', '')}**")
                    st.markdown(f"IV Pctile: **{detail.get('iv_percentile', '')}**")
                    st.markdown(f"Spread: **{detail.get('spread_width', 'N/A')}**")

                with c3:
                    st.markdown("**Market Context**")
                    st.markdown(f"SPY Return: **{detail.get('spy_session_return', '')}**")
                    st.markdown(f"VWAP Slope: **{detail.get('vwap_slope', '')}**")
                    st.markdown(f"ADX: **{detail.get('adx_value', '')}**")
                    st.markdown(f"RSI: **{detail.get('rsi_value', '')}**")
                    st.markdown(f"Vol Ratio: **{detail.get('volume_ratio', '')}**")
                    st.markdown(f"Strength: **{detail.get('signal_strength_score', '')}**")

                st.markdown("---")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("P&L $", f"${detail.get('pnl_dollars', 0):+,.0f}")
                e2.metric("P&L %", f"{detail.get('pnl_percent', 0):+.1%}")
                e3.metric("Max Favorable", f"{detail.get('max_favorable_excursion', 0):.1%}")
                e4.metric("Max Adverse", f"{detail.get('max_adverse_excursion', 0):.1%}")

        # CSV download
        csv = trades.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "trades.csv", "text/csv")


# ──────────────────────────────────────────────
# Tab 4: Strategy Analytics
# ──────────────────────────────────────────────

with tab4:
    if trades.empty:
        st.info("No trades found for the selected filters.")
    else:
        # Per Strategy
        st.subheader("Performance by Strategy")
        strat_rows = []
        for strat in trades["strategy"].unique():
            sdf = trades[trades["strategy"] == strat]
            s = compute_stats(sdf)
            strat_rows.append({
                "Strategy": strat,
                "Bot": STRATEGY_TO_BOT.get(strat, "?"),
                "Trades": s["total"],
                "Win Rate": f"{s['win_rate']:.1%}",
                "P/F": f"{s['profit_factor']:.2f}",
                "Total P&L": f"${s['total_pnl']:+,.0f}",
                "Avg P&L": f"${s['avg_pnl']:+,.0f}",
                "Avg Hold": f"{s['avg_hold_min']:.0f}m",
            })
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Per Ticker
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Performance by Ticker")
            tick_rows = []
            for tick in sorted(trades["ticker"].unique()):
                tdf = trades[trades["ticker"] == tick]
                s = compute_stats(tdf)
                tick_rows.append({
                    "Ticker": tick,
                    "Trades": s["total"],
                    "Win Rate": f"{s['win_rate']:.1%}",
                    "P/F": f"{s['profit_factor']:.2f}",
                    "Total P&L": f"${s['total_pnl']:+,.0f}",
                })
            st.dataframe(pd.DataFrame(tick_rows), use_container_width=True, hide_index=True)

        with c2:
            st.subheader("Win Rate by Strategy")
            wr_data = []
            for strat in trades["strategy"].unique():
                sdf = trades[trades["strategy"] == strat]
                n = len(sdf)
                w = len(sdf[sdf["outcome"] == "WIN"])
                wr_data.append({"Strategy": strat, "Win Rate": w / n if n > 0 else 0, "Trades": n})
            wr_df = pd.DataFrame(wr_data)
            if not wr_df.empty:
                fig = px.bar(wr_df, x="Strategy", y="Win Rate", text="Trades",
                             color="Win Rate",
                             color_continuous_scale=["#ef553b", "#ffa15a", "#00cc96"])
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.divider()
        st.subheader("Strategy × Ticker Heatmap (Win Rate)")

        pivot_data = []
        for strat in trades["strategy"].unique():
            for tick in trades["ticker"].unique():
                combo = trades[(trades["strategy"] == strat) & (trades["ticker"] == tick)]
                n = len(combo)
                if n > 0:
                    w = len(combo[combo["outcome"] == "WIN"])
                    pivot_data.append({"Strategy": strat, "Ticker": tick, "Win Rate": w / n, "Count": n})

        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data)
            heatmap = pivot_df.pivot(index="Strategy", columns="Ticker", values="Win Rate")
            counts = pivot_df.pivot(index="Strategy", columns="Ticker", values="Count")

            fig = go.Figure(go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns.tolist(),
                y=heatmap.index.tolist(),
                text=counts.values,
                texttemplate="%{text} trades",
                colorscale=[[0, "#ef553b"], [0.5, "#ffa15a"], [1, "#00cc96"]],
                zmin=0, zmax=1,
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Day of Week
        st.divider()
        st.subheader("Performance by Day of Week")
        dow_rows = []
        for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            ddf = trades[trades["day_of_week"] == dow]
            if ddf.empty:
                continue
            s = compute_stats(ddf)
            dow_rows.append({
                "Day": dow,
                "Trades": s["total"],
                "Win Rate": f"{s['win_rate']:.1%}",
                "Total P&L": f"${s['total_pnl']:+,.0f}",
                "Avg P&L": f"${s['avg_pnl']:+,.0f}",
            })
        if dow_rows:
            st.dataframe(pd.DataFrame(dow_rows), use_container_width=True, hide_index=True)

        # Time of Day
        st.subheader("P&L by Time of Day")
        if "minutes_since_open" in trades.columns:
            time_df = trades[["minutes_since_open", "pnl_dollars"]].copy()
            time_df["hour_bucket"] = (time_df["minutes_since_open"] // 60).astype(int)
            time_df["time_label"] = time_df["hour_bucket"].apply(
                lambda h: f"{9 + (h + 30) // 60}:{(h + 30) % 60:02d}+")

            time_agg = time_df.groupby("time_label").agg(
                trades=("pnl_dollars", "count"),
                total_pnl=("pnl_dollars", "sum"),
            ).reset_index()

            if not time_agg.empty:
                colors = ["#00cc96" if x >= 0 else "#ef553b" for x in time_agg["total_pnl"]]
                fig = go.Figure(go.Bar(
                    x=time_agg["time_label"], y=time_agg["total_pnl"],
                    marker_color=colors, text=time_agg["trades"],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Entry Hour (ET)", yaxis_title="Total P&L ($)",
                )
                st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 5: Morning Brief (NEW)
# ──────────────────────────────────────────────

with tab5:
    st.header("Morning Brief")

    brief = load_morning_brief()

    if brief is None:
        st.info("No morning brief found. The brief runs at 8:30 AM ET on weekdays.")
    else:
        brief_date = brief.get("date", "Unknown")
        is_today = brief_date == date.today().isoformat()

        if not is_today:
            st.warning(f"Showing brief from **{brief_date}** (not today)")

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sentiment", brief.get("market_sentiment", "?"))
        c2.metric("Risk Level", brief.get("risk_level", "?"))
        c3.metric("Overnight Move", f"{brief.get('overnight_move_pct', 0):+.2f}%")
        c4.metric("VIX", f"{brief.get('vix_level', 0):.1f} ({brief.get('vix_regime', '?')})")
        c5.metric("News Score", f"{brief.get('news_sentiment_score', 0):+.2f}")

        st.divider()

        # Bot adjustments
        st.subheader("Bot Adjustments")
        bcols = st.columns(3)

        for idx, (key, label) in enumerate([
            ("bot1_momentum", "Bot 1 (Momentum)"),
            ("bot2_scalper", "Bot 2 (Scalper)"),
            ("bot3_reversion", "Bot 3 (Mean Reversion)"),
        ]):
            with bcols[idx]:
                adj = brief.get(key, {})
                enabled = adj.get("enabled", True)
                mult = adj.get("size_multiplier", 1.0)
                notes = adj.get("notes", "")

                if enabled:
                    st.metric(label, f"{mult}x size")
                else:
                    st.metric(label, "DISABLED")
                if notes:
                    st.caption(notes)

        # Scheduled Events
        events = brief.get("scheduled_events", [])
        if events:
            st.divider()
            st.subheader("Scheduled Events")
            for e in events:
                impact = e.get("impact", "?")
                icon = "🔴" if impact == "HIGH" else "🟡" if impact == "MEDIUM" else "⚪"
                st.markdown(f"{icon} **{e.get('time', '?')}** — {e.get('event', '?')} [{impact}]")

        # No-trade windows
        windows = brief.get("no_trade_windows", [])
        if windows:
            st.divider()
            st.subheader("No-Trade Windows")
            for w in windows:
                st.markdown(f"🚫 **{w['start']} - {w['end']}** — {w['reason']}")

        # Headlines
        headlines = brief.get("news_headlines", [])
        if headlines:
            st.divider()
            st.subheader("Top Headlines")
            for h in headlines:
                st.markdown(f"- {h}")

        # FOMC flag
        if brief.get("is_fomc_day"):
            st.divider()
            st.error("⚠️ FOMC DAY — All bots have adjusted sizing")


# ──────────────────────────────────────────────
# Tab 6: Event Log
# ──────────────────────────────────────────────

with tab6:
    st.subheader("Event Log Viewer")

    log_files = sorted(LOG_DIR.glob("*.jsonl"), reverse=True) if LOG_DIR.exists() else []
    # Also check for plain .log files
    plain_logs = sorted(LOG_DIR.glob("*.log"), reverse=True) if LOG_DIR.exists() else []

    if not log_files and not plain_logs:
        st.info("No log files found in `logs/`. Logs are created when bots run during market hours.")
    else:
        all_logs = log_files + plain_logs

        if log_files:
            selected_log = st.selectbox(
                "Select log file",
                log_files,
                format_func=lambda p: p.stem,
            )

            categories = ["All", "SYSTEM", "SCANNER", "REGIME", "SIGNAL", "ORDER", "RISK", "DAILY_SUMMARY"]
            selected_cat = st.selectbox("Filter by category", categories)

            events = []
            try:
                with open(selected_log) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            if selected_cat == "All" or event.get("category") == selected_cat:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                st.error(f"Error reading log: {e}")

            if events:
                st.caption(f"Showing {min(len(events), 100)} of {len(events)} events")

                for event in events[-100:]:
                    cat = event.get("category", "UNKNOWN")
                    ts = event.get("timestamp", "")
                    ts_short = ts[11:19] if len(ts) > 19 else ts

                    color_map = {
                        "SYSTEM": "🔵", "SCANNER": "🟣", "REGIME": "🟡",
                        "SIGNAL": "🟢", "ORDER": "🟠", "RISK": "🔴",
                        "DAILY_SUMMARY": "📊",
                    }
                    icon = color_map.get(cat, "⚪")

                    if cat == "SIGNAL":
                        ticker = event.get("ticker", "")
                        direction = event.get("direction", "")
                        strategy = event.get("strategy", "")
                        executed = "✅" if event.get("executed") else "❌"
                        score = event.get("strength_score", 0)
                        summary = f"**{ticker}** {direction} {strategy} (score: {score}) {executed}"
                    elif cat == "ORDER":
                        evt = event.get("event", "")
                        ticker = event.get("ticker", "")
                        if evt == "ENTRY":
                            summary = f"**ENTRY** {ticker} — {event.get('num_contracts', 0)} contracts @ ${event.get('entry_price', 0)}"
                        elif evt == "EXIT":
                            summary = f"**EXIT** {ticker} — {event.get('pnl_pct', 0):+.1%} ({event.get('exit_reason', '')})"
                        else:
                            summary = f"{evt} {ticker}"
                    elif cat == "DAILY_SUMMARY":
                        summary = f"**{event.get('date', '')}** — {event.get('trades_today', 0)} trades, P&L: ${event.get('daily_pnl', 0):,.0f}"
                    else:
                        summary = event.get("message", json.dumps(event)[:100])

                    st.markdown(f"`{ts_short}` {icon} **{cat}** — {summary}")
            else:
                st.info("No events match the selected filter.")

        # Show plain log files
        if plain_logs:
            st.divider()
            st.subheader("Bot Logs")
            selected_plain = st.selectbox(
                "Select bot log",
                plain_logs,
                format_func=lambda p: p.name,
            )
            if selected_plain:
                try:
                    with open(selected_plain) as f:
                        content = f.read()
                    # Show last 100 lines
                    lines = content.strip().split("\n")
                    last_lines = lines[-100:]
                    st.code("\n".join(last_lines), language="text")
                except Exception as e:
                    st.error(f"Error reading log: {e}")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
config = load_config()
risk = config.get("risk", {})
st.caption(
    f"Bot Config: Max risk/trade: {risk.get('max_trade_risk_pct', 0.02):.0%} | "
    f"Daily loss limit: {risk.get('daily_loss_limit_pct', 0.03):.0%} | "
    f"Max trades/day: {risk.get('max_trades_per_day', 6)} | "
    f"DB: {DB_PATH}"
)
