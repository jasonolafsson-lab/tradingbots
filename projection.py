"""
Monthly return projection model.

Uses the bot's actual config parameters to estimate returns
under conservative, moderate, and optimistic scenarios.

THIS IS A GUESS — not financial advice.
"""

# ── Bot Parameters (from settings.yaml) ──
MAX_TRADES_PER_DAY = 6
RISK_PER_TRADE_PCT = 0.02       # 2% of account per trade
STOP_LOSS_PCT = 0.25            # -25% of premium
TAKE_PROFIT_PCT = 0.50          # +50% of premium
TRAILING_ACTIVATION = 0.25      # Trailing stop at +25%
TRAIL_DISTANCE = 0.15           # 15% trail from peak
TRADING_DAYS_PER_MONTH = 21

# ── Scenarios ──
scenarios = {
    "Conservative": {
        "avg_trades_per_day": 3,
        "win_rate": 0.48,
        "avg_win_pct": 0.30,        # Average winner: +30% of premium
        "avg_loss_pct": -0.22,       # Average loser: -22% of premium (tight stops)
        "avg_contracts": 2,
        "avg_premium": 5.00,
        "label": "Choppy markets, learning curve, some signal noise",
    },
    "Moderate": {
        "avg_trades_per_day": 4,
        "win_rate": 0.55,
        "avg_win_pct": 0.38,        # Winners run closer to TP
        "avg_loss_pct": -0.20,       # Stops working well
        "avg_contracts": 3,
        "avg_premium": 5.50,
        "label": "Decent trend days, strategies performing as designed",
    },
    "Optimistic": {
        "avg_trades_per_day": 5,
        "win_rate": 0.60,
        "avg_win_pct": 0.45,        # Trailing stops capturing runners
        "avg_loss_pct": -0.18,       # Quick exits on losers
        "avg_contracts": 3,
        "avg_premium": 6.00,
        "label": "Strong trends, Level 1 active and optimizing sizing",
    },
}

account_sizes = [10_000, 25_000, 50_000, 100_000]

print("=" * 80)
print("OPTIONS BOT — MONTHLY RETURN PROJECTION")
print("=" * 80)
print()
print("Assumptions:")
print(f"  Risk per trade:    {RISK_PER_TRADE_PCT:.0%} of account")
print(f"  Stop loss:         {STOP_LOSS_PCT:.0%} of premium")
print(f"  Take profit:       {TAKE_PROFIT_PCT:.0%} of premium")
print(f"  Trading days/mo:   {TRADING_DAYS_PER_MONTH}")
print(f"  Max trades/day:    {MAX_TRADES_PER_DAY}")
print()

for name, s in scenarios.items():
    trades_per_month = s["avg_trades_per_day"] * TRADING_DAYS_PER_MONTH
    wins_per_month = trades_per_month * s["win_rate"]
    losses_per_month = trades_per_month * (1 - s["win_rate"])

    # Per-trade P&L in terms of premium
    avg_win_dollars = s["avg_win_pct"] * s["avg_premium"] * 100 * s["avg_contracts"]
    avg_loss_dollars = s["avg_loss_pct"] * s["avg_premium"] * 100 * s["avg_contracts"]

    monthly_gross = (wins_per_month * avg_win_dollars) + (losses_per_month * avg_loss_dollars)
    profit_factor = abs((wins_per_month * avg_win_dollars) / (losses_per_month * avg_loss_dollars)) if losses_per_month > 0 else 0

    # Expected value per trade
    ev_per_trade = (s["win_rate"] * avg_win_dollars) + ((1 - s["win_rate"]) * avg_loss_dollars)

    print(f"{'─' * 80}")
    print(f"  {name.upper()} SCENARIO")
    print(f"  {s['label']}")
    print(f"{'─' * 80}")
    print(f"  Win rate:            {s['win_rate']:.0%}")
    print(f"  Avg trades/day:      {s['avg_trades_per_day']}")
    print(f"  Trades/month:        {trades_per_month:.0f}")
    print(f"  Avg contracts:       {s['avg_contracts']}")
    print(f"  Avg premium:         ${s['avg_premium']:.2f}")
    print(f"  Avg winner:          +{s['avg_win_pct']:.0%} → ${avg_win_dollars:+,.0f}")
    print(f"  Avg loser:           {s['avg_loss_pct']:.0%} → ${avg_loss_dollars:+,.0f}")
    print(f"  EV per trade:        ${ev_per_trade:+,.0f}")
    print(f"  Profit factor:       {profit_factor:.2f}")
    print()

    print(f"  {'Account':>12}  {'Monthly P&L':>14}  {'Monthly %':>10}  {'Annual %':>10}  {'Trades to L1':>12}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*12}")

    for acct in account_sizes:
        monthly_return_pct = monthly_gross / acct
        annual_pct = monthly_return_pct * 12
        months_to_l1 = 200 / trades_per_month

        print(f"  ${acct:>11,}  ${monthly_gross:>13,.0f}  {monthly_return_pct:>9.1%}  {annual_pct:>9.1%}  {months_to_l1:>9.1f} mo")

    print()

# ── Summary table ──
print(f"{'=' * 80}")
print("SUMMARY — Monthly Return % by Scenario and Account Size")
print(f"{'=' * 80}")
print(f"  {'Account':>12}  {'Conservative':>14}  {'Moderate':>14}  {'Optimistic':>14}")
print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*14}")

for acct in account_sizes:
    row = f"  ${acct:>11,}"
    for name, s in scenarios.items():
        trades_per_month = s["avg_trades_per_day"] * TRADING_DAYS_PER_MONTH
        wins = trades_per_month * s["win_rate"]
        losses = trades_per_month * (1 - s["win_rate"])
        avg_win_d = s["avg_win_pct"] * s["avg_premium"] * 100 * s["avg_contracts"]
        avg_loss_d = s["avg_loss_pct"] * s["avg_premium"] * 100 * s["avg_contracts"]
        monthly = (wins * avg_win_d) + (losses * avg_loss_d)
        pct = monthly / acct
        row += f"  {pct:>13.1%}"
    print(row)

print()
print("─" * 80)
print("IMPORTANT CAVEATS:")
print("  • These are ESTIMATES based on config parameters, not backtested results")
print("  • Real results depend on market conditions, slippage, and execution quality")
print("  • First 2-4 weeks will likely underperform as the bot calibrates")
print("  • Level 1 (auto-adjust) kicks in at 200 trades and should improve results")
print("  • Options have convex payoffs — a few big winners can skew results up")
print("  • Circuit breaker caps daily losses at -3% of account")
print("  • This is paper trading first — no real money at risk")
print("─" * 80)
