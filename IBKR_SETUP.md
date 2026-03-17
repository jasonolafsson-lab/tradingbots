# IBKR Setup Guide — Options Bot

This guide walks you through connecting the bot to Interactive Brokers for paper trading.

---

## Step 1: Create an IBKR Account

If you don't already have one:

1. Go to [interactivebrokers.com](https://www.interactivebrokers.com)
2. Click **Open Account**
3. Complete the application (basic personal info, financial profile)
4. Once approved, you'll get a username like `jason_olafsson`

**Paper Trading:** Every IBKR account comes with a free paper trading account. Your paper account ID will start with **"DU"** (e.g., `DU1234567`). The bot is hard-coded to only accept paper accounts.

---

## Step 2: Download IB Gateway

IB Gateway is the lightweight, headless connection — better than TWS for automated trading.

1. Go to: [interactivebrokers.com/en/trading/ibgateway-stable.php](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Download the **Stable** version for macOS
3. Install it (drag to Applications)

> **Why Gateway instead of TWS?** Gateway is lighter weight, uses less memory, and is designed for API-only connections. TWS works too but has a full GUI you don't need.

---

## Step 3: Log into IB Gateway (Paper Mode)

1. Open **IB Gateway** from Applications
2. At the login screen:
   - Enter your IBKR **username**
   - Enter your **password**
   - **IMPORTANT:** Click the **"Paper Trading"** button (not "Live Trading")
3. Complete any two-factor authentication if prompted
4. Wait for the gateway to show **"Connected"** status

You should see a small window showing:
- Account: `DU1234567` (your paper account)
- Status: Connected
- Data: Active

---

## Step 4: Configure API Settings

In IB Gateway:

1. Click **Configure** (gear icon or menu bar)
2. Go to **Settings → API → Settings**
3. Set these options:

| Setting | Value |
|---------|-------|
| **Enable ActiveX and Socket Clients** | ✅ Checked |
| **Socket port** | **7497** |
| **Allow connections from localhost only** | ✅ Checked |
| **Read-Only API** | ❌ Unchecked (bot needs to place orders) |
| **Master API client ID** | Leave blank |
| **Trusted IPs** | Leave as 127.0.0.1 |

4. Click **Apply** then **OK**

> **Port 7497** is the standard paper trading port. The bot will refuse to connect to port 7496 (live trading) as a safety measure.

---

## Step 5: Configure the Bot

Edit `config/settings.yaml`:

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497                     # Paper trading port
  client_id: 1                   # Must be unique per connection
  allowed_accounts:
    - "DU1234567"                # Replace with YOUR paper account ID
  timeout_sec: 30
  readonly: false
```

**Find your paper account ID:**
- It's shown in the IB Gateway window title bar
- Or in TWS: Account → Account Window → top right corner
- Starts with "DU"

---

## Step 6: Test the Connection

With IB Gateway running:

```bash
cd ~/Projects/options_bot
source .venv/bin/activate
python main.py --dry-run
```

**What you should see:**
```
[INFO] Connecting to IBKR at 127.0.0.1:7497...
[INFO] Connected. Account: DU1234567
[INFO] Paper account verified.
[INFO] Running in DRY RUN mode — no orders will be placed.
```

**If it works:** The bot will start its event loop, wait for market hours, run the scanner, and generate signals (without placing orders in dry-run mode).

---

## Step 7: Unusual Whales API (Optional)

The bot works without Unusual Whales — it gracefully degrades. But for full intelligence features:

1. Sign up at [unusualwhales.com](https://unusualwhales.com)
2. Get your API token from your account settings
3. Set it as an environment variable:

```bash
export UW_API_TOKEN="your_token_here"
```

Or add it to `config/settings.yaml`:
```yaml
unusual_whales:
  api_token: "your_token_here"
```

---

## Step 8: Run the Bot

```bash
# Dry run first (signals only, no orders)
python main.py --dry-run

# Paper trading (will place real paper orders)
python main.py

# View the dashboard (in a separate terminal)
streamlit run dashboard.py
```

---

## Troubleshooting

### "Connection refused" error
- Make sure IB Gateway is running and logged in
- Verify it's on port **7497** (not 4001 or 4002)
- Check that "Enable ActiveX and Socket Clients" is checked

### "Account not allowed" error
- Add your paper account ID (starts with "DU") to `allowed_accounts` in settings.yaml
- Make sure you logged into Gateway in **Paper Trading** mode

### "Port 7496 blocked" error
- The bot detected you're trying to connect to the live trading port
- Close Gateway, reopen, and log in with **Paper Trading**

### Gateway disconnects after a few hours
- IB Gateway disconnects daily around 11:45 PM ET for a brief restart
- Go to Configure → Settings → Lock and Exit → set "Auto restart" to Yes

### "Client ID already in use"
- Another application (or a previous bot instance) is using client_id 1
- Change `client_id` in settings.yaml to 2 or 3
- Or close the other connection first

### Market data not flowing
- Paper accounts have a 15-minute delay on market data by default
- Subscribe to real-time data in your IBKR account management page (free for paper)
- Or the market might simply be closed (bot only runs 9:30 AM - 4:00 PM ET)

---

## Daily Workflow

1. **Before market open (before 9:30 AM ET):**
   - Start IB Gateway, log in with Paper Trading
   - Start the bot: `python main.py`
   - Bot will auto-run the pre-market scanner at 9:00 AM

2. **During market hours:**
   - Bot runs autonomously
   - Monitor via dashboard: `streamlit run dashboard.py`
   - Check logs at `logs/YYYY-MM-DD.jsonl`

3. **After market close:**
   - Bot auto-shuts down after closing all positions
   - Review the daily summary in the dashboard
   - IB Gateway can stay running for the next day

---

## Architecture Overview

```
┌─────────────┐     Port 7497      ┌──────────────┐
│  IB Gateway  │ ◄─────────────────► │  Options Bot  │
│  (Paper)     │   ib_async library  │  (main.py)    │
└─────────────┘                      └──────┬───────┘
                                            │
                                     ┌──────▼───────┐
                                     │  Dashboard    │
                                     │  (Streamlit)  │
                                     │  Port 8501    │
                                     └──────────────┘
```

The bot connects to IB Gateway for price data and order execution. The dashboard reads from the SQLite trade database — it doesn't connect to IBKR directly.
