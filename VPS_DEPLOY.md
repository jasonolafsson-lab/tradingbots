# VPS Deployment Guide — Options Bot

Your VPS IP: `157.230.55.153`

---

## Step 1: Log into your VPS (from your Mac Terminal)

```bash
ssh root@157.230.55.153
```

It will ask "Are you sure you want to continue connecting?" — type `yes`.
Then enter the password you set when creating the droplet.

Once you see `root@options-bot:~#` you're in.

---

## Step 2: Upload the bot code (open a NEW Terminal tab on your Mac)

Keep the SSH tab open. Open a new Terminal tab (Cmd+T) and run:

```bash
scp -r ~/Projects/options_bot root@157.230.55.153:/tmp/options_bot_upload
```

This copies the entire project to the VPS. Takes ~30 seconds.

---

## Step 3: Run the setup (back in your SSH tab)

```bash
cp -r /tmp/options_bot_upload /root/options_bot_source
cd /root/options_bot_source/deploy
sudo bash setup_vps.sh
```

The script will ask for 3 things:
1. **IBKR Paper Username** — your paper trading login
2. **IBKR Paper Password** — it won't show as you type (that's normal)
3. **Paper Account ID** — starts with "DU" (find it in TWS → Account → Account ID)

Then it runs for about 5 minutes. You'll see green checkmarks as each step completes.

---

## Step 4: Verify it worked

At the end of the script you'll see a summary. Then open your browser:

```
http://157.230.55.153:8501
```

You should see the Streamlit dashboard.

---

## Step 5: Start the bot manually (first time)

```bash
sudo systemctl start optionsbot
```

Watch the logs:

```bash
sudo journalctl -u optionsbot -f
```

Press Ctrl+C to stop watching (the bot keeps running).

After this first test, the bot will auto-start every weekday at 9:00 AM ET.

---

## Daily Operation

You don't need to do anything. Here's what happens automatically:

| Time (ET) | What happens |
|-----------|-------------|
| Always on | IB Gateway connected, Dashboard accessible |
| 9:00 AM | Bot starts, scans market |
| 9:30 AM | Market opens, opening range builds |
| 9:45 AM | Bot begins trading |
| 3:30 PM | No new entries |
| 3:55 PM | All positions force-closed |
| 4:00 PM | Bot sleeps until next trading day |

---

## Useful Commands

```bash
# Check what's running
sudo systemctl status ibgateway
sudo systemctl status optionsbot
sudo systemctl status optionsbot-dashboard

# Start/stop the bot
sudo systemctl start optionsbot
sudo systemctl stop optionsbot

# Watch live logs
sudo journalctl -u optionsbot -f        # Bot
sudo journalctl -u ibgateway -f         # IB Gateway

# Restart everything
sudo systemctl restart ibgateway
sudo systemctl restart optionsbot
sudo systemctl restart optionsbot-dashboard

# Check if bot will start tomorrow
sudo systemctl list-timers optionsbot-start.timer
```

---

## Troubleshooting

**Dashboard won't load:**
```bash
sudo systemctl restart optionsbot-dashboard
sudo journalctl -u optionsbot-dashboard -n 20
```

**IB Gateway won't connect:**
```bash
sudo journalctl -u ibgateway -n 30
# Common fix: restart it
sudo systemctl restart ibgateway
```

**Bot not trading:**
```bash
# Check if it's running
sudo systemctl status optionsbot
# Check logs for errors
sudo journalctl -u optionsbot -n 50
```

**Need to update bot code:**
From your Mac:
```bash
scp -r ~/Projects/options_bot root@157.230.55.153:/tmp/options_bot_upload
```
Then on the VPS:
```bash
sudo systemctl stop optionsbot
rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='data/' --exclude='logs/' --exclude='.env' \
  /tmp/options_bot_upload/ /home/optionsbot/options_bot/
sudo chown -R optionsbot:optionsbot /home/optionsbot/options_bot
sudo systemctl start optionsbot
```

---

## Cost

- DigitalOcean: **$16/month**
- That's it. No other costs.

---

## Architecture

```
┌─────────────────── YOUR MAC ───────────────────┐
│                                                 │
│  TWS (Live Trading) ← You trade here normally   │
│  Browser → http://157.230.55.153:8501           │
│            (check dashboard anytime)            │
│                                                 │
└─────────────────────────────────────────────────┘
              │
              │ internet
              ▼
┌──────────── VPS (DigitalOcean) ────────────────┐
│                                                 │
│  Xvfb (virtual display)                        │
│    └── IB Gateway ← auto-login, paper mode      │
│          └── Options Bot ← trades automatically  │
│                └── SQLite DB (trade history)     │
│                                                 │
│  Streamlit Dashboard ← always on, port 8501     │
│                                                 │
│  Daily Timer → starts bot at 9:00 AM ET         │
│                                                 │
└─────────────────────────────────────────────────┘
```
