# VPS Deployment Guide — Options Bot

> Total time: ~20 minutes. No Linux experience needed.

---

## Step 1: Create a DigitalOcean Account (3 min)

1. Go to [digitalocean.com](https://www.digitalocean.com)
2. Sign up (you can use your Google account)
3. Add a payment method — they often give $200 free credits for 60 days

---

## Step 2: Create a Droplet (3 min)

1. Click **Create → Droplets**
2. Choose these settings:

| Setting | Pick This |
|---------|-----------|
| Region | **New York (NYC1)** — closest to IBKR servers |
| Image | **Ubuntu 24.04 LTS** |
| Size | **Basic → Regular → $6/mo** (1 CPU, 1 GB RAM, 25 GB disk) |
| Authentication | **Password** (pick something strong) |
| Hostname | `options-bot` |

3. Click **Create Droplet**
4. Wait 60 seconds — you'll get an **IP address** (e.g., `164.90.xxx.xxx`)

---

## Step 3: Connect to Your VPS (2 min)

Open Terminal on your Mac and type:

```bash
ssh root@YOUR_IP_ADDRESS
```

It will ask:
- "Are you sure you want to continue?" → type `yes`
- Password → paste the password you set in Step 2

You're now logged into your VPS. You'll see something like:

```
root@options-bot:~#
```

---

## Step 4: Upload the Bot Code (3 min)

**Open a NEW Terminal tab** (Cmd+T) on your Mac. Don't close the VPS tab.

Run this command from your Mac (replace YOUR_IP):

```bash
scp -r ~/Projects/options_bot/deploy root@YOUR_IP:/root/
```

It will ask for your password again. This uploads the deploy folder to the VPS.

---

## Step 5: Run the Setup Script (5 min)

Go back to your **VPS terminal tab** and run:

```bash
cd /root/deploy
bash setup_vps.sh
```

Sit back — it will:
- Install Python 3.12 and all dependencies
- Download and install IB Gateway
- Install IBC (headless controller)
- Copy your bot code
- Set up auto-start services
- Configure the firewall

When it finishes, you'll see `SETUP COMPLETE` with next steps.

---

## Step 6: Add Your IBKR Paper Credentials (2 min)

Still on the VPS, run:

```bash
sudo nano /home/optionsbot/.env
```

Change these two lines:

```
IB_USER=your_paper_username
IB_PASS=your_paper_password
```

To save: press **Ctrl+O**, then **Enter**, then **Ctrl+X** to exit.

> **Your paper trading username/password is separate from your live account.**
> If you haven't created one yet:
> 1. Log in to [ibkr.com](https://www.ibkr.com) with your live account
> 2. Go to **Settings → Account Settings → Paper Trading Account**
> 3. Click **Create** or **Reset Password**
> 4. You'll get a separate username (usually your live username + a number)

---

## Step 7: Add Your Paper Account ID (1 min)

```bash
sudo nano /home/optionsbot/options_bot/config/settings.yaml
```

Find the line:

```yaml
allowed_accounts: []
```

Change it to (replace with your actual paper account ID):

```yaml
allowed_accounts: ["DU1234567"]
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

> Your paper account ID starts with "DU" and is visible in TWS under
> Account → Account Window, or at the top of your paper trading session.

---

## Step 8: Start Everything (2 min)

```bash
# Start IB Gateway (connects to IBKR)
sudo systemctl start ibgateway

# Wait 30 seconds for it to connect
sleep 30

# Check it's running
sudo systemctl status ibgateway

# Start the dashboard (always-on, accessible from your browser)
sudo systemctl start optionsbot-dashboard

# Start the bot
sudo systemctl start optionsbot
```

---

## Step 9: Open the Dashboard 🎉

On your Mac (or phone), open a browser and go to:

```
http://YOUR_VPS_IP:8501
```

You should see the Streamlit dashboard with 4 tabs.

---

## Daily Operation

**You don't need to do anything.** The bot:

- Starts automatically at **9:00 AM ET** on weekdays (systemd timer)
- Trades from 9:45 AM to 3:30 PM ET
- Closes all positions by 3:55 PM ET
- The dashboard runs 24/7 — check it whenever you want
- IB Gateway auto-reconnects if it drops

---

## Useful Commands

Run these from your VPS (ssh in first):

```bash
# Check status of all services
sudo systemctl status ibgateway
sudo systemctl status optionsbot
sudo systemctl status optionsbot-dashboard

# View live bot logs
sudo journalctl -u optionsbot -f

# View IB Gateway logs
sudo journalctl -u ibgateway -f

# Stop the bot
sudo systemctl stop optionsbot

# Restart everything
sudo systemctl restart ibgateway
sudo systemctl restart optionsbot

# Emergency kill (also works by creating a KILL file)
sudo -u optionsbot touch /home/optionsbot/options_bot/KILL

# Check if the daily timer is set
sudo systemctl list-timers | grep optionsbot
```

---

## Updating the Bot Code

When we make changes to the bot, deploy them like this.

**From your Mac:**

```bash
# Upload updated code
scp -r ~/Projects/options_bot/* root@YOUR_IP:/home/optionsbot/options_bot/

# SSH in and restart
ssh root@YOUR_IP
sudo systemctl restart optionsbot
```

---

## Troubleshooting

### "Connection refused" on port 8501
```bash
# Make sure the dashboard is running
sudo systemctl start optionsbot-dashboard
sudo systemctl status optionsbot-dashboard

# Make sure firewall allows it
sudo ufw status
# Should show "8501/tcp ALLOW Anywhere"
```

### IB Gateway won't connect
```bash
# Check the logs
sudo journalctl -u ibgateway --no-pager -n 50

# Common fixes:
# 1. Wrong username/password → edit /home/optionsbot/.env
# 2. Paper account not created yet → create at ibkr.com
# 3. 2FA issue → make sure your paper account doesn't require 2FA
```

### Bot starts but doesn't trade
```bash
# Check bot logs
sudo journalctl -u optionsbot --no-pager -n 100

# Common reasons:
# 1. Market is closed (weekends/holidays)
# 2. No signals meeting criteria (normal on quiet days)
# 3. Circuit breaker triggered (check dashboard)
# 4. allowed_accounts not set in settings.yaml
```

### How to SSH from your phone
Download **Termius** (free) from the App Store. Add your VPS IP, username `root`, and password. You can check on the bot from anywhere.

---

## Cost Summary

| Item | Monthly Cost |
|------|-------------|
| DigitalOcean Droplet (1 CPU, 1 GB) | $6 |
| IBKR Paper Account | Free |
| IB Gateway | Free |
| Total | **$6/month** |

> If you find the bot needs more RAM later (unlikely), you can resize
> the droplet to $12/mo (2 CPU, 2 GB) with one click in DigitalOcean.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                YOUR VPS ($6/mo)                  │
│                                                  │
│  ┌──────────┐    ┌──────────────┐               │
│  │IB Gateway│◄──►│  Options Bot  │               │
│  │ (paper)  │    │  (main.py)   │               │
│  └──────────┘    └──────┬───────┘               │
│       │                 │                        │
│       │          ┌──────▼───────┐               │
│       │          │   SQLite DB   │               │
│       │          │  trades.db    │               │
│       │          └──────┬───────┘               │
│       │                 │                        │
│       │          ┌──────▼───────┐               │
│       │          │  Dashboard    │◄── port 8501  │
│       │          │  (Streamlit)  │               │
│       │          └──────────────┘               │
└───────┼──────────────────────────────┬──────────┘
        │                              │
        ▼                              ▼
   IBKR Servers                Your Browser/Phone
   (paper trading)             http://VPS_IP:8501
```

```
┌─────────────────────┐
│    YOUR MAC          │
│                      │
│  TWS (Live Trading)  │  ← No conflicts. Completely separate.
│                      │
└──────────────────────┘
```

---

*No conflicts. Your live TWS runs on your Mac. The paper bot runs on the VPS. They never touch each other.*
