#!/usr/bin/env bash
# =============================================================================
# Options Bot — VPS Setup Script
# =============================================================================
# Run this ONCE on a fresh Ubuntu 24.04 DigitalOcean droplet.
# Usage:  sudo bash setup_vps.sh
# =============================================================================
set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

# ── Must be root ──
if [[ $EUID -ne 0 ]]; then
    fail "Run this script as root:  sudo bash setup_vps.sh"
fi

echo ""
echo "=============================================="
echo "  OPTIONS BOT — VPS SETUP"
echo "=============================================="
echo ""

# ── Step 1: Collect IBKR credentials ──
info "I need your IBKR paper trading credentials."
info "(These are stored ONLY on this server in a locked-down file.)"
echo ""
read -p "  IBKR Paper Username: " IBKR_USER
read -sp "  IBKR Paper Password: " IBKR_PASS
echo ""
read -p "  IBKR Paper Account ID (e.g. DU1234567): " IBKR_ACCT
echo ""

if [[ -z "$IBKR_USER" || -z "$IBKR_PASS" ]]; then
    fail "Username and password are required."
fi

ok "Credentials captured."
echo ""

# ── Step 2: System updates + core packages ──
info "Updating system and installing dependencies (this takes 2-3 min)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq

# Core packages — two attempts for different Ubuntu package names
apt-get install -y -qq \
  software-properties-common \
  python3.12 python3.12-venv python3.12-dev \
  unzip wget curl git sqlite3 htop tmux \
  ufw \
  xvfb libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
  libgtk-3-0 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libdrm2 libgbm1 libnss3 libxcomposite1 libxdamage1 \
  libxrandr2 libpango-1.0-0 libcairo2 \
  openjdk-17-jre-headless 2>/dev/null || \
apt-get install -y -qq \
  python3.12 python3.12-venv python3.12-dev \
  unzip wget curl git sqlite3 htop tmux \
  ufw xvfb libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
  openjdk-17-jre-headless

ok "System packages installed."

# ── Step 3: Create service user ──
BOT_USER="optionsbot"
BOT_HOME="/home/${BOT_USER}"

info "Creating '${BOT_USER}' user..."
if id "$BOT_USER" &>/dev/null; then
    warn "User '${BOT_USER}' already exists, skipping."
else
    useradd -m -s /bin/bash "$BOT_USER"
    ok "User '${BOT_USER}' created."
fi

# ── Step 4: Create directory structure ──
info "Setting up directories..."
BOT_DIR="${BOT_HOME}/options_bot"
mkdir -p "${BOT_DIR}"
mkdir -p "${BOT_DIR}/config"
mkdir -p "${BOT_DIR}/data/backups"
mkdir -p "${BOT_DIR}/logs"
mkdir -p "${BOT_HOME}/ibc"
mkdir -p "${BOT_HOME}/ibgateway"
mkdir -p /var/log/optbot
chown "${BOT_USER}:${BOT_USER}" /var/log/optbot
ok "Directories created."

# ── Step 5: Copy bot code ──
info "Copying bot code..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_SOURCE="$(dirname "$SCRIPT_DIR")"

rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='*.pyc' --exclude='deploy/' \
  "${BOT_SOURCE}/" "${BOT_DIR}/"

# Inject the paper account ID into settings.yaml
if [[ -n "$IBKR_ACCT" ]]; then
    sed -i "s/allowed_accounts: \[\]/allowed_accounts: [\"${IBKR_ACCT}\"]/" "${BOT_DIR}/config/settings.yaml"
    ok "Paper account ID (${IBKR_ACCT}) written to settings.yaml."
fi

# ── Step 6: Python virtual environment ──
info "Creating Python venv and installing packages..."
python3.12 -m venv "${BOT_DIR}/.venv"
"${BOT_DIR}/.venv/bin/pip" install --upgrade pip -q
"${BOT_DIR}/.venv/bin/pip" install -r "${BOT_DIR}/requirements.txt" -q
ok "Python dependencies installed."

# ── Step 7: Download IB Gateway ──
info "Downloading IB Gateway..."
IB_GATEWAY_URL="https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh"
wget -q -O /tmp/ibgateway-install.sh "$IB_GATEWAY_URL" || {
    warn "IB Gateway download failed — you may need to install manually."
    warn "See: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php"
}

if [[ -f /tmp/ibgateway-install.sh ]]; then
    chmod +x /tmp/ibgateway-install.sh
    info "Installing IB Gateway (silent, ~60 seconds)..."
    /tmp/ibgateway-install.sh -q -dir "${BOT_HOME}/ibgateway" 2>/dev/null || \
        warn "IB Gateway installer had warnings (may still be OK)"
    rm -f /tmp/ibgateway-install.sh
    ok "IB Gateway installed."
fi

# ── Step 8: Download IBC ──
info "Downloading IBC (headless controller)..."
IBC_VERSION="3.19.0"
IBC_URL="https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
wget -q -O /tmp/ibc.zip "$IBC_URL" || {
    warn "IBC ${IBC_VERSION} failed, trying 3.18.0..."
    IBC_VERSION="3.18.0"
    IBC_URL="https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
    wget -q -O /tmp/ibc.zip "$IBC_URL" || warn "IBC download failed — install manually from https://github.com/IbcAlpha/IBC/releases"
}

if [[ -f /tmp/ibc.zip ]]; then
    unzip -q -o /tmp/ibc.zip -d "${BOT_HOME}/ibc/"
    chmod +x "${BOT_HOME}/ibc/"*.sh 2>/dev/null || true
    chmod +x "${BOT_HOME}/ibc/scripts/"*.sh 2>/dev/null || true
    rm -f /tmp/ibc.zip
    ok "IBC ${IBC_VERSION} installed."
fi

# ── Step 9: Write IBC config with credentials ──
info "Writing IBC config..."
cat > "${BOT_HOME}/ibc/config.ini" << IBCEOF
# IBC Configuration — Paper Trading
FIX=no
IbLoginId=${IBKR_USER}
IbPassword=${IBKR_PASS}
TradingMode=paper
AcceptNonBrokerageAccountWarning=yes
AcceptIncomingConnectionAction=accept
DismissNSEComplianceNotice=yes
DismissPasswordExpiryWarning=yes
ExistingSessionDetectedAction=primaryoverride
OverrideTwsApiPort=7497
ReadOnlyLogin=no
AllowBlindTrading=yes
ReloginAfterSecondFactorAuthenticationTimeout=yes
SecondFactorAuthenticationExitInterval=60
IBCEOF
chmod 600 "${BOT_HOME}/ibc/config.ini"
ok "IBC config written (credentials locked down, mode 600)."

# ── Step 10: Create environment file ──
info "Creating environment file..."
cat > "${BOT_HOME}/.env" << ENVEOF
# Options Bot Environment
IB_USER=${IBKR_USER}
IB_PASS=${IBKR_PASS}
DISPLAY=:1
UW_API_TOKEN=
DASHBOARD_PORT=8501
ENVEOF
chmod 600 "${BOT_HOME}/.env"
ok "Environment file created."

# ── Step 11: Create gateway launch script ──
info "Creating IB Gateway launch script..."
cat > "${BOT_HOME}/start_gateway.sh" << 'GWEOF'
#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:1
BOT_HOME="/home/optionsbot"

# Start Xvfb if not running
if ! pgrep -x Xvfb > /dev/null; then
    Xvfb :1 -screen 0 1024x768x24 &
    sleep 2
fi

IBC_DIR="${BOT_HOME}/ibc"
GW_DIR="${BOT_HOME}/ibgateway"

# Source credentials
source "${BOT_HOME}/.env"

# Try different IBC script locations
if [[ -f "${IBC_DIR}/scripts/ibcstart.sh" ]]; then
    exec "${IBC_DIR}/scripts/ibcstart.sh" \
        "${GW_DIR}" \
        "${IBC_DIR}/config.ini" \
        paper \
        "${IB_USER}" \
        "${IB_PASS}"
elif [[ -f "${IBC_DIR}/gatewaystart.sh" ]]; then
    exec "${IBC_DIR}/gatewaystart.sh" \
        --gateway \
        --mode=paper \
        --user="${IB_USER}" \
        --pw="${IB_PASS}" \
        --ibc-path="${IBC_DIR}" \
        --ibc-ini="${IBC_DIR}/config.ini" \
        --gw-path="${GW_DIR}" \
        --on2fatimeout=restart
else
    echo "[ERROR] Cannot find IBC start script"
    ls -la "${IBC_DIR}/"
    exit 1
fi
GWEOF
chmod +x "${BOT_HOME}/start_gateway.sh"
ok "Gateway launch script created."

# ── Step 12: Create systemd services ──
info "Installing systemd services..."

# Xvfb (virtual display)
cat > /etc/systemd/system/xvfb.service << EOF
[Unit]
Description=Virtual Framebuffer (headless display)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :1 -screen 0 1024x768x24
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# IB Gateway
cat > /etc/systemd/system/ibgateway.service << EOF
[Unit]
Description=IB Gateway (Paper Trading via IBC)
After=network-online.target xvfb.service
Wants=network-online.target
Requires=xvfb.service

[Service]
Type=simple
User=${BOT_USER}
EnvironmentFile=${BOT_HOME}/.env
ExecStart=/bin/bash ${BOT_HOME}/start_gateway.sh
Restart=on-failure
RestartSec=30
TimeoutStartSec=120
StandardOutput=append:/var/log/optbot/ibgateway.log
StandardError=append:/var/log/optbot/ibgateway.log

[Install]
WantedBy=multi-user.target
EOF

# Options Bot
cat > /etc/systemd/system/optionsbot.service << EOF
[Unit]
Description=Options Trading Bot
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=${BOT_USER}
WorkingDirectory=${BOT_DIR}
EnvironmentFile=${BOT_HOME}/.env
ExecStart=${BOT_DIR}/.venv/bin/python main.py
Restart=on-failure
RestartSec=60
StandardOutput=append:/var/log/optbot/bot.log
StandardError=append:/var/log/optbot/bot.log

[Install]
WantedBy=multi-user.target
EOF

# Dashboard (always on)
cat > /etc/systemd/system/optionsbot-dashboard.service << EOF
[Unit]
Description=Options Bot Streamlit Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${BOT_USER}
WorkingDirectory=${BOT_DIR}
ExecStart=${BOT_DIR}/.venv/bin/streamlit run dashboard.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
Restart=always
RestartSec=10
StandardOutput=append:/var/log/optbot/dashboard.log
StandardError=append:/var/log/optbot/dashboard.log

[Install]
WantedBy=multi-user.target
EOF

# Daily timer (9:00 AM ET, Mon-Fri)
cat > /etc/systemd/system/optionsbot-start.service << EOF
[Unit]
Description=Start Options Bot (triggered by timer)

[Service]
Type=oneshot
ExecStart=/bin/systemctl start optionsbot.service
EOF

cat > /etc/systemd/system/optionsbot-start.timer << EOF
[Unit]
Description=Start Options Bot Mon-Fri at 9:00 AM ET

[Timer]
OnCalendar=Mon..Fri 09:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
EOF

ok "Systemd services installed."

# ── Step 13: Set ownership ──
info "Setting file ownership..."
chown -R "${BOT_USER}:${BOT_USER}" "${BOT_HOME}"

# ── Step 14: Configure firewall ──
info "Configuring firewall..."
ufw --force reset >/dev/null 2>&1
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp comment "SSH" >/dev/null
ufw allow 8501/tcp comment "Dashboard" >/dev/null
ufw --force enable >/dev/null
ok "Firewall: SSH (22) + Dashboard (8501) open."

# ── Step 15: Set timezone ──
info "Setting timezone to US/Eastern..."
timedatectl set-timezone America/New_York
ok "Timezone: $(date +%Z)"

# ── Step 16: Enable and start services ──
info "Starting services..."
systemctl daemon-reload
systemctl enable --now xvfb.service
sleep 2

systemctl enable ibgateway.service
systemctl start ibgateway.service
sleep 5

systemctl enable --now optionsbot-dashboard.service
systemctl enable --now optionsbot-start.timer

# ── Step 17: Verify everything ──
echo ""
echo "=============================================="
echo "  CHECKING SERVICES"
echo "=============================================="
echo ""

check() {
    if systemctl is-active --quiet "$1" 2>/dev/null; then
        ok "$1"
    else
        warn "$1 — not running. Check: journalctl -u $1 -n 30"
    fi
}

check xvfb.service
check ibgateway.service
check optionsbot-dashboard.service

if systemctl is-active --quiet optionsbot-start.timer 2>/dev/null; then
    ok "optionsbot-start.timer (bot auto-starts Mon-Fri 9 AM ET)"
else
    warn "optionsbot-start.timer not active"
fi

# ── Done ──
VPS_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo "=============================================="
echo -e "  ${GREEN}SETUP COMPLETE!${NC}"
echo "=============================================="
echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │  Dashboard: http://${VPS_IP}:8501        "
echo "  └─────────────────────────────────────────┘"
echo ""
echo "  What's running:"
echo "    ✓ Xvfb         — virtual display (headless)"
echo "    ✓ IB Gateway    — auto-logged in, paper mode"
echo "    ✓ Dashboard     — always on, check from any browser"
echo "    ✓ Daily timer   — bot starts Mon-Fri 9:00 AM ET"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl start optionsbot     # Start bot now"
echo "    sudo systemctl stop optionsbot      # Stop bot"
echo "    sudo systemctl status optionsbot    # Check status"
echo "    sudo journalctl -u ibgateway -f     # IB Gateway logs"
echo "    sudo journalctl -u optionsbot -f    # Bot logs"
echo "    tail -f /var/log/optbot/*.log       # All logs"
echo ""
echo "  Open your browser to: http://${VPS_IP}:8501"
echo ""
