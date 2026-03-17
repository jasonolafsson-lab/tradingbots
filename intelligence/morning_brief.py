"""
Pre-Market Intelligence Module — Morning Brief.

Runs at ~8:30 AM ET before market open. Produces a DailyBrief that all bots
read at startup to adjust their behavior.

Data sources:
  1. Finnhub API (free tier) — Market news headlines + economic calendar
  2. IBKR — ES/SPY futures pre-market move
  3. Built-in calendar — Known FOMC/CPI/NFP dates as fallback
  4. VIX level — Risk environment

Output: DailyBrief JSON saved to data/daily_brief.json
  - market_sentiment: BULLISH / BEARISH / NEUTRAL
  - risk_level: LOW / NORMAL / HIGH / EXTREME
  - scheduled_events: [{time, event, impact}]
  - no_trade_windows: [(start, end)] — times to avoid trading
  - overnight_move_pct: float
  - vix_level: float
  - news_headlines: [str]
  - bot_adjustments: {bot1: {}, bot2: {}, bot3: {}}

Usage:
  python -m intelligence.morning_brief              # Run standalone
  python -m intelligence.morning_brief --dry-run    # Preview without saving
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import aiohttp

logger = logging.getLogger("morning_brief")

PROJECT_ROOT = Path(__file__).parent.parent
BRIEF_PATH = PROJECT_ROOT / "data" / "daily_brief.json"


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ScheduledEvent:
    """A scheduled economic event."""
    time: str           # "14:00" ET or "08:30" ET
    event: str          # "FOMC Rate Decision"
    impact: str         # "HIGH", "MEDIUM", "LOW"
    country: str = "US"


@dataclass
class NoTradeWindow:
    """Time window where bots should not enter new trades."""
    start: str          # "13:30"
    end: str            # "14:30"
    reason: str         # "FOMC announcement"


@dataclass
class BotAdjustment:
    """Adjustments for a specific bot."""
    size_multiplier: float = 1.0    # 0.5 = half size, 1.5 = 1.5x size
    enabled: bool = True            # False = sit out today
    notes: str = ""


@dataclass
class DailyBrief:
    """Complete pre-market intelligence brief."""
    date: str = ""
    generated_at: str = ""

    # Market conditions
    market_sentiment: str = "NEUTRAL"     # BULLISH / BEARISH / NEUTRAL
    risk_level: str = "NORMAL"            # LOW / NORMAL / HIGH / EXTREME
    overnight_move_pct: float = 0.0       # ES futures move
    vix_level: float = 0.0
    vix_regime: str = "NORMAL"            # LOW (<15), NORMAL (15-20), ELEVATED (20-25), HIGH (25-30), EXTREME (>30)

    # Events
    scheduled_events: List[Dict] = field(default_factory=list)
    no_trade_windows: List[Dict] = field(default_factory=list)
    is_fomc_day: bool = False
    is_economic_release_day: bool = False

    # News
    news_headlines: List[str] = field(default_factory=list)
    news_sentiment_score: float = 0.0     # -1 (very bearish) to +1 (very bullish)

    # Bot-specific adjustments
    bot1_momentum: Dict = field(default_factory=lambda: {"size_multiplier": 1.0, "enabled": True, "notes": ""})
    bot2_scalper: Dict = field(default_factory=lambda: {"size_multiplier": 1.0, "enabled": True, "notes": ""})
    bot3_reversion: Dict = field(default_factory=lambda: {"size_multiplier": 1.0, "enabled": True, "notes": ""})


# ============================================================
# Built-in Economic Calendar (FOMC, CPI, NFP for 2026)
# ============================================================

# These are the HIGH-impact scheduled events for 2026.
# Updated manually — these dates are known months in advance.
KNOWN_EVENTS_2026 = [
    # FOMC Rate Decisions (2:00 PM ET)
    {"date": "2026-01-28", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-03-18", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-05-06", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-06-17", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-07-29", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-09-16", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-11-04", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},
    {"date": "2026-12-16", "time": "14:00", "event": "FOMC Rate Decision", "impact": "HIGH"},

    # CPI Reports (8:30 AM ET, typically 2nd or 3rd Tuesday/Wednesday)
    {"date": "2026-01-14", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-02-11", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-03-11", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-04-14", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-05-12", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-06-10", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-07-14", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-08-12", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-09-15", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-10-13", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-11-12", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},
    {"date": "2026-12-10", "time": "08:30", "event": "CPI Report", "impact": "HIGH"},

    # Non-Farm Payrolls (8:30 AM ET, first Friday of month)
    {"date": "2026-01-09", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-02-06", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-03-06", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-04-03", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-05-01", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-06-05", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-07-02", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-08-07", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-09-04", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-10-02", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-11-06", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},
    {"date": "2026-12-04", "time": "08:30", "event": "Non-Farm Payrolls", "impact": "HIGH"},

    # PPI Reports (8:30 AM ET)
    {"date": "2026-01-15", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},
    {"date": "2026-02-13", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},
    {"date": "2026-03-12", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},
    {"date": "2026-04-09", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},
    {"date": "2026-05-14", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},
    {"date": "2026-06-11", "time": "08:30", "event": "PPI Report", "impact": "MEDIUM"},

    # GDP Reports (8:30 AM ET)
    {"date": "2026-01-29", "time": "08:30", "event": "GDP (Advance)", "impact": "MEDIUM"},
    {"date": "2026-02-26", "time": "08:30", "event": "GDP (Second Estimate)", "impact": "MEDIUM"},
    {"date": "2026-03-26", "time": "08:30", "event": "GDP (Third Estimate)", "impact": "MEDIUM"},
    {"date": "2026-04-29", "time": "08:30", "event": "GDP (Advance)", "impact": "MEDIUM"},

    # Retail Sales (8:30 AM ET)
    {"date": "2026-01-16", "time": "08:30", "event": "Retail Sales", "impact": "MEDIUM"},
    {"date": "2026-02-14", "time": "08:30", "event": "Retail Sales", "impact": "MEDIUM"},
    {"date": "2026-03-17", "time": "08:30", "event": "Retail Sales", "impact": "MEDIUM"},
]


def get_known_events_for_date(target_date: date) -> List[Dict]:
    """Get known scheduled events for a specific date."""
    date_str = target_date.isoformat()
    return [e for e in KNOWN_EVENTS_2026 if e["date"] == date_str]


# ============================================================
# Finnhub API Client (Free Tier)
# ============================================================

class FinnhubClient:
    """Lightweight Finnhub API client for news + economic calendar."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to Finnhub."""
        if not self.api_key:
            return None

        if self.session is None:
            self.session = aiohttp.ClientSession()

        if params is None:
            params = {}
        params["token"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Finnhub rate limit hit")
                    return None
                else:
                    logger.warning(f"Finnhub {endpoint}: HTTP {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Finnhub request error: {e}")
            return None

    async def get_market_news(self, category: str = "general") -> List[Dict]:
        """
        Get market news headlines.
        Categories: general, forex, crypto, merger
        """
        data = await self._get("news", {"category": category})
        if not data or not isinstance(data, list):
            return []

        # Return top 10 most recent
        headlines = []
        for article in data[:10]:
            headlines.append({
                "headline": article.get("headline", ""),
                "summary": article.get("summary", "")[:200],
                "source": article.get("source", ""),
                "datetime": article.get("datetime", 0),
                "url": article.get("url", ""),
            })
        return headlines

    async def get_economic_calendar(self, from_date: str, to_date: str) -> List[Dict]:
        """
        Get economic calendar events.
        Dates in YYYY-MM-DD format.
        """
        data = await self._get("calendar/economic", {
            "from": from_date,
            "to": to_date,
        })
        if not data or not isinstance(data, dict):
            return []

        events = data.get("economicCalendar", [])
        # Filter to US events with HIGH/MEDIUM impact
        us_events = []
        for e in events:
            if e.get("country", "") == "US" and e.get("impact", "") in ("high", "medium", "3", "2"):
                us_events.append({
                    "event": e.get("event", "Unknown"),
                    "time": e.get("time", ""),
                    "impact": "HIGH" if e.get("impact") in ("high", "3") else "MEDIUM",
                    "actual": e.get("actual"),
                    "estimate": e.get("estimate"),
                    "prev": e.get("prev"),
                })
        return us_events

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


# ============================================================
# News Sentiment Analyzer (Simple keyword-based)
# ============================================================

BULLISH_KEYWORDS = [
    "rally", "surge", "jump", "gain", "rise", "bullish", "record high",
    "beat", "beats expectations", "strong earnings", "upgrade", "buy",
    "recovery", "optimism", "stimulus", "rate cut", "dovish",
    "jobs added", "growth", "expansion", "breakout",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "drop", "fall", "sell-off", "selloff", "bearish",
    "miss", "misses expectations", "weak earnings", "downgrade", "sell",
    "recession", "fear", "tariff", "rate hike", "hawkish",
    "layoffs", "contraction", "decline", "bankruptcy", "default",
    "war", "conflict", "sanctions", "inflation surges",
]


def score_headline_sentiment(headlines: List[Dict]) -> float:
    """
    Score news sentiment from -1 (bearish) to +1 (bullish).
    Simple keyword counting — not ML, but fast and free.
    """
    if not headlines:
        return 0.0

    bull_count = 0
    bear_count = 0

    for article in headlines:
        text = (article.get("headline", "") + " " + article.get("summary", "")).lower()
        for kw in BULLISH_KEYWORDS:
            if kw in text:
                bull_count += 1
        for kw in BEARISH_KEYWORDS:
            if kw in text:
                bear_count += 1

    total = bull_count + bear_count
    if total == 0:
        return 0.0

    # Score: -1 to +1
    return (bull_count - bear_count) / total


# ============================================================
# VIX Regime Classification
# ============================================================

def classify_vix(vix: float) -> str:
    """Classify VIX into regime buckets."""
    if vix <= 0:
        return "UNKNOWN"
    elif vix < 15:
        return "LOW"           # Complacent — low vol, momentum works
    elif vix < 20:
        return "NORMAL"        # Standard conditions
    elif vix < 25:
        return "ELEVATED"      # Caution — reduce size
    elif vix < 30:
        return "HIGH"          # Fear — reduce size significantly
    else:
        return "EXTREME"       # Panic — consider sitting out


# ============================================================
# No-Trade Window Generator
# ============================================================

def generate_no_trade_windows(events: List[Dict]) -> List[Dict]:
    """Generate no-trade windows around high-impact events."""
    windows = []

    for event in events:
        if event.get("impact") != "HIGH":
            continue

        time_str = event.get("time", "")
        if not time_str or ":" not in time_str:
            continue

        try:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            event_time = dtime(hour, minute)
        except (ValueError, IndexError):
            continue

        # 30 min before, 30 min after for most events
        # 60 min before, 60 min after for FOMC
        if "FOMC" in event.get("event", ""):
            before_mins = 60
            after_mins = 60
        else:
            before_mins = 30
            after_mins = 30

        start_dt = datetime.combine(date.today(), event_time) - timedelta(minutes=before_mins)
        end_dt = datetime.combine(date.today(), event_time) + timedelta(minutes=after_mins)

        # Clamp to market hours
        market_open = dtime(9, 30)
        market_close = dtime(16, 0)

        start_t = max(start_dt.time(), market_open)
        end_t = min(end_dt.time(), market_close)

        windows.append({
            "start": start_t.strftime("%H:%M"),
            "end": end_t.strftime("%H:%M"),
            "reason": event.get("event", "Unknown"),
        })

    return windows


# ============================================================
# Bot Adjustment Logic
# ============================================================

def compute_bot_adjustments(
    brief: DailyBrief,
) -> None:
    """Compute per-bot adjustments based on conditions."""

    # === Risk-level based sizing ===
    if brief.risk_level == "EXTREME":
        # All bots sit out
        brief.bot1_momentum = {"size_multiplier": 0.0, "enabled": False, "notes": "VIX extreme — sitting out"}
        brief.bot2_scalper = {"size_multiplier": 0.0, "enabled": False, "notes": "VIX extreme — sitting out"}
        brief.bot3_reversion = {"size_multiplier": 0.0, "enabled": False, "notes": "VIX extreme — sitting out"}
        return

    if brief.risk_level == "HIGH":
        base_mult = 0.5
    elif brief.risk_level == "ELEVATED":
        base_mult = 0.75
    else:
        base_mult = 1.0

    # === Bot 1: Momentum ===
    bot1_mult = base_mult
    bot1_notes = []

    # Momentum loves trending days (big overnight moves, news catalysts)
    if abs(brief.overnight_move_pct) > 0.5:
        bot1_mult *= 1.25
        bot1_notes.append(f"Overnight move {brief.overnight_move_pct:+.1f}% — trending day likely")

    # News-driven days favor momentum
    if abs(brief.news_sentiment_score) > 0.3:
        bot1_mult *= 1.15
        bot1_notes.append(f"Strong news sentiment ({brief.news_sentiment_score:+.2f})")

    # FOMC days: momentum after the announcement, not before
    if brief.is_fomc_day:
        bot1_mult *= 0.75
        bot1_notes.append("FOMC day — reduced pre-announcement, may ramp post")

    brief.bot1_momentum = {
        "size_multiplier": round(min(bot1_mult, 1.5), 2),
        "enabled": True,
        "notes": "; ".join(bot1_notes) if bot1_notes else "Normal conditions",
    }

    # === Bot 2: Scalper (Power Hour) ===
    bot2_mult = base_mult
    bot2_notes = []

    # Scalper benefits from volatility
    if brief.vix_regime in ("ELEVATED", "HIGH"):
        bot2_mult *= 1.1
        bot2_notes.append("Elevated VIX — wider swings favor scalper")

    if brief.is_fomc_day:
        bot2_mult *= 1.2
        bot2_notes.append("FOMC day — Power Hour likely volatile")

    brief.bot2_scalper = {
        "size_multiplier": round(min(bot2_mult, 1.5), 2),
        "enabled": True,
        "notes": "; ".join(bot2_notes) if bot2_notes else "Normal conditions",
    }

    # === Bot 3: Mean Reversion (buy-the-dip) ===
    bot3_mult = base_mult
    bot3_notes = []

    # Mean reversion loves choppy/no-catalyst days
    if not brief.is_economic_release_day and abs(brief.overnight_move_pct) < 0.3:
        bot3_mult *= 1.2
        bot3_notes.append("Low catalyst day — mean reversion favored")

    # Negative overnight move = dip to buy
    if brief.overnight_move_pct < -0.3:
        bot3_mult *= 1.25
        bot3_notes.append(f"Overnight dip {brief.overnight_move_pct:+.1f}% — dip-buying opportunity")

    # Positive sentiment + dip = high conviction dip buy
    if brief.news_sentiment_score > 0.2 and brief.overnight_move_pct < -0.2:
        bot3_mult *= 1.15
        bot3_notes.append("Bullish sentiment + dip = high conviction")

    # Strong trending day (big move + news) = BAD for mean reversion
    if abs(brief.overnight_move_pct) > 1.0:
        bot3_mult *= 0.5
        bot3_notes.append(f"Large overnight move — trend day, reversion risky")

    if brief.is_fomc_day:
        bot3_mult *= 0.6
        bot3_notes.append("FOMC day — don't fade the move")

    brief.bot3_reversion = {
        "size_multiplier": round(min(bot3_mult, 1.5), 2),
        "enabled": True,
        "notes": "; ".join(bot3_notes) if bot3_notes else "Normal conditions",
    }


# ============================================================
# Main Brief Generator
# ============================================================

class MorningBriefGenerator:
    """Generates the daily pre-market intelligence brief."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        finnhub_key = (
            os.environ.get("FINNHUB_API_KEY", "") or
            self.config.get("finnhub", {}).get("api_key", "")
        )
        self.finnhub = FinnhubClient(finnhub_key)

    async def generate(self) -> DailyBrief:
        """Generate the full morning brief."""
        today = date.today()
        brief = DailyBrief(
            date=today.isoformat(),
            generated_at=datetime.now().isoformat(),
        )

        # 1. Economic calendar — built-in first, then Finnhub overlay
        known_events = get_known_events_for_date(today)
        finnhub_events = await self._fetch_finnhub_calendar(today)

        # Merge: use built-in for known events, add any Finnhub extras
        all_events = list(known_events)
        known_event_names = {e["event"] for e in known_events}
        for fe in finnhub_events:
            if fe["event"] not in known_event_names:
                all_events.append(fe)

        brief.scheduled_events = all_events
        brief.is_fomc_day = any("FOMC" in e.get("event", "") for e in all_events)
        brief.is_economic_release_day = len([
            e for e in all_events if e.get("impact") == "HIGH"
        ]) > 0

        # 2. No-trade windows
        brief.no_trade_windows = generate_no_trade_windows(all_events)

        # 3. News headlines + sentiment
        headlines = await self.finnhub.get_market_news()
        brief.news_headlines = [h["headline"] for h in headlines[:8]]
        brief.news_sentiment_score = score_headline_sentiment(headlines)

        # 4. Market sentiment (from news + overnight move)
        brief.market_sentiment = self._compute_sentiment(brief)

        # 5. Risk level (will be updated with VIX when available)
        brief.risk_level = self._compute_risk_level(brief)

        # 6. Bot adjustments
        compute_bot_adjustments(brief)

        return brief

    async def _fetch_finnhub_calendar(self, today: date) -> List[Dict]:
        """Fetch economic calendar from Finnhub."""
        try:
            events = await self.finnhub.get_economic_calendar(
                today.isoformat(), today.isoformat()
            )
            return events
        except Exception as e:
            logger.warning(f"Finnhub calendar fetch failed: {e}")
            return []

    def _compute_sentiment(self, brief: DailyBrief) -> str:
        """Compute overall market sentiment."""
        score = brief.news_sentiment_score

        # Factor in overnight move
        if brief.overnight_move_pct > 0.3:
            score += 0.2
        elif brief.overnight_move_pct < -0.3:
            score -= 0.2

        if score > 0.15:
            return "BULLISH"
        elif score < -0.15:
            return "BEARISH"
        return "NEUTRAL"

    def _compute_risk_level(self, brief: DailyBrief) -> str:
        """Compute risk level from VIX + events."""
        # VIX-based
        if brief.vix_regime == "EXTREME":
            return "EXTREME"
        elif brief.vix_regime == "HIGH":
            return "HIGH"
        elif brief.vix_regime == "ELEVATED":
            return "ELEVATED" if brief.is_fomc_day else "NORMAL"

        # Event-based escalation
        if brief.is_fomc_day:
            return "ELEVATED"

        return "NORMAL"

    async def close(self):
        await self.finnhub.close()


# ============================================================
# Brief Reader (for bots to load)
# ============================================================

def load_daily_brief() -> Optional[DailyBrief]:
    """
    Load today's daily brief from JSON file.
    Returns None if no brief exists for today.
    Called by each bot at startup.
    """
    if not BRIEF_PATH.exists():
        logger.info("No daily brief found — running with defaults")
        return None

    try:
        with open(BRIEF_PATH) as f:
            data = json.load(f)

        # Check it's for today
        if data.get("date") != date.today().isoformat():
            logger.info(f"Daily brief is from {data.get('date')}, not today — ignoring")
            return None

        brief = DailyBrief(**{k: v for k, v in data.items()
                              if k in DailyBrief.__dataclass_fields__})
        logger.info(
            f"Loaded daily brief: sentiment={brief.market_sentiment}, "
            f"risk={brief.risk_level}, events={len(brief.scheduled_events)}"
        )
        return brief

    except Exception as e:
        logger.error(f"Error loading daily brief: {e}")
        return None


def get_no_trade_windows() -> List[Tuple[dtime, dtime]]:
    """
    Get today's no-trade windows as (start_time, end_time) tuples.
    Convenience function for bots to call.
    """
    brief = load_daily_brief()
    if brief is None:
        return []

    windows = []
    for w in brief.no_trade_windows:
        try:
            start_parts = w["start"].split(":")
            end_parts = w["end"].split(":")
            start = dtime(int(start_parts[0]), int(start_parts[1]))
            end = dtime(int(end_parts[0]), int(end_parts[1]))
            windows.append((start, end))
        except (ValueError, KeyError):
            continue

    return windows


def get_bot_size_multiplier(bot_name: str) -> float:
    """
    Get position size multiplier for a specific bot.
    Returns 1.0 if no brief or bot not found.
    """
    brief = load_daily_brief()
    if brief is None:
        return 1.0

    bot_map = {
        "bot1": brief.bot1_momentum,
        "momentum": brief.bot1_momentum,
        "bot2": brief.bot2_scalper,
        "scalper": brief.bot2_scalper,
        "bot3": brief.bot3_reversion,
        "reversion": brief.bot3_reversion,
        "mean_reversion": brief.bot3_reversion,
    }

    adj = bot_map.get(bot_name.lower(), {})
    if not adj.get("enabled", True):
        return 0.0

    return adj.get("size_multiplier", 1.0)


def is_in_no_trade_window(current_time: dtime = None) -> bool:
    """Check if current time falls within a no-trade window."""
    if current_time is None:
        current_time = datetime.now().time()

    for start, end in get_no_trade_windows():
        if start <= current_time <= end:
            return True
    return False


# ============================================================
# Save Brief
# ============================================================

def save_brief(brief: DailyBrief) -> None:
    """Save brief to JSON file."""
    BRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIEF_PATH, "w") as f:
        json.dump(asdict(brief), f, indent=2, default=str)
    logger.info(f"Daily brief saved to {BRIEF_PATH}")


# ============================================================
# Main (standalone runner)
# ============================================================

async def run_morning_brief(dry_run: bool = False) -> DailyBrief:
    """Generate and optionally save the morning brief."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    generator = MorningBriefGenerator()

    try:
        brief = await generator.generate()

        # Print summary
        print("\n" + "=" * 60)
        print(f"  MORNING BRIEF — {brief.date}")
        print("=" * 60)
        print(f"  Generated: {brief.generated_at}")
        print(f"  Sentiment: {brief.market_sentiment}")
        print(f"  Risk Level: {brief.risk_level}")
        print(f"  Overnight Move: {brief.overnight_move_pct:+.2f}%")
        print(f"  VIX: {brief.vix_level:.1f} ({brief.vix_regime})")
        print(f"  News Sentiment: {brief.news_sentiment_score:+.2f}")

        if brief.scheduled_events:
            print(f"\n  Scheduled Events:")
            for e in brief.scheduled_events:
                print(f"    [{e.get('impact', '?')}] {e.get('time', '?')} — {e.get('event', '?')}")

        if brief.no_trade_windows:
            print(f"\n  No-Trade Windows:")
            for w in brief.no_trade_windows:
                print(f"    {w['start']} - {w['end']} ({w['reason']})")

        if brief.news_headlines:
            print(f"\n  Top Headlines:")
            for h in brief.news_headlines[:5]:
                print(f"    - {h[:80]}")

        print(f"\n  Bot Adjustments:")
        print(f"    Bot 1 (Momentum):  size={brief.bot1_momentum['size_multiplier']}x "
              f"{'ENABLED' if brief.bot1_momentum['enabled'] else 'DISABLED'}")
        if brief.bot1_momentum.get("notes"):
            print(f"      {brief.bot1_momentum['notes']}")

        print(f"    Bot 2 (Scalper):   size={brief.bot2_scalper['size_multiplier']}x "
              f"{'ENABLED' if brief.bot2_scalper['enabled'] else 'DISABLED'}")
        if brief.bot2_scalper.get("notes"):
            print(f"      {brief.bot2_scalper['notes']}")

        print(f"    Bot 3 (Reversion): size={brief.bot3_reversion['size_multiplier']}x "
              f"{'ENABLED' if brief.bot3_reversion['enabled'] else 'DISABLED'}")
        if brief.bot3_reversion.get("notes"):
            print(f"      {brief.bot3_reversion['notes']}")

        print("=" * 60)

        if not dry_run:
            save_brief(brief)
        else:
            print("\n  [DRY RUN — brief not saved]")

        return brief

    finally:
        await generator.close()


def main():
    parser = argparse.ArgumentParser(description="Generate pre-market morning brief")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    asyncio.run(run_morning_brief(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
