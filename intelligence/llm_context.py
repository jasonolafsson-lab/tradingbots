"""
Level 3: LLM-Assisted Market Context.

Uses an LLM (Claude API or local) to process qualitative market information:
- Pre-market regime assessment (RISK_ON, RISK_OFF, EVENT_DAY)
- Earnings proximity check
- Macro event filtering
- End-of-day trade review

This is the last intelligence level to activate.
All LLM outputs are advisory — they never directly place or modify trades.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMContext:
    """Level 3 LLM market context engine (scaffolded for V2.5)."""

    def __init__(self, config: dict):
        self.config = config
        self.active = False
        self.last_regime: str = "NEUTRAL"
        self.earnings_exclusions: List[str] = []
        self.event_window_active: bool = False

    def activate(self) -> None:
        """Activate Level 3 intelligence."""
        self.active = True
        logger.info(
            "Level 3 Intelligence ACTIVATED: LLM context engine online. "
            "NOTE: This requires an LLM API connection (not implemented in V1)."
        )

    async def get_morning_context(self) -> Dict[str, Any]:
        """
        Run the pre-market LLM analysis.
        In V1, this returns a neutral/default context.
        In V2.5, this will call the LLM API with news, calendar, and market data.
        """
        if not self.active:
            return self._default_context()

        # V2.5 implementation would:
        # 1. Fetch news headlines for watchlist tickers
        # 2. Fetch economic calendar (CPI, FOMC, NFP, etc.)
        # 3. Fetch prior-day market summary
        # 4. Send to LLM with structured prompt
        # 5. Parse response into actionable signals

        logger.info("LLM morning context: returning default (V1 scaffolding)")
        return self._default_context()

    async def check_earnings_proximity(
        self, tickers: List[str]
    ) -> List[str]:
        """
        Check which tickers have earnings within 24 hours.
        Returns list of tickers to exclude from trading.
        """
        if not self.active:
            return []

        # V2.5: Call earnings calendar API + LLM parsing
        logger.info("Earnings proximity check: not implemented in V1")
        return []

    async def get_eod_review(
        self, trades_today: List[dict], market_summary: dict
    ) -> Optional[str]:
        """
        Generate end-of-day trade review.
        Returns a written summary for human review.
        """
        if not self.active:
            return None

        # V2.5: Send today's trades + market conditions to LLM
        # Get back a written analysis of what worked, what didn't
        logger.info("EOD review: not implemented in V1")
        return None

    @staticmethod
    def _default_context() -> Dict[str, Any]:
        """Return a neutral default context."""
        return {
            "regime": "NEUTRAL",
            "event_day": False,
            "risk_level": "NORMAL",
            "excluded_tickers": [],
            "notes": "Default context (LLM not active)",
        }
