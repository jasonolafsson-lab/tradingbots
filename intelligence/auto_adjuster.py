"""
Level 1: Auto-Adjuster.

Uses rolling statistics to automatically adjust position sizing
per strategy/ticker combination. Does NOT change entry rules or stops —
only sizing and availability.

Rules (per spec Section 17.1):
- Win rate < 40% → reduce to 50% size
- Win rate < 30% → disable combination
- Win rate > 65% → eligible for 125% size
- Profit factor < 0.8 → disable combination
- Overall PF < 1.0 → halt all trading
- Time bucket win rate < 35% → avoid that time bucket
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Set, Tuple

from intelligence.rolling_stats import RollingStats

logger = logging.getLogger(__name__)


class AutoAdjuster:
    """Level 1 auto-adjustment engine."""

    def __init__(self, config: dict, rolling_stats: RollingStats):
        self.config = config
        self.stats = rolling_stats
        self.active = False

        # Disabled combinations: set of (strategy, ticker) tuples
        self.disabled: Set[Tuple[str, str]] = set()

        # Size multipliers: (strategy, ticker) → multiplier
        self.multipliers: Dict[Tuple[str, str], float] = {}

        # Time buckets to avoid: set of (hour, minute_bucket)
        self.blocked_time_buckets: Set[Tuple[int, int]] = set()

    def activate(self) -> None:
        """Activate Level 1 intelligence."""
        self.active = True
        logger.info("Level 1 Intelligence ACTIVATED: Auto-adjuster online")

    def refresh(self) -> None:
        """
        Refresh all adjustments based on latest rolling stats.
        Call daily after market close.
        """
        if not self.active:
            return

        self.disabled.clear()
        self.multipliers.clear()

        # Check overall profit factor
        overall = self.stats.get_stats("overall")
        if overall.get("count", 0) >= 50 and overall.get("profit_factor", 0) < 1.0:
            logger.critical(
                "HALT: Overall profit factor < 1.0 over 50+ trades. "
                "All trading should stop for review."
            )
            # This would trigger a full halt — handled by activation_manager
            return

        # Per strategy/ticker combinations
        for strategy in ["MOMENTUM", "REVERSION", "DAY2",
                         "TUESDAY_REVERSAL", "GREEN_SECTOR"]:
            for ticker in ["SPY", "QQQ", "NVDA", "TSLA"]:
                key = f"{strategy}:{ticker}"
                stats = self.stats.get_stats(key)
                count = stats.get("count", 0)

                if count < 10:
                    continue  # Not enough data

                win_rate = stats.get("win_rate", 0)
                pf = stats.get("profit_factor", 0)

                combo = (strategy, ticker)

                # Disable if win rate < 30% or PF < 0.8
                if win_rate < 0.30 or (count >= 20 and pf < 0.8):
                    self.disabled.add(combo)
                    logger.warning(
                        f"DISABLED: {strategy}/{ticker} "
                        f"(WR={win_rate:.0%}, PF={pf:.2f}, n={count})"
                    )
                    continue

                # Reduce to 50% if win rate < 40%
                if win_rate < 0.40:
                    self.multipliers[combo] = 0.50
                    logger.info(
                        f"REDUCED: {strategy}/{ticker} → 50% size "
                        f"(WR={win_rate:.0%}, n={count})"
                    )

                # Boost to 125% if win rate > 65%
                elif win_rate > 0.65 and count >= 20:
                    self.multipliers[combo] = 1.25
                    logger.info(
                        f"BOOSTED: {strategy}/{ticker} → 125% size "
                        f"(WR={win_rate:.0%}, n={count})"
                    )

    def get_sizing_multiplier(
        self,
        strategy: str,
        ticker: str,
    ) -> float:
        """
        Get the position sizing multiplier for a strategy/ticker combo.
        Returns 0.0 if disabled, otherwise the multiplier (default 1.0).
        """
        if not self.active:
            return 1.0

        combo = (strategy, ticker)

        if combo in self.disabled:
            return 0.0

        return self.multipliers.get(combo, 1.0)

    def is_combo_disabled(self, strategy: str, ticker: str) -> bool:
        """Check if a strategy/ticker combo has been disabled."""
        return (strategy, ticker) in self.disabled
