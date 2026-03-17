"""
Intelligence Activation Manager.

Controls which intelligence level is active based on trade-count triggers
and graduation criteria. Prevents premature advancement.

Level 1 (200 trades): Statistical Self-Awareness
Level 2 (500 trades + PF > 1.2): ML Confidence Scoring
Level 3 (Level 2 active 200+ trades): LLM Market Context
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from intelligence.trade_memory import TradeMemoryDB
from intelligence.rolling_stats import RollingStats
from intelligence.auto_adjuster import AutoAdjuster
from intelligence.ml_trainer import MLTrainer
from intelligence.ml_scorer import MLScorer
from intelligence.ml_monitor import MLMonitor
from intelligence.llm_context import LLMContext

logger = logging.getLogger(__name__)


class ActivationManager:
    """Manages intelligence level activation based on trade milestones."""

    def __init__(self, config: dict, trade_memory: TradeMemoryDB):
        self.config = config
        self.db = trade_memory

        intel = config.get("intelligence", {})
        self.level1_threshold = intel.get("level1_trade_count", 200)
        self.level2_threshold = intel.get("level2_trade_count", 500)
        self.level2_min_pf = intel.get("level2_min_profit_factor", 1.2)
        self.level2_min_l1_trades = intel.get("level2_min_level1_trades", 100)
        self.level3_min_l2_trades = intel.get("level3_min_level2_trades", 200)

        # Intelligence components
        self.rolling_stats = RollingStats(trade_memory)
        self.auto_adjuster = AutoAdjuster(config, self.rolling_stats)
        self.ml_trainer = MLTrainer(config, trade_memory)
        self.ml_scorer = MLScorer(config)
        self.ml_monitor = MLMonitor(config, trade_memory)
        self.llm_context = LLMContext(config)

        # Tracking
        self.current_level = 0
        self.level1_activation_trade = 0
        self.level2_activation_trade = 0
        self.trades_since_level2 = 0

    def check_and_activate(self) -> Dict[str, Any]:
        """
        Check trade count and graduation criteria.
        Activate the next intelligence level if conditions are met.
        Returns current status.
        """
        total_trades = self.db.get_total_trades()
        overall_stats = self.db.get_overall_stats()
        pf = overall_stats.get("profit_factor", 0)

        status = {
            "total_trades": total_trades,
            "current_level": self.current_level,
            "profit_factor": pf,
        }

        # Level 1 activation check
        if self.current_level < 1 and total_trades >= self.level1_threshold:
            if self._check_level1_criteria(overall_stats):
                self._activate_level1()
                status["current_level"] = 1
                status["event"] = "Level 1 activated"

        # Level 2 activation check
        if self.current_level == 1:
            trades_since_l1 = total_trades - self.level1_activation_trade
            if (total_trades >= self.level2_threshold and
                    trades_since_l1 >= self.level2_min_l1_trades and
                    pf >= self.level2_min_pf):
                if self._check_level2_criteria(overall_stats):
                    self._activate_level2()
                    status["current_level"] = 2
                    status["event"] = "Level 2 activated"

        # Level 3 activation check
        if self.current_level == 2:
            self.trades_since_level2 = total_trades - self.level2_activation_trade
            if self.trades_since_level2 >= self.level3_min_l2_trades:
                ab = self.ml_monitor.get_ab_comparison()
                if ab.get("ml_advantage", 0) > 0:
                    self._activate_level3()
                    status["current_level"] = 3
                    status["event"] = "Level 3 activated"

        return status

    def _check_level1_criteria(self, stats: Dict[str, Any]) -> bool:
        """Check Level 1 graduation criteria."""
        # All 35 fields populated (assumed if we got here)
        # No systematic execution issues (would need separate tracking)
        # Overall PF > 1.0
        return stats.get("profit_factor", 0) > 1.0

    def _check_level2_criteria(self, stats: Dict[str, Any]) -> bool:
        """Check Level 2 graduation criteria."""
        # No strategy fully disabled
        if self.auto_adjuster.disabled:
            logger.info(
                "Level 2 blocked: some strategy/ticker combos disabled. "
                f"Disabled: {self.auto_adjuster.disabled}"
            )
            return False
        return True

    def _activate_level1(self) -> None:
        """Activate Level 1: Rolling stats + auto-adjuster."""
        self.current_level = 1
        self.level1_activation_trade = self.db.get_total_trades()
        self.rolling_stats.refresh()
        self.auto_adjuster.activate()
        logger.info(
            f"LEVEL 1 ACTIVATED at trade #{self.level1_activation_trade}. "
            f"Rolling stats and auto-adjustment online."
        )

    def _activate_level2(self) -> None:
        """Activate Level 2: ML confidence scoring."""
        self.current_level = 2
        self.level2_activation_trade = self.db.get_total_trades()

        # Train initial model
        meta = self.ml_trainer.train()
        if meta:
            model_path = self.ml_trainer.get_latest_model_path()
            if model_path:
                self.ml_scorer.load_model(
                    model_path, meta.get("feature_names", [])
                )

        logger.info(
            f"LEVEL 2 ACTIVATED at trade #{self.level2_activation_trade}. "
            f"ML confidence scoring online."
        )

    def _activate_level3(self) -> None:
        """Activate Level 3: LLM market context."""
        self.current_level = 3
        self.llm_context.activate()
        logger.info(
            f"LEVEL 3 ACTIVATED. LLM market context online."
        )

    def get_sizing_multiplier(
        self, strategy: str, ticker: str, signal=None, ts=None, market_state=None
    ) -> float:
        """
        Get the combined sizing multiplier from all active intelligence levels.
        """
        multiplier = 1.0

        # Level 1: Auto-adjuster
        if self.current_level >= 1:
            l1_mult = self.auto_adjuster.get_sizing_multiplier(strategy, ticker)
            if l1_mult == 0:
                return 0.0  # Disabled combo
            multiplier *= l1_mult

        # Level 2: ML confidence
        if self.current_level >= 2 and signal and ts and market_state:
            prob = self.ml_scorer.score(signal, ts, market_state)
            l2_mult = self.ml_scorer.get_sizing_multiplier(prob)
            if l2_mult == 0:
                return 0.0  # ML says skip
            multiplier *= l2_mult

        # Cap at 1.25x max
        return min(multiplier, 1.25)

    def daily_refresh(self) -> None:
        """Run daily post-market intelligence refresh."""
        if self.current_level >= 1:
            self.rolling_stats.refresh()
            self.auto_adjuster.refresh()
            logger.info("Level 1: daily stats refreshed")

        if self.current_level >= 2:
            # Check if weekly retrain is needed
            retrain_day = self.config.get("intelligence", {}).get(
                "ml_retrain_day", "Sunday"
            )
            from datetime import datetime
            if datetime.now().strftime("%A") == retrain_day:
                meta = self.ml_trainer.train()
                if meta:
                    model_path = self.ml_trainer.get_latest_model_path()
                    if model_path:
                        self.ml_scorer.load_model(
                            model_path, meta.get("feature_names", [])
                        )
                logger.info("Level 2: weekly model retrained")

            # Check model health
            health = self.ml_monitor.check_accuracy()
            if not health.get("should_remain_active", True):
                self.ml_scorer.active = False
                logger.warning("Level 2: ML scorer disabled due to low accuracy")

        self.check_and_activate()
