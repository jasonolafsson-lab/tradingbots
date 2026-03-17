"""
Level 2: ML Confidence Scorer.

Loads the trained XGBoost model and scores new signals in real-time.
Returns a win probability (0.0-1.0) that drives position sizing.
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional, Dict, Any

import numpy as np

from data.market_state import Signal, TickerState, MarketState

logger = logging.getLogger(__name__)


class MLScorer:
    """Real-time ML confidence scoring for trading signals."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.active = False

        # Sizing multiplier thresholds (per spec Section 17.2)
        self.thresholds = {
            "high_confidence": 0.70,     # > 70% → 1.25x
            "standard": 0.55,            # 55-70% → 1.0x
            "low_confidence": 0.45,      # 45-55% → 0.5x
            # < 45% → 0x (skip)
        }

    def load_model(self, model_path: str, feature_names: list) -> bool:
        """Load a trained model from disk."""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.feature_names = feature_names
            self.active = True
            logger.info(f"ML model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False

    def score(
        self,
        signal: Signal,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[float]:
        """
        Score a signal with the ML model.
        Returns win probability (0.0 to 1.0), or None if model not active.
        """
        if not self.active or self.model is None:
            return None

        try:
            import xgboost as xgb

            features = self._extract_features(signal, ts, market_state)
            if features is None:
                return None

            dmatrix = xgb.DMatrix(
                features.reshape(1, -1),
                feature_names=self.feature_names,
            )
            prob = float(self.model.predict(dmatrix)[0])

            logger.info(
                f"ML score for {signal.ticker} {signal.strategy}: "
                f"win_prob={prob:.3f}"
            )
            return prob

        except Exception as e:
            logger.warning(f"ML scoring error: {e}")
            return None

    def get_sizing_multiplier(self, win_probability: Optional[float]) -> float:
        """
        Convert win probability to a position sizing multiplier.

        > 70% → 1.25x (high confidence)
        55-70% → 1.0x (standard)
        45-55% → 0.5x (low confidence)
        < 45% → 0.0x (skip trade)
        """
        if win_probability is None:
            return 1.0  # No ML score → use base sizing

        if win_probability > self.thresholds["high_confidence"]:
            return 1.25
        elif win_probability > self.thresholds["standard"]:
            return 1.0
        elif win_probability > self.thresholds["low_confidence"]:
            return 0.5
        else:
            return 0.0  # Skip trade

    def _extract_features(
        self,
        signal: Signal,
        ts: TickerState,
        market_state: MarketState,
    ) -> Optional[np.ndarray]:
        """Extract feature vector from current signal context."""
        # This must match the encoding used during training
        # For now, return numeric features only
        # Full implementation needs category encoding matching the trainer
        try:
            features = np.array([
                ts.adx_14,
                ts.rsi_7,
                ts.volume_ratio,
                ts.vwap_slope,
                market_state.spy_session_return,
                ts.uw_iv_percentile,
                ts.uw_gex_nearest_wall_distance,
                signal.strength_score,
                signal.day2_score,
            ], dtype=float)
            return features
        except Exception:
            return None
