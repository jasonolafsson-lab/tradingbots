"""
Tests for Level 2 Intelligence: ML Scorer and Monitor.
(Training tests require sufficient data — just test interfaces.)
"""

import pytest

from intelligence.ml_scorer import MLScorer
from intelligence.ml_monitor import MLMonitor


class TestMLScorer:
    def setup_method(self):
        self.config = {
            "intelligence": {"ml_min_accuracy": 0.52}
        }
        self.scorer = MLScorer(self.config)

    def test_inactive_by_default(self):
        assert self.scorer.active is False
        assert self.scorer.model is None

    def test_score_returns_none_when_inactive(self):
        """Score should return None when model is not loaded."""
        result = self.scorer.score(None, None, None)
        assert result is None

    def test_sizing_multiplier_none_score(self):
        """No ML score → base sizing (1.0x)."""
        assert self.scorer.get_sizing_multiplier(None) == 1.0

    def test_sizing_multiplier_high_confidence(self):
        assert self.scorer.get_sizing_multiplier(0.75) == 1.25

    def test_sizing_multiplier_standard(self):
        assert self.scorer.get_sizing_multiplier(0.60) == 1.0

    def test_sizing_multiplier_low_confidence(self):
        assert self.scorer.get_sizing_multiplier(0.50) == 0.5

    def test_sizing_multiplier_skip(self):
        assert self.scorer.get_sizing_multiplier(0.40) == 0.0

    def test_sizing_multiplier_boundary_high(self):
        # Exactly at high threshold boundary
        assert self.scorer.get_sizing_multiplier(0.70) == 1.0  # Not > 0.70

    def test_sizing_multiplier_boundary_standard(self):
        # 0.55 is NOT > 0.55, falls to low_confidence tier
        assert self.scorer.get_sizing_multiplier(0.55) == 0.5
        # 0.56 IS > 0.55, standard tier
        assert self.scorer.get_sizing_multiplier(0.56) == 1.0

    def test_load_model_nonexistent_file(self):
        result = self.scorer.load_model("/nonexistent/model.pkl", ["f1", "f2"])
        assert result is False
        assert self.scorer.active is False


class TestMLMonitor:
    def setup_method(self):
        self.config = {"intelligence": {"ml_min_accuracy": 0.52}}
        # MLMonitor requires a TradeMemoryDB; pass None since we won't query it
        self.monitor = MLMonitor(self.config, trade_memory=None)

    def test_insufficient_predictions(self):
        """With < 20 predictions, should remain active."""
        for i in range(10):
            self.monitor.record_prediction(f"t{i}", 0.6, "WIN")
        result = self.monitor.check_accuracy()
        assert result["should_remain_active"] is True
        assert result["accuracy"] is None
        assert result["sample_size"] == 10

    def test_high_accuracy_stays_active(self):
        # 18 correct out of 20
        for i in range(18):
            self.monitor.record_prediction(f"t{i}", 0.7, "WIN")
        for i in range(2):
            self.monitor.record_prediction(f"t{18+i}", 0.3, "LOSS")
        result = self.monitor.check_accuracy()
        assert result["should_remain_active"] is True
        assert result["accuracy"] == 1.0  # All predictions correct

    def test_low_accuracy_deactivates(self):
        # Many wrong predictions
        for i in range(15):
            self.monitor.record_prediction(f"t{i}", 0.7, "LOSS")  # Predicted WIN, got LOSS
        for i in range(5):
            self.monitor.record_prediction(f"t{15+i}", 0.3, "WIN")  # Predicted LOSS, got WIN
        result = self.monitor.check_accuracy()
        assert result["should_remain_active"] is False
        assert result["accuracy"] == 0.0
        assert self.monitor.ml_should_be_active is False

    def test_ab_comparison_placeholder(self):
        result = self.monitor.get_ab_comparison()
        assert "ml_pnl" in result
        assert "level1_pnl" in result
