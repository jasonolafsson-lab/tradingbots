"""
Level 2: ML Model Trainer.

XGBoost gradient-boosted trees trained on trade memory database.
Binary classification: WIN (1) or LOSS (0).
20 input features from the 35-field trade record.
Retrains weekly (Sunday).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from intelligence.trade_memory import TradeMemoryDB

logger = logging.getLogger(__name__)

# Feature columns (20 features per spec Section 17.2)
FEATURE_COLUMNS = [
    "strategy", "ticker", "direction", "contract_type",
    "day_of_week", "minutes_since_open",
    "dte", "delta_at_entry", "iv_percentile",
    "spy_session_return", "underlying_vs_vwap", "vwap_slope",
    "adx_value", "rsi_value", "volume_ratio",
    "sector_relative_strength", "uw_net_premium_direction",
    "gex_nearest_wall_distance", "day2_score",
    "signal_strength_score",
]

CATEGORICAL_COLUMNS = [
    "strategy", "ticker", "direction", "contract_type",
    "day_of_week", "uw_net_premium_direction",
]


class MLTrainer:
    """XGBoost model training pipeline for Level 2 intelligence."""

    def __init__(self, config: dict, trade_memory: TradeMemoryDB):
        self.config = config
        self.db = trade_memory
        self.models_dir = "models"
        self.min_training_size = config.get("intelligence", {}).get(
            "level2_trade_count", 500
        )
        os.makedirs(self.models_dir, exist_ok=True)

    def train(self) -> Optional[Dict[str, Any]]:
        """
        Train a new XGBoost model on all available trade data.
        Returns model performance metrics, or None if training failed.
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, brier_score_loss
        except ImportError:
            logger.error("xgboost or sklearn not installed")
            return None

        # Load data
        trades = self.db.get_trades()
        if len(trades) < self.min_training_size:
            logger.info(
                f"Not enough trades for ML training: "
                f"{len(trades)} < {self.min_training_size}"
            )
            return None

        df = pd.DataFrame(trades)

        # Prepare features and target
        X, feature_names = self._prepare_features(df)
        y = (df["outcome"] == "WIN").astype(int).values

        if X is None or len(X) == 0:
            return None

        # 80/20 split (most recent trades as validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        params = {
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "seed": 42,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dval, "validation")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        # Evaluate
        y_pred_prob = model.predict(dval)
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_pred_prob)

        # Feature importance
        importance = model.get_score(importance_type="weight")
        top_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = os.path.join(self.models_dir, f"xgb_v{version}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        meta = {
            "version": version,
            "training_size": len(X_train),
            "validation_size": len(X_val),
            "accuracy": accuracy,
            "brier_score": brier,
            "top_features": top_features,
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = os.path.join(self.models_dir, f"xgb_v{version}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(
            f"ML Model trained: v{version} | "
            f"Accuracy={accuracy:.3f} | Brier={brier:.3f} | "
            f"Train={len(X_train)} Val={len(X_val)}"
        )

        return meta

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Prepare feature matrix from trade DataFrame."""
        available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        if len(available_cols) < 10:
            logger.warning(f"Only {len(available_cols)} features available")
            return None, []

        X_df = df[available_cols].copy()

        # Encode categoricals
        feature_names = []
        arrays = []

        for col in available_cols:
            if col in CATEGORICAL_COLUMNS:
                dummies = pd.get_dummies(X_df[col], prefix=col)
                arrays.append(dummies.values)
                feature_names.extend(dummies.columns.tolist())
            else:
                vals = pd.to_numeric(X_df[col], errors="coerce").fillna(0).values
                arrays.append(vals.reshape(-1, 1))
                feature_names.append(col)

        X = np.hstack(arrays)
        return X, feature_names

    def get_latest_model_path(self) -> Optional[str]:
        """Get the path to the most recently trained model."""
        if not os.path.exists(self.models_dir):
            return None

        model_files = [
            f for f in os.listdir(self.models_dir)
            if f.startswith("xgb_v") and f.endswith(".pkl")
        ]

        if not model_files:
            return None

        return os.path.join(self.models_dir, sorted(model_files)[-1])
