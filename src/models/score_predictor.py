"""
First Innings Score Predictor (Model 3)
Predicts final first innings score mid-innings.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import preprocess_all
from src.data.player_registry import PlayerRegistry
from src.data.feature_engineering import FeatureEngineer
from src.evaluation.metrics import (
    evaluate_regressor, temporal_train_test_split,
    get_feature_columns
)


class ScorePredictor:
    """First innings score prediction model."""

    def __init__(self):
        self.xgb_model = None
        self.ridge_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def prepare_data(self, matches: pd.DataFrame, balls: pd.DataFrame,
                     registry: PlayerRegistry, sample_rate: int = 6) -> pd.DataFrame:
        """Build the score prediction dataset."""
        fe = FeatureEngineer(matches, balls, registry)
        df = fe.build_score_prediction_dataset(sample_rate=sample_rate)
        return df

    def train(self, df: pd.DataFrame, feature_cols: list[str] = None):
        """Train the score predictor."""
        if feature_cols is None:
            feature_cols = get_feature_columns(df, exclude=["final_score", "batting_team",
                                                             "bowling_team", "phase"])

        self.feature_cols = feature_cols

        df = df.dropna(subset=["final_score"])

        # Temporal split
        train_df, val_df, test_df = temporal_train_test_split(df, season_col="season")

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["final_score"]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["final_score"]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["final_score"]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost
        print("\nTraining XGBoost Regressor...")
        self.xgb_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=30,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Train Ridge baseline
        print("Training Ridge Regression baseline...")
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n--- Validation Set ---")
        xgb_val_pred = self.xgb_model.predict(X_val)
        evaluate_regressor(y_val, xgb_val_pred, "XGBoost (Val)")

        ridge_val_pred = self.ridge_model.predict(X_val_scaled)
        evaluate_regressor(y_val, ridge_val_pred, "Ridge (Val)")

        print("\n--- Test Set ---")
        xgb_test_pred = self.xgb_model.predict(X_test)
        self.test_results = evaluate_regressor(y_test, xgb_test_pred, "XGBoost (Test)")

        # Accuracy within ranges
        for margin in [10, 20, 30]:
            within = np.abs(xgb_test_pred - y_test.values) <= margin
            print(f"  Within {margin} runs: {within.mean():.1%}")

        self.is_trained = True
        return self

    def predict(self, features: dict) -> dict:
        """Predict final score from current match state."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.feature_cols}])
        X = X[self.feature_cols].fillna(0)

        xgb_pred = self.xgb_model.predict(X)[0]
        ridge_pred = self.ridge_model.predict(self.scaler.transform(X))[0]

        ensemble_pred = 0.7 * xgb_pred + 0.3 * ridge_pred

        return {
            "predicted_score": int(round(ensemble_pred)),
            "predicted_score_range": (int(round(ensemble_pred - 15)), int(round(ensemble_pred + 15))),
            "xgb_prediction": int(round(xgb_pred)),
            "ridge_prediction": int(round(ridge_pred)),
            "current_score": features.get("current_score", 0),
            "overs_bowled": features.get("balls_bowled", 0) / 6,
        }

    def save(self, path: str = "outputs/score_predictor.pkl"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "xgb_model": self.xgb_model,
            "ridge_model": self.ridge_model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "outputs/score_predictor.pkl"):
        """Load model from disk."""
        data = joblib.load(path)
        self.xgb_model = data["xgb_model"]
        self.ridge_model = data["ridge_model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"Model loaded from {path}")
        return self
