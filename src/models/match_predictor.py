"""
IPL Match Outcome Predictor (Model 1)
Predicts match winner before the match starts using team composition,
player form, head-to-head, venue stats, and toss.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import preprocess_all
from src.data.player_registry import PlayerRegistry
from src.data.feature_engineering import FeatureEngineer
from src.evaluation.metrics import (
    evaluate_classifier, temporal_train_test_split,
    get_feature_columns, plot_feature_importance
)


class MatchPredictor:
    """Pre-match outcome prediction model."""

    def __init__(self):
        self.xgb_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def prepare_data(self, matches: pd.DataFrame, balls: pd.DataFrame,
                     registry: PlayerRegistry) -> pd.DataFrame:
        """Build the match-level dataset."""
        fe = FeatureEngineer(matches, balls, registry)
        df = fe.build_match_dataset()
        return df

    def train(self, df: pd.DataFrame, feature_cols: list[str] = None):
        """Train the match predictor."""
        if feature_cols is None:
            feature_cols = get_feature_columns(df, exclude=["team1_last10_wins", "team2_last10_wins",
                                                             "h2h_matches", "h2h_team1_wins"])

        self.feature_cols = feature_cols

        # Drop rows with missing target
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

        # Temporal split
        train_df, val_df, test_df = temporal_train_test_split(df, season_col="season")

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["target"]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["target"]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["target"]

        # Scale features for LR
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost
        print("\nTraining XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Train Logistic Regression baseline
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        self.lr_model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n--- Validation Set ---")
        xgb_val_pred = self.xgb_model.predict(X_val)
        xgb_val_prob = self.xgb_model.predict_proba(X_val)[:, 1]
        evaluate_classifier(y_val, xgb_val_pred, xgb_val_prob, "XGBoost (Val)")

        lr_val_pred = self.lr_model.predict(X_val_scaled)
        lr_val_prob = self.lr_model.predict_proba(X_val_scaled)[:, 1]
        evaluate_classifier(y_val, lr_val_pred, lr_val_prob, "Logistic Regression (Val)")

        print("\n--- Test Set ---")
        xgb_test_pred = self.xgb_model.predict(X_test)
        xgb_test_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        self.test_results = evaluate_classifier(y_test, xgb_test_pred, xgb_test_prob, "XGBoost (Test)")

        self.is_trained = True
        print(f"\nFeature importance (top 10):")
        importances = self.xgb_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:10]
        for name, imp in feat_imp:
            print(f"  {name}: {imp:.4f}")

        return self

    def predict(self, features: dict) -> dict:
        """Predict match outcome from feature dict."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.feature_cols}])
        X = X[self.feature_cols].fillna(0)

        xgb_prob = self.xgb_model.predict_proba(X)[:, 1][0]
        lr_prob = self.lr_model.predict_proba(self.scaler.transform(X))[:, 1][0]

        # Ensemble: weighted average
        ensemble_prob = 0.7 * xgb_prob + 0.3 * lr_prob

        return {
            "team1_win_probability": ensemble_prob,
            "team2_win_probability": 1 - ensemble_prob,
            "predicted_winner": features["team1"] if ensemble_prob > 0.5 else features["team2"],
            "confidence": max(ensemble_prob, 1 - ensemble_prob),
            "xgb_probability": xgb_prob,
            "lr_probability": lr_prob,
        }

    def save(self, path: str = "outputs/match_predictor.pkl"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "xgb_model": self.xgb_model,
            "lr_model": self.lr_model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "outputs/match_predictor.pkl"):
        """Load model from disk."""
        data = joblib.load(path)
        self.xgb_model = data["xgb_model"]
        self.lr_model = data["lr_model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"Model loaded from {path}")
        return self
