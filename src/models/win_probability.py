"""
Live Win Probability Estimator (Model 2)
Ball-by-ball win probability during a match.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
    get_feature_columns, plot_calibration
)


class WinProbabilityEstimator:
    """Ball-by-ball win probability model."""

    def __init__(self):
        self.lgbm_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def prepare_data(self, matches: pd.DataFrame, balls: pd.DataFrame,
                     registry: PlayerRegistry, sample_rate: int = 6) -> pd.DataFrame:
        """Build the ball-by-ball dataset."""
        fe = FeatureEngineer(matches, balls, registry)
        df = fe.build_live_dataset(sample_rate=sample_rate)
        return df

    def train(self, df: pd.DataFrame, feature_cols: list[str] = None):
        """Train the win probability model."""
        if feature_cols is None:
            feature_cols = get_feature_columns(df, exclude=[
                "target_won", "batting_team", "bowling_team", "phase"
            ])

        self.feature_cols = feature_cols

        df = df.dropna(subset=["target_won"])
        df["target_won"] = df["target_won"].astype(int)

        # Temporal split
        train_df, val_df, test_df = temporal_train_test_split(df, season_col="season")

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["target_won"]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["target_won"]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["target_won"]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Train LightGBM
        print("\nTraining LightGBM...")
        self.lgbm_model = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        # Train LR baseline
        print("Training Logistic Regression baseline...")
        self.lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr_model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n--- Validation Set ---")
        lgbm_val_pred = self.lgbm_model.predict(X_val)
        lgbm_val_prob = self.lgbm_model.predict_proba(X_val)[:, 1]
        evaluate_classifier(y_val, lgbm_val_pred, lgbm_val_prob, "LightGBM (Val)")

        print("\n--- Test Set ---")
        lgbm_test_pred = self.lgbm_model.predict(X_test)
        lgbm_test_prob = self.lgbm_model.predict_proba(X_test)[:, 1]
        self.test_results = evaluate_classifier(y_test, lgbm_test_pred, lgbm_test_prob, "LightGBM (Test)")

        # Check calibration
        print("\nCalibration check (predicted vs actual):")
        for threshold in [0.3, 0.5, 0.7]:
            mask = (lgbm_val_prob >= threshold - 0.1) & (lgbm_val_prob < threshold + 0.1)
            if mask.sum() > 0:
                actual = y_val[mask].mean()
                print(f"  Predicted ~{threshold:.1f}: Actual = {actual:.3f} (n={mask.sum()})")

        self.is_trained = True
        return self

    def predict(self, features: dict) -> dict:
        """Predict win probability from ball-level features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.feature_cols}])
        X = X[self.feature_cols].fillna(0)

        lgbm_prob = self.lgbm_model.predict_proba(X)[:, 1][0]
        lr_prob = self.lr_model.predict_proba(self.scaler.transform(X))[:, 1][0]

        # Ensemble with calibration
        ensemble_prob = 0.75 * lgbm_prob + 0.25 * lr_prob

        return {
            "batting_team_win_probability": ensemble_prob,
            "bowling_team_win_probability": 1 - ensemble_prob,
            "batting_team": features.get("batting_team", ""),
            "bowling_team": features.get("bowling_team", ""),
        }

    def predict_match_timeline(self, match_id: int, innings: int,
                                matches: pd.DataFrame, balls: pd.DataFrame,
                                registry: PlayerRegistry) -> pd.DataFrame:
        """Generate win probability timeline for a match innings."""
        fe = FeatureEngineer(matches, balls, registry)
        match_balls = balls[balls["match_id"] == match_id]
        inn_balls = match_balls[match_balls["innings"] == innings]

        timeline = []
        for idx, (_, ball) in enumerate(inn_balls.iterrows()):
            if idx % 6 != 0 and idx != len(inn_balls) - 1:
                continue

            feats = fe.build_ball_features(match_id, innings, ball["over"], ball["ball_in_over"])
            if feats:
                pred = self.predict(feats)
                timeline.append({
                    "over": ball["over"],
                    "ball": ball["ball_in_over"],
                    "score": feats["current_score"],
                    "wickets": feats["wickets_lost"],
                    "win_prob": pred["batting_team_win_probability"],
                })

        return pd.DataFrame(timeline)

    def save(self, path: str = "outputs/win_probability.pkl"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "lgbm_model": self.lgbm_model,
            "lr_model": self.lr_model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "outputs/win_probability.pkl"):
        """Load model from disk."""
        data = joblib.load(path)
        self.lgbm_model = data["lgbm_model"]
        self.lr_model = data["lr_model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"Model loaded from {path}")
        return self
