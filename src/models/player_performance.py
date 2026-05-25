"""
Player Performance Predictor (Model 4)
Predicts individual player performance (runs, wickets, economy).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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


class PlayerPerformancePredictor:
    """Player performance prediction model."""

    def __init__(self):
        self.bat_model = None
        self.bowl_model = None
        self.scaler = StandardScaler()
        self.bat_feature_cols = None
        self.bowl_feature_cols = None
        self.is_trained = False

    def prepare_data(self, matches: pd.DataFrame, balls: pd.DataFrame,
                     registry: PlayerRegistry) -> pd.DataFrame:
        """Build the player performance dataset."""
        fe = FeatureEngineer(matches, balls, registry)
        df = fe.build_player_performance_dataset()
        return df

    def train(self, df: pd.DataFrame):
        """Train the player performance models."""
        # Get feature columns (numeric, excluding targets and identifiers)
        exclude = ["actual_runs", "actual_sr", "actual_wickets", "actual_economy",
                    "match_id", "player", "team", "opposition", "venue", "season"]
        all_features = get_feature_columns(df, exclude=exclude)

        # Split into batter and bowler datasets
        # Batters: players with batting involvement
        bat_df = df[df["actual_runs"].notna()].copy()
        bowl_df = df[df["actual_wickets"].notna()].copy()

        print(f"\nBatter samples: {len(bat_df)}")
        print(f"Bowler samples: {len(bowl_df)}")

        # ---- Batting Model ----
        if len(bat_df) > 100:
            print("\n" + "="*50)
            print("  TRAINING BATTING MODEL")
            print("="*50)

            self.bat_feature_cols = [c for c in all_features if c in bat_df.columns]

            bat_train, bat_val, bat_test = temporal_train_test_split(bat_df, season_col="season")

            X_train = bat_train[self.bat_feature_cols].fillna(0)
            y_train = bat_train["actual_runs"]
            X_val = bat_val[self.bat_feature_cols].fillna(0)
            y_val = bat_val["actual_runs"]
            X_test = bat_test[self.bat_feature_cols].fillna(0)
            y_test = bat_test["actual_runs"]

            self.bat_model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=30,
            )
            self.bat_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            print("\n--- Batting Validation ---")
            bat_val_pred = self.bat_model.predict(X_val)
            evaluate_regressor(y_val, bat_val_pred, "Batting XGBoost (Val)")

            print("\n--- Batting Test ---")
            bat_test_pred = self.bat_model.predict(X_test)
            evaluate_regressor(y_test, bat_test_pred, "Batting XGBoost (Test)")

            for threshold in [10, 20, 30]:
                within = np.abs(bat_test_pred - y_test.values) <= threshold
                print(f"  Within {threshold} runs: {within.mean():.1%}")

        # ---- Bowling Model ----
        if len(bowl_df) > 100:
            print("\n" + "="*50)
            print("  TRAINING BOWLING MODEL")
            print("="*50)

            self.bowl_feature_cols = [c for c in all_features if c in bowl_df.columns]

            bowl_train, bowl_val, bowl_test = temporal_train_test_split(bowl_df, season_col="season")

            X_train = bowl_train[self.bowl_feature_cols].fillna(0)
            y_train = bowl_train["actual_wickets"]
            X_val = bowl_val[self.bowl_feature_cols].fillna(0)
            y_val = bowl_val["actual_wickets"]
            X_test = bowl_test[self.bowl_feature_cols].fillna(0)
            y_test = bowl_test["actual_wickets"]

            self.bowl_model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=30,
            )
            self.bowl_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            print("\n--- Bowling Validation ---")
            bowl_val_pred = self.bowl_model.predict(X_val)
            evaluate_regressor(y_val, bowl_val_pred, "Bowling XGBoost (Val)")

            print("\n--- Bowling Test ---")
            bowl_test_pred = self.bowl_model.predict(X_test)
            self.test_results = evaluate_regressor(y_test, bowl_test_pred, "Bowling XGBoost (Test)")

        self.is_trained = True
        return self

    def predict_batting(self, features: dict) -> dict:
        """Predict batting performance."""
        if not self.is_trained or self.bat_model is None:
            raise ValueError("Batting model not trained")

        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.bat_feature_cols}])
        X = X[self.bat_feature_cols].fillna(0)

        predicted_runs = self.bat_model.predict(X)[0]
        predicted_runs = max(0, predicted_runs)

        return {
            "predicted_runs": int(round(predicted_runs)),
            "predicted_runs_range": (max(0, int(round(predicted_runs - 10))),
                                      int(round(predicted_runs + 10))),
            "player": features.get("player", ""),
            "bat_avg": features.get("long_bat_avg", 0),
            "recent_sr": features.get("short_bat_sr", 0),
        }

    def predict_bowling(self, features: dict) -> dict:
        """Predict bowling performance."""
        if not self.is_trained or self.bowl_model is None:
            raise ValueError("Bowling model not trained")

        X = pd.DataFrame([{k: v for k, v in features.items() if k in self.bowl_feature_cols}])
        X = X[self.bowl_feature_cols].fillna(0)

        predicted_wickets = self.bowl_model.predict(X)[0]
        predicted_wickets = max(0, predicted_wickets)

        return {
            "predicted_wickets": round(predicted_wickets, 1),
            "predicted_wickets_range": (max(0, round(predicted_wickets - 1, 1)),
                                         round(predicted_wickets + 1, 1)),
            "player": features.get("player", ""),
            "bowl_economy": features.get("long_bowl_econ", 0),
            "career_wickets": features.get("long_bowl_wickets", 0),
        }

    def save(self, path: str = "outputs/player_performance.pkl"):
        """Save models to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "bat_model": self.bat_model,
            "bowl_model": self.bowl_model,
            "scaler": self.scaler,
            "bat_feature_cols": self.bat_feature_cols,
            "bowl_feature_cols": self.bowl_feature_cols,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "outputs/player_performance.pkl"):
        """Load models from disk."""
        data = joblib.load(path)
        self.bat_model = data["bat_model"]
        self.bowl_model = data["bowl_model"]
        self.scaler = data["scaler"]
        self.bat_feature_cols = data["bat_feature_cols"]
        self.bowl_feature_cols = data["bowl_feature_cols"]
        self.is_trained = True
        print(f"Model loaded from {path}")
        return self
