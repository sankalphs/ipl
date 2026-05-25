"""
Train all 4 IPL prediction models (Optimized).
Run: python train_all.py
"""

import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.data.preprocessing import preprocess_all
from src.data.player_registry import PlayerRegistry
from src.data.fast_features import FastFeatureEngineer
from src.models.match_predictor import MatchPredictor
from src.models.win_probability import WinProbabilityEstimator
from src.models.score_predictor import ScorePredictor
from src.models.player_performance import PlayerPerformancePredictor
from src.evaluation.metrics import get_feature_columns


def main():
    print("=" * 60)
    print("  IPL ANALYTICS PLATFORM - MODEL TRAINING")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    matches, balls = preprocess_all("dataset")
    print(f"  Matches: {len(matches)}, Ball-by-ball: {len(balls)}")

    # Step 2: Build player registry
    print("\n[2/6] Building player registry...")
    registry = PlayerRegistry(balls, matches)
    registry.build()

    # Step 3: Build fast feature engineer
    print("\n[3/6] Building feature engineering pipeline...")
    ffe = FastFeatureEngineer(matches, balls, registry)

    # Step 4: Train Match Predictor (Model 1)
    print("\n" + "=" * 60)
    print("[4/6] TRAINING MODEL 1: Match Outcome Predictor")
    print("=" * 60)
    match_df = ffe.build_match_dataset_fast()

    match_pred = MatchPredictor()
    feature_cols = [c for c in match_df.columns if c not in [
        "match_id", "season", "venue", "team1", "team2", "target"
    ]]
    match_pred.train(match_df, feature_cols=feature_cols)
    match_pred.save("outputs/match_predictor.pkl")

    # Step 5: Train Win Probability (Model 2)
    print("\n" + "=" * 60)
    print("[5/6] TRAINING MODEL 2: Win Probability Estimator")
    print("=" * 60)
    live_df = ffe.build_live_dataset_fast(sample_rate=15)

    win_prob = WinProbabilityEstimator()
    live_feature_cols = [c for c in live_df.columns if c not in [
        "match_id", "target_won", "season"
    ]]
    win_prob.train(live_df, feature_cols=live_feature_cols)
    win_prob.save("outputs/win_probability.pkl")

    # Step 6: Train Score Predictor (Model 3)
    print("\n" + "=" * 60)
    print("[6/6] TRAINING MODEL 3: Score Predictor")
    print("=" * 60)
    score_df = ffe.build_score_dataset_fast(sample_rate=15)

    score_pred = ScorePredictor()
    score_feature_cols = [c for c in score_df.columns if c not in [
        "match_id", "final_score", "season"
    ]]
    score_pred.train(score_df, feature_cols=score_feature_cols)
    score_pred.save("outputs/score_predictor.pkl")

    # Step 7: Train Player Performance (Model 4)
    print("\n" + "=" * 60)
    print("[7/7] TRAINING MODEL 4: Player Performance Predictor")
    print("=" * 60)
    player_df = ffe.build_player_dataset_fast()

    player_pred = PlayerPerformancePredictor()
    player_pred.train(player_df)
    player_pred.save("outputs/player_performance.pkl")

    # Save registry for app use
    import joblib
    joblib.dump(registry, "outputs/registry.pkl")
    print(f"\nPlayer registry saved to outputs/registry.pkl")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Models saved to: outputs/")
    print(f"    - match_predictor.pkl")
    print(f"    - win_probability.pkl")
    print(f"    - score_predictor.pkl")
    print(f"    - player_performance.pkl")
    print(f"    - registry.pkl")


if __name__ == "__main__":
    main()
