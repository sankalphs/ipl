"""
2026 Season Evaluation (Optimized)
Test models on 2026 data.
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib


def main():
    print("=" * 60)
    print("  2026 SEASON EVALUATION")
    print("=" * 60)

    from src.data.preprocessing import preprocess_all
    from src.data.player_registry import PlayerRegistry
    from src.data.fast_features import FastFeatureEngineer

    # Load data
    print("\n[1] Loading data...")
    matches, balls = preprocess_all("dataset")

    # Load registry and models
    print("[2] Loading models...")
    registry = joblib.load("outputs/registry.pkl")

    from src.models.match_predictor import MatchPredictor
    from src.models.win_probability import WinProbabilityEstimator
    from src.models.score_predictor import ScorePredictor

    match_model = MatchPredictor().load("outputs/match_predictor.pkl")
    wp_model = WinProbabilityEstimator().load("outputs/win_probability.pkl")
    score_model = ScorePredictor().load("outputs/score_predictor.pkl")

    # Filter 2026 matches
    matches_2026 = matches[matches["season"] == "2026"]
    print(f"\n[3] 2026 Season: {len(matches_2026)} matches")

    # Build features
    print("[4] Building features...")
    ffe = FastFeatureEngineer(matches, balls, registry)

    # ============================================================
    # Model 1: Match Predictor on 2026
    # ============================================================
    print("\n" + "=" * 60)
    print("  MODEL 1: MATCH PREDICTOR - 2026")
    print("=" * 60)

    match_df = ffe.build_match_dataset_fast()
    match_2026 = match_df[match_df["season"] == "2026"]

    if len(match_2026) > 0:
        feature_cols = [c for c in match_2026.columns if c not in [
            "match_id", "season", "venue", "team1", "team2", "target"
        ]]

        correct = 0
        total = 0
        predictions = []

        for _, row in match_2026.iterrows():
            features = {k: v for k, v in row.items() if k in feature_cols}
            result = match_model.predict(features, team1=row["team1"], team2=row["team2"])
            actual = row["target"]
            is_correct = (actual == 1 and result["team1_win_probability"] > 0.5) or \
                          (actual == 0 and result["team1_win_probability"] <= 0.5)

            if is_correct:
                correct += 1
            total += 1

            actual_team = row["team1"] if actual == 1 else row["team2"]
            predictions.append({
                "match_id": row["match_id"],
                "team1": row["team1"],
                "team2": row["team2"],
                "predicted_winner": result["predicted_winner"],
                "actual_winner": actual_team,
                "confidence": result["confidence"],
                "correct": is_correct,
            })

        acc = correct / total if total > 0 else 0
        print(f"\n  2026 Match Prediction Accuracy: {acc:.1%} ({correct}/{total})")

        pred_df = pd.DataFrame(predictions)
        print(f"\n  Confidence Analysis:")
        for conf_bin in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            mask = (pred_df["confidence"] >= conf_bin[0]) & (pred_df["confidence"] < conf_bin[1])
            bin_df = pred_df[mask]
            if len(bin_df) > 0:
                bin_acc = bin_df["correct"].mean()
                print(f"    {conf_bin[0]:.0%}-{conf_bin[1]:.0%}: {bin_acc:.1%} (n={len(bin_df)})")

        print(f"\n  All 2026 Predictions:")
        for _, p in pred_df.iterrows():
            status = "OK" if p["correct"] else "XX"
            print(f"    [{status}] {p['team1']} vs {p['team2']} -> {p['predicted_winner']} "
                  f"(conf: {p['confidence']:.1%}) (actual: {p['actual_winner']})")

    # ============================================================
    # Model 2: Win Probability on 2026
    # ============================================================
    print("\n" + "=" * 60)
    print("  MODEL 2: WIN PROBABILITY - 2026")
    print("=" * 60)

    live_df = ffe.build_live_dataset_fast(sample_rate=30)
    live_2026 = live_df[live_df["season"] == "2026"]

    if len(live_2026) > 0:
        live_feature_cols = [c for c in live_2026.columns if c not in [
            "match_id", "target_won", "season"
        ]]

        X_2026 = live_2026[live_feature_cols].fillna(0)
        y_2026 = live_2026["target_won"].astype(int)

        y_prob = wp_model.lgbm_model.predict_proba(X_2026)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

        acc = accuracy_score(y_2026, y_pred)
        auc = roc_auc_score(y_2026, y_prob)
        logloss = log_loss(y_2026, y_prob)
        brier = brier_score_loss(y_2026, y_prob)

        print(f"\n  2026 Win Probability Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    Log Loss:  {logloss:.4f}")
        print(f"    Brier:     {brier:.4f}")

        print(f"\n  Calibration Check:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mask = (y_prob >= threshold - 0.05) & (y_prob < threshold + 0.05)
            if mask.sum() > 0:
                actual = y_2026[mask].mean()
                print(f"    Predicted ~{threshold:.1f}: Actual = {actual:.3f} (n={mask.sum()})")

    # ============================================================
    # Model 3: Score Predictor on 2026
    # ============================================================
    print("\n" + "=" * 60)
    print("  MODEL 3: SCORE PREDICTOR - 2026")
    print("=" * 60)

    score_df = ffe.build_score_dataset_fast(sample_rate=30)
    score_2026 = score_df[score_df["season"] == "2026"]

    if len(score_2026) > 0:
        score_feature_cols = [c for c in score_2026.columns if c not in [
            "match_id", "final_score", "season"
        ]]

        X_score = score_2026[score_feature_cols].fillna(0)
        y_score = score_2026["final_score"]

        y_score_pred = score_model.xgb_model.predict(X_score)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_score, y_score_pred)
        rmse = np.sqrt(mean_squared_error(y_score, y_score_pred))
        r2 = r2_score(y_score, y_score_pred)

        print(f"\n  2026 Score Prediction Results:")
        print(f"    MAE:  {mae:.1f} runs")
        print(f"    RMSE: {rmse:.1f} runs")
        print(f"    R2:   {r2:.4f}")

        for margin in [10, 20, 30]:
            within = np.abs(y_score_pred - y_score.values) <= margin
            print(f"    Within {margin} runs: {within.mean():.1%}")

    # ============================================================
    # 2026 Season Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("  2026 SEASON ANALYSIS")
    print("=" * 60)

    print(f"\n  Team Standings (2026):")
    team_wins = {}
    for _, m in matches_2026.iterrows():
        if m["has_result"]:
            winner = m["winner"]
            team_wins[winner] = team_wins.get(winner, 0) + 1

    for team, wins in sorted(team_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"    {team:30s}: {wins} wins")

    print(f"\n  Top Run Scorers (2026):")
    bat_2026 = balls[balls["season"] == "2026"].groupby("batter")["batter_runs"].sum().reset_index()
    bat_2026.columns = ["player", "runs"]
    bat_2026 = bat_2026.sort_values("runs", ascending=False).head(10)
    for _, row in bat_2026.iterrows():
        print(f"    {row['player']:20s}: {row['runs']} runs")

    print(f"\n  Top Wicket Takers (2026):")
    bowl_2026 = balls[(balls["season"] == "2026") & (balls["is_wicket"] == 1)].groupby("bowler")["is_wicket"].sum().reset_index()
    bowl_2026.columns = ["player", "wickets"]
    bowl_2026 = bowl_2026.sort_values("wickets", ascending=False).head(10)
    for _, row in bowl_2026.iterrows():
        print(f"    {row['player']:20s}: {row['wickets']} wickets")

    print(f"\n  Venue Stats (2026):")
    venue_stats = []
    for venue in matches_2026["venue"].dropna().unique():
        v_matches = matches_2026[matches_2026["venue"] == venue]
        first_inn_scores = []
        for _, m in v_matches.iterrows():
            inn1 = balls[(balls["match_id"] == m["match_id"]) & (balls["innings"] == 1)]
            if not inn1.empty:
                first_inn_scores.append(inn1["total_runs"].sum())
        if first_inn_scores:
            venue_stats.append({
                "venue": venue,
                "matches": len(v_matches),
                "avg_first_inn": np.mean(first_inn_scores),
            })

    for v in sorted(venue_stats, key=lambda x: x["matches"], reverse=True)[:10]:
        print(f"    {v['venue']:50s}: {v['matches']} matches, avg 1st inn: {v['avg_first_inn']:.0f}")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
