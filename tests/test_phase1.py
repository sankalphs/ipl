"""
Test Phase 1: Data preprocessing, player registry, and feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_all
from src.data.player_registry import PlayerRegistry
from src.data.feature_engineering import FeatureEngineer


def test_preprocessing():
    print("=" * 60)
    print("TEST 1: Data Preprocessing")
    print("=" * 60)

    matches, balls = preprocess_all("dataset")

    print(f"Matches loaded: {len(matches)}")
    print(f"Ball-by-ball records: {len(balls)}")
    print(f"Unique teams: {sorted(matches['team1'].dropna().unique())}")
    print(f"Unique venues (sample): {sorted(matches['venue'].dropna().unique())[:10]}")
    print(f"Seasons: {sorted(matches['season'].unique())}")
    print(f"Matches with results: {matches['has_result'].sum()}")

    # Verify team normalization
    all_teams = set(matches["team1"].dropna().unique()) | set(matches["team2"].dropna().unique())
    assert "Delhi Daredevils" not in all_teams, "Delhi Daredevils should be normalized"
    assert "Kings XI Punjab" not in all_teams, "Kings XI Punjab should be normalized"
    assert "Delhi Capitals" in all_teams
    assert "Punjab Kings" in all_teams
    print("\n[OK] Team normalization passed")

    return matches, balls


def test_player_registry(matches, balls):
    print("\n" + "=" * 60)
    print("TEST 2: Player Registry")
    print("=" * 60)

    registry = PlayerRegistry(balls, matches)
    registry.build()

    print(f"Total players: {len(registry.player_profiles)}")
    print(f"Matches with participants: {len(registry.match_participants)}")
    print(f"Impact player matches: {len(registry.impact_player_matches)}")

    # Test player lookup
    test_player = "V Kohli"
    profile = registry.get_player_profile(test_player)
    if profile:
        print(f"\n{test_player} profile:")
        print(f"  Matches: {len(profile['matches'])}")
        print(f"  Bat innings: {profile['batting']['innings']}")
        print(f"  Runs: {profile['batting']['runs']}")
        print(f"  SR: {(profile['batting']['runs']/profile['batting']['balls_faced']*100) if profile['batting']['balls_faced'] > 0 else 0:.1f}")
        print(f"  Bowl innings: {profile['bowling']['innings']}")

    # Test player-team tracking across seasons
    print(f"\n{test_player} team history:")
    for season in sorted(registry.player_profiles.get(test_player, {}).get("teams", {}).keys()):
        teams = registry.player_profiles[test_player]["teams"][season]
        unique_teams = set(teams.values())
        print(f"  Season {season}: {unique_teams}")

    # Test form features
    if profile:
        sample_match = profile["matches"][10] if len(profile["matches"]) > 10 else profile["matches"][0]
        form = registry.get_player_form(test_player, sample_match, window=5)
        print(f"\n{test_player} form at match {sample_match}:")
        print(f"  Short form - bat avg: {form['short_form']['bat_avg']:.1f}, SR: {form['short_form']['bat_sr']:.1f}")
        print(f"  Long form - bat avg: {form['long_form']['bat_avg']:.1f}, SR: {form['long_form']['bat_sr']:.1f}")

    print("\n[OK] Player registry passed")
    return registry


def test_feature_engineering(matches, balls, registry):
    print("\n" + "=" * 60)
    print("TEST 3: Feature Engineering")
    print("=" * 60)

    fe = FeatureEngineer(matches, balls, registry)

    # Test match features
    sample_match = matches[matches["has_result"] == True].iloc[5]["match_id"]
    match_feats = fe.build_match_features(sample_match)
    print(f"\nMatch features for {sample_match}:")
    for k, v in sorted(match_feats.items()):
        if k not in ["match_id", "venue", "team1", "team2", "season"]:
            print(f"  {k}: {v}")

    # Test ball features
    ball_feats = fe.build_ball_features(sample_match, innings=1, over=5, ball_in_over=3)
    if ball_feats:
        print(f"\nBall features (innings 1, over 5, ball 3):")
        for k, v in sorted(ball_feats.items()):
            if k not in ["match_id", "batting_team", "bowling_team", "phase"]:
                print(f"  {k}: {v}")

    print("\n[OK] Feature engineering passed")
    return fe


if __name__ == "__main__":
    matches, balls = test_preprocessing()
    registry = test_player_registry(matches, balls)
    fe = test_feature_engineering(matches, balls, registry)

    print("\n" + "=" * 60)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 60)
