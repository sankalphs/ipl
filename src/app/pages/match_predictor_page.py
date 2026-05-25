"""
Match Predictor Page
Pre-match outcome prediction based on team composition, venue, toss, etc.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render():
    st.title("Match Predictor")
    st.markdown("Predict match outcome **before the match starts** based on team composition, venue, toss, and historical data.")

    from src.app.streamlit_app import load_data, load_registry, load_models

    matches, balls = load_data()
    registry = load_registry()
    models = load_models()

    match_model = models.get("match")
    if match_model is None or not match_model.is_trained:
        st.error("Match predictor model not trained. Run `python train_all.py` first.")
        return

    # Team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", sorted(set(matches["team1"].dropna().unique()) &
                                                 set(matches["team2"].dropna().unique())),
                              index=0)
    with col2:
        team2 = st.selectbox("Team 2", sorted(set(matches["team1"].dropna().unique()) &
                                                 set(matches["team2"].dropna().unique())),
                              index=1)

    if team1 == team2:
        st.warning("Please select two different teams.")
        return

    # Venue and toss
    venues = sorted(matches["venue"].dropna().unique())
    venue = st.selectbox("Venue", venues)

    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col4:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    # Get the latest match_id to use as reference for player stats
    latest_match_id = matches["match_id"].max()

    # Build features for prediction
    from src.data.fast_features import FastFeatureEngineer
    ffe = FastFeatureEngineer(matches, balls, registry)

    # Get participants for both teams (use most recent roster)
    team1_players = []
    team2_players = []
    for mid in sorted(matches["match_id"].unique(), reverse=True)[:50]:
        participants = registry.get_match_players(mid)
        if not team1_players and team1 in participants:
            team1_players = participants[team1]
        if not team2_players and team2 in participants:
            team2_players = participants[team2]
        if team1_players and team2_players:
            break

    # Compute features
    team1_bat, team1_bowl, team1_exp, team1_star, team1_allr = ffe._team_strength_fast(team1_players, latest_match_id)
    team2_bat, team2_bowl, team2_exp, team2_star, team2_allr = ffe._team_strength_fast(team2_players, latest_match_id)

    # H2H
    h2h_matches, h2h_t1_wr, h2h_recent = ffe._h2h_fast(team1, team2, latest_match_id + 1)

    # Team form
    t1_wins, t1_wr, t1_streak = ffe._team_form_fast(team1, latest_match_id + 1)
    t2_wins, t2_wr, t2_streak = ffe._team_form_fast(team2, latest_match_id + 1)

    # Venue stats
    venue_matches = matches[matches["venue"] == venue]
    venue_first_inn = []
    for _, m in venue_matches.iterrows():
        mid = m["match_id"]
        inn1 = balls[(balls["match_id"] == mid) & (balls["innings"] == 1)]
        if not inn1.empty:
            venue_first_inn.append(inn1["total_runs"].sum())
    venue_avg = np.mean(venue_first_inn) if venue_first_inn else 160

    # Toss features
    t1_won_toss = 1 if toss_winner == team1 else 0
    toss_bat = 1 if toss_decision == "bat" else 0
    t1_batting_first = 1 if (toss_winner == team1 and toss_decision == "bat") or \
                             (toss_winner == team2 and toss_decision == "field") else 0

    features = {
        "team1_bat_strength": team1_bat,
        "team1_bowl_strength": team1_bowl,
        "team1_experience_avg": team1_exp,
        "team1_star_power": team1_star,
        "team1_allrounder_count": team1_allr,
        "team2_bat_strength": team2_bat,
        "team2_bowl_strength": team2_bowl,
        "team2_experience_avg": team2_exp,
        "team2_star_power": team2_star,
        "team2_allrounder_count": team2_allr,
        "h2h_matches": h2h_matches,
        "h2h_team1_win_rate": h2h_t1_wr,
        "h2h_recent_form": h2h_recent,
        "team1_last10_win_rate": t1_wr,
        "team1_streak": t1_streak,
        "team2_last10_win_rate": t2_wr,
        "team2_streak": t2_streak,
        "venue_avg_first_inn": venue_avg,
        "venue_chase_success_rate": 0.5,
        "venue_matches": len(venue_matches),
        "venue_is_batting_friendly": 1 if venue_avg > 170 else 0,
        "team1_won_toss": t1_won_toss,
        "toss_decision_bat": toss_bat,
        "team1_batting_first": t1_batting_first,
        "team1_home": 0,
        "team2_home": 0,
        "season_phase": 1,
        "impact_rule_active": 1,
        "team1_has_impact_sub": 0,
        "team2_has_impact_sub": 0,
        "bat_strength_diff": team1_bat - team2_bat,
        "bowl_strength_diff": team1_bowl - team2_bowl,
        "experience_diff": team1_exp - team2_exp,
        "form_diff": t1_wr - t2_wr,
    }

    if st.button("Predict Match Outcome", type="primary"):
        result = match_model.predict(features)

        st.markdown("---")
        st.subheader("Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{team1} Win", f"{result['team1_win_probability']:.1%}")
        with col2:
            st.metric(f"{team2} Win", f"{result['team2_win_probability']:.1%}")
        with col3:
            st.metric("Predicted Winner", result["predicted_winner"])

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["team1_win_probability"] * 100,
            title={"text": f"{team1} Win Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "lightcoral"},
                    {"range": [40, 60], "color": "lightyellow"},
                    {"range": [60, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Key factors
        st.subheader("Key Factors")
        factors = []
        if features["bat_strength_diff"] > 2:
            factors.append(f":green[{team1} has stronger batting lineup]")
        elif features["bat_strength_diff"] < -2:
            factors.append(f":green[{team2} has stronger batting lineup]")
        if features["bowl_strength_diff"] > 0.05:
            factors.append(f":green[{team1} has better bowling attack]")
        elif features["bowl_strength_diff"] < -0.05:
            factors.append(f":green[{team2} has better bowling attack]")
        if features["form_diff"] > 0.1:
            factors.append(f":green[{team1} is in better recent form]")
        elif features["form_diff"] < -0.1:
            factors.append(f":green[{team2} is in better recent form]")
        if features["team1_won_toss"]:
            factors.append(f":blue[{team1} won the toss]")
        else:
            factors.append(f":blue[{team2} won the toss]")
        if features["team1_home"]:
            factors.append(f":green[{team1} has home advantage]")

        for f in factors:
            st.markdown(f"- {f}")

        # Team composition comparison
        st.subheader("Team Composition Comparison")
        comp_data = pd.DataFrame({
            "Metric": ["Batting Strength", "Bowling Strength", "Experience", "Star Power", "All-Rounders"],
            team1: [team1_bat, team1_bowl, team1_exp, team1_star, team1_allr],
            team2: [team2_bat, team2_bowl, team2_exp, team2_star, team2_allr],
        })
        st.dataframe(comp_data, use_container_width=True)
