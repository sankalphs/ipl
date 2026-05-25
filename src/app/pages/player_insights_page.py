"""
Player Insights Page
Career trends, form analysis, phase-wise performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render():
    st.title("Player Insights")
    st.markdown("Explore player career trends, form, and performance.")

    from src.app.streamlit_app import load_data, load_registry, load_models

    matches, balls = load_data()
    registry = load_registry()
    models = load_models()

    # Player search
    all_players = sorted(registry.player_profiles.keys())
    player_name = st.selectbox("Search Player", all_players, index=all_players.index("V Kohli") if "V Kohli" in all_players else 0)

    profile = registry.get_player_profile(player_name)
    if not profile:
        st.warning(f"Player '{player_name}' not found.")
        return

    # Career overview
    st.subheader("Career Overview")
    bat = profile["batting"]
    bowl = profile["bowling"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", len(profile["matches"]))
    with col2:
        st.metric("Runs", bat["runs"])
    with col3:
        st.metric("Highest Score", bat["highest_score"])
    with col4:
        sr = (bat["runs"] / bat["balls_faced"] * 100) if bat["balls_faced"] > 0 else 0
        st.metric("Strike Rate", f"{sr:.1f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Bat Innings", bat["innings"])
    with col6:
        avg = bat["runs"] / bat["outs"] if bat["outs"] > 0 else 0
        st.metric("Average", f"{avg:.1f}")
    with col7:
        st.metric("50s", bat["fifties"])
    with col8:
        st.metric("100s", bat["hundreds"])

    if bowl["innings"] > 0:
        st.markdown("---")
        st.subheader("Bowling Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bowl Innings", bowl["innings"])
        with col2:
            st.metric("Wickets", bowl["wickets"])
        with col3:
            econ = (bowl["runs_conceded"] / bowl["balls_bowled"] * 6) if bowl["balls_bowled"] > 0 else 0
            st.metric("Economy", f"{econ:.2f}")
        with col4:
            st.metric("Overs", f"{bowl['overs_bowled']:.1f}")

    # Batting scores distribution
    st.markdown("---")
    st.subheader("Batting Scores Distribution")
    scores = list(bat["scores_per_match"].values())
    if scores:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, nbinsx=30, name="Scores", marker_color="blue"))
        fig.update_layout(
            xaxis_title="Runs",
            yaxis_title="Frequency",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Team history
    st.subheader("Team History")
    team_history = []
    for season in sorted(profile["teams"].keys()):
        teams = profile["teams"][season]
        unique_teams = list(set(teams.values()))
        team_history.append({"Season": season, "Team": unique_teams[0] if unique_teams else "Unknown"})
    team_df = pd.DataFrame(team_history)
    st.dataframe(team_df, use_container_width=True)

    # Recent form
    st.subheader("Recent Form (Last 10 Matches)")
    recent_matches = profile["matches"][-10:] if len(profile["matches"]) >= 10 else profile["matches"]
    recent_scores = []
    for mid in recent_matches:
        score = bat["scores_per_match"].get(mid, None)
        if score is not None:
            match_info = matches[matches["match_id"] == mid]
            opponent = ""
            if not match_info.empty:
                row = match_info.iloc[0]
                # Determine opponent
                player_team = registry.get_player_team(mid, player_name)
                if player_team == row["team1"]:
                    opponent = row["team2"]
                else:
                    opponent = row["team1"]
            recent_scores.append({
                "Match ID": mid,
                "Score": score,
                "Opponent": opponent,
            })

    if recent_scores:
        recent_df = pd.DataFrame(recent_scores)
        avg_recent = recent_df["Score"].mean()
        career_avg = bat["runs"] / bat["outs"] if bat["outs"] > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recent 10-Match Average", f"{avg_recent:.1f}")
        with col2:
            diff = avg_recent - career_avg
            st.metric("vs Career Average", f"{diff:+.1f}", delta_color="normal")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(recent_scores))),
            y=[s["Score"] for s in recent_scores],
            marker_color=["green" if s["Score"] >= 30 else "red" for s in recent_scores],
            text=[f"{s['Score']} vs {s['Opponent']}" for s in recent_scores],
            textposition="auto",
        ))
        fig.update_layout(
            xaxis_title="Match (Recent to Oldest)",
            yaxis_title="Runs",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Player comparison
    st.markdown("---")
    st.subheader("Player Comparison")
    compare_player = st.selectbox("Compare with", [p for p in all_players if p != player_name],
                                   index=all_players.index("RG Sharma") if "RG Sharma" in all_players and "RG Sharma" != player_name else 0)

    if compare_player:
        comp_profile = registry.get_player_profile(compare_player)
        if comp_profile:
            comp_bat = comp_profile["batting"]
            comp_avg = comp_bat["runs"] / comp_bat["outs"] if comp_bat["outs"] > 0 else 0
            comp_sr = (comp_bat["runs"] / comp_bat["balls_faced"] * 100) if comp_bat["balls_faced"] > 0 else 0

            comp_data = pd.DataFrame({
                "Stat": ["Matches", "Runs", "Average", "Strike Rate", "50s", "100s", "Highest"],
                player_name: [
                    len(profile["matches"]), bat["runs"], f"{avg:.1f}", f"{sr:.1f}",
                    bat["fifties"], bat["hundreds"], bat["highest_score"]
                ],
                compare_player: [
                    len(comp_profile["matches"]), comp_bat["runs"], f"{comp_avg:.1f}", f"{comp_sr:.1f}",
                    comp_bat["fifties"], comp_bat["hundreds"], comp_bat["highest_score"]
                ],
            })
            st.dataframe(comp_data, use_container_width=True)

    # Player Performance Prediction
    st.markdown("---")
    st.subheader("Performance Prediction")
    player_model = models.get("player")
    if player_model and player_model.is_trained:
        st.info("Predict player's likely score in a match based on form, venue, and opposition.")

        venue_options = sorted(matches["venue"].dropna().unique())
        pred_venue = st.selectbox("Venue", venue_options, key="pred_venue")
        opposition = st.selectbox("Opposition", sorted(set(matches["team1"].dropna().unique()) |
                                                         set(matches["team2"].dropna().unique())),
                                   key="pred_opposition")

        if st.button("Predict Performance"):
            latest_match = matches["match_id"].max()
            short_form = registry.get_player_form(player_name, latest_match, window=5)
            long_form = registry.get_player_form(player_name, latest_match, window=100)
            venue_stats = registry.get_player_venue_stats(player_name, pred_venue, latest_match)
            vs_stats = registry.get_player_vs_team_stats(player_name, opposition, latest_match)

            features = {
                "player": player_name,
                "short_bat_avg": short_form["short_form"]["bat_avg"],
                "short_bat_sr": short_form["short_form"]["bat_sr"],
                "short_bowl_econ": short_form["short_form"]["bowl_economy"],
                "short_bowl_wickets": short_form["short_form"]["bowl_wickets"],
                "long_bat_avg": long_form["long_form"]["bat_avg"],
                "long_bat_sr": long_form["long_form"]["bat_sr"],
                "long_bowl_econ": long_form["long_form"]["bowl_economy"],
                "long_bowl_wickets": long_form["long_form"]["bowl_wickets"],
                "career_matches": long_form["long_form"]["matches_played"],
                "venue_bat_avg": venue_stats["bat_avg"],
                "venue_bat_sr": venue_stats["bat_sr"],
                "venue_matches": venue_stats["matches_played"],
                "vs_bat_avg": vs_stats["bat_avg"],
                "vs_bat_sr": vs_stats["bat_sr"],
                "vs_matches": vs_stats["matches_played"],
                "is_allrounder": 1 if long_form["long_form"]["is_allrounder"] else 0,
            }

            try:
                result = player_model.predict_batting(features)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Runs", result["predicted_runs"])
                with col2:
                    st.metric("Range", f"{result['predicted_runs_range'][0]} - {result['predicted_runs_range'][1]}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
