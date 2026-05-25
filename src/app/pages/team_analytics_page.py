"""
Team Analytics Page
Team vs team H2H, venue performance, season form, impact player stats.
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
    st.title("Team Analytics")
    st.markdown("Explore team performance, head-to-head records, and venue analysis.")

    from src.app.streamlit_app import load_data, load_registry

    matches, balls = load_data()
    registry = load_registry()

    # Team selection
    all_teams = sorted(set(matches["team1"].dropna().unique()) & set(matches["team2"].dropna().unique()))
    team = st.selectbox("Select Team", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0)

    # Overall record
    st.subheader(f"{team} - Overall Record")
    team_matches = matches[
        ((matches["team1"] == team) | (matches["team2"] == team)) &
        (matches["has_result"] == True)
    ]
    total = len(team_matches)
    wins = (team_matches["winner"] == team).sum()
    losses = total - wins

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", total)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Losses", losses)
    with col4:
        st.metric("Win Rate", f"{wins/total*100:.1f}%" if total > 0 else "N/A")

    # Season-wise performance
    st.subheader("Season-wise Performance")
    season_stats = []
    for season in sorted(matches["season"].unique()):
        season_matches = team_matches[team_matches["season"] == season]
        if len(season_matches) == 0:
            continue
        season_wins = (season_matches["winner"] == team).sum()
        season_stats.append({
            "Season": season,
            "Matches": len(season_matches),
            "Wins": season_wins,
            "Losses": len(season_matches) - season_wins,
            "Win Rate": season_wins / len(season_matches) * 100,
        })

    if season_stats:
        season_df = pd.DataFrame(season_stats)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=season_df["Season"],
            y=season_df["Wins"],
            name="Wins",
            marker_color="green",
        ))
        fig.add_trace(go.Bar(
            x=season_df["Season"],
            y=season_df["Losses"],
            name="Losses",
            marker_color="red",
        ))
        fig.update_layout(
            barmode="stack",
            title=f"{team} - Season-wise Results",
            xaxis_title="Season",
            yaxis_title="Matches",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Win rate trend
    fig2 = px.line(season_df, x="Season", y="Win Rate", title=f"{team} - Win Rate Trend",
                    markers=True)
    fig2.add_hline(y=50, line_dash="dash", line_color="gray")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Head-to-Head
    st.markdown("---")
    st.subheader("Head-to-Head Records")
    h2h_opponent = st.selectbox("Opponent", [t for t in all_teams if t != team])

    h2h_matches = matches[
        (
            ((matches["team1"] == team) & (matches["team2"] == h2h_opponent)) |
            ((matches["team1"] == h2h_opponent) & (matches["team2"] == team))
        ) &
        (matches["has_result"] == True)
    ]

    h2h_total = len(h2h_matches)
    h2h_wins = (h2h_matches["winner"] == team).sum()
    h2h_losses = h2h_total - h2h_wins

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", h2h_total)
    with col2:
        st.metric(f"{team} Wins", h2h_wins)
    with col3:
        st.metric(f"{h2h_opponent} Wins", h2h_losses)

    # H2H by venue
    if h2h_total > 0:
        h2h_venue = h2h_matches.groupby("venue").apply(
            lambda x: pd.Series({
                "matches": len(x),
                f"{team}_wins": (x["winner"] == team).sum(),
            })
        ).reset_index()
        h2h_venue = h2h_venue.sort_values("matches", ascending=False).head(10)
        st.dataframe(h2h_venue, use_container_width=True)

    # Venue Analysis
    st.markdown("---")
    st.subheader(f"{team} - Venue Analysis")
    venue_stats = []
    for venue in team_matches["venue"].dropna().unique():
        venue_m = team_matches[team_matches["venue"] == venue]
        v_wins = (venue_m["winner"] == team).sum()
        venue_stats.append({
            "Venue": venue,
            "Matches": len(venue_m),
            "Wins": v_wins,
            "Win Rate": v_wins / len(venue_m) * 100 if len(venue_m) > 0 else 0,
        })

    venue_df = pd.DataFrame(venue_stats).sort_values("Matches", ascending=False)
    st.dataframe(venue_df.head(15), use_container_width=True)

    # Toss analysis
    st.markdown("---")
    st.subheader("Toss Analysis")
    toss_matches = team_matches[team_matches["toss_winner"] == team]
    if len(toss_matches) > 0:
        toss_bat = toss_matches[toss_matches["toss_decision"] == "bat"]
        toss_field = toss_matches[toss_matches["toss_decision"] == "field"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toss Win Rate", f"{len(toss_matches)/len(team_matches)*100:.1f}%")
        with col2:
            bat_win = (toss_bat["winner"] == team).sum() / len(toss_bat) * 100 if len(toss_bat) > 0 else 0
            st.metric("Win when Bat First", f"{bat_win:.1f}%")
        with col3:
            field_win = (toss_field["winner"] == team).sum() / len(toss_field) * 100 if len(toss_field) > 0 else 0
            st.metric("Win when Field First", f"{field_win:.1f}%")
        with col4:
            st.metric("Prefer", "Bat" if len(toss_bat) > len(toss_field) else "Field")

    # Impact Player Analysis (2023+)
    st.markdown("---")
    st.subheader("Impact Player Analysis (2023+)")
    impact_matches = team_matches[team_matches["season"].apply(lambda x: str(x) >= "2023")]
    if len(impact_matches) > 0:
        impact_wins = (impact_matches["winner"] == team).sum()
        st.metric(f"Win Rate (2023+)", f"{impact_wins/len(impact_matches)*100:.1f}%")

        # Compare with pre-2023
        pre_impact = team_matches[team_matches["season"].apply(lambda x: str(x) < "2023")]
        if len(pre_impact) > 0:
            pre_wins = (pre_impact["winner"] == team).sum()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pre-2023 Win Rate", f"{pre_wins/len(pre_impact)*100:.1f}%")
            with col2:
                diff = (impact_wins/len(impact_matches) - pre_wins/len(pre_impact)) * 100
                st.metric("Change", f"{diff:+.1f}%", delta_color="normal")
    else:
        st.info("No matches found for 2023+ season.")

    # Top players for team
    st.markdown("---")
    st.subheader(f"Top Players for {team}")
    team_players = {}
    for _, match in team_matches.iterrows():
        mid = match["match_id"]
        participants = registry.get_match_players(mid)
        for p in participants.get(team, []):
            if p not in team_players:
                team_players[p] = 0
            team_players[p] += 1

    if team_players:
        top_players = sorted(team_players.items(), key=lambda x: x[1], reverse=True)[:20]
        top_df = pd.DataFrame(top_players, columns=["Player", "Matches"])
        st.dataframe(top_df, use_container_width=True)
