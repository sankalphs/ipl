"""
IPL Analytics & Prediction Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.cache import load_data, load_registry, load_models

# Page config - no emoji to avoid Windows path issues
st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon="IPL",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main_page():
    st.title("IPL Analytics & Prediction Platform")
    st.markdown("---")

    matches, balls = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(matches))
    with col2:
        st.metric("Seasons", matches["season"].nunique())
    with col3:
        st.metric("Players", len(load_registry().player_profiles))
    with col4:
        st.metric("Ball-by-Ball Records", f"{len(balls):,}")

    st.markdown("---")

    st.subheader("Recent Matches")
    recent = matches[matches["has_result"] == True].sort_values("match_id", ascending=False).head(10)
    display_df = recent[["date", "team1", "team2", "venue", "winner", "season"]].copy()
    display_df.columns = ["Date", "Team 1", "Team 2", "Venue", "Winner", "Season"]
    st.dataframe(display_df, width="stretch")

    st.subheader("Matches per Season")
    season_counts = matches.groupby("season").size().reset_index(name="matches")
    fig = px.bar(season_counts, x="season", y="matches", title="Matches per Season")
    st.plotly_chart(fig, width="stretch")


# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Match Predictor", "Live Simulator", "Player Insights", "Team Analytics"],
    key="nav_selectbox",
)

if page == "Home":
    main_page()
elif page == "Match Predictor":
    from src.app.pages import match_predictor_page
    match_predictor_page.render()
elif page == "Live Simulator":
    from src.app.pages import live_simulator_page
    live_simulator_page.render()
elif page == "Player Insights":
    from src.app.pages import player_insights_page
    player_insights_page.render()
elif page == "Team Analytics":
    from src.app.pages import team_analytics_page
    team_analytics_page.render()
