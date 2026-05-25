"""
IPL Analytics & Prediction Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import preprocess_all, CURRENT_TEAMS
from src.data.player_registry import PlayerRegistry
from src.data.fast_features import FastFeatureEngineer

# Page config
st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon=":cricket_bat_and_ball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Data Loading (cached)
# ============================================================

@st.cache_resource
def load_data():
    """Load and preprocess data."""
    matches, balls = preprocess_all("dataset")
    return matches, balls


@st.cache_resource
def load_registry():
    """Load player registry."""
    try:
        return joblib.load("outputs/registry.pkl")
    except FileNotFoundError:
        matches, balls = load_data()
        registry = PlayerRegistry(balls, matches)
        registry.build()
        return registry


@st.cache_resource
def load_models():
    """Load all trained models."""
    from src.models.match_predictor import MatchPredictor
    from src.models.win_probability import WinProbabilityEstimator
    from src.models.score_predictor import ScorePredictor
    from src.models.player_performance import PlayerPerformancePredictor

    models = {}
    try:
        models["match"] = MatchPredictor().load("outputs/match_predictor.pkl")
    except:
        models["match"] = None
    try:
        models["win_prob"] = WinProbabilityEstimator().load("outputs/win_probability.pkl")
    except:
        models["win_prob"] = None
    try:
        models["score"] = ScorePredictor().load("outputs/score_predictor.pkl")
    except:
        models["score"] = None
    try:
        models["player"] = PlayerPerformancePredictor().load("outputs/player_performance.pkl")
    except:
        models["player"] = None
    return models


# ============================================================
# Main Page
# ============================================================

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

    # Recent matches
    st.subheader("Recent Matches")
    recent = matches[matches["has_result"] == True].sort_values("match_id", ascending=False).head(10)
    display_df = recent[["date", "team1", "team2", "venue", "winner", "season"]].copy()
    display_df.columns = ["Date", "Team 1", "Team 2", "Venue", "Winner", "Season"]
    st.dataframe(display_df, use_container_width=True)

    # Season overview
    st.subheader("Matches per Season")
    season_counts = matches.groupby("season").size().reset_index(name="matches")
    fig = px.bar(season_counts, x="season", y="matches", title="Matches per Season")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Sidebar Navigation
# ============================================================

page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Match Predictor", "Live Simulator", "Player Insights", "Team Analytics"]
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
