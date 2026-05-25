"""
Shared cached data loading for all Streamlit pages.
Separate module to avoid circular imports.
"""

import streamlit as st
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@st.cache_resource
def load_data():
    from src.data.preprocessing import preprocess_all
    matches, balls = preprocess_all("dataset")
    return matches, balls


@st.cache_resource
def load_registry():
    try:
        return joblib.load("outputs/registry.pkl")
    except FileNotFoundError:
        matches, balls = load_data()
        from src.data.player_registry import PlayerRegistry
        registry = PlayerRegistry(balls, matches)
        registry.build()
        return registry


@st.cache_resource
def load_models():
    from src.models.match_predictor import MatchPredictor
    from src.models.win_probability import WinProbabilityEstimator
    from src.models.score_predictor import ScorePredictor
    from src.models.player_performance import PlayerPerformancePredictor

    models = {}
    try:
        models["match"] = MatchPredictor().load("outputs/match_predictor.pkl")
    except Exception:
        models["match"] = None
    try:
        models["win_prob"] = WinProbabilityEstimator().load("outputs/win_probability.pkl")
    except Exception:
        models["win_prob"] = None
    try:
        models["score"] = ScorePredictor().load("outputs/score_predictor.pkl")
    except Exception:
        models["score"] = None
    try:
        models["player"] = PlayerPerformancePredictor().load("outputs/player_performance.pkl")
    except Exception:
        models["player"] = None
    return models
