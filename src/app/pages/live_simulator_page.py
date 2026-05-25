"""
Live Match Simulator Page
Ball-by-ball win probability during a match.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.cache import load_data, load_registry, load_models
from src.data.fast_features import FastFeatureEngineer


def render():
    st.title("Live Match Simulator")
    st.markdown("Track **ball-by-ball win probability** during a match.")

    matches, balls = load_data()
    registry = load_registry()
    models = load_models()

    wp_model = models.get("win_prob")
    score_model = models.get("score")

    if wp_model is None or not wp_model.is_trained:
        st.error("Win probability model not trained. Run `python train_all.py` first.")
        return

    # Match selection
    st.subheader("Select a Match")
    match_options = matches[matches["has_result"] == True].sort_values("match_id", ascending=False).head(50)
    match_labels = [
        f"{row['team1']} vs {row['team2']} ({row['venue']}, {row['season']})"
        for _, row in match_options.iterrows()
    ]
    selected_idx = st.selectbox("Choose Match", range(len(match_labels)),
                                 format_func=lambda x: match_labels[x], key="ls_match")
    selected_match = match_options.iloc[selected_idx]
    match_id = selected_match["match_id"]

    st.info(f"**{selected_match['team1']}** vs **{selected_match['team2']}** at {selected_match['venue']}")
    st.info(f"Winner: **{selected_match['winner']}**")

    innings = st.radio("Innings", [1, 2], horizontal=True, key="ls_innings")

    match_balls_df = balls[balls["match_id"] == match_id]
    inn_balls = match_balls_df[match_balls_df["innings"] == innings].sort_values(["over", "ball_in_over"])

    if inn_balls.empty:
        st.warning("No data for this innings.")
        return

    ffe = FastFeatureEngineer(matches, balls, registry)

    inn_balls = inn_balls.copy()
    inn_balls["cum_runs"] = inn_balls["total_runs"].cumsum()
    inn_balls["cum_wickets"] = inn_balls["is_wicket"].cumsum()

    sample_rate = 6
    sample_indices = list(range(0, len(inn_balls), sample_rate))
    if len(inn_balls) - 1 not in sample_indices:
        sample_indices.append(len(inn_balls) - 1)

    batting_team = inn_balls["batting_team"].iloc[0]
    bowling_team = inn_balls["bowling_team"].iloc[0]
    toss_winner = selected_match.get("toss_winner", None)

    # Pre-compute team strengths
    participants = registry.get_match_players(match_id)
    bat_players = participants.get(batting_team, [])
    bowl_players = participants.get(bowling_team, [])
    bat_str = ffe._quick_team_bat_strength(bat_players, match_id)
    bowl_str = ffe._quick_team_bowl_strength(bowl_players, match_id)
    venue_row = ffe.venue_stats_df[ffe.venue_stats_df["match_id"] == match_id]
    venue_avg = venue_row["venue_avg_first_inn"].values[0] if len(venue_row) > 0 else 160

    # Target for 2nd innings
    target_runs = 0
    if innings == 2:
        first_inn = match_balls_df[match_balls_df["innings"] == 1]
        target_runs = int(first_inn["total_runs"].sum()) + 1 if not first_inn.empty else 0

    progress_bar = st.progress(0)
    timeline = []

    for i, idx in enumerate(sample_indices):
        ball = inn_balls.iloc[idx]
        over = ball["over"]
        total_runs = int(ball["cum_runs"])
        wickets = int(ball["cum_wickets"])
        balls_bowled = idx + 1
        balls_remaining = 120 - balls_bowled
        current_rr = (total_runs / balls_bowled * 6) if balls_bowled > 0 else 0

        is_pp = 1 if over < 6 else 0
        is_mid = 1 if 6 <= over < 16 else 0
        is_death = 1 if over >= 16 else 0

        runs_req = max(0, target_runs - total_runs) if innings == 2 else 0
        req_rr = (runs_req / (balls_remaining / 6)) if balls_remaining > 0 and innings == 2 else 0
        is_chasing = 1 if innings == 2 else 0

        window_start = max(0, idx - 29)
        window = inn_balls.iloc[window_start:idx + 1]
        dots = (window["total_runs"] == 0).sum()
        boundaries = window["batter_runs"].isin([4, 6]).sum()
        dot_pct = dots / len(window) if len(window) > 0 else 0
        boundary_pct = boundaries / len(window) if len(window) > 0 else 0

        last5_start = max(0, over - 4)
        last5 = inn_balls[(inn_balls["over"] >= last5_start) & (inn_balls.index <= inn_balls.index[idx])]
        last5_rr = (last5["total_runs"].sum() / len(last5) * 6) if len(last5) > 0 else 0

        features = {
            "current_score": total_runs, "wickets_lost": wickets,
            "balls_bowled": balls_bowled, "balls_remaining": balls_remaining,
            "current_run_rate": current_rr,
            "is_powerplay": is_pp, "is_middle": is_mid, "is_death": is_death,
            "target": target_runs, "runs_required": runs_req,
            "required_run_rate": req_rr, "is_chasing": is_chasing,
            "dot_ball_pct": dot_pct, "boundary_pct": boundary_pct,
            "last_5_overs_rr": last5_rr,
            "batting_team_strength": bat_str, "bowling_team_strength": bowl_str,
            "venue_avg_first_inn": venue_avg,
            "batting_team_won_toss": 1 if toss_winner == batting_team else 0,
            "batting_team_batting_first": 1 if innings == 1 else 0,
        }

        pred = wp_model.predict(features)
        win_prob = pred["batting_team_win_probability"]

        predicted_score = None
        if innings == 1 and score_model and score_model.is_trained:
            score_feat = {
                "current_score": total_runs, "wickets_lost": wickets,
                "balls_bowled": balls_bowled, "balls_remaining": balls_remaining,
                "current_run_rate": current_rr,
                "is_powerplay": is_pp, "is_middle": is_mid, "is_death": is_death,
                "batting_team_strength": bat_str, "venue_avg_first_inn": venue_avg,
            }
            predicted_score = score_model.predict(score_feat)["predicted_score"]

        timeline.append({
            "over": over, "ball": ball["ball_in_over"],
            "score": total_runs, "wickets": wickets,
            "run_rate": current_rr, "win_prob": win_prob,
            "predicted_score": predicted_score,
        })
        progress_bar.progress((i + 1) / len(sample_indices))

    progress_bar.empty()
    timeline_df = pd.DataFrame(timeline)

    st.markdown("---")
    last = timeline_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Score", f"{int(last['score'])}/{int(last['wickets'])}")
    with c2:
        st.metric("Overs", f"{last['over']}.{int(last['ball'])}")
    with c3:
        st.metric("Run Rate", f"{last['run_rate']:.2f}")
    with c4:
        st.metric(f"{batting_team} Win Prob", f"{last['win_prob']:.1%}")

    if innings == 2:
        c5, c6 = st.columns(2)
        with c5:
            st.metric("Target", f"{target_runs}")
        with c6:
            st.metric("Runs Required", f"{max(0, target_runs - int(last['score']))}")

    st.subheader("Win Probability Timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(timeline_df))), y=timeline_df["win_prob"] * 100,
        mode="lines+markers", name=f"{batting_team} Win %",
        line=dict(color="blue", width=2), marker=dict(size=4),
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.update_layout(xaxis_title="Ball Progression", yaxis_title="Win Probability (%)",
                      yaxis_range=[0, 100], height=400)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Score Progression")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(len(timeline_df))), y=timeline_df["score"],
        mode="lines+markers", name="Runs",
        line=dict(color="green", width=2), fill="tozeroy",
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(len(timeline_df))), y=timeline_df["wickets"] * 10,
        mode="lines+markers", name="Wickets (x10)",
        line=dict(color="red", width=2),
    ))
    if innings == 1 and "predicted_score" in timeline_df.columns:
        fig2.add_trace(go.Scatter(
            x=list(range(len(timeline_df))), y=timeline_df["predicted_score"],
            mode="lines", name="Predicted Final Score",
            line=dict(color="orange", width=2, dash="dash"),
        ))
    fig2.update_layout(xaxis_title="Ball Progression", yaxis_title="Runs", height=350)
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Detailed Timeline"):
        display_df = timeline_df[["over", "ball", "score", "wickets", "run_rate", "win_prob"]].copy()
        display_df["over_display"] = display_df.apply(lambda r: f"{r['over']}.{int(r['ball'])}", axis=1)
        display_df["win_prob_%"] = display_df["win_prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df[["over_display", "score", "wickets", "run_rate", "win_prob_%"]],
                      width="stretch")
