"""
Feature Engineering Pipeline
Builds all features for the 4 ML models.
Ensures no data leakage by using only prior data for each prediction.
"""

import pandas as pd
import numpy as np
from .player_registry import PlayerRegistry
from .preprocessing import normalize_team_name, normalize_venue_name


class FeatureEngineer:
    """
    Builds features for IPL match prediction models.
    All features are computed using only data available BEFORE the prediction point.
    """

    def __init__(self, matches: pd.DataFrame, balls: pd.DataFrame, registry: PlayerRegistry):
        self.matches = matches
        self.balls = balls
        self.registry = registry

        # Pre-compute venue stats (rolling, updated per match)
        self._venue_stats_cache: dict[str, dict] = {}
        # Pre-compute team form (rolling)
        self._team_form_cache: dict[str, list] = {}
        # Head-to-head records
        self._h2h_cache: dict[tuple, list] = {}

    # ============================================================
    # Pre-Match Features (Model 1)
    # ============================================================

    def build_match_features(self, match_id: int) -> dict | None:
        """
        Build pre-match features for a specific match.
        Returns features for both teams (team1 perspective).
        """
        match_info = self.matches[self.matches["match_id"] == match_id]
        if match_info.empty:
            return None

        row = match_info.iloc[0]
        team1 = row["team1"]
        team2 = row["team2"]
        toss_winner = row.get("toss_winner", None)
        toss_decision = row.get("toss_decision", None)
        venue = row.get("venue", "Unknown")
        season = row.get("season", 0)
        date = row.get("date", None)

        # Get participants
        participants = self.registry.get_match_players(match_id)
        team1_players = participants.get(team1, [])
        team2_players = participants.get(team2, [])

        features = {
            "match_id": match_id,
            "season": season,
            "venue": venue,
            "team1": team1,
            "team2": team2,
        }

        # A. Player-level features (team aggregates)
        features.update(self._team_composition_features(team1_players, team2_players, match_id))

        # B. Head-to-head features
        features.update(self._h2h_features(team1, team2, match_id))

        # C. Team form features
        features.update(self._team_form_features(team1, team2, match_id))

        # D. Venue features
        features.update(self._venue_features(venue, match_id))

        # E. Toss features
        features.update(self._toss_features(toss_winner, toss_decision, team1, team2))

        # F. Home game proxy
        features.update(self._home_features(team1, team2, venue, row.get("city", "")))

        # G. Season phase
        features["season_phase"] = self._get_season_phase(match_id, season)

        # H. Impact player features
        features.update(self._impact_player_features(match_id, team1, team2, season))

        # Target
        winner = row.get("winner", None)
        if pd.notna(winner):
            features["target"] = 1 if winner == team1 else 0
        else:
            features["target"] = None

        return features

    def _team_composition_features(self, team1_players: list, team2_players: list, match_id: int) -> dict:
        """Build team composition features from player profiles."""
        features = {}

        for team_label, players in [("team1", team1_players), ("team2", team2_players)]:
            if not players:
                for suffix in ["_bat_strength", "_bowl_strength", "_allrounder_count",
                               "_experience_avg", "_star_power", "_form_avg"]:
                    features[f"{team_label}{suffix}"] = 0
                continue

            # Get player stats before this match
            bat_strengths = []
            bowl_strengths = []
            allrounder_count = 0
            experience_counts = []
            star_count = 0

            for player in players:
                stats = self.registry.get_player_stats_at_match(player, match_id)
                form = self.registry.get_player_form(player, match_id, window=5)

                # Batting strength: blend of career avg and recent form
                career_bat = stats["bat_avg"]
                short_bat = form["short_form"]["bat_avg"]
                bat_strength = 0.6 * short_bat + 0.4 * career_bat if stats["matches_played"] > 3 else career_bat
                bat_strengths.append(bat_strength)

                # Bowling strength: blend of career economy and recent form
                career_econ = stats["bowl_economy"]
                short_econ = form["short_form"]["bowl_economy"]
                bowl_strength = 0.6 * short_econ + 0.4 * career_econ if stats["bowl_innings"] > 3 else career_econ
                # Lower economy is better, so invert
                bowl_strengths.append(1 / (bowl_strength + 1))

                if stats["is_allrounder"]:
                    allrounder_count += 1

                experience_counts.append(stats["matches_played"])
                if stats["matches_played"] >= 50:
                    star_count += 1

            features[f"{team_label}_bat_strength"] = np.mean(bat_strengths) if bat_strengths else 0
            features[f"{team_label}_bowl_strength"] = np.mean(bowl_strengths) if bowl_strengths else 0
            features[f"{team_label}_allrounder_count"] = allrounder_count
            features[f"{team_label}_experience_avg"] = np.mean(experience_counts) if experience_counts else 0
            features[f"{team_label}_star_power"] = star_count

            # Form average (how well team's players are doing recently)
            form_avgs = []
            for player in players:
                form = self.registry.get_player_form(player, match_id, window=5)
                if form["short_form"]["bat_innings"] > 0:
                    form_avgs.append(form["short_form"]["bat_sr"])
            features[f"{team_label}_form_avg"] = np.mean(form_avgs) if form_avgs else 0

        return features

    def _h2h_features(self, team1: str, team2: str, match_id: int) -> dict:
        """Head-to-head features between two teams."""
        h2h_key = (min(team1, team2), max(team1, team2))

        # Get prior H2H matches
        prior_h2h = self.matches[
            (
                ((self.matches["team1"] == team1) & (self.matches["team2"] == team2)) |
                ((self.matches["team1"] == team2) & (self.matches["team2"] == team1))
            ) &
            (self.matches["match_id"] < match_id) &
            (self.matches["has_result"] == True)
        ].sort_values("match_id")

        if prior_h2h.empty:
            return {
                "h2h_matches": 0,
                "h2h_team1_wins": 0,
                "h2h_team1_win_rate": 0.5,
                "h2h_recent_form": 0,
            }

        team1_wins = (prior_h2h["winner"] == team1).sum()
        total_h2h = len(prior_h2h)

        # Recent form (last 5 H2H)
        recent = prior_h2h.tail(5)
        recent_team1_wins = (recent["winner"] == team1).sum()

        return {
            "h2h_matches": total_h2h,
            "h2h_team1_wins": team1_wins,
            "h2h_team1_win_rate": team1_wins / total_h2h,
            "h2h_recent_form": recent_team1_wins / len(recent),
        }

    def _team_form_features(self, team1: str, team2: str, match_id: int) -> dict:
        """Recent team form features."""
        features = {}

        for team_label, team in [("team1", team1), ("team2", team2)]:
            # Last 10 matches
            prior = self.matches[
                (
                    (self.matches["team1"] == team) | (self.matches["team2"] == team)
                ) &
                (self.matches["match_id"] < match_id) &
                (self.matches["has_result"] == True)
            ].sort_values("match_id")

            if prior.empty:
                features[f"{team_label}_last10_wins"] = 0
                features[f"{team_label}_last10_win_rate"] = 0.5
                features[f"{team_label}_streak"] = 0
                continue

            last10 = prior.tail(10)
            wins = (last10["winner"] == team).sum()
            features[f"{team_label}_last10_wins"] = wins
            features[f"{team_label}_last10_win_rate"] = wins / len(last10)

            # Win/loss streak
            streak = 0
            for _, m in prior.iloc[::-1].iterrows():
                if m["winner"] == team:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
            features[f"{team_label}_streak"] = streak

        return features

    def _venue_features(self, venue: str, match_id: int) -> dict:
        """Venue-specific features computed from prior matches at the venue."""
        prior_venue = self.matches[
            (self.matches["venue"] == venue) &
            (self.matches["match_id"] < match_id) &
            (self.matches["has_result"] == True)
        ]

        if prior_venue.empty:
            return {
                "venue_matches": 0,
                "venue_avg_first_inn": 160,
                "venue_chase_success_rate": 0.5,
                "venue_is_batting_friendly": 0,
            }

        # Get first innings scores for these matches
        first_inn_scores = []
        chase_successes = 0
        total_chases = 0

        for _, match in prior_venue.iterrows():
            mid = match["match_id"]
            match_balls = self.balls[self.balls["match_id"] == mid]
            first_inn = match_balls[match_balls["innings"] == 1]
            if not first_inn.empty:
                first_inn_scores.append(first_inn["total_runs"].sum())

            # Chase success
            if match.get("toss_decision") == "field":
                total_chases += 1
                if match["winner"] == match["toss_winner"]:
                    chase_successes += 1

        avg_first_inn = np.mean(first_inn_scores) if first_inn_scores else 160
        chase_rate = chase_successes / total_chases if total_chases > 0 else 0.5

        return {
            "venue_matches": len(prior_venue),
            "venue_avg_first_inn": avg_first_inn,
            "venue_chase_success_rate": chase_rate,
            "venue_is_batting_friendly": 1 if avg_first_inn > 170 else 0,
        }

    def _toss_features(self, toss_winner: str, toss_decision: str, team1: str, team2: str) -> dict:
        """Toss-related features."""
        team1_won_toss = 1 if toss_winner == team1 else 0
        team1_batting_first = 1 if (toss_winner == team1 and toss_decision == "bat") or \
                                    (toss_winner == team2 and toss_decision == "field") else 0

        return {
            "team1_won_toss": team1_won_toss,
            "toss_decision_bat": 1 if toss_decision == "bat" else 0,
            "team1_batting_first": team1_batting_first,
        }

    def _home_features(self, team1: str, team2: str, venue: str, city: str) -> dict:
        """Home game proxy based on city/team name matching."""
        # Simple heuristic: check if city name is in team name or vice versa
        team_cities = {
            "Mumbai Indians": ["Mumbai"],
            "Chennai Super Kings": ["Chennai"],
            "Kolkata Knight Riders": ["Kolkata"],
            "Royal Challengers Bengaluru": ["Bengaluru", "Bangalore"],
            "Delhi Capitals": ["Delhi"],
            "Rajasthan Royals": ["Jaipur"],
            "Punjab Kings": ["Mohali", "Chandigarh", "Dharamsala"],
            "Sunrisers Hyderabad": ["Hyderabad"],
            "Gujarat Titans": ["Ahmedabad", "Rajkot"],
            "Lucknow Super Giants": ["Lucknow"],
        }

        team1_home = 0
        team2_home = 0
        if city and not pd.isna(city):
            city_lower = city.lower()
            for c in team_cities.get(team1, []):
                if c.lower() in city_lower:
                    team1_home = 1
                    break
            for c in team_cities.get(team2, []):
                if c.lower() in city_lower:
                    team2_home = 1
                    break

        return {"team1_home": team1_home, "team2_home": team2_home}

    def _get_season_phase(self, match_id: int, season: int) -> str:
        """Determine if match is early, mid, late, or playoff season."""
        season_matches = self.matches[self.matches["season"] == season].sort_values("match_id")
        if season_matches.empty:
            return "unknown"

        total = len(season_matches)
        match_pos = season_matches[season_matches["match_id"] == match_id].index
        if len(match_pos) == 0:
            return "unknown"

        pos = list(season_matches.index).index(match_pos[0])
        ratio = pos / total

        if ratio < 0.33:
            return "early"
        elif ratio < 0.66:
            return "mid"
        elif ratio < 0.85:
            return "late"
        else:
            return "playoffs"

    def _impact_player_features(self, match_id: int, team1: str, team2: str, season) -> dict:
        """Impact player related features."""
        try:
            season_int = int(season)
        except (ValueError, TypeError):
            season_int = 0
        features = {
            "impact_rule_active": 1 if season_int >= 2023 else 0,
            "team1_has_impact_sub": 0,
            "team2_has_impact_sub": 0,
        }

        impact_info = self.registry.get_is_impact_match(match_id)
        if impact_info:
            features["team1_has_impact_sub"] = 1 if impact_info.get(team1, False) else 0
            features["team2_has_impact_sub"] = 1 if impact_info.get(team2, False) else 0

        return features

    # ============================================================
    # Live Match Features (Model 2 & 3)
    # ============================================================

    def build_ball_features(self, match_id: int, innings: int, over: int, ball_in_over: int) -> dict | None:
        """
        Build features for live win probability / score prediction at a specific ball.
        """
        match_info = self.matches[self.matches["match_id"] == match_id]
        if match_info.empty:
            return None

        row = match_info.iloc[0]
        team1 = row["team1"]
        team2 = row["team2"]
        venue = row.get("venue", "Unknown")
        season = row.get("season", 0)
        toss_winner = row.get("toss_winner", None)
        toss_decision = row.get("toss_decision", None)

        # Get current match state up to this ball
        match_balls = self.balls[self.balls["match_id"] == match_id]
        current_ball = match_balls[
            (match_balls["innings"] == innings) &
            (
                (match_balls["over"] < over) |
                ((match_balls["over"] == over) & (match_balls["ball_in_over"] <= ball_in_over))
            )
        ]

        if current_ball.empty:
            return None

        batting_team = current_ball["batting_team"].iloc[0]
        bowling_team = current_ball["bowling_team"].iloc[0]

        # Current state
        total_runs = current_ball["total_runs"].sum()
        wickets = current_ball["is_wicket"].sum()
        balls_bowled = len(current_ball[current_ball["wide"] == 0]) if "wide" in current_ball.columns else len(current_ball)
        total_legal_balls = innings * 120  # 20 overs * 6 balls

        current_rr = (total_runs / balls_bowled * 6) if balls_bowled > 0 else 0
        balls_remaining = 120 - balls_bowled

        features = {
            "match_id": match_id,
            "innings": innings,
            "over": over,
            "ball_in_over": ball_in_over,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "current_score": total_runs,
            "wickets_lost": wickets,
            "balls_bowled": balls_bowled,
            "balls_remaining": balls_remaining,
            "current_run_rate": current_rr,
        }

        # Phase
        if over < 6:
            features["phase"] = "powerplay"
        elif over < 16:
            features["phase"] = "middle"
        else:
            features["phase"] = "death"

        features["is_powerplay"] = 1 if over < 6 else 0
        features["is_middle"] = 1 if 6 <= over < 16 else 0
        features["is_death"] = 1 if over >= 16 else 0

        # Second innings specific
        if innings == 2:
            first_inn = match_balls[match_balls["innings"] == 1]
            target = first_inn["total_runs"].sum() + 1 if not first_inn.empty else 0
            runs_required = target - total_runs
            features["target"] = target
            features["runs_required"] = max(0, runs_required)
            features["required_run_rate"] = (runs_required / (balls_remaining / 6)) if balls_remaining > 0 else 0
            features["is_chasing"] = 1
        else:
            features["target"] = 0
            features["runs_required"] = 0
            features["required_run_rate"] = 0
            features["is_chasing"] = 0

        # Batter on strike stats
        striker = current_ball.iloc[-1]["batter"] if not current_ball.empty else None
        if striker:
            striker_form = self.registry.get_player_form(striker, match_id, window=5)
            features["striker_bat_avg"] = striker_form["short_form"]["bat_avg"]
            features["striker_bat_sr"] = striker_form["short_form"]["bat_sr"]
            features["striker_matches"] = striker_form["short_form"]["matches_played"]
        else:
            features["striker_bat_avg"] = 0
            features["striker_bat_sr"] = 0
            features["striker_matches"] = 0

        # Current bowler stats
        bowler = current_ball.iloc[-1]["bowler"] if not current_ball.empty else None
        if bowler:
            bowler_form = self.registry.get_player_form(bowler, match_id, window=5)
            features["bowler_economy"] = bowler_form["short_form"]["bowl_economy"]
            features["bowler_wickets"] = bowler_form["short_form"]["bowl_wickets"]
            features["bowler_matches"] = bowler_form["short_form"]["matches_played"]
        else:
            features["bowler_economy"] = 0
            features["bowler_wickets"] = 0
            features["bowler_matches"] = 0

        # Partnership
        features["partnership_runs"] = self._calc_partnership(current_ball)
        features["partnership_balls"] = self._calc_partnership_balls(current_ball)

        # Last 5 overs run rate
        last_5_over_runs = current_ball[current_ball["over"] >= max(0, over - 4)]["total_runs"].sum()
        last_5_over_balls = len(current_ball[current_ball["over"] >= max(0, over - 4)])
        features["last_5_overs_rr"] = (last_5_over_runs / last_5_over_balls * 6) if last_5_over_balls > 0 else 0

        # Dot ball percentage
        dots = len(current_ball[current_ball["total_runs"] == 0])
        features["dot_ball_pct"] = dots / balls_bowled if balls_bowled > 0 else 0

        # Boundary percentage
        boundaries = len(current_ball[current_ball["batter_runs"].isin([4, 6])])
        features["boundary_pct"] = boundaries / balls_bowled if balls_bowled > 0 else 0

        # Team strength features (pre-match)
        participants = self.registry.get_match_players(match_id)
        batting_players = participants.get(batting_team, [])
        bowling_players = participants.get(bowling_team, [])

        bat_strengths = []
        for p in batting_players:
            stats = self.registry.get_player_stats_at_match(p, match_id)
            bat_strengths.append(stats["bat_avg"])
        features["batting_team_strength"] = np.mean(bat_strengths) if bat_strengths else 0

        bowl_strengths = []
        for p in bowling_players:
            stats = self.registry.get_player_stats_at_match(p, match_id)
            if stats["bowl_economy"] > 0:
                bowl_strengths.append(1 / (stats["bowl_economy"] + 1))
        features["bowling_team_strength"] = np.mean(bowl_strengths) if bowl_strengths else 0

        # Venue context
        venue_feats = self._venue_features(venue, match_id)
        features["venue_avg_first_inn"] = venue_feats["venue_avg_first_inn"]
        features["venue_chase_success_rate"] = venue_feats["venue_chase_success_rate"]

        # Toss context
        features["batting_team_won_toss"] = 1 if toss_winner == batting_team else 0
        features["batting_team_batting_first"] = 1 if innings == 1 else 0

        # Target for win probability (if match has result)
        winner = row.get("winner", None)
        if pd.notna(winner):
            features["target_won"] = 1 if winner == batting_team else 0
        else:
            features["target_won"] = None

        return features

    def _calc_partnership(self, current_ball: pd.DataFrame) -> int:
        """Calculate current partnership runs."""
        if current_ball.empty:
            return 0

        # Find last wicket
        wickets = current_ball[current_ball["is_wicket"] == 1]
        if wickets.empty:
            return current_ball["total_runs"].sum()

        last_wicket_idx = wickets.index[-1]
        partnership = current_ball.loc[last_wicket_idx + 1:]["total_runs"].sum()
        return partnership

    def _calc_partnership_balls(self, current_ball: pd.DataFrame) -> int:
        """Calculate current partnership balls."""
        if current_ball.empty:
            return 0

        wickets = current_ball[current_ball["is_wicket"] == 1]
        if wickets.empty:
            return len(current_ball)

        last_wicket_idx = wickets.index[-1]
        return len(current_ball.loc[last_wicket_idx + 1:])

    # ============================================================
    # Dataset Builders
    # ============================================================

    def build_match_dataset(self, match_ids: list[int] | None = None) -> pd.DataFrame:
        """Build complete dataset for Model 1 (pre-match prediction)."""
        if match_ids is None:
            match_ids = sorted(self.matches["match_id"].unique())

        records = []
        for mid in match_ids:
            feats = self.build_match_features(mid)
            if feats and feats.get("target") is not None:
                records.append(feats)

        df = pd.DataFrame(records)
        print(f"Match dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def build_live_dataset(self, match_ids: list[int] | None = None, sample_rate: int = 6) -> pd.DataFrame:
        """
        Build complete dataset for Model 2 (win probability).
        sample_rate: sample every N balls to reduce dataset size.
        """
        if match_ids is None:
            match_ids = sorted(self.matches["match_id"].unique())

        records = []
        for mid in match_ids:
            match_balls = self.balls[self.balls["match_id"] == mid]
            for innings in [1, 2]:
                inn_balls = match_balls[match_balls["innings"] == innings]
                if inn_balls.empty:
                    continue

                # Sample balls
                for idx, (_, ball) in enumerate(inn_balls.iterrows()):
                    if idx % sample_rate != 0 and idx != len(inn_balls) - 1:
                        continue

                    feats = self.build_ball_features(
                        mid, innings, ball["over"], ball["ball_in_over"]
                    )
                    if feats and feats.get("target_won") is not None:
                        records.append(feats)

        df = pd.DataFrame(records)
        print(f"Live dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def build_score_prediction_dataset(self, match_ids: list[int] | None = None, sample_rate: int = 6) -> pd.DataFrame:
        """Build dataset for Model 3 (first innings score prediction)."""
        if match_ids is None:
            match_ids = sorted(self.matches["match_id"].unique())

        records = []
        for mid in match_ids:
            match_balls = self.balls[self.balls["match_id"] == mid]
            first_inn = match_balls[match_balls["innings"] == 1]
            if first_inn.empty:
                continue

            final_score = first_inn["total_runs"].sum()

            for idx, (_, ball) in enumerate(first_inn.iterrows()):
                if idx % sample_rate != 0 and idx != len(first_inn) - 1:
                    continue

                feats = self.build_ball_features(mid, 1, ball["over"], ball["ball_in_over"])
                if feats:
                    feats["final_score"] = final_score
                    # Remove target_won since this is score prediction
                    feats.pop("target_won", None)
                    records.append(feats)

        df = pd.DataFrame(records)
        print(f"Score prediction dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def build_player_performance_dataset(self, match_ids: list[int] | None = None) -> pd.DataFrame:
        """Build dataset for Model 4 (player performance prediction)."""
        if match_ids is None:
            match_ids = sorted(self.matches["match_id"].unique())

        records = []
        for mid in match_ids:
            match_info = self.matches[self.matches["match_id"] == mid]
            if match_info.empty:
                continue

            row = match_info.iloc[0]
            venue = row.get("venue", "Unknown")
            season = row.get("season", 0)
            team1 = row["team1"]
            team2 = row["team2"]

            match_balls = self.balls[self.balls["match_id"] == mid]
            participants = self.registry.get_match_players(mid)

            for team in [team1, team2]:
                opposition = team2 if team == team1 else team1
                players = participants.get(team, [])

                for player in players:
                    # Get player's actual performance in this match
                    bat = match_balls[
                        (match_balls["batter"] == player) &
                        (match_balls["batting_team"] == team)
                    ]
                    bowl = match_balls[
                        (match_balls["bowler"] == player) &
                        (match_balls["bowling_team"] == team)
                    ]

                    actual_runs = bat["batter_runs"].sum() if not bat.empty else None
                    actual_sr = (actual_runs / len(bat) * 100) if not bat.empty and len(bat) > 0 else None
                    actual_wickets = bowl["is_wicket"].sum() if not bowl.empty else None
                    actual_economy = (bowl["total_runs"].sum() / len(bowl) * 6) if not bowl.empty and len(bowl) > 0 else None

                    # Skip players with no involvement
                    if actual_runs is None and actual_wickets is None:
                        continue

                    # Build features
                    short_form = self.registry.get_player_form(player, mid, window=5)
                    long_form = self.registry.get_player_form(player, mid, window=100)
                    venue_stats = self.registry.get_player_venue_stats(player, venue, mid)
                    vs_stats = self.registry.get_player_vs_team_stats(player, opposition, mid)

                    feat = {
                        "match_id": mid,
                        "player": player,
                        "team": team,
                        "opposition": opposition,
                        "venue": venue,
                        "season": season,
                        # Short form
                        "short_bat_avg": short_form["short_form"]["bat_avg"],
                        "short_bat_sr": short_form["short_form"]["bat_sr"],
                        "short_bowl_econ": short_form["short_form"]["bowl_economy"],
                        "short_bowl_wickets": short_form["short_form"]["bowl_wickets"],
                        # Long form
                        "long_bat_avg": long_form["long_form"]["bat_avg"],
                        "long_bat_sr": long_form["long_form"]["bat_sr"],
                        "long_bowl_econ": long_form["long_form"]["bowl_economy"],
                        "long_bowl_wickets": long_form["long_form"]["bowl_wickets"],
                        "career_matches": long_form["long_form"]["matches_played"],
                        # Venue specific
                        "venue_bat_avg": venue_stats["bat_avg"],
                        "venue_bat_sr": venue_stats["bat_sr"],
                        "venue_matches": venue_stats["matches_played"],
                        # Vs opposition
                        "vs_bat_avg": vs_stats["bat_avg"],
                        "vs_bat_sr": vs_stats["bat_sr"],
                        "vs_matches": vs_stats["matches_played"],
                        # Role
                        "is_allrounder": 1 if long_form["long_form"]["is_allrounder"] else 0,
                        # Targets
                        "actual_runs": actual_runs,
                        "actual_sr": actual_sr,
                        "actual_wickets": actual_wickets,
                        "actual_economy": actual_economy,
                    }
                    records.append(feat)

        df = pd.DataFrame(records)
        print(f"Player performance dataset: {len(df)} samples")
        return df
