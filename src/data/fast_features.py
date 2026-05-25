"""
Optimized Feature Engineering - Pre-computes features in batch.
Uses vectorized operations and caching for speed.
"""

import pandas as pd
import numpy as np
from .player_registry import PlayerRegistry
from .preprocessing import normalize_team_name


class FastFeatureEngineer:
    """
    Optimized feature engineering that pre-computes features in batch.
    """

    def __init__(self, matches: pd.DataFrame, balls: pd.DataFrame, registry: PlayerRegistry):
        self.matches = matches.copy()
        self.balls = balls.copy()
        self.registry = registry

        # Pre-sort for efficient lookups
        self.balls = self.balls.sort_values(["match_id", "innings", "over", "ball_in_over"]).reset_index(drop=True)

        # Pre-compute match-level aggregates
        self._precompute_match_aggregates()
        self._precompute_venue_stats()
        self._precompute_player_stats()

    def _precompute_match_aggregates(self):
        """Pre-compute match-level aggregates from ball-by-ball data."""
        # First innings scores
        first_inn = self.balls[self.balls["innings"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
        first_inn.columns = ["match_id", "first_innings_score"]
        self.matches = self.matches.merge(first_inn, on="match_id", how="left")

        # Total balls bowled per innings
        balls_count = self.balls.groupby(["match_id", "innings"]).size().reset_index(name="balls_bowled")
        self.match_balls_count = balls_count

    def _precompute_venue_stats(self):
        """Pre-compute venue statistics."""
        # Merge venue info
        venue_matches = self.matches[self.matches["has_result"] == True].copy()

        # Average first innings score per venue
        venue_first_inn = venue_matches[["match_id", "venue", "first_innings_score"]].copy()
        venue_first_inn = venue_first_inn.dropna(subset=["first_innings_score"])

        # Compute venue stats (expanding window - only use prior matches)
        venue_stats = []
        for _, row in venue_matches.sort_values("match_id").iterrows():
            venue = row["venue"]
            match_id = row["match_id"]

            # Get prior matches at this venue
            prior = venue_matches[
                (venue_matches["venue"] == venue) &
                (venue_matches["match_id"] < match_id) &
                (venue_matches["first_innings_score"].notna())
            ]

            if len(prior) > 0:
                avg_score = prior["first_innings_score"].mean()
                # Chase success
                toss_chasers = prior[prior["toss_decision"] == "field"]
                chase_success = (toss_chasers["winner"] == toss_chasers["toss_winner"]).mean() if len(toss_chasers) > 0 else 0.5
            else:
                avg_score = 160
                chase_success = 0.5

            venue_stats.append({
                "match_id": match_id,
                "venue_avg_first_inn": avg_score,
                "venue_chase_success_rate": chase_success,
                "venue_matches": len(prior),
            })

        self.venue_stats_df = pd.DataFrame(venue_stats)

    def _precompute_player_stats(self):
        """Pre-compute player rolling stats for all players."""
        # Build a player-match sorted list
        player_match_rows = []
        for match_id in sorted(self.balls["match_id"].unique()):
            match_balls = self.balls[self.balls["match_id"] == match_id]
            for player in set(match_balls["batter"].unique()) | set(match_balls["bowler"].unique()):
                player_match_rows.append({"match_id": match_id, "player": player})

        self.player_matches_df = pd.DataFrame(player_match_rows)

    def build_match_dataset_fast(self) -> pd.DataFrame:
        """Build match-level dataset efficiently."""
        records = []

        for _, match in self.matches.sort_values("match_id").iterrows():
            if not match.get("has_result", False):
                continue

            match_id = match["match_id"]
            team1 = match["team1"]
            team2 = match["team2"]
            venue = match.get("venue", "Unknown")
            season = match.get("season", "0")
            toss_winner = match.get("toss_winner", None)
            toss_decision = match.get("toss_decision", None)
            winner = match.get("winner", None)

            if pd.isna(winner):
                continue

            # Get participants
            participants = self.registry.get_match_players(match_id)
            team1_players = participants.get(team1, [])
            team2_players = participants.get(team2, [])

            # Team composition features
            team1_bat, team1_bowl, team1_exp, team1_star, team1_allr = self._team_strength_fast(team1_players, match_id)
            team2_bat, team2_bowl, team2_exp, team2_star, team2_allr = self._team_strength_fast(team2_players, match_id)

            # H2H features
            h2h_matches, h2h_t1_wr, h2h_recent = self._h2h_fast(team1, team2, match_id)

            # Team form
            t1_wins, t1_wr, t1_streak = self._team_form_fast(team1, match_id)
            t2_wins, t2_wr, t2_streak = self._team_form_fast(team2, match_id)

            # Venue stats
            venue_row = self.venue_stats_df[self.venue_stats_df["match_id"] == match_id]
            venue_avg = venue_row["venue_avg_first_inn"].values[0] if len(venue_row) > 0 else 160
            venue_chase = venue_row["venue_chase_success_rate"].values[0] if len(venue_row) > 0 else 0.5
            venue_n = venue_row["venue_matches"].values[0] if len(venue_row) > 0 else 0

            # Toss
            t1_won_toss = 1 if toss_winner == team1 else 0
            toss_bat = 1 if toss_decision == "bat" else 0
            t1_batting_first = 1 if (toss_winner == team1 and toss_decision == "bat") or \
                                     (toss_winner == team2 and toss_decision == "field") else 0

            # Home proxy
            t1_home, t2_home = self._home_proxy(team1, team2, venue, match.get("city", ""))

            # Season phase
            season_phase = self._season_phase(match_id, season)

            # Impact player
            try:
                season_int = int(season)
            except:
                season_int = 0
            impact_active = 1 if season_int >= 2023 else 0
            impact_info = self.registry.get_is_impact_match(match_id)
            t1_impact = 1 if impact_info.get(team1, False) else 0
            t2_impact = 1 if impact_info.get(team2, False) else 0

            # Season phase encoding
            phase_map = {"early": 0, "mid": 1, "late": 2, "playoffs": 3, "unknown": 1}
            phase_enc = phase_map.get(season_phase, 1)

            record = {
                "match_id": match_id,
                "season": season,
                "venue": venue,
                "team1": team1,
                "team2": team2,
                # Team 1 composition
                "team1_bat_strength": team1_bat,
                "team1_bowl_strength": team1_bowl,
                "team1_experience_avg": team1_exp,
                "team1_star_power": team1_star,
                "team1_allrounder_count": team1_allr,
                # Team 2 composition
                "team2_bat_strength": team2_bat,
                "team2_bowl_strength": team2_bowl,
                "team2_experience_avg": team2_exp,
                "team2_star_power": team2_star,
                "team2_allrounder_count": team2_allr,
                # H2H
                "h2h_matches": h2h_matches,
                "h2h_team1_win_rate": h2h_t1_wr,
                "h2h_recent_form": h2h_recent,
                # Team form
                "team1_last10_win_rate": t1_wr,
                "team1_streak": t1_streak,
                "team2_last10_win_rate": t2_wr,
                "team2_streak": t2_streak,
                # Venue
                "venue_avg_first_inn": venue_avg,
                "venue_chase_success_rate": venue_chase,
                "venue_matches": venue_n,
                "venue_is_batting_friendly": 1 if venue_avg > 170 else 0,
                # Toss
                "team1_won_toss": t1_won_toss,
                "toss_decision_bat": toss_bat,
                "team1_batting_first": t1_batting_first,
                # Home
                "team1_home": t1_home,
                "team2_home": t2_home,
                # Season phase
                "season_phase": phase_enc,
                # Impact player
                "impact_rule_active": impact_active,
                "team1_has_impact_sub": t1_impact,
                "team2_has_impact_sub": t2_impact,
                # Strength diff (useful derived features)
                "bat_strength_diff": team1_bat - team2_bat,
                "bowl_strength_diff": team1_bowl - team2_bowl,
                "experience_diff": team1_exp - team2_exp,
                "form_diff": t1_wr - t2_wr,
                # Target
                "target": 1 if winner == team1 else 0,
            }
            records.append(record)

        df = pd.DataFrame(records)
        print(f"Match dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def _team_strength_fast(self, players: list, match_id: int) -> tuple:
        """Fast team strength calculation."""
        bat_strengths = []
        bowl_strengths = []
        experience = []
        star_count = 0
        allrounder_count = 0

        for player in players:
            stats = self.registry.get_player_stats_at_match(player, match_id)
            form = self.registry.get_player_form(player, match_id, window=5)

            # Batting
            career_bat = stats["bat_avg"]
            short_bat = form["short_form"]["bat_avg"]
            bat_val = 0.6 * short_bat + 0.4 * career_bat if stats["matches_played"] > 3 else career_bat
            bat_strengths.append(bat_val)

            # Bowling
            career_econ = stats["bowl_economy"]
            bowl_val = 1 / (career_econ + 1) if career_econ > 0 else 0.5
            bowl_strengths.append(bowl_val)

            experience.append(stats["matches_played"])
            if stats["matches_played"] >= 50:
                star_count += 1
            if stats["is_allrounder"]:
                allrounder_count += 1

        return (
            np.mean(bat_strengths) if bat_strengths else 0,
            np.mean(bowl_strengths) if bowl_strengths else 0,
            np.mean(experience) if experience else 0,
            star_count,
            allrounder_count,
        )

    def _h2h_fast(self, team1: str, team2: str, match_id: int) -> tuple:
        """Fast H2H computation."""
        prior = self.matches[
            (
                ((self.matches["team1"] == team1) & (self.matches["team2"] == team2)) |
                ((self.matches["team1"] == team2) & (self.matches["team2"] == team1))
            ) &
            (self.matches["match_id"] < match_id) &
            (self.matches["has_result"] == True)
        ]

        if prior.empty:
            return 0, 0.5, 0.5

        total = len(prior)
        t1_wins = (prior["winner"] == team1).sum()
        win_rate = t1_wins / total

        recent = prior.tail(5)
        recent_wr = (recent["winner"] == team1).sum() / len(recent)

        return total, win_rate, recent_wr

    def _team_form_fast(self, team: str, match_id: int) -> tuple:
        """Fast team form computation."""
        prior = self.matches[
            (
                (self.matches["team1"] == team) | (self.matches["team2"] == team)
            ) &
            (self.matches["match_id"] < match_id) &
            (self.matches["has_result"] == True)
        ].sort_values("match_id")

        if prior.empty:
            return 0, 0.5, 0

        last10 = prior.tail(10)
        wins = (last10["winner"] == team).sum()
        win_rate = wins / len(last10)

        # Streak
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

        return wins, win_rate, streak

    def _home_proxy(self, team1: str, team2: str, venue: str, city: str) -> tuple:
        """Home game proxy."""
        team_cities = {
            "Mumbai Indians": ["mumbai"],
            "Chennai Super Kings": ["chennai"],
            "Kolkata Knight Riders": ["kolkata"],
            "Royal Challengers Bengaluru": ["bengaluru", "bangalore"],
            "Delhi Capitals": ["delhi"],
            "Rajasthan Royals": ["jaipur"],
            "Punjab Kings": ["mohali", "chandigarh", "dharamsala"],
            "Sunrisers Hyderabad": ["hyderabad"],
            "Gujarat Titans": ["ahmedabad", "rajkot"],
            "Lucknow Super Giants": ["lucknow"],
        }

        t1_home = 0
        t2_home = 0
        if city and not pd.isna(city):
            city_lower = city.lower()
            venue_lower = venue.lower() if venue else ""
            for c in team_cities.get(team1, []):
                if c in city_lower or c in venue_lower:
                    t1_home = 1
                    break
            for c in team_cities.get(team2, []):
                if c in city_lower or c in venue_lower:
                    t2_home = 1
                    break

        return t1_home, t2_home

    def _season_phase(self, match_id: int, season) -> str:
        """Season phase."""
        try:
            season_int = int(season) if not isinstance(season, int) else season
        except:
            return "unknown"

        season_matches = self.matches[self.matches["season"] == season].sort_values("match_id")
        if season_matches.empty:
            return "unknown"

        total = len(season_matches)
        match_positions = season_matches[season_matches["match_id"] == match_id].index
        if len(match_positions) == 0:
            return "unknown"

        pos = list(season_matches.index).index(match_positions[0])
        ratio = pos / total

        if ratio < 0.33:
            return "early"
        elif ratio < 0.66:
            return "mid"
        elif ratio < 0.85:
            return "late"
        else:
            return "playoffs"

    def build_live_dataset_fast(self, sample_rate: int = 12) -> pd.DataFrame:
        """Build live match dataset efficiently with sampling."""
        records = []

        # Pre-compute innings totals
        innings_totals = self.balls.groupby(["match_id", "innings"])["total_runs"].sum().reset_index()
        innings_totals.columns = ["match_id", "innings", "innings_total"]

        # Group balls by match and innings
        grouped = self.balls.groupby(["match_id", "innings"])

        for (match_id, innings), group in grouped:
            match_info = self.matches[self.matches["match_id"] == match_id]
            if match_info.empty:
                continue

            row = match_info.iloc[0]
            winner = row.get("winner", None)
            if pd.isna(winner):
                continue

            # Get innings total for target
            inn_total_row = innings_totals[
                (innings_totals["match_id"] == match_id) &
                (innings_totals["innings"] == innings)
            ]
            inn_total = inn_total_row["innings_total"].values[0] if len(inn_total_row) > 0 else 0

            # Pre-compute cumulative stats for this innings
            group = group.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
            group["cum_runs"] = group["total_runs"].cumsum()
            group["cum_wickets"] = group["is_wicket"].cumsum()
            group["cum_balls"] = range(1, len(group) + 1)

            # Sample at intervals
            sample_indices = list(range(0, len(group), sample_rate))
            if len(group) - 1 not in sample_indices:
                sample_indices.append(len(group) - 1)

            batting_team = group["batting_team"].iloc[0]
            bowling_team = group["bowling_team"].iloc[0]
            venue = row.get("venue", "Unknown")
            toss_winner = row.get("toss_winner", None)
            toss_decision = row.get("toss_decision", None)

            # Get venue stats
            venue_row = self.venue_stats_df[self.venue_stats_df["match_id"] == match_id]
            venue_avg = venue_row["venue_avg_first_inn"].values[0] if len(venue_row) > 0 else 160

            for idx in sample_indices:
                ball = group.iloc[idx]
                over = ball["over"]
                total_runs = ball["cum_runs"]
                wickets = ball["cum_wickets"]
                balls_bowled = idx + 1

                current_rr = (total_runs / balls_bowled * 6) if balls_bowled > 0 else 0
                balls_remaining = 120 - balls_bowled

                # Phase
                is_pp = 1 if over < 6 else 0
                is_mid = 1 if 6 <= over < 16 else 0
                is_death = 1 if over >= 16 else 0

                # Second innings
                if innings == 2:
                    target = inn_total + 1 if inn_total > 0 else 0
                    runs_req = max(0, target - total_runs)
                    req_rr = (runs_req / (balls_remaining / 6)) if balls_remaining > 0 else 0
                    is_chasing = 1
                else:
                    target = 0
                    runs_req = 0
                    req_rr = 0
                    is_chasing = 0

                # Dot/boundary pct (rolling window of last 30 balls)
                window_start = max(0, idx - 29)
                window = group.iloc[window_start:idx + 1]
                dots = (window["total_runs"] == 0).sum()
                boundaries = window["batter_runs"].isin([4, 6]).sum()
                dot_pct = dots / len(window) if len(window) > 0 else 0
                boundary_pct = boundaries / len(window) if len(window) > 0 else 0

                # Last 5 overs RR
                last5_start = max(0, over - 4)
                last5 = group[(group["over"] >= last5_start) & (group.index <= idx)]
                last5_rr = (last5["total_runs"].sum() / len(last5) * 6) if len(last5) > 0 else 0

                # Team strength (pre-match, cached)
                participants = self.registry.get_match_players(match_id)
                bat_players = participants.get(batting_team, [])
                bowl_players = participants.get(bowling_team, [])

                bat_str = self._quick_team_bat_strength(bat_players, match_id)
                bowl_str = self._quick_team_bowl_strength(bowl_players, match_id)

                record = {
                    "match_id": match_id,
                    "innings": innings,
                    "over": over,
                    "current_score": total_runs,
                    "wickets_lost": wickets,
                    "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": current_rr,
                    "is_powerplay": is_pp,
                    "is_middle": is_mid,
                    "is_death": is_death,
                    "target": target,
                    "runs_required": runs_req,
                    "required_run_rate": req_rr,
                    "is_chasing": is_chasing,
                    "dot_ball_pct": dot_pct,
                    "boundary_pct": boundary_pct,
                    "last_5_overs_rr": last5_rr,
                    "batting_team_strength": bat_str,
                    "bowling_team_strength": bowl_str,
                    "venue_avg_first_inn": venue_avg,
                    "batting_team_won_toss": 1 if toss_winner == batting_team else 0,
                    "batting_team_batting_first": 1 if innings == 1 else 0,
                    "target_won": 1 if winner == batting_team else 0,
                    "season": row.get("season", "0"),
                }
                records.append(record)

        df = pd.DataFrame(records)
        print(f"Live dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def _quick_team_bat_strength(self, players: list, match_id: int) -> float:
        """Quick bat strength estimate."""
        strengths = []
        for p in players[:11]:  # Top 11 only for speed
            stats = self.registry.get_player_stats_at_match(p, match_id)
            strengths.append(stats["bat_avg"])
        return np.mean(strengths) if strengths else 0

    def _quick_team_bowl_strength(self, players: list, match_id: int) -> float:
        """Quick bowl strength estimate."""
        strengths = []
        for p in players[:11]:
            stats = self.registry.get_player_stats_at_match(p, match_id)
            if stats["bowl_economy"] > 0:
                strengths.append(1 / (stats["bowl_economy"] + 1))
        return np.mean(strengths) if strengths else 0

    def build_score_dataset_fast(self, sample_rate: int = 12) -> pd.DataFrame:
        """Build score prediction dataset for first innings only."""
        records = []

        first_inn = self.balls[self.balls["innings"] == 1]
        innings_totals = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
        innings_totals.columns = ["match_id", "final_score"]

        grouped = first_inn.groupby("match_id")

        for match_id, group in grouped:
            match_info = self.matches[self.matches["match_id"] == match_id]
            if match_info.empty:
                continue

            final_score_row = innings_totals[innings_totals["match_id"] == match_id]
            if final_score_row.empty:
                continue
            final_score = final_score_row["final_score"].values[0]

            row = match_info.iloc[0]
            venue = row.get("venue", "Unknown")
            venue_row = self.venue_stats_df[self.venue_stats_df["match_id"] == match_id]
            venue_avg = venue_row["venue_avg_first_inn"].values[0] if len(venue_row) > 0 else 160

            batting_team = group["batting_team"].iloc[0]
            participants = self.registry.get_match_players(match_id)
            bat_players = participants.get(batting_team, [])
            bat_str = self._quick_team_bat_strength(bat_players, match_id)

            group = group.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
            group["cum_runs"] = group["total_runs"].cumsum()
            group["cum_wickets"] = group["is_wicket"].cumsum()

            sample_indices = list(range(0, len(group), sample_rate))
            if len(group) - 1 not in sample_indices:
                sample_indices.append(len(group) - 1)

            for idx in sample_indices:
                ball = group.iloc[idx]
                over = ball["over"]
                total_runs = ball["cum_runs"]
                wickets = ball["cum_wickets"]
                balls_bowled = idx + 1
                balls_remaining = 120 - balls_bowled
                current_rr = (total_runs / balls_bowled * 6) if balls_bowled > 0 else 0

                record = {
                    "match_id": match_id,
                    "over": over,
                    "current_score": total_runs,
                    "wickets_lost": wickets,
                    "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": current_rr,
                    "is_powerplay": 1 if over < 6 else 0,
                    "is_middle": 1 if 6 <= over < 16 else 0,
                    "is_death": 1 if over >= 16 else 0,
                    "batting_team_strength": bat_str,
                    "venue_avg_first_inn": venue_avg,
                    "season": row.get("season", "0"),
                    "final_score": final_score,
                }
                records.append(record)

        df = pd.DataFrame(records)
        print(f"Score prediction dataset: {len(df)} samples")
        return df

    def build_player_dataset_fast(self) -> pd.DataFrame:
        """Build player performance dataset."""
        records = []

        for _, match in self.matches.sort_values("match_id").iterrows():
            match_id = match["match_id"]
            team1 = match["team1"]
            team2 = match["team2"]
            venue = match.get("venue", "Unknown")
            season = match.get("season", "0")

            match_balls = self.balls[self.balls["match_id"] == match_id]
            participants = self.registry.get_match_players(match_id)

            for team in [team1, team2]:
                opposition = team2 if team == team1 else team1
                players = participants.get(team, [])

                for player in players:
                    bat = match_balls[
                        (match_balls["batter"] == player) &
                        (match_balls["batting_team"] == team)
                    ]
                    bowl = match_balls[
                        (match_balls["bowler"] == player) &
                        (match_balls["bowling_team"] == team)
                    ]

                    actual_runs = bat["batter_runs"].sum() if not bat.empty else None
                    actual_wickets = bowl["is_wicket"].sum() if not bowl.empty else None

                    if actual_runs is None and actual_wickets is None:
                        continue

                    short_form = self.registry.get_player_form(player, match_id, window=5)
                    long_form = self.registry.get_player_form(player, match_id, window=100)
                    venue_stats = self.registry.get_player_venue_stats(player, venue, match_id)
                    vs_stats = self.registry.get_player_vs_team_stats(player, opposition, match_id)

                    feat = {
                        "match_id": match_id,
                        "player": player,
                        "season": season,
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
                        "actual_runs": actual_runs,
                        "actual_wickets": actual_wickets,
                    }
                    records.append(feat)

        df = pd.DataFrame(records)
        print(f"Player performance dataset: {len(df)} samples")
        return df
