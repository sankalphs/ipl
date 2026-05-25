"""
Player Registry Module
Builds player profiles, tracks player-team assignments per match,
and detects Impact Player substitutions.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


class PlayerRegistry:
    """
    Builds and maintains player profiles from ball-by-ball data.
    Tracks player-team assignments per match (handles trades).
    Detects Impact Player substitutions.
    """

    def __init__(self, balls_df: pd.DataFrame, matches_df: pd.DataFrame):
        self.balls = balls_df
        self.matches = matches_df
        self.player_match_team: dict[int, dict[str, str]] = {}
        self.match_participants: dict[int, dict[str, list[str]]] = {}
        self.player_profiles: dict[str, dict] = {}
        self.impact_player_matches: dict[int, dict[str, bool]] = {}
        self._built = False

    def build(self):
        """Build the complete player registry."""
        print("Building player registry...")
        self._build_match_participants()
        self._build_player_match_team()
        self._detect_impact_players()
        self._build_player_profiles()
        self._built = True
        print(f"Registry built: {len(self.player_profiles)} players, "
              f"{len(self.match_participants)} matches")

    def _build_match_participants(self):
        """For each match, extract unique players per team."""
        for match_id in self.balls["match_id"].unique():
            match_balls = self.balls[self.balls["match_id"] == match_id]
            participants = {}

            for innings in [1, 2]:
                innings_balls = match_balls[match_balls["innings"] == innings]
                if innings_balls.empty:
                    continue

                batting_team = innings_balls["batting_team"].iloc[0]
                bowling_team = innings_balls["bowling_team"].iloc[0]

                # Collect all players from batting team
                bat_players = set()
                for col in ["batter", "non_striker"]:
                    bat_players.update(innings_balls[col].dropna().unique())

                # Collect all players from bowling team
                bowl_players = set(innings_balls["bowler"].dropna().unique())

                # Also check wicket_player_out for batting team players
                out_players = innings_balls[
                    innings_balls["wicket_player_out"].notna() &
                    (innings_balls["wicket_player_out"] != "None")
                ]["wicket_player_out"].unique()
                bat_players.update(out_players)

                if batting_team not in participants:
                    participants[batting_team] = set()
                if bowling_team not in participants:
                    participants[bowling_team] = set()

                participants[batting_team].update(bat_players)
                participants[bowling_team].update(bowl_players)

            # Convert sets to lists
            self.match_participants[match_id] = {
                team: sorted(list(players))
                for team, players in participants.items()
            }

    def _build_player_match_team(self):
        """Build player -> team mapping for each match."""
        for match_id, teams in self.match_participants.items():
            self.player_match_team[match_id] = {}
            for team, players in teams.items():
                for player in players:
                    self.player_match_team[match_id][player] = team

    def _detect_impact_players(self):
        """
        Detect Impact Player substitutions.
        Logic: If a team has 12+ unique participants in a match,
        and the match is from 2023+ season, flag as impact player match.
        """
        for match_id, teams in self.match_participants.items():
            match_info = self.matches[self.matches["match_id"] == match_id]
            if match_info.empty:
                continue

            season = match_info["season"].iloc[0]
            try:
                season_int = int(season)
            except (ValueError, TypeError):
                season_int = 0
            if season_int < 2023:
                continue

            impact_info = {}
            for team, players in teams.items():
                if len(players) >= 12:
                    impact_info[team] = True
                else:
                    impact_info[team] = False

            if any(impact_info.values()):
                self.impact_player_matches[match_id] = impact_info

    def get_impact_sub_candidates(self, match_id: int, team: str) -> list[str]:
        """
        Infer which player is the impact sub.
        Heuristic: The player who appears ONLY in one innings as batter/bowler
        but NOT in the other, and has fewer total appearances, is likely the sub.
        """
        if match_id not in self.impact_player_matches:
            return []
        if not self.impact_player_matches[match_id].get(team, False):
            return []

        match_balls = self.balls[self.balls["match_id"] == match_id]

        # Get team's players in each innings
        team_players_innings = {}
        for innings in [1, 2]:
            inn_balls = match_balls[match_balls["innings"] == innings]
            players = set()
            for col in ["batter", "non_striker", "bowler"]:
                players.update(inn_balls[col].dropna().unique())
            # Filter to team's players
            team_players = set()
            for p in players:
                p_team = self.get_player_team(match_id, p)
                if p_team == team:
                    team_players.add(p)
            team_players_innings[innings] = team_players

        # Players who only appeared in one innings are likely impact subs
        only_inn1 = team_players_innings.get(1, set()) - team_players_innings.get(2, set())
        only_inn2 = team_players_innings.get(2, set()) - team_players_innings.get(1, set())

        # Also check: players with fewer ball appearances
        candidates = list(only_inn1 | only_inn2)
        if not candidates:
            # Fallback: find player with fewest ball appearances
            all_team_players = self.match_participants.get(match_id, {}).get(team, [])
            ball_counts = {}
            for p in all_team_players:
                count = len(match_balls[
                    (match_balls["batter"] == p) |
                    (match_balls["bowler"] == p) |
                    (match_balls["non_striker"] == p)
                ])
                ball_counts[p] = count
            if ball_counts:
                min_count = min(ball_counts.values())
                candidates = [p for p, c in ball_counts.items() if c == min_count]

        return candidates

    def _build_player_profiles(self):
        """Build career profiles for all players."""
        for match_id, team_map in self.player_match_team.items():
            match_info = self.matches[self.matches["match_id"] == match_id]
            if match_info.empty:
                continue

            match_balls = self.balls[self.balls["match_id"] == match_id]
            season = match_info["season"].iloc[0]
            venue = match_info["venue"].iloc[0] if "venue" in match_info.columns else "Unknown"

            for player, team in team_map.items():
                if player not in self.player_profiles:
                    self.player_profiles[player] = {
                        "name": player,
                        "matches": [],
                        "teams": {},
                        "seasons": set(),
                        "venues": set(),
                        "batting": {
                            "innings": 0,
                            "runs": 0,
                            "balls_faced": 0,
                            "fours": 0,
                            "sixes": 0,
                            "outs": 0,
                            "not_outs": 0,
                            "fifties": 0,
                            "hundreds": 0,
                            "highest_score": 0,
                            "scores_per_match": {},
                            "dismissal_types": defaultdict(int),
                        },
                        "bowling": {
                            "innings": 0,
                            "balls_bowled": 0,
                            "runs_conceded": 0,
                            "wickets": 0,
                            "maidens": 0,
                            "overs_bowled": 0,
                            "wickets_per_match": {},
                            "economy_per_match": {},
                        },
                        "phases": {
                            "powerplay": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
                            "middle": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
                            "death": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
                        },
                        "is_allrounder": False,
                    }

                profile = self.player_profiles[player]
                profile["matches"].append(match_id)
                profile["seasons"].add(season)
                profile["venues"].add(venue)

                # Track team per match (handles trades)
                if season not in profile["teams"]:
                    profile["teams"][season] = {}
                profile["teams"][season][match_id] = team

                # Update batting stats from this match
                player_bat = match_balls[
                    (match_balls["batter"] == player) &
                    (match_balls["batting_team"] == team)
                ]
                if not player_bat.empty:
                    runs = player_bat["batter_runs"].sum()
                    balls_faced = len(player_bat[player_bat["wide"] == 0]) if "wide" in player_bat.columns else len(player_bat)
                    fours = len(player_bat[player_bat["batter_runs"] == 4])
                    sixes = len(player_bat[player_bat["batter_runs"] == 6])

                    profile["batting"]["innings"] += 1
                    profile["batting"]["runs"] += runs
                    profile["batting"]["balls_faced"] += balls_faced
                    profile["batting"]["fours"] += fours
                    profile["batting"]["sixes"] += sixes
                    profile["batting"]["scores_per_match"][match_id] = runs

                    if runs > profile["batting"]["highest_score"]:
                        profile["batting"]["highest_score"] = runs
                    if runs >= 50:
                        profile["batting"]["fifties"] += 1
                    if runs >= 100:
                        profile["batting"]["hundreds"] += 1

                    # Phase-wise batting
                    for phase_key, phase_col in [
                        ("powerplay", "is_powerplay"),
                        ("middle", "is_middle_overs"),
                        ("death", "is_death_overs"),
                    ]:
                        if phase_col in player_bat.columns:
                            phase_balls = player_bat[player_bat[phase_col] == 1]
                            profile["phases"][phase_key]["runs"] += phase_balls["batter_runs"].sum()
                            profile["phases"][phase_key]["balls"] += len(phase_balls)

                # Check dismissal
                out_row = match_balls[
                    (match_balls["wicket_player_out"] == player) &
                    (match_balls["batting_team"] == team)
                ]
                if not out_row.empty:
                    profile["batting"]["outs"] += 1
                    dismissal = out_row["wicket_kind"].iloc[0]
                    if dismissal != "None":
                        profile["batting"]["dismissal_types"][dismissal] += 1
                elif not player_bat.empty:
                    profile["batting"]["not_outs"] += 1

                # Update bowling stats from this match
                player_bowl = match_balls[
                    (match_balls["bowler"] == player) &
                    (match_balls["bowling_team"] == team)
                ]
                if not player_bowl.empty:
                    if "wide" in player_bowl.columns:
                        legal_balls = player_bowl[player_bowl["wide"] == 0]
                    else:
                        legal_balls = player_bowl
                    if "noballs" in player_bowl.columns:
                        legal_balls = legal_balls[legal_balls["noballs"] == 0]

                    runs_conceded = player_bowl["total_runs"].sum()
                    wickets = player_bowl["is_wicket"].sum()
                    balls_bowled = len(legal_balls)

                    profile["bowling"]["innings"] += 1
                    profile["bowling"]["balls_bowled"] += balls_bowled
                    profile["bowling"]["runs_conceded"] += runs_conceded
                    profile["bowling"]["wickets"] += wickets
                    profile["bowling"]["overs_bowled"] += balls_bowled / 6
                    profile["bowling"]["wickets_per_match"][match_id] = wickets

                    if balls_bowled > 0:
                        economy = (runs_conceded / balls_bowled) * 6
                        profile["bowling"]["economy_per_match"][match_id] = economy

                    # Phase-wise bowling
                    for phase_key, phase_col in [
                        ("powerplay", "is_powerplay"),
                        ("middle", "is_middle_overs"),
                        ("death", "is_death_overs"),
                    ]:
                        if phase_col in player_bowl.columns:
                            phase_balls = player_bowl[player_bowl[phase_col] == 1]
                            profile["phases"][phase_key]["wickets"] += phase_balls["is_wicket"].sum()
                            profile["phases"][phase_key]["balls_bowled"] += len(phase_balls)

        # Determine all-rounders
        for player, profile in self.player_profiles.items():
            bat_innings = profile["batting"]["innings"]
            bowl_innings = profile["bowling"]["innings"]
            if bat_innings >= 5 and bowl_innings >= 5:
                profile["is_allrounder"] = True

    def get_player_team(self, match_id: int, player: str) -> str | None:
        """Get the team a player was playing for in a specific match."""
        return self.player_match_team.get(match_id, {}).get(player, None)

    def get_match_players(self, match_id: int) -> dict[str, list[str]]:
        """Get all players per team for a match."""
        return self.match_participants.get(match_id, {})

    def get_player_profile(self, player: str) -> dict | None:
        """Get a player's career profile."""
        return self.player_profiles.get(player, None)

    def get_player_team_at_season(self, player: str, season: int) -> str | None:
        """Get the team a player was on for a given season (most frequent team)."""
        profile = self.player_profiles.get(player)
        if not profile:
            return None
        season_teams = profile["teams"].get(season, {})
        if not season_teams:
            return None
        # Return most frequent team in that season
        from collections import Counter
        team_counts = Counter(season_teams.values())
        return team_counts.most_common(1)[0][0]

    def get_is_impact_match(self, match_id: int) -> dict[str, bool]:
        """Check if a match had impact player substitutions."""
        return self.impact_player_matches.get(match_id, {})

    def get_player_stats_at_match(self, player: str, match_id: int) -> dict:
        """
        Get player's cumulative stats BEFORE a specific match.
        Used for feature engineering (avoids data leakage).
        """
        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        # Get match index in player's career
        match_list = profile["matches"]
        if match_id not in match_list:
            return self._empty_stats()

        match_idx = match_list.index(match_id)
        prior_matches = match_list[:match_idx]

        if not prior_matches:
            return self._empty_stats()

        # Calculate stats from prior matches only
        prior_balls = self.balls[
            (self.balls["match_id"].isin(prior_matches)) &
            ((self.balls["batter"] == player) | (self.balls["bowler"] == player))
        ]

        bat = prior_balls[prior_balls["batter"] == player]
        bowl = prior_balls[prior_balls["bowler"] == player]

        stats = {
            "matches_played": len(prior_matches),
            "bat_innings": len(bat["match_id"].unique()) if not bat.empty else 0,
            "bat_runs": bat["batter_runs"].sum() if not bat.empty else 0,
            "bat_balls": len(bat) if not bat.empty else 0,
            "bat_avg": (
                bat["batter_runs"].sum() / max(1, len(bat[bat["is_wicket"] == 1]))
                if not bat.empty else 0
            ),
            "bat_sr": (
                (bat["batter_runs"].sum() / len(bat)) * 100
                if not bat.empty and len(bat) > 0 else 0
            ),
            "bat_fours": len(bat[bat["batter_runs"] == 4]) if not bat.empty else 0,
            "bat_sixes": len(bat[bat["batter_runs"] == 6]) if not bat.empty else 0,
            "bowl_innings": len(bowl["match_id"].unique()) if not bowl.empty else 0,
            "bowl_wickets": bowl["is_wicket"].sum() if not bowl.empty else 0,
            "bowl_balls": len(bowl) if not bowl.empty else 0,
            "bowl_runs_conceded": bowl["total_runs"].sum() if not bowl.empty else 0,
            "bowl_economy": (
                (bowl["total_runs"].sum() / len(bowl)) * 6
                if not bowl.empty and len(bowl) > 0 else 0
            ),
            "is_allrounder": False,
        }

        # Determine all-rounder
        if stats["bat_innings"] >= 5 and stats["bowl_innings"] >= 5:
            stats["is_allrounder"] = True

        return stats

    def _empty_stats(self) -> dict:
        """Return empty stats dict for a new player."""
        return {
            "matches_played": 0,
            "bat_innings": 0,
            "bat_runs": 0,
            "bat_balls": 0,
            "bat_avg": 0,
            "bat_sr": 0,
            "bat_fours": 0,
            "bat_sixes": 0,
            "bowl_innings": 0,
            "bowl_wickets": 0,
            "bowl_balls": 0,
            "bowl_runs_conceded": 0,
            "bowl_economy": 0,
            "is_allrounder": False,
        }

    def get_player_form(self, player: str, match_id: int, window: int = 5) -> dict:
        """
        Get player's recent form (last N matches before match_id).
        Returns both short-term (window) and long-term stats.
        """
        profile = self.player_profiles.get(player)
        if not profile:
            return {"short_form": self._empty_stats(), "long_form": self._empty_stats()}

        match_list = profile["matches"]
        if match_id not in match_list:
            return {"short_form": self._empty_stats(), "long_form": self._empty_stats()}

        match_idx = match_list.index(match_id)
        prior_matches = match_list[:match_idx]

        # Short form (last N matches)
        short_matches = prior_matches[-window:] if len(prior_matches) >= window else prior_matches
        short_stats = self._calc_form_stats(player, short_matches)

        # Long form (all prior matches)
        long_stats = self._calc_form_stats(player, prior_matches)

        return {"short_form": short_stats, "long_form": long_stats}

    def _calc_form_stats(self, player: str, match_ids: list[int]) -> dict:
        """Calculate form stats for a specific set of matches."""
        if not match_ids:
            return self._empty_stats()

        balls_subset = self.balls[
            (self.balls["match_id"].isin(match_ids)) &
            ((self.balls["batter"] == player) | (self.balls["bowler"] == player))
        ]

        bat = balls_subset[balls_subset["batter"] == player]
        bowl = balls_subset[balls_subset["bowler"] == player]

        stats = {
            "matches_played": len(match_ids),
            "bat_innings": len(bat["match_id"].unique()) if not bat.empty else 0,
            "bat_runs": int(bat["batter_runs"].sum()) if not bat.empty else 0,
            "bat_balls": len(bat) if not bat.empty else 0,
            "bat_avg": (
                bat["batter_runs"].sum() / max(1, len(bat[bat["is_wicket"] == 1]))
                if not bat.empty else 0
            ),
            "bat_sr": (
                (bat["batter_runs"].sum() / len(bat)) * 100
                if not bat.empty and len(bat) > 0 else 0
            ),
            "bat_fours": len(bat[bat["batter_runs"] == 4]) if not bat.empty else 0,
            "bat_sixes": len(bat[bat["batter_runs"] == 6]) if not bat.empty else 0,
            "bowl_innings": len(bowl["match_id"].unique()) if not bowl.empty else 0,
            "bowl_wickets": int(bowl["is_wicket"].sum()) if not bowl.empty else 0,
            "bowl_balls": len(bowl) if not bowl.empty else 0,
            "bowl_runs_conceded": int(bowl["total_runs"].sum()) if not bowl.empty else 0,
            "bowl_economy": (
                (bowl["total_runs"].sum() / len(bowl)) * 6
                if not bowl.empty and len(bowl) > 0 else 0
            ),
            "is_allrounder": False,
        }

        if stats["bat_innings"] >= 3 and stats["bowl_innings"] >= 3:
            stats["is_allrounder"] = True

        return stats

    def get_player_venue_stats(self, player: str, venue: str, before_match_id: int) -> dict:
        """Get player's stats at a specific venue before a match."""
        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        match_list = profile["matches"]
        if before_match_id not in match_list:
            return self._empty_stats()

        match_idx = match_list.index(before_match_id)
        prior_matches = match_list[:match_idx]

        # Get venue for each prior match
        venue_matches = []
        for mid in prior_matches:
            match_info = self.matches[self.matches["match_id"] == mid]
            if not match_info.empty and match_info["venue"].iloc[0] == venue:
                venue_matches.append(mid)

        return self._calc_form_stats(player, venue_matches)

    def get_player_vs_team_stats(self, player: str, opposition: str, before_match_id: int) -> dict:
        """Get player's stats against a specific opposition before a match."""
        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        match_list = profile["matches"]
        if before_match_id not in match_list:
            return self._empty_stats()

        match_idx = match_list.index(before_match_id)
        prior_matches = match_list[:match_idx]

        # Filter matches where player played against opposition
        vs_matches = []
        for mid in prior_matches:
            player_team = self.get_player_team(mid, player)
            match_info = self.matches[self.matches["match_id"] == mid]
            if match_info.empty:
                continue
            team1 = match_info["team1"].iloc[0]
            team2 = match_info["team2"].iloc[0]
            # Check if opposition was the other team
            if (player_team == team1 and team2 == opposition) or \
               (player_team == team2 and team1 == opposition):
                vs_matches.append(mid)

        return self._calc_form_stats(player, vs_matches)


def build_registry(data_dir: str | Path = "dataset") -> PlayerRegistry:
    """Convenience function to build registry from data directory."""
    from .preprocessing import preprocess_all

    matches, balls = preprocess_all(data_dir)
    registry = PlayerRegistry(balls, matches)
    registry.build()
    return registry
