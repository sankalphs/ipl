"""
Player Registry Module (Optimized)
Builds player profiles using vectorized pandas operations.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


class PlayerRegistry:
    """
    Builds and maintains player profiles from ball-by-ball data.
    Uses vectorized operations for speed.
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
        self._build_match_participants_fast()
        self._build_player_match_team()
        self._detect_impact_players()
        self._build_player_profiles_fast()
        self._built = True
        print(f"Registry built: {len(self.player_profiles)} players, "
              f"{len(self.match_participants)} matches")

    def _build_match_participants_fast(self):
        """Build match participants using vectorized operations."""
        # Get batting team info per ball (first occurrence per match+innings gives team)
        innings_info = self.balls.groupby(["match_id", "innings"]).agg(
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
        ).reset_index()

        # Build player-team mapping from all ball records
        # Batters
        bat_records = self.balls[["match_id", "innings", "batter", "batting_team"]].drop_duplicates()
        bat_records.columns = ["match_id", "innings", "player", "team"]

        # Non-strikers
        ns_records = self.balls[["match_id", "innings", "non_striker", "batting_team"]].drop_duplicates()
        ns_records.columns = ["match_id", "innings", "player", "team"]

        # Bowlers
        bowl_records = self.balls[["match_id", "innings", "bowler", "bowling_team"]].drop_duplicates()
        bowl_records.columns = ["match_id", "innings", "player", "team"]

        # Wicket player out
        out_mask = self.balls["wicket_player_out"].notna() & (self.balls["wicket_player_out"] != "None")
        out_records = self.balls[out_mask][["match_id", "innings", "wicket_player_out", "batting_team"]].drop_duplicates()
        out_records.columns = ["match_id", "innings", "player", "team"]

        # Combine all
        all_records = pd.concat([bat_records, ns_records, bowl_records, out_records], ignore_index=True)
        all_records = all_records.drop_duplicates(subset=["match_id", "player"])

        # Group by match_id to get participants
        for match_id, group in all_records.groupby("match_id"):
            participants = {}
            for _, row in group.iterrows():
                player = row["player"]
                team = row["team"]
                if team not in participants:
                    participants[team] = set()
                participants[team].add(player)
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
        """Detect Impact Player substitutions."""
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

    def _build_player_profiles_fast(self):
        """Build player profiles using vectorized operations."""
        print("  Computing batting stats...")
        self._build_batting_stats()
        print("  Computing bowling stats...")
        self._build_bowling_stats()
        print("  Computing player-match team mapping...")
        self._finalize_profiles()

    def _build_batting_stats(self):
        """Build batting stats using vectorized operations."""
        # Get all batting records
        bat_df = self.balls[["match_id", "batter", "batting_team", "batter_runs", "is_wicket",
                              "is_powerplay", "is_middle_overs", "is_death_overs"]].copy()
        bat_df.columns = ["match_id", "player", "team", "runs", "is_wicket",
                           "is_pp", "is_mid", "is_death"]

        # Group by player and match
        bat_per_match = bat_df.groupby(["player", "match_id"]).agg(
            runs=("runs", "sum"),
            balls=("runs", "count"),
            outs=("is_wicket", "sum"),
            fours=("runs", lambda x: (x == 4).sum()),
            sixes=("runs", lambda x: (x == 6).sum()),
            pp_runs=("is_pp", lambda x: (bat_df.loc[x.index, "runs"] * x).sum()),
            mid_runs=("is_mid", lambda x: (bat_df.loc[x.index, "runs"] * x).sum()),
            death_runs=("is_death", lambda x: (bat_df.loc[x.index, "runs"] * x).sum()),
        ).reset_index()

        # Store per-match batting for each player
        for _, row in bat_per_match.iterrows():
            player = row["player"]
            if player not in self.player_profiles:
                self.player_profiles[player] = self._empty_profile(player)

            profile = self.player_profiles[player]
            match_id = row["match_id"]

            profile["batting"]["innings"] += 1
            profile["batting"]["runs"] += row["runs"]
            profile["batting"]["balls_faced"] += row["balls"]
            profile["batting"]["outs"] += row["outs"]
            profile["batting"]["fours"] += row["fours"]
            profile["batting"]["sixes"] += row["sixes"]
            profile["batting"]["scores_per_match"][match_id] = row["runs"]

            if row["runs"] > profile["batting"]["highest_score"]:
                profile["batting"]["highest_score"] = row["runs"]
            if row["runs"] >= 50:
                profile["batting"]["fifties"] += 1
            if row["runs"] >= 100:
                profile["batting"]["hundreds"] += 1

    def _build_bowling_stats(self):
        """Build bowling stats using vectorized operations."""
        bowl_df = self.balls[["match_id", "bowler", "bowling_team", "total_runs", "is_wicket",
                               "wides", "noballs", "is_powerplay", "is_middle_overs", "is_death_overs"]].copy()
        bowl_df.columns = ["match_id", "player", "team", "runs_conceded", "is_wicket",
                            "wide", "noball", "is_pp", "is_mid", "is_death"]

        # Legal balls (not wide, not noball)
        bowl_df["is_legal"] = ((bowl_df["wide"] == 0) & (bowl_df["noball"] == 0)).astype(int)

        bowl_per_match = bowl_df.groupby(["player", "match_id"]).agg(
            runs_conceded=("runs_conceded", "sum"),
            wickets=("is_wicket", "sum"),
            balls_bowled=("is_legal", "sum"),
            pp_wickets=("is_pp", lambda x: (bowl_df.loc[x.index, "is_wicket"] * x).sum()),
            mid_wickets=("is_mid", lambda x: (bowl_df.loc[x.index, "is_wicket"] * x).sum()),
            death_wickets=("is_death", lambda x: (bowl_df.loc[x.index, "is_wicket"] * x).sum()),
        ).reset_index()

        for _, row in bowl_per_match.iterrows():
            player = row["player"]
            if player not in self.player_profiles:
                self.player_profiles[player] = self._empty_profile(player)

            profile = self.player_profiles[player]
            match_id = row["match_id"]

            profile["bowling"]["innings"] += 1
            profile["bowling"]["balls_bowled"] += row["balls_bowled"]
            profile["bowling"]["runs_conceded"] += row["runs_conceded"]
            profile["bowling"]["wickets"] += row["wickets"]
            profile["bowling"]["overs_bowled"] += row["balls_bowled"] / 6
            profile["bowling"]["wickets_per_match"][match_id] = row["wickets"]

            if row["balls_bowled"] > 0:
                economy = (row["runs_conceded"] / row["balls_bowled"]) * 6
                profile["bowling"]["economy_per_match"][match_id] = economy

    def _finalize_profiles(self):
        """Finalize player profiles with match lists and team mappings."""
        # Build match list per player from batting and bowling
        bat_matches = self.balls.groupby("batter")["match_id"].apply(list).to_dict()
        bowl_matches = self.balls.groupby("bowler")["match_id"].apply(list).to_dict()

        for player, profile in self.player_profiles.items():
            # Combine match lists
            bat_m = set(bat_matches.get(player, []))
            bowl_m = set(bowl_matches.get(player, []))
            all_matches = sorted(bat_m | bowl_m)
            profile["matches"] = all_matches

            # Build team mapping from match_participants
            for match_id in all_matches:
                team = self.get_player_team(match_id, player)
                if team:
                    season_row = self.matches[self.matches["match_id"] == match_id]
                    if not season_row.empty:
                        season = season_row["season"].iloc[0]
                        if season not in profile["teams"]:
                            profile["teams"][season] = {}
                        profile["teams"][season][match_id] = team

            # Determine all-rounder
            if profile["batting"]["innings"] >= 5 and profile["bowling"]["innings"] >= 5:
                profile["is_allrounder"] = True

    def _empty_profile(self, player: str) -> dict:
        """Create empty player profile."""
        return {
            "name": player,
            "matches": [],
            "teams": {},
            "seasons": set(),
            "venues": set(),
            "batting": {
                "innings": 0, "runs": 0, "balls_faced": 0,
                "fours": 0, "sixes": 0, "outs": 0, "not_outs": 0,
                "fifties": 0, "hundreds": 0, "highest_score": 0,
                "scores_per_match": {}, "dismissal_types": defaultdict(int),
            },
            "bowling": {
                "innings": 0, "balls_bowled": 0, "runs_conceded": 0,
                "wickets": 0, "maidens": 0, "overs_bowled": 0,
                "wickets_per_match": {}, "economy_per_match": {},
            },
            "phases": {
                "powerplay": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
                "middle": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
                "death": {"runs": 0, "balls": 0, "wickets": 0, "balls_bowled": 0},
            },
            "is_allrounder": False,
        }

    def get_player_team(self, match_id: int, player: str) -> str | None:
        return self.player_match_team.get(match_id, {}).get(player, None)

    def get_match_players(self, match_id: int) -> dict[str, list[str]]:
        return self.match_participants.get(match_id, {})

    def get_player_profile(self, player: str) -> dict | None:
        return self.player_profiles.get(player, None)

    def get_is_impact_match(self, match_id: int) -> dict[str, bool]:
        return self.impact_player_matches.get(match_id, {})

    def get_player_stats_at_match(self, player: str, match_id: int) -> dict:
        """Get player's cumulative stats BEFORE a specific match."""
        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        match_list = profile["matches"]
        if match_id not in match_list:
            return self._empty_stats()

        match_idx = match_list.index(match_id)
        prior_matches = match_list[:match_idx]

        if not prior_matches:
            return self._empty_stats()

        # Use pre-computed scores_per_match
        prior_bat_runs = sum(
            profile["batting"]["scores_per_match"].get(m, 0)
            for m in prior_matches
        )
        prior_bat_innings = sum(
            1 for m in prior_matches
            if m in profile["batting"]["scores_per_match"]
        )
        prior_bowl_wickets = sum(
            profile["bowling"]["wickets_per_match"].get(m, 0)
            for m in prior_matches
        )
        prior_bowl_innings = sum(
            1 for m in prior_matches
            if m in profile["bowling"]["economy_per_match"]
        )

        # Calculate average economy from prior matches
        prior_economies = [
            profile["bowling"]["economy_per_match"][m]
            for m in prior_matches
            if m in profile["bowling"]["economy_per_match"]
        ]
        avg_economy = np.mean(prior_economies) if prior_economies else 0

        # Batting average (simplified: runs per innings)
        bat_avg = prior_bat_runs / prior_bat_innings if prior_bat_innings > 0 else 0

        # Strike rate estimate
        # Use career SR as proxy since we don't have per-match balls
        career_sr = (profile["batting"]["runs"] / profile["batting"]["balls_faced"] * 100) \
            if profile["batting"]["balls_faced"] > 0 else 0

        return {
            "matches_played": len(prior_matches),
            "bat_innings": prior_bat_innings,
            "bat_runs": prior_bat_runs,
            "bat_balls": 0,  # Not tracked per match
            "bat_avg": bat_avg,
            "bat_sr": career_sr,
            "bat_fours": 0,
            "bat_sixes": 0,
            "bowl_innings": prior_bowl_innings,
            "bowl_wickets": prior_bowl_wickets,
            "bowl_balls": 0,
            "bowl_runs_conceded": 0,
            "bowl_economy": avg_economy,
            "is_allrounder": profile["is_allrounder"],
        }

    def _empty_stats(self) -> dict:
        return {
            "matches_played": 0, "bat_innings": 0, "bat_runs": 0, "bat_balls": 0,
            "bat_avg": 0, "bat_sr": 0, "bat_fours": 0, "bat_sixes": 0,
            "bowl_innings": 0, "bowl_wickets": 0, "bowl_balls": 0,
            "bowl_runs_conceded": 0, "bowl_economy": 0, "is_allrounder": False,
        }

    def get_player_form(self, player: str, match_id: int, window: int = 5) -> dict:
        """Get player's recent form."""
        profile = self.player_profiles.get(player)
        if not profile:
            return {"short_form": self._empty_stats(), "long_form": self._empty_stats()}

        match_list = profile["matches"]
        if match_id not in match_list:
            return {"short_form": self._empty_stats(), "long_form": self._empty_stats()}

        match_idx = match_list.index(match_id)
        prior_matches = match_list[:match_idx]

        short_matches = prior_matches[-window:] if len(prior_matches) >= window else prior_matches
        short_stats = self._calc_form_stats(player, short_matches)
        long_stats = self._calc_form_stats(player, prior_matches)

        return {"short_form": short_stats, "long_form": long_stats}

    def _calc_form_stats(self, player: str, match_ids: list[int]) -> dict:
        """Calculate form stats for a specific set of matches."""
        if not match_ids:
            return self._empty_stats()

        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        bat_innings = sum(1 for m in match_ids if m in profile["batting"]["scores_per_match"])
        bat_runs = sum(profile["batting"]["scores_per_match"].get(m, 0) for m in match_ids)
        bowl_innings = sum(1 for m in match_ids if m in profile["bowling"]["economy_per_match"])
        bowl_wickets = sum(profile["bowling"]["wickets_per_match"].get(m, 0) for m in match_ids)

        economies = [profile["bowling"]["economy_per_match"][m] for m in match_ids
                      if m in profile["bowling"]["economy_per_match"]]
        avg_economy = np.mean(economies) if economies else 0

        bat_avg = bat_runs / bat_innings if bat_innings > 0 else 0

        # Use career SR
        career_sr = (profile["batting"]["runs"] / profile["batting"]["balls_faced"] * 100) \
            if profile["batting"]["balls_faced"] > 0 else 0

        return {
            "matches_played": len(match_ids),
            "bat_innings": bat_innings,
            "bat_runs": bat_runs,
            "bat_balls": 0,
            "bat_avg": bat_avg,
            "bat_sr": career_sr,
            "bat_fours": 0,
            "bat_sixes": 0,
            "bowl_innings": bowl_innings,
            "bowl_wickets": bowl_wickets,
            "bowl_balls": 0,
            "bowl_runs_conceded": 0,
            "bowl_economy": avg_economy,
            "is_allrounder": profile["is_allrounder"],
        }

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

        # Filter matches at this venue
        venue_matches = []
        for mid in prior_matches:
            match_info = self.matches[self.matches["match_id"] == mid]
            if not match_info.empty and match_info["venue"].iloc[0] == venue:
                venue_matches.append(mid)

        return self._calc_form_stats(player, venue_matches)

    def get_player_vs_team_stats(self, player: str, opposition: str, before_match_id: int) -> dict:
        """Get player's stats against a specific opposition."""
        profile = self.player_profiles.get(player)
        if not profile:
            return self._empty_stats()

        match_list = profile["matches"]
        if before_match_id not in match_list:
            return self._empty_stats()

        match_idx = match_list.index(before_match_id)
        prior_matches = match_list[:match_idx]

        vs_matches = []
        for mid in prior_matches:
            player_team = self.get_player_team(mid, player)
            match_info = self.matches[self.matches["match_id"] == mid]
            if match_info.empty:
                continue
            team1 = match_info["team1"].iloc[0]
            team2 = match_info["team2"].iloc[0]
            if (player_team == team1 and team2 == opposition) or \
               (player_team == team2 and team1 == opposition):
                vs_matches.append(mid)

        return self._calc_form_stats(player, vs_matches)
