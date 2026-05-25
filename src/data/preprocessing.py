"""
IPL Data Preprocessing Module
Handles team normalization, venue normalization, and data cleaning.
"""

import pandas as pd
import numpy as np
from thefuzz import process, fuzz
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Team Name Normalization
# ============================================================

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

# Franchise history mapping (for context, NOT merging)
FRANCHISE_HISTORY = {
    "Delhi Capitals": ["Delhi Daredevils"],
    "Punjab Kings": ["Kings XI Punjab"],
    "Royal Challengers Bengaluru": ["Royal Challengers Bangalore"],
    "Rising Pune Supergiants": ["Rising Pune Supergiant"],
    # These are SEPARATE franchises - do NOT merge
    # "Deccan Chargers" -> defunct
    # "Sunrisers Hyderabad" -> separate franchise
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to canonical form."""
    if pd.isna(name):
        return name
    return TEAM_ALIASES.get(name, name)


def normalize_teams_in_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply team name normalization to specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)
    return df


# ============================================================
# Venue Name Normalization
# ============================================================

VENUE_CANONICAL = {
    # Wankhede variants
    "wankhede stadium": "Wankhede Stadium",
    # Chinnaswamy variants
    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    # Chepauk variants
    "ma chidambaram stadium": "MA Chidambaram Stadium",
    # Eden Gardens
    "eden gardens": "Eden Gardens",
    # Arun Jaitley
    "arun jaitley stadium": "Arun Jaitley Stadium",
    # Feroz Shah Kotla (old name for Arun Jaitley)
    "feroz shah kotla": "Arun Jaitley Stadium",
    # Sawai Mansingh
    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    # DY Patil
    "dr dy patil sports academy": "Dr DY Patil Sports Academy",
    # Maharashtra Cricket Association
    "maharashtra cricket association stadium": "MCA Stadium Pune",
    # Rajiv Gandhi
    "rajiv gandhi international stadium": "Rajiv Gandhi Intl Stadium",
    "rajiv gandhi intl. stadium": "Rajiv Gandhi Intl Stadium",
    # Punjab Cricket Association
    "punjab cricket association is bindra stadium": "PCA Stadium Mohali",
    "punjab cricket association stadium": "PCA Stadium Mohali",
    "is bindra stadium": "PCA Stadium Mohali",
    # Others
    "saurashtra cricket association stadium": "SCA Stadium Rajkot",
    "green park": "Green Park Kanpur",
    "holkar cricket stadium": "Holkar Stadium Indore",
    "vidarbha cricket association stadium": "VCA Stadium Nagpur",
    "newlands": "Newlands Cape Town",
    "superSport park": "SuperSport Park Centurion",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
}


def normalize_venue_name(venue: str) -> str:
    """Normalize venue name to canonical form using fuzzy matching."""
    if pd.isna(venue):
        return venue

    venue_lower = venue.lower().strip()

    # Direct match
    if venue_lower in VENUE_CANONICAL:
        return VENUE_CANONICAL[venue_lower]

    # Check if any canonical key is a substring
    for key, canonical in VENUE_CANONICAL.items():
        if key in venue_lower or venue_lower in key:
            return canonical

    # Return original if no match found (will be handled by fuzzy matching later)
    return venue.strip()


def build_venue_mapping(venues: list[str], threshold: int = 80) -> dict[str, str]:
    """Build venue mapping using fuzzy string matching."""
    canonical_venues = {}
    venue_list = list(set(venues))

    for venue in venue_list:
        if pd.isna(venue):
            continue

        # First try rule-based normalization
        normalized = normalize_venue_name(venue)

        if normalized != venue.strip():
            canonical_venues[venue] = normalized
            continue

        # Try fuzzy matching against existing canonical venues
        if canonical_venues:
            best_match, score = process.extractOne(
                venue, list(set(canonical_venues.values())), scorer=fuzz.token_sort_ratio
            )
            if score >= threshold:
                canonical_venues[venue] = best_match
                continue

        canonical_venues[venue] = venue.strip()

    return canonical_venues


def normalize_venues_in_df(df: pd.DataFrame, column: str = "venue") -> pd.DataFrame:
    """Apply venue normalization to a DataFrame column."""
    df = df.copy()
    if column not in df.columns:
        return df

    # Build mapping from unique venues
    unique_venues = df[column].dropna().unique().tolist()
    venue_map = build_venue_mapping(unique_venues)
    df[column] = df[column].map(venue_map).fillna(df[column])
    return df


# ============================================================
# Data Loading & Cleaning
# ============================================================

def load_raw_data(data_dir: str | Path = "dataset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files."""
    data_dir = Path(data_dir)
    matches = pd.read_csv(data_dir / "ipl_matches.csv")
    balls = pd.read_csv(data_dir / "ipl_ball_by_ball.csv")
    return matches, balls


def load_clean_data(data_dir: str | Path = "dataset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean CSV files."""
    data_dir = Path(data_dir)
    matches = pd.read_csv(data_dir / "ipl_matches_clean.csv")
    balls = pd.read_csv(data_dir / "ipl_ball_by_ball_clean.csv")
    return matches, balls


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize matches DataFrame."""
    df = df.copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize team names
    team_cols = [c for c in ["team1", "team2", "toss_winner", "winner"] if c in df.columns]
    df = normalize_teams_in_df(df, team_cols)

    # Normalize venue
    df = normalize_venues_in_df(df, "venue")

    # Add city to venue if missing
    if "city" in df.columns and "venue" in df.columns:
        df["city"] = df["city"].fillna("Unknown")

    # Drop rows with no winner (no-result matches) for training
    # Keep them in raw data but flag them
    if "winner" in df.columns:
        df["has_result"] = df["winner"].notna()

    return df


def clean_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize ball-by-ball DataFrame."""
    df = df.copy()

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize team names
    team_cols = [
        c for c in ["batting_team", "bowling_team", "toss_winner", "match_winner", "team1", "team2"]
        if c in df.columns
    ]
    df = normalize_teams_in_df(df, team_cols)

    # Normalize venue
    df = normalize_venues_in_df(df, "venue")

    # Ensure numeric columns
    numeric_cols = [
        "match_id", "season", "innings", "over", "ball_in_over",
        "batter_runs", "extra_runs", "total_runs", "wides", "noballs",
        "byes", "legbyes", "penalty", "is_wicket"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN in wicket-related columns
    for col in ["wicket_player_out", "wicket_kind", "wicket_fielders"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


def preprocess_all(data_dir: str | Path = "dataset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline."""
    matches_raw, balls_raw = load_raw_data(data_dir)
    matches = clean_matches(matches_raw)
    balls = clean_balls(balls_raw)
    return matches, balls


# ============================================================
# Constants for downstream use
# ============================================================

# Current IPL teams (as of 2024-2026)
CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

# Defunct teams
DEFUNCT_TEAMS = [
    "Deccan Chargers",
    "Gujarat Lions",
    "Kochi Tuskers Kerala",
    "Pune Warriors",
    "Rising Pune Supergiants",
]
