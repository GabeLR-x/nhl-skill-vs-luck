"""Load the tracked Kaggle CSVs and build one row per team per full season.

Works from the CSVs alone (no nhl.db needed), so the deployed dashboard can
start from a fresh clone. Mirrors the SQL in notebooks/02_skill_vs_luck.ipynb.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FULL_SEASON_GAMES = 82


def load_raw(data_dir=DATA_DIR):
    """Read game, game_teams_stats and team_info, dropping exact duplicate rows
    (the raw CSVs ship with 2,570 and 5,140 of them)."""
    data_dir = Path(data_dir)
    return {
        name: pd.read_csv(data_dir / f"{name}.csv").drop_duplicates().reset_index(drop=True)
        for name in ["game", "game_teams_stats", "team_info"]
    }


def team_seasons(raw, games_required=FULL_SEASON_GAMES):
    """One row per team per regular season with exactly `games_required` games.

    Columns: team_id, season, games, win_rate, shots_for, shots_against,
    goals_for, goals_against, team, abbrev.
    """
    stats = raw["game_teams_stats"][["game_id", "team_id", "won", "shots", "goals"]]
    games = raw["game"][["game_id", "season", "type"]]

    # Self-join: pair each team's row with its opponent's row in the same game
    both = stats.merge(stats, on="game_id", suffixes=("", "_opp"))
    both = both[both.team_id != both.team_id_opp]
    both = both.merge(games[games.type == "R"], on="game_id")

    out = (
        both.groupby(["team_id", "season"])
        .agg(
            games=("won", "size"),
            wins=("won", "sum"),
            shots_for=("shots", "sum"),
            shots_against=("shots_opp", "sum"),
            goals_for=("goals", "sum"),
            goals_against=("goals_opp", "sum"),
        )
        .reset_index()
    )
    out = out[out.games == games_required].copy()
    out["win_rate"] = out.wins / out.games
    out = out.drop(columns="wins")

    names = raw["team_info"][["team_id", "teamName", "abbreviation"]].rename(
        columns={"teamName": "team", "abbreviation": "abbrev"}
    )
    out = out.merge(names, on="team_id")
    cols = ["team_id", "season", "games", "win_rate", "shots_for", "shots_against",
            "goals_for", "goals_against", "team", "abbrev"]
    return out[cols].sort_values(["team_id", "season"]).reset_index(drop=True)
