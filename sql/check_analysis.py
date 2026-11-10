"""Run the window-function queries in sql/analysis.sql and check each one
against the same calculation in pandas.

Run from anywhere:   python sql/check_analysis.py
Builds nhl.db first if it doesn't exist.
"""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "nhl.db"
sys.path.insert(0, str(ROOT / "src"))
import nhl  # noqa: E402


def load_queries(path=ROOT / "sql" / "analysis.sql"):
    """{name: sql} from the '-- name:' markers."""
    parts = re.split(r"^-- name: (\w+)\s*$", path.read_text(), flags=re.M)
    return {name: sql.strip() for name, sql in zip(parts[1::2], parts[2::2])}


def per_game(raw):
    """One row per team per regular-season game, with the opponent's shots, in game order."""
    stats = raw["game_teams_stats"][["game_id", "team_id", "won", "shots"]]
    both = stats.merge(stats, on="game_id", suffixes=("", "_opp"))
    both = both[both.team_id != both.team_id_opp]
    games = raw["game"]
    both = both.merge(games.loc[games.type == "R", ["game_id", "season", "date_time_GMT"]], on="game_id")
    both["won"] = both.won.astype(int)
    return both.sort_values(["team_id", "season", "date_time_GMT", "game_id"]).reset_index(drop=True)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<18} {detail}")
    return ok


def main():
    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "sql" / "load_data.py")], check=True)
    queries = load_queries()
    raw = nhl.load_raw()
    games = per_game(raw)
    teams = nhl.add_rates(nhl.team_seasons(raw))
    results = []

    with sqlite3.connect(DB_PATH) as conn:
        q = {name: pd.read_sql(sql, conn) for name, sql in queries.items()}

    # 1. Running wins: cumulative sum within team-season
    expected = games.groupby(["team_id", "season"]).won.cumsum()
    got = q["running_wins"].running_wins
    results.append(check("running_wins", len(got) == len(expected) and (got.values == expected.values).all(),
                         f"{len(got):,} team-games"))

    # 2. Rolling 10-game shot share
    g = games.groupby(["team_id", "season"])
    sf = g.shots.transform(lambda s: s.rolling(10, min_periods=1).sum())
    sa = g.shots_opp.transform(lambda s: s.rolling(10, min_periods=1).sum())
    expected = sf / (sf + sa)
    got = q["rolling_shot_share"].shot_share_10
    results.append(check("rolling_shot_share", len(got) == len(expected) and np.allclose(got, expected),
                         f"{len(got):,} team-games, max diff {np.abs(got.values - expected.values).max():.1e}"))

    # 3. PDO rank within season
    exp = teams.assign(pdo_rank=teams.groupby("season").pdo.rank(method="min", ascending=False).astype(int))
    m = q["pdo_rank"].merge(exp[["team_id", "season", "pdo", "pdo_rank"]], on=["team_id", "season"],
                            suffixes=("", "_pd"))
    results.append(check("pdo_rank", len(m) == len(teams) and np.allclose(m.pdo, m.pdo_pd)
                         and (m.pdo_rank == m.pdo_rank_pd).all(), f"{len(m)} team-seasons"))

    # 4. LAG() year-over-year change == the season-pair builder
    pairs = nhl.build_pairs(teams)
    m = q["win_rate_change"].merge(pairs, left_on=["team_id", "prev_season"], right_on=["team_id", "season"],
                                   suffixes=("", "_pd"))
    results.append(check("win_rate_change", len(q["win_rate_change"]) == len(pairs) == len(m)
                         and np.allclose(m.change, m.win_rate_next - m.win_rate_pd),
                         f"{len(m)} season pairs (expected 419)"))

    print("\nAll queries match pandas." if all(results) else "\nSome queries did NOT match.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
