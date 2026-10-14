"""Load the NHL CSVs into a SQLite database (nhl.db in the project root).

Run from the project root:   python sql/load_data.py
Re-running rebuilds the tables from scratch.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data"
DB_PATH = ROOT / "nhl.db"

# Only the tables this analysis uses. game_plays etc. are huge; add them later if needed.
TABLES = ["game", "game_teams_stats", "team_info"]

# Join keys get indexes so joins stay fast.
INDEXES = {
    "game": ["game_id"],
    "game_teams_stats": ["game_id", "team_id"],
    "team_info": ["team_id"],
}

with sqlite3.connect(DB_PATH) as conn:
    for name in TABLES:
        raw = pd.read_csv(CSV_DIR / f"{name}.csv")
        df = raw.drop_duplicates()  # raw CSVs ship with exact duplicate rows
        df.to_sql(name, conn, if_exists="replace", index=False)
        for col in INDEXES[name]:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_{col} ON {name}({col})")
        print(f"{name:<18} {len(raw):>7,} raw -> {len(df):>7,} loaded")

print(f"\nWrote {DB_PATH}")
