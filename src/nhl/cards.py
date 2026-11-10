"""Plain-English summaries used by the dashboard's team card and model check."""
import numpy as np
import pandas as pd

from .features import season_label

GAMES = 82

# Stats shown as percentile tiles, in display order
CARD_STATS = ["win_rate", "shot_share", "pdo", "shooting_pct", "save_pct"]


def add_percentiles(teams, stats=CARD_STATS):
    """Percentile (0-100) of each stat among all team-seasons, as `<stat>_pctl`."""
    out = teams.copy()
    for s in stats:
        out[f"{s}_pctl"] = out[s].rank(pct=True) * 100
    return out


def pdo_decile_change(pairs):
    """Average change in wins next season for each PDO decile (1 = unluckiest).

    Returns (bin_edges, Series indexed 1..10)."""
    groups, edges = pd.qcut(pairs.pdo, 10, labels=range(1, 11), retbins=True)
    change = (pairs.win_rate_next - pairs.win_rate) * GAMES
    return edges, change.groupby(groups.astype(int)).mean()


def decile_of(pdo, edges):
    """Which PDO decile (1-10) a value falls in, using edges from pdo_decile_change."""
    return int(np.clip(np.searchsorted(edges, pdo, side="right"), 1, 10))


def spotlight_sentence(row, typical_change):
    """One plain-English paragraph about a team-season.

    `row` needs team, season, win_rate, pdo_pctl, forecast and win_rate_next (NaN if unknown).
    `typical_change` is the average next-season change in wins for teams with similar luck.
    """
    wins = round(row.win_rate * GAMES)
    pctl = int(row.pdo_pctl)
    parts = [f"The {season_label(row.season)} {row.team} won {wins} games."]
    if pctl >= 50:
        parts.append(f"Their shooting and goaltending were luckier than {pctl}% of teams.")
    else:
        parts.append(f"Their shooting and goaltending were unluckier than {100 - pctl}% of teams.")
    d = round(typical_change)
    if d <= -1:
        parts.append(f"Teams like that usually lose about {-d} wins the next year.")
    elif d >= 1:
        parts.append(f"Teams like that usually gain about {d} wins the next year.")
    else:
        parts.append("Teams like that usually win about the same number the next year.")
    parts.append(f"Forecast: {round(row.forecast * GAMES)}.")
    if pd.notna(row.win_rate_next):
        parts.append(f"Actual: {round(row.win_rate_next * GAMES)}.")
    else:
        parts.append("Actual: not in the data (the next season wasn't a full 82-game season).")
    return " ".join(parts)


def model_check(test, forecast, n=10):
    """How often the forecast called the direction right on held-out pairs.

    Returns a dict with the n biggest forecast fallers / risers and how many of
    each actually fell / rose, plus the overall direction hit rate.
    """
    t = test.assign(forecast=forecast)
    t["forecast_change"] = (t.forecast - t.win_rate) * GAMES
    t["actual_change"] = (t.win_rate_next - t.win_rate) * GAMES
    fallers = t.nsmallest(n, "forecast_change")
    risers = t.nlargest(n, "forecast_change")
    same_direction = np.sign(t.forecast_change) == np.sign(t.actual_change)
    return {
        "n": n,
        "fallers": fallers,
        "fallers_hit": int((fallers.actual_change < 0).sum()),
        "risers": risers,
        "risers_hit": int((risers.actual_change > 0).sum()),
        "direction_rate": float(same_direction.mean()),
        "pairs": len(t),
    }
