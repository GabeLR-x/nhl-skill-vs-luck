"""Per team-season rates: shot share, shooting %, save % and PDO."""


def add_rates(teams):
    """Add the four rate stats to a frame with shots/goals for and against.

    shot_share   = shots for / all shots in the team's games
    shooting_pct = goals for / shots for
    save_pct     = 1 - goals against / shots against
    pdo          = shooting_pct + save_pct   (league average is ~1.000)
    """
    out = teams.copy()
    out["shot_share"] = out.shots_for / (out.shots_for + out.shots_against)
    out["shooting_pct"] = out.goals_for / out.shots_for
    out["save_pct"] = 1 - out.goals_against / out.shots_against
    out["pdo"] = out.shooting_pct + out.save_pct
    return out


def season_label(season):
    """20172018 -> '2017-18'."""
    s = str(int(season))
    return f"{s[:4]}-{s[6:]}"
