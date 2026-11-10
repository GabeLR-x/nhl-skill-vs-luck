"""Back-to-back season pairs: a team's season next to its following season."""

SEASON_STEP = 10001  # 20172018 + 10001 = 20182019


def build_pairs(teams):
    """Inner-join each team-season to the same team's next season.

    Only consecutive full seasons pair up, so a season followed by a lockout or
    shortened season (2003-04, 2011-12, 2018-19) has no pair. Columns from the
    next season get a `_next` suffix.
    """
    t = teams.copy()
    t["next_season"] = t.season + SEASON_STEP
    return t.merge(
        t.drop(columns=["next_season"]),
        left_on=["team_id", "next_season"],
        right_on=["team_id", "season"],
        suffixes=("", "_next"),
    )


def attach_next_season(teams, cols=("win_rate", "shot_share", "pdo")):
    """Left-join the next season's `cols` (as `<col>_next`), keeping every season."""
    t = teams.copy()
    t["next_season"] = t.season + SEASON_STEP
    nxt = t[["team_id", "season", *cols]].rename(
        columns={"season": "next_season", **{c: f"{c}_next" for c in cols}}
    )
    return t.merge(nxt, on=["team_id", "next_season"], how="left")
