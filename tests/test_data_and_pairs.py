import pandas as pd

from nhl import build_pairs


def test_team_seasons_are_full_82_game_seasons(teams):
    assert len(teams) == 512
    assert (teams.games == 82).all()
    assert not teams.duplicated(["team_id", "season"]).any()
    # 2012-13 (48 games) and 2019-20 (shortened) are excluded
    assert not teams.season.isin([20122013, 20192020]).any()


def test_pairs_only_join_back_to_back_seasons():
    toy = pd.DataFrame({
        "team_id": [1, 1, 1, 1, 2],
        "season": [20022003, 20032004, 20052006, 20062007, 20052006],
        "win_rate": [0.5, 0.6, 0.4, 0.55, 0.3],
    })
    p = build_pairs(toy)
    # 2003-04 -> 2005-06 skips the cancelled season, and team 2 has no next season
    assert list(zip(p.season, p.season_next)) == [(20022003, 20032004), (20052006, 20062007)]
    assert list(p.win_rate_next) == [0.6, 0.55]
    assert (p.team_id == 1).all()


def test_real_data_has_419_pairs(pairs):
    assert len(pairs) == 419
    assert ((pairs.season_next - pairs.season) == 10001).all()
