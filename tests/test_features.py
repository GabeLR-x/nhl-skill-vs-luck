import pandas as pd
import pytest

from nhl import add_rates, season_label


def test_rates_on_hand_computed_team():
    # 3 goals on 30 shots (10%); 2 goals allowed on 25 shots (.920 save %)
    t = pd.DataFrame({"shots_for": [30], "shots_against": [25], "goals_for": [3], "goals_against": [2]})
    r = add_rates(t).iloc[0]
    assert r.shot_share == pytest.approx(30 / 55)
    assert r.shooting_pct == pytest.approx(0.10)
    assert r.save_pct == pytest.approx(0.92)
    assert r.pdo == pytest.approx(1.02)


def test_league_pdo_averages_to_one(teams):
    # Every goal is one team's shooting and the other's save, so league-wide
    # shooting % + save % is exactly 1 when weighted by shots.
    league_sh = teams.goals_for.sum() / teams.shots_for.sum()
    league_sv = 1 - teams.goals_against.sum() / teams.shots_against.sum()
    assert league_sh + league_sv == pytest.approx(1.0, abs=1e-3)
    assert teams.pdo.mean() == pytest.approx(1.0, abs=0.002)


def test_season_label():
    assert season_label(20172018) == "2017-18"
    assert season_label(20002001) == "2000-01"
