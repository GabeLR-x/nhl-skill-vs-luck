"""The canonical numbers from notebooks/02_skill_vs_luck.ipynb must not move."""
import pytest

import nhl


def test_luck_share(teams):
    assert nhl.luck_share(teams.win_rate) == pytest.approx(0.33, abs=0.005)


def test_repeatability(pairs):
    r = nhl.repeatability(pairs).round(2)
    assert r.to_dict() == {"shot_share": 0.65, "win_rate": 0.52, "save_pct": 0.40,
                           "pdo": 0.32, "shooting_pct": 0.31}


def test_pdo_predicts_fall_back(pairs):
    assert pairs.pdo.corr(pairs.win_rate_next - pairs.win_rate) == pytest.approx(-0.46, abs=0.005)


def test_split_sizes(split):
    train, test = split
    assert (len(train), len(test)) == (358, 61)
    assert train.next_season.max() == 20162017


def test_forecast_beats_baselines(split):
    train, test = split
    errors = nhl.forecast_errors(train, test, nhl.fit_forecast(train))
    assert errors["model"] == pytest.approx(5.6, abs=0.05)
    assert errors["same_as_last_year"] == pytest.approx(7.0, abs=0.05)
    assert errors["model"] < errors["league_average"] < errors["same_as_last_year"]


def test_decline_model_auc(split):
    res = nhl.fit_decline_models(*split)
    assert res["win_rate"]["auc"] == pytest.approx(0.841, abs=0.0005)
    assert res["win_rate+shot_share+pdo"]["auc"] == pytest.approx(0.893, abs=0.0005)
