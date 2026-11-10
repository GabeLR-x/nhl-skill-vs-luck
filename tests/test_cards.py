import nhl
from nhl.cards import add_percentiles, decile_of, model_check, pdo_decile_change, spotlight_sentence


def test_percentiles_are_0_to_100(teams):
    p = add_percentiles(teams)
    assert p.pdo_pctl.between(0, 100).all()
    assert p.pdo_pctl.idxmax() == teams.pdo.idxmax()


def test_spotlight_for_2017_18_predators(teams, pairs, split):
    train, _ = split
    model = nhl.fit_forecast(train)
    t = nhl.attach_next_season(add_percentiles(teams))
    t["forecast"] = model.predict(t[nhl.FEATURES])
    row = t[(t.abbrev == "NSH") & (t.season == 20172018)].iloc[0]
    edges, change = pdo_decile_change(pairs)
    text = spotlight_sentence(row, change[decile_of(row.pdo, edges)])
    assert text.startswith("The 2017-18 Predators won 53 games.")
    assert "luckier than 96% of teams" in text
    assert "lose about 6 wins" in text
    assert text.endswith("Forecast: 45. Actual: 47.")


def test_model_check_on_held_out(split):
    train, test = split
    model = nhl.fit_forecast(train)
    ck = model_check(test, model.predict(test[nhl.FEATURES]))
    assert ck["pairs"] == 61
    assert ck["fallers_hit"] == 9
