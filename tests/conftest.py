import pytest

import nhl


@pytest.fixture(scope="session")
def teams():
    return nhl.add_rates(nhl.team_seasons(nhl.load_raw()))


@pytest.fixture(scope="session")
def pairs(teams):
    return nhl.build_pairs(teams)


@pytest.fixture(scope="session")
def split(pairs):
    return nhl.train_test_split_by_season(pairs)
