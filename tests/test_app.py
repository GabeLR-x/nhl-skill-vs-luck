"""Smoke tests for the Streamlit dashboard."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_loads_without_exceptions(app):
    assert not app.exception
    assert [m.value for m in app.metric][:3] == ["33%", "Strong", "5.6 wins"]


def test_selecting_a_team_renders_its_card(app):
    app.selectbox(key="card_season").select(20152016).run()
    app.selectbox(key="card_team").select("TBL").run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "TBL · Lightning · 2015-16" in html
    assert "The 2015-16 Lightning won 46 games." in html
    assert "Forecast next season" in html
