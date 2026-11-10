"""Shared analysis code for the NHL skill-vs-luck project.

The dashboard and tests import from here; the notebooks keep their own code
so their outputs stay exactly as written.
"""
from .data import load_raw, team_seasons
from .features import add_rates, season_label
from .pairs import build_pairs, attach_next_season
from .models import (
    FEATURES,
    TRAIN_LAST_TARGET,
    luck_share,
    repeatability,
    train_test_split_by_season,
    fit_forecast,
    forecast_errors,
    fit_decline_models,
)

__all__ = [
    "load_raw", "team_seasons", "add_rates", "season_label",
    "build_pairs", "attach_next_season", "FEATURES", "TRAIN_LAST_TARGET",
    "luck_share", "repeatability", "train_test_split_by_season",
    "fit_forecast", "forecast_errors", "fit_decline_models",
]
