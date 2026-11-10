"""Luck share, repeatability, the next-season forecast and the "will decline" model.

Train/test split is by season: pairs whose target (next) season is 2016-17 or
earlier train the models; 2017-18 and 2018-19 targets are held out.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GAMES = 82
TRAIN_LAST_TARGET = 20162017
FEATURES = ["win_rate", "shot_share"]
REPEAT_STATS = ["shot_share", "win_rate", "save_pct", "pdo", "shooting_pct"]


def luck_share(win_rates, games=GAMES):
    """Share of the variance in win rate a pure coin-flip league would produce."""
    luck_var = 0.5 * 0.5 / games
    return luck_var / np.var(win_rates, ddof=1)


def repeatability(pairs, stats=REPEAT_STATS):
    """Correlation of each stat with the same stat the next season."""
    return pd.Series({s: pairs[s].corr(pairs[f"{s}_next"]) for s in stats})


def train_test_split_by_season(pairs, last_train_target=TRAIN_LAST_TARGET):
    return (pairs[pairs.next_season <= last_train_target],
            pairs[pairs.next_season > last_train_target])


def fit_forecast(train, features=FEATURES):
    """Linear regression: next season's win rate from this season's stats."""
    return LinearRegression().fit(train[features], train.win_rate_next)


def forecast_errors(train, test, model, features=FEATURES):
    """Mean absolute error in wins (per 82 games) for the model and two baselines."""
    def wins_off(pred):
        return float(np.mean(np.abs(test.win_rate_next - pred)) * GAMES)
    return {
        "same_as_last_year": wins_off(test.win_rate),
        "league_average": wins_off(train.win_rate_next.mean()),
        "model": wins_off(model.predict(test[features])),
    }


def fit_decline_models(train, test):
    """Logistic regression for 'win rate drops next season'.

    Returns {name: {"model", "accuracy", "auc"}} for win rate only and
    win rate + shot share + PDO.
    """
    def declines(df):
        return (df.win_rate_next < df.win_rate).astype(int)

    out = {}
    for name, cols in {"win_rate": ["win_rate"],
                       "win_rate+shot_share+pdo": ["win_rate", "shot_share", "pdo"]}.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(train[cols], declines(train))
        prob = clf.predict_proba(test[cols])[:, 1]
        out[name] = {
            "model": clf,
            "features": cols,
            "accuracy": float(accuracy_score(declines(test), prob > 0.5)),
            "auc": float(roc_auc_score(declines(test), prob)),
        }
    return out
