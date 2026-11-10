"""NHL skill vs. luck dashboard.

Run from the NHL project folder:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nhl  # noqa: E402
from nhl import season_label  # noqa: E402

# Chart colors: one blue for single series; blue <-> red for up/down
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#9a9890"

st.set_page_config(page_title="NHL: Skill vs. Luck", page_icon="🏒", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    teams = nhl.add_rates(nhl.team_seasons(nhl.load_raw()))
    teams = nhl.attach_next_season(teams)
    teams["label"] = teams.team + " " + teams.season.map(season_label)
    return teams


@st.cache_data
def fit_forecast(teams):
    """Same model as the notebook: trained on pairs whose next season is 2016-17 or earlier."""
    train, _ = nhl.train_test_split_by_season(teams.dropna(subset=["win_rate_next"]))
    model = nhl.fit_forecast(train)
    out = teams.copy()
    out["forecast"] = model.predict(out[nhl.FEATURES])
    out["forecast_change_wins"] = (out.forecast - out.win_rate) * 82
    out["actual_change_wins"] = (out.win_rate_next - out.win_rate) * 82
    return out, model.coef_, model.intercept_


teams = load_data()
teams, coefs, intercept = fit_forecast(teams)
pairs = teams.dropna(subset=["win_rate_next"])
seasons = sorted(teams.season.unique())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("NHL: Which records are real, and which are luck?")
st.caption(
    "Full 82-game regular seasons, 2000-01 to 2018-19. "
    "PDO = shooting % + save %, hockey's standard luck indicator (league average = 1.000)."
)

luck_sd = np.sqrt(0.25 / 82)
luck_share = luck_sd ** 2 / teams.win_rate.var()
test = pairs[pairs.next_season >= 20172018]
model_err = (test.win_rate_next - test.forecast).abs().mean() * 82
naive_err = (test.win_rate_next - test.win_rate).abs().mean() * 82

k1, k2, k3 = st.columns(3)
k1.metric("Luck's share of the spread in records", f"{luck_share:.0%}")
k1.caption("The rest is real differences in team quality.")
k2.metric("Shot share, year-to-year repeat", f"{pairs.shot_share.corr(pairs.shot_share_next):.2f}")
k2.caption(f"vs {pairs.win_rate.corr(pairs.win_rate_next):.2f} for win rate: shots are the steadier signal.")
k3.metric("Forecast error on unseen seasons", f"{model_err:.1f} wins")
k3.caption(f"vs {naive_err:.1f} wins for 'same as last year'.")

tab_watch, tab_luck, tab_similar = st.tabs(["Regression watch", "Skill vs. luck", "Similar teams"])

# ---------------------------------------------------------------------------
# Tab 1: Regression watch
# ---------------------------------------------------------------------------
with tab_watch:
    season = st.selectbox(
        "Season", seasons, index=seasons.index(20172018), format_func=season_label, key="watch_season"
    )
    # Plotly draws the first bar at the bottom, so sort high-to-low to put the biggest fallers on top
    s = teams[teams.season == season].sort_values("forecast_change_wins", ascending=False)
    has_next = s.win_rate_next.notna().any()

    st.subheader(f"Who's due to fall back after {season_label(season)}?")
    st.caption(
        "Forecast change in wins next season, from this season's win rate and shot share. "
        "Teams that won more than their shot share supports are forecast to fall back. "
        "Red = forecast to lose wins, blue = forecast to gain."
    )

    fig = go.Figure(go.Bar(
        x=s.forecast_change_wins,
        y=s.team,
        orientation="h",
        marker_color=np.where(s.forecast_change_wins >= 0, BLUE, RED),
        customdata=np.stack([
            (s.win_rate * 82).round(0),
            (s.forecast * 82).round(0),
            s.pdo.round(3),
            (s.shot_share * 100).round(1),
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Wins this season: %{customdata[0]:.0f}<br>"
            "Forecast next season: %{customdata[1]:.0f}<br>"
            "PDO: %{customdata[2]:.3f}<br>"
            "Shot share: %{customdata[3]:.1f}%<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_width=1, line_color=GRAY)
    fig.update_layout(
        height=max(420, 22 * len(s)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Forecast change in wins next season",
        yaxis_title=None,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

    table = s.sort_values("pdo", ascending=False).assign(
        wins=lambda d: (d.win_rate * 82).round(0).astype(int),
        forecast_wins=lambda d: (d.forecast * 82).round(0).astype(int),
        actual_next=lambda d: (d.win_rate_next * 82).round(0),
    )
    cols = {"team": "Team", "wins": "Wins", "pdo": "PDO", "shot_share": "Shot share",
            "forecast_wins": "Forecast wins next season"}
    if has_next:
        cols["actual_next"] = "Actual wins next season"
    st.dataframe(
        table[list(cols)].rename(columns=cols),
        hide_index=True,
        use_container_width=True,
        column_config={
            "PDO": st.column_config.NumberColumn(format="%.3f"),
            "Shot share": st.column_config.NumberColumn(format="%.3f"),
            "Actual wins next season": st.column_config.NumberColumn(format="%d"),
        },
    )
    if not has_next:
        st.info("The next season isn't in the data as a full 82-game season, so there are no actual results to compare.")
    elif season <= 20152016:
        st.caption("The model was trained on seasons up to 2016-17, so forecasts for this season are in-sample.")

# ---------------------------------------------------------------------------
# Tab 2: Skill vs. luck
# ---------------------------------------------------------------------------
with tab_luck:
    left, right = st.columns(2)

    with left:
        st.subheader("Which stats repeat next season?")
        st.caption("Correlation between a team's stat and the same stat the following season. Higher = more skill, less luck.")
        labels = {"shot_share": "Shot share", "win_rate": "Win rate", "save_pct": "Save %",
                  "pdo": "PDO", "shooting_pct": "Shooting %"}
        tmp = teams.merge(
            teams[["team_id", "season"] + list(labels)].rename(columns={"season": "next_season"}),
            on=["team_id", "next_season"], suffixes=("", "_n"),
        )
        rep = pd.Series({lab: tmp[c].corr(tmp[c + "_n"]) for c, lab in labels.items()}).sort_values()
        fig = go.Figure(go.Bar(
            x=rep.values, y=rep.index, orientation="h", marker_color=BLUE,
            text=[f"{v:.2f}" for v in rep.values], textposition="outside",
            hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=30, t=10, b=10),
                          xaxis=dict(range=[0, 0.8], title="Correlation with next season"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Lucky teams fall back")
        st.caption("Teams split into 10 equal groups by PDO. Average change in wins the following season.")
        p = pairs.copy()
        p["group"] = pd.qcut(p.pdo, 10, labels=range(1, 11)).astype(int)
        g = p.groupby("group").actual_change_wins.mean()
        fig = go.Figure(go.Bar(
            x=g.index, y=g.values,
            marker_color=np.where(g.values >= 0, BLUE, RED),
            hovertemplate="PDO group %{x}<br>Change: %{y:+.1f} wins<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_color=GRAY)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="PDO group (1 = unluckiest 10%, 10 = luckiest)", dtick=1),
                          yaxis_title="Change in wins next season")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Luck vs. record, every team-season")
    st.caption("Each dot is one team in one season. Right of the line = luckier than average.")
    fig = go.Figure(go.Scatter(
        x=teams.pdo, y=teams.win_rate * 82, mode="markers",
        marker=dict(size=8, color=BLUE, opacity=0.55, line=dict(width=1, color="white")),
        text=teams.label,
        hovertemplate="<b>%{text}</b><br>PDO: %{x:.3f}<br>Wins: %{y:.0f}<extra></extra>",
    ))
    fig.add_vline(x=1.0, line_width=1, line_color=GRAY)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="PDO (shooting % + save %)", yaxis_title="Wins")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Similar teams (k-nearest neighbors)
# ---------------------------------------------------------------------------
with tab_similar:
    st.subheader("Which past teams looked most like this one?")
    st.caption(
        "Nearest neighbors on shot share, shooting % and save % (standardized). "
        "Shows what happened to similar teams the following season."
    )
    features = ["shot_share", "shooting_pct", "save_pct"]
    c1, c2 = st.columns(2)
    sim_season = c1.selectbox("Season", seasons, index=len(seasons) - 1,
                              format_func=season_label, key="sim_season")
    team_names = sorted(teams[teams.season == sim_season].team)
    sim_team = c2.selectbox("Team", team_names, key="sim_team")

    target = teams[(teams.season == sim_season) & (teams.team == sim_team)].iloc[0]
    # Compare against other seasons that have a known next season
    pool = pairs[pairs.label != target.label].reset_index(drop=True)
    scaler = StandardScaler().fit(pool[features])
    knn = NearestNeighbors(n_neighbors=5).fit(scaler.transform(pool[features]))
    _, idx = knn.kneighbors(scaler.transform(target[features].to_frame().T))
    similar = pool.iloc[idx[0]]

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{sim_team} {season_label(sim_season)}: wins", f"{target.win_rate * 82:.0f}")
    m2.metric("PDO", f"{target.pdo:.3f}")
    m3.metric("Similar teams: avg change next season",
              f"{similar.actual_change_wins.mean():+.1f} wins")

    show = similar.assign(
        wins=(similar.win_rate * 82).round(0).astype(int),
        wins_next=(similar.win_rate_next * 82).round(0).astype(int),
        change=similar.actual_change_wins.round(1),
    )[["label", "shot_share", "shooting_pct", "save_pct", "wins", "wins_next", "change"]]
    st.dataframe(
        show.rename(columns={"label": "Team-season", "shot_share": "Shot share",
                             "shooting_pct": "Shooting %", "save_pct": "Save %",
                             "wins": "Wins", "wins_next": "Wins next season",
                             "change": "Change"}),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Shot share": st.column_config.NumberColumn(format="%.3f"),
            "Shooting %": st.column_config.NumberColumn(format="%.3f"),
            "Save %": st.column_config.NumberColumn(format="%.3f"),
            "Change": st.column_config.NumberColumn(format="%+d"),
        },
    )

st.divider()
st.caption(
    f"Forecast model: next win rate = {intercept:.3f} + {coefs[0]:.3f} × win rate + {coefs[1]:.3f} × shot share. "
    "Trained on season pairs through 2016-17; tested on 2017-18 and 2018-19. "
    "Limitations: luck is estimated as if every game were a coin flip; no shot-quality data."
)
