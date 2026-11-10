"""NHL skill vs. luck dashboard.

Run from the project folder:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nhl  # noqa: E402
from nhl import season_label  # noqa: E402
from nhl.cards import (  # noqa: E402
    CARD_STATS, add_percentiles, decile_of, model_check, pdo_decile_change, spotlight_sentence,
)

st.set_page_config(page_title="NHL: Skill vs. Luck", page_icon="🏒", layout="wide")

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
# One accent color. Blue = high / gaining, red = low / losing. Both poles were
# checked for red-green colorblind separation on the navy background.
ACCENT = "#2f7fd8"
RED = "#d64545"
INK = "#e6ecf5"
MUTED = "#93a1b8"
GRID = "#24344f"
# Categorical slots for multi-series charts (fixed order, colorblind-checked)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

pio.templates["nhl"] = go.layout.Template(pio.templates["plotly_dark"])
pio.templates["nhl"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=13), colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    hoverlabel=dict(bgcolor="#14213a", bordercolor=GRID, font_color=INK),
    margin=dict(l=10, r=10, t=10, b=10), barcornerradius=4,
)
pio.templates.default = "nhl"

# Team colors for the small identity dot next to a team's name
TEAM_COLORS = {
    "ANA": "#f47a38", "ARI": "#8c2633", "ATL": "#5c88da", "BOS": "#fcb514", "BUF": "#2c5bb8",
    "CAR": "#ce1126", "CBJ": "#4a6fb5", "CGY": "#c8102e", "CHI": "#cf0a2c", "COL": "#8a2b4f",
    "DAL": "#00845a", "DET": "#ce1126", "EDM": "#ff4c00", "FLA": "#c8102e", "LAK": "#a2aaad",
    "MIN": "#2f8f5b", "MTL": "#af1e2d", "NJD": "#ce1126", "NSH": "#ffb81c", "NYI": "#f47d30",
    "NYR": "#0038a8", "OTT": "#c52032", "PHI": "#f74902", "PHX": "#8c2633", "PIT": "#fcb514",
    "SJS": "#00788c", "STL": "#2f5fb3", "TBL": "#2d63b8", "TOR": "#2d5cb3", "VAN": "#00843d",
    "VGK": "#b4975a", "WPG": "#5582c4", "WSH": "#c8102e",
}

LABELS = {
    "win_rate": "Win rate",
    "shot_share": "Share of shots taken",
    "pdo": "Luck score",
    "shooting_pct": "Shooting %",
    "save_pct": "Save %",
}
HELP = {
    "win_rate": "Wins divided by games played, over the 82-game regular season.",
    "shot_share": "Of all shots on goal in this team's games, the share this team took. "
                  "The steadiest sign of real quality: it repeats year to year more than wins do.",
    "pdo": "Luck score = shooting % + save % (hockey analysts call it PDO). League average is "
           "1.000. Far above 1.000 usually means pucks bounced your way, and it rarely lasts.",
    "shooting_pct": "Goals scored divided by shots taken. Mostly luck from year to year.",
    "save_pct": "Share of opponents' shots the goalies stopped. Partly goalie skill, partly luck.",
    "forecast": "Next season's wins from a linear model on this season's win rate and shot share, "
                "trained on seasons through 2016-17.",
    "actual": "Wins the following season, where it was a full 82-game season.",
    "wins": "Regular-season wins out of 82 games.",
}

st.markdown(
    f"""
    <style>
    .tile {{ border-radius: 8px; padding: 10px 6px 8px; text-align: center; color: #fff; }}
    .tile .lab {{ font-size: 0.85rem; opacity: 0.92; min-height: 2.2em; line-height: 1.1em; }}
    .tile .big {{ font-size: 2.1rem; font-weight: 700; line-height: 1.1; }}
    .tile .sub {{ font-size: 0.78rem; opacity: 0.92; }}
    .forecast {{ border: 2px solid {ACCENT}; border-radius: 10px; padding: 14px 10px; text-align: center; }}
    .forecast .big {{ font-size: 3rem; font-weight: 800; line-height: 1; color: {INK}; }}
    .forecast .lab {{ color: {MUTED}; font-size: 0.9rem; }}
    .dot {{ display: inline-block; width: 0.8em; height: 0.8em; border-radius: 50%;
            margin-right: 0.4em; vertical-align: 0.05em; border: 1px solid #ffffff55; }}
    .spot {{ font-size: 1.05rem; line-height: 1.55; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def pctl_band(p):
    """Tile fill and a word, so the tile never relies on color alone."""
    if p >= 80:
        return ACCENT, "High"
    if p >= 60:
        return "#1f5292", "Above average"
    if p > 40:
        return "#3a4458", "Average"
    if p > 20:
        return "#8a3036", "Below average"
    return RED, "Low"


def ordinal(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def strength(r):
    """Plain-language label for a year-to-year correlation."""
    if r >= 0.6:
        return "Strong"
    if r >= 0.45:
        return "Moderate"
    return "Weak"


def dot(abbrev):
    return f'<span class="dot" style="background:{TEAM_COLORS.get(abbrev, MUTED)}"></span>'


# ---------------------------------------------------------------------------
# Data and models (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_teams():
    teams = nhl.add_rates(nhl.team_seasons(nhl.load_raw()))
    teams = nhl.attach_next_season(teams)
    teams["label"] = teams.abbrev + " " + teams.season.map(season_label)
    return add_percentiles(teams)


@st.cache_data
def build_models(teams):
    pairs = nhl.build_pairs(teams[[c for c in teams.columns if not c.endswith("_next")]])
    train, test = nhl.train_test_split_by_season(pairs)
    model = nhl.fit_forecast(train)
    out = teams.copy()
    out["forecast"] = model.predict(out[nhl.FEATURES])
    out["forecast_change_wins"] = (out.forecast - out.win_rate) * 82
    out["actual_change_wins"] = (out.win_rate_next - out.win_rate) * 82
    return {
        "teams": out,
        "pairs": pairs,
        "train": train,
        "test": test,
        "coef": model.coef_,
        "intercept": model.intercept_,
        "errors": nhl.forecast_errors(train, test, model),
        "check": model_check(test, model.predict(test[nhl.FEATURES])),
        "decline": {k: {"accuracy": v["accuracy"], "auc": v["auc"]}
                    for k, v in nhl.fit_decline_models(train, test).items()},
        "repeat": nhl.repeatability(pairs),
        "pdo_change_r": pairs.pdo.corr(pairs.win_rate_next - pairs.win_rate),
        "luck_share": nhl.luck_share(teams.win_rate),
    }


M = build_models(load_teams())
teams, pairs = M["teams"], M["pairs"]
seasons = sorted(teams.season.unique())
pdo_edges, pdo_change = pdo_decile_change(pairs)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("NHL: Which records are real, and which are luck?")
st.caption("Full 82-game regular seasons, 2000-01 to 2018-19 · 512 team-seasons · data: NHL Game Data (Kaggle)")

with st.container(border=True):
    st.markdown("**How to read this**")
    st.markdown(
        "A team's record mixes real quality with luck, and luck doesn't carry over to next season. "
        "The best sign of real quality is how many of the shots in a team's games it takes; "
        "the clearest sign of luck is a *luck score* well above average, from unusually hot "
        "shooting and goaltending. "
        "Teams that won more than their shots support usually fall back the next year, "
        "and this dashboard forecasts by how much."
    )

e = M["errors"]
k1, k2, k3 = st.columns(3)
with k1.container(border=True):
    st.metric("Share of the spread in records that's luck", f"{M['luck_share']:.0%}",
              help="If every game were a coin flip, 82-game records would still spread out. "
                   "That pure-luck spread accounts for this share of the real spread "
                   "(as variance). The rest is real differences between teams.")
with k2.container(border=True):
    st.metric("How well shot share repeats next season", strength(M["repeat"]["shot_share"]),
              help="Year-to-year correlation of shot share (details under 'For the curious'). "
                   f"Win rate repeats less: {strength(M['repeat']['win_rate']).lower()}.")
with k3.container(border=True):
    st.metric("Forecast miss on unseen seasons", f"{e['model']:.1f} wins",
              delta=f"{e['model'] - e['same_as_last_year']:.1f} wins vs. 'same as last year'",
              delta_color="inverse",
              help="Average miss in wins per team on 2017-18 and 2018-19, which the model never "
                   f"saw. Guessing 'same as last year' misses by {e['same_as_last_year']:.1f}.")

tab_card, tab_watch, tab_luck, tab_similar, tab_tech = st.tabs(
    ["Team card", "Regression watch", "Luck vs. skill", "Similar teams", "For the curious"]
)

# ---------------------------------------------------------------------------
# Team card
# ---------------------------------------------------------------------------
with tab_card:
    c1, c2 = st.columns(2)
    card_season = c1.selectbox("Season", seasons, index=seasons.index(20172018),
                               format_func=season_label, key="card_season")
    in_season = teams[teams.season == card_season].sort_values("abbrev")
    options = list(in_season.abbrev)
    card_team = c2.selectbox(
        "Team", options,
        index=options.index("NSH") if "NSH" in options else 0,
        format_func=lambda a: f"{a} · {in_season.set_index('abbrev').team[a]}",
        key="card_team",
    )
    row = in_season[in_season.abbrev == card_team].iloc[0]
    typical = pdo_change[decile_of(row.pdo, pdo_edges)]

    with st.container(border=True):
        st.markdown(
            f"### {dot(row.abbrev)}{row.abbrev} · {row.team} · {season_label(row.season)}",
            unsafe_allow_html=True,
        )
        left, right = st.columns([3, 2])
        with left:
            st.caption("Percentile among all 512 team-seasons. Blue = high, red = low. "
                       "Hover a tile for what it means.")
            raw_fmt = {"win_rate": lambda r: f"{round(r.win_rate * 82)} wins",
                       "shot_share": lambda r: f"{r.shot_share:.1%}",
                       "pdo": lambda r: f"{r.pdo:.3f}",
                       "shooting_pct": lambda r: f"{r.shooting_pct:.1%}",
                       "save_pct": lambda r: f"{r.save_pct:.3f}"}
            tiles = st.columns(len(CARD_STATS))
            for col, s in zip(tiles, CARD_STATS):
                p = row[f"{s}_pctl"]
                fill, word = pctl_band(p)
                col.markdown(
                    f'<div class="tile" style="background:{fill}" title="{HELP[s]}">'
                    f'<div class="lab">{LABELS[s]}</div><div class="big">{ordinal(p)}</div>'
                    f'<div class="sub">{word} · {raw_fmt[s](row)}</div></div>',
                    unsafe_allow_html=True,
                )
        with right:
            actual = (f"Actual: {round(row.win_rate_next * 82)} wins"
                      if pd.notna(row.win_rate_next) else "Actual: not in the data")
            st.markdown(
                f'<div class="forecast" title="{HELP["forecast"]}">'
                f'<div class="lab">Forecast next season</div>'
                f'<div class="big">{round(row.forecast * 82)} wins</div>'
                f'<div class="lab">This season: {round(row.win_rate * 82)} · {actual}</div></div>',
                unsafe_allow_html=True,
            )
            if card_season < 20162017:
                st.caption("This season was part of the model's training data, so the forecast is in-sample.")

        st.markdown(f'<p class="spot">{spotlight_sentence(row, typical)}</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Percentiles over the surrounding seasons**")
        i = seasons.index(card_season)
        window = seasons[max(0, i - 3): i + 4]
        hist = teams[(teams.team_id == row.team_id) & teams.season.isin(window)].sort_values("season")
        fig = go.Figure()
        for s, color in zip(CARD_STATS, SERIES):
            fig.add_trace(go.Scatter(
                x=hist.season.map(season_label), y=hist[f"{s}_pctl"], name=LABELS[s],
                mode="lines+markers", line=dict(width=2, color=color), marker=dict(size=8),
                hovertemplate=f"{LABELS[s]}: %{{y:.0f}}th percentile<extra></extra>",
            ))
        fig.add_vline(x=season_label(card_season), line_width=1, line_dash="dot", line_color=MUTED)
        fig.update_layout(height=300, hovermode="x unified",
                          yaxis=dict(range=[0, 100], title="Percentile", ticksuffix=""),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Regression watch
# ---------------------------------------------------------------------------
with tab_watch:
    season = st.selectbox("Season", seasons, index=seasons.index(20172018),
                          format_func=season_label, key="watch_season")
    s = teams[teams.season == season].sort_values("pdo", ascending=False)
    has_next = s.win_rate_next.notna().any()

    with st.container(border=True):
        st.markdown(f"**Who's due to fall back after {season_label(season)}?**  Sorted luckiest first.")
        table = pd.DataFrame({
            "Team": s.abbrev + " · " + s.team,
            "Wins": (s.win_rate * 82).round().astype(int),
            "Luck score": s.pdo,
            "Forecast": (s.forecast * 82).round().astype(int),
            "Actual next season": (s.win_rate_next * 82).round(),
        })
        luck_pctl = s.pdo_pctl.to_numpy()
        fc_change = (s.forecast_change_wins).to_numpy()
        act_change = (s.actual_change_wins).to_numpy()

        def color_rows(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for j in range(len(df)):
                fill, _ = pctl_band(luck_pctl[j])
                styles.iloc[j, 2] = f"background-color: {fill}; color: white"
                styles.iloc[j, 3] = f"color: {RED if fc_change[j] < -1 else ACCENT if fc_change[j] > 1 else INK}"
                if not np.isnan(act_change[j]):
                    styles.iloc[j, 4] = f"color: {RED if act_change[j] < -1 else ACCENT if act_change[j] > 1 else INK}"
            return styles

        styled = table.reset_index(drop=True).style.apply(color_rows, axis=None).format(
            {"Luck score": "{:.3f}", "Actual next season": lambda v: "—" if pd.isna(v) else f"{v:.0f}"}
        )
        st.dataframe(
            styled, hide_index=True, width="stretch", height=min(38 * len(table) + 40, 760),
            column_config={
                "Team": st.column_config.TextColumn(help="Team abbreviation and name."),
                "Wins": st.column_config.NumberColumn(help=HELP["wins"]),
                "Luck score": st.column_config.NumberColumn(help=HELP["pdo"] + " Cell color: blue = luckier than most, red = unluckier."),
                "Forecast": st.column_config.NumberColumn(help=HELP["forecast"] + " Red = forecast to lose wins, blue = to gain."),
                "Actual next season": st.column_config.NumberColumn(help=HELP["actual"] + " Red = lost wins, blue = gained."),
            },
        )
        if not has_next:
            st.info("The next season isn't in the data as a full 82-game season, so there are no actual results to compare.")
        elif season < 20162017:
            st.caption("The model was trained on seasons through 2016-17, so forecasts for this season are in-sample.")

    with st.container(border=True):
        st.markdown("**This season → forecast → what actually happened**")
        d = s.iloc[::-1]  # Plotly draws the first row at the bottom
        y = d.abbrev
        fig = go.Figure()
        line_x, line_y = [], []
        for _, r in d.iterrows():
            pts = [r.win_rate * 82, r.forecast * 82] + ([r.win_rate_next * 82] if pd.notna(r.win_rate_next) else [])
            line_x += [min(pts), max(pts), None]
            line_y += [r.abbrev, r.abbrev, None]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color="#3d4e6b", width=2),
                                 hoverinfo="skip", showlegend=False))
        marks = [("This season", d.win_rate * 82, "#93a1b8"), ("Forecast", d.forecast * 82, SERIES[0])]
        if has_next:
            marks.append(("Actual next season", d.win_rate_next * 82, SERIES[1]))
        for name, xs, color in marks:
            fig.add_trace(go.Scatter(
                x=xs, y=y, mode="markers", name=name,
                marker=dict(size=11, color=color, line=dict(width=2, color="#0b1422")),
                customdata=d.team, hovertemplate=f"%{{customdata}}<br>{name}: %{{x:.0f}} wins<extra></extra>",
            ))
        fig.update_layout(height=max(420, 24 * len(d)), xaxis_title="Wins",
                          legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    ck = M["check"]
    with st.container(border=True):
        st.markdown("**Was the model right?**  Checked on seasons it never saw (2017-18 and 2018-19).")
        a, b, c = st.columns(3)
        a.metric(f"Of the {ck['n']} teams it said would fall back most", f"{ck['fallers_hit']} did",
                 help="The 10 held-out teams with the biggest forecast drop in wins, and how many "
                      "actually won fewer games the next season.")
        b.metric(f"Of the {ck['n']} it said would bounce back most", f"{ck['risers_hit']} did",
                 help="The 10 held-out teams with the biggest forecast gain, and how many actually won more.")
        c.metric("Right direction, all held-out teams", f"{ck['direction_rate']:.0%}",
                 help=f"Share of all {ck['pairs']} held-out team-seasons where the forecast and the "
                      "actual change went the same way (up or down).")
        f = ck["fallers"].assign(
            Team=lambda x: x.team + " " + x.season.map(season_label),
            Wins=lambda x: (x.win_rate * 82).round().astype(int),
            Forecast=lambda x: (x.forecast * 82).round().astype(int),
            Actual=lambda x: (x.win_rate_next * 82).round().astype(int),
        )[["Team", "Wins", "Forecast", "Actual"]]
        with st.expander("The 10 biggest forecast fallers"):
            st.dataframe(f, hide_index=True, width="stretch", column_config={
                "Wins": st.column_config.NumberColumn(help=HELP["wins"]),
                "Forecast": st.column_config.NumberColumn(help=HELP["forecast"]),
                "Actual": st.column_config.NumberColumn(help=HELP["actual"]),
            })

# ---------------------------------------------------------------------------
# Luck vs. skill
# ---------------------------------------------------------------------------
with tab_luck:
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Which stats carry over to next season?**")
        st.caption("Longer bar = more of it carries over, so it reflects real skill rather than luck.")
        rep = M["repeat"].sort_values()
        fig = go.Figure(go.Bar(
            x=rep.values, y=[LABELS[k] for k in rep.index], orientation="h", marker_color=ACCENT,
            text=[strength(v) for v in rep.values], textposition="outside",
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        fig.update_layout(height=300, bargap=0.45,
                          xaxis=dict(range=[0, 0.85], showticklabels=False, title="How much carries over"))
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    with right.container(border=True):
        st.markdown("**Lucky teams fall back**")
        st.caption("Teams split into 10 equal groups by luck score; average change in wins the next season.")
        fig = go.Figure(go.Bar(
            x=pdo_change.index, y=pdo_change.values,
            marker_color=np.where(pdo_change.values >= 0, ACCENT, RED),
            hovertemplate="Luck group %{x}<br>Change next season: %{y:+.1f} wins<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_color=MUTED)
        fig.update_layout(height=300, bargap=0.35,
                          xaxis=dict(title="Luck group (1 = unluckiest 10%, 10 = luckiest)", dtick=1),
                          yaxis_title="Change in wins next season")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    lucky = teams.nlargest(10, "pdo").iloc[::-1]
    unlucky = teams.nsmallest(10, "pdo").iloc[::-1]
    l2, r2 = st.columns(2)
    for col, df, title, color in [(l2, lucky, "Luckiest team-seasons", ACCENT),
                                  (r2, unlucky, "Unluckiest team-seasons", RED)]:
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption("Luck score measured from the league average of 1.000.")
            fig = go.Figure(go.Bar(
                x=df.pdo - 1, base=1, y=df.label, orientation="h", marker_color=color,
                customdata=np.stack([df.team, (df.win_rate * 82).round(),
                                     (df.win_rate_next * 82).round()], axis=-1),
                hovertemplate="<b>%{customdata[0]} %{y}</b><br>Luck score: %{x:.3f}<br>"
                              "Wins: %{customdata[1]}<br>Wins next season: %{customdata[2]}<extra></extra>",
            ))
            fig.add_vline(x=1.0, line_width=1, line_color=MUTED)
            fig.update_layout(height=340, bargap=0.35, xaxis=dict(title="Luck score", tickformat=".3f"))
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Similar teams (k-nearest neighbors)
# ---------------------------------------------------------------------------
with tab_similar:
    with st.container(border=True):
        st.markdown("**Which past teams looked most like this one, and what happened to them next?**")
        st.caption("The 5 closest team-seasons on share of shots taken, shooting % and save %.")
        features = ["shot_share", "shooting_pct", "save_pct"]
        c1, c2 = st.columns(2)
        sim_season = c1.selectbox("Season", seasons, index=len(seasons) - 1,
                                  format_func=season_label, key="sim_season")
        sim_options = sorted(teams[teams.season == sim_season].abbrev)
        sim_abbrev = c2.selectbox("Team", sim_options, key="sim_team")

        target = teams[(teams.season == sim_season) & (teams.abbrev == sim_abbrev)].iloc[0]
        pool = teams.dropna(subset=["win_rate_next"])
        pool = pool[pool.label != target.label].reset_index(drop=True)
        scaler = StandardScaler().fit(pool[features])
        knn = NearestNeighbors(n_neighbors=5).fit(scaler.transform(pool[features]))
        _, idx = knn.kneighbors(scaler.transform(target[features].to_frame().T))
        similar = pool.iloc[idx[0]]

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{sim_abbrev} {season_label(sim_season)}: wins", f"{target.win_rate * 82:.0f}", help=HELP["wins"])
        m2.metric("Luck score", f"{target.pdo:.3f}", help=HELP["pdo"])
        m3.metric("Similar teams: average change next season",
                  f"{similar.actual_change_wins.mean():+.1f} wins",
                  help="Average change in wins the following season for the 5 most similar team-seasons.")
        show = pd.DataFrame({
            "Team-season": similar.team + " " + similar.season.map(season_label),
            "Share of shots": similar.shot_share,
            "Shooting %": similar.shooting_pct,
            "Save %": similar.save_pct,
            "Wins": (similar.win_rate * 82).round().astype(int),
            "Wins next season": (similar.win_rate_next * 82).round().astype(int),
        })
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "Team-season": st.column_config.TextColumn(help="A past team-season with a known next season."),
            "Share of shots": st.column_config.NumberColumn(format="%.3f", help=HELP["shot_share"]),
            "Shooting %": st.column_config.NumberColumn(format="%.3f", help=HELP["shooting_pct"]),
            "Save %": st.column_config.NumberColumn(format="%.3f", help=HELP["save_pct"]),
            "Wins": st.column_config.NumberColumn(help=HELP["wins"]),
            "Wins next season": st.column_config.NumberColumn(help=HELP["actual"]),
        })

# ---------------------------------------------------------------------------
# For the curious: the technical details
# ---------------------------------------------------------------------------
@st.cache_data
def ols_table(train):
    ols = smf.ols("win_rate_next ~ win_rate + shot_share", data=train).fit()
    ci = ols.conf_int()
    return pd.DataFrame({"Coefficient": ols.params, "95% CI low": ci[0],
                         "95% CI high": ci[1], "p-value": ols.pvalues})


with tab_tech:
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Forecast model**")
        st.latex(
            rf"\widehat{{\text{{win rate}}}}_{{t+1}} = {M['intercept']:.3f} + {M['coef'][0]:.3f}"
            rf"\cdot \text{{win rate}}_t + {M['coef'][1]:.3f}\cdot \text{{shot share}}_t"
        )
        st.caption(f"Linear regression trained on {len(M['train'])} season pairs whose next season is "
                   f"2016-17 or earlier; tested on {len(M['test'])} pairs (2017-18, 2018-19).")
        st.dataframe(ols_table(M["train"]).round(3), width="stretch", column_config={
            "Coefficient": st.column_config.NumberColumn(help="statsmodels OLS fit on the training pairs."),
            "95% CI low": st.column_config.NumberColumn(help="Lower end of the 95% confidence interval."),
            "95% CI high": st.column_config.NumberColumn(help="Upper end of the 95% confidence interval."),
            "p-value": st.column_config.NumberColumn(help="Chance of a coefficient this large if the true effect were zero."),
        })
        err = pd.DataFrame({
            "Forecast": ["Same as last year", "League average", "Win rate + shot share"],
            "Average miss (wins)": [e["same_as_last_year"], e["league_average"], e["model"]],
        })
        st.dataframe(err.round(2), hide_index=True, width="stretch", column_config={
            "Average miss (wins)": st.column_config.NumberColumn(help="Mean absolute error × 82 on the test pairs."),
        })

    with b.container(border=True):
        st.markdown("**Year-to-year correlation (r) behind the Strong / Moderate / Weak labels**")
        rep = M["repeat"].sort_values(ascending=False)
        st.dataframe(pd.DataFrame({"Stat": [LABELS[k] for k in rep.index], "r": rep.values.round(2),
                                   "Label": [strength(v) for v in rep.values]}),
                     hide_index=True, width="stretch", column_config={
                         "r": st.column_config.NumberColumn(help=f"Pearson correlation over {len(pairs)} back-to-back season pairs."),
                         "Label": st.column_config.TextColumn(help="Strong ≥ 0.60, Moderate 0.45–0.60, Weak < 0.45."),
                     })
        st.markdown(f"Luck score vs. next-season change in win rate: **r = {M['pdo_change_r']:.2f}**")
        dec = pd.DataFrame({
            "Model": ["Win rate only", "Win rate + shot share + luck score"],
            "Accuracy": [M["decline"]["win_rate"]["accuracy"], M["decline"]["win_rate+shot_share+pdo"]["accuracy"]],
            "AUC": [M["decline"]["win_rate"]["auc"], M["decline"]["win_rate+shot_share+pdo"]["auc"]],
        }).round(3)
        st.markdown("**Will a team decline next season?** (logistic regression, same split)")
        st.dataframe(dec, hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.NumberColumn(help="Share of test teams classified correctly at a 0.5 threshold."),
            "AUC": st.column_config.NumberColumn(help="Chance the model ranks a random decliner above a random non-decliner."),
        })

    with st.container(border=True):
        st.markdown("**Luck score vs. record, every team-season**")
        fig = go.Figure(go.Scatter(
            x=teams.pdo, y=teams.win_rate * 82, mode="markers",
            marker=dict(size=8, color=ACCENT, opacity=0.6, line=dict(width=1, color="#0b1422")),
            text=teams.team + " " + teams.season.map(season_label),
            hovertemplate="<b>%{text}</b><br>Luck score: %{x:.3f}<br>Wins: %{y:.0f}<extra></extra>",
        ))
        fig.add_vline(x=1.0, line_width=1, line_color=MUTED)
        fig.update_layout(height=380, xaxis_title="Luck score (PDO = shooting % + save %)", yaxis_title="Wins")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        st.caption("Limitations: luck is estimated as if every game were a coin flip; no shot-quality data; "
                   "regression to the mean can't see real improvement (the 2017-18 Lightning were forecast "
                   "to drop to 46 wins and won 62).")
