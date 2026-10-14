# NHL: Which Records Are Real, and Which Are Luck?

An NHL team that wins 50 games looks better than one that wins 44. But how much of that gap is real quality, and how much is luck that will disappear next season?

Using 17 full NHL regular seasons (2000-01 to 2018-19), this project separates repeatable skill from luck in team records, then uses that to **forecast next season's win rate**, tested on seasons the model never saw.

![Dashboard: forecast change in wins for every team after 2017-18](figures/dashboard.png)

## Key findings

1. **About a third of the spread in team records is luck.** A simulated league where every game is a coin flip still produces a win-rate spread (standard deviation) of 0.055 over 82 games. The real NHL spread is 0.096, so luck accounts for about 33% of the variance.

2. **Shot share is a better measure of team quality than the standings.** A team's share of all shots in its games repeats from one season to the next (r = 0.65) more strongly than its win rate does (r = 0.52). Shooting % (0.31) and save % (0.40) barely repeat, so they're mostly luck.

   ![Year-to-year repeatability of each stat](figures/repeatability.png)

3. **Lucky teams fall back.** Teams in the top 10% for PDO (shooting % + save %, hockey's standard luck indicator) lost about 5.5 wins the following season; the bottom 10% gained about 6.5. Both moved only about halfway back to average, because a lucky team is usually also a good one.

   ![Change in wins next season by PDO group](figures/pdo_regression.png)

4. **A simple model beats the obvious forecasts.** Trained on seasons through 2016-17 and tested on 2017-18 and 2018-19, a linear regression on win rate + shot share misses next season's record by **5.6 wins** on average, vs **7.0** for "same as last year". That's about 20% less error. Adding PDO doesn't help further.

5. **A logistic regression flags which teams will decline** with 79% accuracy (vs 54% for always guessing the more common outcome). Adding shot share and PDO raises AUC from 0.84 to 0.89.

6. **Single-game box scores are misleading.** Shots alone can't predict who won a game (50%, a coin flip). The full box score reaches 61%, but its strongest signal is *blocked shots*, which is a symptom of already leading (trailing teams shoot more), not a cause of winning. Adding goals jumps accuracy to 80%, but that's data leakage: goals decide the winner.

## Approach

| Step | What | Where |
|---|---|---|
| Load | Kaggle CSVs into SQLite; removed 2,570 exact duplicate rows in `game.csv` and 5,140 in `game_teams_stats.csv` | `sql/load_data.py` |
| Explore | Does shot volume predict winning, per game and per season? | `notebooks/01_shot_volume.ipynb` |
| Model | Luck simulation, year-to-year repeatability, regression to the mean, linear and logistic regression with a season-based train/test split and baselines | `notebooks/02_skill_vs_luck.ipynb` |
| Dashboard | Streamlit app: regression watch by season, skill vs. luck charts, and a k-nearest-neighbors "most similar past teams" lookup | `dashboard/app.py` |

**Tools:** Python, SQL (SQLite: joins including a self-join, `GROUP BY`, `HAVING`), pandas, NumPy, scikit-learn, statsmodels, matplotlib, Plotly, Streamlit.

**Key decisions:**

- **Regular season only.** Playoff participation depends on team quality, so including it would bias the sample.
- **Full 82-game seasons only.** The 2012-13 lockout and 2019-20 COVID seasons are excluded so every season is comparable. The 2004-05 season was cancelled.
- **Train/test split by season, not at random.** A forecast is always built from the past and used on the future, so the model is never trained on data from after the seasons it's tested on.
- **Every model is compared against simple baselines** ("same as last year" and "league average"). A model is only useful if it beats them.

## Limitations

- The luck estimate treats every game as a 50/50 coin flip, ignoring home ice and overtime/shootout rules.
- The forecast test set is small: 61 team-season pairs.
- Every shot counts the same. There is no shot-quality (expected goals) data.
- Regression to the mean can't see real improvement. The model forecast the 2017-18 Lightning to drop from 54 wins to 46; they won 62 the next season.
- PDO is a well-established idea in hockey analytics. This project tests it on 17 seasons of data and builds a validated forecast on top of it; it doesn't introduce it.

## Run it yourself

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python sql/load_data.py          # builds nhl.db from the CSVs in data/
streamlit run dashboard/app.py   # opens the dashboard in your browser
```

The three CSVs this project uses are included in `data/`. The full dataset is [NHL Game Data](https://www.kaggle.com/datasets/martinellis/nhl-game-data) by Martin Ellis on Kaggle.

## Repo structure

```
data/         game.csv, game_teams_stats.csv, team_info.csv, table_relationships.JPG
sql/          load_data.py
notebooks/    01_shot_volume.ipynb, 02_skill_vs_luck.ipynb
dashboard/    app.py
figures/      charts used in this README
```
