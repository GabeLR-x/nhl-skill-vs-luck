-- Window-function queries on nhl.db (build it with: python sql/load_data.py).
-- Run and check them against pandas with: python sql/check_analysis.py
-- Each query starts with a "-- name:" line so the checker can find it.


-- name: running_wins
-- Running win total for each team through each regular season, game by game.
-- SUM() OVER (... ORDER BY ...) adds up every row from the season's first game to this one.
SELECT s.team_id,
       g.season,
       g.date_time_GMT,
       s.game_id,
       s.won,
       SUM(s.won) OVER (
           PARTITION BY s.team_id, g.season
           ORDER BY g.date_time_GMT, s.game_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_wins
FROM game_teams_stats AS s
JOIN game AS g ON s.game_id = g.game_id
WHERE g.type = 'R'
ORDER BY s.team_id, g.season, g.date_time_GMT, s.game_id;


-- name: rolling_shot_share
-- Share of shots taken over each team's last 10 games (fewer at the start of a season).
-- The frame "9 PRECEDING AND CURRENT ROW" is the 10-game window.
WITH per_game AS (
    SELECT us.team_id,
           g.season,
           g.date_time_GMT,
           us.game_id,
           us.shots   AS shots_for,
           them.shots AS shots_against
    FROM game_teams_stats AS us
    JOIN game_teams_stats AS them
      ON us.game_id = them.game_id AND us.team_id <> them.team_id
    JOIN game AS g ON us.game_id = g.game_id
    WHERE g.type = 'R'
)
SELECT team_id, season, date_time_GMT, game_id,
       SUM(shots_for) OVER w * 1.0
         / (SUM(shots_for) OVER w + SUM(shots_against) OVER w) AS shot_share_10
FROM per_game
WINDOW w AS (
    PARTITION BY team_id, season
    ORDER BY date_time_GMT, game_id
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
)
ORDER BY team_id, season, date_time_GMT, game_id;


-- name: pdo_rank
-- Luck score (PDO = shooting % + save %) for every full 82-game team-season,
-- ranked within its season: 1 = luckiest team that year.
WITH team_seasons AS (
    SELECT us.team_id,
           g.season,
           SUM(us.goals) * 1.0 / SUM(us.shots)
             + 1 - SUM(them.goals) * 1.0 / SUM(them.shots) AS pdo
    FROM game_teams_stats AS us
    JOIN game_teams_stats AS them
      ON us.game_id = them.game_id AND us.team_id <> them.team_id
    JOIN game AS g ON us.game_id = g.game_id
    WHERE g.type = 'R'
    GROUP BY us.team_id, g.season
    HAVING COUNT(*) = 82
)
SELECT team_id, season, pdo,
       RANK() OVER (PARTITION BY season ORDER BY pdo DESC) AS pdo_rank
FROM team_seasons
ORDER BY season, pdo_rank;


-- name: win_rate_change
-- Year-over-year change in win rate with LAG(). Only back-to-back full seasons
-- count (season minus previous season = 10001, e.g. 20172018 - 20162017), the
-- same rule as the 419 season pairs in the notebooks.
WITH team_seasons AS (
    SELECT s.team_id,
           g.season,
           SUM(s.won) * 1.0 / COUNT(*) AS win_rate
    FROM game_teams_stats AS s
    JOIN game AS g ON s.game_id = g.game_id
    WHERE g.type = 'R'
    GROUP BY s.team_id, g.season
    HAVING COUNT(*) = 82
),
with_prev AS (
    SELECT team_id, season, win_rate,
           LAG(season)   OVER (PARTITION BY team_id ORDER BY season) AS prev_season,
           LAG(win_rate) OVER (PARTITION BY team_id ORDER BY season) AS prev_win_rate
    FROM team_seasons
)
SELECT team_id, prev_season, season, prev_win_rate, win_rate,
       win_rate - prev_win_rate AS change
FROM with_prev
WHERE season - prev_season = 10001
ORDER BY team_id, season;
