-- Behavioural features per customer from daily listening logs
--
-- Feature spec (final output columns):
--   total_secs, avg_daily_secs, max_daily_secs
--   total_songs_played, completion_rate, skip_rate, unique_songs_ratio
--   active_days, last_log_date, first_log_date
--   days_since_last_log  (computed in Python, snapshot_date = 2017-03-31)
--
-- Execution note: user_logs.csv (392M rows) + user_logs_v2.csv (18M rows) are processed
-- in two separate passes in the notebook due to RAM constraints (12.5 GB machine).
-- The intermediate aggregation SQL (AGG_SQL) in notebook 03 outputs raw sums/counts
-- so the two passes can be correctly combined in Python before computing final ratios.
-- date column is stored as YYYYMMDD integer. Column ID is 'msno' (not customer_id).
--
-- Reference single-file query (for documentation / smaller datasets):

SELECT
    l.msno,

    -- Volume
    SUM(l.total_secs)                                                            AS total_secs,
    AVG(l.total_secs)                                                            AS avg_daily_secs,
    MAX(l.total_secs)                                                            AS max_daily_secs,

    -- Engagement quality
    SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100)                AS total_songs_played,

    SUM(l.num_100) * 1.0
        / NULLIF(SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100), 0)
                                                                                 AS completion_rate,

    SUM(l.num_25) * 1.0
        / NULLIF(SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100), 0)
                                                                                 AS skip_rate,

    AVG(
        (l.num_unq * 1.0)
        / NULLIF(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100, 0)
    )                                                                            AS unique_songs_ratio,

    -- Recency and consistency
    COUNT(DISTINCT l.date)                                                       AS active_days,
    MAX(l.date)                                                                  AS last_log_date,
    MIN(l.date)                                                                  AS first_log_date

FROM user_logs l
INNER JOIN train_labels lbl ON l.msno = lbl.msno
GROUP BY l.msno
