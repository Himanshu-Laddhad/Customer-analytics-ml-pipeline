-- Transaction features per customer
--
-- Dependencies (created in notebook before this query runs):
--   VIEW  transactions  = UNION ALL of transactions.csv + transactions_v2.csv
--   TABLE train_labels  = train_v2.csv loaded via DuckDB
--
-- Column name is 'msno' (not customer_id) — KKBox-specific.
-- INNER JOIN on train_labels keeps only the 970,960 labelled customers.
-- actual_amount_paid can exceed plan_list_price (QC-flagged anomaly); discount_rate
-- is clamped to avoid negative values misleading the model.

SELECT
    t.msno,

    -- Payment behaviour
    SUM(t.actual_amount_paid)                                                        AS total_paid,
    AVG(t.actual_amount_paid)                                                        AS avg_payment,
    MAX(t.actual_amount_paid)                                                        AS max_payment,
    AVG(
        CASE
            WHEN t.plan_list_price > 0
            THEN GREATEST((t.plan_list_price - t.actual_amount_paid) * 1.0 / t.plan_list_price, 0)
            ELSE NULL
        END
    )                                                                                AS discount_rate,

    -- Subscription behaviour
    COUNT(*)                                                                         AS n_transactions,
    AVG(t.payment_plan_days)                                                         AS avg_plan_days,
    AVG(CAST(t.is_auto_renew AS DOUBLE))                                             AS auto_renew_rate,
    AVG(CAST(t.is_cancel    AS DOUBLE))                                             AS cancel_rate,

    -- Tenure (dates stored as YYYYMMDD integers)
    MIN(t.transaction_date)                                                          AS first_transaction,
    MAX(t.transaction_date)                                                          AS last_transaction,
    DATEDIFF(
        'day',
        STRPTIME(CAST(MIN(t.transaction_date) AS VARCHAR), '%Y%m%d'),
        STRPTIME(CAST(MAX(t.transaction_date) AS VARCHAR), '%Y%m%d')
    )                                                                                AS days_as_member,
    MAX(t.membership_expire_date)                                                    AS last_expire_date,

    -- Payment method diversity
    COUNT(DISTINCT t.payment_method_id)                                              AS n_payment_methods

FROM transactions t
INNER JOIN train_labels lbl ON t.msno = lbl.msno
GROUP BY t.msno
