-- ============================================================
-- Churn Label Queries
-- Source table : transactions  (outputs/retail_clean.parquet)
-- Observation window : InvoiceDate  < 2010-12-01  (features)
-- Prediction window  : InvoiceDate >= 2010-12-01  (target)
-- Churn definition   : 0 purchases in prediction window,
--                      given >= 2 purchases in observation window
-- All queries end with ';' so the notebook can split on ';\n'
-- ============================================================


-- ============================================================
-- Query 1: Observation Window Customers
-- Customers with >= 2 distinct invoices before 2010-12-01.
-- First-time buyers (frequency = 1) are excluded — too little
-- signal to reliably label churn behaviour.
-- ============================================================
SELECT
    "Customer ID"           AS customer_id,
    COUNT(DISTINCT Invoice) AS obs_frequency
FROM transactions
WHERE InvoiceDate < TIMESTAMP '2010-12-01 00:00:00'
GROUP BY "Customer ID"
HAVING COUNT(DISTINCT Invoice) >= 2
ORDER BY customer_id
;

-- ============================================================
-- Query 2: Prediction Window Customers
-- Customers with at least one invoice on or after 2010-12-01.
-- These are the "retained" candidates in the churn definition.
-- ============================================================
SELECT DISTINCT
    "Customer ID" AS customer_id
FROM transactions
WHERE InvoiceDate >= TIMESTAMP '2010-12-01 00:00:00'
ORDER BY customer_id
;

-- ============================================================
-- Query 3: Final Churn Labels
-- LEFT JOIN observation customers onto prediction customers.
--   churned = 1  → no purchase in prediction window  (churned)
--   churned = 0  → at least one purchase in prediction window
-- Only customers with >= 2 obs-window purchases receive a label.
-- Self-contained: re-derives obs and pred via CTEs.
-- ============================================================
WITH obs AS (
    SELECT
        "Customer ID"           AS customer_id,
        COUNT(DISTINCT Invoice) AS obs_frequency
    FROM transactions
    WHERE InvoiceDate < TIMESTAMP '2010-12-01 00:00:00'
    GROUP BY "Customer ID"
    HAVING COUNT(DISTINCT Invoice) >= 2
),
pred AS (
    SELECT DISTINCT
        "Customer ID" AS customer_id
    FROM transactions
    WHERE InvoiceDate >= TIMESTAMP '2010-12-01 00:00:00'
)
SELECT
    o.customer_id,
    o.obs_frequency,
    CASE WHEN p.customer_id IS NULL THEN 1 ELSE 0 END AS churned
FROM obs o
LEFT JOIN pred p USING (customer_id)
ORDER BY customer_id
;
