-- =============================================================================
-- upi_fraud_analytics.sql
-- Advanced SQL analytics for UPI Transaction Fraud Detection
-- Uses: Window Functions · CTEs · Subqueries · Aggregations
-- Compatible with: PostgreSQL 13+ / MySQL 8+
-- =============================================================================


-- ── 1. Overall Fraud Rate Summary ────────────────────────────────────────────
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(is_fraud)                                     AS total_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)        AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount END), 2) AS total_fraud_amount,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN amount END), 2) AS avg_fraud_amount
FROM upi_transactions;


-- ── 2. Fraud by Hour of Day (velocity pattern) ────────────────────────────────
SELECT
    hour_of_day,
    COUNT(*)                                    AS txn_count,
    SUM(is_fraud)                               AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)  AS fraud_rate_pct
FROM upi_transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;


-- ── 3. High-Risk Senders — Window Function ─────────────────────────────────
-- Ranks senders by total fraud amount; flags those above 95th percentile
WITH sender_stats AS (
    SELECT
        sender_upi,
        COUNT(*)                       AS txn_count,
        SUM(is_fraud)                  AS fraud_count,
        SUM(amount)                    AS total_amount,
        SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END) AS fraud_amount,
        ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)       AS fraud_rate_pct
    FROM upi_transactions
    GROUP BY sender_upi
),
ranked AS (
    SELECT *,
        PERCENT_RANK() OVER (ORDER BY fraud_amount) AS fraud_amount_percentile,
        RANK()         OVER (ORDER BY fraud_count DESC) AS fraud_rank
    FROM sender_stats
)
SELECT *
FROM ranked
WHERE fraud_amount_percentile >= 0.95
ORDER BY fraud_amount DESC
LIMIT 50;


-- ── 4. Rolling 7-Day Fraud Rate (time-series) ─────────────────────────────
WITH daily AS (
    SELECT
        DATE(timestamp)                             AS txn_date,
        COUNT(*)                                    AS daily_txns,
        SUM(is_fraud)                               AS daily_fraud
    FROM upi_transactions
    GROUP BY DATE(timestamp)
)
SELECT
    txn_date,
    daily_txns,
    daily_fraud,
    ROUND(
        100.0 * SUM(daily_fraud) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
              / SUM(daily_txns)  OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
        2
    ) AS rolling_7d_fraud_rate_pct,
    SUM(daily_txns)  OVER (ORDER BY txn_date) AS cumulative_txns,
    SUM(daily_fraud) OVER (ORDER BY txn_date) AS cumulative_fraud
FROM daily
ORDER BY txn_date;


-- ── 5. Category × Bank Fraud Heatmap ──────────────────────────────────────
SELECT
    category,
    bank,
    COUNT(*)                                    AS txns,
    SUM(is_fraud)                               AS fraud_txns,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)  AS fraud_pct,
    ROUND(AVG(amount), 2)                       AS avg_amount
FROM upi_transactions
GROUP BY category, bank
ORDER BY fraud_pct DESC
LIMIT 30;


-- ── 6. RFM Segmentation Query ────────────────────────────────────────────────
-- Recency-Frequency-Monetary analysis per sender
WITH rfm_base AS (
    SELECT
        sender_upi,
        DATEDIFF(
            (SELECT MAX(timestamp) FROM upi_transactions),
            MAX(timestamp)
        )                             AS recency_days,
        COUNT(*)                      AS frequency,
        SUM(amount)                   AS monetary
    FROM upi_transactions
    GROUP BY sender_upi
),
rfm_scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,   -- lower = more recent
        NTILE(4) OVER (ORDER BY frequency)          AS f_score,
        NTILE(4) OVER (ORDER BY monetary)           AS m_score
    FROM rfm_base
)
SELECT *,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN (r_score + f_score + m_score) >= 10 THEN 'High Risk'
        WHEN (r_score + f_score + m_score) >= 6  THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_segment
FROM rfm_scored
ORDER BY rfm_total DESC
LIMIT 100;


-- ── 7. Anomaly Detection — Regex-Like Pattern Matching ─────────────────────
-- Flags UPI IDs matching suspicious numeric or keyword patterns
-- (PostgreSQL SIMILAR TO / MySQL REGEXP)
SELECT
    transaction_id,
    sender_upi,
    amount,
    timestamp,
    is_fraud,
    'suspicious_upi_pattern' AS flag_reason
FROM upi_transactions
WHERE sender_upi SIMILAR TO '%(usr|anon|tmp|test|fake)[0-9]+[xzqXZQ]?@%'
   OR sender_upi ~ '^\d{6,}@'        -- numeric-heavy username
UNION ALL
SELECT
    transaction_id,
    sender_upi,
    amount,
    timestamp,
    is_fraud,
    'high_velocity' AS flag_reason
FROM upi_transactions
WHERE velocity_1hr >= 15
  AND is_fraud = 0   -- potential undetected fraud
ORDER BY amount DESC
LIMIT 200;


-- ── 8. False Negative Analysis (subquery) ────────────────────────────────────
-- Transactions the model flagged as legit but turned out fraudulent
-- (requires a predictions table joined to ground truth)
SELECT
    t.transaction_id,
    t.amount,
    t.sender_upi,
    t.hour_of_day,
    t.velocity_1hr,
    t.is_fraud         AS actual_label,
    p.predicted_fraud  AS predicted_label,
    p.fraud_probability
FROM upi_transactions t
JOIN predictions p ON t.transaction_id = p.transaction_id
WHERE t.is_fraud = 1
  AND p.predicted_fraud = 0
ORDER BY t.amount DESC;


-- ── 9. Top 10 Fraud Amount by Bank ───────────────────────────────────────────
SELECT
    bank,
    SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END) AS total_fraud_amount,
    COUNT(CASE WHEN is_fraud=1 THEN 1 END)           AS fraud_count,
    ROUND(
        100.0 * COUNT(CASE WHEN is_fraud=1 THEN 1 END) / COUNT(*), 2
    )                                                AS fraud_rate_pct,
    RANK() OVER (ORDER BY SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END) DESC) AS rnk
FROM upi_transactions
GROUP BY bank
ORDER BY rnk;
