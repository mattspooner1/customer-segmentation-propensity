-- 02_customer_snapshot_features.sql
-- Customer features at one snapshot cutoff, using only lines before the cutoff.
-- Inputs: purchases, cancellations (from 01_clean_lines.sql).
-- Parameters: $cutoff (timestamp string), $lookback_days, $q4_months.
-- Population: customers with at least one purchase in [cutoff - lookback_days, cutoff).
-- Output table: snapshot_features (one row per customer).

CREATE OR REPLACE TABLE snapshot_features AS
WITH params AS (
    SELECT
        CAST($cutoff AS TIMESTAMP) AS cutoff,
        CAST($cutoff AS TIMESTAMP) - to_days(CAST($lookback_days AS INTEGER)) AS lookback_start
),
p AS (
    SELECT pu.*
    FROM purchases pu, params
    WHERE pu.invoice_ts >= params.lookback_start AND pu.invoice_ts < params.cutoff
),
c AS (
    SELECT ca.*
    FROM cancellations ca, params
    WHERE ca.invoice_ts >= params.lookback_start AND ca.invoice_ts < params.cutoff
),
inv AS (
    SELECT customer_id, invoice, min(invoice_ts) AS inv_ts, sum(line_value) AS inv_value, count(*) AS n_lines
    FROM p
    GROUP BY customer_id, invoice
),
gaps AS (
    SELECT
        customer_id,
        date_diff('second', lag(inv_ts) OVER (PARTITION BY customer_id ORDER BY inv_ts, invoice), inv_ts)
            / 86400.0 AS gap_days
    FROM inv
),
inv_agg AS (
    SELECT
        inv.customer_id,
        max(inv_ts) AS last_ts,
        count(*) AS frequency_invoices,
        sum(CASE WHEN inv_ts >= params.cutoff - INTERVAL 90 DAY THEN 1 ELSE 0 END) AS frequency_invoices_last_90d,
        sum(inv_value) AS purchase_value,
        avg(inv_value) AS avg_order_value_gbp,
        avg(n_lines) AS avg_lines_per_invoice
    FROM inv, params
    GROUP BY inv.customer_id
),
gap_agg AS (
    SELECT customer_id, avg(gap_days) AS mean_gap
    FROM gaps
    WHERE gap_days IS NOT NULL
    GROUP BY customer_id
),
line_agg AS (
    SELECT
        customer_id,
        median(quantity) AS median_units_per_line,
        count(DISTINCT stock_code) AS n_distinct_products,
        count(DISTINCT date_trunc('month', invoice_ts)) AS active_months
    FROM p
    GROUP BY customer_id
),
canc AS (
    SELECT customer_id, sum(cancel_value) AS cancel_value
    FROM c
    GROUP BY customer_id
),
first_ever AS (
    SELECT customer_id, min(invoice_ts) AS first_ts
    FROM purchases, params
    WHERE invoice_ts < params.cutoff
    GROUP BY customer_id
),
modal_country AS (
    SELECT customer_id, country
    FROM (
        SELECT
            customer_id,
            country,
            row_number() OVER (PARTITION BY customer_id ORDER BY count(*) DESC, country ASC) AS rn
        FROM p
        GROUP BY customer_id, country
    )
    WHERE rn = 1
)
SELECT
    i.customer_id,
    date_diff('day', CAST(i.last_ts AS DATE), CAST(params.cutoff AS DATE)) AS recency_days,
    i.frequency_invoices,
    i.frequency_invoices_last_90d,
    greatest(i.purchase_value - coalesce(k.cancel_value, 0), 0) AS monetary_net_gbp,
    i.avg_order_value_gbp,
    i.avg_lines_per_invoice,
    l.median_units_per_line,
    l.n_distinct_products,
    l.active_months,
    date_diff('day', CAST(f.first_ts AS DATE), CAST(params.cutoff AS DATE)) AS tenure_days,
    coalesce(g.mean_gap, CAST($lookback_days AS DOUBLE)) AS mean_interpurchase_days,
    CAST(i.frequency_invoices = 1 AS INTEGER) AS single_invoice_flag,
    coalesce(k.cancel_value, 0) / (i.purchase_value + coalesce(k.cancel_value, 0)) AS return_rate_value,
    CAST(m.country = 'United Kingdom' AS INTEGER) AS is_uk,
    CAST(list_contains($q4_months, month(params.cutoff)) AS INTEGER) AS label_window_q4
FROM inv_agg i
CROSS JOIN params
LEFT JOIN gap_agg g USING (customer_id)
LEFT JOIN line_agg l USING (customer_id)
LEFT JOIN canc k USING (customer_id)
LEFT JOIN first_ever f USING (customer_id)
LEFT JOIN modal_country m USING (customer_id)
ORDER BY i.customer_id;
