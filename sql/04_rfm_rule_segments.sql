-- 04_rfm_rule_segments.sql
-- Simple RFM rule: recency, frequency and monetary quintile scores (1-5) with cut points taken
-- from the dev window only, summed (3-15), then collapsed into groups using dev score quantiles.
-- Inputs: rfm_dev (dev snapshot), rfm_target (snapshot to score). Both have customer_id,
--   recency_days, frequency_invoices, monetary_net_gbp.
-- Parameters: $quintile_probs (e.g. [0.2, 0.4, 0.6, 0.8]), $group_probs (e.g. [0.25, 0.5, 0.75]).
-- Output: customer_id, r_score, f_score, m_score, rfm_score, rfm_group (0 = lowest score).
-- A value equal to a cut point falls in the lower band.

WITH cuts AS (
    SELECT
        quantile_cont(-recency_days, $quintile_probs) AS r_cuts,
        quantile_cont(frequency_invoices, $quintile_probs) AS f_cuts,
        quantile_cont(monetary_net_gbp, $quintile_probs) AS m_cuts
    FROM rfm_dev
),
dev_scores AS (
    SELECT
        1 + len(list_filter(r_cuts, lambda c: c < -d.recency_days))
        + 1 + len(list_filter(f_cuts, lambda c: c < d.frequency_invoices))
        + 1 + len(list_filter(m_cuts, lambda c: c < d.monetary_net_gbp)) AS rfm_score
    FROM rfm_dev d, cuts
),
group_cuts AS (
    SELECT quantile_cont(rfm_score, $group_probs) AS g_cuts FROM dev_scores
),
scored AS (
    SELECT
        t.customer_id,
        1 + len(list_filter(r_cuts, lambda c: c < -t.recency_days)) AS r_score,
        1 + len(list_filter(f_cuts, lambda c: c < t.frequency_invoices)) AS f_score,
        1 + len(list_filter(m_cuts, lambda c: c < t.monetary_net_gbp)) AS m_score
    FROM rfm_target t, cuts
)
SELECT
    s.customer_id,
    s.r_score,
    s.f_score,
    s.m_score,
    s.r_score + s.f_score + s.m_score AS rfm_score,
    len(list_filter(g.g_cuts, lambda c: c < s.r_score + s.f_score + s.m_score)) AS rfm_group
FROM scored s, group_cuts g
ORDER BY s.customer_id;
