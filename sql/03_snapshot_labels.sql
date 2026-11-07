-- 03_snapshot_labels.sql
-- Label: did the customer make any purchase in [cutoff, cutoff + label_window_days)?
-- Inputs: snapshot_features (from 02), purchases (from 01).
-- Parameters: $cutoff, $label_window_days.
-- Output: one row per snapshot customer with all features and y.

SELECT
    f.*,
    CASE WHEN b.customer_id IS NULL THEN 0 ELSE 1 END AS y
FROM snapshot_features f
LEFT JOIN (
    SELECT DISTINCT customer_id
    FROM purchases
    WHERE invoice_ts >= CAST($cutoff AS TIMESTAMP)
      AND invoice_ts < CAST($cutoff AS TIMESTAMP) + to_days(CAST($label_window_days AS INTEGER))
) b USING (customer_id)
ORDER BY f.customer_id;
