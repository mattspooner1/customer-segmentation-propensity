-- 01_clean_lines.sql
-- Input view: raw_lines (the raw invoice-line table, original column names).
-- Parameters: $cancel_prefix, $non_product_codes, $non_product_prefixes, $drop_duplicates,
--   $giant_line_qty.
-- Output tables: line_status (every line with one status), purchases, cancellations.
--
-- Each line gets exactly one status, assigned in this priority order:
--   missing_customer_id > non_product_code > price_le_zero > neg_qty_non_cancellation
--   > exact_duplicate (only when $drop_duplicates) > cancellation > purchase
-- Then a purchase of $giant_line_qty+ units and the later cancellation that exactly reverses it
-- (same customer, product, quantity and price, matched one to one in time order) both become
-- reversed_giant_line, so the mistaken order nets to zero everywhere.

CREATE OR REPLACE TABLE flagged_lines AS
SELECT
    line_no,
    Invoice AS invoice,
    StockCode AS stock_code,
    Description AS description,
    Quantity AS quantity,
    CAST(InvoiceDate AS TIMESTAMP) AS invoice_ts,
    Price AS price,
    CAST("Customer ID" AS BIGINT) AS customer_id,
    Country AS country,
    starts_with(Invoice, $cancel_prefix) AS is_cancellation,
    "Customer ID" IS NULL AS is_missing_customer,
    (
        list_contains($non_product_codes, StockCode)
        OR len(list_filter($non_product_prefixes, lambda p: starts_with(StockCode, p))) > 0
    ) AS is_non_product,
    Price <= 0 AS is_price_le_zero,
    (Quantity < 0 AND NOT starts_with(Invoice, $cancel_prefix)) AS is_neg_qty_non_cancellation,
    row_number() OVER (
        PARTITION BY Invoice, StockCode, Description, Quantity, InvoiceDate, Price, "Customer ID", Country
        ORDER BY line_no
    ) > 1 AS is_exact_duplicate
FROM raw_lines;

CREATE OR REPLACE TABLE line_status AS
SELECT
    *,
    quantity * price AS line_value,
    CASE
        WHEN is_missing_customer THEN 'missing_customer_id'
        WHEN is_non_product THEN 'non_product_code'
        WHEN is_price_le_zero THEN 'price_le_zero'
        WHEN is_neg_qty_non_cancellation THEN 'neg_qty_non_cancellation'
        WHEN is_exact_duplicate AND $drop_duplicates THEN 'exact_duplicate'
        WHEN is_cancellation THEN 'cancellation'
        ELSE 'purchase'
    END AS status
FROM flagged_lines;

CREATE OR REPLACE TABLE giant_matches AS
WITH gp AS (
    SELECT
        line_no, customer_id, stock_code, quantity AS qty, price, invoice_ts,
        row_number() OVER (PARTITION BY customer_id, stock_code, quantity, price ORDER BY invoice_ts, line_no) AS k
    FROM line_status
    WHERE status = 'purchase' AND quantity >= $giant_line_qty
),
gc AS (
    SELECT
        line_no, customer_id, stock_code, -quantity AS qty, price, invoice_ts,
        row_number() OVER (PARTITION BY customer_id, stock_code, -quantity, price ORDER BY invoice_ts, line_no) AS k
    FROM line_status
    WHERE status = 'cancellation' AND -quantity >= $giant_line_qty
)
SELECT gp.line_no AS purchase_line, gc.line_no AS cancel_line
FROM gp
JOIN gc USING (customer_id, stock_code, qty, price, k)
WHERE gc.invoice_ts >= gp.invoice_ts;

CREATE OR REPLACE TABLE line_status AS
SELECT
    * EXCLUDE (status),
    CASE
        WHEN line_no IN (SELECT purchase_line FROM giant_matches) THEN 'reversed_giant_line'
        WHEN line_no IN (SELECT cancel_line FROM giant_matches) THEN 'reversed_giant_line'
        ELSE status
    END AS status
FROM line_status;

CREATE OR REPLACE TABLE purchases AS
SELECT invoice, stock_code, quantity, invoice_ts, price, customer_id, country, line_value
FROM line_status
WHERE status = 'purchase';

CREATE OR REPLACE TABLE cancellations AS
SELECT invoice, stock_code, quantity, invoice_ts, price, customer_id, country, abs(line_value) AS cancel_value
FROM line_status
WHERE status = 'cancellation';
