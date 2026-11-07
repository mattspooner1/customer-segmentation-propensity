# Data dictionary

## Raw data (`data/raw/online_retail_II.csv`, not committed)
Online Retail II, UCI Machine Learning Repository (doi:10.24432/C5CG6D, CC BY 4.0). 1,067,371 rows, one per invoice line.

| Column | Type | Description |
|---|---|---|
| Invoice | text | Invoice number. A leading `C` marks a cancellation |
| StockCode | text | Product code. Non-product codes (postage, fees, adjustments, vouchers) are listed in `config.yaml` |
| Description | text | Product description |
| Quantity | integer | Units on the line; negative on cancellations |
| InvoiceDate | timestamp | Invoice date and time |
| Price | decimal | Unit price, GBP |
| Customer ID | integer | Pseudonymous customer number; missing on 22.8% of lines |
| Country | text | Customer's country |

## Cleaned lines (DuckDB tables built by `sql/01_clean_lines.sql`)
| Table | Contents |
|---|---|
| `line_status` | Every raw line with `line_value` = Quantity × Price and one `status`: missing_customer_id, non_product_code, price_le_zero, neg_qty_non_cancellation, exact_duplicate, cancellation, purchase or reversed_giant_line |
| `purchases` | Lines with status purchase |
| `cancellations` | Lines with status cancellation; `cancel_value` is the positive value cancelled |

## Customer snapshot (`data/processed/snapshot_<window>.csv`, not committed)
One row per customer with at least one purchase in the 273 days before the cutoff. Built by `sql/02_customer_snapshot_features.sql` and `sql/03_snapshot_labels.sql`. "Lookback" means the 273 days before the cutoff.

| Column | Type | Definition | Used in |
|---|---|---|---|
| customer_id | integer | Pseudonymous customer number | key |
| recency_days | integer | Days from the last lookback invoice to the cutoff | segmentation (A, B), propensity |
| frequency_invoices | integer | Distinct purchase invoices in the lookback | segmentation (A, B), propensity |
| frequency_invoices_last_90d | integer | Purchase invoices in the 90 days before the cutoff | propensity |
| monetary_net_gbp | decimal | Purchase value minus cancelled value in the lookback, floored at 0 | segmentation (A, B), propensity |
| avg_order_value_gbp | decimal | Mean purchase value per invoice | segmentation (B), propensity, illustrative scenario |
| avg_lines_per_invoice | decimal | Mean number of lines per invoice | propensity |
| median_units_per_line | decimal | Median units per purchase line (high values indicate wholesale buying) | segmentation (B), propensity |
| n_distinct_products | integer | Distinct stock codes bought | segmentation (B), propensity |
| active_months | integer | Distinct calendar months with a purchase | propensity |
| tenure_days | integer | Days from the customer's first purchase in the data to the cutoff | propensity |
| mean_interpurchase_days | decimal | Mean days between consecutive invoices; 273 for single-invoice customers | propensity |
| single_invoice_flag | 0/1 | 1 if the customer has one invoice in the lookback | propensity |
| return_rate_value | decimal | Cancelled value / (purchase value + cancelled value) | segmentation (B), propensity |
| is_uk | 0/1 | 1 if the customer's most frequent country is the United Kingdom | propensity |
| label_window_q4 | 0/1 | 1 if the cutoff month is September-December, so the label period is the seasonal peak | propensity only |
| y | 0/1 | **Label:** 1 if the customer made any purchase in the 90 days from the cutoff | outcome |

Segmentation set A (RFM core) is the one used; set B was evaluated and not selected (see [METHODOLOGY.md](METHODOLOGY.md)).

## Test predictions (`data/processed/test_predictions.csv`, not committed)
| Column | Description |
|---|---|
| customer_id | Pseudonymous customer number |
| y | Test label |
| p | Propensity model score. **Use as a ranking only**; it failed the calibration check |
| segment | Frozen k-means segment: 0 lapsed infrequent low-value, 1 cooling regular mid-value, 2 recent frequent high-value |
| rule_group | RFM quintile rule group: 0 low, 1 middle, 2 high |

## Committed outputs (`outputs/metrics/`)
| File | Contents |
|---|---|
| `results.json` | Every test metric. Each entry has id, model, split, metric, value, ci_low, ci_high, n, n_events, baseline_name, baseline_value, train_value and verdict. Also gate status flags, calibration values, within-segment AUCs, segment shares and rates, and selection context |
| `evidence_gate.md` | Human-readable gate table and outcome, generated from `results.json` |
| `segmentation_selection.json` | k grids, stability, val comparisons and dev profiles from the selection step (dev and val windows only) |
| `data_summary.json` | Cleaning log (lines and value by status), flag counts, customers and events per snapshot, and `window_overlap`: test-window customers who are also in the segment-fitting window (customer IDs counted only) |
| `insights.csv` | The five insights (notebook 04) with evidence, basis (test evaluation in `results.json`, or post hoc and descriptive from the frozen test predictions) and verdict. The verdict column holds only a computed verdict from `results.json` (I1: S1 STRONG); rows without one are blank |
| `supplementary_within_group_auc.csv` | Propensity AUC within each k-means segment (copied from `results.json`) and each RFM rule group (computed post hoc), with CIs |
| `rule_group_rates.csv` | Test-window customers, share and 90-day repeat rate in each RFM rule group |
| `supplementary_list_capture.csv` | Post hoc: share of repeat buyers and repeat rate in a 500-customer list for each ranking, with CIs (illustrative scenario only; not a paired test) |
| `illustrative_scenario.csv` | Contact-list scenario: observed outcomes and break-even arithmetic (assumptions A1-A3) |

## Assumptions (`assumptions.yaml`)
Business figures used only in the illustrative scenario: A1 list size, A2 cost per contact, A3 margin rate. All are unsourced assumptions. A4 (effect of contact) is recorded as not estimable.
