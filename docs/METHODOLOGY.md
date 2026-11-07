# Methodology

> **Illustrative scenario.** The business framing is invented. The data is real (a UK online gift retailer, 2009-2011, with many wholesale buyers, registered customers only) and contains no marketing contact data. "Repeat purchase" means any purchase within 90 days after a cutoff.

The analysis plan (windows, metrics, targets, comparators and outcome rules) was fixed before the segmentation and propensity work began. It was informed by an initial feasibility check that ran a prototype on the same test window; see [section 6](#6-disclosure-use-of-the-test-window). Changes made during the work, and when they were made, are listed in [section 7](#7-changes-to-the-plan).

## 1. Data
Online Retail II (UCI, CC BY 4.0) holds 1,067,371 invoice lines from 2009-12-01 to 2011-12-09. `scripts/download_data.py` verifies a SHA-256 checksum and the row and column counts. All cleaning and feature SQL runs in DuckDB (`sql/`), single-threaded so floating-point sums are reproducible.

### Cleaning
Each line gets exactly one status, in priority order (`sql/01_clean_lines.sql`). Counts come from `outputs/metrics/data_summary.json`.

| Status | Lines | Share of lines | Treatment |
|---|---|---|---|
| missing_customer_id | 243,007 | 22.8% | Excluded: cannot be attributed to a customer |
| non_product_code (postage, fees, adjustments, vouchers) | 3,736 | 0.4% | Excluded: not product demand |
| price_le_zero | 61 | 0.0% | Excluded |
| exact_duplicate | 26,402 | 2.5% | Excluded (refit with duplicates kept as a sensitivity check) |
| reversed_giant_line | 10 | 0.0% | A purchase of 5,000+ units and the cancellation that exactly reverses it are both removed |
| cancellation | 17,581 | 1.6% | Netted against purchases in the same window |
| purchase | 776,574 | 72.8% | Kept |

Statuses lower in the list only count lines not already caught by a higher one. For example, 6,207 lines have a price of zero or less, but only 61 of them have a customer ID and a product code. All 3,457 negative-quantity lines without a cancellation prefix are caught by a higher-priority status.

## 2. Customer snapshots
A snapshot describes every customer with at least one purchase in the 273 days before a cutoff (`sql/02_customer_snapshot_features.sql`). Features use only lines before the cutoff; the label is any purchase in the 90 days from the cutoff (`sql/03_snapshot_labels.sql`). A leakage test asserts that changing data after the cutoff does not change any feature.

| Window | Cutoff | Role | Customers | Repeat buyers | Rate |
|---|---|---|---|---|---|
| seasonal_2010 | 2010-09-11 | Propensity training | 3,243 | 1,898 | 0.585 |
| dev | 2011-03-15 | Segment fitting, propensity tuning and training | 3,856 | 1,539 | 0.399 |
| val | 2011-06-13 | Model and feature-set selection, propensity training | 3,905 | 1,503 | 0.385 |
| test | 2011-09-11 | Held out for evaluation (see section 6) | 3,238 | 1,868 | 0.577 |

Every training label window ends on or before the test cutoff. The seasonal_2010 window gives the propensity model one September-December label period like the test window's.

**Same customers, later behaviour.** 2,443 of the 3,238 test-window customers (75.4%) are also in the dev window the segments were fitted on (`outputs/metrics/data_summary.json`, `window_overlap`). Their features are recomputed at the test cutoff and their labels come from the 90 days after it, so the segment check is out-of-time, not a check on new customers.

## 3. Segmentation
- **Features.** Set A (RFM core): recency_days, frequency_invoices and monetary_net_gbp. Set B adds average order value, median units per line, distinct products and the returned-value share.
- **Preprocessing (dev only, then frozen).** Winsorise at the dev 99.5th percentile, apply log1p to skewed counts and values (not recency), then standardise.
- **Choosing k.** For k = 3 to 8 on dev, refit k-means on 200 bootstrap samples and score agreement with the full-data fit by the adjusted Rand index (ARI). Choose k strictly by the 5th percentile of ARI; every segment must hold at least 5% of customers. For set A, k=3 has ARI 5th percentile 0.882; the next best, k=6, has 0.865.
- **Choosing the feature set** on val by segment-score AUC: set A 0.753 (0.738-0.768) against set B 0.721 (0.706-0.737). Set B was only eligible at k=3, with ARI 5th percentile 0.329, because a small group of customers with high return shares forms its own cluster.
- **Segment score.** Each segment's dev repeat rate becomes the score for every customer in it, so segments can be ranked with AUC.
- **Names** come from the dev centroids: *lapsed infrequent low-value* (dev repeat rate 0.194), *cooling regular mid-value* (0.380) and *recent frequent high-value* (0.785). These dev rates are measured on the window the segments were fitted on, so they serve as scores, not evidence; every segment claim is judged on the later test window.

### Comparators
- **Random segments:** AUC 0.5, the no-skill baseline.
- **Recency only:** ranks customers by days since their last order.
- **RFM quintile rule** (`sql/04_rfm_rule_segments.sql`): quintile scores for recency, frequency and monetary value, using dev cut points, summed and cut into 3 groups at dev tertiles.

### Stability checks (dev window)
- Bootstrap ARI 5th percentile: 0.882 (median 0.927).
- Refit with the top 1% of customers by spend removed: ARI 0.865.
- Refit with exact duplicates kept: ARI 0.938.
- Dev model against the same recipe refitted on val (temporal): ARI 0.788. The segments are reasonably, not fully, stable over time.

## 4. Propensity model
- **Features:** the 15 snapshot features in [DATA_DICTIONARY.md](DATA_DICTIONARY.md), including a flag for a September-December label window.
- **L2 logistic regression.** C is chosen on dev by 5-fold stratified CV from {0.01, 0.1, 1, 10}. Mean CV AUC ranged only from 0.774 to 0.775 across the grid (fold standard deviation about 0.024), so the choice of 0.01, at the edge of the grid, has no practical effect.
- **Gradient boosting** (scikit-learn HistGradientBoosting, depth 3, learning rate 0.05, early stopping), run single-threaded for reproducibility.
- **Choice on val:** gradient boosting AUC 0.793 (0.779-0.808) against logistic 0.784 (0.769-0.798). The rule set in advance picks the higher val AUC.
- **Retraining:** the chosen model is refitted on the seasonal_2010, dev and val windows stacked together, then frozen.

## 5. Evaluation
`segmentation_propensity.gate.evaluate_test` scores every pre-declared metric on the test window in one call (section 6 says how many times it was run):
- **Intervals:** 95% percentile bootstrap CIs over 1,000 customer resamples. Paired differences resample the same customers for both scores.
- **Verdicts, computed in code.** STRONG: CI lower bound at least 0.10 above the baseline and the minimum useful target met. MODEST: CI lower bound above the baseline. WEAK: point estimate above the baseline but the CI includes it. NONE: at or below the baseline. INVALID: below 0.5. An OVERFIT flag is raised if train minus test exceeds 0.15.
- **Calibration:** slope and intercept from a logistic recalibration of the scores, plus calibration-in-the-large (mean score minus observed rate).
- **Outcome rule set in advance:** the project reports findings only if the segment score is STRONG or MODEST and the stability check is met. The k-means versus rule comparison decides which segmentation is recommended; it does not change the outcome.

### Test results (`outputs/metrics/results.json`)
| id | Metric | Value (95% CI) | Baseline | Target (minimum useful) | Verdict |
|---|---|---|---|---|---|
| S1 | Segment-score AUC, k-means k=3 | 0.712 (0.694-0.727); train 0.738 | 0.5 | 0.685 (prototype recency-only AUC, same window) | STRONG |
| ref | RFM rule AUC | 0.717 (0.700-0.733) | 0.5 | – | – |
| ref | Recency-only AUC | 0.685 (0.666-0.703) | 0.5 | – | – |
| S2 | k-means minus RFM rule | -0.005 (-0.014 to +0.004) | 0 | CI lower bound above 0 | Not met: rule recommended |
| S3 | Bootstrap ARI 5th percentile (dev) | 0.882 | 0 | 0.70 | Met |
| S4 | Recency + segments minus recency | +0.033 (+0.027 to +0.040) | 0 | CI lower bound above 0 | Met |
| S5 | Cramér's V, segments vs outcome | 0.395 | 0.044 (95th percentile of 200 permutations) | Above the permuted null | Met |
| P1 | Propensity AUC | 0.755 (0.738-0.771); train 0.804 | 0.5 | 0.725 (prototype segment-level AUC, same window) | STRONG |
| P2 | Calibration slope; mean score minus rate | 1.220; +0.043 | 1; 0 | 0.8-1.2; within 0.03 | Fails |
| P3 | Brier skill vs predicting the test rate | 0.183 (0.163-0.203) | 0 | Above 0 | Met |
| P4 | Propensity train minus test AUC | 0.049 | – | At most 0.15 | Met |

**Outcome: Findings.** Because P2 fails, the model is used for ranking only: no document states a probability or an expected number of buyers. The scores run too high for low- and mid-ranked customers; the September-December test label period is one plausible reason.

The RFM rule has no computed verdict; its interval is clear of chance. No paired test compares the propensity model with the segment or rule scores, so no claim is made that it ranks better than either.

**Propensity AUC within each frozen k-means segment** (computed in the same test evaluation; descriptive, no verdict): lapsed infrequent low-value 0.625 (0.589-0.660), cooling regular mid-value 0.624 (0.596-0.654), recent frequent high-value 0.750 (0.699-0.798). Every lower bound is above 0.5, so the model still orders customers inside each segment.

### Supplementary analyses
These were computed **post hoc**, after the test evaluation, from the frozen test predictions (`notebooks/04_business_insights.ipynb`). They are descriptive, use the same bootstrap, and informed no modelling choice.
- **Propensity AUC within each RFM rule group (post hoc):** low 0.613 (0.580-0.648), middle 0.590 (0.557-0.626), high 0.742 (0.705-0.778).
- **Repeat rate in a 500-customer list (post hoc):** 0.940 (0.910-0.962) for the model's top 500; 0.829 (0.809-0.851) for 500 customers picked at random from the top RFM group. These describe lists for the illustrative scenario. They are not a paired test and do not show that the model ranks better than the rule.

## 6. Disclosure: use of the test window
### Initial feasibility check on the same window
Before the analysis plan was fixed, an initial feasibility check ran a prototype of this approach on the same September 2011 test window. The prototype fitted k-means on recency, frequency and spend on the March 2011 window, then scored it, an RFM rule, recency alone and a logistic propensity model on the test window. Its results were used to:
- set the minimum-useful targets: 0.685 for S1 (the prototype's recency-only AUC) and 0.725 for P1 (the prototype's best segment-level AUC);
- make the k-means-versus-rule comparison (S2) a decision metric, because the prototype's k-means did not beat its rule;
- choose the propensity training design (adding a same-season 2010 window and a September-December flag), after a model trained only on spring and summer windows predicted too few repeat buyers on this window.

So the test window was not untouched when the plan was fixed. What this does and does not affect:
- The S1 and P1 intervals clear chance by more than 0.10 on their own: lower bounds 0.694 and 0.738 against a baseline of 0.5. Only the "minimum useful met" part of each STRONG verdict relied on targets set using this window.
- The S2 result (rule and k-means not distinguishable) agrees with what the prototype found on the same window, so it is not an independent confirmation of it.
- The training design chosen this way did not give calibrated scores: the final model fails P2.

### Six scorings of the frozen pipeline
The test window was scored six times in total: the first run, then five re-runs. One re-run checked a fix to a record-merging step. Four found and fixed run-to-run non-determinism: two after pinning gradient-boosting threads and two after making DuckDB single-threaded, because parallel floating-point sums differed in the last digits between runs. No feature, parameter, model or threshold changed after the first test score was seen, and the final numbers equal the first run exactly. Release checks also re-run the full pipeline from a fresh clone as a reproducibility test; such runs inform no choice, and they reproduce `results.json` byte-for-byte.

## 7. Changes to the plan
**Before segmentation (after data exploration):**
1. `active_months` was dropped from the segmentation feature sets: in dev it correlates at 0.95 with frequency_invoices.
2. Recency and tenure are standardised without a log transform; they are close to symmetric on the raw scale.
3. The plan's "fully cancelled giant lines net to zero" is implemented as a one-to-one match between a 5,000+ unit purchase and the cancellation that exactly reverses it.

**Before the test window was built (after selection on the dev and val windows):**

4. k is chosen strictly by stability. An earlier version allowed a 0.02 tie tolerance on ARI, under which silhouette would have picked k=6 (ARI 5th percentile 0.865 against 0.882 for k=3).
5. A sensitivity check without recency (set C) could not be run: at every k, high-return customers form a cluster below the 5% size rule. Test S4 answers the same question (do segments add to recency?).

**After the test window was scored:**

6. The plan allowed within-segment ranking only if both P1 and P2 passed. After P2 failed, ranking was made to depend on P1 alone (ranking ability), and P2 now governs probability statements, which were dropped.
7. Supplementary descriptive numbers computed from the frozen test predictions are allowed if labelled post hoc (section 5).
8. The illustrative scenario shows what listed customers did in the test window, with no predicted probabilities or expected buyer counts, consistent with ranking-only use.

## 8. Reproducibility
- `scripts/run_pipeline.py` rebuilds everything from the raw file in about two minutes; two runs give byte-identical `results.json`.
- Seeds come from `config.yaml`; DuckDB and gradient boosting run single-threaded.
- No model files are committed. The pipeline retrains from source, and the frozen segment model is saved as plain JSON (gitignored).
- Tests run on synthetic fixtures (`tests/fixtures`) and need no data. They cover cleaning rules, SQL-versus-pandas parity, leakage, segmentation, the gate and the insight helpers.
