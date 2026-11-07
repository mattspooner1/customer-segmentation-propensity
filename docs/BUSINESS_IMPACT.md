# Business impact

> **Illustrative scenario.** The stakeholder and decision are invented: a CRM and trading team at a UK online gift supplier that plans account management each quarter and today groups customers by gut feel and a top-spender list. The data is real (a UK online gift retailer, 2009-2011, with many wholesale buyers) and contains **no marketing contact data**. Nothing here measures the effect of contacting anyone.
>
> **Registered customers only.** 243,007 of 1,067,371 raw invoice lines (22.8%) have no customer ID and are excluded during cleaning. Every segment and rate below describes registered customers only.

## The decision
Which customers should the team review and contact each quarter, and how should it describe its customer base? The analysis answers two narrower questions with evidence from a held-out quarter:
1. Is there a segmentation that reliably separates customers who will order again within 90 days from those who will not?
2. Can individual customers be ranked usefully within those segments?

## What the evidence supports
All results come from a test quarter (cutoff 2011-09-11; 3,238 customers, 1,868 of whom ordered again). The segments were built on an earlier quarter (cutoff 2011-03-15), so none of these claims is judged on the data the segments were built from. Most test customers (2,443 of 3,238, or 75.4%) were also in that earlier quarter: this is a check on later behaviour, not on new customers. How the test quarter was used is described under [How far to trust these numbers](#how-far-to-trust-these-numbers).

| Group (RFM rule) | Share of customers | Ordered again within 90 days |
|---|---|---|
| RFM group 1 (low) | 34% | 35% |
| RFM group 2 (middle) | 33% | 55% |
| RFM group 3 (high) | 33% | 83% |

- **The groups separate repeat buyers on a later quarter.** The k-means segments rank repeat buyers with AUC 0.712 (95% CI 0.694-0.727; computed verdict STRONG), and the RFM rule with 0.717 (0.700-0.733; no verdict computed, interval clear of chance).
- **The simple rule is enough.** The rule is not distinguishable from the k-means segments (difference -0.005, CI -0.014 to +0.004, which spans 0), and it is a few lines of SQL anyone on the team can audit, so it is the one to operate. The data does not show that the rule is better.
- **The k-means segments give the team shared language.** *Lapsed infrequent low-value* (32% of customers, 35% ordered again), *cooling regular mid-value* (46%, 59%) and *recent frequent high-value* (22%, 89%). These are descriptions of who ordered again, not explanations of why.
- **The model can order customers within a group.** It ranks repeat buyers with AUC 0.755 (0.738-0.771; computed verdict STRONG). Inside each k-means segment it still ranks above chance (within-segment AUC 0.625, 0.624 and 0.750, every lower bound above 0.5), and the ordering is weaker in the low and middle segments. No paired test compares the model with the segments or the rule, so this does not show that it ranks better than either.
- **The scores are not probabilities.** They failed a calibration check (slope 1.220, mean score 0.043 above the observed rate). Use their order, never their values.

## Recommendations
1. **Segment with the RFM quintile rule**, recomputed each quarter with fixed cut points, and report the k-means profile names alongside it. Reason: it is not distinguishable from k-means and is simpler to run.
2. **Within each group, order customers by model rank.** Do not convert scores into expected buyer counts or revenue forecasts.
3. **Re-check the segments each quarter.** A refit on a later window agrees with the original at ARI 0.788: reasonably, not fully, stable.
4. **Test before spending.** Contact a random half of each candidate list and leave the other half alone, then compare 90-day order rates. This is the direct way to learn whether contact changes behaviour, and it is the number this data cannot provide.

## How far to trust these numbers
- **Six scorings.** The test quarter was scored six times: the first run, then five re-runs to check a record-merging fix and to remove run-to-run non-determinism. No choice changed after the first score, and the final numbers equal it. Release checks also re-run the full pipeline from a fresh clone as a reproducibility test; such runs inform no choice, and they reproduce `results.json` byte-for-byte.
- **An earlier look at the same quarter.** Before the analysis plan was fixed, an initial feasibility check ran a prototype of this approach on the same test quarter. Its results set the minimum-useful targets (0.685 for the segments, 0.725 for the model), led to the rule-versus-k-means comparison being made a deciding test, and shaped how the model's training data was chosen. The segment and model intervals clear chance by more than 0.10 on their own (lower bounds 0.694 and 0.738 against 0.5), so the main conclusions do not rest on those targets. Only the "minimum useful met" part of each verdict does. Full details are in [METHODOLOGY.md, section 6](METHODOLOGY.md#6-disclosure-use-of-the-test-window).

## Illustrative scenario: a 500-customer contact list

> **Illustrative scenario: unsourced assumptions.** List size, contact cost and margin rate are assumptions, not facts about any business (see `assumptions.yaml`). The incremental effect of contact is **not estimable** from this data. The table shows only (a) what the listed customers did in the test quarter *with no contact programme*, and (b) how many *extra* orders the list would need to cover its cost. It does **not** estimate revenue from contact, and no model score is used as a probability.

| Assumption | Value | Source |
|---|---|---|
| A1: customers the team can contact per quarter | 500 | unsourced assumption |
| A2: cost per contact (staff time and materials) | £2.00 | unsourced assumption |
| A3: gross margin rate on merchandise value | 0.35 | unsourced assumption |
| A4: incremental effect of contact on ordering | unknown | **not estimable** from this data (no contact or treatment records) |

Contact cost per list: 500 × £2.00 = £1,000. Break-even incremental merchandise value: £1,000 / 0.35 = £2,857. Break-even orders use each list's median past order value. Source: `outputs/metrics/illustrative_scenario.csv`.

| List (top 500 by model rank) | Ordered anyway in the test quarter | Did not order | Median past order value | Extra orders needed to break even |
|---|---|---|---|---|
| All customers | 470 | 30 | £348 | 8.2 |
| RFM group 2 (middle) | 316 | 184 | £299 | 9.5 |
| RFM group 1 (low) | 217 | 283 | £252 | 11.3 |

**What this shows.**
- A list built from model rank alone is entirely top-group customers, and 470 of the 500 ordered anyway. At most 30 orders from that list could be incremental, so ranking by likelihood to order is not the same as choosing whom to contact.
- Lists drawn from the middle or low groups hold more customers who did not order on their own (184 and 283), so any effect of contact would have more room to show. Whether any list clears roughly 8-11 extra orders per 500 contacts depends entirely on A4, which is unknown.
- The holdout test in recommendation 4 measures A4 directly. Until it runs, any return-on-investment figure would be invented.

## What this analysis cannot tell you
- The effect of contacting a customer, or which customers respond best to contact.
- Probabilities or expected buyer counts for a quarter (calibration failed).
- Whether the model ranks better than the segments or the rule (no paired test).
- Anything about unregistered buyers (22.8% of raw lines have no customer ID) or businesses unlike a 2009-2011 UK gift wholesaler.
