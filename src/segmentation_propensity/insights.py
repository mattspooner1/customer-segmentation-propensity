"""Insight helpers: interval labels, ranking views and illustrative-scenario arithmetic.

Everything here reads frozen outputs (results.json, test predictions, snapshots). Nothing refits
a model or changes a choice made before the test window was scored. Model scores are used as
rankings only: calibration check P2 failed, so no function returns predicted counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from segmentation_propensity.evaluation import auc_with_ci
from segmentation_propensity.utils import resolve

log = logging.getLogger(__name__)

CONFIDENCE_LABELS = {
    "STRONG": "High",
    "MODEST": "Moderate",
    "WEAK": "Low",
    "NONE": "None",
    "INVALID": "Invalid",
}


def confidence_label(value: float, ci_low: float, baseline: float, strong_margin: float) -> str:
    """Confidence label for a discrimination metric, using the G2 interval rules.

    High: CI lower bound at least ``strong_margin`` above the baseline. Moderate: CI lower bound
    above the baseline. Low: point estimate above the baseline but the CI includes it. None:
    point estimate at or below the baseline. Invalid: below chance (0.5).
    """
    if value < 0.5:
        return CONFIDENCE_LABELS["INVALID"]
    if value <= baseline:
        return CONFIDENCE_LABELS["NONE"]
    if ci_low <= baseline:
        return CONFIDENCE_LABELS["WEAK"]
    if ci_low >= baseline + strong_margin:
        return CONFIDENCE_LABELS["STRONG"]
    return CONFIDENCE_LABELS["MODEST"]


def difference_label(ci_low: float, ci_high: float) -> str:
    """Describe a paired difference by whether its 95% CI excludes zero."""
    if ci_low > 0:
        return "positive (CI excludes 0)"
    if ci_high < 0:
        return "negative (CI excludes 0)"
    return "not distinguishable from 0"


def load_results(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load ``results.json`` and index its entries by id under ``by_id``."""
    with open(resolve(cfg, "results_json"), encoding="utf-8") as f:
        res = json.load(f)
    res["by_id"] = {r["id"]: r for r in res["results"]}
    return res


def load_assumptions(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the assumptions register, indexed by id (F1, A1, ...)."""
    with open(resolve(cfg, "assumptions_yaml"), encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    out = {}
    for section in ("context_facts", "illustrative_scenario"):
        for item in reg.get(section, []):
            out[item["id"]] = item
    return out


def within_group_auc(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
    rng: np.random.Generator,
    n_boot: int = 1000,
    y_col: str = "y",
) -> pd.DataFrame:
    """AUC of ``score_col`` within each group, with bootstrap 95% CIs."""
    rows = []
    for g, part in df.groupby(group_col, sort=True):
        res = auc_with_ci(part[y_col].to_numpy(), part[score_col].to_numpy(), rng, n_boot)
        rows.append({group_col: g, **res})
    return pd.DataFrame(rows)


def capture_curve(y: np.ndarray, score: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Share of all positives found in the top ``fractions`` of customers ranked by ``score``.

    Tied scores (for example coarse segment groups) are handled by taking the expected share
    under random ordering within the tie, so the curve is deterministic.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    n, total = len(y), y.sum()
    blocks = (
        pd.DataFrame({"s": score, "y": y})
        .groupby("s", sort=True)["y"]
        .agg(["size", "sum"])
        .iloc[::-1]
    )
    cum_n = np.concatenate([[0], blocks["size"].cumsum().to_numpy()])
    cum_y = np.concatenate([[0], blocks["sum"].cumsum().to_numpy()])
    found = np.interp(np.asarray(fractions, dtype=float) * n, cum_n, cum_y)
    return found / total


def break_even(
    n_contacts: int, cost_per_contact: float, margin_rate: float, order_value: float
) -> dict[str, float]:
    """Incremental merchandise value and orders needed for contact costs to break even."""
    cost = n_contacts * cost_per_contact
    value_needed = cost / margin_rate
    return {
        "contact_cost_gbp": cost,
        "incremental_value_needed_gbp": value_needed,
        "incremental_orders_needed": value_needed / order_value,
    }


def scenario_list(
    df: pd.DataFrame,
    n: int,
    score_col: str,
    group_col: str | None = None,
    group: Any = None,
) -> pd.DataFrame:
    """Top ``n`` customers by ``score_col``, optionally within one group."""
    pool = df if group_col is None else df[df[group_col] == group]
    if len(pool) < n:
        log.warning("Pool has %d customers, fewer than the list size %d", len(pool), n)
    return pool.sort_values([score_col, "customer_id"], ascending=[False, True]).head(n)


def summarise_list(
    lst: pd.DataFrame,
    name: str,
    cost_per_contact: float,
    margin_rate: float,
    y_col: str = "y",
    order_value_col: str = "avg_order_value_gbp",
) -> dict[str, Any]:
    """Historical outcome and break-even arithmetic for one contact list."""
    n = len(lst)
    ordered = int(lst[y_col].sum())
    aov = float(lst[order_value_col].median())
    be = break_even(n, cost_per_contact, margin_rate, aov)
    return {
        "list": name,
        "customers": n,
        "ordered_in_test_window": ordered,
        "did_not_order": n - ordered,
        "median_past_order_value_gbp": aov,
        **be,
    }


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a small result table as CSV and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote %s", path.name)
    return path
