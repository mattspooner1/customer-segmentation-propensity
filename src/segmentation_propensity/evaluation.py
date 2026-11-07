"""Evaluation helpers: bootstrap confidence intervals, paired differences and association tests."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    n: int,
    rng: np.random.Generator,
    n_boot: int = 1000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat(idx)`` over ``n`` units resampled with replacement.

    Resamples where the statistic is undefined (``nan``) are skipped.
    """
    vals = []
    for _ in range(n_boot):
        v = stat(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC, or ``nan`` when only one class is present."""
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_with_ci(
    y: np.ndarray, score: np.ndarray, rng: np.random.Generator, n_boot: int = 1000
) -> dict[str, float]:
    """AUC with a customer-level bootstrap 95% CI."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    lo, hi = bootstrap_ci(lambda i: safe_auc(y[i], score[i]), len(y), rng, n_boot)
    return {
        "value": safe_auc(y, score),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(y)),
        "n_events": int(y.sum()),
    }


def paired_auc_difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 1000
) -> dict[str, float]:
    """AUC(a) - AUC(b) on the same customers, with a paired bootstrap 95% CI."""
    y = np.asarray(y)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(
        lambda i: safe_auc(y[i], a[i]) - safe_auc(y[i], b[i]), len(y), rng, n_boot
    )
    return {"value": safe_auc(y, a) - safe_auc(y, b), "ci_low": lo, "ci_high": hi, "n": int(len(y))}


def cramers_v(groups: np.ndarray, y: np.ndarray) -> float:
    """Cramer's V for a group-by-binary-outcome table."""
    _, gi = np.unique(groups, return_inverse=True)
    table = np.zeros((gi.max() + 1, 2))
    np.add.at(table, (gi, np.asarray(y, dtype=int)), 1)
    table = table[table.sum(1) > 0]
    if table.shape[0] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def permuted_cramers_v(
    groups: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int = 200
) -> dict[str, float]:
    """Observed Cramer's V with its label-permutation null (95th percentile and p-value)."""
    obs = cramers_v(groups, y)
    null = np.array([cramers_v(groups, rng.permutation(y)) for _ in range(n_perm)])
    return {
        "value": obs,
        "null_p95": float(np.percentile(null, 95)),
        "null_mean": float(null.mean()),
        "perm_p": float((1 + (null >= obs).sum()) / (1 + n_perm)),
    }
