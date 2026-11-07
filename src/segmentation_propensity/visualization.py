"""Publication figures for notebooks 03 and 04.

Colours follow one fixed categorical order (blue, orange, aqua) for segments and model series;
reference lines and no-skill baselines are neutral grey. Every function returns the figure so the
caller decides where it is saved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

log = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
REFERENCE = "#8a8984"


def apply_style() -> None:
    """Shared matplotlib defaults: recessive grid and axes, readable sizes."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": REFERENCE,
            "axes.labelcolor": INK_MUTED,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "font.size": 9,
        }
    )


def save_figure(fig: Figure, path: Path, dpi: int | None = None) -> Path:
    """Save ``fig`` to ``path`` (creating the folder) and return the path.

    ``dpi`` overrides the style default (150); the README hero figures use 300.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        fig.savefig(path)
    else:
        fig.savefig(path, dpi=dpi)
    log.info("Saved figure %s", path.name)
    return path


def plot_intervals(
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    references: dict[str, float] | None = None,
    colors: Sequence[str] | None = None,
) -> Figure:
    """Point estimates with 95% CIs, one row per entry (columns: label, value, ci_low, ci_high).

    Args:
        table: One row per estimate, drawn top to bottom in table order.
        title: Axes title.
        xlabel: Axis label for the metric.
        references: Named vertical reference lines, for example ``{"no skill": 0.5}``.
        colors: Optional colour per row; defaults to the first series colour.
    """
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(table) + 1.2))
    ypos = np.arange(len(table))[::-1]
    cols = list(colors) if colors is not None else [SERIES[0]] * len(table)
    for y, (_, row), c in zip(ypos, table.iterrows(), cols):
        ax.plot([row.ci_low, row.ci_high], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(row.value, y, "o", color=c, ms=8, mec="white", mew=2)
        ax.text(
            row.ci_high,
            y + 0.22,
            f"  {row.value:.3f} ({row.ci_low:.3f}-{row.ci_high:.3f})",
            fontsize=8,
            color=INK_MUTED,
            va="bottom",
            ha="right",
        )
    for i, (name, x) in enumerate((references or {}).items()):
        ax.axvline(x, color=REFERENCE, lw=1, ls="--" if i == 0 else ":")
        ax.text(x, ypos.max() + 0.6, f" {name}", fontsize=8, color=INK_MUTED, va="bottom")
    ax.set_yticks(ypos, table["label"])
    ax.set_ylim(-0.6, len(table) - 0.1 + 0.6)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_cv_grid(cv: pd.DataFrame) -> Figure:
    """Cross-validated AUC (mean +/- 1 SD) across the logistic regularisation grid."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.errorbar(
        cv["C"],
        cv["cv_auc_mean"],
        yerr=cv["cv_auc_sd"],
        color=SERIES[0],
        marker="o",
        ms=7,
        lw=2,
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set(
        title="Logistic regression: dev-window CV AUC by C (mean +/- 1 SD)",
        xlabel="C (inverse regularisation strength)",
        ylabel="CV AUC",
    )
    fig.tight_layout()
    return fig


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted score and observed rate in equal-count bins of the score."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    out = df.groupby("bin").agg(
        mean_score=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    )
    return out.reset_index()


def plot_reliability(table: pd.DataFrame, slope: float, bias: float) -> Figure:
    """Reliability diagram from :func:`reliability_table`, annotated with the P2 check values."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="perfect calibration")
    ax.plot(
        table.mean_score,
        table.observed_rate,
        "-o",
        color=SERIES[0],
        lw=2,
        ms=7,
        mec="white",
        mew=2,
        label="test window, score deciles",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean model score in decile",
        ylabel="Observed 90-day repeat rate",
        title="Calibration check (P2): fails, so scores are rankings only",
    )
    ax.text(
        0.03,
        0.93,
        f"slope {slope:.2f} (target 0.8-1.2)\nmean score minus rate {bias:+.3f} "
        "(target within 0.03)",
        fontsize=8,
        color=INK_MUTED,
        va="top",
        transform=ax.transAxes,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_capture(fractions: np.ndarray, curves: dict[str, np.ndarray]) -> Figure:
    """Share of repeat buyers found when contacting the top fraction of customers, per ranking."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="random order (no skill)")
    for (name, cap), c in zip(curves.items(), SERIES):
        ax.plot(fractions, cap, color=c, lw=2, label=name)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Share of customers contacted, best-ranked first",
        ylabel="Share of the window's repeat buyers in the list",
        title="Ranking customers: test window",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_group_rates(
    names: Sequence[str], shares: Sequence[float], rates: Sequence[float], title: str
) -> Figure:
    """Repeat-purchase rate per group, with each group's share of customers in the label."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(names))
    ax.bar(x, rates, width=0.6, color=SERIES[: len(names)], edgecolor="white", linewidth=2)
    for i, (r, s) in enumerate(zip(rates, shares)):
        ax.text(i, r + 0.02, f"{100 * r:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(
        x, [f"{n}\n({100 * s:.0f}% of customers)" for n, s in zip(names, shares)], fontsize=8
    )
    ax.set(ylim=(0, 1.05), ylabel="Ordered again within 90 days", title=title)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
