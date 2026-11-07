"""Insight helpers and figures on synthetic inputs (no raw data, no results files)."""

from __future__ import annotations

import copy
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from segmentation_propensity import insights as ins  # noqa: E402
from segmentation_propensity import visualization as viz  # noqa: E402


@pytest.mark.parametrize(
    "value,ci_low,expected",
    [
        (0.45, 0.40, "Invalid"),
        (0.50, 0.45, "None"),
        (0.55, 0.49, "Low"),
        (0.58, 0.55, "Moderate"),
        (0.72, 0.69, "High"),
    ],
)
def test_confidence_label_follows_interval_rules(value, ci_low, expected):
    assert ins.confidence_label(value, ci_low, 0.5, 0.10) == expected


@pytest.mark.parametrize(
    "lo,hi,expected",
    [(0.01, 0.05, "positive"), (-0.05, -0.01, "negative"), (-0.01, 0.01, "not distinguishable")],
)
def test_difference_label(lo, hi, expected):
    assert ins.difference_label(lo, hi).startswith(expected)


def test_capture_curve_endpoints_and_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    f = np.array([0.0, 0.5, 1.0])
    assert np.allclose(ins.capture_curve(y, np.array([4, 3, 2, 1]), f), [0, 1, 1])
    # all scores tied: expected capture equals the share contacted
    assert np.allclose(ins.capture_curve(y, np.ones(4), f), [0, 0.5, 1])


def test_capture_curve_handles_ties_inside_blocks():
    y = np.array([1, 0, 1, 1])
    s = np.array([2, 2, 1, 1])  # top block: 1 buyer in 2 customers
    assert ins.capture_curve(y, s, np.array([0.25]))[0] == pytest.approx(1 / 6)


def test_break_even_arithmetic():
    be = ins.break_even(500, 2.0, 0.35, 250.0)
    assert be["contact_cost_gbp"] == 1000.0
    assert be["incremental_value_needed_gbp"] == pytest.approx(1000 / 0.35)
    assert be["incremental_orders_needed"] == pytest.approx(1000 / 0.35 / 250)


@pytest.fixture()
def preds():
    rng = np.random.default_rng(0)
    n = 400
    seg = rng.integers(0, 3, n)
    p = np.clip(0.2 + 0.25 * seg + rng.normal(0, 0.15, n), 0.01, 0.99)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "y": (rng.random(n) < p).astype(int),
            "p": p,
            "segment": seg,
            "avg_order_value_gbp": rng.lognormal(5, 0.5, n),
        }
    )


def test_within_group_auc_one_row_per_group(preds):
    out = ins.within_group_auc(preds, "segment", "p", np.random.default_rng(1), n_boot=30)
    assert list(out.segment) == [0, 1, 2]
    assert (out.ci_low <= out.value).all() and (out.value <= out.ci_high).all()
    assert out.n.sum() == len(preds)


def test_scenario_list_and_summary(preds):
    lst = ins.scenario_list(preds, 20, "p", group_col="segment", group=1)
    assert len(lst) == 20 and (lst.segment == 1).all()
    assert lst.p.min() >= preds.loc[preds.segment == 1].p.nlargest(20).min()
    s = ins.summarise_list(lst, "test", 2.0, 0.35)
    assert s["ordered_in_test_window"] + s["did_not_order"] == 20
    assert s["contact_cost_gbp"] == 40.0


def test_scenario_list_warns_on_small_pool(preds, caplog):
    ins.scenario_list(preds, 10_000, "p")
    assert "fewer than the list size" in caplog.text


def test_loaders(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["paths"]["results_json"] = str(tmp_path / "r.json")
    (tmp_path / "r.json").write_text(json.dumps({"results": [{"id": "S1", "value": 0.7}]}))
    assert ins.load_results(c)["by_id"]["S1"]["value"] == 0.7
    reg = ins.load_assumptions(cfg)
    assert reg["A4"]["value"] is None and {"A1", "A2", "A3", "F1"} <= set(reg)


def test_save_table(tmp_path):
    p = ins.save_table(pd.DataFrame({"a": [1]}), tmp_path / "sub" / "t.csv")
    assert p.exists()


def test_figures_render_and_save(preds, tmp_path):
    viz.apply_style()
    table = pd.DataFrame(
        {"label": ["a", "b"], "value": [0.7, 0.6], "ci_low": [0.65, 0.55], "ci_high": [0.75, 0.65]}
    )
    rel = viz.reliability_table(preds.y, preds.p, n_bins=5)
    assert len(rel) == 5 and rel.n.sum() == len(preds)
    f = np.linspace(0, 1, 11)
    figs = [
        viz.plot_intervals(table, "t", "AUC", references={"no skill": 0.5, "min": 0.6}),
        viz.plot_cv_grid(
            pd.DataFrame({"C": [0.1, 1.0], "cv_auc_mean": [0.7, 0.71], "cv_auc_sd": [0.01, 0.01]})
        ),
        viz.plot_reliability(rel, 1.1, 0.02),
        viz.plot_capture(f, {"model": ins.capture_curve(preds.y, preds.p, f)}),
        viz.plot_group_rates(["x", "y", "z"], [0.3, 0.4, 0.3], [0.2, 0.5, 0.8], "t"),
    ]
    for i, fig in enumerate(figs):
        assert viz.save_figure(fig, tmp_path / f"f{i}.png").stat().st_size > 0
