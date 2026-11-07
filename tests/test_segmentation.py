"""Segmentation, RFM rule and evaluation helpers on synthetic snapshots (no raw data)."""

from __future__ import annotations

import copy

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from segmentation_propensity.evaluation import (
    auc_with_ci,
    cramers_v,
    paired_auc_difference,
    permuted_cramers_v,
)
from segmentation_propensity.features import FEATURES
from segmentation_propensity.segmentation import (
    Preprocessor,
    SegmentModel,
    choose_k,
    fit_segment_model,
    name_segments,
    rfm_rule_groups,
    rfm_rule_pandas,
    rule_score,
    select_segmentation,
)


def synthetic_snapshot(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Three well-separated customer groups with different repeat-purchase rates."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    df = pd.DataFrame({f: rng.random(n) for f in FEATURES})
    df["recency_days"] = np.choose(g, [20, 120, 240]) + rng.integers(0, 15, n)
    df["frequency_invoices"] = np.choose(g, [12, 4, 1]) + rng.integers(0, 2, n)
    df["monetary_net_gbp"] = np.choose(g, [5000.0, 800.0, 90.0]) * rng.uniform(0.8, 1.2, n)
    df["median_units_per_line"] = rng.integers(1, 30, n).astype(float)
    df["return_rate_value"] = rng.uniform(0, 0.1, n)
    df["y"] = (rng.random(n) < np.choose(g, [0.85, 0.45, 0.15])).astype(int)
    df.insert(0, "customer_id", np.arange(10000, 10000 + n))
    df["true_group"] = g
    return df


@pytest.fixture()
def fast_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["segmentation"]["bootstrap_refits"] = 5
    c["segmentation"]["k_range"] = [2, 4]
    c["segmentation"]["n_init"] = 3
    c["evaluation"]["bootstrap_resamples"] = 50
    return c


def test_preprocessor_caps_logs_and_standardises(cfg):
    dev = synthetic_snapshot()
    feats = ["recency_days", "monetary_net_gbp"]
    pre = Preprocessor.fit(dev, feats, cfg)
    assert pre.log1p == ["monetary_net_gbp"]  # recency is not log-transformed
    x = pre.transform(dev)
    np.testing.assert_allclose(x.mean(0), 0, atol=1e-9)
    later = dev.copy()
    later["monetary_net_gbp"] = 1e9  # capped at the dev quantile, not refitted
    capped = (np.log1p(pre.caps["monetary_net_gbp"]) - pre.mean["monetary_net_gbp"]) / pre.std[
        "monetary_net_gbp"
    ]
    np.testing.assert_allclose(pre.transform(later)[:, 1], capped)


def test_choose_k_rules(cfg):
    grid = pd.DataFrame(
        {
            "k": [3, 4, 5, 6],
            "ari_p05": [0.90, 0.40, 0.895, 0.95],
            "silhouette": [0.30, 0.50, 0.35, 0.20],
            "size_ok": [True, True, True, False],
        }
    )
    tol = copy.deepcopy(cfg)
    tol["segmentation"]["ari_tie_tolerance"] = 0.02
    # k=6 fails the size rule; 3 and 5 tie on ARI (within 0.02); 5 has the better silhouette
    assert choose_k(grid, tol) == 5
    strict = copy.deepcopy(cfg)
    strict["segmentation"]["ari_tie_tolerance"] = 0.0
    assert choose_k(grid, strict) == 3  # strict stability-first: best eligible ARI wins
    grid.loc[grid.k == 5, "silhouette"] = 0.30
    assert choose_k(grid, tol) == 3  # remaining tie goes to the smaller k
    grid["size_ok"] = False
    assert choose_k(grid, tol) is None


def test_model_recovers_groups_orders_by_rate_and_round_trips(cfg):
    dev = synthetic_snapshot()
    model = fit_segment_model(dev, "A_rfm_core", 3, cfg)
    labels = model.predict(dev)
    assert adjusted_rand_score(dev.true_group, labels) > 0.95
    rates = [model.segment_rates[j] for j in range(3)]
    assert rates == sorted(rates)
    again = SegmentModel.from_json(model.to_json())
    np.testing.assert_array_equal(again.predict(dev), labels)
    np.testing.assert_allclose(again.score(dev), model.score(dev))


def test_segment_score_uses_dev_rates_only(cfg):
    dev, later = synthetic_snapshot(seed=0), synthetic_snapshot(seed=1)
    model = fit_segment_model(dev, "A_rfm_core", 3, cfg)
    flipped = later.assign(y=1 - later.y)
    np.testing.assert_allclose(model.score(flipped), model.score(later))


def test_names_are_unique(cfg):
    dev = synthetic_snapshot()
    model = fit_segment_model(dev, "A_rfm_core", 3, cfg)
    assert len(set(model.names.values())) == 3
    assert len(set(name_segments(dev, np.zeros(len(dev), int), cfg).values())) == 1


@pytest.mark.parametrize("k", [3, 4, 6])
def test_rfm_rule_sql_matches_pandas(cfg, k):
    dev, target = synthetic_snapshot(seed=2), synthetic_snapshot(seed=3)
    dev["frequency_invoices"] = dev.frequency_invoices.clip(upper=3)  # force tied cut points
    sql = rfm_rule_groups(duckdb.connect(), dev, target, k, cfg)
    ref = rfm_rule_pandas(dev, target, k, cfg)
    pd.testing.assert_frame_equal(sql.astype(float), ref.astype(float))
    assert sql.rfm_group.between(0, k - 1).all()
    means = sql.groupby("rfm_group").rfm_score.mean()
    assert means.is_monotonic_increasing


def test_rule_score_maps_dev_group_rates(cfg):
    dev = synthetic_snapshot()
    con = duckdb.connect()
    g = rfm_rule_groups(con, dev, dev, 3, cfg)
    score = rule_score(dev, g, g)
    expected = dev.merge(g, on="customer_id").groupby("rfm_group").y.transform("mean")
    np.testing.assert_allclose(score, expected.to_numpy())


def test_selection_never_needs_the_test_window(fast_cfg):
    snaps = {"dev": synthetic_snapshot(seed=4), "val": synthetic_snapshot(seed=5)}
    result = select_segmentation(snaps, fast_cfg)  # no "test" key: any access would raise
    assert result["chosen_set"] in fast_cfg["features"]["segmentation_sets"]
    assert 0 <= result["stability"]["S3_ari_p05"] <= 1
    assert result["val_segment_score_auc"][result["chosen_set"]]["value"] > 0.6


def test_evaluation_helpers():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    perfect = auc_with_ci(y, y.astype(float), rng, 50)
    assert perfect["value"] == 1.0 and perfect["ci_low"] == 1.0
    same = paired_auc_difference(y, y + 0.0, y + 0.0, rng, 50)
    assert same["value"] == 0 and same["ci_low"] == 0 == same["ci_high"]
    assert cramers_v(np.zeros(100), y) == 0.0
    strong = permuted_cramers_v(y, y, rng, 50)
    assert strong["value"] == pytest.approx(1.0) and strong["perm_p"] < 0.05
