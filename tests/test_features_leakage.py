"""Leakage controls: features use only data before the cutoff; labels only the label window."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from segmentation_propensity.data import clean_lines
from segmentation_propensity.features import FEATURES, build_snapshot, window_overlap
from tests.fixtures import random_invoice_lines

CUTOFF = "2011-03-15"


def _snapshot(cfg, raw, cutoff=CUTOFF):
    con = duckdb.connect()
    clean_lines(con, raw, cfg)
    return build_snapshot(con, cfg, cutoff).set_index("customer_id")


@pytest.fixture()
def raw():
    return random_invoice_lines(seed=3)


def test_features_ignore_everything_after_cutoff(cfg, raw):
    base = _snapshot(cfg, raw)
    after = raw.InvoiceDate >= pd.Timestamp(CUTOFF)
    changed = raw.copy()
    changed.loc[after, "Quantity"] = changed.loc[after, "Quantity"] * 7
    changed.loc[after, "Price"] = changed.loc[after, "Price"] + 3.0
    pd.testing.assert_frame_equal(_snapshot(cfg, changed)[FEATURES], base[FEATURES])
    pd.testing.assert_frame_equal(_snapshot(cfg, raw[~after])[FEATURES], base[FEATURES])


def test_label_ignores_data_after_label_window(cfg, raw):
    end = pd.Timestamp(CUTOFF) + pd.Timedelta(days=cfg["windows"]["label_window_days"])
    base = _snapshot(cfg, raw)
    trimmed = _snapshot(cfg, raw[raw.InvoiceDate < end])
    pd.testing.assert_series_equal(trimmed["y"], base["y"])


def test_label_uses_label_window(cfg, raw):
    c = pd.Timestamp(CUTOFF)
    end = c + pd.Timedelta(days=cfg["windows"]["label_window_days"])
    in_window = (raw.InvoiceDate >= c) & (raw.InvoiceDate < end)
    without = _snapshot(cfg, raw[~in_window])
    assert without["y"].sum() == 0
    assert _snapshot(cfg, raw)["y"].sum() > 0


def test_population_is_lookback_buyers_only(cfg, raw):
    snap = _snapshot(cfg, raw)
    assert (snap.recency_days >= 0).all()
    assert (snap.recency_days <= cfg["windows"]["lookback_days"]).all()
    assert (snap.tenure_days >= snap.recency_days).all()
    assert (snap.frequency_invoices >= 1).all()


def test_window_overlap_counts_test_customers_seen_at_fit(cfg):
    snaps = {n: pd.DataFrame({"customer_id": [1, 2, 3]}) for n in cfg["windows"]["snapshots"]}
    snaps["dev"] = pd.DataFrame({"customer_id": [2, 3, 9]})
    snaps["test"] = pd.DataFrame({"customer_id": [1, 2, 3, 4]})
    out = window_overlap(snaps, cfg)
    assert (out["segment_fit_window"], out["test_window"]) == ("dev", "test")
    assert out["test_customers"] == 4 and out["test_customers_in_segment_fit_window"] == 2
    assert out["share_of_test_customers"] == 0.5
