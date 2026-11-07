"""SQL pipeline equals an independent pandas implementation of the same rules."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from segmentation_propensity.data import clean_lines, clean_lines_pandas
from segmentation_propensity.features import FEATURES, build_snapshot, build_snapshot_pandas
from tests.fixtures import invoice_lines, random_invoice_lines

CUTOFFS = ["2010-09-11", "2011-03-15", "2011-06-13", "2011-09-11"]


def _both(cfg, raw, cutoff, drop_duplicates=None):
    con = duckdb.connect()
    clean_lines(con, raw, cfg, drop_duplicates)
    sql = build_snapshot(con, cfg, cutoff).reset_index(drop=True)
    p, c, _ = clean_lines_pandas(raw, cfg, drop_duplicates)
    ref = build_snapshot_pandas(p, c, cfg, cutoff).reset_index(drop=True)
    return con, sql, ref


def _assert_frames_equal(sql, ref):
    assert sql.customer_id.tolist() == ref.customer_id.tolist()
    for col in FEATURES + ["y"]:
        np.testing.assert_allclose(
            sql[col].astype(float), ref[col].astype(float), rtol=1e-9, atol=1e-9, err_msg=col
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("drop_duplicates", [True, False])
def test_cleaning_status_parity(cfg, seed, drop_duplicates):
    raw = random_invoice_lines(seed=seed)
    con = duckdb.connect()
    clean_lines(con, raw, cfg, drop_duplicates)
    sql_status = con.execute("SELECT status FROM line_status ORDER BY line_no").df()["status"]
    _, _, ref_status = clean_lines_pandas(raw, cfg, drop_duplicates)
    assert sql_status.tolist() == ref_status.tolist()


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feature_parity_random(cfg, cutoff, seed):
    _, sql, ref = _both(cfg, random_invoice_lines(seed=seed), cutoff)
    assert len(sql) > 10
    _assert_frames_equal(sql, ref)


def test_feature_parity_fixture(cfg):
    _, sql, ref = _both(cfg, invoice_lines(), "2011-09-11")
    _assert_frames_equal(sql, ref)


def test_fixture_hand_checked_values(cfg):
    _, sql, _ = _both(cfg, invoice_lines(), "2011-09-11")
    row = sql.set_index("customer_id").loc[10002]
    # the 80,995-unit line and its cancellation net to zero; only the 288-unit order remains
    assert row.frequency_invoices == 1
    assert row.monetary_net_gbp == pytest.approx(288 * 1.06)
    assert row.return_rate_value == 0
    row = sql.set_index("customer_id").loc[10003]
    assert row.return_rate_value == pytest.approx(6 * 1.65 / (24 * 1.65 + 6 * 1.65))
    assert row.is_uk == 0
    assert sql.set_index("customer_id").loc[10001, "y"] == 1
    assert sql.label_window_q4.eq(1).all()


def test_snapshot_columns(cfg):
    _, sql, _ = _both(cfg, invoice_lines(), "2011-06-13")
    assert list(sql.columns) == ["customer_id"] + FEATURES + ["y"]
    assert sql.label_window_q4.eq(0).all()
    assert isinstance(sql, pd.DataFrame)


def test_all_snapshots_and_summary(cfg):
    from segmentation_propensity.features import build_all_snapshots, snapshot_summary

    con = duckdb.connect()
    clean_lines(con, random_invoice_lines(seed=4), cfg)
    snaps = build_all_snapshots(con, cfg)
    assert set(snaps) == set(cfg["windows"]["snapshots"])
    s = snapshot_summary(snaps, cfg).set_index("snapshot")
    for name, df in snaps.items():
        assert s.loc[name, "customers"] == len(df)
        assert s.loc[name, "events"] + s.loc[name, "non_events"] == len(df)
        assert s.loc[name, "events_per_fold"] == df.y.sum() // cfg["propensity"]["cv_folds"]


def test_snapshot_is_bit_reproducible(cfg):
    from segmentation_propensity.data import connect

    raw = random_invoice_lines(seed=5)
    builds = []
    for _ in range(2):
        con = connect(cfg)
        clean_lines(con, raw, cfg)
        builds.append(build_snapshot(con, cfg, "2011-03-15"))
    pd.testing.assert_frame_equal(builds[0], builds[1], check_exact=True)
