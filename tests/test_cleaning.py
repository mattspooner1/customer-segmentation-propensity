"""Cleaning rules (sql/01_clean_lines.sql) on the hand-built fixture."""

from __future__ import annotations

import duckdb
import pytest

from segmentation_propensity.data import clean_lines, flag_counts

EXPECTED_STATUS = {
    "missing_customer_id": "missing_customer_id",
    "non_product_code": "non_product_code",
    "non_product_prefix": "non_product_code",
    "price_zero": "price_le_zero",
    "neg_qty_non_cancellation": "neg_qty_non_cancellation",
    "giant_line": "reversed_giant_line",
    "purchase_label_window": "purchase",
    "wholesale": "purchase",
}


@pytest.fixture()
def cleaned(cfg, lines_with_case):
    con = duckdb.connect()
    log = clean_lines(con, lines_with_case.drop(columns="case"), cfg)
    status = con.execute("SELECT line_no, status FROM line_status ORDER BY line_no").df()
    return con, log, lines_with_case.assign(status=status["status"].to_numpy())


def test_each_case_gets_expected_status(cleaned):
    _, _, df = cleaned
    for case, expected in EXPECTED_STATUS.items():
        assert set(df.loc[df.case == case, "status"]) == {expected}, case


def test_duplicate_keeps_first_copy(cleaned):
    _, _, df = cleaned
    dup = df[df.Invoice == "500030"]
    assert dup.status.tolist() == ["purchase", "exact_duplicate"]


def test_giant_line_and_its_cancellation_both_reversed(cleaned):
    _, _, df = cleaned
    assert set(df.loc[df.Invoice.isin(["500010", "C500011"]), "status"]) == {"reversed_giant_line"}
    # the partial return of an ordinary line stays a cancellation
    assert df.loc[df.Invoice == "C500021", "status"].item() == "cancellation"


def test_every_line_has_one_status(cleaned):
    _, log, df = cleaned
    assert log["lines"].sum() == len(df)
    assert df.status.notna().all()


def test_duplicates_kept_when_disabled(cfg, lines):
    con = duckdb.connect()
    log = clean_lines(con, lines, cfg, drop_duplicates=False)
    assert "exact_duplicate" not in set(log.status.astype(str))


def test_purchases_and_cancellations_tables(cleaned):
    con, _, _ = cleaned
    p = con.execute("SELECT * FROM purchases").df()
    c = con.execute("SELECT * FROM cancellations").df()
    assert (p.quantity > 0).all() and (p.price > 0).all() and p.customer_id.notna().all()
    assert (c.cancel_value > 0).all()
    assert not set(p.stock_code) & {"POST", "ADJUST2", "gift_0001_20"}


def test_flag_counts_are_independent(cleaned):
    con, _, _ = cleaned
    flags = flag_counts(con)
    assert flags["cancellation"] == 2
    assert flags["exact_duplicate"] == 1
