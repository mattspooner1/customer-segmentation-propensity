"""Synthetic invoice-line fixtures matching the raw Online Retail II schema.

Tests use these tables only, so they pass on a fresh clone with no downloaded data.
Every cleaning case in the cleaning rules appears at least once.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

# (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country, case)
_ROWS = [
    # customer 10001: regular UK buyer, before and after the test cutoff
    (
        "500001",
        "85048",
        "GLASS BALL LIGHTS",
        12,
        "2011-01-10 10:00",
        6.95,
        10001.0,
        "United Kingdom",
        "purchase",
    ),
    (
        "500001",
        "22041",
        "RECORD FRAME",
        6,
        "2011-01-10 10:00",
        2.10,
        10001.0,
        "United Kingdom",
        "purchase",
    ),
    (
        "500050",
        "85048",
        "GLASS BALL LIGHTS",
        24,
        "2011-06-01 09:30",
        6.95,
        10001.0,
        "United Kingdom",
        "purchase",
    ),
    (
        "500090",
        "22041",
        "RECORD FRAME",
        6,
        "2011-10-05 14:00",
        2.10,
        10001.0,
        "United Kingdom",
        "purchase_label_window",
    ),
    # customer 10002: wholesale profile with a giant line that is fully cancelled
    (
        "500010",
        "23843",
        "PAPER CRAFT",
        80995,
        "2011-05-01 09:15",
        2.08,
        10002.0,
        "United Kingdom",
        "giant_line",
    ),
    (
        "C500011",
        "23843",
        "PAPER CRAFT",
        -80995,
        "2011-05-01 09:27",
        2.08,
        10002.0,
        "United Kingdom",
        "cancellation",
    ),
    (
        "500012",
        "21232",
        "TRINKET BOX",
        288,
        "2011-05-20 11:00",
        1.06,
        10002.0,
        "United Kingdom",
        "wholesale",
    ),
    # customer 10003: non-UK buyer with a partial return and postage
    (
        "500020",
        "22064",
        "DOUGHNUT POT",
        24,
        "2011-04-02 12:00",
        1.65,
        10003.0,
        "France",
        "purchase",
    ),
    (
        "500020",
        "POST",
        "POSTAGE",
        1,
        "2011-04-02 12:00",
        18.00,
        10003.0,
        "France",
        "non_product_code",
    ),
    (
        "C500021",
        "22064",
        "DOUGHNUT POT",
        -6,
        "2011-04-10 12:00",
        1.65,
        10003.0,
        "France",
        "cancellation",
    ),
    # customer 10004: single invoice, duplicated line
    (
        "500030",
        "84879",
        "BIRD ORNAMENT",
        8,
        "2011-02-14 16:00",
        1.69,
        10004.0,
        "United Kingdom",
        "purchase",
    ),
    (
        "500030",
        "84879",
        "BIRD ORNAMENT",
        8,
        "2011-02-14 16:00",
        1.69,
        10004.0,
        "United Kingdom",
        "exact_duplicate",
    ),
    # lines without a customer, free items, stock adjustments and other fee codes
    (
        "500040",
        "85123A",
        "HEART HOLDER",
        6,
        "2011-03-01 08:00",
        2.55,
        None,
        "United Kingdom",
        "missing_customer_id",
    ),
    (
        "500041",
        "22139",
        "RETRO SPOT TEA SET",
        1,
        "2011-03-02 08:00",
        0.00,
        10001.0,
        "United Kingdom",
        "price_zero",
    ),
    (
        "500042",
        "22139",
        "damaged",
        -3,
        "2011-03-03 08:00",
        1.25,
        10001.0,
        "United Kingdom",
        "neg_qty_non_cancellation",
    ),
    (
        "500043",
        "ADJUST2",
        "Manual adjustment",
        1,
        "2011-03-04 08:00",
        50.00,
        10003.0,
        "France",
        "non_product_prefix",
    ),
    (
        "500044",
        "gift_0001_20",
        "Dotcomgiftshop Gift Voucher",
        1,
        "2011-03-05 08:00",
        16.67,
        10004.0,
        "United Kingdom",
        "non_product_prefix",
    ),
]


def invoice_lines(with_case: bool = False) -> pd.DataFrame:
    """Return the synthetic invoice table, optionally with a ``case`` column per row."""
    df = pd.DataFrame(_ROWS, columns=COLUMNS + ["case"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df if with_case else df.drop(columns="case")


def random_invoice_lines(n_customers: int = 80, seed: int = 0) -> pd.DataFrame:
    """Larger random table for parity tests: returns, duplicates, ties and giant lines."""
    import numpy as np

    rng = np.random.default_rng(seed)
    codes = ["85048", "22041", "21232", "22064", "84879", "85123A", "POST", "ADJUST", "M"]
    countries = ["United Kingdom", "France", "Germany"]
    start = pd.Timestamp("2010-01-01")
    rows = []
    inv = 600000
    for cid in range(20000, 20000 + n_customers):
        cust = float(cid) if rng.random() > 0.05 else None
        for _ in range(rng.integers(1, 12)):
            inv += 1
            ts = start + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 700)))
            country = countries[int(rng.integers(0, 3))]
            for _ in range(rng.integers(1, 5)):
                qty = int(rng.choice([1, 2, 6, 12, 24, 48, 6000]))
                price = float(rng.choice([0.0, 0.85, 1.25, 2.10, 6.95]))
                row = (str(inv), str(rng.choice(codes)), "ITEM", qty, ts, price, cust, country)
                rows.append(row)
                if rng.random() < 0.05:
                    rows.append(row)  # exact duplicate
                if rng.random() < 0.15:  # cancellation of this line, sometimes partial
                    back = qty if rng.random() < 0.6 else max(1, qty // 2)
                    later = ts + pd.Timedelta(days=int(rng.integers(0, 30)))
                    rows.append(
                        ("C" + str(inv), row[1], "ITEM", -back, later, price, cust, country)
                    )
            if rng.random() < 0.05:
                rows.append((str(inv), "22139", "damaged", -2, ts, 1.0, cust, country))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
