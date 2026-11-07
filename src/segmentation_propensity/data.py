"""Load raw invoice lines and clean them with the SQL rules in ``sql/01_clean_lines.sql``.

A pandas reference implementation of the same rules (``clean_lines_pandas``) exists only so tests
can check that the SQL applies the documented cleaning rules.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from segmentation_propensity.utils import resolve

log = logging.getLogger(__name__)

RAW_DTYPES = {"Invoice": str, "StockCode": str, "Description": str, "Country": str}
STATUS_ORDER = [
    "missing_customer_id",
    "non_product_code",
    "price_le_zero",
    "neg_qty_non_cancellation",
    "exact_duplicate",
    "reversed_giant_line",
    "cancellation",
    "purchase",
]


def connect(cfg: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with the configured thread count.

    One thread keeps floating-point aggregation order fixed, so features are bit-reproducible.
    """
    con = duckdb.connect()
    con.execute(f"SET threads TO {int(cfg['engine']['duckdb_threads'])}")
    return con


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the raw CSV written by ``scripts/download_data.py`` and check its schema."""
    df = pd.read_csv(resolve(cfg, "raw_csv"), dtype=RAW_DTYPES, parse_dates=["InvoiceDate"])
    if list(df.columns) != cfg["data"]["expected_columns"]:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    log.info("Loaded %s raw lines", len(df))
    return df


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path, params: dict[str, Any]):
    """Execute each statement in a SQL file, passing only the parameters it references.

    Returns the relation produced by the last statement.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("--"))
    result = None
    for stmt in (s.strip() for s in text.split(";")):
        if not stmt:
            continue
        names = set(re.findall(r"\$(\w+)", stmt))
        missing = names - params.keys()
        if missing:
            raise KeyError(f"Missing SQL parameters: {sorted(missing)}")
        result = con.execute(stmt, {k: params[k] for k in names})
    return result


def cleaning_params(cfg: dict[str, Any], drop_duplicates: bool | None = None) -> dict[str, Any]:
    """Build SQL parameters for the cleaning step from config."""
    c = cfg["cleaning"]
    return {
        "cancel_prefix": c["cancellation_prefix"],
        "non_product_codes": list(c["non_product_codes"]),
        "non_product_prefixes": list(c["non_product_prefixes"]),
        "drop_duplicates": (
            c["drop_exact_duplicates"] if drop_duplicates is None else drop_duplicates
        ),
        "giant_line_qty": int(c["giant_line_qty"]),
    }


def clean_lines(
    con: duckdb.DuckDBPyConnection,
    raw: pd.DataFrame,
    cfg: dict[str, Any],
    drop_duplicates: bool | None = None,
) -> pd.DataFrame:
    """Run the SQL cleaning rules and return the cleaning log (lines and value by status).

    Creates the tables ``line_status``, ``purchases`` and ``cancellations`` in ``con``.
    """
    con.register("raw_lines", raw.assign(line_no=np.arange(len(raw))))
    sql = resolve(cfg, "sql_dir") / "01_clean_lines.sql"
    run_sql_file(con, sql, cleaning_params(cfg, drop_duplicates))
    log_df = con.execute("""
        SELECT status, count(*) AS lines, sum(abs(line_value)) AS abs_value
        FROM line_status GROUP BY status
        """).df()
    log_df["status"] = pd.Categorical(log_df["status"], STATUS_ORDER, ordered=True)
    log_df = log_df.sort_values("status").reset_index(drop=True)
    log_df["share_of_lines"] = log_df["lines"] / log_df["lines"].sum()
    log_df["share_of_value"] = log_df["abs_value"] / log_df["abs_value"].sum()
    return log_df


def flag_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Independent (overlapping) counts of each raw-line flag, before priority is applied."""
    row = con.execute("""
        SELECT
            sum(is_cancellation::INT) AS cancellation,
            sum(is_missing_customer::INT) AS missing_customer_id,
            sum(is_non_product::INT) AS non_product_code,
            sum(is_price_le_zero::INT) AS price_le_zero,
            sum(is_neg_qty_non_cancellation::INT) AS neg_qty_non_cancellation,
            sum(is_exact_duplicate::INT) AS exact_duplicate
        FROM flagged_lines
        """).df()
    return {k: int(v) for k, v in row.iloc[0].items()}


def clean_lines_pandas(
    raw: pd.DataFrame, cfg: dict[str, Any], drop_duplicates: bool | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Pandas reference implementation of ``01_clean_lines.sql`` (used by parity tests).

    Returns:
        purchases, cancellations and the per-line status.
    """
    p = cleaning_params(cfg, drop_duplicates)
    raw = raw.reset_index(drop=True)  # positional line numbers, as in the SQL
    code = raw["StockCode"].astype(str)
    is_cancel = raw["Invoice"].astype(str).str.startswith(p["cancel_prefix"])
    non_product = code.isin(p["non_product_codes"]) | code.str.startswith(
        tuple(p["non_product_prefixes"])
    )
    conditions = [
        ("missing_customer_id", raw["Customer ID"].isna()),
        ("non_product_code", non_product),
        ("price_le_zero", raw["Price"] <= 0),
        ("neg_qty_non_cancellation", (raw["Quantity"] < 0) & ~is_cancel),
        ("exact_duplicate", raw.duplicated(keep="first") & p["drop_duplicates"]),
        ("cancellation", is_cancel),
    ]
    status = pd.Series("purchase", index=raw.index)
    assigned = pd.Series(False, index=raw.index)
    for name, cond in conditions:
        hit = cond.fillna(False).astype(bool) & ~assigned
        status[hit] = name
        assigned |= hit

    # Matched reversal of giant lines, one to one in time order (same rule as the SQL).
    qty, giant = raw["Quantity"], p["giant_line_qty"]
    key_cols = ["Customer ID", "StockCode", "Price", "q"]
    sides = []
    for mask, q in [
        ((status == "purchase") & (qty >= giant), qty),
        ((status == "cancellation") & (-qty >= giant), -qty),
    ]:
        d = raw[mask].assign(q=q[mask], line_no=raw.index[mask])
        d = d.sort_values(["InvoiceDate", "line_no"])
        d["k"] = d.groupby(key_cols).cumcount()
        sides.append(d)
    m = sides[0].merge(sides[1], on=key_cols + ["k"], suffixes=("_p", "_c"))
    m = m[m["InvoiceDate_c"] >= m["InvoiceDate_p"]]
    status[m["line_no_p"].tolist() + m["line_no_c"].tolist()] = "reversed_giant_line"

    def _frame(mask: pd.Series) -> pd.DataFrame:
        d = raw[mask]
        return pd.DataFrame(
            {
                "invoice": d["Invoice"],
                "stock_code": d["StockCode"],
                "quantity": d["Quantity"],
                "invoice_ts": pd.to_datetime(d["InvoiceDate"]),
                "price": d["Price"],
                "customer_id": d["Customer ID"].astype("int64"),
                "country": d["Country"],
                "line_value": d["Quantity"] * d["Price"],
            }
        )

    purchases = _frame(status == "purchase")
    cancellations = _frame(status == "cancellation").rename(columns={"line_value": "cancel_value"})
    cancellations["cancel_value"] = cancellations["cancel_value"].abs()
    return purchases, cancellations, status
