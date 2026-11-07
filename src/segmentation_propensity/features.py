"""Build customer snapshots (features before a cutoff, 90-day label after it) with SQL.

``build_snapshot_pandas`` is a reference implementation used only by parity tests.

Usage:
    python -m segmentation_propensity.features [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from segmentation_propensity.data import (
    clean_lines,
    connect,
    flag_counts,
    load_raw,
    run_sql_file,
)
from segmentation_propensity.utils import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

FEATURES = [
    "recency_days",
    "frequency_invoices",
    "frequency_invoices_last_90d",
    "monetary_net_gbp",
    "avg_order_value_gbp",
    "avg_lines_per_invoice",
    "median_units_per_line",
    "n_distinct_products",
    "active_months",
    "tenure_days",
    "mean_interpurchase_days",
    "single_invoice_flag",
    "return_rate_value",
    "is_uk",
    "label_window_q4",
]


def snapshot_params(cfg: dict[str, Any], cutoff: str) -> dict[str, Any]:
    """SQL parameters for one snapshot."""
    w = cfg["windows"]
    return {
        "cutoff": str(pd.Timestamp(cutoff)),
        "lookback_days": int(w["lookback_days"]),
        "label_window_days": int(w["label_window_days"]),
        "q4_months": [int(m) for m in w["q4_months"]],
    }


def build_snapshot(
    con: duckdb.DuckDBPyConnection, cfg: dict[str, Any], cutoff: str
) -> pd.DataFrame:
    """Build features and labels for one cutoff. Requires ``purchases`` and ``cancellations``."""
    sql_dir = resolve(cfg, "sql_dir")
    params = snapshot_params(cfg, cutoff)
    run_sql_file(con, sql_dir / "02_customer_snapshot_features.sql", params)
    df = run_sql_file(con, sql_dir / "03_snapshot_labels.sql", params).df()
    if list(df.columns) != ["customer_id"] + FEATURES + ["y"]:
        raise ValueError(f"Unexpected snapshot columns: {list(df.columns)}")
    return df


def build_all_snapshots(
    con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Build every snapshot named in config."""
    return {
        name: build_snapshot(con, cfg, s["cutoff"])
        for name, s in cfg["windows"]["snapshots"].items()
    }


def snapshot_summary(snapshots: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Customers, events and events per CV fold for each snapshot."""
    folds = cfg["propensity"]["cv_folds"]
    rows = []
    for name, df in snapshots.items():
        events = int(df["y"].sum())
        rows.append(
            {
                "snapshot": name,
                "cutoff": cfg["windows"]["snapshots"][name]["cutoff"],
                "customers": len(df),
                "events": events,
                "non_events": len(df) - events,
                "event_rate": events / len(df) if len(df) else np.nan,
                "events_per_fold": events // folds,
            }
        )
    return pd.DataFrame(rows)


def build_snapshot_pandas(
    purchases: pd.DataFrame, cancellations: pd.DataFrame, cfg: dict[str, Any], cutoff: str
) -> pd.DataFrame:
    """Pandas reference implementation of ``02`` and ``03`` (used by parity tests)."""
    w = cfg["windows"]
    c = pd.Timestamp(cutoff)
    start = c - pd.Timedelta(days=int(w["lookback_days"]))
    p = purchases[(purchases.invoice_ts >= start) & (purchases.invoice_ts < c)]
    k = cancellations[(cancellations.invoice_ts >= start) & (cancellations.invoice_ts < c)]
    inv = (
        p.groupby(["customer_id", "invoice"])
        .agg(
            inv_ts=("invoice_ts", "min"),
            inv_value=("line_value", "sum"),
            n_lines=("stock_code", "size"),
        )
        .reset_index()
        .sort_values(["customer_id", "inv_ts", "invoice"])
    )
    inv["gap"] = inv.groupby("customer_id")["inv_ts"].diff().dt.total_seconds() / 86400.0
    recent = inv[inv.inv_ts >= c - pd.Timedelta(days=90)]
    g = inv.groupby("customer_id")
    out = pd.DataFrame(
        {
            "last_ts": g.inv_ts.max(),
            "frequency_invoices": g.size(),
            "purchase_value": g.inv_value.sum(),
            "avg_order_value_gbp": g.inv_value.mean(),
            "avg_lines_per_invoice": g.n_lines.mean(),
            "mean_gap": g.gap.mean(),
        }
    )
    out["frequency_invoices_last_90d"] = (
        recent.groupby("customer_id").size().reindex(out.index).fillna(0)
    )
    lg = p.groupby("customer_id")
    out["median_units_per_line"] = lg.quantity.median()
    out["n_distinct_products"] = lg.stock_code.nunique()
    out["active_months"] = lg.invoice_ts.agg(lambda s: s.dt.to_period("M").nunique())
    cancel = k.groupby("customer_id").cancel_value.sum().reindex(out.index).fillna(0)
    first = purchases[purchases.invoice_ts < c].groupby("customer_id").invoice_ts.min()
    cc = p.groupby(["customer_id", "country"]).size().rename("n").reset_index()
    modal = cc.sort_values(["customer_id", "n", "country"], ascending=[True, False, True])
    modal = modal.drop_duplicates("customer_id").set_index("customer_id").country

    res = pd.DataFrame(index=out.index)
    res["recency_days"] = (c.normalize() - out.last_ts.dt.normalize()).dt.days
    res["frequency_invoices"] = out.frequency_invoices
    res["frequency_invoices_last_90d"] = out.frequency_invoices_last_90d
    res["monetary_net_gbp"] = (out.purchase_value - cancel).clip(lower=0)
    res["avg_order_value_gbp"] = out.avg_order_value_gbp
    res["avg_lines_per_invoice"] = out.avg_lines_per_invoice
    res["median_units_per_line"] = out.median_units_per_line
    res["n_distinct_products"] = out.n_distinct_products
    res["active_months"] = out.active_months
    res["tenure_days"] = (c.normalize() - first.reindex(out.index).dt.normalize()).dt.days
    res["mean_interpurchase_days"] = out.mean_gap.fillna(float(w["lookback_days"]))
    res["single_invoice_flag"] = (out.frequency_invoices == 1).astype(int)
    res["return_rate_value"] = cancel / (out.purchase_value + cancel)
    res["is_uk"] = (modal.reindex(out.index) == "United Kingdom").astype(int)
    res["label_window_q4"] = int(c.month in w["q4_months"])
    end = c + pd.Timedelta(days=int(w["label_window_days"]))
    buyers = purchases[
        (purchases.invoice_ts >= c) & (purchases.invoice_ts < end)
    ].customer_id.unique()
    res["y"] = res.index.isin(buyers).astype(int)
    return res.reset_index().rename(columns={"index": "customer_id"}).sort_values("customer_id")


def window_overlap(snaps: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Count test-window customers who are also in the window the segments were fitted on.

    Customer IDs only; nothing is scored. The segment check is out-of-time, not on new customers.
    """
    roles = {name: s["role"] for name, s in cfg["windows"]["snapshots"].items()}
    fit = next(name for name, role in roles.items() if role.startswith("segmentation_fit"))
    test = next(name for name, role in roles.items() if role == "held_out_once")
    fit_ids, test_ids = set(snaps[fit].customer_id), set(snaps[test].customer_id)
    both = len(fit_ids & test_ids)
    return {
        "segment_fit_window": fit,
        "test_window": test,
        "test_customers": len(test_ids),
        "test_customers_in_segment_fit_window": both,
        "share_of_test_customers": both / len(test_ids),
    }


def write_data_summary(
    con: duckdb.DuckDBPyConnection,
    clog: pd.DataFrame,
    snaps: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Save aggregate cleaning and snapshot counts (no customer rows).

    Written to ``processed_dir/phase1_summary.json`` and to ``metrics_dir/data_summary.json``;
    the second copy is committed so the docs can cite it on a fresh clone.
    """
    summary = snapshot_summary(snaps, cfg)
    record = {
        "cleaning_log": clog.assign(status=clog.status.astype(str)).to_dict(orient="records"),
        "flag_counts": flag_counts(con),
        "snapshots": summary.to_dict(orient="records"),
        "window_overlap": window_overlap(snaps, cfg),
    }
    text = json.dumps(record, indent=2, default=float)
    for path in (
        resolve(cfg, "processed_dir") / "phase1_summary.json",
        resolve(cfg, "metrics_dir") / "data_summary.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    log.info("Snapshots:\n%s", summary.to_string(index=False))
    return record


def main(config_path: str | None = None) -> None:
    """Clean the raw data, build every snapshot and save them with a cleaning log."""
    cfg = load_config(config_path)
    setup_logging(cfg["logging"]["level"])
    con = connect(cfg)
    clog = clean_lines(con, load_raw(cfg), cfg)
    snaps = build_all_snapshots(con, cfg)
    out = resolve(cfg, "processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in snaps.items():
        df.to_csv(out / f"snapshot_{name}.csv", index=False)
    write_data_summary(con, clog, snaps, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
