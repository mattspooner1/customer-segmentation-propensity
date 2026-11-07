"""Customer segmentation: preprocessing, k-means selection on dev, RFM rule and profiles.

Everything here is fitted on the dev snapshot only and applied frozen to later windows.
Selection uses the dev and val snapshots; the test snapshot is never read in this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from segmentation_propensity.data import connect, run_sql_file
from segmentation_propensity.evaluation import auc_with_ci
from segmentation_propensity.utils import resolve

log = logging.getLogger(__name__)
RFM_COLS = ["recency_days", "frequency_invoices", "monetary_net_gbp"]


# --------------------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    """Winsorise, log1p and standardise with parameters fitted on one window (dev)."""

    features: list[str]
    caps: dict[str, float] = field(default_factory=dict)
    log1p: list[str] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str], cfg: dict[str, Any]) -> "Preprocessor":
        q = cfg["cleaning"]["winsorise_quantile"]
        logs = [f for f in features if f in cfg["features"]["log1p"]]
        pre = cls(features=list(features), log1p=logs)
        pre.caps = {f: float(df[f].quantile(q)) for f in features}
        z = pre._clip_log(df)
        pre.mean = {f: float(z[f].mean()) for f in features}
        pre.std = {f: float(z[f].std(ddof=0)) or 1.0 for f in features}
        return pre

    def _clip_log(self, df: pd.DataFrame) -> pd.DataFrame:
        z = pd.DataFrame({f: df[f].astype(float).clip(upper=self.caps[f]) for f in self.features})
        for f in self.log1p:
            z[f] = np.log1p(z[f].clip(lower=0))
        return z

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = self._clip_log(df)
        return np.column_stack([(z[f] - self.mean[f]) / self.std[f] for f in self.features])


# --------------------------------------------------------------------------- k-means model
@dataclass
class SegmentModel:
    """Frozen segmentation: preprocessing, centroids and dev outcome rates per segment."""

    feature_set: str
    k: int
    preprocessor: Preprocessor
    centroids: list[list[float]]
    segment_rates: dict[int, float] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Assign each customer to the nearest centroid."""
        x = self.preprocessor.transform(df)
        c = np.asarray(self.centroids)
        return ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Segment-level score: the dev repeat-purchase rate of each customer's segment."""
        return np.array([self.segment_rates[int(s)] for s in self.predict(df)])

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "SegmentModel":
        d = json.loads(text)
        d["preprocessor"] = Preprocessor(**d["preprocessor"])
        d["segment_rates"] = {int(k): v for k, v in d["segment_rates"].items()}
        d["names"] = {int(k): v for k, v in d["names"].items()}
        return cls(**d)


def fit_kmeans(x: np.ndarray, k: int, seed: int, n_init: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)


def relabel_by_rate(km: KMeans, labels: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order segments by dev repeat-purchase rate (0 = lowest) so labels are stable and readable."""
    rates = np.array([y[labels == j].mean() for j in range(km.n_clusters)])
    order = np.argsort(rates)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return km.cluster_centers_[order], remap[labels]


def bootstrap_ari(
    x: np.ndarray, ref_labels: np.ndarray, k: int, seed: int, n_refits: int, n_init: int
) -> np.ndarray:
    """ARI between the reference labels and labels from k-means refitted on bootstrap resamples."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_refits)
    for b in range(n_refits):
        idx = rng.integers(0, len(x), len(x))
        km = fit_kmeans(x[idx], k, seed + b + 1, n_init)
        out[b] = adjusted_rand_score(ref_labels, km.predict(x))
    return out


def evaluate_k_grid(dev: pd.DataFrame, set_name: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Silhouette, bootstrap ARI and smallest segment share for each k on the dev window."""
    s = cfg["segmentation"]
    seed = cfg["project"]["seed"]
    feats = cfg["features"]["segmentation_sets"][set_name]
    x = Preprocessor.fit(dev, feats, cfg).transform(dev)
    rows = []
    for k in range(s["k_range"][0], s["k_range"][1] + 1):
        km = fit_kmeans(x, k, seed, s["n_init"])
        ari = bootstrap_ari(x, km.labels_, k, seed, s["bootstrap_refits"], s["n_init"])
        shares = np.bincount(km.labels_, minlength=k) / len(x)
        rows.append(
            {
                "feature_set": set_name,
                "k": k,
                "silhouette": float(silhouette_score(x, km.labels_)),
                "ari_median": float(np.median(ari)),
                "ari_p05": float(np.percentile(ari, 5)),
                "min_segment_share": float(shares.min()),
            }
        )
    grid = pd.DataFrame(rows)
    grid["size_ok"] = grid.min_segment_share >= s["min_segment_share"]
    return grid


def choose_k(grid: pd.DataFrame, cfg: dict[str, Any]) -> int | None:
    """Pick k: segments >= min share; ari_p05 within tolerance of the best; then best silhouette;
    remaining ties go to the smaller k. Returns None when no k meets the size rule."""
    tol = cfg["segmentation"]["ari_tie_tolerance"]
    ok = grid[grid.size_ok]
    if ok.empty:
        return None
    near = ok[ok.ari_p05 >= ok.ari_p05.max() - tol]
    best = near[near.silhouette == near.silhouette.max()]
    return int(best.k.min())


def fit_segment_model(
    dev: pd.DataFrame, set_name: str, k: int, cfg: dict[str, Any]
) -> SegmentModel:
    """Fit the frozen model on dev and record dev outcome rates per segment."""
    feats = cfg["features"]["segmentation_sets"][set_name]
    pre = Preprocessor.fit(dev, feats, cfg)
    x = pre.transform(dev)
    km = fit_kmeans(x, k, cfg["project"]["seed"], cfg["segmentation"]["n_init"])
    centroids, labels = relabel_by_rate(km, km.labels_, dev.y.to_numpy())
    rates = {j: float(dev.y.to_numpy()[labels == j].mean()) for j in range(k)}
    model = SegmentModel(set_name, k, pre, centroids.tolist(), rates)
    model.names = name_segments(dev, labels, cfg)
    return model


# --------------------------------------------------------------------------- profiles and names
def name_segments(df: pd.DataFrame, labels: np.ndarray, cfg: dict[str, Any]) -> dict[int, str]:
    """Plain-language names from segment medians against the window's thirds (and 90th percentile).

    Recency: recent / cooling / lapsed. Frequency: infrequent / regular / frequent / very frequent.
    Value: low / mid / high / top.
    """
    qs = df[RFM_COLS].quantile([1 / 3, 2 / 3, 0.9])

    def level(col: str, v: float, words: list[str]) -> str:
        cuts = qs[col].to_numpy()
        return words[int((v > cuts[:2]).sum()) + int(len(words) == 4 and v > cuts[2])]

    names: dict[int, str] = {}
    for j in sorted(set(int(v) for v in labels)):
        m = df.loc[labels == j, RFM_COLS + ["median_units_per_line"]].median()
        parts = [
            level("recency_days", m.recency_days, ["recent", "cooling", "lapsed"]),
            level(
                "frequency_invoices",
                m.frequency_invoices,
                ["infrequent", "regular", "frequent", "very frequent"],
            ),
            level(
                "monetary_net_gbp",
                m.monetary_net_gbp,
                ["low-value", "mid-value", "high-value", "top-value"],
            ),
        ]
        if m.median_units_per_line >= cfg["cleaning"]["wholesale_units_per_line"]:
            parts.append("bulk")
        names[j] = " ".join(parts)
    # Segments sharing a name are told apart by their median time since last order.
    for name in {n for n in names.values() if list(names.values()).count(n) > 1}:
        for j in [j for j, n in names.items() if n == name]:
            months = df.loc[labels == j, "recency_days"].median() / 30.4
            names[j] = f"{name}, last order ~{months:.0f} months ago"
    # Any remaining exact collision gets the segment number.
    seen: set[str] = set()
    for j in sorted(names):
        if names[j] in seen:
            names[j] = f"{names[j]} ({j})"
        seen.add(names[j])
    return names


def segment_profiles(df: pd.DataFrame, labels: np.ndarray, model: SegmentModel) -> pd.DataFrame:
    """Size, outcome rate, value share and feature medians per segment."""
    cols = model.preprocessor.features
    g = df.assign(segment=labels).groupby("segment")
    prof = g[cols].median().add_prefix("median_")
    prof.insert(0, "repeat_rate", g.y.mean())
    prof.insert(0, "value_share", g.monetary_net_gbp.sum() / df.monetary_net_gbp.sum())
    prof.insert(0, "share", g.size() / len(df))
    prof.insert(0, "customers", g.size())
    prof.insert(0, "name", [model.names[int(s)] for s in prof.index])
    return prof


# --------------------------------------------------------------------------- RFM rule (SQL)
def rfm_rule_groups(
    con: duckdb.DuckDBPyConnection,
    dev: pd.DataFrame,
    target: pd.DataFrame,
    k: int,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Score ``target`` with the RFM quintile rule using dev cut points (sql/04)."""
    q = cfg["rfm_rule"]["quintiles"]
    con.register("rfm_dev", dev[["customer_id"] + RFM_COLS])
    con.register("rfm_target", target[["customer_id"] + RFM_COLS])
    params = {
        "quintile_probs": [i / q for i in range(1, q)],
        "group_probs": [i / k for i in range(1, k)],
    }
    return run_sql_file(con, resolve(cfg, "sql_dir") / "04_rfm_rule_segments.sql", params).df()


def rfm_rule_pandas(
    dev: pd.DataFrame, target: pd.DataFrame, k: int, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Pandas reference implementation of sql/04 (used by parity tests)."""
    q = cfg["rfm_rule"]["quintiles"]
    probs = [i / q for i in range(1, q)]

    def scores(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"customer_id": d.customer_id.to_numpy()})
        for name, col, sign in [
            ("r_score", "recency_days", -1),
            ("f_score", "frequency_invoices", 1),
            ("m_score", "monetary_net_gbp", 1),
        ]:
            cuts = np.quantile(sign * dev[col].to_numpy(float), probs)
            v = sign * d[col].to_numpy(float)
            out[name] = 1 + (cuts[None, :] < v[:, None]).sum(1)
        out["rfm_score"] = out.r_score + out.f_score + out.m_score
        return out

    g_cuts = np.quantile(scores(dev).rfm_score.to_numpy(float), [i / k for i in range(1, k)])
    out = scores(target)
    out["rfm_group"] = (g_cuts[None, :] < out.rfm_score.to_numpy()[:, None]).sum(1)
    return out.sort_values("customer_id").reset_index(drop=True)


def rule_score(
    dev: pd.DataFrame, dev_groups: pd.DataFrame, target_groups: pd.DataFrame
) -> np.ndarray:
    """Dev repeat-purchase rate of each target customer's RFM group."""
    rates = dev.merge(dev_groups, on="customer_id").groupby("rfm_group").y.mean()
    return target_groups.rfm_group.map(rates).fillna(dev.y.mean()).to_numpy()


# --------------------------------------------------------------------------- selection (dev + val)
def select_segmentation(snaps: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run segmentation selection (feature set and k) on the dev and val windows only.

    Returns grids, chosen k per set, val comparison of sets A and B, the chosen frozen model,
    the RFM rule, stability and sensitivity results. ``snaps["test"]`` is never accessed.
    """
    s = cfg["segmentation"]
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    dev, val = snaps["dev"], snaps[s["set_selection_snapshot"]]
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    sets = list(cfg["features"]["segmentation_sets"])
    primary_sets = [n for n in sets if n != s["sensitivity_set"]]

    grids = {n: evaluate_k_grid(dev, n, cfg) for n in sets}
    chosen_k = {n: choose_k(g, cfg) for n, g in grids.items()}
    eligible = [n for n in sets if chosen_k[n] is not None]
    candidates = [n for n in primary_sets if n in eligible]
    if not candidates:
        raise ValueError("No primary feature set has a k meeting the size rule: use fallback")
    models = {n: fit_segment_model(dev, n, chosen_k[n], cfg) for n in eligible}
    yv = val.y.to_numpy()
    val_auc = {n: auc_with_ci(yv, models[n].score(val), rng, n_boot) for n in eligible}
    best_set = max(candidates, key=lambda n: val_auc[n]["value"])
    model = models[best_set]
    grid_row = grids[best_set].set_index("k").loc[model.k]

    con = connect(cfg)
    rule_dev = rfm_rule_groups(con, dev, dev, model.k, cfg)
    rule_val = rfm_rule_groups(con, dev, val, model.k, cfg)
    rule_val_auc = auc_with_ci(yv, rule_score(dev, rule_dev, rule_val), rng, n_boot)
    recency_val_auc = auc_with_ci(yv, -val.recency_days.to_numpy(), rng, n_boot)

    # temporal stability: dev-fitted model vs the same recipe refitted on val, on val customers
    refit_val = fit_segment_model(val, best_set, model.k, cfg)
    temporal_ari = float(adjusted_rand_score(model.predict(val), refit_val.predict(val)))

    # sensitivity: refit without the top share of customers by dev monetary value
    top = s["sensitivity"]["top_share_removed"]
    keep = dev.monetary_net_gbp < dev.monetary_net_gbp.quantile(1 - top)
    no_top = fit_segment_model(dev[keep], best_set, model.k, cfg)
    no_top_ari = float(adjusted_rand_score(model.predict(dev[keep]), no_top.predict(dev[keep])))

    stability = {
        "S3_ari_p05": float(grid_row.ari_p05),
        "ari_median": float(grid_row.ari_median),
        "minimum_useful": s["stability_min"],
        "meets_minimum": bool(grid_row.ari_p05 >= s["stability_min"]),
        "temporal_ari_dev_model_vs_val_refit": temporal_ari,
        "top_share_removed_ari": no_top_ari,
    }
    log.info("Chosen set %s with k=%s; S3 ari_p05=%.3f", best_set, model.k, grid_row.ari_p05)
    return {
        "grids": grids,
        "chosen_k": chosen_k,
        "ineligible_sets": [n for n in sets if n not in eligible],
        "models": models,
        "val_segment_score_auc": val_auc,
        "chosen_set": best_set,
        "model": model,
        "rfm_rule_val_auc": rule_val_auc,
        "recency_only_val_auc": recency_val_auc,
        "rule_groups_dev": rule_dev,
        "rule_groups_val": rule_val,
        "stability": stability,
    }


def save_selection(result: dict[str, Any], cfg: dict[str, Any], extra: dict | None = None) -> Path:
    """Write the selection record (JSON) and the frozen model (JSON, gitignored folder)."""
    metrics = resolve(cfg, "metrics_dir")
    metrics.mkdir(parents=True, exist_ok=True)
    record = {
        "chosen_set": result["chosen_set"],
        "chosen_k": result["chosen_k"],
        "ineligible_sets": result["ineligible_sets"],
        "grids": {n: g.to_dict(orient="records") for n, g in result["grids"].items()},
        "val_segment_score_auc": result["val_segment_score_auc"],
        "rfm_rule_val_auc": result["rfm_rule_val_auc"],
        "recency_only_val_auc": result["recency_only_val_auc"],
        "stability": result["stability"],
        "segment_names": result["model"].names,
        "dev_segment_rates": result["model"].segment_rates,
        "note": "Selection used dev and val windows only. Test window not evaluated.",
        **(extra or {}),
    }
    path = metrics / "segmentation_selection.json"
    if path.exists():  # keep fields written by other steps (for example notebook diagnostics)
        previous = json.loads(path.read_text(encoding="utf-8"))
        record = {**{k: v for k, v in previous.items() if k not in record}, **record}
    path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    models_dir = Path(cfg["_root"]) / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "segmentation_frozen.json").write_text(
        result["model"].to_json(), encoding="utf-8"
    )
    return path
