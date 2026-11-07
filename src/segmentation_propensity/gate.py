"""G2 evidence gate: one evaluation call on the held-out test window, verdicts computed in code.

Verdict rules (quality gates section 5):
    INVALID: discrimination metric below 0.5.
    STRONG: CI lower bound >= baseline + strong_margin and the minimum useful target is met.
    MODEST: CI lower bound above the baseline.
    WEAK: point estimate above the baseline but the CI includes it.
    NONE: point estimate at or below the baseline.
    OVERFIT flag: train minus test above overfit_gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from segmentation_propensity.evaluation import (
    auc_with_ci,
    bootstrap_ci,
    paired_auc_difference,
    permuted_cramers_v,
    safe_auc,
)
from segmentation_propensity.data import connect
from segmentation_propensity.propensity import PropensityModel
from segmentation_propensity.segmentation import SegmentModel, rfm_rule_groups, rule_score
from segmentation_propensity.utils import resolve


def verdict(value: float, ci_low: float, baseline: float, minimum_useful: float, cfg: dict) -> str:
    """Compute the G2 verdict for a discrimination metric."""
    if value < 0.5:
        return "INVALID"
    if value <= baseline:
        return "NONE"
    if ci_low <= baseline:
        return "WEAK"
    if ci_low >= baseline + cfg["evaluation"]["strong_margin"] and value >= minimum_useful:
        return "STRONG"
    return "MODEST"


def calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration slope and intercept (logistic recalibration) and calibration-in-the-large."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit, y)
    return {
        "slope": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
        "mean_predicted": float(p.mean()),
        "event_rate": float(y.mean()),
        "calibration_in_the_large": float(p.mean() - y.mean()),
    }


def _entry(id_, model, metric, res, baseline_name, baseline_value, train_value=None, verdict_=None):
    return {
        "id": id_,
        "model": model,
        "split": "test",
        "metric": metric,
        "value": res["value"],
        "ci_low": res.get("ci_low"),
        "ci_high": res.get("ci_high"),
        "n": res.get("n"),
        "n_events": res.get("n_events"),
        "baseline_name": baseline_name,
        "baseline_value": baseline_value,
        "train_value": train_value,
        "verdict": verdict_,
    }


def evaluate_test(
    snaps: dict[str, pd.DataFrame],
    seg: SegmentModel,
    prop: PropensityModel,
    prop_train: pd.DataFrame,
    stability: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Score every pre-declared metric on the test window in one call."""
    rng = np.random.default_rng(cfg["project"]["seed"] + 1)
    nb, t = cfg["evaluation"]["bootstrap_resamples"], cfg["targets"]
    dev, test = snaps["dev"], snaps["test"]
    y = test.y.to_numpy()

    # S1: frozen segments, dev rates, scored on test
    seg_test = seg.predict(test)
    s1 = auc_with_ci(y, seg.score(test), rng, nb)
    s1_train = safe_auc(dev.y.to_numpy(), seg.score(dev))
    s1_v = verdict(s1["value"], s1["ci_low"], 0.5, t["S1_segment_score_auc"]["minimum_useful"], cfg)

    # S2: segments minus the RFM rule with the same number of groups (dev cut points)
    con = connect(cfg)
    rule_dev = rfm_rule_groups(con, dev, dev, seg.k, cfg)
    rule_test = rfm_rule_groups(con, dev, test, seg.k, cfg)
    rule_s = rule_score(dev, rule_dev, rule_test)
    rule_auc = auc_with_ci(y, rule_s, rng, nb)
    s2 = paired_auc_difference(y, seg.score(test), rule_s, rng, nb)

    # S4: does segment membership add to recency alone? (logistic fitted on dev)
    def design(df, with_seg):
        r = ((df.recency_days - dev.recency_days.mean()) / dev.recency_days.std()).to_numpy()
        cols = [r]
        if with_seg:
            lab = seg.predict(df)
            cols += [(lab == j).astype(float) for j in range(1, seg.k)]
        return np.column_stack(cols)

    lr_r = LogisticRegression(max_iter=1000).fit(design(dev, False), dev.y)
    lr_rs = LogisticRegression(max_iter=1000).fit(design(dev, True), dev.y)
    p_r = lr_r.predict_proba(design(test, False))[:, 1]
    p_rs = lr_rs.predict_proba(design(test, True))[:, 1]
    s4 = paired_auc_difference(y, p_rs, p_r, rng, nb)
    recency_auc = auc_with_ci(y, -test.recency_days.to_numpy(), rng, nb)

    # S5: association of frozen segments with the outcome on test
    s5 = permuted_cramers_v(seg_test, y, rng, cfg["evaluation"]["permutations"])

    # P1-P4: propensity
    p = prop.predict_proba(test)
    p1 = auc_with_ci(y, p, rng, nb)
    p1_train = safe_auc(prop_train.y.to_numpy(), prop.predict_proba(prop_train))
    p1_v = verdict(p1["value"], p1["ci_low"], 0.5, t["P1_propensity_auc"]["minimum_useful"], cfg)
    cal = calibration(y, p)
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((y.mean() - y) ** 2))
    bss_ci = bootstrap_ci(
        lambda i: 1 - np.mean((p[i] - y[i]) ** 2) / max(np.mean((y[i].mean() - y[i]) ** 2), 1e-12),
        len(y),
        rng,
        nb,
    )
    within = {
        seg.names[j]: auc_with_ci(y[seg_test == j], p[seg_test == j], rng, nb)
        for j in range(seg.k)
        if 0 < y[seg_test == j].sum() < (seg_test == j).sum()
    }
    gap = p1_train - p1["value"]

    entries = [
        _entry(
            "S1",
            f"k-means {seg.feature_set} k={seg.k}",
            "segment_score_auc",
            s1,
            "random segments",
            0.5,
            s1_train,
            s1_v,
        ),
        _entry(
            "S1_ref_rule",
            f"RFM quintile rule, {seg.k} groups",
            "segment_score_auc",
            rule_auc,
            "random segments",
            0.5,
        ),
        _entry("S1_ref_recency", "recency only", "auc", recency_auc, "random ranking", 0.5),
        _entry("S2", "segments minus RFM rule", "paired_auc_difference", s2, "no improvement", 0.0),
        _entry(
            "S4",
            "recency + segments minus recency (logistic)",
            "paired_auc_difference",
            s4,
            "no improvement",
            0.0,
        ),
        _entry(
            "S5",
            f"k-means {seg.feature_set} k={seg.k}",
            "cramers_v",
            {**s5, "n": int(len(y)), "n_events": int(y.sum())},
            "permuted-label null p95",
            s5["null_p95"],
        ),
        _entry("P1", prop.kind, "auc", p1, "random ranking", 0.5, p1_train, p1_v),
        _entry(
            "P3",
            prop.kind,
            "brier_skill_score",
            {
                "value": 1 - brier / brier_ref,
                "ci_low": bss_ci[0],
                "ci_high": bss_ci[1],
                "n": int(len(y)),
                "n_events": int(y.sum()),
            },
            "test prevalence",
            0.0,
        ),
    ]
    status = {
        "S1_verdict": s1_v,
        "P1_verdict": p1_v,
        "S1_minimum_useful_met": s1["value"] >= t["S1_segment_score_auc"]["minimum_useful"],
        "S2_minimum_useful_met": s2["ci_low"] > 0,
        "S3_ari_p05": stability["S3_ari_p05"],
        "S3_minimum_useful_met": stability["S3_ari_p05"]
        >= t["S3_bootstrap_ari_p05"]["minimum_useful"],
        "S4_minimum_useful_met": s4["ci_low"] > 0,
        "S5_above_null": s5["value"] > s5["null_p95"],
        "P1_minimum_useful_met": p1["value"] >= t["P1_propensity_auc"]["minimum_useful"],
        "P2_slope_ok": t["P2_calibration"]["slope_range"][0]
        <= cal["slope"]
        <= t["P2_calibration"]["slope_range"][1],
        "P2_bias_ok": abs(cal["calibration_in_the_large"]) <= t["P2_calibration"]["max_abs_bias"],
        "P3_brier_skill_positive": (1 - brier / brier_ref) > 0,
        "P4_train_test_gap": gap,
        "P4_overfit": gap > cfg["evaluation"]["overfit_gap"],
        "S1_overfit": (s1_train - s1["value"]) > cfg["evaluation"]["overfit_gap"],
    }
    req = cfg["outcome"]["findings_requires"]
    if s1_v in req["S1_verdict"] and status["S3_minimum_useful_met"]:
        mode = "Findings"
    elif s1_v in ("NONE", "INVALID"):
        mode = "Negative result"
    else:
        mode = "Exploratory"
    status["mode"] = mode
    status["recommended_segmentation"] = (
        "k-means segments" if status["S2_minimum_useful_met"] else "RFM quintile rule"
    )
    return {
        "entries": entries,
        "status": status,
        "calibration": cal,
        "within_segment_propensity_auc": within,
        "segment_test_rates": pd.Series(y)
        .groupby(seg_test)
        .mean()
        .rename(index=seg.names)
        .to_dict(),
        "segment_test_shares": pd.Series(seg_test)
        .value_counts(normalize=True)
        .sort_index()
        .rename(index=seg.names)
        .to_dict(),
        "predictions": pd.DataFrame(
            {
                "customer_id": test.customer_id,
                "y": y,
                "p": p,
                "segment": seg_test,
                "rule_group": rule_test.rfm_group.to_numpy(),
            }
        ),
    }


def _fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))


def write_gate(
    ev: dict[str, Any], cfg: dict[str, Any], context: dict[str, Any]
) -> tuple[Path, Path]:
    """Write outputs/metrics/results.json and evidence_gate.md from the evaluation."""
    results_path, gate_path = resolve(cfg, "results_json"), resolve(cfg, "evidence_gate_md")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "results": ev["entries"],
        "status": ev["status"],
        "calibration": ev["calibration"],
        "within_segment_propensity_auc": ev["within_segment_propensity_auc"],
        "segment_test_rates": ev["segment_test_rates"],
        "segment_test_shares": ev["segment_test_shares"],
        "context": context,
    }
    results_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    gate_path.write_text(render_gate(record, cfg), encoding="utf-8")
    return results_path, gate_path


def rerender_gate(cfg: dict[str, Any]) -> Path:
    """Rewrite evidence_gate.md from the saved results.json without scoring anything."""
    record = json.loads(resolve(cfg, "results_json").read_text(encoding="utf-8"))
    gate_path = resolve(cfg, "evidence_gate_md")
    gate_path.write_text(render_gate(record, cfg), encoding="utf-8")
    return gate_path


def render_gate(record: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Build the evidence_gate.md text from a results.json record."""
    st, cal, ev_cfg = record["status"], record["calibration"], cfg["evaluation"]
    test_cutoff = cfg["windows"]["snapshots"]["test"]["cutoff"]
    s3_min = cfg["targets"]["S3_bootstrap_ari_p05"]["minimum_useful"]
    lines = [
        "# Evidence gate (G2)",
        "",
        "Generated by `segmentation_propensity.gate` from `outputs/metrics/results.json`.",
        f"Test window: cutoff {test_cutoff}. All modelling choices were frozen on the dev and val",
        "windows before the first test score. The test window was scored six times in total: the",
        "first run, then five re-runs to check a record-merging fix and to remove run-to-run",
        "non-determinism. No choice changed after the first score, and the final numbers equal it.",
        "Before the plan was fixed, a feasibility check also ran a prototype of the approach on",
        "the same window; its results informed the minimum-useful targets and the training-window",
        "design.",
        f"Bootstrap: {ev_cfg['bootstrap_resamples']} customer resamples.",
        "",
        "> Illustrative scenario: real 2009-2011 retail data with no marketing contact records.",
        "> The outcome is ordering again within 90 days.",
        "",
        "| id | model | metric | value | 95% CI | baseline | train | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in record["results"]:
        ci = f"{_fmt(e['ci_low'])} to {_fmt(e['ci_high'])}"
        base = f"{e['baseline_name']} ({_fmt(e['baseline_value'])})"
        lines.append(
            f"| {e['id']} | {e['model']} | {e['metric']} | {_fmt(e['value'])} | {ci} | {base} "
            f"| {_fmt(e['train_value'])} | {e['verdict'] or '–'} |"
        )
    met = "met" if st["S3_minimum_useful_met"] else "not met"
    lines += [
        "",
        f"**Stability (S3, dev bootstrap):** ARI 5th percentile {st['S3_ari_p05']:.3f}"
        f" (minimum useful {s3_min}): {met}.",
        "",
        f"**Calibration (P2):** slope {cal['slope']:.3f}, calibration-in-the-large"
        f" {cal['calibration_in_the_large']:+.3f} (mean predicted {cal['mean_predicted']:.3f}"
        f" vs event rate {cal['event_rate']:.3f}). Slope in range: {st['P2_slope_ok']};"
        f" bias within 0.03: {st['P2_bias_ok']}.",
        "",
        "**Targets:**",
        f"- S1 minimum useful met: {st['S1_minimum_useful_met']}",
        f"- S2 segments beat the RFM rule (CI lower bound > 0): {st['S2_minimum_useful_met']}",
        f"- S4 segments add to recency (CI lower bound > 0): {st['S4_minimum_useful_met']}",
        f"- S5 above permuted null: {st['S5_above_null']}",
        f"- P1 minimum useful met: {st['P1_minimum_useful_met']}",
        f"- P3 Brier skill positive: {st['P3_brier_skill_positive']}",
        f"- Overfit flags: S1 {st['S1_overfit']}, P1 {st['P4_overfit']}"
        f" (P1 train-test gap {st['P4_train_test_gap']:.3f})",
        "",
        "## Gate outcome",
        f"- S1 verdict: **{st['S1_verdict']}**; P1 verdict: **{st['P1_verdict']}**",
        f"- Mode: **{st['mode']}**",
        f"- Recommended segmentation (decided by S2): **{st['recommended_segmentation']}**",
    ]
    if st["mode"] != "Findings":
        lines += [
            "",
            "## Options (owner decides)",
            "- Alternative dataset (UCI Bank Marketing) after a dataset viability check",
            "- Fallback plan: RFM rule as the segmentation of record, descriptive profiles",
            "- The matching outcome-path write-up",
        ]
    return "\n".join(lines) + "\n"
