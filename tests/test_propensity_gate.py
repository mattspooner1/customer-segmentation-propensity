"""Propensity selection and the G2 gate on synthetic snapshots (no raw data)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from segmentation_propensity.gate import (
    calibration,
    evaluate_test,
    rerender_gate,
    verdict,
    write_gate,
)
from segmentation_propensity.propensity import select_propensity, vif
from segmentation_propensity.segmentation import fit_segment_model
from tests.test_segmentation import synthetic_snapshot


@pytest.fixture()
def fast_cfg(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["evaluation"]["bootstrap_resamples"] = 50
    c["evaluation"]["permutations"] = 20
    c["propensity"]["logistic"]["C_grid"] = [0.1, 1.0]
    c["propensity"]["hist_gradient_boosting"]["max_iter"] = 30
    c["paths"]["results_json"] = str(tmp_path / "results.json")
    c["paths"]["evidence_gate_md"] = str(tmp_path / "evidence_gate.md")
    return c


@pytest.fixture()
def snaps():
    s = {
        n: synthetic_snapshot(seed=i) for i, n in enumerate(["seasonal_2010", "dev", "val", "test"])
    }
    s["seasonal_2010"]["label_window_q4"] = 1
    s["test"]["label_window_q4"] = 1
    return s


@pytest.mark.parametrize(
    "value,ci_low,expected",
    [
        (0.45, 0.40, "INVALID"),
        (0.50, 0.45, "NONE"),
        (0.55, 0.49, "WEAK"),
        (0.58, 0.53, "MODEST"),
        (0.70, 0.62, "STRONG"),
        (0.66, 0.61, "MODEST"),
    ],
)
def test_verdict_rules(cfg, value, ci_low, expected):
    assert verdict(value, ci_low, 0.5, 0.68, cfg) == expected  # 0.66 misses minimum useful 0.68


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    c = calibration(y, p)
    assert c["slope"] == pytest.approx(1.0, abs=0.08)
    assert abs(c["calibration_in_the_large"]) < 0.02


def test_propensity_selection_never_needs_test(fast_cfg, snaps):
    sel = {k: v for k, v in snaps.items() if k != "test"}
    res = select_propensity(sel, fast_cfg)
    assert res["chosen"] in {"logistic", "hist_gradient_boosting"}
    assert "test" not in set(res["train"].window)
    assert res["pre_val_windows"] == ["seasonal_2010", "dev"]


def test_gate_end_to_end(fast_cfg, snaps):
    sel = {k: v for k, v in snaps.items() if k != "test"}
    seg = fit_segment_model(snaps["dev"], "A_rfm_core", 3, fast_cfg)
    prop = select_propensity(sel, fast_cfg)
    ev = evaluate_test(snaps, seg, prop["model"], prop["train"], {"S3_ari_p05": 0.9}, fast_cfg)
    ids = [e["id"] for e in ev["entries"]]
    assert ids[:2] == ["S1", "S1_ref_rule"] and "P1" in ids
    for e in ev["entries"]:
        assert set(e) >= {
            "id",
            "model",
            "split",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "n",
            "n_events",
            "baseline_name",
            "baseline_value",
            "train_value",
            "verdict",
        }
    assert ev["status"]["S1_verdict"] in {"STRONG", "MODEST"}  # separable synthetic groups
    assert ev["status"]["mode"] == "Findings"
    results, gate = write_gate(ev, fast_cfg, {})
    assert results.exists() and "Mode: **Findings**" in gate.read_text(encoding="utf-8")
    before, first_gate = results.read_bytes(), gate.read_text(encoding="utf-8")
    gate.unlink()
    rerender_gate(fast_cfg)  # markdown only, from the saved results.json
    assert results.read_bytes() == before
    assert gate.read_text(encoding="utf-8") == first_gate
    assert "scored six times" in first_gate and "evaluated once" not in first_gate


def test_vif_flags_collinear_features(cfg, snaps):
    df = snaps["dev"].copy()
    df["active_months"] = df["frequency_invoices"] * 1.0 + 1e-6 * np.arange(len(df))
    assert vif(df, cfg)["active_months"] > 10
