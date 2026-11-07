"""Setup checks: config contract, window design, fixtures and download safety helpers."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from segmentation_propensity.utils import resolve, set_seed
from tests.fixtures import COLUMNS


def _load_download_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "download_data", root / "scripts" / "download_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_python_version_supported():
    assert (3, 11) <= sys.version_info[:2] < (3, 14)


def test_config_has_every_section(cfg):
    for key in [
        "project",
        "paths",
        "data",
        "cleaning",
        "windows",
        "features",
        "segmentation",
        "rfm_rule",
        "propensity",
        "evaluation",
        "targets",
        "outcome",
    ]:
        assert key in cfg, key


def test_paths_are_relative(cfg):
    for key, value in cfg["paths"].items():
        assert not Path(value).is_absolute(), key
        assert resolve(cfg, key).is_relative_to(cfg["_root"])


def test_training_label_windows_end_by_test_cutoff(cfg):
    w = cfg["windows"]
    label = pd.Timedelta(days=w["label_window_days"])
    test_cutoff = pd.Timestamp(w["snapshots"]["test"]["cutoff"])
    for name in cfg["propensity"]["train_snapshots"]:
        assert pd.Timestamp(w["snapshots"][name]["cutoff"]) + label <= test_cutoff, name
    assert test_cutoff + label <= pd.Timestamp(w["data_end"])


def test_earliest_lookback_inside_data(cfg):
    w = cfg["windows"]
    earliest = min(pd.Timestamp(s["cutoff"]) for s in w["snapshots"].values())
    assert earliest - pd.Timedelta(days=w["lookback_days"]) >= pd.Timestamp("2009-12-01")


def test_feature_budget(cfg):
    assert len(cfg["features"]["all"]) <= 150  # feature budget: dev events / 10
    assert "label_window_q4" not in sum(cfg["features"]["segmentation_sets"].values(), [])


def test_every_target_has_no_skill_reference(cfg):
    for name, t in cfg["targets"].items():
        assert "no_skill" in t or "max" in t or "slope_range" in t, name


def test_fixture_schema_matches_raw(cfg, lines):
    assert list(lines.columns) == COLUMNS == cfg["data"]["expected_columns"]


def test_fixture_covers_every_cleaning_case(lines_with_case):
    needed = {
        "cancellation",
        "missing_customer_id",
        "non_product_code",
        "non_product_prefix",
        "exact_duplicate",
        "giant_line",
        "price_zero",
        "neg_qty_non_cancellation",
        "purchase_label_window",
    }
    assert needed <= set(lines_with_case["case"])


def test_seed_is_reproducible():
    assert set_seed(7).random() == set_seed(7).random()


def test_unsafe_zip_members_detected(root, tmp_path):
    mod = _load_download_module(root)
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("online_retail_II.xlsx", b"x")
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.txt", b"x")
        z.writestr("/abs.txt", b"x")
    with zipfile.ZipFile(good) as z:
        assert mod.unsafe_members(z) == []
    with zipfile.ZipFile(bad) as z:
        assert len(mod.unsafe_members(z)) == 2


def test_sha256_and_https_only(root, tmp_path):
    mod = _load_download_module(root)
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert mod.sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        mod.download("http://example.com/file.zip", tmp_path / "x")
