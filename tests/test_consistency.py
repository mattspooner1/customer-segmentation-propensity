"""Doc-number consistency check on synthetic documents and sources."""

from __future__ import annotations

import json

import pytest

from segmentation_propensity.consistency import check_documents, check_text, renderings


@pytest.fixture()
def rendered():
    return renderings([0.71190, 0.6944, -0.00482, 0.3496, 1067371, 2857.14])


@pytest.mark.parametrize(
    "text",
    [
        "AUC 0.712 (lower bound 0.694)",
        "difference -0.005 and rate 35% and 34.96%",
        "1,067,371 lines; break-even £2,857",
        "Use 3 groups, k=8, from 2009 to 2011-12-09, see `0.999` and [x](https://a.b/7777)",
        "SHA-256, doi:10.24432/C5CG6D, the 34.96th percentile and 0.71",
        "1. A numbered item\n## 2. A numbered heading",
    ],
)
def test_traceable_numbers_pass(text, rendered):
    assert check_text(text, "doc", rendered, set()) == []


def test_untraceable_numbers_are_reported(rendered):
    bad = check_text("intro\nAUC 0.713 and 88% and 4,321 customers", "doc", rendered, set())
    assert [b.token for b in bad] == ["0.713", "88%", "4,321"]
    assert all(b.line == 2 for b in bad)


def test_allowlist(rendered):
    assert check_text("Python 3.11", "doc", rendered, {"3.11"}) == []


def test_check_documents_reads_all_source_types(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"a": [{"v": 0.755}], "flag": True}))
    (tmp_path / "t.csv").write_text("name,value\nx,0.94\n")
    (tmp_path / "a.yaml").write_text("items:\n  - value: 500\n")
    (tmp_path / "notes.txt").write_text("ignored source type 0.123")
    doc = tmp_path / "README.md"
    doc.write_text("AUC 0.755, 94% of 500, and 0.123 is not sourced")
    srcs = [tmp_path / n for n in ("r.json", "t.csv", "a.yaml", "notes.txt")]
    bad = check_documents([doc], srcs, root=tmp_path)
    assert [(b.file, b.token) for b in bad] == [("README.md", "0.123")]
