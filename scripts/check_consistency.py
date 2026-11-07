"""Check that every number in the README and docs traces to a committed results file.

Sources: outputs/metrics (JSON and CSV), assumptions.yaml and config.yaml. Numbers that are not
results (software versions, the licence version, design choices discussed in the text) are listed
with a reason in docs/consistency_allowlist.yaml.

Usage:
    python scripts/check_consistency.py
Exits with status 1 and lists each unmatched number if any are found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation_propensity.consistency import check_documents  # noqa: E402


def main() -> int:
    """Run the check and print a report."""
    docs = [ROOT / "README.md", ROOT / "data" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    metrics = ROOT / "outputs" / "metrics"
    sources = [
        *sorted(metrics.glob("*.json")),
        *sorted(metrics.glob("*.csv")),
        ROOT / "assumptions.yaml",
        ROOT / "config.yaml",
    ]
    allow_file = ROOT / "docs" / "consistency_allowlist.yaml"
    allow = [e["value"] for e in yaml.safe_load(allow_file.read_text(encoding="utf-8"))["allow"]]
    bad = check_documents(docs, sources, allow, root=ROOT)
    for m in bad:
        print(f"{m.file}:{m.line}: {m.token!r} not found in sources | {m.context}")
    print(f"Checked {len(docs)} documents against {len(sources)} sources: {len(bad)} unmatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
