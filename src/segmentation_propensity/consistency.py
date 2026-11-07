"""Check that every number quoted in the docs traces to a committed results file.

A number in a Markdown document passes if it equals some value from the sources (results JSON,
metrics CSVs, the assumptions register or the config) at the precision it is written, either as
written or as a percentage. Skipped as structural: integers up to ``SMALL_INT``, four-digit years,
ISO dates, URLs and DOIs, numbers inside code spans, code blocks and link targets, numbers
after a letter and hyphen (SHA-256), and numbers attached to an identifier such as S1 or A3.
Anything else needs an allowlist entry.

The check proves that each number exists in a source, not that it is attached to the right
claim; the claims ledger review does that.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

SMALL_INT = 20
MAX_DECIMALS = 3

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_CODE_SPAN = re.compile(r"`[^`]*`")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_URL = re.compile(r"(?:https?://|doi:)\S+|\b10\.\d{4,}/\S+")
_NUMBER = re.compile(r"(?<![\w.])(?<![A-Za-z]-)\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?(?![\w.])(%?)")
_LIST_MARKER = re.compile(r"^\s*\d+\.\s")


@dataclass
class Mismatch:
    """A number in a document with no matching source value."""

    file: str
    line: int
    token: str
    context: str


def _leaves(obj: Any) -> Iterable[float]:
    """Yield every numeric leaf of a nested JSON/YAML structure."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)


def _csv_numbers(path: Path) -> Iterable[float]:
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            for cell in row:
                try:
                    yield float(cell)
                except ValueError:
                    continue


def source_values(paths: Iterable[Path]) -> list[float]:
    """Collect numeric values from JSON, YAML and CSV source files."""
    out: list[float] = []
    for p in paths:
        if p.suffix == ".json":
            out.extend(_leaves(json.loads(p.read_text(encoding="utf-8"))))
        elif p.suffix in (".yaml", ".yml"):
            out.extend(_leaves(yaml.safe_load(p.read_text(encoding="utf-8"))))
        elif p.suffix == ".csv":
            out.extend(_csv_numbers(p))
    return out


def renderings(values: Iterable[float]) -> dict[bool, set[str]]:
    """All ways a value may be written: plain and as a percentage, at 0-3 decimals."""
    plain: set[str] = set()
    pct: set[str] = set()
    for v in values:
        a = abs(v)
        for d in range(MAX_DECIMALS + 1):
            plain.add(f"{a:.{d}f}")
            pct.add(f"{100 * a:.{d}f}")
    return {False: plain, True: pct}


def _strip(text: str) -> str:
    text = _CODE_BLOCK.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = _CODE_SPAN.sub(" ", text)
    text = _LINK_TARGET.sub("]", text)
    text = _URL.sub(" ", text)
    return _DATE.sub(" ", text)


def check_text(
    text: str, name: str, rendered: dict[bool, set[str]], allow: set[str]
) -> list[Mismatch]:
    """Return numbers in ``text`` that match no source value and are not allowlisted."""
    bad = []
    for i, line in enumerate(_strip(text).splitlines(), start=1):
        body = _LIST_MARKER.sub(" ", line)
        if body.lstrip().startswith("#"):
            body = re.sub(r"^\s*#+\s*\d+\.", " ", body)
        for m in _NUMBER.finditer(body):
            raw = m.group(0)
            is_pct = raw.endswith("%")
            num = re.sub(r"(st|nd|rd|th)$", "", raw.rstrip("%")).replace(",", "")
            if raw in allow or num in allow:
                continue
            if "." not in num and not is_pct:
                n = int(num)
                if n <= SMALL_INT or 1900 <= n <= 2100:
                    continue
            if num in rendered[is_pct] or (not is_pct and num in rendered[True]):
                continue
            bad.append(Mismatch(name, i, raw, line.strip()[:120]))
    return bad


def check_documents(
    docs: Iterable[Path],
    sources: Iterable[Path],
    allow: Iterable[str] = (),
    root: Path | None = None,
) -> list[Mismatch]:
    """Check every document against the sources."""
    rendered = renderings(source_values(sources))
    allow_set = {str(a) for a in allow}
    out: list[Mismatch] = []
    for d in docs:
        name = d.relative_to(root).as_posix() if root else d.name
        out.extend(check_text(d.read_text(encoding="utf-8"), name, rendered, allow_set))
    return out
