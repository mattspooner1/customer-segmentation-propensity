"""Download UCI Online Retail II from the official host, verify it and convert it to CSV.

Steps: HTTPS download, SHA-256 check against config, zip member path check, extraction into an
empty folder, file type check, xlsx to CSV conversion, row and column count check.

Usage:
    python scripts/download_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_propensity.utils import load_config, resolve, setup_logging  # noqa: E402

log = setup_logging()


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unsafe_members(zf: zipfile.ZipFile) -> list[str]:
    """Return archive members whose paths are absolute or escape the extraction folder."""
    bad = []
    for name in zf.namelist():
        p = Path(name)
        if p.is_absolute() or name.startswith(("/", "\\")) or ".." in p.parts or ":" in name:
            bad.append(name)
    return bad


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` over HTTPS only."""
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS downloads are allowed")
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:  # noqa: S310
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main(config_path: str | None = None) -> None:
    """Run the download and verification pipeline."""
    import pandas as pd

    cfg = load_config(config_path)
    d = cfg["data"]
    zip_path, extract_dir, csv_path = (
        resolve(cfg, k) for k in ("raw_zip", "raw_extract_dir", "raw_csv")
    )
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        download(d["download_url"], zip_path)
    digest = sha256_of(zip_path)
    if digest != d["sha256"]:
        raise ValueError(f"Checksum mismatch: got {digest}")
    log.info("SHA-256 verified")

    with zipfile.ZipFile(zip_path) as zf:
        bad = unsafe_members(zf)
        if bad:
            raise ValueError(f"Unsafe archive members: {bad}")
        if zf.namelist() != [d["expected_member"]]:
            raise ValueError(f"Unexpected archive contents: {zf.namelist()}")
        if extract_dir.exists() and any(extract_dir.iterdir()):
            raise FileExistsError("Extraction folder is not empty; remove it to re-extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    xlsx = extract_dir / d["expected_member"]
    with open(xlsx, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            raise ValueError("Extracted file is not an xlsx (zip) container")

    sheets = pd.read_excel(
        xlsx, sheet_name=None, engine="openpyxl", dtype={"Invoice": str, "StockCode": str}
    )
    df = pd.concat(sheets.values(), ignore_index=True)
    if list(df.columns) != d["expected_columns"]:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != d["expected_rows"]:
        raise ValueError(f"Unexpected row count: {len(df)}")
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s rows x %s columns to %s", len(df), df.shape[1], csv_path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
