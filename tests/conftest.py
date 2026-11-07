"""Shared pytest fixtures. Paths resolve from this file, never from the working directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from segmentation_propensity.utils import load_config  # noqa: E402
from tests.fixtures import invoice_lines  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "config.yaml")


@pytest.fixture()
def lines():
    return invoice_lines()


@pytest.fixture()
def lines_with_case():
    return invoice_lines(with_case=True)
