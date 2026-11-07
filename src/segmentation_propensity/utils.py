"""Shared helpers: configuration, paths, logging and seeds."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and record its directory as the project root.

    Args:
        path: Path to a config file. Defaults to ``config.yaml`` at the repository root.

    Returns:
        The parsed configuration with an added ``_root`` key holding the config's directory.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = cfg_path.resolve().parent
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Return an absolute path for ``cfg["paths"][key]``, relative to the config's directory."""
    return Path(cfg["_root"]) / cfg["paths"][key]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package logger."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("segmentation_propensity")


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global generators and return a fresh NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
