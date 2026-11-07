"""Run the full pipeline: clean, build snapshots, select and freeze segments, select the
propensity model, then evaluate the test window once and write the evidence gate.

The selection steps receive only the seasonal_2010, dev and val snapshots. The test snapshot is
passed to a single evaluation call after everything is frozen.

Usage:
    python scripts/run_pipeline.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from segmentation_propensity.data import clean_lines, connect, load_raw  # noqa: E402
from segmentation_propensity.features import build_all_snapshots, write_data_summary  # noqa: E402
from segmentation_propensity.gate import evaluate_test, write_gate  # noqa: E402
from segmentation_propensity.propensity import select_propensity  # noqa: E402
from segmentation_propensity.segmentation import save_selection, select_segmentation  # noqa: E402
import segmentation_propensity.utils as u  # noqa: E402


def main(config_path: str | None = None) -> dict:
    """Run every step and return the evaluation record."""
    cfg = u.load_config(config_path)
    log = u.setup_logging(cfg["logging"]["level"])
    u.set_seed(cfg["project"]["seed"])
    con = connect(cfg)
    clog = clean_lines(con, load_raw(cfg), cfg)
    snaps = build_all_snapshots(con, cfg)
    write_data_summary(con, clog, snaps, cfg)

    selection_windows = {k: v for k, v in snaps.items() if k != "test"}
    seg = select_segmentation(selection_windows, cfg)
    save_selection(seg, cfg)
    prop = select_propensity(selection_windows, cfg)

    ev = evaluate_test(snaps, seg["model"], prop["model"], prop["train"], seg["stability"], cfg)
    context = {
        "segmentation": {
            "feature_set": seg["chosen_set"],
            "k": seg["model"].k,
            "names": seg["model"].names,
            "dev_rates": seg["model"].segment_rates,
        },
        "propensity": {
            "chosen": prop["chosen"],
            "C": prop["C"],
            "cv": prop["cv"].to_dict(orient="records"),
            "val_auc": prop["val_auc"],
            "train_windows": prop["train_windows"],
        },
    }
    write_gate(ev, cfg, context)
    ev["predictions"].to_csv(u.resolve(cfg, "processed_dir") / "test_predictions.csv", index=False)
    log.info("Gate: %s", json.dumps(ev["status"], default=float))
    return ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
