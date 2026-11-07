"""Repeat-purchase propensity: tuning on dev, model choice on val, retraining for the test window.

The test snapshot is not used here. It is scored only by ``gate.evaluate_test``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from segmentation_propensity.evaluation import auc_with_ci, safe_auc
from segmentation_propensity.features import FEATURES
from segmentation_propensity.segmentation import Preprocessor

log = logging.getLogger(__name__)


@dataclass
class PropensityModel:
    """A fitted propensity model with its own preprocessing (fitted on its training rows)."""

    kind: str
    preprocessor: Preprocessor | None
    estimator: Any

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self.preprocessor.transform(df) if self.preprocessor else df[FEATURES].to_numpy(float)
        with threadpool_limits(1):  # single thread: multithreaded histograms are not reproducible
            return self.estimator.predict_proba(x)[:, 1]


def stack(snaps: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """Concatenate snapshots (a customer can appear once per window)."""
    return pd.concat([snaps[n].assign(window=n) for n in names], ignore_index=True)


def fit_logistic(train: pd.DataFrame, C: float, cfg: dict[str, Any]) -> PropensityModel:
    pre = Preprocessor.fit(train, FEATURES, cfg)
    lr = LogisticRegression(C=C, max_iter=cfg["propensity"]["logistic"]["max_iter"])
    return PropensityModel("logistic", pre, lr.fit(pre.transform(train), train.y))


def fit_hgb(train: pd.DataFrame, cfg: dict[str, Any]) -> PropensityModel:
    h = cfg["propensity"]["hist_gradient_boosting"]
    est = HistGradientBoostingClassifier(
        max_depth=h["max_depth"],
        learning_rate=h["learning_rate"],
        max_iter=h["max_iter"],
        early_stopping=h["early_stopping"],
        random_state=cfg["project"]["seed"],
    )
    with threadpool_limits(1):  # single thread: multithreaded histograms are not reproducible
        est.fit(train[FEATURES].to_numpy(float), train.y)
    return PropensityModel("hist_gradient_boosting", None, est)


def tune_logistic_C(dev: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """5-fold stratified CV AUC on dev for each C (preprocessing refitted inside each fold)."""
    p = cfg["propensity"]
    folds = StratifiedKFold(p["cv_folds"], shuffle=True, random_state=cfg["project"]["seed"])
    rows = []
    for C in p["logistic"]["C_grid"]:
        aucs = []
        for tr, te in folds.split(dev, dev.y):
            m = fit_logistic(dev.iloc[tr], C, cfg)
            aucs.append(safe_auc(dev.y.to_numpy()[te], m.predict_proba(dev.iloc[te])))
        rows.append({"C": C, "cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs))})
    return pd.DataFrame(rows)


def select_propensity(snaps: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Tune C on dev, compare logistic and HGB on val (trained on the windows before val), then
    refit the chosen model on all training windows. ``snaps["test"]`` is never accessed."""
    rng = np.random.default_rng(cfg["project"]["seed"])
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    cv = tune_logistic_C(snaps["dev"], cfg)
    best = cv.sort_values(["cv_auc_mean", "C"], ascending=[False, True]).iloc[0]
    C = float(best.C)
    pre_val = [n for n in cfg["propensity"]["train_snapshots"] if n != "val"]
    train_pre_val = stack(snaps, pre_val)
    yv = snaps["val"].y.to_numpy()
    candidates = {
        "logistic": fit_logistic(train_pre_val, C, cfg),
        "hist_gradient_boosting": fit_hgb(train_pre_val, cfg),
    }
    val_auc = {
        k: auc_with_ci(yv, m.predict_proba(snaps["val"]), rng, n_boot)
        for k, m in candidates.items()
    }
    chosen = max(val_auc, key=lambda k: val_auc[k]["value"])
    train_all = stack(snaps, cfg["propensity"]["train_snapshots"])
    final = fit_logistic(train_all, C, cfg) if chosen == "logistic" else fit_hgb(train_all, cfg)
    log.info(
        "Propensity: C=%s, val AUC %s, chosen %s",
        C,
        {k: round(v["value"], 4) for k, v in val_auc.items()},
        chosen,
    )
    return {
        "cv": cv,
        "C": C,
        "val_auc": val_auc,
        "chosen": chosen,
        "model": final,
        "train": train_all,
        "train_windows": cfg["propensity"]["train_snapshots"],
        "pre_val_windows": pre_val,
    }


def vif(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Variance inflation factor of each propensity feature after preprocessing."""
    x = Preprocessor.fit(df, FEATURES, cfg).transform(df)
    out = {}
    for i, f in enumerate(FEATURES):
        others = np.delete(x, i, axis=1)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(x)), others], x[:, i], rcond=None)
        resid = x[:, i] - np.c_[np.ones(len(x)), others] @ beta
        r2 = 1 - resid.var() / x[:, i].var() if x[:, i].var() > 0 else 0.0
        out[f] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return pd.Series(out)
