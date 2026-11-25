"""Metrics, timing and threshold selection."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def scores_of(model, X) -> np.ndarray:
    """Continuous score for the positive class (higher = more likely above limits)."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    p = model.predict_proba(X)
    return np.asarray(p[:, 1], dtype=float)


def metrics_from(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = (int(v) for v in cm.ravel())
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def best_threshold(y_true, y_score, metric: str = "macro_f1") -> float:
    """Pick the decision threshold on VALIDATION data (never on the test set)."""
    grid = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best, best_val = grid[len(grid) // 2], -1.0
    for t in grid:
        pred = (y_score >= t).astype(int)
        val = f1_score(y_true, pred, average="macro", zero_division=0) if metric == "macro_f1" \
            else balanced_accuracy_score(y_true, pred)
        if val > best_val:
            best, best_val = float(t), float(val)
    return best


def default_threshold(model) -> float:
    """0.5 for probability scores, 0 for margin scores."""
    return 0.0 if hasattr(model, "decision_function") else 0.5


def evaluate(model, X, y, threshold: float | None = None) -> dict:
    """Predict + score with timing. ``threshold`` is on the continuous score."""
    t0 = time.perf_counter()
    s = scores_of(model, X)
    pred_time = time.perf_counter() - t0
    thr = default_threshold(model) if threshold is None else threshold
    pred = (s >= thr).astype(int)
    m = metrics_from(y, pred, s)
    m["predict_time_s"] = pred_time
    m["threshold"] = float(thr)
    return m


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
