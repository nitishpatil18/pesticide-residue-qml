"""Bootstrap confidence intervals and paired AUC differences on the held-out test set.

With only 64 positive test samples a single AUC is imprecise; this module quantifies how
much. Scores are the seed-42 models saved by the evaluate stage. Stratified bootstrap:
positives and negatives are resampled separately so every replicate keeps both classes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config

N_BOOT = 2000

REFERENCE = "1_classical|rf_tuned"
COMPARED = ["1_classical|logreg_tuned", "1_classical|xgb_tuned", "1b_classical_matched_tuned|logreg",
            "3_basic_hybrid|hybrid", "4_optimized_quantum|qsvm", "5_optimized_hybrid|hybrid",
            "2_basic_quantum|vqc", "4_optimized_quantum|vqc"]


def _boot_indices(y, rng, n_boot):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def run(seed: int = config.SEED, n_boot: int = N_BOOT) -> dict:
    met = config.path("metrics_dir")
    y = np.asarray(config.load_json(met / "test_labels.json"))
    scores = config.load_json(met / "test_scores_seed0.json")
    keys = [REFERENCE] + [k for k in COMPARED if k in scores]
    S = {k: np.asarray(scores[k], dtype=float) for k in keys}
    rng = np.random.default_rng(seed)
    aucs = {k: [] for k in keys}
    for idx in _boot_indices(y, rng, n_boot):
        for k in keys:
            aucs[k].append(roc_auc_score(y[idx], S[k][idx]))
    out = {"n_test": int(len(y)), "n_positive": int(y.sum()), "n_boot": n_boot, "models": {}, "vs_reference": {}}
    for k in keys:
        a = np.asarray(aucs[k])
        out["models"][k] = {"auc": float(roc_auc_score(y, S[k])),
                            "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    ref = np.asarray(aucs[REFERENCE])
    for k in keys[1:]:
        d = np.asarray(aucs[k]) - ref
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["vs_reference"][k] = {"auc_diff": float(roc_auc_score(y, S[k]) - roc_auc_score(y, S[REFERENCE])),
                                  "ci95": [float(lo), float(hi)],
                                  "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    out["reference"] = REFERENCE
    config.save_json(out, met / "bootstrap_ci.json")
    return out


if __name__ == "__main__":
    r = run()
    for k, v in r["models"].items():
        print(f"{k:42s} AUC {v['auc']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("paired difference vs", r["reference"])
    for k, v in r["vs_reference"].items():
        print(f"{k:42s} {v['auc_diff']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  sig={v['ci_excludes_zero']}")
