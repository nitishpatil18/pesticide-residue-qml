"""Experiment stages (called from main.py). Every stage writes its results to disk."""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd

from . import classical_models as C
from . import config, optimization as O, preprocessing as P
from .evaluation import Timer, best_threshold, evaluate, scores_of
from .hybrid_model import (BASIC_CONFIGS, CIRCUIT_KEYS, FEATURE_KEYS, HYPER_KEYS, build_pipeline,
                           balanced_subsample, config_summary, with_defaults_from)

warnings.filterwarnings("ignore")
MET = config.path("metrics_dir")
EXP = config.ROOT / "experiments"
EXP.mkdir(exist_ok=True)
MODELS = config.path("models_dir")
SEEDS = [42, 43, 44, 45, 46]
VQC_SEEDS = [42, 43, 44]
KEYS = ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "f1", "precision", "recall",
        "accuracy"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------- helpers
def fit_and_score(pipe, train_df, val_df, test_df) -> dict:
    """Fit on train_df, pick the threshold on validation, score validation and test."""
    Xtr, ytr = P.xy(train_df)
    Xva, yva = P.xy(val_df)
    Xte, yte = P.xy(test_df)
    with Timer() as t:
        pipe.fit(Xtr, ytr)
    sv = scores_of(pipe, Xva)
    thr = best_threshold(yva, sv)
    out = {"fit_time_s": t.elapsed,
           "default_threshold": evaluate(pipe, Xte, yte),
           "calibrated_threshold": evaluate(pipe, Xte, yte, thr),
           "val_default": evaluate(pipe, Xva, yva)}
    out["predict_ms_per_sample"] = 1000 * out["default_threshold"]["predict_time_s"] / len(yte)
    out["test_scores"] = scores_of(pipe, Xte).tolist()
    return out


def aggregate(runs: list[dict]) -> dict:
    """mean / std across seeds for both operating points."""
    agg = {"n_runs": len(runs)}
    for op in ("default_threshold", "calibrated_threshold"):
        for k in KEYS:
            v = np.array([r[op][k] for r in runs], dtype=float)
            agg[f"{op}.{k}.mean"], agg[f"{op}.{k}.std"] = float(v.mean()), float(v.std())
    for k in ("fit_time_s", "predict_ms_per_sample"):
        v = np.array([r[k] for r in runs], dtype=float)
        agg[f"{k}.mean"], agg[f"{k}.std"] = float(v.mean()), float(v.std())
    return agg


def strip_scores(runs):
    return [{k: v for k, v in r.items() if k != "test_scores"} for r in runs]


def _splits():
    parts = P.load_splits()
    return parts["train"], parts["val"], parts["test"]


def run_config_over_seeds(cfg: dict, train, val, test, seeds, n_per_class, save_name=None):
    runs = []
    for s in seeds:
        sub = balanced_subsample(train, n_per_class, s)
        pipe = build_pipeline(cfg, seed=s)
        r = fit_and_score(pipe, sub, val, test)
        r["seed"] = s
        runs.append(r)
        if save_name and s == seeds[0]:
            try:
                joblib.dump({"pipeline": pipe, "config": cfg, "seed": s,
                             "threshold": best_threshold(P.xy(val)[1], scores_of(pipe, P.xy(val)[0]))},
                            MODELS / f"{save_name}.joblib")
            except Exception as e:  # e.g. an unpicklable simulator object
                log("could not pickle", save_name, e)
    return runs


# ----------------------------------------------------------------------------- stage: data
def stage_data():
    from .data_loader import build_and_save
    build_and_save()
    parts = P.prepare()
    log("data ready", {k: (len(v), int(v.label.sum())) for k, v in parts.items()})


# ------------------------------------------------------------------------ stage: classical
def stage_classical():
    train, val, test = _splits()
    n_trials = config.CFG["search_budget"]["classical_trials"]
    pw = float((train.label == 0).sum() / (train.label == 1).sum())
    results, rows = {}, []
    for name in C.CLASSICAL_NAMES:
        cached = EXP / f"classical_{name}_best.json"
        if cached.exists():                       # resume support: reuse a finished search
            log("reusing finished search", cached.name)
            study = config.load_json(cached)
        else:
            log("tuning classical", name, n_trials, "trials")
            study = O.tune_classical(name, train, n_trials)
        best = study["best_params"]
        for tag, params in (("default", None), ("tuned", best)):
            pipe = C.build_pipeline(name, params, "onehot", pw)
            r = fit_and_score(pipe, train, val, test)
            r["params"] = params or C.DEFAULTS[name]
            results[f"{name}_{tag}"] = r
            if tag == "tuned":
                joblib.dump({"pipeline": pipe, "config": {"model": name, **best}},
                            MODELS / f"classical_{name}.joblib")
            log(name, tag, "test AUC %.3f" % r["default_threshold"]["roc_auc"],
                "cv", round(study["best_value_cv_auc"], 3))
    config.save_json({k: {**{kk: vv for kk, vv in v.items() if kk != "test_scores"}}
                      for k, v in results.items()}, MET / "classical_results.json")
    config.save_json({k: v["test_scores"] for k, v in results.items()}, MET / "classical_test_scores.json")


# -------------------------------------------------------------------------- stage: optimize
def stage_optimize():
    train, val, _ = _splits()
    b = config.CFG["search_budget"]
    nq = config.CFG["quantum"]["train_size_per_class"]
    sub = balanced_subsample(train, nq, config.SEED)
    log("balanced subsample", sub.shape, "positives", int(sub.label.sum()))

    log("stage 1: KTA screening", b["kta_candidates"], "candidates")
    archs, table = O.kta_screen(sub, b["kta_candidates"], b["kta_top_k"])
    config.save_json(archs, EXP / "kta_top_architectures.json")
    log("top KTA", float(table.kta.iloc[0]), [a["feature_map"] for a in archs])

    for model in ("qsvm", "hybrid"):
        log("stage 2:", model, b["qsvm_trials"] if model == "qsvm" else b["hybrid_trials"], "trials")
        n = b["qsvm_trials"] if model == "qsvm" else b["hybrid_trials"]
        r = O.tune_kernel_model(model, sub, archs, n)
        log(model, "best CV AUC", round(r["best_value_cv_auc"], 4), "basic", round(r["basic_cv_auc"], 4))

    log("stage 2b: hybrid without KTA shortlist (ablation of stage 1)")
    O.tune_kernel_model("hybrid", sub, [], b["hybrid_trials"], tag="_no_kta")

    log("stage 3: VQC", b["vqc_trials"], "trials, maxiter", b["vqc_maxiter"])
    r = O.tune_vqc(sub, archs, b["vqc_trials"], b["vqc_maxiter"])
    log("vqc best CV AUC", round(r["best_value_cv_auc"], 4), "basic", round(r["basic_cv_auc"], 4))

    best_h = config.load_json(EXP / "optuna_hybrid_best.json")["best_config"]
    arch = {k: best_h[k] for k in FEATURE_KEYS}
    log("matched classical controls on", arch)
    for name in C.CLASSICAL_NAMES:
        if not (EXP / f"classical_matched_{name}_best.json").exists():
            O.tune_classical_compact(name, sub, arch, 20)


# ---------------------------------------------------------------------------- stage: evaluate
def _load_best(model: str, suffix: str = ""):
    return config.load_json(EXP / f"optuna_{model}{suffix}_best.json")["best_config"]


def stage_evaluate():
    train, val, test = _splits()
    nq = config.CFG["quantum"]["train_size_per_class"]
    table, curves = {}, {}

    cres = config.load_json(MET / "classical_results.json")
    for k, r in cres.items():
        group = "1_classical_default" if k.endswith("_default") else "1_classical_tuned"
        table[f"{group}|{k.rsplit('_', 1)[0]}"] = {"runs": [r], "agg": aggregate([r]),
                                                  "n_train": len(train)}

    cfgs = {
        "2_basic_quantum|qsvm": (BASIC_CONFIGS["qsvm"], SEEDS),
        "2_basic_quantum|vqc": (BASIC_CONFIGS["vqc"], VQC_SEEDS),
        "3_basic_hybrid|hybrid": (BASIC_CONFIGS["hybrid"], SEEDS),
        "4_optimized_quantum|qsvm": (_load_best("qsvm"), SEEDS),
        "4_optimized_quantum|vqc": (_load_best("vqc"), VQC_SEEDS),
        "5_optimized_hybrid|hybrid": (_load_best("hybrid"), SEEDS),
    }
    for key, (cfg, seeds) in cfgs.items():
        log("evaluating", key, config_summary(cfg))
        runs = run_config_over_seeds(cfg, train, val, test, seeds, nq,
                                     save_name=key.replace("|", "_"))
        table[key] = {"runs": strip_scores(runs), "agg": aggregate(runs), "config": cfg,
                      "n_train": 2 * nq}
        curves[key] = runs[0]["test_scores"]
        log(key, "test AUC %.3f +- %.3f" % (table[key]["agg"]["default_threshold.roc_auc.mean"],
                                           table[key]["agg"]["default_threshold.roc_auc.std"]))

    # matched classical controls: same compact inputs, same balanced subsamples
    arch = {k: _load_best("hybrid")[k] for k in FEATURE_KEYS}
    for name in C.CLASSICAL_NAMES:
        best = config.load_json(EXP / f"classical_matched_{name}_best.json")["best_params"]
        runs = []
        for s in SEEDS:
            sub = balanced_subsample(train, nq, s)
            pipe = C.build_compact_pipeline(name, best, arch["n_qubits"], arch["reduction"],
                                            arch["view"], arch.get("k_select"), 1.0, s)
            r = fit_and_score(pipe, sub, val, test)
            r["seed"] = s
            runs.append(r)
        key = f"1b_classical_matched_tuned|{name}"
        table[key] = {"runs": strip_scores(runs), "agg": aggregate(runs), "config": {**arch, **best},
                      "n_train": 2 * nq}
        curves[key] = runs[0]["test_scores"]
        log(key, "test AUC %.3f" % table[key]["agg"]["default_threshold.roc_auc.mean"])

    for k, v in config.load_json(MET / "classical_test_scores.json").items():
        curves[f"1_classical|{k}"] = v
    config.save_json(table, MET / "final_results.json")
    config.save_json(curves, MET / "test_scores_seed0.json")
    config.save_json(test["label"].to_numpy(), MET / "test_labels.json")
    flat = []
    for key, v in table.items():
        group, model = key.split("|")
        a = v["agg"]
        row = {"group": group, "model": model, "n_train": v["n_train"], "n_runs": a["n_runs"],
               "fit_time_s": a["fit_time_s.mean"], "predict_ms_per_sample": a["predict_ms_per_sample.mean"]}
        for op, short in (("default_threshold", "def"), ("calibrated_threshold", "cal")):
            for k in KEYS:
                row[f"{short}_{k}"] = a[f"{op}.{k}.mean"]
                row[f"{short}_{k}_std"] = a[f"{op}.{k}.std"]
        flat.append(row)
    pd.DataFrame(flat).sort_values(["group", "model"]).to_csv(MET / "comparison_table.csv", index=False)
    log("saved comparison table")


# ------------------------------------------------------------------------------ stage: ablation
def stage_ablation():
    train, val, test = _splits()
    nq = config.CFG["quantum"]["train_size_per_class"]
    basic, opt = BASIC_CONFIGS["hybrid"], _load_best("hybrid")
    groups = {"features": FEATURE_KEYS, "circuit": CIRCUIT_KEYS, "hyperparameters": HYPER_KEYS}
    variants = {"A0 basic (nothing optimised)": basic,
                "A1 + feature optimisation only": with_defaults_from(basic, opt, groups["features"]),
                "A2 + circuit optimisation only": with_defaults_from(basic, opt, groups["circuit"]),
                "A3 + hyper-parameter optimisation only": with_defaults_from(basic, opt, groups["hyperparameters"]),
                "A4 features + circuit": with_defaults_from(basic, opt, groups["features"] + groups["circuit"]),
                "A5 features + hyper-parameters": with_defaults_from(basic, opt, groups["features"] + groups["hyperparameters"]),
                "A6 circuit + hyper-parameters": with_defaults_from(basic, opt, groups["circuit"] + groups["hyperparameters"]),
                "A7 all optimised (final hybrid)": opt,
                "A8 final, alpha=1 (pure quantum kernel)": {**opt, "alpha": 1.0},
                "A9 final, alpha=0 (pure classical RBF kernel)": {**opt, "alpha": 0.0}}
    rows = []
    for name, cfg in variants.items():
        log("ablation", name)
        runs = run_config_over_seeds(cfg, train, val, test, SEEDS, nq)
        a = aggregate(runs)
        rows.append({"variant": name, "config": config_summary(cfg),
                     **{f"{k}": a[f"default_threshold.{k}.mean"] for k in KEYS},
                     **{f"{k}_std": a[f"default_threshold.{k}.std"] for k in KEYS},
                     **{f"cal_{k}": a[f"calibrated_threshold.{k}.mean"] for k in ("macro_f1", "balanced_accuracy")}})
    # stage-1 ablation: Optuna without the KTA shortlist (same trial budget)
    no_kta = _load_best("hybrid", "_no_kta")
    runs = run_config_over_seeds(no_kta, train, val, test, SEEDS, nq)
    a = aggregate(runs)
    rows.append({"variant": "B1 Optuna only, no KTA shortlist (same budget)",
                 "config": config_summary(no_kta),
                 **{k: a[f"default_threshold.{k}.mean"] for k in KEYS},
                 **{f"{k}_std": a[f"default_threshold.{k}.std"] for k in KEYS},
                 **{f"cal_{k}": a[f"calibrated_threshold.{k}.mean"] for k in ("macro_f1", "balanced_accuracy")}})
    pd.DataFrame(rows).to_csv(MET / "ablation_hybrid.csv", index=False)
    log("ablation saved")


# ------------------------------------------------------------------------------ stage: scaling
def stage_scaling():
    """Learning curve: does more (balanced) training data help quantum / hybrid / RBF-SVM?"""
    train, val, test = _splits()
    hyb, qs = _load_best("hybrid"), _load_best("qsvm")
    arch = {k: hyb[k] for k in FEATURE_KEYS}
    svm = config.load_json(EXP / "classical_matched_svm_best.json")["best_params"]
    rows = []
    for n in (25, 50, 100, 150, 200):
        for label, builder in (
            ("optimized hybrid", lambda s: build_pipeline(hyb, s)),
            ("optimized qsvm", lambda s: build_pipeline(qs, s)),
            ("matched classical SVM", lambda s: C.build_compact_pipeline(
                "svm", svm, arch["n_qubits"], arch["reduction"], arch["view"], arch.get("k_select"), 1.0, s))):
            aucs = []
            for s in SEEDS:
                sub = balanced_subsample(train, n, s)
                pipe = builder(s)
                Xtr, ytr = P.xy(sub)
                pipe.fit(Xtr, ytr)
                Xte, yte = P.xy(test)
                aucs.append(evaluate(pipe, Xte, yte)["roc_auc"])
            rows.append({"n_per_class": n, "model": label, "auc_mean": float(np.mean(aucs)),
                         "auc_std": float(np.std(aucs))})
            log("scaling", n, label, round(float(np.mean(aucs)), 3))
    pd.DataFrame(rows).to_csv(MET / "scaling_results.csv", index=False)
