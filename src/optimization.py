"""Staged optimisation of the hybrid quantum-classical framework.

Stage 0  Tuned classical baselines (Optuna TPE, 5-fold CV on the training split).
Stage 1  Architecture screening with kernel-target alignment (KTA). Random candidates
         over feature optimisation (encoding view, reduction, #qubits) and circuit design
         (feature map, repetitions, entanglement, bandwidth) are scored by KTA, a label-aware
         but training-free proxy, and only the top-k are kept. No SVM is trained here.
Stage 2  Optuna TPE over the shortlisted architectures plus kernel hyper-parameters
         (bandwidth, C, mixing weight alpha, RBF gamma), scored by repeated stratified CV.
Stage 3  VQC search: ansatz, depth, optimizer, learning-rate / perturbation.

The objective for every model family is the mean cross-validated ROC-AUC, which does not
depend on the class ratio of the training data, so classical models trained on the full
imbalanced training set and quantum models trained on a balanced subsample are scored on
the same scale. The test set is never touched here.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config, classical_models as C
from .evaluation import scores_of
from .feature_selection import make_reducer_steps
from .hybrid_model import BASIC_CONFIGS, build_pipeline
from .preprocessing import make_feature_builder, xy
from .quantum_models import (ENTANGLEMENTS, FEATURE_MAPS, kernel_target_alignment,
                             make_feature_map, quantum_kernel_matrix)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _exp_dir():
    """Experiment output folder (override with PDPQML_EXP_DIR, used by the tests)."""
    import os
    from pathlib import Path
    d = Path(os.environ.get("PDPQML_EXP_DIR", config.ROOT / "experiments"))
    d.mkdir(parents=True, exist_ok=True)
    return d
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------ CV helper
def cv_auc(pipe, df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 1,
           seed: int = config.SEED) -> tuple[float, float]:
    """Mean/std ROC-AUC over repeated stratified folds. Fit failures count as AUC 0.5."""
    X, y = xy(df)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = []
    for tr, va in rskf.split(X, y):
        try:
            m = clone(pipe)
            m.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[va], scores_of(m, X.iloc[va])))
        except Exception:  # an invalid configuration must not stop the search
            aucs.append(0.5)
    return float(np.mean(aucs)), float(np.std(aucs))


def _dump_study(study: optuna.Study, name: str, extra: dict | None = None) -> dict:
    exp = _exp_dir()
    df = study.trials_dataframe()
    df.to_csv(exp / f"{name}_trials.csv", index=False)
    out = {"name": name, "n_trials": len(study.trials), "best_value_cv_auc": study.best_value,
           "best_params": study.best_params, **(extra or {})}
    config.save_json(out, exp / f"{name}_best.json")
    return out


# --------------------------------------------------------------------- Stage 0: classical
def tune_classical(name: str, train_df: pd.DataFrame, n_trials: int,
                   seed: int = config.SEED) -> dict:
    X, y = xy(train_df)
    pw = float((y == 0).sum() / (y == 1).sum())

    def objective(trial):
        pipe = C.build_pipeline(name, C.suggest_params(name, trial), "onehot", pw, seed)
        return cv_auc(pipe, train_df, n_splits=config.CFG["split"]["cv_folds"], seed=seed)[0]

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    default_cv = cv_auc(C.build_pipeline(name, None, "onehot", pw, seed), train_df,
                        n_splits=config.CFG["split"]["cv_folds"], seed=seed)[0]
    study.optimize(objective, n_trials=n_trials)
    return _dump_study(study, f"classical_{name}",
                       {"search_seconds": time.perf_counter() - t0, "pos_weight": pw,
                        "default_cv_auc": default_cv})


def tune_classical_compact(name: str, sub_df: pd.DataFrame, arch: dict, n_trials: int,
                           seed: int = config.SEED) -> dict:
    """Tune a classical model on EXACTLY the reduced inputs + training subsample the quantum
    model uses. Provides the matched control: does the quantum circuit add anything?"""
    def objective(trial):
        pipe = C.build_compact_pipeline(name, C.suggest_params(name, trial), arch["n_qubits"],
                                        arch["reduction"], arch["view"], arch.get("k_select"),
                                        1.0, seed)
        return cv_auc(pipe, sub_df, n_splits=3, n_repeats=2, seed=seed)[0]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return _dump_study(study, f"classical_matched_{name}", {"arch": arch})


# ------------------------------------------------------------- Stage 1: KTA screening
def sample_architecture(rng: np.random.Generator) -> dict:
    n_q = int(rng.choice([3, 4, 5, 6]))
    reduction = str(rng.choice(["pca", "kbest", "kbest_pca"]))
    view = str(rng.choice(["onehot", "target"]))
    fm = str(rng.choice(FEATURE_MAPS))
    arch = {"view": view, "reduction": reduction, "n_qubits": n_q,
            "k_select": int(rng.integers(n_q, n_q + 7)) if reduction == "kbest_pca" else None,
            "feature_map": fm, "reps": int(rng.integers(1, 4)),
            "entanglement": "linear" if fm == "Z" else str(rng.choice(ENTANGLEMENTS)),
            "bandwidth": float(np.exp(rng.uniform(np.log(0.25), np.log(2.5))))}
    return arch


def transform_features(arch: dict, df: pd.DataFrame, seed: int = config.SEED):
    """Fit the feature chain (no model) on df and return transformed training matrix."""
    from sklearn.pipeline import Pipeline
    steps = [("features", make_feature_builder(arch["view"], seed))]
    steps += make_reducer_steps(arch["reduction"], arch["n_qubits"], arch.get("k_select"), seed)
    pipe = Pipeline(steps)
    X, y = xy(df)
    return pipe.fit_transform(X, y), y


def kta_of(arch: dict, df: pd.DataFrame, seed: int = config.SEED) -> float:
    Z, y = transform_features(arch, df, seed)
    fm = make_feature_map(arch["feature_map"], Z.shape[1], arch["reps"], arch["entanglement"])
    K = quantum_kernel_matrix(fm, arch["bandwidth"] * Z)
    return kernel_target_alignment(K, y)


def kta_screen(sub_df: pd.DataFrame, n_candidates: int, top_k: int,
               seed: int = config.SEED) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_candidates):
        arch = sample_architecture(rng)
        t0 = time.perf_counter()
        try:
            arch["kta"] = kta_of(arch, sub_df, seed)
        except Exception as e:  # invalid combination, e.g. fewer features than qubits
            arch["kta"] = float("nan")
            arch["error"] = str(e)[:80]
        arch["seconds"] = time.perf_counter() - t0
        rows.append(arch)
    table = pd.DataFrame(rows).sort_values("kta", ascending=False).reset_index(drop=True)
    table.to_csv(_exp_dir() / "kta_screening.csv", index=False)
    top = table.dropna(subset=["kta"]).head(top_k)
    keys = ["view", "reduction", "n_qubits", "k_select", "feature_map", "reps", "entanglement",
            "bandwidth"]
    archs = []
    for _, r in top.iterrows():
        a = {k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else r[k]) for k in keys}
        a["n_qubits"] = int(a["n_qubits"])
        a["reps"] = int(a["reps"])
        a["k_select"] = None if a["k_select"] is None else int(a["k_select"])
        archs.append(a)
    return archs, table


# ----------------------------------------------------------- Stage 2: kernel models
def tune_kernel_model(model: str, sub_df: pd.DataFrame, archs: list[dict], n_trials: int,
                      seed: int = config.SEED, tag: str = "") -> dict:
    """Optuna search for 'qsvm' or 'hybrid'. The basic configuration is always one of the
    candidate architectures and is enqueued first, so the search can only improve on it."""
    basic = BASIC_CONFIGS[model]
    candidates = [{k: basic.get(k) for k in ["view", "reduction", "n_qubits", "k_select",
                                              "feature_map", "reps", "entanglement"]}] + archs
    cv_cfg = config.CFG["search_budget"]["cv_folds_quantum"]

    def build_cfg(trial):
        arch = candidates[trial.suggest_int("arch_id", 0, len(candidates) - 1)]
        cfg = {"model": model, **arch,
               "bandwidth": trial.suggest_float("bandwidth", 0.1, 3.0, log=True),
               "C": trial.suggest_float("C", 0.01, 100.0, log=True)}
        if model == "hybrid":
            cfg["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
            cfg["gamma"] = trial.suggest_float("gamma", 0.01, 5.0, log=True)
        return cfg

    def objective(trial):
        cfg = build_cfg(trial)
        return cv_auc(build_pipeline(cfg, seed), sub_df, n_splits=cv_cfg, n_repeats=2,
                      seed=seed)[0]

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    first = {"arch_id": 0, "bandwidth": basic["bandwidth"], "C": basic["C"]}
    if model == "hybrid":
        first.update({"alpha": basic["alpha"], "gamma": basic["gamma"]})
    study.enqueue_trial(first)
    study.optimize(objective, n_trials=n_trials)
    best_cfg = build_cfg(optuna.trial.FixedTrial(study.best_params))
    out = _dump_study(study, f"optuna_{model}{tag}",
                      {"search_seconds": time.perf_counter() - t0, "best_config": best_cfg,
                       "basic_cv_auc": study.trials[0].value, "architectures": candidates})
    return out


# ------------------------------------------------------------------ Stage 3: VQC
def tune_vqc(sub_df: pd.DataFrame, archs: list[dict], n_trials: int, maxiter: int,
             seed: int = config.SEED) -> dict:
    basic = BASIC_CONFIGS["vqc"]
    feats = [{k: basic[k] for k in ["view", "reduction", "n_qubits", "k_select"]}]
    for a in archs:
        if a["n_qubits"] <= 5:
            feats.append({k: a[k] for k in ["view", "reduction", "n_qubits", "k_select"]})

    def build_cfg(trial):
        f = feats[trial.suggest_int("feat_id", 0, len(feats) - 1)]
        opt = trial.suggest_categorical("optimizer", ["COBYLA", "SPSA"])
        cfg = {"model": "vqc", **f,
               "feature_map": trial.suggest_categorical("feature_map", ["Z", "ZZ"]),
               "fm_reps": trial.suggest_int("fm_reps", 1, 2), "fm_entanglement": "linear",
               "ansatz": trial.suggest_categorical("ansatz", ["RealAmplitudes", "EfficientSU2"]),
               "ansatz_reps": trial.suggest_int("ansatz_reps", 1, 3),
               "ansatz_entanglement": trial.suggest_categorical("ansatz_entanglement",
                                                                 ["linear", "full"]),
               "optimizer": opt, "maxiter": maxiter}
        if opt == "SPSA":
            cfg["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.6, log=True)
            cfg["perturbation"] = trial.suggest_float("perturbation", 0.05, 0.4)
        return cfg

    def objective(trial):
        cfg = build_cfg(trial)
        return cv_auc(build_pipeline(cfg, seed), sub_df, n_splits=3, n_repeats=1, seed=seed)[0]

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))
    study.enqueue_trial({"feat_id": 0, "optimizer": basic["optimizer"],
                         "feature_map": basic["feature_map"], "fm_reps": basic["fm_reps"],
                         "ansatz": basic["ansatz"], "ansatz_reps": basic["ansatz_reps"],
                         "ansatz_entanglement": basic["ansatz_entanglement"]})
    study.optimize(objective, n_trials=n_trials)
    best_cfg = build_cfg(optuna.trial.FixedTrial(study.best_params))
    return _dump_study(study, "optuna_vqc",
                       {"search_seconds": time.perf_counter() - t0, "best_config": best_cfg,
                        "basic_cv_auc": study.trials[0].value, "maxiter": maxiter})
