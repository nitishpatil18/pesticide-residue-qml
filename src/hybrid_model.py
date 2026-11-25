"""Hybrid quantum-classical framework: configuration -> fitted-on-train pipeline.

Architecture (every box is fitted on training data only)

  samples -> feature builder (one-hot or target encoding)
          -> standardise -> feature selection (mutual information) and/or PCA
          -> rotation-angle scaling to [0, pi]
          -> quantum model  (QSVM | Hybrid quantum+RBF kernel SVM | VQC)
          -> score -> threshold -> compliant / above limits

The "framework" part is that ONE configuration dictionary describes the whole chain
(classical feature optimisation + quantum circuit design + model hyper-parameters),
so the optimiser can search all three groups jointly and the ablation study can
switch each group back to its default independently.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config
from .feature_selection import make_reducer_steps
from .preprocessing import make_feature_builder
from .quantum_models import HybridKernelSVC, QuantumKernelSVC, VQCClassifier

# Parameter groups used by the optimiser and by the ablation study.
FEATURE_KEYS = ["view", "reduction", "n_qubits", "k_select"]
CIRCUIT_KEYS = ["feature_map", "reps", "entanglement", "fm_reps", "fm_entanglement", "ansatz",
                "ansatz_reps", "ansatz_entanglement"]
HYPER_KEYS = ["bandwidth", "C", "alpha", "gamma", "optimizer", "maxiter", "learning_rate",
              "perturbation"]

# "Basic" = textbook Qiskit defaults with the architecture from the project brief:
# one-hot -> PCA(4) -> angle encoding -> ZZFeatureMap(reps=2, linear) -> classifier.
BASIC_CONFIGS = {
    "qsvm": {"model": "qsvm", "view": "onehot", "reduction": "pca", "n_qubits": 4,
             "k_select": None, "feature_map": "ZZ", "reps": 2, "entanglement": "linear",
             "bandwidth": 1.0, "C": 1.0},
    "hybrid": {"model": "hybrid", "view": "onehot", "reduction": "pca", "n_qubits": 4,
               "k_select": None, "feature_map": "ZZ", "reps": 2, "entanglement": "linear",
               "bandwidth": 1.0, "C": 1.0, "alpha": 0.5, "gamma": 0.25},
    "vqc": {"model": "vqc", "view": "onehot", "reduction": "pca", "n_qubits": 4,
            "k_select": None, "feature_map": "ZZ", "fm_reps": 1, "fm_entanglement": "linear",
            "ansatz": "RealAmplitudes", "ansatz_reps": 2, "ansatz_entanglement": "linear",
            "optimizer": "COBYLA", "maxiter": 60, "learning_rate": 0.1, "perturbation": 0.1},
}


def make_quantum_estimator(cfg: dict):
    m = cfg["model"]
    if m == "qsvm":
        return QuantumKernelSVC(feature_map=cfg["feature_map"], reps=cfg["reps"],
                                entanglement=cfg["entanglement"], bandwidth=cfg["bandwidth"],
                                C=cfg["C"])
    if m == "hybrid":
        return HybridKernelSVC(feature_map=cfg["feature_map"], reps=cfg["reps"],
                               entanglement=cfg["entanglement"], bandwidth=cfg["bandwidth"],
                               alpha=cfg["alpha"], gamma=cfg["gamma"], C=cfg["C"])
    if m == "vqc":
        return VQCClassifier(feature_map=cfg["feature_map"], fm_reps=cfg["fm_reps"],
                             fm_entanglement=cfg["fm_entanglement"], ansatz=cfg["ansatz"],
                             ansatz_reps=cfg["ansatz_reps"],
                             ansatz_entanglement=cfg["ansatz_entanglement"],
                             optimizer=cfg["optimizer"], maxiter=cfg["maxiter"],
                             learning_rate=cfg.get("learning_rate", 0.1),
                             perturbation=cfg.get("perturbation", 0.1), seed=config.SEED)
    raise ValueError(m)


def build_pipeline(cfg: dict, seed: int = config.SEED) -> Pipeline:
    steps = [("features", make_feature_builder(cfg["view"], seed))]
    steps += make_reducer_steps(cfg["reduction"], cfg["n_qubits"], cfg.get("k_select"), seed)
    steps.append(("model", make_quantum_estimator(cfg)))
    return Pipeline(steps)


def balanced_subsample(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Class-balanced random subsample of the TRAINING split (simulators scale ~O(n^2))."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        idx = rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)
        parts.append(sub.iloc[idx])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def with_defaults_from(base: dict, other: dict, keys: list[str]) -> dict:
    """Copy ``keys`` from ``other`` into a copy of ``base`` (used by the ablation study)."""
    out = copy.deepcopy(base)
    for k in keys:
        if k in other:
            out[k] = other[k]
    return out


def config_summary(cfg: dict) -> str:
    keys = ["model", "view", "reduction", "n_qubits", "feature_map", "reps", "entanglement",
            "bandwidth", "C", "alpha", "gamma"]
    return ", ".join(f"{k}={cfg[k]}" for k in keys if k in cfg and cfg[k] is not None)
