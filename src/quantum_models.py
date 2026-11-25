"""Quantum models built with Qiskit and Qiskit Machine Learning.

* ``QuantumKernelSVC`` : Quantum Support Vector Machine. A fidelity quantum kernel
  k(x, x') = |<phi(x)|phi(x')>|^2 is evaluated exactly with statevector simulation
  (qiskit_machine_learning FidelityStatevectorKernel) and fed to a classical SVM.
* ``VQCClassifier``    : Variational Quantum Classifier = feature map + trainable ansatz,
  trained with a classical optimizer (COBYLA or SPSA) on a cross-entropy loss.

Both are scikit-learn compatible so they plug into Pipeline / cross-validation.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from qiskit.circuit.library import (efficient_su2, pauli_feature_map, real_amplitudes,
                                    z_feature_map, zz_feature_map)
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from . import config

warnings.filterwarnings("ignore", category=DeprecationWarning)

FEATURE_MAPS = ("Z", "ZZ", "Pauli_ZXX", "Pauli_XYY")
ENTANGLEMENTS = ("linear", "circular", "full")
ANSATZE = ("RealAmplitudes", "EfficientSU2")
OPTIMIZERS = ("COBYLA", "SPSA")


# ----------------------------------------------------------------------------- circuits
def make_feature_map(kind: str, n_qubits: int, reps: int = 2, entanglement: str = "linear"):
    """Data-encoding circuit. ``Z`` has no entanglement; the others do."""
    if kind == "Z":
        return z_feature_map(n_qubits, reps=reps)
    if kind == "ZZ":
        return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    if kind == "Pauli_ZXX":
        return pauli_feature_map(n_qubits, reps=reps, entanglement=entanglement,
                                 paulis=["Z", "XX"])
    if kind == "Pauli_XYY":
        return pauli_feature_map(n_qubits, reps=reps, entanglement=entanglement,
                                 paulis=["X", "YY"])
    raise ValueError(f"unknown feature map {kind}")


def make_ansatz(kind: str, n_qubits: int, reps: int = 2, entanglement: str = "linear"):
    if kind == "RealAmplitudes":
        return real_amplitudes(n_qubits, reps=reps, entanglement=entanglement)
    if kind == "EfficientSU2":
        return efficient_su2(n_qubits, reps=reps, entanglement=entanglement)
    raise ValueError(f"unknown ansatz {kind}")


def circuit_stats(qc) -> dict:
    d = qc.decompose().decompose()
    return {"n_qubits": qc.num_qubits, "n_parameters": qc.num_parameters,
            "depth_decomposed": d.depth(), "gate_counts": dict(d.count_ops())}


# ------------------------------------------------------------------------------- kernels
def quantum_kernel_matrix(feature_map, X, Y=None) -> np.ndarray:
    kern = FidelityStatevectorKernel(feature_map=feature_map, enforce_psd=False)
    return kern.evaluate(X, Y) if Y is not None else kern.evaluate(X)


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment in [-1, 1]; higher = kernel separates classes better.

    Cheap proxy used to screen circuit architectures before any SVM is trained.
    """
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    yy = np.where(np.asarray(y) == 1, 1.0, -1.0)
    Y = np.outer(yy, yy)
    Yc = H @ Y @ H
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-12))


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Quantum SVM: quantum fidelity kernel + classical SVM (precomputed kernel)."""

    def __init__(self, feature_map="ZZ", reps=2, entanglement="linear", bandwidth=1.0, C=1.0,
                 class_weight="balanced"):
        self.feature_map = feature_map
        self.reps = reps
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.C = C
        self.class_weight = class_weight

    def _circuit(self, n):
        return make_feature_map(self.feature_map, n, self.reps, self.entanglement)

    def _K(self, X, Y=None):
        return quantum_kernel_matrix(self._qc_, self.bandwidth * X,
                                     None if Y is None else self.bandwidth * Y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.array([0, 1])
        self._qc_ = self._circuit(X.shape[1])
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight=self.class_weight)
        self.svc_.fit(self._K(X), y)
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self._K(np.asarray(X, dtype=float), self.X_train_))

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class HybridKernelSVC(BaseEstimator, ClassifierMixin):
    """Hybrid quantum-classical kernel SVM.

        K(x, x') = alpha * K_quantum(x, x') + (1 - alpha) * K_rbf(x, x')

    A convex combination of two valid kernels is a valid kernel. alpha=1 is a pure QSVM,
    alpha=0 is a classical RBF SVM; the optimizer finds the mixture that works best.
    """

    def __init__(self, feature_map="ZZ", reps=2, entanglement="linear", bandwidth=1.0,
                 alpha=0.5, gamma=0.5, C=1.0, class_weight="balanced"):
        self.feature_map = feature_map
        self.reps = reps
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.gamma = gamma
        self.C = C
        self.class_weight = class_weight

    def _K(self, X, Y=None):
        Kq = quantum_kernel_matrix(self._qc_, self.bandwidth * X,
                                   None if Y is None else self.bandwidth * Y)
        Kc = rbf_kernel(X, X if Y is None else Y, gamma=self.gamma)
        return self.alpha * Kq + (1.0 - self.alpha) * Kc

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.array([0, 1])
        self._qc_ = make_feature_map(self.feature_map, X.shape[1], self.reps, self.entanglement)
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight=self.class_weight)
        self.svc_.fit(self._K(X), y)
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self._K(np.asarray(X, dtype=float), self.X_train_))

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# --------------------------------------------------------------------------------- VQC
def _parity(bitstring_int: int) -> int:
    return bin(bitstring_int).count("1") % 2


class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational quantum classifier (feature map + ansatz + classical optimizer)."""

    def __init__(self, feature_map="ZZ", fm_reps=1, fm_entanglement="linear",
                 ansatz="RealAmplitudes", ansatz_reps=2, ansatz_entanglement="linear",
                 optimizer="COBYLA", maxiter=60, learning_rate=0.1, perturbation=0.1,
                 seed=config.SEED):
        self.feature_map = feature_map
        self.fm_reps = fm_reps
        self.fm_entanglement = fm_entanglement
        self.ansatz = ansatz
        self.ansatz_reps = ansatz_reps
        self.ansatz_entanglement = ansatz_entanglement
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.seed = seed

    def _optimizer(self):
        if self.optimizer == "COBYLA":
            return COBYLA(maxiter=self.maxiter)
        if self.optimizer == "SPSA":
            return SPSA(maxiter=self.maxiter, learning_rate=self.learning_rate,
                        perturbation=self.perturbation)
        raise ValueError(self.optimizer)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.loss_history_: list[float] = []
        rng = np.random.default_rng(self.seed)
        ansatz = make_ansatz(self.ansatz, n, self.ansatz_reps, self.ansatz_entanglement)
        init = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        self.vqc_ = VQC(feature_map=make_feature_map(self.feature_map, n, self.fm_reps,
                                                     self.fm_entanglement),
                        ansatz=ansatz, optimizer=self._optimizer(), initial_point=init,
                        interpret=_parity, output_shape=2,
                        callback=lambda w, loss: self.loss_history_.append(float(loss)))
        t0 = time.perf_counter()
        self.vqc_.fit(X, np.asarray(y))
        self.fit_seconds_ = time.perf_counter() - t0
        self.weights_ = np.asarray(self.vqc_.weights, dtype=float).copy()
        self.n_features_ = n
        return self

    def _qnn(self):
        """Rebuild the circuit network from config so the model can be pickled as plain
        data (weights + hyper-parameters) and re-used without the training objects."""
        from qiskit_machine_learning.neural_networks import SamplerQNN
        n = self.n_features_
        fm = make_feature_map(self.feature_map, n, self.fm_reps, self.fm_entanglement)
        an = make_ansatz(self.ansatz, n, self.ansatz_reps, self.ansatz_entanglement)
        circuit = fm.compose(an)
        return SamplerQNN(circuit=circuit, input_params=fm.parameters, weight_params=an.parameters,
                          interpret=_parity, output_shape=2)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("vqc_", None)          # training object holds un-picklable primitives
        state.pop("_qnn_cache", None)
        return state

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if getattr(self, "_qnn_cache", None) is None:
            self._qnn_cache = self._qnn()
        return np.asarray(self._qnn_cache.forward(X, self.weights_))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
