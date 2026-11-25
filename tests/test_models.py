import numpy as np
import pytest

from src import classical_models as C, evaluation as E, hybrid_model as H, preprocessing as P
from src import quantum_models as Q


@pytest.mark.parametrize("name", C.CLASSICAL_NAMES)
def test_classical_train_predict(name, splits):
    tr = splits["train"].sample(1500, random_state=1)
    Xtr, ytr = P.xy(tr)
    Xva, yva = P.xy(splits["val"])
    pipe = C.build_pipeline(name, None, "onehot", float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    pipe.fit(Xtr, ytr)
    s = E.scores_of(pipe, Xva)
    assert s.shape == (len(Xva),) and np.isfinite(s).all()
    assert E.metrics_from(yva, (s >= E.default_threshold(pipe)).astype(int), s)["roc_auc"] > 0.55


@pytest.mark.parametrize("kind", Q.FEATURE_MAPS)
def test_quantum_kernel_is_valid(kind):
    """Fidelity kernel: symmetric, unit diagonal, entries in [0, 1], positive semi-definite."""
    X = np.random.default_rng(0).uniform(0, np.pi, (12, 3))
    fm = Q.make_feature_map(kind, 3, 2, "linear")
    K = Q.quantum_kernel_matrix(fm, X)
    assert np.allclose(K, K.T, atol=1e-8)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    assert K.min() >= -1e-9 and K.max() <= 1 + 1e-9
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_circuit_execution_and_stats():
    fm = Q.make_feature_map("ZZ", 4, 2, "full")
    an = Q.make_ansatz("RealAmplitudes", 4, 2, "linear")
    st = Q.circuit_stats(fm)
    assert st["n_qubits"] == 4 and st["depth_decomposed"] > 0
    assert an.num_parameters == 4 * 3


def test_pipeline_qsvm_hybrid_predict(small_train, splits):
    Xtr, ytr = P.xy(small_train)
    Xva, yva = P.xy(splits["val"].head(200))
    for name in ("qsvm", "hybrid"):
        pipe = H.build_pipeline(H.BASIC_CONFIGS[name])
        pipe.fit(Xtr, ytr)
        s = E.scores_of(pipe, Xva)
        assert s.shape == (200,) and np.isfinite(s).all()
        assert set(pipe.predict(Xva)) <= {0, 1}


def test_hybrid_alpha_limits(small_train):
    """alpha=1 reproduces the pure quantum SVM; alpha=0 the classical RBF SVM."""
    Xtr, ytr = P.xy(small_train)
    cfg = dict(H.BASIC_CONFIGS["hybrid"])
    qsvm = H.build_pipeline({**H.BASIC_CONFIGS["qsvm"]}).fit(Xtr, ytr)
    hyb1 = H.build_pipeline({**cfg, "alpha": 1.0}).fit(Xtr, ytr)
    Xq = Xtr.head(30)
    assert np.allclose(E.scores_of(qsvm, Xq), E.scores_of(hyb1, Xq), atol=1e-6)


def test_vqc_trains_tiny(small_train):
    Xtr, ytr = P.xy(small_train.head(30))
    cfg = {**H.BASIC_CONFIGS["vqc"], "maxiter": 3, "n_qubits": 3}
    pipe = H.build_pipeline(cfg).fit(Xtr, ytr)
    proba = pipe.predict_proba(Xtr.head(5))
    assert proba.shape == (5, 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
