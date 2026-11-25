import numpy as np
import pytest

from src import optimization as O


@pytest.fixture(autouse=True)
def tmp_exp(tmp_path, monkeypatch):
    monkeypatch.setenv("PDPQML_EXP_DIR", str(tmp_path))
    return tmp_path


def test_kta_range_and_ordering():
    from src.quantum_models import kernel_target_alignment
    y = np.array([0, 0, 1, 1])
    perfect = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], float)
    assert kernel_target_alignment(perfect, y) > 0.99
    assert kernel_target_alignment(np.ones((4, 4)), y) < 1e-6


def test_screen_and_tune_tiny(small_train, tmp_exp):
    archs, table = O.kta_screen(small_train, n_candidates=4, top_k=2)
    assert 1 <= len(archs) <= 2 and (tmp_exp / "kta_screening.csv").exists()
    res = O.tune_kernel_model("hybrid", small_train, archs, n_trials=2)
    assert 0.0 <= res["best_value_cv_auc"] <= 1.0
    assert res["best_config"]["model"] == "hybrid"
    assert (tmp_exp / "optuna_hybrid_trials.csv").exists()


def test_tune_classical_tiny(splits, tmp_exp):
    tr = splits["train"].sample(1200, random_state=0)
    res = O.tune_classical("logreg", tr, n_trials=2)
    assert "C" in res["best_params"]
