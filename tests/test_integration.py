import json

import joblib
import numpy as np
import pytest

from src import config, evaluation as E, preprocessing as P


def test_end_to_end_small(splits):
    """data -> features -> hybrid model -> thresholded prediction on unseen test rows."""
    from src.hybrid_model import BASIC_CONFIGS, balanced_subsample, build_pipeline
    sub = balanced_subsample(splits["train"], 60, 3)
    Xtr, ytr = P.xy(sub)
    Xte, yte = P.xy(splits["test"].head(300))
    pipe = build_pipeline(BASIC_CONFIGS["hybrid"]).fit(Xtr, ytr)
    m = E.evaluate(pipe, Xte, yte)
    assert 0 <= m["balanced_accuracy"] <= 1 and m["tn"] + m["fp"] + m["fn"] + m["tp"] == 300


def test_saved_models_load_and_predict(splits):
    mdir = config.path("models_dir")
    files = sorted(mdir.glob("*.joblib"))
    if not files:
        pytest.skip("models not trained yet (run main.py evaluate)")
    Xte, _ = P.xy(splits["test"].head(20))
    for f in files:
        obj = joblib.load(f)
        s = E.scores_of(obj["pipeline"], Xte)
        assert len(s) == 20 and np.isfinite(s).all(), f.name


def test_results_files_consistent():
    f = config.path("metrics_dir") / "final_results.json"
    if not f.exists():
        pytest.skip("final results not generated yet")
    res = json.load(open(f))
    for k, v in res.items():
        a = v["agg"]
        assert 0 <= a["default_threshold.roc_auc.mean"] <= 1, k
        assert v["runs"][0]["calibrated_threshold"]["tp"] >= 0


def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception


def test_bootstrap_ci_contains_point_estimate():
    f = config.path("metrics_dir") / "bootstrap_ci.json"
    if not f.exists():
        pytest.skip("run main.py stats first")
    res = json.load(open(f))
    for k, v in res["models"].items():
        lo, hi = v["ci95"]
        assert lo <= v["auc"] <= hi, k
