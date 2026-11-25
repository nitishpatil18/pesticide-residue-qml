"""Streamlit demo:  streamlit run app.py

Predictions come from models trained in this project on USDA PDP 2022 sample metadata.
The app is a research demonstration of risk screening; it does NOT replace certified
laboratory pesticide residue testing.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config, evaluation as E, preprocessing as P
from src.hybrid_model import build_pipeline, balanced_subsample, config_summary

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Pesticide residue QML", page_icon=None, layout="wide")

MET, FIG = config.path("metrics_dir"), config.path("figures_dir")
MODELS, DIAG = config.path("models_dir"), config.ROOT / "docs" / "diagrams"
PROC = config.path("processed_dir")

DISPLAY = {
    "classical_logreg": "Classical - Logistic Regression (tuned)",
    "classical_svm": "Classical - SVM RBF (tuned)",
    "classical_rf": "Classical - Random Forest (tuned)",
    "classical_xgb": "Classical - XGBoost (tuned)",
    "2_basic_quantum_qsvm": "Quantum - QSVM (basic)",
    "2_basic_quantum_vqc": "Quantum - VQC (basic)",
    "3_basic_hybrid_hybrid": "Hybrid - quantum+RBF kernel (basic)",
    "4_optimized_quantum_qsvm": "Quantum - QSVM (optimized)",
    "4_optimized_quantum_vqc": "Quantum - VQC (optimized)",
    "5_optimized_hybrid_hybrid": "Hybrid - quantum+RBF kernel (optimized)",
}
SCORE_KEY = {"classical_logreg": "1_classical|logreg_tuned", "classical_svm": "1_classical|svm_tuned",
             "classical_rf": "1_classical|rf_tuned", "classical_xgb": "1_classical|xgb_tuned"}

DISCLAIMER = ("Research demonstration only. The model estimates how likely a sample is to be flagged "
              "as above permissible limits from collection metadata (commodity, origin, claim, "
              "facility, state, month). It has not seen any chemical measurement of the sample and "
              "does not replace certified laboratory pesticide residue testing.")


# --------------------------------------------------------------------------------- loaders
@st.cache_data
def splits():
    return P.load_splits()


@st.cache_data
def ref(name):
    return pd.read_csv(PROC / f"ref_{name}.csv", dtype=str, keep_default_na=False)


@st.cache_data
def table():
    f = MET / "comparison_table.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_resource
def load_model(key: str):
    obj = joblib.load(MODELS / f"{key}.joblib")
    parts = splits()
    Xva, yva = P.xy(parts["val"])
    thr = E.best_threshold(yva, E.scores_of(obj["pipeline"], Xva))
    return obj["pipeline"], thr, obj.get("config", {})


def available_models():
    return [k for k in DISPLAY if (MODELS / f"{k}.joblib").exists()]


def show_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{path.name} has not been generated yet (run `python main.py figures`).")


# ----------------------------------------------------------------------------------- pages
def page_intro():
    st.title("Quantum machine learning for pesticide residue classification")
    st.warning(DISCLAIMER)
    st.markdown("""
**Task.** Classify a food sample as **within permissible limits** or **above permissible limits**
(a residue exceeds its EPA tolerance, or a residue is present where no tolerance exists), using the
USDA Pesticide Data Program 2022 database.

**Methods compared.** Logistic Regression, SVM, Random Forest, XGBoost; a Quantum SVM (quantum fidelity
kernel), a Variational Quantum Classifier, and a **hybrid quantum-classical kernel SVM** whose feature
encoding, circuit design and hyper-parameters are tuned jointly (KTA screening + Optuna).

**Honest framing.** Inputs are sample metadata only, so absolute accuracy is modest, and quantum models
are simulated classically on small training sets. Results are reported with the classical controls.
""")
    show_image(DIAG / "system_architecture.png", "System architecture")


def page_dataset():
    st.header("Dataset")
    rep = MET / "data_quality_report.json"
    if rep.exists():
        r = config.load_json(rep)
        c = st.columns(4)
        c[0].metric("Samples", f"{r['n_samples']:,}")
        c[1].metric("Residue results", f"{r['n_result_rows']:,}")
        c[2].metric("Above limits", f"{r['class_counts']['1']:,}", f"{r['positive_rate']:.1%}")
        c[3].metric("Commodities", r["unique_commodities"])
        st.json(r["label_verification"], expanded=False)
    st.markdown("""
**Source:** USDA Agricultural Marketing Service, Pesticide Data Program (PDP) 2022 database,
<https://www.ams.usda.gov/datasets/pdp>. Public-domain US government data. Limits are EPA tolerances
(pesticide- and commodity-specific). **Label:** official PDP annotations X, V, QV (exceeds tolerance /
no tolerance), cross-checked against concentration vs. tolerance. **Inputs:** sample metadata only;
no concentration is used as a feature.
""")
    show_image(FIG / "eda_overview.png")
    st.subheader("Preview of the training split")
    st.dataframe(splits()["train"][P.RAW_FEATURES + ["label"]].head(200), use_container_width=True)


def page_comparison():
    st.header("Classical vs quantum vs hybrid")
    t = table()
    if t is None:
        st.info("Run `python main.py evaluate` first.")
        return
    cols = ["group", "model", "n_train", "n_runs", "def_roc_auc", "def_roc_auc_std", "def_pr_auc",
            "cal_macro_f1", "cal_balanced_accuracy", "cal_precision", "cal_recall", "fit_time_s",
            "predict_ms_per_sample"]
    st.dataframe(t[cols].round(3), use_container_width=True, hide_index=True)
    st.caption("Test set: 2,133 samples, 64 above limits. cal_* = threshold chosen on the validation set. "
               "Quantum rows: mean over training subsamples (5 seeds, VQC 3); std in def_roc_auc_std.")
    show_image(FIG / "model_comparison.png")
    c = st.columns(2)
    with c[0]:
        show_image(FIG / "roc_pr_curves.png")
    with c[1]:
        show_image(FIG / "confusion_matrices.png")
    show_image(FIG / "runtime.png")


def page_optimization():
    st.header("Optimization results")
    exp = config.ROOT / "experiments"
    for name in ("hybrid", "qsvm", "vqc"):
        f = exp / f"optuna_{name}_best.json"
        if f.exists():
            r = config.load_json(f)
            st.subheader(name.upper())
            st.write(f"Best CV ROC-AUC **{r['best_value_cv_auc']:.3f}** vs basic configuration "
                     f"**{r['basic_cv_auc']:.3f}** ({r['n_trials']} trials, {r['search_seconds']:.0f} s)")
            st.code(config_summary(r["best_config"]) if name != "vqc" else str(r["best_config"]))
    show_image(FIG / "optuna_history.png")
    show_image(FIG / "kta_screening.png")
    ab = MET / "ablation_hybrid.csv"
    if ab.exists():
        st.subheader("Ablation study (hybrid model)")
        st.dataframe(pd.read_csv(ab)[["variant", "roc_auc", "roc_auc_std", "cal_macro_f1",
                                      "cal_balanced_accuracy"]].round(3), hide_index=True,
                     use_container_width=True)
        show_image(FIG / "ablation_hybrid.png")
    show_image(FIG / "scaling_curve.png")


def page_circuits():
    st.header("Quantum circuits and architecture")
    for n in ("hybrid_architecture", "quantum_circuit_architecture", "optimization_workflow",
              "circuit_quantum_kernel_feature_map", "circuit_vqc"):
        show_image(DIAG / f"{n}.png")


def page_classify():
    st.header("Classify a sample")
    st.warning(DISCLAIMER)
    keys = available_models()
    if not keys:
        st.info("No trained models found. Run `python main.py classical` and `python main.py evaluate`.")
        return
    default = keys.index("5_optimized_hybrid_hybrid") if "5_optimized_hybrid_hybrid" in keys else 0
    key = st.selectbox("Model", keys, index=default, format_func=lambda k: DISPLAY[k])
    parts = splits()
    allrows = pd.concat(parts.values())
    commod = ref("commodity")
    commod = commod[commod.COMMOD.isin(allrows.COMMOD.unique())]
    country = ref("country").set_index("COUNTRY")["COUNTRY_NAME"].to_dict()
    c = st.columns(3)
    cm = c[0].selectbox("Commodity", commod.COMMOD, format_func=lambda x: commod.set_index("COMMOD").loc[x, "COMMOD_NAME"])
    origin = c[1].selectbox("Origin", ["1", "2", "3"], format_func={"1": "Domestic (US)", "2": "Imported",
                                                                     "3": "Unknown"}.get)
    ctry_opts = ["NONE"] + sorted(set(allrows.COUNTRY) - {"NONE"})
    ctry = c[2].selectbox("Country of origin (imports)", ctry_opts, format_func=lambda x: country.get(x, "n/a") if x != "NONE" else "n/a")
    c = st.columns(3)
    claim = c[0].selectbox("Marketing claim", ["NC", "PO", "PP"],
                           format_func={"NC": "No claim", "PO": "Organic", "PP": "Pesticide free"}.get)
    dist = c[1].selectbox("Collection facility", sorted(allrows.DISTTYPE.unique()),
                          format_func={"D": "Distribution center", "F": "Farmgate", "G": "Grain lot", "H": "Wholesale",
                                       "L": "Wholesale + retail", "R": "Retail", "T": "Terminal market"}.get)
    state = c[2].selectbox("Collection state", sorted(allrows.STATE.unique()))
    c = st.columns(3)
    month = c[0].slider("Month", 1, 12, 6)
    qty = c[1].number_input("Units in sample (0 = unknown)", 0, 500, 0)
    commtype = allrows[allrows.COMMOD == cm].COMMTYPE.mode().iloc[0]
    row = pd.DataFrame([{"COMMOD": cm, "ORIGIN": origin, "DISTTYPE": dist, "COMMTYPE": commtype,
                         "CLAIM": claim, "STATE": state, "COUNTRY": ctry, "GROWST": "NONE",
                         "PACKST": "NONE", "DISTST": "NONE", "QUANTITY": np.nan if qty == 0 else float(qty),
                         "MONTH": month}])
    X, _ = P.xy(P.add_derived(row).assign(label=0))
    if st.button("Predict", type="primary"):
        pipe, thr, cfg = load_model(key)
        s = float(E.scores_of(pipe, X)[0])
        above = s >= thr
        (st.error if above else st.success)(
            f"Predicted: **{'ABOVE permissible limits (flag for lab testing)' if above else 'Within permissible limits'}**")
        scores = config.load_json(MET / "test_scores_seed0.json") if (MET / "test_scores_seed0.json").exists() else {}
        g, m = key.rsplit("_", 1)
        sk = SCORE_KEY.get(key, f"{g}|{m}")
        if sk in scores:
            pct = (np.array(scores[sk]) <= s).mean() * 100
            st.write(f"Risk score {s:.3f}, higher than **{pct:.0f}%** of the 2,133 test samples "
                     f"(decision threshold {thr:.3f}).")
        else:
            st.write(f"Risk score {s:.3f} (decision threshold {thr:.3f}).")
        st.caption("The score ranks risk; it is not a probability. Only about 3% of PDP samples are above "
                   "limits, and the model sees metadata only, so most flagged samples will be false alarms "
                   "(see precision in the comparison page).")
    with st.expander("Train / retrain a model now"):
        st.write("Retrains the selected quantum or hybrid configuration from scratch on a class-balanced "
                 "subsample of the training split (a few seconds for kernel models).")
        if st.button("Retrain selected model") and cfg_of(key):
            cfg = cfg_of(key)
            sub = balanced_subsample(parts["train"], config.CFG["quantum"]["train_size_per_class"], config.SEED)
            Xs, ys = P.xy(sub)
            with st.spinner("Training..."):
                pipe = build_pipeline(cfg).fit(Xs, ys)
            Xv, yv = P.xy(parts["val"])
            Xt, yt = P.xy(parts["test"])
            thr = E.best_threshold(yv, E.scores_of(pipe, Xv))
            m = E.evaluate(pipe, Xt, yt, thr)
            st.json({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


def cfg_of(key):
    try:
        obj = joblib.load(MODELS / f"{key}.joblib")
        cfg = obj.get("config", {})
        return cfg if cfg.get("model") in ("qsvm", "hybrid", "vqc") else None
    except Exception:
        return None


PAGES = {"Introduction": page_intro, "Dataset": page_dataset, "Classify a sample": page_classify,
         "Classical vs quantum vs hybrid": page_comparison, "Optimization": page_optimization,
         "Circuits and architecture": page_circuits}

with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.caption("Data: USDA PDP 2022. " + "Not a substitute for laboratory testing.")
PAGES[choice]()
