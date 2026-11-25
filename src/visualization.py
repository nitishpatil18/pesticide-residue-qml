"""Result figures and architecture diagrams (Matplotlib / Seaborn)."""
from __future__ import annotations

import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config

FIG = config.path("figures_dir")
DIAG = config.ROOT / "docs" / "diagrams"
DIAG.mkdir(parents=True, exist_ok=True)
MET = config.path("metrics_dir")
EXP = config.ROOT / "experiments"

INK, GREY, SURFACE = "#191919", "#666666", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
GROUP_COLORS = {"1_classical": BLUE, "1b_classical_matched": AQUA, "2_basic_quantum": ORANGE,
                "3_basic_hybrid": MAGENTA, "4_optimized_quantum": VIOLET,
                "5_optimized_hybrid": GREEN}
GROUP_LABELS = {"1_classical": "Classical (full data)",
                "1b_classical_matched": "Classical (matched inputs)",
                "2_basic_quantum": "Basic quantum", "3_basic_hybrid": "Basic hybrid",
                "4_optimized_quantum": "Optimized quantum", "5_optimized_hybrid": "Optimized hybrid"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 160,
                     "savefig.bbox": "tight"})


def _save(fig, name, folder=FIG):
    fig.savefig(folder / name)
    plt.close(fig)
    print("saved", folder / name, flush=True)


def _load_table() -> pd.DataFrame:
    return pd.read_csv(MET / "comparison_table.csv")


def _group_key(g: str) -> str:
    return "1_classical" if g.startswith("1_classical") else g


# ---------------------------------------------------------------------------------- EDA
def fig_eda():
    df = pd.read_csv(config.path("processed_dir") / "samples_labeled.csv", dtype=str,
                     keep_default_na=False)
    df["label"] = df["label"].astype(int)
    names = pd.read_csv(config.path("processed_dir") / "ref_commodity.csv", dtype=str)
    names["COMMOD_NAME"] = names["COMMOD_NAME"].str.replace(", Fresh", "", regex=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [0.7, 2.2, 1]})
    counts = df["label"].value_counts().sort_index()
    ax[0].bar(["Within limits", "Above limits"], counts.values, color=[BLUE, ORANGE], width=0.6)
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax[0].set_ylim(0, counts.max() * 1.18)
    ax[0].set_title("Class balance (10,665 samples)")
    r = df.groupby("COMMOD")["label"].agg(["size", "mean"]).reset_index().merge(names, on="COMMOD")
    r = r.sort_values("mean", ascending=False)
    ax[1].bar(r["COMMOD_NAME"], r["mean"] * 100, color=BLUE, width=0.7)
    ax[1].set_xticks(range(len(r)))
    ax[1].set_xticklabels(r["COMMOD_NAME"], rotation=60, ha="right")
    ax[1].set_ylabel("% of samples above limits")
    ax[1].set_title("Violation rate by commodity (limits are commodity-specific)")
    o = df.groupby("ORIGIN")["label"].mean() * 100
    lab = {"1": "Domestic", "2": "Imported", "3": "Unknown"}
    ax[2].bar([lab[i] for i in o.index], o.values, color=[BLUE, ORANGE, GREY], width=0.6)
    ax[2].set_title("Violation rate by origin")
    ax[2].set_ylabel("% above limits")
    _save(fig, "eda_overview.png")


# ------------------------------------------------------------------------- model results
def _entries():
    scores = config.load_json(MET / "test_scores_seed0.json")
    y = np.array(config.load_json(MET / "test_labels.json"))
    return scores, y


def _pick_models(tab: pd.DataFrame):
    """One representative per group: best (by mean test ROC-AUC) classical, plus quantum groups."""
    picks = {}
    for g, sub in tab.groupby("group"):
        picks[g] = sub.sort_values("def_roc_auc", ascending=False).iloc[0]["model"]
    return picks


def fig_roc_pr():
    scores, y = _entries()
    tab = _load_table()
    picks = _pick_models(tab)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for g, m in picks.items():
        key = f"{g}|{m}" if f"{g}|{m}" in scores else f"1_classical|{m}_tuned"
        if key not in scores:
            continue
        s = np.array(scores[key])
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        lbl = f"{GROUP_LABELS[_group_key(g) if g != '1b_classical_matched_tuned' else '1b_classical_matched']} - {m}"
        col = GROUP_COLORS.get(_group_key(g), GREY)
        ax[0].plot(fpr, tpr, color=col, lw=2, label=lbl)
        ax[1].plot(rc, pr, color=col, lw=2, label=lbl)
    ax[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curves (test set, seed-42 models)")
    ax[1].axhline(y.mean(), color=GREY, lw=1, ls="--")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-recall curves (dashed = prevalence 3%)")
    ax[0].legend(fontsize=8, loc="lower right")
    _save(fig, "roc_pr_curves.png")


def fig_comparison():
    tab = _load_table()
    tab["label"] = tab["group"].map(lambda g: GROUP_LABELS[_group_key(g) if not g.startswith("1b") else "1b_classical_matched"]
                                    if not g.startswith("1_classical_") else "Classical (" + g.split("_", 2)[2] + ")")
    tab["name"] = tab["label"] + " | " + tab["model"]
    tab = tab.sort_values(["group", "model"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    for a, (col, title) in zip(ax, [("def_roc_auc", "ROC-AUC (threshold-free)"),
                                    ("cal_macro_f1", "Macro-F1 (validation-calibrated threshold)"),
                                    ("cal_balanced_accuracy", "Balanced accuracy (calibrated)")]):
        colors = [GROUP_COLORS.get(_group_key(g) if not g.startswith("1b") else "1b_classical_matched", GREY)
                  for g in tab["group"]]
        a.barh(tab["name"], tab[col], xerr=tab[col + "_std"], color=colors, height=0.7,
               error_kw={"ecolor": GREY, "lw": 1})
        a.set_title(title, fontsize=10)
        a.set_xlim(0.4 if "auc" in col or "bal" in col else 0.3, max(tab[col].max() * 1.08, 0.8))
        a.invert_yaxis()
    fig.suptitle("Test-set comparison of all model groups (mean +- std over training subsamples)",
                 y=1.0)
    _save(fig, "model_comparison.png")


def fig_confusion():
    import json
    res = config.load_json(MET / "final_results.json")
    want = [("1_classical_tuned|xgb", "Classical XGBoost (tuned)"),
            ("4_optimized_quantum|qsvm", "Optimized QSVM"),
            ("4_optimized_quantum|vqc", "Optimized VQC"),
            ("5_optimized_hybrid|hybrid", "Optimized hybrid")]
    want = [(k, t) for k, t in want if k in res]
    fig, ax = plt.subplots(1, len(want), figsize=(4.2 * len(want), 4))
    ax = np.atleast_1d(ax)
    for a, (k, t) in zip(ax, want):
        r = res[k]["runs"][0]["calibrated_threshold"]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=["Within", "Above"], yticklabels=["Within", "Above"])
        a.set_title(t, fontsize=10)
        a.set_xlabel("Predicted")
        a.set_ylabel("Actual")
        a.grid(False)
    fig.suptitle("Confusion matrices on the test set (2,133 samples, 64 above limits; threshold from validation)", y=1.03)
    _save(fig, "confusion_matrices.png")


def fig_ablation():
    ab = pd.read_csv(MET / "ablation_hybrid.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = [GREEN if v.startswith("A7") else (VIOLET if v.startswith("B1") else BLUE) for v in ab["variant"]]
    for a, (col, title) in zip(ax, [("roc_auc", "ROC-AUC"), ("cal_macro_f1", "Macro-F1 (calibrated threshold)")]):
        err = ab[col + "_std"] if col + "_std" in ab else None
        a.barh(ab["variant"], ab[col], xerr=err, color=colors, height=0.65, error_kw={"ecolor": GREY})
        a.set_title(title)
        a.set_xlim(0.4 if col == "roc_auc" else 0.3, ab[col].max() * 1.06)
        a.invert_yaxis()
    fig.suptitle("Ablation of the hybrid framework (test set, mean over 5 training subsamples)", y=1.0)
    _save(fig, "ablation_hybrid.png")


def fig_optuna():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (name, title, col) in zip(ax, [("optuna_qsvm", "QSVM", VIOLET), ("optuna_hybrid", "Hybrid", GREEN),
                                          ("optuna_vqc", "VQC", ORANGE)]):
        try:
            t = pd.read_csv(EXP / f"{name}_trials.csv")
        except FileNotFoundError:
            continue
        best = t["value"].cummax()
        a.scatter(t["number"], t["value"], color=col, s=26, alpha=0.7, label="trial")
        a.plot(t["number"], best, color=INK, lw=1.8, label="best so far")
        a.axhline(t["value"].iloc[0], color=GREY, ls="--", lw=1, label="basic configuration")
        a.set_title(f"{title}: Optuna search history")
        a.set_xlabel("trial")
        a.set_ylabel("cross-validated ROC-AUC")
        a.legend(fontsize=8)
    _save(fig, "optuna_history.png")


def fig_kta():
    t = pd.read_csv(EXP / "kta_screening.csv").dropna(subset=["kta"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    pal = {"Z": BLUE, "ZZ": ORANGE, "Pauli_ZXX": GREEN, "Pauli_XYY": VIOLET}
    for fm, s in t.groupby("feature_map"):
        ax[0].scatter(s["bandwidth"], s["kta"], color=pal.get(fm, GREY), label=fm, s=34, alpha=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("bandwidth (data scaling)")
    ax[0].set_ylabel("kernel-target alignment")
    ax[0].set_title("Stage 1: KTA of 60 random architectures")
    ax[0].legend(title="feature map", fontsize=8)
    t2 = t.groupby("n_qubits")["kta"].agg(["mean", "max"]).reset_index()
    ax[1].bar(t2["n_qubits"].astype(str), t2["max"], color=BLUE, width=0.55, label="best")
    ax[1].plot(t2["n_qubits"].astype(str), t2["mean"], color=INK, marker="o", label="mean")
    ax[1].set_xlabel("number of qubits")
    ax[1].set_ylabel("kernel-target alignment")
    ax[1].set_title("KTA vs. number of qubits")
    ax[1].legend(fontsize=8)
    _save(fig, "kta_screening.png")


def fig_scaling():
    s = pd.read_csv(MET / "scaling_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = {"optimized hybrid": GREEN, "optimized qsvm": VIOLET, "matched classical SVM": BLUE}
    for m, g in s.groupby("model"):
        ax.errorbar(2 * g["n_per_class"], g["auc_mean"], yerr=g["auc_std"], marker="o", color=pal[m],
                    lw=2, capsize=3, label=m)
    ax.set_xlabel("training samples (balanced)")
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Learning curve")
    ax.legend()
    _save(fig, "scaling_curve.png")


def fig_runtime():
    tab = _load_table()
    tab["name"] = tab["group"].str.replace("_", " ", n=1).str.replace("_", " ") + " | " + tab["model"]
    tab = tab.sort_values("fit_time_s")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax[0].barh(tab["name"], tab["fit_time_s"], color=BLUE, height=0.7)
    ax[0].set_xscale("log")
    ax[0].set_title("Training time (s, log scale)")
    ax[1].barh(tab["name"], tab["predict_ms_per_sample"], color=ORANGE, height=0.7)
    ax[1].set_xscale("log")
    ax[1].set_title("Prediction time (ms / sample, log scale)")
    _save(fig, "runtime.png")


def fig_circuits():
    from .hybrid_model import BASIC_CONFIGS
    from .quantum_models import make_ansatz, make_feature_map
    try:
        h = config.load_json(EXP / "optuna_hybrid_best.json")["best_config"]
    except FileNotFoundError:
        h = BASIC_CONFIGS["hybrid"]
    v = BASIC_CONFIGS["vqc"]
    try:
        v = config.load_json(EXP / "optuna_vqc_best.json")["best_config"]
    except FileNotFoundError:
        pass
    n = min(h["n_qubits"], 4)
    fm = make_feature_map(h["feature_map"], n, min(h["reps"], 2), h["entanglement"])
    fm_v = make_feature_map(v["feature_map"], v["n_qubits"], v["fm_reps"], v["fm_entanglement"])
    an = make_ansatz(v["ansatz"], v["n_qubits"], v["ansatz_reps"], v["ansatz_entanglement"])
    vqc = fm_v.compose(an)
    for qc, name, ttl in [(fm, "circuit_quantum_kernel_feature_map.png",
                           f"Feature map of the optimized quantum kernel ({h['feature_map']}, {n} qubits shown, reps={min(h['reps'], 2)}, {h['entanglement']})"),
                          (vqc, "circuit_vqc.png",
                           f"VQC = {v['feature_map']} feature map + {v['ansatz']} ansatz ({v['n_qubits']} qubits)")]:
        fig = qc.decompose().draw("mpl", fold=40, style={"backgroundcolor": SURFACE})
        fig.suptitle(ttl, fontsize=10)
        fig.savefig(DIAG / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", DIAG / name, flush=True)


# --------------------------------------------------------------------------- diagrams
def _flow(name, title, nodes, edges, size=(14, 6), notes=None):
    fig, ax = plt.subplots(figsize=size)
    ymax = max(v[1] + v[3] / 2 for v in nodes.values()) + 6
    ymin = min(v[1] - v[3] / 2 for v in nodes.values()) - 4
    ax.set_xlim(0, 100)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    ax.grid(False)
    pos = {}
    for k, (x, y, w, h, txt, col) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=col, ec=INK, lw=1.0, alpha=0.95))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, color=INK, wrap=True)
        pos[k] = (x, y, w, h)
    for a, b, *lab in edges:
        xa, ya, wa, ha = pos[a]
        xb, yb, wb, hb = pos[b]
        dx, dy = xb - xa, yb - ya
        if abs(dx) >= abs(dy) * 0.8:           # mostly horizontal: leave from the facing sides
            sgn = 1 if dx > 0 else -1
            s, e = (xa + sgn * (wa / 2 + 0.6), ya), (xb - sgn * (wb / 2 + 0.6), yb)
        else:                                   # mostly vertical: leave from top / bottom
            sgn = 1 if dy > 0 else -1
            s, e = (xa, ya + sgn * (ha / 2 + 0.6)), (xb, yb - sgn * (hb / 2 + 0.6))
        ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.2))
        if lab:
            ax.text((s[0] + e[0]) / 2, (s[1] + e[1]) / 2 + 1.6, lab[0], ha="center", fontsize=7.5, color=GREY)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if notes:
        ax.text(0, ymin + 1, notes, fontsize=8, color=GREY)
    _save(fig, name, DIAG)


C_DATA, C_PRE, C_Q, C_C, C_H, C_OUT = "#dbe9fb", "#e5f6ef", "#e8e2fb", "#fdebd8", "#d7efd9", "#fde3e3"


def diagrams():
    _flow("system_architecture.png", "Overall system architecture",
          {"d": (10, 72, 16, 14, "USDA PDP 2022\nsamples (10,665)\n+ results (2.89M)\n+ EPA tolerances", C_DATA),
           "l": (29, 72, 16, 14, "Label builder\nX / V / QV annotation\n= above limits\n(verified vs. tolerance)", C_DATA),
           "p": (48, 72, 16, 14, "Preprocessing\nclean, encode, split\nfit on train only", C_PRE),
           "f": (67, 72, 16, 14, "Feature optimisation\nselection / PCA\nangle scaling", C_PRE),
           "m": (86, 72, 16, 14, "Quantum & hybrid\nQSVM - hybrid kernel\n- VQC", C_Q),
           "o": (86, 98, 16, 12, "Optuna + KTA\n(staged search)", C_H),
           "c": (48, 42, 16, 14, "Classical baselines\nLR - SVM - RF - XGBoost\n(tuned with Optuna)", C_C),
           "e": (77, 42, 16, 14, "Evaluation\nmetrics, ablation,\nstatistics", C_OUT),
           "a": (77, 14, 16, 12, "Streamlit app\n+ saved models", C_OUT)},
          [("d", "l"), ("l", "p"), ("p", "f"), ("f", "m"), ("o", "m"), ("p", "c"), ("m", "e"), ("c", "e"), ("e", "a")],
          size=(15, 7.5))
    _flow("preprocessing_workflow.png", "Data preprocessing workflow (no leakage: every fitted step sees training data only)",
          {"a": (9, 62, 14, 12, "Load PDP samples\n+ results + tolerances", C_DATA),
           "b": (27, 62, 14, 12, "Quality checks\nduplicates, orphans,\nblank rates", C_PRE),
           "c": (45, 62, 14, 12, "Label from\nannotations\n(+ recompute check)", C_PRE),
           "d": (63, 62, 14, 12, "Select metadata\nfeatures only\n(no concentrations)", C_PRE),
           "e": (81, 62, 14, 12, "Stratified split\n65 / 15 / 20", C_PRE),
           "f": (81, 30, 16, 12, "Encoding\none-hot  or\ntarget (cross-fitted)", C_Q),
           "g": (63, 30, 16, 12, "Standardise\nselect (MI) / PCA", C_Q),
           "h": (45, 30, 16, 12, "Angle scaling\n[0, pi], clip", C_Q),
           "i": (27, 30, 14, 12, "Model input\n3-6 features", C_OUT)},
          [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h"), ("h", "i")], size=(15, 5.2),
          notes="Missing values: QUANTITY median-imputed with indicator; blank state/country -> 'NONE'. Outliers: IQR analysis, kept.")
    _flow("classical_ml_architecture.png", "Classical ML architecture",
          {"x": (10, 50, 14, 14, "Sample metadata\n(commodity, origin,\nclaim, facility...)", C_DATA),
           "o": (30, 50, 14, 14, "One-hot encoding\n(~100 columns)", C_PRE),
           "s": (50, 50, 14, 14, "StandardScaler", C_PRE),
           "m": (75, 74, 20, 12, "Logistic Regression", C_C), "v": (75, 58, 20, 12, "SVM (RBF kernel)", C_C),
           "r": (75, 42, 20, 12, "Random Forest", C_C), "g": (75, 26, 20, 12, "XGBoost", C_C),
           "z": (95, 50, 8, 14, "score\n> threshold", C_OUT)},
          [("x", "o"), ("o", "s"), ("s", "m"), ("s", "v"), ("s", "r"), ("s", "g"), ("m", "z"), ("v", "z"), ("r", "z"), ("g", "z")], size=(14, 5.6),
          notes="Each model: class-weighted, 25-trial Optuna search, 5-fold CV ROC-AUC; trained on the full 6,932-sample training split.")
    _flow("hybrid_architecture.png", "Hybrid quantum-classical architecture (optimised kernel framework)",
          {"x": (9, 50, 13, 14, "Sample\nmetadata", C_DATA),
           "f": (27, 50, 15, 14, "Classical encoder\ntarget encoding\n-> select / PCA", C_PRE),
           "a": (45, 50, 13, 14, "Angle scaling\nx -> [0, pi]\n(bandwidth)", C_PRE),
           "q": (66, 70, 20, 14, "Quantum kernel\nK_q = |<phi(x)|phi(x')>|^2\n(statevector simulation)", C_Q),
           "r": (66, 30, 20, 14, "Classical RBF kernel\nK_c = exp(-gamma |x-x'|^2)", C_C),
           "k": (86, 50, 14, 14, "K = a K_q + (1-a) K_c\n-> SVM (C)\n-> score", C_H)},
          [("x", "f"), ("f", "a"), ("a", "q"), ("a", "r"), ("q", "k"), ("r", "k")], size=(14, 5.6),
          notes="Edges: encoded features feed both kernels; the convex combination is a valid kernel. a, gamma, C, bandwidth, circuit and features are optimised jointly.")
    _flow("optimization_workflow.png", "Optimisation workflow (staged search; test set never used)",
          {"0": (10, 50, 15, 16, "Training split\nbalanced subsample\n(150 + 150)", C_DATA),
           "1": (31, 50, 16, 16, "Stage 1: KTA screening\n60 random architectures\n-> top 6", C_PRE),
           "2": (53, 50, 17, 16, "Stage 2: Optuna TPE\narch id, bandwidth, C,\nalpha, gamma\n(repeated 3-fold CV)", C_H),
           "3": (75, 50, 16, 16, "Stage 3: VQC search\nansatz, depth,\noptimizer, lr", C_Q),
           "4": (92, 50, 12, 16, "Best configs\n-> JSON\n-> final eval", C_OUT)},
          [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")], size=(15, 4.8),
          notes="Objective: mean cross-validated ROC-AUC (class-ratio independent). Basic configuration is enqueued first, so the search can only improve on it.")
    _flow("quantum_circuit_architecture.png", "Quantum circuit architecture (QSVM kernel and VQC)",
          {"i": (10, 62, 14, 14, "Classical features\nx1..xn in [0, pi]", C_DATA),
           "p": (30, 62, 15, 14, "Feature map U(x)\nH + P(x_i) + ZZ\n(entanglement: linear /\ncircular / full)", C_Q),
           "k": (55, 80, 22, 12, "QSVM: kernel = |<0|U(x')^+ U(x)|0>|^2\n-> classical SVM", C_H),
           "w": (55, 44, 18, 14, "VQC ansatz W(theta)\nRy/Rz + CX layers\n(depth = reps)", C_Q),
           "m": (78, 44, 14, 14, "Measure all qubits\nparity -> class", C_OUT),
           "o": (94, 44, 10, 14, "COBYLA /\nSPSA\nupdates theta", C_C)},
          [("i", "p"), ("p", "k"), ("p", "w"), ("w", "m"), ("m", "o")], size=(15, 5.6))


def make_all():
    jobs = [("eda", fig_eda), ("roc_pr", fig_roc_pr), ("comparison", fig_comparison),
            ("confusion", fig_confusion), ("ablation", fig_ablation), ("optuna", fig_optuna),
            ("kta", fig_kta), ("scaling", fig_scaling), ("runtime", fig_runtime),
            ("circuits", fig_circuits), ("diagrams", diagrams)]
    for name, fn in jobs:
        try:
            fn()
        except Exception:
            print(f"figure '{name}' skipped:", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    make_all()
