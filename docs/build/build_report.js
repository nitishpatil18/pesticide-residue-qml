const L = require("./lib");
const { D, R, f2, f3, pm, P, B, I, H1, H2, bullet, num, code, table, figure, gap, pb } = L;

const GROUP = { "1_classical_default": "1 Classical, default", "1_classical_tuned": "1 Classical, tuned (full train)",
  "1b_classical_matched_tuned": "1b Classical, tuned, matched inputs", "2_basic_quantum": "2 Basic quantum",
  "3_basic_hybrid": "3 Basic hybrid", "4_optimized_quantum": "4 Optimized quantum", "5_optimized_hybrid": "5 Optimized hybrid" };
const MODEL = { logreg: "LogReg", svm: "SVM", rf: "RF", xgb: "XGBoost", qsvm: "QSVM", vqc: "VQC", hybrid: "Hybrid kernel" };
const bsKey = (g, m) => ({ "1_classical_tuned": `1_classical|${m}_tuned` })[g] || null;
const BS = R.bootstrap;

function mainTable() {
  const rows = R.comparison.map((r) => [GROUP[r.group], MODEL[r.model], r.n_train,
    r.n_runs > 1 ? pm(r.cal_roc_auc, r.cal_roc_auc_std) : f3(r.cal_roc_auc), f3(r.cal_pr_auc),
    f3(r.cal_balanced_accuracy), f3(r.cal_macro_f1), f2(r.fit_time_s), f3(r.predict_ms_per_sample)]);
  return table([1900, 1000, 640, 1450, 700, 800, 780, 700, 820].map((x) => x),
    ["Group", "Model", "n train", "ROC-AUC", "PR-AUC", "Bal. acc.", "Macro-F1", "Fit s", "Pred ms"], rows, { size: 15 });
}
function detailTable() {
  const rows = R.comparison.map((r) => [GROUP[r.group], MODEL[r.model], f3(r.cal_precision), f3(r.cal_recall), f3(r.cal_f1),
    f3(r.cal_accuracy), f3(r.def_balanced_accuracy), f3(r.def_macro_f1)]);
  return table([2500, 1000, 850, 800, 800, 900, 1100, 1076],
    ["Group", "Model", "Precision", "Recall", "F1 (pos)", "Accuracy", "Bal. acc. (default thr.)", "Macro-F1 (default thr.)"], rows, { size: 15 });
}
function ablationTable() {
  const rows = R.ablation.map((r) => [r.variant, pm(r.roc_auc, r.roc_auc_std), f3(r.pr_auc), f3(r.cal_balanced_accuracy), f3(r.cal_macro_f1)]);
  return table([4300, 1700, 900, 1060, 1066], ["Variant", "ROC-AUC", "PR-AUC", "Bal. acc.", "Macro-F1"], rows, { size: 16 });
}
function bootTable() {
  const names = { "1_classical|rf_tuned": "Tuned RF (reference)", "1_classical|logreg_tuned": "Tuned LogReg (full train)", "1_classical|xgb_tuned": "Tuned XGBoost (full train)",
    "1b_classical_matched_tuned|logreg": "Matched-input LogReg (300 train)", "3_basic_hybrid|hybrid": "Basic hybrid", "4_optimized_quantum|qsvm": "Optimized QSVM",
    "5_optimized_hybrid|hybrid": "Optimized hybrid", "2_basic_quantum|vqc": "Basic VQC", "4_optimized_quantum|vqc": "Optimized VQC" };
  const rows = Object.entries(BS.models).map(([k, v]) => {
    const d = BS.vs_reference[k];
    return [names[k] || k, f3(v.auc), `[${f3(v.ci95[0])}, ${f3(v.ci95[1])}]`,
      d ? `${d.auc_diff >= 0 ? "+" : ""}${f3(d.auc_diff)} [${f3(d.ci95[0])}, ${f3(d.ci95[1])}]` : "-", d ? (d.ci_excludes_zero ? "yes" : "no") : "-"];
  });
  return table([2900, 900, 1700, 2600, 926], ["Model (seed-42 run)", "AUC", "95% CI", "AUC difference vs tuned RF", "CI excludes 0"], rows, { size: 16 });
}
function scalingTable() {
  const by = {}; R.scaling.forEach((r) => { (by[r.n_per_class] = by[r.n_per_class] || {})[r.model] = r; });
  const rows = Object.keys(by).map((n) => [2 * n, ...["optimized hybrid", "optimized qsvm", "matched classical SVM"].map((m) => pm(by[n][m].auc_mean, by[n][m].auc_std))]);
  return table([1800, 2400, 2400, 2426], ["Training samples (balanced)", "Optimized hybrid", "Optimized QSVM", "Matched classical SVM"], rows, { size: 17 });
}
const cfgRows = [
  ["Basic QSVM", "one-hot, PCA, 4 qubits, ZZ feature map (2 reps, linear), bandwidth 1, C = 1"],
  ["Optimized QSVM", `target encoding, MI selection + PCA (10 -> 5 features), 5 qubits, Z feature map (1 rep), bandwidth ${f3(R.opt.qsvm.best_params.bandwidth)}, C = ${f2(R.opt.qsvm.best_params.C)}`],
  ["Basic hybrid", "as basic QSVM; alpha = 0.5, RBF gamma = 0.25"],
  ["Optimized hybrid", `target encoding, MI selection (6 features), 6 qubits, Z feature map (1 rep), bandwidth ${f3(R.opt.hybrid.best_params.bandwidth)}, C = ${f3(R.opt.hybrid.best_params.C)}, alpha = ${f3(R.opt.hybrid.best_params.alpha)}, gamma = ${f3(R.opt.hybrid.best_params.gamma)}`],
  ["Basic VQC", "one-hot, PCA, 4 qubits, ZZ map (1 rep) + RealAmplitudes (2 reps), COBYLA, 60 iterations"],
  ["Optimized VQC", "target encoding, MI selection + PCA (10 -> 5 features), 5 qubits, ZZ map (2 reps) + EfficientSU2 (1 rep), COBYLA, 60 iterations"],
];

(async () => {
  const c = [];
  c.push(new D.Paragraph({ style: "Title", children: [L.run("Quantum Machine Learning for Pesticide Residue Classification")] }));
  c.push(P("An optimization-driven hybrid quantum-classical framework, evaluated honestly against tuned classical baselines", { run: { size: 26, color: "595959" } }));
  c.push(P("Technical report  |  Final-year project, B.E. Computer Science and Engineering (AI and ML)", { run: { size: 20, color: "595959" } }));
  c.push(gap());

  // 1 Abstract
  c.push(H1("1. Abstract"));
  c.push(P(["We build and evaluate a complete quantum machine learning pipeline that screens food samples for the risk of pesticide residues above regulatory limits. The data are the real, public USDA Pesticide Data Program (PDP) 2022 database: 10,665 samples and 2,890,842 residue results. A sample is labelled above-limit when any of its results carries the official violation annotation (X, V or QV); 320 samples (3.0%) are positive. Models see only collection metadata (commodity, origin, claim, states, month, quantity), never residue concentrations, so the label cannot leak into the inputs."]));
  c.push(P(["We compare tuned classical models (logistic regression, SVM, random forest, XGBoost) with a quantum kernel SVM (QSVM), a variational quantum classifier (VQC) and a hybrid quantum-classical kernel, each in a basic and an optimized form. The optimization framework searches features, circuit architecture and hyper-parameters (kernel-target-alignment screening followed by Optuna). All quantum models are classically simulated on small balanced training subsamples (300 samples)."]));
  c.push(P([B("Main result. "), `Optimization helps the quantum models: QSVM test ROC-AUC rises from ${f3(R.comparison.find(r=>r.group==="2_basic_quantum"&&r.model==="qsvm").cal_roc_auc)} to ${f3(R.comparison.find(r=>r.group==="4_optimized_quantum"&&r.model==="qsvm").cal_roc_auc)}, and the optimized hybrid reaches ${f3(R.comparison.find(r=>r.group==="5_optimized_hybrid").cal_roc_auc)}. It does not beat classical models. A tuned random forest reaches 0.800 and a logistic regression given the same compact inputs reaches 0.787. The ablation shows the hybrid's gain comes from hyper-parameter tuning and its classical RBF component, and the winning quantum kernel uses a non-entangling feature map that a classical computer evaluates trivially. We make no claim of quantum advantage.`]));
  c.push(P(["The system is a pre-screening aid. It does not replace certified laboratory residue testing."], { run: { italics: true } }));

  // 2 Intro
  c.push(H1("2. Introduction"));
  c.push(P("Pesticide residues in food are controlled through legal limits per pesticide and commodity. Enforcement relies on laboratory analysis, which is accurate but slow and costly, so only a fraction of consignments can be tested. A model that ranks samples by the risk of exceeding limits, using information available at the time of collection, could help decide where to look first."));
  c.push(P("Quantum machine learning (QML) proposes that quantum circuits can define feature spaces that are hard to reach classically. Whether this helps on realistic tabular data is an open question. This project is a controlled experiment on one real dataset, designed so that a negative result is as informative as a positive one."));

  // 3 Problem
  c.push(H1("3. Problem statement and objectives"));
  c.push(P(["Binary classification: given the metadata of a collected sample, predict whether any residue found in it exceeds permissible limits (class 1) or not (class 0). Objectives:"]));
  [["O1", "Use a real, documented public dataset and avoid target leakage."],
   ["O2", "Implement classical, quantum and hybrid classifiers under one evaluation protocol."],
   ["O3", "Add an optimization framework for features, circuit and hyper-parameters, and measure what each part contributes (ablation)."],
   ["O4", "Compare five groups fairly: classical, basic quantum, basic hybrid, optimized quantum, optimized hybrid."],
   ["O5", "Deliver a tested, reproducible code base and an interactive application, and report limitations honestly."]].forEach(([a, b]) => c.push(bullet([B(a + " "), b])));

  // 4 Background
  c.push(H1("4. Literature and background"));
  c.push(P(["Quantum kernel methods embed data in a Hilbert space with a feature map and use the state overlap as a kernel; Havlicek et al. proposed and demonstrated this with a classifier on a superconducting processor [1], and Schuld and Killoran showed that such models are kernel methods in a feature Hilbert space [2]. Variational classifiers train circuit parameters with a classical optimizer; they face trainability problems such as barren plateaus [3]."]));
  c.push(P(["Whether quantum kernels help depends on the data. Huang et al. showed that classical learners can match quantum models unless the data have specific structure [4], and Kubler et al. analysed the inductive bias of quantum kernels and concluded that useful quantum kernels are hard to find for classical data [5]. Thanasilp et al. showed that fidelity kernel values can concentrate exponentially with the number of qubits, which makes trained models trivial, and that training an embedding with kernel alignment is also affected [6]. A large benchmark of 12 quantum models on 6 tasks by Bowles, Ahmed and Schuld found that out-of-the-box classical models generally outperform the quantum ones and that removing entanglement often gives equal or better results [7]. Our findings are consistent with that picture."]));
  c.push(P(["Kernel-target alignment (KTA) measures how well a kernel matrix matches the label structure and is a cheap model-selection signal [8]. We use it to shortlist circuit architectures before the more expensive cross-validated search with Optuna [9]."]));

  // 5 Regulation
  c.push(H1("5. Regulatory framework"));
  c.push(P("Permissible pesticide residue levels are set per pesticide and commodity. In the United States the EPA establishes tolerances (40 CFR Part 180) [10]; the PDP reference tables contain the EPA tolerance list used by the programme. In India the Food Safety and Standards (Contaminants, Toxins and Residues) Regulations, 2011 under FSSAI define maximum residue limits [11], and the Codex Alimentarius sets international limits [12]."));
  c.push(P(["The PDP dataset is US sample-level data, so the applicable limits are the US tolerances and the official PDP annotations derived from them. FSSAI and Codex limits are the corresponding frameworks in India and internationally, but sample-level Indian monitoring data are not publicly available, so we do not apply FSSAI limits to these samples. Applying the method to FSSAI data would only need a labelled sample table."]));

  // 6 Dataset
  c.push(H1("6. Dataset"));
  c.push(P(["Source: USDA Agricultural Marketing Service, Pesticide Data Program, 2022 database (public domain) [13]. Official URL: ams.usda.gov/sites/default/files/media/2022PDPDatabase.zip. Because ams.usda.gov was not reachable from the build environment, the archive was obtained from a public GitHub mirror (bdyapa9/Commodities-and-pesticides-data). Its SHA-256 is "]));
  c.push(code(R.dq.dataset_sha256));
  c.push(P("Row counts match the official data dictionary exactly. The checksum could not be compared with the official download from the build environment; the README gives the command to verify it on a machine with internet access."));
  c.push(table([3400, 5626], ["Property", "Value"], [
    ["Samples", `${R.dq.n_samples.toLocaleString()} (18 columns, 23 commodities)`],
    ["Residue results", `${R.dq.n_result_rows.toLocaleString()} rows`],
    ["Duplicate sample keys / orphan result rows", `${R.dq.duplicate_sample_pk} / ${R.dq.orphan_result_rows}`],
    ["Samples without results", String(R.dq.samples_without_results)],
    ["Positive class (above limits)", `${R.dq.class_counts["1"]} samples (${(100 * R.dq.positive_rate).toFixed(1)}%)`],
    ["Result annotations", `X ${R.dq.annotation_counts.X}, V ${R.dq.annotation_counts.V}, QV ${R.dq.annotation_counts.QV}, Q ${R.dq.annotation_counts.Q}, none ${R.dq.annotation_counts.none.toLocaleString()}`],
  ]));
  c.push(gap());

  // 7 Label
  c.push(H1("7. Label construction and verification"));
  c.push(P("The PDP marks a result with annotation X when it exceeds the tolerance, V when a residue is found for which no tolerance exists, and QV when a residue is found below the quantifiable level on a commodity with no tolerance. We label a sample 1 if any of its results has annotation X, V or QV. Annotation Q (below the quantifiable level, tolerance exists) does not count."));
  c.push(P(`Independent check. We recomputed violations from detected concentration versus the EPA tolerance (unit-normalised to ppm; detections on pesticide-commodity pairs without a tolerance are violations). The recomputation yields ${R.dq.label_verification.violations_recomputed_from_concentration} positives and contains all ${R.dq.label_verification.violations_official_annotation} official positives (agreement ${(100 * R.dq.label_verification.agreement_rate).toFixed(2)}%). The 17 extra recomputed positives come from analytical rules we do not replicate. The official annotation is the training label; the recomputation is only a consistency check and is not used as a feature.`));

  // 8 Leakage
  c.push(H1("8. Feature design and leakage control"));
  c.push(P("Because the label is defined from residue measurements, any residue-derived field would leak the answer. Inputs are limited to attributes known at collection time:"));
  c.push(bullet([B("Used: "), "commodity, origin (domestic/import), distribution type, commodity type, claim (e.g. organic), state, country, growing/packing/distribution states, month, quantity; derived: log quantity, quantity-missing flag, month sine/cosine, import flag, organic flag, commodity x origin and commodity x distribution interactions."]));
  c.push(bullet([B("Excluded: "), "sample key, year, day, site, source id, variety, the recomputed label, and every residue-derived quantity (concentration, LOD, LOQ, detect flag, pesticide code, annotation, laboratory)."]));
  c.push(P("The task is therefore pre-screening risk from provenance, not detection. Absolute accuracy is modest by design: commodity and origin explain part of the risk (green beans 16.6% positive, tomatoes 7.5%; imports 6.1% versus domestic 1.9%) but not which individual sample is contaminated."));

  // 9 Preproc
  c.push(H1("9. Preprocessing pipeline"));
  c.push(...await figure("docs/diagrams/preprocessing_workflow.png", 600, "Figure 1. Preprocessing workflow. Every fitted step sees training data only."));
  c.push(bullet("Missing values: QUANTITY is missing for 4,317 samples (40%); it is median-imputed with a missing indicator. Blank state or country fields become the category NONE."));
  c.push(bullet("Outliers: 101 QUANTITY values exceed the IQR upper fence (33.5; max 177). They are valid counts and are kept; log transform and scaling limit their influence."));
  c.push(bullet("Split: stratified 65/15/20 into train (6,932; 208 positive), validation (1,600; 48) and test (2,133; 64). The test set is used only for final reporting. The validation set is used only to choose the decision threshold (maximum macro-F1). All hyper-parameter searches use cross-validation inside the training split."));
  c.push(bullet("Encoding: one-hot (rare categories pooled, about 100 columns) or cross-fitted target encoding (18 dense columns). The target encoder is fitted with internal 5-fold cross-fitting so a training row never sees its own label."));
  c.push(bullet("Quantum inputs: standardisation, then mutual-information feature selection or PCA to 3-6 features, then scaling to [0, pi] (clipped) as rotation angles."));
  c.push(bullet("Class imbalance (32:1): class weights for classical models; class-balanced subsamples (150 + 150) for quantum models; evaluation on the untouched imbalanced test set."));
  c.push(bullet("Reproducibility: global seed 42; subsample seeds 42-46 (VQC: 42-44)."));

  // 10 EDA
  c.push(H1("10. Exploratory analysis"));
  c.push(...await figure("results/figures/eda_overview.png", 600, "Figure 2. Exploratory overview: class balance and positive rate by commodity and origin."));
  c.push(P("Positive rates differ strongly by commodity (0% for several commodities, 16.6% for green beans) and by origin. This explains why metadata carries real signal, and also why the signal is limited."));

  // 11 Classical
  c.push(H1("11. Classical models"));
  c.push(...await figure("docs/diagrams/classical_ml_architecture.png", 600, "Figure 3. Classical pipeline."));
  c.push(P("Logistic regression, RBF-SVM, random forest and XGBoost, each in a default and a tuned form. Tuning uses Optuna (TPE, 25 trials, 5-fold stratified CV, objective ROC-AUC) on the full training split with class weighting. Search-space and best parameters are in experiments/classical_*_best.json."));
  c.push(table([1300, 1500, 1500, 4726], ["Model", "Default CV AUC", "Tuned CV AUC", "Best parameters"],
    ["logreg", "svm", "rf", "xgb"].map((m) => { const d = R.cls[m]; return [MODEL[m], f3(d.default_cv_auc), f3(d.best_value_cv_auc), JSON.stringify(Object.fromEntries(Object.entries(d.best_params).map(([k, v]) => [k, typeof v === "number" && !Number.isInteger(v) ? Number(v.toPrecision(3)) : v])))]; }), { size: 16 }));
  c.push(gap());
  c.push(P("To make a fair comparison with the quantum models, which see only 3-6 compact features and 300 training samples, we also tune all four classical models on exactly the same compact inputs and the same subsample (matched controls, 20 trials each)."));

  // 12 Quantum
  c.push(H1("12. Quantum models"));
  c.push(...await figure("docs/diagrams/quantum_circuit_architecture.png", 600, "Figure 4. Quantum circuit architecture for the QSVM kernel and the VQC."));
  c.push(bullet([B("QSVM. "), "A feature map U(x) (Z, ZZ, Pauli-ZXX or Pauli-XYY; 1-3 repetitions; linear, circular or full entanglement; bandwidth scaling) defines the kernel K(x, x') = |<0|U(x')^+ U(x)|0>|^2, computed with Qiskit Machine Learning's fidelity kernel on exact statevector simulation. A classical SVM with a precomputed kernel is trained on it."]));
  c.push(bullet([B("VQC. "), "Feature map followed by a RealAmplitudes or EfficientSU2 ansatz; measurement parity gives the class; weights are trained with COBYLA or SPSA (60 iterations) on cross-entropy loss."]));
  c.push(P("Simulation is noise-free statevector simulation (qiskit 2.6, qiskit-machine-learning 0.9). No quantum hardware was used. Kernel evaluation scales quadratically with training size, which is why quantum models use 300 training samples."));
  c.push(...await figure("docs/diagrams/circuit_quantum_kernel_feature_map.png", 520, "Figure 5. Example feature-map circuit used by the QSVM."));

  // 13 Hybrid
  c.push(H1("13. Hybrid architecture"));
  c.push(...await figure("docs/diagrams/hybrid_architecture.png", 600, "Figure 6. Hybrid kernel: convex combination of a quantum kernel and a classical RBF kernel."));
  c.push(P("The hybrid kernel is K = alpha K_quantum + (1 - alpha) K_RBF, with alpha in [0, 1]. A convex combination of two positive semi-definite kernels is a valid kernel, so it plugs into a standard SVM. Setting alpha = 1 gives a pure QSVM and alpha = 0 a pure classical RBF-SVM on the same inputs, which lets the ablation isolate the quantum contribution."));

  // 14 Optimization
  c.push(H1("14. Optimization-driven hybrid framework (novelty)"));
  c.push(...await figure("docs/diagrams/optimization_workflow.png", 600, "Figure 7. Staged optimization workflow."));
  c.push(P("The contribution is a staged, budget-bounded search over three groups of choices, evaluated with an ablation:"));
  c.push(num([B("Feature optimization. "), "encoding view (one-hot or target), reduction method (PCA, mutual-information selection, selection then PCA), number of qubits (3-6), selected features."]));
  c.push(num([B("Circuit optimization. "), "feature map, repetitions, entanglement pattern. 60 random architectures are screened by kernel-target alignment on the training subsample; the 6 best (plus the basic configuration) enter the next stage."]));
  c.push(num([B("Hyper-parameter optimization. "), "Optuna TPE picks an architecture id, bandwidth, SVM C, alpha and RBF gamma (QSVM: 30 trials; hybrid: 40 trials) with repeated stratified 3x2 CV on a balanced 150 + 150 subsample. The basic configuration is enqueued as the first candidate, so the search cannot do worse than it on the CV objective. VQC: 8 trials over input features, optimizer, feature map and ansatz."]));
  c.push(P("The objective is mean cross-validated ROC-AUC, which does not depend on the class ratio. The test set is never touched by any search."));
  c.push(table([2100, 6926], ["Model", "Configuration"], cfgRows, { size: 16 }));
  c.push(gap());

  // 15 Protocol
  c.push(H1("15. Experimental protocol"));
  c.push(bullet("Five comparison groups: (1) tuned classical, (2) basic quantum, (3) basic hybrid, (4) optimized quantum, (5) optimized hybrid; plus (1b) classical models tuned on the same compact inputs and 300-sample subsample as the quantum models."));
  c.push(bullet("Quantum and matched-control results are means over 5 independent balanced training subsamples (3 for VQC, which is slower); all use the same fixed validation and test sets."));
  c.push(bullet("Threshold-free metrics (ROC-AUC, PR-AUC) are the primary comparison. Threshold-dependent metrics (accuracy, precision, recall, F1, macro-F1, balanced accuracy) are reported at a validation-calibrated threshold that maximises macro-F1; default-threshold values are in the appendix tables."));
  c.push(bullet("Fit time and prediction time per sample are measured on a 2-core CPU."));

  // 16 Results
  c.push(H1("16. Results"));
  c.push(P("Table 1. Test-set results (2,133 samples, 64 positive). Mean ± standard deviation over training subsamples for quantum and matched models; single run for full-data classical models. Metrics other than AUC use the validation-calibrated threshold. PR-AUC of a random model is 0.030."));
  c.push(mainTable());
  c.push(gap());
  c.push(...await figure("results/figures/model_comparison.png", 620, "Figure 8. Test-set comparison of all model groups."));
  c.push(...await figure("results/figures/roc_pr_curves.png", 600, "Figure 9. ROC and precision-recall curves (seed-42 models)."));
  c.push(...await figure("results/figures/confusion_matrices.png", 600, "Figure 10. Confusion matrices at the calibrated threshold."));
  c.push(H2("16.1 What the results say"));
  c.push(bullet("Tuning matters for classical models: default RF 0.713 to tuned 0.800, default SVM 0.686 to 0.772."));
  c.push(bullet(`Optimization matters for quantum models: QSVM ${f3(R.comparison.find(r=>r.group==="2_basic_quantum"&&r.model==="qsvm").cal_roc_auc)} to ${f3(R.comparison.find(r=>r.group==="4_optimized_quantum"&&r.model==="qsvm").cal_roc_auc)}; hybrid ${f3(R.comparison.find(r=>r.group==="3_basic_hybrid").cal_roc_auc)} to ${f3(R.comparison.find(r=>r.group==="5_optimized_hybrid").cal_roc_auc)}.`));
  c.push(bullet("Classical models are at least as good as every quantum or hybrid model. The best quantum-family model (optimized hybrid or QSVM, AUC 0.772) is below tuned RF (0.800) and about equal to a logistic regression or SVM on the same compact inputs (0.774-0.787)."));
  c.push(bullet("The VQC is the weakest model, and optimization did not help it on the test set (basic 0.667, optimized 0.632), although its CV score improved from 0.668 to 0.701. With 8 trials, 60 optimizer iterations and 300 samples, the search overfits the CV estimate."));
  c.push(bullet("Precision is low for every model (about 0.06-0.28 at the calibrated threshold) because only 3% of samples are positive. These models rank risk; they cannot certify a sample."));
  c.push(P("Table 2. Remaining metrics (appendix view)."));
  c.push(detailTable());
  c.push(gap());

  // 17 Ablation
  c.push(H1("17. Ablation study"));
  c.push(P("Table 3. Hybrid-kernel ablation on the test set (mean ± std over 5 subsamples). Each variant switches one group of optimized choices (features, circuit, hyper-parameters) on or off relative to the basic configuration."));
  c.push(ablationTable());
  c.push(gap());
  c.push(...await figure("results/figures/ablation_hybrid.png", 620, "Figure 11. Ablation of the hybrid framework."));
  c.push(H2("17.1 Interpretation"));
  c.push(bullet("Hyper-parameter optimization alone (A3, 0.771) recovers essentially all of the final score (A7, 0.772). Feature or circuit optimization alone does not help (A1 0.754, A2 0.755), and features + circuit without hyper-parameters is worse (A4 0.735). Differences of about 0.01 are within one standard deviation (0.011-0.017), so only the larger gaps are reliable."));
  c.push(bullet("Removing the quantum part from the final model (A9, alpha = 0, pure RBF kernel on the same inputs) gives 0.770, indistinguishable from the full hybrid (0.772). The quantum kernel contributes no measurable gain. Using the quantum kernel alone (A8, alpha = 1) is clearly worse (0.691)."));
  c.push(bullet("KTA shortlisting versus an Optuna-only search with the same trial budget (B1): 0.772 versus 0.752. This favours the KTA stage, with the caveat that the difference is about one standard deviation and comes from one search run each."));
  c.push(bullet("The optimizer chose the Z feature map (single-qubit rotations, no entanglement) for both the QSVM and the hybrid. Its kernel is a product over qubits and is simulated efficiently on a classical computer; the result is therefore a classically simulable kernel, in line with the finding that entanglement is often not the useful ingredient [7]."));
  c.push(...await figure("results/figures/optuna_history.png", 560, "Figure 12. Optuna optimization history."));
  c.push(...await figure("results/figures/kta_screening.png", 520, "Figure 13. Kernel-target alignment of the 60 screened architectures."));

  // 18 Stats
  c.push(H1("18. Statistical analysis"));
  c.push(P(`The test set has only ${BS.n_positive} positive samples, so single AUC values are imprecise. We computed stratified bootstrap confidence intervals (${BS.n_boot} resamples, positives and negatives resampled separately) and paired AUC differences against the tuned random forest, using the seed-42 models. Because they use one trained model each, values differ slightly from the 5-subsample means in Table 1.`));
  c.push(bootTable());
  c.push(gap());
  c.push(P("The intervals are wide (about ±0.06). The tuned RF is not significantly better than XGBoost, the matched-input logistic regression or the optimized QSVM (paired intervals include zero), and it is significantly better than the full-data logistic regression, the basic hybrid, the optimized hybrid and both VQCs. Within the single test set we therefore cannot rank the top classical and quantum models with confidence; we can say that no quantum or hybrid model is better than the tuned classical models, and that the VQC is clearly worse."));

  // 19 Scaling
  c.push(H1("19. Scaling and runtime"));
  c.push(P("Table 4. Test ROC-AUC against balanced training-set size (mean ± std over 5 subsamples)."));
  c.push(scalingTable());
  c.push(gap());
  c.push(...await figure("results/figures/scaling_curve.png", 480, "Figure 14. Learning curve."));
  c.push(P("All three curves rise to about 300 training samples and then flatten or dip within noise; there is no sign that the quantum or hybrid kernel benefits more from additional data. Runtime (Figure 15): the QSVM on 300 samples fits in about 0.4-0.5 s and predicts in 1.0-1.5 ms per sample under statevector simulation, versus about 0.02 ms for a random forest; the VQC needs 24-46 s per fit. Quantum timings are simulator timings and say nothing about hardware."));
  c.push(...await figure("results/figures/runtime.png", 520, "Figure 15. Fit and prediction time."));

  // 20 Discussion
  c.push(H1("20. Discussion"));
  c.push(P("Why do quantum models not win here? The data are low-dimensional tabular metadata whose predictive structure is mostly additive (commodity and origin effects). Classical models capture this directly. A kernel can only help if its feature space contains the target function in a way classical kernels cannot match [5]; the optimizer's preference for a non-entangling Z map suggests it found no such structure. Fidelity kernels with entangling maps can also concentrate as qubits increase [6]; we did not test whether this affects our ZZ-map configurations, so it is a possible explanation, not a finding."));
  c.push(P("What did the framework achieve? It reliably moves a poorly chosen quantum configuration (basic QSVM, 0.733) to a competitive one (0.772) at small cost (one minute of search for the QSVM). The cost is that the final model is nearly classical. This is a legitimate and useful outcome for a controlled study: the framework acts as a model-selection tool that also reveals when the quantum component is not needed."));

  // 21 Limitations
  c.push(H1("21. Limitations and threats to validity"));
  [["Few positives.", "64 positive test samples give wide confidence intervals (Section 18); many differences between models are not statistically resolved."],
   ["Simulated, small-scale quantum models.", "Noise-free statevector simulation, 3-6 qubits, 300 training samples. Larger circuits, noise, finite-shot sampling and real hardware were not studied; finite-shot kernel estimation would add error."],
   ["Different training sizes.", "Full-data classical models use 6,932 samples and quantum models 300. We added matched-input controls to separate this effect; matched controls still match or exceed the quantum models."],
   ["Selection bias in CV scores.", "The best CV AUC of searched models (0.81) is higher than their test AUC (0.77) because the search selects on the CV score. The test set was not used for selection."],
   ["Weak label-feature link.", "Features are provenance metadata, so the task is risk screening. Precision at 3% prevalence is low; the model cannot identify contaminated samples."],
   ["Single dataset and year.", "PDP 2022, US market. Results may not transfer to other years, countries or regulatory frameworks. FSSAI and EFSA sample-level data were not available; EFSA monitoring data are future work."],
   ["Dataset provenance.", "The archive was fetched from a GitHub mirror. Row counts match the official data dictionary but the official checksum was not verified."],
   ["Search budget.", "Search budgets were fixed and small (VQC: 8 trials). Larger budgets might change the VQC result. KTA-based training of embeddings can itself suffer from concentration [6]."],
   ["Label subtleties.", "17 samples have a recomputed violation without an official annotation; the official annotation is used."]].forEach(([a, b]) => c.push(bullet([B(a + " "), b])));

  // 22 Conclusion
  c.push(H1("22. Conclusion and future work"));
  c.push(P("We delivered a leakage-controlled, reproducible pipeline on real public data, with classical, quantum and hybrid models, an optimization framework with ablation, a tested code base and a Streamlit application. Optimization improved the quantum models but no quantum or hybrid model outperformed tuned classical baselines; the best hybrid is effectively classical. Reporting this negative result clearly is the main scientific outcome."));
  c.push(P("Future work: (i) larger positive counts by pooling several PDP years; (ii) the EFSA monitoring data and, where available, FSSAI data under the same label definition; (iii) shot-noise and hardware-noise simulation and, optionally, hardware runs; (iv) projected quantum kernels and data re-uploading circuits, which target the concentration problem; (v) cost-sensitive thresholds chosen with laboratory capacity in mind."));

  // References
  c.push(H1("References"));
  const refs = [
    "[1] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, J. M. Gambetta. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209-212 (2019).",
    "[2] M. Schuld, N. Killoran. Quantum machine learning in feature Hilbert spaces. Physical Review Letters 122, 040504 (2019).",
    "[3] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven. Barren plateaus in quantum neural network training landscapes. Nature Communications 9, 4812 (2018).",
    "[4] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, J. R. McClean. Power of data in quantum machine learning. Nature Communications 12, 2631 (2021).",
    "[5] J. M. Kubler, S. Buchholz, B. Scholkopf. The inductive bias of quantum kernels. Advances in Neural Information Processing Systems 34 (NeurIPS 2021), 12661-12673.",
    "[6] S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes. Exponential concentration in quantum kernel methods. Nature Communications 15, 5200 (2024). doi:10.1038/s41467-024-49287-w.",
    "[7] J. Bowles, S. Ahmed, M. Schuld. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059 (2024).",
    "[8] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, J. Kandola. On kernel-target alignment. Advances in Neural Information Processing Systems 14 (2001).",
    "[9] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. Optuna: a next-generation hyperparameter optimization framework. Proceedings of KDD 2019.",
    "[10] United States Code of Federal Regulations, Title 40, Part 180: Tolerances and exemptions for pesticide chemical residues in food (U.S. Environmental Protection Agency).",
    "[11] Food Safety and Standards Authority of India. Food Safety and Standards (Contaminants, Toxins and Residues) Regulations, 2011.",
    "[12] FAO/WHO Codex Alimentarius Commission. Pesticide residues in food: maximum residue limits (Codex pesticide MRL database).",
    "[13] U.S. Department of Agriculture, Agricultural Marketing Service. Pesticide Data Program, 2022 database and data dictionary. https://www.ams.usda.gov/datasets/pdp",
    "[14] A. Javadi-Abhari et al. Quantum computing with Qiskit. arXiv:2405.08810 (2024).",
    "[15] F. Pedregosa et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12, 2825-2830 (2011).",
    "[16] T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. Proceedings of KDD 2016, 785-794.",
    "[17] L. Breiman. Random forests. Machine Learning 45, 5-32 (2001).",
    "[18] D. Micci-Barreca. A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. SIGKDD Explorations 3(1), 27-32 (2001).",
  ];
  refs.forEach((r) => c.push(P(r, { run: { size: 19 }, para: { spacing: { after: 80, line: 276 } } })));

  // Appendix
  c.push(H1("Appendix A. Reproduction"));
  c.push(P("From the project root (Python 3.13 was used for all reported runs):"));
  ["pip install -r requirements.txt", "python main.py data        # download/verify, label, split", "python main.py classical", "python main.py optimize",
   "python main.py evaluate", "python main.py ablation", "python main.py scaling", "python main.py stats", "python main.py figures",
   "python -m pytest -q tests", "streamlit run app.py"].forEach((x) => c.push(code(x)));
  c.push(gap());
  c.push(P("All metrics, configurations, search trials and logs are saved under results/ and experiments/."));
  await L.pack(c, "QML for pesticide residue classification", path.join(L.ROOT, "docs", "Technical_Report.docx"));
})();
const path = require("path");
