const path = require("path");
const L = require("./lib");
const { D, R, f3, P, B, H1, H2, bullet, pack } = L;
const cmp = (g, m) => R.comparison.find((r) => r.group === g && r.model === m);
const auc = (g, m) => f3(cmp(g, m).cal_roc_auc);
const QA = [
 ["Project and data", [
  ["What problem does the project solve?", "It ranks food samples by the risk that at least one pesticide residue exceeds permissible limits, using only information known at collection time (commodity, origin, claim, states, month, quantity). It is a pre-screening aid and does not replace laboratory testing."],
  ["Which dataset did you use and is it real?", "The USDA AMS Pesticide Data Program 2022 database: 10,665 samples and 2,890,842 residue results, public domain. Nothing is synthetic. The archive came from a GitHub mirror because ams.usda.gov was unreachable from the build environment; its SHA-256 is recorded and row counts match the official data dictionary. The checksum was not compared against the official download."],
  ["How is the label defined?", "A sample is positive if any of its results carries the official PDP annotation X (exceeds tolerance), V (residue with no tolerance) or QV (below quantifiable level, no tolerance). 320 of 10,665 samples (3.0%) are positive."],
  ["How do you know the label is right?", "We recomputed violations from detected concentration and the EPA tolerance table. That gives 337 positives and contains all 320 official ones (99.84% agreement). The 17 extra come from analytical rules we do not replicate. The official annotation is the label."],
  ["Why not use FSSAI limits?", "The dataset is US sample-level data, so US EPA tolerances apply. FSSAI regulations are the Indian framework, but sample-level Indian monitoring data are not public. The method would work unchanged on any table with an above-limit label."],
  ["What is target leakage and how did you avoid it?", "Leakage means an input contains information derived from the label. The label comes from residue concentrations and annotations, so none of those, nor LOD, detect flags, pesticide codes or laboratory, is used as input. Only collection metadata is used. Target encoding is cross-fitted so a row never sees its own label, and all fitted transforms use training data only."],
  ["Why is accuracy modest?", "Provenance metadata explains part of the risk (green beans 16.6% positive versus 0% for several commodities; imports 6.1% versus domestic 1.9%) but cannot say which individual sample is contaminated. Best ROC-AUC is about 0.80."],
 ]],
 ["Methodology", [
  ["Why ROC-AUC and PR-AUC as main metrics?", "With 3% positives accuracy is meaningless (predicting all negative gives 97%). ROC-AUC and PR-AUC do not depend on a threshold. Threshold-dependent metrics (macro-F1, balanced accuracy, precision, recall) use a threshold chosen on the validation set."],
  ["How did you prevent test-set tuning?", "Stratified 65/15/20 split. Every search uses cross-validation inside the training split. The validation split only selects the decision threshold. The test split is used once for reporting."],
  ["How did you handle class imbalance?", "Class weights for classical models, class-balanced subsamples (150 + 150) for quantum models, and evaluation on the untouched imbalanced test set."],
  ["Why do quantum models train on only 300 samples?", "Quantum kernels need a pairwise statevector simulation, which grows quadratically with training size, and the VQC needs many circuit evaluations. We therefore added classical controls tuned on exactly the same inputs and subsample, to keep the comparison fair."],
  ["Why five subsamples?", "A 300-sample subsample is random, so one draw could be lucky. We report mean and standard deviation over five (three for VQC)."],
 ]],
 ["Quantum and hybrid models", [
  ["What is a quantum kernel?", "A feature map circuit U(x) encodes a data point into a quantum state. The kernel between two points is the squared overlap of their states. A classical SVM is then trained on that kernel matrix."],
  ["What is the VQC?", "A feature map followed by a trainable ansatz; the measurement parity gives the class. A classical optimizer (COBYLA here) updates the circuit weights to minimise cross-entropy."],
  ["What is the hybrid model?", "A kernel K = alpha K_quantum + (1 - alpha) K_RBF fed to an SVM. A convex combination of valid kernels is a valid kernel. alpha = 0 and alpha = 1 give the pure classical and pure quantum versions, which makes the ablation clean."],
  ["Did you use real quantum hardware?", "No. All quantum models are noise-free statevector simulations with Qiskit. Timings are simulator timings. Noise, finite shots and hardware are future work."],
  ["Why angles in [0, pi]?", "Features are encoded as rotation angles. Scaling to [0, pi] with clipping keeps each rotation within one period so different values map to different states."],
 ]],
 ["Novelty and optimization", [
  ["What is novel in your framework?", "A staged optimization-driven hybrid framework that searches features, circuit architecture and hyper-parameters together: kernel-target-alignment screening of 60 random architectures, then Optuna over the shortlist, bandwidth, C, alpha and gamma, with the basic configuration included as a candidate, plus a full ablation and matched classical controls. We make no claim that the circuits are new."],
  ["What is kernel-target alignment?", "A measure of how well a kernel matrix agrees with the label structure. It is cheap (no SVM training), so it is used to shortlist architectures before expensive cross-validated search."],
  ["Why Optuna and what objective?", "Optuna's TPE sampler handles mixed categorical and continuous spaces efficiently. The objective is mean cross-validated ROC-AUC (repeated stratified 3x2 CV), which does not depend on the class ratio."],
  ["Does optimization help?", `For quantum models, yes: QSVM ${auc("2_basic_quantum", "qsvm")} to ${auc("4_optimized_quantum", "qsvm")} and hybrid ${auc("3_basic_hybrid", "hybrid")} to ${auc("5_optimized_hybrid", "hybrid")} test ROC-AUC. For the VQC, no: ${auc("2_basic_quantum", "vqc")} to ${auc("4_optimized_quantum", "vqc")}, although its CV score rose, so the small search overfit the CV estimate.`],
 ]],
 ["Results and honesty", [
  ["Did quantum beat classical?", `No. Tuned random forest reaches ${auc("1_classical_tuned", "rf")} and a logistic regression on the same compact inputs reaches ${auc("1b_classical_matched_tuned", "logreg")}. The best quantum-family models reach ${auc("4_optimized_quantum", "qsvm")}. Bootstrap intervals are wide (about ±0.06), so top models are not all statistically separable, but no quantum model is better than the classical ones and the VQC is clearly worse.`],
  ["What does the ablation show?", "Hyper-parameter optimization alone gets essentially all of the gain (0.771 versus 0.772 for everything). Removing the quantum kernel from the final hybrid (alpha = 0) gives 0.770, so the quantum part adds nothing measurable. The pure quantum kernel (alpha = 1) is worse (0.691). KTA shortlisting beat Optuna-only search (0.772 versus 0.752), by about one standard deviation."],
  ["Which feature map won and why does that matter?", "The Z map, which has no entanglement. Its kernel factorises over qubits and is classically cheap to simulate. So the winning model is classically simulable, consistent with benchmark findings that removing entanglement often does not hurt."],
  ["Is it a problem that quantum lost?", "No. A controlled negative result on real data is a valid outcome. The literature reports similar findings: a large benchmark found classical models generally ahead of quantum ones on small tabular tasks."],
  ["Why is precision so low?", "Prevalence is 3%, and the features describe provenance rather than the sample's actual residue. Precision is 0.06-0.28 at the calibrated threshold. The model ranks risk; it cannot certify a sample."],
  ["Why is CV AUC (0.81) higher than test AUC (0.77)?", "The search chooses the configuration with the best CV score, which biases that score upward (selection bias). The test set was not used for selection, so the test value is the honest one."],
 ]],
 ["Limitations and future work", [
  ["What are the main limitations?", "Only 64 positive test samples; simulated 3-6 qubit models on 300 samples; one dataset and year; dataset taken from a mirror without official checksum verification; small search budgets, especially for the VQC."],
  ["Could this run on a real quantum computer?", "The code uses Qiskit, so circuits could be sent to hardware, but kernel estimation needs many shots and noise would add error. The study does not claim hardware results."],
  ["What would you do next?", "Pool several PDP years for more positives, add EFSA (and FSSAI, if available) data under the same label rule, simulate shot and hardware noise, and test projected kernels and data re-uploading circuits that address kernel concentration."],
  ["How is the system used responsibly?", "The Streamlit app states that it gives a risk score for pre-screening and does not replace certified laboratory testing."],
 ]],
];
(async () => {
  const c = [new D.Paragraph({ style: "Title", children: [L.run("Viva preparation: questions and answers")] }),
    P("Quantum Machine Learning for Pesticide Residue Classification. Numbers come from the saved results files in the project.", { run: { color: "595959" } })];
  let n = 0;
  QA.forEach(([sec, items]) => {
    c.push(H1(sec));
    items.forEach(([q, a]) => {
      n += 1;
      c.push(new D.Paragraph({ spacing: { before: 140, after: 40 }, keepNext: true, children: [L.run(`Q${n}. ${q}`, { bold: true, color: "1F3A5F" })] }));
      c.push(P(a));
    });
  });
  await pack(c, "Viva Q&A", path.join(L.ROOT, "docs", "Viva_QA.docx"));
})();
