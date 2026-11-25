const path = require("path"); const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
// applyTheme writes the theme colours into the saved deck. Set PPTX_THEME_HELPER to the path of an
// apply_theme.js module; without it the deck is still written but keeps the stock Office colours.
const applyTheme = process.env.PPTX_THEME_HELPER ? require(process.env.PPTX_THEME_HELPER).applyTheme : async () => {};
const R = JSON.parse(fs.readFileSync(path.join(__dirname, "results_summary.json"), "utf8"));
const ROOT = path.resolve(__dirname, "..", "..");
const f3 = (x) => Number(x).toFixed(3);
const cmp = (g, m) => R.comparison.find((r) => r.group === g && r.model === m);

const THEME = { name: "PesticideQML", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A34", lt1: "FFFFFF", dk2: "0E3B43", lt2: "EEF4F3", accent1: "0B7A75", accent2: "E0A100", accent3: "5B6C8F",
    accent4: "C8553D", accent5: "7FB7A9", accent6: "8A8D91", hlink: "0B7A75", folHlink: "5B6C8F" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Quantum Machine Learning for Pesticide Residue Classification";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.8, fontSize: 28, bold: true, color: C.text2, fontFace: "Cambria", valign: "middle", margin: 0 }, text: "" } }],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: C.accent6 } });
pres.defineSlideMaster({ title: "DARK", background: { color: "0E3B43" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.5, w: 8.6, h: 1.4, fontSize: 38, bold: true, color: C.background1, fontFace: "Cambria", valign: "top", margin: 0 }, text: "" } }] });

let sec = "";
const section = (t) => { sec = t; pres.addSection({ title: t }); };
const slide = (master = "CONTENT") => pres.addSlide({ masterName: master, sectionTitle: sec });
const text = (s, t, o) => s.addText(t, { isTextBox: true, fontFace: "Calibri", color: C.text1, margin: 0, valign: "top", ...o });
const img = async (s, rel, x, y, w, h) => {
  const m = await sharp(path.join(ROOT, rel)).metadata(); const r = m.width / m.height;
  let ww = w, hh = w / r; if (hh > h) { hh = h; ww = h * r; }
  s.addImage({ path: path.join(ROOT, rel), x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh, altText: rel });
};
const card = (s, x, y, w, h, head, body, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: name + " card" });
  text(s, head, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.4, fontSize: 16, bold: true, color: C.accent1, fontFace: "Cambria", objectName: name + " head" });
  text(s, body, { x: x + 0.2, y: y + 0.6, w: w - 0.4, h: h - 0.75, fontSize: 14, paraSpaceAfter: 4, objectName: name + " body" });
};
const stat = (s, x, y, w, big, small, name) => {
  text(s, big, { x, y, w, h: 0.9, fontSize: 48, bold: true, color: C.accent1, fontFace: "Cambria", objectName: name + " number" });
  text(s, small, { x, y: y + 0.95, w, h: 0.6, fontSize: 14, color: C.text1, objectName: name + " label" });
};
const bl = (items) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } }));

(async () => {
  // 1 title
  section("Opening");
  let s = slide("DARK");
  s.addText("Quantum Machine Learning for Pesticide Residue Classification", { placeholder: "title" });
  text(s, "An optimization-driven hybrid framework, evaluated honestly against tuned classical baselines", { x: 0.7, y: 3.1, w: 7.6, h: 0.8, fontSize: 18, color: "CFE3E0", objectName: "subtitle" });
  text(s, "Final-year project  |  B.E. Computer Science and Engineering (AI and ML)", { x: 0.7, y: 4.6, w: 8, h: 0.4, fontSize: 14, color: "9FC3BE", objectName: "byline" });
  s.addNotes("Real data, real quantum simulation, honest comparison. The main finding is a negative one for quantum advantage.");

  // 2 problem
  s = slide(); s.addText("Screening for residue risk before the lab", { placeholder: "title" });
  text(s, "Lab testing is accurate but slow and costly, so only some consignments are tested. A model that ranks samples by risk of exceeding limits, using information known at collection, could help decide where to look first.", { x: 0.5, y: 1.3, w: 4.6, h: 2.0, fontSize: 16, objectName: "problem text" });
  text(s, "Question: do quantum kernels add anything over tuned classical models on this real task?", { x: 0.5, y: 3.5, w: 4.6, h: 1.2, fontSize: 16, bold: true, color: C.accent1, objectName: "research question" });
  stat(s, 5.5, 1.3, 4, "10,665", "samples in PDP 2022", "samples");
  stat(s, 5.5, 2.75, 4, "3.0%", "above permissible limits (320)", "prevalence");
  stat(s, 5.5, 4.0, 4, "2.89M", "residue results behind the label", "results");
  s.addNotes("The system is a pre-screening aid and does not replace certified laboratory testing.");

  // 3 dataset + label
  s = slide(); s.addText("Real data and a verified label", { placeholder: "title" });
  card(s, 0.5, 1.3, 4.35, 3.1, "USDA PDP 2022 database", bl(["Public domain, official EPA tolerance table included", "10,665 samples, 23 commodities", "No duplicates, no orphan results", "Mirror copy; SHA-256 recorded, official checksum not verifiable from the build host"]), "dataset");
  card(s, 5.15, 1.3, 4.35, 3.1, "Label = official annotation", bl(["Above limits if any result is X, V or QV", "Check: recomputing from concentration vs tolerance gives 337 positives, covering all 320 official ones (99.84%)", "Class ratio 32 : 1"]), "label");

  // 4 leakage
  s = slide(); s.addText("No leakage: metadata only", { placeholder: "title" });
  card(s, 0.5, 1.3, 4.35, 3.1, "Inputs (known at collection)", bl(["Commodity, origin, claim", "State, country, facility states", "Month, quantity (+ derived)", "Commodity x origin interactions"]), "inputs");
  card(s, 5.15, 1.3, 4.35, 3.1, "Never used as input", bl(["Concentration, LOD, LOQ", "Detect flag, pesticide, annotation", "Laboratory, sample key", "Recomputed label"]), "excluded");
  s.addNotes("Because the label is defined from residue measurements, any residue-derived field would leak the answer. The task is risk from provenance.");

  // 5 architecture
  section("Method");
  s = slide(); s.addText("System architecture", { placeholder: "title" });
  await img(s, "docs/diagrams/system_architecture.png", 0.5, 1.2, 9, 3.9);

  // 6 preprocessing
  s = slide(); s.addText("Preprocessing without leakage", { placeholder: "title" });
  await img(s, "docs/diagrams/preprocessing_workflow.png", 0.5, 1.2, 9, 2.5);
  text(s, bl(["Stratified 65 / 15 / 20 split: 6,932 train, 1,600 validation, 2,133 test (64 positives)", "Encoders and reducers fitted on training data only; target encoding cross-fitted", "Validation selects only the decision threshold; the test set is used once"]), { x: 0.5, y: 3.85, w: 9, h: 1.3, fontSize: 14, paraSpaceAfter: 4, objectName: "preprocessing bullets" });

  // 7 models
  s = slide(); s.addText("Three model families, one protocol", { placeholder: "title" });
  card(s, 0.5, 1.3, 2.85, 3.1, "Classical", bl(["LogReg, SVM, RF, XGBoost", "Optuna, 25 trials, 5-fold CV", "Full 6,932 training rows", "Matched controls on same 300 rows"]), "classical");
  card(s, 3.55, 1.3, 2.85, 3.1, "Quantum", bl(["QSVM: fidelity kernel + SVM", "VQC: feature map + ansatz, COBYLA", "Statevector simulation, 3-6 qubits", "300 balanced training rows"]), "quantum");
  card(s, 6.6, 1.3, 2.9, 3.1, "Hybrid", bl(["K = a K_quantum + (1 - a) K_RBF", "a = 0: classical; a = 1: quantum", "Lets the ablation isolate the quantum part"]), "hybrid");

  // 8 circuits
  s = slide(); s.addText("Quantum circuits", { placeholder: "title" });
  await img(s, "docs/diagrams/quantum_circuit_architecture.png", 0.4, 1.2, 5.6, 3.2);
  await img(s, "docs/diagrams/circuit_quantum_kernel_feature_map.png", 6.1, 1.2, 3.5, 2.6);
  text(s, "Feature maps Z, ZZ, Pauli-ZXX, Pauli-XYY; 1-3 repetitions; linear, circular or full entanglement; ansatz RealAmplitudes or EfficientSU2.", { x: 0.5, y: 4.5, w: 9, h: 0.7, fontSize: 14, objectName: "circuit caption" });

  // 9 optimization
  s = slide(); s.addText("Optimization-driven hybrid framework", { placeholder: "title" });
  await img(s, "docs/diagrams/optimization_workflow.png", 0.5, 1.15, 9, 1.9);
  text(s, bl(["Features: encoding, reduction method, 3-6 qubits", "Circuit: 60 architectures screened by kernel-target alignment, top 6 kept", "Hyper-parameters: Optuna over architecture, bandwidth, C, alpha, gamma (repeated 3x2 CV)", "Basic configuration is a candidate, so the search cannot do worse on CV; the test set is never searched"]), { x: 0.5, y: 3.25, w: 9, h: 1.9, fontSize: 15, paraSpaceAfter: 5, objectName: "optimization bullets" });

  // 10 results chart
  section("Results");
  s = slide(); s.addText("Classical models stay ahead", { placeholder: "title" });
  const rows = [["Tuned RF", 0.800], ["Tuned XGBoost", 0.790], ["Matched LogReg", +cmp("1b_classical_matched_tuned", "logreg").cal_roc_auc],
    ["Optimized hybrid", +cmp("5_optimized_hybrid", "hybrid").cal_roc_auc], ["Optimized QSVM", +cmp("4_optimized_quantum", "qsvm").cal_roc_auc],
    ["Basic hybrid", +cmp("3_basic_hybrid", "hybrid").cal_roc_auc], ["Basic QSVM", +cmp("2_basic_quantum", "qsvm").cal_roc_auc],
    ["Basic VQC", +cmp("2_basic_quantum", "vqc").cal_roc_auc], ["Optimized VQC", +cmp("4_optimized_quantum", "vqc").cal_roc_auc]];
  s.addChart(pres.charts.BAR, [{ name: "Test ROC-AUC", labels: rows.map((r) => r[0]), values: rows.map((r) => Number(r[1].toFixed(3))) }], {
    x: 0.4, y: 1.2, w: 5.9, h: 3.9, barDir: "bar", chartColors: ["0B7A75"], showTitle: true, title: "Test ROC-AUC (higher is better)", titleFontSize: 12, titleFontFace: "Calibri",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontFace: "Calibri", dataLabelFormatCode: "0.000", showLegend: false,
    catAxisLabelFontSize: 10, catAxisLabelFontFace: "Calibri", catAxisOrientation: "maxMin", valAxisMinVal: 0.5, valAxisMaxVal: 0.85, valAxisLabelFontSize: 10, valAxisLabelFontFace: "Calibri",
    valGridLine: { color: "E1E6E6", size: 0.5 }, catGridLine: { style: "none" } });
  text(s, bl(["Optimization lifts QSVM 0.733 to 0.772 and the hybrid 0.765 to 0.772", "No quantum or hybrid model beats tuned classical models", "VQC is weakest; optimization did not help it", "Quantum rows: mean over 5 subsamples (VQC 3), 300 training rows"]), { x: 6.5, y: 1.4, w: 3.0, h: 3.6, fontSize: 14, paraSpaceAfter: 6, objectName: "results bullets" });
  s.addNotes("Full-data classical models use 6,932 training rows; quantum and matched models use 300.");

  // 11 ablation
  s = slide(); s.addText("Ablation: where the gain comes from", { placeholder: "title" });
  const ab = ["A0 basic", "A1 features", "A2 circuit", "A3 hyper-params", "A4 feat + circ", "A5 feat + hyp", "A6 circ + hyp", "A7 all optimized", "A8 alpha = 1", "A9 alpha = 0", "B1 no KTA"];
  s.addChart(pres.charts.BAR, [{ name: "Test ROC-AUC", labels: ab, values: R.ablation.map((r) => Number(r.roc_auc.toFixed(3))) }], {
    x: 0.4, y: 1.2, w: 5.9, h: 3.9, barDir: "bar", chartColors: ["0B7A75"], showTitle: true, title: "Hybrid ablation, test ROC-AUC", titleFontSize: 12, titleFontFace: "Calibri",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontFace: "Calibri", dataLabelFormatCode: "0.000", showLegend: false,
    catAxisLabelFontSize: 10, catAxisLabelFontFace: "Calibri", catAxisOrientation: "maxMin", valAxisMinVal: 0.5, valAxisMaxVal: 0.85, valAxisLabelFontSize: 10, valAxisLabelFontFace: "Calibri",
    valGridLine: { color: "E1E6E6", size: 0.5 }, catGridLine: { style: "none" } });
  text(s, bl(["Hyper-parameters alone (A3, 0.771) recover the full gain (A7, 0.772)", "Removing the quantum kernel (A9) gives 0.770: no measurable quantum contribution", "Winner uses the Z map: no entanglement, classically simulable", "Differences near 0.01 are within one standard deviation"]), { x: 6.5, y: 1.4, w: 3.0, h: 3.6, fontSize: 14, paraSpaceAfter: 6, objectName: "ablation bullets" });

  // 12 statistics
  s = slide(); s.addText("How sure can we be? 64 positives", { placeholder: "title" });
  const bn = { "1_classical|rf_tuned": "Tuned RF (reference)", "1b_classical_matched_tuned|logreg": "Matched LogReg", "4_optimized_quantum|qsvm": "Optimized QSVM", "5_optimized_hybrid|hybrid": "Optimized hybrid", "4_optimized_quantum|vqc": "Optimized VQC" };
  const hdr = ["Model (seed-42 run)", "AUC", "95% CI", "Diff vs RF [95% CI]"].map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "0E3B43" }, fontSize: 13 } }));
  const tr = Object.keys(bn).map((k) => { const m = R.bootstrap.models[k], d = R.bootstrap.vs_reference[k];
    return [bn[k], f3(m.auc), `[${f3(m.ci95[0])}, ${f3(m.ci95[1])}]`, d ? `${d.auc_diff >= 0 ? "+" : ""}${f3(d.auc_diff)} [${f3(d.ci95[0])}, ${f3(d.ci95[1])}]` : "-"].map((t) => ({ text: t, options: { fontSize: 13 } })); });
  s.addTable([hdr, ...tr], { x: 0.5, y: 1.3, w: 9, colW: [2.6, 1.0, 2.1, 3.3], rowH: 0.42, border: { type: "solid", color: "C9D3D3", pt: 0.75 }, fontFace: "Calibri", color: "1B2A34", valign: "middle", objectName: "bootstrap table" });
  text(s, "Stratified bootstrap, 2,000 resamples. Tuned RF vs optimized QSVM and matched LogReg: intervals include zero. Vs optimized hybrid and both VQCs: RF is significantly better. We cannot rank the top models with confidence, and no quantum model is better.", { x: 0.5, y: 3.95, w: 9, h: 1.1, fontSize: 14, objectName: "stats note" });

  // 13 scaling
  s = slide(); s.addText("More data does not favour quantum", { placeholder: "title" });
  const ns = [...new Set(R.scaling.map((r) => r.n_per_class))];
  const ser = ["optimized hybrid", "optimized qsvm", "matched classical SVM"].map((m) => ({ name: m, labels: ns.map((n) => String(2 * n)), values: ns.map((n) => Number(R.scaling.find((r) => r.n_per_class === n && r.model === m).auc_mean.toFixed(3))) }));
  s.addChart(pres.charts.LINE, ser, { x: 0.4, y: 1.2, w: 5.9, h: 3.9, chartColors: ["0B7A75", "E0A100", "5B6C8F"], showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "Calibri",
    showTitle: true, title: "Test ROC-AUC vs training rows (balanced)", titleFontSize: 12, titleFontFace: "Calibri", lineSize: 2.5, lineDataSymbolSize: 7,
    catAxisLabelFontSize: 10, catAxisLabelFontFace: "Calibri", valAxisLabelFontSize: 10, valAxisLabelFontFace: "Calibri", valAxisMinVal: 0.6, valAxisMaxVal: 0.8, valGridLine: { color: "E1E6E6", size: 0.5 }, catGridLine: { style: "none" } });
  text(s, bl(["All curves rise to about 300 rows, then flatten within noise", "QSVM fits 300 rows in about 0.4 s but predicts at 1-1.5 ms per sample (simulator); RF 0.02 ms", "VQC needs 24-46 s per fit"]), { x: 6.5, y: 1.4, w: 3.0, h: 3.6, fontSize: 14, paraSpaceAfter: 6, objectName: "scaling bullets" });

  // 14 app
  section("System and wrap-up");
  s = slide(); s.addText("Streamlit application", { placeholder: "title" });
  card(s, 0.5, 1.3, 4.35, 3.1, "Pages", bl(["Introduction and disclaimer", "Dataset and visualization", "Classify a sample with a saved model", "Classical vs quantum vs hybrid", "Optimization results and circuits"]), "pages");
  card(s, 5.15, 1.3, 4.35, 3.1, "Behaviour", bl(["Model selection, saved or retrained", "Validation-calibrated threshold", "Risk percentile versus test scores", "States that it does not replace laboratory testing"]), "behaviour");

  // 15 limitations
  s = slide(); s.addText("Limitations", { placeholder: "title" });
  card(s, 0.5, 1.3, 4.35, 3.1, "Data and statistics", bl(["Only 64 positive test samples: wide CIs", "One dataset, one year, US market", "Mirror copy of the data; official checksum unverified", "CV score optimistic after selection (0.81 vs test 0.77)"]), "limits data");
  card(s, 5.15, 1.3, 4.35, 3.1, "Quantum scope", bl(["Noise-free simulation, 3-6 qubits", "300 training rows vs 6,932 classical", "No shot noise, no hardware", "Small search budgets (VQC: 8 trials)"]), "limits quantum");

  // 16 conclusion
  s = slide("DARK"); s.addText("Conclusion", { placeholder: "title" });
  text(s, bl(["Leakage-controlled pipeline on real public data, fully reproducible and tested (28 tests)", "Optimization improves quantum models but not beyond tuned classical ones", "Winning hybrid is effectively classical: no evidence of quantum advantage", "Next: more positives (pool PDP years), EFSA / FSSAI data, noise and hardware"]), { x: 0.7, y: 2.5, w: 8.6, h: 2.6, fontSize: 18, color: "FFFFFF", paraSpaceAfter: 8, objectName: "conclusion bullets" });

  // 17 references
  s = slide(); s.addText("Key references", { placeholder: "title" });
  text(s, bl(["Havlicek et al., Supervised learning with quantum-enhanced feature spaces, Nature 567 (2019)", "Schuld and Killoran, Quantum machine learning in feature Hilbert spaces, PRL 122 (2019)", "Kubler, Buchholz, Scholkopf, The inductive bias of quantum kernels, NeurIPS 2021", "Thanasilp et al., Exponential concentration in quantum kernel methods, Nat. Commun. 15 (2024)", "Bowles, Ahmed, Schuld, Better than classical? arXiv:2403.07059 (2024)", "Akiba et al., Optuna, KDD 2019", "USDA AMS, Pesticide Data Program 2022 database; 40 CFR Part 180"]), { x: 0.5, y: 1.3, w: 9, h: 3.7, fontSize: 14, paraSpaceAfter: 8, objectName: "references" });

  const file = path.join(ROOT, "docs", "Presentation.pptx");
  await pres.writeFile({ fileName: file });
  await applyTheme(file, THEME);
  console.log("wrote", file);
})();
