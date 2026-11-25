# Quantum Machine Learning for Pesticide Residue Classification

Pre-screening of food samples for the risk of pesticide residues above permissible limits, using
classical, quantum (QSVM, VQC) and hybrid quantum-classical models on the real USDA Pesticide Data
Program (PDP) 2022 database, with an optimization-driven hybrid framework and an ablation study.

> **Scope.** This is a risk-screening research prototype. It does not replace certified laboratory
> pesticide residue testing and must not be used to approve or reject a consignment.

## Headline result (honest summary)

| Group | Model | Test ROC-AUC |
|---|---|---|
| Classical, tuned (full training split) | Random forest | 0.800 |
| Classical, tuned, same compact inputs as quantum | Logistic regression | 0.787 |
| Basic quantum | QSVM | 0.733 |
| Basic hybrid | Hybrid kernel | 0.765 |
| Optimized quantum | QSVM | 0.772 |
| Optimized hybrid | Hybrid kernel | 0.772 |
| Basic / optimized VQC | VQC | 0.667 / 0.632 |

Optimization improves the quantum models, but no quantum or hybrid model beats tuned classical
models. The ablation shows the hybrid's gain comes from hyper-parameter tuning and its classical RBF
component; the winning quantum kernel is non-entangling and classically simulable. **No quantum
advantage is claimed.** Only 64 test samples are positive, so confidence intervals are wide
(about ±0.06 AUC); see `results/metrics/bootstrap_ci.json` and the report.

## Dataset

- **USDA AMS Pesticide Data Program, 2022 database** (public domain): 10,665 samples,
  2,890,842 residue results, EPA tolerance table. Official page: https://www.ams.usda.gov/datasets/pdp
  (file: https://www.ams.usda.gov/sites/default/files/media/2022PDPDatabase.zip).
- **Label:** 1 if any result of the sample has the official annotation X (exceeds tolerance),
  V (no tolerance) or QV (below quantifiable level, no tolerance); 320 of 10,665 samples (3.0%).
  Recomputing violations from concentration and tolerance covers all 320 (337 recomputed).
- **Inputs:** collection metadata only (commodity, origin, claim, states, month, quantity and
  derived fields). No concentration or other residue-derived field is used, so there is no target leakage.
- **Regulatory framework:** EPA tolerances (40 CFR Part 180) apply to this US data. FSSAI and Codex
  limits are the Indian/international equivalents; sample-level Indian data are not public, so they
  are not applied here.
- **Provenance caveat:** the build environment could not reach ams.usda.gov, so the zip was fetched
  from a public GitHub mirror (see `config/default.json`). Row counts match the official data
  dictionary. SHA-256 of the file used: `9e599c741c6122d1ea7f6ec7ed0509ed5c2dbe34084cd5531b05303cafa433af`.
  Verify it against your own download of the official file with `shasum -a 256 data/raw/2022PDPDatabase.zip`.

## Architecture

```
PDP samples + results + EPA tolerances
  -> label builder (X / V / QV)           src/data_loader.py
  -> preprocessing, stratified split      src/preprocessing.py
  -> feature optimisation (MI / PCA, angle scaling)   src/feature_selection.py
  -> classical models                     src/classical_models.py
  -> quantum + hybrid models              src/quantum_models.py, src/hybrid_model.py
  -> staged optimisation (KTA + Optuna)   src/optimization.py
  -> evaluation, ablation, scaling, bootstrap   src/runner.py, src/evaluation.py, src/statistics_tests.py
  -> figures + diagrams                   src/visualization.py
  -> Streamlit app                        app.py
```

Diagrams are in `docs/diagrams/` (system, preprocessing, classical, quantum circuit, hybrid, optimization).

## Folder structure

```
app.py               Streamlit application
main.py              command line for every pipeline stage
config/default.json  seed, paths, dataset URLs, search budgets
src/                 library code (one module per concern)
tests/               unit and integration tests (30)
notebooks/           01_colab_quickstart.ipynb
data/raw, data/processed   dataset (raw zip is downloaded on first run)
experiments/         Optuna trials and best configurations (csv / json)
results/metrics      all metrics, comparison table, ablation, scaling, bootstrap CIs
results/figures      result figures
results/models       saved models (joblib)
docs/                Technical_Report.docx, Presentation.pptx, Viva_QA.docx, diagrams/, build/
logs/                run logs
WORKLOG.md           progress, decisions, failures, open limitations
```

## Prerequisites and installation

Python 3.13 was used for every reported run (Qiskit Machine Learning 0.9 needs Qiskit 2.x).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ranges
# pip install -r requirements-lock.txt   # the exact versions used
```

Google Colab: open `notebooks/01_colab_quickstart.ipynb`.

## Run the pipeline

```bash
python main.py data        # download (official URL, then mirror), checksum, label, split
python main.py classical   # tune and evaluate LR, SVM, RF, XGBoost
python main.py optimize    # KTA screening, Optuna searches, matched classical controls
python main.py evaluate    # five comparison groups over training subsamples, save models
python main.py ablation    # A0-A9 and B1
python main.py scaling     # learning curve
python main.py stats       # bootstrap CIs and paired AUC differences
python main.py figures     # figures and diagrams
python main.py all         # everything above in order
```

All stages are seeded (seed 42; subsamples 42-46). Saved results are already in `results/` and
`experiments/`, so the app and figures work without re-running the searches.

## Streamlit app

```bash
streamlit run app.py
```

Pages: introduction and disclaimer, dataset and visualization, classify a sample (choose a saved
model or retrain; prediction uses the validation-calibrated threshold and shows the risk percentile
against test scores), classical vs quantum vs hybrid comparison, optimization results, circuits and architecture.

## Tests

```bash
python -m pytest -q tests
```

Covers data loading and download fallback, label logic, leakage guards, preprocessing, model
behaviour (including pickling a trained VQC), optimization, saved-model loading, result consistency
and an automated run of the Streamlit app.

## Limitations

- 64 positive test samples: many model differences are not statistically resolved.
- Quantum models are noise-free simulations with 3-6 qubits on 300 training samples; no shot noise or hardware.
- One dataset, one year (US market). Features describe provenance, not the sample's actual residue, so precision at 3% prevalence is low.
- Search budgets are small (VQC: 8 trials). CV scores of searched models are optimistic.
- Dataset file obtained from a mirror; official checksum not verified.

## References

See the reference list in `docs/Technical_Report.docx`.
