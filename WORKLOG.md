# WORKLOG

Chronological record of progress, technical decisions, failures and open limitations.
Times are local (IST), 2026-11-25.

## Environment and constraints (read first)

- The build was started inside the user's project folder on their Mac through a remote shell.
  That shell could only reach PyPI and GitHub (Kaggle, UCI, USDA, EFSA, Zenodo, Mendeley were all
  refused by the egress proxy). After the Mac's bridge connection dropped, the work was
  continued in an isolated cloud workspace (Python 3.13, 2 CPU cores, 7 GB RAM) and the
  finished project is delivered as a zip file. All numbers in this project come from runs in
  that workspace; nothing was copied from papers.
- Dataset bytes were downloaded in both places and have the same SHA-256
  (`9e599c74...a433af`), see `results/metrics/data_quality_report.json`.
- Every experiment is seeded (`config/default.json`, seed 42; subsample seeds 42-46).

## Phase 1 - research and dataset selection (done)

Searched Kaggle/UCI/government/Zenodo sources by web search. Findings:

| Candidate | Verdict |
|---|---|
| USDA AMS Pesticide Data Program (PDP) 2022 database | **Selected.** Sample-level, 10,665 samples, 2.89 M residue results, official EPA tolerance table, official violation annotations, public domain. |
| EFSA monitoring data on Zenodo (SSD2 files) | Real and large, but the hosts are blocked in the build environment, and records are per country/year; kept as future scope. |
| India FSSAI/MPRNL monitoring | Only aggregate statistics are published (sample-level data is not public). The MRL framework is cited; no sample-level data exists to download. |
| Kaggle / UCI "pesticide" datasets | Search found image/spectral/thermal datasets and unrelated tables, not sample-level residue-vs-MRL data. Hosts unreachable anyway. |
| Papers with supplementary tables | Small (tens to hundreds of rows), mostly concentration summaries; not suitable for a train/test split. |

The PDP 2022 zip was obtained from a GitHub repository that mirrors the file
(`raw.githubusercontent.com/bdyapa9/Commodities-and-pesticides-data`), because ams.usda.gov was not
reachable. Provenance checks: the archive contains the official data dictionary, reference tables
and MS-Access database; row counts match the official dictionary exactly (10,665 samples,
2,890,842 results, 6,757-row tolerance table). **A checksum against the official file could not
be verified** (no network path to ams.usda.gov); the README gives the official URL and the SHA-256
so the user can verify on a machine with internet access.

## Phase 2 - problem definition (done)

- Regulatory reference: EPA tolerances stored in the PDP `ToleranceRef` table (pesticide x commodity
  specific). FSSAI/Codex MRLs are the Indian/international analogues but cannot be applied to this
  US sample-level dataset.
- Label: sample is **above permissible limits** if any result is annotated X (exceeds tolerance),
  V (no tolerance) or QV (below quantifiable level, no tolerance); else **within limits**.
- Verification: independent recomputation from concentration and tolerance reproduces all 320
  official violations (337 recomputed positives in total, 17 extra; agreement 99.84%).
- Prevalence: 320 / 10,665 = 3.0% positives. Strong class imbalance is a core property of the task.
- **Leakage control:** features are sample metadata known at collection time (commodity, origin,
  country, claim, facility type, states, month, quantity). No concentration, LOD, detect count,
  laboratory or annotation is used as input. Because the label is defined from residue
  measurements, the task is "pre-screening risk" and cannot be expected to be highly accurate.

## Phase 3-6 - structure, environment, data, preprocessing (done)

- Structure follows the brief; `src/` modules each contain working code (no placeholders).
- Dependencies verified together: qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-machine-learning 0.9.1
  (requires qiskit>=2), scikit-learn 1.9.1, xgboost 3.4.1, optuna 5.1.0, streamlit 1.67.0.
  `requirements.txt` has ranges; `requirements-lock.txt` has the exact versions used.
- Splits: stratified 65/15/20 (train 6,932 / val 1,600 / test 2,133; positives 208 / 48 / 64).
- Test set is used only for final reporting; validation is used only to choose decision thresholds.
- Decision (design): one-hot features for classical models; "target" view (cross-fitted target
  encoding, 18 dense columns) as a compact input that suits few-qubit circuits.
- Decision: quantum models train on a class-balanced subsample of the training split (150+150),
  because kernel evaluation and VQC training cost grows quickly in simulation. The classical
  baselines are reported both on the full training split and under the identical subsample
  and inputs ("matched controls"), so the effect of the quantum circuit can be isolated.
- Decision: search objective is cross-validated ROC-AUC (independent of class ratio of the training
  data), so models trained on full and balanced data are comparable.

## Failures and fixes

- pip install launched with `nohup ... &` in the Mac shell was killed when the call returned and
  a foreground install exceeded the tool timeout; then the Mac connection dropped. Fixed by
  moving to the cloud workspace and installing in the foreground.
- sklearn 1.9 deprecated `TargetEncoder(shuffle=, random_state=)`; replaced by passing a seeded `KFold`
  as `cv`; `requirements.txt` pins scikit-learn>=1.9. Tests re-run after the change.
- XGBoost tuning with `n_jobs=-1` was extremely slow (about 20 minutes with no finish) because two
  background jobs oversubscribed the 2 CPU cores. Both jobs were stopped, XGBoost was set to
  `n_jobs=1`, the already finished Optuna searches were reused (resume support added to the
  classical stage), and XGBoost then finished in about a minute.
- The first run of `main.py figures` drew the system diagram clipped (axis limit assumed the figure
  aspect ratio, not the node coordinates) and with missing edges in three diagrams. Fixed the axis
  limits and edge routing, and added the missing edges; all diagrams were re-rendered and inspected.
- `pytest` found that the two VQC model files saved by the evaluate stage were truncated
  (`EOFError` on load): the qiskit-machine-learning training object cannot be pickled. The VQC now
  stores only its trained weights and hyper-parameters and rebuilds the circuit network on demand
  (verified numerically identical to the training object, max probability difference 0.0).
  The two files were regenerated with the same seed; their AUCs are identical to the recorded
  values (0.7093 and 0.6356), which also confirms reproducibility.
- The data stage did not download the dataset although the documentation said it did. A download
  with official-URL-then-mirror fallback and a checksum report was added and tested (the official
  host is unreachable from the build environment; the mirror download matched the recorded SHA-256).

## Phase 7-10 - experiments (done)

Searches (seed 42, balanced 150+150 subsample, repeated stratified 3x2 CV unless noted):
- KTA screening of 60 random architectures; top 6 kept (experiments/kta_screening.csv).
- QSVM: 30 trials, best CV AUC 0.816 (basic configuration 0.717), 63 s.
- Hybrid: 40 trials, best CV AUC 0.814 (basic 0.784), 78 s. Same budget without the KTA shortlist: 0.808, 126 s.
- VQC: 8 trials, best CV AUC 0.701 (basic 0.668), 337 s.
- Classical: 25 trials each, 5-fold CV on the full training split; tuned CV AUC 0.797-0.801.

## Phase 11-12 - evaluation, ablation, scaling, statistics (done)

Test ROC-AUC (2,133 samples, 64 positive). Quantum and matched models: mean over 5 subsamples (VQC: 3).
- Classical tuned (full train): LogReg 0.775, SVM 0.772, RF 0.800, XGBoost 0.790 (defaults 0.745 / 0.686 / 0.713 / 0.752).
- Matched classical (300 train, same compact inputs): LogReg 0.787, SVM 0.774, RF 0.781, XGBoost 0.775.
- Basic quantum: QSVM 0.733, VQC 0.667. Basic hybrid: 0.765.
- Optimized quantum: QSVM 0.772, VQC 0.632. Optimized hybrid: 0.772.
- Ablation: hyper-parameter optimization alone 0.771 (all three 0.772); feature-only 0.754,
  circuit-only 0.755, features + circuit 0.735; hybrid with alpha = 0 (pure RBF) 0.770; alpha = 1 (pure quantum) 0.691;
  Optuna without the KTA shortlist 0.752.
- Bootstrap (2,000 resamples) 95% CIs are about +-0.06 wide; tuned RF is not significantly better
  than XGBoost, matched LogReg or the optimized QSVM, and is significantly better than the optimized hybrid and both VQCs.
- Learning curve: all curves flatten at about 300 training rows; no sign of a quantum data-efficiency advantage.

## Phase 13-16 - app, tests, documents (done)

- Streamlit app tested with Streamlit's AppTest; 30 tests pass (`python -m pytest -q tests`).
- Documents in docs/: Technical_Report.docx (22 sections + references), Presentation.pptx (17 slides),
  Viva_QA.docx; generator scripts in docs/build (Node, optional). Every number in them is read from
  results/ and experiments/ files via docs/build/results_summary.json.

## Unresolved limitations (honest list)

1. Only 64 positives in the test set: most differences between top models are within noise.
2. The winning quantum kernel (Z feature map) has no entanglement and is classically simulable; the
   quantum component adds no measurable gain over the classical RBF kernel in the hybrid.
3. Quantum models are noise-free simulations on 3-6 qubits with 300 training rows. No hardware, no shot noise.
4. Search budgets are small; the VQC search overfit its CV estimate (CV 0.701 vs test 0.632).
5. Selection bias: best CV AUCs (0.81) are higher than test AUCs (0.77).
6. Dataset obtained from a GitHub mirror; row counts match the official dictionary but the official checksum is unverified.
7. EFSA and FSSAI sample-level data were not available; only US PDP 2022 was used.
8. Features are provenance metadata, so precision at 3% prevalence is low (0.06-0.28).
9. The project was built in a cloud workspace (the user's Mac connection dropped), and is delivered as a zip file.
