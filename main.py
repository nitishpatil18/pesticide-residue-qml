"""Command line entry point.

    python main.py data        # build labelled dataset, cleaning report and splits
    python main.py classical   # tune + evaluate Logistic Regression, SVM, Random Forest, XGBoost
    python main.py optimize    # KTA screening, Optuna search for QSVM / hybrid / VQC, matched controls
    python main.py evaluate    # evaluate the five model groups on validation and test data
    python main.py ablation    # ablation study of the hybrid framework
    python main.py scaling     # learning curve over training-set size
    python main.py stats       # bootstrap CIs and paired AUC differences
    python main.py figures     # all result figures and diagrams
    python main.py all         # everything above, in order
"""
from __future__ import annotations

import argparse
import sys

STAGES = ["data", "classical", "optimize", "evaluate", "ablation", "scaling", "stats", "figures"]


def run(stage: str) -> None:
    from src import runner
    if stage == "data":
        runner.stage_data()
    elif stage == "classical":
        runner.stage_classical()
    elif stage == "optimize":
        runner.stage_optimize()
    elif stage == "evaluate":
        runner.stage_evaluate()
    elif stage == "ablation":
        runner.stage_ablation()
    elif stage == "scaling":
        runner.stage_scaling()
    elif stage == "stats":
        from src import statistics_tests
        statistics_tests.run()
    elif stage == "figures":
        from src import visualization
        visualization.make_all()
    else:
        raise SystemExit(f"unknown stage {stage}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    args = ap.parse_args(argv)
    for st in (STAGES if args.stage == "all" else [args.stage]):
        print(f"=== stage: {st} ===", flush=True)
        run(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
