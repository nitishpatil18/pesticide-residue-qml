"""Classical baselines: Logistic Regression, SVM (RBF), Random Forest, XGBoost."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .feature_selection import make_reducer_steps
from .preprocessing import make_feature_builder

CLASSICAL_NAMES = ("logreg", "svm", "rf", "xgb")

DEFAULTS = {
    "logreg": {"C": 1.0},
    "svm": {"C": 1.0, "gamma": "scale"},
    "rf": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "xgb": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0,
            "colsample_bytree": 1.0},
}


def make_estimator(name: str, params: dict | None = None, seed: int = config.SEED,
                   pos_weight: float = 1.0):
    p = {**DEFAULTS[name], **(params or {})}
    if name == "logreg":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000,
                                  random_state=seed)
    if name == "svm":
        return SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight="balanced",
                   random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"],
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"], scale_pos_weight=pos_weight,
                             eval_metric="logloss", tree_method="hist", n_jobs=1,
                             random_state=seed, verbosity=0)
    raise ValueError(name)


def build_pipeline(name: str, params: dict | None = None, view: str = "onehot",
                   pos_weight: float = 1.0, seed: int = config.SEED) -> Pipeline:
    """feature builder -> scaler -> estimator (all fitted on training data only)."""
    return Pipeline([("features", make_feature_builder(view, seed)),
                     ("scale", StandardScaler()),
                     ("model", make_estimator(name, params, seed, pos_weight))])


def build_compact_pipeline(name: str, params: dict | None, n_components: int, reduction: str,
                           view: str = "target", k_select: int | None = None,
                           pos_weight: float = 1.0, seed: int = config.SEED) -> Pipeline:
    """Classical model on exactly the reduced inputs the quantum models see (fair comparison)."""
    steps = [("features", make_feature_builder(view, seed))]
    steps += make_reducer_steps(reduction, n_components, k_select, seed)
    steps.append(("model", make_estimator(name, params, seed, pos_weight)))
    return Pipeline(steps)


def suggest_params(name: str, trial) -> dict:
    """Optuna search spaces for tuned classical baselines."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "svm":
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)}
    if name == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}
    raise ValueError(name)
