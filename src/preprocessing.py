"""Cleaning, feature engineering, splitting and imbalance/outlier analysis.

Model inputs are limited to attributes that are known when a sample is collected
(commodity, origin, marketing claim, facility type, state, month, quantity, ...).
No analytical result (concentration, LOD, number of detects, laboratory) is used,
so the label cannot leak into the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from . import config

CAT_LOW = ["COMMOD", "ORIGIN", "DISTTYPE", "COMMTYPE", "CLAIM", "STATE"]
CAT_HIGH = ["COUNTRY", "GROWST", "PACKST", "DISTST", "COMMOD_ORIGIN", "COMMOD_DISTTYPE"]
NUMERIC = ["QUANTITY_LOG", "QTY_MISSING", "MONTH_SIN", "MONTH_COS", "IS_IMPORT", "IS_ORGANIC"]
RAW_FEATURES = ["COMMOD", "ORIGIN", "DISTTYPE", "COMMTYPE", "CLAIM", "STATE", "COUNTRY",
                "GROWST", "PACKST", "DISTST", "QUANTITY", "MONTH"]
# Columns that exist in the sample table but are deliberately NOT used as features.
EXCLUDED = {
    "SAMPLE_PK": "row identifier",
    "YEAR": "constant (22) apart from 25 samples",
    "DAY": "day of month carries no agronomic meaning",
    "SITE": "arbitrary site code, near-unique per sample",
    "SOURCE_ID": "technical code used to make the sample id unique",
    "VARIETY": "free text; also records days held in lab for P90 samples",
    "label_recomputed": "derived from analytical results (leakage)",
}


def clean_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardise blanks, fix types and remove duplicates. Returns (clean_df, log)."""
    log: dict = {"rows_in": int(len(df))}
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    for c in ["COUNTRY", "GROWST", "PACKST", "DISTST"]:
        df[c] = df[c].replace("", "NONE")
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    df["label_recomputed"] = pd.to_numeric(df["label_recomputed"], errors="coerce").astype(int)
    log["missing_before"] = {c: int(df[c].isna().sum()) for c in ["QUANTITY", "MONTH"]}
    feature_cols = RAW_FEATURES + ["label"]
    dup = df.duplicated(subset=feature_cols, keep=False)
    # Rows identical in every model input AND label are exact duplicates for modelling.
    log["rows_with_identical_features_and_label"] = int(dup.sum())
    log["note"] = ("Samples with identical metadata are different physical samples (distinct "
                   "SAMPLE_PK), so they are kept. Exact duplicate PDP records do not exist "
                   "(see data_quality_report.json: duplicate_rows_excluding_pk = 0).")
    log["rows_out"] = int(len(df))
    return df, log


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (stateless) feature engineering - safe to apply before splitting."""
    df = df.copy()
    df["QUANTITY_LOG"] = np.log1p(df["QUANTITY"])
    df["QTY_MISSING"] = df["QUANTITY"].isna().astype(float)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12.0)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12.0)
    df["IS_IMPORT"] = (df["ORIGIN"] == "2").astype(float)
    df["IS_ORGANIC"] = (df["CLAIM"] == "PO").astype(float)
    df["COMMOD_ORIGIN"] = df["COMMOD"] + "_" + df["ORIGIN"]
    df["COMMOD_DISTTYPE"] = df["COMMOD"] + "_" + df["DISTTYPE"]
    return df


def outlier_report(df: pd.DataFrame) -> dict:
    """IQR outlier analysis for the only continuous input (QUANTITY)."""
    q = df["QUANTITY"].dropna()
    q1, q3 = q.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = q[(q < lo) | (q > hi)]
    return {"feature": "QUANTITY", "n_non_missing": int(len(q)), "q1": float(q1), "q3": float(q3),
            "iqr_lower_fence": float(lo), "iqr_upper_fence": float(hi),
            "n_outliers": int(len(out)), "max": float(q.max()),
            "treatment": "kept; log1p transform and later scaling limit their influence "
                         "(they are valid unit counts, not measurement errors)"}


def imbalance_report(df: pd.DataFrame) -> dict:
    vc = df["label"].value_counts().sort_index()
    by_commod = df.groupby("COMMOD")["label"].agg(["size", "sum", "mean"]).rename(
        columns={"size": "n", "sum": "n_above", "mean": "rate"})
    by_origin = df.groupby("ORIGIN")["label"].agg(["size", "sum", "mean"]).rename(
        columns={"size": "n", "sum": "n_above", "mean": "rate"})
    return {"class_counts": {int(k): int(v) for k, v in vc.items()},
            "imbalance_ratio_neg_to_pos": float(vc.get(0, 0) / max(vc.get(1, 1), 1)),
            "positive_rate_by_commodity": by_commod.round(4).to_dict("index"),
            "positive_rate_by_origin": by_origin.round(4).to_dict("index"),
            "handling": ("class_weight='balanced' for classical models; class-balanced "
                         "training subsample for quantum models; evaluation on the untouched, "
                         "imbalanced test set with balanced accuracy, macro-F1, ROC-AUC, PR-AUC")}


def split_data(df: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split (65 / 15 / 20 by default)."""
    s = config.CFG["split"]
    train_val, test = train_test_split(df, test_size=s["test_size"], stratify=df["label"],
                                       random_state=seed)
    val_frac = s["val_size"] / (1 - s["test_size"])
    train, val = train_test_split(train_val, test_size=val_frac, stratify=train_val["label"],
                                  random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def make_feature_builder(view: str = "target", seed: int = config.SEED) -> ColumnTransformer:
    """Fit-on-train feature builder.

    view='onehot' : one-hot of every categorical field (about 100 columns).
    view='target' : one out-of-fold target-encoded column per categorical field
                    (18 dense columns) - compact input for quantum circuits.
    """
    cats = CAT_LOW + CAT_HIGH
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    if view == "onehot":
        cat_enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15,
                                sparse_output=False)
    elif view == "target":
        cat_enc = TargetEncoder(target_type="binary", smooth="auto",
                                cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    else:
        raise ValueError(view)
    ct = ColumnTransformer([("cat", cat_enc, cats), ("num", num_pipe, NUMERIC)],
                           verbose_feature_names_out=False)
    ct.set_output(transform="default")
    return ct


def prepare(seed: int = config.SEED, save: bool = True):
    """Load labelled samples, clean, add features, split, write analysis reports."""
    from .data_loader import load_labeled
    raw = load_labeled()
    clean, log = clean_samples(raw)
    feat = add_derived(clean)
    parts = split_data(feat, seed)
    if save:
        mdir = config.path("metrics_dir")
        ddir = config.path("processed_dir")
        report = {"cleaning": log, "outliers": outlier_report(feat),
                  "imbalance_full": imbalance_report(feat),
                  "excluded_columns": EXCLUDED,
                  "model_inputs": {"categorical": CAT_LOW + CAT_HIGH, "numeric": NUMERIC},
                  "split_sizes": {k: {"n": int(len(v)), "positives": int(v["label"].sum()),
                                      "positive_rate": float(v["label"].mean())}
                                  for k, v in parts.items()}}
        config.save_json(report, mdir / "preprocessing_report.json")
        for k, v in parts.items():
            v.to_parquet(ddir / f"{k}.parquet", index=False)
    return parts


def load_splits() -> dict[str, pd.DataFrame]:
    d = config.path("processed_dir")
    if not (d / "train.parquet").exists():
        return prepare()
    return {k: pd.read_parquet(d / f"{k}.parquet") for k in ["train", "val", "test"]}


def xy(df: pd.DataFrame):
    return df[CAT_LOW + CAT_HIGH + NUMERIC], df["label"].to_numpy()
