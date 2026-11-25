"""Load the USDA Pesticide Data Program (PDP) 2022 database and build sample-level labels.

Data source (see config/default.json and README):
  USDA Agricultural Marketing Service, Pesticide Data Program, 2022 database ZIP.
  Official page: https://www.ams.usda.gov/datasets/pdp
  Official file: https://www.ams.usda.gov/sites/default/files/media/2022PDPDatabase.zip

Label definition (sample level)
-------------------------------
PDP annotates every analytical result with a regulatory flag. A sample is labelled
``1 = above permissible limits`` when at least one of its results carries
  X  = residue exceeds the EPA tolerance
  V  = residue with no EPA tolerance on that commodity (presumptive violation)
  QV = residue below the quantifiable level, with no tolerance
and ``0 = within permissible limits`` otherwise (no residue, or every residue is
inside its tolerance / exempt). The tolerances are the pesticide- and
commodity-specific limits stored in the PDP ``ToleranceRef`` table (EPA tolerances,
the US equivalent of FSSAI/Codex MRLs).

The label is cross-checked by an independent recomputation from concentration
and tolerance (``verify_label``). Concentration is NEVER used as a model input.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

from . import config

SAMPLE_COLS = ["SAMPLE_PK", "STATE", "YEAR", "MONTH", "DAY", "SITE", "COMMOD", "SOURCE_ID",
               "VARIETY", "ORIGIN", "COUNTRY", "DISTTYPE", "COMMTYPE", "CLAIM", "QUANTITY",
               "GROWST", "PACKST", "DISTST"]
RESULT_COLS = ["SAMPLE_PK", "COMMOD", "COMMTYPE", "LAB", "PESTCODE", "TESTCLASS", "CONCEN",
               "LOD", "CONUNIT", "CONFMETHOD", "CONFMETHOD2", "ANNOTATE", "QUANTITATE", "MEAN",
               "EXTRACT", "DETERMIN"]


def sha256(file: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_zip(dest: Path, urls: list[str]) -> str:
    """Download the PDP archive from the first URL that works. Returns the URL used."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return url
        except Exception as e:  # network errors, 404, blocked hosts
            errors.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("could not download the PDP archive:\n  " + "\n  ".join(errors) +
                       "\nDownload it manually and place it at " + str(dest))


def ensure_raw_data() -> Path:
    """Download (if needed), extract and checksum the PDP zip. Returns the raw directory.

    The expected SHA-256 in config/default.json is that of the file used for the reported
    results. A different checksum is reported as a warning, because the official site may
    re-publish the file; the row counts are checked in data_quality_report.json."""
    raw_dir = config.path("raw_dir")
    zip_path = config.ROOT / config.CFG["paths"]["raw_zip"]
    if not (raw_dir / "PDP22Samples.txt").exists():
        if not zip_path.exists():
            ds = config.CFG["dataset"]
            used = download_zip(zip_path, [ds["official_url"], ds["download_mirror_url"]])
            print(f"downloaded PDP archive from {used}")
        digest = sha256(zip_path)
        if digest != config.CFG["dataset"]["expected_sha256_of_zip"]:
            print(f"WARNING: SHA-256 {digest} differs from the one used for the reported results")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
    return raw_dir


def load_samples(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or ensure_raw_data()
    return pd.read_csv(raw_dir / "PDP22Samples.txt", sep="|", header=None, names=SAMPLE_COLS,
                       dtype=str, keep_default_na=False, encoding="latin-1")


def load_results(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or ensure_raw_data()
    return pd.read_csv(raw_dir / "PDP22Results.txt", sep="|", header=None, names=RESULT_COLS,
                       dtype=str, keep_default_na=False, encoding="latin-1")


def _sheet_table(wb: xlrd.Book, name: str, header_row: int) -> pd.DataFrame:
    sh = wb.sheet_by_name(name)
    header = [str(c.value) for c in sh.row(header_row)]
    rows = [[c.value for c in sh.row(i)] for i in range(header_row + 1, sh.nrows)]
    return pd.DataFrame(rows, columns=header)


def load_reference_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """EPA tolerances plus code -> name look-ups from the official reference workbook."""
    raw_dir = raw_dir or ensure_raw_data()
    wb = xlrd.open_workbook(str(raw_dir / "PDP ReferenceTables 2022.xls"))
    tol = _sheet_table(wb, "Tolerance", 3)
    tol.columns = ["PESTCODE", "COMMOD", "EPATOL", "TOLUNIT", "NOTE", "COMMENT"]
    for c in ["PESTCODE", "COMMOD", "EPATOL", "TOLUNIT", "NOTE", "COMMENT"]:
        tol[c] = tol[c].astype(str).str.strip()
    tol = tol[tol.PESTCODE != ""].drop_duplicates(["PESTCODE", "COMMOD"])

    commod = _sheet_table(wb, "Commodity", 4).iloc[:, :2]
    commod.columns = ["COMMOD", "COMMOD_NAME"]
    country = _sheet_table(wb, "Country", 3)
    country.columns = ["COUNTRY", "COUNTRY_NAME"]
    pest = _sheet_table(wb, "Pest Code", 3).iloc[:, :3]
    pest.columns = ["PESTCODE", "PEST_NAME", "TESTCLASS"]
    for df in (commod, country, pest):
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return {"tolerance": tol, "commodity": commod, "country": country, "pesticide": pest}


def _tol_to_ppm(tol: pd.DataFrame) -> pd.DataFrame:
    tol = tol.copy()
    val = pd.to_numeric(tol["EPATOL"], errors="coerce")
    unit = tol["TOLUNIT"].map({"M": 1.0, "B": 1e-3}).fillna(1.0)
    tol["TOL_PPM"] = val * unit
    tol["TOL_KIND"] = np.where(val.notna(), "numeric",
                               np.where(tol["EPATOL"] == "NT", "none",
                                        np.where(tol["EPATOL"] == "EX", "exempt", "other")))
    return tol


def build_sample_table(samples: pd.DataFrame, results: pd.DataFrame, tol: pd.DataFrame,
                       violation_codes=("X", "V", "QV")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labelled samples, per-sample residue summary used only for analysis)."""
    res = results.copy()
    res["CONC"] = pd.to_numeric(res["CONCEN"], errors="coerce")
    res["LOD_N"] = pd.to_numeric(res["LOD"], errors="coerce")
    res["detect"] = res["CONC"].fillna(0) > 0
    res["official_violation"] = res["ANNOTATE"].isin(violation_codes)

    tolp = _tol_to_ppm(tol)[["PESTCODE", "COMMOD", "TOL_PPM", "TOL_KIND"]]
    res = res.merge(tolp, on=["PESTCODE", "COMMOD"], how="left")
    res["recomputed_exceeds"] = (res["TOL_KIND"] == "numeric") & (res["CONC"] > res["TOL_PPM"])
    res["recomputed_no_tol"] = res["detect"] & (res["TOL_KIND"] == "none")
    res["recomputed_violation"] = res["recomputed_exceeds"] | res["recomputed_no_tol"]

    grp = res.groupby("SAMPLE_PK")
    summary = pd.DataFrame({
        "n_analytes_tested": grp.size(),
        "n_detects": grp["detect"].sum(),
        "n_official_violations": grp["official_violation"].sum(),
        "n_recomputed_violations": grp["recomputed_violation"].sum(),
        "max_conc_over_tol": (res["CONC"] / res["TOL_PPM"]).replace([np.inf], np.nan)
                              .groupby(res["SAMPLE_PK"]).max(),
    }).reset_index()

    out = samples.merge(summary[["SAMPLE_PK", "n_official_violations", "n_recomputed_violations"]],
                        on="SAMPLE_PK", how="left")
    out["label"] = (out["n_official_violations"].fillna(0) > 0).astype(int)
    out["label_recomputed"] = (out["n_recomputed_violations"].fillna(0) > 0).astype(int)
    out = out.drop(columns=["n_official_violations", "n_recomputed_violations"])
    return out, summary


def verify_label(df: pd.DataFrame) -> dict:
    """Agreement between the official-annotation label and the independent recomputation."""
    a, b = df["label"], df["label_recomputed"]
    both = int(((a == 1) & (b == 1)).sum())
    only_official = int(((a == 1) & (b == 0)).sum())
    only_recomputed = int(((a == 0) & (b == 1)).sum())
    return {
        "samples": int(len(df)),
        "violations_official_annotation": int(a.sum()),
        "violations_recomputed_from_concentration": int(b.sum()),
        "both": both,
        "only_official": only_official,
        "only_recomputed": only_recomputed,
        "agreement_rate": float((a == b).mean()),
        "note": ("Official PDP annotations are the training label. The recomputation compares "
                 "detected concentration with the EPA tolerance and is reported only as a "
                 "consistency check; differences come from analytical rules such as "
                 "measurement-uncertainty handling and below-quantification annotations."),
    }


def data_quality_report(samples: pd.DataFrame, results: pd.DataFrame, labelled: pd.DataFrame) -> dict:
    rep = {
        "n_samples": int(len(samples)),
        "n_result_rows": int(len(results)),
        "n_sample_columns": int(samples.shape[1]),
        "duplicate_sample_pk": int(samples["SAMPLE_PK"].duplicated().sum()),
        "duplicate_rows_excluding_pk": int(samples.drop(columns=["SAMPLE_PK"]).duplicated().sum()),
        "samples_without_results": int((~samples["SAMPLE_PK"].isin(results["SAMPLE_PK"])).sum()),
        "orphan_result_rows": int((~results["SAMPLE_PK"].isin(samples["SAMPLE_PK"])).sum()),
        "blank_rate_per_column": {c: float((samples[c].str.strip() == "").mean())
                                   for c in samples.columns},
        "class_counts": {int(k): int(v) for k, v in labelled["label"].value_counts().items()},
        "positive_rate": float(labelled["label"].mean()),
        "unique_commodities": int(samples["COMMOD"].nunique()),
        "result_unit_values": results["CONUNIT"].value_counts().to_dict(),
        "annotation_counts": results["ANNOTATE"].replace("", "none").value_counts().to_dict(),
        "dataset_sha256": None,
    }
    zp = config.ROOT / config.CFG["paths"]["raw_zip"]
    if zp.exists():
        rep["dataset_sha256"] = sha256(zp)
        rep["sha256_matches_config"] = rep["dataset_sha256"] == config.CFG["dataset"]["expected_sha256_of_zip"]
    return rep


def build_and_save() -> pd.DataFrame:
    """End-to-end data build used by ``main.py data``."""
    raw_dir = ensure_raw_data()
    samples, results = load_samples(raw_dir), load_results(raw_dir)
    ref = load_reference_tables(raw_dir)
    labelled, summary = build_sample_table(samples, results, ref["tolerance"],
                                           tuple(config.CFG["dataset"]["violation_annotations"]))
    out_dir = config.path("processed_dir")
    labelled.to_csv(out_dir / "samples_labeled.csv", index=False)
    summary.to_csv(out_dir / "sample_residue_summary_ANALYSIS_ONLY.csv", index=False)
    for name, df in ref.items():
        df.to_csv(out_dir / f"ref_{name}.csv", index=False)
    rep = data_quality_report(samples, results, labelled)
    rep["label_verification"] = verify_label(labelled)
    config.save_json(rep, config.path("metrics_dir") / "data_quality_report.json")
    return labelled


def load_labeled() -> pd.DataFrame:
    f = config.path("processed_dir") / "samples_labeled.csv"
    if not f.exists():
        build_and_save()
    return pd.read_csv(f, dtype=str, keep_default_na=False).assign()
