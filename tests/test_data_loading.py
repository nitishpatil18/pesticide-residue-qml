import pandas as pd

from src import config, data_loader as D


def test_raw_files_present_and_checksum():
    zp = config.ROOT / config.CFG["paths"]["raw_zip"]
    assert zp.exists()
    assert D.sha256(zp) == config.CFG["dataset"]["expected_sha256_of_zip"]


def test_samples_and_results_shapes():
    s, r = D.load_samples(), D.load_results()
    assert s.shape == (10665, 18)          # matches the official data dictionary
    assert len(r) == 2890842
    assert s["SAMPLE_PK"].is_unique
    assert set(r["SAMPLE_PK"]) == set(s["SAMPLE_PK"])


def test_tolerance_table():
    tol = D.load_reference_tables()["tolerance"]
    assert len(tol) >= 6000
    assert {"PESTCODE", "COMMOD", "EPATOL"} <= set(tol.columns)


def test_label_counts_and_verification():
    df = D.load_labeled()
    assert len(df) == 10665
    assert pd.to_numeric(df["label"]).sum() == 320
    rep = config.load_json(config.path("metrics_dir") / "data_quality_report.json")
    v = rep["label_verification"]
    assert v["only_official"] == 0            # every official violation is reproduced
    assert v["agreement_rate"] > 0.995


def test_download_zip_falls_back_to_second_url(tmp_path):
    from src import data_loader as dl
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc" * 1000)
    dest = tmp_path / "out" / "x.zip"
    used = dl.download_zip(dest, [(tmp_path / "missing.bin").as_uri(), src.as_uri()])
    assert used == src.as_uri() and dest.read_bytes() == src.read_bytes()
    assert not dest.with_suffix(".part").exists()


def test_download_zip_reports_all_failures(tmp_path):
    import pytest
    from src import data_loader as dl
    with pytest.raises(RuntimeError, match="could not download"):
        dl.download_zip(tmp_path / "x.zip", [(tmp_path / "nope").as_uri()])
