import numpy as np

from src import preprocessing as P


def test_splits_are_disjoint_stratified(splits):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert len(tr) + len(va) + len(te) == 10665
    ids = [set(x["SAMPLE_PK"]) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [x["label"].mean() for x in (tr, va, te)]
    assert max(rates) - min(rates) < 0.005


def test_no_leaky_columns_in_features():
    X_cols = set(P.CAT_LOW + P.CAT_HIGH + P.NUMERIC)
    for bad in ["label", "label_recomputed", "SAMPLE_PK", "CONCEN", "LOD", "ANNOTATE", "SITE"]:
        assert bad not in X_cols


def test_feature_builders_fit_on_train_only(splits):
    Xtr, ytr = P.xy(splits["train"])
    Xte, _ = P.xy(splits["test"])
    for view, width in (("onehot", None), ("target", 18)):
        b = P.make_feature_builder(view)
        Z = b.fit_transform(Xtr, ytr)
        Zt = b.transform(Xte)
        assert Z.shape[0] == len(Xtr) and Zt.shape[1] == Z.shape[1]
        assert np.isfinite(Z).all() and np.isfinite(Zt).all()
        if width:
            assert Z.shape[1] == width


def test_derived_features_row_wise():
    import pandas as pd
    df = pd.DataFrame({"QUANTITY": [np.nan, 3.0], "MONTH": [1, 7], "ORIGIN": ["2", "1"],
                       "CLAIM": ["PO", "NC"], "COMMOD": ["TO", "BB"], "DISTTYPE": ["R", "D"]})
    out = P.add_derived(df)
    assert out["IS_IMPORT"].tolist() == [1.0, 0.0] and out["IS_ORGANIC"].tolist() == [1.0, 0.0]
    assert out["QTY_MISSING"].tolist() == [1.0, 0.0]
    assert out["COMMOD_ORIGIN"].tolist() == ["TO_2", "BB_1"]
