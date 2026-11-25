"""Feature selection, dimensionality reduction and quantum-friendly scaling."""
from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config

REDUCTION_METHODS = ("none", "kbest", "pca", "kbest_pca")


class AngleScaler(MinMaxScaler):
    """Scale each feature to [0, pi] using training statistics; clip unseen values.

    Rotation-angle encoding is periodic, so values outside the training range are
    clipped rather than wrapped around.
    """

    def __init__(self, high: float = np.pi):
        super().__init__(feature_range=(0.0, high), clip=True)
        self.high = high


def make_reducer_steps(method: str, n_components: int, k_select: int | None = None,
                       seed: int = config.SEED) -> list:
    """Return sklearn pipeline steps that reduce to ``n_components`` features.

    none      : keep the first ``n_components`` columns (only valid if input has that many)
    kbest     : keep the ``n_components`` features with highest mutual information
    pca       : PCA to ``n_components`` components
    kbest_pca : keep ``k_select`` features by mutual information, then PCA
    """
    if method not in REDUCTION_METHODS:
        raise ValueError(f"unknown reduction {method}")
    mi = partial(mutual_info_classif, random_state=seed)
    steps = [("std", StandardScaler())]
    if method == "kbest":
        steps.append(("select", SelectKBest(mi, k=n_components)))
    elif method == "pca":
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    elif method == "kbest_pca":
        steps.append(("select", SelectKBest(mi, k=max(k_select or n_components, n_components))))
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    steps.append(("angles", AngleScaler()))
    return steps
