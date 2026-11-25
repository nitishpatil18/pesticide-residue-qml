import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def splits():
    from src import preprocessing as P
    return P.load_splits()


@pytest.fixture(scope="session")
def small_train(splits):
    from src.hybrid_model import balanced_subsample
    return balanced_subsample(splits["train"], 40, 7)
