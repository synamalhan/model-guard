import numpy as np
import pandas as pd
import pytest

from modelguard.data import CATEGORICAL, COLUMNS, NUMERIC, TARGET


def make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """Adult-shaped synthetic frame so tests never need the real download."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    for c in CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], n)
    df[TARGET] = np.where(df["age"] + rng.normal(scale=0.5, size=n) > 0, ">50K", "<=50K")
    return df[COLUMNS]


@pytest.fixture
def synthetic():
    return make_frame(400)


@pytest.fixture(scope="module")
def big():
    return make_frame(10000)
