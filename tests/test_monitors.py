import numpy as np
import pandas as pd
import pytest

from modelguard.monitors import (
    categorical_drift,
    load_config,
    missing_rate,
    numeric_drift,
    run_drift,
)

CFG = load_config()


def rng():
    return np.random.default_rng(0)


def by(results):
    return {r.monitor: r for r in results}


def test_numeric_no_drift_not_flagged():
    r = rng()
    out = by(numeric_drift(pd.Series(r.normal(0, 1, 5000)), pd.Series(r.normal(0, 1, 2000)), CFG, "x"))
    assert not out["psi"].flagged and not out["ks"].flagged


def test_numeric_drift_flagged():
    r = rng()
    out = by(numeric_drift(pd.Series(r.normal(0, 1, 5000)), pd.Series(r.normal(1, 1, 2000)), CFG, "x"))
    assert out["psi"].flagged and out["ks"].flagged and out["ks"].p_value < CFG["drift"]["alpha"]


def test_numeric_constant_column_and_nan_ok():
    ref = pd.Series([1.0] * 100 + [np.nan] * 5)
    out = by(numeric_drift(ref, pd.Series([1.0] * 50), CFG, "c"))
    assert not out["psi"].flagged


def test_categorical_no_drift_not_flagged():
    r = rng()
    p = [0.5, 0.3, 0.2]
    out = by(categorical_drift(pd.Series(r.choice(list("abc"), 5000, p=p)), pd.Series(r.choice(list("abc"), 2000, p=p)), CFG, "c"))
    assert not out["psi"].flagged and not out["chi2"].flagged


def test_categorical_drift_flagged():
    r = rng()
    ref = pd.Series(r.choice(list("abc"), 5000, p=[0.5, 0.3, 0.2]))
    live = pd.Series(r.choice(list("abc"), 2000, p=[0.2, 0.3, 0.5]))
    out = by(categorical_drift(ref, live, CFG, "c"))
    assert out["psi"].flagged and out["chi2"].flagged


def test_categorical_unseen_values_flagged():
    ref = pd.Series(list("ab") * 500)
    out = by(categorical_drift(ref, pd.Series(["z"] * 200), CFG, "c"))
    assert out["psi"].flagged and out["chi2"].flagged


def test_missing_rate():
    r = rng()
    ref = pd.Series(r.normal(size=1000))
    ok = pd.Series(r.normal(size=1000))
    bad = ok.copy()
    bad[:300] = np.nan
    assert not missing_rate(ref, ok, CFG, "x").flagged
    res = missing_rate(ref, bad, CFG, "x")
    assert res.flagged and res.statistic == pytest.approx(0.3)


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        numeric_drift(pd.Series([1.0, 2.0]), pd.Series([], dtype=float), CFG, "x")
    with pytest.raises(ValueError):
        missing_rate(pd.Series([1.0]), pd.Series([], dtype=float), CFG, "x")


def test_run_drift_covers_every_feature(synthetic):
    X = synthetic.drop(columns="income")
    res = run_drift(X.iloc[:200], X.iloc[200:], CFG)
    assert len(res) == 6 * 2 + 8 * 2 + 14


def test_rare_categories_merged_not_flagged():
    r = rng()
    cats, p = list("abcdefghij"), [0.5, 0.3, 0.1, 0.05, 0.03, 0.01] + [0.0025] * 4
    ref = pd.Series(r.choice(cats, 20000, p=p))
    live = pd.Series(r.choice(cats, 500, p=p))
    assert not any(x.flagged for x in categorical_drift(ref, live, CFG, "c"))
