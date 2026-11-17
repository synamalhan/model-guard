from dataclasses import replace

import pytest

from modelguard.data import POSITIVE, split_reference
from modelguard.simulator import Scenario, load_scenario, stream

BASE = Scenario("t", seed=7, batch_size=100, n_batches=10, label_delay=2, drift={"type": "none"})


def run(sc, big):
    ref, live = split_reference(big)
    return list(stream(live, ref, sc)), ref


def test_deterministic_and_fixed_size(big):
    a, _ = run(BASE, big)
    b, _ = run(BASE, big)
    assert [len(x.X) for x in a] == [100] * 10
    assert all(x.X.equals(y.X) and x.y.equals(y.y) for x, y in zip(a, b))
    assert a[3].batch_id == "t-003"


def test_different_seed_differs(big):
    a, _ = run(BASE, big)
    b, _ = run(replace(BASE, seed=8), big)
    assert not a[0].X.equals(b[0].X)


def test_sudden_shift_starts_exactly_at_start_batch(big):
    clean, ref = run(BASE, big)
    sc = replace(BASE, drift={"type": "sudden_shift", "start_batch": 4, "feature": "age", "magnitude": 2.0})
    drift, _ = run(sc, big)
    for i in range(10):
        diff = (drift[i].X["age"] - clean[i].X["age"]).mean()
        assert diff == pytest.approx(0 if i < 4 else 2.0 * ref["age"].std())


def test_gradual_shift_ramps_to_full(big):
    clean, ref = run(BASE, big)
    sc = replace(BASE, drift={"type": "gradual_shift", "start_batch": 2, "feature": "age", "magnitude": 1.0, "ramp_batches": 4})
    drift, _ = run(sc, big)
    d = [(drift[i].X["age"] - clean[i].X["age"]).mean() / ref["age"].std() for i in range(10)]
    assert d[:2] == pytest.approx([0, 0]) and d[2:6] == pytest.approx([0.25, 0.5, 0.75, 1.0]) and d[9] == pytest.approx(1.0)


def test_label_shift_changes_positive_rate_only_after_start(big):
    sc = replace(BASE, drift={"type": "label_shift", "start_batch": 5, "positive_rate": 0.9})
    batches, _ = run(sc, big)
    rate = [(b.y == POSITIVE).mean() for b in batches]
    assert all(r < 0.8 for r in rate[:5]) and all(r == pytest.approx(0.9) for r in rate[5:])


def test_missing_injection_rate(big):
    sc = replace(BASE, batch_size=1000, n_batches=6, drift={"type": "missing_feature", "start_batch": 3, "feature": "workclass", "rate": 0.4})
    batches, _ = run(sc, big)
    miss = [b.X["workclass"].isna().mean() for b in batches]
    assert max(miss[:3]) == 0 and all(0.35 < m < 0.45 for m in miss[3:])


def test_validation(big):
    with pytest.raises(ValueError):
        run(replace(BASE, batch_size=10**6), big)
    p = "configs/scenarios/clean.yaml"
    assert load_scenario(p).drift["type"] == "none"
