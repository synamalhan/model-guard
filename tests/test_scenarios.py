"""Acceptance: every configured scenario runs end to end on synthetic data."""
from dataclasses import replace
from pathlib import Path

import pytest

from modelguard import store
from modelguard.data import split_reference
from modelguard.monitors import load_config
from modelguard.runner import run_scenario
from modelguard.simulator import batch_id, load_scenario
from modelguard.train import train

PATHS = sorted(Path("configs/scenarios").glob("*.yaml"))
CFG = load_config()


def test_scenarios_exist():
    assert {p.stem for p in PATHS} >= {"clean", "sudden_shift", "gradual_shift", "label_shift", "missing_feature"}


@pytest.fixture(scope="module")
def trained(big):
    model, acc, f1 = train(big)
    return model, {"holdout_accuracy": acc, "holdout_f1": f1}


def execute(path, big, trained):
    # Smaller batches keep the test fast; synthetic data is stationary and balanced.
    sc = replace(load_scenario(path), batch_size=150)
    ref, live = split_reference(big)
    con = store.connect(":memory:")
    run_scenario(sc, live, ref, trained[0], trained[1], CFG, con)
    return sc, con


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.stem)
def test_scenario_end_to_end(path, big, trained):
    sc, con = execute(path, big, trained)
    gt = store.read_ground_truth(con, sc.name)
    assert gt["start_batch"] == sc.start_batch and gt["drift_type"] == sc.drift["type"]
    assert len(store.read_batch(con, batch_id(sc.name, sc.n_batches - 1))) == 150
    # delayed labels: the last label_delay batches never got theirs
    for i in range(sc.n_batches):
        labeled = store.read_batch(con, batch_id(sc.name, i))["label"].notna().all()
        assert labeled == (i < sc.n_batches - sc.label_delay)
    alerts = store.read_alerts(con, sc.name)
    if sc.start_batch is not None:
        assert (alerts["batch_index"] >= sc.start_batch).all()  # nothing alerts before drift begins


def test_clean_stream_raises_no_alert(big, trained):
    sc, con = execute(Path("configs/scenarios/clean.yaml"), big, trained)
    assert store.read_alerts(con, sc.name).empty


def test_sudden_shift_alerts_with_reason(big, trained):
    sc, con = execute(Path("configs/scenarios/sudden_shift.yaml"), big, trained)
    a = store.read_alerts(con, sc.name)
    assert "age" in set(a["feature"]) and a["reason"].str.contains("consecutive").all()
