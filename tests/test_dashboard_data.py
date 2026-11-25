from dataclasses import replace

import pytest

from modelguard import dashboard_data as dd
from modelguard import store
from modelguard.data import CATEGORICAL, NUMERIC, split_reference
from modelguard.monitors import load_config
from modelguard.runner import run_scenario
from modelguard.simulator import load_scenario
from modelguard.train import train

CFG = load_config()
SC = replace(load_scenario("configs/scenarios/sudden_shift.yaml"), batch_size=150)


@pytest.fixture(scope="module")
def db_file(big, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "demo.db"
    model, acc, f1 = train(big)
    ref, live = split_reference(big)
    con = store.connect(str(path))
    run_scenario(SC, live, ref, model, {"holdout_accuracy": acc, "holdout_f1": f1}, CFG, con)
    con.close()
    return path


@pytest.fixture
def con(db_file):
    c = store.connect(str(db_file))
    yield c
    c.close()


def test_scenarios_and_drift_start(con):
    assert dd.list_scenarios(con) == ["sudden_shift"]
    assert dd.drift_start(con, "sudden_shift") == 15


def test_drift_series_has_thresholds_and_flags_the_shift(con):
    s = dd.drift_series(con, "sudden_shift", "age")
    assert set(s["monitor"]) == {"psi", "ks"}
    psi = s[s["monitor"] == "psi"]
    assert len(psi) == SC.n_batches and (psi["threshold"] == CFG["drift"]["psi_threshold"]).all()
    assert not psi[psi["batch_index"] < 15]["flagged"].any() and psi[psi["batch_index"] >= 15]["flagged"].all()
    ks = s[s["monitor"] == "ks"]
    assert (ks["value"] == ks["p_value"].clip(lower=dd.P_FLOOR)).all() and (ks["threshold"] == CFG["drift"]["alpha"]).all()


def test_missing_matrix_is_one_cell_per_feature_per_batch(con):
    m = dd.missing_matrix(con, "sudden_shift")
    assert len(m) == (len(NUMERIC) + len(CATEGORICAL)) * SC.n_batches
    assert m["rate"].between(0, 1).all()


def test_performance_series_starts_when_labels_arrive(con):
    p = dd.performance_series(con, "sudden_shift")
    assert set(p["monitor"]) == {"rolling_accuracy", "rolling_f1"}
    assert p["batch_index"].min() == SC.label_delay  # batch 0's labels arrive at batch K


def test_alert_feed_has_reasons(con):
    a = dd.alert_feed(con, "sudden_shift")
    assert not a.empty and a["reason"].str.contains("consecutive").all() and (a["batch_index"] >= 15).all()
    assert a["timestamp"].str.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC").all()


def test_features_lists_drift_monitored_columns(con):
    assert dd.features(con, "sudden_shift") == sorted(NUMERIC + CATEGORICAL)


def test_unknown_scenario_raises(con):
    for fn in (dd.features, dd.missing_matrix, dd.performance_series, dd.alert_feed, dd.drift_start):
        with pytest.raises(ValueError):
            fn(con, "nope")
    with pytest.raises(ValueError):
        dd.drift_series(con, "nope", "age")


def test_plotted_p_values_are_floored_but_raw_kept(con):
    ks = dd.drift_series(con, "sudden_shift", "age").query("monitor == 'ks'")
    assert ks["value"].min() == dd.P_FLOOR and ks["p_value"].min() < dd.P_FLOOR  # drifted batches floor out
    assert ks["value"].max() <= 1 and (ks["value"] >= dd.P_FLOOR).all()


def test_summary_counts_delay_from_true_start(con):
    s = dd.summary(con, "sudden_shift")
    feed = dd.alert_feed(con, "sudden_shift")
    assert s["drift_start"] == 15 and s["n_batches"] == SC.n_batches and s["n_alerts"] == len(feed)
    assert s["first_alert"] == feed["batch_index"].min() and s["delay"] == s["first_alert"] - 15
    assert s["false_alarms"] == 0 and s["drift_type"] == "sudden_shift"
    with pytest.raises(ValueError):
        dd.summary(con, "nope")


def test_summary_of_scenario_without_monitor_results(tmp_path):
    c = store.connect(str(tmp_path / "sim.db"))
    c.execute("INSERT INTO ground_truth (scenario, drift_type, start_batch, params) VALUES ('clean', 'none', NULL, '{}')")
    s = dd.summary(c, "clean")
    assert s["n_batches"] == 0 and s["n_alerts"] == 0 and s["delay"] is None and s["drift_feature"] is None
