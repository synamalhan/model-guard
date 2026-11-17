import math

import numpy as np
import pandas as pd
import pytest

from modelguard import evaluation as ev
from modelguard.data import split_reference
from modelguard.monitors import load_config
from modelguard.simulator import Scenario
from modelguard.train import train

COLS = ["batch_index", "monitor", "feature", "statistic", "p_value", "flagged"]


def mk(scenario, seed, start, n_batches, rows):
    df = pd.DataFrame(rows, columns=COLS)
    return df.assign(scenario=scenario, seed=seed, drift_start=np.nan if start is None else start, n_batches=n_batches, batch_size=100)


def psi_rows(batches, stat):
    return [(b, "psi", "age", stat, np.nan, stat > 0.2) for b in batches]


# --- seed split ---------------------------------------------------------------------------------------

def test_seed_guards():
    ev.check_seeds("tune", range(10))
    ev.check_seeds("report", range(10, 30))
    with pytest.raises(ValueError):
        ev.check_seeds("tune", [9, 10])
    with pytest.raises(ValueError):
        ev.check_seeds("report", [5])
    assert set(ev.TUNE_SEEDS).isdisjoint(ev.REPORT_SEEDS) and len(ev.REPORT_SEEDS) == 20


def test_tune_and_report_reject_wrong_seeds():
    ecfg = {"psi_grid": [0.2], "alpha_grid": [0.01], "false_alarm_budget_per_100": 1.0}
    with pytest.raises(ValueError):
        ev.tune(mk("d", 10, 5, 12, psi_rows(range(12), 1.0)), ecfg, 3)
    ok = {"tuned_on_seeds": list(ev.TUNE_SEEDS), "consecutive": 3, "settings": {}}
    with pytest.raises(ValueError):
        ev.report(mk("d", 5, 5, 12, psi_rows(range(12), 1.0)), ok)
    bad = {"tuned_on_seeds": [0, 1], "consecutive": 3, "settings": {}}
    with pytest.raises(ValueError):
        ev.report(mk("d", 10, 5, 12, psi_rows(range(12), 1.0)), bad)


# --- maths --------------------------------------------------------------------------------------------

def test_wilson():
    lo, hi = ev.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert ev.wilson(0, 20)[0] == 0 and ev.wilson(20, 20)[1] == 1
    assert math.isnan(ev.wilson(0, 0)[0])


def test_apply_setting_rethresholds_only_drift_statistics():
    rows = [(0, "psi", "age", 0.5, np.nan, True), (0, "ks", "age", 0.4, 0.001, True), (0, "chi2", "sex", 5.0, 0.2, False),
            (0, "missing", "age", 0.3, np.nan, True), (0, "rolling_f1", "model", 0.5, np.nan, False)]
    s = mk("d", 0, 5, 12, rows)
    names = lambda df: sorted(df["monitor"])
    assert names(ev.apply_setting(s, "psi", 0.3, None)) == ["missing", "psi"]
    assert names(ev.apply_setting(s, "psi", 0.6, None)) == ["missing"]
    assert names(ev.apply_setting(s, "ks", None, 0.01)) == ["ks", "missing"]
    assert names(ev.apply_setting(s, "ks", None, 0.0001)) == ["missing"]  # p=0.001 no longer breaches
    assert names(ev.apply_setting(s, "both", 0.3, 0.01)) == ["ks", "missing", "psi"]


def test_metrics_on_a_hand_built_run():
    # false streak at batches 0-2 (alert at 2, before drift), real drift from batch 5 (alert at 7)
    stats = mk("d", 0, 5, 12, psi_rows([0, 1, 2], 1.0) + psi_rows(range(3, 5), 0.0) + psi_rows(range(5, 12), 1.0))
    runs, alerts = ev.evaluate(stats, "psi", 0.2, None, 3)
    r = runs.iloc[0]
    assert (r.n_alerts, r.n_correct, r.detected, r.first_correct, r.delay) == (2, 1, True, 7, 2)
    assert alerts["correct"].tolist() == [False, True]
    s = ev.summarize(runs).iloc[0]
    assert (s.recall, s.precision, s.delay_mean) == (1.0, 0.5, 2.0)


def test_undetected_run_and_clean_false_alarm_rate():
    runs, _ = ev.evaluate(mk("d", 0, 5, 12, psi_rows(range(12), 0.0)), "psi", 0.2, None, 3)
    s = ev.summarize(runs).iloc[0]
    assert s.recall == 0 and math.isnan(s.delay_mean) and math.isnan(s.precision)
    runs, _ = ev.evaluate(mk("c", 0, None, 40, psi_rows(range(10, 13), 1.0)), "psi", 0.2, None, 3)
    s = ev.summarize(runs).iloc[0]
    assert (s.alerts, s.clean_batches, s.false_alarms_per_100) == (1, 40, 2.5)


def test_tune_picks_most_conservative_threshold_within_budget():
    clean = [mk("clean", sd, None, 40, psi_rows(range(40), 0.12)) for sd in (0, 1)]  # breaches only at thr 0.10
    drift = [mk("shift", sd, 15, 40, psi_rows(range(15), 0.0) + psi_rows(range(15, 40), 1.0)) for sd in (0, 1)]
    ecfg = {"psi_grid": [0.10, 0.15, 0.20], "alpha_grid": [0.05, 0.01], "false_alarm_budget_per_100": 1.0}
    th, grid = ev.tune(pd.concat(clean + drift), ecfg, 3)
    assert th["tuned_on_seeds"] == list(ev.TUNE_SEEDS)
    assert th["settings"]["psi"]["psi_threshold"] == 0.20 and th["settings"]["psi"]["within_budget"]
    over = grid[(grid["setting"] == "psi") & (grid["psi_threshold"] == 0.10)].iloc[0]
    assert not over.within_budget and over.false_alarms_per_100 == 2.5


# --- end to end on synthetic data ----------------------------------------------------------------------

def test_pipeline_end_to_end_small(big):
    cfg = load_config()
    model, acc, f1 = train(big)
    base = {"holdout_accuracy": acc, "holdout_f1": f1}
    ref, live = split_reference(big)
    drift = {"type": "sudden_shift", "start_batch": 3, "feature": "age", "magnitude": 1.0}
    scs = [Scenario("clean", 0, 150, 8, 2, {"type": "none"}), Scenario("shift", 0, 150, 8, 2, drift)]
    with pytest.raises(ValueError):
        ev.collect("tune", [10], scs, live, ref, model, base, cfg, log=lambda *_: None)
    quiet = {"log": lambda *_: None}
    tune_stats = ev.collect("tune", [0, 1], scs, live, ref, model, base, cfg, **quiet)
    assert set(tune_stats["seed"]) == {0, 1} and tune_stats.loc[tune_stats.scenario == "clean", "drift_start"].isna().all()
    ecfg = {"psi_grid": [0.2], "alpha_grid": [0.01], "false_alarm_budget_per_100": 1.0}
    th, _ = ev.tune(tune_stats, ecfg, 3)
    rep_stats = ev.collect("report", [10, 11], scs, live, ref, model, base, cfg, **quiet)
    runs, _ = ev.report(rep_stats, th)
    assert len(runs) == 2 * 2 * 3 and set(runs["setting"]) == set(ev.SETTINGS)
    block = ev.render(runs, th)
    assert "10-11" in block and "| shift | psi |" in block and "False alarms" in block
