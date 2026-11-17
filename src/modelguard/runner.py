"""Run one scenario end to end: stream, predict, log, delayed labels, monitors, alerts."""
import pandas as pd

from modelguard import store
from modelguard.alerts import AlertEngine
from modelguard.monitors import run_drift
from modelguard.performance import rolling_metrics
from modelguard.simulator import Scenario, batch_id, stream


def threshold(r, ref: pd.DataFrame, baseline: dict, cfg: dict) -> float:
    """The value a result is compared against: PSI/p-value/rate ceilings, or a metric floor."""
    d, p = cfg["drift"], cfg["performance"]
    if r.monitor == "psi":
        return d["psi_threshold"]
    if r.monitor in ("ks", "chi2"):
        return d["alpha"]
    if r.monitor == "missing":
        return float(ref[r.feature].isna().mean()) + cfg["missing"]["margin"]
    return baseline["holdout_" + r.monitor.removeprefix("rolling_")] - p["max_drop"]


def monitored_batches(sc: Scenario, live: pd.DataFrame, ref: pd.DataFrame, model, baseline: dict, cfg: dict, con):
    """Stream a scenario: log predictions, fill delayed labels, and yield (batch, monitor results) per batch."""
    truth: dict[int, list] = {}
    for b in stream(live, ref, sc):
        store.log_batch(con, b.batch_id, b.X, model.predict(b.X))
        truth[b.index] = b.y.tolist()
        k = b.index - sc.label_delay  # labels arrive label_delay batches after their predictions
        if k >= 0:
            store.set_labels(con, batch_id(sc.name, k), truth.pop(k))
        yield b, run_drift(ref, b.X, cfg) + rolling_metrics(store.read_log(con, sc.name), baseline, cfg)


def run_scenario(sc: Scenario, live: pd.DataFrame, ref: pd.DataFrame, model, baseline: dict, cfg: dict, con) -> None:
    """Run one scenario end to end: stream, monitor, persist results, raise alerts."""
    store.record_ground_truth(con, sc.name, sc.drift)
    engine = AlertEngine(cfg["alerts"]["consecutive"])
    for b, results in monitored_batches(sc, live, ref, model, baseline, cfg, con):
        store.log_monitor_results(con, sc.name, b.batch_id, b.index, [(r, threshold(r, ref, baseline, cfg)) for r in results])
        for a in engine.update(b.index, results):
            store.log_alert(con, sc.name, b.batch_id, a.batch_index, a.monitor, a.feature, a.consecutive, a.reason)
