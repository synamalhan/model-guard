"""Read-only data layer for the dashboard: SQLite in, DataFrames out. No Streamlit here."""
import pandas as pd

from modelguard import store

P_FLOOR = 1e-6


def list_scenarios(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT scenario FROM ground_truth ORDER BY scenario")]


def _results(con, scenario: str) -> pd.DataFrame:
    if scenario not in list_scenarios(con):
        raise ValueError(f"unknown scenario {scenario!r}")
    return pd.read_sql_query(
        "SELECT batch_index, monitor, feature, statistic, p_value, threshold, flagged FROM monitor_results"
        " WHERE scenario=? ORDER BY batch_index", con, params=(scenario,)).astype({"flagged": bool})


def features(con, scenario: str) -> list[str]:
    r = _results(con, scenario)
    return sorted(r.loc[r["monitor"].isin(["psi", "ks", "chi2"]), "feature"].unique())


def drift_series(con, scenario: str, feature: str) -> pd.DataFrame:
    """PSI, KS and chi-square rows for one feature. `value` is what to plot against `threshold`:
    the PSI score, or the p-value for KS/chi2 (flagged when below the alpha threshold).
    Plotted p-values are floored at P_FLOOR so a log axis over 1e-6..1 keeps the useful range readable;
    the unfloored value stays in `p_value`."""
    r = _results(con, scenario)
    r = r[(r["feature"] == feature) & r["monitor"].isin(["psi", "ks", "chi2"])].copy()
    r["value"] = r["p_value"].clip(lower=P_FLOOR).fillna(r["statistic"])
    return r.reset_index(drop=True)


def missing_matrix(con, scenario: str) -> pd.DataFrame:
    """Long format (feature, batch_index, rate), one cell per feature per batch."""
    r = _results(con, scenario)
    return r.loc[r["monitor"] == "missing", ["feature", "batch_index", "statistic"]].rename(columns={"statistic": "rate"}).reset_index(drop=True)


def performance_series(con, scenario: str) -> pd.DataFrame:
    """Rolling accuracy/F1 per batch; batches before the first labels arrive have no rows."""
    r = _results(con, scenario)
    return r[r["monitor"].str.startswith("rolling_")].reset_index(drop=True)


def alert_feed(con, scenario: str) -> pd.DataFrame:
    _results(con, scenario)  # validates the scenario
    feed = store.read_alerts(con, scenario)[["batch_index", "batch_id", "monitor", "feature", "reason", "timestamp"]]
    feed["timestamp"] = pd.to_datetime(feed["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S UTC")  # stored as UTC ISO
    return feed


def drift_start(con, scenario: str) -> int | None:
    gt = store.read_ground_truth(con, scenario)
    if gt is None:
        raise ValueError(f"unknown scenario {scenario!r}")
    return gt["start_batch"]


def summary(con, scenario: str) -> dict:
    """Headline numbers for one scenario. `delay` counts batches from the true drift start to the first
    alert at or after it; alerts before the start (or any alert on a clean stream) are `false_alarms`."""
    gt = store.read_ground_truth(con, scenario)
    if gt is None:
        raise ValueError(f"unknown scenario {scenario!r}")
    feed, start, r = alert_feed(con, scenario), gt["start_batch"], _results(con, scenario)
    hits = feed["batch_index"] if start is None else feed.loc[feed["batch_index"] >= start, "batch_index"]
    first = None if start is None or hits.empty else int(hits.min())
    return {
        "drift_type": gt["drift_type"],
        "drift_feature": gt["params"].get("feature"),
        "n_batches": 0 if r.empty else int(r["batch_index"].max()) + 1,  # 0 = simulated but monitors not run yet
        "drift_start": start,
        "n_alerts": len(feed),
        "first_alert": first,
        "delay": None if first is None else first - start,
        "false_alarms": len(feed) if start is None else int((feed["batch_index"] < start).sum()),
    }
