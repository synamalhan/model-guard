"""SQLite prediction log. Features are stored as JSON; labels arrive later."""
import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    features TEXT NOT NULL,
    prediction TEXT NOT NULL,
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_batch ON predictions(batch_id);
CREATE TABLE IF NOT EXISTS ground_truth (
    scenario TEXT PRIMARY KEY,
    drift_type TEXT NOT NULL,
    start_batch INTEGER,
    params TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitor_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    monitor TEXT NOT NULL,
    feature TEXT NOT NULL,
    statistic REAL NOT NULL,
    p_value REAL,
    threshold REAL NOT NULL,
    flagged INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_monitor_results_scenario ON monitor_results(scenario, batch_index);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    monitor TEXT NOT NULL,
    feature TEXT NOT NULL,
    consecutive INTEGER NOT NULL,
    reason TEXT NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def log_batch(con, batch_id: str, features: pd.DataFrame, predictions, timestamp: str | None = None) -> None:
    """Log one batch. Row order of `features` and `predictions` must match."""
    if len(features) != len(predictions):
        raise ValueError("features and predictions differ in length")
    ts = timestamp or datetime.now(UTC).isoformat()
    rows = [
        (batch_id, ts, json.dumps(rec, default=str), str(p))
        for rec, p in zip(features.to_dict("records"), predictions)
    ]
    with con:
        con.executemany(
            "INSERT INTO predictions (batch_id, timestamp, features, prediction) VALUES (?,?,?,?)", rows
        )


def read_batch(con, batch_id: str) -> pd.DataFrame:
    """Return the batch in insert order: id, batch_id, timestamp, prediction, label + feature columns."""
    df = pd.read_sql_query(
        "SELECT id, batch_id, timestamp, features, prediction, label FROM predictions WHERE batch_id=? ORDER BY id",
        con,
        params=(batch_id,),
    )
    feats = pd.DataFrame([json.loads(f) for f in df.pop("features")], index=df.index)
    return pd.concat([df, feats], axis=1)


def set_labels(con, batch_id: str, labels) -> None:
    """Fill labels for a batch, in the same row order as it was logged."""
    ids = [r[0] for r in con.execute("SELECT id FROM predictions WHERE batch_id=? ORDER BY id", (batch_id,))]
    if len(ids) != len(labels):
        raise ValueError(f"batch {batch_id} has {len(ids)} rows, got {len(labels)} labels")
    with con:
        con.executemany("UPDATE predictions SET label=? WHERE id=?", [(str(l), i) for l, i in zip(labels, ids)])


def read_log(con, scenario: str) -> pd.DataFrame:
    """id, prediction, label for every row of a scenario's batches (batch ids are '<scenario>-NNN')."""
    return pd.read_sql_query(
        "SELECT id, prediction, label FROM predictions WHERE substr(batch_id, 1, ?) = ? ORDER BY id",
        con, params=(len(scenario) + 1, scenario + "-"))


def record_ground_truth(con, scenario: str, drift: dict) -> None:
    """One ground-truth row per scenario per database (re-recording replaces it)."""
    with con:
        con.execute("INSERT OR REPLACE INTO ground_truth VALUES (?,?,?,?)",
                    (scenario, drift["type"], drift.get("start_batch"), json.dumps(drift)))


def read_ground_truth(con, scenario: str) -> dict | None:
    row = con.execute("SELECT drift_type, start_batch, params FROM ground_truth WHERE scenario=?", (scenario,)).fetchone()
    return None if row is None else {"drift_type": row[0], "start_batch": row[1], "params": json.loads(row[2])}


def log_alert(con, scenario: str, batch_id: str, batch_index: int, monitor: str, feature: str,
              consecutive: int, reason: str) -> None:
    with con:
        con.execute(
            "INSERT INTO alerts (scenario, batch_id, batch_index, timestamp, monitor, feature, consecutive, reason)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (scenario, batch_id, batch_index, datetime.now(UTC).isoformat(), monitor, feature, consecutive, reason))


def read_alerts(con, scenario: str) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM alerts WHERE scenario=? ORDER BY id", con, params=(scenario,))


def log_monitor_results(con, scenario: str, batch_id: str, batch_index: int, results_with_thresholds) -> None:
    """Persist one batch of (MonitorResult, threshold) pairs so a UI can chart them without recomputing."""
    rows = [(scenario, batch_id, batch_index, r.monitor, r.feature, r.statistic, r.p_value, t, int(r.flagged))
            for r, t in results_with_thresholds]
    with con:
        con.executemany(
            "INSERT INTO monitor_results (scenario, batch_id, batch_index, monitor, feature, statistic, p_value,"
            " threshold, flagged) VALUES (?,?,?,?,?,?,?,?,?)", rows)
