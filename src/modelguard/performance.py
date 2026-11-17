"""Rolling accuracy and F1 (positive class) over the most recent labeled rows."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from modelguard.data import POSITIVE
from modelguard.monitors import MonitorResult


def rolling_metrics(log: pd.DataFrame, baseline: dict, cfg: dict) -> list[MonitorResult]:
    """`log` is store.read_batch output (or a concat); `baseline` has holdout_accuracy and holdout_f1.
    Returns [] when no labels have arrived yet."""
    p = cfg["performance"]
    lab = log[log["label"].notna()].sort_values("id").tail(p["window"])
    if lab.empty:
        return []
    acc = float(accuracy_score(lab["label"], lab["prediction"]))
    f1 = float(f1_score(lab["label"], lab["prediction"], pos_label=POSITIVE, zero_division=0))
    return [
        MonitorResult("rolling_accuracy", "model", acc, None, baseline["holdout_accuracy"] - acc > p["max_drop"]),
        MonitorResult("rolling_f1", "model", f1, None, baseline["holdout_f1"] - f1 > p["max_drop"]),
    ]
