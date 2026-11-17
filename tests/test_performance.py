import pandas as pd

from modelguard.monitors import load_config
from modelguard.performance import rolling_metrics

CFG = load_config()
BASE = {"holdout_accuracy": 0.85, "holdout_f1": 0.66}


def log(wrong_every, n=1000):
    """Labels alternate; every `wrong_every`-th prediction is flipped."""
    labels = [">50K" if i % 2 else "<=50K" for i in range(n)]
    preds = [("<=50K" if l == ">50K" else ">50K") if i % wrong_every == 0 else l for i, (l) in enumerate(labels)]
    return pd.DataFrame({"id": range(n), "prediction": preds, "label": labels})


def test_healthy_not_flagged():
    out = {r.monitor: r for r in rolling_metrics(log(8), BASE, CFG)}  # 87.5% accurate
    assert not out["rolling_accuracy"].flagged and not out["rolling_f1"].flagged


def test_degraded_flagged():
    df = log(8).assign(prediction="<=50K")  # always majority class: 50% accurate, F1 = 0
    out = {r.monitor: r for r in rolling_metrics(df, BASE, CFG)}
    assert out["rolling_accuracy"].flagged and out["rolling_f1"].flagged


def test_window_uses_latest_rows():
    df = pd.concat([log(2, 1000), log(100, 1000).assign(id=lambda d: d["id"] + 1000)])
    out = {r.monitor: r for r in rolling_metrics(df, BASE, CFG)}  # last 500 rows are healthy
    assert not out["rolling_accuracy"].flagged


def test_no_labels_returns_empty():
    df = log(2).assign(label=None)
    assert rolling_metrics(df, BASE, CFG) == []
