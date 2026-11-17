"""Run every monitor on a clean and a synthetically drifted batch; print a table."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelguard import store
from modelguard.data import CATEGORICAL, NUMERIC, TARGET, load, split_reference
from modelguard.monitors import load_config, run_drift
from modelguard.performance import rolling_metrics

SEED, N = 0, 1000
FEATURES = NUMERIC + CATEGORICAL


def drifted(batch: pd.DataFrame) -> pd.DataFrame:
    """SYNTHETIC drift injected for the demo: shifted age, skewed workclass, nulled occupation."""
    b, rng = batch.copy(), np.random.default_rng(SEED)
    b["age"] += 15
    b.loc[rng.random(len(b)) < 0.6, "workclass"] = "Self-emp-inc"
    b.loc[rng.random(len(b)) < 0.3, "occupation"] = np.nan
    return b


def main() -> None:
    cfg = load_config()
    model = joblib.load("models/baseline.joblib")
    baseline = json.loads(Path("results/baseline.json").read_text())
    ref, live = split_reference(load())
    clean = live.sample(N, random_state=SEED)
    con = store.connect(":memory:")
    rows = []
    for name, batch in [("clean", clean), ("drifted (synthetic)", drifted(clean))]:
        X = batch[FEATURES]
        store.log_batch(con, name, X, model.predict(X))
        store.set_labels(con, name, batch[TARGET].tolist())
        logged = store.read_batch(con, name)
        results = run_drift(ref, X, cfg) + rolling_metrics(logged, baseline, cfg)
        rows += [{"batch": name, **r.__dict__} for r in results]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", None)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nflagged per batch:\n" + df.groupby("batch")["flagged"].sum().to_string())


if __name__ == "__main__":
    main()
