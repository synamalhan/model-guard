"""Run every scenario in configs/scenarios on real Adult; print detection results computed from the DB."""
import json
from pathlib import Path

import joblib
import pandas as pd

from modelguard import store
from modelguard.data import load, split_reference
from modelguard.monitors import load_config
from modelguard.registry import write_run_record
from modelguard.runner import run_scenario
from modelguard.simulator import load_scenario

DB = Path("data/simulation.db")


def main() -> None:
    cfg, model = load_config(), joblib.load("models/baseline.joblib")
    baseline = json.loads(Path("results/baseline.json").read_text())
    ref, live = split_reference(load())
    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = store.connect(str(DB))
    rows = []
    for path in sorted(Path("configs/scenarios").glob("*.yaml")):
        sc = load_scenario(path)
        run_scenario(sc, live, ref, model, baseline, cfg, con)
        write_run_record(sc, cfg, str(DB))
        gt, alerts = store.read_ground_truth(con, sc.name), store.read_alerts(con, sc.name)
        first = int(alerts["batch_index"].min()) if len(alerts) else None
        rows.append({"scenario": sc.name, "drift_start": gt["start_batch"], "first_alert": first,
                     "delay": None if first is None or gt["start_batch"] is None else first - gt["start_batch"],
                     "alerts": len(alerts), "alerted_on": ", ".join(sorted(set(alerts["monitor"] + ":" + alerts["feature"])))})
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
