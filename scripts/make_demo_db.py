"""Simulate the sudden-shift scenario on real Adult into data/demo.db; fail if no alert fires."""
import json
import sys
from pathlib import Path

import joblib

from modelguard import store
from modelguard.data import load, split_reference
from modelguard.monitors import load_config
from modelguard.registry import write_run_record
from modelguard.runner import run_scenario
from modelguard.simulator import load_scenario

DB = Path("data/demo.db")


def main() -> int:
    cfg, sc = load_config(), load_scenario("configs/scenarios/sudden_shift.yaml")
    ref, live = split_reference(load())
    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = store.connect(str(DB))
    run_scenario(sc, live, ref, joblib.load("models/baseline.joblib"), json.loads(Path("results/baseline.json").read_text()), cfg, con)
    print(f"run record: {write_run_record(sc, cfg, str(DB))}")
    n = len(store.read_alerts(con, sc.name))
    if n == 0:
        print("no alert was raised; refusing to start an empty demo", file=sys.stderr)
        return 1
    print(f"{n} alert(s) stored in {DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
