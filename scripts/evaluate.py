"""Evaluation. `tune` uses seeds 0-9 only; `report` uses seeds 10-29 only. Run via `make eval`."""
import json
import sys
from pathlib import Path

import joblib
import yaml

from modelguard.data import load, split_reference
from modelguard.evaluation import REPORT_SEEDS, TUNE_SEEDS, collect, report, tune
from modelguard.monitors import load_config
from modelguard.simulator import load_scenario

OUT = Path("results/eval")


def main(mode: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, ecfg = load_config(), yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    n = cfg["alerts"]["consecutive"]
    model, baseline = joblib.load("models/baseline.joblib"), json.loads(Path("results/baseline.json").read_text())
    ref, live = split_reference(load())
    scenarios = [load_scenario(p) for p in sorted(Path("configs/scenarios").glob("*.yaml"))]
    if mode == "tune":
        stats = collect("tune", TUNE_SEEDS, scenarios, live, ref, model, baseline, cfg)
        stats.to_csv(OUT / "tune_stats.csv.gz", index=False)
        thresholds, grid = tune(stats, ecfg, n)
        grid.to_csv(OUT / "tune_grid.csv", index=False)
        (OUT / "tuned_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        print(json.dumps(thresholds["settings"], indent=2))
    elif mode == "report":
        path = OUT / "tuned_thresholds.json"
        if not path.exists():
            print("run `evaluate.py tune` first: report needs tuned thresholds", file=sys.stderr)
            return 1
        stats = collect("report", REPORT_SEEDS, scenarios, live, ref, model, baseline, cfg)
        stats.to_csv(OUT / "report_stats.csv.gz", index=False)
        runs, alerts = report(stats, json.loads(path.read_text()))
        runs.to_csv(OUT / "runs.csv", index=False)
        alerts.to_csv(OUT / "alerts.csv", index=False)
        print(f"wrote {OUT/'runs.csv'} ({len(runs)} rows) and {OUT/'alerts.csv'} ({len(alerts)} rows)")
    else:
        print("usage: evaluate.py tune|report", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
