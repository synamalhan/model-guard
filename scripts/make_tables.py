"""Generate results/tables/results.md and splice it into the README from results/eval only."""
import json
from pathlib import Path

import pandas as pd

from modelguard.evaluation import render, splice

RUNS, THRESHOLDS, README = Path("results/eval/runs.csv"), Path("results/eval/tuned_thresholds.json"), Path("README.md")


def main() -> None:
    block = render(pd.read_csv(RUNS), json.loads(THRESHOLDS.read_text()))
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/results.md").write_text(block + "\n")
    README.write_text(splice(README.read_text(), block))
    print("wrote results/tables/results.md and updated README")


if __name__ == "__main__":
    main()
