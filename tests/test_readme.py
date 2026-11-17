import json
import re
from pathlib import Path

import pandas as pd

from modelguard.evaluation import README_END, README_START, render, splice

README = Path("README.md").read_text()
RUNS, TH = Path("results/eval/runs.csv"), Path("results/eval/tuned_thresholds.json")


def outside_block(text):
    return text.split(README_START)[0] + text.split(README_END)[1]


def test_readme_has_markers():
    assert README_START in README and README_END in README


def test_no_result_numbers_outside_generated_block():
    assert not re.search(r"\d+\.\d+|\d+\s*%", outside_block(README))


def test_readme_block_is_exactly_the_generated_table():
    block = README.split(README_START)[1].split(README_END)[0].strip()
    if RUNS.exists():
        assert block == render(pd.read_csv(RUNS), json.loads(TH.read_text())).strip()
    else:
        assert "TODO" in block  # rule 4: no numbers until a script produced them


def test_splice_replaces_only_the_block():
    out = splice(f"a\n{README_START}\nold\n{README_END}\nz", "NEW")
    assert out == f"a\n{README_START}\nNEW\n{README_END}\nz"


def test_relative_links_resolve():
    for doc in ("README.md", "docs/writeup.md"):
        base = Path(doc).parent
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", Path(doc).read_text()):
            if not target.startswith(("http://", "https://", "mailto:")):
                assert (base / target).exists(), f"{doc} links to missing {target}"
