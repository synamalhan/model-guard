import json

from modelguard.monitors import load_config
from modelguard.registry import versions, write_run_record
from modelguard.simulator import load_scenario


def test_run_record_has_everything_needed_to_reproduce(tmp_path):
    sc, cfg = load_scenario("configs/scenarios/sudden_shift.yaml"), load_config()
    rec = json.loads(write_run_record(sc, cfg, "data/x.db", out_dir=str(tmp_path)).read_text())
    assert rec["run_id"].startswith("sudden_shift-") and rec["seed"] == sc.seed
    assert rec["scenario"]["drift"]["start_batch"] == 15 and rec["monitor_config"] == cfg
    assert rec["git_commit"]  # a hash, or the string "unknown" before the first commit
    assert all(rec["versions"][k] for k in ("python", "pandas", "scikit-learn", "scipy", "streamlit", "modelguard"))


def test_versions_flags_missing_libs():
    assert "python" in versions()
