"""Run registry: one JSON record per simulation run with everything needed to reproduce it."""
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from modelguard.simulator import Scenario

LIBS = ["pandas", "numpy", "scipy", "scikit-learn", "PyYAML", "streamlit", "altair", "modelguard"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        return out
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    return out


def write_run_record(sc: Scenario, cfg: dict, db_path: str, out_dir: str = "results/runs") -> Path:
    """Write results/runs/<run_id>.json. The commit is 'unknown' outside a git repo or before the first commit."""
    now = datetime.now(UTC)
    run_id = f"{sc.name}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    status = _git("status", "--porcelain")
    record = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "scenario": asdict(sc),
        "seed": sc.seed,
        "monitor_config": cfg,
        "database": db_path,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "versions": versions(),
    }
    path = Path(out_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
