"""Replay a dataset as a stream of fixed-size batches with injected drift."""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modelguard.data import CATEGORICAL, NUMERIC, POSITIVE, TARGET

FEATURES = NUMERIC + CATEGORICAL
DRIFT_TYPES = {"none", "sudden_shift", "gradual_shift", "label_shift", "missing_feature"}


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    batch_size: int
    n_batches: int
    label_delay: int
    drift: dict

    @property
    def start_batch(self) -> int | None:
        return self.drift.get("start_batch")


@dataclass(frozen=True)
class Batch:
    batch_id: str
    index: int
    X: pd.DataFrame
    y: pd.Series


def load_scenario(path) -> Scenario:
    sc = Scenario(**yaml.safe_load(Path(path).read_text()))
    if sc.drift.get("type") not in DRIFT_TYPES:
        raise ValueError(f"unknown drift type {sc.drift.get('type')!r}")
    if sc.drift["type"] != "none" and sc.start_batch is None:
        raise ValueError("drifting scenarios need drift.start_batch")
    return sc


def batch_id(name: str, index: int) -> str:
    return f"{name}-{index:03d}"


def stream(live: pd.DataFrame, ref: pd.DataFrame, sc: Scenario) -> Iterator[Batch]:
    """Yield `n_batches` batches from a seeded shuffle of `live`; drift starts at `sc.start_batch`.
    Covariate shifts are numeric-only. shortcut: label_shift batches resample from the whole pool, so rows
    can repeat across batches; upgrade if disjointness matters."""
    d, n = sc.drift, sc.batch_size
    kind, start = d["type"], sc.start_batch
    if sc.n_batches * n > len(live):
        raise ValueError(f"need {sc.n_batches * n} live rows, have {len(live)}")
    rng = np.random.default_rng(sc.seed)
    pool = live.iloc[rng.permutation(len(live))].reset_index(drop=True)
    is_pos = (pool[TARGET] == POSITIVE).to_numpy()
    pos, neg = np.flatnonzero(is_pos), np.flatnonzero(~is_pos)
    for i in range(sc.n_batches):
        active = kind != "none" and i >= start
        if active and kind == "label_shift":
            k = round(n * d["positive_rate"])
            idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])
            rng.shuffle(idx)
            b = pool.iloc[idx]
        else:
            b = pool.iloc[i * n:(i + 1) * n]
        X, y = b[FEATURES].reset_index(drop=True), b[TARGET].reset_index(drop=True)
        if active and kind in ("sudden_shift", "gradual_shift"):
            scale = 1.0 if kind == "sudden_shift" else min(1.0, (i - start + 1) / d["ramp_batches"])
            X[d["feature"]] = X[d["feature"]] + scale * d["magnitude"] * ref[d["feature"]].std()
        if active and kind == "missing_feature":
            X[d["feature"]] = X[d["feature"]].where(rng.random(n) >= d["rate"])
        yield Batch(batch_id(sc.name, i), i, X, y)
