"""Per-feature monitors comparing a live batch to the reference window."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from modelguard.data import CATEGORICAL, NUMERIC

EPS = 1e-4
OTHER = "__other__"


@dataclass(frozen=True)
class MonitorResult:
    monitor: str
    feature: str
    statistic: float
    p_value: float | None  # None for score-only metrics (PSI, rates)
    flagged: bool


def load_config(path: str = "configs/monitors.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _psi(ref_p: np.ndarray, live_p: np.ndarray) -> float:
    ref_p, live_p = np.clip(ref_p, EPS, None), np.clip(live_p, EPS, None)
    return float(np.sum((live_p - ref_p) * np.log(live_p / ref_p)))


def _check(ref: pd.Series, live: pd.Series) -> tuple[pd.Series, pd.Series]:
    ref, live = ref.dropna(), live.dropna()
    if ref.empty or live.empty:
        raise ValueError("reference and live batch each need at least one non-null value")
    return ref, live


def numeric_drift(ref: pd.Series, live: pd.Series, cfg: dict, name: str | None = None) -> list[MonitorResult]:
    name = name or str(ref.name)
    ref, live = _check(ref, live)
    d = cfg["drift"]
    # Quantile edges come from the reference only; duplicates (ties) collapse.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, d["bins"] + 1)[1:-1]))
    k = len(edges) + 1
    ref_p = np.bincount(np.digitize(ref, edges), minlength=k) / len(ref)
    live_p = np.bincount(np.digitize(live, edges), minlength=k) / len(live)
    psi = _psi(ref_p, live_p)
    ks = stats.ks_2samp(ref, live)
    return [
        MonitorResult("psi", name, psi, None, psi > d["psi_threshold"]),
        MonitorResult("ks", name, float(ks.statistic), float(ks.pvalue), bool(ks.pvalue < d["alpha"])),
    ]


def categorical_drift(ref: pd.Series, live: pd.Series, cfg: dict, name: str | None = None) -> list[MonitorResult]:
    name = name or str(ref.name)
    ref, live = _check(ref, live)
    d = cfg["drift"]
    # Rare and unseen categories share OTHER: tiny expected counts make chi-square unreliable.
    share = ref.value_counts(normalize=True)
    cats = sorted(share[share >= d["min_category_share"]].index) + [OTHER]
    ref_c = ref.where(ref.isin(cats), OTHER).value_counts().reindex(cats, fill_value=0).to_numpy()
    live_c = live.where(live.isin(cats), OTHER).value_counts().reindex(cats, fill_value=0).to_numpy()
    ref_p = (ref_c + 1) / (ref_c.sum() + len(cats))  # add-one smoothing: no zero expected counts
    live_p = live_c / live_c.sum()
    psi = _psi(ref_p, live_p)
    chi = stats.chisquare(live_c, ref_p * live_c.sum())
    return [
        MonitorResult("psi", name, psi, None, psi > d["psi_threshold"]),
        MonitorResult("chi2", name, float(chi.statistic), float(chi.pvalue), bool(chi.pvalue < d["alpha"])),
    ]


def missing_rate(ref: pd.Series, live: pd.Series, cfg: dict, name: str | None = None) -> MonitorResult:
    name = name or str(ref.name)
    if ref.empty or live.empty:
        raise ValueError("reference and live batch must be non-empty")
    rate = float(live.isna().mean())
    return MonitorResult("missing", name, rate, None, rate - float(ref.isna().mean()) > cfg["missing"]["margin"])


def run_drift(ref: pd.DataFrame, live: pd.DataFrame, cfg: dict) -> list[MonitorResult]:
    out = []
    for c in NUMERIC:
        out += numeric_drift(ref[c], live[c], cfg)
    for c in CATEGORICAL:
        out += categorical_drift(ref[c], live[c], cfg)
    for c in NUMERIC + CATEGORICAL:
        out.append(missing_rate(ref[c], live[c], cfg))
    return out
