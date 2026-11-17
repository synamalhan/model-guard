"""Evaluation: run scenarios over seeds, tune thresholds on tuning seeds, report on held-out seeds.

Monitors run once per (scenario, seed); detector settings, thresholds and the alert rule are then applied
offline to the stored statistics, so a grid search costs no extra model or monitor runs."""
import math
from itertools import product

import numpy as np
import pandas as pd

from modelguard import store
from modelguard.alerts import AlertEngine
from modelguard.monitors import MonitorResult
from modelguard.runner import monitored_batches
from modelguard.simulator import Scenario

TUNE_SEEDS = tuple(range(10))
REPORT_SEEDS = tuple(range(10, 30))
ALWAYS = {"missing", "rolling_accuracy", "rolling_f1"}  # on in every setting; settings differ in drift statistics only
FAMILIES = {"psi": {"psi"}, "ks": {"ks", "chi2"}, "both": {"psi", "ks", "chi2"}}
SETTINGS = list(FAMILIES)
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def check_seeds(purpose: str, seeds) -> None:
    """Keep tuning and reporting seeds disjoint. This blocks accidental leakage, not deliberate edits."""
    allowed = {"tune": TUNE_SEEDS, "report": REPORT_SEEDS}[purpose]
    bad = sorted(set(map(int, seeds)) - set(allowed))
    if bad:
        raise ValueError(f"seeds {bad} are not allowed for '{purpose}' (allowed: {allowed[0]}-{allowed[-1]})")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --- raw statistics -------------------------------------------------------------------------------------

def collect_run(sc: Scenario, live, ref, model, baseline: dict, cfg: dict) -> pd.DataFrame:
    """Per-batch monitor statistics for one scenario run (flags use the default config)."""
    con = store.connect(":memory:")
    rows = [(sc.name, sc.seed, b.index, r.monitor, r.feature, r.statistic, r.p_value, r.flagged)
            for b, results in monitored_batches(sc, live, ref, model, baseline, cfg, con) for r in results]
    df = pd.DataFrame(rows, columns=["scenario", "seed", "batch_index", "monitor", "feature", "statistic", "p_value", "flagged"])
    return df.assign(drift_start=np.nan if sc.start_batch is None else sc.start_batch, n_batches=sc.n_batches, batch_size=sc.batch_size)


def collect(purpose: str, seeds, scenarios: list[Scenario], live, ref, model, baseline: dict, cfg: dict, log=print) -> pd.DataFrame:
    """Run every scenario once per seed. Refuses seeds that do not belong to `purpose`."""
    check_seeds(purpose, seeds)
    from dataclasses import replace
    parts = []
    for seed in seeds:
        for sc in scenarios:
            parts.append(collect_run(replace(sc, seed=seed), live, ref, model, baseline, cfg))
        log(f"[{purpose}] seed {seed} done")
    return pd.concat(parts, ignore_index=True)


# --- detector settings and alerts -----------------------------------------------------------------------

def apply_setting(stats: pd.DataFrame, setting: str, psi_threshold: float | None, alpha: float | None) -> pd.DataFrame:
    """Keep only rows that breach under this setting. PSI and p-value flags are recomputed from the stored
    statistic; missing-rate and performance flags keep the default-config flag."""
    s = stats[stats["monitor"].isin(FAMILIES[setting] | ALWAYS)]
    m, flag = s["monitor"].to_numpy(), s["flagged"].to_numpy()
    if psi_threshold is not None:
        flag = np.where(m == "psi", s["statistic"].to_numpy() > psi_threshold, flag)
    if alpha is not None:
        flag = np.where(np.isin(m, ["ks", "chi2"]), s["p_value"].to_numpy() < alpha, flag)
    return s[flag.astype(bool)]


def _run_alerts(flagged: pd.DataFrame | None, n_batches: int, consecutive: int):
    engine, out = AlertEngine(consecutive), []
    by = {} if flagged is None else {i: g for i, g in flagged.groupby("batch_index")}
    for i in range(n_batches):
        g = by.get(i)
        results = [] if g is None else [MonitorResult(r.monitor, r.feature, r.statistic, None if pd.isna(r.p_value) else r.p_value, True)
                                        for r in g.itertuples()]
        out += engine.update(i, results)
    return out


def evaluate(stats: pd.DataFrame, setting: str, psi_threshold, alpha, consecutive: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (runs, alerts): one runs row per scenario/seed, one alerts row per alert. An alert is correct
    when raised at or after the true drift batch."""
    flagged = apply_setting(stats, setting, psi_threshold, alpha)
    groups = dict(tuple(flagged.groupby(["scenario", "seed"])))
    meta = stats.drop_duplicates(["scenario", "seed"])[["scenario", "seed", "drift_start", "n_batches", "batch_size"]]
    runs, alerts = [], []
    for m in meta.itertuples(index=False):
        found = _run_alerts(groups.get((m.scenario, m.seed)), int(m.n_batches), consecutive)
        correct = [a for a in found if not pd.isna(m.drift_start) and a.batch_index >= m.drift_start]
        first = min((a.batch_index for a in correct), default=math.nan)
        runs.append({"scenario": m.scenario, "setting": setting, "seed": m.seed, "drift_start": m.drift_start,
                     "n_batches": m.n_batches, "batch_size": m.batch_size, "n_alerts": len(found), "n_correct": len(correct),
                     "detected": bool(correct), "first_correct": first, "delay": first - m.drift_start if correct else math.nan})
        alerts += [{"scenario": m.scenario, "setting": setting, "seed": m.seed, "batch_index": a.batch_index,
                    "monitor": a.monitor, "feature": a.feature, "correct": a in correct} for a in found]
    return pd.DataFrame(runs), pd.DataFrame(alerts, columns=["scenario", "setting", "seed", "batch_index", "monitor", "feature", "correct"])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Per scenario and setting: recall, delay, precision for drifting scenarios; false alarms per 100 clean batches."""
    rows = []
    for (scen, setting), g in runs.groupby(["scenario", "setting"], sort=False):
        row = {"scenario": scen, "setting": setting, "runs": len(g), "alerts": int(g["n_alerts"].sum())}
        if g["drift_start"].isna().all():
            clean = int(g["n_batches"].sum())
            row |= {"clean_batches": clean, "false_alarms_per_100": 100 * row["alerts"] / clean}
        else:
            k, n, c = int(g["detected"].sum()), len(g), int(g["n_correct"].sum())
            d = g.loc[g["detected"], "delay"]
            row |= {"detected": k, "recall": k / n, "recall_ci": wilson(k, n), "delay_mean": d.mean(), "delay_std": d.std(ddof=1),
                    "correct_alerts": c, "precision": c / row["alerts"] if row["alerts"] else math.nan,
                    "precision_ci": wilson(c, row["alerts"])}
        rows.append(row)
    return pd.DataFrame(rows)


# --- tuning (seeds 0-9) and reporting (seeds 10-29) -----------------------------------------------------

def _candidates(setting: str, ecfg: dict):
    if setting == "psi":
        return [(p, None) for p in ecfg["psi_grid"]]
    if setting == "ks":
        return [(None, a) for a in ecfg["alpha_grid"]]
    return list(product(ecfg["psi_grid"], ecfg["alpha_grid"]))


def tune(stats: pd.DataFrame, ecfg: dict, consecutive: int) -> tuple[dict, pd.DataFrame]:
    """Pick thresholds per setting from tuning seeds only. Rule: among candidates within the false-alarm budget,
    highest mean recall over drift scenarios, then lowest mean delay, then the most conservative threshold."""
    check_seeds("tune", stats["seed"].unique())
    budget, grid, chosen = ecfg["false_alarm_budget_per_100"], [], {}
    for setting in SETTINGS:
        rows = []
        for psi_t, alpha in _candidates(setting, ecfg):
            runs, _ = evaluate(stats, setting, psi_t, alpha, consecutive)
            s = summarize(runs)
            drift, clean = s[s["recall"].notna()], s[s["false_alarms_per_100"].notna()]
            fa = float(clean["false_alarms_per_100"].mean())
            delay = runs.loc[runs["detected"], "delay"].mean()
            rows.append({"setting": setting, "psi_threshold": psi_t, "alpha": alpha, "false_alarms_per_100": fa,
                         "mean_recall": float(drift["recall"].mean()), "mean_delay": delay, "within_budget": fa <= budget})
        df = pd.DataFrame(rows)
        grid.append(df)

        def key(r):
            ok = r.within_budget
            return (0 if ok else 1, 0 if ok else r.false_alarms_per_100, -r.mean_recall,
                    math.inf if pd.isna(r.mean_delay) else r.mean_delay,
                    -(0 if pd.isna(r.psi_threshold) else r.psi_threshold), 0 if pd.isna(r.alpha) else r.alpha)
        best = min(df.itertuples(), key=key)
        chosen[setting] = {"psi_threshold": None if pd.isna(best.psi_threshold) else float(best.psi_threshold),
                           "alpha": None if pd.isna(best.alpha) else float(best.alpha),
                           "within_budget": bool(best.within_budget)}
    thresholds = {"tuned_on_seeds": list(TUNE_SEEDS), "false_alarm_budget_per_100": budget, "consecutive": consecutive,
                  "settings": chosen}
    return thresholds, pd.concat(grid, ignore_index=True)


def report(stats: pd.DataFrame, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate on reporting seeds with thresholds that were tuned on tuning seeds. Never tunes anything."""
    check_seeds("report", stats["seed"].unique())
    if thresholds.get("tuned_on_seeds") != list(TUNE_SEEDS):
        raise ValueError("thresholds were not tuned on the designated tuning seeds")
    parts = [evaluate(stats, s, t["psi_threshold"], t["alpha"], thresholds["consecutive"]) for s, t in thresholds["settings"].items()]
    return pd.concat([p[0] for p in parts], ignore_index=True), pd.concat([p[1] for p in parts], ignore_index=True)


# --- table and README -----------------------------------------------------------------------------------

def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(p, ci):
    return "n/a" if p is None or math.isnan(p) else f"{p:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def render(runs: pd.DataFrame, thresholds: dict) -> str:
    """Markdown results block generated only from runs.csv and tuned_thresholds.json."""
    s = summarize(runs)
    seeds = sorted(runs["seed"].unique())
    nb, bs = int(runs["n_batches"].iloc[0]), int(runs["batch_size"].iloc[0])
    start = int(runs["drift_start"].dropna().iloc[0])
    th = thresholds
    out = [
        (
            f"Method: every scenario was run on {len(seeds)} held-out seeds ({seeds[0]}-{seeds[-1]}); thresholds were tuned "
            f"on separate seeds ({th['tuned_on_seeds'][0]}-{th['tuned_on_seeds'][-1]}) with a budget of at most "
            f"{_f(th['false_alarm_budget_per_100'], 1)} false alarms per 100 clean batches. Each stream has {nb} batches of {bs} rows, "
            f"drift starts at batch {start}, and an alert needs {th['consecutive']} consecutive breaching batches. "
            "An alert is correct if it is raised at or after the true drift batch; this counts alerts on any feature, "
            "so precision is lenient. Recall is the share of runs with at least one correct alert. Delay is the first correct "
            "alert minus the drift batch, over detected runs. Intervals are 95% Wilson. Seeds change the stream "
            "(shuffle, resampling, missing mask); the model and reference window are fixed."
        ),
        "",
        "| Scenario | Setting | Detected | Recall [95% CI] | Delay, batches (mean ± sd) | Precision [95% CI] | Alerts |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s[s["recall"].notna()].itertuples():
        out.append(f"| {r.scenario} | {r.setting} | {int(r.detected)}/{int(r.runs)} | {_ci(r.recall, r.recall_ci)} | "
                   f"{_f(r.delay_mean)} ± {_f(r.delay_std)} | {_ci(r.precision, r.precision_ci)} | {int(r.correct_alerts)}/{int(r.alerts)} correct |")
    out += ["", "| Clean stream, setting | False alarms | Clean batches | Per 100 clean batches |", "|---|---|---|---|"]
    for r in s[s["false_alarms_per_100"].notna()].itertuples():
        out.append(f"| {r.setting} | {int(r.alerts)} | {int(r.clean_batches)} | {_f(r.false_alarms_per_100)} |")
    out += ["", "| Setting | PSI threshold | KS / chi-square alpha | Within false-alarm budget (tuning seeds) |", "|---|---|---|---|"]
    for name, t in th["settings"].items():
        out.append(f"| {name} | {'n/a' if t['psi_threshold'] is None else t['psi_threshold']} | "
                   f"{'n/a' if t['alpha'] is None else t['alpha']} | {'yes' if t['within_budget'] else 'no'} |")
    return "\n".join(out)


def splice(readme: str, block: str) -> str:
    if README_START not in readme or README_END not in readme:
        raise ValueError("README is missing the RESULTS markers")
    head, rest = readme.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    return f"{head}{README_START}\n{block}\n{README_END}{tail}"
