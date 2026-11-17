<!-- DRAFT: written from the repo history and committed results. Edit into your own voice before publishing. -->

# ModelGuard write-up

## What I built

ModelGuard watches a deployed classifier for three failures: input drift, missing features, and falling accuracy.

- **Baseline.** A logistic regression on the UCI Adult income data, trained on a reference window made of the leading rows of the file.
- **Prediction log.** A SQLite table that stores each batch's features, predictions and a nullable label that is filled in later.
- **Monitors.** PSI and the two-sample KS test for numeric features, PSI and chi-square for categorical ones, the missing-value rate per feature, and rolling accuracy and F1. Reference quantile bins are used for the numeric tests.
- **Alert engine.** A monitor raises an alert only after N consecutive breaching batches (N is 3 in the committed config). Each alert is stored with its batch and a plain-text reason.
- **Simulator.** It replays the live rows as seeded batches and injects sudden shift, gradual shift, label shift or missing values at a known batch, which is recorded as ground truth. Labels arrive a few batches after their predictions.
- **Dashboard and registry.** A read-only Streamlit view of the drift series, a missing-rate heatmap, rolling performance and an alert feed. Every run writes a record with its config, seed, commit and library versions.

## What I measured

The question was how quickly and how reliably each detector setting notices injected drift. I compared PSI only, KS only (with chi-square for categorical features), and both combined.

Thresholds were tuned on seeds 0 to 9. Everything reported comes from seeds 10 to 29, and the code refuses to mix the two. Each scenario ran 20 times, and clean streams were included to count false alarms. The full table, with the method, is in the [README](../README.md#results). Raw outcomes are in [results/eval/runs.csv](../results/eval/runs.csv) and [results/eval/alerts.csv](../results/eval/alerts.csv). Two full `make eval` runs produced byte-identical output files.

In the committed run:

- Every drift scenario was detected in every run, under every setting.
- The settings differed in delay, not in recall. KS detected gradual shift and label shift sooner than PSI. Sudden shift and the missing-feature scenario were detected at the earliest delay the alert rule allows under all settings.
- The clean streams produced a single alert (the same one under all three settings), and it came from the rolling F1 monitor, not from a drift statistic.

## What surprised me

- **A clean batch looked drifted.** My first clean demo batch flagged `native_country` on chi-square. Merging rare categories into one bucket helped, but I never fully isolated the rest. The alert rule hid it: it asks for several breaches in a row, and a lone noisy batch does not satisfy that. I did not tune the threshold to make the problem go away.
- **Accuracy did not move.** Shifting `age` and skewing a categorical feature in my first demo left rolling accuracy and F1 healthy. The model is more robust to that shift than I expected, so performance monitoring would not have caught it. Only the feature monitors did.
- **Label shift was not pure.** Resampling rows to change the positive rate also changed the feature distributions, because the features correlate with income. Feature monitors fired on it.
- **The chart hid the signal.** The first KS chart plotted p-values on a linear axis. The drifted points sat far below the threshold, so everything collapsed onto zero. Then my log-axis fix squeezed the clean points onto the threshold line. I only saw either problem when I looked at a screenshot.
- **The evaluation was too easy.** Recall and precision came out perfect everywhere, so they could not separate the detectors. The combined setting ended up behaving like KS: in the tuning grid, the PSI threshold made no difference once KS was on, so the value it chose is a tie-break. The tuned PSI threshold and the tuned KS alpha both landed on the edge of their grids ([results/eval/tune_grid.csv](../results/eval/tune_grid.csv)).

## What I would change

- **Harder scenarios.** Add small shifts, so detectors are separated by how much drift they can catch, not only by how fast they catch an obvious one. Choose them before looking at any new results.
- **A real held-out split.** Tune on one set of rows and report on a disjoint set, instead of two seed ranges over the same rows.
- **Vary the model and reference window across seeds,** so training variance is part of the error bars.
- **More drift types.** Categorical covariate shift and a concept-drift scenario, where the labelling rule changes.
- **Stricter precision.** Count an alert as correct only if it is on the feature that was actually changed, and report that next to the lenient number.
- **Realistic label delays,** drawn from a distribution.
- **A dataset with real timestamps,** so the reference window is a genuine earlier period.
- **Multiple-testing control.** With many features tested per batch, some false flags are expected; I rely on the consecutive-breach rule instead of correcting for it.
- **Log prediction probabilities,** so I could track AUC and calibration, not only label-based metrics.
