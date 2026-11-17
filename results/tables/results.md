Method: every scenario was run on 20 held-out seeds (10-29); thresholds were tuned on separate seeds (0-9) with a budget of at most 1.0 false alarms per 100 clean batches. Each stream has 40 batches of 500 rows, drift starts at batch 15, and an alert needs 3 consecutive breaching batches. An alert is correct if it is raised at or after the true drift batch; this counts alerts on any feature, so precision is lenient. Recall is the share of runs with at least one correct alert. Delay is the first correct alert minus the drift batch, over detected runs. Intervals are 95% Wilson. Seeds change the stream (shuffle, resampling, missing mask); the model and reference window are fixed.

| Scenario | Setting | Detected | Recall [95% CI] | Delay, batches (mean ± sd) | Precision [95% CI] | Alerts |
|---|---|---|---|---|---|---|
| gradual_shift | psi | 20/20 | 1.00 [0.84, 1.00] | 3.95 ± 0.22 | 1.00 [0.85, 1.00] | 21/21 correct |
| label_shift | psi | 20/20 | 1.00 [0.84, 1.00] | 4.50 ± 1.28 | 1.00 [0.93, 1.00] | 50/50 correct |
| missing_feature | psi | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.85, 1.00] | 22/22 correct |
| sudden_shift | psi | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.84, 1.00] | 20/20 correct |
| gradual_shift | ks | 20/20 | 1.00 [0.84, 1.00] | 2.25 ± 0.44 | 1.00 [0.85, 1.00] | 21/21 correct |
| label_shift | ks | 20/20 | 1.00 [0.84, 1.00] | 1.55 ± 0.69 | 1.00 [0.99, 1.00] | 318/318 correct |
| missing_feature | ks | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.86, 1.00] | 23/23 correct |
| sudden_shift | ks | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.84, 1.00] | 20/20 correct |
| gradual_shift | both | 20/20 | 1.00 [0.84, 1.00] | 2.25 ± 0.44 | 1.00 [0.91, 1.00] | 41/41 correct |
| label_shift | both | 20/20 | 1.00 [0.84, 1.00] | 1.55 ± 0.69 | 1.00 [0.99, 1.00] | 318/318 correct |
| missing_feature | both | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.86, 1.00] | 23/23 correct |
| sudden_shift | both | 20/20 | 1.00 [0.84, 1.00] | 2.00 ± 0.00 | 1.00 [0.91, 1.00] | 40/40 correct |

| Clean stream, setting | False alarms | Clean batches | Per 100 clean batches |
|---|---|---|---|
| psi | 1 | 800 | 0.12 |
| ks | 1 | 800 | 0.12 |
| both | 1 | 800 | 0.12 |

| Setting | PSI threshold | KS / chi-square alpha | Within false-alarm budget (tuning seeds) |
|---|---|---|---|
| psi | 0.1 | n/a | yes |
| ks | n/a | 0.05 | yes |
| both | 0.3 | 0.05 | yes |
