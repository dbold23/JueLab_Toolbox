# Benchmark snapshot v1 (pre-rebuild)

This directory is the tracked, clean-proof baseline for the synthetic
validation suite **as it was published before the rebuild**.  It is not a step-9
run directory: `make clean` runs `git clean` over
`synthetic_data/output/comprehensive_test/validation*/` only, and every file
here is git-tracked, so nothing in this directory is ever deleted.

Nothing in this directory is regenerated.  It is a frozen copy, kept so the
v1 -> v2 movement in `docs/BENCHMARK_LEDGER.md` can be re-derived from files in
the working tree instead of from git archaeology.

| file | snapshot of |
|---|---|
| `ground_truth.csv` | `.../comprehensive_test/test_data/ground_truth.csv` |
| `generation_summary.json` | `.../comprehensive_test/generation_summary.json` |
| `validation_metrics.json` | `.../comprehensive_test/validation_metrics.json` |
| `VALIDATION_SUMMARY.md` | `.../comprehensive_test/VALIDATION_SUMMARY.md` |
| `processing_results.csv` | `.../comprehensive_test/validation_latest/processing_results.csv` |
| `validation_report.json` | `.../comprehensive_test/validation_latest/validation_report.json` |
| `classification_summary.png` | `.../comprehensive_test/validation_latest/classification_summary.png` |
| `DATA/*.csv` | `.../comprehensive_test/test_data/DATA/*` (the six v1 plates) |
| `config.json` | `.../comprehensive_test/test_data/config.json` |
| `HEADLINE.json` | the v1 headline, written by WP12 (not a copy of anything) |
| `MANIFEST.json` | sha256 of every file here |

Deliberately **not** snapshotted: `validation_latest/plots/` (~96 MB of
per-curve PNGs), which was never tracked.

## Why the v1 plates are kept

`DATA/` is here so the pipeline's **non-regression** claim stays checkable
forever, without git archaeology and without depending on the current
`test_data/`, which now holds the v2 suite.  Re-running today's pipeline over
these six plates in first-peak truncation reproduces the v1 confusion matrix
exactly -- 319/100/35/26, accuracy 0.8729166666666667 -- and every non-ML cell
of the re-run matches the committed `processing_results.csv`.  The only
difference is WP9's appended `failed_gates` column.

```bash
MPLBACKEND=Agg python scripts/01_growth_curve_analysis.py \
    synthetic_data/benchmark/v1/DATA -o /tmp/v1_rerun \
    --no-adaptive --first-peak --no-plots -q
MPLBACKEND=Agg python - <<'PY'
import sys, pandas as pd
sys.path.insert(0, 'synthetic_data')
from validation.matching import match_results
m = match_results(pd.read_csv('synthetic_data/benchmark/v1/ground_truth.csv'),
                  pd.read_csv('/tmp/v1_rerun/processing_results.csv'))
tp = int((m.gt_is_good & m.pred_is_good).sum())
tn = int(((~m.gt_is_good) & (~m.pred_is_good)).sum())
fp = int(((~m.gt_is_good) & m.pred_is_good).sum())
fn = int((m.gt_is_good & (~m.pred_is_good)).sum())
print(f'TP/TN/FP/FN {tp}/{tn}/{fp}/{fn}  accuracy {(tp+tn)/len(m)!r}')
PY
```

## The v1 headline

480 curves over 32 scenarios, 345 GOOD / 135 BAD, first-peak truncation, seed 42.

```
accuracy 0.8729  precision 0.9011  recall 0.9246  specificity 0.7407
balanced accuracy 0.8327     TP/TN/FP/FN 319/100/35/26
pooled recovery  A R2 0.8251 RMSE 0.14185 | mu RMSE 0.02820 | lambda RMSE 0.91512
misses  borderline_noise 14, borderline_r2_bad 14, logistic_generated 13,
        sparse_data 10, high_noise 7, fast_growth 2, low_A 1
```

Recompute the confusion matrix straight from the two CSVs here:

```bash
MPLBACKEND=Agg python - <<'PY'
import sys, pandas as pd
sys.path.insert(0, 'synthetic_data')
from validation.matching import match_results
m = match_results(pd.read_csv('synthetic_data/benchmark/v1/ground_truth.csv'),
                  pd.read_csv('synthetic_data/benchmark/v1/processing_results.csv'))
tp = ((m.gt_is_good) & (m.pred_is_good)).sum()
tn = ((~m.gt_is_good) & (~m.pred_is_good)).sum()
fp = ((~m.gt_is_good) & (m.pred_is_good)).sum()
fn = ((m.gt_is_good) & (~m.pred_is_good)).sum()
print(f"TP/TN/FP/FN {tp}/{tn}/{fp}/{fn}  accuracy {(tp+tn)/len(m):.4f}")
PY
```
