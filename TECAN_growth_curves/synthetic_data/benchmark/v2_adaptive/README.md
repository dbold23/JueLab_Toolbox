# Benchmark snapshot v2, adaptive truncation

The same 1139-curve v2 dataset scored under the **adaptive (Monte-Carlo
cross-validated) truncation** that `scripts/config.yaml` makes the batch
pipeline's default, alongside the first-peak headline in
`synthetic_data/benchmark/v2/`. Everything else is identical: same seed (42),
same dataset (`760a04062cf2`), same gates, same validator.

It exists because the two modes disagree, and the repository publishes
first-peak while the batch pipeline runs adaptive. Until this run, no
adaptive synthetic number existed at all.

Cost: 5 h 40 min for 1139 curves on 4 CPUs, against 6 minutes for first peak.
That ratio is why the lab interface defaults to first peak.

Regenerate with:

```bash
MPLBACKEND=Agg python scripts/01_growth_curve_analysis.py \
    synthetic_data/output/comprehensive_test/test_data/DATA \
    -o <run dir> --adaptive --no-plots -q
MPLBACKEND=Agg python synthetic_data/validation/benchmark.py \
    --results-csv <run dir>/processing_results.csv \
    --ground-truth synthetic_data/output/comprehensive_test/test_data/ground_truth.csv \
    --out-dir <report dir> --truncation-mode adaptive
```

## The two modes side by side

Both over the same 960 curves the gate set can decide fairly (the 179
policy-excluded curves are scored in their own blocks in each report).

| | first peak | adaptive |
|---|---:|---:|
| accuracy | 0.8760 | **0.8958** |
| balanced accuracy | 0.8790 | **0.8928** |
| sensitivity | 0.8558 | **0.9168** |
| specificity | **0.9021** | 0.8687 |
| MCC | 0.7528 | **0.7879** |
| TP / TN / FP / FN | 463 / 378 / 41 / 78 | 496 / 364 / 55 / 45 |
| core-32 accuracy | **0.9605** | 0.9511 |
| core-32 specificity | **1.0000** | 0.9179 |
| A recovery R² (accepted GOOD) | 0.4048 (n=555) | **0.6463** (n=588) |
| μ recovery R² | 0.9910 | **0.9957** |
| λ recovery R² | 0.9753 | 0.9752 |
| wall clock | ~6 min | ~5 h 40 min |

Adaptive wins overall and loses on specificity: it fits 33 more curves
correctly but accepts 14 more that should have been rejected.

## Where they differ by more than 15 points

| scenario | n | first peak | adaptive |
|---|---:|---:|---:|
| `diauxic_second_larger` | 12 | 0.00 | **1.00** |
| `sparse_data` | 15 | 0.47 | **0.87** |
| `fast_growth` | 10 | 0.60 | **1.00** |
| `noise_ladder_1` | 15 | 0.47 | **0.73** |
| `bubble_spike` | 20 | 0.75 | **0.95** |
| `contamination` | 25 | **1.00** | 0.52 |

The pattern is one trade, not noise. Searching the window by cross-validated
prediction finds the growth phase when the first-peak rule cannot: it fits the
first phase of a two-phase curve (`diauxic_second_larger` goes from 0/12 to
12/12), it keeps enough points on an hourly grid or a fast grower to clear the
20-point minimum, and it steps over a bubble. The cost is `contamination`,
where the search happily fits the plateau of a well that started high and never
grew, and 13 of 25 such wells are accepted that the first-peak rule rejects.

For the lab that means: first peak is the safer default and the honest one for
controls, adaptive is worth the wait when a plate is dominated by multi-phase
or sparsely sampled growers. The interface offers both per curve.
