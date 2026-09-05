# Synthetic Data Validation Summary

*Generated: 2026-09-05 21:55:46*

## Executive Summary

The TECAN growth-curve processing pipeline was validated against **480 synthetic curves** spanning **32 test scenarios** (9 good, 8 bad, 15 edge-case).

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.3% |
| **Precision** | 90.1% |
| **Recall (sensitivity)** | 92.5% |
| **F1 score** | 91.3% |

Out of 480 curves, **419** were classified correctly and **61** were misclassified. The pipeline achieves high recall (92.5%), meaning it rarely misses genuinely good curves. Precision (90.1%) is slightly lower, indicating a modest tendency to accept borderline or bad curves as good (false positives).

## Confusion Matrix

Positive = GOOD curve, Negative = BAD curve.

|  | **Predicted GOOD** | **Predicted BAD** | **Total** |
|--|---:|---:|---:|
| **Actually GOOD** | 319 (TP) | 26 (FN) | 345 |
| **Actually BAD** | 35 (FP) | 100 (TN) | 135 |
| **Total** | 354 | 126 | 480 |

- **True Positives (TP):** 319 -- good curves correctly accepted.
- **True Negatives (TN):** 100 -- bad curves correctly rejected.
- **False Positives (FP):** 35 -- bad curves incorrectly accepted.
- **False Negatives (FN):** 26 -- good curves incorrectly rejected.

## Classification Metrics

| Metric | Formula | Value |
|--------|---------|------:|
| Accuracy | (TP+TN) / Total | 87.3% |
| Precision | TP / (TP+FP) | 90.1% |
| Recall | TP / (TP+FN) | 92.5% |
| F1 Score | 2*P*R / (P+R) | 91.3% |
| Specificity | TN / (TN+FP) | 74.1% |
| Neg. Predictive Value | TN / (TN+FN) | 79.4% |

## Parameter Recovery (True-Positive Curves Only)

For the 319 curves correctly classified as GOOD, the Gompertz parameters estimated by the pipeline were compared to the known ground-truth values.

| Parameter | R-squared | RMSE | Mean Error (bias) | N |
|-----------|----------:|-----:|------------------:|--:|
| **A (max OD)** | 0.8251 | 0.1418 | 0.0371 | 319 |
| **mu (growth rate)** | 0.9084 | 0.0282 | -0.0066 | 319 |
| **lambda (lag time)** | 0.9867 | 0.9151 | -0.3162 | 319 |

**Interpretation:**

- **A** (R-squared = 0.825): Excellent recovery. The pipeline estimates maximum OD very accurately.
- **mu** (R-squared = 0.908): Good recovery, though growth rate is inherently harder to pin down because it depends on the slope of a narrow exponential window.
- **lambda** (R-squared = 0.987): Excellent recovery. The small negative bias (-0.32 h) suggests the pipeline estimates lag time slightly earlier than ground truth on average, likely due to truncation heuristics.

## Per-Scenario Accuracy

Scenarios are sorted from lowest to highest accuracy. Each scenario contains 15 synthetic curves.

| Scenario | Category | Expected | Accuracy | Correct | Wrong | FP | FN |
|----------|----------|----------|-------:|--------:|------:|---:|---:|
| borderline_r2_bad | Edge / borderline | BAD | 6.7% | 1 | 14 | 14 | 0 |
| borderline_noise | Bad (no / failed growth) | BAD | 6.7% | 1 | 14 | 14 | 0 |
| logistic_generated | Edge / borderline | GOOD | 13.3% | 2 | 13 | 0 | 13 |
| sparse_data | Edge / borderline | GOOD | 33.3% | 5 | 10 | 0 | 10 |
| high_noise | Bad (no / failed growth) | BAD | 53.3% | 8 | 7 | 7 | 0 |
| fast_growth | Good (standard growth) | GOOD | 86.7% | 13 | 2 | 0 | 2 |
| low_A | Good (standard growth) | GOOD | 93.3% | 14 | 1 | 0 | 1 |
| borderline_r2_good | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| death_phase_severe | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| death_phase_moderate | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| drift_positive | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| dense_data | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| erratic | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| contamination | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| borderline_delta_od | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| baranyi_generated | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| high_A | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| flat_no_growth | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| fit_failure | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| long_experiment | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| minimal_growth | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| outlier_contaminated | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| pesticide_lb_typical | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| long_lag | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| pesticide_only | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| richards_asymmetric | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| short_lag | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| short_experiment | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| slow_growth | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| standard | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| truncation_challenge | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| very_clean | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |

**25 of 32 scenarios achieved 100% accuracy.** The remaining 7 scenarios had at least one misclassification.

## Failure Analysis

### Misclassifications by Scenario

| Scenario | Misclassifications | Category | Dominant Error |
|----------:|---:|----------|---------------|
| borderline_noise | 14 | Bad (no / failed growth) | FP (14) |
| borderline_r2_bad | 14 | Edge / borderline | FP (14) |
| logistic_generated | 13 | Edge / borderline | FN (13) |
| sparse_data | 10 | Edge / borderline | FN (10) |
| high_noise | 7 | Bad (no / failed growth) | FP (7) |
| fast_growth | 2 | Good (standard growth) | FN (2) |
| low_A | 1 | Good (standard growth) | FN (1) |

### Detailed Failure Discussion

**False Positives (bad curves accepted as good):**

- **borderline_r2_bad** (14 FP): R-squared hovering just below the GOOD threshold. (expected BAD)
- **borderline_noise** (14 FP): Noise level at the boundary between acceptable and unacceptable. (expected BAD)
- **high_noise** (7 FP): Very high noise that obscures any underlying growth signal. (expected BAD)

**False Negatives (good curves rejected as bad):**

- **logistic_generated** (13 FN): Logistic growth model instead of Gompertz. Another model-mismatch test. (expected GOOD)
- **sparse_data** (10 FN): Reduced measurement density (fewer time points). Tests robustness to low temporal resolution. (expected GOOD)
- **fast_growth** (2 FN): High growth rate (mu > 0.3). Tests fitting of steep exponential phases. (expected GOOD)
- **low_A** (1 FN): Low maximum OD (A ~ 0.3-0.5). Tests sensitivity to small but real growth signals. (expected GOOD)

## Scenario Descriptions

### Good (standard growth)

- **fast_growth**: High growth rate (mu > 0.3). Tests fitting of steep exponential phases. (expected GOOD)
- **high_A**: High maximum OD (A > 1.5). Tests whether the pipeline handles large growth amplitudes. (expected GOOD)
- **long_lag**: Extended lag phase (lambda > 10 h). Tests correct identification of delayed growth onset. (expected GOOD)
- **low_A**: Low maximum OD (A ~ 0.3-0.5). Tests sensitivity to small but real growth signals. (expected GOOD)
- **pesticide_lb_typical**: Typical LB-media pesticide experiment curves: real but modest growth. (expected GOOD)
- **short_lag**: Very short lag phase (lambda < 2 h). Tests fitting when growth begins almost immediately. (expected GOOD)
- **slow_growth**: Low growth rate (mu < 0.1). Tests detection of gradual growth that may be mistaken for noise. (expected GOOD)
- **standard**: Baseline Gompertz curves with medium noise -- the bread-and-butter case. (expected GOOD)
- **very_clean**: Very low noise. Ensures the pipeline does not introduce artefacts on pristine data. (expected GOOD)

### Bad (no / failed growth)

- **borderline_noise**: Noise level at the boundary between acceptable and unacceptable. (expected BAD)
- **contamination**: Sudden OD spikes simulating contamination events. (expected BAD)
- **erratic**: Randomly fluctuating OD with no biological growth pattern. (expected BAD)
- **fit_failure**: Curves deliberately designed to defeat Gompertz fitting (e.g. non-sigmoidal shapes). (expected BAD)
- **flat_no_growth**: Flat OD traces with no growth. Pipeline should classify as BAD. (expected BAD)
- **high_noise**: Very high noise that obscures any underlying growth signal. (expected BAD)
- **minimal_growth**: Tiny delta-OD near the detection threshold. Should be classified BAD. (expected BAD)
- **pesticide_only**: Flat or declining curves representing pesticide-inhibited wells (no growth). (expected BAD)

### Edge / borderline

- **baranyi_generated**: Curves generated with the Baranyi model rather than Gompertz. Tests model-mismatch robustness. (expected GOOD)
- **borderline_delta_od**: Delta-OD just above the minimum threshold for GOOD classification. (expected GOOD)
- **borderline_r2_bad**: R-squared hovering just below the GOOD threshold. (expected BAD)
- **borderline_r2_good**: R-squared hovering just above the GOOD threshold. (expected GOOD)
- **bubble_spike**: Growth curve with a single large bubble artifact (outlier spike) (expected GOOD) *(not present in data)*
- **death_phase_moderate**: Moderate post-peak decline. The pipeline should truncate before the death phase and still fit. (expected GOOD)
- **death_phase_severe**: Severe post-peak decline. Truncation has to cut the death phase away for the fit to succeed. (expected GOOD)
- **dense_data**: High measurement density. Ensures the pipeline does not over-fit to noise. (expected GOOD)
- **drift_positive**: Slow upward baseline drift without real growth. (expected GOOD)
- **evaporation_drift**: Flat curve with upward drift mimicking edge-well evaporation (false positive risk) (expected BAD) *(not present in data)*
- **haldane_mild_inhibition**: Haldane model with mild substrate inhibition (Ki >> Ks) (expected GOOD) *(not present in data)*
- **haldane_strong_inhibition**: Haldane model with strong substrate inhibition (Ki ~ Ks) (expected GOOD) *(not present in data)*
- **haldane_total_inhibition**: Haldane with very high substrate causing total inhibition (no growth) (expected BAD) *(not present in data)*
- **logistic_generated**: Logistic growth model instead of Gompertz. Another model-mismatch test. (expected GOOD)
- **long_experiment**: Extended experiment duration (~150 h). Tests handling of prolonged stationary phase and potential drift. (expected GOOD)
- **outlier_contaminated**: Good growth curves with a few large outlier points injected. (expected GOOD)
- **richards_asymmetric**: Richards (asymmetric sigmoid) growth model. Tests generalisation beyond symmetric sigmoids. (expected GOOD)
- **short_experiment**: Reduced total experiment duration (~20 h). Tests whether the pipeline copes with limited data windows. (expected GOOD)
- **sparse_data**: Reduced measurement density (fewer time points). Tests robustness to low temporal resolution. (expected GOOD)
- **truncation_challenge**: Experiment ended before stationary phase was fully reached. (expected GOOD)

## Recommendations

1. **Tighten borderline thresholds.** 28 false positives arose from borderline scenarios (borderline_r2_bad, borderline_noise). Consider adding a secondary quality check or slightly raising the R-squared / delta-OD acceptance thresholds.

2. **Improve noise handling.** 21 misclassifications came from noise-related scenarios. A pre-processing smoothing step or SNR-based filter could help discriminate noisy growth from noisy non-growth.

3. **Improve model-mismatch robustness.** 13 misclassifications occurred on curves generated with non-Gompertz models (logistic_generated). Using a multi-model fitting approach or model-averaging could improve classification for diverse growth dynamics.

4. **Expand the test suite over time.** As the pipeline evolves, add new synthetic scenarios targeting any newly-discovered failure modes. Consider adding real-data validation as a complementary benchmark.

---

## Appendix: Summary Statistics

- **Total curves:** 480
- **Total scenarios:** 32
- **Curves per scenario:** 15
- **Good scenarios:** 9 (135 curves)
- **Bad scenarios:** 8 (120 curves)
- **Edge scenarios:** 15 (225 curves)
- **Ground truth GOOD curves:** 345
- **Ground truth BAD curves:** 135
- **Pipeline GOOD predictions:** 354
- **Pipeline BAD predictions:** 126
