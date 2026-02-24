# Synthetic Data Validation Summary

*Done: 2026-02-12 15:09:33*

## Executive Summary

The TECAN growth-curve processing pipeline was validated against **480 synthetic curves** spanning **32 test scenarios** (9 good, 8 bad, 15 edge-case).

| Metric | Value |
|--------|-------|
| **Accuracy** | 84.4% |
| **Precision** | 85.3% |
| **Recall (sensitivity)** | 94.5% |
| **F1 score** | 89.7% |

Out of 480 curves, **405** were classified correctly and **75** were misclassified. The pipeline achieves high recall (94.5%), meaning it rarely misses genuinely good curves. Precision (85.3%) is slightly lower, indicating a modest tendency to accept borderline or bad curves as good (false positives).

## Confusion Matrix

Positive = GOOD curve, Negative = BAD curve.

|  | **Predicted GOOD** | **Predicted BAD** | **Total** |
|--|---:|---:|---:|
| **Actually GOOD** | 326 (TP) | 19 (FN) | 345 |
| **Actually BAD** | 56 (FP) | 79 (TN) | 135 |
| **Total** | 382 | 98 | 480 |

- **True Positives (TP):** 326 -- good curves correctly accepted.
- **True Negatives (TN):** 79 -- bad curves correctly rejected.
- **False Positives (FP):** 56 -- bad curves incorrectly accepted.
- **False Negatives (FN):** 19 -- good curves incorrectly rejected.

## Classification Metrics

| Metric | Formula | Value |
|--------|---------|------:|
| Accuracy | (TP+TN) / Total | 84.4% |
| Precision | TP / (TP+FP) | 85.3% |
| Recall | TP / (TP+FN) | 94.5% |
| F1 Score | 2*P*R / (P+R) | 89.7% |
| Specificity | TN / (TN+FP) | 58.5% |
| Neg. Predictive Value | TN / (TN+FN) | 80.6% |

## Parameter Recovery (True-Positive Curves Only)

For the 326 curves correctly classified as GOOD, the Gompertz parameters estimated by the pipeline were compared to the known ground-truth values.

| Parameter | R-squared | RMSE | Mean Error (bias) | N |
|-----------|----------:|-----:|------------------:|--:|
| **A (max OD)** | 0.9849 | 0.0418 | 0.0199 | 326 |
| **mu (growth rate)** | 0.9058 | 0.0304 | -0.0074 | 326 |
| **lambda (lag time)** | 0.9834 | 1.0127 | -0.3663 | 326 |

**Interpretation:**

- **A** (R-squared = 0.985): Excellent recovery. The pipeline estimates maximum OD very accurately.
- **mu** (R-squared = 0.906): Good recovery, though growth rate is inherently harder to pin down because it depends on the slope of a narrow exponential window.
- **lambda** (R-squared = 0.983): Excellent recovery. The small negative bias (-0.37 h) suggests the pipeline estimates lag time slightly earlier than ground truth on average, likely due to truncation heuristics.

## Per-Scenario Accuracy

Scenarios are sorted from lowest to highest accuracy. Each scenario contains 15 synthetic curves.

| Scenario | Category | Expected | Accuracy | Correct | Wrong | FP | FN |
|----------|----------|----------|-------:|--------:|------:|---:|---:|
| borderline_noise | Edge / borderline | BAD | 0.0% | 0 | 15 | 15 | 0 |
| borderline_r2_bad | Edge / borderline | BAD | 0.0% | 0 | 15 | 15 | 0 |
| high_noise | Bad (no / failed growth) | BAD | 33.3% | 5 | 10 | 10 | 0 |
| minimal_growth | Bad (no / failed growth) | BAD | 46.7% | 7 | 8 | 8 | 0 |
| logistic_generated | Edge / borderline | GOOD | 53.3% | 8 | 7 | 0 | 7 |
| truncation_challenge | Edge / borderline | GOOD | 53.3% | 8 | 7 | 0 | 7 |
| pesticide_only | Bad (no / failed growth) | BAD | 60.0% | 9 | 6 | 6 | 0 |
| baranyi_generated | Good (standard growth) | GOOD | 80.0% | 12 | 3 | 0 | 3 |
| erratic | Bad (no / failed growth) | BAD | 86.7% | 13 | 2 | 2 | 0 |
| short_lag | Good (standard growth) | GOOD | 86.7% | 13 | 2 | 0 | 2 |
| dense_data | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| standard | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| sparse_data | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| slow_growth | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| short_experiment | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| richards_asymmetric | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| borderline_delta_od | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| pesticide_lb_typical | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| outlier_contaminated | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| borderline_r2_good | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| low_A | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| long_lag | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| long_experiment | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| contamination | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| death_phase_moderate | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| high_A | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| flat_no_growth | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| fit_failure | Bad (no / failed growth) | BAD | 100.0% | 15 | 0 | 0 | 0 |
| fast_growth | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| drift_positive | Edge / borderline | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| death_phase_severe | Bad (no / failed growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |
| very_clean | Good (standard growth) | GOOD | 100.0% | 15 | 0 | 0 | 0 |

**22 of 32 scenarios achieved 100% accuracy.** The remaining 10 scenarios had at least one misclassification.

## Failure Analysis

### Misclassifications by Scenario

| Scenario | Misclassifications | Category | Dominant Error |
|----------:|---:|----------|---------------|
| borderline_noise | 15 | Edge / borderline | FP (15) |
| borderline_r2_bad | 15 | Edge / borderline | FP (15) |
| high_noise | 10 | Bad (no / failed growth) | FP (10) |
| minimal_growth | 8 | Bad (no / failed growth) | FP (8) |
| truncation_challenge | 7 | Edge / borderline | FN (7) |
| logistic_generated | 7 | Edge / borderline | FN (7) |
| pesticide_only | 6 | Bad (no / failed growth) | FP (6) |
| baranyi_generated | 3 | Good (standard growth) | FN (3) |
| short_lag | 2 | Good (standard growth) | FN (2) |
| erratic | 2 | Bad (no / failed growth) | FP (2) |

### Detailed Failure Discussion

**False Positives (bad curves accepted as good):**

- **borderline_noise** (15 FP): Noise level at the boundary between acceptable and unacceptable.
- **borderline_r2_bad** (15 FP): R-squared hovering just below the GOOD threshold.
- **high_noise** (10 FP): Very high noise that obscures any underlying growth signal.
- **minimal_growth** (8 FP): Tiny delta-OD near the detection threshold. Should be classified BAD.
- **pesticide_only** (6 FP): Flat or declining curves representing pesticide-inhibited wells (no growth).
- **erratic** (2 FP): Randomly fluctuating OD with no biological growth pattern.

**False Negatives (good curves rejected as bad):**

- **logistic_generated** (7 FN): Logistic growth model instead of Gompertz. Another model-mismatch test.
- **truncation_challenge** (7 FN): Experiment ended before stationary phase was fully reached.
- **baranyi_generated** (3 FN): Curves generated with the Baranyi model rather than Gompertz. Tests model-mismatch robustness.
- **short_lag** (2 FN): Very short lag phase (lambda < 2 h). Tests fitting when growth begins almost immediately.

## Scenario Descriptions

### Good (standard growth)

- **standard**: Baseline Gompertz curves with medium noise -- the bread-and-butter case.
- **high_A**: High maximum OD (A > 1.5). Tests whether the pipeline handles large growth amplitudes.
- **low_A**: Low maximum OD (A ~ 0.3-0.5). Tests sensitivity to small but real growth signals.
- **fast_growth**: High growth rate (mu > 0.3). Tests fitting of steep exponential phases.
- **slow_growth**: Low growth rate (mu < 0.1). Tests detection of gradual growth that may be mistaken for noise.
- **long_lag**: Extended lag phase (lambda > 10 h). Tests correct identification of delayed growth onset.
- **short_lag**: Very short lag phase (lambda < 2 h). Tests fitting when growth begins almost immediately.
- **very_clean**: Very low noise. Ensures the pipeline does not introduce artefacts on pristine data.
- **baranyi_generated**: Curves generated with the Baranyi model rather than Gompertz. Tests model-mismatch robustness.

### Bad (no / failed growth)

- **flat_no_growth**: Flat OD traces with no growth. Pipeline should classify as BAD.
- **erratic**: Randomly fluctuating OD with no biological growth pattern.
- **fit_failure**: Curves deliberately designed to defeat Gompertz fitting (e.g. non-sigmoidal shapes).
- **death_phase_severe**: Severe decline after stationary phase. Tests whether large post-peak drops are flagged.
- **contamination**: Sudden OD spikes simulating contamination events.
- **pesticide_only**: Flat or declining curves representing pesticide-inhibited wells (no growth).
- **minimal_growth**: Tiny delta-OD near the detection threshold. Should be classified BAD.
- **high_noise**: Very high noise that obscures any underlying growth signal.

### Edge / borderline

- **borderline_noise**: Noise level at the boundary between acceptable and unacceptable.
- **borderline_r2_good**: R-squared hovering just above the GOOD threshold.
- **borderline_r2_bad**: R-squared hovering just below the GOOD threshold.
- **borderline_delta_od**: Delta-OD just above the minimum threshold for GOOD classification.
- **death_phase_moderate**: Moderate post-peak decline. Ambiguous -- could be GOOD or BAD depending on pipeline settings.
- **drift_positive**: Slow upward baseline drift without real growth.
- **outlier_contaminated**: Good growth curves with a few large outlier points injected.
- **truncation_challenge**: Experiment ended before stationary phase was fully reached.
- **short_experiment**: Reduced total experiment duration (~20 h). Tests whether the pipeline copes with limited data windows.
- **long_experiment**: Extended experiment duration (~150 h). Tests handling of prolonged stationary phase and potential drift.
- **sparse_data**: Reduced measurement density (fewer time points). Tests robustness to low temporal resolution.
- **dense_data**: High measurement density. Ensures the pipeline does not over-fit to noise.
- **logistic_generated**: Logistic growth model instead of Gompertz. Another model-mismatch test.
- **richards_asymmetric**: Richards (asymmetric sigmoid) growth model. Tests generalisation beyond symmetric sigmoids.
- **pesticide_lb_typical**: Typical LB-media pesticide experiment curves: real but modest growth.

## Recommendations

1. **Tighten borderline thresholds.** 30 false positives arose from borderline scenarios (borderline_noise, borderline_r2_bad). Consider adding a secondary quality check or slightly raising the R-squared / delta-OD acceptance thresholds.

2. **Improve noise handling.** 25 misclassifications came from noise-related scenarios. A pre-processing smoothing step or SNR-based filter could help discriminate noisy growth from noisy non-growth.

3. **Handle truncated experiments.** 7 errors in the truncation_challenge scenario suggest the pipeline struggles when stationary phase is not fully observed. Fitting with bounded parameters or detecting incomplete curves could reduce these errors.

4. **Improve model-mismatch robustness.** 10 misclassifications occurred on curves generated with non-Gompertz models (logistic_generated, baranyi_generated). Using a multi-model fitting approach or model-averaging could improve classification for diverse growth dynamics.

5. **Refine low-growth detection.** 14 errors came from minimal_growth and/or pesticide_only scenarios. These involve very small or absent growth signals that the pipeline sometimes over-fits. Adding a minimum delta-OD confidence interval could help.

6. **Expand the test suite over time.** As the pipeline evolves, add new synthetic scenarios targeting any newly-discovered failure modes. Consider adding real-data validation as a complementary benchmark.

---

## Advanced Analysis Validation

The advanced statistical methods in `06_advanced_fitting.py` were validated via
unit tests (`tests/test_advanced.py`) and end-to-end runs on real data.

### Unit Test Coverage (18 tests, all passing)

| Test Class | Tests | What is validated |
|-----------|-------|-------------------|
| TestGPTruncation | 4 | GP R² > 0.99 on clean sigmoid, truncation in correct region, 3-phase identification, sparse data handling |
| TestBootstrap | 3 | 95% CI contains true params, P(good) high for growth curves, P(good) low for flat curves |
| TestHaldaneODE | 3 | Biomass increases, substrate depletes, strong inhibition slows growth |
| TestDataLoading | 3 | Pesticide name extraction, strain identification, config loading |
| TestAdvancedOutputs | 2 | GP/bootstrap output files exist with expected columns |
| TestThinning | 3 | Short data preserved, long data reduced, endpoints kept |

### GP Truncation (57 strains)

- Successfully fit RBF + WhiteKernel GP to all 57 good strains
- Identifies lag, exponential, and stationary phases from GP derivative
- No heuristic parameters needed (GP learns noise level automatically)

### Bootstrap CIs (57 strains)

- Produced 95% confidence intervals on A, mu, lambda for all good strains
- P(good) classification confidence computed for each strain
- Clean growth curves: P(good) >= 0.90; flat/noisy curves: P(good) < 0.50

### Bayesian Gompertz (verified on 3 strains)

- NUTS sampler produced posterior summaries with mean, median, and 95% HDI
- Convergence diagnostics: ESS values 25-85 (low due to minimal test config: 1 chain, 50 draws)
- PPC (posterior predictive check) plots generated for each strain
- Bayesian classification: P(good) computed from posterior draws

### Bayesian Haldane

- Custom PyTensor Op wrapping scipy solve_ivp validated against direct `solve_haldane()`
- DEMetropolisZ sampler handles ODE-based likelihood without gradients
- Hierarchical Ki posteriors separate pesticide groups

### Recommendations for Advanced Methods

1. **Accept Xcode license** (`sudo xcodebuild -license`) to enable PyTensor C compilation. Without it, NUTS sampling is ~100x slower (Python-only mode).
2. **Production Bayesian runs** should use 4 chains, 2000 draws, 1000 tune for proper convergence diagnostics (R-hat, ESS).
3. **Data thinning** (`--thin 50`) is recommended for Bayesian models to reduce observation count while preserving curve shape.

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
- **Pipeline GOOD predictions:** 382
- **Pipeline BAD predictions:** 98
