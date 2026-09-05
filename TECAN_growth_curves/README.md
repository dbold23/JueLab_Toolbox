# TECAN Growth Curve Analysis Pipeline

**BIO380SP25 — Pesticide Bioremediating Bacteria**

Automated pipeline for fitting growth models to TECAN plate reader data, classifying curve quality with ML-assisted classification, quantifying pesticide substrate inhibition via Haldane kinetics, performing advanced Bayesian and GP-based statistical analysis, and comparing truncation methods to rescue misclassified strains.

## Quick Start

```bash
pip install -r scripts/requirements.txt

# Recommended: run the full 10-step pipeline
make full-pipeline    # Complete end-to-end (10 steps)
make core-pipeline    # Core only (skip GP/Ensemble)
make dry-run          # Preview without running
make step-2           # Run individual step
```

## Pipeline Flow

The pipeline is orchestrated by `scripts/run_full_pipeline.py` and executes 10 steps in order:

```
Step 0: Preprocess raw data       (Group 1 only, if needed)
Step 1: Train ML classifier       (auto-skips if models/ exist)
Step 2: Gompertz curve analysis    (per group, --adaptive, with ML)
Step 3: Combine group results      (-> all_groups_results.csv)
Step 4: Haldane inhibition         (ODE + AICc comparison)
Step 5: Advanced fitting           (GP, Bootstrap, Ensemble)
Step 6: Statistical analysis       (ANOVA, pairwise, figures)
Step 7: Truncation comparison      (5-method ranking + rescue)
Step 8: Export for collaboration    (clean CSV + methodology)
Step 9: Synthetic validation       (480 curves -> accuracy metrics)
```

## What This Does

1. **Gompertz Growth Model** — Fits `y(t) = A * exp(-exp((mu*e/A)*(lambda-t)+1))` to each strain's OD600 curve. Extracts maximum growth (A), growth rate (mu), and lag time (lambda).

2. **Two-Stage ML Classification** — Rule-based gates (R² >= 0.95, parameter error < 20%, secondary noise/SNR gates) plus an ML classifier (HistGradientBoosting) that adds biological metadata features (is-control, concentration) for three-class output: GOOD / BORDERLINE / BAD.

3. **Haldane Feedback Inhibition** — Fits a mechanistic ODE model (`mu(S) = mu_max * S / (Ks + S + S²/Ki)`) to pesticide+LB strains. Compares with Gompertz via AIC to test whether substrate inhibition kinetics explain growth better than a purely phenomenological model.

4. **GP-Based Truncation** — Fits a Gaussian Process (RBF + WhiteKernel) to each curve and identifies growth phases from the GP derivative, replacing heuristic peak-finding.

5. **Ensemble Truncation** — Combines 5 independent truncation methods (First Peak, Stationary Phase, Adaptive R², GP Derivative, Changepoint) via weighted-median consensus.

6. **Bayesian Hierarchical Models** — Fits Gompertz (NUTS) and Haldane (DEMetropolisZ) models with partial pooling by pesticide group, producing full posterior distributions on all parameters.

7. **Bootstrap Uncertainty** — Residual resampling on NLS Gompertz fits (1000 resamples) to produce 95% CIs and P(good) classification confidence.

8. **Truncation Method Comparison** — Fits Gompertz at each of 5 truncation points per strain, ranks methods by R², and generates per-strain overlay plots. Rescues bad strains where suboptimal truncation was the cause of poor fits.

9. **Interactive Truncation Validator** — matplotlib-based tool for human review of automated truncation method selections, with resumable annotations.

10. **ML Classifier** — Two-stage HistGradientBoosting classifier trained on 480 synthetic + 85 real audited curves with proper 70/30 held-out validation. Pre-fit gate rejects obvious junk before expensive fitting; post-fit classifier uses 24 features including biological metadata (is-control, concentration).

## Key Results

### Pipeline (92 strains across 4 groups)

| Group | Pesticides | Total | Good | Bad |
|-------|-----------|-------|------|-----|
| Group 1 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| Group 2 | Malathion, Lambda-Cyhalothrin | 24 | 11 | 13 |
| Group 3 | Imidacloprid, Lambda-Cyhalothrin | 24 | 12 | 12 |
| Group 4 | Permethrin, Lambda-Cyhalothrin | 20 | 9 | 11 |
| **Total** | | **92** | **44** | **48** |

Manual visual audit: **91.3%** of classifications validated correct (84/92).

### Truncation Method Comparison (57 good strains)

| Rank | Method | Mean R² | % Good (R²≥0.95) |
|:---:|--------|:-------:|:-----------------:|
| 1 | Adaptive R² | 0.9959 | 100% |
| 2 | Consensus | 0.9942 | 100% |
| 3 | Stationary Phase | 0.9911 | 100% |
| 4 | First Peak | 0.8355 | 93% |

**Bad strain rescue:** 17 of 22 candidate strains rescued by alternative truncation.

### Haldane Analysis (21 pesticide+LB strains) (uses S0=1.0 until real substrate [] known)

- Haldane model preferred over Gompertz for **16/21 strains (76%)** by AIC
- All 6 pesticides show measurable inhibition constants (Ki)
- Confirms substrate inhibition kinetics are present in pesticide treatment conditions

### Rule-Based Validation (480 synthetic curves)

| Metric | Value |
|--------|-------|
| Accuracy | 99.6% |
| Precision | 100% |
| Recall | 99.4% |
| F1 Score | 99.7% |

Only 2 failures out of 480 synthetic curves.

### ML Classifier (held-out 30% test set, 170 curves never seen during training)

| Metric | Value |
|--------|-------|
| Accuracy | 95.9% |
| Precision | 97.4% |
| Recall | 96.6% |
| F1 Score | 97.0% |
| Specificity | 94.3% |

Pre-fit gate at threshold 0.20 rejects 40/170 with 0 false rejects.

### Advanced Analysis

| Method | What it adds |
|--------|-------------|
| GP Truncation | Data-driven phase detection (no heuristic parameters) |
| Ensemble Truncation | 5-method consensus via weighted median |
| Bootstrap CIs | 95% confidence intervals on A, mu, lambda for all 57 good strains |
| Bayesian Gompertz | Full posteriors with partial pooling by pesticide group (NUTS) |
| Bayesian Haldane | Hierarchical Ki posteriors by pesticide (DEMetropolisZ) |
| Bayesian Classification | P(good) probability replacing hard GOOD/BAD threshold |
| Model Comparison | WAIC/LOO replacing AIC for Gompertz vs Haldane selection |
| ML Classifier | Two-stage HistGBT with metadata features (is-control, concentration) |

## Folder Structure

```
TECAN_growth_curves/
├── scripts/
│   ├── 01_growth_curve_analysis.py      # Gompertz pipeline
│   ├── 02_preprocess_raw_plate_data.py  # 96-well data preprocessor
│   ├── 03_mle_model_fitting.py          # Multi-model MLE fitting
│   ├── 05_haldane_analysis.py           # Haldane inhibition analysis
│   ├── 06_advanced_fitting.py           # Advanced stats (GP, Bayesian, Bootstrap, Ensemble)
│   ├── 07_compare_truncation_methods.py # Truncation method comparison & bad strain rescue
│   ├── 08_validate_truncation.py        # Interactive truncation method validator
│   ├── 09_train_classifier.py           # ML classifier training (70/30 split)
│   ├── run_full_pipeline.py             # Master orchestrator (10-step pipeline)
│   ├── ml_classifier.py                 # ML classifier runtime module
│   ├── validate_results_interactive.py  # Interactive curve auditor
│   ├── run_all_groups.sh                # Batch runner
│   ├── config.yaml                      # All thresholds in one place
│   └── requirements.txt                 # Python deps + pymc, arviz, scikit-learn
├── data/raw/                            # Raw TECAN plate reader data (Groups 1-4)
├── results/tables/
│   ├── all_groups_results.csv           # Consolidated Gompertz results
│   ├── validation_audit.csv             # Manual audit annotations
│   ├── Group{1-4}_Results/              # Per-group results + diagnostic plots
│   ├── Haldane_Analysis/                # Ki values, AIC comparisons, plots
│   └── Advanced_Analysis/               # GP, Bayesian, Bootstrap, Comparison results
│       ├── ensemble_truncation/         # Ensemble truncation results
│       └── truncation_comparison/       # Method comparison study
│           ├── method_comparison.csv    # Per-strain × per-method fit results
│           ├── method_summary.csv       # Aggregate method rankings
│           ├── rescued_strains.csv      # Bad strains rescued by better truncation
│           └── overlay_plots/           # Per-strain method overlay plots
├── models/                              # Trained ML classifiers (joblib, gitignored)
│   └── feature_config.json              # Feature names + training metrics (committed)
├── synthetic_data/                      # 480-curve validation suite
├── tests/                               # pytest suite (118 tests)
├── Makefile                             # make install / test / full-pipeline (20+ targets)
└── .github/workflows/test.yml           # CI (Python 3.10-3.12)
```

## Detailed Documentation

See [PIPELINE_METHODOLOGY.md](PIPELINE_METHODOLOGY.md) for comprehensive methodology documentation covering all 10 pipeline stages, mathematical formulations, algorithm details, and results.

See [scripts/README.md](scripts/README.md) for full usage, all CLI options, model equations, classification methods, truncation strategies, output column descriptions, and troubleshooting.
