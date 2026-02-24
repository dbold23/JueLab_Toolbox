# TECAN Growth Curve Analysis Pipeline

**BIO380SP25 — Pesticide Bioremediating Bacteria**

Automated pipeline for fitting growth models to TECAN plate reader data, classifying curve quality, quantifying pesticide substrate inhibition via Haldane kinetics, and performing advanced Bayesian and GP-based statistical analysis.

## Quick Start

```bash
pip install -r scripts/requirements.txt

# Run full pipeline (Gompertz fits + Haldane analysis)
make run-all

# Or step by step
make run-pipeline    # Gompertz fits for all 4 groups
make run-haldane     # Haldane feedback inhibition analysis
make run-advanced    # Advanced stats (GP, Bayesian, Bootstrap)
make test            # Run test suite
```

## What This Does

1. **Gompertz Growth Model** — Fits `y(t) = A * exp(-exp((mu*e/A)*(lambda-t)+1))` to each strain's OD600 curve. Extracts maximum growth (A), growth rate (mu), and lag time (lambda).

2. **Automated Classification** — Labels each curve GOOD or BAD based on fit quality (R² >= 0.95, parameter error < 20%) with secondary noise/SNR gates.

3. **Haldane Feedback Inhibition** — Fits a mechanistic ODE model (`mu(S) = mu_max * S / (Ks + S + S²/Ki)`) to pesticide+LB strains. Compares with Gompertz via AIC to test whether substrate inhibition kinetics explain growth better than a purely phenomenological model.

4. **GP-Based Truncation** — Fits a Gaussian Process (RBF + WhiteKernel) to each curve and identifies growth phases from the GP derivative, replacing heuristic peak-finding.

5. **Bayesian Hierarchical Models** — Fits Gompertz (NUTS) and Haldane (DEMetropolisZ) models with partial pooling by pesticide group, producing full posterior distributions on all parameters.

6. **Bootstrap Uncertainty** — Residual resampling on NLS Gompertz fits (1000 resamples) to produce 95% CIs and P(good) classification confidence.

7. **Bayesian Model Comparison** — WAIC/LOO-CV comparison of Gompertz vs Haldane with proper uncertainty quantification.

## Key Results

### Pipeline (92 strains across 4 groups)

| Group | Pesticides | Total | Good | Bad |
|-------|-----------|-------|------|-----|
| Group 1 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| Group 2 | Malathion, Lambda-Cyhalothrin | 24 | 16 | 8 |
| Group 3 | Imidacloprid, Lambda-Cyhalothrin | 24 | 17 | 7 |
| Group 4 | Permethrin, Lambda-Cyhalothrin | 20 | 12 | 8 |
| **Total** | | **92** | **57** | **35** |

Manual visual audit: **91.3%** of classifications validated correct (84/92).

### Haldane Analysis (23 pesticide+LB strains) (uses S0=1.0 until real substrat [] known)

- Haldane model preferred over Gompertz for **15/23 strains (65%)** by AIC
- All 6 pesticides show measurable inhibition constants (Ki)
- Confirms substrate inhibition kinetics are present in pesticide treatment conditions

### Validation (480 synthetic curves)

| Metric | Value |
|--------|-------|
| Accuracy | 89.0% |
| Precision | 89.5% |
| Recall | 95.9% |
| F1 Score | 92.6% |

25/32 test scenarios achieve 100% accuracy.

### Advanced Analysis

| Method | What it adds |
|--------|-------------|
| GP Truncation | Data-driven phase detection (no heuristic parameters) |
| Bootstrap CIs | 95% confidence intervals on A, mu, lambda for all 57 good strains |
| Bayesian Gompertz | Full posteriors with partial pooling by pesticide group (NUTS) |
| Bayesian Haldane | Hierarchical Ki posteriors by pesticide (DEMetropolisZ) |
| Bayesian Classification | P(good) probability replacing hard GOOD/BAD threshold |
| Model Comparison | WAIC/LOO replacing AIC for Gompertz vs Haldane selection |

## Folder Structure

```
TECAN_growth_curves/
├── scripts/
│   ├── 01_growth_curve_analysis.py      # Gompertz pipeline
│   ├── 02_preprocess_raw_plate_data.py  # 96-well data preprocessor
│   ├── 03_mle_model_fitting.py          # Multi-model MLE fitting
│   ├── 05_haldane_analysis.py           # Haldane inhibition analysis
│   ├── 06_advanced_fitting.py           # Advanced stats (GP, Bayesian, Bootstrap)
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
│   └── Advanced_Analysis/               # GP, Bayesian, Bootstrap results
├── synthetic_data/                      # 480-curve validation suite
├── tests/                               # pytest suite (18+ tests)
├── Makefile                             # make install / test / run-all
└── .github/workflows/test.yml           # CI (Python 3.10-3.12)
```

## Detailed Documentation

See [scripts/README.md](scripts/README.md) for full usage, all CLI options, model equations, classification methods, truncation strategies, output column descriptions, and troubleshooting.

