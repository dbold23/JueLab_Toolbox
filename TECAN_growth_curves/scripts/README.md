# Growth Curve Analysis Pipeline

**BIO380SP25 - Pesticide Bioremediating Bacteria Research Project**

This folder is **self-contained** with all scripts and raw data needed to reproduce the analysis.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
./run_all_groups.sh

# Results will be in OUTPUT/
```

---

## Folder Structure

```
TECAN_growth_curves/
├── scripts/
│   ├── 01_growth_curve_analysis.py      # Main Gompertz analysis pipeline
│   ├── 02_preprocess_raw_plate_data.py  # Raw 96-well data preprocessor
│   ├── 03_mle_model_fitting.py          # MLE multi-model fitting (Gompertz, Baranyi, Logistic, Richards, Haldane)
│   ├── 05_haldane_analysis.py           # Haldane feedback inhibition analysis
│   ├── 06_advanced_fitting.py           # Advanced stats (GP, Bayesian, Bootstrap, Ensemble)
│   ├── 07_compare_truncation_methods.py # Truncation method comparison & bad strain rescue
│   ├── 08_validate_truncation.py        # Interactive truncation method validator
│   ├── 09_train_classifier.py           # ML classifier training (70/30 holdout)
│   ├── ml_classifier.py                 # ML classifier runtime (PreFitGate + PostFitClassifier)
│   ├── validate_results_interactive.py  # Interactive curve validation tool
│   ├── run_all_groups.sh                # Batch processing script
│   ├── config.yaml                      # Centralized threshold configuration
│   └── requirements.txt                 # Python dependencies
├── data/raw/                            # Raw experimental data
│   ├── Group1/  (Bifenthrin, Flupyradifurone, LambdaCyhalothrin)
│   ├── Group 2/ (Malathion, LambdaCyhalothrin)
│   ├── Group 3/ (Imidacloprid, LambdaCyhalothrin)
│   └── Group 4/ (Permethrin, LambdaCyhalothrin)
├── results/tables/                      # Pipeline outputs
│   ├── all_groups_results.csv           # Consolidated Gompertz results (92 strains)
│   ├── validation_audit.csv             # Manual validation annotations
│   ├── Group{1-4}_Results/              # Per-group results + diagnostic plots
│   ├── Haldane_Analysis/                # Haldane feedback inhibition results
│   └── Advanced_Analysis/               # GP, Bayesian, Bootstrap, Comparison results
├── synthetic_data/                      # Validation test suite (480 synthetic curves)
├── tests/                               # pytest test suite
├── Makefile                             # Build/run automation
└── .github/workflows/test.yml           # CI/CD pipeline
```

---

## Scripts

### 1. `01_growth_curve_analysis.py` - Main Analysis Script

Processes growth curve data to:
- **Classify** curves as GOOD (suitable for fitting) or BAD (insufficient growth)
- **Truncate** curves at optimal point (removes death/decline phase)
- **Fit** Gompertz growth model to extract parameters
- **Generate** visualizations and summary statistics

#### Gompertz Model
```
y(t) = A * exp(-exp((mu * e / A) * (lambda - t) + 1))

Parameters:
  A      = Maximum OD600 (asymptotic value)
  mu     = Maximum specific growth rate (OD/hour)
  lambda = Lag phase duration (hours)
```

#### Usage
```bash
# Basic usage (fit-based classification + adaptive truncation)
python 01_growth_curve_analysis.py DATA/<GROUP_FOLDER> -o OUTPUT/<GROUP>_Final --adaptive

# Example
python 01_growth_curve_analysis.py "DATA/Group 2/Group_2_DATA" -o "OUTPUT/Group2_Final" --adaptive
```

#### Key Options
| Option | Description | Default |
|--------|-------------|---------|
| `--adaptive` | Optimize truncation point for best R² | Off |
| `--ml-classify` | Enable ML classifier (requires trained models in models/) | Off |
| `--od-based` | Use OD-threshold classification instead of fit-based | Fit-based |
| `--min-r2` | Minimum R² for good classification | 0.95 |
| `--max-param-error` | Maximum parameter error % | 20.0 |
| `--global-max` | Truncate at global max instead of first peak | First peak |
| `-q, --quiet` | Suppress progress output | Verbose |

#### Output Files
```
OUTPUT/Group_Final/
├── processing_results.csv      # All results in tabular format
├── classification_summary.png  # Overview visualization
└── plots/                      # Individual strain plots
    ├── STRAIN_truncation_analysis.png  # Good curves
    └── STRAIN_BAD_fit_analysis.png     # Bad curves
```

---

### 2. `02_preprocess_raw_plate_data.py` - Raw Data Preprocessor

Converts raw 96-well plate reader format (like Group 1) to the analysis-ready format.

**Input format:**
- Raw data: Columns A1-H12 (96-well positions), Time in seconds
- Key file: Maps wells to Media and Strain

**Output format:**
- Separate CSV per media type
- Columns: `TIME[H]`, `{MEDIA}_{STRAIN}_blanked`
- Triplicates averaged, blanks subtracted

#### Usage
```bash
python 02_preprocess_raw_plate_data.py <DATA_FILE> <KEY_FILE> -o <OUTPUT_DIR>

# Example for Group 1
python 02_preprocess_raw_plate_data.py \
    "DATA/Group1/GrowthRate_ Group1_Values.csv" \
    "DATA/Group1/GROUP1_KEY_V2.csv" \
    -o "DATA/Group1/Group_1_DATA"
```

---

### 3. `run_all_groups.sh` - Batch Processing Script

Runs the complete analysis pipeline for all groups automatically.

```bash
./run_all_groups.sh
```

This will:
1. Preprocess Group 1 raw data
2. Run Gompertz analysis on all groups
3. Combine results into `OUTPUT/all_groups_results.csv`

---

### 4. `05_haldane_analysis.py` - Feedback Inhibition Analysis

Fits the Haldane/Andrews substrate inhibition model to pesticide treatment curves and compares with Gompertz via AIC.

#### Haldane Model (Mechanistic)
```
dX/dt = mu(S) * X * (1 - X/X_max)    (biomass growth)
dS/dt = -q * mu(S) * X                (substrate depletion)

mu(S) = mu_max * S / (Ks + S + S²/Ki) (Haldane kinetics)

Parameters:
  mu_max = Maximum specific growth rate
  Ks     = Half-saturation constant
  Ki     = Substrate inhibition constant (lower = more inhibitory)
  X_max  = Carrying capacity
  q      = Substrate consumption coefficient
  S0     = Initial substrate concentration
```

Unlike Gompertz (which fits shape), Haldane explains **why** growth slows: substrate inhibition at high pesticide concentrations.

#### Usage
```bash
python 05_haldane_analysis.py                        # Default (all pesticide+LB strains)
python 05_haldane_analysis.py --s0 2.5               # Custom substrate concentration
python 05_haldane_analysis.py --all-strains           # Fit all strains (not just pesticide)
```

#### Output Files
```
Haldane_Analysis/
├── haldane_comparison.csv   # Per-strain Gompertz vs Haldane comparison
├── haldane_summary.csv      # Per-pesticide Ki rankings
├── haldane_overview.png     # Summary dashboard (4-panel figure)
└── plots/                   # Individual strain comparison plots
```

---

### 5. `validate_results_interactive.py` - Curve Validation Tool

Interactive matplotlib tool for visually auditing all growth curves. Shows each curve's diagnostic plot and lets you quickly classify the pipeline's decision as correct or wrong.

```bash
python validate_results_interactive.py
```

Keyboard shortcuts: `y` = correct, `n` = wrong, `u` = unsure, `space` = skip, `b` = back, `q` = quit

Saves annotations to `validation_audit.csv` (resumable).

---

### 6. `07_compare_truncation_methods.py` - Truncation Method Comparison

Compares all 5 ensemble truncation methods + consensus by fitting Gompertz at each method's truncation point. Generates per-strain overlay plots and aggregate method rankings.

```bash
# Compare methods on good strains only
python 07_compare_truncation_methods.py

# Also attempt to rescue bad strains with better truncation
python 07_compare_truncation_methods.py --include-bad

# Custom input path
python 07_compare_truncation_methods.py --ensemble-csv path/to/ensemble_results.csv
```

**Output:**
```
truncation_comparison/
├── method_comparison.csv        # Per-strain × per-method fit results
├── method_summary.csv           # Aggregate method rankings
├── strain_best_method.csv       # Best method per strain
├── rescued_strains.csv          # Bad strains rescued (with --include-bad)
├── overlay_plots/               # Per-strain method overlay plots
├── bad_strain_overlay_plots/    # Bad strain overlay plots (with --include-bad)
└── *.png                        # Aggregate comparison figures
```

---

### 7. `08_validate_truncation.py` - Interactive Truncation Validator

Interactive matplotlib tool for human review of truncation method selections. Shows each strain's raw data overlaid with all 6 Gompertz fits and lets you pick which method looks best.

```bash
python 08_validate_truncation.py
python 08_validate_truncation.py --comparison-csv path/to/method_comparison.csv
```

Keyboard shortcuts: `1-5` = select method, `c` = consensus, `n` = none good, `b` = back, `q` = quit & save

Saves annotations to `truncation_validation_audit.csv` (resumable).

---

## Makefile Commands

```bash
make help                  # Show all available commands
make install               # Install Python dependencies
make test                  # Run full test suite (114 tests)
make train-classifier      # Train ML classifier on synthetic + real data
make test-quick            # Run unit tests only (no integration)
make run-pipeline          # Run Gompertz pipeline on all groups
make run-haldane           # Run Haldane analysis on pesticide strains
make run-advanced          # Run all advanced fitting (GP + Bootstrap + Bayesian + Ensemble)
make run-advanced-gp       # GP truncation only
make run-advanced-bootstrap # Bootstrap CIs only
make run-advanced-gompertz # Bayesian Gompertz only
make run-advanced-haldane  # Bayesian Haldane only
make run-advanced-ensemble # Ensemble truncation only
make run-compare-methods   # Compare truncation methods (requires ensemble run first)
make run-compare-bad       # Compare methods + rescue bad strains
make validate              # Launch interactive curve validation tool
make validate-truncation   # Launch interactive truncation method validator
make validate-synthetic    # Run pipeline on synthetic data
make run-all               # Full pipeline + Haldane + Advanced
```

---

## Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
scipy>=1.7.0

# Advanced analysis (06_advanced_fitting.py)
pymc>=5.10.0
arviz>=0.17.0
scikit-learn>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

Or create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Classification Methods

### Fit-Based Classification (Default, Recommended)
Attempts to fit Gompertz model first, then classifies based on fit quality:
- **R² >= 0.95** (good fit to model)
- **Parameter error < 20%** for A and mu

This is more scientifically rigorous because it judges whether the curve actually follows Gompertz growth, regardless of absolute OD magnitude.

### OD-Threshold Classification (Legacy)
Classifies based on raw OD measurements:
- Delta OD >= 0.3
- Max OD >= 0.5
- Signal-to-noise ratio >= 5.0

Use `--od-based` flag to enable this mode.

---

## Truncation Methods

### Adaptive Truncation (Recommended)
Searches for the truncation point that maximizes R². Most scientifically rigorous.
```bash
python 01_growth_curve_analysis.py DATA_DIR -o OUTPUT_DIR --adaptive
```

### First Peak Truncation (Default without --adaptive)
Truncates at first local maximum after growth phase. Good for standard sigmoid curves.

### Global Max Truncation
Truncates at the absolute maximum OD600.
```bash
python 01_growth_curve_analysis.py DATA_DIR -o OUTPUT_DIR --global-max
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| `strain` | Strain identifier |
| `is_good` | Boolean classification result |
| `classification_reason` | Why curve was classified as good/bad |
| `delta_od` | OD change (max - initial) |
| `max_od` | Maximum OD600 reached |
| `r_squared` | Gompertz fit R² value |
| `gompertz_a` | Fitted A parameter (max OD) |
| `gompertz_a_err` | Standard error of A |
| `gompertz_mu` | Fitted mu parameter (growth rate, OD/hour) |
| `gompertz_mu_err` | Standard error of mu |
| `gompertz_lambda` | Fitted lambda parameter (lag phase, hours) |
| `gompertz_lambda_err` | Standard error of lambda |
| `fit_rmse` | Root mean square error of fit |
| `truncation_time` | Time (hours) where data was truncated |
| `points_used` | Number of data points used in fit |

---

## Expected Results

When you run the pipeline, you should see approximately:

| Group | Total | Good | Bad |
|-------|-------|------|-----|
| Group 1 | 24 | 12 | 12 |
| Group 2 | 24 | 16 | 8 |
| Group 3 | 24 | 17 | 7 |
| Group 4 | 20 | 12 | 8 |
| **Total** | **92** | **57** | **35** |

**Key findings:**
- LB controls show excellent growth (R² > 0.97)
- Pesticide+LB conditions show strong growth for most strains
- Pesticide-only conditions (without LB) show no/poor growth (expected)
- H2O controls show no growth (expected)

### Validation Metrics

**Rule-based (480 synthetic curves):**

| Metric | Value |
|--------|-------|
| Accuracy | 89.8% |
| Precision | 91.3% |
| Recall | 94.8% |
| F1 Score | 93.0% |
| Specificity | 77.0% |

**ML Classifier (held-out 30% test, 170 curves — synthetic + real mixed):**

| Metric | Value |
|--------|-------|
| Accuracy | 95.3% |
| Precision | 95.8% |
| Recall | 97.4% |
| F1 Score | 96.6% |
| Specificity | 90.6% |

25/32 test scenarios achieve 100% accuracy. Manual audit of 92 real curves: 91.3% validated correct.

### Truncation Method Comparison (57 good strains)

The weighted-median consensus achieves the highest mean R² (0.9804). All top-5 methods perform within 0.002 of each other. 6 of 9 candidate bad strains were rescued by alternative truncation, potentially increasing good strain count from 57 to 63.

### Haldane Analysis Results

The Haldane model was preferred over Gompertz for **15/23** (65%) pesticide+LB strains by AIC, confirming substrate inhibition kinetics are present. All 6 pesticides tested (Bifenthrin, Flupyradifurone, Imidacloprid, Lambda-Cyhalothrin, Malathion, Permethrin) show measurable inhibition constants (Ki).

---

## Advanced Statistical Methods (`06_advanced_fitting.py`)

Upgrades every pipeline stage with modern statistical methods, addressing PI recommendations for Monte Carlo, Gaussian Processes, HMC, and Bayesian Hierarchical Models.

| Pipeline Stage | Before | After | Method |
|---------------|--------|-------|--------|
| **Truncation** | Heuristic peak-finding | GP + 5-method ensemble consensus | Gaussian Process + Ensemble |
| **Gompertz fitting** | scipy curve_fit (point estimates) | Full posteriors, partial pooling | Bayesian HMC (NUTS) |
| **Haldane fitting** | scipy minimize (point estimates) | Hierarchical posteriors, Ki by pesticide | Bayesian DEMetropolisZ |
| **Classification** | Hard R²/SNR thresholds | P(good) from posterior | Bayesian posterior |
| **Model comparison** | AIC (ΔAIC > 2) | WAIC/LOO with uncertainty | Bayesian model comparison |
| **Uncertainty** | None | Bootstrap CIs + full posteriors | Monte Carlo bootstrap |

### Usage

```bash
# Run everything
python 06_advanced_fitting.py

# Run specific phases
python 06_advanced_fitting.py --gp-only          # GP truncation only
python 06_advanced_fitting.py --bootstrap-only    # Bootstrap CIs only
python 06_advanced_fitting.py --gompertz-only     # Bayesian Gompertz only
python 06_advanced_fitting.py --haldane-only      # Bayesian Haldane only
python 06_advanced_fitting.py --ensemble-only     # Ensemble truncation only
python 06_advanced_fitting.py --no-haldane        # Everything except slow Haldane ODE

# Performance tuning
python 06_advanced_fitting.py --chains 2 --draws 500 --thin 50    # Quick test
python 06_advanced_fitting.py --max-strains 10 --thin 30          # Limit strains
```

### GP-Based Truncation

Fits a Gaussian Process (RBF + WhiteKernel) to each growth curve and identifies growth phases from the GP derivative. The truncation point is where dOD/dt first crosses zero after reaching its maximum (end of exponential growth). No smoothing window or heuristic parameters needed — the GP learns the noise level automatically.

### Bayesian Hierarchical Gompertz

Three-level hierarchy: Population → Pesticide group → Strain. Uses non-centered parameterization to avoid funnel geometry. Sampled with NUTS (No U-Turn Sampler). Produces full posterior distributions on A, mu, and lambda for every strain with partial pooling by treatment group.

### Bayesian Hierarchical Haldane

Ki (substrate inhibition constant) is partially pooled by pesticide via a LogNormal hierarchy. Uses a custom PyTensor Op wrapping scipy's ODE solver and DEMetropolisZ (gradient-free sampler). Produces P(Ki < threshold) for each pesticide — directly answering "which pesticide is most inhibitory?"

### Bootstrap Uncertainty

Residual resampling on NLS Gompertz fits (1000 resamples). Produces 95% CIs on all parameters and P(good) classification confidence. Runs in seconds per strain (vs minutes for full Bayesian).

### Output

```
results/tables/Advanced_Analysis/
├── gp_truncation/                    # GP phase detection results + plots
├── ensemble_truncation/              # 5-method ensemble truncation results
├── bayesian_gompertz/                # Posterior summaries, traces, convergence
├── bayesian_haldane/                 # Ki posteriors by pesticide
├── bootstrap/                       # Bootstrap CIs and P(good)
├── classification/                  # Bayesian classification (P(good) per strain)
├── model_comparison.csv             # WAIC/LOO Gompertz vs Haldane
└── truncation_comparison/           # Method comparison study (07_compare)
    ├── method_comparison.csv        # Per-strain × per-method fits
    ├── method_summary.csv           # Aggregate rankings
    ├── rescued_strains.csv          # Rescued bad strains
    └── overlay_plots/               # Visual method overlays
```

### Additional Dependencies

```
pymc>=5.10.0          # Bayesian modeling (NUTS, DEMetropolisZ)
arviz>=0.17.0         # Posterior analysis, diagnostics (R-hat, ESS, HDI)
scikit-learn>=1.0.0   # Gaussian Process regression
```

### Configuration

All advanced settings are in `config.yaml` under the `advanced:` section (GP kernel parameters, Bayesian sampling settings, prior hyperparameters, bootstrap resamples, convergence thresholds).

---

### 8. `09_train_classifier.py` - ML Classifier Training

Trains a two-stage ML classifier for growth curve quality classification:

1. **Pre-fit gate** (8 raw features + 2 metadata) — rejects obvious junk before expensive truncation/fitting
2. **Post-fit classifier** (22 fit features + 2 metadata) — three-class output: GOOD / BORDERLINE / BAD

Features include biological metadata (`is_control`, `concentration_numeric`) parsed from strain names, enabling the model to learn that water/LB controls with growth signals are suspicious.

```bash
# Train with 70/30 holdout validation
python 09_train_classifier.py

# Compare GBT vs Random Forest
python 09_train_classifier.py --compare

# Use the classifier in the pipeline
python 01_growth_curve_analysis.py DATA_DIR -o OUTPUT_DIR --ml-classify
```

**Training data:** 480 synthetic + 85 real audited curves (7 "unsure" excluded).
**Held-out test (30%):** 95.3% accuracy, 90.6% specificity, 97.4% recall.

### `ml_classifier.py` - ML Classifier Runtime Module

Runtime module loaded by `01_growth_curve_analysis.py` when `--ml-classify` is enabled. Contains:
- `extract_prefit_features(time, od600, strain_name)` — 10 raw features from OD signal + metadata
- `extract_postfit_features(fit_result, metrics, ...)` — 24 features from Gompertz fit
- `extract_metadata_features(strain_name)` — parses is-control and concentration from strain names
- `PreFitGate` — loads joblib model, `should_skip()` returns bool
- `PostFitClassifier` — loads joblib model, `classify()` returns GOOD/BORDERLINE/BAD with P(good)

---

## Troubleshooting

### "python3: command not found"
Edit `run_all_groups.sh` and change `PYTHON="python3"` to your Python path (e.g., `PYTHON="python"` or `PYTHON="/usr/bin/python3"`).

### Permission denied on run_all_groups.sh
```bash
chmod +x run_all_groups.sh
```

### Missing dependencies
```bash
pip install pandas numpy matplotlib scipy
```

---

## Contact

For questions about this analysis pipeline, contact the BIO380SP25 research team.
