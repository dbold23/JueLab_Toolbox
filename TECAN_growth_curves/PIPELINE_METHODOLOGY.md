# TECAN Growth Curve Analysis Pipeline: Complete Methodology & Documentation

## BIO380SP25 — Pesticide Bioremediating Bacteria Research Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure & File Map](#2-directory-structure--file-map)
3. [Pipeline Architecture (10-Step Data Flow)](#3-pipeline-architecture-10-step-data-flow)
4. [Step 0 — Raw Data Preprocessing](#4-step-0--raw-data-preprocessing)
5. [Step 1 — Train ML Classifier](#5-step-1--train-ml-classifier)
6. [Step 2 — Gompertz Curve Analysis](#6-step-2--gompertz-curve-analysis)
7. [Step 3 — Combine Group Results](#7-step-3--combine-group-results)
8. [Step 4 — Haldane Substrate Inhibition](#8-step-4--haldane-substrate-inhibition)
9. [Step 5 — Advanced Fitting](#9-step-5--advanced-fitting)
10. [Step 6 — Statistical Analysis](#10-step-6--statistical-analysis)
11. [Step 7 — Truncation Comparison](#11-step-7--truncation-comparison)
12. [Step 8 — Export for Collaboration](#12-step-8--export-for-collaboration)
13. [Step 9 — Synthetic Validation](#13-step-9--synthetic-validation)
14. [Configuration Reference](#14-configuration-reference)
15. [Execution & Reproducibility](#15-execution--reproducibility)
16. [Results Summary](#16-results-summary)

---

## 1. Project Overview

This pipeline analyzes bacterial growth curves from **TECAN plate reader** experiments measuring the optical density (OD600) of bacteria over time. The goal is to determine which bacterial strains can grow in the presence of various pesticides — testing their potential as **pesticide bioremediators**.

**Core research question:** Can these bacteria metabolize common pesticides (bifenthrin, flupyradifurone, imidacloprid, lambda-cyhalothrin, malathion, permethrin) as a carbon/energy source?

**What the pipeline does (10 steps):**

0. Preprocesses raw 96-well plate reader output into clean, blank-subtracted time series
1. Trains a two-stage ML classifier (pre-fit gate + post-fit HistGBT) on the synthetic suite only (1059 curves enter training); the 85 audited real curves are an external test set
2. Truncates, fits modified Gompertz growth model, and classifies each curve as "GOOD" or "BAD" (per group)
3. Combines per-group results into a consolidated master results file
4. Fits a mechanistic Haldane substrate-inhibition model to pesticide-treated strains
5. Applies advanced Bayesian hierarchical models, Gaussian processes, and bootstrap methods
6. Performs statistical analysis and generates publication-quality figures
7. Compares 5 truncation methods via Gompertz fit quality, rescues misclassified bad strains
8. Exports clean CSV files and methodology documentation for collaboration
9. Validates the entire pipeline against 1139 synthetic ground-truth curves

**Scale:** 92 bacterial strains across 4 experimental groups, each tested in multiple media conditions (LB, pesticide+LB, pesticide-only, H2O control).

---

## 2. Directory Structure & File Map

```
TECAN_growth_curves/
│
├── scripts/                              # ──── MAIN ANALYSIS PIPELINE ────
│   ├── 01_growth_curve_analysis.py       # PRIMARY: Truncation + Gompertz + classification
│   ├── 02_preprocess_raw_plate_data.py   # Raw 96-well plate → analysis-ready CSVs
│   ├── 03_mle_model_fitting.py           # Multi-model MLE (Gompertz, Baranyi, Logistic, etc.)
│   ├── 03_statistical_analysis.py        # Statistical tests across groups
│   ├── 04_export_for_collaboration.py    # Export formatting for collaborators
│   ├── 05_haldane_analysis.py            # Substrate inhibition kinetics (Haldane/Andrews)
│   ├── 06_advanced_fitting.py            # GP, Bayesian hierarchical, Bootstrap methods
│   ├── 07_compare_truncation_methods.py  # Truncation method comparison study
│   ├── 08_validate_truncation.py         # Interactive truncation validator
│   ├── 09_train_classifier.py            # ML classifier training (70/30 stratified split)
│   ├── ml_classifier.py                  # ML classifier runtime (PreFitGate, PostFitClassifier)
│   ├── validate_results_interactive.py   # Interactive curve auditor
│   ├── run_all_groups.sh                 # Batch processing shell script
│   ├── config.yaml                       # Central configuration (all thresholds)
│   └── requirements.txt                  # Python dependencies
│
├── data/raw/                             # ──── RAW EXPERIMENTAL DATA ────
│   ├── Group1/
│   │   ├── GrowthRate_ Group1_Values.csv # Raw plate reader output (96-well)
│   │   ├── GROUP1_KEY_V2.csv             # Well-to-strain mapping key
│   │   └── Group_1_DATA/                 # Preprocessed CSVs (per media)
│   ├── Group 2/
│   │   └── Group_2_DATA/                 # Preprocessed CSVs
│   ├── Group 3/
│   │   └── Group_3_DATA/                 # Preprocessed CSVs
│   └── Group 4/
│       └── Group4_DATA/                  # Preprocessed CSVs
│
├── results/tables/                       # ──── PIPELINE OUTPUTS ────
│   ├── all_groups_results.csv            # Consolidated 92-strain master results
│   ├── results_for_R_import.csv          # R-compatible export
│   ├── Group1_Results/ ... Group4_Results/
│   ├── Haldane_Analysis/                 # Substrate inhibition results + plots
│   └── Advanced_Analysis/                # GP, Bayesian, Bootstrap outputs
│       ├── ensemble_truncation/          #   Ensemble truncation results per strain
│       └── truncation_comparison/        #   Method comparison study outputs
│           ├── method_comparison.csv     #     Per-strain × per-method fit results
│           ├── method_summary.csv        #     Aggregate method rankings
│           ├── strain_best_method.csv    #     Best method per strain
│           ├── rescued_strains.csv       #     Bad strains rescued by better truncation
│           ├── overlay_plots/            #     Per-strain overlay plots (good strains)
│           └── bad_strain_overlay_plots/ #     Per-strain overlay plots (bad strains)
│
├── synthetic_data/                       # ──── VALIDATION FRAMEWORK ────
│   ├── src/                              # Core synthetic generation modules
│   │   ├── growth_models.py              # 5 growth models implemented
│   │   ├── noise_models.py               # 4 noise types
│   │   ├── curve_scenarios.py            # 32 test scenarios
│   │   └── data_generator.py             # Orchestrator
│   ├── validation/                       # Validation metrics & reporting
│   └── output/comprehensive_test/        # 1139 synthetic curves + validation reports
│
├── tests/                                # ──── UNIT & INTEGRATION TESTS ────
│   ├── test_gompertz.py
│   ├── test_haldane.py
│   ├── test_classification.py
│   ├── test_integration.py
│   ├── test_advanced.py                  # Ensemble truncation, GP, Bootstrap tests
│   ├── test_truncation_improvements.py   # MCCV truncation, incomplete curve detection
│   └── test_ml_classifier.py             # ML classifier feature extraction, metadata, gates
│
├── models/                               # ──── TRAINED ML MODELS ────
│   ├── prefit_gate.joblib                # Pre-fit gate model (gitignored)
│   ├── postfit_classifier.joblib         # Post-fit classifier (gitignored)
│   └── feature_config.json              # Feature names + training metrics (committed)
│
├── Makefile                              # Build automation (18 targets)
└── .github/workflows/test.yml            # CI/CD pipeline
```

### Which file does what

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `02_preprocess_raw_plate_data.py` | Converts raw 96-well plate data | Raw CSV + key file | Per-media CSVs |
| `01_growth_curve_analysis.py` | Core pipeline: truncate → fit → classify | Per-media CSVs | `processing_results.csv` + plots |
| `05_haldane_analysis.py` | Mechanistic inhibition kinetics | Results CSV + raw data | Haldane comparison tables + plots |
| `06_advanced_fitting.py` | GP, Bayesian, Bootstrap, Ensemble methods | Results CSV + raw data | Advanced analysis outputs |
| `07_compare_truncation_methods.py` | Truncation method comparison & bad strain rescue | Ensemble results + raw data | Method rankings, overlay plots, rescued strains |
| `08_validate_truncation.py` | Interactive truncation method validator | `method_comparison.csv` + raw data | `truncation_validation_audit.csv` |
| `09_train_classifier.py` | Train ML classifier (70/30 holdout) | Synthetic curves (real audited curves held out as an external test set) | `models/*.joblib` + `feature_config.json` |
| `ml_classifier.py` | ML runtime (feature extraction, gates) | Raw OD data + fit results | Classification with probabilities |

---

## 3. Pipeline Architecture (10-Step Data Flow)

```
Step 0: Preprocess raw data       → *_DATA.csv
Step 1: Train ML classifier       → models/*.joblib
Step 2: Gompertz curve analysis    → processing_results.csv (per group)
Step 3: Combine group results      → all_groups_results.csv
Step 4: Haldane inhibition         → haldane_comparison.csv
Step 5: Advanced fitting           → Advanced_Analysis/
Step 6: Statistical analysis       → publication figures
Step 7: Truncation comparison      → method rankings + rescued strains
Step 8: Export for collaboration    → clean CSV + methodology
Step 9: Synthetic validation       → accuracy / recall / F1 metrics
```

**Detailed data flow:**

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     RAW DATA ACQUISITION                           │
  │  TECAN plate reader → 96-well OD600 readings every 15 min         │
  │  + Well mapping key (which well = which strain + media)            │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 0: PREPROCESSING  (02_preprocess_raw_plate_data.py)          │
  │  • Time: seconds → hours                                          │
  │  • Triplicate averaging                                            │
  │  • Blank subtraction (per media)                                   │
  │  • Output: {MEDIA}_DATA.csv files                                  │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 1: TRAIN ML CLASSIFIER  (09_train_classifier.py)             │
  │  • 1059 synthetic curves train it; 85 real audited curves are      │
  │    an external test set and are never trained on                   │
  │  • Two-stage: PreFitGate + PostFitClassifier (HistGBT)             │
  │  • Output: models/*.joblib + feature_config.json                   │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 2: GOMPERTZ CURVE ANALYSIS  (01_growth_curve_analysis.py)    │
  │  • Truncation: rolling average smoothing + first local max         │
  │  • Gompertz model: y(t) = A·exp(-exp((ue/A)(l-t)+1))              │
  │  • Classification: R² >= 0.95, param error < 20%, quality gates    │
  │  • Output: processing_results.csv + diagnostic plots (per group)   │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 3: COMBINE GROUP RESULTS                                     │
  │  • Merge per-group processing_results.csv files                    │
  │  • Output: all_groups_results.csv (92 strains)                     │
  └────────┬────────────────────────────────────────────┬──────────────┘
           │                                            │
           ▼                                            ▼
  ┌────────────────────┐                    ┌───────────────────────────┐
  │  STEP 4: HALDANE   │                    │  STEP 5: ADVANCED         │
  │  (05_haldane.py)   │                    │  (06_advanced_fitting.py) │
  │  • ODE substrate   │                    │  • GP truncation          │
  │    inhibition      │                    │  • Ensemble truncation    │
  │  • AICc comparison │                    │  • Bayesian Gompertz      │
  │    vs Gompertz     │                    │  • Bayesian Haldane       │
  └────────────────────┘                    │  • Bootstrap CIs          │
                                            │  • WAIC/LOO comparison    │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 6: STATISTICAL ANALYSIS  (03_statistical_analysis.py)        │
  │  • Cross-group statistical tests                                   │
  │  • Publication-quality figures                                      │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 7: TRUNCATION COMPARISON & RESCUE                            │
  │  (07_compare_truncation_methods.py / 08_validate_truncation.py)    │
  │  • Compare 5 methods + consensus via per-strain Gompertz R²        │
  │  • Overlay plots: raw data + 6 Gompertz curves per strain          │
  │  • Bad strain rescue: refit BAD strains, keep only gate-passing     │
  │  • Interactive validator: human review of method selections         │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 8: EXPORT FOR COLLABORATION  (04_export_for_collaboration.py)│
  │  • Clean CSV formatting for collaborators                          │
  │  • Methodology documentation export                                │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 9: SYNTHETIC VALIDATION  (synthetic_data/)                   │
  │  • 1139 synthetic ground-truth curves across 64 scenarios          │
  │  • First-peak rule, 960 scored: balanced accuracy 87.9%,           │
  │    sensitivity 85.6%, specificity 90.2%, accuracy 87.6%            │
  │  • Visual audit of real data: 74/85 confident calls reproduced     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1 — Raw Data Preprocessing

**Script:** `scripts/02_preprocess_raw_plate_data.py`

### 4.1 What the raw data looks like

The TECAN plate reader produces a CSV file with 96 columns (one per well: A1, A2, ..., H12) and rows representing time points. A separate **key file** maps each well to a `Media` type and `Strain` identifier.

**Raw plate reader format:**
```
Time [s],A1,A2,A3,...,H12
0,0.0893,0.0901,0.0876,...,0.0912
900,0.0901,0.0913,0.0888,...,0.0920
1800,0.0915,0.0927,0.0901,...,0.0935
...
```

**Key file format:**
```
Cell,Media,Strain
A1,LB,MAL8
A2,LB,MAL8
A3,LB,MAL8
A4,LB,Blank
...
```

### 4.2 Preprocessing steps

The script performs four operations in sequence:

#### Step 1 — Time conversion (seconds → hours)

```python
# From 02_preprocess_raw_plate_data.py, line 85
time_hours = data_df['Time [s]'] / 3600.0
```

This converts the TECAN's native seconds to hours for biological interpretability.

#### Step 2 — Identify blank wells

For each media type, find wells labeled "Blank" in the key file and compute their average OD at each time point. This represents the background absorbance of the media itself.

```python
# From 02_preprocess_raw_plate_data.py, lines 114-118
blank_wells = get_wells_for_condition(key_df, media, 'Blank')
if len(blank_wells) > 0:
    blank_data = data_df[blank_wells].values
    blank_avg = np.mean(blank_data, axis=1)
```

#### Step 3 — Average triplicates

Each strain is measured in 3 replicate wells. These are averaged to reduce technical noise:

```python
# From 02_preprocess_raw_plate_data.py, lines 140-143
strain_data = data_df[wells].values
strain_avg = np.mean(strain_data, axis=1)
```

#### Step 4 — Blank subtraction and floor to zero

Subtract the blank average from each strain's triplicate average. Negative values (noise artifacts) are clipped to zero:

```python
# From 02_preprocess_raw_plate_data.py, lines 146-149
strain_blanked = strain_avg - blank_avg
strain_blanked = np.maximum(strain_blanked, 0)  # Ensure non-negative
```

### 4.3 Output format

One CSV file is produced per media type (e.g., `LB_DATA.csv`, `MalathionANDLB_DATA.csv`):

```csv
TIME[H],LB_MAL8_blanked,LB_MAL10_blanked,LB_MAL1_blanked,...
0.0,0.0275,0.0353,0.0,...
0.25,0.0341,0.0327,0.0,...
0.50,0.0405,0.0339,0.0,...
```

Each column name follows the pattern `{MEDIA}_{STRAIN}_blanked`, making it trivially parseable by downstream scripts.

---

## 5. Stage 2 — Growth Curve Truncation

**Script:** `scripts/01_growth_curve_analysis.py` — Truncation section (lines 298–747)

### 5.1 Why truncation is necessary

Bacterial growth curves have distinct phases:

1. **Lag phase** — bacteria adapt to new media (flat OD)
2. **Exponential phase** — rapid growth (steep rise)
3. **Stationary phase** — nutrients depleted, growth stops (plateau)
4. **Death phase** — cells die, OD declines

The Gompertz model describes phases 1–3 as a **monotonic sigmoid**. If the death phase (4) is included, the model fit degrades because it tries to fit a non-monotonic decline with a monotonic curve. Truncation removes post-peak data to ensure clean fitting.

### 5.2 Smoothing

Before finding the peak, the raw OD data is smoothed using a centered rolling average to prevent noise spikes from being mistaken for the true maximum:

```python
# From 01_growth_curve_analysis.py, lines 789-793
smoothed_od = pd.Series(od600).rolling(
    window=smoothing_window,    # default: 5
    center=True,
    min_periods=1
).mean().values
```

A window of 5 time points (typically 1.25 hours) provides enough smoothing to remove measurement noise without blurring real biological transitions.

### 5.3 Method 1: First Local Maximum Detection (Default)

This is the **primary truncation method**. Rather than finding the global maximum (which could be a noise spike late in the experiment), it finds the **first biologically meaningful peak** — the end of the primary growth phase.

```python
# From 01_growth_curve_analysis.py, lines 298-370
def find_first_local_maximum(
    od600: np.ndarray,
    smoothed_od: np.ndarray,
    min_growth_threshold: float = 0.1,
    plateau_window: int = 10
) -> int:
    # Calculate derivative of smoothed data
    derivative = np.diff(smoothed_od)

    # Find where growth has meaningfully started (above threshold)
    growth_started_idx = 0
    for i, val in enumerate(smoothed_od):
        if val > min_growth_threshold:
            growth_started_idx = i
            break

    # Find the first point where derivative goes from positive to negative
    # after growth has started
    first_peak_idx = None
    for i in range(growth_started_idx, len(derivative) - plateau_window):
        if i < 5:
            continue

        recent_growth = np.mean(derivative[max(0, i-5):i])
        upcoming_trend = np.mean(derivative[i:i+plateau_window])

        # Detect peak: was growing, now flattening/declining
        if recent_growth > 0.001 and upcoming_trend <= 0.005:
            if smoothed_od[i] > 0.5 * np.max(smoothed_od):
                first_peak_idx = i
                break

    # Fallback to global max if no local peak found
    if first_peak_idx is None:
        first_peak_idx = np.argmax(smoothed_od)

    return first_peak_idx
```

**Algorithm logic:**
1. Find where OD exceeds 0.1 (growth has started — ignore lag-phase noise)
2. Compute the derivative (rate of change) of the smoothed curve
3. Scan forward looking for where the derivative transitions from positive (growing) to near-zero/negative (plateau/decline)
4. Require the peak to be at least 50% of the global maximum (prevents premature triggering on minor fluctuations)
5. Fall back to global maximum if no local peak is detected

### 5.4 Method 2: Adaptive R²-Maximizing Truncation (Optional)

This method searches for the truncation point that produces the best Gompertz fit, balancing biological relevance with statistical fit quality:

```python
# From 01_growth_curve_analysis.py, lines 567-747
def find_optimal_truncation(time, od600, min_points=20, trim_noisy_start=True):
    # Step 0: Find where real growth begins (trim noisy start)
    start_idx, start_metadata = find_growth_start(time, od600)

    # Step 1: Find biologically-motivated truncation point
    bio_truncation_idx, bio_metadata = find_stationary_phase_start(time, od600)

    # Step 2: Define search window around the biological truncation point
    search_start = max(start_idx + min_points, int(bio_truncation_idx * 0.70))
    search_end = min(n_points - 1, int(bio_truncation_idx * 1.20))

    # Step 3: Test Gompertz fits at each candidate truncation point
    for end_idx in range(search_start, search_end, step):
        t_trunc = time[start_idx:end_idx]
        od_trunc = od600[start_idx:end_idx]

        popt, _ = curve_fit(gompertz_model, t_trunc, od_trunc, ...)

        predicted = gompertz_model(t_trunc, *popt)
        ss_res = np.sum((od_trunc - predicted)**2)
        ss_tot = np.sum((od_trunc - np.mean(od_trunc))**2)
        r2 = 1 - ss_res/ss_tot

        # Prefer higher R², but when R² is similar, prefer
        # truncation points closer to the biological estimate
        if r2 > best_r2 + 0.005:
            best_r2 = r2
            best_end_idx = end_idx
        elif r2 >= best_r2 - 0.005:
            if dist_to_bio < best_dist_to_bio:
                best_r2 = r2
                best_end_idx = end_idx

    return start_idx, best_end_idx, best_r2, metadata
```

**Key design decisions:**
- The search window is centered on the biologically-estimated truncation point (±30%), not the entire curve
- When two candidate points yield similar R² (within 0.005), the one closer to the biological estimate is preferred — this prevents overfitting to noise
- A coarse search (step size ~1/30 of range) followed by fine-tuning (every point around best) keeps computation tractable

### 5.5 Truncation validation

After truncation, the result is checked:

```python
# From 01_growth_curve_analysis.py, lines 845-871
def validate_truncation(result, min_points=20):
    if len(result.od_truncated) < min_points:
        return False, f"Too few points after truncation: {len(result.od_truncated)} < {min_points}"

    total_duration = result.time_original[-1] - result.time_original[0]
    time_to_max_ratio = result.truncation_time / total_duration

    if time_to_max_ratio < 0.05:  # Max within first 5% of experiment
        return False, "Maximum too early in experiment"

    return True, "Truncation valid"
```

This catches two failure modes:
- **Too few points**: Fewer than 20 data points after truncation means the fit would be unreliable
- **Maximum too early**: If the peak is in the first 5% of the experiment, something is wrong (e.g., contamination caused a high initial OD)

---

## 6. Stage 3 — Gompertz Model Fitting

**Script:** `scripts/01_growth_curve_analysis.py` — Fitting section (lines 878–1003)

### 6.1 The Modified Gompertz Model

The modified Gompertz model describes a sigmoid growth curve with three biologically interpretable parameters:

**Mathematical form:**

```
y(t) = A × exp( -exp( (μ × e / A) × (λ - t) + 1 ) )
```

**Parameters:**

| Symbol | Name | Biological meaning | Units |
|--------|------|-------------------|-------|
| A | Asymptote | Maximum OD600 (carrying capacity) | OD units |
| μ | Maximum specific growth rate | Steepest slope of the sigmoid | OD/hour |
| λ | Lag time | Duration before exponential growth begins | hours |

**Python implementation:**

```python
# From 01_growth_curve_analysis.py, lines 878-900
def gompertz_model(t, a, mu, lam):
    """
    Modified Gompertz growth model.
    y(t) = a * exp(-exp((mu * e / a) * (lambda - t) + 1))
    """
    return a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1))
```

The `e` in the equation is Euler's number (≈2.718), ensuring that μ represents the **actual** maximum growth rate at the inflection point (not a model-specific scaling).

### 6.2 Fitting procedure

Fitting uses `scipy.optimize.curve_fit`, which implements the **Levenberg-Marquardt** algorithm — a damped least-squares optimizer well-suited for nonlinear models:

```python
# From 01_growth_curve_analysis.py, lines 903-1003
def fit_gompertz(time, od600, max_iterations=5000):
    # ---- STEP 1: Generate intelligent initial guesses ----
    a_init = np.max(od600)                                      # max OD as A estimate

    diff = np.diff(od600)
    dt = np.diff(time)
    growth_rates = diff / dt
    mu_init = max(np.max(growth_rates), 0.01)                   # steepest slope as μ

    max_growth_idx = np.argmax(growth_rates)
    lambda_init = time[max_growth_idx]                           # time of steepest slope as λ

    p0 = [a_init, mu_init, lambda_init]

    # ---- STEP 2: Set parameter bounds ----
    bounds = (
        [0.01,           0.001,              0          ],      # lower bounds
        [3 * a_init,     10 * mu_init + 1,   time[-1]   ]       # upper bounds
    )

    # ---- STEP 3: Fit using Levenberg-Marquardt ----
    popt, pcov = curve_fit(
        gompertz_model, time, od600,
        p0=p0, bounds=bounds, maxfev=max_iterations
    )

    a_opt, mu_opt, lambda_opt = popt

    # ---- STEP 4: Extract parameter uncertainties ----
    # Standard errors from the covariance matrix diagonal
    perr = np.sqrt(np.diag(pcov))
    a_err, mu_err, lambda_err = perr

    # ---- STEP 5: Compute fit quality metrics ----
    predicted = gompertz_model(time, *popt)
    residuals = od600 - predicted

    rmse = np.sqrt(np.mean(residuals**2))      # Root Mean Square Error
    mae  = np.mean(np.abs(residuals))           # Mean Absolute Error

    # R-squared (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((od600 - np.mean(od600))**2)
    r_squared = 1 - (ss_res / ss_tot)
```

### 6.3 Initial guess generation — why it matters

Nonlinear optimization requires a starting point. Poor initial guesses can cause the optimizer to converge to a local minimum or fail entirely. The pipeline uses data-driven heuristics:

| Parameter | Initial guess strategy | Rationale |
|-----------|----------------------|-----------|
| A | `max(OD)` | The asymptote can't be less than the observed maximum |
| μ | `max(dOD/dt)` | The steepest observed slope approximates the inflection point growth rate |
| λ | `time[argmax(dOD/dt)]` | The time of fastest growth is near the end of the lag phase |

### 6.4 Parameter bounds

Bounds prevent the optimizer from exploring physically impossible parameter space:

| Parameter | Lower | Upper | Reason |
|-----------|-------|-------|--------|
| A | 0.01 | 3 × max(OD) | Must be positive; allow headroom above observed max |
| μ | 0.001 | 10 × max(slope) + 1 | Must be positive; generous upper bound |
| λ | 0 | last time point | Can't be negative; can't exceed experiment duration |

### 6.5 Error estimation

The covariance matrix `pcov` from `curve_fit` contains the estimated variance of each parameter. The standard error is its square root:

```python
perr = np.sqrt(np.diag(pcov))
```

This gives the ±1σ confidence interval on each parameter. The **relative error** (as a percentage) is then `error / value × 100%`.

### 6.6 R² calculation

R-squared measures how much variance the model explains:

```
R² = 1 - SS_residual / SS_total
    = 1 - Σ(observed - predicted)² / Σ(observed - mean(observed))²
```

- R² = 1.0 → perfect fit
- R² = 0.95 → model explains 95% of variance (pipeline threshold for "GOOD")
- R² < 0.0 → model is worse than a horizontal line at the mean

---

## 7. Stage 4 — Classification (Good vs Bad)

**Script:** `scripts/01_growth_curve_analysis.py` — Classification section (lines 176–291)

### 7.1 Classification philosophy

Rather than using absolute OD thresholds ("did the bacteria grow past OD 0.5?"), the pipeline uses a **fit-quality-based** approach: "can the growth curve be well-described by the Gompertz model?" This is more scientifically rigorous because:

- A strain with low maximum OD but a clean sigmoid shape represents real (albeit limited) growth
- A strain with high OD but noisy/erratic readings represents poor data quality
- The Gompertz parameters (A, μ, λ) are biologically meaningful only when the fit is good

### 7.2 Primary classification criteria

```python
# From 01_growth_curve_analysis.py, lines 36-40
FIT_QUALITY_THRESHOLDS = {
    'min_r_squared': 0.95,         # Minimum R² for a good fit
    'max_param_error_pct': 20.0,   # Maximum relative error (%) for parameters
    'min_points_for_fit': 15,      # Minimum data points to attempt fit
}
```

A curve is classified as **GOOD** if and only if:
1. The Gompertz fit R² ≥ 0.95
2. The relative error of A is ≤ 20%
3. The relative error of μ is ≤ 20%
4. There were at least 15 data points after truncation

### 7.3 Secondary quality gates

Even if the Gompertz fit looks good on paper, additional checks catch edge cases:

```python
# From 01_growth_curve_analysis.py, lines 250-291
# (inside classify_by_fit_quality function)

# --- Signal-to-Noise Ratio ---
baseline = od600[:n_baseline]
baseline_std = np.std(baseline)
snr = (max_od - baseline_mean) / baseline_std

# Only enforce noise-based gates when fit quality is NOT excellent
fit_is_excellent = fit_result.r_squared >= 0.98

if not fit_is_excellent:
    # SNR filter: signal must meaningfully exceed noise floor
    if snr < 5.0:
        reasons.append(f"Low SNR: {snr:.1f} < 5.0")

    # Delta-OD confidence interval: growth must be statistically significant
    delta_od_ci_lower = delta_od - 2 * baseline_std
    if delta_od_ci_lower < 0.1:
        reasons.append(f"Delta OD not significant: CI lower={delta_od_ci_lower:.3f} < 0.1")

# Absolute minimum delta-OD (always enforced, even for excellent R²)
if delta_od < 0.15:
    reasons.append(f"Insufficient growth: delta_od={delta_od:.3f} < 0.15")

# Flatness check: fitted A must exceed noise level (always enforced)
if fit_result.a_opt < 3 * baseline_std:
    reasons.append(f"Growth amplitude below noise: A={fit_result.a_opt:.3f}")
```

### 7.4 The "excellent R² bypass" logic

A key design decision: **if R² ≥ 0.98, the noise-based gates (SNR, delta-OD CI) are bypassed**. The rationale is that if the Gompertz model explains 98%+ of variance, the signal is clearly above noise — enforcing SNR thresholds would be redundant and could reject valid low-OD curves with clean sigmoid shapes.

However, two gates are **always enforced** regardless of R²:
- Absolute minimum delta-OD (0.15): Even a perfect fit to a flat line is not growth
- Flatness check (A > 3×noise): The fitted amplitude must be biologically meaningful

### 7.5 Classification decision tree

```
START
  │
  ├─ Fit failed? ──────────────────────────→ BAD (fit failure)
  │
  ├─ R² < 0.95? ──────────────────────────→ BAD (poor fit)
  │
  ├─ A error > 20% or μ error > 20%? ─────→ BAD (uncertain parameters)
  │
  ├─ delta_OD < 0.15? ────────────────────→ BAD (insufficient growth)
  │
  ├─ A < 3 × baseline noise? ─────────────→ BAD (below noise floor)
  │
  ├─ R² ≥ 0.98? ──────────────────────────→ GOOD (excellent fit bypasses noise gates)
  │
  ├─ SNR < 5.0? ──────────────────────────→ BAD (poor signal-to-noise)
  │
  ├─ delta_OD CI lower < 0.1? ────────────→ BAD (growth not statistically significant)
  │
  └─ All checks passed ───────────────────→ GOOD
```

### 7.6 Output: processing_results.csv

Each row in the output file contains:

```csv
strain,is_good,classification_reason,delta_od,max_od,initial_od,
r_squared,gompertz_a,gompertz_a_err,gompertz_mu,gompertz_mu_err,
gompertz_lambda,gompertz_lambda_err,truncation_time,points_used,fit_rmse,fit_mae
```

Example:
```csv
BifenthrinANDLB-BIF2,True,GOOD: Fit quality passed,1.619,1.632,0.013,
0.988,1.608,0.015,0.170,0.005,0.438,0.141,19.5,79,0.057,0.040
```

---

## 8. Stage 5 — Haldane Substrate Inhibition Analysis

**Script:** `scripts/05_haldane_analysis.py`

### 8.1 Biological motivation

The Gompertz model is **phenomenological** — it describes the shape of growth but says nothing about the mechanism. For strains growing on pesticide+LB, we want to ask a **mechanistic** question: is the pesticide acting as a substrate that inhibits growth at high concentrations?

The **Haldane/Andrews model** describes this exactly: at low substrate concentrations, growth rate increases (Monod kinetics); at high concentrations, substrate inhibits growth (feedback inhibition). This produces a bell-shaped curve of growth rate vs. substrate concentration.

### 8.2 The Haldane ODE system

The model is a system of two coupled **ordinary differential equations** (ODEs):

```
dX/dt = μ(S) × X × (1 - X/X_max)     (biomass growth with carrying capacity)
dS/dt = -q × μ(S) × X                  (substrate consumption)

where:
μ(S) = μ_max × S / (Ks + S + S²/Ki)   (Haldane/Andrews kinetics)
```

**Python implementation:**

```python
# From 05_haldane_analysis.py, lines 46-67
def haldane_ode(t, y, mu_max, Ks, Ki, X_max, q):
    """
    Coupled ODE: biomass growth + substrate depletion with Haldane kinetics.
    """
    X, S = y
    S = max(S, 0)    # Substrate can't go negative
    X = max(X, 0)    # Biomass can't go negative

    if S < 1e-10:
        mu_S = 0.0   # No growth without substrate
    else:
        mu_S = mu_max * S / (Ks + S + S**2 / Ki)

    dXdt = mu_S * X * (1 - X / X_max)
    dSdt = -q * mu_S * X

    return [dXdt, dSdt]
```

### 8.3 Parameter definitions

| Parameter | Symbol | Range | Biological meaning |
|-----------|--------|-------|--------------------|
| μ_max | Maximum growth rate | 0.001–5.0 h⁻¹ | Growth rate if substrate were optimal |
| Ks | Half-saturation constant | 0.001–10.0 | Substrate conc. giving half μ_max |
| Ki | Inhibition constant | 0.1–1000.0 | Higher Ki = less inhibition |
| X_max | Carrying capacity | 0.05–5.0 OD | Maximum biomass |
| q | Yield coefficient | 0.001–10.0 | Substrate consumed per unit growth |
| X0 | Initial biomass | 10⁻⁶–0.5 OD | Starting cell density |
| S0 | Initial substrate | 0.01–100.0 | Pesticide concentration |

**Ki is the key parameter**: a low Ki means the bacteria are strongly inhibited by the pesticide, while a high Ki means they tolerate the pesticide well.

### 8.4 Numerical ODE solving

The coupled ODEs are solved numerically using the **Runge-Kutta 45** method (adaptive step size):

```python
# From 05_haldane_analysis.py, lines 70-88
def solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0):
    sol = solve_ivp(
        haldane_ode,
        [time[0], time[-1]],      # Integration interval
        [X0, S0],                  # Initial conditions [biomass, substrate]
        args=(mu_max, Ks, Ki, X_max, q),
        t_eval=time,               # Evaluate at observed time points
        method='RK45',             # Runge-Kutta 4(5) with adaptive step
        max_step=0.5,              # Maximum step size (hours)
        rtol=1e-8, atol=1e-10     # Tight tolerances for accuracy
    )
    if sol.success:
        return sol.y[0], sol.y[1]  # X(t), S(t)
```

### 8.5 Fitting procedure

The Haldane model is fitted using **L-BFGS-B** (bounded Limited-memory BFGS) on
**log(parameter)**, with **S0 held fixed** at the dosing unit passed in (`--s0`),
so six parameters are free. Three details matter:

* **S0 is not fitted.** The curve samples μ(S) along a single trajectory, so a
  change in Ki can be absorbed by rescaling S0 (with Ks and q). With S0 free,
  every strain came back at the Ki initial guess and the "inhibition ranking"
  ranked nothing.
* **The search is in log space.** L-BFGS-B takes finite differences with an
  absolute step of ~1e-8; on a linear Ki of 10 that is a 1e-9 relative step, the
  gradient underflows and Ki never moves. In log space every parameter gets the
  same relative step.
* **Ki is multi-started and profiled.** The fit is repeated from several Ki
  starting values (best SSE wins) and the winner is profiled over 0.1x–10x its
  value. Unless the SSE rises on *both* sides, the data cannot distinguish those
  values: `Ki_identified` is False and `Ki` is reported as NaN (the raw optimum
  is kept as `Ki_fitted`).

```python
# From 05_haldane_analysis.py, fit_haldane()
bounds = [
    (0.001, 5.0),    # mu_max
    (0.001, 10.0),   # Ks
    (0.1, 1000.0),   # Ki
    (0.05, 5.0),     # X_max
    (0.001, 10.0),   # q
    (1e-6, 0.5),     # X0
]                    # S0 is FIXED, not fitted
log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in bounds]

def objective(log_params):
    mu_max, Ks, Ki, X_max, q, X0 = np.exp(log_params)
    X_pred, _S, solved = solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0,
                                       return_status=True)
    sse = np.sum((od - X_pred)**2)
    return sse if solved else sse + SOLVER_FAILURE_PENALTY   # flat fallback is not a fit

best = min((minimize(objective, clip_log(start(ki0)), method='L-BFGS-B',
                     bounds=log_bounds, options={'maxiter': 5000})
            for ki0 in KI_MULTISTART), key=lambda r: r.fun)
```

### 8.6 Gompertz vs. Haldane comparison (AIC)

Both models are compared using the **Akaike Information Criterion** (AIC), which penalizes model complexity:

```
AIC = n × ln(SS_res / n) + 2k

where:
  n = number of data points
  k = number of free parameters (3 for Gompertz, 6 for Haldane — S0 is fixed)
```

Lower AIC is better. The Haldane model has 3 more parameters, so it must explain substantially more variance to justify its added complexity. Both models are scored on **the same window**: the Gompertz model is refitted on the points the Haldane fit sees (the pipeline's own parameters are kept if they describe that window better), because scoring truncated-window parameters against the full curve made "Haldane preferred" a foregone conclusion.

```python
# From 05_haldane_analysis.py
n = len(od)
k = 6  # Haldane has 6 free parameters (S0 is held fixed)
aic = n * np.log(ss_res / n) + 2 * k
bic = n * np.log(ss_res / n) + k * np.log(n)
```

For Gompertz comparison:

```python
# From 05_haldane_analysis.py, lines 180-190
k = 3  # Gompertz has 3 parameters
aic = n * np.log(ss_res / n) + 2 * k
```

**Result:** see `results/tables/Haldane_Analysis/haldane_comparison.csv`. The
16/21 (76%) "Haldane preferred" figure quoted here previously was produced
before the same-window comparison and the S0/Ki fixes above, and does not
describe the current code.

---

## 9. Stage 6 — Advanced Statistical Methods

**Script:** `scripts/06_advanced_fitting.py` (2107 lines)

### 9.1 Gaussian Process (GP) Truncation

**Purpose:** Replace the heuristic peak-finding in Stage 2 with a data-driven, non-parametric method.

A **Gaussian Process** is a flexible, non-parametric regression model that provides both a smooth estimate of the curve and principled uncertainty bands.

**Kernel specification:**

```python
# From 06_advanced_fitting.py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=2.0) + WhiteKernel(noise_level=0.01)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp.fit(time.reshape(-1, 1), od)
```

- **RBF kernel** (Radial Basis Function): Encodes the assumption that nearby time points have similar OD values. The `length_scale` (2.0 hours) controls how smooth the fit is.
- **WhiteKernel**: Models independent measurement noise at each time point (noise level 0.01 OD).

**Truncation via derivative analysis:**

Once the GP is fitted, its derivative dOD/dt is computed analytically. The truncation point is where the derivative crosses zero after its maximum — i.e., the transition from exponential to stationary phase.

### 9.2 Ensemble Truncation

Instead of relying on a single truncation method, the pipeline can combine **five independent methods** and take a weighted consensus:

```yaml
# From config.yaml, lines 133-138
methods:
  first_peak: true           # Heuristic rolling-average peak
  stationary_phase: true     # Growth rate drops to <5% of max
  adaptive_r2: true          # R²-maximizing search
  gp_derivative: true        # Gaussian process derivative zero-crossing
  changepoint: true          # Ruptures PELT changepoint detection
```

The consensus is computed via **weighted median** of the five truncation points:

```python
# From 06_advanced_fitting.py
consensus_method: weighted_median
max_disagreement_hours: 4.0     # Flag if methods disagree by >4 hours
min_methods_required: 3         # Flag if fewer than 3 methods succeed
```

If the methods disagree by more than 4 hours, the strain is flagged for manual inspection.

### 9.3 Bayesian Hierarchical Gompertz Model

**Purpose:** Instead of fitting each strain independently, fit all strains simultaneously with **partial pooling** — strains within the same pesticide group share information, improving estimates for noisy curves.

**Model structure:**

```
Population level (hyperpriors):
    μ_A   ~ Normal(1.3, 0.5)         # Population mean of A
    σ_A   ~ HalfNormal(0.5)          # Between-group variability of A
    μ_μ   ~ Normal(0.25, 0.15)       # Population mean of growth rate
    σ_μ   ~ HalfNormal(0.15)         # Between-group variability of growth rate
    μ_λ   ~ Normal(2.0, 2.0)         # Population mean of lag time
    σ_λ   ~ HalfNormal(2.0)          # Between-group variability of lag time
    σ_obs ~ HalfNormal(0.05)         # Observation noise

Group level (partial pooling):
    A_group[j]   ~ Normal(μ_A, σ_A)           # Group j mean A
    μ_group[j]   ~ Normal(μ_μ, σ_μ)           # Group j mean growth rate
    λ_group[j]   ~ Normal(μ_λ, σ_λ)           # Group j mean lag time

Strain level:
    A_strain[i]  ~ Normal(A_group[j], σ_A_within)
    μ_strain[i]  ~ Normal(μ_group[j], σ_μ_within)
    λ_strain[i]  ~ Normal(λ_group[j], σ_λ_within)

Likelihood:
    OD_observed[i,t] ~ Normal(Gompertz(t; A_strain[i], μ_strain[i], λ_strain[i]), σ_obs)
```

**Non-centered parameterization** (prevents sampling issues in hierarchical models):

```python
# From 06_advanced_fitting.py
# Instead of: A_group ~ Normal(mu_A, sigma_A)
# Use: A_group_offset ~ Normal(0, 1); A_group = mu_A + sigma_A * A_group_offset
```

**Sampling:** NUTS (No U-Turn Sampler) — a state-of-the-art Hamiltonian Monte Carlo variant that automatically adapts step size and trajectory length:

```yaml
# From config.yaml, lines 88-92
bayesian:
  gompertz:
    draws: 2000          # Posterior samples per chain
    tune: 1000           # Warmup/adaptation samples (discarded)
    chains: 4            # Independent Markov chains
    target_accept: 0.90  # Target acceptance probability
```

**Key outputs:**
- **Posterior summaries**: Mean, median, and 95% HDI (Highest Density Interval) for every parameter of every strain
- **Group-level estimates**: Pesticide-group average growth parameters with uncertainty
- **Convergence diagnostics**: R̂ (Gelman-Rubin statistic, should be < 1.05) and ESS (Effective Sample Size)

### 9.4 Bayesian Hierarchical Haldane Model

Extends the Bayesian approach to the Haldane substrate inhibition model. The key innovation is **partial pooling of Ki (inhibition constant) across pesticide groups**:

```
Population:
    μ_log_Ki ~ Normal(2.0, 1.0)           # Population mean of log(Ki)
    σ_log_Ki ~ HalfNormal(1.0)            # Between-pesticide variability

Pesticide group:
    log_Ki_pest[j] ~ Normal(μ_log_Ki, σ_log_Ki)   # Pesticide j mean Ki

Strain:
    log_Ki_strain[i] ~ Normal(log_Ki_pest[j], σ_within)
```

Because the Haldane model involves solving an ODE at each evaluation, gradient-based samplers (like NUTS) are impractical. Instead, the pipeline uses **DEMetropolisZ** — a gradient-free differential evolution sampler:

```yaml
# From config.yaml, lines 93-97
haldane:
  draws: 5000                # More draws needed for gradient-free sampler
  tune: 3000                 # Longer warmup
  chains: 4
  sampler: DEMetropolisZ     # Gradient-free (ODE model)
```

**Key output:** P(Ki < threshold) — the posterior probability that the inhibition constant is below a given threshold, providing a principled measure of inhibition strength for each pesticide.

### 9.5 Bootstrap Uncertainty Quantification

**Purpose:** A computationally cheaper alternative to full Bayesian inference, providing confidence intervals via **residual resampling**:

```python
# From 06_advanced_fitting.py (bootstrap_gompertz function)
def bootstrap_gompertz(time, od, n_resamples=1000, ci_level=0.95):
    # 1. Fit Gompertz to original data
    popt, _ = curve_fit(gompertz_model, time, od, ...)
    predicted = gompertz_model(time, *popt)
    residuals = od - predicted

    # 2. Resample residuals 1000 times
    boot_params = []
    for _ in range(n_resamples):
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        synthetic_od = predicted + resampled_residuals

        # 3. Refit Gompertz to synthetic data
        try:
            popt_boot, _ = curve_fit(gompertz_model, time, synthetic_od, ...)
            boot_params.append(popt_boot)
        except:
            continue

    # 4. Compute 95% confidence intervals from bootstrap distribution
    boot_params = np.array(boot_params)
    ci_low = np.percentile(boot_params, 2.5, axis=0)    # [A_low, mu_low, lam_low]
    ci_high = np.percentile(boot_params, 97.5, axis=0)  # [A_high, mu_high, lam_high]

    # 5. Compute P(good): fraction of bootstrap samples with R² ≥ 0.95
    p_good = np.mean(boot_r2_values >= 0.95)
```

**Advantages over analytical error estimates:** The covariance matrix from `curve_fit` assumes the model is correct and errors are normally distributed. Bootstrap makes no such assumption — it directly estimates the sampling distribution of each parameter.

### 9.6 Bayesian Classification

**Purpose:** Replace hard yes/no classification with **P(good)** — the probability that a strain shows real growth.

```yaml
# From config.yaml, lines 116-119
classification:
  p_good_high: 0.90       # ≥ 0.90 → "GOOD (high confidence)"
  p_good_low: 0.50        # ≥ 0.50 → "GOOD (moderate confidence)"
  borderline_range: [0.30, 0.70]  # Flag for manual review
```

This eliminates the cliff effect where R² = 0.949 is "BAD" but R² = 0.951 is "GOOD". Instead, borderline cases get a probability and are flagged for review.

### 9.7 Model Comparison (WAIC/LOO)

The pipeline compares models using proper Bayesian information criteria:

- **WAIC** (Widely Applicable Information Criterion): A fully Bayesian generalization of AIC that averages over the posterior distribution
- **LOO** (Leave-One-Out Cross-Validation): Estimated via Pareto-Smoothed Importance Sampling (PSIS) without actually refitting the model n times

Both are computed via the **ArviZ** library and provide uncertainty estimates on the comparison itself.

---

## 10. Stage 7 — Truncation Method Comparison & Rescue

**Scripts:** `scripts/07_compare_truncation_methods.py`, `scripts/08_validate_truncation.py`

### 10.1 Motivation

The ensemble truncation system (Section 9.2) produces five independent truncation points plus a weighted-median consensus. But which method actually produces the best downstream Gompertz fit? And can strains initially classified as "BAD" be rescued by using a better truncation?

This stage answers both questions by fitting the Gompertz model at **each method's truncation point** and comparing R², RMSE, and parameter errors across all methods for every strain.

### 10.2 Method comparison study

For each of the 57 good strains, the comparison script:

1. Reads the ensemble truncation results (truncation times per method)
2. Truncates the raw OD data at each method's point
3. Fits the Gompertz model at each truncation
4. Records R², RMSE, parameter values, and parameter errors

This produces a `method_comparison.csv` with one row per strain × method (57 strains × 6 methods = 342 rows).

**Aggregate method rankings (57 good strains):**

| Rank | Method | Mean R² | Median R² | % Good (R²≥0.95) | % Excellent (R²≥0.99) |
|:---:|--------|:-------:|:---------:|:-----------------:|:---------------------:|
| 1 | Consensus | 0.9804 | 0.9962 | 96.5% | 86.0% |
| 2 | Adaptive R² | 0.9800 | 0.9924 | 98.2% | 82.5% |
| 3 | Stationary Phase | 0.9799 | 0.9924 | 98.2% | 82.5% |
| 4 | First Peak | 0.9790 | 0.9973 | 94.7% | 89.5% |
| 5 | GP Derivative | 0.9787 | 0.9968 | 94.7% | 84.2% |
| 6 | Changepoint | 0.7099 | 0.9892 | 87.5% | 73.2% |

**Key finding:** The weighted-median consensus achieves the highest mean R² overall. All top-5 methods perform similarly (mean R² 0.978–0.980), but the Changepoint method has substantially higher variance and one failure, dragging its mean down.

### 10.3 Per-strain overlay plots

For each strain, the script generates an overlay plot showing:

- Raw OD data as gray scatter points
- 6 Gompertz fit curves, color-coded by method (solid through fitted region, dotted for extrapolation beyond truncation)
- Vertical dashed lines at each method's truncation point
- Legend with method name and R² value

These plots are saved to `truncation_comparison/overlay_plots/` and allow visual inspection of how each method's truncation affects the fitted curve.

### 10.4 Bad strain rescue analysis

When invoked with `--include-bad`, the script attempts to rescue strains that were classified as BAD in the original pipeline:

1. Load all 92 strains from `all_groups_results.csv`
2. Filter bad strains with delta_OD > 0.015 (skip genuinely flat H₂O controls)
3. For each candidate: run `ensemble_truncate()` → fit Gompertz at each method's truncation
4. A strain is "rescued" if any method produces R² ≥ 0.95 with parameter errors < 20%

**Rescue results:**

| Strain | Group | Best Rescue Method | Rescue R² | Original Reason |
|--------|-------|--------------------|:---------:|-----------------|
| Bifenthrin-BIF6 | Group1 | Stationary Phase | 0.987 | Low R²: 0.908 |
| Flupyradifurone-FLUPC8 | Group1 | Changepoint | 0.986 | Low R²: 0.467 |
| H2O-IMID5 | Group3 | Adaptive R² | 0.957 | Low R²: 0.945 |
| LB-CISPERM1 | Group4 | Stationary Phase | 0.978 | Low R²: 0.788; High errors |
| MALATHION-MAL1 | Group2 | First Peak | 0.999 | Low R²: 0.919 |
| MALATHION-MAL8 | Group2 | First Peak | 0.999 | Low R²: -0.323; High A error |

A strain now counts as rescued only when the refit passes every quality gate of the main pipeline (R², parameter errors, growth, noise and residual gates), not merely when its R² improves. The table above predates that rule; the current rescues are listed in `results/tables/Advanced_Analysis/truncation_comparison/rescued_strains.csv`.

### 10.5 Interactive truncation validator

`08_validate_truncation.py` provides a matplotlib-based interactive tool for human review of the automated method selections. For each strain it:

1. Loads raw data and all 6 method fits from `method_comparison.csv`
2. Renders a live overlay plot (same as Section 10.3 but interactive)
3. Accepts user input via keyboard or buttons:

| Key | Action |
|-----|--------|
| 1–5 | Select a specific method as best |
| c | Consensus is correct |
| n | None of the methods look good |
| b | Go back to previous strain |
| q | Quit and save |

4. Records annotations to `truncation_validation_audit.csv` (resumable — already-reviewed strains are skipped on relaunch)

**Output CSV schema:**

```csv
strain, group, user_best_method, auto_best_method, auto_best_r2, consensus_r2, user_notes, timestamp
```

This enables systematic comparison of human expert judgment against the automated method selection, providing ground-truth validation of the ensemble approach.

---

## 11. Stage 8 — Synthetic Data Validation

**Directory:** `synthetic_data/`

### 11.1 Purpose

To validate the pipeline's accuracy, we generate 1139 synthetic growth curves
across 64 scenarios with **known ground-truth parameters**, run the pipeline on
them, and measure how well it recovers the true classifications and parameters.

The ground truth describes the curve that was actually written, not the
sampler's request: `true_A`, `true_mu` and `true_lambda` are measured on the
emitted trace, while the sampled targets are kept alongside as `requested_*`.
`label_rule` records where each label came from (`scenario`, asserted by the
catalog, or `oracle`, decided per curve by whether A and mu are recoverable),
and `scoring_mode` carries the difficulty policy so `expected_class` stays a
plain binary GOOD/BAD.

### 11.2 Growth models available for synthetic data

Five mathematical models are implemented in `synthetic_data/src/growth_models.py`:

```python
# 1. Modified Gompertz (matches pipeline)
class GompertzModel:
    @staticmethod
    def compute(t, A, mu, lambda_):
        return A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))

# 2. Baranyi-Roberts (better stationary phase)
class BaranyiModel:
    @staticmethod
    def compute(t, y0, y_max, mu_max, lambda_):
        A_t = t + (1/mu_max) * np.log(np.exp(-mu_max*t) +
              np.exp(-mu_max*lambda_) - np.exp(-mu_max*(t+lambda_)))
        return y_max + np.log((-1 + np.exp(mu_max*A_t)) /
              (1 + (np.exp(mu_max*A_t) - 1) / np.exp(y_max - y0)))

# 3. Logistic (symmetric S-curve)
class LogisticModel:
    @staticmethod
    def compute(t, A, k, t_mid):
        return A / (1 + np.exp(-k * (t - t_mid)))

# 4. Richards (flexible asymmetry parameter ν)
class RichardsModel:
    @staticmethod
    def compute(t, A, k, t_mid, nu):
        return A / (1 + nu * np.exp(-k * (t - t_mid)))**(1/nu)

# 5. Haldane (coupled ODE with substrate inhibition)
class HaldaneModel:
    @staticmethod
    def compute(t, mu_max, Ks, Ki, X_max, q, X0, S0):
        # Uses scipy.integrate.solve_ivp internally
        ...
```

### 11.3 Noise models

`synthetic_data/src/noise_models.py` builds every trace's error term from
components that are each anchored to a measured quantity on the four plates in
this study, rather than to a chosen multiplier:

* **`ColoredNoise`** — white noise plus a fast (`tau_fast_h=0.5`) and a slow
  (`tau_slow_h=7.5`) correlated component. Real TECAN residuals are correlated
  in time; white residuals made the pipeline's lag-1 autocorrelation gate
  almost impossible to fire, which flattered the benchmark.
* **`RMSEBasedNoise`** — solves for the component that delivers a requested RMSE
  exactly, so a scenario can ask for a target R² and get it.
* **`BlankModel`** — samples a blank trace per media, so blank subtraction is
  exercised rather than assumed.
* **`inject_spikes` / `apply_signed_drift`** — bubble artefacts and evaporation
  or condensation drift, at the magnitudes and rates measured on real wells.
* **`InstrumentNoise` / `GaussianNoise` / `ODDependentNoise`** — kept for the
  simpler scenarios and for backwards compatibility.

Nothing in the module calls `np.random.seed`; every generator takes an explicit
`rng`, which is what makes the suite reproducible scenario by scenario.

Noise presets (`NOISE_PRESETS`, names unchanged). Each is a real well, named:

| Preset | Anchored to | Baseline σ | RMSE range |
|--------|-------------|-----------|------------|
| very_low | cleanest real well (Group 3 A1, σ̂ 0.00036) | 0.0003 | 0.0006–0.0012 |
| low | p10 real well (σ̂ 0.00073, Group 3 B2) | 0.0005 | 0.0010–0.0020 |
| medium | median of 96 Group 3 wells (σ̂ 0.00127) | 0.0009 | 0.0018–0.0036 |
| high | p90 real well (σ̂ 0.00200, Group 3 C1) | 0.0020 | 0.0040–0.0080 |
| very_high | noisiest well observed in 96 real wells (Group 3 H12) | 0.0020 | see the module |

The `very_high` preset is the noisiest *well* observed, not a multiple of the
worst curve: the worst real condition-average whole-curve σ̂ is 0.00443.

### 11.4 The 64 test scenarios

The scenarios are defined in `synthetic_data/src/curve_scenarios.py`, which is
the only definition of what each one generates. They fall into three catalog
buckets:

| Bucket | Scenarios | What it holds |
|--------|-----------|---------------|
| `GOOD_GROWTH_SCENARIOS` | 10 | plain growth the pipeline should accept |
| `BAD_CURVE_SCENARIOS` | 15 | curves it should reject — dead wells, contamination, replicate artefacts, and curves whose fit is not recoverable |
| `EDGE_CASE_SCENARIOS` | 39 | awkward but mostly legitimate curves: multi-phase shapes, decline and recovery, run-length geometries, model mismatch, noise ladders |

The full per-scenario table, with parameter ranges and per-scenario counts, is
in `synthetic_data/README.md` and is generated from the catalog itself
(`python scripts/generate_comprehensive.py --print-readme-tables`); it is not
duplicated here, so it cannot go stale in two places at once.

A `scoring_mode` other than `standard` marks a block the present gate set
cannot decide fairly, and those curves are scored separately from the headline
rather than folded into it:

| `scoring_mode` | n | Why it is held out |
|---|---:|---|
| `standard` | 867 | in the headline |
| `model_mismatch` | 93 | in the headline; flagged because the generating model is not Gompertz |
| `preprocessing_blind` | 99 | the preprocessing step hands the pipeline the replicate mean only, so a per-well artefact is invisible to it |
| `a_not_identifiable` | 80 | the run stops before the plateau, so the asymptote is not identifiable from the data |

### 11.5 Validation metrics

`make step-9` runs the ordinary pipeline over the generated plates and scores
the resulting `processing_results.csv` against `ground_truth.csv` with
`synthetic_data/validation/benchmark.py`, which writes `report.json` and the
per-curve, per-scenario, per-scenario-recovery and gate-attribution tables.

The headline below is first-peak truncation (the interactive default), seed 42,
scored over the 960 curves the present gate set can decide fairly. Because the
suite is deliberately unbalanced, balanced accuracy leads and raw accuracy is
reported with its prevalence:

| Metric | Full suite (n=960) | The original 32 scenarios (n=532) |
|--------|-------|-------|
| **Balanced accuracy** | 87.9% | 96.8% |
| **Sensitivity** | 85.6% | 93.5% |
| **Specificity** | 90.2% | 100.0% |
| Accuracy (prevalence 56.4% GOOD) | 87.6% | 96.1% |
| Precision | 91.9% | 100.0% |
| F1 | 88.6% | 96.7% |
| MCC | 0.753 | 0.922 |
| TP / TN / FP / FN | 463 / 378 / 41 / 78 | 304 / 207 / 0 / 21 |

Over all 1139 curves under one flat rule, ignoring the scoring policy: 81.9%
accuracy, 81.3% balanced. The two held-out blocks score 85.0%
(`a_not_identifiable`, 80 curves) and 24.2% (`preprocessing_blind`, 99 curves).
Both are flat-rule figures -- correct means the accept/reject verdict matched
the truth. `report.json` also scores `a_not_identifiable` under a stricter
block rule (correct if the curve was rejected, or the fitted A is within 15 %
of the observed final OD and no more than 1.20 x the last observed point),
which it passes 12 times out of 80 -- 15.0 %: the verdict is right on 68 of
the 80 while the reported amplitude is supportable on 12, and that gap is why
the block is held out. Both numbers appear together in
`docs/BENCHMARK_LEDGER.md` and in the rendered summary.
The right-hand column is the subset comparable with the number this document
used to quote for a 480-curve suite; `docs/BENCHMARK_LEDGER.md` decomposes the
difference between the two, and no number in this section is typed by hand —
they are all in `synthetic_data/benchmark/v2/report.json`, which
`synthetic_data/scripts/validation_summary.py` renders into a markdown summary
on demand (`make validation-summary`; the rendering is a build artefact and is
not committed).

The misses concentrate in: hourly-sampled and fast-growing curves that fall
under the 20-point window minimum; curves whose residuals are correlated enough
to trip the 0.7 autocorrelation gate (the residuals became coloured when the
noise model was recalibrated, and the gate now fires on 17% of fitted curves
against 5.8% before, with the threshold untouched); diauxic curves whose second
phase the Gompertz shape cannot absorb; and contamination at the measured
acceptance boundary, which the pipeline accepts outright.

### 11.6 Parameter recovery accuracy

Pooled over the curves the pipeline accepted and the truth calls GOOD, against
the **measured** parameters of the emitted curve:

| Set | n | A R² | A RMSE | μ R² | μ RMSE | λ R² | λ RMSE |
|-----|--:|-----:|-------:|-----:|-------:|-----:|-------:|
| The original 32 scenarios | 304 | 0.919 | 0.101 | 0.987 | 0.0093 | 0.999 | 0.306 |
| Full headline | 463 | 0.740 | 0.181 | 0.988 | 0.0091 | 0.995 | 0.878 |
| All accepted GOOD, including the held-out blocks | 555 | 0.405 | 0.262 | 0.991 | 0.0083 | 0.975 | 1.772 |

A recovery falls as the held-out blocks enter because those families
deliberately contain amplitudes a Gompertz asymptote is not meant to recover;
μ and λ recovery improved roughly threefold over the previous suite.

Two cautions belong with these numbers and are printed in the generated summary
as well:

* **They are conditioned on the pipeline's own verdict.** Over every fitted
  GOOD curve, without the pipeline's filter, A RMSE is 0.786 against 0.262 on
  the accepted subset — a factor of 3.0. The gap is selection, not accuracy:
  tightening any gate would raise the accepted-only figure without changing the
  estimator.
* **The pipeline's parameter standard errors are not confidence intervals.** A
  nominal 95% interval covers the truth for 11.9% of accepted curves for A,
  25.4% for μ and 10.3% for λ; the median |z| is 8.2, 3.7 and 9.5. The
  least-squares covariance assumes independent residuals and the residuals are
  correlated, so the intervals are too narrow by roughly an order of magnitude
  for A and λ. Treat them as a relative precision indicator.

---

## 12. Stage 9 — ML Classifier

### Architecture

Two-stage classification system augmenting the rule-based pipeline:

```
Raw OD data
  → Stage 1: PRE-FIT GATE (10 features: 8 raw signal + 2 metadata)
      → P(good) <= 0.20 → REJECT (skip expensive truncation/fitting)
      → P(good) > 0.20  → PASS to pipeline
  → Truncation + Gompertz fit (expensive)
  → Rule-based classification (R², parameter errors, quality gates)
  → Stage 2: POST-FIT CLASSIFIER (24 features: 22 fit quality + 2 metadata)
      → P(good) >= 0.7 → GOOD
      → P(good) <= 0.3 → BAD
      → otherwise      → BORDERLINE (flagged for manual review)
```

### Feature Sets

**Pre-fit gate (10 features):** `raw_delta_od`, `raw_max_od`, `raw_snr`, `raw_monotone_fraction`, `raw_baseline_std`, `raw_baseline_mean`, `n_points`, `time_span`, `is_control`, `concentration_numeric`

**Post-fit classifier (24 features):** 13 direct fit features + 4 secondary quality metrics + 5 derived ratios + 2 metadata features. Key additions over rule-based: `is_control` (parsed from strain name — H2O/LB prefixes detected as controls), `concentration_numeric` (trailing number from strain name).

### Training

- **Model:** `HistGradientBoostingClassifier` (max_depth=5, min_samples_leaf=5, class_weight='balanced')
- **Data:** the synthetic suite only. 1059 curves enter training (565 GOOD / 494 BAD), 936 of them reaching the post-fit stage (533 GOOD / 403 BAD). 80 curves are excluded as not trainable — their asymptote is not identifiable from the observed window, so a label on `gompertz_a` would be noise — and 15 train-split rows are dropped by a label-agreement guard, from `short_experiment` and `short_lag`, whose labels are not predictable from their own post-fit features. The 85 audited real curves (7 "unsure" excluded) are an **external test set** and are never trained on.
- **Split:** stratified holdout plus 5-fold cross-validation
- **Pre-fit features:** Extracted from actual raw OD data files (not CSV proxies) to avoid train/inference mismatch
- **Truncation:** features are built in a named truncation mode (`first_peak` for the committed model) and the mode is recorded in `feature_config.json`
- **Deployment:** After holdout evaluation, the final model is retrained on the synthetic training rows

### Results (synthetic training set, real audited curves held out)

5-fold cross-validation on the synthetic suite (features built with first-peak truncation): post-fit classifier accuracy 91.9 %, precision 91.3 %, recall 94.9 %, specificity 87.9 %; pre-fit gate accuracy 88.3 %. On the 85 audited real curves, which are never trained on (66 of them reach the post-fit stage): accuracy 81.8 %, precision 100.0 %, recall 78.2 %, specificity 100.0 %.

Retraining on the rebuilt suite is a **regression in accuracy on real data and is published as one**: scored by the same code on the same 66 rows, the previous model gave accuracy 90.9 %, precision 98.0 %, recall 90.9 %, specificity 90.9 % (confusion 50/5/1/10) against 81.8 % / 100.0 % / 78.2 % / 100.0 % (43/12/0/11) now. The new model is markedly more conservative: it makes no false-GOOD calls at all on real plates, at the cost of calling 12 real GOOD curves BAD. The training suite went from 71.9 % GOOD to 56.6 % GOOD and gained several families of hard GOOD curves, and the model followed that prior. Whether the trade is wanted is a policy question; the deployed thresholds (0.70 / 0.30) were **not** changed, and the threshold sweep in `feature_config.json` is reported as advisory only.

The exact numbers of the committed model are in `models/feature_config.json`, and section 9 of `docs/BENCHMARK_LEDGER.md` gives the before/after with the command that reproduces it. Two earlier figures are **not** comparable and should not be quoted: 95.9 % / 94.3 % came from a model trained on part of the real audit set, and 94.0 % over 67 fitted curves came from the pre-rebuild training code with a different `fit_attempted` mask.

**Probability distribution:** Strongly bimodal — all predictions are p<0.05 or p>0.95. No borderline cases in current dataset.

### Files

| File | Purpose |
|------|---------|
| `scripts/ml_classifier.py` | Runtime module (feature extraction, PreFitGate, PostFitClassifier) |
| `scripts/09_train_classifier.py` | Training script (70/30 split on synthetic curves, real audit held out, metadata) |
| `models/prefit_gate.joblib` | Trained pre-fit gate model (gitignored) |
| `models/postfit_classifier.joblib` | Trained post-fit classifier (gitignored) |
| `models/feature_config.json` | Feature names, training metrics, timestamp (committed) |
| `tests/test_ml_classifier.py` | 22 unit tests |

---

## 13. Configuration Reference

All tunable parameters are centralized in `scripts/config.yaml`:

```yaml
# =============================================================================
# PRIMARY CLASSIFICATION THRESHOLDS
# =============================================================================
classification:
  min_r_squared: 0.95          # Minimum R² for GOOD classification
  max_param_error_pct: 20.0    # Maximum relative parameter error (%)
  min_points_for_fit: 15       # Minimum data points to attempt fitting

# =============================================================================
# SECONDARY QUALITY GATES
# =============================================================================
quality_gates:
  min_snr: 5.0                 # Minimum signal-to-noise ratio
  min_delta_od_ci: 0.1         # Minimum delta-OD confidence interval lower bound
  min_absolute_delta_od: 0.15  # Absolute minimum delta-OD for growth detection
  pre_fit_snr_threshold: 3.0   # Pre-fit SNR threshold (skip fitting below this)
  pre_fit_delta_od_override: 0.5  # If delta-OD > this, bypass low pre-fit SNR
  excellent_r2_threshold: 0.98 # R² above which noise-based gates are bypassed

# =============================================================================
# TRUNCATION PARAMETERS
# =============================================================================
truncation:
  smoothing_window: 5          # Rolling average window for peak detection
  buffer_points: 3             # Points to keep after max for fit stability
  min_points_for_fit: 20       # Minimum points after truncation
  use_first_peak: true         # Truncate at first local max (vs global max)
  use_adaptive: false          # Enable adaptive R²-optimizing truncation

# =============================================================================
# HALDANE MODEL SETTINGS
# =============================================================================
haldane:
  default_S0: 1.0              # Default substrate concentration (arbitrary units)
  parameter_bounds:
    mu_max: [0.001, 5.0]
    Ks: [0.001, 10.0]
    Ki: [0.1, 1000.0]
    X_max: [0.05, 5.0]
    q: [0.001, 10.0]
    X0: [0.000001, 0.5]
    S0: [0.01, 100.0]

# =============================================================================
# ADVANCED STATISTICAL METHODS
# =============================================================================
advanced:
  # Gaussian Process truncation
  gp:
    kernel_length_scale: 2.0       # RBF initial length scale (hours)
    kernel_noise: 0.01             # WhiteKernel noise level
    derivative_threshold: 0.01     # dOD/dt threshold for phase boundary
    n_restarts: 5                  # Kernel optimization restarts

  # Bayesian MCMC sampling
  bayesian:
    gompertz:
      draws: 2000                  # Posterior samples per chain
      tune: 1000                   # Warmup samples (discarded)
      chains: 4                    # Independent Markov chains
      target_accept: 0.90          # NUTS target acceptance rate
    haldane:
      draws: 5000                  # More draws for gradient-free sampler
      tune: 3000
      chains: 4
      sampler: DEMetropolisZ       # Gradient-free (ODE model)

  # Bayesian priors (informed by observed data distributions)
  priors:
    mu_A: [1.3, 0.5]              # Normal(1.3, 0.5) for max OD
    mu_mu: [0.25, 0.15]           # Normal(0.25, 0.15) for growth rate
    mu_lam: [2.0, 2.0]            # Normal(2.0, 2.0) for lag time
    sigma_obs: 0.05               # HalfNormal(0.05) for observation noise
    mu_log_Ki: [2.0, 1.0]         # Normal(2.0, 1.0) for log(Ki) population mean
    sigma_log_Ki: 1.0             # HalfNormal(1.0) for Ki between-group variance

  # Bootstrap
  bootstrap:
    n_resamples: 1000
    ci_level: 0.95

  # Bayesian classification thresholds
  classification:
    p_good_high: 0.90             # High confidence GOOD
    p_good_low: 0.50              # Moderate confidence GOOD
    borderline_range: [0.30, 0.70]

  # Convergence diagnostics
  convergence:
    max_rhat: 1.05                # Gelman-Rubin convergence criterion
    min_ess_bulk: 400             # Minimum effective sample size (bulk)
    min_ess_tail: 200             # Minimum effective sample size (tails)

  # Ensemble truncation
  ensemble:
    enabled: true
    consensus_method: weighted_median
    max_disagreement_hours: 4.0
    min_methods_required: 3
    methods:
      first_peak: true
      stationary_phase: true
      adaptive_r2: true
      gp_derivative: true
      changepoint: true
```

---

## 14. Execution & Reproducibility

### 13.1 Dependencies

```
# Core pipeline
pandas>=1.3.0           # Data frames
numpy>=1.20.0           # Numerical arrays
matplotlib>=3.4.0       # Plotting
scipy>=1.7.0            # Optimization, ODE solving
pyyaml>=5.4             # Configuration files

# Advanced methods
pymc>=5.10.0            # Bayesian modeling (NUTS, DEMetropolisZ)
arviz>=0.17.0           # Posterior analysis, diagnostics
scikit-learn>=1.0.0     # Gaussian Processes
ruptures>=1.1.5         # Changepoint detection (optional)

# Testing
pytest>=6.0.0
pytest-cov>=2.12.0
```

### 13.2 Running the pipeline

**Quick start (via `run_full_pipeline.py`):**

```bash
python scripts/run_full_pipeline.py              # Full pipeline (all 10 steps)
python scripts/run_full_pipeline.py --dry-run    # Preview steps without executing
python scripts/run_full_pipeline.py --no-ml      # Rule-based only (skip ML classifier)
python scripts/run_full_pipeline.py --steps 2,3  # Run specific steps only
```

**Quick start (via Makefile):**

```bash
make install              # Install all Python dependencies
make run-pipeline         # Run Steps 0-3: preprocess → truncate → fit → classify → combine
make run-haldane          # Run Step 4: Haldane substrate inhibition analysis
make run-advanced         # Run Step 5: GP + Bootstrap + Bayesian + Ensemble
make run-all              # Run everything (Steps 0-9)
make test                 # Run pytest test suite (830 tests)
make validate             # Interactive curve auditor
```

**Truncation comparison & validation (Step 7):**

```bash
make run-compare-methods  # Compare truncation methods (requires ensemble run first)
make run-compare-bad      # Compare methods + rescue bad strains (--include-bad)
make validate-truncation  # Launch interactive truncation method validator
```

**Advanced fitting sub-targets (Step 5):**

```bash
make run-advanced-gp          # GP truncation only
make run-advanced-bootstrap   # Bootstrap CIs only
make run-advanced-gompertz    # Bayesian Gompertz only
make run-advanced-haldane     # Bayesian Haldane only
make run-advanced-ensemble    # Ensemble truncation only
```

**Individual scripts:**

```bash
# Step 0: Preprocess raw plate data (Group 1 example)
python scripts/02_preprocess_raw_plate_data.py \
    data/raw/Group1/GrowthRate_Group1_Values.csv \
    data/raw/Group1/GROUP1_KEY_V2.csv \
    -o data/raw/Group1/Group_1_DATA

# Step 2: Core analysis (single group)
python scripts/01_growth_curve_analysis.py \
    data/raw/Group1/Group_1_DATA \
    -o results/tables/Group1_Results \
    --adaptive   # Enable adaptive truncation (optional)

# Step 4: Haldane analysis
python scripts/05_haldane_analysis.py \
    --results-dir results/tables \
    --output results/tables/Haldane_Analysis \
    --s0 2.5  # Initial substrate concentration

# Step 5: Advanced methods
python scripts/06_advanced_fitting.py \
    --gompertz-only    # Skip Haldane Bayesian (faster)
    --chains 2         # Reduce chains for testing
    --draws 500        # Reduce draws for testing

# Batch processing (all 4 groups)
bash scripts/run_all_groups.sh
```

**Step 7: Truncation method comparison & rescue:**

```bash
# Compare all 5 methods + consensus on 57 good strains
python scripts/07_compare_truncation_methods.py

# Also attempt to rescue bad strains
python scripts/07_compare_truncation_methods.py --include-bad

# Interactive truncation method validator
python scripts/08_validate_truncation.py
python scripts/08_validate_truncation.py --comparison-csv path/to/method_comparison.csv
```

**Step 9: Synthetic validation:**

Step 9 does not re-run the analysis on its own: it runs the ordinary pipeline
over the generated plates and scores the resulting `processing_results.csv`
against `ground_truth.csv` with `synthetic_data/validation/benchmark.py`. (The
older `synthetic_data/scripts/validate_pipeline.py`, which re-ran the analysis
from a path that did not exist and wrote a second, incompatible metrics schema,
has been removed.)

```bash
# Generate the comprehensive test suite (writes test_data/DATA + ground_truth.csv)
python synthetic_data/scripts/generate_comprehensive.py \
    --output synthetic_data/output/comprehensive_test

# Run the pipeline over it and score it, first-peak truncation (the headline)
make step-9              # = python scripts/run_full_pipeline.py --steps 9
make step-9-all-modes    # first-peak and adaptive, one run directory each

# Or score a finished run directly
python synthetic_data/validation/benchmark.py \
    --results-csv  synthetic_data/output/comprehensive_test/validation/first_peak/processing_results.csv \
    --ground-truth synthetic_data/output/comprehensive_test/test_data/ground_truth.csv \
    --out-dir      synthetic_data/output/comprehensive_test/validation/first_peak \
    --truncation-mode first_peak

# Render the human-readable summary from a scored run (report.json only)
make validation-summary
python synthetic_data/scripts/validation_summary.py \
    --run-dir synthetic_data/output/comprehensive_test/validation/first_peak
```

Each mode writes its own directory (`validation/first_peak/`,
`validation/adaptive/`), so an adaptive run can no longer overwrite the
first-peak headline in place, and the mode is recorded in the run's
`run_settings.json` sidecar and in the emitted `report.json`.

### 13.3 CI/CD

The `.github/workflows/test.yml` file configures automatic testing on GitHub:
- Tests against Python 3.10, 3.11, and 3.12
- Runs the full pytest suite on every push/PR
- Validates core functionality (Gompertz fitting, classification, Haldane ODE solving)

---

## 15. Results Summary

### 15.1 Real experimental data (92 strains)

| Group | Pesticides | Total Strains | Good | Bad |
|-------|-----------|:---:|:---:|:---:|
| 1 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| 2 | Malathion, Lambda-Cyhalothrin | 24 | 11 | 13 |
| 3 | Imidacloprid, Lambda-Cyhalothrin | 24 | 12 | 12 |
| 4 | Permethrin, Lambda-Cyhalothrin | 20 | 10 | 10 |
| **Total** | | **92** | **45 (49%)** | **47 (51%)** |

These are the committed tables, produced with adaptive (cross-validated) truncation and rule-based classification. The visual audit (`validation_audit.csv`) was recorded against an earlier version of the pipeline; of its 85 confident calls the current adaptive run reproduces 73 and the interactive first-peak rule 74. Almost every difference is a well that rose by 0.05 to 0.13 OD, which that earlier version accepted and the present 0.15 OD minimum-growth gate rejects.

### 15.2 Growth parameter distributions (GOOD curves; see growth_rate_summary_for_collaboration.csv for the current values)

| Parameter | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| A (Max OD) | 0.025 | 1.87 | 0.72 | 0.45 |
| μ (Growth rate, h⁻¹) | 0.001 | 0.75 | 0.18 | 0.15 |
| λ (Lag time, h) | 0.035 | 76.7 | 8.2 | 12.5 |
| R² | 0.95 | 0.9998 | 0.985 | 0.015 |

### 15.3 Haldane analysis (21 pesticide+LB strains)

- **Haldane preferred over Gompertz by AICc:** 7/22, all in strains whose comparison window is the whole 95-100 h run, where a single Gompertz cannot describe the decline phase (Malathion + LB: Gompertz R² < 0 on the full window against 0.998 on its own growth window)
- **Ki identified:** 0/22. With S0 fixed and the profile-likelihood test, no strain constrains Ki; the previously reported Ki values (all ~10) were the optimiser's starting value, which it could not move from.
- The data therefore do not support a substrate-inhibition claim; the model comparison is kept as a documented negative result.

### 15.4 Truncation method comparison

- Methods are now ranked by median R² with failure counts alongside; see `method_summary.csv`
- A BAD strain counts as rescued only when its refit passes every quality gate; see `rescued_strains.csv`

### 15.5 Bad strain rescue

See `results/tables/Advanced_Analysis/truncation_comparison/rescued_strains.csv` for the current list; the earlier figure of 17 of 22 counted any R² improvement as a rescue, including water controls.

### 15.6 Pipeline validation

| Validation type | Metric | Value |
|----------------|--------|-------|
| **Synthetic validation** (1139 curves, 64 scenarios, first-peak rule; 960 scored in the headline) | Balanced accuracy | 87.9% |
| | Sensitivity | 85.6% |
| | Specificity | 90.2% |
| | Accuracy | 87.6% |
| | Accuracy, the original 32 scenarios (n=532) | 96.1% |
| | Accuracy, all 1139 under one flat rule | 81.9% |
| **ML Classifier** (5-fold CV on the synthetic suite, post-fit) | Accuracy | 91.9% |
| | Precision | 91.3% |
| | Recall | 94.9% |
| | Specificity | 87.9% |
| **ML Classifier** (85 audited real curves, held out; 66 fitted) | Accuracy | 81.8% |
| | Precision | 100.0% |
| | Recall | 78.2% |
| | Specificity | 100.0% |
| **Visual audit** (85 confident calls, earlier pipeline version) | Reproduced by first-peak rule | 74 |
| | Reproduced by adaptive rule | 73 |

The ML classifier uses a two-stage HistGradientBoosting architecture trained on the synthetic suite only; the 85 audited real curves are an external test set. Its real-data accuracy fell from 90.9 % to 81.8 % when it was retrained on the rebuilt suite, while its specificity rose to 100 % — see section 12. It is off by default (`ml_classifier.enabled: false`) and, in the lab interface, only ever a second opinion next to the rule-based call.

### 15.7 Test suite

830 tests across 31 test files covering Gompertz fitting, Haldane ODE, classification logic, ensemble truncation, GP truncation, bootstrap, MCCV truncation, incomplete curve detection, ML classifier feature extraction, metadata parsing, the lab interface, the documentation, and — under `tests/synthetic/` — the synthetic generator, its truth schema, its noise and shape libraries, the benchmark validator and the pipeline's run provenance. `python -m pytest tests --collect-only -q` prints the current count.

---

*Generated for BIO380SP25 — Pesticide Bioremediating Bacteria Research Project*
*Pipeline version: February 2026*