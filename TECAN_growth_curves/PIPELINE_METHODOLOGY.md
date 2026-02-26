# TECAN Growth Curve Analysis Pipeline: Complete Methodology & Documentation

## BIO380SP25 — Pesticide Bioremediating Bacteria Research Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure & File Map](#2-directory-structure--file-map)
3. [Pipeline Architecture (Data Flow)](#3-pipeline-architecture-data-flow)
4. [Stage 1 — Raw Data Preprocessing](#4-stage-1--raw-data-preprocessing)
5. [Stage 2 — Growth Curve Truncation](#5-stage-2--growth-curve-truncation)
6. [Stage 3 — Gompertz Model Fitting](#6-stage-3--gompertz-model-fitting)
7. [Stage 4 — Classification (Good vs Bad)](#7-stage-4--classification-good-vs-bad)
8. [Stage 5 — Haldane Substrate Inhibition Analysis](#8-stage-5--haldane-substrate-inhibition-analysis)
9. [Stage 6 — Advanced Statistical Methods](#9-stage-6--advanced-statistical-methods)
10. [Stage 7 — Truncation Method Comparison & Rescue](#10-stage-7--truncation-method-comparison--rescue)
11. [Stage 8 — Synthetic Data Validation](#11-stage-8--synthetic-data-validation)
12. [Configuration Reference](#12-configuration-reference)
13. [Execution & Reproducibility](#13-execution--reproducibility)
14. [Results Summary](#14-results-summary)

---

## 1. Project Overview

This pipeline analyzes bacterial growth curves from **TECAN plate reader** experiments measuring the optical density (OD600) of bacteria over time. The goal is to determine which bacterial strains can grow in the presence of various pesticides — testing their potential as **pesticide bioremediators**.

**Core research question:** Can these bacteria metabolize common pesticides (bifenthrin, flupyradifurone, imidacloprid, lambda-cyhalothrin, malathion, permethrin) as a carbon/energy source?

**What the pipeline does, step by step:**

1. Converts raw 96-well plate reader output into clean, blank-subtracted time series
2. Truncates curves to isolate the growth phase (removing death/decline)
3. Fits the modified Gompertz growth model to quantify growth parameters
4. Classifies each curve as "GOOD" (real growth) or "BAD" (no meaningful growth)
5. Fits a mechanistic Haldane substrate-inhibition model to pesticide-treated strains
6. Applies advanced Bayesian hierarchical models, Gaussian processes, and bootstrap methods
7. Compares 5 truncation methods via Gompertz fit quality, rescues misclassified bad strains
8. Validates the entire pipeline against 480 synthetic ground-truth curves

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
│   └── output/comprehensive_test/        # 480 synthetic curves + validation reports
│
├── tests/                                # ──── UNIT & INTEGRATION TESTS ────
│   ├── test_gompertz.py
│   ├── test_haldane.py
│   ├── test_classification.py
│   ├── test_integration.py
│   └── test_advanced.py                  # Ensemble truncation, GP, Bootstrap tests
│
├── Makefile                              # Build automation (17 targets)
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

---

## 3. Pipeline Architecture (Data Flow)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     RAW DATA ACQUISITION                           │
  │  TECAN plate reader → 96-well OD600 readings every 15 min         │
  │  + Well mapping key (which well = which strain + media)            │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 1: PREPROCESSING  (02_preprocess_raw_plate_data.py)         │
  │  • Time: seconds → hours                                          │
  │  • Triplicate averaging                                            │
  │  • Blank subtraction (per media)                                   │
  │  • Output: {MEDIA}_DATA.csv files                                  │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 2: TRUNCATION  (01_growth_curve_analysis.py)                │
  │  • Rolling average smoothing (window=5)                            │
  │  • First local maximum detection                                   │
  │  • Optional: Adaptive R²-maximizing truncation                     │
  │  • Remove death/decline phase                                      │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 3: GOMPERTZ FITTING  (01_growth_curve_analysis.py)          │
  │  • Modified Gompertz model: y(t) = A·exp(-exp((μe/A)(λ-t)+1))     │
  │  • Levenberg-Marquardt optimization (scipy.optimize.curve_fit)     │
  │  • Extract: A (max OD), μ (growth rate), λ (lag time)             │
  │  • Compute: R², RMSE, parameter standard errors                   │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 4: CLASSIFICATION  (01_growth_curve_analysis.py)            │
  │  • R² ≥ 0.95 AND parameter error < 20% → GOOD                     │
  │  • Secondary quality gates: SNR, delta-OD CI, absolute delta-OD   │
  │  • Output: processing_results.csv + diagnostic plots               │
  └────────┬────────────────────────────────────────────┬──────────────┘
           │                                            │
           ▼                                            ▼
  ┌────────────────────┐                    ┌───────────────────────────┐
  │  STAGE 5: HALDANE  │                    │  STAGE 6: ADVANCED        │
  │  (05_haldane.py)   │                    │  (06_advanced_fitting.py) │
  │  • ODE substrate   │                    │  • GP truncation          │
  │    inhibition      │                    │  • Ensemble truncation    │
  │  • AIC comparison  │                    │  • Bayesian Gompertz      │
  │    vs Gompertz     │                    │  • Bayesian Haldane       │
  └────────────────────┘                    │  • Bootstrap CIs          │
                                            │  • WAIC/LOO comparison    │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 7: TRUNCATION COMPARISON & RESCUE                           │
  │  (07_compare_truncation_methods.py / 08_validate_truncation.py)    │
  │  • Compare 5 methods + consensus via per-strain Gompertz R²        │
  │  • Overlay plots: raw data + 6 Gompertz curves per strain          │
  │  • Bad strain rescue: re-truncate failed strains, recover 6/9      │
  │  • Interactive validator: human review of method selections         │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 8: VALIDATION  (synthetic_data/)                            │
  │  • 480 synthetic ground-truth curves across 32 scenarios           │
  │  • Pipeline accuracy: 84.4%, Sensitivity: 94.5%                   │
  │  • Manual audit of real data: 91.3% agreement                      │
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

The Haldane model is fitted using **L-BFGS-B** (bounded Limited-memory BFGS), a quasi-Newton optimizer suitable for bounded parameter spaces:

```python
# From 05_haldane_analysis.py, lines 91-168
def fit_haldane(time, od, S0=1.0, gompertz_params=None):
    # Use Gompertz parameters to inform initial guesses
    X0_init = max(float(np.mean(od[:3])), 0.001)
    X_max_init = float(np.max(od)) * 1.1

    if gompertz_params:
        mu_init = gompertz_params.get('mu', 0.2) * 2.0   # Haldane μ_max > Gompertz μ

    p0 = np.array([mu_init, 0.1, 10.0, X_max_init, 0.1, X0_init, S0])

    bounds = [
        (0.001, 5.0),    # mu_max
        (0.001, 10.0),   # Ks
        (0.1, 1000.0),   # Ki
        (0.05, 5.0),     # X_max
        (0.001, 10.0),   # q
        (1e-6, 0.5),     # X0
        (0.01, 100.0),   # S0
    ]

    def objective(params):
        mu_max, Ks, Ki, X_max, q, X0, S0_fit = params
        X_pred, _ = solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0_fit)
        return np.sum((od - X_pred)**2)    # Sum of squared residuals

    result = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000})
```

### 8.6 Gompertz vs. Haldane comparison (AIC)

Both models are compared using the **Akaike Information Criterion** (AIC), which penalizes model complexity:

```
AIC = n × ln(SS_res / n) + 2k

where:
  n = number of data points
  k = number of free parameters (3 for Gompertz, 7 for Haldane)
```

Lower AIC is better. The Haldane model has 4 more parameters, so it must explain substantially more variance to justify its added complexity.

```python
# From 05_haldane_analysis.py, lines 150-153
n = len(od)
k = 7  # Haldane has 7 parameters
aic = n * np.log(ss_res / n) + 2 * k
bic = n * np.log(ss_res / n) + k * np.log(n)
```

For Gompertz comparison:

```python
# From 05_haldane_analysis.py, lines 180-190
k = 3  # Gompertz has 3 parameters
aic = n * np.log(ss_res / n) + 2 * k
```

**Result:** 15/23 (65%) pesticide+LB strains preferred the Haldane model over Gompertz by AIC, confirming that substrate inhibition kinetics are at play.

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

**6 of 9 candidate bad strains were rescued** — their poor original fits were caused by suboptimal truncation, not by lack of growth. This changes the dataset from 57/35 (good/bad) to potentially 63/29.

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

To validate the pipeline's accuracy, we generated 480 synthetic growth curves with **known ground-truth parameters**, ran the pipeline on them, and measured how well it recovered the true classifications and parameters.

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

Four noise types simulate different real-world noise sources:

```python
# From synthetic_data/src/noise_models.py

# 1. Gaussian: constant σ across all OD values
noise = np.random.normal(0, sigma, len(clean_curve))

# 2. OD-Dependent: noise scales with signal (heteroscedastic)
noise = np.random.normal(0, sigma_base + sigma_scale * clean_curve)

# 3. Instrument noise: combines baseline, proportional, drift, and outliers
noise = (baseline_noise +                           # constant floor
         proportional_noise * clean_curve +          # scales with signal
         drift * np.linspace(0, 1, n) +             # systematic drift
         outlier_spikes)                              # rare large errors

# 4. RMSE-Based: adjusts noise σ iteratively to achieve a target R²
while abs(achieved_r2 - target_r2) > tolerance:
    sigma = adjust(sigma, achieved_r2, target_r2)
```

Noise presets (from `synthetic_data/config/default_config.yaml`):

| Preset | σ | Target R² | Description |
|--------|---|-----------|-------------|
| very_low | 0.002 | 0.995 | Clean instrument data |
| low | 0.005 | 0.985 | Typical TECAN data |
| medium | 0.015 | 0.96 | Moderate noise |
| high | 0.03 | 0.92 | Noisy conditions |
| very_high | 0.06 | 0.85 | Extreme noise |

### 11.4 The 32 test scenarios

The scenarios are defined in `synthetic_data/src/curve_scenarios.py` and fall into three categories:

**Good growth scenarios (7 types, 105 curves):**

| Scenario | A range | μ range | λ range | Purpose |
|----------|---------|---------|---------|---------|
| standard | 0.8–1.5 | 0.1–0.3 | 2–8h | Typical LB growth |
| high_A | 1.5–2.0 | 0.1–0.3 | 2–8h | Dense culture |
| low_A | 0.15–0.35 | 0.05–0.15 | 2–8h | Minimal but valid |
| fast_growth | 0.5–1.2 | 0.4–0.75 | 1–3h | Rapid growers |
| slow_growth | 0.3–0.8 | 0.02–0.08 | 5–15h | Slow growers |
| short_lag | 0.5–1.5 | 0.1–0.3 | 0.1–1h | No lag phase |
| long_lag | 0.3–1.0 | 0.05–0.2 | 20–50h | Extended lag |

**Bad curve scenarios (8 types, 120 curves):**

| Scenario | Expected classification | Description |
|----------|----------------------|-------------|
| flat_no_growth | BAD | H₂O control — no growth at all |
| contamination | BAD | High initial OD (0.15–0.4) |
| minimal_growth | BAD | ΔOD < 0.05 |
| high_noise | BAD | Valid growth obscured by noise (R² < 0.90) |
| erratic | BAD | Random non-biological pattern |
| declining | BAD | Only decreasing OD |
| sparse_data | BAD | Only 20 time points |
| truncated_early | BAD | 15h experiment (no stationary phase) |

**Edge cases (17 types, 255 curves):**

| Scenario | Challenge for the pipeline |
|----------|---------------------------|
| borderline_r2 | R² exactly near the 0.95 threshold |
| truncation_challenge | Multiple local maxima |
| death_phase | Clear decline after plateau |
| diauxic_growth | Two distinct growth phases |
| baranyi_generated | Non-Gompertz growth model |
| logistic_generated | Symmetric instead of asymmetric sigmoid |

### 11.5 Validation metrics

The pipeline was run on all 480 synthetic curves and compared against ground truth:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 84.4% | Correct classifications / total |
| **Precision** | 85.3% | True GOOD / (True GOOD + False GOOD) |
| **Recall (Sensitivity)** | 94.5% | True GOOD / All actually GOOD |
| **Specificity** | 58.5% | True BAD / All actually BAD |
| **F1 Score** | 89.7% | Harmonic mean of precision and recall |

**Confusion matrix:**
```
                    Pipeline says GOOD    Pipeline says BAD
Actually GOOD            326 (TP)              19 (FN)
Actually BAD              56 (FP)              79 (TN)
```

The pipeline is intentionally **biased toward acceptance** (high sensitivity, lower specificity) — it's better to include a marginal curve for manual review than to miss real growth.

### 11.6 Parameter recovery accuracy

For the 326 correctly classified GOOD curves, how well did the pipeline recover the true parameters?

| Parameter | R² | RMSE | Mean Bias | Interpretation |
|-----------|-----|------|-----------|----------------|
| A (max OD) | 0.985 | 0.042 | +0.020 | Excellent — slight overestimate |
| μ (growth rate) | 0.906 | 0.030 | -0.007 | Good — slight underestimate |
| λ (lag time) | 0.983 | 1.013h | -0.366h | Excellent — slight early estimate |

---

## 12. Configuration Reference

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

## 13. Execution & Reproducibility

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

**Quick start (via Makefile):**

```bash
make install              # Install all Python dependencies
make run-pipeline         # Run Stage 2-4: preprocess → truncate → fit → classify
make run-haldane          # Run Stage 5: Haldane substrate inhibition analysis
make run-advanced         # Run Stage 6: GP + Bootstrap + Bayesian + Ensemble
make run-all              # Run everything (Stages 2-6)
make test                 # Run pytest test suite (64 tests)
make validate             # Interactive curve auditor
```

**Truncation comparison & validation (Stage 7):**

```bash
make run-compare-methods  # Compare truncation methods (requires ensemble run first)
make run-compare-bad      # Compare methods + rescue bad strains (--include-bad)
make validate-truncation  # Launch interactive truncation method validator
```

**Advanced fitting sub-targets (Stage 6):**

```bash
make run-advanced-gp          # GP truncation only
make run-advanced-bootstrap   # Bootstrap CIs only
make run-advanced-gompertz    # Bayesian Gompertz only
make run-advanced-haldane     # Bayesian Haldane only
make run-advanced-ensemble    # Ensemble truncation only
```

**Individual scripts:**

```bash
# Stage 1: Preprocess raw plate data (Group 1 example)
python scripts/02_preprocess_raw_plate_data.py \
    data/raw/Group1/GrowthRate_Group1_Values.csv \
    data/raw/Group1/GROUP1_KEY_V2.csv \
    -o data/raw/Group1/Group_1_DATA

# Stage 2-4: Core analysis (single group)
python scripts/01_growth_curve_analysis.py \
    data/raw/Group1/Group_1_DATA \
    -o results/tables/Group1_Results \
    --adaptive   # Enable adaptive truncation (optional)

# Stage 5: Haldane analysis
python scripts/05_haldane_analysis.py \
    --results-dir results/tables \
    --output results/tables/Haldane_Analysis \
    --s0 2.5  # Initial substrate concentration

# Stage 6: Advanced methods
python scripts/06_advanced_fitting.py \
    --gompertz-only    # Skip Haldane Bayesian (faster)
    --chains 2         # Reduce chains for testing
    --draws 500        # Reduce draws for testing

# Batch processing (all 4 groups)
bash scripts/run_all_groups.sh
```

**Stage 7: Truncation method comparison & rescue:**

```bash
# Compare all 5 methods + consensus on 57 good strains
python scripts/07_compare_truncation_methods.py

# Also attempt to rescue bad strains
python scripts/07_compare_truncation_methods.py --include-bad

# Interactive truncation method validator
python scripts/08_validate_truncation.py
python scripts/08_validate_truncation.py --comparison-csv path/to/method_comparison.csv
```

**Synthetic validation:**

```bash
# Generate comprehensive test suite (480 curves)
python synthetic_data/scripts/generate_comprehensive.py \
    --output-dir synthetic_data/output/comprehensive_test

# Run pipeline on synthetic data and compare to ground truth
python synthetic_data/scripts/validate_pipeline.py \
    --test-dir synthetic_data/output/comprehensive_test
```

### 13.3 CI/CD

The `.github/workflows/test.yml` file configures automatic testing on GitHub:
- Tests against Python 3.10, 3.11, and 3.12
- Runs the full pytest suite on every push/PR
- Validates core functionality (Gompertz fitting, classification, Haldane ODE solving)

---

## 14. Results Summary

### 14.1 Real experimental data (92 strains)

| Group | Pesticides | Total Strains | Good | Bad |
|-------|-----------|:---:|:---:|:---:|
| 1 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| 2 | Malathion, Lambda-Cyhalothrin | 24 | 16 | 8 |
| 3 | Imidacloprid, Lambda-Cyhalothrin | 24 | 17 | 7 |
| 4 | Permethrin, Lambda-Cyhalothrin | 20 | 12 | 8 |
| **Total** | | **92** | **57 (62%)** | **35 (38%)** |

### 14.2 Growth parameter distributions (57 GOOD curves)

| Parameter | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| A (Max OD) | 0.025 | 1.87 | 0.72 | 0.45 |
| μ (Growth rate, h⁻¹) | 0.001 | 0.75 | 0.18 | 0.15 |
| λ (Lag time, h) | 0.035 | 76.7 | 8.2 | 12.5 |
| R² | 0.95 | 0.9998 | 0.985 | 0.015 |

### 14.3 Haldane analysis (23 pesticide+LB strains)

- **Haldane preferred over Gompertz by AIC:** 15/23 (65%)
- All 6 pesticides show measurable Ki (inhibition constant)
- Confirms substrate inhibition kinetics are present in pesticide bioremediators

### 14.4 Truncation method comparison (57 good strains)

| Rank | Method | Mean R² | % Good (R²≥0.95) |
|:---:|--------|:-------:|:-----------------:|
| 1 | Consensus (weighted median) | 0.9804 | 96.5% |
| 2 | Adaptive R² | 0.9800 | 98.2% |
| 3 | Stationary Phase | 0.9799 | 98.2% |
| 4 | First Peak | 0.9790 | 94.7% |
| 5 | GP Derivative | 0.9787 | 94.7% |
| 6 | Changepoint | 0.7099 | 87.5% |

### 14.5 Bad strain rescue

6 of 9 candidate bad strains were rescued by alternative truncation methods, potentially increasing the good strain count from 57 to 63.

### 14.6 Pipeline validation

| Validation type | Accuracy |
|----------------|----------|
| Synthetic data (480 curves) | 84.4% overall, 94.5% sensitivity |
| Manual audit (92 real curves) | 91.3% agreement |

### 14.7 Test suite

64 tests across 5 test files covering Gompertz fitting, Haldane ODE, classification logic, ensemble truncation, GP truncation, bootstrap, and integration tests.

---

*Generated for BIO380SP25 — Pesticide Bioremediating Bacteria Research Project*
*Pipeline version: February 2025*