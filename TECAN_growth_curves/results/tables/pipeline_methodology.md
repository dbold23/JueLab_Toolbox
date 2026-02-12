# TECAN Growth Curve Analysis Pipeline - Methodology

## Growth Model

All growth curves were fit to the **modified Gompertz equation**:

```
y(t) = A * exp( -exp( (mu * e / A) * (lambda - t) + 1 ) )
```

Where:

| Parameter | Symbol | Biological Meaning |
|-----------|--------|--------------------|
| A | A | Maximum population density (asymptotic OD600). Represents the carrying capacity of the culture under the given conditions. |
| mu | mu | Maximum specific growth rate (OD600 / hour). The steepest slope of the growth curve during exponential phase. |
| lambda | lambda | Lag phase duration (hours). The time before exponential growth begins, during which cells adapt to their environment. |

The modified Gompertz model is widely used for microbial growth because it
captures the three canonical phases of batch culture (lag, exponential,
stationary) in a single sigmoidal equation with biologically interpretable
parameters.

The **doubling time** is derived from the Gompertz parameters as:

```
t_d = ln(2) / (mu * e / A)
```

where `mu * e / A` gives the maximum instantaneous specific growth rate.

---

## Data Processing

### Preprocessing Steps

1. **Blank subtraction** -- For each media type, blank wells (no inoculum)
   were averaged across replicates at each time point. The blank average was
   subtracted from every strain well to correct for media absorbance.

2. **Triplicate averaging** -- Each strain/condition was measured in
   triplicate wells. The three replicates were averaged to produce a single
   representative growth curve per strain per condition.

3. **Time conversion** -- Raw plate reader times (in seconds) were converted
   to hours.

4. **Non-negativity enforcement** -- After blank subtraction, any negative
   OD600 values (arising from instrument noise) were set to zero.

---

## Classification Criteria

Growth curves were classified using a **fit-quality-based** approach, which
is more scientifically rigorous than raw OD thresholds because it evaluates
whether the curve actually follows Gompertz growth kinetics.

### Classification Thresholds

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| R-squared (R^2) | >= 0.95 | The Gompertz fit must explain at least 95% of variance in the truncated curve. |
| A parameter error | < 20% relative | The standard error of the fitted maximum OD must be less than 20% of the parameter value. |
| mu parameter error | < 20% relative | The standard error of the fitted growth rate must be less than 20% of the parameter value. |

A curve is classified as **Good** only if ALL three criteria are satisfied.
Otherwise it is classified as **Bad** with a specific reason recorded.

---

## Truncation Strategy

Before fitting, each growth curve was truncated to retain only the portion
that conforms to the Gompertz sigmoid shape (lag through early stationary
phase). Two complementary strategies were used:

### Adaptive Truncation (Primary)

1. **Biological estimate** -- The stationary phase onset was estimated by
   finding where the curve reached 90% of maximum OD and where the growth
   rate dropped below 5% of its maximum.

2. **R-squared optimization** -- A search window around the biological
   estimate was scanned. At each candidate truncation point, a quick
   Gompertz fit was performed, and the truncation point yielding the highest
   R^2 was selected. Ties in R^2 were broken by preferring the point closest
   to the biological estimate.

3. **Noisy-start trimming** -- The start of the curve was also evaluated.
   Data points before the onset of measurable growth (baseline noise) were
   trimmed to improve fit quality.

### First-Peak Truncation (Fallback)

For curves where adaptive truncation did not converge, the data was truncated
at the **first local maximum** after exponential growth. This prevents the
inclusion of diauxic shifts, death phase, or other non-Gompertz behavior.

---

## Validation Results

The pipeline was validated on **480 synthetic growth curves** generated with
known Gompertz parameters across a range of A, mu, and lambda values, with
realistic levels of biological noise.

| Metric | Value |
|--------|-------|
| Overall classification accuracy | 84.4% |
| Parameter recovery R^2 (A) | > 0.90 |
| Parameter recovery R^2 (mu) | > 0.90 |
| Parameter recovery R^2 (lambda) | > 0.90 |

Synthetic curves with parameters near classification boundaries (e.g., very
low growth, high noise) accounted for most misclassifications. For curves
with clear sigmoid growth, accuracy exceeded 95%.

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib scipy
# or
pip install -r requirements.txt
```

### Quick Start

```bash
# Run the complete pipeline (all groups, preprocessing + analysis + combine)
cd ANALYSIS_SCRIPTS/
./run_all_groups.sh

# Run analysis on a single group
python 01_growth_curve_analysis.py DATA/<GROUP_FOLDER> -o OUTPUT/<GROUP>_Final --adaptive

# Preprocess raw Group 1 plate reader data
python 02_preprocess_raw_plate_data.py \
    "DATA/Group1/GrowthRate_ Group1_Values.csv" \
    "DATA/Group1/GROUP1_KEY_V2.csv" \
    -o "DATA/Group1/Group_1_DATA"

# Generate collaboration exports (this script)
python 04_export_for_collaboration.py
# or with custom paths:
python 04_export_for_collaboration.py --input ../OUTPUT/all_groups_results.csv --output-dir ../OUTPUT
```

---

## CSV Column Definitions

The collaboration CSV (`growth_rate_summary_for_collaboration.csv`) contains
only **Good** fits and uses human-readable column names:

| Column | Description |
|--------|-------------|
| Strain | Full strain identifier (TREATMENT-STRAINID) |
| Group | Experimental group (Group1 -- Group4) |
| Treatment | Treatment condition parsed from strain name (e.g., LB, BifenthrinANDLB, H2O) |
| Strain_ID | Short strain identifier parsed from strain name (e.g., BIF2, MAL8) |
| Classification | Good or Bad based on fit quality criteria |
| Max_OD600_A | Gompertz A parameter -- fitted maximum OD600 (asymptote) |
| A_StdErr | Standard error of the A parameter estimate |
| Growth_Rate_mu_OD_per_hr | Gompertz mu parameter -- maximum specific growth rate (OD600/hour) |
| mu_StdErr | Standard error of the mu parameter estimate |
| Lag_Phase_lambda_hrs | Gompertz lambda parameter -- lag phase duration (hours) |
| lambda_StdErr | Standard error of the lambda parameter estimate |
| Doubling_Time_hrs | Estimated doubling time: ln(2) / (mu * e / A). NaN for bad fits |
| R_squared | Coefficient of determination (R^2) of the Gompertz fit |
| RMSE | Root mean squared error of the Gompertz fit |

---

## Experimental Groups

| Group | Pesticide(s) | Strains | Total Curves | Good | Bad |
|-------|-------------|---------|--------------|------|-----|
| Group1 | Bifenthrin, Flupyradifurone, LambdaCyhalothrin | BIF2, BIF6, BIF9, BiF7, FLUPC8, LCY4 | 24 | 12 | 12 |
| Group2 | LAMBDACYHALOTHRIN, MALATHION | LCY1, MAL1, MAL10, MAL4, MAL5, MAL8 | 24 | 16 | 8 |
| Group3 | IMIDACLOPRID, LAMBDACYHALOTHRIN | IMID2, IMID3, IMID5, IMID8, LCY01, LCY2 | 24 | 16 | 8 |
| Group4 | LAMBDACYHALOTHRIN, PERMETHRIN | CISPERM1, CYPERM2, LCY3, PERM4, PERM7 | 20 | 13 | 7 |
| **Total** | | | **92** | **57** | **35** |

### Treatment Conditions per Strain

Each strain was measured under multiple conditions:

- **LB** -- LB broth only (positive control; bacteria should grow normally)
- **Pesticide + LB** -- Pesticide dissolved in LB broth (test condition)
- **Pesticide only** -- Pesticide in water without nutrients (negative control)
- **H2O** -- Water only (negative control; no growth expected)
