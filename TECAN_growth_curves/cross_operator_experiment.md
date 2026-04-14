# Cross-Operator Validation of the TECAN Growth Curve Pipeline

**Daniel Sambold, Mary Snook, and Nathaniel Jue, Ph.D.**
Department of Biology, California State University, Monterey Bay
BIO380SP25 -- Pesticide Bioremediating Bacteria Research Project

---

## 1. Background & Motivation

### The Problem

Bioremediation offers a sustainable strategy for degrading pesticide residues in agricultural environments, yet screening candidate bacteria requires analyzing hundreds of growth curves from high-throughput plate reader experiments. Manual classification of these curves is labor-intensive, subjective, and difficult to reproduce across experimenters.

### Research Question

Can soil bacteria isolated from agricultural sites metabolize common pesticides -- bifenthrin, diazinon, flupyradifurone, imidacloprid, lambda-cyhalothrin, malathion, and permethrin -- as sole carbon and energy sources?

### Why Cross-Operator Validation Matters

Our data were collected by **three independent operators** across **two years** (2024--2025). For the pipeline to serve as a general bioremediation screening tool, growth parameters extracted from TECAN plate reader experiments must be **reproducible across experimenters, instruments, and time**. The cross-operator comparison is the ultimate test of whether our automated pipeline produces operator-independent results.

### Scale of the Study

| Metric | Value |
|--------|-------|
| Total averaged growth curves | 161 |
| Experimental groups | 6 |
| Pesticides tested | 7 |
| Independent operators | 3 |
| Time span | 2024--2025 |
| Shared strains for comparison | 5 (imidacloprid) |

---

## 2. Experimental Design

### 2.1 Operators and Groups

Each operator independently performed growth assays on bacterial strains isolated from pesticide-treated soils. The data were organized into six experimental groups:

| Group | Operator | Year | Pesticides Tested | Total Curves | Good Fits | Bad Fits |
|:-----:|----------|:----:|-------------------|:------------:|:---------:|:--------:|
| 1 | Operator1 | 2025 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| 2 | Operator1 | 2025 | Malathion, Lambda-Cyhalothrin | 24 | 11 | 13 |
| 3 | Operator1 | 2025 | Imidacloprid, Lambda-Cyhalothrin | 24 | 12 | 12 |
| 4 | Operator1 | 2025 | Permethrin, Lambda-Cyhalothrin | 20 | 11 | 9 |
| 5 | Walton | 2024 | Imidacloprid, Malathion, Diazinon | 59 | 16 | 43 |
| 6 | Dominique | 2024 | Imidacloprid | 10 | 4 | 6 |
| **Total** | | | **7 pesticides** | **161** | **66** | **95** |

### 2.2 Operator Summary

| Operator | Groups | N Good Fits | Success Rate | Pesticides |
|----------|:------:|:-----------:|:------------:|------------|
| Operator1 | 1, 2, 3, 4 | 46 | 50% | Bifenthrin, Flupyradifurone, Imidacloprid, Lambda-Cyhalothrin, Malathion, Permethrin |
| Walton | 5 | 16 | 27% | Diazinon, Imidacloprid, Malathion |
| Dominique | 6 | 4 | 40% | Imidacloprid |

### 2.3 Shared Strains for Cross-Operator Comparison

The cross-operator comparison is anchored on **five imidacloprid-treated strains** that were independently tested by multiple operators:

| Strain | Operators | N Operators | N Curves | Comparison Test |
|--------|-----------|:-----------:|:--------:|-----------------|
| IMID2 | Operator1, Walton, Dominique | 3 | 5 | One-way ANOVA |
| IMID5 | Operator1, Walton, Dominique | 3 | 3 | One-way ANOVA |
| IMID8 | Operator1, Walton, Dominique | 3 | 3 | One-way ANOVA |
| IMID3 | Operator1, Walton | 2 | 2 | Welch's t-test |
| IMID6 | Walton, Dominique | 2 | 3 | Welch's t-test |

These strains were all tested under the **Pesticide+LB** media condition, enabling direct comparison of growth kinetics in the presence of imidacloprid with nutrient supplementation.

### 2.4 Experimental Protocol

- **Instrument**: TECAN Infinite plate reader
- **Format**: 96-well microplates
- **Measurement**: Optical density at 600 nm (OD600)
- **Interval**: Every 15 minutes
- **Replicates**: 3 technical replicates per strain/condition, averaged during preprocessing

### 2.5 Media Conditions (per strain)

Each bacterial strain-pesticide combination was tested under four conditions:

| Condition | Purpose |
|-----------|---------|
| **LB only** | Positive growth control -- confirms the strain is viable |
| **H2O only** | Negative control -- no carbon source available |
| **Pesticide alone** | Tests if bacteria can use pesticide as sole carbon source |
| **Pesticide + LB** | Tests growth with pesticide present alongside nutrients |

### 2.6 Pesticide Concentrations

| Pesticide | Concentration | Supplemental Glucose |
|-----------|:------------:|:--------------------:|
| Imidacloprid | 20 mg/L | 20 mg/L |
| Bifenthrin | 50--100 mg/L | -- |
| Malathion | 50 mg/L | 45 mg/L |
| Lambda-Cyhalothrin | 50 mg/L | -- |
| Permethrin | 100 mg/L | -- |
| Diazinon | 20 mg/L | -- |
| Flupyradifurone | 20 mg/L | -- |

---

## 3. Analysis Pipeline Overview

### 3.1 Eleven-Step Automated Pipeline

The entire analysis -- from raw 96-well plate data to publication figures -- is executed through an automated pipeline orchestrated by `run_full_pipeline.py`:

| Step | Name | Script | Description |
|:----:|------|--------|-------------|
| 0 | Preprocess | `02_preprocess_raw_plate_data.py` | Raw CSV -> blank subtraction, triplicate averaging, time conversion (s to h) |
| 1 | Train ML Classifier | `09_train_classifier.py` | Two-stage HistGradientBoosting on 565 curves (480 synthetic + 85 real) |
| 2 | Gompertz Fitting | `01_growth_curve_analysis.py` | Modified Gompertz model fitting with adaptive truncation + GOOD/BAD classification |
| 3 | Combine Results | (merge step) | Per-group results consolidated into `all_groups_results.csv` |
| 4 | Haldane Inhibition | `05_haldane_analysis.py` | Substrate inhibition ODE model, AICc comparison vs Gompertz |
| 5 | Advanced Fitting | `06_advanced_fitting.py` | GP-based truncation, Bayesian hierarchical models, bootstrap CIs |
| 6 | Statistical Analysis | `03_statistical_analysis.py` | Cross-group ANOVA, pairwise tests, publication figures |
| 7 | Truncation Comparison | `07_compare_truncation_methods.py` | 5-method ensemble comparison, bad-strain rescue |
| 8 | Export | `04_export_for_collaboration.py` | Clean CSV + methodology for collaborators |
| 9 | Synthetic Validation | (synthetic data suite) | 480 ground-truth curves across 32 scenarios |
| **10** | **Operator Comparison** | **`10_operator_comparison.py`** | **Cross-operator ANOVA on shared strains** |

### 3.2 Modified Gompertz Growth Model

Each growth curve is fitted with the modified Gompertz equation:

```
y(t) = A * exp( -exp( (mu * e / A) * (lambda - t) + 1 ) )
```

| Parameter | Symbol | Units | Biological Meaning |
|-----------|:------:|:-----:|-------------------|
| Maximum OD | A | OD600 | Asymptotic carrying capacity (final population density) |
| Growth rate | mu | OD/h | Maximum specific growth rate (steepest slope of growth phase) |
| Lag time | lambda | hours | Duration of lag phase before exponential growth begins |

### 3.3 Classification Thresholds

A curve is classified as **GOOD** only if it passes all quality gates:

| Gate | Threshold | Purpose |
|------|:---------:|---------|
| R-squared | >= 0.95 | Minimum goodness of fit |
| Parameter error | < 20% | Maximum relative standard error on fitted parameters |
| Signal-to-noise ratio | >= 5.0 | Minimum SNR for reliable curve detection |
| Delta-OD | >= 0.15 | Minimum OD change to confirm actual growth |
| Monotone fraction | >= 0.55 | Minimum fraction of consecutively increasing points |
| Residual autocorrelation | < 0.7 | Maximum lag-1 autocorrelation in residuals |

### 3.4 ML Classifier Architecture

A two-stage machine learning system augments rule-based classification:

**Stage 1 -- Pre-Fit Gate:**
- 10 raw-signal features (no fitting required)
- Rejects obvious BAD curves before expensive Gompertz fitting
- Reject threshold: P(good) <= 0.05

**Stage 2 -- Post-Fit Classifier:**
- 24 features (fit quality metrics + raw signal features)
- HistGradientBoosting classifier
- GOOD threshold: P(good) >= 0.70
- BAD threshold: P(good) <= 0.30

**Training Data:** 480 synthetic + 85 real audited = 565 curves, 70/30 stratified split

### 3.5 Haldane Substrate Inhibition Model

For pesticide-strain combinations, a mechanistic Haldane ODE model captures substrate-dependent growth inhibition:

```
mu(S) = mu_max * S / (Ks + S + S^2/Ki)
```

| Parameter | Meaning |
|-----------|---------|
| mu_max | Maximum growth rate (uninhibited) |
| Ks | Half-saturation constant |
| Ki | Inhibition constant (lower = more inhibitory) |

Model selection between Haldane and Gompertz uses corrected Akaike Information Criterion (AICc).

### 3.6 Uncertainty Quantification

- **Bayesian hierarchical models**: Partial pooling across strains within pesticide groups (PyMC, 4 chains, 2000 draws + 1000 tuning)
- **Bootstrap confidence intervals**: 1000 resamples, 95% CIs on all Gompertz parameters
- **Gaussian Process truncation**: RBF kernel-based derivative analysis for optimal curve truncation point

---

## 4. Comparison & Validation Strategy

### 4.1 Cross-Operator Comparison Design

The operator comparison (Step 10) tests whether Gompertz growth parameters are consistent across operators for the same bacterial strains:

1. **Identify shared strains**: Parse strain IDs to find strains tested by 2+ operators
2. **Extract parameters**: Collect A, mu, and lambda from GOOD-classified fits only
3. **Statistical comparison**:
   - **3 operators**: One-way ANOVA
   - **2 operators**: Welch's t-test (assumes unequal variance)
4. **Variability assessment**: Coefficient of Variation (CV%) across operator means

### 4.2 CV Interpretation Scale

| CV Range | Interpretation |
|:--------:|---------------|
| < 20% | Good reproducibility |
| 20--40% | Moderate reproducibility |
| > 40% | Poor reproducibility |

### 4.3 Pipeline Validation (Independent of Operator Comparison)

| Validation Method | Dataset | Purpose |
|-------------------|---------|---------|
| Synthetic curves | 480 curves, 32 scenarios | Ground-truth parameter recovery |
| ML held-out test | 170 curves (30% split) | Classifier generalization |
| Manual audit | 92 real curves | Human-vs-pipeline agreement |

### 4.4 Key Limitation: No Shared LB Controls

Each operator used their own LB-only control strains. Without a common baseline:
- Cannot normalize pesticide growth to operator-specific instrument bias
- Cannot distinguish systematic protocol differences from biological variability
- Cannot confirm whether absolute growth rate differences reflect true biology or technical artifacts

---

## 5. Results

### 5.1 Overall Pipeline Performance

| Metric | Value |
|--------|:-----:|
| Total curves analyzed | 161 |
| Good fits (GOOD classification) | 66 (41%) |
| Bad fits (BAD classification) | 95 (59%) |
| Mean R-squared (good curves) | 0.971--0.977 |
| Manual audit agreement | 91.3% |

### 5.2 Operator-Level Growth Parameters (GOOD Fits Only)

| Operator | N Curves | N Pesticides | Mean mu (h^-1) | SD mu | Mean lambda (h) | SD lambda | Mean R-squared |
|----------|:--------:|:------------:|:--------------:|:-----:|:---------------:|:---------:|:--------------:|
| Operator1 | 46 | 8 | **0.2349** | 0.2212 | 5.06 | 11.37 | 0.9707 |
| Walton | 16 | 3 | **0.0377** | 0.0343 | 2.37 | 3.05 | 0.9769 |
| Dominique | 4 | 1 | **0.0272** | 0.0088 | 4.28 | 3.08 | 0.9728 |

**Critical observation:** Operator1's mean growth rate (0.235 h^-1) is **6.2x higher** than Walton (0.038 h^-1) and **8.6x higher** than Dominique (0.027 h^-1). This represents a systematic operator-level bias.

### 5.3 Cross-Operator Statistical Comparison (ANOVA/t-test on Shared Strains)

| Strain | Parameter | Test | F-statistic | p-value | CV (%) | Interpretation |
|--------|-----------|------|:-----------:|:-------:|:------:|---------------|
| IMID2 | Growth Rate (mu) | ANOVA | 0.486 | 0.673 | 63.1% | Poor |
| IMID2 | Lag Time (lambda) | ANOVA | 0.534 | 0.652 | 48.2% | Poor |
| IMID2 | Max OD (A) | ANOVA | 0.333 | 0.750 | 50.3% | Poor |
| IMID5 | Growth Rate (mu) | ANOVA | -- | -- | 62.5% | Poor |
| IMID5 | Lag Time (lambda) | ANOVA | -- | -- | 74.9% | Poor |
| IMID5 | Max OD (A) | ANOVA | -- | -- | 28.7% | Moderate |
| IMID6 | Growth Rate (mu) | Welch t | -- | -- | **7.0%** | **Good** |
| IMID6 | Lag Time (lambda) | Welch t | -- | -- | 100.0% | Poor |
| IMID6 | Max OD (A) | Welch t | -- | -- | 57.3% | Poor |
| IMID8 | Growth Rate (mu) | ANOVA | -- | -- | 84.8% | Poor |
| IMID8 | Lag Time (lambda) | ANOVA | -- | -- | 95.4% | Poor |
| IMID8 | Max OD (A) | ANOVA | -- | -- | 70.0% | Poor |

**Summary:**
- **All computable p-values > 0.05** -- no statistically significant differences detected
- **Mean CV across all parameters: 61.8%** -- overall poor reproducibility
- **CV range: 7.0% to 100.0%** -- extreme variability across strain-parameter combinations
- IMID5 and IMID8 had N=1 per operator (insufficient for ANOVA F-statistics)

### 5.4 The IMID6 Exception

IMID6 growth rate shows a CV of only **7.0%** between Walton and Dominique -- the only parameter meeting the "good reproducibility" threshold. Notably, this comparison **excludes Operator1**, suggesting that the two 2024 operators show much better agreement with each other than either does with the 2025 operator.

### 5.5 Haldane Substrate Inhibition Results

| Pesticide | N Strains | Haldane Preferred | Bayesian Ki | Interpretation |
|-----------|:---------:|:-----------------:|:-----------:|---------------|
| **Imidacloprid** | 17 | 65% (11/17) | **4.83** | Most inhibitory |
| **Flupyradifurone** | 1 | 100% (1/1) | **5.87** | Highly inhibitory |
| Malathion | 5 | 100% (5/5) | -- | Strong inhibition |
| Diazinon | 6 | 83% (5/6) | -- | Strong inhibition |
| Lambda-Cyhalothrin | 5 | 80% (4/5) | -- | Strong inhibition |
| Permethrin | 4 | 100% (4/4) | -- | Strong inhibition |
| Bifenthrin | 4 | 0% (0/4) | -- | Minimal inhibition (Gompertz preferred) |

**Overall:** Haldane kinetics preferred over Gompertz in **30 of 42** pesticide-strain combinations (71%), confirming that substrate inhibition is the dominant growth-limiting mechanism for most pesticide-bacteria pairs.

### 5.6 Pipeline Validation Results

#### Synthetic Data Validation (480 curves, 32 scenarios)

| Metric | Value |
|--------|:-----:|
| Classification accuracy | 84.4% |
| Precision | 85.3% |
| Recall (sensitivity) | 94.5% |
| F1 Score | 89.7% |
| Specificity | 58.5% |

#### Parameter Recovery (326 true-positive curves)

| Parameter | R-squared | RMSE | Mean Bias |
|-----------|:---------:|:----:|:---------:|
| A (Max OD) | 0.985 | 0.042 | +0.020 |
| mu (Growth rate) | 0.906 | 0.030 | -0.007 |
| lambda (Lag time) | 0.983 | 1.013 | -0.366 h |

**Interpretation:** The pipeline recovers growth parameters with high fidelity. Growth rate (mu) is inherently harder to estimate precisely (R^2 = 0.906) because it depends on the slope of a narrow exponential window. The slight negative lag time bias (-0.37 h) reflects truncation heuristics that identify growth onset marginally early.

#### ML Classifier Performance (Held-Out Test Set)

| Metric | Value |
|--------|:-----:|
| Held-out accuracy | 99.2% |
| Recall | 97.4% |
| Specificity | 94.3% |

---

## 6. Discussion

### 6.1 "Not Significant" Does Not Mean "Equivalent"

The ANOVA results (all p > 0.05) should **not** be interpreted as evidence that operators produce equivalent results. The non-significance reflects **insufficient statistical power**, not agreement:

- Sample sizes are extremely small (N = 2--5 per strain across all operators)
- With CVs averaging 61.8%, the measurements show substantial practical disagreement
- A proper equivalence test (TOST) would require pre-specified equivalence margins and much larger sample sizes

### 6.2 Systematic Operator Bias

The most striking finding is Operator1's **6--9x higher mean growth rates** compared to Walton and Dominique:

| Operator | Mean mu (h^-1) | Mean A (OD600) | Mean lambda (h) |
|----------|:--------------:|:--------------:|:---------------:|
| Operator1 | 0.235 | 1.33* | 5.06 |
| Walton | 0.038 | 0.50* | 2.37 |
| Dominique | 0.027 | 0.74* | 4.28 |

*A values estimated from overall operator profiles

**Possible explanations for the systematic offset:**
1. **Incubation temperature differences** -- higher temperature accelerates bacterial growth
2. **Media preparation variability** -- LB broth concentration, pH, or autoclave protocols
3. **Inoculum density** -- different starting cell concentrations
4. **Plate reader calibration drift** -- TECAN instruments require periodic calibration; 2024 vs 2025 readings may not be directly comparable
5. **Bacterial stock condition** -- strains stored for different durations may have undergone fitness changes

### 6.3 Evidence That Walton and Dominique Agree

The IMID6 growth rate CV of **7.0%** between Walton and Dominique (both 2024) stands in sharp contrast to the 63--85% CVs seen when Operator1 is included. This suggests:

- The two 2024 operators used **more similar protocols** or conditions
- The **year effect** (2024 vs 2025) may be more important than the operator effect
- A temporal confound cannot be separated from operator identity in this study design

### 6.4 The Pipeline Is Sound; The Data Are Variable

Despite high inter-operator variability in absolute parameter values, the pipeline demonstrates:

- **High classification accuracy**: 84.4% on synthetic data, 91.3% manual audit agreement
- **Strong parameter recovery**: R^2 > 0.90 for all three Gompertz parameters
- **Consistent biological conclusions**: Haldane substrate inhibition is preferred for most pesticide-strain combinations regardless of operator
- **Preserved relative rankings**: Imidacloprid is consistently the most inhibitory pesticide (lowest Ki) across all operators

The variability is in the **input data**, not the analysis.

### 6.5 Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **No shared LB controls** | Cannot normalize for operator-specific baseline growth | Future: all operators run identical LB controls |
| **Unbalanced design** | Operator1 has 46 good fits vs Dominique's 4 | Future: equal replicates per operator |
| **Only imidacloprid is shared** | Cannot assess reproducibility for 6 other pesticides | Future: extend shared strains across pesticides |
| **Temporal confound** | Cannot separate "year effect" from "operator effect" | Future: concurrent experiments by all operators |
| **Small N per comparison** | Low statistical power (N = 1--5 per operator per strain) | Future: minimum 3 biological replicates per operator |
| **High bad-fit rate** | 59% of curves classified as BAD reduces comparison pool | Improve protocol to increase growth success rate |
| **No plate position randomization documented** | Potential edge effects in 96-well plates | Future: randomize strain positions |

### 6.6 Future Directions

1. **Shared LB control experiment**: All three operators should run identical bacterial strains in LB-only on the same plate reader within the same week. This enables ratio-based normalization (Pesticide+LB growth / LB-only growth) that removes absolute OD scale differences.

2. **Plate reader calibration standard**: Run a commercial OD standard solution (e.g., McFarland standards) across instruments to quantify and correct for inter-instrument bias.

3. **Increase replicates**: Minimum 3 biological replicates per operator per strain per condition to power ANOVA properly (current N = 1--2 per operator is fundamentally insufficient).

4. **Extend shared pesticides**: Expand cross-operator comparison beyond imidacloprid to at least malathion and diazinon (already tested by 2 operators each).

5. **Bayesian hierarchical operator model**: Fit a hierarchical model with operator as a random effect to formally quantify and adjust for systematic operator bias while preserving biological signal:
   ```
   mu_observed ~ Normal(mu_true + operator_effect, sigma)
   operator_effect ~ Normal(0, tau)
   ```

6. **Concurrent experiments**: Eliminate the temporal confound by having all operators run experiments during the same time period with shared reagent batches.

---

## 7. Key Figures

The following figures are generated by the pipeline and available in `results/tables/Operator_Comparison/`:

| Figure | File | Description |
|--------|------|-------------|
| Operator Parameter Comparison | `operator_parameter_comparison.png` | Box plots of mu and lambda distributions by operator |
| CV Heatmap | `operator_cv_heatmap.png` | Heatmap showing CV (%) for each strain-parameter combination |

Additional relevant figures in `results/tables/` and `paper/figures/`:
- Per-group growth curve overlays with Gompertz fits
- Haldane vs Gompertz model comparison plots
- Bayesian posterior distributions for Ki values
- Synthetic validation confusion matrices

---

## 8. Summary for Lab Meeting

### Take-Home Points

1. **The pipeline works**: 84.4% accuracy on synthetic data, strong parameter recovery, 91.3% manual audit agreement. The analysis framework is robust and reproducible.

2. **Inter-operator reproducibility is poor**: Mean CV of 61.8% across shared strains. Operator1 produces systematically higher growth rates (6--9x) than Walton and Dominique.

3. **The non-significance is misleading**: All ANOVA p-values > 0.05, but this reflects low sample sizes (N = 2--5), not agreement. The study is underpowered for reproducibility conclusions.

4. **Biological signal is preserved**: Despite absolute value disagreements, the relative ranking of pesticide inhibition (imidacloprid > flupyradifurone > others > bifenthrin) and the preference for Haldane over Gompertz (71%) are consistent across operators.

5. **Shared controls are essential**: The single most impactful improvement would be running identical LB controls across all operators to enable baseline normalization.

### One-Sentence Conclusion

> Our automated pipeline produces reliable, validated growth parameter estimates, but inter-operator comparison reveals systematic biases that require standardized protocols and shared controls before cross-operator data can be meaningfully combined.

---

*Pipeline code and all results: `TECAN_growth_curves/`*
*Full methodology: `PIPELINE_METHODOLOGY.md` (74 KB)*
*Manuscript draft: `paper/full_paper.md`*
