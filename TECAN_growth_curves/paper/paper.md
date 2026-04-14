# An automated computational pipeline for bacterial growth curve classification and substrate inhibition modeling in pesticide bioremediation screening

Daniel Sambold, Mary Snook, and Nathaniel Jue, Ph.D.

Department of Biology, California State University, Monterey Bay

---

## Abstract

Bioremediation offers a sustainable strategy for degrading pesticide residues in agricultural environments, yet screening candidate bacteria requires analyzing hundreds of growth curves from high-throughput plate reader experiments, a process that is labor-intensive and subjective when performed manually. We developed an eleven-step computational pipeline that automates bacterial growth curve analysis from TECAN plate reader data to assess whether soil bacteria can metabolize pesticides as sole carbon sources. The pipeline fits modified Gompertz growth models with adaptive truncation, classifies curves using a two-stage machine learning classifier (99.2% held-out accuracy), applies Haldane substrate-inhibition ordinary differential equations to estimate inhibition constants (Ki), and quantifies uncertainty through Bayesian hierarchical models with partial pooling and bootstrap confidence intervals. Validation against 480 synthetic curves yielded 98.3% classification accuracy with parameter recovery R-squared values exceeding 0.87 for growth rate and lag time. Applied to 161 averaged growth curves from three independent operators spanning 2018 to 2025 and testing seven pesticides (bifenthrin, diazinon, flupyradifurone, imidacloprid, lambda-cyhalothrin, malathion, and permethrin), the pipeline identified imidacloprid (Bayesian Ki = 4.83) and flupyradifurone (Ki = 5.87) as the most inhibitory pesticides and found Haldane kinetics preferred over Gompertz in 30 of 42 pesticide-strain combinations. Gompertz growth parameters (maximum density, growth rate, lag time) were compared across operators using five shared imidacloprid strains to assess inter-operator reproducibility across experimenters and years. This pipeline provides a reproducible, validated framework for operator-independent bioremediation screening.

---

## 1. Introduction

Pesticide contamination of agricultural soils represents a persistent and growing environmental challenge worldwide. The intensive application of organophosphates, neonicotinoids, pyrethroids, and other synthetic pesticides has led to the accumulation of residues that threaten soil microbial communities, contaminate groundwater, and disrupt ecosystem function [Tudi et al., 2021]. Conventional remediation strategies, including chemical oxidation and physical removal, are often prohibitively expensive and may introduce secondary pollutants into already degraded soils. Bioremediation, the use of microorganisms to degrade or transform contaminants into less toxic products, has emerged as a cost-effective and environmentally sustainable alternative [Parte et al., 2017]. Among bioremediation approaches, the exploitation of soil bacteria capable of metabolizing pesticide compounds as sole carbon and energy sources is particularly promising, because such organisms can be isolated directly from contaminated sites and scaled for field application.

A critical step in identifying candidate bioremediation strains is high-throughput growth screening, in which bacterial isolates are cultured in minimal media supplemented with target pesticides and monitored for growth over time. Automated microplate readers, such as the TECAN Infinite series, enable the simultaneous measurement of optical density at 600 nm (OD600) across 96-well plates at regular intervals, generating dense time-series data that capture the lag, exponential, and stationary phases of bacterial growth [Sprouffske and Wagner, 2016]. These instruments have made data acquisition routine: a single experiment can yield hundreds of growth curves spanning multiple pesticide treatments, concentrations, and biological replicates. However, the downstream analysis of these data remains a significant bottleneck. Extracting biologically meaningful growth parameters, distinguishing genuine growth from noise or instrument artifacts, and comparing results across experiments and operators requires substantial manual effort that does not scale with the throughput of modern plate readers.

The standard approach to growth curve analysis involves fitting sigmoidal models, most commonly the modified Gompertz equation [Zwietering et al., 1990] or the model of Baranyi and Roberts [1994], to estimate parameters such as maximum specific growth rate (mu), lag phase duration (lambda), and carrying capacity (A). Several software tools have been developed to automate this fitting process. Growthcurver, an R package, fits a logistic equation to optical density data and extracts key growth parameters in a high-throughput manner [Sprouffske and Wagner, 2016]. The grofit package provides a flexible framework for parametric and non-parametric growth curve fitting with model selection [Kahm et al., 2010]. GCAT offers a web-based interface for growth curve analysis with automated outlier detection [Mikolajczyk et al., 2015]. More recently, AMiGA introduced a Gaussian process framework that avoids parametric model assumptions and includes statistical testing for differential growth [Midani et al., 2021].

While these tools have advanced the field considerably, they share several important limitations when applied to pesticide bioremediation screening. First, none provides automated quality classification of growth curves. In practice, a substantial fraction of wells in any plate reader experiment produce unusable data due to contamination, bubble formation, evaporation, or failed inoculation. Identifying these problematic curves currently requires manual visual inspection, which is subjective, time-consuming, and poorly reproducible across analysts. Second, existing tools fit standard sigmoidal growth models but do not incorporate substrate inhibition kinetics. In bioremediation contexts, many pesticides inhibit bacterial growth at elevated concentrations, producing concentration-response relationships that are not captured by simple Monod kinetics. The Haldane substrate inhibition model [Andrews, 1968], which includes an inhibition constant (Ki) governing growth suppression at high substrate concentrations, is the appropriate framework for these systems, yet no current growth curve tool integrates this modeling step. Third, parameter uncertainty is typically reported as point estimates with standard errors from nonlinear regression, rather than as full posterior distributions that would enable rigorous propagation of uncertainty into downstream analyses. Fourth, when growth data span multiple operators, instruments, or years, there is no automated mechanism for assessing inter-operator reproducibility or harmonizing results across datasets. Fifth, raw plate reader data frequently contain post-stationary-phase artifacts, including optical density decline from cell lysis or pigment changes, that corrupt model fits if the full time series is used. Intelligent truncation of these artifacts prior to fitting is essential for reliable parameter estimation, yet no existing tool offers automated truncation with systematic comparison of alternative strategies.

In this study, we developed an automated computational pipeline that addresses each of these gaps. The pipeline comprises eleven sequential steps that transform raw TECAN plate reader exports into publication-ready results without manual intervention. Raw optical density data are first preprocessed with baseline correction and blank subtraction, then classified as suitable or unsuitable for analysis using a two-stage machine learning classifier. Curves classified as suitable are fit to the modified Gompertz model [Zwietering et al., 1990] using nonlinear least squares regression, with seven truncation strategies compared and the adaptive R-squared method selected as optimal. Growth parameters are then used to fit a Haldane substrate inhibition ordinary differential equation model [Andrews, 1968] to estimate the inhibition constant Ki for each pesticide-strain combination. Bayesian inference via Markov chain Monte Carlo sampling provides full posterior distributions for all kinetic parameters. The pipeline concludes with inter-operator reproducibility analysis and automated generation of summary tables and figures. We validated the classification module on 480 synthetic growth curves and then applied the complete pipeline to 161 real growth curves generated by three independent operators over a seven-year period (2018-2025), encompassing seven pesticides from three chemical classes. All code is implemented in Python and is freely available as open-source software.

---

## 2. Materials and Methods

### 2.1 Experimental Design and Data Collection

Soil bacterial isolates were screened for their capacity to degrade seven pesticides representing three chemical classes: bifenthrin, permethrin, and lambda-cyhalothrin (pyrethroids), malathion and diazinon (organophosphates), and imidacloprid and flupyradifurone (neonicotinoids). Growth kinetics were measured using a TECAN Infinite plate reader monitoring optical density at 600 nm (OD600) in 96-well microplates at 15-minute measurement intervals over the duration of each experiment.

Four media conditions were employed for each bacterial isolate: (i) Luria-Bertani broth (LB; positive growth control), (ii) sterile water (H2O; negative control), (iii) pesticide as sole carbon source in minimal medium (pesticide-only), and (iv) pesticide supplemented with LB (pesticide+LB). Each condition was measured in triplicate wells. Raw OD600 readings were blank-subtracted against uninoculated media controls and averaged across the three replicate wells to yield a single growth curve per strain-condition combination.

Experiments were organized into six groups collected by three independent operators over a multi-year period. Operator 1 collected Groups 1 through 4 in 2025, with Group 1 containing bifenthrin, flupyradifurone, and lambda-cyhalothrin treatments; Group 2 containing malathion and lambda-cyhalothrin; Group 3 containing imidacloprid and lambda-cyhalothrin; and Group 4 containing permethrin and lambda-cyhalothrin. Group 5 was collected by Operator 2 (Walton) in 2024 with imidacloprid, malathion, and diazinon treatments, and Group 6 was collected by Operator 3 (Dominique) in 2024 with imidacloprid treatments. The complete dataset comprised 161 averaged growth curves across all groups and conditions.

### 2.2 Pipeline Architecture

All analyses were performed using a custom automated pipeline implemented in Python (version 3.x) and orchestrated by a master script that executed 11 sequential steps (Steps 0 through 10): (0) preprocessing of raw plate reader data; (1) training of a machine learning classifier; (2) modified Gompertz model fitting with adaptive truncation and ML classification; (3) combination of per-group results; (4) Haldane substrate inhibition modeling; (5) advanced fitting including Gaussian process truncation, bootstrap confidence intervals, and Bayesian hierarchical models; (6) statistical analysis with publication-quality figure generation; (7) truncation method comparison and rescue of initially rejected curves; (8) export for collaboration; (9) synthetic data validation; and (10) inter-operator reproducibility analysis (Figure 1). The pipeline was configuration-driven, with all thresholds, hyperparameters, and file paths specified in a central YAML configuration file, ensuring full reproducibility across runs.

### 2.3 Data Preprocessing

Raw TECAN plate reader output consisted of comma-separated files containing time stamps (in seconds) and OD600 absorbance values for each well position. A key file mapped well positions to bacterial strain identifiers, media conditions, and operator metadata. Preprocessing converted time from seconds to hours and applied blank subtraction according to:

OD_blanked(t) = max(0, OD_strain(t) - mean(OD_blank(t)))

where mean(OD_blank(t)) denotes the mean OD600 of uninoculated blank wells at time t. Negative values resulting from subtraction were clipped to zero. Triplicate wells for each strain-condition combination were then averaged to produce a single representative growth curve.

### 2.4 Modified Gompertz Growth Model

Bacterial growth curves were fitted to the modified Gompertz model [Zwietering et al., 1990]:

y(t) = A * exp(-exp((mu_m * e / A) * (lambda - t) + 1))

where A is the maximum population density (carrying capacity, OD600 units), mu_m is the maximum specific growth rate (OD600/h), lambda is the lag phase duration (h), and e is Euler's number. Parameters were estimated via nonlinear least-squares regression using the Levenberg-Marquardt algorithm as implemented in scipy.optimize.curve_fit with a maximum of 5000 iterations. Initial parameter estimates were derived from the data: A_0 = max(OD), mu_0 = max(dOD/dt), and lambda_0 = t at the maximum growth rate. Parameter bounds were set as follows: A in [0.01, 3 * max(OD)], mu_m in [0.001, 10 * max(dOD/dt) + 1], and lambda in [0, t_max].

Goodness of fit was assessed using the coefficient of determination (R-squared), root mean squared error (RMSE), and mean absolute error (MAE). Parameter standard errors were obtained from the square roots of the diagonal elements of the covariance matrix returned by the optimizer.

### 2.5 Adaptive Data Truncation

The modified Gompertz model assumes monotonically increasing growth approaching an asymptote, an assumption violated when cultures exhibit post-stationary phase decline. To address this, data were truncated prior to fitting. Seven truncation methods were implemented and compared:

**Method 1: First Local Maximum.** The OD600 time series was smoothed with a rolling mean (window = 5 time points), and the first local maximum was identified as the truncation point, with 3 additional buffer points retained.

**Method 2: Stationary Phase Detection.** The time derivative of the smoothed OD600 was computed, and the onset of stationary phase was defined as the first time point after the inflection point where the derivative fell below a threshold.

**Method 3: Adaptive R-squared Optimization.** The truncation endpoint was treated as a hyperparameter. A coarse scan of 50 candidate endpoints was performed, with Gompertz fits evaluated using 20-fold Monte Carlo cross-validation. The top candidates were refined using a fine-grained search. The endpoint maximizing cross-validated R-squared was selected.

**Method 4: Gaussian Process Derivative.** A Gaussian process with a composite RBF + WhiteKernel was fitted to the growth curve. The truncation point was defined as the first zero-crossing of the GP derivative after the maximum growth rate.

**Method 5: Changepoint Detection.** Bayesian changepoint detection was performed using the PELT algorithm with a radial basis function cost model (ruptures library).

**Method 6: Ensemble Consensus.** A weighted median of truncation points from Methods 1 through 5, with outlier detection via median absolute deviation.

**Method 7: Minimum Growth Segment.** A minimum of 20 data points after truncation was enforced.

The adaptive R-squared method (Method 3) was used as the primary truncation strategy, achieving the highest success rate (98.5%) and highest mean R-squared (0.978) in systematic benchmarking.

### 2.6 Two-Stage Machine Learning Classifier

**Stage 1: Pre-Fit Gate.** A HistGradientBoostingClassifier was applied to 10 features extracted from the raw OD600 signal prior to model fitting: delta-OD, maximum OD, SNR, monotone fraction, baseline standard deviation, baseline mean, number of data points, time span, control indicator, and pesticide concentration. Curves with P(good) <= 0.05 were rejected before fitting.

**Stage 2: Post-Fit Classifier.** A second HistGradientBoostingClassifier was applied after Gompertz fitting using 24 features encompassing signal characteristics, fit quality metrics, Gompertz parameters, and derived ratios. Curves were classified as GOOD if P(good) >= 0.7, BAD if P(good) <= 0.3, and BORDERLINE otherwise.

**Training Data.** The classifier was trained on 565 labeled curves (480 synthetic + 85 manually audited real curves), split 70/30 with stratified sampling.

**Performance.** The post-fit classifier achieved 99.2% accuracy, 99.6% precision, and 99.3% recall on the held-out test set. Because the classifier was trained on features extracted by the same pipeline, these metrics reflect learned pipeline decision boundaries rather than fully independent classification. The synthetic validation (Section 2.10) provided a complementary ground-truth-based assessment.

### 2.7 Haldane Substrate Inhibition Model

For strains grown in pesticide-containing media, a Haldane substrate inhibition model was fitted:

dX/dt = mu(S) * X * (1 - X/X_max)
dS/dt = -(1/q) * mu(S) * X
mu(S) = mu_max * S / (Ks + S + S^2/Ki)

where X is biomass (OD600), S is substrate concentration, X_max is carrying capacity, q is yield coefficient, mu_max is maximum growth rate, Ks is half-saturation constant, and Ki is the substrate inhibition constant. Lower Ki values indicate greater sensitivity to substrate inhibition.

The ODE system was integrated numerically using scipy.integrate.solve_ivp and parameters estimated via L-BFGS-B optimization. Model selection between Haldane (k=7 parameters) and Gompertz (k=3) used corrected Akaike Information Criterion (AICc), with Haldane preferred when delta-AICc > 2 [Burnham and Anderson, 2002].

### 2.8 Bayesian Hierarchical Analysis

**Bayesian Gompertz Model.** A three-level hierarchy was specified using PyMC (version 5): population hyperpriors, pesticide-group parameters, and strain-level parameters with non-centered parameterization. Posterior samples were drawn using NUTS with 4 chains, 2000 draws, and 1000 tuning steps.

**Bayesian Haldane Model.** A hierarchical model for Ki was specified across pesticide groups with log-normal population priors and partial pooling. Because the ODE-based likelihood precluded gradient computation, posterior samples were drawn using DEMetropolisZ with 4 chains, 5000 draws, and 3000 tuning steps.

**Convergence.** R-hat < 1.05 and ESS_bulk > 400 were required for all parameters.

### 2.9 Bootstrap Confidence Intervals

Nonparametric 95% confidence intervals were computed via residual-resampling bootstrap with 1000 replicates per curve, using the percentile method.

### 2.10 Synthetic Data Validation

480 synthetic curves were generated across 32 scenarios using five growth models (Gompertz, Baranyi-Roberts, logistic, Richards, death-phase) and four noise models (Gaussian, OD-dependent, instrument, RMSE-targeted). The pipeline was run with identical settings to real data analysis, and accuracy, precision, recall, F1, and parameter recovery R-squared were computed against known ground truth.

### 2.11 Inter-Operator Comparison

Five imidacloprid strains shared across two or three operators were compared using Welch's t-test (two operators) or one-way ANOVA (three operators). Coefficient of variation (CV) was computed as the summary reproducibility metric.

### 2.12 Software and Reproducibility

All analyses were implemented in Python using NumPy, pandas, SciPy, scikit-learn, PyMC, ArviZ, ruptures, and matplotlib. All random number generators used fixed seeds. The complete pipeline source code and configuration files are available as open-source software.

---

## 3. Results

### 3.1 Dataset Overview

The complete dataset comprised 161 averaged growth curves distributed across six experimental groups and generated by three independent operators (Table 1). Seven pesticides spanning three chemical classes were represented. The pipeline classified 66 of 161 curves (41%) as GOOD and 95 (59%) as BAD. Classification outcomes varied across groups: Group 1 yielded 12 of 24 GOOD (50%), Group 2 yielded 11 of 24 (46%), Group 3 yielded 12 of 24 (50%), Group 4 yielded 11 of 20 (55%), Group 5 yielded 16 of 59 (27%), and Group 6 yielded 4 of 10 (40%) (Figure 2). The lower GOOD rate in Group 5 reflected the larger number of replicates and the presence of strains with marginal growth. The high overall BAD rate was consistent with expectations for bioremediation screening, in which many isolates fail to grow on pesticide substrates.

### 3.2 Machine Learning Classification Performance

The post-fit classifier achieved 99.2% accuracy, 99.6% precision, 99.3% recall, and F1 = 0.994 on held-out test data (Table 2). The pre-fit gate achieved 94.2% accuracy, 99.6% precision, 91.9% recall, and F1 = 0.956. The near-perfect precision of both stages (99.6%) indicated that false positive classifications were exceedingly rare.

### 3.3 Synthetic Validation

The classifier achieved 98.3% accuracy on 480 synthetic curves, with 337 true positives, 135 true negatives, zero false positives, and eight false negatives (Figure 3, Table 4). The eight failures were distributed across six scenarios: long lag (2), long experiment (2), slow growth (1), short lag (1), pesticide-LB typical (1), and Baranyi-generated (1).

Parameter recovery analysis showed strong recovery of growth rate (mu R-squared = 0.871) and lag time (lambda R-squared = 0.970), but poor recovery of carrying capacity (A R-squared = 0.419). The degraded A performance was attributable to adaptive truncation removing the stationary-phase plateau.

### 3.4 Truncation Method Comparison

Seven truncation strategies were compared on the 66 GOOD curves (Table 3, Figure 4). Adaptive R-squared achieved the highest success rate (98.5%) with mean R-squared = 0.978. GP derivative produced the highest excellent-fit rate but was vulnerable to outliers (min R-squared = -4.58). Changepoint detection exhibited 100% failure. Ensemble consensus was robust but computationally expensive without improving upon adaptive R-squared.

### 3.5 Haldane versus Gompertz Model Selection

The Haldane model was preferred in 30 of 42 pesticide-strain combinations (71.4%) by AICc (Table 5, Figure 5). Model preference varied by pesticide: permethrin 4/4 (100%), malathion 5/5 (100%), lambda-cyhalothrin 4/5 (80%), diazinon 5/6 (83%), imidacloprid 11/17 (65%), flupyradifurone 1/1 (100%), bifenthrin 0/4 (0%). Bifenthrin was the sole pesticide for which Gompertz was universally preferred.

### 3.6 Bayesian Inhibition Constant Estimates

Bayesian hierarchical modeling estimated Ki for each pesticide (Figure 6). Imidacloprid exhibited the lowest median Ki (4.83; HDI: 0.05-16.13; P(Ki<5) = 50.5%), identifying it as the most inhibitory pesticide. Flupyradifurone was second (Ki = 5.87; HDI: 3.03-13.82; P(Ki<5) = 34.5%). The organophosphates showed intermediate values: diazinon Ki = 9.32, malathion Ki = 11.84. The pyrethroids showed higher values: lambda-cyhalothrin Ki = 12.28, permethrin Ki = 14.79. Bifenthrin exhibited the highest Ki (17.55; HDI: 4.62-255.47) with the widest uncertainty, consistent with the absence of Haldane preference.

### 3.7 Inter-Operator Reproducibility

Five shared imidacloprid strains were compared across operators (Figure 7, Table 6). None of the 12 statistical tests reached significance at p < 0.05, supporting operator-independent pipeline performance. However, the mean CV was 61.8% (range: 7.0%-100.0%), indicating substantial quantitative variability attributable to biological and experimental heterogeneity across years.

### 3.8 Bootstrap Confidence Intervals

Of the 66 GOOD curves, 65 received bootstrap P(good) = 1.0. One curve (H2O-CYPERM2) received P(good) = 0.0, correctly identifying a water control with low-amplitude OD drift as non-growth.

---

## 4. Discussion

This study presents an end-to-end computational pipeline for automated bacterial growth curve analysis in pesticide bioremediation screening. By integrating machine learning classification, modified Gompertz fitting with adaptive truncation, Haldane substrate inhibition modeling, and Bayesian uncertainty quantification into a single workflow, the pipeline addresses critical gaps left by existing tools. Applied to 161 growth curves spanning seven pesticides, three operators, and a seven-year experimental window, the pipeline identified imidacloprid and flupyradifurone as the most inhibitory compounds and demonstrated that substrate inhibition kinetics dominate these bioremediation systems.

### Substrate inhibition as the dominant growth mechanism

The prevalence of Haldane kinetics across pesticide-strain combinations (30/42, 71%) indicates that substrate inhibition is a pervasive feature of bacterial growth on pesticide substrates. This finding has direct implications for bioremediation practice: strains that degrade a given pesticide may nonetheless exhibit reduced growth at the concentrations encountered in contaminated soils, and screening protocols that test only a single high concentration risk mischaracterizing strain performance.

The ranking of pesticides by Ki values aligns with expectations from their physicochemical properties. Imidacloprid, a neonicotinoid with high water solubility (~610 mg/L), yielded the lowest Ki (4.83), indicating the strongest inhibition. Its high aqueous bioavailability likely ensures that the effective concentration experienced by cells approaches the nominal concentration. Flupyradifurone exhibited the second-lowest Ki (5.87). In contrast, bifenthrin, a pyrethroid with extremely low water solubility (~0.1 ug/L), produced the highest Ki (17.55), suggesting that hydrophobic partitioning substantially reduces bioavailability.

It is essential to emphasize that the Ki values reported here are relative inhibition indices in arbitrary units, not absolute inhibition constants. Because substrate concentration was not measured directly, these Ki values should be interpreted as comparative rankings within this experimental system.

### Truncation strategy and its consequences

Adaptive R-squared truncation provided the most reliable balance between data retention and fit quality. Post-stationary decline violates Gompertz monotonicity assumptions and, if included, distorts all parameter estimates. GP-based truncation was elegant but sensitive to outliers. Changepoint detection failed entirely, as growth transitions are gradual rather than sharp.

A direct consequence of truncation is degraded carrying capacity (A) estimation. While growth rate and lag time were recovered with R-squared > 0.87, A recovery was substantially poorer (R-squared = 0.419). This is inherent to the approach and researchers should treat A estimates with caution.

### Machine learning classification: strengths and circularity

The two-stage classifier achieved 99.2% accuracy on held-out data and 98.3% on synthetic curves. However, an inherent circularity exists: the classifier was trained on features extracted by the same pipeline. The high accuracy therefore reflects consistency with rule-based labels, not necessarily independent biological assessment.

The synthetic validation provides a more defensible assessment because ground-truth labels were assigned during scenario generation. The eight false negatives, concentrated among long-lag and slow-growth edge cases, expose genuine limitations. Addressing this will require expanding training data and incorporating independently labeled external datasets.

### Inter-operator reproducibility

No statistically significant differences were detected across operators for any shared strain (0/12 tests, p < 0.05), supporting operator-independent pipeline performance. However, the high CV (61.8%) reflects genuine biological and experimental heterogeneity across years. The lack of significance should not be interpreted as evidence of negligible variability but rather as a reflection of limited statistical power with small sample sizes.

### Comparison with existing tools

The pipeline occupies a distinct niche. Growthcurver provides logistic fitting without classification or kinetics. grofit adds model selection but lacks quality control. AMiGA introduces GP methods but no inhibition modeling. GCAT offers basic Gompertz fitting through a web interface. None integrates classification, kinetic modeling, uncertainty quantification, and reproducibility assessment into a single workflow.

### Limitations

Several limitations should be acknowledged: (1) single pesticide concentration per treatment, precluding true dose-response curves; (2) Ki in arbitrary units; (3) modest strain diversity; (4) ML classifier circularity; (5) Bayesian Gompertz convergence issues (2420 divergences); (6) bootstrap residual exchangeability assumption; (7) TECAN-only input format.

### Future directions

Dose-response experiments will enable absolute Ki estimation. External ML validation using independently labeled datasets will address circularity. Adaptation to other plate reader formats (Bioscreen C, BioTek Epoch) will broaden applicability. A web interface is under development for non-computational users. Integration with 16S rRNA sequencing would link growth phenotypes to taxonomic identity.

In summary, this pipeline provides a validated, reproducible framework for automated growth curve analysis in pesticide bioremediation screening. The identification of substrate inhibition as the dominant growth mechanism underscores the importance of incorporating inhibition kinetics into screening protocols, a capability that no prior tool has offered.

---

## References

1. Andrews JF. A mathematical model for continuous culture of microorganisms utilizing inhibitory substrates. Biotechnol Bioeng. 1968;10(5):707-723.
2. Baranyi J, Roberts TA. A dynamic approach to predicting bacterial growth in food. Int J Food Microbiol. 1994;23(3-4):277-294.
3. Burnham KP, Anderson DR. Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach. 2nd ed. New York: Springer; 2002.
4. Kahm M, Hasenbrink G, Lichtenberg-Frate H, Ludwig J, Kschischo M. grofit: Fitting biological growth curves with R. J Stat Softw. 2010;33(7):1-21.
5. Midani FS, Collins J, Britton RA. AMiGA: Software for automated analysis of microbial growth assays. mSystems. 2021;6(4):e00508-21.
6. Mikolajczyk K, Kaczmarek M, Krawczyk B. GCAT: Galaxy-based analysis toolkit for growth curve analysis. Bioinformatics. 2015;31(10):1683-1685.
7. Parte SG, Mohekar AD, Kharat AS. Microbial degradation of pesticide: A review. Afr J Microbiol Res. 2017;11(24):992-1012.
8. Sprouffske K, Wagner A. Growthcurver: An R package for obtaining interpretable metrics from microbial growth curves. BMC Bioinformatics. 2016;17:172.
9. Tudi M, Raza A, Breez NT, et al. Agriculture development, pesticide application and its impact on the environment. Int J Environ Res Public Health. 2021;18(3):1112.
10. Zwietering MH, Jongenburger I, Rombouts FM, van 't Riet K. Modeling of the bacterial growth curve. Appl Environ Microbiol. 1990;56(6):1875-1881.

---

## Figure Legends

**Figure 1.** Pipeline schematic showing the eleven sequential analysis steps. Steps are color-coded by function: preprocessing (blue), model fitting (green), machine learning (orange), statistical analysis (purple), and validation (red). Key dataset metrics are annotated.

**Figure 2.** Dataset overview. (A) Classification counts by experimental group showing GOOD and BAD curves. (B) Boxplots of Gompertz growth parameters (mu, lambda, A) by treatment type (LB control, H2O control, pesticide+LB, pesticide-only). (C) Scatter plot of lag time versus growth rate for all GOOD-classified curves, colored by pesticide.

**Figure 3.** Synthetic validation results. (A) Confusion matrix for the ML classifier on 480 synthetic curves (337 TP, 135 TN, 0 FP, 8 FN). (B) Parameter recovery scatter plots showing fitted versus true values for growth rate (mu, R-squared = 0.871), lag time (lambda, R-squared = 0.970), and carrying capacity (A, R-squared = 0.419).

**Figure 4.** Truncation method comparison. (A) Boxplot of R-squared values across seven truncation methods. (B) Method ranking by mean performance metrics. (C) Representative curve showing all truncation methods overlaid on a single growth curve (BifenthrinANDLB-BIF2).

**Figure 5.** Haldane versus Gompertz model comparison. (A) Overview of model preference across all 42 pesticide-strain combinations. (B-D) Representative individual strain comparisons showing (B) Haldane preferred, (C) Gompertz preferred, and (D) Haldane preferred with intermediate fit quality.

**Figure 6.** Bayesian inhibition constant (Ki) forest plot. Posterior median and 95% highest density interval for Ki by pesticide, estimated from the hierarchical Haldane model with partial pooling. Lower Ki indicates stronger substrate inhibition.

**Figure 7.** Inter-operator reproducibility. (A) Boxplots of Gompertz growth parameters by operator for shared imidacloprid strains. (B) Heatmap of coefficient of variation (%) by parameter and strain across operators.

**Figure 8.** Representative growth curve analyses. (A) High-quality GOOD fit showing clear sigmoidal growth. (B) BAD classification showing flat, noisy signal with no detectable growth. (C) Pesticide+nutrient curve from an external operator (Walton). (D) LB control curve from Group 3.

---

## Tables

**Table 1.** Experimental design summary showing group, operator, year, pesticides tested, total curves, and classification outcomes.

**Table 2.** Machine learning classifier performance metrics for pre-fit gate and post-fit classifier stages.

**Table 3.** Truncation method comparison: success rate, mean R-squared, and failure characteristics for seven methods.

**Table 4.** Synthetic validation results: accuracy, precision, recall, F1, and parameter recovery R-squared values.

**Table 5.** Haldane model selection by pesticide: number of strains, Haldane preference rate, median Ki, and mean R-squared for both models.

**Table 6.** Inter-operator comparison: ANOVA/Welch test results and coefficient of variation for shared imidacloprid strains.

---

## Supporting Information

**S1 Table.** Bootstrap confidence intervals for all 66 GOOD-classified curves.

**S2 Table.** Complete Haldane versus Gompertz comparison for all 42 pesticide-strain combinations.

**S3 Table.** Full pipeline results for all 161 curves with extracted features and classifications.

**S4 Table.** Bayesian Haldane posterior summaries for all parameters by pesticide group.

**S5 Table.** Synthetic validation per-scenario breakdown (32 scenarios).

**S6 Table.** ML classifier feature lists with descriptions for pre-fit (10 features) and post-fit (24 features) stages.

**S1 Figure.** Bayesian Gompertz MCMC trace diagnostics.

**S2 Figure.** Bayesian Gompertz posterior predictive checks for five representative strains.

**S3 Figure.** Bootstrap classification summary.

**S4 Figure.** Truncation method pairwise scatter matrix.

**S5 Figure.** Per-strain truncation performance heatmap.

**S6 Figure.** All 35 Haldane versus Gompertz individual strain comparison plots.

**S7 Figure.** Per-group classification summaries for Groups 1-6.
