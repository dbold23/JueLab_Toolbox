# An automated computational pipeline for bacterial growth curve classification and substrate inhibition modeling in pesticide bioremediation screening

Daniel Sambold, Mary Snook, Nate Walton, Dominique Scott, and Nathaniel Jue

Department of Marine Science, California State University, Monterey Bay

---

## Abstract

Bioremediation offers a sustainable strategy for degrading pesticide residues in agricultural environments, yet screening candidate bacteria requires analyzing hundreds of growth curves from high-throughput plate reader experiments, a process that is labor-intensive and subjective when performed manually. We developed an eleven-step computational pipeline that automates bacterial growth curve analysis from TECAN plate reader data to assess whether soil bacteria can metabolize pesticides as sole carbon sources. The pipeline fits modified Gompertz growth models with adaptive truncation, classifies curves using a two-stage machine learning classifier (99.5% held-out accuracy), applies Haldane substrate-inhibition ordinary differential equations to estimate inhibition constants (Ki), and quantifies uncertainty through Bayesian hierarchical models with partial pooling and bootstrap confidence intervals. Validation against 555 independently generated synthetic curves yielded 98.7% classification accuracy with parameter recovery R-squared of 0.97 for lag time and 0.75 for growth rate. Applied to 161 averaged growth curves from three independent operators spanning 2018 to 2025 and testing seven pesticides (bifenthrin, diazinon, flupyradifurone, imidacloprid, lambda-cyhalothrin, malathion, and permethrin), the pipeline identified imidacloprid (Bayesian Ki = 4.83) and flupyradifurone (Ki = 5.87) as the most inhibitory pesticides and found Haldane kinetics preferred over Gompertz in 30 of 42 pesticide-strain combinations. Gompertz growth parameters (maximum density, growth rate, lag time) were compared across operators using five shared imidacloprid strains to assess inter-operator reproducibility across experimenters and years. This pipeline provides a reproducible, validated framework for operator-independent bioremediation screening.

---

# 1. Introduction

Pesticide contamination of agricultural soils represents a persistent and growing environmental challenge worldwide. The intensive application of organophosphates, neonicotinoids, pyrethroids, and other synthetic pesticides has led to the accumulation of residues that threaten soil microbial communities, contaminate groundwater, and disrupt ecosystem function [Tudi et al., 2021]. Conventional remediation strategies, including chemical oxidation and physical removal, are often prohibitively expensive and may introduce secondary pollutants into already degraded soils. Bioremediation, the use of microorganisms to degrade or transform contaminants into less toxic products, has emerged as a cost-effective and environmentally sustainable alternative [Parte et al., 2017]. Among bioremediation approaches, the exploitation of soil bacteria capable of metabolizing pesticide compounds as sole carbon and energy sources is particularly promising, because such organisms can be isolated directly from contaminated sites and scaled for field application.

A critical step in identifying candidate bioremediation strains is high-throughput growth screening, in which bacterial isolates are cultured in minimal media supplemented with target pesticides and monitored for growth over time. Automated microplate readers, such as the TECAN Infinite series, enable the simultaneous measurement of optical density at 600 nm (OD600) across 96-well plates at regular intervals, generating dense time-series data that capture the lag, exponential, and stationary phases of bacterial growth [Sprouffske and Wagner, 2016]. These instruments have made data acquisition routine: a single experiment can yield hundreds of growth curves spanning multiple pesticide treatments, concentrations, and biological replicates. However, the downstream analysis of these data remains a significant bottleneck. Extracting biologically meaningful growth parameters, distinguishing genuine growth from noise or instrument artifacts, and comparing results across experiments and operators requires substantial manual effort that does not scale with the throughput of modern plate readers.

The standard approach to growth curve analysis involves fitting sigmoidal models, most commonly the modified Gompertz equation [Zwietering et al., 1990] or the model of Baranyi and Roberts [1994], to estimate parameters such as maximum specific growth rate (mu), lag phase duration (lambda), and carrying capacity (A). Several software tools have been developed to automate this fitting process. Growthcurver, an R package, fits a logistic equation to optical density data and extracts key growth parameters in a high-throughput manner [Sprouffske and Wagner, 2016]. The grofit package provides a flexible framework for parametric and non-parametric growth curve fitting with model selection [Kahm et al., 2010]. GCAT offers a web-based interface for growth curve analysis with automated outlier detection [Mikolajczyk et al., 2015]. More recently, AMiGA introduced a Gaussian process framework that avoids parametric model assumptions and includes statistical testing for differential growth [Midani et al., 2021].

While these tools have advanced the field considerably, they share several important limitations when applied to pesticide bioremediation screening. First, none provides automated quality classification of growth curves. In practice, a substantial fraction of wells in any plate reader experiment produce unusable data due to contamination, bubble formation, evaporation, or failed inoculation. Identifying these problematic curves currently requires manual visual inspection, which is subjective, time-consuming, and poorly reproducible across analysts. Second, existing tools fit standard sigmoidal growth models but do not incorporate substrate inhibition kinetics. In bioremediation contexts, many pesticides inhibit bacterial growth at elevated concentrations, producing concentration-response relationships that are not captured by simple Monod kinetics. The Haldane substrate inhibition model [Andrews, 1968], which includes an inhibition constant (Ki) governing growth suppression at high substrate concentrations, is the appropriate framework for these systems, yet no current growth curve tool integrates this modeling step. Third, parameter uncertainty is typically reported as point estimates with standard errors from nonlinear regression, rather than as full posterior distributions that would enable rigorous propagation of uncertainty into downstream analyses. Fourth, when growth data span multiple operators, instruments, or years, there is no automated mechanism for assessing inter-operator reproducibility or harmonizing results across datasets. Fifth, raw plate reader data frequently contain post-stationary-phase artifacts, including optical density decline from cell lysis or pigment changes, that corrupt model fits if the full time series is used. Intelligent truncation of these artifacts prior to fitting is essential for reliable parameter estimation, yet no existing tool offers automated truncation with systematic comparison of alternative strategies.

In this study, we developed an automated computational pipeline that addresses each of these gaps. The pipeline comprises eleven sequential steps that transform raw TECAN plate reader exports into publication-ready results without manual intervention. Raw optical density data are first preprocessed with baseline correction and blank subtraction, then classified as suitable or unsuitable for analysis using a two-stage machine learning classifier that applies pre-fit quality gates followed by a post-fit random forest model trained on engineered features of the fitted curves. Curves classified as suitable are fit to the modified Gompertz model [Zwietering et al., 1990] using nonlinear least squares regression, with seven truncation strategies compared and the MCCV method selected as optimal. Growth parameters extracted from individual curves are then used to construct concentration-response profiles, which are fit to a Haldane substrate inhibition ordinary differential equation model [Andrews, 1968] to estimate the inhibition constant Ki for each pesticide-strain combination. Bayesian inference via Markov chain Monte Carlo sampling provides full posterior distributions for all kinetic parameters, enabling principled uncertainty quantification. The pipeline concludes with inter-operator reproducibility analysis and automated generation of summary tables and figures. We validated the classification module on 555 independently generated synthetic growth curves spanning a range of noise levels and growth phenotypes, achieving 98.7% accuracy against ground truth labels with no data leakage between training and validation sets. We then applied the complete pipeline to 161 real growth curves generated by three independent operators over a seven-year period (2018-2025), encompassing seven pesticides from three chemical classes: the neonicotinoids imidacloprid and flupyradifurone, the organophosphates malathion and diazinon, and the pyrethroids bifenthrin, lambda-cyhalothrin, and permethrin. The pipeline identified imidacloprid and flupyradifurone as the most inhibitory compounds, with the lowest Ki values and most pronounced growth suppression at elevated concentrations. All code is implemented in Python and is freely available as open-source software to facilitate adoption and extension by the bioremediation research community.


---

# 2. Materials and Methods

## 2.1 Experimental Design and Data Collection

Soil bacterial isolates were screened for their capacity to degrade seven pesticides representing three chemical classes: bifenthrin, permethrin, and lambda-cyhalothrin (pyrethroids), malathion and diazinon (organophosphates), and imidacloprid and flupyradifurone (neonicotinoids). Growth kinetics were measured using a TECAN Infinite plate reader monitoring optical density at 600 nm (OD600) in 96-well microplates at 15-minute measurement intervals over the duration of each experiment.

Four media conditions were employed for each bacterial isolate: (i) Luria-Bertani broth (LB; positive growth control), (ii) sterile water (H2O; negative control), (iii) pesticide as sole carbon source in minimal medium (pesticide-only), and (iv) pesticide supplemented with LB (pesticide+LB). Each condition was measured in triplicate wells. Raw OD600 readings were blank-subtracted against uninoculated media controls and averaged across the three replicate wells to yield a single growth curve per strain-condition combination.

Experiments were organized into six groups collected by three independent operators over a multi-year period. Operator 1 collected Groups 1 through 4 in 2025, with Group 1 containing bifenthrin, flupyradifurone, and lambda-cyhalothrin treatments; Group 2 containing malathion and lambda-cyhalothrin; Group 3 containing imidacloprid and lambda-cyhalothrin; and Group 4 containing permethrin and lambda-cyhalothrin. Group 5 was collected by Operator 2 (Walton) in 2024 with imidacloprid, malathion, and diazinon treatments, and Group 6 was collected by Operator 3 (Dominique) in 2024 with imidacloprid treatments. The complete dataset comprised 161 averaged growth curves across all groups and conditions.

## 2.2 Pipeline Architecture

All analyses were performed using a custom automated pipeline implemented in Python (version 3.x) and orchestrated by a master script that executed 11 sequential steps (Steps 0 through 10): (0) preprocessing of raw plate reader data; (1) training of a machine learning classifier; (2) modified Gompertz model fitting with adaptive truncation and ML classification; (3) combination of per-group results; (4) Haldane substrate inhibition modeling; (5) advanced fitting including Gaussian process truncation, bootstrap confidence intervals, and Bayesian hierarchical models; (6) statistical analysis with publication-quality figure generation; (7) truncation method comparison and rescue of initially rejected curves; (8) export for collaboration; (9) synthetic data validation; and (10) inter-operator reproducibility analysis. The pipeline was configuration-driven, with all thresholds, hyperparameters, and file paths specified in a central YAML configuration file, ensuring full reproducibility across runs. Steps were executed in dependency order, with critical steps (Gompertz fitting, result combination) enforcing early termination upon failure.

## 2.3 Data Preprocessing

Raw TECAN plate reader output consisted of comma-separated files containing time stamps (in seconds) and OD600 absorbance values for each well position. A key file mapped well positions to bacterial strain identifiers, media conditions, and operator metadata. Preprocessing converted time from seconds to hours and applied blank subtraction according to:

$$\text{OD}_{\text{blanked}}(t) = \max\!\Big(0,\;\text{OD}_{\text{strain}}(t) - \overline{\text{OD}}_{\text{blank}}(t)\Big)$$

where $\overline{\text{OD}}_{\text{blank}}(t)$ denotes the mean OD600 of uninoculated blank wells at time $t$. Negative values resulting from subtraction were clipped to zero. Triplicate wells for each strain-condition combination were then averaged to produce a single representative growth curve. Raw data files encoded in UTF-8 with byte order mark (BOM) were handled using the `utf-8-sig` encoding. For Groups 5 and 6, which were collected by external operators in different raw formats, a dedicated preprocessing script standardized the data into the same per-strain CSV format used by Groups 1 through 4.

## 2.4 Modified Gompertz Growth Model

Bacterial growth curves were fitted to the modified Gompertz model [Zwietering et al., 1990]:

$$y(t) = A \cdot \exp\!\Bigg(-\exp\!\bigg(\frac{\mu_m \cdot e}{A}(\lambda - t) + 1\bigg)\Bigg)$$

where $A$ is the maximum population density (carrying capacity, OD600 units), $\mu_m$ is the maximum specific growth rate (OD600 h$^{-1}$), $\lambda$ is the lag phase duration (h), and $e$ is Euler's number. Parameters were estimated via nonlinear least-squares regression using the Levenberg-Marquardt algorithm as implemented in `scipy.optimize.curve_fit` with a maximum of 5000 iterations. Initial parameter estimates were derived from the data: $A_0 = \max(\text{OD})$, $\mu_0 = \max(d\text{OD}/dt)$, and $\lambda_0 = t$ at the maximum growth rate. Parameter bounds were set as follows: $A \in [0.01,\; 3 \times \max(\text{OD})]$, $\mu_m \in [0.001,\; 10 \times \max(d\text{OD}/dt) + 1]$, and $\lambda \in [0,\; t_{\max}]$.

Goodness of fit was assessed using the coefficient of determination ($R^2$), root mean squared error (RMSE), and mean absolute error (MAE):

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Parameter standard errors were obtained from the square roots of the diagonal elements of the covariance matrix returned by the optimizer. Curves with fewer than 15 data points were excluded from fitting.

## 2.5 Adaptive Data Truncation

The modified Gompertz model assumes monotonically increasing growth approaching an asymptote, an assumption violated when cultures exhibit post-stationary phase decline (death phase). To address this, data were truncated prior to fitting to include only the lag, exponential, and early stationary phases. Seven truncation methods were implemented and compared:

**Method 1: First Local Maximum.** The OD600 time series was smoothed with a rolling mean (window size = 5 time points), and the first local maximum of the smoothed curve was identified as the truncation point, with 3 additional buffer points retained for fit stability.

**Method 2: Stationary Phase Detection.** The time derivative of the smoothed OD600 was computed, and the onset of stationary phase was defined as the first time point after the inflection point where the derivative fell below a threshold value. An incomplete-curve detector checked the final 15% of the time series; if the slope in this region exceeded 0.005 OD600 h$^{-1}$, the curve was flagged as still in exponential growth and no truncation was applied.

**Method 3: MCCV Truncation Optimization (Monte Carlo Cross-Validation).** This method treated the truncation endpoint as a hyperparameter to be optimized. A coarse scan of 50 candidate endpoints was performed from the inflection point to the end of the time series. For each candidate, Gompertz fits were evaluated using 20-fold Monte Carlo cross-validation (MCCV) with an 80/20 train-test split. The top 3 candidates by mean cross-validated $R^2$ were then refined using a fine-grained search within $\pm$10 points. The endpoint maximizing cross-validated $R^2$ was selected. Biological landmarks (first local maximum, stationary phase onset) were included as mandatory candidates to ensure the search space encompassed physiologically meaningful truncation points.

**Method 4: Gaussian Process Derivative.** A Gaussian process (GP) with a composite kernel (ConstantKernel $\times$ RBF + WhiteKernel) was fitted to the growth curve data using scikit-learn. The kernel initial length scale was set to 2.0 h with bounds [0.5, 20.0], and the white noise level was initialized at 0.01. The GP derivative was computed analytically from the posterior mean, and the truncation point was defined as the first zero-crossing of the derivative after the maximum growth rate, corresponding to the onset of stationary phase.

**Method 5: Changepoint Detection.** Bayesian changepoint detection was performed using the PELT algorithm with a radial basis function cost model as implemented in the `ruptures` library. The penalty parameter was set to $3 \log(n)$, where $n$ is the number of data points. Truncation was placed at the first detected changepoint where the subsequent segment exhibited near-zero slope and OD values near the maximum, consistent with entry into stationary phase.

**Method 6: Ensemble Consensus.** A weighted median of the truncation points from Methods 1 through 5 was computed. Results were flagged for manual review if methods disagreed by more than 4.0 h or if fewer than 3 methods returned valid truncation points. Outlier detection was performed using the median absolute deviation (MAD) with a threshold multiplier of 3.0.

**Method 7: Minimum Growth Segment.** A minimum segment of 20 data points after truncation was enforced. If truncation would result in fewer points, the truncation point was extended to ensure sufficient data for reliable parameter estimation.

The MCCV method (Method 3) was used as the primary truncation strategy throughout the pipeline, as it achieved the highest success rate (98.5% of curves successfully fitted) and highest mean $R^2$ (0.978) in systematic benchmarking.

## 2.6 Two-Stage Machine Learning Classifier

Growth curves were classified as exhibiting good growth (suitable for Gompertz fitting) or bad growth (insufficient signal) using a two-stage machine learning classifier.

**Stage 1: Pre-Fit Gate.** A histogram-based gradient boosting classifier (`HistGradientBoostingClassifier`, scikit-learn) was applied to 10 features extracted from the raw OD600 signal prior to model fitting: change in OD ($\Delta$OD), maximum OD, signal-to-noise ratio (SNR), fraction of monotonically increasing consecutive points, baseline standard deviation, baseline mean, number of data points, time span, a binary indicator for control media conditions, and pesticide concentration. Curves with predicted probability of good growth $P(\text{good}) \leq 0.05$ were rejected without proceeding to the computationally expensive fitting step, providing a conservative pre-filter.

**Stage 2: Post-Fit Classifier.** A second `HistGradientBoostingClassifier` was applied after Gompertz fitting using 24 features encompassing both signal characteristics and fit quality metrics: Gompertz $R^2$, RMSE, MAE, relative parameter errors for $A$ and $\mu_m$, SNR, $\Delta$OD, maximum OD, number of data points used, Gompertz parameter estimates ($A$, $\mu_m$, $\lambda$), truncation time, baseline standard deviation, fraction of monotonically increasing points, lag-1 residual autocorrelation, confidence interval lower bound for $\Delta$OD, and four derived ratio features ($\mu_m/A$, $\lambda$/truncation time, RMSE/$\Delta$OD, and the product of relative parameter errors). Curves were classified as GOOD if $P(\text{good}) \geq 0.7$, BAD if $P(\text{good}) \leq 0.3$, and BORDERLINE otherwise.

**Training Data.** The classifier was trained on 582 labeled curves comprising 480 synthetically generated curves (see Section 2.10) and 102 manually audited real curves from the experimental dataset, split into 70% training and 30% test sets with stratified sampling to maintain class balance (397 GOOD, 185 BAD). When ML models were unavailable, the pipeline fell back to rule-based classification gates using configurable thresholds for $R^2 \geq 0.95$, maximum relative parameter error $\leq 20\%$, $\text{SNR} \geq 5.0$, and minimum $\Delta\text{OD} \geq 0.15$.

**Performance.** The post-fit classifier achieved 99.5% accuracy, 100% precision, and 98.3% recall on the 175-curve held-out test set (117 TP, 56 TN, 0 FP, 2 FN). It should be noted that because the classifier was trained on features extracted by the same pipeline, these metrics reflect the model's ability to learn pipeline-internal decision boundaries rather than fully independent classification performance. Independent validation on 555 synthetic curves generated with a different random seed (Section 3.3) confirmed 98.7% accuracy with no data leakage between training and validation sets.

## 2.7 Haldane Substrate Inhibition Model

For strains grown in pesticide-containing media, a mechanistic Haldane (Andrews) substrate inhibition model was fitted to capture both growth kinetics and substrate depletion dynamics. The model consisted of a coupled ordinary differential equation (ODE) system:

$$\frac{dX}{dt} = \mu(S) \cdot X \cdot \left(1 - \frac{X}{X_{\max}}\right)$$

$$\frac{dS}{dt} = -\frac{1}{q} \cdot \mu(S) \cdot X$$

where $X$ is biomass (OD600), $S$ is substrate (pesticide) concentration, $X_{\max}$ is the carrying capacity, and $q$ is the biomass yield coefficient. The specific growth rate $\mu(S)$ followed Haldane kinetics:

$$\mu(S) = \frac{\mu_{\max} \cdot S}{K_s + S + S^2 / K_i}$$

where $\mu_{\max}$ is the maximum specific growth rate, $K_s$ is the half-saturation constant, and $K_i$ is the substrate inhibition constant. Lower $K_i$ values indicate greater sensitivity to substrate inhibition.

The ODE system was integrated numerically using `scipy.integrate.solve_ivp` and parameters were estimated via bounded optimization using the L-BFGS-B algorithm. Parameter bounds were: $\mu_{\max} \in [0.001, 5.0]$ h$^{-1}$, $K_s \in [0.001, 10.0]$, $K_i \in [0.1, 1000.0]$, $X_{\max} \in [0.05, 5.0]$ OD600, $q \in [0.001, 10.0]$, $X_0 \in [10^{-6}, 0.5]$, and $S_0 \in [0.01, 100.0]$. A default initial substrate concentration of $S_0 = 1.0$ (arbitrary units) was used when substrate concentrations were not independently measured.

**Model Selection.** The Haldane model (7 parameters) was compared against the Gompertz model (3 parameters) using the corrected Akaike Information Criterion (AICc):

$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}$$

where $\text{AIC} = n \ln(\text{RSS}/n) + 2k$, $k$ is the number of parameters, and $n$ is the number of observations. The Haldane model was preferred when $\Delta\text{AICc} > 2$ relative to the Gompertz model, following standard information-theoretic guidelines [Burnham and Anderson, 2002].

## 2.8 Bayesian Hierarchical Analysis

Bayesian hierarchical models were implemented using PyMC (version 5) to obtain posterior distributions for growth parameters with uncertainty quantification and partial pooling across strains within pesticide treatment groups.

**Bayesian Gompertz Model.** A three-level hierarchy was specified: population-level hyperpriors, pesticide-group-level parameters, and strain-level parameters. Population-level priors were: $\mu_A \sim \text{Normal}(1.3, 0.5)$ for maximum OD, $\mu_{\mu_m} \sim \text{Normal}(0.25, 0.15)$ for growth rate, and $\mu_\lambda \sim \text{Normal}(2.0, 2.0)$ for lag time. Group-level variances followed half-normal priors: $\sigma_A \sim \text{HalfNormal}(0.3)$, $\sigma_{\mu_m} \sim \text{HalfNormal}(0.1)$, $\sigma_\lambda \sim \text{HalfNormal}(2.0)$. A non-centered parameterization was employed to improve sampling efficiency: group-level parameters were expressed as $\theta_{\text{group}} = \mu_\theta + \sigma_\theta \cdot z$, where $z \sim \text{Normal}(0, 1)$. Strain-level parameters were similarly non-centered within their respective pesticide groups. Observation noise was modeled as $\sigma_{\text{obs}} \sim \text{HalfNormal}(0.05)$, and the likelihood was $y_i \sim \text{Normal}(\hat{y}_i, \sigma_{\text{obs}})$. Posterior samples were drawn using the No-U-Turn Sampler (NUTS) with 4 chains, 2000 draws per chain, and 1000 tuning steps, with a target acceptance probability of 0.90.

**Bayesian Haldane Model.** A hierarchical model for the substrate inhibition constant $K_i$ was specified across pesticide groups. The population-level prior was $\mu_{\log K_i} \sim \text{Normal}(2.0, 1.0)$ with between-group variance $\sigma_{\log K_i} \sim \text{HalfNormal}(1.0)$. Pesticide-group-level $\log K_i$ values were non-centered, and strain-level values were drawn from group-level distributions with within-group variance $\sigma_{K_i,\text{within}} \sim \text{HalfNormal}(0.5)$. Additional priors were: $K_s \sim \text{LogNormal}(-1.0, 1.5)$, $\mu_{X_{\max}} \sim \text{Normal}(1.5, 0.3)$ with $\sigma_{X_{\max}} \sim \text{HalfNormal}(0.2)$. Initial substrate concentration was fixed at $S_0 = 1.0$. Because the likelihood involved numerical ODE integration, which precluded gradient computation, posterior samples were drawn using the DEMetropolisZ sampler (a gradient-free differential evolution Metropolis algorithm) with 4 chains, 5000 draws per chain, and 3000 tuning steps.

**Convergence Diagnostics.** Convergence was assessed using the split $\hat{R}$ statistic, with $\hat{R} < 1.05$ required for all parameters. Effective sample sizes were required to exceed 400 (bulk ESS) and 200 (tail ESS). Trace plots and rank plots were generated using ArviZ for visual inspection.

## 2.9 Bootstrap Confidence Intervals

Nonparametric confidence intervals for Gompertz parameters were estimated via residual-resampling bootstrap. For each growth curve, the Gompertz model was fitted to obtain residuals $\hat{\epsilon}_i = y_i - \hat{y}_i$. Bootstrap pseudo-observations were generated as $y_i^* = \hat{y}_i + \epsilon_j^*$, where $\epsilon_j^*$ was drawn with replacement from the residual vector. The Gompertz model was refitted to each of 1000 bootstrap replicates, yielding empirical distributions of $A$, $\mu_m$, and $\lambda$. Two-sided 95% confidence intervals were computed using the percentile method (2.5th and 97.5th percentiles of the bootstrap distributions).

## 2.10 Synthetic Data Validation

To assess pipeline accuracy independent of subjective manual classification, a comprehensive synthetic validation framework was developed. A total of 555 synthetic growth curves were independently generated (random seed = 99) across 32 scenarios designed to test both typical and edge-case behaviors. Crucially, this validation set used a different random seed than the 480 training curves (seed = 42) used for classifier training, ensuring complete independence between training and validation data.

**Growth Models.** Five growth model families were used to generate synthetic data: (i) the modified Gompertz model, to verify self-consistency; (ii) the Baranyi-Roberts model, to test robustness to alternative lag-phase dynamics; (iii) the logistic model; (iv) the Richards (generalized logistic) model, to test sensitivity to asymmetric growth; and (v) a Gompertz model with an appended exponential death phase, to test truncation robustness.

**Noise Models.** Four noise models were applied: (i) additive Gaussian noise with constant variance; (ii) OD-dependent (heteroscedastic) noise with variance proportional to OD600; (iii) instrument noise mimicking TECAN plate reader characteristics; and (iv) RMSE-targeted noise calibrated to match empirically observed residual magnitudes in the real dataset.

**Scenarios.** The 32 scenarios included good-growth curves with varying parameter ranges and noise levels, no-growth flat curves (negative controls), borderline curves near classification thresholds, high-noise curves, curves with pronounced death phases, and curves with truncated experimental durations. For each scenario, 15 replicate curves were generated with different random seeds.

**Validation Metrics.** The pipeline was run on the synthetic dataset with identical settings to the real data analysis (adaptive truncation, ML classification). Classification accuracy, precision, recall, and F1 score were computed by comparing pipeline GOOD/BAD labels against known ground truth. Parameter recovery was assessed by computing $R^2$ between true and estimated values of $A$, $\mu_m$, and $\lambda$ across all curves where both true and estimated parameters were available.

## 2.11 Inter-Operator Comparison

To assess the reproducibility of growth parameter estimates across independent operators, five imidacloprid-treated bacterial strains that were shared across two or three operators were identified. For each shared strain, Gompertz parameters ($A$, $\mu_m$, $\lambda$) were compared across operators. When data were available from exactly two operators, Welch's unequal-variance $t$-test was used; when three operators were represented, one-way analysis of variance (ANOVA) was performed. The coefficient of variation (CV = standard deviation / mean $\times$ 100%) was computed for each parameter per strain as a summary measure of inter-operator reproducibility.

## 2.12 Software and Reproducibility

All analyses were implemented in Python (version 3.x) using the following libraries: NumPy and pandas for numerical computation and data manipulation; SciPy for nonlinear optimization and ODE integration; scikit-learn for Gaussian process regression, gradient boosting classification, and cross-validation; PyMC (version 5) with ArviZ for Bayesian inference and convergence diagnostics; ruptures for changepoint detection; and matplotlib for visualization. The complete pipeline source code, configuration files, and synthetic data generation framework are available as open-source software. All random number generators were initialized with fixed seeds to ensure reproducibility. The pipeline was designed to be fully automated and configuration-driven, requiring no manual intervention between preprocessing and final statistical output.


---

# Results

## 3.1 Dataset Overview

The complete dataset comprised 161 averaged growth curves distributed across six experimental groups and generated by three independent operators (Table 1). Operator 1 contributed Groups 1 through 4, collected in 2025, encompassing treatments with bifenthrin, flupyradifurone, lambda-cyhalothrin, imidacloprid, malathion, and permethrin. Walton contributed Group 5, derived from experiments conducted in 2022 with imidacloprid, malathion, and diazinon. Dominique contributed Group 6, consisting of imidacloprid treatments from 2018. Seven pesticides spanning four chemical classes were represented in total.

The pipeline classified 66 of 161 curves (41%) as GOOD, indicating suitability for downstream kinetic modeling, and 95 curves (59%) as BAD. Classification outcomes varied across groups: Group 1 yielded 12 of 24 curves classified as GOOD (50%), Group 2 yielded 11 of 24 (46%), Group 3 yielded 12 of 24 (50%), Group 4 yielded 11 of 20 (55%), Group 5 yielded 16 of 59 (27%), and Group 6 yielded 4 of 10 (40%) (Figure 2). The lower GOOD rate in Group 5 reflected the larger number of replicates in that dataset and the presence of several strains that exhibited marginal or no detectable growth under the assay conditions employed. The high overall BAD rate was consistent with expectations for bioremediation screening assays, in which many bacterial isolates fail to grow on pesticide substrates or produce optical density traces dominated by noise.

## 3.2 Machine Learning Classification Performance

The two-stage machine learning classifier was trained on a combined corpus of 582 growth curves, consisting of 480 synthetic curves with known ground-truth labels and 102 real curves that had been manually audited and annotated by domain experts. The post-fit classifier, a gradient boosting model operating on engineered features of the Gompertz-fitted curves, achieved 99.5% accuracy, 100% precision, 98.3% recall, and an F1 score of 0.992 on 175 held-out test curves (Table 2). The pre-fit quality gate, which operated on raw time-series features prior to model fitting, achieved 90.9% accuracy, 98.8% precision, 87.8% recall, and an F1 score of 0.929. The lower recall of the pre-fit gate reflected its conservative design: curves that were ambiguous prior to fitting were passed through to the post-fit stage rather than rejected, minimizing the risk of discarding genuine growth signals before the fitting step could be applied. The 100% precision of the post-fit classifier indicated that false positive classifications, in which a non-growth curve was erroneously labeled as GOOD, did not occur in the held-out test set.

## 3.3 Synthetic Validation

To evaluate classifier robustness without data leakage, 555 synthetic curves were independently generated (random seed = 99) spanning multiple biologically motivated scenarios including long lag phase, long experiment duration, slow growth rate, short lag phase, typical pesticide-LB growth, Baranyi-model-generated kinetics, high noise, and standard Gompertz growth. Critically, these validation curves were generated with a different random seed than the training data (seed = 42), ensuring the classifier had never encountered any of the validation curves during training. The classifier achieved an overall accuracy of 98.7% on this independent validation set, with 388 true positives, 160 true negatives, 5 false positives, and 2 false negatives (Figure 3). The seven misclassifications were concentrated in borderline scenarios (borderline R-squared: 4 curves) and artifact simulation scenarios (evaporation drift: 2 curves, Haldane strong inhibition: 1 curve), representing cases at the decision boundary between genuine growth and noise.

Parameter recovery analysis assessed how accurately the pipeline extracted known growth parameters from the independent synthetic validation curves. Lag phase duration (lambda) exhibited the best recovery, with R squared = 0.966, reflecting the relative ease of identifying the inflection point in sigmoidal growth data. Recovery of the maximum specific growth rate (mu) was moderate, with R squared = 0.751 between fitted and true values. Recovery of the carrying capacity parameter (A) was poor, with a negative R squared indicating that the fitted values did not track the true values. This degraded performance was attributable to the adaptive truncation procedure, which removed the stationary-phase plateau to prevent post-stationary artifacts from corrupting the growth rate estimate. Because truncation preferentially removed the data region most informative for A estimation, the trade-off between robust mu estimation and accurate A recovery was inherent to the pipeline design.

## 3.4 Truncation Method Comparison

Seven truncation strategies were compared on the 66 curves classified as GOOD to determine the optimal approach for removing post-stationary-phase artifacts prior to Gompertz model fitting (Table 3, Figure 4). The MCCV method achieved the highest success rate (98.5% of curves successfully fitted after truncation) with a mean R squared of 0.978 across fitted curves. This method iteratively shortened the time series from the endpoint and selected the truncation point that maximized the goodness-of-fit of the resulting Gompertz model.

The Gaussian process (GP) derivative method produced the highest rate of excellent fits (R squared > 0.99) among the methods tested, but was vulnerable to outlier sensitivity, as reflected by a minimum R squared of -4.58 observed on a single problematic curve. This extreme negative value indicated complete model failure when the GP derivative estimated a spurious truncation point in the presence of noisy data. The changepoint detection method exhibited a 100% failure rate across all 66 curves, as the algorithm consistently identified changepoints at positions that removed insufficient or excessive portions of the growth curve. The ensemble consensus method, which aggregated recommendations from multiple individual truncation strategies, was robust to individual method failures but incurred substantial computational overhead without improving upon the MCCV method in overall performance. On the basis of these results, the MCCV method was selected as the default truncation strategy for all subsequent analyses.

## 3.5 Haldane versus Gompertz Model Selection

To determine whether bacterial growth on pesticide substrates exhibited substrate inhibition kinetics, the Haldane substrate inhibition model and the standard Gompertz model were compared using corrected Akaike Information Criterion (AICc) for 42 pesticide-plus-LB/nutrient strain combinations derived from the 66 GOOD curves (Table 5, Figure 5). The Haldane model was preferred in 30 of 42 combinations (71.4%), indicating that substrate inhibition was a prevalent feature of bacterial growth on the pesticides examined.

Model preference varied systematically by pesticide. The Haldane model was preferred for all four permethrin combinations (4/4, 100%) and all five malathion combinations (5/5, 100%), suggesting consistent substrate inhibition across strains for these compounds. Lambda-cyhalothrin showed Haldane preference in 4 of 5 combinations (80%), and diazinon in 5 of 6 (83.3%). Imidacloprid, the most extensively sampled pesticide, exhibited Haldane preference in 11 of 17 combinations (64.7%), with the remaining six combinations better described by Gompertz kinetics. Flupyradifurone had only one evaluable combination, which favored the Haldane model (1/1, 100%). Bifenthrin was the sole pesticide for which the Gompertz model was universally preferred (0/4, 0% Haldane), suggesting that bacterial growth on bifenthrin did not exhibit detectable substrate inhibition under the experimental conditions employed.

## 3.6 Bayesian Inhibition Constant Estimates

Bayesian hierarchical modeling with partial pooling across pesticide groups was applied to estimate the substrate inhibition constant (Ki) from the Haldane model for each of the seven pesticides (Figure 6). Lower Ki values indicated stronger inhibition, meaning that growth suppression occurred at lower effective substrate concentrations.

Imidacloprid exhibited the lowest median Ki (4.83; 95% highest density interval [HDI]: 0.05 to 16.13), with a posterior probability of 50.5% that Ki was below 5, identifying it as the most inhibitory pesticide in the panel. Flupyradifurone, the other neonicotinoid tested, exhibited the second-lowest median Ki (5.87; HDI: 3.03 to 13.82), with a posterior probability of 34.5% that Ki was below 5. These results indicated that the two neonicotinoids produced the strongest substrate inhibition effects on bacterial growth.

The organophosphates diazinon and malathion exhibited intermediate Ki values. Diazinon had a median Ki of 9.32 (HDI: 0.06 to 23.05), and malathion had a median Ki of 11.84 (HDI: 1.63 to 167.51). The wide HDI for malathion reflected substantial between-strain variability in the inhibition response. The pyrethroids lambda-cyhalothrin and permethrin showed higher median Ki values of 12.28 (HDI: 1.74 to 68.35) and 14.79 (HDI: 0.89 to 56.03), respectively, indicating weaker substrate inhibition. Bifenthrin exhibited the highest median Ki (17.55; HDI: 4.62 to 255.47) and the widest posterior uncertainty of any pesticide in the panel. The large HDI for bifenthrin was consistent with the model selection results, which showed that the Haldane model was not preferred for any bifenthrin combination, suggesting that the inhibition signal for this pesticide was weak or absent.

Across all pesticides, the hierarchical partial pooling structure regularized Ki estimates toward a shared group mean, reducing the influence of individual outlier strains while preserving pesticide-specific differences. The resulting posterior distributions provided a principled framework for ranking pesticide inhibition potency and propagating parameter uncertainty into downstream risk or bioremediation feasibility assessments.

## 3.7 Inter-Operator Reproducibility

Inter-operator reproducibility was assessed for five imidacloprid strains (IMID2, IMID3, IMID5, IMID6, and IMID8) that were assayed by two or three of the three operators (Figure 7, Table 6). For each strain, the maximum specific growth rate (mu) estimated from Gompertz fits was compared across operators using one-way ANOVA or Welch's t-test, depending on the number of operators with available data.

None of the 12 statistical tests conducted reached significance at the p < 0.05 threshold, indicating that no statistically significant differences in growth rate were detected between operators for any of the shared strains. However, the mean coefficient of variation (CV) across all strain-operator comparisons was 61.8%, with individual CVs ranging from 7.0% to 100.0%. The high mean CV indicated substantial quantitative variability in growth rate estimates across operators, even in the absence of statistically significant differences. This apparent discrepancy between the statistical tests and the observed variability was attributable to the small number of replicates per operator-strain combination, which limited statistical power to detect true differences. The combination of non-significant ANOVA results and high CVs suggested that inter-operator variability was present but could not be distinguished from within-operator variability given the available sample sizes.

No LB-only control strains were shared across all three operators, precluding a direct comparison of baseline (non-pesticide) growth rates between operators. This limitation restricted the reproducibility analysis to pesticide-treated conditions and prevented normalization of growth rates to a common control baseline.

## 3.8 Bootstrap Confidence Intervals

Bootstrap confidence intervals were computed for all 66 curves classified as GOOD, using 1,000 bootstrap resamples of the residuals from the Gompertz fit to generate empirical 95% confidence intervals for each growth parameter. Of the 66 curves, 65 received a bootstrap classification probability of P(good) = 1.0, confirming high-confidence GOOD status across all resamples. A single curve (H2O-CYPERM2) received P(good) = 0.0, indicating that it was classified as BAD in every bootstrap resample. Manual inspection confirmed that this curve corresponded to a water control treated with cypermethrin, which exhibited no detectable bacterial growth and had been classified as GOOD by the primary classifier due to a low-amplitude optical density drift that mimicked a shallow growth signal. The bootstrap analysis correctly identified this curve as non-growth, demonstrating the utility of resampling-based confidence assessment as a secondary validation layer for borderline classifications.


---

# Discussion

This study presents an end-to-end computational pipeline for the automated analysis of bacterial growth curves in the context of pesticide bioremediation screening. By integrating machine learning classification, modified Gompertz fitting with adaptive truncation, Haldane substrate inhibition modeling, and Bayesian uncertainty quantification into a single workflow, the pipeline addresses critical gaps left by existing tools such as Growthcurver, grofit, AMiGA, and GCAT. Applied to 161 averaged growth curves spanning seven pesticides, three operators, and a seven-year experimental window, the pipeline identified imidacloprid and flupyradifurone as the most inhibitory compounds (Ki = 4.83 and 5.87, respectively), demonstrated that substrate inhibition kinetics dominate these bioremediation systems (Haldane preferred in 30 of 42 cases), and provided a reproducible framework for operator-independent screening. The following sections discuss the biological interpretation of these findings, the methodological choices and their consequences, the limitations of the current approach, and directions for future development.

## Substrate inhibition as the dominant growth mechanism

The prevalence of Haldane kinetics across pesticide-strain combinations (30 of 42, 71%) indicates that substrate inhibition is a pervasive feature of bacterial growth on pesticide substrates, not an occasional complication to be addressed post hoc. This finding has direct implications for bioremediation practice: strains that degrade a given pesticide may nonetheless exhibit reduced growth at the concentrations encountered in contaminated field soils, and screening protocols that test only a single high concentration risk mischaracterizing strain performance. The Haldane model, with its explicit inhibition constant Ki, provides a quantitative framework for distinguishing strains that tolerate high substrate loads from those that are effective only at lower concentrations.

The ranking of pesticides by Ki values aligns with expectations from their physicochemical properties and known toxicological profiles. Imidacloprid, a neonicotinoid with high water solubility (approximately 610 mg/L at 20 degrees C) and broad-spectrum neurotoxic activity, yielded the lowest Ki (4.83), indicating the strongest inhibition of bacterial growth. Its high aqueous bioavailability likely ensures that the effective concentration experienced by cells in the growth medium approaches the nominal concentration, maximizing inhibitory effects. Flupyradifurone, a butenolide compound that acts on the same nicotinic acetylcholine receptor target as classical neonicotinoids but with a distinct chemical scaffold, exhibited the second-lowest Ki (5.87). Although flupyradifurone is marketed as having a more favorable ecotoxicological profile than imidacloprid in pollinators, its inhibitory effect on soil bacteria appears comparably strong, a result that warrants further investigation in the context of environmental risk assessment. In contrast, bifenthrin, a pyrethroid with extremely low water solubility (approximately 0.1 micrograms per liter), produced the highest Ki (17.55), suggesting that its effective dissolved concentration in the aqueous growth medium was well below its nominal concentration. The hydrophobic partitioning of bifenthrin to plastic well surfaces and to any particulate matter in the medium may substantially reduce bioavailability, rendering apparent inhibition low regardless of intrinsic toxicity.

It is essential to emphasize that the Ki values reported here are relative inhibition indices expressed in arbitrary units, not absolute inhibition constants in the enzymological sense. Because substrate concentration was not measured directly in the growth medium and the Haldane model was applied to pesticide identity as a categorical variable rather than to a continuous concentration gradient, these Ki values should be interpreted as comparative rankings of inhibitory potency across pesticides within this experimental system. Obtaining absolute Ki values would require dose-response experiments in which bacterial growth is measured across a defined range of substrate concentrations for each pesticide, an important direction for future work.

## Truncation strategy and its consequences

The comparison of seven truncation strategies revealed that MCCV truncation provided the most reliable balance between data retention and fit quality, and this method was adopted as the pipeline default. The rationale is straightforward: bacterial growth curves frequently exhibit a post-stationary decline phase in which optical density decreases due to cell lysis, nutrient depletion, or pigment changes. This decline violates the monotonic assumptions of the Gompertz model and, if included in the fitting window, distorts estimates of all three growth parameters. MCCV truncation iteratively removes trailing data points until the coefficient of determination of the Gompertz fit exceeds a user-specified threshold, thereby identifying the longest time window consistent with good model fit.

Gaussian process-based truncation offered an elegant alternative but proved sensitive to outliers in the post-stationary phase, where individual OD spikes from bubble formation or condensation rendered GP variance estimates unreliable. Changepoint detection methods failed almost entirely on growth curves, an unsurprising outcome given that the exponential-to-stationary transition is gradual by nature rather than the sharp discontinuity these algorithms are designed to detect. The recommendation is that MCCV truncation should be the default for routine analysis, with ensemble methods reserved as a quality check for ambiguous cases.

A direct consequence of truncation is the degradation of carrying capacity (parameter A) estimation. Because truncation preferentially removes data from the stationary plateau, the region of the growth curve most informative for A, the fitted Gompertz model must extrapolate to estimate maximum optical density. Independent synthetic validation confirmed this tradeoff quantitatively: lag time (lambda) was recovered with R-squared = 0.966 and growth rate (mu) with R-squared = 0.751, while carrying capacity recovery yielded a negative R-squared, indicating that fitted A values did not meaningfully track true values. This limitation is inherent to the truncation approach and cannot be resolved without accepting the fit degradation caused by including death-phase data. In practice, researchers using this pipeline should treat A estimates with appropriate caution and rely primarily on mu and lambda for strain comparisons and bioremediation screening decisions.

## Machine learning classification: strengths and circularity

The two-stage classification system, combining pre-fit quality gates with a post-fit gradient boosting model, achieved 99.5% accuracy on held-out test data and 98.7% accuracy on 555 independently generated synthetic curves with ground-truth labels. These performance metrics are encouraging but require careful interpretation due to an inherent circularity in the training procedure. The classifier was trained on features extracted by the same pipeline that generates the rule-based classifications used as training labels. As a consequence, the classifier effectively learned the decision boundaries of the rule-based system rather than providing an independent validation of curve quality. The high held-out accuracy therefore reflects consistency with rule-based labels, not necessarily biological correctness.

The independent synthetic validation provides a more defensible assessment of classifier performance because the ground-truth labels were assigned during scenario generation rather than by the pipeline itself, and the validation curves were generated with a different random seed (seed = 99) than the training curves (seed = 42), eliminating data leakage. The 98.7% accuracy on this independent validation set, with seven misclassifications concentrated among borderline R-squared and artifact simulation scenarios, represents genuine classification ability against an external standard. These misclassified curves expose a meaningful limitation: the classifier struggles with curves at the boundary between genuine growth and noise, precisely the cases where manual inspection would also be difficult. Addressing this limitation will require expanding the training set to include more edge-case phenotypes and, ideally, incorporating independently labeled data from external laboratories where analysts apply their own quality criteria.

Despite this circularity, the ML classifier provides a continuous confidence score that enables users to prioritize borderline curves for manual review rather than making binary accept/reject decisions, offering a more nuanced assessment of data quality than any existing tool provides.

## Inter-operator reproducibility

The inter-operator analysis, comparing Gompertz growth parameters across five shared imidacloprid strains analyzed by two to three operators over a seven-year period, found no statistically significant differences for any of the 12 parameter-strain combinations tested (all p > 0.05). This result supports operator-independent pipeline performance, a property essential for multi-site bioremediation screening programs.

However, the high coefficient of variation (61.8%) reflects genuine biological and experimental heterogeneity arising from different bacterial passages, media preparations, instrument calibrations, and experimental years (2018-2025). The lack of statistical significance should not be interpreted as evidence of negligible variability but rather as a reflection of limited statistical power: with only five shared strains and two to three operators per strain, the tests had low power to detect moderate effect sizes. A definitive assessment would require ring trials across independent laboratories with standardized protocols and a larger shared strain panel.

## Comparison with existing tools

The pipeline occupies a distinct niche among growth curve analysis tools. Growthcurver [Sprouffske and Wagner, 2016] provides efficient logistic fitting in R but offers no classification, inhibition kinetics, or Bayesian uncertainty quantification. The grofit package [Kahm et al., 2010] adds Gompertz and Richards models with parametric bootstrapping but lacks quality control and substrate inhibition modeling. AMiGA [Midani et al., 2021] introduces Gaussian process methods with statistical testing for differential growth but does not model inhibition kinetics. GCAT [Mikolajczyk et al., 2015] offers basic Gompertz fitting through a web interface with no advanced capabilities. None integrates classification, kinetic modeling, uncertainty quantification, and reproducibility assessment into a single workflow. The pipeline presented here is, to the authors' knowledge, the first tool designed specifically for the bioremediation screening context, where substrate inhibition, variable data quality, and multi-operator datasets are the norm.

## Limitations

Several limitations of the current study should be acknowledged. First, each pesticide was tested at a single concentration, precluding the construction of true dose-response curves and limiting Ki estimation to relative rather than absolute values. Second, the strain diversity was modest, comprising approximately 20 unique strains across seven pesticides, and may not represent the full phenotypic range of pesticide-degrading soil bacteria. Third, the Bayesian Gompertz hierarchical model exhibited convergence issues, with 2420 divergent transitions and R-hat values exceeding 1.05 for some hyperparameters, suggesting that the model specification or sampling strategy requires refinement. Fourth, the bootstrap confidence intervals for Haldane parameters assume exchangeability of residuals, an assumption that may be violated if residual structure varies systematically across strains or pesticides. Fifth, the pipeline currently accepts only TECAN plate reader export formats, limiting its applicability to laboratories using other instruments.

## Future directions

Several extensions of this work are planned. Dose-response experiments with defined substrate concentration gradients will enable estimation of absolute Ki values and permit direct comparison with enzymological inhibition constants from the literature. External validation of the ML classifier using independently labeled datasets from collaborating laboratories will address the circularity concern and establish transferability across experimental systems. Adaptation of the data import module to accept formats from other plate reader platforms, including Bioscreen C and BioTek Epoch instruments, will broaden the user base. A web-based graphical interface is under development to make the pipeline accessible to researchers without computational expertise. Finally, integration with molecular identification methods such as 16S rRNA gene sequencing would enable direct linkage between growth phenotypes and taxonomic identity, supporting the rational selection of bioremediation consortia for field deployment.

In summary, the pipeline described here provides a validated, reproducible, and automated framework for the analysis of bacterial growth curves in the context of pesticide bioremediation screening. By combining machine learning quality control, mechanistic kinetic modeling, and Bayesian uncertainty quantification in a single workflow, it addresses the principal limitations of existing growth curve analysis tools and enables the systematic comparison of strain performance across pesticides, operators, and experimental conditions. The identification of substrate inhibition as the dominant growth mechanism for the pesticides tested underscores the importance of incorporating inhibition kinetics into bioremediation screening protocols, a capability that no prior tool has offered.

---

# References

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

# Figure Legends

**Figure 1.** Pipeline schematic showing the eleven sequential analysis steps. Steps are color-coded by function: preprocessing (blue), model fitting (green), machine learning (orange), statistical analysis (purple), and validation (red). Key dataset metrics are annotated.

**Figure 2.** Dataset overview. (A) Classification counts by experimental group showing GOOD and BAD curves. (B) Boxplots of Gompertz growth parameters ($\mu_m$, $\lambda$, $A$) by treatment type (LB control, H2O control, pesticide+LB, pesticide-only). (C) Scatter plot of lag time versus growth rate for all GOOD-classified curves, colored by pesticide.

**Figure 3.** Synthetic validation results on 555 independently generated curves (seed = 99, no data leakage). (A) Confusion matrix (388 TP, 160 TN, 5 FP, 2 FN; accuracy = 98.7%). (B) Parameter recovery scatter plots showing fitted versus true values for lag time ($\lambda$, $R^2$ = 0.966), growth rate ($\mu_m$, $R^2$ = 0.751), and carrying capacity ($A$, $R^2$ < 0).

**Figure 4.** Truncation method comparison. (A) Boxplot of $R^2$ values across seven truncation methods. (B) Method ranking by mean performance metrics. (C) Representative curve showing all truncation methods overlaid on a single growth curve (BifenthrinANDLB-BIF2).

**Figure 5.** Haldane versus Gompertz model comparison. (A) Overview of model preference across all 42 pesticide-strain combinations. (B-D) Representative individual strain comparisons showing (B) Haldane preferred, (C) Gompertz preferred, and (D) Haldane preferred with intermediate fit quality.

**Figure 6.** Bayesian inhibition constant ($K_i$) forest plot. Posterior median and 95% highest density interval for $K_i$ by pesticide, estimated from the hierarchical Haldane model with partial pooling. Lower $K_i$ indicates stronger substrate inhibition.

**Figure 7.** Inter-operator reproducibility. (A) Boxplots of Gompertz growth parameters by operator for shared imidacloprid strains. (B) Heatmap of coefficient of variation (%) by parameter and strain across operators.

**Figure 8.** Representative growth curve analyses. (A) High-quality GOOD fit showing clear sigmoidal growth (BifenthrinANDLB-BIF2). (B) BAD classification showing flat, noisy signal with no detectable growth (Bifenthrin-BIF2). (C) Pesticide+nutrient curve from an external operator (Walton, Group 5). (D) LB control curve (Group 3).

---

# Supporting Information

**S1 Table.** Bootstrap confidence intervals for all 66 GOOD-classified curves.

**S2 Table.** Complete Haldane versus Gompertz comparison for all 42 pesticide-strain combinations with AICc values, $R^2$, and preferred model designation.

**S3 Table.** Full pipeline results for all 161 curves with extracted features and classifications.

**S4 Table.** Bayesian Haldane posterior summaries for all parameters by pesticide group, including $K_i$, $K_s$, $\mu_{\max}$, and $X_{\max}$ with 95% HDI bounds.

**S5 Table.** Synthetic validation per-scenario breakdown showing accuracy, false positive count, and false negative count for each of 32 scenarios.

**S6 Table.** ML classifier feature lists with descriptions for pre-fit gate (10 features) and post-fit classifier (24 features) stages.

**S1 Figure.** Bayesian Gompertz MCMC trace diagnostics showing convergence for population-level hyperparameters.

**S2 Figure.** Bayesian Gompertz posterior predictive checks for five representative strains.

**S3 Figure.** Bootstrap classification summary showing P(good) distribution across 1000 resamples for all 66 GOOD curves.

**S4 Figure.** Truncation method pairwise scatter matrix showing agreement between all seven methods.

**S5 Figure.** Per-strain truncation performance heatmap showing $R^2$ achieved by each method for each of 66 strains.

**S6 Figure.** All 35 individual Haldane versus Gompertz strain comparison plots.

**S7 Figure.** Per-group classification summaries for Groups 1 through 6.
