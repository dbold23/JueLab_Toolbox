# 2. Materials and Methods

## 2.1 Experimental Design and Data Collection

Soil bacterial isolates were screened for their capacity to degrade seven pesticides representing four chemical classes: bifenthrin and permethrin (pyrethroids), lambda-cyhalothrin (pyrethroid), malathion and diazinon (organophosphates), and imidacloprid and flupyradifurone (neonicotinoids). Growth kinetics were measured using a TECAN Infinite plate reader monitoring optical density at 600 nm (OD600) in 96-well microplates at 15-minute measurement intervals over the duration of each experiment.

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

**Method 3: Adaptive $R^2$ Optimization (Monte Carlo Cross-Validation).** This method treated the truncation endpoint as a hyperparameter to be optimized. A coarse scan of 50 candidate endpoints was performed from the inflection point to the end of the time series. For each candidate, Gompertz fits were evaluated using 20-fold Monte Carlo cross-validation (MCCV) with an 80/20 train-test split. The top 3 candidates by mean cross-validated $R^2$ were then refined using a fine-grained search within $\pm$10 points. The endpoint maximizing cross-validated $R^2$ was selected. Biological landmarks (first local maximum, stationary phase onset) were included as mandatory candidates to ensure the search space encompassed physiologically meaningful truncation points.

**Method 4: Gaussian Process Derivative.** A Gaussian process (GP) with a composite kernel (ConstantKernel $\times$ RBF + WhiteKernel) was fitted to the growth curve data using scikit-learn. The kernel initial length scale was set to 2.0 h with bounds [0.5, 20.0], and the white noise level was initialized at 0.01. The GP derivative was computed analytically from the posterior mean, and the truncation point was defined as the first zero-crossing of the derivative after the maximum growth rate, corresponding to the onset of stationary phase.

**Method 5: Changepoint Detection.** Bayesian changepoint detection was performed using the PELT algorithm with a radial basis function cost model as implemented in the `ruptures` library. The penalty parameter was set to $3 \log(n)$, where $n$ is the number of data points. Truncation was placed at the first detected changepoint where the subsequent segment exhibited near-zero slope and OD values near the maximum, consistent with entry into stationary phase.

**Method 6: Ensemble Consensus.** A weighted median of the truncation points from Methods 1 through 5 was computed. Results were flagged for manual review if methods disagreed by more than 4.0 h or if fewer than 3 methods returned valid truncation points. Outlier detection was performed using the median absolute deviation (MAD) with a threshold multiplier of 3.0.

**Method 7: Minimum Growth Segment.** A minimum segment of 20 data points after truncation was enforced. If truncation would result in fewer points, the truncation point was extended to ensure sufficient data for reliable parameter estimation.

The adaptive $R^2$ method (Method 3) was used as the primary truncation strategy throughout the pipeline, as it achieved the highest success rate (98.5% of curves successfully fitted) and highest mean $R^2$ (0.978) in systematic benchmarking.

### 2.5.1 Per-Parameter Truncation Refinement

The whole-curve truncation strategy (Section 2.5) selects a single endpoint that optimizes overall fit quality. However, different Gompertz parameters are best constrained by different regions of the growth curve: the lag time $\lambda$ is most informative before and around the inflection point, the maximum growth rate $\mu_m$ in the exponential phase, and the carrying capacity $A$ as the curve approaches its asymptote. A single truncation cannot simultaneously minimize the uncertainty of all three parameters.

To address this, we implemented a per-parameter truncation refinement applied after the primary truncated fit. For each candidate endpoint along the curve (scanned from the inflection point onward), a Gompertz fit was performed and the relative standard error of each parameter was recorded as $\text{rel\_err}_\theta = \sigma_\theta / |\theta|$, where $\sigma_\theta$ was obtained from the covariance matrix. The refined estimate for parameter $\theta$ was taken from the endpoint minimizing $\text{rel\_err}_\theta$, independently for $A$, $\mu_m$, and $\lambda$. Each refined parameter was classified as **identifiable** ($\text{rel\_err} < 0.10$), **weakly identifiable** ($0.10 \leq \text{rel\_err} < 0.50$), or **unidentifiable** ($\text{rel\_err} \geq 0.50$).

A final per-parameter value was selected as the refined estimate when the corresponding identifiability flag was *identifiable*, otherwise falling back to the whole-curve value. A \texttt{usable\_for} field was derived per curve, listing the parameters whose identifiability flag was at least *weakly identifiable*. Downstream analyses consumed refined values gated by this field: the Haldane substrate-inhibition regression (Section 2.7) was restricted to strains with identifiable growth rate, and treatment-wise tests on $\lambda$ (Section 2.6) were restricted to strains with identifiable lag time.

Validation against 345 synthetic growth curves with known ground-truth parameters (Section 2.10) demonstrated that per-parameter refinement reduced recovery root mean squared error by factors of $6.6\times$ for $A$, $13.8\times$ for $\mu_m$, and $3.4\times$ for $\lambda$ (with identifiability gating) relative to the whole-curve fit.

## 2.6 Two-Stage Machine Learning Classifier

Growth curves were classified as exhibiting good growth (suitable for Gompertz fitting) or bad growth (insufficient signal) using a two-stage machine learning classifier.

**Stage 1: Pre-Fit Gate.** A histogram-based gradient boosting classifier (`HistGradientBoostingClassifier`, scikit-learn) was applied to 10 features extracted from the raw OD600 signal prior to model fitting: change in OD ($\Delta$OD), maximum OD, signal-to-noise ratio (SNR), fraction of monotonically increasing consecutive points, baseline standard deviation, baseline mean, number of data points, time span, a binary indicator for control media conditions, and pesticide concentration. Curves with predicted probability of good growth $P(\text{good}) \leq 0.05$ were rejected without proceeding to the computationally expensive fitting step, providing a conservative pre-filter.

**Stage 2: Post-Fit Classifier.** A second `HistGradientBoostingClassifier` was applied after Gompertz fitting using 24 features encompassing both signal characteristics and fit quality metrics: Gompertz $R^2$, RMSE, MAE, relative parameter errors for $A$ and $\mu_m$, SNR, $\Delta$OD, maximum OD, number of data points used, Gompertz parameter estimates ($A$, $\mu_m$, $\lambda$), truncation time, baseline standard deviation, fraction of monotonically increasing points, lag-1 residual autocorrelation, confidence interval lower bound for $\Delta$OD, and four derived ratio features ($\mu_m/A$, $\lambda$/truncation time, RMSE/$\Delta$OD, and the product of relative parameter errors). Curves were classified as GOOD if $P(\text{good}) \geq 0.7$, BAD if $P(\text{good}) \leq 0.3$, and BORDERLINE otherwise.

**Training Data.** The classifier was trained on 565 labeled curves comprising 480 synthetically generated curves (see Section 2.10) and 85 manually audited real curves from the experimental dataset, split into 70% training and 30% test sets with stratified sampling to maintain class balance (389 GOOD, 176 BAD). When ML models were unavailable, the pipeline fell back to rule-based classification gates using configurable thresholds for $R^2 \geq 0.95$, maximum relative parameter error $\leq 20\%$, $\text{SNR} \geq 5.0$, and minimum $\Delta\text{OD} \geq 0.15$.

**Performance.** The post-fit classifier achieved 99.2% accuracy, 99.6% precision, and 99.3% recall on the held-out test set. It should be noted that because the classifier was trained on features extracted by the same pipeline, these metrics reflect the model's ability to learn pipeline-internal decision boundaries rather than fully independent classification performance. The synthetic validation described in Section 2.10 provided a complementary, ground-truth-based assessment of classification accuracy.

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

**Bayesian Gompertz Model.** A three-level hierarchy was specified: population-level hyperpriors, pesticide-group-level parameters, and strain-level parameters. Population-level priors were: $\mu_A \sim \text{Normal}(1.3, 0.5)$ for maximum OD, $\mu_{\mu_m} \sim \text{Normal}(0.25, 0.15)$ for growth rate, and $\mu_\lambda \sim \text{Normal}(2.0, 2.0)$ for lag time. Group-level variances followed half-normal priors: $\sigma_A \sim \text{HalfNormal}(0.3)$, $\sigma_{\mu_m} \sim \text{HalfNormal}(0.1)$, $\sigma_\lambda \sim \text{HalfNormal}(2.0)$. A non-centered parameterization was employed to improve sampling efficiency: group-level parameters were expressed as $\theta_{\text{group}} = \mu_\theta + \sigma_\theta \cdot z$, where $z \sim \text{Normal}(0, 1)$. Strain-level parameters were similarly non-centered within their respective pesticide groups.

**Informative priors from per-parameter refinement.** When a strain's refined per-parameter estimate (Section 2.5.1) was classified as identifiable, its prior mean was shifted toward the refined value; strains with weakly identifiable or unidentifiable parameters retained the population-level prior for that parameter. A preliminary variant of this scheme additionally narrowed the per-strain prior spread to a function of the refined relative standard error, but this produced sampler divergences (NUTS max-tree-depth stalls) because the refined point estimate is a truncation-MLE from a single window whereas the Bayesian likelihood uses the full observation vector. The retained mean-shift-only scheme injects the truncation-informed accuracy gains into the hierarchical inference without constraining the data likelihood.

Observation noise was modeled as $\sigma_{\text{obs}} \sim \text{HalfNormal}(0.05)$, and the likelihood was $y_i \sim \text{Normal}(\hat{y}_i, \sigma_{\text{obs}})$. Posterior samples were drawn using the No-U-Turn Sampler (NUTS) with 4 chains, 2000 draws per chain, and 1000 tuning steps, with a target acceptance probability of 0.90.

**Bayesian Haldane Model.** A hierarchical model for the substrate inhibition constant $K_i$ was specified across pesticide groups. The population-level prior was $\mu_{\log K_i} \sim \text{Normal}(2.0, 1.0)$ with between-group variance $\sigma_{\log K_i} \sim \text{HalfNormal}(1.0)$. Pesticide-group-level $\log K_i$ values were non-centered, and strain-level values were drawn from group-level distributions with within-group variance $\sigma_{K_i,\text{within}} \sim \text{HalfNormal}(0.5)$. Additional priors were: $K_s \sim \text{LogNormal}(-1.0, 1.5)$, $\mu_{X_{\max}} \sim \text{Normal}(1.5, 0.3)$ with $\sigma_{X_{\max}} \sim \text{HalfNormal}(0.2)$. Initial substrate concentration was fixed at $S_0 = 1.0$. Because the likelihood involved numerical ODE integration, which precluded gradient computation, posterior samples were drawn using the DEMetropolisZ sampler (a gradient-free differential evolution Metropolis algorithm) with 4 chains, 5000 draws per chain, and 3000 tuning steps.

**Convergence Diagnostics.** Convergence was assessed using the split $\hat{R}$ statistic, with $\hat{R} < 1.05$ required for all parameters. Effective sample sizes were required to exceed 400 (bulk ESS) and 200 (tail ESS). Trace plots and rank plots were generated using ArviZ for visual inspection.

## 2.9 Bootstrap Confidence Intervals

Nonparametric confidence intervals for Gompertz parameters were estimated via residual-resampling bootstrap. For each growth curve, the Gompertz model was fitted to obtain residuals $\hat{\epsilon}_i = y_i - \hat{y}_i$. Bootstrap pseudo-observations were generated as $y_i^* = \hat{y}_i + \epsilon_j^*$, where $\epsilon_j^*$ was drawn with replacement from the residual vector. The Gompertz model was refitted to each of 1000 bootstrap replicates, yielding empirical distributions of $A$, $\mu_m$, and $\lambda$. Two-sided 95% confidence intervals were computed using the percentile method (2.5th and 97.5th percentiles of the bootstrap distributions).

## 2.10 Synthetic Data Validation

To assess pipeline accuracy independent of subjective manual classification, a comprehensive synthetic validation framework was developed. A total of 480 synthetic growth curves were generated across 32 scenarios designed to test both typical and edge-case behaviors.

**Growth Models.** Five growth model families were used to generate synthetic data: (i) the modified Gompertz model, to verify self-consistency; (ii) the Baranyi-Roberts model, to test robustness to alternative lag-phase dynamics; (iii) the logistic model; (iv) the Richards (generalized logistic) model, to test sensitivity to asymmetric growth; and (v) a Gompertz model with an appended exponential death phase, to test truncation robustness.

**Noise Models.** Four noise models were applied: (i) additive Gaussian noise with constant variance; (ii) OD-dependent (heteroscedastic) noise with variance proportional to OD600; (iii) instrument noise mimicking TECAN plate reader characteristics; and (iv) RMSE-targeted noise calibrated to match empirically observed residual magnitudes in the real dataset.

**Scenarios.** The 32 scenarios included good-growth curves with varying parameter ranges and noise levels, no-growth flat curves (negative controls), borderline curves near classification thresholds, high-noise curves, curves with pronounced death phases, and curves with truncated experimental durations. For each scenario, 15 replicate curves were generated with different random seeds.

**Validation Metrics.** The pipeline was run on the synthetic dataset with identical settings to the real data analysis (adaptive truncation, ML classification). Classification accuracy, precision, recall, and F1 score were computed by comparing pipeline GOOD/BAD labels against known ground truth. Parameter recovery was assessed by computing $R^2$ between true and estimated values of $A$, $\mu_m$, and $\lambda$ across all curves where both true and estimated parameters were available.

## 2.11 Inter-Operator Comparison

To assess the reproducibility of growth parameter estimates across independent operators, five imidacloprid-treated bacterial strains that were shared across two or three operators were identified. For each shared strain, Gompertz parameters ($A$, $\mu_m$, $\lambda$) were compared across operators. When data were available from exactly two operators, Welch's unequal-variance $t$-test was used; when three operators were represented, one-way analysis of variance (ANOVA) was performed. The coefficient of variation (CV = standard deviation / mean $\times$ 100%) was computed for each parameter per strain as a summary measure of inter-operator reproducibility.

## 2.12 Software and Reproducibility

All analyses were implemented in Python (version 3.x) using the following libraries: NumPy and pandas for numerical computation and data manipulation; SciPy for nonlinear optimization and ODE integration; scikit-learn for Gaussian process regression, gradient boosting classification, and cross-validation; PyMC (version 5) with ArviZ for Bayesian inference and convergence diagnostics; ruptures for changepoint detection; and matplotlib for visualization. The complete pipeline source code, configuration files, and synthetic data generation framework are available as open-source software. All random number generators were initialized with fixed seeds to ensure reproducibility. The pipeline was designed to be fully automated and configuration-driven, requiring no manual intervention between preprocessing and final statistical output.
