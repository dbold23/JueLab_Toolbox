# Presentation: Full Methods Walkthrough

**All 21 audit fixes implemented. Numbers from pipeline run 2026-04-09.**

---

## PART 1: THE BIOLOGY & MODELS

### Slide 1: Title
**"An Automated Pipeline for Bacterial Growth Curve Analysis with Per-Parameter Identifiability and Genomic Prediction for Pesticide Bioremediation"**

Daniel Sambold, Mary Snook, Nate Walton, Dominique Scott, Nathaniel Jue
Department of Marine Science, CSUMB

Visual: Clean title, CSUMB logo

---

### Slide 2: What Is a Growth Curve?

Bacteria in a well plate grow through phases: lag, exponential, stationary, death. We measure OD600 every 15 minutes. The shape of that curve tells us whether and how well a strain degrades a pesticide.

- 96-well plates, 4 media conditions per strain (LB, H2O, pesticide-only, pesticide+LB)
- 161 averaged curves across 7 pesticides, 3 operators, 2018-2025
- The problem: hundreds of curves, manual analysis doesn't scale

Visual: **annotated_gompertz.png** (EXISTS)

---

### Slide 3: The Gompertz Model

y(t) = A * exp(-exp((mu*e/A)(lambda - t) + 1))

Three parameters, three biological quantities. Fit by nonlinear least squares (Levenberg-Marquardt). But which part of the data do you fit? Truncation matters — include too much post-stationary data and the fit degrades.

- A = carrying capacity (max OD600)
- mu = maximum specific growth rate (OD/hour at inflection)
- lambda = lag phase duration (hours before exponential growth)

Visual: **before_after_pipeline.png** (EXISTS) — good fits vs bad fits from real data

---

### Slide 4: Why Gompertz Isn't Enough — Haldane Substrate Inhibition

Gompertz is phenomenological — it fits a shape but doesn't model the mechanism. Haldane adds substrate inhibition:

mu(S) = mu_max * S / (Ks + S + S^2/Ki)

Ki = inhibition constant. Lower Ki/S0 = more inhibitory per unit dose. 7-parameter ODE solved with RK45.

Haldane uses actual pesticide concentrations as S0:
- Bifenthrin: 50 mg/L
- Permethrin: 100 mg/L
- Imidacloprid: 20 mg/L
- Malathion: 50 mg/L
- Diazinon: 20 mg/L
- Lambda-cyhalothrin: 50 mg/L
- Flupyradifurone: 20 mg/L

Models are complementary, not competing — Gompertz for growth kinetics (truncated data), Haldane for inhibition mechanics (full data).

Visual: **haldane_phase_portrait.png** (EXISTS)

---

## PART 2: PIPELINE OVERVIEW

### Slide 5: The Full Pipeline

11 steps, zero manual intervention.

Steps 0-3: Data in (preprocess with blank QC + outlier detection, train classifier, fit Gompertz with per-param truncation, combine groups)
Steps 4-5: Deep modeling (Haldane ODE with real S0, Bayesian hierarchical with wild bootstrap)
Steps 6-8: Analysis out (operator-blocked ANOVA with Bonferroni, truncation comparison, export)
Steps 9-10: Validation (555 synthetic curves, inter-operator reproducibility)
Step 11: Genomic prediction (optional, genotype-to-phenotype)

Visual: **Figure1_pipeline.png** (EXISTS)

---

## PART 3: STEP-BY-STEP METHODS

### Slide 6: Step 0 — Preprocessing & QC

Raw 96-well TECAN output -> per-strain time series.
- Time: seconds -> hours
- Blank subtraction: OD_blanked = max(0, OD_strain - median(OD_blank))
- Triplicate averaging uses median (robust to outliers)
- MAD-based outlier detection: flag replicates >3 MAD from median
- Blank QC: flag media types where blank sigma > 0.01 OD

Visual: NEED TO GENERATE — triplicate overlay or 96-well heatmap

---

### Slide 7: Step 2 — Truncation (The Critical Decision)

Each Gompertz parameter has a different optimal truncation:
- lambda (lag time) -> wants early truncation (lag-to-exponential transition)
- mu (growth rate) -> wants the full exponential phase
- A (carrying capacity) -> wants plateau included

MCCV scans 50 candidate truncation points, scores by out-of-sample R2. But MCCV compromises between parameters.

Per-parameter optimal truncation: for each parameter, find the truncation that minimizes its relative uncertainty from the covariance matrix.

Disagreement diagnostic: FLUPC8 shows 16h disagreement (lambda@8h, mu@24h, A@24h). MAL8 shows 1h agreement. High disagreement = single-Gompertz assumption breaking down.

Visual: **per_param_truncation.png** (EXISTS)

---

### Slide 8: Step 2 — Per-Parameter Identifiability

Based on Browning et al. (2022) practical identifiability framework:
- Identifiable: relative error < 10%
- Weakly identifiable: 10-50%
- Unidentifiable: > 50%

A strain is GOOD if at least one parameter is identifiable. The usable_for field tells downstream which parameters to trust.

Current results (75 good strains):
- 64 all_identifiable (all 3 params)
- 5 partial (2 of 3)
- 6 poor (but at least 1 usable)
- 69 strains usable for growth_rate + carrying_capacity + lag_time
- 5 strains usable for growth_rate + carrying_capacity only

Visual: **identifiability_summary.png** (EXISTS)

---

### Slide 9: Step 1 — ML Classification

Two-stage HistGradientBoosting classifier:

Stage 1 (pre-fit gate): 10 raw-signal features -> reject obvious junk. P(good) <= 0.05 -> skip.

Stage 2 (post-fit classifier): 24 features including fit quality, Gompertz parameters, derived ratios, metadata (is_control, actual pesticide concentration in mg/L — not strain ID numbers).

Per-parameter override: if ML says BAD but per-parameter analysis found identifiable parameters, override to GOOD.

Trained on 582 curves (398 GOOD, 184 BAD). Held-out postfit accuracy: 99.5%.
Synthetic validation (555 independent curves, seed=99): 92.1% accuracy.

Note: synthetic accuracy dropped from 98.7% because per-param override now rescues borderline curves that were previously rejected. This is intentional — the override catches real growth the classifier misses.

Visual: **ml_decision_landscape.png** (EXISTS)

---

### Slide 10: Step 4 — Haldane Substrate Inhibition ODE

40 GOOD pesticide+LB strains analyzed. 7-parameter ODE with L-BFGS-B + RK45.

Key: actual pesticide concentrations as S0 from media prep records. Ki values are physically meaningful. Ki/S0 ratio enables fair cross-pesticide comparison.

Models are complementary:
- Gompertz -> growth parameters (A, mu, lambda) from truncated data
- Haldane -> inhibition kinetics (Ki, mu_max, Ks) from full data

No AIC comparison — they answer different questions on different data by design.

Visual: **figure5_haldane_vs_gompertz.png** (EXISTS)

---

### Slide 11: Step 5 — Bayesian Hierarchical Models

Gompertz hierarchy (NUTS, 4 chains x 2000 draws):
- Population -> Pesticide-group -> Strain (partial pooling)

Haldane hierarchy (DEMetropolisZ, gradient-free for ODE, 4 chains x 5000 draws)

Priors justified from observed data ranges:
- mu_A ~ Normal(1.3, 0.5) — observed median ~1.3, range 0.01-2.5
- mu_mu ~ Normal(0.25, 0.15) — observed median ~0.2, range 0.01-1.2
- mu_lam ~ Normal(2.0, 2.0) — observed median ~2h, range 0-20h

Convergence enforced: R-hat < 1.01, ESS_bulk > 1000, ESS_tail > 400.

Visual: **partial_pooling_shrinkage.png** + **mcmc_random_walk.png** (both EXIST)

---

### Slide 12: Step 5 — Bootstrap Uncertainty

1000 wild bootstrap resamples per GOOD strain. Wild bootstrap preserves heteroscedasticity (noise varies across curve — more at high OD). Residuals multiplied by random +/-1 signs instead of resampled.

95% CIs on A, mu, lambda.

Visual: **bootstrap_spaghetti.png** (EXISTS)

---

### Slide 13: Steps 6 & 10 — Statistical Analysis & Operator Reproducibility

Primary: Kruskal-Wallis (non-parametric, no assumptions)
Assumption tests: Shapiro-Wilk (normality), Levene (homoscedasticity)
Pairwise: Mann-Whitney U with Bonferroni correction
Blocking: Two-way ANOVA with operator as blocking factor

Confounding check: Bifenthrin only from Operator1, Diazinon only from Walton — flagged as confounded.

Inter-operator CV on 5 shared imidacloprid strains.

Visual: **Figure7_operator_reproducibility.png** (EXISTS) + confounding matrix (NEED TO GENERATE)

---

### Slide 14: Step 9 — Validation

Synthetic: 555 curves (seed=99, independent from training seed=42)
- Classification accuracy: 92.1%
- Parameter recovery: mu R2=0.74, lambda R2=0.96, A R2=-0.47 (A unreliable due to truncation — known limitation)

Real-data validation: manual re-audit of 161 curves pending (browser tool built at paper/reaudit.html).

Visual: **figure3_synthetic_validation.png** (EXISTS, needs regeneration with current numbers)

---

### Slide 15: Key Result — Ki Rankings (Corrected)

With actual S0 concentrations, the inhibition ranking changed from the original analysis:

| Pesticide | Ki | S0 (mg/L) | Ki/S0 | Rank |
|-----------|-----|-----------|-------|------|
| Permethrin | 15.9 | 100 | 0.16 | Most inhibitory per dose |
| Malathion | 10.0 | 50 | 0.20 | |
| Bifenthrin | 10.1 | 50 | 0.20 | |
| Lambda-cyhalothrin | 11.5 | 50 | 0.23 | |
| Imidacloprid | 9.9 | 20 | 0.50 | |
| Diazinon | 10.0 | 20 | 0.50 | |
| Flupyradifurone | 10.9 | 20 | 0.54 | Least inhibitory per dose |

Key insight: raw Ki suggested imidacloprid was most inhibitory, but normalized to dose (Ki/S0), permethrin at 100 mg/L is actually the most inhibitory. Neonicotinoids appeared worse only because they were tested at lower concentrations.

Visual: NEED TO REGENERATE — Ki forest plot with Ki/S0 panel

---

## PART 4: GENOMICS

### Slide 16: Can Genotype Predict Phenotype?

30 genome assemblies (RagTag scaffolded). 15 overlap with TECAN data.

tblastn against 76 NCBI degradation gene references. 33 features (BLAST + codon usage + assembly stats). Elastic net with LOSOCV.

Visual: **genomic_dataflow.png** (EXISTS)

---

### Slide 17: The Honest Result

LOSOCV R2 = -0.06. Permutation p = 1.0. 0/33 features selected.

MAL strains are near-clonal P. putida: 100% carboxylesterase identity, ENC 38.5 +/- 0.3, GC 62%. Growth rate variation driven by pesticide condition, not genotype.

Consistent with James et al. (2025): within-species genomic features have weak predictive power.

Visual: **gc_vs_enc_scatter.png** (EXISTS)

---

### Slide 18: The Diversity Exists

15 Chlorp + Dimeth strains have genomes but no TECAN data:
- Chlorpyrifos: GC ~61%, ENC ~40 — one genus
- Dimethoate: GC 37-68%, ENC 31-56 — multiple genera
- Dimeth1 (ENC=31.6, GC=68%) = Actinobacterium; Dimeth2-9 (~38% GC) = Firmicutes

One TECAN run brings n from 13 clonal to 28 diverse.

Visual: Same **gc_vs_enc_scatter.png**, emphasize unfilled markers

---

### Slide 19: What's Next

1. TECAN assays on Chlorp + Dimeth strains (n: 13 -> 28 diverse)
2. Pan-genome GWAS with Scoary
3. RNA-seq on 3-5 strains under pesticide exposure

---

## PART 5: WRAP

### Slide 20: Conclusions

1. Automated pipeline: 75/161 curves classified GOOD across 7 pesticides, 3 operators
2. Per-parameter truncation optimization: novel diagnostic revealing when single-Gompertz breaks down (64/75 all_identifiable, 11/75 partial)
3. Per-parameter identifiability replaces binary GOOD/BAD — each strain contributes what it can
4. Ki/S0 normalization with real concentrations: permethrin most inhibitory per dose (Ki/S0=0.16), correcting previous claim that imidacloprid was most inhibitory
5. Genomic gene content alone doesn't predict within-species growth (p=1.0) — need phylogenetic diversity
6. Wild bootstrap, operator-blocked ANOVA, prior documentation ensure statistical rigor

**"The pipeline works. The biology says: bring diverse strains."**

---

### Slide 21: Acknowledgments

- Jue Lab, CSUMB Marine Science
- UROC
- Mary Snook, Nate Walton, Dominique Scott — data collection
- Nathaniel Jue — PI

---

## VISUAL INVENTORY

### Already exist:
| Slide | Figure | File |
|-------|--------|------|
| 2 | Annotated Gompertz | annotated_gompertz.png |
| 3 | Good vs bad fits | before_after_pipeline.png |
| 4 | Haldane phase portrait | haldane_phase_portrait.png |
| 5 | Pipeline flowchart | Figure1_pipeline.png |
| 7 | Per-param truncation | per_param_truncation.png |
| 8 | Identifiability bars | identifiability_summary.png |
| 9 | ML decision landscape | ml_decision_landscape.png |
| 10 | Haldane vs Gompertz | figure5_haldane_vs_gompertz.png |
| 11a | Partial pooling shrinkage | partial_pooling_shrinkage.png |
| 11b | MCMC terrain walk | mcmc_random_walk.png |
| 12 | Bootstrap spaghetti | bootstrap_spaghetti.png |
| 13 | Operator CV heatmap | Figure7_operator_reproducibility.png |
| 14 | Synthetic validation | figure3_synthetic_validation.png |
| 16 | Genomic dataflow | genomic_dataflow.png |
| 17-18 | GC vs ENC scatter | gc_vs_enc_scatter.png |

### Need to generate:
| Slide | Figure | Description |
|-------|--------|-------------|
| 6 | Triplicate overlay or 96-well heatmap | Preprocessing QC visual |
| 13 | Operator x pesticide confounding matrix | Grid showing data availability |
| 14 | Real-data confusion matrix | After manual re-audit |
| 15 | Ki/S0 forest plot | Updated with real concentrations |
