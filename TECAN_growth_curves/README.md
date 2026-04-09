# TECAN Growth Curve Analysis Pipeline

**BIO380SP25 — Pesticide Bioremediating Bacteria**

Automated pipeline for fitting growth models to TECAN plate reader data, classifying curve quality with ML-assisted classification, quantifying pesticide substrate inhibition via Haldane kinetics, performing advanced Bayesian and GP-based statistical analysis, and comparing truncation methods to rescue misclassified strains.

## Quick Start

```bash
pip install -r scripts/requirements.txt

# Recommended: run the full pipeline
python scripts/run_full_pipeline.py              # All 11 steps
python scripts/run_full_pipeline.py --steps 2,3,4  # Specific steps
python scripts/run_full_pipeline.py --dry-run      # Preview
python scripts/run_full_pipeline.py --no-genomic   # Skip genomic prediction
```

## Pipeline Flow

The pipeline is orchestrated by `scripts/run_full_pipeline.py` and executes 11 steps in order:

```
Step  0: Preprocess raw data       (Groups 1, 5, 6 if needed)
Step  1: Train ML classifier       (auto-skips if models/ exist)
Step  2: Gompertz curve analysis    (per group, --adaptive, with ML)
Step  3: Combine group results      (-> all_groups_results.csv)
Step  4: Haldane inhibition         (ODE + AICc comparison)
Step  5: Advanced fitting           (GP, Bootstrap, Ensemble, Bayesian)
Step  6: Statistical analysis       (ANOVA, pairwise, figures)
Step  7: Truncation comparison      (5-method ranking + rescue)
Step  8: Export for collaboration    (clean CSV + methodology)
Step  9: Synthetic validation       (555 curves -> accuracy metrics)
Step 10: Inter-operator comparison  (ANOVA + CV heatmap)
Step 11: Genomic prediction         (optional, requires data/genomic/)
```

## What This Does

1. **Gompertz Growth Model** — Fits `y(t) = A * exp(-exp((mu*e/A)*(lambda-t)+1))` to each strain's OD600 curve. Extracts maximum growth (A), growth rate (mu), and lag time (lambda).

2. **Two-Stage ML Classification** — Rule-based gates (R² >= 0.95, parameter error < 20%, secondary noise/SNR gates) plus an ML classifier (HistGradientBoosting) that adds biological metadata features (is-control, concentration) for three-class output: GOOD / BORDERLINE / BAD.

3. **Haldane Feedback Inhibition** — Fits a mechanistic ODE model (`mu(S) = mu_max * S / (Ks + S + S²/Ki)`) to pesticide+LB strains. Compares with Gompertz via AIC to test whether substrate inhibition kinetics explain growth better than a purely phenomenological model.

4. **GP-Based Truncation** — Fits a Gaussian Process (RBF + WhiteKernel) to each curve and identifies growth phases from the GP derivative, replacing heuristic peak-finding.

5. **Ensemble Truncation** — Combines 5 independent truncation methods (First Peak, Stationary Phase, Adaptive R², GP Derivative, Changepoint) via weighted-median consensus.

6. **Bayesian Hierarchical Models** — Fits Gompertz (NUTS) and Haldane (DEMetropolisZ) models with partial pooling by pesticide group, producing full posterior distributions on all parameters.

7. **Bootstrap Uncertainty** — Residual resampling on NLS Gompertz fits (1000 resamples) to produce 95% CIs and P(good) classification confidence.

8. **Truncation Method Comparison** — Fits Gompertz at each of 5 truncation points per strain, ranks methods by R², and generates per-strain overlay plots. Rescues bad strains where suboptimal truncation was the cause of poor fits.

9. **Interactive Truncation Validator** — matplotlib-based tool for human review of automated truncation method selections, with resumable annotations.

10. **ML Classifier** — Two-stage HistGradientBoosting classifier trained on 480 synthetic + 85 real audited curves with proper 70/30 held-out validation. Pre-fit gate rejects obvious junk before expensive fitting; post-fit classifier uses 24 features including biological metadata (is-control, concentration).

## Key Results

### Pipeline (161 averaged curves, 6 groups, 3 operators, 2018-2025)

| Group | Operator | Pesticides | Total | Good | Bad |
|-------|----------|-----------|-------|------|-----|
| Group 1 | Operator1 | Bifenthrin, Flupyradifurone, Lambda-Cyhalothrin | 24 | 12 | 12 |
| Group 2 | Operator1 | Malathion, Lambda-Cyhalothrin | 24 | 11 | 13 |
| Group 3 | Operator1 | Imidacloprid, Lambda-Cyhalothrin | 24 | 12 | 12 |
| Group 4 | Operator1 | Permethrin, Lambda-Cyhalothrin | 20 | 9 | 11 |
| Group 5 | Walton | Imidacloprid, Malathion, Diazinon | — | — | — |
| Group 6 | Dominique | Imidacloprid | — | — | — |
| **Total** | | 7 pesticides | **161** | **66** | **95** |

### ML Classifier (held-out 30% test set)

| Metric | Value |
|--------|-------|
| Accuracy | 99.5% |
| Precision | 100% |
| Recall | 99.3% |
| F1 Score | 99.6% |

Independent synthetic validation (555 curves, separate seed, no data leakage): **98.7% accuracy**.

### Haldane Substrate Inhibition

- Haldane preferred over Gompertz in **30/42 (71%)** pesticide-strain combinations
- Most inhibitory: imidacloprid (Ki=4.83), flupyradifurone (Ki=5.87)
- Least inhibitory: bifenthrin (Ki=17.55)

### Genomic Prediction (30 genome assemblies)

Gene presence, codon usage, and assembly statistics do not predict within-species growth rate variation under pesticide stress (n=13 overlapping strains, LOSOCV R²=-0.056, permutation p=1.0). MAL strains are near-clonal *P. putida* with no discriminating genomic variation. Phenotypic variation is environment-driven. Phylogenetically diverse strains (Chlorp, Dimeth — GC 37-68%) are available and await TECAN assays.

## Folder Structure

```
TECAN_growth_curves/
├── scripts/
│   ├── 01_growth_curve_analysis.py      # Gompertz pipeline
│   ├── 02_preprocess_raw_plate_data.py  # 96-well data preprocessor
│   ├── 05_haldane_analysis.py           # Haldane inhibition analysis
│   ├── 06_advanced_fitting.py           # Advanced stats (GP, Bayesian, Bootstrap, Ensemble)
│   ├── 07_compare_truncation_methods.py # Truncation method comparison & bad strain rescue
│   ├── 09_train_classifier.py           # ML classifier training (70/30 split)
│   ├── 10_operator_comparison.py        # Inter-operator reproducibility
│   ├── 11_genomic_prediction.py         # Genotype-to-phenotype prediction
│   ├── genomic_features.py              # Genomic feature extraction (BLAST/GFF)
│   ├── ml_classifier.py                 # ML classifier runtime module
│   ├── run_full_pipeline.py             # Master orchestrator (11-step pipeline)
│   ├── config.yaml                      # All thresholds in one place
│   └── requirements.txt                 # Python deps + pymc, arviz, scikit-learn
├── data/
│   ├── raw/                             # Raw TECAN plate reader data (Groups 1-6)
│   └── genomic/                         # Genome assemblies + BLAST results
│       ├── assemblies/                  # 30 RagTag scaffolded FASTAs
│       ├── annotations/                 # Pyrodigal gene predictions (.ffn, .faa)
│       ├── blast_results/               # tblastn output per strain
│       ├── reference_databases/         # NCBI degradation gene references
│       ├── strain_mapping.csv           # Strain ID -> genome path mapping
│       ├── genomic_features.csv         # Extracted BLAST features (30 strains)
│       └── codon_usage_features.csv     # ENC, GC3s, codon entropy (30 strains)
├── results/tables/
│   ├── all_groups_results.csv           # Consolidated Gompertz results (161 curves)
│   ├── Haldane_Analysis/                # Ki values, AIC comparisons, plots
│   ├── Advanced_Analysis/               # GP, Bayesian, Bootstrap results
│   ├── Genomic_Analysis/                # Genotype-phenotype predictions + LOSOCV
│   └── Operator_Comparison/             # Inter-operator reproducibility
├── paper/                               # Manuscript, poster content, figures
├── models/                              # Trained ML classifiers (joblib, gitignored)
│   └── feature_config.json              # Feature names + training metrics (committed)
├── synthetic_data/                      # 555-curve validation suite
├── tests/                               # pytest suite (74 tests)
└── PIPELINE_METHODOLOGY.md              # Comprehensive methodology (16 sections)
```

## Detailed Documentation

See [PIPELINE_METHODOLOGY.md](PIPELINE_METHODOLOGY.md) for comprehensive methodology documentation covering all pipeline stages (including genomic prediction), mathematical formulations, algorithm details, and empirical results.

See [scripts/README.md](scripts/README.md) for full usage, all CLI options, model equations, classification methods, truncation strategies, output column descriptions, and troubleshooting.
