# Jue Lab Toolbox

A shared collection of scripts, pipelines, and utilities developed for the Jue Lab. This repository serves as a reusable toolbox across multiple projects, organized by tool or workflow.

## Repository Structure

```
JueLab_toolbox/
├── TECAN_growth_curves/       # TECAN plate reader growth curve pipeline
│   ├── scripts/               # Core pipeline scripts (11 steps)
│   │   ├── 01_growth_curve_analysis.py    # Gompertz fitting & classification
│   │   ├── 05_haldane_analysis.py         # Haldane inhibition analysis
│   │   ├── 06_advanced_fitting.py         # GP, Bayesian, Bootstrap
│   │   ├── 11_genomic_prediction.py       # Genotype-to-phenotype prediction
│   │   └── genomic_features.py            # Genomic feature extraction (BLAST/GFF)
│   ├── data/
│   │   ├── raw/               # Raw TECAN plate reader output (Groups 1-6)
│   │   └── genomic/           # Genome assemblies, BLAST results, annotations
│   ├── results/tables/        # Summary statistics, fitted parameters, plots
│   │   ├── Haldane_Analysis/              # Ki values, AIC comparisons
│   │   ├── Advanced_Analysis/             # GP, Bayesian, Bootstrap results
│   │   └── Genomic_Analysis/             # Genotype-phenotype predictions
│   ├── paper/                 # Manuscript, poster content, figures
│   ├── synthetic_data/        # 555-curve validation suite
│   └── tests/                 # pytest test suite (74 tests)
└── README.md
```

## Projects

### TECAN Growth Curves

End-to-end pipeline for TECAN plate reader growth curve analysis of pesticide-bioremediating bacteria. 161 averaged growth curves from 92 strains across 7 pesticides, 3 operators (2018-2025).

1. **Preprocess** — Blank-subtract, average triplicates, normalize time
2. **Classify** — Two-stage ML classifier (99.5% held-out accuracy, HistGradientBoosting)
3. **Fit** — Modified Gompertz growth model with adaptive truncation
4. **Haldane** — Mechanistic substrate inhibition ODE (Ki ranking across 7 pesticides)
5. **Bayesian** — Hierarchical models (NUTS/DEMetropolisZ) with partial pooling
6. **Validate** — 555 synthetic curves, independent holdout, 98.7% accuracy
7. **Genomic** — BLAST-based gene feature extraction + elastic net genotype-to-phenotype prediction (30 genome assemblies)

Key finding: Imidacloprid (Ki=4.83) and flupyradifurone (Ki=5.87) are the most inhibitory pesticides. Haldane preferred over Gompertz in 71% of strain-pesticide combinations.

## Getting Started

```bash
git clone <https://github.com/dbold23/JueLab_Toolbox>
cd JueLab_toolbox
```

## Contributing

Lab members can add new tools by creating a new top-level directory (e.g., `flow_cytometry/`, `microscopy/`) following the same general structure.
