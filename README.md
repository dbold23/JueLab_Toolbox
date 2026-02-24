# Jue Lab Toolbox

A shared collection of scripts, pipelines, and utilities developed for the Jue Lab. This repository serves as a reusable toolbox across multiple projects, organized by tool or workflow.

## Repository Structure

```
JueLab_toolbox/
├── TECAN_growth_curves/       # TECAN plate reader growth curve pipeline
│   ├── scripts/               # Core pipeline scripts
│   │   ├── 01_growth_curve_analysis.py    # Gompertz fitting & classification
│   │   ├── 05_haldane_analysis.py         # Haldane inhibition analysis
│   │   └── 06_advanced_fitting.py         # GP, Bayesian, Bootstrap
│   ├── data/raw/              # Raw TECAN plate reader output files (Groups 1-4)
│   ├── results/tables/        # Summary statistics, fitted parameters, plots
│   │   ├── Haldane_Analysis/              # Ki values, AIC comparisons
│   │   └── Advanced_Analysis/             # GP, Bayesian, Bootstrap results
│   ├── synthetic_data/        # 480-curve validation suite
│   └── tests/                 # pytest test suite
└── README.md
```

## Projects

### TECAN Growth Curves

End-to-end pipeline for TECAN plate reader growth curve analysis:

1. **Parse** — Read raw TECAN output files, preprocess 96-well plate data
2. **Process** — Clean, normalize, blank-subtract, and average triplicates
3. **Analyze** — Fit Gompertz growth models, classify curves (GOOD/BAD)
4. **Haldane** — Fit mechanistic substrate inhibition ODE to pesticide strains
5. **Advanced Stats** — GP truncation, Bayesian hierarchical models (NUTS/DEMetropolisZ), bootstrap CIs, WAIC/LOO model comparison
6. **Plot** — Generate publication-ready figures and diagnostic plots
7. **Validate** — 480 synthetic curves + interactive manual audit tool

## Getting Started

```bash
git clone <https://github.com/dbold23/JueLab_Toolbox>
cd JueLab_toolbox
```

## Contributing

Lab members can add new tools by creating a new top-level directory (e.g., `flow_cytometry/`, `microscopy/`) following the same general structure.
