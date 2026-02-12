# Jue Lab Toolbox

A shared collection of scripts, pipelines, and utilities developed for the Jue Lab. This repository serves as a reusable toolbox across multiple projects, organized by tool or workflow.

## Repository Structure

```
JueLab_toolbox/
├── TECAN_growth_curves/       # TECAN plate reader growth curve pipeline
│   ├── scripts/               # Core pipeline scripts (parsing, analysis, plotting)
│   ├── data/
│   │   ├── raw/               # Raw TECAN plate reader output files
│   │   └── processed/         # Cleaned and formatted data
│   ├── results/
│   │   ├── figures/           # Generated plots and growth curves
│   │   └── tables/            # Summary statistics and fitted parameters
│   ├── synthetic_data/        # Synthetic/test data generation
│   └── tests/                 # Unit tests
└── README.md
```

## Projects

### TECAN Growth Curves

End-to-end pipeline for TECAN plate reader growth curve analysis + view all synthetic data/test set:

1. **Parse** - Read raw TECAN output files
2. **Process** - Clean, normalize, and format plate reader data
3. **Analyze** - Fit growth curves and extract parameters
4. **Plot** - Generate publication-ready figures
5. **Synthetic data** - Generate test datasets for pipeline validation

## Getting Started

```bash
git clone <repo-url>
cd JueLab_toolbox
```

## Contributing

Lab members can add new tools by creating a new top-level directory (e.g., `flow_cytometry/`, `microscopy/`) following the same general structure.
