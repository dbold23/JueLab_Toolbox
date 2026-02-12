# Growth Curve Analysis Pipeline

**BIO380SP25 - Pesticide Bioremediating Bacteria Research Project**

This folder is **self-contained** with all scripts and raw data needed to reproduce the analysis.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
./run_all_groups.sh

# Results will be in OUTPUT/
```

---

## Folder Structure

```
ANALYSIS_SCRIPTS/
├── 01_growth_curve_analysis.py      # Main analysis script
├── 02_preprocess_raw_plate_data.py  # Raw data preprocessor
├── run_all_groups.sh                # Batch processing script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── DATA/                            # Raw data (included)
│   ├── Group1/
│   │   ├── GrowthRate_ Group1_Values.csv  # Raw 96-well format
│   │   ├── GROUP1_KEY_V2.csv              # Well mapping
│   │   └── Group_1_DATA/                  # Generated after preprocessing
│   ├── Group 2/
│   │   └── Group_2_DATA/                  # Pre-processed CSVs
│   ├── Group 3/
│   │   └── Group_3_DATA/
│   └── Group 4/
│       └── Group4_DATA/
└── OUTPUT/                          # Generated results
    ├── Group1_Final/
    ├── Group2_Final/
    ├── Group3_Final/
    ├── Group4_Final/
    └── all_groups_results.csv
```

---

## Scripts

### 1. `01_growth_curve_analysis.py` - Main Analysis Script

Processes growth curve data to:
- **Classify** curves as GOOD (suitable for fitting) or BAD (insufficient growth)
- **Truncate** curves at optimal point (removes death/decline phase)
- **Fit** Gompertz growth model to extract parameters
- **Generate** visualizations and summary statistics

#### Gompertz Model
```
y(t) = A * exp(-exp((mu * e / A) * (lambda - t) + 1))

Parameters:
  A      = Maximum OD600 (asymptotic value)
  mu     = Maximum specific growth rate (OD/hour)
  lambda = Lag phase duration (hours)
```

#### Usage
```bash
# Basic usage (fit-based classification + adaptive truncation)
python 01_growth_curve_analysis.py DATA/<GROUP_FOLDER> -o OUTPUT/<GROUP>_Final --adaptive

# Example
python 01_growth_curve_analysis.py "DATA/Group 2/Group_2_DATA" -o "OUTPUT/Group2_Final" --adaptive
```

#### Key Options
| Option | Description | Default |
|--------|-------------|---------|
| `--adaptive` | Optimize truncation point for best R² | Off |
| `--od-based` | Use OD-threshold classification instead of fit-based | Fit-based |
| `--min-r2` | Minimum R² for good classification | 0.95 |
| `--max-param-error` | Maximum parameter error % | 20.0 |
| `--global-max` | Truncate at global max instead of first peak | First peak |
| `-q, --quiet` | Suppress progress output | Verbose |

#### Output Files
```
OUTPUT/Group_Final/
├── processing_results.csv      # All results in tabular format
├── classification_summary.png  # Overview visualization
└── plots/                      # Individual strain plots
    ├── STRAIN_truncation_analysis.png  # Good curves
    └── STRAIN_BAD_fit_analysis.png     # Bad curves
```

---

### 2. `02_preprocess_raw_plate_data.py` - Raw Data Preprocessor

Converts raw 96-well plate reader format (like Group 1) to the analysis-ready format.

**Input format:**
- Raw data: Columns A1-H12 (96-well positions), Time in seconds
- Key file: Maps wells to Media and Strain

**Output format:**
- Separate CSV per media type
- Columns: `TIME[H]`, `{MEDIA}_{STRAIN}_blanked`
- Triplicates averaged, blanks subtracted

#### Usage
```bash
python 02_preprocess_raw_plate_data.py <DATA_FILE> <KEY_FILE> -o <OUTPUT_DIR>

# Example for Group 1
python 02_preprocess_raw_plate_data.py \
    "DATA/Group1/GrowthRate_ Group1_Values.csv" \
    "DATA/Group1/GROUP1_KEY_V2.csv" \
    -o "DATA/Group1/Group_1_DATA"
```

---

### 3. `run_all_groups.sh` - Batch Processing Script

Runs the complete analysis pipeline for all groups automatically.

```bash
./run_all_groups.sh
```

This will:
1. Preprocess Group 1 raw data
2. Run Gompertz analysis on all groups
3. Combine results into `OUTPUT/all_groups_results.csv`

---

## Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
scipy>=1.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

Or create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Classification Methods

### Fit-Based Classification (Default, Recommended)
Attempts to fit Gompertz model first, then classifies based on fit quality:
- **R² >= 0.95** (good fit to model)
- **Parameter error < 20%** for A and mu

This is more scientifically rigorous because it judges whether the curve actually follows Gompertz growth, regardless of absolute OD magnitude.

### OD-Threshold Classification (Legacy)
Classifies based on raw OD measurements:
- Delta OD >= 0.3
- Max OD >= 0.5
- Signal-to-noise ratio >= 5.0

Use `--od-based` flag to enable this mode.

---

## Truncation Methods

### Adaptive Truncation (Recommended)
Searches for the truncation point that maximizes R². Most scientifically rigorous.
```bash
python 01_growth_curve_analysis.py DATA_DIR -o OUTPUT_DIR --adaptive
```

### First Peak Truncation (Default without --adaptive)
Truncates at first local maximum after growth phase. Good for standard sigmoid curves.

### Global Max Truncation
Truncates at the absolute maximum OD600.
```bash
python 01_growth_curve_analysis.py DATA_DIR -o OUTPUT_DIR --global-max
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| `strain` | Strain identifier |
| `is_good` | Boolean classification result |
| `classification_reason` | Why curve was classified as good/bad |
| `delta_od` | OD change (max - initial) |
| `max_od` | Maximum OD600 reached |
| `r_squared` | Gompertz fit R² value |
| `gompertz_a` | Fitted A parameter (max OD) |
| `gompertz_a_err` | Standard error of A |
| `gompertz_mu` | Fitted mu parameter (growth rate, OD/hour) |
| `gompertz_mu_err` | Standard error of mu |
| `gompertz_lambda` | Fitted lambda parameter (lag phase, hours) |
| `gompertz_lambda_err` | Standard error of lambda |
| `fit_rmse` | Root mean square error of fit |
| `truncation_time` | Time (hours) where data was truncated |
| `points_used` | Number of data points used in fit |

---

## Expected Results

When you run the pipeline, you should see approximately:

| Group | Total | Good | Bad |
|-------|-------|------|-----|
| Group 1 | 24 | 12 | 12 |
| Group 2 | 24 | 16 | 8 |
| Group 3 | 24 | 17 | 7 |
| Group 4 | 20 | 12 | 8 |
| **Total** | **92** | **57** | **35** |

**Key findings:**
- LB controls show excellent growth (R² > 0.97)
- Pesticide+LB conditions show strong growth for most strains
- Pesticide-only conditions (without LB) show no/poor growth (expected)
- H2O controls show no growth (expected)

---

## Troubleshooting

### "python3: command not found"
Edit `run_all_groups.sh` and change `PYTHON="python3"` to your Python path (e.g., `PYTHON="python"` or `PYTHON="/usr/bin/python3"`).

### Permission denied on run_all_groups.sh
```bash
chmod +x run_all_groups.sh
```

### Missing dependencies
```bash
pip install pandas numpy matplotlib scipy
```

---

## Contact

For questions about this analysis pipeline, contact the BIO380SP25 research team.
