# Synthetic Bacterial Growth Curve Data Generator

A comprehensive tool for generating synthetic bacterial growth curve data to validate and stress-test the TECAN growth curve analysis pipeline.

## Overview

This module generates realistic synthetic OD600 growth curves covering every possible scenario (good growth, bad curves, edge cases) to ensure the analysis pipeline handles all situations correctly.

### Key Features

- **Multiple Growth Models**: Gompertz, Baranyi, Logistic, and Richards models
- **Death Phase Modeling**: Exponential decay after stationary phase
- **Realistic Noise**: OD-dependent, instrument-based, and RMSE-targeted noise
- **Comprehensive Scenarios**: ~400 curves covering 27 different scenarios
- **Pipeline Validation**: Automated comparison of expected vs actual classifications
- **TECAN-Compatible Output**: Generates CSV files matching the expected pipeline format

## Installation

```bash
cd TECAN/SYNTHETIC_DATA_GENERATOR
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib
- pyyaml

## Quick Start

### 1. Generate Comprehensive Test Data

```bash
# Generate ~400 curves covering all scenarios
python scripts/generate_comprehensive.py --output-dir output/full_test

# Quick test with fewer curves
python scripts/generate_comprehensive.py --n-per-scenario 5 --output-dir output/quick_test
```

### 2. Visualize Real Data Parameters

```bash
# Extract parameter distributions from real experimental data
python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv
```

### 3. Generate Custom Synthetic Data

```bash
# Generate specific number of curves
python scripts/generate_synthetic.py --n-curves 100 --model gompertz --output-dir output/custom

# Generate with specific noise level
python scripts/generate_synthetic.py --n-curves 50 --noise-level high
```

### 4. Validate Pipeline

```bash
# Run validation after processing synthetic data through pipeline
python scripts/validate_pipeline.py \
    --synthetic-dir output/full_test/DATA \
    --ground-truth output/full_test/ground_truth.csv \
    --report-dir validation_report
```

## Project Structure

```
SYNTHETIC_DATA_GENERATOR/
├── README.md
├── requirements.txt
├── config/
│   ├── default_config.yaml        # Default generation settings
│   └── extracted_params.yaml      # Parameters extracted from real data
├── src/
│   ├── __init__.py
│   ├── growth_models.py           # Gompertz, Baranyi, Logistic, Richards
│   ├── noise_models.py            # Noise generation algorithms
│   ├── parameter_extractor.py     # Extract params from real data
│   ├── data_generator.py          # Main generator class
│   ├── curve_scenarios.py         # Scenario definitions
│   └── output_formatter.py        # TECAN CSV format output
├── visualization/
│   ├── __init__.py
│   ├── triplicate_visualizer.py   # Real data visualizations
│   └── synthetic_plotter.py       # Synthetic data plots
├── validation/
│   ├── __init__.py
│   ├── pipeline_validator.py      # Validation framework
│   └── comparison_report.py       # Report generation
├── scripts/
│   ├── extract_parameters.py      # CLI: Extract from real data
│   ├── generate_synthetic.py      # CLI: Generate synthetic data
│   ├── generate_comprehensive.py  # CLI: Full test suite
│   ├── visualize_real_data.py     # CLI: Visualize real data
│   └── validate_pipeline.py       # CLI: Validate pipeline
└── output/                        # Generated data output
```

## Growth Models

### 1. Modified Gompertz Model (Default)

The primary model used by the analysis pipeline:

```
y(t) = A × exp(-exp((μ×e/A) × (λ - t) + 1))
```

Where:
- **A**: Maximum OD600 increase (carrying capacity)
- **μ**: Maximum specific growth rate (1/hour)
- **λ**: Lag phase duration (hours)

### 2. Baranyi-Roberts Model

Better fits for stationary phase transitions:

```
y(t) = y_max + ln((-1 + exp(μ_max × A(t))) / (1 + (exp(μ_max × A(t)) - 1) / exp(y_max - y0)))
```

### 3. Logistic Model

Symmetric S-curve growth:

```
y(t) = A / (1 + exp(-k × (t - t_mid)))
```

### 4. Richards Model

Generalized sigmoid with flexible asymmetry:

```
y(t) = A / (1 + ν × exp(-k × (t - t_mid)))^(1/ν)
```

### 5. Death Phase Extension

Exponential decay after stationary phase:

```
y_death(t) = y_plateau × exp(-k_death × (t - t_death))
```

## Scenario Coverage

### Good Growth Scenarios (Expected: GOOD classification)

| Scenario | A Range | μ Range | λ Range | Description |
|----------|---------|---------|---------|-------------|
| standard | 0.8-1.5 | 0.1-0.3 | 2-8 | Typical bacterial growth |
| high_A | 1.5-2.0 | 0.15-0.35 | 1-5 | High maximum density |
| low_A | 0.1-0.3 | 0.05-0.15 | 3-10 | Low but valid growth |
| fast_growth | 1.0-1.5 | 0.4-0.75 | 0.5-2 | Rapid exponential phase |
| slow_growth | 0.8-1.2 | 0.02-0.08 | 5-15 | Slow growth rate |
| short_lag | 1.0-1.5 | 0.15-0.25 | 0.1-1 | Minimal lag phase |
| long_lag | 1.0-1.4 | 0.1-0.2 | 20-50 | Extended lag phase |

### Bad Curve Scenarios (Expected: BAD classification)

| Scenario | Description |
|----------|-------------|
| flat_no_growth | H2O control, OD ≈ 0 |
| contamination | High initial OD (>0.15) |
| minimal_growth | ΔOD < 0.05 |
| high_noise | Valid growth but R² < 0.90 |
| erratic | Random non-biological pattern |
| declining | Only decreasing OD |
| sparse_data | Insufficient data points |
| truncated_early | Experiment ended too soon |

### Edge Case Scenarios (Challenging cases)

| Scenario | Description |
|----------|-------------|
| borderline_r2 | R² near 0.95 threshold |
| truncation_challenge | Multiple local maxima (diauxic-like) |
| death_phase | Clear decline after plateau |
| diauxic_growth | Two distinct growth phases |
| very_short/long_experiment | 20h or 120h duration |
| baranyi/logistic/richards_generated | Cross-model fitting tests |
| high_initial_od_valid | Higher OD but valid growth |
| parameter_boundary | Parameters at detection limits |
| realistic_noise_patterns | Instrument noise simulation |

## Noise Models

### 1. Gaussian Noise
Constant standard deviation across all measurements.

### 2. OD-Dependent Noise
Heteroscedastic noise that scales with OD:
```
σ(OD) = σ_base + σ_scale × OD
```

### 3. Instrument Noise
Realistic simulation including:
- Baseline noise
- Proportional noise
- Thermal drift
- Random outliers

### 4. RMSE-Based Noise
Generate noise to achieve a target R² value (most realistic).

## Output Format

Generated CSV files match the TECAN pipeline input format:

```csv
TIME[H],TREATMENT_CURVE0001_blanked,TREATMENT_CURVE0002_blanked,...
0.0,0.052,0.048,...
0.5,0.055,0.051,...
```

Ground truth CSV contains expected results:

```csv
curve_id,strain_name,scenario,expected_class,true_A,true_mu,true_lambda,...
1,CURVE0001,standard,GOOD,1.23,0.18,4.5,...
```

## Validation Metrics

The validation framework computes:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Parameter Recovery**: RMSE, R², and bias for A, μ, λ
- **Scenario-Level Analysis**: Performance breakdown by scenario type

## Usage Examples

### Extract Parameters from Real Data

```python
from src.parameter_extractor import ParameterExtractor

extractor = ParameterExtractor('../OUTPUT/all_groups_results.csv')
params = extractor.extract_parameter_stats()
extractor.save_to_yaml('config/extracted_params.yaml')
```

### Generate Custom Curves

```python
from src.data_generator import SyntheticGrowthCurveGenerator

generator = SyntheticGrowthCurveGenerator(seed=42)

# Generate single curve
curve = generator.generate_single_curve(
    A=1.2, mu=0.2, lambda_=5.0,
    model_type='gompertz',
    noise_level='low'
)

# Generate from scenario
curves_df = generator.generate_from_scenario('standard', n_curves=20)
```

### Export to TECAN Format

```python
from src.output_formatter import TECANFormatWriter

writer = TECANFormatWriter('output/my_test')
result = writer.export_for_analysis_pipeline(curves_df)
```

### Validate Pipeline

```python
from validation.pipeline_validator import PipelineValidator

validator = PipelineValidator(ground_truth_df, pipeline_results_df)
report = validator.validate_classification()
print(f"Accuracy: {report['accuracy']:.1%}")
```

## Configuration

Edit `config/default_config.yaml` to customize:

- Default parameter ranges
- Noise levels and types
- Scenario definitions
- Output format settings

## Research Background

### Why Multiple Models?

- **Baranyi model** often outperforms Gompertz for bacterial growth, especially for stationary phase transitions
- **Death phase** causes observable OD decline in long experiments due to cell lysis
- Testing the pipeline with different model-generated data ensures robustness

### Real Data Statistics (from 57 good curves)

| Parameter | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| R² | 0.95 | 0.9998 | 0.985 | 0.015 |
| A | 0.025 | 1.87 | 0.72 | 0.45 |
| μ | 0.001 | 0.75 | 0.18 | 0.15 |
| λ | 0.035 | 76.7 | 8.2 | 12.5 |
| RMSE | 0.0006 | 0.087 | 0.015 | 0.018 |

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the `SYNTHETIC_DATA_GENERATOR` directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Path issues**: Use absolute paths or run scripts from the project root

### Getting Help

Check the analysis pipeline documentation in `TECAN/ANALYSIS_SCRIPTS/README.md` for pipeline-specific questions.

## License

Part of the TECAN Growth Curve Analysis project.
