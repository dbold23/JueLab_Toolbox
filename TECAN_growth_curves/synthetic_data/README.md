# Synthetic Bacterial Growth Curve Data Generator

A comprehensive tool for generating synthetic bacterial growth curve data to validate and stress-test the TECAN growth curve analysis pipeline.

## Overview

This module generates realistic synthetic OD600 growth curves covering every possible scenario (good growth, bad curves, edge cases) to ensure the analysis pipeline handles all situations correctly.

### Key Features

- **Multiple Growth Models**: Gompertz, Baranyi, Logistic, and Richards models
- **Death Phase Modeling**: Exponential decay after stationary phase
- **Realistic Noise**: OD-dependent, instrument-based, and RMSE-targeted noise
- **Comprehensive Scenarios**: 64 scenarios / 1057 requested curves (1139 rows at seed 42, once artefact scenarios add their artefact-free twin), 10 good, 15 bad and 39 edge-case
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
# Generate the full suite (1139 rows at seed 42) covering all 64 scenarios
python scripts/generate_comprehensive.py --output output/full_test

# Quick test with fewer curves
python scripts/generate_comprehensive.py --n-per-scenario 5 --output output/quick_test
```

### 2. Visualize Real Data Parameters

```bash
# Extract parameter distributions from real experimental data
python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv
```

### 3. Generate Custom Synthetic Data

```bash
# Generate specific number of curves
python scripts/generate_synthetic.py --n-curves 100 --model gompertz --output output/custom

# Generate with specific noise level
python scripts/generate_synthetic.py --n-curves 50 --noise high
```

### 4. Score a Pipeline Run

`scripts/validate_pipeline.py` is gone; scoring is now `validation/benchmark.py`,
which reads the pipeline's `processing_results.csv` (and the `run_settings.json`
sidecar it writes beside it) rather than re-running the analysis itself.

```bash
# 1. run the real pipeline over the generated DATA/ directory, then
python validation/benchmark.py \
    --results-csv  <run>/processing_results.csv \
    --ground-truth output/full_test/test_data/ground_truth.csv \
    --out-dir      output/full_test/benchmark \
    --data-dir     output/full_test/test_data/DATA
```

The truncation mode comes from `run_settings.json`; pass
`--truncation-mode first_peak|adaptive` when the run wrote no sidecar. The
benchmark writes `report.json`, `per_curve.csv`, `per_scenario.csv`,
`gate_attribution.csv`, `per_scenario_recovery.csv` and the comparison figures.

### 5. Render the markdown summary

`scripts/validation_summary.py` renders `VALIDATION_SUMMARY.md` from one scored
run. It reads that run's `report.json` and nothing else, so the markdown cannot
disagree with the JSON, and it takes a single `--run-dir`:

```bash
# render the run make step-9 just produced
python scripts/validation_summary.py --run-dir output/comprehensive_test/validation/first_peak

# no arguments: the first run directory that exists, falling back to the
# tracked copy of the published run (benchmark/v2)
python scripts/validation_summary.py
```

Each truncation mode is a separate benchmark and gets its own rendering inside
its own run directory: the rendering goes beside the `report.json` it came from
(`<run-dir>/VALIDATION_SUMMARY.md`) unless `--output` says otherwise, so the
renderer never writes outside the directory it was pointed at. Rendering a
committed snapshot (`benchmark/v1`, `benchmark/v2`) writes to **stdout** rather
than dropping an extra file into a snapshot whose `MANIFEST.json` enumerates it;
pass `--output` to save it. `make validation-summary` does this for you.

The rendered markdown is a build artefact and **no copy of it is committed**:
render it when you want it, and cite `report.json` when you want to quote a
number. That is why nothing in the repository can hold a stale summary. The one
committed `VALIDATION_SUMMARY.md`, `benchmark/v1/VALIDATION_SUMMARY.md`, states
the pre-rebuild v1 numbers and is labelled as history. Never hand-edit either.

## Project Structure

```
synthetic_data/
├── README.md
├── requirements.txt
├── config/
│   ├── default_config.yaml        # Default generation settings
│   └── lab_prior.json             # Measured composition of the 92 real conditions
├── src/
│   ├── __init__.py
│   ├── growth_models.py           # Gompertz, Baranyi, Logistic, Richards
│   ├── noise_models.py            # Noise generation algorithms
│   ├── parameter_extractor.py     # Extract params from real data
│   ├── data_generator.py          # Main generator class
│   ├── curve_scenarios.py         # Scenario definitions
│   ├── curve_shapes.py            # Multi-phase / death curve shapes
│   ├── truth.py                   # Ground-truth schema and label rules
│   ├── plate_writer.py            # TECAN plate layout writer
│   └── output_formatter.py        # TECAN CSV format output
├── visualization/
│   ├── __init__.py
│   ├── triplicate_visualizer.py   # Real data visualizations
│   └── synthetic_plotter.py       # Synthetic data plots
├── validation/
│   ├── __init__.py
│   ├── matching.py                # curve_id join between truth and results
│   ├── benchmark.py               # CLI: score a pipeline run against truth
│   ├── pipeline_validator.py      # Validation framework
│   └── comparison_report.py       # Report generation
├── scripts/
│   ├── extract_parameters.py      # CLI: Extract from real data
│   ├── generate_synthetic.py      # CLI: Generate synthetic data
│   ├── generate_comprehensive.py  # CLI: Full test suite (+ README tables)
│   ├── derive_lab_prior.py        # CLI: Measure the lab prior from real plates
│   ├── visualize_real_data.py     # CLI: Visualize real data
│   └── validation_summary.py      # CLI: Render the summary from report.json
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

The three tables below are generated from the catalog itself; regenerate them
with `python scripts/generate_comprehensive.py --print-readme-tables` after any
change to `src/curve_scenarios.py`.

### Good Growth Scenarios (Expected: GOOD classification)

10 scenarios, defined in `src/curve_scenarios.py` (`GOOD_GROWTH_SCENARIOS`).

| Scenario | A Range | μ Range | λ Range | n | Description |
|----------|---------|---------|---------|---|-------------|
| standard | 0.8-1.5 | 0.1-0.3 | 2-8 | 10 | Typical good growth curve (LB media) |
| high_A | 1.5-2 | 0.15-0.35 | 1-5 | 10 | Very high final OD (dense culture) |
| low_A | 0.15-0.5 | 0.05-0.15 | 3-10 | 15 | Low but valid growth (still above thresholds) |
| fast_growth | 1-1.5 | 0.4-0.75 | 0.5-2 | 10 | Rapid exponential phase (high mu) |
| slow_growth | 0.8-1.2 | 0.02-0.08 | 5-15 | 10 | Slow exponential phase (low mu) |
| short_lag | 1-1.5 | 0.15-0.25 | 0-0.5 | 10 | Very short lag phase (pre-adapted cells) |
| long_lag | 1-1.4 | 0.1-0.2 | 20-77 | 10 | Extended lag phase (stressed cells) |
| very_clean | 1-1.5 | 0.15-0.25 | 2-6 | 10 | Very clean data with minimal noise (R² > 0.99) |
| pesticide_lb_typical | 1.2-1.7 | 0.12-0.22 | 2-5 | 10 | Typical pesticide+LB growth (slightly lower A) |
| high_baseline_good | 1-1.5 | 0.12-0.25 | 1-5 | 10 | Good growth from a high but legitimate baseline (0.08-0.15 OD) |

### Bad Curve Scenarios (Expected: BAD classification)

15 scenarios, defined in `src/curve_scenarios.py` (`BAD_CURVE_SCENARIOS`).

| Scenario | n | Description |
|----------|---|-------------|
| flat_no_growth | 45 | H2O control - no growth at all |
| minimal_growth | 40 | Barely detectable growth (delta OD < 0.1) |
| contamination | 25 | High initial OD indicating contamination (offset fraction 0.34-0.60) |
| high_noise | 15 | Valid growth pattern under the noisiest real well (label by oracle rule) |
| borderline_noise | 15 | Growth with borderline noise (label by oracle rule) |
| erratic | 25 | Non-biological erratic pattern |
| pesticide_only | 35 | Pesticide without nutrients - no growth |
| fit_failure | 35 | Curve that causes fit convergence failure |
| contamination_offset | 35 | Contamination at the measured acceptance boundary (offset fraction 0.10-0.20) |
| clipped_baseline | 30 | Dead well whose blank is over-subtracted: the opening segment is clipped at zero |
| all_zero | 18 | Identically zero trace (12/92 real conditions) |
| negative_drift | 20 | Dead well drifting DOWN (G3 H2O_IMID3: 0.0431 -> 0.0293 over 99 h) |
| replicate_outlier_high | 20 | One well grows, two are dead (G4 PERMETHRINANDLB-CISPERM1 = [0.016, 1.063, 0.010]) |
| replicate_dropout | 25 | One of three wells never inoculated: the condition mean is 2/3 of the truth |
| contaminated_blank | 30 | The blank well itself grows, so every condition on the plate sags after subtraction |

### Edge Case Scenarios (Challenging cases)

39 scenarios, defined in `src/curve_scenarios.py` (`EDGE_CASE_SCENARIOS`).
Edge cases carry their own expected class -- most of them are curves the
pipeline is expected to handle correctly and classify GOOD.  A scenario
with `label_rule=oracle` (the noise ladder, the borderline-noise pair and
the haldane trio) has its class decided per curve by whether its
parameters are recoverable from the curve that was written, so the column
below is the scenario's intent, not a guarantee about every row.

| Scenario | Expected | n | Description |
|----------|----------|---|-------------|
| borderline_r2_good | GOOD | 15 | Noise just below the R² 0.95 threshold (label by oracle rule) |
| borderline_r2_bad | BAD | 15 | Noise just above the R² 0.95 threshold (label by oracle rule) |
| truncation_challenge | GOOD | 12 | Diauxic-like second phase (sampled, not the old hard-coded one) |
| death_phase_moderate | GOOD | 12 | Moderate decline after the plateau (5-20 % of peak, anchored to t95) |
| death_phase_severe | GOOD | 12 | Severe decline (25-45 % of peak) beginning right after t95 |
| short_experiment | GOOD | 10 | Very short experiment (20 hours) |
| long_experiment | GOOD | 10 | Extended experiment (150 hours) |
| sparse_data | GOOD | 15 | Sparse time resolution (1 hour intervals) |
| dense_data | GOOD | 10 | Dense time resolution (5 min intervals) |
| baranyi_generated | GOOD | 12 | Baranyi model data fitted with Gompertz |
| logistic_generated | GOOD | 12 | Logistic model data fitted with Gompertz |
| richards_asymmetric | GOOD | 12 | Highly asymmetric Richards curve |
| outlier_contaminated | GOOD | 10 | Good curve with 3-8 real outlier spikes inside the growth window |
| drift_positive | GOOD | 10 | Good curve with positive baseline drift (0.0008-0.002 OD/h) |
| borderline_delta_od | GOOD | 15 | Delta OD just above the 0.15 OD gate (0.16-0.24 realised) |
| haldane_mild_inhibition | GOOD | 12 | Haldane, mild substrate inhibition (Ki=10 >> Ks=0.1, S0=1.0) |
| haldane_strong_inhibition | GOOD | 15 | Haldane, strong substrate inhibition (Ki=0.15 ~ Ks=0.1, S0=1.0) |
| haldane_total_inhibition | BAD | 18 | Haldane at inhibitory substrate (Ki=0.02, S0=5.0): no growth phase |
| evaporation_drift | BAD | 20 | Dead edge well drifting up 0.0008-0.002 OD/h - the false positive its name promises |
| bubble_spike | GOOD | 10 | Growth curve with 1-2 large bubble artefacts (0.25-0.8 OD) in the exponential phase |
| noise_ladder_1 | GOOD | 15 | Noise ladder rung 1: per-well sigma 0.06 OD (target oracle error ~5 %) |
| noise_ladder_2 | GOOD | 15 | Noise ladder rung 2: per-well sigma 0.10 OD (target oracle error ~10 %) |
| noise_ladder_3 | BAD | 20 | Noise ladder rung 3: per-well sigma 0.25 OD (target oracle error ~35 %, out of distribution) |
| noise_ladder_4 | BAD | 20 | Noise ladder rung 4: per-well sigma 0.35 OD (target oracle error ~60 %, out of distribution) |
| incomplete_24h | GOOD | 50 | Group 1 geometry (24 h, 96 points): the run ends at 60-85 % of A |
| run_24h_plateaued | GOOD | 10 | Group 1 geometry that DOES plateau inside 24 h |
| incomplete_growth_richards | GOOD | 15 | 100 h run still rising at the end, Richards-generated so A is genuinely non-identifiable |
| incomplete_growth_baranyi | GOOD | 15 | 100 h run still rising at the end, Baranyi-generated so A is genuinely non-identifiable |
| late_growth_still_rising | GOOD | 12 | 60 h flat, then growth (the G4 PERMETHRINANDLB_CISPERM1 shape, t50 = 78 h) |
| run_ends_at_plateau_edge | GOOD | 10 | Run stops 4 samples after t99, leaving the peak-finder no plateau window |
| slow_secondary_drift | GOOD | 15 | Plateau, then a slow 30-80 h creep - the dominant real grower shape |
| diauxic_second_larger | GOOD | 12 | Diauxic shift whose SECOND phase is the larger one (A1 0.4-0.7, A2 0.8-1.2) |
| clipped_then_grows | GOOD | 10 | Flat zero for 30-40 h, then real growth (G4 PERMETHRINANDLB_CYPERM2, 26.4 % exact zeros) |
| edge_well_evaporation | GOOD | 12 | One of three wells evaporating at +0.0015 OD/h; the condition mean carries a third of it |
| high_autocorr_good | GOOD | 12 | Model mismatch with structured residuals: autocorr 0.7-0.95 at R² 0.96-0.99, still GOOD |
| death_phase_extreme | GOOD | 5 | The historic exponential collapse (median 72-98 % of peak lost) - out of distribution |
| partial_lysis | GOOD | 12 | Plateau then a gentle 20-35 % sigmoid loss (G4 LB_PERM7 0.253, G2 LB_MAL4 0.263) |
| sudden_crash | GOOD | 10 | Hold, then a ~10 h collapse of 25-40 % (G4 LB_PERM4: 1.880 -> 1.180 across 57-68 h) |
| dip_and_recover | GOOD | 12 | Peak, 0.3-0.5 OD drawdown, then partial recovery (G2 MALATHIONANDLB_MAL8) |

64 scenarios and 1057 requested curves in total;
artefact scenarios add an artefact-free twin per curve, so the default
suite is larger than that (about 1140 rows at seed 42).  Its expected-GOOD
share is held in the lab band, 49-60 % - the 92 real conditions are 30.4 %
dead and 13.0 % trace (`config/lab_prior.json`) - against 71.9 % GOOD in
the benchmark this replaces.  MANIFEST.json records the realised numbers
of the run that produced the published headline.

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
Generate noise to hit a target RMSE / R² exactly (`RMSEBasedNoise`, which solves
for the scale rather than assuming it).

### 5. Colored Noise
`ColoredNoise` adds the correlated component the real reader shows: a fast
AR(1) term (tau 0.5 h), a slow drift term (tau 7.5 h) and a white term, so
residual autocorrelation is a property of the curve rather than an accident.

### 6. Blank and Artefact Models
`BlankModel.sample_blank` draws a measured media blank; `inject_spikes`,
`apply_signed_drift` and `add_target_rmse_component` add bubbles, evaporation /
condensation drift and a calibrated RMSE component. All of them take an
explicit `rng` (`None`, an integer seed or a `numpy.random.Generator`) -- the
module calls `np.random.seed` nowhere, so two scenarios can never share a
global stream.

## Output Format

Generated CSV files match the TECAN pipeline input format:

```csv
TIME[H],TREATMENT_CURVE0001_blanked,TREATMENT_CURVE0002_blanked,...
0.0,0.052,0.048,...
0.5,0.055,0.051,...
```

Ground truth CSV describes the curve that was actually written. The schema is
`src/truth.py` (`GROUND_TRUTH_COLUMNS`, `TRUTH_SCHEMA_VERSION = '2'`, 48
columns); it keeps the sampler's request as `requested_*` and the values
measured back off the written curve as `true_*`:

```csv
curve_id,strain_name,scenario,expected_class,label_rule,label_rule_version,scoring_mode,lab_band,model_type,shape,requested_A,requested_mu,requested_lambda,true_A,true_mu,true_lambda,...
CURVE0001,CURVE0001,standard,GOOD,scenario,1,standard,0.6-1.2,gompertz,gompertz,1.23,0.18,4.5,1.23,0.18,4.5,...
```

`label_rule` says where the label came from: `scenario` (the catalog asserts it)
or `oracle` (decided per curve by whether A and mu are recoverable from the
curve that was written). `scoring_mode` carries the difficulty policy --
`standard`, `a_not_identifiable`, `preprocessing_blind` or `model_mismatch` --
so `expected_class` stays a plain binary GOOD/BAD.

## Validation Metrics

`validation/benchmark.py` writes `report.json` and the per-curve / per-scenario
tables. The headline is **balanced accuracy with sensitivity and specificity**,
because the suite is deliberately not balanced; raw accuracy is reported second,
with its prevalence stated next to it.

- **Classification**: balanced accuracy, sensitivity, specificity, then
  accuracy, precision, recall, F1, NPV and MCC
- **Parameter Recovery**: RMSE, R² and bias for A, μ and λ, stratified and
  reported per scenario with tolerance pass rates
- **Gate Attribution**: which quality gate fired on each rejected curve
  (`gate_attribution.csv`, from the pipeline's `failed_gates` column)
- **Scenario-Level Analysis**: performance per scenario, with
  `preprocessing_blind` and `a_not_identifiable` scenarios reported separately
  from the headline rather than folded into it

Every published number names its truncation mode, seed, dataset hash and git
SHA; those come from the run's `run_settings.json` and the suite's
`MANIFEST.json`.

### The published headline

First-peak truncation, seed 42, 1139 curves. The headline is scored over the
960 curves the present gate set can decide fairly; the other 179 are scored
separately, not dropped.

| | full suite (n=960) | the original 32 scenarios (n=532) |
|---|---:|---:|
| balanced accuracy | 87.9 % | 96.8 % |
| sensitivity | 85.6 % | 93.5 % |
| specificity | 90.2 % | 100.0 % |
| accuracy | 87.6 % | 96.1 % |
| TP / TN / FP / FN | 463 / 378 / 41 / 78 | 304 / 207 / 0 / 21 |

All 1139 curves under one flat rule, nothing excluded: 81.9 % accuracy, 81.3 %
balanced.

- `<run-dir>/VALIDATION_SUMMARY.md` -- the markdown summary of a run, rendered
  from its `report.json` by `scripts/validation_summary.py`
  (`make validation-summary`). A build artefact: rendered on demand, never
  hand-edited and never committed, so no stale copy of it can exist here.
- `benchmark/v2/` -- the tracked copy of the run itself (`report.json`,
  `per_curve.csv`, `per_scenario.csv`, `gate_attribution.csv`, `HEADLINE.json`).
- `benchmark/v1/` -- the frozen pre-rebuild baseline, kept so the older
  published numbers stay checkable. Its numbers describe v1 data and are not
  current.
- `../docs/BENCHMARK_LEDGER.md` -- what moved between the two, and why.

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

### Score a Pipeline Run

```python
from validation.benchmark import run_benchmark

report = run_benchmark(
    results_csv='<run>/processing_results.csv',
    ground_truth_csv='output/full_test/test_data/ground_truth.csv',
    out_dir='output/full_test/benchmark',
    # omit when the run wrote run_settings.json beside its results
    truncation_mode='first_peak',
)
cls = report['classification']
print(f"balanced accuracy: {cls['balanced_accuracy']:.1%} "
      f"(sens {cls['sensitivity']:.1%}, spec {cls['specificity']:.1%})")
```

`validation/pipeline_validator.py` is still there for the end-to-end path -- it
runs `scripts/01_growth_curve_analysis.py` itself -- and takes paths, not
frames: `PipelineValidator(analysis_script_path, synthetic_data_dir,
ground_truth_csv)`.

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

1. **Import errors**: Run from the `synthetic_data/` directory, or put it on `sys.path` (`sys.path.insert(0, 'synthetic_data')`) and import `from src...` / `from validation...`; there is no top-level `synthetic_data/__init__.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Path issues**: Use absolute paths or run scripts from the project root

### Getting Help

Check the analysis pipeline documentation in `scripts/README.md` and `PIPELINE_METHODOLOGY.md` for pipeline-specific questions.

## License

Part of the TECAN Growth Curve Analysis project.
