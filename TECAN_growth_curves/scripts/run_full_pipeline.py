#!/usr/bin/env python3
"""
TECAN Growth Curve Full Analysis Pipeline
==========================================
BIO380SP25 - Pesticide Bioremediating Bacteria Research Project

Master orchestrator that runs the complete analysis pipeline in correct order.
Edit config.yaml to configure paths, thresholds, and analysis options.

PIPELINE FLOWCHART:
===================

    ┌─────────────────────────────────────────────────────┐
    │           RAW TECAN PLATE READER DATA                │
    │  (96-well format or pre-processed *_DATA.csv files) │
    └──────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  STEP 0     │  (only if raw 96-well format)
                    │  Preprocess │  02_preprocess_raw_plate_data.py
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────┐
                    │  STEP 1                  │
                    │  Train ML Classifier     │  09_train_classifier.py
                    │  (if models don't exist) │  → models/*.joblib
                    └──────┬──────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │  STEP 2                          │
                    │  Gompertz Growth Curve Analysis   │  01_growth_curve_analysis.py
                    │  (per group, --adaptive)          │
                    │  • Pre-fit ML gate (reject junk)  │
                    │  • Truncation (adaptive R²)       │
                    │  • Gompertz fitting                │
                    │  • Post-fit ML classification      │
                    │  • Rule-based fallback gates       │
                    │  • QC plots per strain             │
                    └──────┬──────────────────────────-┘
                           │
                    ┌──────▼──────────────────┐
                    │  STEP 3                  │
                    │  Combine Group Results    │  → all_groups_results.csv
                    │  (92 strains, 4 groups)   │
                    └──────┬──────────────────-┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───────┐ ┌──▼──────────────────┐
       │  STEP 4     │ │ STEP 5   │ │  STEP 6              │
       │  Haldane    │ │ Advanced │ │  Statistical Analysis │
       │  Inhibition │ │ Fitting  │ │  ANOVA, pairwise     │
       │  ODE model  │ │ GP/Boot/ │ │  publication figures  │
       │  AICc comp. │ │ Ensemble │ │  R-format export      │
       │  Ki ranking │ │          │ │                       │
       └─────────────┘ └──┬───────┘ └───────────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 7                  │
                   │  Truncation Comparison   │  5-method ranking
                   │  + Bad Strain Rescue     │  --include-bad
                   └──────┬──────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 8                  │
                   │  Export for Collaboration │  clean CSV + methodology
                   └──────┬──────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 9                  │
                   │  Synthetic Validation     │  555 synthetic curves
                   │  Accuracy / Recall / F1   │  parameter recovery
                   └──────┬──────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 10                 │
                   │  Inter-Operator          │  ANOVA + CV heatmap
                   │  Reproducibility         │  shared strains
                   └──────┬──────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 11 (optional)      │
                   │  Genomic Prediction      │  BLAST → gene features
                   │  Genotype → Phenotype    │  elastic net + priors
                   └──────────────────────────┘

Usage:
    python run_full_pipeline.py                    # Run all steps
    python run_full_pipeline.py --steps 2,3,4      # Run specific steps
    python run_full_pipeline.py --from-step 4      # Resume from step 4
    python run_full_pipeline.py --dry-run           # Show what would run
    python run_full_pipeline.py --no-ml             # Skip ML (rule-based only)
    python run_full_pipeline.py --skip-advanced     # Skip steps 5,7
    python run_full_pipeline.py --config my.yaml    # Custom config file
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def load_config(config_path=None):
    """Load pipeline configuration from YAML."""
    import yaml

    if config_path is None:
        config_path = SCRIPT_DIR / 'config.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"WARNING: Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def ml_models_exist():
    """Check if trained ML classifier models are available."""
    prefit = PROJECT_DIR / 'models' / 'prefit_gate.joblib'
    postfit = PROJECT_DIR / 'models' / 'postfit_classifier.joblib'
    return prefit.exists() and postfit.exists()


def synthetic_data_exists():
    """Check if synthetic test data is available for validation."""
    # Use validation_holdout (independent from training data) to avoid data leakage
    synth_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'validation_holdout' / 'test_data'
    if (synth_dir / 'DATA').exists() and (synth_dir / 'ground_truth.csv').exists():
        return True
    # Fall back to comprehensive_test if holdout not generated yet
    synth_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data'
    return (synth_dir / 'DATA').exists() and (synth_dir / 'ground_truth.csv').exists()


def run_step(cmd, description, dry_run=False):
    """Run a pipeline step with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    else:
        print(f"  Completed in {elapsed:.1f}s")
        return True


# ─── Step Implementations ────────────────────────────────────────────────────

def step0_preprocess(config, dry_run=False, **kwargs):
    """Step 0: Preprocess raw 96-well plate reader data (Groups 1, 5, 6)."""
    all_ok = True

    # Group 1
    groups = config.get('groups', {})
    g1 = groups.get('Group1', {})
    g1_data_dir = PROJECT_DIR / g1.get('data_dir', 'data/raw/Group1/Group_1_DATA')

    raw_csv = PROJECT_DIR / 'data' / 'raw' / 'Group1' / 'GrowthRate_ Group1_Values.csv'
    key_csv = PROJECT_DIR / 'data' / 'raw' / 'Group1' / 'GROUP1_KEY_V2.csv'

    if raw_csv.exists() and key_csv.exists() and not g1_data_dir.exists():
        ok = run_step(
            [sys.executable, str(SCRIPT_DIR / '02_preprocess_raw_plate_data.py'),
             str(raw_csv), str(key_csv), '-o', str(g1_data_dir)],
            "STEP 0a: Preprocess Group 1 raw plate data → CSV",
            dry_run=dry_run
        )
        if not ok:
            all_ok = False
    else:
        if g1_data_dir.exists():
            print("\n  Step 0a: Group 1 already preprocessed, skipping")
        else:
            print("\n  Step 0a: No Group 1 raw data found, skipping")

    # Groups 5-6 (Walton/Dominique)
    needs_preprocess = False
    for gname in ('Group5', 'Group6'):
        gcfg = groups.get(gname, {})
        if gcfg:
            data_dir = PROJECT_DIR / gcfg.get('data_dir', '')
            if not data_dir.exists():
                needs_preprocess = True
                break

    if needs_preprocess:
        ok = run_step(
            [sys.executable, str(SCRIPT_DIR / '02b_preprocess_groups5_6.py')],
            "STEP 0b: Preprocess Groups 5-6 raw plate data → CSV",
            dry_run=dry_run
        )
        if not ok:
            all_ok = False
    else:
        print("\n  Step 0b: Groups 5-6 already preprocessed, skipping")

    return all_ok


def step1_train_classifier(config, dry_run=False, **kwargs):
    """Step 1: Train ML classifier (if models don't exist yet)."""
    no_ml = kwargs.get('no_ml', False)

    if no_ml:
        print("\n  Step 1: ML disabled (--no-ml), skipping classifier training")
        return True

    if ml_models_exist():
        print("\n  Step 1: ML models already exist, skipping training")
        print(f"    Pre-fit gate:  {PROJECT_DIR / 'models' / 'prefit_gate.joblib'}")
        print(f"    Post-fit clf:  {PROJECT_DIR / 'models' / 'postfit_classifier.joblib'}")
        return True

    if not synthetic_data_exists():
        print("\n  Step 1: No synthetic training data found, skipping ML training")
        print("    (Pipeline will use rule-based classification as fallback)")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '09_train_classifier.py'), '--compare'],
        "STEP 1: Train ML Classifier (pre-fit gate + post-fit classifier)",
        dry_run=dry_run
    )


def step2_gompertz(config, dry_run=False, **kwargs):
    """Step 2: Run Gompertz growth curve analysis on all groups."""
    groups = config.get('groups', {})
    no_ml = kwargs.get('no_ml', False)
    use_ml = not no_ml and ml_models_exist()
    all_ok = True

    for group_name, group_cfg in sorted(groups.items()):
        data_dir = PROJECT_DIR / group_cfg['data_dir']
        output_dir = PROJECT_DIR / group_cfg['output_dir']

        if not data_dir.exists():
            print(f"\n  WARNING: {group_name} data not found at {data_dir}, skipping")
            continue

        cmd = [
            sys.executable, str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
            str(data_dir),
            '-o', str(output_dir),
            '--adaptive',
        ]
        if use_ml:
            cmd.append('--ml-classify')

        label = f"STEP 2: Gompertz Analysis — {group_name}"
        if use_ml:
            label += " (with ML classifier)"

        ok = run_step(cmd, label, dry_run=dry_run)
        if not ok:
            all_ok = False

    return all_ok


def step3_combine(config, dry_run=False, **kwargs):
    """Step 3: Combine all group results into all_groups_results.csv."""
    import pandas as pd

    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    groups = config.get('groups', {})

    print(f"\n{'='*60}")
    print(f"  STEP 3: Combine Group Results → all_groups_results.csv")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    all_dfs = []
    for group_name, group_cfg in sorted(groups.items()):
        csv_path = PROJECT_DIR / group_cfg['output_dir'] / 'processing_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['group'] = group_name
            df['operator'] = group_cfg.get('operator', group_name)
            df['year'] = group_cfg.get('year', '')
            all_dfs.append(df)
            good = df['is_good'].sum()
            bad = len(df) - good
            print(f"  {group_name}: {len(df)} curves ({good} good, {bad} bad)")
        else:
            print(f"  WARNING: {group_name} results not found at {csv_path}")

    if not all_dfs:
        print("  ERROR: No group results found to combine")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = results_base / 'all_groups_results.csv'
    combined.to_csv(out_path, index=False)

    total_good = combined['is_good'].sum()
    total_bad = len(combined) - total_good
    print(f"\n  Combined: {len(combined)} curves ({total_good} good, {total_bad} bad)")
    print(f"  Saved: {out_path}")

    # Phase A.5: schema assertion — downstream stages (03, 05, 06, classifier)
    # rely on the unified *_final/*_source columns emitted by the post-Phase A.1
    # version of 01_growth_curve_analysis.py. Fail fast if the schema is stale.
    required_schema = {
        'mu_final', 'mu_source', 'mu_final_err',
        'A_final',  'A_source',  'A_final_err',
        'lam_final', 'lam_source', 'lam_final_err',
        'usable_for',
    }
    missing = required_schema - set(combined.columns)
    if missing:
        print(f"\n  ERROR: combined CSV missing expected refined-param columns: {sorted(missing)}")
        print("         This means one or more per-group processing_results.csv files "
              "were produced by an older (pre-Phase A.1) version of 01_growth_curve_analysis.py.")
        print("         Re-run step 2 (Gompertz analysis) for all groups before proceeding.")
        return False
    return True


def step4_haldane(config, dry_run=False, **kwargs):
    """Step 4: Run Haldane feedback inhibition analysis."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '05_haldane_analysis.py')],
        "STEP 4: Haldane Feedback Inhibition Analysis (ODE + AICc)",
        dry_run=dry_run
    )


def step5_advanced(config, dry_run=False, **kwargs):
    """Step 5: Run advanced fitting (GP, Bootstrap, Ensemble)."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '06_advanced_fitting.py')],
        "STEP 5: Advanced Fitting (GP Truncation + Bootstrap CIs + Ensemble)",
        dry_run=dry_run
    )


def step6_statistics(config, dry_run=False, **kwargs):
    """Step 6: Run statistical analysis and generate figures."""
    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    all_results = results_base / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 6: all_groups_results.csv not found, skipping statistics")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '03_statistical_analysis.py'),
         '--input', str(all_results),
         '--output-dir', str(results_base)],
        "STEP 6: Statistical Analysis (ANOVA + pairwise + publication figures)",
        dry_run=dry_run
    )


def step7_compare_methods(config, dry_run=False, **kwargs):
    """Step 7: Compare truncation methods + rescue bad strains."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '07_compare_truncation_methods.py'),
         '--include-bad'],
        "STEP 7: Truncation Method Comparison + Bad Strain Rescue",
        dry_run=dry_run
    )


def step8_export(config, dry_run=False, **kwargs):
    """Step 8: Export clean results for collaboration."""
    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    all_results = results_base / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 8: all_groups_results.csv not found, skipping export")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '04_export_for_collaboration.py'),
         '--input', str(all_results),
         '--output-dir', str(results_base)],
        "STEP 8: Export for Collaboration (clean CSV + methodology)",
        dry_run=dry_run
    )


def step9_validate(config, dry_run=False, **kwargs):
    """Step 9: Validate pipeline on synthetic data."""
    no_ml = kwargs.get('no_ml', False)

    if not synthetic_data_exists():
        print("\n  Step 9: No synthetic test data found, skipping validation")
        return True

    # Use validation_holdout (seed=99) — independent from training data (seed=42)
    # This prevents data leakage: the classifier never trained on these curves
    holdout_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'validation_holdout' / 'test_data'
    if (holdout_dir / 'DATA').exists():
        synth_data = holdout_dir / 'DATA'
        synth_gt = holdout_dir / 'ground_truth.csv'
        out_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'validation_holdout' / 'validation_latest'
        print("  Using independent validation holdout set (seed=99, no data leakage)")
    else:
        synth_data = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data' / 'DATA'
        synth_gt = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data' / 'ground_truth.csv'
        out_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'validation_latest'
        print("  WARNING: Using training data for validation (holdout not found)")

    # Run pipeline on synthetic data (same flags as step 2: adaptive + ML)
    cmd = [
        sys.executable, str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
        str(synth_data),
        '-o', str(out_dir),
        '-q',
        '--adaptive',
    ]
    if not no_ml and ml_models_exist():
        cmd.append('--ml-classify')

    ok = run_step(cmd, "STEP 9a: Run pipeline on 480 synthetic curves", dry_run=dry_run)
    if not ok:
        return False

    # Run validator to compute accuracy metrics
    results_csv = out_dir / 'processing_results.csv'
    validator_script = PROJECT_DIR / 'synthetic_data' / 'validation' / 'pipeline_validator.py'

    if validator_script.exists() and (results_csv.exists() or dry_run):
        ok = run_step(
            [sys.executable, str(validator_script),
             '--pipeline', str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
             '--data-dir', str(synth_data),
             '--ground-truth', str(synth_gt),
             '--output-dir', str(out_dir),
             '--skip-run',
             '--results-csv', str(results_csv)],
            "STEP 9b: Compute validation metrics (accuracy, recall, F1, parameter recovery)",
            dry_run=dry_run
        )
        return ok

    return True


def step10_operator_comparison(config, dry_run=False, **kwargs):
    """Step 10: Inter-operator reproducibility analysis."""
    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    all_results = results_base / 'all_groups_results.csv'
    if not all_results.exists() and not dry_run:
        print("\n  Step 10: all_groups_results.csv not found, skipping")
        return True
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '10_operator_comparison.py'),
         '--input', str(all_results),
         '--output-dir', str(results_base / 'Operator_Comparison')],
        "STEP 10: Inter-Operator Reproducibility Analysis",
        dry_run=dry_run
    )


def step11_genomic_prediction(config, dry_run=False, **kwargs):
    """Step 11: Genomic feature extraction + genotype-to-phenotype prediction."""
    no_genomic = kwargs.get('no_genomic', False)
    if no_genomic:
        print("\n  Step 11: Genomic prediction disabled (--no-genomic), skipping")
        return True

    genomic_dir = PROJECT_DIR / 'data' / 'genomic'
    genomic_csv = genomic_dir / 'genomic_features.csv'

    if not genomic_dir.exists():
        print("\n  Step 11: No data/genomic/ directory found, skipping")
        print("    (Create data/genomic/ with BLAST results to enable genomic prediction)")
        return True

    # Step 11a: Extract genomic features from BLAST/GFF if not already done
    if not genomic_csv.exists():
        blast_dir = genomic_dir / 'blast_results'
        strain_mapping = genomic_dir / 'strain_mapping.csv'
        annotations_dir = genomic_dir / 'annotations'

        if blast_dir.exists() and strain_mapping.exists():
            cmd = [
                sys.executable, str(SCRIPT_DIR / 'genomic_features.py'),
                '--blast-dir', str(blast_dir),
                '--strain-mapping', str(strain_mapping),
                '--output', str(genomic_csv),
            ]
            if annotations_dir.exists():
                cmd.extend(['--annotations-dir', str(annotations_dir)])

            ok = run_step(cmd,
                          "STEP 11a: Extract Genomic Features (BLAST/GFF)",
                          dry_run=dry_run)
            if not ok:
                return False
        else:
            print("\n  Step 11a: No BLAST results or strain mapping found, skipping")
            print(f"    Expected: {blast_dir} and {strain_mapping}")
            return True

    # Step 11b: Run genotype-to-phenotype prediction
    cmd = [
        sys.executable, str(SCRIPT_DIR / '11_genomic_prediction.py'),
        '--validate',
    ]

    # Use pesticide filter from config if specified
    genomic_cfg = config.get('genomic', {})
    pesticide = genomic_cfg.get('pesticide_filter', None)
    if pesticide:
        cmd.extend(['--pesticide', pesticide])

    return run_step(cmd,
                    "STEP 11b: Genotype-to-Phenotype Prediction (Elastic Net + Bayesian Ridge)",
                    dry_run=dry_run)


# ─── Step Registry ───────────────────────────────────────────────────────────
#  (step_num, function, short_name, is_advanced, is_critical)

STEPS = [
    (0, step0_preprocess,       "Preprocess raw data",                False, False),
    (1, step1_train_classifier, "Train ML classifier",                False, False),
    (2, step2_gompertz,         "Gompertz curve analysis (per group)", False, True),
    (3, step3_combine,          "Combine group results",              False, True),
    (4, step4_haldane,          "Haldane inhibition analysis (AICc)", False, False),
    (5, step5_advanced,         "Advanced fitting (GP/Bootstrap)",    True,  False),
    (6, step6_statistics,       "Statistical analysis + figures",     False, False),
    (7, step7_compare_methods,  "Truncation comparison + rescue",     True,  False),
    (8, step8_export,           "Export for collaboration",           False, False),
    (9, step9_validate,         "Synthetic validation (480 curves)",  False, False),
    (10, step10_operator_comparison, "Inter-operator reproducibility", False, False),
    (11, step11_genomic_prediction, "Genomic prediction (genotype→phenotype)", False, False),
]


def main():
    parser = argparse.ArgumentParser(
        description="TECAN Growth Curve Full Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  0  Preprocess raw plate data (Group 1 only, if needed)
  1  Train ML classifier (skipped if models/ already exists)
  2  Gompertz growth curve analysis (all groups, --adaptive, with ML)
  3  Combine group results → all_groups_results.csv
  4  Haldane feedback inhibition analysis (ODE + AICc comparison)
  5  Advanced fitting (GP truncation, Bootstrap CIs, Ensemble)
  6  Statistical analysis + publication figures (ANOVA, pairwise)
  7  Truncation method comparison + bad strain rescue
  8  Export clean results for collaboration
  9  Synthetic validation (accuracy, recall, F1, parameter recovery)
  10 Inter-operator reproducibility analysis
  11 Genomic prediction (genotype-to-phenotype, requires data/genomic/)

Examples:
  python run_full_pipeline.py                     # Run everything
  python run_full_pipeline.py --steps 2,3,4       # Core + Haldane only
  python run_full_pipeline.py --from-step 4       # Resume from Haldane
  python run_full_pipeline.py --skip-advanced      # Skip GP/Ensemble (steps 5,7)
  python run_full_pipeline.py --no-ml              # Rule-based only (no ML)
  python run_full_pipeline.py --dry-run            # Preview without running
        """
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: scripts/config.yaml)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated step numbers to run (e.g., "2,3,4")')
    parser.add_argument('--from-step', type=int, default=None,
                        help='Start from this step number (inclusive)')
    parser.add_argument('--skip-advanced', action='store_true',
                        help='Skip advanced fitting steps (5, 7)')
    parser.add_argument('--no-ml', action='store_true',
                        help='Disable ML classifier (use rule-based classification only)')
    parser.add_argument('--no-genomic', action='store_true',
                        help='Skip genomic prediction step even if data exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')

    args = parser.parse_args()
    config = load_config(args.config)

    # Determine which steps to run
    max_step = max(s[0] for s in STEPS)
    if args.steps:
        run_steps = set(int(s.strip()) for s in args.steps.split(','))
    elif args.from_step is not None:
        run_steps = set(range(args.from_step, max_step + 1))
    else:
        run_steps = set(range(0, max_step + 1))

    if args.skip_advanced:
        run_steps -= {5, 7}

    # Header
    ml_status = "disabled (--no-ml)" if args.no_ml else (
        "enabled (models exist)" if ml_models_exist() else "will train in step 1"
    )

    print("\n" + "=" * 60)
    print("  TECAN GROWTH CURVE FULL ANALYSIS PIPELINE")
    print("  BIO380SP25 - Pesticide Bioremediating Bacteria")
    print("=" * 60)
    print(f"  Project dir:  {PROJECT_DIR}")
    print(f"  Config:       {args.config or 'scripts/config.yaml'}")
    print(f"  Steps:        {sorted(run_steps)}")
    print(f"  ML classify:  {ml_status}")
    if args.dry_run:
        print(f"  Mode:         DRY RUN (no execution)")
    print()

    # Run steps in order
    results = {}
    pipeline_start = time.time()

    for step_num, step_fn, step_name, is_advanced, is_critical in STEPS:
        if step_num not in run_steps:
            continue

        ok = step_fn(config, dry_run=args.dry_run, no_ml=args.no_ml,
                     no_genomic=args.no_genomic)
        results[step_num] = ok

        if not ok:
            print(f"\n  WARNING: Step {step_num} ({step_name}) failed")
            if is_critical:
                print("  ABORTING: Critical pipeline step failed")
                break

    # Summary
    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for step_num, _, step_name, _, _ in STEPS:
        if step_num in results:
            status = "OK" if results[step_num] else "FAILED"
            print(f"  Step {step_num}: {step_name:45s} [{status}]")
        elif step_num in run_steps:
            print(f"  Step {step_num}: {step_name:45s} [SKIPPED]")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results:    {PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')}")
    print()

    # Exit with failure if any critical step failed
    if any(not v for k, v in results.items()
           if any(s[0] == k and s[4] for s in STEPS)):
        sys.exit(1)


if __name__ == '__main__':
    main()
