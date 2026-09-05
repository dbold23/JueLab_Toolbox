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
                    │  STEP 0     │  (per group, if *_DATA.csv missing)
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
                    │  (per group, config-driven)       │
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
                   │  Synthetic Validation     │  480 synthetic curves
                   │  Accuracy / Recall / F1   │  parameter recovery
                   └──────────────────────────┘

Usage:
    python run_full_pipeline.py                    # Run all steps
    python run_full_pipeline.py --steps 2,3,4      # Run specific steps
    python run_full_pipeline.py --from-step 4      # Resume from step 4
    python run_full_pipeline.py --dry-run           # Show what would run
    python run_full_pipeline.py --no-ml             # Skip ML (rule-based only)
    python run_full_pipeline.py --skip-advanced     # Skip steps 5,7
    python run_full_pipeline.py --config my.yaml    # Custom config file
    python run_full_pipeline.py --no-adaptive       # Override truncation.use_adaptive
    python run_full_pipeline.py --force-preprocess  # Rebuild every group's *_DATA.csv

Truncation and ML classification come from config.yaml (truncation.use_adaptive,
ml_classifier.enabled); --adaptive/--no-adaptive and --ml/--no-ml override them.
The exit code is non-zero if any requested step failed.
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
    synth_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data'
    return (synth_dir / 'DATA').exists() and (synth_dir / 'ground_truth.csv').exists()


def resolve_use_adaptive(config, **kwargs):
    """
    Adaptive (cross-validated) truncation on or off.

    ``--adaptive`` / ``--no-adaptive`` win; otherwise config.yaml
    ``truncation.use_adaptive`` decides (default: off).
    """
    override = kwargs.get('adaptive')
    if override is not None:
        return bool(override)
    return bool((config.get('truncation') or {}).get('use_adaptive', False))


def ml_requested(config, **kwargs):
    """Whether ML classification is wanted at all (ignoring whether models exist)."""
    if kwargs.get('no_ml', False):
        return False
    if kwargs.get('force_ml', False):
        return True
    return bool((config.get('ml_classifier') or {}).get('enabled', False))


def resolve_use_ml(config, **kwargs):
    """
    ML classification on or off: enabled in config.yaml *and* the trained models
    exist *and* ``--no-ml`` was not given. ``--ml`` forces it on regardless of
    config.yaml, but still needs the models.
    """
    return ml_requested(config, **kwargs) and ml_models_exist()


def config_args(**kwargs):
    """``['--config', PATH]`` for the sub-scripts when a custom config was given."""
    path = kwargs.get('config_path')
    return ['--config', str(path)] if path else []


def results_base(config):
    """``paths.results`` from config.yaml as an absolute path."""
    return PROJECT_DIR / (config.get('paths', {}) or {}).get('results', 'results/tables')


def advanced_output_dir(config):
    """``advanced.output_dir`` from config.yaml as an absolute path."""
    return PROJECT_DIR / (config.get('advanced', {}) or {}).get(
        'output_dir', 'results/tables/Advanced_Analysis')


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
    """
    Step 0: Rebuild every group's processed ``{MEDIA}_DATA.csv`` files from its
    raw plate-reader export.

    A group is (re)built when its ``data_dir`` is missing, or always when
    ``--force-preprocess`` is given. Groups without ``raw_values``/``raw_key``
    in config.yaml are skipped -- their processed files are the only copy.
    """
    groups = config.get('groups', {}) or {}
    force = kwargs.get('force_preprocess', False)
    all_ok = True

    if not groups:
        print("\n  Step 0: No groups configured, skipping")
        return True

    for group_name, group_cfg in sorted(groups.items()):
        group_cfg = group_cfg or {}
        if not group_cfg.get('data_dir'):
            print(f"\n  Step 0: {group_name} has no data_dir in config.yaml, skipping")
            continue
        data_dir = PROJECT_DIR / group_cfg['data_dir']
        raw_values = group_cfg.get('raw_values')
        raw_key = group_cfg.get('raw_key')

        if data_dir.exists() and not force:
            print(f"\n  Step 0: {group_name} already preprocessed, skipping ({data_dir})")
            continue

        if not raw_values or not raw_key:
            print(f"\n  Step 0: {group_name} has no raw_values/raw_key in config.yaml, skipping")
            if not data_dir.exists():
                print(f"    WARNING: {data_dir} does not exist and cannot be rebuilt")
            continue

        values_path = PROJECT_DIR / raw_values
        key_path = PROJECT_DIR / raw_key
        if not dry_run and not (values_path.exists() and key_path.exists()):
            missing = [str(p) for p in (values_path, key_path) if not p.exists()]
            print(f"\n  Step 0: {group_name} raw files not found: {', '.join(missing)}")
            all_ok = False
            continue

        ok = run_step(
            [sys.executable, str(SCRIPT_DIR / '02_preprocess_raw_plate_data.py'),
             str(values_path), str(key_path), '-o', str(data_dir)],
            f"STEP 0: Preprocess {group_name} raw plate data → {data_dir.name}/",
            dry_run=dry_run
        )
        if not ok:
            all_ok = False

    return all_ok


def step1_train_classifier(config, dry_run=False, **kwargs):
    """Step 1: Train ML classifier (if models don't exist yet)."""
    if not ml_requested(config, **kwargs):
        reason = "--no-ml" if kwargs.get('no_ml') else "ml_classifier.enabled is false in config.yaml"
        print(f"\n  Step 1: ML disabled ({reason}), skipping classifier training")
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
    groups = config.get('groups', {}) or {}
    if not groups:
        print("\n  ERROR: config.yaml defines no groups; nothing to analyse")
        return False

    use_ml = resolve_use_ml(config, **kwargs)
    use_adaptive = resolve_use_adaptive(config, **kwargs)
    all_ok = True
    ran_any = False

    for group_name, group_cfg in sorted(groups.items()):
        data_dir = PROJECT_DIR / group_cfg['data_dir']
        output_dir = PROJECT_DIR / group_cfg['output_dir']

        if not data_dir.exists():
            # A configured group with no data is a pipeline failure, not a skip:
            # step 3 would otherwise silently combine fewer groups than asked for.
            print(f"\n  ERROR: {group_name} data not found at {data_dir}")
            all_ok = False
            continue

        cmd = [
            sys.executable, str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
            str(data_dir),
            '-o', str(output_dir),
        ]
        cmd += config_args(**kwargs)
        cmd.append('--adaptive' if use_adaptive else '--no-adaptive')
        if use_ml:
            cmd.append('--ml-classify')

        label = f"STEP 2: Gompertz Analysis — {group_name}"
        label += " (adaptive truncation)" if use_adaptive else " (first-peak truncation)"
        if use_ml:
            label += " (with ML classifier)"

        ran_any = True
        ok = run_step(cmd, label, dry_run=dry_run)
        if not ok:
            all_ok = False

    if not ran_any:
        print("\n  ERROR: none of the configured group data directories exist; nothing was analysed")

    return all_ok


def step3_combine(config, dry_run=False, **kwargs):
    """Step 3: Combine all group results into all_groups_results.csv."""
    import pandas as pd

    results_dir = results_base(config)
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
    out_path = results_dir / 'all_groups_results.csv'
    combined.to_csv(out_path, index=False)

    total_good = combined['is_good'].sum()
    total_bad = len(combined) - total_good
    print(f"\n  Combined: {len(combined)} curves ({total_good} good, {total_bad} bad)")
    print(f"  Saved: {out_path}")
    return True


def step4_haldane(config, dry_run=False, **kwargs):
    """Step 4: Run Haldane feedback inhibition analysis."""
    # 05 has no --config flag: hand it the configured results directory
    # explicitly so it reads this run's all_groups_results.csv and writes
    # Haldane_Analysis/ beside it instead of the hard-coded results/tables.
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '05_haldane_analysis.py'),
         '--results-dir', str(results_base(config))],
        "STEP 4: Haldane Feedback Inhibition Analysis (ODE + AICc)",
        dry_run=dry_run
    )


def step5_advanced(config, dry_run=False, **kwargs):
    """Step 5: Run advanced fitting (GP, Bootstrap, Ensemble)."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '06_advanced_fitting.py'),
         '--results-dir', str(results_base(config))] + config_args(**kwargs),
        "STEP 5: Advanced Fitting (GP Truncation + Bootstrap CIs + Ensemble)",
        dry_run=dry_run
    )


def step6_statistics(config, dry_run=False, **kwargs):
    """Step 6: Run statistical analysis and generate figures."""
    results_dir = results_base(config)
    all_results = results_dir / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 6: all_groups_results.csv not found, skipping statistics")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '03_statistical_analysis.py'),
         '--input', str(all_results),
         '--output-dir', str(results_dir)],
        "STEP 6: Statistical Analysis (ANOVA + pairwise + publication figures)",
        dry_run=dry_run
    )


def step7_compare_methods(config, dry_run=False, **kwargs):
    """Step 7: Compare truncation methods + rescue bad strains."""
    adv_dir = advanced_output_dir(config)
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '07_compare_truncation_methods.py'),
         '--include-bad',
         '--results-csv', str(results_base(config) / 'all_groups_results.csv'),
         '--ensemble-csv', str(adv_dir / 'ensemble_truncation' / 'ensemble_truncation_results.csv'),
         '--output-dir', str(adv_dir / 'truncation_comparison')] + config_args(**kwargs),
        "STEP 7: Truncation Method Comparison + Bad Strain Rescue",
        dry_run=dry_run
    )


def step8_export(config, dry_run=False, **kwargs):
    """Step 8: Export clean results for collaboration."""
    results_dir = results_base(config)
    all_results = results_dir / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 8: all_groups_results.csv not found, skipping export")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '04_export_for_collaboration.py'),
         '--input', str(all_results),
         '--output-dir', str(results_dir)],
        "STEP 8: Export for Collaboration (clean CSV + methodology)",
        dry_run=dry_run
    )


def step9_validate(config, dry_run=False, **kwargs):
    """Step 9: Validate pipeline on synthetic data."""
    if not synthetic_data_exists():
        print("\n  Step 9: No synthetic test data found, skipping validation")
        return True

    synth_data = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data' / 'DATA'
    synth_gt = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'test_data' / 'ground_truth.csv'
    out_dir = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test' / 'validation_latest'

    # Run pipeline on synthetic data (same flags as step 2)
    cmd = [
        sys.executable, str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
        str(synth_data),
        '-o', str(out_dir),
        '-q',
    ]
    cmd += config_args(**kwargs)
    cmd.append('--adaptive' if resolve_use_adaptive(config, **kwargs) else '--no-adaptive')
    if resolve_use_ml(config, **kwargs):
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
]


def main():
    parser = argparse.ArgumentParser(
        description="TECAN Growth Curve Full Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  0  Preprocess raw plate data (every group whose processed data is missing)
  1  Train ML classifier (skipped if models/ already exists or ML is off)
  2  Gompertz growth curve analysis (all groups; truncation/ML from config.yaml)
  3  Combine group results → all_groups_results.csv
  4  Haldane feedback inhibition analysis (ODE + AICc comparison)
  5  Advanced fitting (GP truncation, Bootstrap CIs, Ensemble)
  6  Statistical analysis + publication figures (ANOVA, pairwise)
  7  Truncation method comparison + bad strain rescue
  8  Export clean results for collaboration
  9  Synthetic validation (accuracy, recall, F1, parameter recovery)

Examples:
  python run_full_pipeline.py                     # Run everything
  python run_full_pipeline.py --steps 2,3,4       # Core + Haldane only
  python run_full_pipeline.py --from-step 4       # Resume from Haldane
  python run_full_pipeline.py --skip-advanced      # Skip GP/Ensemble (steps 5,7)
  python run_full_pipeline.py --no-ml              # Rule-based only (no ML)
  python run_full_pipeline.py --no-adaptive        # First-peak truncation this run
  python run_full_pipeline.py --force-preprocess   # Rebuild all groups from raw
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
    parser.add_argument('--ml', dest='force_ml', action='store_true',
                        help='Force the ML classifier on even if ml_classifier.enabled is false '
                             '(still needs models/*.joblib)')
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', default=None,
                        help='Force adaptive (cross-validated) truncation, overriding '
                             'truncation.use_adaptive in config.yaml')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false',
                        help='Force first-peak/global-max truncation, overriding '
                             'truncation.use_adaptive in config.yaml')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='Re-run step 0 for every group even when its processed data already exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')

    args = parser.parse_args()
    if args.config is not None and not Path(args.config).exists():
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(2)
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

    # Options every step sees (CLI overrides on top of config.yaml)
    step_kwargs = dict(
        no_ml=args.no_ml,
        force_ml=args.force_ml,
        adaptive=args.adaptive,
        force_preprocess=args.force_preprocess,
        config_path=args.config,
    )

    # Header
    if not ml_requested(config, **step_kwargs):
        ml_status = "disabled (--no-ml)" if args.no_ml else "disabled (ml_classifier.enabled: false)"
    else:
        ml_status = "enabled (models exist)" if ml_models_exist() else "will train in step 1"

    adaptive_status = "adaptive (cross-validated)" if resolve_use_adaptive(config, **step_kwargs) else "first peak"
    if args.adaptive is not None:
        adaptive_status += "  [CLI override]"

    print("\n" + "=" * 60)
    print("  TECAN GROWTH CURVE FULL ANALYSIS PIPELINE")
    print("  BIO380SP25 - Pesticide Bioremediating Bacteria")
    print("=" * 60)
    print(f"  Project dir:  {PROJECT_DIR}")
    print(f"  Config:       {args.config or 'scripts/config.yaml'}")
    print(f"  Steps:        {sorted(run_steps)}")
    print(f"  Truncation:   {adaptive_status}")
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

        ok = step_fn(config, dry_run=args.dry_run, **step_kwargs)
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

    # Exit non-zero if ANY requested step failed. 'critical' only decides whether
    # the remaining steps still run -- a failed non-critical step is still a
    # failed pipeline run, and CI/Makefile must not read it as success.
    failed = sorted(k for k, v in results.items() if not v)
    if failed:
        print(f"  FAILED steps: {', '.join(str(k) for k in failed)}")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
