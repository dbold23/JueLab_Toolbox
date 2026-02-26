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
                    │  Gompertz Curve Analysis  │  01_growth_curve_analysis.py
                    │  (per group)              │  --adaptive [--ml-classify]
                    │  • Truncation             │
                    │  • Gompertz fitting       │
                    │  • Classification (GOOD/  │
                    │    BAD)                   │
                    │  • QC plots               │
                    └──────┬──────────────────-┘
                           │
                    ┌──────▼──────────────────┐
                    │  STEP 2                  │
                    │  Combine Group Results    │  → all_groups_results.csv
                    └──────┬──────────────────-┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───────┐ ┌──▼──────────────────┐
       │  STEP 3     │ │ STEP 4   │ │  STEP 5              │
       │  Haldane    │ │ Advanced │ │  Statistical Analysis │
       │  Inhibition │ │ Fitting  │ │  03_statistical_      │
       │  Analysis   │ │ GP/Boot/ │ │  analysis.py          │
       │  05_haldane │ │ Ensemble │ │                       │
       └─────────────┘ │ 06_adv.  │ └───────────────────────┘
                       └──┬───────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 6                  │
                   │  Truncation Comparison   │  07_compare_truncation_methods.py
                   │  + Bad Strain Rescue     │  --include-bad
                   └──────────────────────────┘
                          │
                   ┌──────▼──────────────────┐
                   │  STEP 7                  │
                   │  Export for Collaboration │  04_export_for_collaboration.py
                   └──────────────────────────┘

    OPTIONAL (run separately):
    ┌─────────────────────────────────────────────┐
    │  ML Classifier Training (09_train_classifier)│
    │  Interactive Validation (08_validate_trunc.) │
    │  Synthetic Data Validation                   │
    └─────────────────────────────────────────────┘

Usage:
    python run_full_pipeline.py                    # Run all steps
    python run_full_pipeline.py --steps 1,2,3      # Run specific steps
    python run_full_pipeline.py --from-step 3      # Resume from step 3
    python run_full_pipeline.py --dry-run           # Show what would run
    python run_full_pipeline.py --ml-classify       # Enable ML classifier
    python run_full_pipeline.py --skip-advanced     # Skip steps 4,6
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


def step0_preprocess(config, dry_run=False):
    """Step 0: Preprocess raw 96-well plate reader data (Group 1 only)."""
    groups = config.get('groups', {})
    g1 = groups.get('Group1', {})
    g1_data_dir = PROJECT_DIR / g1.get('data_dir', 'data/raw/Group1/Group_1_DATA')

    # Only preprocess if Group 1 needs it
    raw_csv = PROJECT_DIR / 'data' / 'raw' / 'Group1' / 'GrowthRate_ Group1_Values.csv'
    key_csv = PROJECT_DIR / 'data' / 'raw' / 'Group1' / 'GROUP1_KEY_V2.csv'

    if raw_csv.exists() and key_csv.exists() and not g1_data_dir.exists():
        return run_step(
            [sys.executable, str(SCRIPT_DIR / '02_preprocess_raw_plate_data.py'),
             str(raw_csv), str(key_csv), '-o', str(g1_data_dir)],
            "STEP 0: Preprocess Group 1 raw plate data → CSV",
            dry_run=dry_run
        )
    else:
        if g1_data_dir.exists():
            print("\n  Step 0: Group 1 already preprocessed, skipping")
        else:
            print("\n  Step 0: No raw data found to preprocess, skipping")
        return True


def step1_gompertz(config, ml_classify=False, dry_run=False):
    """Step 1: Run Gompertz growth curve analysis on all groups."""
    groups = config.get('groups', {})
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
        if ml_classify:
            cmd.append('--ml-classify')

        ok = run_step(cmd, f"STEP 1: Gompertz Analysis — {group_name}", dry_run=dry_run)
        if not ok:
            all_ok = False

    return all_ok


def step2_combine(config, dry_run=False):
    """Step 2: Combine all group results into all_groups_results.csv."""
    import pandas as pd

    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    groups = config.get('groups', {})

    print(f"\n{'='*60}")
    print(f"  STEP 2: Combine Group Results → all_groups_results.csv")
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
    out_path = results_base / 'all_groups_results.csv'
    combined.to_csv(out_path, index=False)

    total_good = combined['is_good'].sum()
    total_bad = len(combined) - total_good
    print(f"\n  Combined: {len(combined)} curves ({total_good} good, {total_bad} bad)")
    print(f"  Saved: {out_path}")
    return True


def step3_haldane(config, dry_run=False):
    """Step 3: Run Haldane feedback inhibition analysis."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '05_haldane_analysis.py')],
        "STEP 3: Haldane Feedback Inhibition Analysis",
        dry_run=dry_run
    )


def step4_advanced(config, dry_run=False):
    """Step 4: Run advanced fitting (GP, Bootstrap, Ensemble)."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '06_advanced_fitting.py')],
        "STEP 4: Advanced Fitting (GP + Bootstrap + Ensemble)",
        dry_run=dry_run
    )


def step5_statistics(config, dry_run=False):
    """Step 5: Run statistical analysis and generate figures."""
    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    all_results = results_base / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 5: all_groups_results.csv not found, skipping statistics")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '03_statistical_analysis.py'),
         '--input', str(all_results),
         '--output-dir', str(results_base)],
        "STEP 5: Statistical Analysis + Publication Figures",
        dry_run=dry_run
    )


def step6_compare_methods(config, dry_run=False):
    """Step 6: Compare truncation methods + rescue bad strains."""
    return run_step(
        [sys.executable, str(SCRIPT_DIR / '07_compare_truncation_methods.py'),
         '--include-bad'],
        "STEP 6: Truncation Method Comparison + Bad Strain Rescue",
        dry_run=dry_run
    )


def step7_export(config, dry_run=False):
    """Step 7: Export clean results for collaboration."""
    results_base = PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')
    all_results = results_base / 'all_groups_results.csv'

    if not all_results.exists() and not dry_run:
        print("\n  Step 7: all_groups_results.csv not found, skipping export")
        return True

    return run_step(
        [sys.executable, str(SCRIPT_DIR / '04_export_for_collaboration.py'),
         '--input', str(all_results),
         '--output-dir', str(results_base)],
        "STEP 7: Export for Collaboration",
        dry_run=dry_run
    )


# Step registry: (step_number, function, short_name, requires_advanced)
STEPS = [
    (0, step0_preprocess,       "Preprocess raw data",              False),
    (1, step1_gompertz,         "Gompertz curve analysis",          False),
    (2, step2_combine,          "Combine group results",            False),
    (3, step3_haldane,          "Haldane inhibition analysis",      False),
    (4, step4_advanced,         "Advanced fitting (GP/Bootstrap)",  True),
    (5, step5_statistics,       "Statistical analysis + figures",   False),
    (6, step6_compare_methods,  "Truncation comparison + rescue",   True),
    (7, step7_export,           "Export for collaboration",         False),
]


def main():
    parser = argparse.ArgumentParser(
        description="TECAN Growth Curve Full Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  0  Preprocess raw plate data (Group 1 only, if needed)
  1  Gompertz growth curve analysis (all groups, --adaptive)
  2  Combine group results → all_groups_results.csv
  3  Haldane feedback inhibition analysis
  4  Advanced fitting (GP truncation, Bootstrap CIs, Ensemble)
  5  Statistical analysis + publication figures
  6  Truncation method comparison + bad strain rescue
  7  Export clean results for collaboration

Examples:
  python run_full_pipeline.py                     # Run everything
  python run_full_pipeline.py --steps 1,2,3       # Core pipeline only
  python run_full_pipeline.py --from-step 3       # Resume from Haldane
  python run_full_pipeline.py --skip-advanced      # Skip GP/Ensemble steps
  python run_full_pipeline.py --ml-classify        # Use ML classifier in step 1
  python run_full_pipeline.py --dry-run            # Preview without running
        """
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: scripts/config.yaml)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated step numbers to run (e.g., "1,2,3")')
    parser.add_argument('--from-step', type=int, default=None,
                        help='Start from this step number (inclusive)')
    parser.add_argument('--skip-advanced', action='store_true',
                        help='Skip advanced fitting steps (4, 6)')
    parser.add_argument('--ml-classify', action='store_true',
                        help='Use ML classifier during Gompertz analysis')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')

    args = parser.parse_args()
    config = load_config(args.config)

    # Determine which steps to run
    if args.steps:
        run_steps = set(int(s.strip()) for s in args.steps.split(','))
    elif args.from_step is not None:
        run_steps = set(range(args.from_step, 8))
    else:
        run_steps = set(range(0, 8))

    if args.skip_advanced:
        run_steps -= {4, 6}

    # Header
    print("\n" + "=" * 60)
    print("  TECAN GROWTH CURVE FULL ANALYSIS PIPELINE")
    print("  BIO380SP25 - Pesticide Bioremediating Bacteria")
    print("=" * 60)
    print(f"  Project dir:  {PROJECT_DIR}")
    print(f"  Config:       {args.config or 'scripts/config.yaml'}")
    print(f"  Steps:        {sorted(run_steps)}")
    print(f"  ML classify:  {'yes' if args.ml_classify else 'no'}")
    if args.dry_run:
        print(f"  Mode:         DRY RUN (no execution)")
    print()

    # Run steps in order
    results = {}
    pipeline_start = time.time()

    for step_num, step_fn, step_name, is_advanced in STEPS:
        if step_num not in run_steps:
            continue

        # Pass ml_classify flag to step 1
        if step_num == 1:
            ok = step_fn(config, ml_classify=args.ml_classify, dry_run=args.dry_run)
        else:
            ok = step_fn(config, dry_run=args.dry_run)

        results[step_num] = ok

        if not ok:
            print(f"\n  WARNING: Step {step_num} ({step_name}) failed")
            # Steps 0-2 are critical; others can fail gracefully
            if step_num <= 2:
                print("  ABORTING: Core pipeline step failed")
                break

    # Summary
    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for step_num, _, step_name, _ in STEPS:
        if step_num in results:
            status = "OK" if results[step_num] else "FAILED"
            print(f"  Step {step_num}: {step_name:42s} [{status}]")
        elif step_num in run_steps:
            print(f"  Step {step_num}: {step_name:42s} [SKIPPED]")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results:    {PROJECT_DIR / config.get('paths', {}).get('results', 'results/tables')}")
    print()

    # Exit with failure if any critical step failed
    if any(not v for k, v in results.items() if k <= 2):
        sys.exit(1)


if __name__ == '__main__':
    main()
