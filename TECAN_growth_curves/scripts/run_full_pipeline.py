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
                   │  STEP 9                  │  suite size read from
                   │  Synthetic Validation     │  MANIFEST.json at run time
                   │  Accuracy / Recall / F1   │  one directory per
                   │  parameter recovery       │  truncation mode
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
    python run_full_pipeline.py --steps 9 --all-modes        # Benchmark both truncation modes
    python run_full_pipeline.py --steps 9 --allow-in-sample-ml  # Extra IN-SAMPLE ML pass

Step 9 (synthetic validation) does NOT follow truncation.use_adaptive: that
setting is the batch default for the real plates (step 2). The published
synthetic headline is first-peak truncation -- the lab interface default -- so
step 9 benchmarks first-peak unless --adaptive or --all-modes says otherwise,
and each mode gets its own run directory so one can never overwrite the other.
Step 9 is always rule-based; the ML classifier was trained on these very curves,
so an ML pass is in-sample and needs the explicit --allow-in-sample-ml flag.

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


#: Root of the committed synthetic benchmark (data in ``test_data/``, run
#: directories beside it). Overridable per run with ``--synthetic-out-dir``
#: so a scratch run never writes into the committed tree.
SYNTHETIC_ROOT = PROJECT_DIR / 'synthetic_data' / 'output' / 'comprehensive_test'

#: The truncation mode the published synthetic headline is measured under.
#: It is the lab interface default; ``truncation.use_adaptive`` is the *batch*
#: default for the real plates and deliberately does not reach step 9 (M14:
#: `make step-9` used to run adaptive and overwrite the first-peak headline in
#: place, with nothing in any output naming the mode).
HEADLINE_TRUNCATION_MODE = 'first_peak'

#: Every truncation mode step 9 can benchmark, headline first.
STEP9_TRUNCATION_MODES = ('first_peak', 'adaptive')


def synthetic_test_data_dir(**kwargs):
    """Directory holding the generated plate CSVs and ``ground_truth.csv``."""
    return synthetic_root(**kwargs) / 'test_data'


def synthetic_root(**kwargs):
    """Where step 9 reads its data from and writes its run directories."""
    override = kwargs.get('synthetic_out_dir')
    return Path(override).resolve() if override else SYNTHETIC_ROOT


def synthetic_data_exists(**kwargs):
    """Check if synthetic test data is available for validation."""
    synth_dir = synthetic_test_data_dir(**kwargs)
    return (synth_dir / 'DATA').exists() and (synth_dir / 'ground_truth.csv').exists()


def _read_json(path):
    """Parse a small JSON file, or ``None`` if it is missing or unreadable."""
    import json
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def synthetic_suite_counts(root=None):
    """
    ``(n_curves, n_scenarios, source)`` for the synthetic suite.

    Read from ``MANIFEST.json`` (written by generate_comprehensive.py), falling
    back to ``generation_summary.json`` and finally to counting the rows of
    ``ground_truth.csv``. Nothing here is hand-typed: a printed suite size that
    disagrees with the data on disk is how a literal curve count outlives the
    suite it was written for.
    """
    root = Path(root) if root is not None else SYNTHETIC_ROOT

    for name in ('MANIFEST.json', 'generation_summary.json'):
        data = _read_json(root / name)
        if not isinstance(data, dict):
            continue
        per_scenario = data.get('n_per_scenario')
        if not isinstance(per_scenario, dict):
            per_scenario = data.get('scenarios') if isinstance(
                data.get('scenarios'), dict) else None

        n_curves = data.get('total_curves') or data.get('n_curves')
        if n_curves is None and per_scenario:
            try:
                n_curves = sum(int(v) for v in per_scenario.values())
            except (TypeError, ValueError):
                n_curves = None

        scenarios = data.get('scenario_names')
        if scenarios is None:
            scenarios = data.get('scenarios')
        if scenarios is None:
            scenarios = per_scenario
        if isinstance(scenarios, dict):
            n_scenarios = len(scenarios)
        elif isinstance(scenarios, (list, tuple)):
            n_scenarios = len(scenarios)
        else:
            n_scenarios = None

        if n_curves is not None or n_scenarios is not None:
            return (int(n_curves) if n_curves is not None else None,
                    n_scenarios, name)

    ground_truth = root / 'test_data' / 'ground_truth.csv'
    if ground_truth.exists():
        import csv
        try:
            with open(ground_truth, newline='') as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            rows = []
        if rows:
            scenarios = {r.get('scenario') for r in rows if r.get('scenario')}
            return len(rows), len(scenarios) or None, 'ground_truth.csv'

    return None, None, None


def describe_synthetic_suite(root=None):
    """``'<n> synthetic curves (<k> scenarios)'``, or an honest hedge."""
    n_curves, n_scenarios, source = synthetic_suite_counts(root)
    if n_curves is None and n_scenarios is None:
        return "the synthetic curves (suite size unknown: no MANIFEST.json)"
    if n_curves is None:
        return f"the synthetic curves ({n_scenarios} scenarios, per {source})"
    text = f"{n_curves} synthetic curves"
    if n_scenarios:
        text += f" ({n_scenarios} scenarios)"
    return text


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


def resolve_step9_modes(config, **kwargs):
    """
    Which truncation modes step 9 benchmarks, in run order.

    ``--all-modes`` runs both. Otherwise an explicit ``--adaptive`` /
    ``--no-adaptive`` on the command line picks one, and with neither the
    headline mode (first peak) is used. ``truncation.use_adaptive`` is
    deliberately *not* consulted: it is the batch default for the real plates,
    and letting it reach step 9 is how a config flag silently replaced the
    published first-peak headline with adaptive numbers (M14).
    """
    if kwargs.get('all_modes'):
        return list(STEP9_TRUNCATION_MODES)
    override = kwargs.get('adaptive')
    if override is None:
        return [HEADLINE_TRUNCATION_MODE]
    return ['adaptive' if override else 'first_peak']


def step9_run_dir(mode, **kwargs):
    """``<root>/validation/<mode>/`` -- one directory per truncation mode."""
    return synthetic_root(**kwargs) / 'validation' / mode


def step9_ml_run_dir(mode, n_modes, **kwargs):
    """
    Where the opt-in in-sample ML pass writes.

    ``validation_latest_ml/`` for a single mode; ``validation_latest_ml/<mode>/``
    when ``--all-modes`` asked for more than one, so neither mode's ML pass can
    overwrite the other's.
    """
    base = synthetic_root(**kwargs) / 'validation_latest_ml'
    return base if n_modes == 1 else base / mode


def _analysis_cmd(mode, data_dir, out_dir, ml_classify, **kwargs):
    """The 01_growth_curve_analysis.py command line for one benchmark pass."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / '01_growth_curve_analysis.py'),
        str(data_dir),
        '-o', str(out_dir),
        '-q',
    ]
    cmd += config_args(**kwargs)
    # Spell the mode out in full instead of leaving half of it to config.yaml:
    # --no-adaptive alone still lets truncation.use_first_peak decide between
    # first-peak and global-max, and the run_settings.json sidecar (and so the
    # report header) would then name a mode nobody asked for.
    if mode == 'adaptive':
        cmd.append('--adaptive')
    else:
        cmd += ['--no-adaptive', '--first-peak']
    if ml_classify:
        cmd.append('--ml-classify')
    return cmd


def _score_run(out_dir, data_dir, ground_truth, mode, dry_run=False):
    """Score one finished pipeline run with WP7's ``run_benchmark``."""
    print(f"\n{'='*60}")
    print(f"  STEP 9: Score the {mode} run against ground truth")
    print(f"{'='*60}")
    print(f"  run_benchmark(results_csv={out_dir / 'processing_results.csv'}, "
          f"ground_truth_csv={ground_truth}, out_dir={out_dir}, "
          f"truncation_mode={mode!r})")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    results_csv = out_dir / 'processing_results.csv'
    if not results_csv.exists():
        print(f"  FAILED: {results_csv} was not written")
        return False

    # Agg before the benchmark imports pyplot for its comparison figures.
    os.environ.setdefault('MPLBACKEND', 'Agg')
    synth_pkg = str(PROJECT_DIR / 'synthetic_data')
    if synth_pkg not in sys.path:
        sys.path.insert(0, synth_pkg)
    try:
        from validation.benchmark import run_benchmark
    except ImportError as exc:
        print(f"  FAILED: cannot import the benchmark ({exc})")
        return False

    start = time.time()
    try:
        report = run_benchmark(
            results_csv=results_csv,
            ground_truth_csv=ground_truth,
            out_dir=out_dir,
            truncation_mode=mode,
            data_dir=data_dir,
            verbose=False,
        )
    except Exception as exc:
        print(f"  FAILED after {time.time() - start:.1f}s: {type(exc).__name__}: {exc}")
        return False

    # Every printed number names its mode; the headline carries seed, dataset
    # digest and git SHA so it cannot be quoted without its provenance.
    print(f"  Completed in {time.time() - start:.1f}s")
    print(f"  [{mode}] {report.get('headline', '')}")
    print(f"  [{mode}] report: {out_dir / 'report.json'}")
    return True


def _benchmark_pass(mode, out_dir, label, ml_classify, dry_run=False, **kwargs):
    """Run the pipeline over the synthetic curves and score it. One mode."""
    data_dir = synthetic_test_data_dir(**kwargs) / 'DATA'
    ground_truth = synthetic_test_data_dir(**kwargs) / 'ground_truth.csv'

    ok = run_step(
        _analysis_cmd(mode, data_dir, out_dir, ml_classify, **kwargs),
        label,
        dry_run=dry_run,
    )
    if not ok:
        return False
    return _score_run(out_dir, data_dir, ground_truth, mode, dry_run=dry_run)


def _in_sample_ml_note(config, **kwargs):
    """Why the in-sample ML pass is not being run. Exactly one line, always."""
    if not ml_requested(config, **kwargs):
        reason = ("ML classification is off, and the rule-based benchmark is "
                  "the headline")
    elif not ml_models_exist():
        reason = "ML is enabled but models/*.joblib are missing"
    else:
        reason = ("pass --allow-in-sample-ml to run it; the synthetic curves are "
                  "the classifier's own training set")
    return f"  step 9b: in-sample ML pass skipped ({reason})"


def step9_validate(config, dry_run=False, **kwargs):
    """
    Step 9: score the pipeline against the synthetic ground truth.

    One run directory per truncation mode (``validation/first_peak/``,
    ``validation/adaptive/``) so a run in one mode can never overwrite the
    other's numbers, and the mode is named on every line this step prints.

    The benchmark is always rule-based. The ML classifier was trained on these
    exact curves, so an ML pass measures the model on its own training rows
    (measured: accuracy 0.975 / precision 1.000 / recall 0.965 in-sample versus
    0.873 / 0.901 / 0.925 rule-based on the same curves, M26). It runs only
    when ML is resolved on *and* ``--allow-in-sample-ml`` is given, into its own
    directory, under a banner saying what it is.
    """
    if not synthetic_data_exists(**kwargs):
        print("\n  Step 9: No synthetic test data found, skipping validation")
        return True

    modes = resolve_step9_modes(config, **kwargs)
    suite = describe_synthetic_suite(synthetic_root(**kwargs))

    print(f"\n  Step 9: {suite}")
    print(f"  Truncation modes: {', '.join(modes)}"
          f"{'  (headline: ' + HEADLINE_TRUNCATION_MODE + ')' if len(modes) > 1 else ''}")

    all_ok = True
    for mode in modes:
        out_dir = step9_run_dir(mode, **kwargs)
        ok = _benchmark_pass(
            mode, out_dir,
            f"STEP 9a [{mode}]: Run pipeline on {suite} → {out_dir}",
            ml_classify=False, dry_run=dry_run, **kwargs)
        all_ok = all_ok and ok

    if resolve_use_ml(config, **kwargs) and kwargs.get('allow_in_sample_ml'):
        for mode in modes:
            ml_dir = step9_ml_run_dir(mode, len(modes), **kwargs)
            print("\n  step 9b: ML pass is IN-SAMPLE (models were trained on "
                  "these curves)")
            print(f"  step 9b: [{mode}] writing to {ml_dir}")
            ok = _benchmark_pass(
                mode, ml_dir,
                f"STEP 9b [{mode}]: IN-SAMPLE ML pass over {suite} → {ml_dir}",
                ml_classify=True, dry_run=dry_run, **kwargs)
            all_ok = all_ok and ok
    else:
        print(_in_sample_ml_note(config, **kwargs))

    return all_ok


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
    (9, step9_validate,         f"Synthetic validation, {describe_synthetic_suite()}",
                                                                      False, False),
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
     first-peak truncation by default -- the published headline; --all-modes
     adds adaptive in its own directory. Rule-based unless --allow-in-sample-ml.

Examples:
  python run_full_pipeline.py                     # Run everything
  python run_full_pipeline.py --steps 2,3,4       # Core + Haldane only
  python run_full_pipeline.py --from-step 4       # Resume from Haldane
  python run_full_pipeline.py --skip-advanced      # Skip GP/Ensemble (steps 5,7)
  python run_full_pipeline.py --no-ml              # Rule-based only (no ML)
  python run_full_pipeline.py --no-adaptive        # First-peak truncation this run
  python run_full_pipeline.py --force-preprocess   # Rebuild all groups from raw
  python run_full_pipeline.py --dry-run            # Preview without running
  python run_full_pipeline.py --steps 9 --all-modes  # Both truncation modes, one dir each
  python run_full_pipeline.py --steps 9 --allow-in-sample-ml  # Extra IN-SAMPLE ML pass
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
    parser.add_argument('--all-modes', action='store_true',
                        help='Step 9: benchmark every truncation mode '
                             f'({", ".join(STEP9_TRUNCATION_MODES)}), each in its own '
                             'run directory. Without it step 9 runs the headline mode '
                             f'({HEADLINE_TRUNCATION_MODE}) unless --adaptive/--no-adaptive says otherwise')
    parser.add_argument('--allow-in-sample-ml', action='store_true',
                        help='Step 9: additionally run the ML classifier over the synthetic '
                             'curves. These curves are the classifier\'s training set, so the '
                             'numbers are IN-SAMPLE; they are written to their own directory '
                             'and never replace the rule-based benchmark')
    parser.add_argument('--synthetic-out-dir', type=str, default=None,
                        help='Step 9: root holding test_data/ and the per-mode run '
                             f'directories (default: {SYNTHETIC_ROOT})')
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
        all_modes=args.all_modes,
        allow_in_sample_ml=args.allow_in_sample_ml,
        synthetic_out_dir=args.synthetic_out_dir,
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
    if 9 in run_steps:
        modes = resolve_step9_modes(config, **step_kwargs)
        ml_note = ("IN-SAMPLE ML pass requested"
                   if (resolve_use_ml(config, **step_kwargs)
                       and args.allow_in_sample_ml) else "rule-based")
        print(f"  Step 9:       {', '.join(modes)} truncation, {ml_note}")
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
