#!/usr/bin/env python3
"""
Generate Comprehensive Synthetic Test Set

Generates a full test suite covering all scenarios to stress-test
the TECAN growth curve analysis pipeline.

The composition is the catalog's, not this script's: every scenario carries its
own ``n_default`` (curve_scenarios.py), so "how many curves of what" is part of
the definition a reader can check, instead of the ``--extra-good`` /
``--extra-borderline`` flags this script used to apply to names matching a
substring.  ``--weighting lab`` re-allocates the same budget to the measured
composition of the real plates (config/lab_prior.json, derived by
scripts/derive_lab_prior.py); ``uniform`` (the default) keeps the catalog's own
counts so regression runs stay comparable.

This script generates ~1050 curves covering:
- All good growth scenarios (10 types)
- All bad curve scenarios (15 types)
- All edge case scenarios (39 types)
- Various noise levels, artefacts, shapes and model types

Before anything is written the suite goes through a REALISATION GATE
(``curve_scenarios.audit_realisation``): every artefact, per-well sigma and
dead-well fraction a scenario declares has to be measurable in the curves that
were produced.  A ScenarioConfig field is a request to the generator, and a
request nothing consumes is dropped silently - which is how four noise rungs
once ran at one preset and 35 curves labelled BAD for "two of three wells are
dead" shipped as ordinary growers.  Generation now stops instead, and names the
field and the file that has to read it.

Every run writes MANIFEST.json next to the data: the seed, the per-scenario
counts, the sha256 of every emitted CSV, a hash of the scenario definitions
themselves, the git SHA, the numpy version, the truncation mode the headline is
to be measured under and the exact analysis command to reproduce it.  A number
published without that block cannot be attributed to anything.

Usage:
    python generate_comprehensive.py -o output/comprehensive_test/
    python generate_comprehensive.py --n-per-scenario 30 -o output/large_test/
    python generate_comprehensive.py --weighting lab -o output/lab_weighted/
    python generate_comprehensive.py --emit-plate -o output/with_plates/
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import SyntheticGrowthCurveGenerator
from src.output_formatter import BatchExporter
from src.plate_writer import PlateFormatWriter
from src.curve_scenarios import (
    ALL_SCENARIOS,
    GOOD_GROWTH_SCENARIOS,
    BAD_CURVE_SCENARIOS,
    EDGE_CASE_SCENARIOS,
    audit_realisation,
    get_comprehensive_test_config,
    get_lab_weighted_config,
    load_lab_prior,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Truncation modes the headline may be measured under.  The committed 87.3 %
#: was produced with --no-adaptive (first_peak) while config.yaml sets
#: use_adaptive: true, so a rerun of `make step-9` silently overwrote a
#: first-peak headline with adaptive results.  The mode is recorded here and
#: the manifest prints the command that reproduces it.
TRUNCATION_MODES = {
    'first_peak': '--no-adaptive',
    'adaptive': '--adaptive',
    'global_max': '--no-adaptive --global-max',
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scenario_definitions_sha256() -> str:
    """Hash of the scenario catalog itself.

    Over ``repr`` of the sorted (name, ScenarioConfig) pairs, so a changed
    parameter range invalidates the dataset's provenance even when the counts
    and the seed are identical.
    """
    payload = repr(sorted(ALL_SCENARIOS.items(), key=lambda kv: kv[0]))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def git_sha() -> dict:
    """Current commit and whether the tree is dirty (never fatal)."""
    def run(*args):
        try:
            return subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError):    # pragma: no cover
            return ''
    sha = run('git', 'rev-parse', 'HEAD')
    dirty = bool(run('git', 'status', '--porcelain'))
    return {'git_sha': sha or None, 'git_dirty': dirty}


def class_balance(curves_df) -> dict:
    """GOOD share of the suite, next to the lab's own prevalence."""
    good = int((curves_df['expected_class'] == 'GOOD').sum())
    total = int(len(curves_df))
    return {
        'n_curves': total,
        'n_good': good,
        'n_bad': total - good,
        'good_fraction': round(good / total, 4) if total else None,
    }


def check_realisation(curves_df, allow: bool) -> list:
    """
    Refuse to write a suite whose scenarios declare more than the data carries.

    A ScenarioConfig field is a REQUEST to the generator.  When nothing
    consumes it the request is dropped silently and the ground truth then
    justifies its labels with a property the curves do not have - the four
    ``noise_ladder`` rungs all ran at one preset, and 35 curves labelled BAD
    for "two of three wells are dead" were ordinary growers.  Rather than ship
    that, generation stops and names the field and its consumer.
    """
    problems = audit_realisation(curves_df)
    if not problems:
        print("  Realisation gate: every declared artefact is in the data.")
        return problems
    print("\n" + "!" * 60)
    print("REALISATION GATE FAILED - declared but not realised:")
    for problem in problems:
        print(f"  - {problem}")
    print("!" * 60)
    if allow:
        print("--allow-unrealised given: writing the suite anyway.  Do NOT "
              "publish a headline from it; the labels of the scenarios above "
              "are not supported by their curves.")
        return problems
    raise SystemExit(
        "\nRefusing to write a suite whose labels the curves do not support.\n"
        "Fix the generator (synthetic_data/src/data_generator.py must read the\n"
        "Contract A fields listed above) or pass --allow-unrealised for an\n"
        "exploratory run that must not be published.")


def _range_text(pair) -> str:
    lo, hi = float(pair[0]), float(pair[1])
    return f"{lo:g}-{hi:g}" if lo != hi else f"{lo:g}"


def readme_tables() -> str:
    """The three ``## Scenario Coverage`` tables of synthetic_data/README.md.

    Rendered from the catalog rather than maintained by hand: the README went
    stale the moment the catalog grew (it documented 9 GOOD, 8 BAD and 20 edge
    scenarios against 10/15/39, and tests/test_audit_fixes_ml.py asserts the
    two agree).  WP8 owns the README; this command gives it the exact block to
    paste between ``## Scenario Coverage`` and ``## Noise Models``.
    """
    out = []
    out.append('### Good Growth Scenarios (Expected: GOOD classification)\n')
    out.append(f'{len(GOOD_GROWTH_SCENARIOS)} scenarios, defined in '
               '`src/curve_scenarios.py` (`GOOD_GROWTH_SCENARIOS`).\n')
    out.append('| Scenario | A Range | \u03bc Range | \u03bb Range | n | Description |')
    out.append('|----------|---------|---------|---------|---|-------------|')
    for name, cfg in GOOD_GROWTH_SCENARIOS.items():
        out.append(f'| {name} | {_range_text(cfg.A)} | {_range_text(cfg.mu)} | '
                   f'{_range_text(cfg.lambda_)} | {cfg.n_default} | {cfg.description} |')
    out.append('')
    out.append('### Bad Curve Scenarios (Expected: BAD classification)\n')
    out.append(f'{len(BAD_CURVE_SCENARIOS)} scenarios, defined in '
               '`src/curve_scenarios.py` (`BAD_CURVE_SCENARIOS`).\n')
    out.append('| Scenario | n | Description |')
    out.append('|----------|---|-------------|')
    for name, cfg in BAD_CURVE_SCENARIOS.items():
        out.append(f'| {name} | {cfg.n_default} | {cfg.description} |')
    out.append('')
    out.append('### Edge Case Scenarios (Challenging cases)\n')
    out.append(f'{len(EDGE_CASE_SCENARIOS)} scenarios, defined in '
               '`src/curve_scenarios.py` (`EDGE_CASE_SCENARIOS`).')
    out.append('Edge cases carry their own expected class -- most of them are curves the')
    out.append('pipeline is expected to handle correctly and classify GOOD.  A scenario')
    out.append('with `label_rule=oracle` (the noise ladder, the borderline-noise pair and')
    out.append('the haldane trio) has its class decided per curve by whether its')
    out.append('parameters are recoverable from the curve that was written, so the column')
    out.append("below is the scenario's intent, not a guarantee about every row.\n")
    out.append('| Scenario | Expected | n | Description |')
    out.append('|----------|----------|---|-------------|')
    for name, cfg in EDGE_CASE_SCENARIOS.items():
        out.append(f'| {name} | {cfg.expected_class} | {cfg.n_default} | '
                   f'{cfg.description} |')
    out.append('')
    total = sum(cfg.n_default for cfg in ALL_SCENARIOS.values())
    out.append(f'{len(ALL_SCENARIOS)} scenarios and {total} requested curves in total;')
    out.append('artefact scenarios add an artefact-free twin per curve, so the default')
    out.append('suite is larger than that (about 1140 rows at seed 42).  Its expected-GOOD')
    out.append('share is held in the lab band, 49-60 % - the 92 real conditions are 30.4 %')
    out.append('dead and 13.0 % trace (`config/lab_prior.json`) - against 71.9 % GOOD in')
    out.append('the benchmark this replaces.  MANIFEST.json records the realised numbers')
    out.append('of the run that produced the published headline.')
    out.append('')
    return '\n'.join(out)


def build_config(args) -> dict:
    """Curves per scenario, from the catalog or from the lab prior."""
    if args.weighting == 'lab':
        prior = load_lab_prior(args.lab_prior) if args.lab_prior else load_lab_prior()
        total = args.total_curves or sum(
            get_comprehensive_test_config(args.n_per_scenario).values())
        return get_lab_weighted_config(total, prior=prior,
                                       min_per_scenario=args.min_per_scenario)
    return get_comprehensive_test_config(args.n_per_scenario)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive synthetic test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates a comprehensive test set covering all scenarios
defined in curve_scenarios.py.

Scenarios include:
  GOOD: standard, high_A, low_A, fast_growth, slow_growth, short_lag,
        long_lag, very_clean, pesticide_lb_typical, high_baseline_good
  BAD:  flat_no_growth, minimal_growth, contamination, contamination_offset,
        clipped_baseline, all_zero, negative_drift, replicate_outlier_high,
        replicate_dropout, contaminated_blank, high_noise, ...
  EDGE: borderline_r2_good/bad, truncation_challenge, death_phase_*,
        noise_ladder_1..4, incomplete_24h, incomplete_growth_*,
        late_growth_still_rising, run_ends_at_plateau_edge,
        slow_secondary_drift, diauxic_second_larger, clipped_then_grows,
        edge_well_evaporation, high_autocorr_good, partial_lysis,
        sudden_crash, dip_and_recover, haldane_*, evaporation_drift,
        bubble_spike, ...

A scenario that declares an artefact also yields an artefact-free TWIN of every
curve (same seed, same noise realisation), so the validator can measure the
shift the artefact caused; twins are extra rows and are cross-referenced by
twin_curve_id.

Examples:
    # Default comprehensive test (catalog counts, ~1100 curves with twins)
    python generate_comprehensive.py -o output/comprehensive/

    # Uniform 30 per scenario
    python generate_comprehensive.py --n-per-scenario 30 -o output/large/

    # Lab-weighted composition (for ML training sets)
    python generate_comprehensive.py --weighting lab -o output/lab/

    # Quick validation set (5 per scenario)
    python generate_comprehensive.py --n-per-scenario 5 -o output/quick/
        """
    )

    parser.add_argument(
        '-o', '--output',
        default='output/comprehensive_test',
        help='Output directory (default: output/comprehensive_test/)'
    )

    parser.add_argument(
        '--n-per-scenario',
        type=int,
        default=None,
        help='Override every scenario n_default with this uniform count'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--weighting',
        choices=('uniform', 'lab'),
        default='uniform',
        help="'uniform' uses each scenario's n_default (default); 'lab' "
             "re-allocates the budget to the measured delta-OD composition of "
             "the real plates (config/lab_prior.json)"
    )

    parser.add_argument(
        '--total-curves',
        type=int,
        default=None,
        help='Target suite size for --weighting lab (default: the uniform total)'
    )

    parser.add_argument(
        '--min-per-scenario',
        type=int,
        default=5,
        help='Floor for --weighting lab so no scenario leaves the headline (default: 5)'
    )

    parser.add_argument(
        '--lab-prior',
        default=None,
        help='Path to a lab_prior.json (default: synthetic_data/config/lab_prior.json)'
    )

    parser.add_argument(
        '--truncation-mode',
        choices=sorted(TRUNCATION_MODES),
        default='first_peak',
        help='Truncation mode the headline is to be measured under; recorded '
             'in MANIFEST.json together with the exact analysis command '
             '(default: first_peak)'
    )

    parser.add_argument(
        '--emit-plate',
        action='store_true',
        help='Also write the curves as TECAN plates (PLATES/plate_NN/) so the '
             'benchmark can go through the real preprocessing stage'
    )

    parser.add_argument(
        '--test-name',
        default='test_data',
        help='Sub-directory holding DATA/ and ground_truth.csv (default: test_data)'
    )

    parser.add_argument(
        '--allow-unrealised',
        action='store_true',
        help='Write the suite even when a scenario declares an artefact the '
             'generator did not produce (exploratory runs only; such a suite '
             'must not be published)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--print-readme-tables',
        action='store_true',
        help='Print the scenario tables for synthetic_data/README.md and exit '
             '(the README is rendered from the catalog, not maintained by hand)'
    )

    args = parser.parse_args()

    if args.print_readme_tables:
        print(readme_tables())
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 60)
    print("COMPREHENSIVE SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Weighting: {args.weighting}")
    print(f"Random seed: {args.seed}")

    config = build_config(args)

    for label, group in (("GOOD GROWTH SCENARIOS", GOOD_GROWTH_SCENARIOS),
                         ("BAD CURVE SCENARIOS", BAD_CURVE_SCENARIOS),
                         ("EDGE CASE SCENARIOS", EDGE_CASE_SCENARIOS)):
        print(f"\n{label}:")
        for name in group:
            print(f"  {name}: {config[name]} curves")

    requested = sum(config.values())
    print(f"\nCurves requested: {requested} "
          f"(artefact scenarios add an artefact-free twin per curve)")

    # Generate curves
    print("\n" + "-" * 60)
    print("Generating curves...")
    print("-" * 60)

    generator = SyntheticGrowthCurveGenerator(seed=args.seed)
    curves_df = generator.generate_comprehensive_test_set(config=config, seed=args.seed,
                                                         verbose=args.verbose)

    # ------------------------------------------------- declared vs realised
    print("\n" + "-" * 60)
    print("Checking that the data carries what the scenarios declare...")
    print("-" * 60)
    unrealised = check_realisation(curves_df, allow=args.allow_unrealised)

    balance = class_balance(curves_df)
    prior = load_lab_prior(args.lab_prior) if args.lab_prior else load_lab_prior()
    print(f"  Class balance: {balance['n_good']}/{balance['n_curves']} GOOD "
          f"({balance['good_fraction']:.1%}); over the {prior['n_conditions']} "
          f"real conditions the GOOD share is 49-60 %, depending on where the "
          f"marginal 0.1-0.3 OD band falls.")

    # Export
    print("\n" + "-" * 60)
    print("Exporting data...")
    print("-" * 60)

    exporter = BatchExporter(str(output_dir))
    result = exporter.export_comprehensive_test(curves_df, test_name=args.test_name)

    plates = None
    if args.emit_plate:
        writer = PlateFormatWriter(Path(result['data_dir']).parent)
        plates = writer.write_plates(curves_df)
        print(f"  Plates: {plates['plates_dir']} ({len(plates['plates'])} plates)")

    # Create summary
    summary = {
        'total_curves': len(curves_df),
        'expected_good': int((curves_df['expected_class'] == 'GOOD').sum()),
        'expected_bad': int((curves_df['expected_class'] == 'BAD').sum()),
        'scenarios': curves_df['scenario'].value_counts().to_dict(),
        'models_used': curves_df['model_type'].value_counts().to_dict(),
        'seed': args.seed,
        'weighting': args.weighting,
        'files': {
            'data_dir': str(result['data_dir']),
            'ground_truth': str(result['ground_truth']),
            'config': str(result.get('config', ''))
        }
    }

    # Save summary
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---------------------------------------------------------------- manifest
    data_dir = Path(result['data_dir'])
    ground_truth = Path(result['ground_truth'])
    files = {}
    for path in sorted(data_dir.glob('*.csv')):
        files[str(path.relative_to(output_dir))] = sha256_file(path)
    files[str(ground_truth.relative_to(output_dir))] = sha256_file(ground_truth)

    trunc_flags = TRUNCATION_MODES[args.truncation_mode]
    analysis_command = (
        f"MPLBACKEND=Agg python scripts/01_growth_curve_analysis.py "
        f"{data_dir} -o <results-dir> {trunc_flags} --no-plots -q"
    )

    manifest = {
        'schema': 'synthetic_manifest/1',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seed': args.seed,
        'weighting': args.weighting,
        'n_per_scenario': {name: int(config[name]) for name in sorted(config)},
        'scenarios': sorted(config),
        'n_scenarios': len(config),
        'n_curves_requested': requested,
        'n_curves_written': int(len(curves_df)),
        'n_twin_curves': int((curves_df['twin_curve_id'].astype(str) != '').sum())
                         if 'twin_curve_id' in curves_df.columns else 0,
        'expected_good': summary['expected_good'],
        'expected_bad': summary['expected_bad'],
        'good_fraction': balance['good_fraction'],
        'unrealised_declarations': unrealised,
        'scenario_definitions_sha256': scenario_definitions_sha256(),
        'lab_prior_sha256': (sha256_file(Path(args.lab_prior)) if args.lab_prior
                             else sha256_file(
                                 Path(__file__).resolve().parent.parent /
                                 'config' / 'lab_prior.json')),
        'files_sha256': files,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'truncation_mode': args.truncation_mode,
        'analysis_command': analysis_command,
        'generation_command': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
        'wall_clock_s': round(time.time() - started, 3),
    }
    manifest.update(git_sha())
    if plates is not None:
        manifest['plates'] = {
            'plates_dir': str(Path(plates['plates_dir']).relative_to(output_dir)),
            'n_plates': len(plates['plates']),
        }

    manifest_path = output_dir / 'MANIFEST.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')

    # Print final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total curves generated: {summary['total_curves']}")
    print(f"  Expected GOOD: {summary['expected_good']}")
    print(f"  Expected BAD: {summary['expected_bad']}")
    print(f"\nOutput files:")
    print(f"  Data directory: {result['data_dir']}")
    print(f"  Ground truth: {result['ground_truth']}")
    print(f"  Summary: {summary_path}")
    print(f"  Manifest: {manifest_path}")

    print("\n" + "-" * 60)
    print("Scenario breakdown:")
    print("-" * 60)
    for scenario, count in sorted(summary['scenarios'].items()):
        expected = ALL_SCENARIOS.get(scenario)
        if expected:
            print(f"  {scenario:30s} {count:4d} curves  [{expected.expected_class}]")

    print("\n" + "-" * 60)
    print(f"Headline is to be measured under truncation_mode="
          f"{args.truncation_mode}:")
    print(f"  {analysis_command}")
    print("=" * 60)


if __name__ == "__main__":
    main()
