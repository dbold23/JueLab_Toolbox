#!/usr/bin/env python3
"""
Generate Comprehensive Synthetic Test Set

Generates a full test suite covering all scenarios to stress-test
the TECAN growth curve analysis pipeline.

This script generates ~500 curves covering:
- All good growth scenarios (7 types)
- All bad curve scenarios (5 types)
- All edge case scenarios (12 types)
- Various noise levels and model types

Usage:
    python generate_comprehensive.py -o output/comprehensive_test/
    python generate_comprehensive.py --n-per-scenario 30 -o output/large_test/
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import SyntheticGrowthCurveGenerator
from src.output_formatter import BatchExporter
from src.curve_scenarios import (
    GOOD_GROWTH_SCENARIOS,
    BAD_CURVE_SCENARIOS,
    EDGE_CASE_SCENARIOS,
    get_comprehensive_test_config
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive synthetic test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates a comprehensive test set covering all scenarios
defined in curve_scenarios.py.

Scenarios include:
  GOOD (7): standard, high_A, low_A, fast_growth, slow_growth, short_lag, long_lag
  BAD (8): flat_no_growth, minimal_growth, contamination, high_noise, etc.
  EDGE (12): borderline_r2, truncation_challenge, death_phase, etc.

Examples:
    # Default comprehensive test (~400 curves)
    python generate_comprehensive.py -o output/comprehensive/

    # Larger test set (30 curves per scenario, ~700 curves)
    python generate_comprehensive.py --n-per-scenario 30 -o output/large/

    # Quick validation set (5 per scenario, ~120 curves)
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
        default=15,
        help='Number of curves per scenario (default: 15)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--extra-good',
        type=int,
        default=0,
        help='Extra curves to add to standard/typical good scenarios'
    )

    parser.add_argument(
        '--extra-borderline',
        type=int,
        default=0,
        help='Extra curves to add to borderline scenarios (for threshold testing)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Base curves per scenario: {args.n_per_scenario}")
    print(f"Random seed: {args.seed}")

    # Build custom configuration
    config = {}

    # Good scenarios
    print("\nGOOD GROWTH SCENARIOS:")
    for name in GOOD_GROWTH_SCENARIOS:
        n = args.n_per_scenario
        if name in ['standard', 'very_clean']:
            n += args.extra_good
        config[name] = n
        print(f"  {name}: {n} curves")

    # Bad scenarios
    print("\nBAD CURVE SCENARIOS:")
    for name in BAD_CURVE_SCENARIOS:
        n = args.n_per_scenario
        config[name] = n
        print(f"  {name}: {n} curves")

    # Edge cases
    print("\nEDGE CASE SCENARIOS:")
    for name in EDGE_CASE_SCENARIOS:
        n = args.n_per_scenario
        if 'borderline' in name:
            n += args.extra_borderline
        config[name] = n
        print(f"  {name}: {n} curves")

    total_curves = sum(config.values())
    print(f"\nTotal curves to generate: {total_curves}")

    # Generate curves
    print("\n" + "-" * 60)
    print("Generating curves...")
    print("-" * 60)

    generator = SyntheticGrowthCurveGenerator(seed=args.seed)
    curves_df = generator.generate_comprehensive_test_set(config=config, seed=args.seed)

    # Export
    print("\n" + "-" * 60)
    print("Exporting data...")
    print("-" * 60)

    exporter = BatchExporter(str(output_dir))
    result = exporter.export_comprehensive_test(curves_df, test_name='test_data')

    # Create summary
    summary = {
        'total_curves': len(curves_df),
        'expected_good': len(curves_df[curves_df['expected_class'] == 'GOOD']),
        'expected_bad': len(curves_df[curves_df['expected_class'] == 'BAD']),
        'scenarios': curves_df['scenario'].value_counts().to_dict(),
        'models_used': curves_df['model_type'].value_counts().to_dict(),
        'seed': args.seed,
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

    print("\n" + "-" * 60)
    print("Scenario breakdown:")
    print("-" * 60)
    for scenario, count in sorted(summary['scenarios'].items()):
        expected = GOOD_GROWTH_SCENARIOS.get(scenario,
                   BAD_CURVE_SCENARIOS.get(scenario,
                   EDGE_CASE_SCENARIOS.get(scenario)))
        if expected:
            exp_class = expected.expected_class
            print(f"  {scenario:30s} {count:4d} curves  [{exp_class}]")

    print("\n" + "=" * 60)
    print("Done! Data ready for analysis pipeline validation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
