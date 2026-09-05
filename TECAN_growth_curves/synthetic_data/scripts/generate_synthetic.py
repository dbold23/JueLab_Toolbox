#!/usr/bin/env python3
"""
Generate Synthetic Growth Curve Data

CLI tool to generate synthetic bacterial growth curves for testing
the analysis pipeline.

Usage:
    # Generate from scenario
    python generate_synthetic.py --scenario standard --n-curves 50

    # Generate comprehensive test set
    python generate_synthetic.py --comprehensive

    # Generate with specific parameters
    python generate_synthetic.py --A 1.5 --mu 0.2 --lambda 3.0 --n-curves 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import (
    SyntheticGrowthCurveGenerator,
    generate_quick_test_set
)
from src.output_formatter import TECANFormatWriter, BatchExporter
from src.curve_scenarios import list_scenarios, ALL_SCENARIOS


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic bacterial growth curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 50 standard growth curves
    python generate_synthetic.py --scenario standard --n-curves 50 -o output/

    # Generate comprehensive test set (~400 curves)
    python generate_synthetic.py --comprehensive -o output/comprehensive/

    # Generate curves with specific parameters
    python generate_synthetic.py --A 1.5 --mu 0.2 --lambda 3.0 --n-curves 20

    # List available scenarios
    python generate_synthetic.py --list-scenarios

    # Quick test set
    python generate_synthetic.py --quick-test -o output/quick_test/
        """
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory (default: output/)'
    )

    # Generation modes
    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        '--scenario',
        type=str,
        help='Generate from a specific scenario (use --list-scenarios to see options)'
    )

    mode_group.add_argument(
        '--comprehensive',
        action='store_true',
        help='Generate comprehensive test set covering all scenarios'
    )

    mode_group.add_argument(
        '--quick-test',
        action='store_true',
        help='Generate quick test set (5 curves per category)'
    )

    mode_group.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List all available scenarios and exit'
    )

    # Manual parameter specification
    parser.add_argument('--A', type=float, help='Maximum OD600 (asymptotic value)')
    parser.add_argument('--mu', type=float, help='Maximum growth rate (OD/hour)')
    parser.add_argument('--lambda', dest='lambda_', type=float, help='Lag phase duration (hours)')

    # Generation options
    parser.add_argument(
        '--n-curves',
        type=int,
        default=10,
        help='Number of curves to generate (default: 10)'
    )

    parser.add_argument(
        '--n-per-scenario',
        type=int,
        default=15,
        help='Curves per scenario for comprehensive mode (default: 15)'
    )

    parser.add_argument(
        '--model',
        choices=['gompertz', 'baranyi', 'logistic', 'richards'],
        default='gompertz',
        help='Growth model to use (default: gompertz)'
    )

    parser.add_argument(
        '--noise',
        choices=['very_low', 'low', 'medium', 'high', 'very_high'],
        default='medium',
        help='Noise level (default: medium)'
    )

    parser.add_argument(
        '--include-death-phase',
        action='store_true',
        help='Include death/decline phase in curves'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=100.0,
        help='Experiment duration in hours (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Handle --list-scenarios
    if args.list_scenarios:
        list_scenarios()
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SyntheticGrowthCurveGenerator(seed=args.seed)

    if args.verbose:
        print(f"Output directory: {output_dir}")
        print(f"Random seed: {args.seed}")

    # Generate based on mode
    if args.comprehensive:
        print("Generating comprehensive test set...")

        # Custom config with specified n_per_scenario
        from src.curve_scenarios import get_comprehensive_test_config
        config = get_comprehensive_test_config()

        # Scale to requested size
        scale = args.n_per_scenario / 15  # Default is 15
        config = {k: max(1, int(v * scale)) for k, v in config.items()}

        curves_df = generator.generate_comprehensive_test_set(config=config, seed=args.seed)

        # Export
        exporter = BatchExporter(str(output_dir))
        result = exporter.export_comprehensive_test(curves_df, test_name='comprehensive')

        print(f"\nGenerated {len(curves_df)} curves")
        print(f"Output directory: {result['test_dir']}")

    elif args.quick_test:
        print("Generating quick test set...")
        curves_df = generate_quick_test_set(n_per_category=5, seed=args.seed)

        # Export
        writer = TECANFormatWriter(str(output_dir))
        result = writer.export_for_analysis_pipeline(curves_df)

        print(f"\nGenerated {len(curves_df)} curves")
        print(f"Output directory: {output_dir}")

    elif args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Error: Unknown scenario '{args.scenario}'")
            print("Use --list-scenarios to see available options")
            sys.exit(1)

        print(f"Generating {args.n_curves} curves from scenario: {args.scenario}")

        curves = generator.generate_from_scenario(
            args.scenario,
            n_curves=args.n_curves,
            seed=args.seed
        )

        # Convert to DataFrame for export
        rows = []
        for i, curve in enumerate(curves):
            rows.append({
                'curve_id': i,
                'scenario': args.scenario,
                'expected_class': curve.metadata.get('expected_class', 'UNKNOWN'),
                'true_A': curve.parameters['A'],
                'true_mu': curve.parameters['mu'],
                'true_lambda': curve.parameters['lambda_'],
                'model_type': curve.parameters['model_type'],
                'initial_od': curve.parameters['initial_od'],
                'noise_level': curve.metadata.get('noise_level'),
                'actual_r_squared': curve.metadata.get('actual_r_squared'),
                'rmse': curve.metadata.get('rmse'),
                'max_od': curve.metadata.get('max_od'),
                'delta_od': curve.metadata.get('delta_od'),
                'n_points': len(curve.time),
                'time': curve.time.tolist(),
                'od600': curve.od600.tolist(),
            })

        import pandas as pd
        curves_df = pd.DataFrame(rows)

        # Export
        writer = TECANFormatWriter(str(output_dir))
        result = writer.export_for_analysis_pipeline(curves_df)

        print(f"\nGenerated {len(curves)} curves")
        print(f"Output directory: {output_dir}")

    elif args.A is not None and args.mu is not None and args.lambda_ is not None:
        print(f"Generating {args.n_curves} curves with specified parameters...")
        print(f"  A={args.A}, mu={args.mu}, lambda={args.lambda_}")

        curves = []
        for i in range(args.n_curves):
            curve = generator.generate_single_curve(
                A=args.A,
                mu=args.mu,
                lambda_=args.lambda_,
                model_type=args.model,
                noise_level=args.noise,
                include_death_phase=args.include_death_phase,
                duration_hours=args.duration,
                seed=args.seed + i
            )
            curves.append(curve)

        # Convert to DataFrame
        rows = []
        for i, curve in enumerate(curves):
            rows.append({
                'curve_id': i,
                'scenario': 'custom',
                'expected_class': 'GOOD',
                'true_A': curve.parameters['A'],
                'true_mu': curve.parameters['mu'],
                'true_lambda': curve.parameters['lambda_'],
                'model_type': curve.parameters['model_type'],
                'initial_od': curve.parameters['initial_od'],
                'noise_level': curve.metadata.get('noise_level'),
                'actual_r_squared': curve.metadata.get('actual_r_squared'),
                'rmse': curve.metadata.get('rmse'),
                'max_od': curve.metadata.get('max_od'),
                'delta_od': curve.metadata.get('delta_od'),
                'n_points': len(curve.time),
                'time': curve.time.tolist(),
                'od600': curve.od600.tolist(),
            })

        import pandas as pd
        curves_df = pd.DataFrame(rows)

        # Export
        writer = TECANFormatWriter(str(output_dir))
        result = writer.export_for_analysis_pipeline(curves_df)

        print(f"\nGenerated {len(curves)} curves")
        print(f"Output directory: {output_dir}")

    else:
        print("Error: Must specify --scenario, --comprehensive, --quick-test, or manual parameters (--A, --mu, --lambda)")
        print("Use --help for more information")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
