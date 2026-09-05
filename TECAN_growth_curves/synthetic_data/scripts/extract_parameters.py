#!/usr/bin/env python3
"""
Extract Gompertz Parameters from Real Growth Curve Results

CLI tool to extract parameter distributions from good curves
in all_groups_results.csv or processing_results.csv.

Usage:
    python extract_parameters.py ../OUTPUT/all_groups_results.csv
    python extract_parameters.py ../OUTPUT/all_groups_results.csv -o config/extracted_params.yaml
    python extract_parameters.py ../OUTPUT/all_groups_results.csv --samples 20
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parameter_extractor import ParameterExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract Gompertz parameters from growth curve analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic extraction with summary
    python extract_parameters.py ../OUTPUT/all_groups_results.csv

    # Save to YAML config
    python extract_parameters.py ../OUTPUT/all_groups_results.csv -o params.yaml

    # Generate sample parameters
    python extract_parameters.py ../OUTPUT/all_groups_results.csv --samples 20
        """
    )

    parser.add_argument(
        'input',
        help='Path to all_groups_results.csv or processing_results.csv'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output YAML file path (default: extracted_params.yaml in config/)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=0,
        help='Generate N sample parameter sets and print them'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample generation (default: 42)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress summary output'
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create extractor
    extractor = ParameterExtractor(str(input_path))

    # Print summary unless quiet
    if not args.quiet:
        extractor.print_summary()

    # Save to YAML if output specified
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output location
        config_dir = Path(__file__).parent.parent / 'config'
        config_dir.mkdir(exist_ok=True)
        output_path = config_dir / 'extracted_params.yaml'

    extractor.save_to_yaml(str(output_path))

    # Generate sample parameters if requested
    if args.samples > 0:
        print(f"\nGenerated {args.samples} sample parameter sets:")
        print("-" * 60)
        samples = extractor.generate_sample_parameters(args.samples, seed=args.seed)
        print(samples.to_string(index=False))

    print(f"\nDone! Config saved to: {output_path}")


if __name__ == "__main__":
    main()
