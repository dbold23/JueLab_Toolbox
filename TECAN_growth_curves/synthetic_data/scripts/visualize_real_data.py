#!/usr/bin/env python3
"""
Visualize Real Data CLI

Generate visualizations of real growth curve data including
triplicate variance bands and parameter distributions.

Usage:
    python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv
    python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv --output-dir viz_output
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.triplicate_visualizer import TriplicateVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Visualize real TECAN growth curve data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization from pipeline results
    python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv

    # Save to custom output directory
    python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv --output-dir my_viz

    # Only plot good curves
    python scripts/visualize_real_data.py --results ../OUTPUT/all_groups_results.csv --good-only
        """
    )

    parser.add_argument(
        '--results', '-r',
        required=True,
        help='Path to all_groups_results.csv from analysis pipeline'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='real_data_visualizations',
        help='Output directory for plots (default: real_data_visualizations)'
    )

    parser.add_argument(
        '--good-only',
        action='store_true',
        help='Only visualize curves classified as good'
    )

    parser.add_argument(
        '--overlay-count',
        type=int,
        default=20,
        help='Number of curves to show in overlay plots (default: 20)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )

    args = parser.parse_args()

    # Validate input
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real Data Visualizer")
    print("=" * 60)
    print(f"\nInput: {results_path}")
    print(f"Output: {output_dir}")
    print(f"Good curves only: {args.good_only}")

    # Load data
    print("\nLoading data...")
    visualizer = TriplicateVisualizer(str(results_path))

    # Apply filter if requested
    if args.good_only:
        original_count = len(visualizer.results_df)
        visualizer.results_df = visualizer.results_df[
            visualizer.results_df['is_good'] == True
        ]
        print(f"Filtered to good curves: {len(visualizer.results_df)}/{original_count}")

    # Print summary
    print(f"\nData summary:")
    print(f"  Total curves: {len(visualizer.results_df)}")
    if 'is_good' in visualizer.results_df.columns:
        n_good = (visualizer.results_df['is_good'] == True).sum()
        n_bad = (visualizer.results_df['is_good'] == False).sum()
        print(f"  Good curves: {n_good}")
        print(f"  Bad curves: {n_bad}")

    print("\n" + "-" * 60)
    print("Generating visualizations...")
    print("-" * 60)

    # Generate plots
    plots_generated = []

    # 1. Parameter distributions
    try:
        print("\n1. Parameter distributions...")
        fig = visualizer.plot_parameter_distributions(
            output_path=str(output_dir / 'parameter_distributions.png')
        )
        plots_generated.append('parameter_distributions.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Could not generate parameter distributions: {e}")

    # 2. Good curves overlay
    try:
        print("\n2. Good curves overlay...")
        fig = visualizer.plot_good_curves_overlay(
            n_curves=args.overlay_count,
            output_path=str(output_dir / 'good_curves_overlay.png')
        )
        plots_generated.append('good_curves_overlay.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Could not generate overlay: {e}")

    # 3. Parameter scatter matrix
    try:
        print("\n3. Parameter scatter matrix...")
        fig = visualizer.plot_parameter_scatter(
            output_path=str(output_dir / 'parameter_scatter.png')
        )
        plots_generated.append('parameter_scatter.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Could not generate scatter matrix: {e}")

    # 4. Quality metrics
    try:
        print("\n4. Quality metrics...")
        fig = visualizer.plot_fit_quality_metrics(
            output_path=str(output_dir / 'fit_quality_metrics.png')
        )
        plots_generated.append('fit_quality_metrics.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Could not generate quality metrics: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nGenerated {len(plots_generated)} visualizations:")
    for plot in plots_generated:
        print(f"  - {output_dir / plot}")

    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
