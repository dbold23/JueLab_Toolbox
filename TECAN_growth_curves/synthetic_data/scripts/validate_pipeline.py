#!/usr/bin/env python3
"""
Pipeline Validation CLI

Run synthetic data through the TECAN analysis pipeline and
compare results against ground truth.

Usage:
    python scripts/validate_pipeline.py --synthetic-dir output/test_data --ground-truth output/ground_truth.csv
    python scripts/validate_pipeline.py --pipeline-results path/to/processing_results.csv --ground-truth output/ground_truth.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import os

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.pipeline_validator import PipelineValidator
from validation.comparison_report import ComparisonReport


def main():
    parser = argparse.ArgumentParser(
        description='Validate analysis pipeline with synthetic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate with generated test data directory (runs pipeline automatically)
    python scripts/validate_pipeline.py \\
        --synthetic-dir output/comprehensive_test/test_data/DATA \\
        --ground-truth output/comprehensive_test/test_data/ground_truth.csv \\
        --pipeline-script ../ANALYSIS_SCRIPTS/01_growth_curve_analysis.py

    # Validate with pre-existing pipeline results
    python scripts/validate_pipeline.py \\
        --pipeline-results path/to/processing_results.csv \\
        --ground-truth output/comprehensive_test/test_data/ground_truth.csv

    # Quick validation without detailed report
    python scripts/validate_pipeline.py \\
        --pipeline-results path/to/processing_results.csv \\
        --ground-truth output/comprehensive_test/test_data/ground_truth.csv \\
        --quick
        """
    )

    parser.add_argument(
        '--synthetic-dir', '-s',
        help='Directory containing synthetic TECAN-format CSV files'
    )

    parser.add_argument(
        '--pipeline-results', '-p',
        help='Path to pipeline results CSV (if already processed)'
    )

    parser.add_argument(
        '--pipeline-script',
        help='Path to 01_growth_curve_analysis.py (required if --synthetic-dir used without --pipeline-results)'
    )

    parser.add_argument(
        '--ground-truth', '-g',
        required=True,
        help='Path to ground truth CSV with expected classifications'
    )

    parser.add_argument(
        '--report-dir', '-r',
        default='validation_report',
        help='Output directory for validation report (default: validation_report)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation without detailed visualizations'
    )

    parser.add_argument(
        '--output-json',
        help='Save validation metrics to JSON file'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.pipeline_results and not args.synthetic_dir:
        print("Error: Must provide either --pipeline-results or --synthetic-dir")
        sys.exit(1)

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        sys.exit(1)

    print("=" * 60)
    print("Pipeline Validation")
    print("=" * 60)

    # Determine the analysis script path
    if args.pipeline_script:
        pipeline_script = args.pipeline_script
    else:
        # Default: look for it relative to this script
        default_path = Path(__file__).parent.parent.parent / 'ANALYSIS_SCRIPTS' / '01_growth_curve_analysis.py'
        pipeline_script = str(default_path)

    # Determine synthetic data directory
    synthetic_dir = args.synthetic_dir or '.'

    # Create the validator
    validator = PipelineValidator(
        analysis_script_path=pipeline_script,
        synthetic_data_dir=synthetic_dir,
        ground_truth_csv=str(ground_truth_path)
    )

    # Load or generate pipeline results
    if args.pipeline_results:
        print(f"\nLoading pipeline results from: {args.pipeline_results}")
        validator.load_results(args.pipeline_results)
        print(f"  Loaded {len(validator.pipeline_results)} pipeline results")
    elif args.synthetic_dir:
        print(f"\nRunning pipeline on synthetic data in: {args.synthetic_dir}")
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        pipeline_output = str(report_dir / 'pipeline_output')

        result_path = validator.run_pipeline(output_dir=pipeline_output)
        if result_path is None:
            print("Error: Pipeline failed to produce results")
            sys.exit(1)
        print(f"  Pipeline produced {len(validator.pipeline_results)} results")

    # Print ground truth summary
    print(f"\nGround truth: {len(validator.ground_truth)} expected results")
    print(f"  Expected GOOD: {(validator.ground_truth['expected_class'] == 'GOOD').sum()}")
    print(f"  Expected BAD: {(validator.ground_truth['expected_class'] == 'BAD').sum()}")

    # Create output directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    print("\n" + "-" * 60)
    print("Running Validation...")
    print("-" * 60)

    # Classification metrics
    class_metrics = validator.compute_classification_metrics()

    print(f"\nClassification Results:")
    print(f"  Accuracy:  {class_metrics.accuracy:.1%}")
    print(f"  Precision: {class_metrics.precision:.1%}")
    print(f"  Recall:    {class_metrics.recall:.1%}")
    print(f"  F1 Score:  {class_metrics.f1_score:.1%}")

    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:  {class_metrics.true_positives}")
    print(f"    True Negatives:  {class_metrics.true_negatives}")
    print(f"    False Positives: {class_metrics.false_positives}")
    print(f"    False Negatives: {class_metrics.false_negatives}")

    # Parameter recovery
    param_report = validator.compute_parameter_recovery()
    if param_report:
        print(f"\nParameter Recovery (Good Curves):")
        for param_name, pr in param_report.items():
            print(f"  {param_name}: RMSE={pr.rmse:.4f}, R2={pr.r_squared:.3f}, Bias={pr.mean_error:.4f}")

    # Failure analysis
    failures = validator.identify_failure_cases()
    if len(failures) > 0:
        print(f"\nFailure Analysis: {len(failures)} misclassifications")
        if 'scenario' in failures.columns:
            print("  By scenario:")
            for scenario, count in failures['scenario'].value_counts().items():
                print(f"    {scenario}: {count}")

    # Generate visualizations if not quick mode
    if not args.quick:
        print("\n" + "-" * 60)
        print("Generating Report Visualizations...")
        print("-" * 60)

        comparison = ComparisonReport(
            ground_truth=validator.ground_truth,
            pipeline_results=validator.pipeline_results
        )

        outputs = comparison.generate_full_report(
            output_dir=str(report_dir),
            prefix='validation'
        )

        print(f"\nGenerated {len(outputs)} visualizations in {report_dir}")

    # Save JSON report
    json_report = {
        'classification': {
            'accuracy': class_metrics.accuracy,
            'precision': class_metrics.precision,
            'recall': class_metrics.recall,
            'f1_score': class_metrics.f1_score,
            'confusion_matrix': {
                'TP': class_metrics.true_positives,
                'TN': class_metrics.true_negatives,
                'FP': class_metrics.false_positives,
                'FN': class_metrics.false_negatives
            }
        },
        'parameters': {
            name: {
                'mean_error': pr.mean_error,
                'rmse': pr.rmse,
                'r_squared': pr.r_squared,
                'bias': pr.bias,
                'n_samples': pr.n_samples
            }
            for name, pr in param_report.items()
        } if param_report else {},
        'summary': {
            'total_ground_truth': len(validator.ground_truth),
            'total_pipeline_results': len(validator.pipeline_results),
            'n_failures': len(failures)
        }
    }

    # Always save to report dir
    json_path = report_dir / 'validation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\nSaved metrics to: {json_path}")

    # Also save to custom path if specified
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(json_report, f, indent=2)
        print(f"Also saved metrics to: {args.output_json}")

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    if class_metrics.accuracy >= 0.95:
        print("\nPipeline performs well (accuracy >= 95%)")
    elif class_metrics.accuracy >= 0.90:
        print("\nPipeline performs adequately (90-95% accuracy)")
    else:
        print("\nPipeline needs improvement (accuracy < 90%)")

    if class_metrics.false_negatives > 0:
        print(f"\n  {class_metrics.false_negatives} good curves misclassified as bad")
        print("    Consider: Adjusting R2 threshold or truncation parameters")

    if class_metrics.false_positives > 0:
        print(f"\n  {class_metrics.false_positives} bad curves misclassified as good")
        print("    Consider: Adding stricter quality filters")

    # Generate full validation report JSON
    full_report_path = report_dir / 'validation_report.json'
    validator.generate_validation_report(str(full_report_path))

    print(f"\nFull report saved to: {report_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
