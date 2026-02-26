"""
Pipeline Validator

Validates synthetic data against the TECAN analysis pipeline.
Runs synthetic data through the pipeline and compares results
to ground truth expectations.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: List[List[int]]


@dataclass
class ParameterRecovery:
    """Container for parameter recovery metrics."""
    parameter: str
    mean_error: float
    rmse: float
    r_squared: float
    bias: float
    n_samples: int


class PipelineValidator:
    """
    Validate synthetic data against the analysis pipeline.
    """

    def __init__(
        self,
        analysis_script_path: str,
        synthetic_data_dir: str,
        ground_truth_csv: str
    ):
        """
        Initialize validator.

        Args:
            analysis_script_path: Path to 01_growth_curve_analysis.py
            synthetic_data_dir: Directory with synthetic *_DATA.csv files
            ground_truth_csv: CSV with expected classifications and parameters
        """
        self.analysis_script = Path(analysis_script_path)
        self.data_dir = Path(synthetic_data_dir)
        self.ground_truth_path = Path(ground_truth_csv)

        # Load ground truth
        self.ground_truth = pd.read_csv(ground_truth_csv)

        # Results (populated after running pipeline)
        self.pipeline_results = None
        self.output_dir = None

    def run_pipeline(
        self,
        output_dir: str,
        use_adaptive: bool = True,
        verbose: bool = True
    ) -> Path:
        """
        Execute the analysis pipeline on synthetic data.

        Args:
            output_dir: Directory for pipeline output
            use_adaptive: Use adaptive truncation mode
            verbose: Print progress

        Returns:
            Path to processing_results.csv
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            str(self.analysis_script),
            str(self.data_dir),
            '-o', str(self.output_dir)
        ]

        if use_adaptive:
            cmd.append('--adaptive')

        cmd.append('--no-plots')  # Skip per-curve plots for faster validation

        if verbose:
            print(f"Running pipeline: {' '.join(cmd)}")

        # Run pipeline
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout (MCCV truncation is compute-intensive)
            )

            if verbose:
                print("Pipeline stdout:")
                print(result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout)

            if result.returncode != 0:
                print(f"Pipeline failed with code {result.returncode}")
                print("stderr:", result.stderr)

        except subprocess.TimeoutExpired:
            print("Pipeline timed out!")
            return None

        # Load results
        results_path = self.output_dir / 'processing_results.csv'
        if results_path.exists():
            self.pipeline_results = pd.read_csv(results_path)
            if verbose:
                print(f"Loaded {len(self.pipeline_results)} results from pipeline")
            return results_path
        else:
            print(f"Results file not found: {results_path}")
            return None

    def load_results(self, results_csv: str):
        """Load pipeline results from CSV."""
        self.pipeline_results = pd.read_csv(results_csv)

    def _match_results_to_ground_truth(self) -> pd.DataFrame:
        """
        Match pipeline results to ground truth by strain name.

        Handles the case where pipeline strain names include a prefix
        (e.g., 'SYNTHETIC-400pts-99h-CURVE0001') while ground truth
        has just 'CURVE0001'.

        Returns:
            DataFrame with matched ground truth and predictions
        """
        if self.pipeline_results is None:
            raise ValueError("No pipeline results loaded")

        import re

        # Create merged DataFrame
        # Ground truth has: curve_id, strain_name, expected_class, true_A, true_mu, true_lambda
        # Pipeline has: strain, is_good, gompertz_a, gompertz_mu, gompertz_lambda

        # Standardize strain column in ground truth
        gt = self.ground_truth.copy()
        if 'strain_name' in gt.columns:
            gt['strain_match'] = gt['strain_name'].str.upper().str.strip()
        elif 'strain' in gt.columns:
            gt['strain_match'] = gt['strain'].str.upper().str.strip()

        # For pipeline results, extract the CURVE#### identifier
        # Pipeline strain format: "SYNTHETIC-400pts-99h-CURVE0001" or just "CURVE0001"
        pr = self.pipeline_results.copy()
        pr['strain_match'] = pr['strain'].str.upper().str.strip().apply(
            lambda s: re.search(r'(CURVE\d+)', s).group(1) if re.search(r'(CURVE\d+)', s) else s
        )

        # Merge on the extracted curve identifier
        merged = gt.merge(pr, on='strain_match', how='inner', suffixes=('_gt', '_pred'))

        print(f"Matched {len(merged)} curves between ground truth and results")

        return merged

    def compute_classification_metrics(self) -> ValidationMetrics:
        """
        Compare predicted vs expected classifications.

        Returns:
            ValidationMetrics with accuracy, precision, recall, F1
        """
        matched = self._match_results_to_ground_truth()

        if len(matched) == 0:
            print("Warning: No matched curves found!")
            return ValidationMetrics(
                accuracy=0, precision=0, recall=0, f1_score=0,
                true_positives=0, true_negatives=0, false_positives=0, false_negatives=0,
                confusion_matrix=[[0, 0], [0, 0]]
            )

        # Expected and predicted
        y_true = (matched['expected_class'] == 'GOOD').astype(int).values
        y_pred = matched['is_good'].astype(int).values

        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            confusion_matrix=[[int(tp), int(fn)], [int(fp), int(tn)]]  # standard: [[TP,FN],[FP,TN]]
        )

    def compute_parameter_recovery(self) -> Dict[str, ParameterRecovery]:
        """
        Compare fitted parameters to ground truth.

        Only uses curves that were correctly classified as GOOD.

        Returns:
            Dict mapping parameter name to ParameterRecovery
        """
        matched = self._match_results_to_ground_truth()

        # Filter to correctly classified good curves
        correct_good = matched[
            (matched['expected_class'] == 'GOOD') &
            (matched['is_good'] == True)
        ]

        results = {}

        # Parameter mappings
        param_pairs = [
            ('A', 'true_A', 'gompertz_a'),
            ('mu', 'true_mu', 'gompertz_mu'),
            ('lambda', 'true_lambda', 'gompertz_lambda')
        ]

        for name, gt_col, pred_col in param_pairs:
            if gt_col not in correct_good.columns or pred_col not in correct_good.columns:
                continue

            # Joint dropna to preserve row alignment
            valid = correct_good[[gt_col, pred_col]].dropna()
            if len(valid) == 0:
                continue

            gt_vals = valid[gt_col].values
            pred_vals = valid[pred_col].values

            # Compute metrics
            errors = pred_vals - gt_vals
            mean_error = np.mean(errors)
            rmse = np.sqrt(np.mean(errors**2))

            # R² for parameter recovery
            ss_res = np.sum(errors**2)
            ss_tot = np.sum((gt_vals - np.mean(gt_vals))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results[name] = ParameterRecovery(
                parameter=name,
                mean_error=mean_error,
                rmse=rmse,
                r_squared=r_squared,
                bias=mean_error,
                n_samples=len(valid)
            )

        return results

    def identify_failure_cases(self) -> pd.DataFrame:
        """
        Identify where pipeline gave unexpected results.

        Returns:
            DataFrame with misclassified curves
        """
        matched = self._match_results_to_ground_truth()

        # Find misclassifications
        failures = matched[
            ((matched['expected_class'] == 'GOOD') & (matched['is_good'] == False)) |
            ((matched['expected_class'] == 'BAD') & (matched['is_good'] == True))
        ].copy()

        failures['error_type'] = failures.apply(
            lambda r: 'False Negative' if r['expected_class'] == 'GOOD' else 'False Positive',
            axis=1
        )

        return failures

    def generate_validation_report(self, output_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save report (JSON)

        Returns:
            Dict with all validation metrics
        """
        # Compute all metrics
        class_metrics = self.compute_classification_metrics()
        param_recovery = self.compute_parameter_recovery()
        failures = self.identify_failure_cases()

        report = {
            'summary': {
                'n_ground_truth': len(self.ground_truth),
                'n_pipeline_results': len(self.pipeline_results) if self.pipeline_results is not None else 0,
                'n_matched': len(self._match_results_to_ground_truth()),
                'n_failures': len(failures)
            },
            'classification': {
                'accuracy': class_metrics.accuracy,
                'precision': class_metrics.precision,
                'recall': class_metrics.recall,
                'f1_score': class_metrics.f1_score,
                'confusion_matrix': {
                    'true_positives': class_metrics.true_positives,
                    'true_negatives': class_metrics.true_negatives,
                    'false_positives': class_metrics.false_positives,
                    'false_negatives': class_metrics.false_negatives
                }
            },
            'parameter_recovery': {
                name: {
                    'mean_error': pr.mean_error,
                    'rmse': pr.rmse,
                    'r_squared': pr.r_squared,
                    'bias': pr.bias,
                    'n_samples': pr.n_samples
                }
                for name, pr in param_recovery.items()
            },
            'failure_analysis': {
                'n_false_positives': class_metrics.false_positives,
                'n_false_negatives': class_metrics.false_negatives,
                'failure_scenarios': failures['scenario'].value_counts().to_dict() if 'scenario' in failures.columns else {}
            }
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Validation report saved to: {output_path}")

        return report

    def print_summary(self):
        """Print a summary of validation results."""
        class_metrics = self.compute_classification_metrics()
        param_recovery = self.compute_parameter_recovery()
        failures = self.identify_failure_cases()

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print("\nCLASSIFICATION PERFORMANCE:")
        print(f"  Accuracy:  {class_metrics.accuracy:.3f}")
        print(f"  Precision: {class_metrics.precision:.3f}")
        print(f"  Recall:    {class_metrics.recall:.3f}")
        print(f"  F1 Score:  {class_metrics.f1_score:.3f}")

        print("\n  Confusion Matrix:")
        print(f"                    Predicted GOOD  Predicted BAD")
        print(f"    Actual GOOD:         {class_metrics.true_positives:4d}           {class_metrics.false_negatives:4d}")
        print(f"    Actual BAD:          {class_metrics.false_positives:4d}           {class_metrics.true_negatives:4d}")

        print("\nPARAMETER RECOVERY (good curves):")
        for name, pr in param_recovery.items():
            print(f"  {name}:")
            print(f"    RMSE: {pr.rmse:.4f}, R²: {pr.r_squared:.3f}, Bias: {pr.bias:.4f}")

        print(f"\nFAILURE ANALYSIS:")
        print(f"  Total failures: {len(failures)}")
        if 'scenario' in failures.columns:
            print("  By scenario:")
            for scenario, count in failures['scenario'].value_counts().items():
                print(f"    {scenario}: {count}")

        print("\n" + "=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate synthetic data against analysis pipeline'
    )

    parser.add_argument(
        '--pipeline',
        required=True,
        help='Path to 01_growth_curve_analysis.py'
    )

    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory with synthetic *_DATA.csv files'
    )

    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to ground_truth.csv'
    )

    parser.add_argument(
        '--output-dir',
        default='validation_output',
        help='Directory for validation output'
    )

    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running pipeline, just load existing results'
    )

    parser.add_argument(
        '--results-csv',
        help='Path to existing processing_results.csv (with --skip-run)'
    )

    args = parser.parse_args()

    # Initialize validator
    validator = PipelineValidator(
        analysis_script_path=args.pipeline,
        synthetic_data_dir=args.data_dir,
        ground_truth_csv=args.ground_truth
    )

    # Run or load pipeline
    if args.skip_run:
        if args.results_csv:
            validator.load_results(args.results_csv)
        else:
            print("Error: --results-csv required with --skip-run")
            sys.exit(1)
    else:
        validator.run_pipeline(args.output_dir)

    # Generate report
    validator.print_summary()
    report_path = Path(args.output_dir) / 'validation_report.json'
    validator.generate_validation_report(str(report_path))
