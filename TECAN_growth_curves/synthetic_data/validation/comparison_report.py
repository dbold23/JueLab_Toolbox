"""
Comparison Report Generator

Generates visual reports comparing expected vs actual results
from pipeline validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json


class ComparisonReport:
    """
    Generate detailed comparison reports for validation results.
    """

    def __init__(
        self,
        ground_truth: pd.DataFrame,
        pipeline_results: pd.DataFrame,
        validation_report: Optional[Dict] = None
    ):
        """
        Initialize report generator.

        Args:
            ground_truth: DataFrame with expected values
            pipeline_results: DataFrame with pipeline output
            validation_report: Optional dict from PipelineValidator
        """
        self.ground_truth = ground_truth
        self.pipeline_results = pipeline_results
        self.validation_report = validation_report

        # Match datasets
        self.matched = self._match_datasets()

    def _match_datasets(self) -> pd.DataFrame:
        """Match ground truth to pipeline results."""
        import re

        gt = self.ground_truth.copy()
        pr = self.pipeline_results.copy()

        # Standardize strain names in ground truth
        if 'strain_name' in gt.columns:
            gt['strain_match'] = gt['strain_name'].str.upper().str.strip()
        elif 'strain' in gt.columns:
            gt['strain_match'] = gt['strain'].str.upper().str.strip()
        else:
            gt['strain_match'] = gt.index.astype(str)

        # For pipeline results, extract the CURVE#### identifier
        # Pipeline strain format: "SYNTHETIC-400pts-99h-CURVE0001" or just "CURVE0001"
        pr['strain_match'] = pr['strain'].str.upper().str.strip().apply(
            lambda s: re.search(r'(CURVE\d+)', s).group(1) if re.search(r'(CURVE\d+)', s) else s
        )

        # Merge
        merged = gt.merge(pr, on='strain_match', how='inner', suffixes=('_gt', '_pred'))

        return merged

    def plot_confusion_matrix(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        """
        from matplotlib.colors import LinearSegmentedColormap

        # Calculate confusion matrix
        y_true = (self.matched['expected_class'] == 'GOOD').astype(int)
        y_pred = self.matched['is_good'].astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        cm = np.array([[tp, fn], [fp, tn]])

        fig, ax = plt.subplots(figsize=figsize)

        # Custom colormap
        cmap = plt.cm.Blues

        im = ax.imshow(cm, cmap=cmap)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm[i, j]}',
                             ha='center', va='center', fontsize=20,
                             color='white' if cm[i, j] > cm.max()/2 else 'black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted GOOD', 'Predicted BAD'])
        ax.set_yticklabels(['Actual GOOD', 'Actual BAD'])
        ax.set_xlabel('Predicted Classification')
        ax.set_ylabel('Actual Classification')
        ax.set_title('Classification Confusion Matrix')

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics_text = f'Accuracy: {accuracy:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}'
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes, ha='center', fontsize=12)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to: {output_path}")

        return fig

    def plot_parameter_scatter(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Scatter plots comparing true vs fitted parameters.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Filter to correctly classified good curves
        good_mask = (
            (self.matched['expected_class'] == 'GOOD') &
            (self.matched['is_good'] == True)
        )
        good = self.matched[good_mask]

        param_pairs = [
            ('A', 'true_A', 'gompertz_a'),
            ('μ', 'true_mu', 'gompertz_mu'),
            ('λ', 'true_lambda', 'gompertz_lambda')
        ]

        for ax, (name, gt_col, pred_col) in zip(axes, param_pairs):
            if gt_col not in good.columns or pred_col not in good.columns:
                ax.text(0.5, 0.5, f'{name} data not available',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            x = good[gt_col].dropna()
            y = good[pred_col].dropna()

            # Align
            n = min(len(x), len(y))
            x, y = x.iloc[:n].values, y.iloc[:n].values

            ax.scatter(x, y, alpha=0.5, edgecolors='black', linewidth=0.5)

            # Perfect prediction line
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'r--', label='Perfect recovery')

            # Calculate R²
            if len(x) > 1:
                ss_res = np.sum((y - x)**2)
                ss_tot = np.sum((x - np.mean(x))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                rmse = np.sqrt(np.mean((y - x)**2))
                ax.set_title(f'{name} Recovery (R²={r2:.3f}, RMSE={rmse:.4f})')
            else:
                ax.set_title(f'{name} Recovery')

            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Fitted {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Parameter Recovery: True vs Fitted', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved parameter scatter to: {output_path}")

        return fig

    def plot_error_distributions(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Histograms of parameter errors.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        good_mask = (
            (self.matched['expected_class'] == 'GOOD') &
            (self.matched['is_good'] == True)
        )
        good = self.matched[good_mask]

        param_pairs = [
            ('A', 'true_A', 'gompertz_a'),
            ('μ', 'true_mu', 'gompertz_mu'),
            ('λ', 'true_lambda', 'gompertz_lambda')
        ]

        for ax, (name, gt_col, pred_col) in zip(axes, param_pairs):
            if gt_col not in good.columns or pred_col not in good.columns:
                continue

            true_vals = good[gt_col].dropna().values
            pred_vals = good[pred_col].dropna().values

            n = min(len(true_vals), len(pred_vals))
            errors = pred_vals[:n] - true_vals[:n]

            ax.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', label='Zero error')
            ax.axvline(np.mean(errors), color='green', linestyle='--',
                      label=f'Mean: {np.mean(errors):.4f}')

            ax.set_xlabel(f'{name} Error (fitted - true)')
            ax.set_ylabel('Count')
            ax.set_title(f'{name} Error Distribution')
            ax.legend()

        plt.suptitle('Parameter Error Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved error distributions to: {output_path}")

        return fig

    def plot_scenario_performance(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Show classification performance by scenario.
        """
        if 'scenario' not in self.matched.columns:
            print("No scenario information available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Calculate accuracy per scenario
        scenario_stats = []
        for scenario in self.matched['scenario'].unique():
            subset = self.matched[self.matched['scenario'] == scenario]

            correct = (
                ((subset['expected_class'] == 'GOOD') & (subset['is_good'] == True)) |
                ((subset['expected_class'] == 'BAD') & (subset['is_good'] == False))
            ).sum()

            accuracy = correct / len(subset) if len(subset) > 0 else 0

            scenario_stats.append({
                'scenario': scenario,
                'accuracy': accuracy,
                'n_curves': len(subset),
                'expected_class': subset['expected_class'].iloc[0] if len(subset) > 0 else 'UNKNOWN'
            })

        stats_df = pd.DataFrame(scenario_stats)
        stats_df = stats_df.sort_values('accuracy')

        # Bar plot of accuracy by scenario
        ax1 = axes[0]
        colors = ['green' if c == 'GOOD' else 'red' for c in stats_df['expected_class']]
        bars = ax1.barh(stats_df['scenario'], stats_df['accuracy'], color=colors, alpha=0.7)

        ax1.axvline(0.95, color='orange', linestyle='--', label='95% threshold')
        ax1.set_xlabel('Classification Accuracy')
        ax1.set_ylabel('Scenario')
        ax1.set_title('Accuracy by Scenario')
        ax1.set_xlim(0, 1.05)
        ax1.legend()

        # Failure count by scenario
        ax2 = axes[1]
        failures = self.matched[
            ((self.matched['expected_class'] == 'GOOD') & (self.matched['is_good'] == False)) |
            ((self.matched['expected_class'] == 'BAD') & (self.matched['is_good'] == True))
        ]

        if len(failures) > 0 and 'scenario' in failures.columns:
            failure_counts = failures['scenario'].value_counts()
            ax2.barh(failure_counts.index, failure_counts.values, color='coral', alpha=0.7)
            ax2.set_xlabel('Number of Misclassifications')
            ax2.set_ylabel('Scenario')
            ax2.set_title('Misclassifications by Scenario')
        else:
            ax2.text(0.5, 0.5, 'No misclassifications!',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Misclassifications by Scenario')

        plt.suptitle('Scenario-Level Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved scenario performance to: {output_path}")

        return fig

    def generate_full_report(
        self,
        output_dir: str,
        prefix: str = 'validation'
    ) -> Dict[str, Path]:
        """
        Generate all report visualizations.

        Args:
            output_dir: Directory for output files
            prefix: Prefix for filenames

        Returns:
            Dict mapping plot name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        print("Generating validation report...")

        # Confusion matrix
        path = output_dir / f"{prefix}_confusion_matrix.png"
        self.plot_confusion_matrix(str(path))
        outputs['confusion_matrix'] = path
        plt.close()

        # Parameter scatter
        path = output_dir / f"{prefix}_parameter_scatter.png"
        self.plot_parameter_scatter(str(path))
        outputs['parameter_scatter'] = path
        plt.close()

        # Error distributions
        path = output_dir / f"{prefix}_error_distributions.png"
        self.plot_error_distributions(str(path))
        outputs['error_distributions'] = path
        plt.close()

        # Scenario performance
        if 'scenario' in self.matched.columns:
            path = output_dir / f"{prefix}_scenario_performance.png"
            fig = self.plot_scenario_performance(str(path))
            if fig:
                outputs['scenario_performance'] = path
                plt.close()

        print(f"\nGenerated {len(outputs)} report plots in {output_dir}")
        return outputs


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("ComparisonReport module - use with PipelineValidator")
