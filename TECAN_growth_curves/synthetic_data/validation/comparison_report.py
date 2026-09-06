"""
Comparison Report Generator

Figures for the synthetic benchmark: confusion matrix, parameter recovery
scatter, error distributions and per-scenario performance.

Two defects fixed here (M25, M19)
---------------------------------
1. The parameter pair list ``[('A', 'true_A', 'gompertz_a'), ...]`` was written
   out three times (once per plot, plus twice more in the validator), so a
   change of ground-truth column had four places to miss.  It is now the
   validator's single ``PARAM_SPECS``.
2. ``plot_parameter_scatter`` and ``plot_error_distributions`` dropped NaNs from
   x and y *independently* and then sliced both to ``iloc[:n]``: any curve with
   a fitted value but no truth (or the reverse) shifted one series against the
   other, so the plotted points -- and the R2/RMSE printed in the title -- paired
   the wrong curves.  Both now use the joint dropna the validator uses.

The join is delegated to :mod:`matching`, so this module can no longer produce
its own, differently-broken version of "matched".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # package import
    from .matching import match_results
    from .pipeline_validator import PARAM_SPECS, confusion, core_classification
except ImportError:  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.matching import match_results  # type: ignore
    from validation.pipeline_validator import (  # type: ignore
        PARAM_SPECS, confusion, core_classification,
    )


#: Display names for the three recovered parameters, in ``PARAM_SPECS`` order.
PARAM_LABELS = {"A": "A", "mu": "μ", "lambda": "λ"}


class ComparisonReport:
    """Generate detailed comparison reports for validation results."""

    def __init__(
        self,
        ground_truth: pd.DataFrame,
        pipeline_results: pd.DataFrame,
        validation_report: Optional[Dict] = None,
        matched: Optional[pd.DataFrame] = None,
        strict: bool = True,
    ):
        """
        Args:
            ground_truth: DataFrame with expected values
            pipeline_results: DataFrame with pipeline output
            validation_report: Optional dict from PipelineValidator
            matched: An already-computed match (``benchmark.py`` passes the one
                it scored, so the figures cannot disagree with the numbers).
            strict: Passed to :func:`matching.match_results` when ``matched``
                is not supplied.
        """
        self.ground_truth = ground_truth
        self.pipeline_results = pipeline_results
        self.validation_report = validation_report
        self.matched = (
            matched if matched is not None
            else match_results(ground_truth, pipeline_results, strict=strict))

    # -- helpers ------------------------------------------------------------

    def _match_datasets(self) -> pd.DataFrame:
        """Backwards-compatible alias for the cached match."""
        return self.matched

    def _accepted_good(self) -> pd.DataFrame:
        return self.matched[self.matched["gt_is_good"] & self.matched["pred_is_good"]]

    @staticmethod
    def _pairs(frame: pd.DataFrame, gt_col: str, pred_col: str) -> pd.DataFrame:
        """Jointly dropped truth/fit pairs -- never two independent dropnas."""
        if gt_col not in frame.columns or pred_col not in frame.columns:
            return pd.DataFrame(columns=[gt_col, pred_col])
        valid = frame[[gt_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
        return valid[np.isfinite(valid[gt_col]) & np.isfinite(valid[pred_col])]

    # -- plots --------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        block = core_classification(
            self.matched["gt_is_good"].to_numpy(dtype=bool),
            self.matched["pred_is_good"].to_numpy(dtype=bool))
        cm_dict = block["confusion_matrix"]
        cm = np.array([[cm_dict["true_positives"], cm_dict["false_negatives"]],
                       [cm_dict["false_positives"], cm_dict["true_negatives"]]])

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cm, cmap=plt.cm.Blues)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=20,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted GOOD', 'Predicted BAD'])
        ax.set_yticklabels(['Actual GOOD', 'Actual BAD'])
        ax.set_xlabel('Predicted Classification')
        ax.set_ylabel('Actual Classification')
        ax.set_title('Classification Confusion Matrix')

        # Balanced accuracy and specificity lead: on a 71.9 %-GOOD set raw
        # accuracy is four points higher for reasons of prevalence alone.
        metrics_text = (
            f"Balanced acc: {block['balanced_accuracy']:.3f}  "
            f"Sens: {block['sensitivity']:.3f}  Spec: {block['specificity']:.3f}\n"
            f"Accuracy: {block['accuracy']:.3f} (prevalence {block['prevalence']:.3f})  "
            f"Precision: {block['precision']:.3f}  Recall: {block['recall']:.3f}")
        ax.text(0.5, -0.18, metrics_text, transform=ax.transAxes, ha='center', fontsize=10)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to: {output_path}")
        return fig

    def plot_parameter_scatter(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Scatter plots comparing true vs fitted parameters."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        good = self._accepted_good()

        for ax, (name, gt_col, pred_col, _se) in zip(axes, PARAM_SPECS):
            label = PARAM_LABELS.get(name, name)
            pairs = self._pairs(good, gt_col, pred_col)
            if len(pairs) == 0:
                ax.text(0.5, 0.5, f'{label} data not available',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            x = pairs[gt_col].to_numpy(dtype=float)
            y = pairs[pred_col].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.5, edgecolors='black', linewidth=0.5)

            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'r--', label='Perfect recovery')

            if len(x) > 1:
                ss_res = float(np.sum((y - x) ** 2))
                ss_tot = float(np.sum((x - np.mean(x)) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                rmse = float(np.sqrt(np.mean((y - x) ** 2)))
                ax.set_title(f'{label} Recovery (n={len(x)}, R²={r2:.3f}, RMSE={rmse:.4f})')
            else:
                ax.set_title(f'{label} Recovery (n={len(x)})')

            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Fitted {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Parameter Recovery: True vs Fitted (accepted GOOD only)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved parameter scatter to: {output_path}")
        return fig

    def plot_error_distributions(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Histograms of parameter errors."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        good = self._accepted_good()

        for ax, (name, gt_col, pred_col, _se) in zip(axes, PARAM_SPECS):
            label = PARAM_LABELS.get(name, name)
            pairs = self._pairs(good, gt_col, pred_col)
            if len(pairs) == 0:
                ax.text(0.5, 0.5, f'{label} data not available',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            errors = (pairs[pred_col].to_numpy(dtype=float)
                      - pairs[gt_col].to_numpy(dtype=float))
            ax.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', label='Zero error')
            ax.axvline(float(np.mean(errors)), color='green', linestyle='--',
                       label=f'Mean: {np.mean(errors):.4f}')
            ax.set_xlabel(f'{label} Error (fitted - true)')
            ax.set_ylabel('Count')
            ax.set_title(f'{label} Error Distribution (n={len(errors)})')
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
        figsize: Tuple[int, int] = (14, 8),
    ) -> Optional[plt.Figure]:
        """Show classification performance by scenario."""
        if 'scenario' not in self.matched.columns:
            print("No scenario information available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        stats: List[Dict[str, Any]] = []
        for scenario, subset in self.matched.groupby('scenario', dropna=False):
            gt = subset['gt_is_good'].to_numpy(dtype=bool)
            pred = subset['pred_is_good'].to_numpy(dtype=bool)
            stats.append({
                'scenario': str(scenario),
                'accuracy': float(np.mean(gt == pred)) if len(subset) else 0.0,
                'n_curves': int(len(subset)),
                'expected_class': (subset['expected_class'].iloc[0]
                                   if 'expected_class' in subset.columns and len(subset)
                                   else 'UNKNOWN'),
            })
        stats_df = pd.DataFrame(stats).sort_values('accuracy')

        ax1 = axes[0]
        colors = ['green' if c == 'GOOD' else 'red' for c in stats_df['expected_class']]
        ax1.barh(stats_df['scenario'], stats_df['accuracy'], color=colors, alpha=0.7)
        ax1.axvline(0.95, color='orange', linestyle='--', label='95% threshold')
        ax1.set_xlabel('Classification Accuracy')
        ax1.set_ylabel('Scenario')
        ax1.set_title('Accuracy by Scenario')
        ax1.set_xlim(0, 1.05)
        ax1.legend()

        ax2 = axes[1]
        failures = self.matched[self.matched['gt_is_good'] != self.matched['pred_is_good']]
        if len(failures) > 0 and 'scenario' in failures.columns:
            failure_counts = failures['scenario'].value_counts()
            ax2.barh(failure_counts.index, failure_counts.values, color='coral', alpha=0.7)
            ax2.set_xlabel('Number of Misclassifications')
            ax2.set_ylabel('Scenario')
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
        prefix: str = 'validation',
    ) -> Dict[str, Path]:
        """Generate all report visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Path] = {}

        print("Generating validation report figures...")

        path = output_dir / f"{prefix}_confusion_matrix.png"
        self.plot_confusion_matrix(str(path))
        outputs['confusion_matrix'] = path
        plt.close('all')

        path = output_dir / f"{prefix}_parameter_scatter.png"
        self.plot_parameter_scatter(str(path))
        outputs['parameter_scatter'] = path
        plt.close('all')

        path = output_dir / f"{prefix}_error_distributions.png"
        self.plot_error_distributions(str(path))
        outputs['error_distributions'] = path
        plt.close('all')

        if 'scenario' in self.matched.columns:
            path = output_dir / f"{prefix}_scenario_performance.png"
            if self.plot_scenario_performance(str(path)) is not None:
                outputs['scenario_performance'] = path
            plt.close('all')

        print(f"\nGenerated {len(outputs)} report plots in {output_dir}")
        return outputs


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("ComparisonReport module - use with benchmark.py / PipelineValidator")
