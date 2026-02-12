"""
Triplicate Visualizer for Real Growth Curve Data

Visualizes real good curves showing triplicate ranges and envelopes
to establish validation reference for synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings


class TriplicateVisualizer:
    """
    Visualize real good curves with triplicate ranges and parameter distributions.
    """

    def __init__(
        self,
        results_csv: str,
        data_dir: Optional[str] = None
    ):
        """
        Initialize visualizer with results data.

        Args:
            results_csv: Path to all_groups_results.csv
            data_dir: Directory containing raw *_DATA.csv files (optional)
        """
        self.results_path = Path(results_csv)
        self.results_df = pd.read_csv(results_csv)

        self.data_dir = Path(data_dir) if data_dir else None

        # Filter to good curves
        self.good_curves = self.results_df[self.results_df['is_good'] == True].copy()

        print(f"Loaded {len(self.results_df)} curves, {len(self.good_curves)} good")

    def plot_parameter_distributions(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot distributions of Gompertz parameters from good curves.

        Creates histograms for A, mu, lambda and their correlations.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        good = self.good_curves

        # Extract parameters (handle missing columns)
        A = good['gompertz_a'].dropna().values
        mu = good['gompertz_mu'].dropna().values
        lambda_ = good['gompertz_lambda'].dropna().values
        r2 = good['fit_r_squared'].dropna().values if 'fit_r_squared' in good.columns else np.array([])

        # Plot 1: A distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(A, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.median(A), color='red', linestyle='--', label=f'Median: {np.median(A):.3f}')
        ax1.set_xlabel('A (Maximum OD600)')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of A Parameter')
        ax1.legend()

        # Plot 2: mu distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(mu, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(mu), color='red', linestyle='--', label=f'Median: {np.median(mu):.3f}')
        ax2.set_xlabel('μ (Growth Rate, OD/hour)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of μ Parameter')
        ax2.legend()

        # Plot 3: lambda distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(lambda_, bins=20, color='darkorange', edgecolor='black', alpha=0.7)
        ax3.axvline(np.median(lambda_), color='red', linestyle='--', label=f'Median: {np.median(lambda_):.1f}')
        ax3.set_xlabel('λ (Lag Phase, hours)')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of λ Parameter')
        ax3.legend()

        # Plot 4: A vs mu scatter
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(A, mu, c='steelblue', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('A (Maximum OD600)')
        ax4.set_ylabel('μ (Growth Rate)')
        ax4.set_title('A vs μ Correlation')
        ax4.grid(True, alpha=0.3)

        # Plot 5: A vs lambda scatter
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(A, lambda_, c='forestgreen', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('A (Maximum OD600)')
        ax5.set_ylabel('λ (Lag Phase)')
        ax5.set_title('A vs λ Correlation')
        ax5.grid(True, alpha=0.3)

        # Plot 6: R² distribution
        ax6 = fig.add_subplot(gs[1, 2])
        if len(r2) > 0:
            ax6.hist(r2, bins=20, color='purple', edgecolor='black', alpha=0.7)
            ax6.axvline(0.95, color='red', linestyle='--', label='Threshold (0.95)')
            ax6.set_xlabel('R² (Fit Quality)')
            ax6.set_ylabel('Count')
            ax6.set_title('Distribution of R²')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'R² data not available', ha='center', va='center')

        plt.suptitle('Gompertz Parameter Distributions from Good Curves', fontsize=14, fontweight='bold')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved parameter distributions to: {output_path}")

        return fig

    def plot_parameter_space_3d(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        3D scatter plot of good curves in (A, μ, λ) parameter space.

        Shows the "valid" region where good growth curves exist.
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        good = self.good_curves

        A = good['gompertz_a'].values
        mu = good['gompertz_mu'].values
        lambda_ = good['gompertz_lambda'].values

        # Color by R² if available
        if 'fit_r_squared' in good.columns:
            colors = good['fit_r_squared'].values
            scatter = ax.scatter(A, mu, lambda_, c=colors, cmap='viridis',
                               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='R²')
        else:
            ax.scatter(A, mu, lambda_, c='steelblue', alpha=0.7, s=50,
                      edgecolors='black', linewidth=0.5)

        ax.set_xlabel('A (Max OD600)')
        ax.set_ylabel('μ (Growth Rate)')
        ax.set_zlabel('λ (Lag Phase)')
        ax.set_title('Good Curves in Parameter Space')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D parameter space to: {output_path}")

        return fig

    def plot_all_good_curves_overlay(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        max_curves: int = 100
    ) -> plt.Figure:
        """
        Overlay all good curves on a single plot to show the envelope.

        Uses reconstructed Gompertz curves from fitted parameters.
        """
        fig, ax = plt.subplots(figsize=figsize)

        good = self.good_curves.head(max_curves)

        # Time array for reconstruction
        t = np.linspace(0, 100, 400)

        for idx, row in good.iterrows():
            A = row['gompertz_a']
            mu = row['gompertz_mu']
            lambda_ = row['gompertz_lambda']

            if pd.isna(A) or pd.isna(mu) or pd.isna(lambda_):
                continue

            # Reconstruct Gompertz curve
            y = A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))

            ax.plot(t, y, alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('OD600')
        ax.set_title(f'Overlay of {len(good)} Good Growth Curves (from Gompertz fits)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2.5)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved curve overlay to: {output_path}")

        return fig

    def plot_by_treatment_group(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot parameter distributions grouped by treatment type.
        """
        good = self.good_curves.copy()

        # Parse treatment from strain name
        def get_treatment(strain):
            strain = str(strain).upper()
            if strain.startswith('LB-'):
                return 'LB Control'
            elif 'ANDLB-' in strain:
                return 'Pesticide+LB'
            elif strain.startswith('H2O-'):
                return 'H2O Control'
            else:
                return 'Pesticide Only'

        good['treatment_type'] = good['strain'].apply(get_treatment)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot A by treatment
        ax1 = axes[0, 0]
        for treatment in good['treatment_type'].unique():
            subset = good[good['treatment_type'] == treatment]
            ax1.hist(subset['gompertz_a'].dropna(), bins=15, alpha=0.5, label=treatment)
        ax1.set_xlabel('A (Max OD600)')
        ax1.set_ylabel('Count')
        ax1.set_title('A by Treatment Type')
        ax1.legend()

        # Plot mu by treatment
        ax2 = axes[0, 1]
        for treatment in good['treatment_type'].unique():
            subset = good[good['treatment_type'] == treatment]
            ax2.hist(subset['gompertz_mu'].dropna(), bins=15, alpha=0.5, label=treatment)
        ax2.set_xlabel('μ (Growth Rate)')
        ax2.set_ylabel('Count')
        ax2.set_title('μ by Treatment Type')
        ax2.legend()

        # Box plot for A
        ax3 = axes[1, 0]
        treatment_groups = [good[good['treatment_type'] == t]['gompertz_a'].dropna()
                          for t in good['treatment_type'].unique()]
        ax3.boxplot(treatment_groups, labels=good['treatment_type'].unique())
        ax3.set_ylabel('A (Max OD600)')
        ax3.set_title('A Distribution by Treatment')
        ax3.tick_params(axis='x', rotation=45)

        # Box plot for mu
        ax4 = axes[1, 1]
        treatment_groups = [good[good['treatment_type'] == t]['gompertz_mu'].dropna()
                          for t in good['treatment_type'].unique()]
        ax4.boxplot(treatment_groups, labels=good['treatment_type'].unique())
        ax4.set_ylabel('μ (Growth Rate)')
        ax4.set_title('μ Distribution by Treatment')
        ax4.tick_params(axis='x', rotation=45)

        plt.suptitle('Parameter Distributions by Treatment Type', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved treatment comparison to: {output_path}")

        return fig

    def plot_good_vs_bad_comparison(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Compare parameter distributions between good and bad curves.
        """
        good = self.results_df[self.results_df['is_good'] == True]
        bad = self.results_df[self.results_df['is_good'] == False]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # delta_od comparison
        ax1 = axes[0]
        ax1.hist(good['delta_od'].dropna(), bins=20, alpha=0.5, label='Good', color='green')
        ax1.hist(bad['delta_od'].dropna(), bins=20, alpha=0.5, label='Bad', color='red')
        ax1.axvline(0.3, color='black', linestyle='--', label='Threshold (0.3)')
        ax1.set_xlabel('Delta OD')
        ax1.set_ylabel('Count')
        ax1.set_title('Delta OD: Good vs Bad')
        ax1.legend()

        # max_od comparison
        ax2 = axes[1]
        ax2.hist(good['max_od'].dropna(), bins=20, alpha=0.5, label='Good', color='green')
        ax2.hist(bad['max_od'].dropna(), bins=20, alpha=0.5, label='Bad', color='red')
        ax2.set_xlabel('Max OD')
        ax2.set_ylabel('Count')
        ax2.set_title('Max OD: Good vs Bad')
        ax2.legend()

        # R² comparison (for curves with fits)
        ax3 = axes[2]
        if 'fit_r_squared' in self.results_df.columns:
            good_r2 = good['fit_r_squared'].dropna()
            bad_r2 = bad['fit_r_squared'].dropna()
            bad_r2 = bad_r2[bad_r2 > -10]  # Filter extreme negative values

            ax3.hist(good_r2, bins=20, alpha=0.5, label='Good', color='green')
            ax3.hist(bad_r2, bins=20, alpha=0.5, label='Bad', color='red')
            ax3.axvline(0.95, color='black', linestyle='--', label='Threshold (0.95)')
            ax3.set_xlabel('R²')
            ax3.set_ylabel('Count')
            ax3.set_title('R²: Good vs Bad')
            ax3.legend()

        plt.suptitle('Good vs Bad Curve Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved good vs bad comparison to: {output_path}")

        return fig

    def generate_summary_report(
        self,
        output_dir: str,
        prefix: str = 'real_data'
    ) -> Dict[str, Path]:
        """
        Generate all visualization plots and save to output directory.

        Args:
            output_dir: Directory for output files
            prefix: Prefix for output filenames

        Returns:
            Dict mapping plot name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        print("Generating visualization report...")

        # Parameter distributions
        path = output_dir / f"{prefix}_parameter_distributions.png"
        self.plot_parameter_distributions(str(path))
        outputs['parameter_distributions'] = path
        plt.close()

        # 3D parameter space
        path = output_dir / f"{prefix}_parameter_space_3d.png"
        self.plot_parameter_space_3d(str(path))
        outputs['parameter_space_3d'] = path
        plt.close()

        # Curve overlay
        path = output_dir / f"{prefix}_curve_overlay.png"
        self.plot_all_good_curves_overlay(str(path))
        outputs['curve_overlay'] = path
        plt.close()

        # By treatment
        path = output_dir / f"{prefix}_by_treatment.png"
        self.plot_by_treatment_group(str(path))
        outputs['by_treatment'] = path
        plt.close()

        # Good vs bad
        path = output_dir / f"{prefix}_good_vs_bad.png"
        self.plot_good_vs_bad_comparison(str(path))
        outputs['good_vs_bad'] = path
        plt.close()

        print(f"\nGenerated {len(outputs)} plots in {output_dir}")
        return outputs


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python triplicate_visualizer.py <results_csv> [output_dir]")
        print("Example: python triplicate_visualizer.py ../OUTPUT/all_groups_results.csv ./viz_output")
        sys.exit(1)

    results_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'visualization_output'

    visualizer = TriplicateVisualizer(results_csv)
    outputs = visualizer.generate_summary_report(output_dir)

    print("\nGenerated files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
