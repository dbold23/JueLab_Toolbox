"""
Synthetic Data Plotter

Visualization tools for synthetic growth curve data.
Includes comparison plots between synthetic and real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.growth_models import GompertzModel, BaranyiModel, LogisticModel, RichardsModel


class SyntheticPlotter:
    """
    Visualization tools for synthetic growth curve data.
    """

    def __init__(self, synthetic_df: Optional[pd.DataFrame] = None):
        """
        Initialize plotter with synthetic data.

        Args:
            synthetic_df: DataFrame from SyntheticGrowthCurveGenerator
        """
        self.synthetic_df = synthetic_df

    def load_synthetic_data(self, csv_path: str):
        """Load synthetic data from CSV."""
        self.synthetic_df = pd.read_csv(csv_path)

    def plot_scenario_examples(
        self,
        n_per_scenario: int = 3,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 15)
    ) -> plt.Figure:
        """
        Plot example curves from each scenario.

        Args:
            n_per_scenario: Number of curves to show per scenario
            output_path: Path to save figure
            figsize: Figure size
        """
        if self.synthetic_df is None:
            raise ValueError("No synthetic data loaded")

        scenarios = self.synthetic_df['scenario'].unique()
        n_scenarios = len(scenarios)

        # Calculate grid size
        n_cols = min(4, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_scenarios > 1 else [axes]

        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]

            scenario_data = self.synthetic_df[self.synthetic_df['scenario'] == scenario]
            sample = scenario_data.head(n_per_scenario)

            for _, row in sample.iterrows():
                # Parse time and OD
                time = row['time']
                od = row['od600']

                if isinstance(time, str):
                    time = np.array(eval(time))
                if isinstance(od, str):
                    od = np.array(eval(od))

                ax.plot(time, od, alpha=0.7, linewidth=0.8)

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('OD600')
            expected = scenario_data['expected_class'].iloc[0] if 'expected_class' in scenario_data.columns else '?'
            ax.set_title(f'{scenario}\n(Expected: {expected})', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(scenarios), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Synthetic Growth Curves by Scenario', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved scenario examples to: {output_path}")

        return fig

    def plot_model_comparison(
        self,
        A: float = 1.5,
        mu: float = 0.2,
        lambda_: float = 3.0,
        duration: float = 50.0,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Compare different growth models with same base parameters.
        """
        t = np.linspace(0, duration, 200)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: All models
        ax1 = axes[0, 0]
        y_gomp = GompertzModel.compute(t, A, mu, lambda_)
        baranyi_params = BaranyiModel.from_gompertz_params(A, mu, lambda_)
        y_bar = BaranyiModel.compute(t, **baranyi_params)
        logistic_params = LogisticModel.from_gompertz_params(A, mu, lambda_)
        y_log = LogisticModel.compute(t, **logistic_params)
        richards_params = RichardsModel.from_gompertz_params(A, mu, lambda_)
        y_rich = RichardsModel.compute(t, **richards_params)

        ax1.plot(t, y_gomp, 'b-', label='Gompertz', linewidth=2)
        ax1.plot(t, y_bar, 'r--', label='Baranyi', linewidth=2)
        ax1.plot(t, y_log, 'g-.', label='Logistic', linewidth=2)
        ax1.plot(t, y_rich, 'm:', label='Richards', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('OD600')
        ax1.set_title(f'Model Comparison (A={A}, μ={mu}, λ={lambda_})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Growth rate comparison
        ax2 = axes[0, 1]
        dy_gomp = GompertzModel.derivative(t, A, mu, lambda_)
        dy_log = LogisticModel.derivative(t, **logistic_params)
        ax2.plot(t, dy_gomp, 'b-', label='Gompertz', linewidth=2)
        ax2.plot(t, dy_log, 'g--', label='Logistic', linewidth=2)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('dOD/dt')
        ax2.set_title('Growth Rate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Effect of varying A
        ax3 = axes[1, 0]
        for A_val in [0.5, 1.0, 1.5, 2.0]:
            y = GompertzModel.compute(t, A_val, mu, lambda_)
            ax3.plot(t, y, label=f'A={A_val}', linewidth=1.5)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('OD600')
        ax3.set_title(f'Effect of A (μ={mu}, λ={lambda_})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Effect of varying mu
        ax4 = axes[1, 1]
        for mu_val in [0.05, 0.1, 0.2, 0.4]:
            y = GompertzModel.compute(t, A, mu_val, lambda_)
            ax4.plot(t, y, label=f'μ={mu_val}', linewidth=1.5)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('OD600')
        ax4.set_title(f'Effect of μ (A={A}, λ={lambda_})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Growth Model Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved model comparison to: {output_path}")

        return fig

    def plot_noise_effect(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Show effect of different noise levels on curves.
        """
        if self.synthetic_df is None:
            raise ValueError("No synthetic data loaded")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        noise_levels = ['very_low', 'low', 'medium', 'high', 'very_high']

        for idx, noise_level in enumerate(noise_levels):
            if idx >= 6:
                break

            ax = axes.flatten()[idx]

            # Find curves with this noise level
            subset = self.synthetic_df[self.synthetic_df['noise_level'] == noise_level]

            if len(subset) == 0:
                ax.text(0.5, 0.5, f'No curves with\n{noise_level} noise',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Plot up to 5 curves
            for _, row in subset.head(5).iterrows():
                time = row['time']
                od = row['od600']

                if isinstance(time, str):
                    time = np.array(eval(time))
                if isinstance(od, str):
                    od = np.array(eval(od))

                ax.plot(time, od, alpha=0.7, linewidth=0.8)

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('OD600')
            r2_mean = subset['actual_r_squared'].mean() if 'actual_r_squared' in subset.columns else 0
            ax.set_title(f'{noise_level}\n(Avg R²: {r2_mean:.3f})')
            ax.grid(True, alpha=0.3)

        # Hide last subplot if only 5 noise levels
        if len(noise_levels) < 6:
            axes.flatten()[-1].set_visible(False)

        plt.suptitle('Effect of Noise Level', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved noise effect to: {output_path}")

        return fig

    def plot_expected_class_distribution(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Show distribution of expected classifications.
        """
        if self.synthetic_df is None:
            raise ValueError("No synthetic data loaded")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Pie chart
        ax1 = axes[0]
        class_counts = self.synthetic_df['expected_class'].value_counts()
        colors = {'GOOD': 'green', 'BAD': 'red'}
        ax1.pie(class_counts.values,
               labels=class_counts.index,
               autopct='%1.1f%%',
               colors=[colors.get(c, 'gray') for c in class_counts.index],
               startangle=90)
        ax1.set_title('Expected Classification Distribution')

        # Scenario breakdown
        ax2 = axes[1]
        scenario_counts = self.synthetic_df.groupby(['scenario', 'expected_class']).size().unstack(fill_value=0)

        if len(scenario_counts.columns) > 0:
            scenario_counts.plot(kind='barh', stacked=True, ax=ax2,
                               color=['green' if c == 'GOOD' else 'red' for c in scenario_counts.columns])
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Scenario')
            ax2.set_title('Curves per Scenario')
            ax2.legend(title='Expected')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved class distribution to: {output_path}")

        return fig

    def plot_real_vs_synthetic_comparison(
        self,
        real_df: pd.DataFrame,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Side-by-side comparison of real vs synthetic parameter distributions.
        """
        if self.synthetic_df is None:
            raise ValueError("No synthetic data loaded")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Get good curves from each
        real_good = real_df[real_df['is_good'] == True]
        syn_good = self.synthetic_df[self.synthetic_df['expected_class'] == 'GOOD']

        # Compare A
        ax1 = axes[0, 0]
        ax1.hist(real_good['gompertz_a'].dropna(), bins=15, alpha=0.5, label='Real', density=True)
        ax1.hist(syn_good['true_A'].dropna(), bins=15, alpha=0.5, label='Synthetic', density=True)
        ax1.set_xlabel('A (Max OD600)')
        ax1.set_ylabel('Density')
        ax1.set_title('A Parameter')
        ax1.legend()

        # Compare mu
        ax2 = axes[0, 1]
        ax2.hist(real_good['gompertz_mu'].dropna(), bins=15, alpha=0.5, label='Real', density=True)
        ax2.hist(syn_good['true_mu'].dropna(), bins=15, alpha=0.5, label='Synthetic', density=True)
        ax2.set_xlabel('μ (Growth Rate)')
        ax2.set_ylabel('Density')
        ax2.set_title('μ Parameter')
        ax2.legend()

        # Compare lambda
        ax3 = axes[0, 2]
        ax3.hist(real_good['gompertz_lambda'].dropna(), bins=15, alpha=0.5, label='Real', density=True)
        ax3.hist(syn_good['true_lambda'].dropna(), bins=15, alpha=0.5, label='Synthetic', density=True)
        ax3.set_xlabel('λ (Lag Phase)')
        ax3.set_ylabel('Density')
        ax3.set_title('λ Parameter')
        ax3.legend()

        # Compare R²
        ax4 = axes[1, 0]
        if 'fit_r_squared' in real_good.columns and 'actual_r_squared' in syn_good.columns:
            ax4.hist(real_good['fit_r_squared'].dropna(), bins=15, alpha=0.5, label='Real', density=True)
            ax4.hist(syn_good['actual_r_squared'].dropna(), bins=15, alpha=0.5, label='Synthetic', density=True)
            ax4.set_xlabel('R²')
            ax4.set_ylabel('Density')
            ax4.set_title('Fit Quality (R²)')
            ax4.legend()

        # Compare max_od
        ax5 = axes[1, 1]
        ax5.hist(real_good['max_od'].dropna(), bins=15, alpha=0.5, label='Real', density=True)
        ax5.hist(syn_good['max_od'].dropna(), bins=15, alpha=0.5, label='Synthetic', density=True)
        ax5.set_xlabel('Max OD')
        ax5.set_ylabel('Density')
        ax5.set_title('Maximum OD')
        ax5.legend()

        # Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')

        stats_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
        stats_text += f"Real good curves: {len(real_good)}\n"
        stats_text += f"Synthetic good curves: {len(syn_good)}\n\n"

        stats_text += "Parameter Means:\n"
        stats_text += f"  A: Real={real_good['gompertz_a'].mean():.3f}, Syn={syn_good['true_A'].mean():.3f}\n"
        stats_text += f"  μ: Real={real_good['gompertz_mu'].mean():.3f}, Syn={syn_good['true_mu'].mean():.3f}\n"
        stats_text += f"  λ: Real={real_good['gompertz_lambda'].mean():.2f}, Syn={syn_good['true_lambda'].mean():.2f}\n"

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Real vs Synthetic Data Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {output_path}")

        return fig


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from src.data_generator import generate_quick_test_set

    print("Generating test data...")
    test_df = generate_quick_test_set(n_per_category=5, seed=42)

    print("Creating plots...")
    plotter = SyntheticPlotter(test_df)

    # Create plots
    plotter.plot_scenario_examples(output_path='synthetic_scenarios.png')
    plotter.plot_model_comparison(output_path='model_comparison.png')
    plotter.plot_expected_class_distribution(output_path='class_distribution.png')

    print("Done!")
