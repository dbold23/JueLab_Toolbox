#!/usr/bin/env python3
"""
Haldane Feedback Inhibition Analysis

Fits the Haldane/Andrews substrate inhibition model to pesticide treatment
growth curves and compares with Gompertz fits from the main pipeline.

The Haldane model is mechanistic:
    dX/dt = mu(S) * X * (1 - X/X_max)
    dS/dt = -q * mu(S) * X
    mu(S) = mu_max * S / (Ks + S + S^2/Ki)

Key outputs:
    - haldane_comparison.csv: Per-strain Gompertz vs Haldane comparison
    - haldane_summary.csv: Per-pesticide summary with Ki rankings
    - plots/: Dual-panel comparison plots (biomass + substrate depletion)
    - haldane_overview.png: Summary dashboard

Usage:
    python 05_haldane_analysis.py [--results-dir DIR] [--output DIR]

BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# Haldane Model (self-contained, no dependency on 03_mle_model_fitting.py)
# =============================================================================

def haldane_ode(t, y, mu_max, Ks, Ki, X_max, q):
    """
    Coupled ODE: biomass growth + substrate depletion with Haldane kinetics.

    dX/dt = mu(S) * X * (1 - X/X_max)
    dS/dt = -q * mu(S) * X

    where mu(S) = mu_max * S / (Ks + S + S^2/Ki)
    """
    X, S = y
    S = max(S, 0)
    X = max(X, 0)

    if S < 1e-10:
        mu_S = 0.0
    else:
        mu_S = mu_max * S / (Ks + S + S**2 / Ki)

    dXdt = mu_S * X * (1 - X / X_max)
    dSdt = -q * mu_S * X

    return [dXdt, dSdt]


def solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0):
    """Solve the Haldane ODE system. Returns X(t), S(t)."""
    try:
        sol = solve_ivp(
            haldane_ode,
            [time[0], time[-1]],
            [X0, S0],
            args=(mu_max, Ks, Ki, X_max, q),
            t_eval=time,
            method='RK45',
            max_step=0.5,
            rtol=1e-8,
            atol=1e-10
        )
        if sol.success:
            return sol.y[0], sol.y[1]
    except Exception:
        pass
    return np.full_like(time, X0, dtype=float), np.full_like(time, S0, dtype=float)


def fit_haldane(time, od, S0=1.0, gompertz_params=None):
    """
    Fit the Haldane model to OD data via least-squares.

    Args:
        time: Time array (hours)
        od: OD600 values
        S0: Initial substrate concentration (arbitrary units)
        gompertz_params: Dict with 'a', 'mu', 'lambda' from Gompertz fit

    Returns:
        Dict with fit results
    """
    X0_init = max(float(np.mean(od[:3])), 0.001)
    X_max_init = float(np.max(od)) * 1.1

    if gompertz_params:
        mu_init = gompertz_params.get('mu', 0.2) * 2.0
    else:
        diff = np.diff(od)
        dt = np.diff(time)
        mu_init = max(float(np.max(diff / np.maximum(dt, 1e-6))), 0.01) * 2.0

    # Initial guesses
    p0 = np.array([mu_init, 0.1, 10.0, X_max_init, 0.1, X0_init, S0])

    bounds = [
        (0.001, 5.0),    # mu_max
        (0.001, 10.0),   # Ks
        (0.1, 1000.0),   # Ki
        (0.05, 5.0),     # X_max
        (0.001, 10.0),   # q
        (1e-6, 0.5),     # X0
        (0.01, 100.0),   # S0
    ]

    def objective(params):
        mu_max, Ks, Ki, X_max, q, X0, S0_fit = params
        X_pred, _ = solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0_fit)
        residuals = od - X_pred
        return np.sum(residuals**2)

    try:
        result = minimize(
            objective, p0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000}
        )

        if result.success or result.fun < 1e10:
            mu_max, Ks, Ki, X_max, q, X0_fit, S0_fit = result.x
            X_pred, S_pred = solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0_fit, S0_fit)

            residuals = od - X_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((od - np.mean(od))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            n = len(od)
            k = 7  # number of parameters
            aic = n * np.log(ss_res / n) + 2 * k
            bic = n * np.log(ss_res / n) + k * np.log(n)

            return {
                'success': True,
                'mu_max': mu_max, 'Ks': Ks, 'Ki': Ki,
                'X_max': X_max, 'q': q, 'X0': X0_fit, 'S0': S0_fit,
                'r_squared': r_squared,
                'aic': aic, 'bic': bic,
                'rmse': np.sqrt(np.mean(residuals**2)),
                'predicted_od': X_pred,
                'predicted_substrate': S_pred,
            }
    except Exception:
        pass

    return {'success': False, 'r_squared': 0, 'aic': np.inf, 'bic': np.inf}


# =============================================================================
# Gompertz Model (for AIC-fair comparison)
# =============================================================================

def gompertz_model(t, A, mu, lambda_):
    """Modified Gompertz growth model."""
    return A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))


def gompertz_aic(time, od, A, mu, lambda_):
    """Compute AIC for Gompertz fit using same n."""
    pred = gompertz_model(time, A, mu, lambda_)
    residuals = od - pred
    ss_res = np.sum(residuals**2)
    n = len(od)
    k = 3  # Gompertz has 3 parameters
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    r_squared = 1 - ss_res / np.sum((od - np.mean(od))**2) if np.sum((od - np.mean(od))**2) > 0 else 0
    return aic, bic, r_squared, np.sqrt(np.mean(residuals**2))


# =============================================================================
# Data Loading
# =============================================================================

def identify_pesticide_strains(results_df):
    """
    Identify which strains are pesticide treatment curves.

    Pesticide treatment strains have format: [PESTICIDE]ANDLB-[STRAIN_ID]
    Controls: LB-*, H2O-*, [PESTICIDE]-* (no LB)

    Returns:
        DataFrame filtered to pesticide+LB treatment strains
    """
    pesticide_mask = results_df['strain'].str.contains('ANDLB', case=False, na=False)
    return results_df[pesticide_mask].copy()


def extract_pesticide_name(strain):
    """Extract pesticide name from strain like 'BifenthrinANDLB-BIF2'."""
    if 'ANDLB' in strain.upper():
        return strain.upper().split('ANDLB')[0]
    return 'UNKNOWN'


def load_raw_data(data_dir, strain_name, group):
    """
    Load raw time-series data for a specific strain.

    Args:
        data_dir: Base data directory (e.g., data/raw)
        strain_name: e.g., 'BifenthrinANDLB-BIF2'
        group: e.g., 'Group1'

    Returns:
        (time, od) arrays or (None, None) if not found
    """
    # Map group names to directory structures
    group_dirs = {
        'Group1': 'Group1/Group_1_DATA',
        'Group2': 'Group 2/Group_2_DATA',
        'Group3': 'Group 3/Group_3_DATA',
        'Group4': 'Group 4/Group4_DATA',
    }

    group_path = data_dir / group_dirs.get(group, group)
    if not group_path.exists():
        return None, None

    # Search for the CSV file containing this strain
    csv_files = list(group_path.glob("*_DATA.csv"))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            time_col = [c for c in df.columns if 'TIME' in c.upper()]
            if not time_col:
                continue

            time = df[time_col[0]].values

            # Look for a column matching this strain
            for col in df.columns:
                clean_col = col.replace('_blanked', '').replace('_', '-')
                if clean_col.upper() == strain_name.upper().replace('-', '_').replace('_blanked', ''):
                    return time, df[col].values

                # Also try with underscore-to-dash conversion
                col_normalized = col.replace('_blanked', '').replace('_', '-')
                strain_normalized = strain_name.replace('_', '-')
                if col_normalized.upper() == strain_normalized.upper():
                    return time, df[col].values

                # Partial match: column contains the strain key parts
                strain_parts = strain_name.upper().split('-')
                col_upper = col.upper().replace('_BLANKED', '')
                if len(strain_parts) >= 2:
                    prefix = strain_parts[0].replace('-', '')
                    suffix = strain_parts[-1]
                    if prefix in col_upper and suffix in col_upper:
                        return time, df[col].values

        except Exception:
            continue

    return None, None


# =============================================================================
# Plotting
# =============================================================================

def plot_haldane_comparison(
    time, od, haldane_result, gompertz_params, strain_name, output_path
):
    """
    Create dual-panel comparison plot: Gompertz vs Haldane.

    Left: Biomass fits (data + both model predictions)
    Right: Haldane substrate depletion prediction
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Biomass comparison
    ax = axes[0]
    ax.scatter(time, od, s=10, alpha=0.4, color='gray', label='Data', zorder=1)

    # Gompertz fit
    if gompertz_params:
        gomp_pred = gompertz_model(
            time, gompertz_params['a'], gompertz_params['mu'], gompertz_params['lambda']
        )
        ax.plot(time, gomp_pred, 'b-', linewidth=2, label='Gompertz', zorder=2)

    # Haldane fit
    if haldane_result['success']:
        ax.plot(time, haldane_result['predicted_od'], 'r--', linewidth=2,
                label='Haldane', zorder=3)

    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('OD600 (Biomass)', fontsize=11)
    ax.set_title(f'{strain_name}: Model Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add R²/AIC annotation
    info_parts = []
    if gompertz_params:
        info_parts.append(f"Gompertz: R²={gompertz_params.get('r2', 0):.4f}")
    if haldane_result['success']:
        info_parts.append(f"Haldane:  R²={haldane_result['r_squared']:.4f}")
    if info_parts:
        ax.text(0.02, 0.98, '\n'.join(info_parts), transform=ax.transAxes,
                verticalalignment='top', fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: Substrate depletion
    ax = axes[1]
    if haldane_result['success']:
        ax.plot(time, haldane_result['predicted_substrate'], 'b-', linewidth=2)
        ax.fill_between(time, 0, haldane_result['predicted_substrate'], alpha=0.1, color='blue')
        ax.set_xlabel('Time (hours)', fontsize=11)
        ax.set_ylabel('Substrate Concentration (a.u.)', fontsize=11)

        Ki = haldane_result['Ki']
        Ks = haldane_result['Ks']
        mu_max = haldane_result['mu_max']
        q = haldane_result['q']

        ax.set_title(
            f'Predicted Pesticide Depletion\n'
            f'Ki={Ki:.2f} (inhibition constant)',
            fontsize=12, fontweight='bold'
        )

        param_text = (
            f"mu_max = {mu_max:.3f} h⁻¹\n"
            f"Ks     = {Ks:.3f}\n"
            f"Ki     = {Ki:.1f}\n"
            f"q      = {q:.3f}\n"
            f"S0     = {haldane_result['S0']:.2f}"
        )
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Haldane fit failed', ha='center', va='center',
                fontsize=14, color='red')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overview_dashboard(comparison_df, output_path):
    """
    Create summary dashboard showing key Haldane results.

    4-panel figure:
    1. Ki by pesticide (lower = more inhibitory)
    2. Gompertz vs Haldane R² scatter
    3. AIC comparison (which model is preferred)
    4. Substrate depletion rate by pesticide
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Filter to successful Haldane fits
    df = comparison_df[comparison_df['haldane_success']].copy()
    if len(df) == 0:
        fig.text(0.5, 0.5, 'No successful Haldane fits', ha='center', va='center', fontsize=16)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    df['pesticide'] = df['strain'].apply(extract_pesticide_name)

    # Panel 1: Ki by pesticide (boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    pesticides = df.groupby('pesticide')['haldane_Ki'].median().sort_values().index.tolist()
    ki_data = [df[df['pesticide'] == p]['haldane_Ki'].values for p in pesticides]
    if ki_data:
        bp = ax1.boxplot(ki_data, tick_labels=pesticides, patch_artist=True, vert=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(pesticides)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax1.set_ylabel('Ki (Inhibition Constant)', fontsize=11)
    ax1.set_title('Substrate Inhibition by Pesticide\n(lower Ki = stronger inhibition)',
                   fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Gompertz vs Haldane R²
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['gompertz_r2'], df['haldane_r2'], s=40, alpha=0.7,
                c=df['pesticide'].astype('category').cat.codes, cmap='Set1')
    lims = [min(df['gompertz_r2'].min(), df['haldane_r2'].min()) - 0.02, 1.01]
    ax2.plot(lims, lims, 'k--', alpha=0.3, label='Equal fit')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Gompertz R²', fontsize=11)
    ax2.set_ylabel('Haldane R²', fontsize=11)
    ax2.set_title('Model Fit Quality Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: AIC preference (bar chart)
    ax3 = fig.add_subplot(gs[1, 0])
    df['preferred_model'] = df.apply(
        lambda r: 'Haldane' if r['haldane_aic'] < r['gompertz_aic'] else 'Gompertz', axis=1
    )
    pref_counts = df.groupby('pesticide')['preferred_model'].value_counts().unstack(fill_value=0)
    if 'Haldane' not in pref_counts.columns:
        pref_counts['Haldane'] = 0
    if 'Gompertz' not in pref_counts.columns:
        pref_counts['Gompertz'] = 0
    pref_counts = pref_counts[['Gompertz', 'Haldane']]
    pref_counts.plot(kind='bar', ax=ax3, color=['#4C72B0', '#DD8452'], width=0.7)
    ax3.set_ylabel('Number of Strains', fontsize=11)
    ax3.set_title('Preferred Model by AIC\n(per pesticide)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Preferred')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Effective growth rate under inhibition
    ax4 = fig.add_subplot(gs[1, 1])
    # mu_effective = mu_max * S0 / (Ks + S0 + S0^2/Ki)
    df['mu_effective'] = df.apply(
        lambda r: r['haldane_mu_max'] * r['haldane_S0'] / (
            r['haldane_Ks'] + r['haldane_S0'] + r['haldane_S0']**2 / r['haldane_Ki']
        ) if r['haldane_Ki'] > 0 else 0,
        axis=1
    )
    pest_mu = df.groupby('pesticide')['mu_effective'].agg(['mean', 'std']).sort_values('mean')
    ax4.barh(pest_mu.index, pest_mu['mean'], xerr=pest_mu['std'],
             color=plt.cm.viridis(np.linspace(0.3, 0.8, len(pest_mu))),
             capsize=3, alpha=0.8)
    ax4.set_xlabel('Effective Growth Rate (h⁻¹)', fontsize=11)
    ax4.set_title('Growth Rate Under Inhibition\n(higher = less inhibited)',
                   fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Haldane Feedback Inhibition Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Haldane feedback inhibition analysis for pesticide growth curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analyzes pesticide treatment growth curves using the Haldane/Andrews substrate
inhibition model. Compares mechanistic Haldane fits with phenomenological
Gompertz fits using AIC.

Key outputs:
    haldane_comparison.csv  - Per-strain model comparison
    haldane_summary.csv     - Per-pesticide Ki rankings
    haldane_overview.png    - Summary dashboard
    plots/                  - Individual strain comparison plots
        """)
    parser.add_argument(
        '--results-dir', '-r',
        default=None,
        help='Path to results/tables directory (default: auto-detect)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default=None,
        help='Path to data/raw directory for loading raw time series'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: results/tables/Haldane_Analysis)'
    )
    parser.add_argument(
        '--s0', type=float, default=1.0,
        help='Initial substrate concentration in arbitrary units (default: 1.0)'
    )
    parser.add_argument(
        '--all-strains', action='store_true',
        help='Fit all strains, not just pesticide+LB treatments'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress detailed progress output'
    )

    args = parser.parse_args()

    # Auto-detect paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = base_dir / "results" / "tables"

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = base_dir / "data" / "raw"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_dir / "Haldane_Analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load pipeline results
    results_csv = results_dir / "all_groups_results.csv"
    if not results_csv.exists():
        print(f"ERROR: Pipeline results not found at {results_csv}")
        print("Run the main pipeline (01_growth_curve_analysis.py) first.")
        sys.exit(1)

    all_results = pd.read_csv(results_csv)
    print(f"Loaded {len(all_results)} strains from pipeline results")

    # Identify pesticide treatment strains
    if args.all_strains:
        target_strains = all_results[all_results['is_good'] == True].copy()
        print(f"  Analyzing ALL {len(target_strains)} good-fit strains")
    else:
        target_strains = identify_pesticide_strains(all_results)
        # Only analyze strains with good Gompertz fits
        target_strains = target_strains[target_strains['is_good'] == True].copy()
        print(f"  Found {len(target_strains)} pesticide+LB strains with good fits")

    if len(target_strains) == 0:
        print("No target strains found. Check pipeline results.")
        sys.exit(1)

    # Analyze each strain
    print(f"\n{'='*60}")
    print(f"  HALDANE FEEDBACK INHIBITION ANALYSIS")
    print(f"  S0 = {args.s0} (arbitrary units)")
    print(f"{'='*60}\n")

    comparison_rows = []
    n_total = len(target_strains)

    for i, (_, row) in enumerate(target_strains.iterrows()):
        strain = row['strain']
        group = row['group']

        if not args.quiet:
            print(f"  [{i+1}/{n_total}] {strain}...", end=' ')

        # Load raw time-series data
        time, od = load_raw_data(data_dir, strain, group)

        if time is None:
            if not args.quiet:
                print("SKIPPED (raw data not found)")
            continue

        # Clean data: remove NaN, ensure numeric
        mask = np.isfinite(time) & np.isfinite(od)
        time = time[mask].astype(float)
        od = od[mask].astype(float)

        if len(time) < 20:
            if not args.quiet:
                print("SKIPPED (too few points)")
            continue

        # Get Gompertz parameters from pipeline results
        gompertz_params = {
            'a': row.get('gompertz_a', 1.0),
            'mu': row.get('gompertz_mu', 0.15),
            'lambda': row.get('gompertz_lambda', 3.0),
            'r2': row.get('fit_r_squared', row.get('r_squared', 0)),
        }

        # Compute Gompertz AIC on full (non-truncated) data for fair comparison
        gomp_aic, gomp_bic, gomp_r2_full, gomp_rmse = gompertz_aic(
            time, od, gompertz_params['a'], gompertz_params['mu'], gompertz_params['lambda']
        )

        # Fit Haldane model
        haldane = fit_haldane(
            time, od, S0=args.s0,
            gompertz_params={'mu': gompertz_params['mu']}
        )

        # Determine preferred model
        if haldane['success']:
            delta_aic = gomp_aic - haldane['aic']  # positive = Haldane preferred
            preferred = 'Haldane' if delta_aic > 2 else ('Gompertz' if delta_aic < -2 else 'Equivalent')
        else:
            delta_aic = float('nan')
            preferred = 'Gompertz'

        # Collect results
        comp_row = {
            'strain': strain,
            'group': group,
            'pesticide': extract_pesticide_name(strain),
            # Gompertz
            'gompertz_a': gompertz_params['a'],
            'gompertz_mu': gompertz_params['mu'],
            'gompertz_lambda': gompertz_params['lambda'],
            'gompertz_r2': gomp_r2_full,
            'gompertz_aic': gomp_aic,
            'gompertz_bic': gomp_bic,
            'gompertz_rmse': gomp_rmse,
            # Haldane
            'haldane_success': haldane['success'],
            'haldane_r2': haldane.get('r_squared', float('nan')),
            'haldane_aic': haldane.get('aic', float('nan')),
            'haldane_bic': haldane.get('bic', float('nan')),
            'haldane_rmse': haldane.get('rmse', float('nan')),
            'haldane_mu_max': haldane.get('mu_max', float('nan')),
            'haldane_Ks': haldane.get('Ks', float('nan')),
            'haldane_Ki': haldane.get('Ki', float('nan')),
            'haldane_X_max': haldane.get('X_max', float('nan')),
            'haldane_q': haldane.get('q', float('nan')),
            'haldane_S0': haldane.get('S0', float('nan')),
            # Comparison
            'delta_aic': delta_aic,
            'preferred_model': preferred,
        }
        comparison_rows.append(comp_row)

        if not args.quiet:
            if haldane['success']:
                print(f"Haldane R²={haldane['r_squared']:.4f} Ki={haldane['Ki']:.1f} "
                      f"| Gompertz R²={gomp_r2_full:.4f} | {preferred} (ΔAIC={delta_aic:+.1f})")
            else:
                print(f"Haldane FAILED | Gompertz R²={gomp_r2_full:.4f}")

        # Generate comparison plot
        safe_name = strain.replace('/', '_').replace(' ', '_')
        plot_haldane_comparison(
            time, od, haldane, gompertz_params, strain,
            str(plots_dir / f'{safe_name}_haldane_vs_gompertz.png')
        )

    # Save comparison results
    if not comparison_rows:
        print("\nNo strains were successfully analyzed!")
        sys.exit(1)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / 'haldane_comparison.csv', index=False)
    print(f"\nSaved comparison to {output_dir / 'haldane_comparison.csv'}")

    # Generate summary by pesticide
    haldane_ok = comparison_df[comparison_df['haldane_success']].copy()

    if len(haldane_ok) > 0:
        summary = haldane_ok.groupby('pesticide').agg({
            'haldane_Ki': ['median', 'mean', 'std', 'count'],
            'haldane_mu_max': ['median', 'mean'],
            'haldane_Ks': ['median', 'mean'],
            'haldane_r2': ['median', 'mean'],
            'gompertz_r2': ['median', 'mean'],
            'preferred_model': lambda x: (x == 'Haldane').sum(),
        }).reset_index()

        # Flatten column names
        summary.columns = [
            'pesticide',
            'Ki_median', 'Ki_mean', 'Ki_std', 'n_strains',
            'mu_max_median', 'mu_max_mean',
            'Ks_median', 'Ks_mean',
            'haldane_r2_median', 'haldane_r2_mean',
            'gompertz_r2_median', 'gompertz_r2_mean',
            'n_haldane_preferred',
        ]
        summary = summary.sort_values('Ki_median')
        summary.to_csv(output_dir / 'haldane_summary.csv', index=False)
        print(f"Saved summary to {output_dir / 'haldane_summary.csv'}")

    # Generate overview dashboard
    plot_overview_dashboard(comparison_df, output_dir / 'haldane_overview.png')
    print(f"Saved overview to {output_dir / 'haldane_overview.png'}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  HALDANE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total strains analyzed: {len(comparison_df)}")
    print(f"  Successful Haldane fits: {len(haldane_ok)}")
    if len(haldane_ok) > 0:
        n_haldane_pref = (comparison_df['preferred_model'] == 'Haldane').sum()
        n_gomp_pref = (comparison_df['preferred_model'] == 'Gompertz').sum()
        n_equiv = (comparison_df['preferred_model'] == 'Equivalent').sum()
        print(f"  Preferred model (by AIC):")
        print(f"    Haldane:    {n_haldane_pref}")
        print(f"    Gompertz:   {n_gomp_pref}")
        print(f"    Equivalent: {n_equiv}")

        print(f"\n  Inhibition Ranking (lower Ki = stronger inhibition):")
        for _, srow in summary.iterrows():
            print(f"    {srow['pesticide']:<30s} Ki={srow['Ki_median']:.1f} "
                  f"(n={int(srow['n_strains'])})")

        print(f"\n  Biological Interpretation:")
        most_inhibitory = summary.iloc[0]['pesticide']
        least_inhibitory = summary.iloc[-1]['pesticide']
        print(f"    Most inhibitory pesticide:  {most_inhibitory} (lowest Ki)")
        print(f"    Least inhibitory pesticide: {least_inhibitory} (highest Ki)")
    print(f"{'='*60}")

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
