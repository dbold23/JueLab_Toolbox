#!/usr/bin/env python3
"""
Truncation Method Comparison

Reads ensemble truncation results, fits Gompertz at each method's truncation
point, and compares which method produces the best downstream fits.

Usage:
    python 07_compare_truncation_methods.py
    python 07_compare_truncation_methods.py --ensemble-csv path/to/csv

BIO380SP25 — Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import importlib.util
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---- Dynamic imports from pipeline scripts ----
SCRIPTS_DIR = Path(__file__).parent

# Import 06_advanced_fitting for load_raw_data, load_config, gompertz_model
_adv_spec = importlib.util.spec_from_file_location(
    "advanced_fitting", str(SCRIPTS_DIR / "06_advanced_fitting.py")
)
_adv = importlib.util.module_from_spec(_adv_spec)
_adv_spec.loader.exec_module(_adv)

# Import 01_growth_curve_analysis for fit_gompertz
_gca_spec = importlib.util.spec_from_file_location(
    "growth_curve_analysis", str(SCRIPTS_DIR / "01_growth_curve_analysis.py")
)
_gca = importlib.util.module_from_spec(_gca_spec)
_gca_spec.loader.exec_module(_gca)

load_config = _adv.load_config
load_raw_data = _adv.load_raw_data
gompertz_model = _adv.gompertz_model
ensemble_truncate = _adv.ensemble_truncate
fit_gompertz = _gca.fit_gompertz

METHODS = ['first_peak', 'stationary_phase', 'adaptive_r2',
           'gp_derivative', 'changepoint', 'consensus']

METHOD_COLORS = {
    'first_peak': '#2ca02c',
    'stationary_phase': '#1f77b4',
    'adaptive_r2': '#ff7f0e',
    'gp_derivative': '#d62728',
    'changepoint': '#9467bd',
    'consensus': '#000000',
}

METHOD_LABELS = {
    'first_peak': 'First Peak',
    'stationary_phase': 'Stationary Phase',
    'adaptive_r2': 'Adaptive R²',
    'gp_derivative': 'GP Derivative',
    'changepoint': 'Changepoint',
    'consensus': 'Consensus',
}


# =============================================================================
# Core Comparison Logic
# =============================================================================

def fit_at_truncation(time, od, trunc_idx):
    """Truncate data and fit Gompertz. Returns FitResult or None."""
    if trunc_idx is None or np.isnan(trunc_idx):
        return None
    idx = int(trunc_idx)
    idx = max(0, min(idx, len(time) - 1))
    t_trunc = time[:idx + 1]
    od_trunc = od[:idx + 1]
    if len(t_trunc) < 10:
        return None
    try:
        result = fit_gompertz(t_trunc, od_trunc)
        if result.success:
            return result
    except Exception:
        pass
    return None


def run_comparison(ensemble_df, results_df, data_dir, config):
    """Run Gompertz fits at each method's truncation for each strain."""
    good = results_df[results_df['is_good'] == True].copy()

    rows = []
    for _, erow in ensemble_df.iterrows():
        strain = erow['strain']
        # Find group
        match = good[good['strain'] == strain]
        if match.empty:
            continue
        group = match.iloc[0]['group']

        # Load raw data
        t, od = load_raw_data(data_dir, strain, group)
        if t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(od)
        t, od = t[mask].astype(float), od[mask].astype(float)

        for method in METHODS:
            if method == 'consensus':
                idx_col = 'consensus_idx'
            else:
                idx_col = f'{method}_idx'

            trunc_idx = erow.get(idx_col)
            if pd.isna(trunc_idx) if isinstance(trunc_idx, float) else trunc_idx is None:
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': None, 'trunc_time': None, 'n_points': None,
                    'fit_success': False, 'r_squared': None, 'rmse': None,
                    'gompertz_A': None, 'gompertz_mu': None, 'gompertz_lambda': None,
                    'a_err_pct': None, 'mu_err_pct': None,
                })
                continue

            trunc_idx_int = int(trunc_idx)
            fit = fit_at_truncation(t, od, trunc_idx_int)

            if fit is not None and fit.success:
                a_err_pct = abs(fit.a_err / fit.a_opt * 100) if fit.a_opt != 0 else None
                mu_err_pct = abs(fit.mu_err / fit.mu_opt * 100) if fit.mu_opt != 0 else None
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': trunc_idx_int,
                    'trunc_time': t[min(trunc_idx_int, len(t)-1)],
                    'n_points': trunc_idx_int + 1,
                    'fit_success': True,
                    'r_squared': fit.r_squared,
                    'rmse': fit.rmse,
                    'gompertz_A': fit.a_opt,
                    'gompertz_mu': fit.mu_opt,
                    'gompertz_lambda': fit.lambda_opt,
                    'a_err_pct': a_err_pct,
                    'mu_err_pct': mu_err_pct,
                })
            else:
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': trunc_idx_int,
                    'trunc_time': t[min(trunc_idx_int, len(t)-1)] if trunc_idx_int < len(t) else None,
                    'n_points': trunc_idx_int + 1,
                    'fit_success': False, 'r_squared': None, 'rmse': None,
                    'gompertz_A': None, 'gompertz_mu': None, 'gompertz_lambda': None,
                    'a_err_pct': None, 'mu_err_pct': None,
                })

    return pd.DataFrame(rows)


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_method_summary(comp_df, config):
    """Compute per-method aggregate statistics."""
    min_r2 = config.get('classification', {}).get('min_r_squared', 0.95)
    excellent_r2 = config.get('quality_gates', {}).get('excellent_r2_threshold', 0.98)

    rows = []
    for method in METHODS:
        mdf = comp_df[comp_df['method'] == method]
        n_total = len(mdf)
        success = mdf[mdf['fit_success'] == True]
        n_success = len(success)
        n_fail = n_total - n_success

        r2_vals = success['r_squared'].dropna()
        if len(r2_vals) > 0:
            rows.append({
                'method': method,
                'label': METHOD_LABELS.get(method, method),
                'n_strains': n_total,
                'n_success': n_success,
                'n_failures': n_fail,
                'mean_r2': r2_vals.mean(),
                'median_r2': r2_vals.median(),
                'std_r2': r2_vals.std(),
                'min_r2': r2_vals.min(),
                'pct_good': (r2_vals >= min_r2).mean() * 100,
                'pct_excellent': (r2_vals >= excellent_r2).mean() * 100,
                'mean_rmse': success['rmse'].mean(),
                'mean_a_err_pct': success['a_err_pct'].dropna().mean(),
                'mean_mu_err_pct': success['mu_err_pct'].dropna().mean(),
            })
        else:
            rows.append({
                'method': method, 'label': METHOD_LABELS.get(method, method),
                'n_strains': n_total, 'n_success': 0, 'n_failures': n_fail,
                'mean_r2': None, 'median_r2': None, 'std_r2': None,
                'min_r2': None, 'pct_good': 0, 'pct_excellent': 0,
                'mean_rmse': None, 'mean_a_err_pct': None, 'mean_mu_err_pct': None,
            })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values('mean_r2', ascending=False).reset_index(drop=True)
    summary.index.name = 'rank'
    return summary


def compute_pairwise_agreement(comp_df):
    """Compute pairwise method agreement on truncation times and R²."""
    rows = []
    for m1, m2 in combinations(METHODS, 2):
        df1 = comp_df[comp_df['method'] == m1].set_index('strain')
        df2 = comp_df[comp_df['method'] == m2].set_index('strain')
        shared = df1.index.intersection(df2.index)

        # Filter to both succeeded
        mask = df1.loc[shared, 'fit_success'] & df2.loc[shared, 'fit_success']
        shared_ok = shared[mask]

        if len(shared_ok) < 3:
            rows.append({
                'method_a': m1, 'method_b': m2, 'n_shared': len(shared_ok),
                'time_corr': None, 'time_mae_hours': None,
                'r2_corr': None, 'classification_agreement_pct': None,
            })
            continue

        times1 = df1.loc[shared_ok, 'trunc_time'].values.astype(float)
        times2 = df2.loc[shared_ok, 'trunc_time'].values.astype(float)
        r2_1 = df1.loc[shared_ok, 'r_squared'].values.astype(float)
        r2_2 = df2.loc[shared_ok, 'r_squared'].values.astype(float)

        time_corr, _ = pearsonr(times1, times2) if np.std(times1) > 0 and np.std(times2) > 0 else (0, 1)
        r2_corr, _ = pearsonr(r2_1, r2_2) if np.std(r2_1) > 0 and np.std(r2_2) > 0 else (0, 1)

        good1 = r2_1 >= 0.95
        good2 = r2_2 >= 0.95
        agree = (good1 == good2).mean() * 100

        rows.append({
            'method_a': m1, 'method_b': m2, 'n_shared': len(shared_ok),
            'time_corr': time_corr, 'time_mae_hours': np.mean(np.abs(times1 - times2)),
            'r2_corr': r2_corr, 'classification_agreement_pct': agree,
        })

    return pd.DataFrame(rows)


def compute_strain_best(comp_df):
    """Find which method produced the best R² for each strain."""
    rows = []
    for strain, sdf in comp_df.groupby('strain'):
        success = sdf[sdf['fit_success'] == True].copy()
        if success.empty:
            rows.append({'strain': strain, 'best_method': None, 'best_r2': None,
                         'consensus_r2': None, 'consensus_rank': None})
            continue
        best_idx = success['r_squared'].idxmax()
        best = success.loc[best_idx]

        consensus_row = success[success['method'] == 'consensus']
        consensus_r2 = consensus_row['r_squared'].values[0] if not consensus_row.empty else None

        # Rank consensus among all methods for this strain
        ranked = success.sort_values('r_squared', ascending=False)
        rank_list = ranked['method'].tolist()
        consensus_rank = rank_list.index('consensus') + 1 if 'consensus' in rank_list else None

        rows.append({
            'strain': strain,
            'group': sdf['group'].iloc[0],
            'best_method': best['method'],
            'best_r2': best['r_squared'],
            'consensus_r2': consensus_r2,
            'consensus_rank': consensus_rank,
            'n_methods_succeeded': len(success),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_r2_boxplot(comp_df, summary_df, output_path):
    """Box plot of R² per method — the money plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by mean R² (best first)
    order = summary_df['method'].tolist()
    data = []
    labels = []
    colors = []
    for method in order:
        vals = comp_df[(comp_df['method'] == method) &
                       (comp_df['fit_success'] == True)]['r_squared'].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(METHOD_LABELS.get(method, method))
            colors.append(METHOD_COLORS.get(method, '#888888'))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Good threshold (0.95)')
    ax.axhline(y=0.98, color='orange', linestyle=':', alpha=0.5, label='Excellent (0.98)')
    ax.set_ylabel('Gompertz R²')
    ax.set_title('Truncation Method Comparison: Gompertz Fit Quality')
    ax.legend(loc='lower left')
    ax.set_ylim(max(0.8, ax.get_ylim()[0]), 1.005)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ranking_bar(summary_df, output_path):
    """Horizontal bar chart of mean R² per method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = summary_df['method'].tolist()[::-1]  # reverse for horizontal
    means = [summary_df[summary_df['method'] == m]['mean_r2'].values[0] for m in methods]
    stds = [summary_df[summary_df['method'] == m]['std_r2'].values[0] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    bars = ax.barh(range(len(methods)), means, xerr=stds, color=colors, alpha=0.7,
                   capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean Gompertz R²')
    ax.set_title('Method Ranking by Mean R²')
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5)

    # Annotate bars
    for i, (m, s) in enumerate(zip(means, stds)):
        if m is not None:
            ax.text(m + s + 0.002, i, f'{m:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_truncation_scatter_matrix(comp_df, output_path):
    """Pairwise scatter of truncation times."""
    trunc_methods = [m for m in METHODS if m != 'consensus']
    n = len(trunc_methods)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i, m1 in enumerate(trunc_methods):
        for j, m2 in enumerate(trunc_methods):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                vals = comp_df[(comp_df['method'] == m1) &
                               (comp_df['fit_success'] == True)]['trunc_time'].dropna()
                ax.hist(vals, bins=15, color=METHOD_COLORS.get(m1, '#888'), alpha=0.7)
                ax.set_title(METHOD_LABELS.get(m1, m1), fontsize=8)
            elif i > j:
                # Lower triangle: scatter
                df1 = comp_df[(comp_df['method'] == m1) & (comp_df['fit_success'] == True)].set_index('strain')
                df2 = comp_df[(comp_df['method'] == m2) & (comp_df['fit_success'] == True)].set_index('strain')
                shared = df1.index.intersection(df2.index)
                if len(shared) > 2:
                    t1 = df1.loc[shared, 'trunc_time'].values.astype(float)
                    t2 = df2.loc[shared, 'trunc_time'].values.astype(float)
                    ax.scatter(t2, t1, s=10, alpha=0.6)
                    r, _ = pearsonr(t1, t2) if np.std(t1) > 0 and np.std(t2) > 0 else (0, 1)
                    ax.set_title(f'r={r:.2f}', fontsize=8)
                    lims = [min(t1.min(), t2.min()), max(t1.max(), t2.max())]
                    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
            else:
                ax.axis('off')

            if i == n - 1:
                ax.set_xlabel(METHOD_LABELS.get(m2, m2), fontsize=7)
            if j == 0:
                ax.set_ylabel(METHOD_LABELS.get(m1, m1), fontsize=7)
            ax.tick_params(labelsize=6)

    plt.suptitle('Truncation Time: Pairwise Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_calibration(comp_df, ensemble_df, output_path):
    """Confidence vs R² calibration curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method in [m for m in METHODS if m != 'consensus']:
        # Get confidence from ensemble_df
        if method == 'consensus':
            conf_col = 'consensus_confidence'
        else:
            conf_col = f'{method}_confidence'

        if conf_col not in ensemble_df.columns:
            continue

        merged = comp_df[comp_df['method'] == method].merge(
            ensemble_df[['strain', conf_col]].rename(columns={conf_col: 'confidence'}),
            on='strain', how='left'
        )
        merged = merged[merged['fit_success'] == True].dropna(subset=['confidence', 'r_squared'])

        if len(merged) < 4:
            continue

        # Bin by confidence quartiles
        bins = [0, 0.25, 0.5, 0.75, 1.01]
        merged['conf_bin'] = pd.cut(merged['confidence'], bins=bins,
                                     labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0'])
        grouped = merged.groupby('conf_bin', observed=True)['r_squared'].mean()

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped.values, 'o-',
                    color=METHOD_COLORS.get(method, '#888'),
                    label=METHOD_LABELS.get(method, method), markersize=6)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, fontsize=9)

    ax.set_xlabel('Method Confidence Bin')
    ax.set_ylabel('Mean Gompertz R²')
    ax.set_title('Confidence Calibration: Does Higher Confidence → Better Fit?')
    ax.legend(fontsize=8)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_strain_heatmap(comp_df, strain_best_df, output_path):
    """Heatmap: rows=strains, cols=methods, color=R²."""
    pivot = comp_df.pivot_table(index='strain', columns='method',
                                 values='r_squared', aggfunc='first')
    # Reorder columns
    cols = [m for m in METHODS if m in pivot.columns]
    pivot = pivot[cols]

    # Sort by consensus R² descending
    if 'consensus' in pivot.columns:
        pivot = pivot.sort_values('consensus', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.25)))

    # Use TwoSlopeNorm centered at 0.95
    data = pivot.values
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        plt.close()
        return

    vmin = max(0.8, np.nanmin(data))
    vmax = min(1.0, np.nanmax(data))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.95, vmax=vmax)

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', norm=norm)

    # Mark best method per strain with a star
    for i, strain in enumerate(pivot.index):
        best = strain_best_df[strain_best_df['strain'] == strain]
        if not best.empty:
            bm = best.iloc[0]['best_method']
            if bm in cols:
                j = cols.index(bm)
                ax.text(j, i, '*', ha='center', va='center', fontsize=10,
                        fontweight='bold', color='black')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([METHOD_LABELS.get(c, c) for c in cols], rotation=30, ha='right')
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title('Per-Strain R² by Truncation Method (* = best)')

    plt.colorbar(im, ax=ax, label='Gompertz R²', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_consensus_vs_best(strain_best_df, output_path):
    """Scatter: consensus R² vs best individual R²."""
    df = strain_best_df.dropna(subset=['best_r2', 'consensus_r2'])
    if len(df) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df['best_r2'], df['consensus_r2'], s=30, alpha=0.6, c='steelblue')

    lims = [min(df['best_r2'].min(), df['consensus_r2'].min()) - 0.01, 1.005]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='y = x')
    ax.set_xlabel('Best Individual Method R²')
    ax.set_ylabel('Consensus (Ensemble) R²')
    ax.set_title('Does Ensemble Consensus Add Value?')

    # Count wins
    wins = (df['consensus_r2'] > df['best_r2'] + 0.001).sum()
    ties = ((df['consensus_r2'] - df['best_r2']).abs() <= 0.001).sum()
    losses = (df['consensus_r2'] < df['best_r2'] - 0.001).sum()
    ax.legend([f'Consensus wins: {wins}, ties: {ties}, loses: {losses}'], loc='lower right')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Overlay Plots — per-strain visual comparison
# =============================================================================

def plot_method_overlay(strain, group, time, od, comp_df, output_path):
    """Plot raw data + all 6 Gompertz curves overlaid for a single strain.

    For each method: draw a vertical dashed line at the truncation point,
    then the Gompertz curve (solid through fitted region, dotted beyond).
    """
    strain_data = comp_df[(comp_df['strain'] == strain) & (comp_df['fit_success'] == True)]
    if strain_data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Raw data
    ax.scatter(time, od, s=12, c='#bbbbbb', alpha=0.6, zorder=1, label='Raw OD')

    t_fine = np.linspace(time.min(), time.max(), 500)
    legend_entries = []

    for _, row in strain_data.iterrows():
        method = row['method']
        color = METHOD_COLORS.get(method, '#888888')
        label = METHOD_LABELS.get(method, method)
        A = row['gompertz_A']
        mu = row['gompertz_mu']
        lam = row['gompertz_lambda']
        trunc_time = row['trunc_time']
        r2 = row['r_squared']

        if pd.isna(A) or pd.isna(mu) or pd.isna(lam):
            continue

        # Gompertz curve: solid in fitted region, dotted beyond
        y_fit = gompertz_model(t_fine, A, mu, lam)
        mask_fit = t_fine <= trunc_time
        mask_extrap = t_fine >= trunc_time

        lw = 2.5 if method == 'consensus' else 1.8
        ax.plot(t_fine[mask_fit], y_fit[mask_fit], color=color, linewidth=lw,
                linestyle='-', zorder=3)
        ax.plot(t_fine[mask_extrap], y_fit[mask_extrap], color=color, linewidth=lw * 0.7,
                linestyle=':', alpha=0.5, zorder=2)

        # Vertical truncation line
        ax.axvline(x=trunc_time, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

        legend_entries.append(f"{label} (R²={r2:.4f})")

    # Build legend manually
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='#bbbbbb', marker='o', linestyle='None',
                       markersize=5, label='Raw OD')]
    for _, row in strain_data.iterrows():
        method = row['method']
        color = METHOD_COLORS.get(method, '#888888')
        label = METHOD_LABELS.get(method, method)
        r2 = row['r_squared']
        lw = 2.5 if method == 'consensus' else 1.8
        handles.append(Line2D([0], [0], color=color, linewidth=lw,
                               label=f"{label} (R²={r2:.4f})"))

    ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('OD', fontsize=11)
    ax.set_title(f'{strain}  ({group})\nTruncation Method Overlay', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_overlay_plots(comp_df, results_df, data_dir, output_dir, is_bad=False):
    """Generate per-strain overlay plots."""
    subdir = 'bad_strain_overlay_plots' if is_bad else 'overlay_plots'
    overlay_dir = output_dir / subdir
    overlay_dir.mkdir(exist_ok=True)

    strains = comp_df['strain'].unique()
    n = len(strains)
    print(f"\n  Generating {n} overlay plots → {overlay_dir}/")

    for i, strain in enumerate(strains):
        sdf = comp_df[comp_df['strain'] == strain]
        group = sdf['group'].iloc[0]

        t, od_raw = load_raw_data(data_dir, strain, group)
        if t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(od_raw)
        t, od_raw = t[mask].astype(float), od_raw[mask].astype(float)

        out_path = str(overlay_dir / f'{strain}_method_overlay.png')
        plot_method_overlay(strain, group, t, od_raw, comp_df, out_path)

        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"    [{i+1}/{n}] plots done")


# =============================================================================
# Bad Strain Rescue Analysis
# =============================================================================

def run_bad_strain_analysis(results_df, data_dir, config, output_dir):
    """Run ensemble truncation + Gompertz fitting on bad strains to check for rescues."""
    bad_df = results_df[results_df['is_good'] == False].copy()
    min_delta_od = 0.015

    # Filter to strains with some growth signal
    candidates = bad_df[bad_df['delta_od'].astype(float) > min_delta_od]
    n_flat = len(bad_df) - len(candidates)
    print(f"\n  Bad strains: {len(bad_df)} total")
    print(f"    Flat/negligible (dOD <= {min_delta_od}): {n_flat} — skipping")
    print(f"    Candidates for rescue (dOD > {min_delta_od}): {len(candidates)}")

    if candidates.empty:
        print("  No candidates to analyze.")
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for _, brow in candidates.iterrows():
        strain = brow['strain']
        group = brow['group']

        t, od = load_raw_data(data_dir, strain, group)
        if t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(od)
        t, od = t[mask].astype(float), od[mask].astype(float)

        # Run ensemble truncation
        try:
            ens_result = ensemble_truncate(t, od, config)
        except Exception as e:
            print(f"    {strain}: ensemble failed ({e})")
            continue

        # Fit Gompertz at each method's truncation
        method_results = ens_result.method_results
        for method in METHODS:
            if method == 'consensus':
                trunc_idx = ens_result.consensus_idx
                trunc_time = ens_result.consensus_time
            elif method in method_results:
                trunc_idx = method_results[method].get('idx')
                trunc_time = method_results[method].get('time')
            else:
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': None, 'trunc_time': None, 'n_points': None,
                    'fit_success': False, 'r_squared': None, 'rmse': None,
                    'gompertz_A': None, 'gompertz_mu': None, 'gompertz_lambda': None,
                    'a_err_pct': None, 'mu_err_pct': None,
                    'original_reason': brow['classification_reason'],
                    'delta_od': brow['delta_od'],
                })
                continue

            if trunc_idx is None:
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': None, 'trunc_time': None, 'n_points': None,
                    'fit_success': False, 'r_squared': None, 'rmse': None,
                    'gompertz_A': None, 'gompertz_mu': None, 'gompertz_lambda': None,
                    'a_err_pct': None, 'mu_err_pct': None,
                    'original_reason': brow['classification_reason'],
                    'delta_od': brow['delta_od'],
                })
                continue

            fit = fit_at_truncation(t, od, int(trunc_idx))

            if fit is not None and fit.success:
                a_err_pct = abs(fit.a_err / fit.a_opt * 100) if fit.a_opt != 0 else None
                mu_err_pct = abs(fit.mu_err / fit.mu_opt * 100) if fit.mu_opt != 0 else None
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': int(trunc_idx), 'trunc_time': trunc_time,
                    'n_points': int(trunc_idx) + 1,
                    'fit_success': True,
                    'r_squared': fit.r_squared,
                    'rmse': fit.rmse,
                    'gompertz_A': fit.a_opt,
                    'gompertz_mu': fit.mu_opt,
                    'gompertz_lambda': fit.lambda_opt,
                    'a_err_pct': a_err_pct,
                    'mu_err_pct': mu_err_pct,
                    'original_reason': brow['classification_reason'],
                    'delta_od': brow['delta_od'],
                })
            else:
                rows.append({
                    'strain': strain, 'group': group, 'method': method,
                    'trunc_idx': int(trunc_idx), 'trunc_time': trunc_time,
                    'n_points': int(trunc_idx) + 1,
                    'fit_success': False, 'r_squared': None, 'rmse': None,
                    'gompertz_A': None, 'gompertz_mu': None, 'gompertz_lambda': None,
                    'a_err_pct': None, 'mu_err_pct': None,
                    'original_reason': brow['classification_reason'],
                    'delta_od': brow['delta_od'],
                })

    bad_comp_df = pd.DataFrame(rows)
    if bad_comp_df.empty:
        print("  No fits produced.")
        return bad_comp_df, pd.DataFrame()

    # Identify rescued strains: any method gives R² >= 0.95 and param errors < 20%
    rescued_rows = []
    for strain, sdf in bad_comp_df.groupby('strain'):
        success = sdf[(sdf['fit_success'] == True) &
                      (sdf['r_squared'] >= 0.95)]
        if success.empty:
            continue
        # Check param errors
        for _, frow in success.iterrows():
            a_ok = frow['a_err_pct'] is not None and frow['a_err_pct'] < 20
            mu_ok = frow['mu_err_pct'] is not None and frow['mu_err_pct'] < 20
            if a_ok and mu_ok:
                rescued_rows.append({
                    'strain': strain,
                    'group': frow['group'],
                    'rescue_method': frow['method'],
                    'rescue_r2': frow['r_squared'],
                    'rescue_rmse': frow['rmse'],
                    'a_err_pct': frow['a_err_pct'],
                    'mu_err_pct': frow['mu_err_pct'],
                    'original_reason': frow['original_reason'],
                    'delta_od': frow['delta_od'],
                })

    rescued_df = pd.DataFrame(rescued_rows)

    # Print summary
    n_candidates = bad_comp_df['strain'].nunique()
    n_rescued = rescued_df['strain'].nunique() if not rescued_df.empty else 0
    print(f"\n  Bad strain rescue results:")
    print(f"    Candidates analyzed: {n_candidates}")
    print(f"    Rescued (R²>=0.95, errors<20%): {n_rescued}")
    if not rescued_df.empty:
        print(f"\n    Rescued strains:")
        for strain in rescued_df['strain'].unique():
            sr = rescued_df[rescued_df['strain'] == strain].iloc[0]
            print(f"      {strain}: {sr['rescue_method']} R²={sr['rescue_r2']:.4f}")

    return bad_comp_df, rescued_df


# =============================================================================
# Terminal Output
# =============================================================================

def print_summary(summary_df, strain_best_df, comp_df):
    """Print comparison results to terminal."""
    n_strains = comp_df['strain'].nunique()

    print(f"\n{'='*70}")
    print(f"  TRUNCATION METHOD COMPARISON ({n_strains} strains)")
    print(f"{'='*70}\n")

    print(f"{'Rank':<6}{'Method':<22}{'Mean R²':<10}{'Med R²':<10}"
          f"{'%Good':<8}{'%Excl':<8}{'Fail':<8}{'RMSE':<10}")
    print('-' * 70)
    for i, row in summary_df.iterrows():
        mr2 = f"{row['mean_r2']:.4f}" if row['mean_r2'] is not None else 'N/A'
        mdr2 = f"{row['median_r2']:.4f}" if row['median_r2'] is not None else 'N/A'
        pg = f"{row['pct_good']:.0f}%" if row['pct_good'] is not None else 'N/A'
        pe = f"{row['pct_excellent']:.0f}%" if row['pct_excellent'] is not None else 'N/A'
        fail = f"{row['n_failures']}/{row['n_strains']}"
        rmse = f"{row['mean_rmse']:.4f}" if row['mean_rmse'] is not None else 'N/A'
        print(f"{i+1:<6}{row['label']:<22}{mr2:<10}{mdr2:<10}{pg:<8}{pe:<8}{fail:<8}{rmse:<10}")

    # Best method per strain
    print(f"\nBest method per strain:")
    if not strain_best_df.empty:
        counts = strain_best_df['best_method'].value_counts()
        for method, count in counts.items():
            pct = count / len(strain_best_df) * 100
            label = METHOD_LABELS.get(method, method)
            print(f"  {label}: {count}/{len(strain_best_df)} ({pct:.0f}%)")

    # Consensus vs best
    both = strain_best_df.dropna(subset=['best_r2', 'consensus_r2'])
    if len(both) > 0:
        wins = (both['consensus_r2'] > both['best_r2'] + 0.001).sum()
        ties = ((both['consensus_r2'] - both['best_r2']).abs() <= 0.001).sum()
        losses = (both['consensus_r2'] < both['best_r2'] - 0.001).sum()
        print(f"\nConsensus vs best individual method:")
        print(f"  Consensus wins: {wins}  |  Ties (±0.001): {ties}  |  Best individual wins: {losses}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare truncation methods')
    parser.add_argument('--ensemble-csv', type=str, default=None,
                        help='Path to ensemble_truncation_results.csv')
    parser.add_argument('--results-csv', type=str, default=None,
                        help='Path to all_groups_results.csv')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for comparison results')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--include-bad', action='store_true',
                        help='Also analyze bad strains to check for rescues')
    args = parser.parse_args()

    # Resolve paths
    base = Path(__file__).parent.parent
    config = load_config(args.config) if args.config else load_config()

    ensemble_csv = Path(args.ensemble_csv) if args.ensemble_csv else \
        base / 'results' / 'tables' / 'Advanced_Analysis' / 'ensemble_truncation' / 'ensemble_truncation_results.csv'
    results_csv = Path(args.results_csv) if args.results_csv else \
        base / 'results' / 'tables' / 'all_groups_results.csv'
    output_dir = Path(args.output_dir) if args.output_dir else \
        base / 'results' / 'tables' / 'Advanced_Analysis' / 'truncation_comparison'
    data_dir = base / 'data' / 'raw'
    plot_dir = output_dir / 'plots'

    # Check inputs
    if not ensemble_csv.exists():
        print(f"ERROR: Ensemble results not found at {ensemble_csv}")
        print("Run: python 06_advanced_fitting.py --ensemble-only")
        return 1
    if not results_csv.exists():
        print(f"ERROR: Pipeline results not found at {results_csv}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    ensemble_df = pd.read_csv(ensemble_csv)
    results_df = pd.read_csv(results_csv)
    print(f"  Ensemble: {len(ensemble_df)} strains")
    print(f"  Pipeline: {len(results_df)} strains ({results_df['is_good'].sum()} good)")

    # Run comparison
    print("\nFitting Gompertz at each method's truncation point...")
    comp_df = run_comparison(ensemble_df, results_df, data_dir, config)
    print(f"  {len(comp_df)} total fits ({comp_df['fit_success'].sum()} successful)")

    # Save raw comparison
    comp_df.to_csv(output_dir / 'method_comparison.csv', index=False)

    # Compute summaries
    summary_df = compute_method_summary(comp_df, config)
    summary_df.to_csv(output_dir / 'method_summary.csv', index=False)

    pairwise_df = compute_pairwise_agreement(comp_df)
    pairwise_df.to_csv(output_dir / 'pairwise_agreement.csv', index=False)

    strain_best_df = compute_strain_best(comp_df)
    strain_best_df.to_csv(output_dir / 'strain_best_method.csv', index=False)

    # Aggregate plots
    print("\nGenerating aggregate plots...")
    plot_r2_boxplot(comp_df, summary_df, str(plot_dir / 'r2_boxplot.png'))
    plot_ranking_bar(summary_df, str(plot_dir / 'method_ranking.png'))
    plot_truncation_scatter_matrix(comp_df, str(plot_dir / 'truncation_scatter_matrix.png'))
    plot_confidence_calibration(comp_df, ensemble_df, str(plot_dir / 'confidence_calibration.png'))
    plot_strain_heatmap(comp_df, strain_best_df, str(plot_dir / 'strain_heatmap.png'))
    plot_consensus_vs_best(strain_best_df, str(plot_dir / 'consensus_vs_best.png'))

    # Per-strain overlay plots
    generate_overlay_plots(comp_df, results_df, data_dir, output_dir)

    # Terminal summary
    print_summary(summary_df, strain_best_df, comp_df)

    # Bad strain rescue analysis
    if args.include_bad:
        print(f"\n{'='*70}")
        print(f"  BAD STRAIN RESCUE ANALYSIS")
        print(f"{'='*70}")

        bad_comp_df, rescued_df = run_bad_strain_analysis(
            results_df, data_dir, config, output_dir
        )

        if not bad_comp_df.empty:
            bad_comp_df.to_csv(output_dir / 'bad_strain_comparison.csv', index=False)
            print(f"\n  Bad strain comparison saved to {output_dir / 'bad_strain_comparison.csv'}")

            # Generate overlay plots for bad strains
            generate_overlay_plots(bad_comp_df, results_df, data_dir, output_dir, is_bad=True)

        if not rescued_df.empty:
            rescued_df.to_csv(output_dir / 'rescued_strains.csv', index=False)
            print(f"  Rescued strains saved to {output_dir / 'rescued_strains.csv'}")
        else:
            print("  No strains rescued.")

    print(f"\nResults saved to {output_dir}")
    print(f"Plots saved to {plot_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
