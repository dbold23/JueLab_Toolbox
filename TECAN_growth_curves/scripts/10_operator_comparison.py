#!/usr/bin/env python3
"""
Step 10: Inter-Operator Reproducibility Analysis

Compares Gompertz growth parameters across operators for shared strains
and controls to assess inter-operator reproducibility.

Usage:
    python 10_operator_comparison.py --input all_groups_results.csv --output-dir Operator_Comparison/
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_TUKEY = True
except ImportError:
    HAS_TUKEY = False


# ---------------------------------------------------------------------------
# Strain parsing (mirrors 03_statistical_analysis.py logic)
# ---------------------------------------------------------------------------

PESTICIDE_MAP = {
    "FLUPYRADIFURONEANDLB": "Flupyradifurone",
    "FLUPYRADIFURONEANDG":  "Flupyradifurone",
    "FLUPYRADIFURONE":      "Flupyradifurone",
    "BIFENTHRINANDLB":      "Bifenthrin",
    "BIFENTHRINANDG":       "Bifenthrin",
    "BIFENTHRIN":           "Bifenthrin",
    "LAMBDACYHALOTHRINANDLB": "Lambda-Cyhalothrin",
    "LAMBDACYHALOTHRINANDG": "Lambda-Cyhalothrin",
    "LAMBDACYHALOTHRIN":    "Lambda-Cyhalothrin",
    "DIAZINONANDLB":        "Diazinon",
    "DIAZINONANDG":         "Diazinon",
    "DIAZINON":             "Diazinon",
    "MALATHIONANDLB":       "Malathion",
    "MALATHIONANDG":        "Malathion",
    "MALATHION":            "Malathion",
    "IMIDACLOPRIDANDLB":    "Imidacloprid",
    "IMIDACLOPRIDANDG":     "Imidacloprid",
    "IMIDACLOPRID":         "Imidacloprid",
    "PERMETHRINANDLB":      "Permethrin",
    "PERMETHRINANDG":       "Permethrin",
    "PERMETHRIN":           "Permethrin",
    "LB":                   "Control-LB",
    "H2O":                  "Control-H2O",
}


def parse_strain(strain_str):
    """Parse strain string into (treatment, strain_id, pesticide)."""
    treatment = strain_str.rsplit("-", 1)[0]
    strain_id = strain_str.rsplit("-", 1)[1] if "-" in strain_str else strain_str

    key = treatment.upper()
    pesticide = "Unknown"
    for pattern, name in PESTICIDE_MAP.items():
        if key == pattern:
            pesticide = name
            break

    # Normalize strain_id: remove trailing Y/Z suffixes for cross-group matching
    base_id = re.sub(r'[YZ]$', '', strain_id.upper())

    return treatment, strain_id, pesticide, base_id


def classify_treatment(treatment):
    """Classify treatment type."""
    t = treatment.upper()
    if t == "LB":
        return "LB_only"
    if t == "H2O":
        return "H2O_only"
    if "ANDLB" in t or "ANDG" in t:
        return "Pesticide_plus_nutrient"
    return "Pesticide_only"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path):
    """Load results CSV and add parsed columns."""
    df = pd.read_csv(csv_path)

    # Parse strain info
    parsed = df['strain'].apply(parse_strain)
    df['treatment'] = parsed.apply(lambda x: x[0])
    df['strain_id'] = parsed.apply(lambda x: x[1])
    df['pesticide'] = parsed.apply(lambda x: x[2])
    df['base_id'] = parsed.apply(lambda x: x[3])
    df['treatment_type'] = df['treatment'].apply(classify_treatment)

    # Compute doubling time
    df['doubling_time'] = np.where(
        df['gompertz_mu'] > 0,
        np.log(2) / df['gompertz_mu'],
        np.nan
    )

    return df


def find_shared_strains(df):
    """Find strain base IDs that appear in multiple operators."""
    good = df[df['is_good'] == True].copy()

    # Group by pesticide + base_id, find which operators have each
    shared = []
    for (pest, bid), grp in good.groupby(['pesticide', 'base_id']):
        operators = grp['operator'].unique()
        if len(operators) > 1:
            shared.append({
                'pesticide': pest,
                'base_id': bid,
                'operators': list(operators),
                'n_operators': len(operators),
                'n_curves': len(grp),
            })

    return pd.DataFrame(shared) if shared else pd.DataFrame()


def run_operator_anova(df, shared_df, output_dir):
    """Run ANOVA on Gompertz parameters across operators for shared strains."""
    good = df[(df['is_good'] == True)].copy()
    params = ['gompertz_mu', 'gompertz_lambda', 'gompertz_a']
    param_labels = {'gompertz_mu': 'Growth Rate (mu)', 'gompertz_lambda': 'Lag Time (lambda)', 'gompertz_a': 'Max OD (A)'}

    results = []

    if shared_df.empty:
        print("  No shared strains found across operators.")
        return pd.DataFrame()

    for _, row in shared_df.iterrows():
        pest = row['pesticide']
        bid = row['base_id']
        subset = good[(good['pesticide'] == pest) & (good['base_id'] == bid)]

        if len(subset) < 3:
            continue

        for param in params:
            vals = subset[param].dropna()
            if len(vals) < 3:
                continue

            groups = [grp[param].dropna().values for _, grp in subset.groupby('operator')]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                continue

            # ANOVA (or t-test for 2 groups)
            if len(groups) == 2:
                stat, pval = stats.ttest_ind(*groups, equal_var=False)
                test = "Welch t-test"
            else:
                stat, pval = stats.f_oneway(*groups)
                test = "ANOVA"

            # CV across operators
            operator_means = [np.mean(g) for g in groups]
            cv = np.std(operator_means) / np.mean(operator_means) * 100 if np.mean(operator_means) != 0 else np.nan

            results.append({
                'pesticide': pest,
                'strain_base_id': bid,
                'parameter': param_labels.get(param, param),
                'test': test,
                'statistic': stat,
                'p_value': pval,
                'cv_percent': cv,
                'n_operators': len(groups),
                'n_total': sum(len(g) for g in groups),
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


def run_lb_control_comparison(df):
    """Compare LB control parameters across all operators."""
    lb = df[(df['is_good'] == True) & (df['treatment'].str.upper() == 'LB')].copy()

    if lb.empty or lb['operator'].nunique() < 2:
        print("  Insufficient LB control data across operators.")
        return pd.DataFrame()

    params = ['gompertz_mu', 'gompertz_lambda', 'gompertz_a', 'doubling_time']
    param_labels = {
        'gompertz_mu': 'Growth Rate (mu)',
        'gompertz_lambda': 'Lag Time (lambda)',
        'gompertz_a': 'Max OD (A)',
        'doubling_time': 'Doubling Time (h)',
    }

    results = []
    for param in params:
        groups = [grp[param].dropna().values for _, grp in lb.groupby('operator')]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        if len(groups) == 2:
            stat, pval = stats.ttest_ind(*groups, equal_var=False)
            test = "Welch t-test"
        else:
            stat, pval = stats.f_oneway(*groups)
            test = "ANOVA"

        operator_means = [np.mean(g) for g in groups]
        cv = np.std(operator_means) / np.mean(operator_means) * 100 if np.mean(operator_means) != 0 else np.nan

        results.append({
            'parameter': param_labels.get(param, param),
            'test': test,
            'statistic': stat,
            'p_value': pval,
            'cv_percent': cv,
            'n_operators': len(groups),
            'n_total': sum(len(g) for g in groups),
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_operator_boxplots(df, output_dir):
    """Box/violin plots of mu and lambda by operator for shared pesticides."""
    good = df[df['is_good'] == True].copy()

    # Focus on pesticides with data from multiple operators
    multi_op = good.groupby('pesticide')['operator'].nunique()
    shared_pesticides = multi_op[multi_op > 1].index.tolist()

    if not shared_pesticides:
        print("  No pesticides with multi-operator data for plotting.")
        return

    params = [('gompertz_mu', 'Growth Rate (mu, h$^{-1}$)'),
              ('gompertz_lambda', 'Lag Time (lambda, h)')]

    fig, axes = plt.subplots(len(params), 1, figsize=(10, 5 * len(params)))
    if len(params) == 1:
        axes = [axes]

    for ax, (param, label) in zip(axes, params):
        plot_data = good[good['pesticide'].isin(shared_pesticides)].copy()
        plot_data = plot_data.dropna(subset=[param])

        if plot_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Create grouped boxplot
        operators = sorted(plot_data['operator'].unique())
        pesticides = sorted(plot_data['pesticide'].unique())
        n_pest = len(pesticides)
        width = 0.8 / len(operators)

        for i, op in enumerate(operators):
            positions = np.arange(n_pest) + (i - len(operators)/2 + 0.5) * width
            op_data = [plot_data[(plot_data['operator'] == op) & (plot_data['pesticide'] == p)][param].dropna().values
                       for p in pesticides]
            bp = ax.boxplot(op_data, positions=positions, widths=width * 0.8,
                           patch_artist=True, showfliers=True)
            color = plt.cm.Set2(i / max(len(operators) - 1, 1))
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            # Legend entry
            ax.plot([], [], 's', color=color, label=op, markersize=10)

        ax.set_xticks(range(n_pest))
        ax.set_xticklabels(pesticides, rotation=30, ha='right')
        ax.set_ylabel(label)
        ax.legend(title='Operator', loc='upper right')
        ax.set_title(f'{label} by Operator')

    plt.tight_layout()
    fig.savefig(output_dir / 'operator_parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: operator_parameter_comparison.png")


def figure_lb_controls(df, output_dir):
    """Compare LB control growth across operators."""
    lb = df[(df['is_good'] == True) & (df['treatment'].str.upper() == 'LB')].copy()

    if lb.empty or lb['operator'].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    params = [('gompertz_mu', 'Growth Rate (h$^{-1}$)'),
              ('gompertz_lambda', 'Lag Time (h)'),
              ('doubling_time', 'Doubling Time (h)')]

    for ax, (param, label) in zip(axes, params):
        operators = sorted(lb['operator'].unique())
        data = [lb[lb['operator'] == op][param].dropna().values for op in operators]
        bp = ax.boxplot(data, labels=operators, patch_artist=True, showfliers=True)
        colors = [plt.cm.Set2(i / max(len(operators) - 1, 1)) for i in range(len(operators))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(label)
        ax.set_title(f'LB Controls: {label}')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig.savefig(output_dir / 'lb_control_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: lb_control_comparison.png")


def figure_cv_heatmap(anova_results, output_dir):
    """Heatmap of coefficient of variation by parameter and pesticide."""
    if anova_results.empty:
        return

    pivot = anova_results.pivot_table(
        values='cv_percent', index='pesticide', columns='parameter', aggfunc='mean'
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.8)))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                       color='white' if val > 30 else 'black', fontsize=10)

    ax.set_title('Inter-Operator CV (%) by Parameter and Pesticide')
    plt.colorbar(im, ax=ax, label='CV (%)')
    plt.tight_layout()
    fig.savefig(output_dir / 'operator_cv_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: operator_cv_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inter-operator reproducibility analysis")
    parser.add_argument('--input', type=str, required=True, help="Path to all_groups_results.csv")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Inter-Operator Reproducibility Analysis")
    print(f"{'='*60}\n")

    # Load and prepare data
    df = load_and_prepare(input_path)
    print(f"  Loaded {len(df)} curves from {input_path.name}")
    print(f"  Operators: {sorted(df['operator'].unique())}")
    print(f"  Groups: {sorted(df['group'].unique())}")

    good = df[df['is_good'] == True]
    print(f"  Good fits: {len(good)} / {len(df)}")

    # Summary by operator
    print(f"\n  --- Curves by Operator ---")
    for op, grp in good.groupby('operator'):
        pesticides = sorted(grp['pesticide'].unique())
        print(f"  {op}: {len(grp)} good curves, pesticides: {', '.join(pesticides)}")

    # Find shared strains
    shared = find_shared_strains(df)
    if not shared.empty:
        print(f"\n  --- Shared Strains Across Operators ---")
        for _, row in shared.iterrows():
            print(f"  {row['pesticide']} {row['base_id']}: {row['operators']} ({row['n_curves']} curves)")
        shared.to_csv(output_dir / 'shared_strains.csv', index=False)
    else:
        print("\n  No shared strains found across operators.")

    # ANOVA on shared strains
    print(f"\n  --- ANOVA on Shared Strains ---")
    anova_results = run_operator_anova(df, shared, output_dir)
    if not anova_results.empty:
        for _, row in anova_results.iterrows():
            sig = "*" if row['p_value'] < 0.05 else "ns"
            print(f"  {row['pesticide']} {row['strain_base_id']} | "
                  f"{row['parameter']}: p={row['p_value']:.4f} {sig}, CV={row['cv_percent']:.1f}%")
        anova_results.to_csv(output_dir / 'operator_anova_results.csv', index=False)
        print(f"  Saved: operator_anova_results.csv")

    # LB control comparison
    print(f"\n  --- LB Control Comparison ---")
    lb_results = run_lb_control_comparison(df)
    if not lb_results.empty:
        for _, row in lb_results.iterrows():
            sig = "*" if row['p_value'] < 0.05 else "ns"
            print(f"  {row['parameter']}: p={row['p_value']:.4f} {sig}, CV={row['cv_percent']:.1f}%")
        lb_results.to_csv(output_dir / 'lb_control_anova.csv', index=False)
        print(f"  Saved: lb_control_anova.csv")

    # Operator summary table
    summary_rows = []
    for op, grp in good.groupby('operator'):
        summary_rows.append({
            'operator': op,
            'n_curves': len(grp),
            'n_pesticides': grp['pesticide'].nunique(),
            'pesticides': ', '.join(sorted(grp['pesticide'].unique())),
            'mean_mu': grp['gompertz_mu'].mean(),
            'std_mu': grp['gompertz_mu'].std(),
            'mean_lambda': grp['gompertz_lambda'].mean(),
            'std_lambda': grp['gompertz_lambda'].std(),
            'mean_r2': grp['fit_r_squared'].mean() if 'fit_r_squared' in grp.columns else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'operator_summary.csv', index=False)
    print(f"\n  Saved: operator_summary.csv")

    # Figures
    print(f"\n  --- Generating Figures ---")
    figure_operator_boxplots(df, output_dir)
    figure_lb_controls(df, output_dir)
    if not anova_results.empty:
        figure_cv_heatmap(anova_results, output_dir)

    # Overall reproducibility assessment
    print(f"\n  --- Reproducibility Assessment ---")
    if not anova_results.empty:
        sig_count = (anova_results['p_value'] < 0.05).sum()
        total = len(anova_results)
        print(f"  Significant differences (p<0.05): {sig_count}/{total} comparisons")
        mean_cv = anova_results['cv_percent'].mean()
        print(f"  Mean CV across parameters: {mean_cv:.1f}%")
        if mean_cv < 20:
            print(f"  Assessment: GOOD reproducibility (CV < 20%)")
        elif mean_cv < 40:
            print(f"  Assessment: MODERATE reproducibility (CV 20-40%)")
        else:
            print(f"  Assessment: POOR reproducibility (CV > 40%)")

    print(f"\n  All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
