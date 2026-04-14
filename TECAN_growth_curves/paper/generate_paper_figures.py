#!/usr/bin/env python3
"""
Generate all 8 publication figures for the TECAN growth curve analysis paper.

Reads raw CSV result files and produces matplotlib figures with PLOS ONE
formatting.  Each figure is saved as both PNG (300 DPI) and PDF (vector).

Usage:
    python generate_paper_figures.py              # Generate all 8 figures
    python generate_paper_figures.py --fig 3      # Generate only Figure 3
    python generate_paper_figures.py --fig 1,3,6  # Generate specific figures

Output directory:  paper/figures/
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent          # TECAN_growth_curves/
RESULTS = BASE / "results" / "tables"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palettes and mappings
# ---------------------------------------------------------------------------
COLORBLIND_PALETTE = [
    '#0173B2', '#DE8F05', '#029E73', '#D55E00',
    '#CC78BC', '#CA9161', '#FBAFE4', '#949494',
    '#ECE133', '#56B4E9',
]

PESTICIDE_COLORS = {
    'IMIDACLOPRID':       '#9B59B6',
    'FLUPYRADIFURONE':    '#8E44AD',
    'BIFENTHRIN':         '#E67E22',
    'LAMBDACYHALOTHRIN':  '#F39C12',
    'PERMETHRIN':         '#D35400',
    'MALATHION':          '#27AE60',
    'DIAZINON':           '#2ECC71',
    'LB Control':         '#3498DB',
    'H2O Control':        '#85C1E9',
    'Other':              '#95A5A6',
}

PESTICIDE_CLASS = {
    'IMIDACLOPRID':       'Neonicotinoid',
    'FLUPYRADIFURONE':    'Neonicotinoid',
    'BIFENTHRIN':         'Pyrethroid',
    'LAMBDACYHALOTHRIN':  'Pyrethroid',
    'PERMETHRIN':         'Pyrethroid',
    'MALATHION':          'Organophosphate',
    'DIAZINON':           'Organophosphate',
}

CLASS_COLORS = {
    'Neonicotinoid':     '#9B59B6',
    'Organophosphate':   '#27AE60',
    'Pyrethroid':        '#E67E22',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_global_style():
    """Configure matplotlib rcParams for publication quality."""
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          8,
        'axes.titlesize':     10,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth':     0.5,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.facecolor':     'white',
        'figure.facecolor':   'white',
        'axes.grid':          False,
        'pdf.fonttype':       42,   # embed TrueType in PDF
        'ps.fonttype':        42,
    })


def save_figure(fig, name):
    """Save as PNG (300 DPI) and PDF, then close."""
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f"  -> {png.name}  |  {pdf.name}")


def panel_label(ax, label, x=-0.08, y=1.08):
    """Bold (A)/(B)/... label at the upper-left corner of an axes."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


def get_pesticide(strain):
    """Parse pesticide name from a strain identifier string."""
    s = strain.upper().split('-')[0]
    for p in ['FLUPYRADIFURONE', 'BIFENTHRIN', 'LAMBDACYHALOTHRIN',
              'DIAZINON', 'MALATHION', 'IMIDACLOPRID', 'PERMETHRIN']:
        if p in s:
            return p
    if s == 'LB':
        return 'LB Control'
    if s == 'H2O':
        return 'H2O Control'
    return 'Other'


def read_csv_safe(path, **kwargs):
    """Read CSV with graceful handling of missing files."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: file not found -- {p}")
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as exc:
        print(f"  WARNING: error reading {p}: {exc}")
        return None


# ===================================================================
# FIGURE 1 -- Pipeline Schematic  (7.2 x 5 in, double column)
# ===================================================================
def figure1_pipeline():
    """Two-column flowchart of the 11-step pipeline."""
    print("Figure 1: Pipeline Schematic")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 9.5)
    ax.axis('off')

    # Category colour map
    CAT = {
        'pre':  '#4A90D9',   # Preprocessing
        'fit':  '#7ED321',   # Fitting
        'ml':   '#F5A623',   # ML
        'ana':  '#9B59B6',   # Analysis
        'out':  '#E74C3C',   # Output
    }

    # Single-column linear flow: what a user actually runs
    steps = [
        (1,  "Raw TECAN Data\n(96-well CSV)",                'pre'),
        (2,  "Preprocessing\n(blank subtraction, averaging)",'pre'),
        (3,  "MCCV Truncation\n+ Gompertz Fitting",         'fit'),
        (4,  "ML Classification\n(two-stage: pre-fit + post-fit)", 'ml'),
        (5,  "Combine Results\n(all groups)",                'fit'),
        (6,  "Haldane Inhibition\nModeling (ODE + AICc)",   'ana'),
        (7,  "Bayesian Analysis\n(hierarchical Ki + CIs)",  'ana'),
        (8,  "Statistical Summary\n+ Publication Figures",   'out'),
    ]

    bw, bh = 3.5, 0.82
    cx = 5.0
    y_start = 8.5
    y_step = 1.2

    for i, (num, label, cat) in enumerate(steps):
        cy = y_start - i * y_step
        color = CAT[cat]
        rect = FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor='#333333', linewidth=0.8, alpha=0.88)
        ax.add_patch(rect)
        ax.text(cx, cy + 0.02, label, ha='center', va='center',
                fontsize=7, color='white', linespacing=1.15)
        # Step number badge
        ax.text(cx - bw / 2 + 0.2, cy + bh / 2 - 0.17, str(num),
                ha='left', va='top', fontsize=7.5, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.12',
                          facecolor='#00000030', edgecolor='none'))

        # Arrow to next
        if i < len(steps) - 1:
            cy_next = y_start - (i + 1) * y_step
            ax.annotate('', xy=(cx, cy_next + bh / 2 + 0.02),
                         xytext=(cx, cy - bh / 2 - 0.02),
                         arrowprops=dict(arrowstyle='->', color='#555',
                                         lw=1.2, mutation_scale=12))

    # Right-side annotations (key outputs at each stage)
    annotations = [
        (0, "OD600 time series"),
        (1, "161 averaged curves\n6 groups, 3 operators"),
        (2, "Gompertz A, mu, lambda\nR-squared, RMSE"),
        (3, "66 GOOD / 95 BAD\nP(good) scores"),
        (4, "all_groups_results.csv"),
        (5, "Ki per pesticide\n30/42 Haldane preferred"),
        (6, "Posterior Ki with HDI\nBootstrap CIs"),
        (7, "Figures, tables,\nR-ready export"),
    ]
    for i, (_, note) in enumerate(annotations):
        cy = y_start - i * y_step
        ax.text(cx + bw / 2 + 0.2, cy, note,
                ha='left', va='center', fontsize=5.5, color='#555',
                fontstyle='italic', linespacing=1.2)

    # Category legend (bottom-left)
    leg_items = [
        ('Preprocessing',     CAT['pre']),
        ('ML Classification', CAT['ml']),
        ('Curve Fitting',     CAT['fit']),
        ('Analysis',          CAT['ana']),
        ('Output',            CAT['out']),
    ]
    for i, (lbl, c) in enumerate(leg_items):
        yy = -0.3 + i * 0.35
        ax.add_patch(FancyBboxPatch(
            (0.3, yy), 0.3, 0.22, boxstyle="round,pad=0.05",
            facecolor=c, edgecolor='#333', linewidth=0.5, alpha=0.88))
        ax.text(0.75, yy + 0.11, lbl, fontsize=6.5, va='center')

    save_figure(fig, 'Figure1_pipeline')


# ===================================================================
# FIGURE 2 -- Dataset Overview  (7.2 x 6 in, 2x2)
# ===================================================================
def figure2_dataset_overview():
    """GOOD/BAD bar, mu boxplot, lambda boxplot, mu-lambda scatter."""
    print("Figure 2: Dataset Overview")

    df = read_csv_safe(RESULTS / "all_groups_results.csv")
    if df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))

    # ---- (A) Stacked bar: GOOD vs BAD per group ----
    ax = axes[0, 0]
    panel_label(ax, 'A')

    groups = sorted(df['group'].unique())
    good_n = [int((df['group'] == g).sum() - (~df['is_good'].astype(bool) & (df['group'] == g)).sum())
              for g in groups]
    bad_n  = [int((~df['is_good'].astype(bool) & (df['group'] == g)).sum())
              for g in groups]

    x = np.arange(len(groups))
    ax.bar(x, good_n, 0.6, label='GOOD', color='#27AE60',
           edgecolor='white', linewidth=0.5)
    ax.bar(x, bad_n, 0.6, bottom=good_n, label='BAD', color='#E74C3C',
           edgecolor='white', linewidth=0.5)

    for i in range(len(groups)):
        if good_n[i]:
            ax.text(i, good_n[i] / 2, str(good_n[i]),
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold')
        if bad_n[i]:
            ax.text(i, good_n[i] + bad_n[i] / 2, str(bad_n[i]),
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel('Number of Curves')
    ax.legend(fontsize=6, loc='upper right')

    # ---- helpers: GOOD-only subset, operator colours ----
    good = df[df['is_good'].astype(bool)].copy()
    op_colors = {'Operator1': '#0173B2', 'Walton': '#DE8F05', 'Dominique': '#029E73'}
    good_groups = sorted(good['group'].unique())

    def _boxplot_with_jitter(ax, col, ylabel, show_legend=False):
        data = [good.loc[good['group'] == g, col].dropna().values
                for g in good_groups]
        ax.boxplot(data, positions=range(len(good_groups)), widths=0.5,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='#D5E8D4', edgecolor='#333',
                                 linewidth=0.5),
                   medianprops=dict(color='#333', linewidth=1))
        rng = np.random.default_rng(42)
        for gi, g in enumerate(good_groups):
            sub = good[good['group'] == g]
            jit = rng.normal(0, 0.08, len(sub))
            for op in sub['operator'].unique():
                m = sub['operator'] == op
                c = op_colors.get(op, '#999')
                ax.scatter(gi + jit[m.values], sub.loc[m, col],
                           s=10, alpha=0.6, color=c, edgecolors='none',
                           zorder=3,
                           label=op if (gi == 0 and show_legend) else None)
        ax.set_xticks(range(len(good_groups)))
        ax.set_xticklabels(good_groups, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        if show_legend:
            handles = [mpatches.Patch(color=c, label=o) for o, c in op_colors.items()]
            ax.legend(handles=handles, fontsize=5, loc='upper right')

    # ---- (B) mu boxplot ----
    ax = axes[0, 1]
    panel_label(ax, 'B')
    _boxplot_with_jitter(ax, 'gompertz_mu', r'Growth Rate ($\mu$, hr$^{-1}$)',
                         show_legend=True)

    # ---- (C) lambda boxplot ----
    ax = axes[1, 0]
    panel_label(ax, 'C')
    _boxplot_with_jitter(ax, 'gompertz_lambda', r'Lag Time ($\lambda$, hr)')

    # ---- (D) lambda vs mu scatter coloured by pesticide ----
    ax = axes[1, 1]
    panel_label(ax, 'D')

    good = good.copy()
    good['pesticide'] = good['strain'].apply(get_pesticide)

    order = [p for p in PESTICIDE_COLORS if p in good['pesticide'].unique()]
    order += [p for p in good['pesticide'].unique() if p not in order]

    for pest in order:
        sub = good[good['pesticide'] == pest]
        c = PESTICIDE_COLORS.get(pest, '#95A5A6')
        ax.scatter(sub['gompertz_mu'], sub['gompertz_lambda'],
                   s=15, alpha=0.7, color=c, edgecolors='none', label=pest)

    ax.set_xlabel(r'Growth Rate ($\mu$, hr$^{-1}$)')
    ax.set_ylabel(r'Lag Time ($\lambda$, hr)')
    ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)

    fig.tight_layout()
    save_figure(fig, 'Figure2_dataset_overview')


# ===================================================================
# FIGURE 3 -- Synthetic Validation  (7.2 x 3.5 in, 1x4)
# ===================================================================
def figure3_synthetic_validation():
    """Confusion matrix + parameter-recovery scatter (mu, lambda, A)."""
    print("Figure 3: Synthetic Validation")

    gt_path  = (BASE / "synthetic_data" / "output" / "validation_holdout"
                / "test_data" / "ground_truth.csv")
    val_path = (BASE / "synthetic_data" / "output" / "validation_holdout"
                / "validation_latest" / "processing_results.csv")

    gt  = read_csv_safe(gt_path)
    val = read_csv_safe(val_path)
    if gt is None or val is None:
        return

    # Match on CURVE#### pattern
    def _curve_id(s):
        m = re.search(r'(CURVE\d+)', str(s))
        return m.group(1) if m else None

    gt['cid']  = gt['strain_name'].apply(_curve_id)
    val['cid'] = val['strain'].apply(_curve_id)

    merged = pd.merge(gt, val, on='cid', how='inner', suffixes=('_gt', '_val'))
    if merged.empty:
        print("  WARNING: No matching curves. Skipping Figure 3.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.5))

    # ---- (A) Confusion matrix ----
    ax = axes[0]
    panel_label(ax, 'A', x=-0.15)

    pred_good = merged['is_good'].astype(bool)
    true_good = merged['expected_class'] == 'GOOD'

    tp = int((pred_good &  true_good).sum())
    fn = int((~pred_good & true_good).sum())
    fp = int((pred_good & ~true_good).sum())
    tn = int((~pred_good & ~true_good).sum())

    cm = np.array([[tp, fp],
                   [fn, tn]])
    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks([0, 1]);  ax.set_xticklabels(['GOOD', 'BAD'], fontsize=7)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(['GOOD', 'BAD'], fontsize=7)
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_ylabel('True', fontsize=7)

    for i in range(2):
        for j in range(2):
            clr = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=clr)

    # ---- (B-D) Parameter recovery ----
    both_good = merged[(merged['expected_class'] == 'GOOD') &
                       (merged['is_good'].astype(bool))]

    params = [
        ('B', 'true_mu',     'gompertz_mu',     r'$\mu$'),
        ('C', 'true_lambda', 'gompertz_lambda', r'$\lambda$'),
        ('D', 'true_A',      'gompertz_a',      'A'),
    ]

    for idx, (lbl, tcol, pcol, nice) in enumerate(params):
        ax = axes[idx + 1]
        panel_label(ax, lbl, x=-0.15)

        xv = both_good[tcol].values.astype(float)
        yv = both_good[pcol].values.astype(float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]

        ax.scatter(xv, yv, s=12, alpha=0.5,
                   color=COLORBLIND_PALETTE[0], edgecolors='none')

        lo, hi = min(xv.min(), yv.min()), max(xv.max(), yv.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                '--', color='#888', lw=0.8, zorder=0)

        # R-squared (1 - SSres/SStot)
        if len(xv) > 2:
            ss_res = np.sum((yv - xv) ** 2)
            ss_tot = np.sum((xv - xv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.05, 0.92, f"R$^2$ = {r2:.3f}",
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(facecolor='white', edgecolor='#CCC',
                              linewidth=0.5, pad=2))

        ax.set_xlabel(f'True {nice}', fontsize=7)
        ax.set_ylabel(f'Predicted {nice}', fontsize=7)

    fig.tight_layout(w_pad=1.0)
    save_figure(fig, 'Figure3_synthetic_validation')


# ===================================================================
# FIGURE 4 -- Truncation Method Comparison  (7.2 x 4 in, 1x2)
# ===================================================================
def figure4_truncation_comparison():
    """Horizontal R-squared bars + grouped good/excellent bars."""
    print("Figure 4: Truncation Method Comparison")

    df = read_csv_safe(
        RESULTS / "Advanced_Analysis" / "truncation_comparison" / "method_summary.csv")
    if df is None:
        return

    # Drop methods with 100 % failure (changepoint)
    df = df[df['n_success'] > 0].copy()
    df = df.sort_values('mean_r2', ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4))

    y = np.arange(len(df))

    # ---- (A) Mean R-squared horizontal bars ----
    ax = axes[0]
    panel_label(ax, 'A')

    colours = plt.cm.YlGn(df['pct_excellent'].values / 100.0)
    ax.barh(y, df['mean_r2'], xerr=df['std_r2'], height=0.6,
            color=colours, edgecolor='#333', linewidth=0.5,
            error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('Mean R$^2$')
    ax.set_xlim(0.5, 1.08)

    for i, r2 in enumerate(df['mean_r2']):
        ax.text(r2 + 0.012, i, f"{r2:.3f}", va='center', fontsize=6)

    sm = plt.cm.ScalarMappable(cmap='YlGn', norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('% Excellent (R$^2$ > 0.99)', fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    # ---- (B) Good / Excellent grouped bars ----
    ax = axes[1]
    panel_label(ax, 'B')

    w = 0.35
    ax.barh(y - w / 2, df['pct_good'], height=w,
            color='#85C1E9', edgecolor='#333', linewidth=0.5,
            label='% Good (R$^2$ > 0.95)')
    ax.barh(y + w / 2, df['pct_excellent'], height=w,
            color='#2471A3', edgecolor='#333', linewidth=0.5,
            label='% Excellent (R$^2$ > 0.99)')

    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('Percentage of Strains (%)')
    ax.set_xlim(0, 110)
    ax.legend(fontsize=6, loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'Figure4_truncation_comparison')


# ===================================================================
# FIGURE 5 -- Complementary Models  (7.2 x 5 in, 1x2)
# ===================================================================
def figure5_haldane_vs_gompertz():
    """Side-by-side: Gompertz on truncated data vs Haldane on full data."""
    from scipy.integrate import solve_ivp

    print("Figure 5: Complementary Models (Gompertz + Haldane)")

    comp = read_csv_safe(RESULTS / "Haldane_Analysis" / "haldane_comparison.csv")
    if comp is None:
        return

    # Pick DIAZ28 in pesticide+LB: visible inhibition, 31h per-param disagreement
    target = 'DIAZINONANDLB-DIAZ28'
    match = comp[comp['strain'] == target]
    if match.empty:
        # Fallback: best Haldane fit
        match = comp.sort_values('haldane_r2', ascending=False).head(1)
    row = match.iloc[0]
    strain_name = row['strain']
    pesticide = row['pesticide']
    group = row['group']

    # Load per-parameter truncation times from processing results
    proc_csv = RESULTS / f"{group}_Results" / "processing_results.csv"
    proc_df = read_csv_safe(proc_csv)
    per_param = {}
    if proc_df is not None:
        pmatch = proc_df[proc_df['strain'] == strain_name]
        if len(pmatch):
            pr = pmatch.iloc[0]
            for p, col in [('$\\lambda$', 'lam_trunc_time'),
                           ('$\\mu$', 'mu_trunc_time'),
                           ('A', 'A_trunc_time')]:
                val = pr.get(col)
                if pd.notna(val):
                    per_param[p] = float(val)

    # Load raw time-series data
    raw_dir = BASE / "data" / "raw"
    # Build the expected CSV filename
    media_type = strain_name.split('-')[0]  # e.g. BifenthrinANDLB
    group_dirs = {
        'Group1': raw_dir / 'Group1' / 'Group_1_DATA',
        'Group2': raw_dir / 'Group 2' / 'Group_2_DATA',
        'Group3': raw_dir / 'Group 3' / 'Group_3_DATA',
        'Group4': raw_dir / 'Group 4' / 'Group4_DATA',
        'Group5': raw_dir / 'Group5' / 'Group5_DATA_processed',
    }
    data_dir = group_dirs.get(group)
    strain_short = strain_name.split('-')[-1]  # e.g. BIF2

    time_data = od_data = None
    if data_dir and data_dir.exists():
        csv_name = f"{media_type}_DATA.csv"
        csv_path = data_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            time_col = [c for c in df.columns if 'TIME' in c.upper()][0]
            # Find column matching this strain
            strain_cols = [c for c in df.columns if strain_short.upper() in c.upper()
                           and c != time_col]
            if strain_cols:
                time_data = df[time_col].values
                od_data = df[strain_cols[0]].values
                od_data = np.maximum(od_data, 0)  # blank-subtracted

    if time_data is None:
        print(f"  WARNING: Could not load raw data for {strain_name}, using synthetic")
        # Fallback: generate from fit parameters
        time_data = np.linspace(0, 48, 200)
        A, mu, lam = row['gompertz_a'], row['gompertz_mu'], row['gompertz_lambda']
        od_data = A * np.exp(-np.exp((mu * np.e / A) * (lam - time_data) + 1))
        od_data += np.random.default_rng(42).normal(0, 0.02, len(od_data))
        od_data = np.maximum(od_data, 0)

    # Gompertz model function
    def gompertz(t, A, mu, lam):
        return A * np.exp(-np.exp((mu * np.e / A) * (lam - t) + 1))

    # Get truncation point from processing results
    trunc_time = row.get('truncation_time', None)
    if trunc_time is None or pd.isna(trunc_time):
        # Estimate: find where OD first reaches ~95% of max
        max_od = np.max(od_data)
        idx_95 = np.where(od_data >= 0.95 * max_od)[0]
        trunc_time = time_data[idx_95[0]] if len(idx_95) > 0 else time_data[-1] * 0.7

    # Gompertz fit parameters from CSV
    A_g = row['gompertz_a']
    mu_g = row['gompertz_mu']
    lam_g = row['gompertz_lambda']

    # Haldane ODE parameters from CSV
    mu_max_h = row['haldane_mu_max']
    Ks_h = row['haldane_Ks']
    Ki_h = row['haldane_Ki']
    X_max_h = row['haldane_X_max']
    q_h = row['haldane_q']
    S0_h = row['haldane_S0']
    X0_h = max(float(np.mean(od_data[:3])), 0.001)

    # Generate model curves
    t_fine = np.linspace(0, time_data[-1], 500)
    # Use the latest per-param truncation for the Gompertz curve extent
    max_trunc = max(per_param.values()) if per_param else float(trunc_time)
    t_trunc_fine = np.linspace(0, max_trunc, 300)

    gompertz_curve = gompertz(t_trunc_fine, A_g, mu_g, lam_g)

    # Solve Haldane ODE
    def haldane_rhs(t, y):
        X, S = y
        S = max(S, 0)
        X = max(X, 0)
        if S < 1e-10:
            mu_S = 0.0
        else:
            mu_S = mu_max_h * S / (Ks_h + S + S**2 / Ki_h)
        dXdt = mu_S * X * (1 - X / X_max_h)
        dSdt = -q_h * mu_S * X
        return [dXdt, dSdt]

    sol = solve_ivp(haldane_rhs, [0, time_data[-1]], [X0_h, S0_h],
                    t_eval=t_fine, method='RK45', max_step=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4), sharey=True)

    # ---- (A) Gompertz on truncated data ----
    ax = axes[0]
    panel_label(ax, 'A')

    # Show all data, Gompertz curve, and per-parameter truncation lines
    ax.scatter(time_data, od_data,
               s=8, alpha=0.4, color='#333', zorder=2, label='Data')
    ax.plot(t_trunc_fine, gompertz_curve, '-', color='#E67E22', lw=2.0,
            label=f'Gompertz (R$^2$={row["gompertz_r2"]:.3f})', zorder=3)

    # Per-parameter truncation lines (the novel contribution)
    pp_colors = {'$\\lambda$': '#2ECC71', '$\\mu$': '#E74C3C', 'A': '#3498DB'}
    pp_styles = {'$\\lambda$': ':', '$\\mu$': '--', 'A': '-.'}
    for pname, ptime in sorted(per_param.items(), key=lambda x: x[1]):
        ax.axvline(ptime, color=pp_colors.get(pname, '#888'),
                   ls=pp_styles.get(pname, '--'), lw=1.2, alpha=0.8,
                   label=f'{pname} optimal ({ptime:.0f}h)')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD$_{600}$')
    ax.set_title('Gompertz: Growth Kinetics\n(per-parameter truncation)', fontsize=9)
    ax.legend(fontsize=5.5, loc='lower right', frameon=True, framealpha=0.9)

    # Annotate parameters + disagreement
    disagree = max(per_param.values()) - min(per_param.values()) if per_param else 0
    param_text = (f'A = {A_g:.2f} OD\n'
                  f'$\\mu$ = {mu_g:.3f} OD/h\n'
                  f'$\\lambda$ = {lam_g:.1f} h\n'
                  f'Disagreement: {disagree:.0f}h')
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
            fontsize=6, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                      edgecolor='#E67E22', alpha=0.9))

    # ---- (B) Haldane on full data ----
    ax = axes[1]
    panel_label(ax, 'B')

    ax.scatter(time_data, od_data, s=8, alpha=0.5, color='#333', zorder=2,
               label='Data (full)')

    if sol.success:
        ax.plot(sol.t, sol.y[0], '-', color='#3498DB', lw=2.0,
                label=f'Haldane ODE (R$^2$={row["haldane_r2"]:.3f})', zorder=3)
    else:
        ax.text(0.5, 0.5, 'ODE solve failed', transform=ax.transAxes,
                fontsize=10, ha='center', color='red')

    ax.set_xlabel('Time (hours)')
    ax.set_title(f'Haldane: Substrate Inhibition\n(full data, S$_0$={S0_h:.0f} mg/L)',
                 fontsize=9)
    ax.legend(fontsize=6, loc='lower right', frameon=True, framealpha=0.9)

    # Annotate parameters
    param_text = (f'$\\mu_{{max}}$ = {mu_max_h:.2f} h$^{{-1}}$\n'
                  f'K$_s$ = {Ks_h:.3f}\n'
                  f'K$_i$ = {Ki_h:.1f}\n'
                  f'K$_i$/S$_0$ = {Ki_h/S0_h:.2f}')
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
            fontsize=6, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB',
                      edgecolor='#3498DB', alpha=0.9))

    fig.suptitle(f'Complementary Models: {strain_name}', fontsize=10,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, 'figure5_haldane_vs_gompertz')


# ===================================================================
# FIGURE 6 -- Bayesian Ki Forest Plot  (3.5 x 4 in, single column)
# ===================================================================
def figure6_ki_forest_plot():
    """Horizontal forest plot of Bayesian Ki by pesticide."""
    print("Figure 6: Bayesian Ki Forest Plot")

    df = read_csv_safe(
        RESULTS / "Advanced_Analysis" / "bayesian_haldane"
        / "haldane_Ki_by_pesticide.csv")
    if df is None:
        return

    # Sort: most inhibitory (lowest Ki_median) at top
    df = df.sort_values('Ki_median', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.5, 4))

    for i, row in df.iterrows():
        pest = row['pesticide'].upper()
        cls = PESTICIDE_CLASS.get(pest, 'Other')
        clr = CLASS_COLORS.get(cls, '#95A5A6')
        lo, hi, med = row['Ki_hdi_low'], row['Ki_hdi_high'], row['Ki_median']

        ax.plot([lo, hi], [i, i], '-', color=clr, lw=2,
                solid_capstyle='round')
        ax.scatter(med, i, s=50, color=clr, edgecolors='#333',
                   linewidth=0.5, zorder=5)
        # value label just past the HDI bar
        ax.text(hi * 1.15, i, f"{med:.1f}", va='center', fontsize=6,
                color='#333')

    # Y labels with sample size
    labels = [f"{row['pesticide']}  (n={int(row['n_strains'])})"
              for _, row in df.iterrows()]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)

    # Reference threshold lines
    ax.axvline(5, ls='--', color='#E74C3C', lw=0.8, alpha=0.7, zorder=0)
    ax.axvline(10, ls='--', color='#F39C12', lw=0.8, alpha=0.7, zorder=0)
    top = len(df) - 0.3
    ax.text(5, top, 'Ki = 5', fontsize=6, color='#E74C3C', ha='center',
            va='bottom')
    ax.text(10, top, 'Ki = 10', fontsize=6, color='#F39C12', ha='center',
            va='bottom')

    ax.set_xscale('log')
    ax.set_xlabel('Inhibition Constant (Ki)')

    # Direction annotation along x-axis
    ax.text(0.02, -0.09, r'$\leftarrow$ More inhibitory',
            transform=ax.transAxes, fontsize=6, color='#555', ha='left')
    ax.text(0.98, -0.09, r'Less inhibitory $\rightarrow$',
            transform=ax.transAxes, fontsize=6, color='#555', ha='right')

    # Class legend
    handles = [mpatches.Patch(color=c, label=cl) for cl, c in CLASS_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc='lower right', framealpha=0.9)

    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, 'Figure6_Ki_forest_plot')


# ===================================================================
# FIGURE 7 -- Inter-Operator Reproducibility  (7.2 x 4 in, 1x2)
# ===================================================================
def figure7_operator_reproducibility():
    """CV% heatmap + mean CV bar chart with threshold bands."""
    print("Figure 7: Inter-Operator Reproducibility")

    df = read_csv_safe(RESULTS / "Operator_Comparison" / "operator_anova_results.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4))

    # Nicer parameter names
    pmap = {
        'Growth Rate (mu)': 'Growth Rate',
        'Lag Time (lambda)': 'Lag Time',
        'Max OD (A)':        'Max OD',
    }
    df['param_short'] = df['parameter'].map(pmap).fillna(df['parameter'])

    hm = df.pivot_table(index='strain_base_id', columns='param_short',
                         values='cv_percent', aggfunc='first')
    strain_order = [s for s in ['IMID2', 'IMID5', 'IMID6', 'IMID8']
                    if s in hm.index]
    col_order = [c for c in ['Growth Rate', 'Lag Time', 'Max OD']
                 if c in hm.columns]
    hm = hm.reindex(index=strain_order, columns=col_order)

    # ---- (A) Heatmap ----
    ax = axes[0]
    panel_label(ax, 'A')

    im = ax.imshow(hm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index)

    for r in range(hm.shape[0]):
        for c in range(hm.shape[1]):
            v = hm.iloc[r, c]
            if np.isfinite(v):
                clr = 'white' if v > 60 else 'black'
                ax.text(c, r, f"{v:.0f}%", ha='center', va='center',
                        fontsize=8, fontweight='bold', color=clr)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('CV (%)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ---- (B) Mean CV by parameter with threshold bands ----
    ax = axes[1]
    panel_label(ax, 'B')

    mean_cv = hm.mean(axis=0)
    std_cv  = hm.std(axis=0)
    xp = np.arange(len(mean_cv))

    ax.bar(xp, mean_cv, yerr=std_cv, width=0.5,
           color=COLORBLIND_PALETTE[:len(mean_cv)],
           edgecolor='#333', linewidth=0.5,
           error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8))
    ax.set_xticks(xp)
    ax.set_xticklabels(mean_cv.index, rotation=30, ha='right')
    ax.set_ylabel('Mean CV (%)')

    # Threshold lines and shaded bands
    ax.axhline(20, ls='--', color='#27AE60', lw=0.8, alpha=0.7)
    ax.axhline(40, ls='--', color='#F39C12', lw=0.8, alpha=0.7)
    ax.text(len(mean_cv) - 0.5, 21, 'Good (<20%)', fontsize=6,
            color='#27AE60', ha='right', va='bottom')
    ax.text(len(mean_cv) - 0.5, 41, 'Moderate (<40%)', fontsize=6,
            color='#F39C12', ha='right', va='bottom')
    ax.axhspan(0,  20, alpha=0.05, color='#27AE60', zorder=0)
    ax.axhspan(20, 40, alpha=0.05, color='#F39C12', zorder=0)
    ax.axhspan(40, 120, alpha=0.05, color='#E74C3C', zorder=0)

    ymax = float((mean_cv + std_cv).max()) * 1.3
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    save_figure(fig, 'Figure7_operator_reproducibility')


# ===================================================================
# FIGURE 8 -- Representative Curve Fits  (7.2 x 7 in, 2x2 PNGs)
# ===================================================================
def figure8_representative_curves():
    """Composite existing per-strain QC plots into a 2x2 panel figure."""
    print("Figure 8: Representative Curve Fits")

    specs = [
        ('A', 'Good fit (Bifenthrin, BIF2)',
         RESULTS / "Group1_Results" / "plots"
         / "BifenthrinANDLB-BIF2_truncation_analysis.png"),
        ('B', 'Bad fit (Bifenthrin, BIF6)',
         RESULTS / "Group1_Results" / "plots"
         / "Bifenthrin-BIF6_BAD_fit_analysis.png"),
    ]

    # (C) external-operator plot
    g5 = RESULTS / "Group5_Results" / "plots"
    c_candidates = sorted(g5.glob("IMIDACLOPRIDANDLB-*_truncation_analysis.png")) if g5.exists() else []
    if c_candidates:
        name = c_candidates[0].stem.split('_')[0]
        specs.append(('C', f'External operator ({name})', c_candidates[0]))
    else:
        print("  WARNING: No Group5 IMIDACLOPRIDANDLB plot found for panel C.")

    # (D) LB control
    g3 = RESULTS / "Group3_Results" / "plots"
    d_candidates = sorted(g3.glob("LB-*_truncation_analysis.png")) if g3.exists() else []
    if d_candidates:
        name = d_candidates[0].stem.split('_')[0]
        specs.append(('D', f'LB Control ({name})', d_candidates[0]))
    else:
        print("  WARNING: No Group3 LB control plot found for panel D.")

    if not specs:
        print("  ERROR: No images found at all. Skipping Figure 8.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7))
    flat = axes.flatten()

    for idx, (lbl, subtitle, img_path) in enumerate(specs):
        ax = flat[idx]
        if img_path.exists():
            ax.imshow(mpimg.imread(str(img_path)))
        else:
            ax.text(0.5, 0.5, f"Not found:\n{img_path.name}",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=7, color='red')
        ax.axis('off')
        panel_label(ax, lbl, x=-0.02, y=1.02)
        ax.set_title(subtitle, fontsize=7, pad=4, fontstyle='italic')

    for idx in range(len(specs), 4):
        flat[idx].axis('off')

    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    save_figure(fig, 'Figure8_representative_curves')


# ===================================================================
# Dispatch + CLI
# ===================================================================
FIGURES = {
    1: figure1_pipeline,
    2: figure2_dataset_overview,
    3: figure3_synthetic_validation,
    4: figure4_truncation_comparison,
    5: figure5_haldane_vs_gompertz,
    6: figure6_ki_forest_plot,
    7: figure7_operator_reproducibility,
    8: figure8_representative_curves,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for the TECAN paper.")
    parser.add_argument('--fig', type=str, default=None,
                        help='Comma-separated figure numbers (e.g. 1,3,6)')
    args = parser.parse_args()

    apply_global_style()

    nums = ([int(f.strip()) for f in args.fig.split(',')]
            if args.fig else sorted(FIGURES.keys()))

    print(f"Output directory: {OUT}")
    print(f"Generating: {nums}\n")

    for n in nums:
        fn = FIGURES.get(n)
        if fn is None:
            print(f"WARNING: Figure {n} not defined. Skipping.")
            continue
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR on Figure {n}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. All figures in: {OUT}")


if __name__ == '__main__':
    main()
