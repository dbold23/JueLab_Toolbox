#!/usr/bin/env python3
"""
Interactive Truncation Method Validator

Displays each strain's raw data overlaid with all 6 Gompertz fits (one per
truncation method) and lets you pick which method looks best.

Keys:
    1-5   = Pick a specific method as best
    c     = Consensus is correct
    n     = None look good
    b     = Go back to previous strain
    q     = Quit & Save

Saves annotations to truncation_validation_audit.csv (resumable).

Usage:
    python 08_validate_truncation.py
    python 08_validate_truncation.py --comparison-csv path/to/method_comparison.csv

BIO380SP25 — Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Force interactive backend BEFORE any other matplotlib import (including
# from 06_advanced_fitting.py which sets Agg at module level).
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# ---- Dynamic imports from pipeline scripts ----
SCRIPTS_DIR = Path(__file__).parent

# Temporarily allow backend switch so the 06 import's matplotlib.use('Agg')
# doesn't crash — our 'macosx' is already locked in and will be kept.
_adv_spec = importlib.util.spec_from_file_location(
    "advanced_fitting", str(SCRIPTS_DIR / "06_advanced_fitting.py")
)
_adv = importlib.util.module_from_spec(_adv_spec)
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*cannot be set.*')
    _adv_spec.loader.exec_module(_adv)

load_config = _adv.load_config
load_raw_data = _adv.load_raw_data
gompertz_model = _adv.gompertz_model

# Method display config (same as 07_compare)
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

# Keys mapped to method indices (1-5 for individual methods, c for consensus)
KEY_TO_METHOD = {
    '1': 'first_peak',
    '2': 'stationary_phase',
    '3': 'adaptive_r2',
    '4': 'gp_derivative',
    '5': 'changepoint',
}


def load_existing_audit(audit_path):
    """Load existing audit CSV if it exists (for resume support)."""
    if audit_path.exists():
        df = pd.read_csv(audit_path)
        return set(df['strain'].tolist()), df
    return set(), pd.DataFrame()


class TruncationValidator:
    """Interactive matplotlib-based truncation method validator."""

    def __init__(self, comp_df, data_dir, audit_path):
        self.comp_df = comp_df
        self.data_dir = data_dir
        self.audit_path = audit_path
        self.current_note = ""
        self.annotations = []
        self.closed = False

        # Get unique strains (ordered as in comp_df)
        self.all_strains = comp_df['strain'].unique().tolist()

        # Load existing audit for resume
        audited_strains, existing_df = load_existing_audit(audit_path)
        if not existing_df.empty:
            self.annotations = existing_df.to_dict('records')

        # Filter out already-audited strains
        self.pending_strains = [s for s in self.all_strains if s not in audited_strains]

        if not self.pending_strains:
            print("\nAll strains already validated! Delete truncation_validation_audit.csv to start over.")
            self.show_summary()
            sys.exit(0)

        n_done = len(audited_strains)
        n_total = len(self.all_strains)
        n_remaining = len(self.pending_strains)

        print(f"\n{'='*60}")
        print(f"  TRUNCATION METHOD VALIDATOR")
        print(f"  {n_done} already reviewed, {n_remaining} remaining ({n_total} total)")
        print(f"{'='*60}")
        print(f"  1 = First Peak    2 = Stationary Phase  3 = Adaptive R²")
        print(f"  4 = GP Derivative 5 = Changepoint       c = Consensus OK")
        print(f"  n = None good     b = Back              q = Quit & Save")
        print(f"{'='*60}\n")

        self.pos = 0
        self.fig = None

    def save_audit(self):
        """Save all annotations to CSV."""
        if self.annotations:
            df = pd.DataFrame(self.annotations)
            df.to_csv(self.audit_path, index=False)
            print(f"\nSaved {len(self.annotations)} annotations to {self.audit_path}")

    def show_summary(self):
        """Print summary of validation results."""
        if not self.annotations:
            print("No annotations yet.")
            return

        df = pd.DataFrame(self.annotations)
        total = len(df)

        print(f"\n{'='*60}")
        print(f"  VALIDATION SUMMARY ({total} strains)")
        print(f"{'='*60}")

        # Method preference counts
        counts = df['user_best_method'].value_counts()
        for method, count in counts.items():
            pct = count / total * 100
            label = METHOD_LABELS.get(method, method)
            print(f"  {label:<25s}: {count:>3d}  ({pct:.0f}%)")

        # Agreement with auto-detected best
        if 'auto_best_method' in df.columns:
            agree = (df['user_best_method'] == df['auto_best_method']).sum()
            print(f"\n  Agrees with auto-best: {agree}/{total} ({100*agree/total:.0f}%)")

        print(f"{'='*60}\n")

    def annotate(self, user_method):
        """Record an annotation for the current strain."""
        if self.pos >= len(self.pending_strains):
            return

        strain = self.pending_strains[self.pos]
        sdf = self.comp_df[self.comp_df['strain'] == strain]
        group = sdf['group'].iloc[0]

        # Auto-detected best method (highest R²)
        success = sdf[sdf['fit_success'] == True]
        auto_best = success.loc[success['r_squared'].idxmax(), 'method'] if not success.empty else None
        auto_best_r2 = success['r_squared'].max() if not success.empty else None

        # Consensus R²
        cons = sdf[sdf['method'] == 'consensus']
        consensus_r2 = cons['r_squared'].values[0] if not cons.empty and cons['fit_success'].values[0] else None

        annotation = {
            'strain': strain,
            'group': group,
            'user_best_method': user_method,
            'auto_best_method': auto_best,
            'auto_best_r2': auto_best_r2,
            'consensus_r2': consensus_r2,
            'user_notes': self.current_note,
            'timestamp': datetime.now().isoformat(),
        }
        self.annotations.append(annotation)
        self.current_note = ""

        # Auto-save
        self.save_audit()

        # Advance
        self.pos += 1
        if self.pos >= len(self.pending_strains):
            print("\nAll strains validated!")
            self.show_summary()
            plt.close(self.fig)
            self.closed = True
        else:
            self.show_current()

    def on_key(self, event):
        """Handle keyboard input."""
        if self.closed:
            return

        # Don't capture keys when typing in the note box
        if hasattr(self, 'note_box') and self.note_box.ax == event.inaxes:
            return

        # Grab note
        if hasattr(self, 'note_box'):
            self.current_note = self.note_box.text

        if event.key in KEY_TO_METHOD:
            self.annotate(KEY_TO_METHOD[event.key])
        elif event.key == 'c':
            self.annotate('consensus')
        elif event.key == 'n':
            self.annotate('none')
        elif event.key == 'b':
            self.go_back()
        elif event.key == 'q':
            self.save_audit()
            self.show_summary()
            plt.close(self.fig)
            self.closed = True

    def go_back(self):
        """Go back to previous strain (undo last annotation)."""
        if self.annotations and self.pos > 0:
            self.annotations.pop()
            self.pos -= 1
            self.save_audit()
            self.show_current()

    def show_current(self):
        """Render the current strain's overlay plot live in the figure."""
        if self.pos >= len(self.pending_strains):
            return

        strain = self.pending_strains[self.pos]
        sdf = self.comp_df[self.comp_df['strain'] == strain]
        group = sdf['group'].iloc[0]

        # Load raw data
        t, od = load_raw_data(self.data_dir, strain, group)
        if t is not None:
            mask = np.isfinite(t) & np.isfinite(od)
            t, od = t[mask].astype(float), od[mask].astype(float)

        # Clear figure
        self.fig.clf()

        # Progress
        n_done = len(self.annotations)
        n_total = len(self.all_strains)
        n_remaining = len(self.pending_strains) - self.pos

        # Main plot area
        ax = self.fig.add_axes([0.06, 0.18, 0.88, 0.72])

        if t is not None:
            ax.scatter(t, od, s=12, c='#bbbbbb', alpha=0.6, zorder=1)

        t_fine = np.linspace(t.min(), t.max(), 500) if t is not None else np.array([])

        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color='#bbbbbb', marker='o', linestyle='None',
                           markersize=5, label='Raw OD')]

        success_data = sdf[sdf['fit_success'] == True]
        for _, row in success_data.iterrows():
            method = row['method']
            color = METHOD_COLORS.get(method, '#888888')
            label = METHOD_LABELS.get(method, method)
            A = row['gompertz_A']
            mu = row['gompertz_mu']
            lam = row['gompertz_lambda']
            trunc_time = row['trunc_time']
            r2 = row['r_squared']

            if pd.isna(A) or pd.isna(mu) or pd.isna(lam) or len(t_fine) == 0:
                continue

            y_fit = gompertz_model(t_fine, A, mu, lam)
            mask_fit = t_fine <= trunc_time
            mask_extrap = t_fine >= trunc_time

            lw = 2.5 if method == 'consensus' else 1.8
            ax.plot(t_fine[mask_fit], y_fit[mask_fit], color=color, linewidth=lw,
                    linestyle='-', zorder=3)
            ax.plot(t_fine[mask_extrap], y_fit[mask_extrap], color=color, linewidth=lw * 0.7,
                    linestyle=':', alpha=0.5, zorder=2)
            ax.axvline(x=trunc_time, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

            # Key number for the method
            key = {v: k for k, v in KEY_TO_METHOD.items()}.get(method, 'c')
            handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                   label=f"[{key}] {label} (R²={r2:.4f})"))

        ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
        ax.set_xlabel('Time (hours)', fontsize=11)
        ax.set_ylabel('OD', fontsize=11)
        ax.set_title(f'[{n_done+1}/{n_total}]  {strain}  ({group})\n'
                     f'Pick best truncation method — {n_remaining} remaining',
                     fontsize=11, fontweight='bold')

        # Buttons
        btn_specs = [
            (0.02, '1: Peak', '#2ca02c'),
            (0.13, '2: Stat', '#1f77b4'),
            (0.24, '3: AdR²', '#ff7f0e'),
            (0.35, '4: GP', '#d62728'),
            (0.46, '5: Chg', '#9467bd'),
            (0.57, 'c: Cons', '#D3D3D3'),
            (0.68, 'n: None', '#FFB6B6'),
            (0.80, 'b: Back', '#B0C4DE'),
        ]

        self._buttons = []
        for x, text, color in btn_specs:
            bax = self.fig.add_axes([x, 0.02, 0.10, 0.05])
            btn = Button(bax, text, color=color, hovercolor='#FFFACD')
            self._buttons.append(btn)

        # Wire button clicks
        self._buttons[0].on_clicked(lambda e: self.annotate('first_peak'))
        self._buttons[1].on_clicked(lambda e: self.annotate('stationary_phase'))
        self._buttons[2].on_clicked(lambda e: self.annotate('adaptive_r2'))
        self._buttons[3].on_clicked(lambda e: self.annotate('gp_derivative'))
        self._buttons[4].on_clicked(lambda e: self.annotate('changepoint'))
        self._buttons[5].on_clicked(lambda e: self.annotate('consensus'))
        self._buttons[6].on_clicked(lambda e: self.annotate('none'))
        self._buttons[7].on_clicked(lambda e: self.go_back())

        # Note textbox
        ax_note = self.fig.add_axes([0.1, 0.09, 0.75, 0.04])
        self.note_box = TextBox(ax_note, 'Note: ', initial='')
        self.note_box.on_submit(lambda text: setattr(self, 'current_note', text))

        self.fig.canvas.draw()

        # Terminal progress
        print(f"  [{n_done+1}/{n_total}] {strain} ({group}) — {n_remaining} remaining")

    def run(self):
        """Launch the interactive validator."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Truncation Method Validator')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.show_current()
        plt.show()

        # Final save
        self.save_audit()
        self.show_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive truncation method validator',
        epilog='Keys: 1-5=method, c=consensus, n=none, b=back, q=quit'
    )
    parser.add_argument('--comparison-csv', '-c', default=None,
                        help='Path to method_comparison.csv')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path for truncation_validation_audit.csv')
    args = parser.parse_args()

    # Resolve paths
    base = Path(__file__).parent.parent
    results_dir = base / 'results' / 'tables'
    data_dir = base / 'data' / 'raw'

    if args.comparison_csv:
        comp_csv = Path(args.comparison_csv)
    else:
        comp_csv = results_dir / 'Advanced_Analysis' / 'truncation_comparison' / 'method_comparison.csv'

    if args.output:
        audit_path = Path(args.output)
    else:
        audit_path = results_dir / 'Advanced_Analysis' / 'truncation_comparison' / 'truncation_validation_audit.csv'

    if not comp_csv.exists():
        print(f"ERROR: Comparison CSV not found at {comp_csv}")
        print("Run: python 07_compare_truncation_methods.py")
        sys.exit(1)

    comp_df = pd.read_csv(comp_csv)
    n_strains = comp_df['strain'].nunique()
    n_methods = comp_df['method'].nunique()
    print(f"Loaded {len(comp_df)} rows ({n_strains} strains × {n_methods} methods)")

    validator = TruncationValidator(comp_df, data_dir, audit_path)
    validator.run()


if __name__ == '__main__':
    main()
