#!/usr/bin/env python3
"""
Interactive Growth Curve Validation Annotator

Displays each curve's diagnostic plot and lets you quickly mark it as:
  y = Correct (pipeline classification is right)
  n = Wrong (pipeline classification is wrong)
  u = Unsure (needs further review)
  space = Skip (come back later)

Saves annotations to validation_audit.csv (resumable).

Usage:
    python validate_results_interactive.py [--results-dir RESULTS_DIR]

Keyboard shortcuts:
    y / Enter  = Correct
    n          = Wrong
    u          = Unsure
    space      = Skip
    q          = Quit and save
    b          = Go back to previous curve

BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.image import imread


def find_plot_for_strain(strain, is_good, group, results_dir):
    """Find the diagnostic plot PNG for a given strain."""
    group_dir = results_dir / f"{group}_Results" / "plots"

    if is_good:
        plot_name = f"{strain}_truncation_analysis.png"
    else:
        plot_name = f"{strain}_BAD_fit_analysis.png"

    plot_path = group_dir / plot_name
    if plot_path.exists():
        return plot_path

    # Fallback: try the other type (in case classification changed)
    alt_name = f"{strain}_BAD_fit_analysis.png" if is_good else f"{strain}_truncation_analysis.png"
    alt_path = group_dir / alt_name
    if alt_path.exists():
        return alt_path

    # Fallback: search for any file containing the strain name
    if group_dir.exists():
        for f in group_dir.iterdir():
            if strain in f.stem:
                return f

    return None


def load_existing_audit(audit_path):
    """Load existing audit CSV if it exists (for resume support)."""
    if audit_path.exists():
        df = pd.read_csv(audit_path)
        return set(df['strain'].tolist()), df
    return set(), pd.DataFrame()


class CurveAnnotator:
    """Interactive matplotlib-based curve annotation tool."""

    def __init__(self, results_df, results_dir, audit_path):
        self.results_df = results_df
        self.results_dir = results_dir
        self.audit_path = audit_path
        self.current_idx = 0
        self.annotations = []
        self.current_note = ""

        # Load existing audit for resume
        audited_strains, existing_df = load_existing_audit(audit_path)
        if not existing_df.empty:
            self.annotations = existing_df.to_dict('records')

        # Filter out already-audited strains
        self.pending_indices = []
        for i, row in self.results_df.iterrows():
            if row['strain'] not in audited_strains:
                self.pending_indices.append(i)

        if not self.pending_indices:
            print("\nAll curves already audited! Delete validation_audit.csv to start over.")
            self.show_summary()
            sys.exit(0)

        n_done = len(audited_strains)
        n_total = len(self.results_df)
        n_remaining = len(self.pending_indices)
        print(f"\n{'='*60}")
        print(f"  CURVE VALIDATION ANNOTATOR")
        print(f"  {n_done} already audited, {n_remaining} remaining ({n_total} total)")
        print(f"{'='*60}")
        print(f"  y/Enter = Correct | n = Wrong | u = Unsure | space = Skip")
        print(f"  b = Back | q = Quit & Save")
        print(f"{'='*60}\n")

        self.pos_in_pending = 0
        self.fig = None
        self.closed = False

    def save_audit(self):
        """Save all annotations to CSV."""
        if self.annotations:
            df = pd.DataFrame(self.annotations)
            df.to_csv(self.audit_path, index=False)
            print(f"\nSaved {len(self.annotations)} annotations to {self.audit_path}")

    def show_summary(self):
        """Print summary of audit results."""
        if not self.annotations:
            print("No annotations yet.")
            return

        df = pd.DataFrame(self.annotations)
        total = len(df)
        correct = len(df[df['audit_result'] == 'correct'])
        wrong = len(df[df['audit_result'] == 'wrong'])
        unsure = len(df[df['audit_result'] == 'unsure'])
        skipped = len(df[df['audit_result'] == 'skip'])

        print(f"\n{'='*60}")
        print(f"  AUDIT SUMMARY")
        print(f"{'='*60}")
        print(f"  Total audited:  {total}")
        print(f"  Correct:        {correct}  ({100*correct/max(total,1):.1f}%)")
        print(f"  Wrong:          {wrong}  ({100*wrong/max(total,1):.1f}%)")
        print(f"  Unsure:         {unsure}  ({100*unsure/max(total,1):.1f}%)")
        print(f"  Skipped:        {skipped}")
        print(f"{'='*60}")

        if wrong > 0:
            print(f"\n  WRONG classifications:")
            wrong_df = df[df['audit_result'] == 'wrong']
            for _, row in wrong_df.iterrows():
                note = f" -- {row['notes']}" if row.get('notes') else ""
                print(f"    {row['strain']} (pipeline said {'GOOD' if row['is_good'] else 'BAD'}){note}")

        if unsure > 0:
            print(f"\n  UNSURE classifications:")
            unsure_df = df[df['audit_result'] == 'unsure']
            for _, row in unsure_df.iterrows():
                note = f" -- {row['notes']}" if row.get('notes') else ""
                print(f"    {row['strain']} (pipeline said {'GOOD' if row['is_good'] else 'BAD'}){note}")

        print()

    def annotate(self, result):
        """Record an annotation for the current curve."""
        if self.pos_in_pending >= len(self.pending_indices):
            return

        idx = self.pending_indices[self.pos_in_pending]
        row = self.results_df.iloc[idx]

        annotation = {
            'strain': row['strain'],
            'group': row['group'],
            'is_good': row['is_good'],
            'classification_reason': row.get('classification_reason', ''),
            'audit_result': result,
            'notes': self.current_note,
            'timestamp': datetime.now().isoformat(),
            'r_squared': row.get('fit_r_squared', row.get('r_squared', '')),
            'gompertz_a': row.get('gompertz_a', ''),
            'gompertz_mu': row.get('gompertz_mu', ''),
            'gompertz_lambda': row.get('gompertz_lambda', ''),
        }
        self.annotations.append(annotation)
        self.current_note = ""

        # Auto-save after each annotation
        self.save_audit()

        # Advance to next
        self.pos_in_pending += 1
        if self.pos_in_pending >= len(self.pending_indices):
            print("\nAll curves audited!")
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

        if event.key == 'y' or event.key == 'enter':
            # Grab note text before annotating
            if hasattr(self, 'note_box'):
                self.current_note = self.note_box.text
            self.annotate('correct')
        elif event.key == 'n':
            if hasattr(self, 'note_box'):
                self.current_note = self.note_box.text
            self.annotate('wrong')
        elif event.key == 'u':
            if hasattr(self, 'note_box'):
                self.current_note = self.note_box.text
            self.annotate('unsure')
        elif event.key == ' ':
            if hasattr(self, 'note_box'):
                self.current_note = self.note_box.text
            self.annotate('skip')
        elif event.key == 'q':
            self.save_audit()
            self.show_summary()
            plt.close(self.fig)
            self.closed = True
        elif event.key == 'b':
            self.go_back()

    def go_back(self):
        """Go back to previous curve (undo last annotation)."""
        if self.annotations and self.pos_in_pending > 0:
            self.annotations.pop()
            self.pos_in_pending -= 1
            self.save_audit()
            self.show_current()

    def on_correct(self, event):
        if hasattr(self, 'note_box'):
            self.current_note = self.note_box.text
        self.annotate('correct')

    def on_wrong(self, event):
        if hasattr(self, 'note_box'):
            self.current_note = self.note_box.text
        self.annotate('wrong')

    def on_unsure(self, event):
        if hasattr(self, 'note_box'):
            self.current_note = self.note_box.text
        self.annotate('unsure')

    def on_skip(self, event):
        if hasattr(self, 'note_box'):
            self.current_note = self.note_box.text
        self.annotate('skip')

    def on_note_submit(self, text):
        self.current_note = text

    def show_current(self):
        """Display the current curve's diagnostic plot."""
        if self.pos_in_pending >= len(self.pending_indices):
            return

        idx = self.pending_indices[self.pos_in_pending]
        row = self.results_df.iloc[idx]
        strain = row['strain']
        is_good = row['is_good']
        group = row['group']

        # Find the plot
        plot_path = find_plot_for_strain(strain, is_good, group, self.results_dir)

        # Clear the figure
        self.fig.clf()

        # Progress info
        n_done = len(self.annotations)
        n_total = len(self.results_df)
        n_remaining = len(self.pending_indices) - self.pos_in_pending

        # Title with key info
        classification = "GOOD" if is_good else "BAD"
        reason = row.get('classification_reason', 'N/A')
        r2 = row.get('fit_r_squared', row.get('r_squared', 'N/A'))
        if isinstance(r2, float):
            r2 = f"{r2:.4f}"

        title = (f"[{n_done+1}/{n_total}] {strain} | Pipeline: {classification} | "
                f"R²={r2} | {group}\n"
                f"Reason: {reason}")

        if plot_path and plot_path.exists():
            # Load and display the plot image
            img = imread(str(plot_path))
            ax = self.fig.add_axes([0.02, 0.15, 0.96, 0.75])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        else:
            ax = self.fig.add_axes([0.02, 0.15, 0.96, 0.75])
            ax.text(0.5, 0.5, f"Plot not found for:\n{strain}\n\nExpected at:\n{plot_path}",
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

        # Buttons at the bottom
        ax_back = self.fig.add_axes([0.02, 0.02, 0.1, 0.06])
        ax_correct = self.fig.add_axes([0.15, 0.02, 0.15, 0.06])
        ax_wrong = self.fig.add_axes([0.33, 0.02, 0.15, 0.06])
        ax_unsure = self.fig.add_axes([0.51, 0.02, 0.15, 0.06])
        ax_skip = self.fig.add_axes([0.69, 0.02, 0.15, 0.06])

        self.btn_back = Button(ax_back, '<< Back (b)', color='#B0C4DE', hovercolor='#6495ED')
        self.btn_correct = Button(ax_correct, 'Correct (y)', color='#90EE90', hovercolor='#50C878')
        self.btn_wrong = Button(ax_wrong, 'Wrong (n)', color='#FFB6B6', hovercolor='#FF6B6B')
        self.btn_unsure = Button(ax_unsure, 'Unsure (u)', color='#FFFACD', hovercolor='#FFD700')
        self.btn_skip = Button(ax_skip, 'Skip (space)', color='#D3D3D3', hovercolor='#A9A9A9')

        self.btn_back.on_clicked(lambda e: self.go_back())
        self.btn_correct.on_clicked(self.on_correct)
        self.btn_wrong.on_clicked(self.on_wrong)
        self.btn_unsure.on_clicked(self.on_unsure)
        self.btn_skip.on_clicked(self.on_skip)

        # Note textbox
        ax_note = self.fig.add_axes([0.1, 0.09, 0.75, 0.04])
        self.note_box = TextBox(ax_note, 'Note: ', initial='')
        self.note_box.on_submit(self.on_note_submit)

        self.fig.canvas.draw()

        # Print to terminal too
        print(f"  [{n_done+1}/{n_total}] {strain} -- {classification} (R²={r2}) -- {n_remaining} remaining")

    def run(self):
        """Launch the interactive annotator."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Growth Curve Validation Annotator')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.show_current()
        plt.show()

        # Final save on close
        self.save_audit()
        self.show_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive growth curve validation annotator',
        epilog='Keyboard: y=correct, n=wrong, u=unsure, space=skip, b=back, q=quit'
    )
    parser.add_argument(
        '--results-dir', '-r',
        default=None,
        help='Path to results/tables directory (default: auto-detect)'
    )
    parser.add_argument(
        '--results-csv', '-c',
        default=None,
        help='Path to all_groups_results.csv (default: auto-detect)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output path for validation_audit.csv (default: results/tables/validation_audit.csv)'
    )

    args = parser.parse_args()

    # Auto-detect paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "results" / "tables"

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = base_dir

    if args.results_csv:
        results_csv = Path(args.results_csv)
    else:
        results_csv = results_dir / "all_groups_results.csv"

    if args.output:
        audit_path = Path(args.output)
    else:
        audit_path = results_dir / "validation_audit.csv"

    # Validate inputs
    if not results_csv.exists():
        print(f"ERROR: Results CSV not found at {results_csv}")
        sys.exit(1)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found at {results_dir}")
        sys.exit(1)

    # Load results
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} curves from {results_csv}")
    print(f"  GOOD: {len(df[df['is_good'] == True])}")
    print(f"  BAD:  {len(df[df['is_good'] == False])}")

    # Check that plots exist
    missing = 0
    for _, row in df.iterrows():
        plot = find_plot_for_strain(row['strain'], row['is_good'], row['group'], results_dir)
        if not plot:
            missing += 1
            print(f"  WARNING: No plot found for {row['strain']} ({row['group']})")

    if missing > 0:
        print(f"\n  {missing} curves have missing plots. They will show a placeholder.")

    # Launch annotator
    annotator = CurveAnnotator(df, results_dir, audit_path)
    annotator.run()


if __name__ == '__main__':
    main()
