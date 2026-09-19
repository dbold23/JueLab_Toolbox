#!/usr/bin/env python3
"""
Turn a raw TECAN plate-reader export into the analysis-ready ``{MEDIA}_DATA.csv``
files that ``01_growth_curve_analysis.py`` reads.

Input
    * a values table  — SparkControl/i-control export (``.xlsx``) or the same
      kinetic table saved as CSV (``Cycle Nr., Time [s], Temp. [°C], A1 … H12``)
    * a key file      — ``Cell, Media, Strain`` (CSV or a sheet of the export)

Output (one file per media, matching the committed Group_*_DATA layout)
    ``TIME[H]`` plus one ``{MEDIA}_{STRAIN}_blanked`` column per condition:
    the media's blank wells averaged and subtracted at every time point, then
    the replicate wells averaged.

This script is a thin command-line wrapper: every numerical step lives in
:mod:`tecan_growth.io` and :mod:`tecan_growth.preprocess`, which the lab
interface and the test-suite use as well, so the CLI and the app cannot drift
apart. Going through the package also fixes what the old stand-alone
implementation got wrong on real plates:

  * blank labels match case-insensitively (``Blank`` and ``BLANK``),
  * key rows with no media/strain are ignored instead of writing ``nan_DATA.csv``,
  * the time column is found by header (``Time [s]``, ``Time [min]``, ``Time [h]``),
  * a key well with no data column is a readable error, not a bare ``KeyError``,
  * missing/non-numeric cells are ignored in the averages instead of poisoning
    the whole curve with ``NaN``.

Usage:
    python 02_preprocess_raw_plate_data.py VALUES.csv KEY.csv -o OUTPUT_DIR
"""

import os
import sys
from pathlib import Path
from typing import List

# Allow running the script directly from anywhere (scripts/ is not a package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tecan_growth.io import load_plate           # noqa: E402
from tecan_growth.preprocess import preprocess_plate  # noqa: E402


class PreprocessError(RuntimeError):
    """Raised when the values/key pair cannot be preprocessed."""


def preprocess_group1(
    data_path: str,
    key_path: str,
    output_dir: str,
    verbose: bool = True,
) -> List[Path]:
    """
    Convert a raw plate-reader export to the analysis-ready ``{MEDIA}_DATA.csv``
    files.

    The name is historical (this started as a Group 1 helper); it works for any
    group's export.

    Parameters
    ----------
    data_path : str
        Raw values file — ``.xlsx`` export or values CSV.
    key_path : str
        Key file mapping wells to media/strain (``Cell, Media, Strain``).
    output_dir : str
        Directory to write ``{MEDIA}_DATA.csv`` into (created if needed).
    verbose : bool
        Print progress and QC messages.

    Returns
    -------
    list of Path
        The files written, one per media.

    Raises
    ------
    PreprocessError
        If a file is missing or the key does not match the data.
    """
    for label, p in (("data", data_path), ("key", key_path)):
        if not Path(p).exists():
            raise PreprocessError(f"{label} file not found: {p}")

    if verbose:
        print(f"Loading data file: {data_path}")
        print(f"Loading key file:  {key_path}")

    try:
        run = load_plate(data_path, key_path)
    except (ValueError, KeyError) as exc:
        raise PreprocessError(f"Could not read {data_path} / {key_path}: {exc}") from exc

    for w in run.warnings:
        if verbose:
            print(f"  NOTE: {w}")

    try:
        processed = preprocess_plate(run)
    except (ValueError, KeyError) as exc:
        raise PreprocessError(str(exc)) from exc

    if verbose:
        t = processed.time_h
        print(f"Time range: {t.min():.2f} to {t.max():.2f} hours")
        print(f"Number of timepoints: {len(t)}")
        media_names = list(dict.fromkeys(processed.conditions.media))
        print(f"\nMedia types found: {media_names}")
        print(f"Strains found: {sorted(set(processed.conditions.strain))}")

        for media in media_names:
            print(f"\n{'=' * 50}")
            print(f"Processing media: {media}")
            b = processed.blanks[processed.blanks.media == media]
            if len(b) and b.blank_source.iloc[0] != "none":
                print(f"  Blank wells: {int(b.n_blank_wells.iloc[0])} ({b.blank_source.iloc[0]})")
                print(f"  Blank average OD range: {b.blank_mean.min():.4f} to {b.blank_mean.max():.4f}")
            else:
                print(f"  WARNING: No blank wells found for {media}, using zeros")
            for _, row in processed.conditions[processed.conditions.media == media].iterrows():
                _, od = processed.curve(row.curve_id)
                print(f"  {row.strain}: wells {row.wells}, "
                      f"blanked OD range: {od.min():.4f} to {od.max():.4f}")

        for msg in processed.qc:
            print(f"  QC: {msg}")

    paths = processed.write_legacy_csvs(output_dir)

    if verbose:
        print(f"\n{'=' * 50}")
        for p in paths:
            print(f"  Saved: {p}")
        print("Preprocessing complete!")
        print(f"Output directory: {os.fspath(output_dir)}")

    return paths


#: Backwards-compatible alias for the group-agnostic name.
preprocess_plate_export = preprocess_group1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess raw plate reader data for growth curve analysis'
    )
    parser.add_argument(
        'data_file',
        help='Path to raw data (.xlsx export or values CSV, e.g. GrowthRate_ Group1_Values.csv)'
    )
    parser.add_argument(
        'key_file',
        help='Path to key CSV mapping wells to media/strain'
    )
    parser.add_argument(
        '-o', '--output',
        default='./Group1_Processed',
        help='Output directory for processed files (default: ./Group1_Processed)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    try:
        preprocess_group1(
            data_path=args.data_file,
            key_path=args.key_file,
            output_dir=args.output,
            verbose=not args.quiet,
        )
    except PreprocessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
