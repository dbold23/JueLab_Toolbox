#!/usr/bin/env python3
"""
Preprocess Group 1 data from raw plate reader format to analysis-ready format.

Group 1 data format:
- Raw file: 96-well plate format with columns A1-H12, Time in seconds
- Key file: Maps wells to Media and Strain

Output format (matches Groups 2-4):
- Separate CSV per media type
- Columns: TIME[H], {MEDIA}_{STRAIN}_blanked (averaged triplicates, blank-subtracted)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import os


def load_key_file(key_path: str) -> pd.DataFrame:
    """Load the well mapping key file."""
    key_df = pd.read_csv(key_path)
    return key_df


def load_data_file(data_path: str) -> pd.DataFrame:
    """Load the raw plate reader data."""
    data_df = pd.read_csv(data_path)
    return data_df


def get_wells_for_condition(key_df: pd.DataFrame, media: str, strain: str) -> list:
    """Get all wells matching a media+strain combination."""
    mask = (key_df['Media'] == media) & (key_df['Strain'] == strain)
    return key_df[mask]['Cell'].tolist()


def preprocess_group1(
    data_path: str,
    key_path: str,
    output_dir: str,
    verbose: bool = True
):
    """
    Convert Group 1 raw plate reader data to analysis-ready format.

    Steps:
    1. Load key file and data file
    2. Convert time from seconds to hours
    3. Group wells by Media type
    4. For each Media type:
       a. Find blank wells (Strain == 'Blank')
       b. Calculate blank average for each timepoint
       c. For each non-blank strain:
          - Find triplicate wells
          - Average the triplicates
          - Subtract the blank average
       d. Save as {MEDIA}_DATA.csv

    Parameters
    ----------
    data_path : str
        Path to raw data CSV (e.g., GrowthRate_ Group1_Values.csv)
    key_path : str
        Path to key CSV (e.g., GROUP1_KEY_V2.csv)
    output_dir : str
        Directory to save processed files
    verbose : bool
        Print progress messages
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load files
    if verbose:
        print(f"Loading key file: {key_path}")
    key_df = load_key_file(key_path)

    if verbose:
        print(f"Loading data file: {data_path}")
    data_df = load_data_file(data_path)

    # Convert time from seconds to hours
    time_hours = data_df['Time [s]'] / 3600.0

    if verbose:
        print(f"Time range: {time_hours.min():.2f} to {time_hours.max():.2f} hours")
        print(f"Number of timepoints: {len(time_hours)}")

    # Get unique media types and strains
    media_types = key_df['Media'].unique()
    all_strains = key_df['Strain'].unique()
    non_blank_strains = [s for s in all_strains if s != 'Blank']

    if verbose:
        print(f"\nMedia types found: {list(media_types)}")
        print(f"Strains found: {list(non_blank_strains)}")

    # Process each media type
    for media in media_types:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing media: {media}")

        # Get strains for this media
        media_key = key_df[key_df['Media'] == media]
        strains_in_media = media_key['Strain'].unique()

        # Start building output dataframe
        output_data = {'TIME[H]': time_hours.values}

        # Get blank wells and calculate blank average
        blank_wells = get_wells_for_condition(key_df, media, 'Blank')

        if len(blank_wells) > 0:
            blank_data = data_df[blank_wells].values
            blank_avg = np.mean(blank_data, axis=1)
            blank_std = np.std(blank_data, axis=1)
            if verbose:
                print(f"  Blank wells: {blank_wells}")
                print(f"  Blank average OD range: {blank_avg.min():.4f} to {blank_avg.max():.4f}")
            # QC: flag high blank variability
            max_blank_std = np.max(blank_std) if len(blank_std) > 0 else 0
            if max_blank_std > 0.01:
                print(f"  WARNING: High blank variability for {media} "
                      f"(max sigma={max_blank_std:.4f} > 0.01 OD)")
        else:
            blank_avg = np.zeros(len(time_hours))
            if verbose:
                print(f"  WARNING: No blank wells found for {media}, using zeros")

        # Process each non-blank strain
        for strain in strains_in_media:
            if strain == 'Blank':
                continue

            wells = get_wells_for_condition(key_df, media, strain)

            if len(wells) == 0:
                if verbose:
                    print(f"  No wells found for {strain}, skipping")
                continue

            # Get data for these wells
            strain_data = data_df[wells].values

            # Outlier replicate detection (flag if any well >3σ from median)
            if strain_data.shape[1] >= 3:
                median_vals = np.median(strain_data, axis=1, keepdims=True)
                deviations = np.abs(strain_data - median_vals)
                mad = np.median(deviations, axis=1, keepdims=True)
                mad = np.maximum(mad, 1e-6)  # avoid division by zero
                outlier_mask = deviations > 3 * 1.4826 * mad  # MAD-based outlier
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    print(f"  WARNING: {strain} has {n_outliers} outlier replicate "
                          f"timepoints (>3 MAD from median)")

            # Average triplicates (use median for robustness against outliers)
            strain_avg = np.median(strain_data, axis=1) if strain_data.shape[1] >= 3 else np.mean(strain_data, axis=1)

            # Subtract blank
            strain_blanked = strain_avg - blank_avg

            # Ensure non-negative (can happen with noise)
            strain_blanked = np.maximum(strain_blanked, 0)

            # Column name matches expected format
            col_name = f"{media}_{strain}_blanked"
            output_data[col_name] = strain_blanked

            if verbose:
                print(f"  {strain}: wells {wells}, blanked OD range: {strain_blanked.min():.4f} to {strain_blanked.max():.4f}")

        # Create output dataframe
        output_df = pd.DataFrame(output_data)

        # Save to file
        output_path = os.path.join(output_dir, f"{media}_DATA.csv")
        output_df.to_csv(output_path, index=False)

        if verbose:
            print(f"  Saved: {output_path}")
            print(f"  Columns: {list(output_df.columns)}")

    if verbose:
        print(f"\n{'='*50}")
        print(f"Preprocessing complete!")
        print(f"Output directory: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess Group 1 plate reader data for growth curve analysis'
    )
    parser.add_argument(
        'data_file',
        help='Path to raw data CSV (e.g., GrowthRate_ Group1_Values.csv)'
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

    preprocess_group1(
        data_path=args.data_file,
        key_path=args.key_file,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
