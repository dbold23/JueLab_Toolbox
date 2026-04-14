#!/usr/bin/env python3
"""
Preprocess Group 5 (Walton/Nate) and Group 6 (Dominique) raw plate reader data.

These datasets have a different format from Group 1:
- Multiple dates (plates) per dataset, each with different well-strain mappings
- Time in seconds (Walton) or hours (Dominique)
- Wells mapped via a key file: Name, Date, Cell, Media, Strain

Output format matches existing pipeline:
- Separate CSV per media type (pesticide or LB)
- Columns: TIME[H], {MEDIA}_{STRAIN}_blanked
- Triplicates averaged, blank-subtracted, non-negative
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import sys


def normalize_strain(s):
    """Normalize strain names: strip whitespace, standardize blanks."""
    if pd.isna(s):
        return None
    s = s.strip()
    if not s:
        return None
    return s


def get_pesticide_from_strain(strain):
    """Extract pesticide prefix from strain name."""
    if strain is None:
        return None
    strain_lower = strain.lower()
    if strain_lower.startswith('imid'):
        return 'Imidacloprid'
    elif strain_lower.startswith('mal'):
        return 'Malathion'
    elif strain_lower.startswith('diaz'):
        return 'Diazinon'
    elif strain_lower.startswith('blank'):
        return 'Blank'
    return None


def get_strain_id(strain):
    """Extract the strain ID number from a strain name like 'Imid 6' -> 'IMID6'."""
    if strain is None:
        return None
    strain = strain.strip()
    # Handle blanks
    if strain.lower() == 'blank':
        return None
    # Handle "Blank Imid", "Blank Mal", etc.
    if strain.lower().startswith('blank'):
        return None
    # Handle "Mal Blank", "Diaz Blank"
    if strain.lower().endswith('blank'):
        return None
    # Extract prefix and number: "Imid 6" -> "IMID6", "Imid 1Y" -> "IMID1Y"
    match = re.match(r'(Imid|Mal|Diaz)\s*(.+)', strain, re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        num = match.group(2).strip()
        return f"{prefix}{num}"
    return strain.upper().replace(' ', '')


def is_blank(strain):
    """Check if a strain is a blank/control."""
    if strain is None:
        return True
    s = strain.strip().lower()
    return s == 'blank' or s.startswith('blank') or s.endswith('blank')


def get_blank_for_pesticide(key_date, pesticide_name, media):
    """Find appropriate blank wells for a pesticide in a given media on a given date."""
    blanks = []
    for _, row in key_date.iterrows():
        if row['Media'] != media:
            continue
        strain = normalize_strain(row['Strain'])
        if strain is None:
            continue
        s_lower = strain.lower()
        pest_lower = pesticide_name.lower()[:4]  # imid, mal, diaz

        # Match blanks: "Blank", "Blank Imid", "Imid Blank", etc.
        if s_lower == 'blank':
            blanks.append(row['Cell'])
        elif 'blank' in s_lower and pest_lower in s_lower:
            blanks.append(row['Cell'])
    return blanks


def preprocess_group(data_path, key_path, output_dir, group_name, verbose=True):
    """
    Preprocess a raw plate reader dataset into pipeline-ready *_DATA.csv files.

    Handles:
    - Multiple dates (plates) per dataset
    - Time in seconds or hours
    - Triplicate averaging and blank subtraction
    - Splitting by pesticide media type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and key
    data_df = pd.read_csv(data_path, encoding='utf-8-sig')
    key_df = pd.read_csv(key_path, encoding='utf-8-sig')

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Preprocessing {group_name}")
        print(f"{'='*60}")
        print(f"  Data shape: {data_df.shape}")
        print(f"  Key entries: {len(key_df)}")

    # Determine time column and convert to hours
    if 'Time [s]' in data_df.columns:
        time_col = 'Time [s]'
        time_scale = 3600.0
    elif 'timeh' in data_df.columns:
        time_col = 'timeh'
        time_scale = 1.0
    else:
        raise ValueError(f"No recognized time column in {data_path}")

    # Get well columns (everything that's a plate well like A1, B2, etc.)
    well_pattern = re.compile(r'^[A-H]\d{1,2}$')
    well_cols = [c for c in data_df.columns if well_pattern.match(c)]

    if verbose:
        print(f"  Time column: {time_col} (scale: {'seconds' if time_scale > 1 else 'hours'})")
        print(f"  Well columns: {len(well_cols)}")

    dates = sorted(key_df['Date'].unique())
    if verbose:
        print(f"  Dates (plates): {dates}")

    # Collect all curves across all dates, grouped by (media_type, pesticide, strain_id)
    # media_type = the output file type: IMIDACLOPRID, MALATHION, DIAZINON, LB
    # We accumulate across dates since strains may appear on multiple plates

    # Structure: {output_media: {strain_id: [(time_hours, blanked_values), ...]}}
    all_curves = defaultdict(lambda: defaultdict(list))
    total_wells = 0

    for date in dates:
        # Filter data and key for this date
        if 'Date' in data_df.columns:
            date_data = data_df[data_df['Date'] == date].copy()
        else:
            date_data = data_df.copy()

        date_key = key_df[key_df['Date'] == date].copy()

        if len(date_data) == 0:
            if verbose:
                print(f"\n  WARNING: No data rows for date {date}, skipping")
            continue

        # Convert time to hours
        time_hours = date_data[time_col].values / time_scale

        if verbose:
            print(f"\n  Date {date}: {len(date_data)} timepoints, "
                  f"time range: {time_hours[0]:.1f} - {time_hours[-1]:.1f} h")

        # Get unique media types for this date
        media_types = date_key['Media'].dropna().unique()

        for media in media_types:
            media_key = date_key[date_key['Media'] == media]

            # Identify strains and blanks in this media
            strains_in_media = {}
            blank_wells = []

            for _, row in media_key.iterrows():
                strain = normalize_strain(row['Strain'])
                if strain is None:
                    continue
                well = row['Cell']
                if well not in well_cols:
                    if verbose:
                        print(f"    WARNING: Well {well} not in data columns, skipping")
                    continue

                if is_blank(strain):
                    blank_wells.append(well)
                else:
                    strain_id = get_strain_id(strain)
                    pesticide = get_pesticide_from_strain(strain)
                    if strain_id and pesticide:
                        key_tuple = (pesticide, strain_id)
                        if key_tuple not in strains_in_media:
                            strains_in_media[key_tuple] = []
                        strains_in_media[key_tuple].append(well)

            # If no blanks found for this media, try finding generic blanks
            if not blank_wells:
                # Look for any blank on this date in this media
                for _, row in date_key[date_key['Media'] == media].iterrows():
                    strain = normalize_strain(row['Strain'])
                    if strain and is_blank(strain):
                        well = row['Cell']
                        if well in well_cols:
                            blank_wells.append(well)

            # Calculate blank average
            if blank_wells:
                blank_data = date_data[blank_wells].values.astype(float)
                blank_avg = np.nanmean(blank_data, axis=1)
            else:
                blank_avg = np.zeros(len(time_hours))
                if verbose:
                    print(f"    WARNING: No blanks for media={media} date={date}, using zeros")

            if verbose:
                print(f"    Media {media}: {len(blank_wells)} blank wells, "
                      f"{len(strains_in_media)} strain groups")

            # Process each strain
            for (pesticide, strain_id), wells in strains_in_media.items():
                # Average replicates
                strain_data = date_data[wells].values.astype(float)
                strain_avg = np.nanmean(strain_data, axis=1)

                # Subtract blank
                blanked = strain_avg - blank_avg
                blanked = np.maximum(blanked, 0)

                # Determine output media name
                if media == 'LB':
                    # Strains in LB = pesticide+LB condition
                    output_media = f"{pesticide.upper()}ANDLB"
                elif media == 'P':
                    output_media = pesticide.upper()
                elif media == 'G':
                    # G media - treat as a separate condition
                    output_media = f"{pesticide.upper()}ANDG"
                else:
                    output_media = f"{pesticide.upper()}_{media}"

                col_name = f"{output_media}_{strain_id}_blanked"
                all_curves[output_media][strain_id].append(
                    (time_hours, blanked, col_name)
                )
                total_wells += len(wells)

            # Also save LB-only blanks as LB_DATA if this is LB media
            if media == 'LB' and blank_wells:
                blank_data_vals = date_data[blank_wells].values.astype(float)
                blank_avg_curve = np.nanmean(blank_data_vals, axis=1)
                # LB blanks don't get blank-subtracted (they ARE the blank)
                # But for consistency, we note them
                all_curves['LB']['Blank'].append(
                    (time_hours, blank_avg_curve, 'LB_Blank_blanked')
                )

    if verbose:
        print(f"\n  Total wells processed: {total_wells}")

    # Now write output files — one per output_media type
    # For strains that appear on multiple dates, we take the average across dates
    # if timepoints match, otherwise keep the longest run
    files_written = 0
    total_curves = 0

    for output_media, strains_dict in sorted(all_curves.items()):
        if output_media == 'LB' and len(strains_dict) == 1 and 'Blank' in strains_dict:
            # Skip LB file that only has blank (no actual strains)
            continue

        output_data = {}

        for strain_id, curve_list in sorted(strains_dict.items()):
            if strain_id == 'Blank':
                continue

            # If multiple dates have this strain, pick the one with best data
            # (longest time series or highest max OD)
            best_curve = max(curve_list, key=lambda x: (len(x[0]), np.max(x[1])))
            time_hours, blanked, col_name = best_curve

            # Set time if not yet set
            if 'TIME[H]' not in output_data:
                output_data['TIME[H]'] = time_hours
            elif len(time_hours) != len(output_data['TIME[H]']):
                # Different time grids — interpolate to match
                from scipy.interpolate import interp1d
                f = interp1d(time_hours, blanked, bounds_error=False, fill_value=0)
                blanked = f(output_data['TIME[H]'])

            output_data[col_name] = blanked
            total_curves += 1

        if len(output_data) <= 1:
            continue

        output_df = pd.DataFrame(output_data)
        output_path = output_dir / f"{output_media}_DATA.csv"
        output_df.to_csv(output_path, index=False)
        files_written += 1

        if verbose:
            strain_cols = [c for c in output_df.columns if c != 'TIME[H]']
            print(f"  Saved: {output_path.name} ({len(strain_cols)} strains)")

    # Also create LB_DATA.csv with strains grown in LB
    # These are in PESTICIDE_ANDLB files already, but some pipelines expect an LB_DATA.csv
    # with control strains. We'll skip if no pure LB data exists.

    if verbose:
        print(f"\n  Files written: {files_written}")
        print(f"  Total curves: {total_curves}")
        print(f"  Output dir: {output_dir}")

    return total_curves


def main():
    base = Path(__file__).resolve().parent.parent
    raw_dir = base / 'data' / 'raw'

    total_curves = 0

    # Group 5 (Walton/Nate)
    g5_data = raw_dir / 'Group5' / 'Group5_DATA.csv'
    g5_key = raw_dir / 'Group5' / 'Group5_KEY.csv'
    g5_out = raw_dir / 'Group5' / 'Group5_DATA_processed'

    if g5_data.exists() and g5_key.exists():
        n = preprocess_group(g5_data, g5_key, g5_out, 'Group5 (Walton/Nate)')
        total_curves += n
    else:
        print(f"WARNING: Group5 data not found at {g5_data}")

    # Group 6 (Dominique)
    g6_data = raw_dir / 'Group6' / 'Group6_DATA.csv'
    g6_key = raw_dir / 'Group6' / 'Group6_KEY.csv'
    g6_out = raw_dir / 'Group6' / 'Group6_DATA_processed'

    if g6_data.exists() and g6_key.exists():
        n = preprocess_group(g6_data, g6_key, g6_out, 'Group6 (Dominique)')
        total_curves += n
    else:
        print(f"WARNING: Group6 data not found at {g6_data}")

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Total curves across Groups 5-6: {total_curves}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
