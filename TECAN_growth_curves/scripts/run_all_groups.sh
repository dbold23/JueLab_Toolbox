#!/bin/bash
#
# Run Growth Curve Analysis Pipeline for All Groups
# BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
#
# Usage: ./run_all_groups.sh
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter (adjust if needed)
PYTHON="python3"

echo "=============================================="
echo "Growth Curve Analysis Pipeline"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo ""

# Create output directory
mkdir -p "$SCRIPT_DIR/OUTPUT"

# Step 1: Preprocess Group 1 (raw plate reader format)
echo "----------------------------------------------"
echo "STEP 1: Preprocessing Group 1 data..."
echo "----------------------------------------------"

if [ -f "$SCRIPT_DIR/DATA/Group1/GrowthRate_ Group1_Values.csv" ]; then
    $PYTHON "$SCRIPT_DIR/02_preprocess_raw_plate_data.py" \
        "$SCRIPT_DIR/DATA/Group1/GrowthRate_ Group1_Values.csv" \
        "$SCRIPT_DIR/DATA/Group1/GROUP1_KEY_V2.csv" \
        -o "$SCRIPT_DIR/DATA/Group1/Group_1_DATA"
    echo "Group 1 preprocessing complete."
else
    echo "Group 1 raw data not found, skipping preprocessing."
fi

echo ""

# Step 2: Process all groups with fit-based classification + adaptive truncation
echo "----------------------------------------------"
echo "STEP 2: Running Gompertz analysis..."
echo "----------------------------------------------"

# Group 1
if [ -d "$SCRIPT_DIR/DATA/Group1/Group_1_DATA" ]; then
    echo ""
    echo "Processing Group 1..."
    $PYTHON "$SCRIPT_DIR/01_growth_curve_analysis.py" \
        "$SCRIPT_DIR/DATA/Group1/Group_1_DATA" \
        -o "$SCRIPT_DIR/OUTPUT/Group1_Final" \
        --adaptive
fi

# Group 2
if [ -d "$SCRIPT_DIR/DATA/Group 2/Group_2_DATA" ]; then
    echo ""
    echo "Processing Group 2..."
    $PYTHON "$SCRIPT_DIR/01_growth_curve_analysis.py" \
        "$SCRIPT_DIR/DATA/Group 2/Group_2_DATA" \
        -o "$SCRIPT_DIR/OUTPUT/Group2_Final" \
        --adaptive
fi

# Group 3
if [ -d "$SCRIPT_DIR/DATA/Group 3/Group_3_DATA" ]; then
    echo ""
    echo "Processing Group 3..."
    $PYTHON "$SCRIPT_DIR/01_growth_curve_analysis.py" \
        "$SCRIPT_DIR/DATA/Group 3/Group_3_DATA" \
        -o "$SCRIPT_DIR/OUTPUT/Group3_Final" \
        --adaptive
fi

# Group 4
if [ -d "$SCRIPT_DIR/DATA/Group 4/Group4_DATA" ]; then
    echo ""
    echo "Processing Group 4..."
    $PYTHON "$SCRIPT_DIR/01_growth_curve_analysis.py" \
        "$SCRIPT_DIR/DATA/Group 4/Group4_DATA" \
        -o "$SCRIPT_DIR/OUTPUT/Group4_Final" \
        --adaptive
fi

echo ""

# Step 3: Combine all results
echo "----------------------------------------------"
echo "STEP 3: Combining results..."
echo "----------------------------------------------"

$PYTHON << EOF
import pandas as pd
import os

output_dir = "$SCRIPT_DIR/OUTPUT"

groups = {
    'Group1': f'{output_dir}/Group1_Final/processing_results.csv',
    'Group2': f'{output_dir}/Group2_Final/processing_results.csv',
    'Group3': f'{output_dir}/Group3_Final/processing_results.csv',
    'Group4': f'{output_dir}/Group4_Final/processing_results.csv',
}

all_results = []
for group, path in groups.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['group'] = group
        all_results.append(df)
        good = df['is_good'].sum()
        bad = len(df) - good
        print(f"  {group}: {len(df)} curves ({good} good, {bad} bad)")

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f'{output_dir}/all_groups_results.csv', index=False)

    print(f"\nCombined: {len(combined)} total curves")
    print(f"  Good: {combined['is_good'].sum()}")
    print(f"  Bad: {(~combined['is_good']).sum()}")
    print(f"\nSaved to: {output_dir}/all_groups_results.csv")
else:
    print("No results found to combine.")
EOF

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo ""
echo "Results saved to: $SCRIPT_DIR/OUTPUT/"
echo "  - Group1_Final/"
echo "  - Group2_Final/"
echo "  - Group3_Final/"
echo "  - Group4_Final/"
echo "  - all_groups_results.csv"
