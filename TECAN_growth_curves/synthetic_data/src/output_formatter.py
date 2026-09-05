"""
Output Formatter for TECAN-Compatible CSV Files

Formats synthetic growth curve data to match the exact format
expected by the TECAN analysis pipeline:
- TIME[H] column
- {MEDIA}_{STRAIN}_blanked columns

Also generates ground truth CSV for validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json


@dataclass
class CurveExportConfig:
    """Configuration for exporting a curve."""
    curve_id: int
    treatment: str
    strain: str
    expected_class: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class TECANFormatWriter:
    """
    Format synthetic data to match real TECAN output format.
    Output format: TIME[H], {TREATMENT}_{STRAIN}_blanked columns
    """

    def __init__(self, output_dir: str):
        """
        Initialize writer with output directory.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_single_treatment(
        self,
        time: np.ndarray,
        curves: Dict[str, np.ndarray],
        treatment_name: str
    ) -> pd.DataFrame:
        """
        Format curves for a single treatment into TECAN-style DataFrame.

        Args:
            time: Time array in hours
            curves: Dict mapping strain_name to OD600 values
            treatment_name: Name of treatment (e.g., 'LB', 'PESTICIDE')

        Returns:
            DataFrame with TIME[H] and {treatment}_{strain}_blanked columns
        """
        df = pd.DataFrame()
        df['TIME[H]'] = time

        for strain_name, od_values in curves.items():
            col_name = f"{treatment_name}_{strain_name}_blanked"
            df[col_name] = od_values

        return df

    def write_treatment_csv(
        self,
        time: np.ndarray,
        curves: Dict[str, np.ndarray],
        treatment_name: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Write curves for a single treatment to CSV file.

        Args:
            time: Time array
            curves: Dict mapping strain to OD values
            treatment_name: Treatment name
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        df = self.format_single_treatment(time, curves, treatment_name)

        if filename is None:
            filename = f"{treatment_name}_DATA.csv"

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        return output_path

    def export_generated_curves(
        self,
        curves_df: pd.DataFrame,
        group_by: str = 'scenario',
        treatment_prefix: str = 'SYNTHETIC'
    ) -> Dict[str, Path]:
        """
        Export generated curves to TECAN-format CSV files.

        Groups curves by specified column and creates one CSV per group.

        Args:
            curves_df: DataFrame from SyntheticGrowthCurveGenerator.generate_comprehensive_test_set()
            group_by: Column to group curves by ('scenario', 'expected_class', etc.)
            treatment_prefix: Prefix for treatment names

        Returns:
            Dict mapping group name to output file path
        """
        output_files = {}

        for group_name, group_df in curves_df.groupby(group_by):
            # Create treatment name
            treatment = f"{treatment_prefix}_{group_name}".upper().replace(' ', '_')

            # Get common time array (assume all curves have same time)
            # Parse time from first row
            first_time = group_df.iloc[0]['time']
            if isinstance(first_time, str):
                time = np.array(eval(first_time))
            else:
                time = np.array(first_time)

            # Build curves dict
            curves = {}
            for idx, row in group_df.iterrows():
                strain_name = f"CURVE{row['curve_id']:04d}"

                # Parse OD values
                od = row['od600']
                if isinstance(od, str):
                    od = np.array(eval(od))
                else:
                    od = np.array(od)

                curves[strain_name] = od

            # Write to CSV
            output_path = self.write_treatment_csv(
                time, curves, treatment,
                filename=f"{treatment}_DATA.csv"
            )

            output_files[group_name] = output_path

        return output_files

    def create_ground_truth_csv(
        self,
        curves_df: pd.DataFrame,
        output_filename: str = 'ground_truth.csv'
    ) -> Path:
        """
        Create CSV with ground truth parameters for validation.

        Columns:
        - curve_id, strain_name
        - expected_class
        - true_A, true_mu, true_lambda
        - model_type, scenario
        - actual_r_squared, rmse, max_od, delta_od

        Args:
            curves_df: DataFrame from generator
            output_filename: Output filename

        Returns:
            Path to ground truth CSV
        """
        # Select columns for ground truth
        columns_to_keep = [
            'curve_id', 'scenario', 'expected_class',
            'true_A', 'true_mu', 'true_lambda',
            'model_type', 'initial_od',
            'noise_level', 'target_r_squared', 'actual_r_squared',
            'rmse', 'max_od', 'delta_od', 'n_points', 'duration_hours'
        ]

        # Filter to available columns
        available = [c for c in columns_to_keep if c in curves_df.columns]
        gt_df = curves_df[available].copy()

        # Add strain name for matching
        gt_df['strain_name'] = gt_df['curve_id'].apply(lambda x: f"CURVE{x:04d}")

        # Reorder columns
        cols = ['curve_id', 'strain_name'] + [c for c in available if c != 'curve_id']
        gt_df = gt_df[[c for c in cols if c in gt_df.columns]]

        output_path = self.output_dir / output_filename
        gt_df.to_csv(output_path, index=False)

        print(f"Ground truth saved to: {output_path}")
        return output_path

    def export_for_analysis_pipeline(
        self,
        curves_df: pd.DataFrame,
        data_subdir: str = 'DATA',
        create_ground_truth: bool = True
    ) -> Dict[str, Any]:
        """
        Export curves in a format ready for the analysis pipeline.

        Creates:
        - DATA/ subdirectory with *_DATA.csv files (grouped by time array length)
        - ground_truth.csv with expected results

        Curves with different time resolutions or durations are exported to
        separate CSV files since they cannot share the same TIME[H] column.

        Args:
            curves_df: DataFrame from generator
            data_subdir: Subdirectory name for data files
            create_ground_truth: Whether to create ground truth file

        Returns:
            Dict with paths to created files
        """
        # Create data subdirectory
        data_dir = self.output_dir / data_subdir
        data_dir.mkdir(exist_ok=True)

        result = {
            'data_dir': data_dir,
            'data_files': {},
            'ground_truth': None
        }

        data_writer = TECANFormatWriter(str(data_dir))

        # Group curves by their time array length to ensure compatible CSVs
        curves_df = curves_df.copy()
        curves_df['_n_points'] = curves_df['time'].apply(
            lambda t: len(eval(t)) if isinstance(t, str) else len(t)
        )

        for n_points, length_group in curves_df.groupby('_n_points'):
            # Get the time array from first curve in this length group
            first_time = length_group.iloc[0]['time']
            if isinstance(first_time, str):
                time = np.array(eval(first_time))
            else:
                time = np.array(first_time)

            # Build curves dict for all curves in this length group
            curves = {}
            for _, row in length_group.iterrows():
                strain = f"CURVE{row['curve_id']:04d}"

                od = row['od600']
                if isinstance(od, str):
                    od = np.array(eval(od))
                else:
                    od = np.array(od)

                curves[strain] = od

            # Create a descriptive group name
            duration = time[-1] if len(time) > 0 else 0
            group_name = f"SYNTHETIC_{int(n_points)}pts_{int(duration)}h"

            # Write CSV
            output_path = data_writer.write_treatment_csv(
                time, curves, group_name,
                filename=f"{group_name}_DATA.csv"
            )

            result['data_files'][group_name] = output_path

        # Clean up temp column
        curves_df.drop('_n_points', axis=1, inplace=True)

        # Create ground truth
        if create_ground_truth:
            result['ground_truth'] = self.create_ground_truth_csv(curves_df)

        return result


class BatchExporter:
    """
    Export large batches of synthetic data with proper organization.
    """

    def __init__(self, base_output_dir: str):
        """
        Initialize batch exporter.

        Args:
            base_output_dir: Base directory for all exports
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def export_by_scenario(
        self,
        curves_df: pd.DataFrame,
        scenarios_per_file: int = 1
    ) -> Dict[str, Path]:
        """
        Export curves organized by scenario.

        Creates separate directories for each scenario or group of scenarios.

        Args:
            curves_df: DataFrame from generator
            scenarios_per_file: Number of scenarios to combine per output file

        Returns:
            Dict mapping scenario to output directory
        """
        results = {}

        scenarios = curves_df['scenario'].unique()

        for scenario in scenarios:
            scenario_df = curves_df[curves_df['scenario'] == scenario]

            # Create scenario directory
            scenario_dir = self.base_dir / scenario
            writer = TECANFormatWriter(str(scenario_dir))

            # Export data
            export_result = writer.export_for_analysis_pipeline(
                scenario_df,
                data_subdir='.',
                create_ground_truth=True
            )

            results[scenario] = scenario_dir

        return results

    def export_comprehensive_test(
        self,
        curves_df: pd.DataFrame,
        test_name: str = 'comprehensive_test'
    ) -> Dict[str, Any]:
        """
        Export comprehensive test set with full organization.

        Creates:
        - {test_name}/
            - DATA/
                - SYNTHETIC_GOOD_DATA.csv
                - SYNTHETIC_BAD_DATA.csv
                - SYNTHETIC_ALL_DATA.csv
            - ground_truth.csv
            - config.json (test configuration)

        Args:
            curves_df: DataFrame from comprehensive generator
            test_name: Name for this test set

        Returns:
            Dict with all paths and metadata
        """
        test_dir = self.base_dir / test_name
        test_dir.mkdir(exist_ok=True)

        writer = TECANFormatWriter(str(test_dir))

        # Export data
        export_result = writer.export_for_analysis_pipeline(
            curves_df,
            data_subdir='DATA',
            create_ground_truth=True
        )

        # Create config file with test metadata
        config = {
            'test_name': test_name,
            'n_curves': len(curves_df),
            'n_good': len(curves_df[curves_df['expected_class'] == 'GOOD']),
            'n_bad': len(curves_df[curves_df['expected_class'] == 'BAD']),
            'scenarios': curves_df['scenario'].value_counts().to_dict(),
            'data_files': {k: str(v) for k, v in export_result['data_files'].items()},
            'ground_truth': str(export_result['ground_truth']),
        }

        config_path = test_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        export_result['config'] = config_path
        export_result['test_dir'] = test_dir

        print(f"\nExported comprehensive test to: {test_dir}")
        print(f"  Total curves: {config['n_curves']}")
        print(f"  Expected GOOD: {config['n_good']}")
        print(f"  Expected BAD: {config['n_bad']}")

        return export_result


# =============================================================================
# Convenience Functions
# =============================================================================

def export_quick_test(
    curves_df: pd.DataFrame,
    output_dir: str = 'synthetic_test_output'
) -> Dict[str, Any]:
    """
    Quick export for testing.

    Args:
        curves_df: DataFrame from generator
        output_dir: Output directory

    Returns:
        Export results
    """
    writer = TECANFormatWriter(output_dir)
    return writer.export_for_analysis_pipeline(curves_df)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from data_generator import generate_quick_test_set

    print("Testing output formatter...")

    # Generate test data
    print("\n1. Generating test data...")
    test_df = generate_quick_test_set(n_per_category=5, seed=42)
    print(f"   Generated {len(test_df)} curves")

    # Export to TECAN format
    print("\n2. Exporting to TECAN format...")
    exporter = BatchExporter('test_output')
    result = exporter.export_comprehensive_test(test_df, test_name='demo_test')

    print("\n3. Created files:")
    for key, value in result.items():
        if isinstance(value, Path):
            print(f"   {key}: {value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"   {key}/{k}: {v}")

    print("\nDone!")
