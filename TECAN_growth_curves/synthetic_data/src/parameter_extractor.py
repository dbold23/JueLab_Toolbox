"""
Parameter Extractor for Growth Curve Analysis

Extracts Gompertz parameter distributions from good curves in the
all_groups_results.csv file. Only uses curves classified as is_good=True.

Outputs:
- Parameter statistics (min, max, mean, std, percentiles)
- Fitted distributions for each parameter
- Treatment-grouped parameters
- YAML config file for synthetic data generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import yaml
from scipy import stats


@dataclass
class ParameterStats:
    """Statistics for a single parameter."""
    name: str
    min: float
    max: float
    mean: float
    std: float
    median: float
    p25: float  # 25th percentile
    p75: float  # 75th percentile
    p05: float  # 5th percentile
    p95: float  # 95th percentile
    count: int
    distribution: str
    distribution_params: Dict[str, float]


@dataclass
class ExtractedParameters:
    """Container for all extracted parameters."""
    A: ParameterStats
    mu: ParameterStats
    lambda_: ParameterStats
    r_squared: ParameterStats
    rmse: ParameterStats
    n_good_curves: int
    n_total_curves: int
    treatment_groups: Dict[str, Dict[str, ParameterStats]]


class ParameterExtractor:
    """
    Extract Gompertz parameter distributions from all_groups_results.csv.
    Only uses is_good=True curves.
    """

    def __init__(self, results_csv_path: str):
        """
        Initialize extractor with path to results CSV.

        Args:
            results_csv_path: Path to all_groups_results.csv or processing_results.csv
        """
        self.results_path = Path(results_csv_path)
        self.df = pd.read_csv(results_csv_path)

        # Column name mapping (handle variations)
        self._init_column_names()

    def _init_column_names(self):
        """Initialize column name mappings."""
        # Find the actual column names in the CSV
        cols = self.df.columns.tolist()

        self.col_is_good = 'is_good'
        self.col_A = 'gompertz_a'
        self.col_mu = 'gompertz_mu'
        self.col_lambda = 'gompertz_lambda'
        self.col_A_err = 'gompertz_a_err'
        self.col_mu_err = 'gompertz_mu_err'
        self.col_lambda_err = 'gompertz_lambda_err'
        self.col_r2 = 'fit_r_squared'
        self.col_rmse = 'fit_rmse'
        self.col_strain = 'strain'
        self.col_group = 'group' if 'group' in cols else None

    def filter_good_curves(self) -> pd.DataFrame:
        """
        Filter to only good curves (is_good=True) with valid parameters.

        Returns:
            DataFrame containing only good curves with valid Gompertz parameters
        """
        # Filter by is_good
        good = self.df[self.df[self.col_is_good] == True].copy()

        # Remove rows with NaN in key parameters
        param_cols = [self.col_A, self.col_mu, self.col_lambda]
        good = good.dropna(subset=param_cols)

        # Remove rows with invalid values (negative, inf)
        for col in param_cols:
            good = good[good[col] > 0]
            good = good[good[col] < np.inf]

        return good

    def _fit_distribution(self, values: np.ndarray, param_name: str) -> Tuple[str, Dict[str, float]]:
        """
        Fit the best distribution to parameter values.

        Tests: normal, lognormal, exponential, gamma

        Args:
            values: Array of parameter values
            param_name: Name of parameter (for context)

        Returns:
            (distribution_name, distribution_parameters)
        """
        if len(values) < 5:
            # Not enough data to fit
            return 'uniform', {'low': values.min(), 'high': values.max()}

        # Clean data
        values = values[~np.isnan(values)]
        values = values[values > 0]  # Most params must be positive
        values = values[values < np.inf]

        if len(values) < 5:
            return 'uniform', {'low': 0.01, 'high': 1.0}

        # Test different distributions
        distributions = {}

        # Normal
        try:
            loc, scale = stats.norm.fit(values)
            _, p_norm = stats.normaltest(values)
            distributions['normal'] = {
                'params': {'loc': loc, 'scale': scale},
                'p_value': p_norm
            }
        except Exception:
            pass

        # Lognormal (for positive-only parameters)
        try:
            shape, loc, scale = stats.lognorm.fit(values, floc=0)
            distributions['lognormal'] = {
                'params': {'shape': shape, 'loc': loc, 'scale': scale},
                'ks_stat': stats.kstest(values, 'lognorm', args=(shape, loc, scale)).statistic
            }
        except Exception:
            pass

        # Exponential (for parameters like lambda that might be exponentially distributed)
        try:
            loc, scale = stats.expon.fit(values)
            distributions['exponential'] = {
                'params': {'loc': loc, 'scale': scale},
                'ks_stat': stats.kstest(values, 'expon', args=(loc, scale)).statistic
            }
        except Exception:
            pass

        # Gamma
        try:
            a, loc, scale = stats.gamma.fit(values, floc=0)
            distributions['gamma'] = {
                'params': {'a': a, 'loc': loc, 'scale': scale},
                'ks_stat': stats.kstest(values, 'gamma', args=(a, loc, scale)).statistic
            }
        except Exception:
            pass

        # Select best distribution based on KS test (lower is better)
        if not distributions:
            return 'uniform', {'low': values.min(), 'high': values.max()}

        # For simplicity, prefer lognormal for A and mu, exponential for lambda
        if param_name in ['A', 'mu'] and 'lognormal' in distributions:
            return 'lognormal', distributions['lognormal']['params']
        elif param_name == 'lambda' and 'exponential' in distributions:
            return 'exponential', distributions['exponential']['params']
        elif 'normal' in distributions:
            return 'normal', distributions['normal']['params']

        # Fallback to first available
        dist_name = list(distributions.keys())[0]
        return dist_name, distributions[dist_name]['params']

    def _compute_stats(self, values: np.ndarray, name: str) -> ParameterStats:
        """Compute comprehensive statistics for a parameter."""
        # Clean values
        values = values[~np.isnan(values)]
        values = values[values < np.inf]
        values = values[values > -np.inf]

        if len(values) == 0:
            return ParameterStats(
                name=name, min=0, max=0, mean=0, std=0, median=0,
                p25=0, p75=0, p05=0, p95=0, count=0,
                distribution='uniform', distribution_params={'low': 0, 'high': 1}
            )

        dist_name, dist_params = self._fit_distribution(values, name)

        return ParameterStats(
            name=name,
            min=float(np.min(values)),
            max=float(np.max(values)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            median=float(np.median(values)),
            p25=float(np.percentile(values, 25)),
            p75=float(np.percentile(values, 75)),
            p05=float(np.percentile(values, 5)),
            p95=float(np.percentile(values, 95)),
            count=len(values),
            distribution=dist_name,
            distribution_params={k: float(v) for k, v in dist_params.items()}
        )

    def extract_parameter_stats(self) -> ExtractedParameters:
        """
        Extract statistics for all Gompertz parameters from good curves.

        Returns:
            ExtractedParameters object with all statistics
        """
        good = self.filter_good_curves()

        A_stats = self._compute_stats(good[self.col_A].values, 'A')
        mu_stats = self._compute_stats(good[self.col_mu].values, 'mu')
        lambda_stats = self._compute_stats(good[self.col_lambda].values, 'lambda')

        # R² and RMSE stats
        r2_values = good[self.col_r2].values if self.col_r2 in good.columns else np.array([0.95])
        rmse_values = good[self.col_rmse].values if self.col_rmse in good.columns else np.array([0.02])

        r2_stats = self._compute_stats(r2_values, 'r_squared')
        rmse_stats = self._compute_stats(rmse_values, 'rmse')

        # Extract by treatment group if available
        treatment_groups = self._extract_by_treatment_group(good)

        return ExtractedParameters(
            A=A_stats,
            mu=mu_stats,
            lambda_=lambda_stats,
            r_squared=r2_stats,
            rmse=rmse_stats,
            n_good_curves=len(good),
            n_total_curves=len(self.df),
            treatment_groups=treatment_groups
        )

    def _extract_by_treatment_group(self, good: pd.DataFrame) -> Dict[str, Dict[str, ParameterStats]]:
        """
        Extract parameters grouped by treatment type.

        Groups curves by treatment prefix in strain name:
        - LB-* → LB control
        - *ANDLB-* → Pesticide + LB
        - H2O-* → H2O control
        - Other → Pesticide only

        Returns:
            Dict mapping group name to parameter stats
        """
        groups = {}

        # Parse treatment from strain name
        def get_treatment_type(strain: str) -> str:
            strain = str(strain).upper()
            if strain.startswith('LB-'):
                return 'LB_control'
            elif 'ANDLB-' in strain:
                return 'Pesticide_LB'
            elif strain.startswith('H2O-'):
                return 'H2O_control'
            else:
                return 'Pesticide_only'

        good = good.copy()
        good['treatment_type'] = good[self.col_strain].apply(get_treatment_type)

        for treatment_type in good['treatment_type'].unique():
            subset = good[good['treatment_type'] == treatment_type]

            if len(subset) < 3:
                continue

            groups[treatment_type] = {
                'A': self._compute_stats(subset[self.col_A].values, 'A'),
                'mu': self._compute_stats(subset[self.col_mu].values, 'mu'),
                'lambda': self._compute_stats(subset[self.col_lambda].values, 'lambda'),
                'count': len(subset)
            }

        return groups

    def generate_sample_parameters(
        self,
        n_samples: int,
        use_distribution: bool = True,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate n random parameter sets from fitted distributions.

        Args:
            n_samples: Number of parameter sets to generate
            use_distribution: If True, sample from fitted distributions.
                            If False, sample uniformly from observed range.
            seed: Random seed for reproducibility

        Returns:
            DataFrame with columns: A, mu, lambda, expected_class
        """
        if seed is not None:
            np.random.seed(seed)

        params = self.extract_parameter_stats()

        samples = []
        for _ in range(n_samples):
            if use_distribution:
                A = self._sample_from_distribution(params.A)
                mu = self._sample_from_distribution(params.mu)
                lambda_ = self._sample_from_distribution(params.lambda_)
            else:
                A = np.random.uniform(params.A.p05, params.A.p95)
                mu = np.random.uniform(params.mu.p05, params.mu.p95)
                lambda_ = np.random.uniform(params.lambda_.p05, params.lambda_.p95)

            samples.append({
                'A': A,
                'mu': mu,
                'lambda': lambda_,
                'expected_class': 'GOOD'  # These are sampled from good curve params
            })

        return pd.DataFrame(samples)

    def _sample_from_distribution(self, param_stats: ParameterStats) -> float:
        """Sample a single value from the fitted distribution."""
        dist_name = param_stats.distribution
        params = param_stats.distribution_params

        if dist_name == 'normal':
            value = np.random.normal(params['loc'], params['scale'])
        elif dist_name == 'lognormal':
            value = np.random.lognormal(
                mean=np.log(params['scale']),
                sigma=params['shape']
            )
        elif dist_name == 'exponential':
            value = np.random.exponential(params['scale']) + params.get('loc', 0)
        elif dist_name == 'gamma':
            value = np.random.gamma(params['a'], params['scale']) + params.get('loc', 0)
        else:  # uniform
            value = np.random.uniform(params['low'], params['high'])

        # Ensure positive and within reasonable bounds
        value = max(0.001, value)
        value = min(value, param_stats.max * 1.5)

        return value

    def save_to_yaml(self, output_path: str):
        """
        Save extracted parameters to YAML config file.

        Args:
            output_path: Path for output YAML file
        """
        params = self.extract_parameter_stats()

        config = {
            'metadata': {
                'source_file': str(self.results_path),
                'n_good_curves': params.n_good_curves,
                'n_total_curves': params.n_total_curves
            },
            'parameters': {
                'A': {
                    'description': 'Maximum OD600 (asymptotic value)',
                    'min': params.A.min,
                    'max': params.A.max,
                    'mean': params.A.mean,
                    'std': params.A.std,
                    'median': params.A.median,
                    'p05': params.A.p05,
                    'p95': params.A.p95,
                    'distribution': params.A.distribution,
                    'distribution_params': params.A.distribution_params
                },
                'mu': {
                    'description': 'Maximum specific growth rate (OD/hour)',
                    'min': params.mu.min,
                    'max': params.mu.max,
                    'mean': params.mu.mean,
                    'std': params.mu.std,
                    'median': params.mu.median,
                    'p05': params.mu.p05,
                    'p95': params.mu.p95,
                    'distribution': params.mu.distribution,
                    'distribution_params': params.mu.distribution_params
                },
                'lambda': {
                    'description': 'Lag phase duration (hours)',
                    'min': params.lambda_.min,
                    'max': params.lambda_.max,
                    'mean': params.lambda_.mean,
                    'std': params.lambda_.std,
                    'median': params.lambda_.median,
                    'p05': params.lambda_.p05,
                    'p95': params.lambda_.p95,
                    'distribution': params.lambda_.distribution,
                    'distribution_params': params.lambda_.distribution_params
                }
            },
            'fit_quality': {
                'r_squared': {
                    'min': params.r_squared.min,
                    'max': params.r_squared.max,
                    'mean': params.r_squared.mean,
                    'threshold': 0.95  # Classification threshold
                },
                'rmse': {
                    'min': params.rmse.min,
                    'max': params.rmse.max,
                    'mean': params.rmse.mean,
                    'typical_range': [params.rmse.p05, params.rmse.p95]
                }
            },
            'treatment_groups': {}
        }

        # Add treatment group stats
        for group_name, group_stats in params.treatment_groups.items():
            if isinstance(group_stats, dict) and 'A' in group_stats:
                config['treatment_groups'][group_name] = {
                    'count': group_stats.get('count', 0),
                    'A_range': [group_stats['A'].min, group_stats['A'].max],
                    'mu_range': [group_stats['mu'].min, group_stats['mu'].max],
                    'lambda_range': [group_stats['lambda'].min, group_stats['lambda'].max]
                }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Saved parameter config to: {output_path}")

    def print_summary(self):
        """Print a summary of extracted parameters."""
        params = self.extract_parameter_stats()

        print("=" * 60)
        print("PARAMETER EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Source: {self.results_path}")
        print(f"Good curves: {params.n_good_curves} / {params.n_total_curves} total")
        print()

        print("GOMPERTZ PARAMETERS (from good curves):")
        print("-" * 60)

        for name, stat in [('A (max OD)', params.A),
                          ('mu (growth rate)', params.mu),
                          ('lambda (lag phase)', params.lambda_)]:
            print(f"\n{name}:")
            print(f"  Range: [{stat.min:.4f}, {stat.max:.4f}]")
            print(f"  Mean ± Std: {stat.mean:.4f} ± {stat.std:.4f}")
            print(f"  Median: {stat.median:.4f}")
            print(f"  5th-95th percentile: [{stat.p05:.4f}, {stat.p95:.4f}]")
            print(f"  Best-fit distribution: {stat.distribution}")

        print("\n" + "-" * 60)
        print("FIT QUALITY METRICS:")
        print(f"  R² range: [{params.r_squared.min:.4f}, {params.r_squared.max:.4f}]")
        print(f"  RMSE range: [{params.rmse.min:.4f}, {params.rmse.max:.4f}]")

        if params.treatment_groups:
            print("\n" + "-" * 60)
            print("BY TREATMENT GROUP:")
            for group, stats in params.treatment_groups.items():
                if isinstance(stats, dict) and 'A' in stats:
                    print(f"\n  {group} (n={stats.get('count', '?')}):")
                    print(f"    A: [{stats['A'].min:.3f}, {stats['A'].max:.3f}]")
                    print(f"    mu: [{stats['mu'].min:.3f}, {stats['mu'].max:.3f}]")
                    print(f"    lambda: [{stats['lambda'].min:.2f}, {stats['lambda'].max:.2f}]")

        print("\n" + "=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract Gompertz parameters from growth curve analysis results'
    )
    parser.add_argument(
        'input',
        help='Path to all_groups_results.csv or processing_results.csv'
    )
    parser.add_argument(
        '-o', '--output',
        default='extracted_params.yaml',
        help='Output YAML file path (default: extracted_params.yaml)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=0,
        help='Generate N sample parameter sets and print them'
    )

    args = parser.parse_args()

    # Extract parameters
    extractor = ParameterExtractor(args.input)
    extractor.print_summary()
    extractor.save_to_yaml(args.output)

    if args.samples > 0:
        print(f"\nGenerated {args.samples} sample parameter sets:")
        samples = extractor.generate_sample_parameters(args.samples, seed=42)
        print(samples.to_string(index=False))
