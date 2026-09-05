"""
Synthetic Growth Curve Data Generator

Main module for generating synthetic bacterial growth curves.
Combines growth models, noise models, and scenario definitions
to create comprehensive test datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

from .growth_models import (
    GompertzModel, BaranyiModel, LogisticModel, RichardsModel, HaldaneModel,
    DeathPhaseExtension, get_model, convert_parameters, generate_curve
)
from .noise_models import (
    GaussianNoise, ODDependentNoise, InstrumentNoise, RMSEBasedNoise,
    get_noise_model, get_rmse_noise, NOISE_PRESETS
)
from .curve_scenarios import (
    ScenarioConfig, ScenarioSampler, ALL_SCENARIOS,
    GOOD_GROWTH_SCENARIOS, BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS,
    get_scenario, get_comprehensive_test_config
)


@dataclass
class GeneratedCurve:
    """Container for a single generated curve with metadata."""
    time: np.ndarray
    od600: np.ndarray
    clean_od600: np.ndarray  # Without noise
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class SyntheticGrowthCurveGenerator:
    """
    Main class for generating synthetic bacterial growth curves.

    Example usage:
        generator = SyntheticGrowthCurveGenerator()
        curve = generator.generate_single_curve(A=1.5, mu=0.2, lambda_=3.0)
        batch = generator.generate_from_scenario('standard', n=50)
        full_test = generator.generate_comprehensive_test_set()
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Default time parameters
        self.default_duration = 100.0
        self.default_resolution = 0.25

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_time_array(
        self,
        duration_hours: float = None,
        time_resolution: float = None
    ) -> np.ndarray:
        """Generate time array with specified parameters."""
        duration = duration_hours or self.default_duration
        resolution = time_resolution or self.default_resolution
        return np.arange(0, duration, resolution)

    def generate_single_curve(
        self,
        A: float = 1.0,
        mu: float = 0.15,
        lambda_: float = 3.0,
        model_type: str = 'gompertz',
        noise_level: str = 'medium',
        target_r_squared: Optional[float] = None,
        include_death_phase: bool = False,
        t_death: Optional[float] = None,
        k_death: float = 0.02,
        initial_od: float = 0.0,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        seed: Optional[int] = None
    ) -> GeneratedCurve:
        """
        Generate a single synthetic growth curve.

        Args:
            A: Maximum OD600 (asymptotic value)
            mu: Maximum specific growth rate (OD/hour)
            lambda_: Lag phase duration (hours)
            model_type: 'gompertz', 'baranyi', 'logistic', or 'richards'
            noise_level: 'very_low', 'low', 'medium', 'high', 'very_high'
            target_r_squared: If specified, adjust noise to achieve this R²
            include_death_phase: Whether to add death/decline phase
            t_death: Time when death phase begins (auto if None)
            k_death: Death rate constant
            initial_od: Initial/baseline OD600
            duration_hours: Experiment duration in hours
            time_resolution: Time between measurements (hours)
            seed: Random seed for this specific curve

        Returns:
            GeneratedCurve with time, OD600, and metadata
        """
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        # Generate time array
        time = self._get_time_array(duration_hours, time_resolution)

        # Generate clean curve based on model type
        if model_type == 'gompertz':
            clean_od = GompertzModel.compute(time, A, mu, lambda_)
        elif model_type == 'baranyi':
            baranyi_params = BaranyiModel.from_gompertz_params(A, mu, lambda_, initial_od)
            clean_od = BaranyiModel.compute(time, **baranyi_params)
        elif model_type == 'logistic':
            logistic_params = LogisticModel.from_gompertz_params(A, mu, lambda_)
            clean_od = LogisticModel.compute(time, **logistic_params)
        elif model_type == 'richards':
            richards_params = RichardsModel.from_gompertz_params(A, mu, lambda_)
            clean_od = RichardsModel.compute(time, **richards_params)
        elif model_type == 'haldane':
            haldane_params = HaldaneModel.from_gompertz_params(A, mu, lambda_)
            clean_od = HaldaneModel.compute(time, **haldane_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Add initial OD offset
        clean_od = clean_od + initial_od

        # Add death phase if requested
        if include_death_phase:
            if t_death is None:
                # Default: death starts at 70% of experiment
                t_death = duration_hours * 0.7

            # Find OD at death start
            death_idx = np.searchsorted(time, t_death)
            if death_idx < len(time):
                od_at_death = clean_od[death_idx]
                # Apply exponential decay after t_death
                death_mask = time > t_death
                clean_od[death_mask] = od_at_death * np.exp(-k_death * (time[death_mask] - t_death))

        # Add noise
        if target_r_squared is not None:
            # Use noise to achieve target R²
            noise_model = RMSEBasedNoise()
            noisy_od, actual_r2, rmse = noise_model.generate_noise_to_target_r2(
                clean_od, target_r_squared, seed=local_rng.integers(0, 2**31)
            )
        else:
            # Use preset noise level
            noise_model = get_noise_model(noise_level)
            noisy_od = noise_model.apply(
                clean_od, time=time, seed=local_rng.integers(0, 2**31)
            )
            # Calculate actual metrics
            rmse = np.sqrt(np.mean((noisy_od - clean_od)**2))
            ss_res = np.sum((noisy_od - clean_od)**2)
            ss_tot = np.sum((clean_od - np.mean(clean_od))**2)
            actual_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # Ensure non-negative
        noisy_od = np.maximum(0, noisy_od)

        # Compile metadata
        parameters = {
            'A': A,
            'mu': mu,
            'lambda_': lambda_,
            'model_type': model_type,
            'initial_od': initial_od,
            'include_death_phase': include_death_phase,
            't_death': t_death if include_death_phase else None,
            'k_death': k_death if include_death_phase else None,
        }

        metadata = {
            'noise_level': noise_level,
            'target_r_squared': target_r_squared,
            'actual_r_squared': actual_r2,
            'rmse': rmse,
            'duration_hours': duration_hours,
            'time_resolution': time_resolution,
            'n_points': len(time),
            'max_od': float(np.max(noisy_od)),
            'delta_od': float(np.max(noisy_od) - np.mean(noisy_od[:5])),
        }

        return GeneratedCurve(
            time=time,
            od600=noisy_od,
            clean_od600=clean_od,
            parameters=parameters,
            metadata=metadata
        )

    def generate_flat_curve(
        self,
        initial_od: float = 0.01,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        noise_level: str = 'very_low',
        seed: Optional[int] = None
    ) -> GeneratedCurve:
        """
        Generate a flat curve (no growth, like H2O control).

        Args:
            initial_od: Baseline OD600
            duration_hours: Experiment duration
            time_resolution: Measurement interval
            noise_level: Noise level to add
            seed: Random seed

        Returns:
            GeneratedCurve representing no-growth control
        """
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        time = self._get_time_array(duration_hours, time_resolution)

        # Flat baseline
        clean_od = np.full_like(time, initial_od)

        # Add small noise
        noise_model = get_noise_model(noise_level)
        noisy_od = noise_model.apply(clean_od, time=time, seed=local_rng.integers(0, 2**31))
        noisy_od = np.maximum(0, noisy_od)

        parameters = {
            'A': 0.0,
            'mu': 0.0,
            'lambda_': 0.0,
            'model_type': 'flat',
            'initial_od': initial_od,
            'include_death_phase': False,
        }

        metadata = {
            'noise_level': noise_level,
            'pattern': 'flat',
            'actual_r_squared': 0.0,  # No fit possible
            'max_od': float(np.max(noisy_od)),
            'delta_od': float(np.max(noisy_od) - np.mean(noisy_od[:5])),
        }

        return GeneratedCurve(
            time=time,
            od600=noisy_od,
            clean_od600=clean_od,
            parameters=parameters,
            metadata=metadata
        )

    def generate_random_walk_curve(
        self,
        initial_od: float = 0.1,
        walk_sigma: float = 0.02,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        seed: Optional[int] = None
    ) -> GeneratedCurve:
        """
        Generate an erratic random walk curve (non-biological pattern).

        Args:
            initial_od: Starting OD600
            walk_sigma: Step size standard deviation
            duration_hours: Experiment duration
            time_resolution: Measurement interval
            seed: Random seed

        Returns:
            GeneratedCurve with erratic pattern
        """
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        time = self._get_time_array(duration_hours, time_resolution)

        # Random walk
        steps = local_rng.normal(0, walk_sigma, len(time))
        od = initial_od + np.cumsum(steps)

        # Clip to reasonable range
        od = np.clip(od, 0, 2.0)

        parameters = {
            'A': 0.0,
            'mu': 0.0,
            'lambda_': 0.0,
            'model_type': 'random_walk',
            'initial_od': initial_od,
            'include_death_phase': False,
        }

        metadata = {
            'pattern': 'random_walk',
            'walk_sigma': walk_sigma,
            'noise_level': 'high',
            'actual_r_squared': -1.0,  # Definitely bad fit
            'max_od': float(np.max(od)),
            'delta_od': float(np.max(od) - od[0]),
        }

        return GeneratedCurve(
            time=time,
            od600=od,
            clean_od600=od.copy(),  # No "clean" version
            parameters=parameters,
            metadata=metadata
        )

    def generate_diauxic_curve(
        self,
        A1: float = 0.8,
        mu1: float = 0.25,
        lambda1: float = 2.0,
        A2: float = 0.5,
        mu2: float = 0.15,
        gap_hours: float = 10.0,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        noise_level: str = 'medium',
        seed: Optional[int] = None
    ) -> GeneratedCurve:
        """
        Generate a diauxic-like curve with two growth phases.

        This tests the truncation algorithm's ability to find the first peak.

        Args:
            A1: Max OD of first growth phase
            mu1: Growth rate of first phase
            lambda1: Lag time of first phase
            A2: Additional OD from second growth phase
            mu2: Growth rate of second phase
            gap_hours: Gap between first plateau and second growth
            duration_hours: Total experiment duration
            time_resolution: Measurement interval
            noise_level: Noise level
            seed: Random seed

        Returns:
            GeneratedCurve with diauxic pattern
        """
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        time = self._get_time_array(duration_hours, time_resolution)

        # First growth phase
        od1 = GompertzModel.compute(time, A1, mu1, lambda1)

        # Second growth phase (shifted)
        # Find when first phase reaches ~90% of A1
        t_first_plateau = lambda1 + A1 / (mu1 * np.e) + 5
        t_second_start = t_first_plateau + gap_hours

        # Second phase starts from A1 level
        od2_times = time - t_second_start
        od2 = np.zeros_like(time)
        second_phase_mask = time > t_second_start
        od2[second_phase_mask] = A2 * (1 - np.exp(-mu2 * od2_times[second_phase_mask]))

        # Combine: take max of first phase or sum where second kicks in
        clean_od = od1 + od2

        # Add noise
        noise_model = get_noise_model(noise_level)
        noisy_od = noise_model.apply(clean_od, time=time, seed=local_rng.integers(0, 2**31))
        noisy_od = np.maximum(0, noisy_od)

        parameters = {
            'A': A1,  # Report first phase params (what pipeline should find)
            'mu': mu1,
            'lambda_': lambda1,
            'model_type': 'diauxic',
            'initial_od': 0.0,
            'include_death_phase': False,
            'A2': A2,
            'mu2': mu2,
            'gap_hours': gap_hours,
        }

        metadata = {
            'pattern': 'diauxic',
            'noise_level': noise_level,
            'max_od': float(np.max(noisy_od)),
            'delta_od': float(np.max(noisy_od) - np.mean(noisy_od[:5])),
            'first_plateau_time': t_first_plateau,
        }

        return GeneratedCurve(
            time=time,
            od600=noisy_od,
            clean_od600=clean_od,
            parameters=parameters,
            metadata=metadata
        )

    def generate_from_scenario(
        self,
        scenario_name: str,
        n_curves: int = 10,
        seed: Optional[int] = None
    ) -> List[GeneratedCurve]:
        """
        Generate curves according to a scenario specification.

        Args:
            scenario_name: Name of scenario from curve_scenarios.py
            n_curves: Number of curves to generate
            seed: Random seed

        Returns:
            List of GeneratedCurve objects
        """
        scenario = get_scenario(scenario_name)
        sampler = ScenarioSampler(scenario, seed=seed)

        curves = []
        for i in range(n_curves):
            params = sampler.sample_parameters()

            # Handle special patterns
            if params.get('pattern') == 'flat':
                curve = self.generate_flat_curve(
                    initial_od=params['initial_od'],
                    duration_hours=params['duration_hours'],
                    time_resolution=params['time_resolution'],
                    noise_level=params['noise_level'],
                    seed=seed + i if seed else None
                )
            elif params.get('pattern') == 'random_walk':
                curve = self.generate_random_walk_curve(
                    initial_od=params['initial_od'],
                    duration_hours=params['duration_hours'],
                    time_resolution=params['time_resolution'],
                    seed=seed + i if seed else None
                )
            elif params.get('pattern') == 'diauxic':
                curve = self.generate_diauxic_curve(
                    A1=params['A'],
                    mu1=params['mu'],
                    lambda1=params['lambda_'],
                    duration_hours=params['duration_hours'],
                    time_resolution=params['time_resolution'],
                    noise_level=params['noise_level'],
                    seed=seed + i if seed else None
                )
            else:
                # Standard growth curve
                curve = self.generate_single_curve(
                    A=params['A'],
                    mu=params['mu'],
                    lambda_=params['lambda_'],
                    model_type=params['model'],
                    noise_level=params['noise_level'],
                    target_r_squared=params.get('target_r_squared'),
                    include_death_phase=params['include_death_phase'],
                    t_death=(params['duration_hours'] * params.get('t_death_fraction', 0.7)
                            if params['include_death_phase'] else None),
                    k_death=params.get('k_death', 0.02),
                    initial_od=params['initial_od'],
                    duration_hours=params['duration_hours'],
                    time_resolution=params['time_resolution'],
                    seed=seed + i if seed else None
                )

            # Add scenario metadata
            curve.metadata['scenario'] = scenario_name
            curve.metadata['expected_class'] = params['expected_class']
            curve.metadata['curve_index'] = i

            curves.append(curve)

        return curves

    def generate_triplicate(
        self,
        A: float = 1.0,
        mu: float = 0.15,
        lambda_: float = 3.0,
        replicate_variance: float = 0.05,
        noise_level: str = 'medium',
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[GeneratedCurve]]:
        """
        Generate biological triplicates with slight parameter variation.

        Simulates realistic biological variability between replicates.

        Args:
            A, mu, lambda_: Base parameters
            replicate_variance: Fraction of parameter variation (e.g., 0.05 = 5%)
            noise_level: Noise level for each replicate
            seed: Random seed
            **kwargs: Additional parameters passed to generate_single_curve

        Returns:
            Tuple of (time, od_mean, od_std, individual_curves)
        """
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        curves = []
        for i in range(3):
            # Vary parameters slightly for each replicate
            A_rep = A * (1 + local_rng.uniform(-replicate_variance, replicate_variance))
            mu_rep = mu * (1 + local_rng.uniform(-replicate_variance, replicate_variance))
            lambda_rep = lambda_ * (1 + local_rng.uniform(-replicate_variance, replicate_variance))

            curve = self.generate_single_curve(
                A=A_rep,
                mu=mu_rep,
                lambda_=lambda_rep,
                noise_level=noise_level,
                seed=local_rng.integers(0, 2**31),
                **kwargs
            )
            curves.append(curve)

        # Calculate mean and std
        time = curves[0].time
        od_stack = np.array([c.od600 for c in curves])
        od_mean = np.mean(od_stack, axis=0)
        od_std = np.std(od_stack, axis=0)

        return time, od_mean, od_std, curves

    def generate_comprehensive_test_set(
        self,
        config: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive test data covering all scenarios.

        Args:
            config: Dict mapping scenario name to number of curves.
                   If None, uses default configuration.
            seed: Random seed for reproducibility

        Returns:
            DataFrame with all generated curves and metadata
        """
        if config is None:
            config = get_comprehensive_test_config()

        if seed is not None:
            self.set_seed(seed)

        all_rows = []
        curve_id = 0

        for scenario_name, n_curves in config.items():
            print(f"Generating {n_curves} curves for scenario: {scenario_name}")

            curves = self.generate_from_scenario(
                scenario_name,
                n_curves=n_curves,
                seed=seed + curve_id if seed else None
            )

            for curve in curves:
                row = {
                    'curve_id': curve_id,
                    'scenario': scenario_name,
                    'expected_class': curve.metadata.get('expected_class', 'UNKNOWN'),

                    # Parameters
                    'true_A': curve.parameters.get('A', 0.0),
                    'true_mu': curve.parameters.get('mu', 0.0),
                    'true_lambda': curve.parameters.get('lambda_', 0.0),
                    'model_type': curve.parameters.get('model_type', 'unknown'),
                    'initial_od': curve.parameters.get('initial_od', 0.0),

                    # Metadata
                    'noise_level': curve.metadata.get('noise_level', 'medium'),
                    'target_r_squared': curve.metadata.get('target_r_squared'),
                    'actual_r_squared': curve.metadata.get('actual_r_squared'),
                    'rmse': curve.metadata.get('rmse'),
                    'max_od': curve.metadata.get('max_od'),
                    'delta_od': curve.metadata.get('delta_od'),
                    'n_points': curve.metadata.get('n_points', len(curve.time)),
                    'duration_hours': curve.parameters.get('duration_hours',
                                                          curve.time[-1] if len(curve.time) > 0 else 0),

                    # Store time series as lists (for later export)
                    'time': curve.time.tolist(),
                    'od600': curve.od600.tolist(),
                }

                all_rows.append(row)
                curve_id += 1

        df = pd.DataFrame(all_rows)

        print(f"\nGenerated {len(df)} total curves:")
        print(f"  Expected GOOD: {len(df[df['expected_class'] == 'GOOD'])}")
        print(f"  Expected BAD: {len(df[df['expected_class'] == 'BAD'])}")

        return df

    def generate_batch_from_parameters(
        self,
        parameter_df: pd.DataFrame,
        noise_level: str = 'medium',
        seed: Optional[int] = None
    ) -> List[GeneratedCurve]:
        """
        Generate curves from a DataFrame of parameters.

        Useful for generating curves from extracted real data parameters.

        Args:
            parameter_df: DataFrame with columns 'A', 'mu', 'lambda'
            noise_level: Noise level for all curves
            seed: Random seed

        Returns:
            List of GeneratedCurve objects
        """
        curves = []

        for i, row in parameter_df.iterrows():
            curve = self.generate_single_curve(
                A=row['A'],
                mu=row['mu'],
                lambda_=row.get('lambda', row.get('lambda_', 3.0)),
                noise_level=noise_level,
                seed=seed + i if seed else None
            )
            curves.append(curve)

        return curves


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_quick_test_set(n_per_category: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small test set for quick validation.

    Args:
        n_per_category: Curves per scenario category
        seed: Random seed

    Returns:
        DataFrame with test curves
    """
    generator = SyntheticGrowthCurveGenerator(seed=seed)

    config = {
        'standard': n_per_category,
        'flat_no_growth': n_per_category,
        'high_noise': n_per_category,
        'death_phase_moderate': n_per_category,
        'borderline_r2_good': n_per_category,
        'borderline_r2_bad': n_per_category,
    }

    return generator.generate_comprehensive_test_set(config=config, seed=seed)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing SyntheticGrowthCurveGenerator...")

    generator = SyntheticGrowthCurveGenerator(seed=42)

    # Test single curve generation
    print("\n1. Single curve generation:")
    curve = generator.generate_single_curve(A=1.5, mu=0.2, lambda_=3.0)
    print(f"   Generated curve with {len(curve.time)} points")
    print(f"   Max OD: {curve.metadata['max_od']:.3f}")
    print(f"   R²: {curve.metadata['actual_r_squared']:.4f}")

    # Test different scenarios
    print("\n2. Scenario generation:")
    for scenario in ['standard', 'flat_no_growth', 'death_phase_moderate']:
        curves = generator.generate_from_scenario(scenario, n_curves=3)
        print(f"   {scenario}: {len(curves)} curves, "
              f"expected={curves[0].metadata['expected_class']}")

    # Test comprehensive generation
    print("\n3. Quick test set generation:")
    test_df = generate_quick_test_set(n_per_category=3, seed=42)
    print(f"   Generated {len(test_df)} curves")
    print(f"   Scenarios: {test_df['scenario'].unique().tolist()}")

    # Plot examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    scenarios_to_plot = [
        'standard', 'flat_no_growth', 'high_noise',
        'death_phase_moderate', 'diauxic', 'borderline_r2_bad'
    ]

    # Generate diauxic separately since it's not in quick test
    generator2 = SyntheticGrowthCurveGenerator(seed=42)

    for ax, scenario in zip(axes.flat, scenarios_to_plot):
        if scenario == 'diauxic':
            curve = generator2.generate_diauxic_curve(seed=42)
        else:
            curves = generator2.generate_from_scenario(scenario, n_curves=1, seed=42)
            curve = curves[0]

        ax.plot(curve.time, curve.od600, 'b-', alpha=0.7, label='Noisy')
        ax.plot(curve.time, curve.clean_od600, 'r--', alpha=0.5, label='Clean')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('OD600')
        ax.set_title(f"{scenario}\n(Expected: {curve.metadata.get('expected_class', 'N/A')})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_generator_demo.png', dpi=150)
    print("\n4. Saved demo plot to data_generator_demo.png")

    plt.show()
