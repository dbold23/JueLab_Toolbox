"""
Curve Scenarios for Comprehensive Testing

Defines test scenarios covering all possible growth curve patterns:
- Good growth scenarios (should classify as GOOD)
- Bad curve scenarios (should classify as BAD)
- Edge case scenarios (test boundary conditions)

Each scenario specifies:
- Parameter ranges (A, mu, lambda)
- Noise level
- Expected classification
- Description for documentation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario."""
    name: str
    description: str
    expected_class: str  # 'GOOD' or 'BAD'

    # Gompertz parameters (can be fixed or range)
    A: Tuple[float, float] = (1.0, 1.5)
    mu: Tuple[float, float] = (0.1, 0.3)
    lambda_: Tuple[float, float] = (2.0, 8.0)

    # Initial OD (for contamination scenarios)
    initial_od: Tuple[float, float] = (0.0, 0.02)

    # Noise configuration
    noise_level: str = 'medium'  # 'very_low', 'low', 'medium', 'high', 'very_high'
    target_r_squared: Optional[Tuple[float, float]] = None  # Override noise to target R²

    # Death phase
    include_death_phase: bool = False
    t_death_fraction: Tuple[float, float] = (0.6, 0.8)  # Fraction of experiment
    k_death: Tuple[float, float] = (0.01, 0.05)

    # Experiment parameters
    duration_hours: float = 100.0
    time_resolution: float = 0.25

    # Special patterns
    pattern: Optional[str] = None  # 'flat', 'random_walk', 'diauxic', etc.

    # Model to use for generation
    model: str = 'gompertz'

    # Additional metadata
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Good Growth Scenarios (Should classify as GOOD)
# =============================================================================

GOOD_GROWTH_SCENARIOS = {
    'standard': ScenarioConfig(
        name='standard',
        description='Typical good growth curve (LB media)',
        expected_class='GOOD',
        A=(0.8, 1.5),
        mu=(0.1, 0.3),
        lambda_=(2.0, 8.0),
        noise_level='medium',
        tags=['typical', 'LB']
    ),

    'high_A': ScenarioConfig(
        name='high_A',
        description='Very high final OD (dense culture)',
        expected_class='GOOD',
        A=(1.5, 2.0),
        mu=(0.15, 0.35),
        lambda_=(1.0, 5.0),
        noise_level='medium',
        tags=['high_density']
    ),

    'low_A': ScenarioConfig(
        name='low_A',
        description='Low but valid growth (still above thresholds)',
        expected_class='GOOD',
        A=(0.15, 0.35),
        mu=(0.05, 0.15),
        lambda_=(3.0, 10.0),
        noise_level='low',
        tags=['low_density', 'borderline']
    ),

    'fast_growth': ScenarioConfig(
        name='fast_growth',
        description='Rapid exponential phase (high mu)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.4, 0.75),
        lambda_=(0.5, 2.0),
        noise_level='medium',
        tags=['fast', 'high_growth_rate']
    ),

    'slow_growth': ScenarioConfig(
        name='slow_growth',
        description='Slow exponential phase (low mu)',
        expected_class='GOOD',
        A=(0.8, 1.2),
        mu=(0.02, 0.08),
        lambda_=(5.0, 15.0),
        noise_level='low',
        duration_hours=120.0,  # Needs longer experiment
        tags=['slow', 'low_growth_rate']
    ),

    'short_lag': ScenarioConfig(
        name='short_lag',
        description='Very short lag phase (pre-adapted cells)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(0.1, 1.0),
        noise_level='medium',
        tags=['short_lag', 'pre_adapted']
    ),

    'long_lag': ScenarioConfig(
        name='long_lag',
        description='Extended lag phase (stressed cells)',
        expected_class='GOOD',
        A=(1.0, 1.4),
        mu=(0.1, 0.2),
        lambda_=(20.0, 50.0),
        noise_level='medium',
        duration_hours=120.0,
        tags=['long_lag', 'stressed']
    ),

    'very_clean': ScenarioConfig(
        name='very_clean',
        description='Very clean data with minimal noise (R² > 0.99)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        noise_level='very_low',
        tags=['clean', 'high_quality']
    ),

    'pesticide_lb_typical': ScenarioConfig(
        name='pesticide_lb_typical',
        description='Typical pesticide+LB growth (slightly lower A)',
        expected_class='GOOD',
        A=(1.2, 1.7),
        mu=(0.12, 0.22),
        lambda_=(2.0, 5.0),
        noise_level='medium',
        tags=['pesticide', 'treatment']
    ),
}


# =============================================================================
# Bad Curve Scenarios (Should classify as BAD)
# =============================================================================

BAD_CURVE_SCENARIOS = {
    'flat_no_growth': ScenarioConfig(
        name='flat_no_growth',
        description='H2O control - no growth at all',
        expected_class='BAD',
        A=(0.0, 0.0),  # No growth
        mu=(0.0, 0.0),
        lambda_=(0.0, 0.0),
        initial_od=(0.0, 0.02),
        noise_level='very_low',
        pattern='flat',
        tags=['negative_control', 'H2O', 'no_growth']
    ),

    'minimal_growth': ScenarioConfig(
        name='minimal_growth',
        description='Barely detectable growth (delta OD < 0.1)',
        expected_class='BAD',
        A=(0.03, 0.08),
        mu=(0.005, 0.02),
        lambda_=(5.0, 20.0),
        noise_level='low',
        tags=['minimal', 'borderline_bad']
    ),

    'contamination': ScenarioConfig(
        name='contamination',
        description='High initial OD indicating contamination',
        expected_class='BAD',
        A=(0.3, 0.6),
        mu=(0.05, 0.15),
        lambda_=(0.0, 1.0),
        initial_od=(0.2, 0.5),  # High starting OD
        noise_level='medium',
        tags=['contamination', 'high_initial_od']
    ),

    'high_noise': ScenarioConfig(
        name='high_noise',
        description='Valid growth pattern but excessive noise (R² < 0.90)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        noise_level='very_high',
        target_r_squared=(0.70, 0.90),
        tags=['noisy', 'poor_fit']
    ),

    'borderline_noise': ScenarioConfig(
        name='borderline_noise',
        description='Growth with borderline noise (R² ~0.90-0.94)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        target_r_squared=(0.90, 0.94),
        tags=['noisy', 'borderline']
    ),

    'erratic': ScenarioConfig(
        name='erratic',
        description='Non-biological erratic pattern',
        expected_class='BAD',
        pattern='random_walk',
        initial_od=(0.05, 0.15),
        noise_level='high',
        tags=['erratic', 'non_biological']
    ),

    'pesticide_only': ScenarioConfig(
        name='pesticide_only',
        description='Pesticide without nutrients - no growth',
        expected_class='BAD',
        A=(0.02, 0.08),
        mu=(0.001, 0.01),
        lambda_=(10.0, 50.0),
        noise_level='low',
        tags=['pesticide', 'no_nutrients', 'inhibited']
    ),

    'fit_failure': ScenarioConfig(
        name='fit_failure',
        description='Curve that causes fit convergence failure',
        expected_class='BAD',
        A=(0.01, 0.02),  # Very low signal
        mu=(0.001, 0.005),
        lambda_=(0.0, 1.0),
        noise_level='medium',
        tags=['fit_failure', 'low_signal']
    ),
}


# =============================================================================
# Edge Case Scenarios (Test boundary conditions)
# =============================================================================

EDGE_CASE_SCENARIOS = {
    'borderline_r2_good': ScenarioConfig(
        name='borderline_r2_good',
        description='R² just above 0.95 threshold (should be GOOD)',
        expected_class='GOOD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        target_r_squared=(0.95, 0.97),
        tags=['borderline', 'threshold_test']
    ),

    'borderline_r2_bad': ScenarioConfig(
        name='borderline_r2_bad',
        description='R² just below 0.95 threshold (should be BAD)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        target_r_squared=(0.93, 0.949),
        tags=['borderline', 'threshold_test']
    ),

    'truncation_challenge': ScenarioConfig(
        name='truncation_challenge',
        description='Multiple local maxima (diauxic-like growth)',
        expected_class='GOOD',  # Should still fit the first phase
        A=(1.0, 1.3),
        mu=(0.15, 0.25),
        lambda_=(2.0, 5.0),
        pattern='diauxic',
        noise_level='medium',
        tags=['truncation', 'diauxic', 'multi_peak']
    ),

    'death_phase_moderate': ScenarioConfig(
        name='death_phase_moderate',
        description='Clear death phase with moderate decline',
        expected_class='GOOD',  # Should truncate before death
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 5.0),
        include_death_phase=True,
        t_death_fraction=(0.5, 0.7),
        k_death=(0.02, 0.04),
        noise_level='medium',
        tags=['death_phase', 'decline']
    ),

    'death_phase_severe': ScenarioConfig(
        name='death_phase_severe',
        description='Severe death phase with rapid decline',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 4.0),
        include_death_phase=True,
        t_death_fraction=(0.4, 0.5),
        k_death=(0.05, 0.1),
        noise_level='medium',
        tags=['death_phase', 'severe_decline']
    ),

    'short_experiment': ScenarioConfig(
        name='short_experiment',
        description='Very short experiment (20 hours)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.2, 0.4),  # Fast growth to reach plateau
        lambda_=(1.0, 3.0),
        duration_hours=20.0,
        noise_level='medium',
        tags=['short_duration', 'limited_data']
    ),

    'long_experiment': ScenarioConfig(
        name='long_experiment',
        description='Extended experiment (150 hours)',
        expected_class='GOOD',
        A=(1.0, 1.4),
        mu=(0.08, 0.15),
        lambda_=(5.0, 15.0),
        duration_hours=150.0,
        noise_level='medium',
        tags=['long_duration', 'extended']
    ),

    'sparse_data': ScenarioConfig(
        name='sparse_data',
        description='Sparse time resolution (1 hour intervals)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        time_resolution=1.0,  # 1 hour instead of 0.25
        noise_level='medium',
        tags=['sparse', 'low_resolution']
    ),

    'dense_data': ScenarioConfig(
        name='dense_data',
        description='Dense time resolution (5 min intervals)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        time_resolution=0.083,  # 5 minutes
        noise_level='medium',
        tags=['dense', 'high_resolution']
    ),

    'baranyi_generated': ScenarioConfig(
        name='baranyi_generated',
        description='Baranyi model data fitted with Gompertz',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        model='baranyi',
        noise_level='medium',
        tags=['model_mismatch', 'robustness']
    ),

    'logistic_generated': ScenarioConfig(
        name='logistic_generated',
        description='Logistic model data fitted with Gompertz',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        model='logistic',
        noise_level='medium',
        tags=['model_mismatch', 'robustness']
    ),

    'richards_asymmetric': ScenarioConfig(
        name='richards_asymmetric',
        description='Highly asymmetric Richards curve',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        model='richards',
        noise_level='medium',
        tags=['model_mismatch', 'asymmetric']
    ),

    'outlier_contaminated': ScenarioConfig(
        name='outlier_contaminated',
        description='Good curve with occasional outlier spikes',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        noise_level='medium',  # Will add extra outliers
        tags=['outliers', 'robust_fit']
    ),

    'drift_positive': ScenarioConfig(
        name='drift_positive',
        description='Good curve with positive baseline drift',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        noise_level='medium',
        tags=['drift', 'systematic_error']
    ),

    'borderline_delta_od': ScenarioConfig(
        name='borderline_delta_od',
        description='Delta OD just above minimum threshold',
        expected_class='GOOD',
        A=(0.32, 0.38),  # Just above 0.3 threshold
        mu=(0.08, 0.12),
        lambda_=(3.0, 8.0),
        noise_level='low',
        tags=['borderline', 'low_signal']
    ),
}


# =============================================================================
# Scenario Collections
# =============================================================================

ALL_SCENARIOS = {
    **GOOD_GROWTH_SCENARIOS,
    **BAD_CURVE_SCENARIOS,
    **EDGE_CASE_SCENARIOS
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get a scenario by name."""
    if name not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. "
                        f"Available: {list(ALL_SCENARIOS.keys())}")
    return ALL_SCENARIOS[name]


def get_scenarios_by_class(expected_class: str) -> Dict[str, ScenarioConfig]:
    """Get all scenarios with a given expected classification."""
    return {
        name: config for name, config in ALL_SCENARIOS.items()
        if config.expected_class == expected_class
    }


def get_scenarios_by_tag(tag: str) -> Dict[str, ScenarioConfig]:
    """Get all scenarios with a given tag."""
    return {
        name: config for name, config in ALL_SCENARIOS.items()
        if tag in config.tags
    }


def list_scenarios() -> None:
    """Print a summary of all available scenarios."""
    print("=" * 70)
    print("AVAILABLE SCENARIOS")
    print("=" * 70)

    for category_name, scenarios in [
        ("GOOD GROWTH SCENARIOS", GOOD_GROWTH_SCENARIOS),
        ("BAD CURVE SCENARIOS", BAD_CURVE_SCENARIOS),
        ("EDGE CASE SCENARIOS", EDGE_CASE_SCENARIOS)
    ]:
        print(f"\n{category_name}:")
        print("-" * 70)
        for name, config in scenarios.items():
            print(f"  {name:30s} [{config.expected_class}] - {config.description}")

    print("\n" + "=" * 70)
    print(f"Total scenarios: {len(ALL_SCENARIOS)}")
    print(f"  Good: {len(get_scenarios_by_class('GOOD'))}")
    print(f"  Bad: {len(get_scenarios_by_class('BAD'))}")


def get_comprehensive_test_config() -> Dict[str, int]:
    """
    Get recommended number of curves per scenario for comprehensive testing.

    Returns:
        Dict mapping scenario name to number of curves to generate
    """
    config = {}

    # More samples for standard/common scenarios
    for name in GOOD_GROWTH_SCENARIOS:
        if name in ['standard', 'very_clean']:
            config[name] = 30
        else:
            config[name] = 15

    # Fewer samples for bad scenarios (they're expected to fail)
    for name in BAD_CURVE_SCENARIOS:
        config[name] = 10

    # Edge cases need good coverage
    for name in EDGE_CASE_SCENARIOS:
        if 'borderline' in name:
            config[name] = 20  # Extra samples near thresholds
        else:
            config[name] = 10

    return config


# =============================================================================
# Scenario Generator Helper
# =============================================================================

class ScenarioSampler:
    """Sample parameters from a scenario configuration."""

    def __init__(self, scenario: ScenarioConfig, seed: Optional[int] = None):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

    def sample_parameters(self) -> Dict[str, Any]:
        """Sample a single set of parameters from the scenario."""
        s = self.scenario

        params = {
            'A': self._sample_range(s.A),
            'mu': self._sample_range(s.mu),
            'lambda_': self._sample_range(s.lambda_),
            'initial_od': self._sample_range(s.initial_od),
            'model': s.model,
            'noise_level': s.noise_level,
            'duration_hours': s.duration_hours,
            'time_resolution': s.time_resolution,
            'include_death_phase': s.include_death_phase,
            'pattern': s.pattern,
            'expected_class': s.expected_class,
            'scenario_name': s.name,
        }

        if s.include_death_phase:
            params['t_death_fraction'] = self._sample_range(s.t_death_fraction)
            params['k_death'] = self._sample_range(s.k_death)

        if s.target_r_squared:
            params['target_r_squared'] = self._sample_range(s.target_r_squared)

        return params

    def _sample_range(self, range_tuple: Tuple[float, float]) -> float:
        """Sample uniformly from a range."""
        low, high = range_tuple
        if low == high:
            return low
        return self.rng.uniform(low, high)

    def sample_batch(self, n: int) -> List[Dict[str, Any]]:
        """Sample n parameter sets from the scenario."""
        return [self.sample_parameters() for _ in range(n)]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    list_scenarios()

    print("\n\nExample: Sampling from 'standard' scenario:")
    sampler = ScenarioSampler(get_scenario('standard'), seed=42)
    for i in range(3):
        params = sampler.sample_parameters()
        print(f"  Sample {i+1}: A={params['A']:.3f}, mu={params['mu']:.3f}, "
              f"lambda={params['lambda_']:.2f}")

    print("\n\nComprehensive test config:")
    config = get_comprehensive_test_config()
    total = sum(config.values())
    print(f"Total curves to generate: {total}")
