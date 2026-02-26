"""
Noise Models for Synthetic Growth Curve Generation

Implements realistic noise patterns observed in TECAN plate reader data:
- GaussianNoise: Simple constant-variance noise
- ODDependentNoise: Heteroscedastic noise that scales with OD
- InstrumentNoise: Comprehensive model including baseline, proportional, outliers, drift
- RMSEBasedNoise: Generates noise to achieve target RMSE values

Real TECAN data characteristics (from analysis of good fits):
- RMSE range: 0.0006 - 0.087
- Typical RMSE: 0.005 - 0.05
- Noise often proportional to OD reading
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class NoiseCharacteristics:
    """Container for extracted noise characteristics."""
    baseline_sigma: float
    proportional_sigma: float
    outlier_probability: float
    drift_rate: float
    rmse_range: Tuple[float, float]


class GaussianNoise:
    """
    Simple Gaussian noise with constant standard deviation.

    Good for testing basic robustness but doesn't capture
    the full complexity of real instrument noise.
    """

    def __init__(self, sigma: float = 0.01):
        """
        Initialize Gaussian noise model.

        Args:
            sigma: Standard deviation of noise
        """
        self.sigma = sigma

    def apply(self, signal: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Add Gaussian noise to signal.

        Args:
            signal: Clean OD600 values
            seed: Random seed for reproducibility

        Returns:
            Noisy OD600 values
        """
        if seed is not None:
            np.random.seed(seed)

        noise = np.random.normal(0, self.sigma, size=signal.shape)
        return np.maximum(0, signal + noise)  # OD can't be negative

    def get_expected_rmse(self) -> float:
        """Return expected RMSE from this noise model."""
        return self.sigma


class ODDependentNoise:
    """
    Heteroscedastic noise that scales with OD reading.

    Real OD600 measurements often have noise proportional to the signal:
    - Low OD: dominated by baseline noise (instrument sensitivity limit)
    - High OD: dominated by proportional noise (photometric variability)

    Model:
        sigma(OD) = sigma_base + sigma_scale * OD
    """

    def __init__(self, sigma_base: float = 0.002, sigma_scale: float = 0.02):
        """
        Initialize OD-dependent noise model.

        Args:
            sigma_base: Baseline noise at OD=0 (typically 0.001-0.005)
            sigma_scale: Proportional noise coefficient (typically 0.01-0.05)
        """
        self.sigma_base = sigma_base
        self.sigma_scale = sigma_scale

    def apply(self, signal: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Add OD-dependent noise to signal.

        Args:
            signal: Clean OD600 values
            seed: Random seed for reproducibility

        Returns:
            Noisy OD600 values
        """
        if seed is not None:
            np.random.seed(seed)

        # Calculate point-wise standard deviation
        sigma = self.sigma_base + self.sigma_scale * np.abs(signal)

        # Generate heteroscedastic noise
        noise = np.random.normal(0, 1, size=signal.shape) * sigma

        return np.maximum(0, signal + noise)

    def get_sigma_at_od(self, od: float) -> float:
        """Calculate noise standard deviation at a given OD."""
        return self.sigma_base + self.sigma_scale * od


class InstrumentNoise:
    """
    Comprehensive TECAN instrument noise model.

    Combines multiple noise sources observed in real plate reader data:
    1. Baseline noise: Low OD measurement uncertainty
    2. Proportional noise: Noise that scales with OD
    3. Occasional outliers: Random spikes (bubbles, condensation, etc.)
    4. Drift: Slow systematic change over time (temperature, evaporation)

    This model produces the most realistic synthetic data.
    """

    def __init__(
        self,
        baseline_sigma: float = 0.002,
        proportional_sigma: float = 0.015,
        outlier_prob: float = 0.005,
        outlier_magnitude: float = 0.1,
        drift_rate: float = 0.0001
    ):
        """
        Initialize comprehensive instrument noise model.

        Args:
            baseline_sigma: Baseline noise at OD=0
            proportional_sigma: Proportional noise coefficient
            outlier_prob: Probability of outlier at each point (0-1)
            outlier_magnitude: Typical magnitude of outliers (OD units)
            drift_rate: Systematic drift per hour (OD/hour)
        """
        self.baseline_sigma = baseline_sigma
        self.proportional_sigma = proportional_sigma
        self.outlier_prob = outlier_prob
        self.outlier_magnitude = outlier_magnitude
        self.drift_rate = drift_rate

    def apply(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add comprehensive instrument noise to signal.

        Args:
            signal: Clean OD600 values
            time: Time values in hours (for drift calculation)
            seed: Random seed for reproducibility

        Returns:
            Noisy OD600 values
        """
        if seed is not None:
            np.random.seed(seed)

        noisy = signal.copy()

        # 1. Add OD-dependent noise
        sigma = self.baseline_sigma + self.proportional_sigma * np.abs(signal)
        gaussian_noise = np.random.normal(0, 1, size=signal.shape) * sigma
        noisy += gaussian_noise

        # 2. Add occasional outliers
        if self.outlier_prob > 0:
            outlier_mask = np.random.random(signal.shape) < self.outlier_prob
            outlier_values = np.random.normal(0, self.outlier_magnitude, size=signal.shape)
            noisy[outlier_mask] += outlier_values[outlier_mask]

        # 3. Add systematic drift
        if time is not None and self.drift_rate != 0:
            # Drift direction can be positive (evaporation) or negative (settling)
            drift = self.drift_rate * time
            noisy += drift

        # Ensure non-negative OD
        return np.maximum(0, noisy)

    def apply_with_random_drift_direction(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply noise with randomly chosen drift direction.

        Some experiments show increasing drift (evaporation concentrating cells),
        others show decreasing drift (settling, photobleaching).
        """
        if seed is not None:
            np.random.seed(seed)

        # Randomly choose drift direction
        drift_direction = np.random.choice([-1, 1])
        actual_drift_rate = self.drift_rate * drift_direction

        # Temporarily modify drift rate
        original_drift = self.drift_rate
        self.drift_rate = actual_drift_rate
        result = self.apply(signal, time, seed=None)  # Don't re-seed
        self.drift_rate = original_drift

        return result


class RMSEBasedNoise:
    """
    Generate noise to achieve a target RMSE value.

    Uses the RMSE distribution from real good curve fits (0.001 - 0.08)
    to generate realistic noise levels.

    This is useful for generating synthetic data that matches the
    noise characteristics of real experiments.
    """

    def __init__(
        self,
        target_rmse: Optional[float] = None,
        rmse_range: Tuple[float, float] = (0.001, 0.08)
    ):
        """
        Initialize RMSE-based noise model.

        Args:
            target_rmse: Specific target RMSE (if None, randomly sampled)
            rmse_range: Range to sample RMSE from (min, max)
        """
        self.target_rmse = target_rmse
        self.rmse_range = rmse_range

    def apply(
        self,
        signal: np.ndarray,
        seed: Optional[int] = None,
        distribution: str = 'uniform'
    ) -> Tuple[np.ndarray, float]:
        """
        Add noise to achieve target RMSE.

        Args:
            signal: Clean OD600 values
            seed: Random seed for reproducibility
            distribution: How to sample RMSE if not specified
                - 'uniform': Uniform distribution over range
                - 'log_uniform': Log-uniform (more low values)
                - 'realistic': Match observed distribution from real data

        Returns:
            Tuple of (noisy_signal, actual_rmse)
        """
        if seed is not None:
            np.random.seed(seed)

        # Determine target RMSE
        if self.target_rmse is not None:
            rmse = self.target_rmse
        else:
            rmse = self._sample_rmse(distribution)

        # For Gaussian noise, RMSE ≈ sigma
        # Add some OD-dependent component for realism
        base_sigma = rmse * 0.7
        prop_sigma = rmse * 0.3 / max(0.1, np.max(signal))

        sigma = base_sigma + prop_sigma * np.abs(signal)
        noise = np.random.normal(0, 1, size=signal.shape) * sigma

        noisy = np.maximum(0, signal + noise)

        # Calculate actual RMSE
        actual_rmse = np.sqrt(np.mean((noisy - signal)**2))

        return noisy, actual_rmse

    def _sample_rmse(self, distribution: str) -> float:
        """Sample an RMSE value from the specified distribution."""
        low, high = self.rmse_range

        if distribution == 'uniform':
            return np.random.uniform(low, high)

        elif distribution == 'log_uniform':
            # Log-uniform: more probability at lower values
            log_low, log_high = np.log10(low), np.log10(high)
            return 10 ** np.random.uniform(log_low, log_high)

        elif distribution == 'realistic':
            # Approximate the observed RMSE distribution from real data
            # Most values cluster around 0.02-0.04, with tails
            return np.random.lognormal(mean=np.log(0.025), sigma=0.6)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def generate_noise_to_target_r2(
        self,
        signal: np.ndarray,
        target_r2: float,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate noise to achieve approximately a target R² value.

        Useful for testing classification thresholds.

        Args:
            signal: Clean OD600 values (the "true" model curve)
            target_r2: Target R² value (e.g., 0.95 for borderline)
            seed: Random seed

        Returns:
            Tuple of (noisy_signal, actual_r2, rmse)
        """
        if seed is not None:
            np.random.seed(seed)

        # R² = 1 - SS_res/SS_tot
        # SS_tot is fixed (variance of signal)
        # SS_res = n * RMSE²
        # So: RMSE = sqrt((1 - R²) * var(signal))

        ss_tot = np.sum((signal - np.mean(signal))**2)

        if ss_tot == 0:
            # Flat signal - can't define R²
            return signal.copy(), 1.0, 0.0

        target_ss_res = (1 - target_r2) * ss_tot
        target_rmse = np.sqrt(target_ss_res / len(signal))

        # Generate noise with this RMSE (use local override, don't mutate instance)
        saved_rmse = self.target_rmse
        self.target_rmse = target_rmse
        noisy, actual_rmse = self.apply(signal, seed=None)  # Don't re-seed
        self.target_rmse = saved_rmse  # Restore original

        # Calculate actual R²
        ss_res = np.sum((noisy - signal)**2)
        actual_r2 = 1 - ss_res / ss_tot

        return noisy, actual_r2, actual_rmse


# =============================================================================
# Noise Level Presets
# =============================================================================

NOISE_PRESETS = {
    'very_low': {
        'description': 'Very clean data (R² > 0.99)',
        'baseline_sigma': 0.001,
        'proportional_sigma': 0.005,
        'outlier_prob': 0.0,
        'drift_rate': 0.0,
        'rmse_range': (0.0005, 0.005)
    },
    'low': {
        'description': 'Clean data typical of LB controls',
        'baseline_sigma': 0.002,
        'proportional_sigma': 0.01,
        'outlier_prob': 0.001,
        'drift_rate': 0.00005,
        'rmse_range': (0.005, 0.02)
    },
    'medium': {
        'description': 'Typical experimental noise',
        'baseline_sigma': 0.003,
        'proportional_sigma': 0.02,
        'outlier_prob': 0.005,
        'drift_rate': 0.0001,
        'rmse_range': (0.02, 0.05)
    },
    'high': {
        'description': 'Noisy data (borderline R² ~0.95)',
        'baseline_sigma': 0.005,
        'proportional_sigma': 0.03,
        'outlier_prob': 0.01,
        'drift_rate': 0.0002,
        'rmse_range': (0.05, 0.08)
    },
    'very_high': {
        'description': 'Very noisy data (R² < 0.95, likely BAD)',
        'baseline_sigma': 0.01,
        'proportional_sigma': 0.05,
        'outlier_prob': 0.02,
        'drift_rate': 0.0003,
        'rmse_range': (0.08, 0.15)
    }
}


def get_noise_model(preset: str) -> InstrumentNoise:
    """
    Get a pre-configured noise model.

    Args:
        preset: One of 'very_low', 'low', 'medium', 'high', 'very_high'

    Returns:
        Configured InstrumentNoise instance
    """
    if preset not in NOISE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available: {list(NOISE_PRESETS.keys())}")

    params = NOISE_PRESETS[preset]
    return InstrumentNoise(
        baseline_sigma=params['baseline_sigma'],
        proportional_sigma=params['proportional_sigma'],
        outlier_prob=params['outlier_prob'],
        drift_rate=params['drift_rate']
    )


def get_rmse_noise(preset: str) -> RMSEBasedNoise:
    """
    Get an RMSE-based noise model for a preset level.

    Args:
        preset: One of 'very_low', 'low', 'medium', 'high', 'very_high'

    Returns:
        Configured RMSEBasedNoise instance
    """
    if preset not in NOISE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    params = NOISE_PRESETS[preset]
    return RMSEBasedNoise(rmse_range=params['rmse_range'])


# =============================================================================
# Testing / Demonstration
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from growth_models import GompertzModel

    # Generate a clean curve
    t = np.arange(0, 50, 0.25)
    clean = GompertzModel.compute(t, A=1.5, mu=0.2, lambda_=3.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Gaussian noise at different levels
    ax1 = axes[0, 0]
    ax1.plot(t, clean, 'k-', label='Clean', linewidth=2)
    for sigma in [0.01, 0.03, 0.05]:
        noise_model = GaussianNoise(sigma=sigma)
        noisy = noise_model.apply(clean, seed=42)
        ax1.plot(t, noisy, alpha=0.7, label=f'σ={sigma}')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('OD600')
    ax1.set_title('Gaussian Noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: OD-dependent noise
    ax2 = axes[0, 1]
    ax2.plot(t, clean, 'k-', label='Clean', linewidth=2)
    noise_model = ODDependentNoise(sigma_base=0.002, sigma_scale=0.02)
    noisy = noise_model.apply(clean, seed=42)
    ax2.plot(t, noisy, 'r-', alpha=0.7, label='OD-dependent')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('OD600')
    ax2.set_title('OD-Dependent Noise')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Instrument noise with different presets
    ax3 = axes[0, 2]
    ax3.plot(t, clean, 'k-', label='Clean', linewidth=2)
    for preset in ['low', 'medium', 'high']:
        noise_model = get_noise_model(preset)
        noisy = noise_model.apply(clean, time=t, seed=42)
        ax3.plot(t, noisy, alpha=0.7, label=preset)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('OD600')
    ax3.set_title('Instrument Noise Presets')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: RMSE-based noise
    ax4 = axes[1, 0]
    ax4.plot(t, clean, 'k-', label='Clean', linewidth=2)
    for target_rmse in [0.01, 0.03, 0.06]:
        noise_model = RMSEBasedNoise(target_rmse=target_rmse)
        noisy, actual = noise_model.apply(clean, seed=42)
        ax4.plot(t, noisy, alpha=0.7, label=f'Target={target_rmse:.2f}, Actual={actual:.3f}')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('OD600')
    ax4.set_title('RMSE-Based Noise')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Noise to target R²
    ax5 = axes[1, 1]
    ax5.plot(t, clean, 'k-', label='Clean', linewidth=2)
    for target_r2 in [0.99, 0.95, 0.90]:
        noise_model = RMSEBasedNoise()
        noisy, actual_r2, rmse = noise_model.generate_noise_to_target_r2(clean, target_r2, seed=42)
        ax5.plot(t, noisy, alpha=0.7, label=f'Target R²={target_r2}, Actual={actual_r2:.3f}')
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('OD600')
    ax5.set_title('Noise to Target R²')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Effect of outliers
    ax6 = axes[1, 2]
    ax6.plot(t, clean, 'k-', label='Clean', linewidth=2)
    noise_model = InstrumentNoise(
        baseline_sigma=0.002,
        proportional_sigma=0.01,
        outlier_prob=0.03,
        outlier_magnitude=0.15,
        drift_rate=0.0
    )
    noisy = noise_model.apply(clean, time=t, seed=42)
    ax6.plot(t, noisy, 'r-', alpha=0.7, label='With outliers (3%)')
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('OD600')
    ax6.set_title('Effect of Outliers')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noise_models_demo.png', dpi=150)
    plt.show()

    print("Demo complete! Saved to noise_models_demo.png")
