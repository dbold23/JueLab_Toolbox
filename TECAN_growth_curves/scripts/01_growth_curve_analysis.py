"""
Growth Curve Analysis: Classification, Truncation, and Gompertz Fitting

This script processes bacterial growth curves from TECAN plate reader data.
It classifies curves as good/bad, truncates at peak OD600, and fits Gompertz models.

Author: Generated for BIO380SP25 Research Project
Date: December 2024
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


# =============================================================================
# Configuration / Thresholds
# =============================================================================

# Old threshold-based classification (kept for reference)
CLASSIFICATION_THRESHOLDS = {
    'min_delta_od': 0.3,           # Minimum OD600 change required
    'min_max_od': 0.5,             # Minimum peak OD600 required
    'min_snr': 5.0,                # Minimum signal-to-noise ratio
    'max_initial_od': 0.15,        # Maximum starting OD600 (contamination check)
}

# New fit-quality-based classification thresholds
FIT_QUALITY_THRESHOLDS = {
    'min_r_squared': 0.95,         # Minimum R² for a good fit
    'max_param_error_pct': 20.0,   # Maximum relative error (%) for parameters
    'min_points_for_fit': 15,      # Minimum data points to attempt fit
}

TRUNCATION_PARAMS = {
    'smoothing_window': 5,         # Rolling average window size
    'buffer_points': 3,            # Points to keep after max for fit stability
    'min_points_for_fit': 20,      # Minimum points required for fitting
}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of binary curve classification."""
    is_good: bool
    reason: str
    metrics: Dict[str, float]


@dataclass
class TruncationResult:
    """Result of curve truncation."""
    truncation_index: int
    truncation_time: float
    max_od: float
    time_truncated: np.ndarray
    od_truncated: np.ndarray
    time_original: np.ndarray
    od_original: np.ndarray
    smoothed_od: np.ndarray
    start_index: int = 0  # Start index (for trimming noisy baseline)


@dataclass
class FitResult:
    """Result of Gompertz model fitting."""
    success: bool
    a_opt: float           # Maximum OD600
    mu_opt: float          # Maximum growth rate
    lambda_opt: float      # Lag phase duration
    a_err: float           # Standard error of a
    mu_err: float          # Standard error of mu
    lambda_err: float      # Standard error of lambda
    mae: float             # Mean Absolute Error
    mse: float             # Mean Squared Error
    rmse: float            # Root Mean Squared Error
    r_squared: float       # R-squared value
    predicted: np.ndarray  # Predicted values
    residuals: np.ndarray  # Residuals
    error_message: str     # Error message if fitting failed


@dataclass
class ProcessingResult:
    """Complete result from processing a single strain."""
    strain_name: str
    classification: ClassificationResult
    truncation: Optional[TruncationResult]
    fit: Optional[FitResult]


# =============================================================================
# Classification Functions
# =============================================================================

def classify_growth_curve(
    time: np.ndarray,
    od600: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None
) -> ClassificationResult:
    """
    Binary classify a growth curve as good (worth fitting) or bad (not worth fitting).

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values (blanked)
    thresholds : Optional[Dict[str, float]]
        Custom classification thresholds. Uses defaults if None.

    Returns
    -------
    ClassificationResult
        Contains is_good (bool), reason (str), and metrics (dict)
    """
    if thresholds is None:
        thresholds = CLASSIFICATION_THRESHOLDS

    metrics = {}

    # Calculate basic statistics
    # Use first 5 points for initial OD (avoid single-point noise)
    metrics['initial_od'] = np.mean(od600[:5])
    metrics['max_od'] = np.max(od600)
    metrics['final_od'] = np.mean(od600[-5:])
    metrics['delta_od'] = metrics['max_od'] - metrics['initial_od']

    # Calculate noise estimate (std of first 10 points in lag phase)
    noise_estimate = np.std(od600[:10])
    metrics['noise_estimate'] = noise_estimate

    # Calculate signal-to-noise ratio
    signal = metrics['delta_od']
    metrics['snr'] = signal / noise_estimate if noise_estimate > 0 else float('inf')

    # Find time to max
    max_idx = np.argmax(od600)
    time_to_max = time[max_idx] - time[0]
    metrics['time_to_max'] = time_to_max
    metrics['max_idx'] = max_idx

    # Apply classification criteria
    reasons = []

    if metrics['delta_od'] < thresholds['min_delta_od']:
        reasons.append(f"Insufficient OD change: {metrics['delta_od']:.3f} < {thresholds['min_delta_od']}")

    if metrics['max_od'] < thresholds['min_max_od']:
        reasons.append(f"Maximum OD too low: {metrics['max_od']:.3f} < {thresholds['min_max_od']}")

    if metrics['snr'] < thresholds['min_snr']:
        reasons.append(f"Low signal-to-noise: {metrics['snr']:.1f} < {thresholds['min_snr']}")

    if metrics['initial_od'] > thresholds['max_initial_od']:
        reasons.append(f"High initial OD (contamination?): {metrics['initial_od']:.3f} > {thresholds['max_initial_od']}")

    is_good = len(reasons) == 0
    reason = "GOOD: All criteria passed" if is_good else "BAD: " + "; ".join(reasons)

    return ClassificationResult(is_good=is_good, reason=reason, metrics=metrics)


def classify_by_fit_quality(
    fit_result: 'FitResult',
    thresholds: Optional[Dict[str, float]] = None,
    time: Optional[np.ndarray] = None,
    od600: Optional[np.ndarray] = None
) -> ClassificationResult:
    """
    Classify a growth curve based on Gompertz fit quality.

    This is more scientifically rigorous than OD-threshold-based classification
    because it judges whether the curve is actually fittable, regardless of
    the absolute OD magnitude.

    Includes secondary quality gates:
    - SNR filter (signal must exceed noise floor)
    - Delta-OD confidence interval (growth must be statistically significant)
    - Flatness check (fitted amplitude must exceed noise)

    Parameters
    ----------
    fit_result : FitResult
        Result from fit_gompertz()
    thresholds : Optional[Dict[str, float]]
        Custom fit quality thresholds. Uses FIT_QUALITY_THRESHOLDS if None.
    time : Optional[np.ndarray]
        Original time array (for secondary quality checks)
    od600 : Optional[np.ndarray]
        Original OD600 array (for secondary quality checks)

    Returns
    -------
    ClassificationResult
        Contains is_good (bool), reason (str), and metrics (dict)
    """
    if thresholds is None:
        thresholds = FIT_QUALITY_THRESHOLDS

    metrics = {
        'r_squared': fit_result.r_squared,
        'rmse': fit_result.rmse,
        'a_opt': fit_result.a_opt,
        'mu_opt': fit_result.mu_opt,
        'lambda_opt': fit_result.lambda_opt,
        'a_err_pct': (fit_result.a_err / fit_result.a_opt * 100) if fit_result.a_opt > 0 else float('inf'),
        'mu_err_pct': (fit_result.mu_err / fit_result.mu_opt * 100) if fit_result.mu_opt > 0 else float('inf'),
        'lambda_err_pct': (fit_result.lambda_err / fit_result.lambda_opt * 100) if fit_result.lambda_opt > 0 else float('inf'),
    }

    reasons = []

    # Check if fit failed entirely
    if not fit_result.success:
        return ClassificationResult(
            is_good=False,
            reason=f"BAD: Fit failed - {fit_result.error_message}",
            metrics=metrics
        )

    # Check R²
    if fit_result.r_squared < thresholds['min_r_squared']:
        reasons.append(f"Low R²: {fit_result.r_squared:.3f} < {thresholds['min_r_squared']}")

    # Check parameter errors (only A and mu - lambda can be tricky near zero)
    max_err = thresholds['max_param_error_pct']
    if metrics['a_err_pct'] > max_err:
        reasons.append(f"High A error: {metrics['a_err_pct']:.1f}% > {max_err}%")
    if metrics['mu_err_pct'] > max_err:
        reasons.append(f"High mu error: {metrics['mu_err_pct']:.1f}% > {max_err}%")

    # =====================================================================
    # Secondary quality gates (require raw data)
    # Tightened: the R² >= 0.98 bypass was too permissive — noisy non-
    # growth curves could get a decent R² by chance. Now raised to 0.995.
    # SNR gate raised to 5.0 and always enforced with min delta-OD.
    # Added: residual autocorrelation check and monotonicity gate.
    # =====================================================================
    if od600 is not None and len(od600) > 10:
        n_baseline = min(10, len(od600) // 5)
        baseline = od600[:n_baseline]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline)) if float(np.std(baseline)) > 1e-8 else 1e-8
        max_od = float(np.max(od600))
        delta_od = max_od - baseline_mean

        snr = (max_od - baseline_mean) / baseline_std

        # Only bypass noise gates when fit is truly excellent (raised from 0.98)
        excellent_threshold = thresholds.get('excellent_r2_threshold', 0.995)
        fit_is_excellent = fit_result.r_squared >= excellent_threshold

        if not fit_is_excellent:
            # SNR filter: signal must meaningfully exceed noise floor
            min_snr = thresholds.get('min_snr', 5.0)
            if snr < min_snr:
                reasons.append(f"Low SNR: {snr:.1f} < {min_snr}")

            # Delta-OD confidence interval: growth must be statistically significant
            delta_od_ci_lower = delta_od - 2 * baseline_std
            min_delta_od_ci = thresholds.get('min_delta_od_ci', 0.1)
            if delta_od_ci_lower < min_delta_od_ci:
                reasons.append(f"Delta OD not significant: CI lower={delta_od_ci_lower:.3f} < {min_delta_od_ci}")

        # Absolute minimum delta-OD (always enforced, even for excellent fits)
        min_delta_od = thresholds.get('min_absolute_delta_od', 0.15)
        if delta_od < min_delta_od:
            reasons.append(f"Insufficient growth: delta_od={delta_od:.3f} < {min_delta_od}")

        # Flatness check: fitted A must exceed noise level (always enforced)
        if fit_result.a_opt < 3 * baseline_std:
            reasons.append(f"Growth amplitude below noise: A={fit_result.a_opt:.3f} < 3*noise={3*baseline_std:.3f}")

        # Monotonicity check: real growth curves have sustained upward trends.
        # Compute fraction of consecutive smoothed points that increase over
        # the first 60% of data (lag + exponential phase). Pure noise is ~50%.
        smoothed_check = pd.Series(od600).rolling(
            window=5, center=True, min_periods=1
        ).mean().values
        check_end = max(10, int(len(smoothed_check) * 0.6))
        diffs = np.diff(smoothed_check[:check_end])
        monotone_frac = float(np.sum(diffs > 0) / max(len(diffs), 1))
        min_monotone = thresholds.get('min_monotone_fraction', 0.55)
        if monotone_frac < min_monotone and not fit_is_excellent:
            reasons.append(
                f"Non-monotonic signal: {monotone_frac:.2f} < {min_monotone}"
            )

        # Residual autocorrelation check: structured residuals indicate the
        # Gompertz model is fitting noise rather than real sigmoid growth.
        if fit_result.success and len(fit_result.residuals) > 10:
            resid = fit_result.residuals
            n_res = len(resid)
            resid_mean = np.mean(resid)
            c0 = np.sum((resid - resid_mean) ** 2) / n_res
            if c0 > 1e-12:
                c1 = np.sum(
                    (resid[:-1] - resid_mean) * (resid[1:] - resid_mean)
                ) / n_res
                lag1_autocorr = c1 / c0
            else:
                lag1_autocorr = 0.0
            max_autocorr = thresholds.get('max_residual_autocorr', 0.7)
            if lag1_autocorr > max_autocorr and not fit_is_excellent:
                reasons.append(
                    f"High residual autocorrelation: {lag1_autocorr:.2f} > {max_autocorr}"
                )
            metrics['residual_autocorr'] = lag1_autocorr

        metrics['snr'] = snr
        metrics['delta_od'] = delta_od
        metrics['baseline_std'] = baseline_std
        metrics['monotone_fraction'] = monotone_frac

    is_good = len(reasons) == 0
    reason = "GOOD: Fit quality passed" if is_good else "BAD: " + "; ".join(reasons)

    return ClassificationResult(is_good=is_good, reason=reason, metrics=metrics)


# =============================================================================
# Truncation Functions
# =============================================================================

def detect_incomplete_curve(
    time: np.ndarray,
    od600: np.ndarray,
    smoothing_window: int = 7,
    tail_fraction: float = 0.15,
    slope_threshold: float = 0.005
) -> Tuple[bool, Dict]:
    """
    Detect if a growth curve is still rising at the end of the experiment
    (i.e., stationary phase was never reached).

    These curves should NOT be truncated early, since the peak hasn't occurred
    yet. Forcing truncation causes Gompertz fits to fail (truncation_challenge
    scenario).

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    smoothing_window : int
        Window for smoothing (default: 7)
    tail_fraction : float
        Fraction of data points at the end to check (default: 0.15 = last 15%)
    slope_threshold : float
        Minimum dOD/dt to consider "still rising" (default: 0.005 OD/h)

    Returns
    -------
    Tuple[bool, Dict]
        (is_incomplete, metadata)
        is_incomplete is True if the curve is still rising at experiment end
    """
    smoothed = pd.Series(od600).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    n = len(smoothed)
    tail_start = max(1, int(n * (1 - tail_fraction)))

    # Compute derivative in the tail region
    dt = np.diff(time[tail_start:])
    dod = np.diff(smoothed[tail_start:])
    # Avoid division by zero
    valid = dt > 0
    if not np.any(valid):
        return False, {'tail_slope': 0.0, 'tail_start_idx': tail_start}

    tail_rates = dod[valid] / dt[valid]
    mean_tail_slope = float(np.mean(tail_rates))

    # Check: is the tail still rising meaningfully?
    # Also check that the final OD is near the global max (within 5%)
    max_od = np.max(smoothed)
    final_od = smoothed[-1]
    near_max = final_od >= 0.95 * max_od

    # Curve is incomplete if:
    # 1. Mean tail slope is positive and above threshold, AND
    # 2. The final OD is near the global maximum (not a decline after peak)
    is_incomplete = mean_tail_slope > slope_threshold and near_max

    metadata = {
        'tail_slope': mean_tail_slope,
        'tail_start_idx': tail_start,
        'final_od': float(final_od),
        'max_od': float(max_od),
        'near_max': near_max,
    }

    return is_incomplete, metadata


def find_first_local_maximum(
    od600: np.ndarray,
    smoothed_od: np.ndarray,
    min_growth_threshold: float = 0.1,
    plateau_window: int = 10
) -> int:
    """
    Find the FIRST local maximum after exponential growth phase.

    This is more scientifically appropriate for Gompertz fitting because:
    1. Gompertz model assumes monotonic sigmoid growth
    2. First peak = end of primary growth phase (biologically meaningful)
    3. Avoids fitting noisy plateau or diauxic growth phases

    Parameters
    ----------
    od600 : np.ndarray
        Original OD600 values
    smoothed_od : np.ndarray
        Smoothed OD600 values
    min_growth_threshold : float
        Minimum OD600 value to consider as "growth started" (default: 0.1)
    plateau_window : int
        Number of consecutive points with decreasing/flat derivative to confirm plateau

    Returns
    -------
    int
        Index of the first local maximum
    """
    # Calculate derivative of smoothed data
    derivative = np.diff(smoothed_od)

    # Find where growth has meaningfully started (above threshold)
    # This avoids detecting noise in the lag phase as a "peak"
    growth_started_idx = 0
    for i, val in enumerate(smoothed_od):
        if val > min_growth_threshold:
            growth_started_idx = i
            break

    # Find the first point where derivative goes from positive to negative/zero
    # after growth has started
    first_peak_idx = None

    for i in range(growth_started_idx, len(derivative) - plateau_window):
        # Check if this is a local maximum:
        # - Current derivative is near zero or negative
        # - Previous derivatives were positive (we were growing)
        # - Following derivatives are negative or flat (entering plateau/decline)

        if i < 5:  # Need some history
            continue

        # Look at recent growth trend (was positive)
        recent_growth = np.mean(derivative[max(0, i-5):i])

        # Look at current and upcoming trend (should be flat or negative)
        upcoming_trend = np.mean(derivative[i:i+plateau_window])

        # Detect peak: was growing, now flattening/declining
        if recent_growth > 0.001 and upcoming_trend <= 0.005:
            # Verify this is a meaningful peak (not just noise)
            # The OD at this point should be at least 50% of max OD
            if smoothed_od[i] > 0.5 * np.max(smoothed_od):
                first_peak_idx = i
                break

    # Fallback to global max if no local peak found
    if first_peak_idx is None:
        first_peak_idx = np.argmax(smoothed_od)

    return first_peak_idx


def find_growth_start(
    time: np.ndarray,
    od600: np.ndarray,
    smoothing_window: int = 7,
    noise_threshold_multiplier: float = 3.0
) -> Tuple[int, Dict]:
    """
    Find where real growth begins, trimming noisy baseline data.

    Strategy:
    1. Calculate baseline noise from early data points
    2. Find where signal consistently exceeds baseline + noise threshold
    3. Also detect where growth rate first becomes consistently positive

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    smoothing_window : int
        Window for smoothing (default: 7)
    noise_threshold_multiplier : float
        How many std devs above baseline to consider "real signal" (default: 3.0)

    Returns
    -------
    Tuple[int, Dict]
        (start_index, metadata_dict)
    """
    # Smooth the data
    smoothed = pd.Series(od600).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    # Calculate baseline statistics from first 10% of data (lag phase)
    n_baseline = max(10, len(od600) // 10)
    baseline_mean = np.mean(smoothed[:n_baseline])
    baseline_std = np.std(smoothed[:n_baseline])

    # Threshold for "real growth" - above baseline noise
    growth_threshold = baseline_mean + noise_threshold_multiplier * baseline_std

    # Calculate growth rate
    dt = np.diff(time)
    dod = np.diff(smoothed)
    growth_rate = dod / dt
    growth_rate_smooth = pd.Series(growth_rate).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    # Find where growth rate becomes consistently positive
    # (indicates transition from lag to exponential phase)
    rate_start_idx = 0
    consecutive_positive = 0
    min_consecutive = 5

    for i in range(len(growth_rate_smooth)):
        if growth_rate_smooth[i] > 0:
            consecutive_positive += 1
            if consecutive_positive >= min_consecutive:
                rate_start_idx = max(0, i - min_consecutive + 1)
                break
        else:
            consecutive_positive = 0

    # Find where OD first exceeds the noise threshold
    od_start_idx = 0
    for i in range(len(smoothed)):
        if smoothed[i] > growth_threshold:
            od_start_idx = i
            break

    # Use the EARLIER of the two (to capture start of growth)
    # But add a small lookback buffer to include lag phase
    start_idx = min(rate_start_idx, od_start_idx)
    start_idx = max(0, start_idx - 5)  # Include a few points before growth starts

    # Don't trim too much - keep at least first 5% of data
    max_trim = len(od600) // 20
    start_idx = min(start_idx, max_trim)

    metadata = {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'growth_threshold': growth_threshold,
        'rate_start_idx': rate_start_idx,
        'od_start_idx': od_start_idx,
        'final_start_idx': start_idx,
    }

    return start_idx, metadata


def find_stationary_phase_start(
    time: np.ndarray,
    od600: np.ndarray,
    smoothing_window: int = 7
) -> Tuple[int, Dict]:
    """
    Find where the curve enters stationary phase using multiple criteria.

    Strategy:
    1. Find where curve reaches 90% of max OD (late transition phase)
    2. Also find where growth rate drops to <5% of max rate
    3. Take the LATER of these two (conservative approach)

    This ensures we capture most of the sigmoid growth curve without
    truncating too early during transition phase.

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    smoothing_window : int
        Window for smoothing (default: 7)

    Returns
    -------
    Tuple[int, Dict]
        (truncation_index, metadata_dict)
    """
    # Smooth the data
    smoothed = pd.Series(od600).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    # Calculate growth rate (dOD/dt)
    dt = np.diff(time)
    dod = np.diff(smoothed)
    growth_rate = dod / dt

    # Smooth the growth rate
    growth_rate_smooth = pd.Series(growth_rate).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    # Find maximum growth rate and its position
    max_rate = np.max(growth_rate_smooth)
    max_rate_idx = np.argmax(growth_rate_smooth)

    # Find max OD position
    max_od_idx = np.argmax(smoothed)
    max_od = smoothed[max_od_idx]

    # Criterion 1: Where curve reaches 90% of max OD
    threshold_90 = 0.90 * max_od
    reached_90_idx = max_od_idx  # default
    for i in range(max_rate_idx, len(smoothed)):
        if smoothed[i] >= threshold_90:
            reached_90_idx = i
            break

    # Criterion 2: Where growth rate drops to <5% of max rate
    # (more conservative than 10%)
    plateau_threshold = max_rate * 0.05
    rate_plateau_idx = max_od_idx  # default
    consecutive_low = 0
    min_plateau_points = 5

    for i in range(max_rate_idx, len(growth_rate_smooth)):
        if growth_rate_smooth[i] < plateau_threshold:
            consecutive_low += 1
            if consecutive_low >= min_plateau_points:
                rate_plateau_idx = i - min_plateau_points + 1
                break
        else:
            consecutive_low = 0

    # Take the LATER of the two criteria (conservative - capture more growth)
    plateau_start = max(reached_90_idx, rate_plateau_idx)

    # But don't go past the max OD point + small buffer
    plateau_start = min(plateau_start, max_od_idx + 5)

    # Add buffer to include some plateau region
    truncation_idx = min(plateau_start + 10, len(od600) - 1)

    metadata = {
        'max_rate': max_rate,
        'max_rate_idx': max_rate_idx,
        'max_rate_time': time[max_rate_idx] if max_rate_idx < len(time) else time[-1],
        'max_od_idx': max_od_idx,
        'max_od': max_od,
        'reached_90_idx': reached_90_idx,
        'rate_plateau_idx': rate_plateau_idx,
        'plateau_start_idx': plateau_start,
    }

    return truncation_idx, metadata


def find_optimal_truncation(
    time: np.ndarray,
    od600: np.ndarray,
    min_points: int = 20,
    trim_noisy_start: bool = True
) -> Tuple[int, int, float, Dict]:
    """
    Find the optimal truncation point using a multi-step approach:

    1. Optionally trim noisy data from the start (baseline noise)
    2. Use biological plateau detection to find candidate truncation region
    3. Search around that region for best Gompertz fit R²
    4. Validate that the fit captures the sigmoid shape properly

    This balances biological relevance (truncate at stationary phase)
    with statistical fit quality (good R²).

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    min_points : int
        Minimum points required for fitting (default: 20)
    trim_noisy_start : bool
        If True, also find and trim noisy baseline data from start (default: True)

    Returns
    -------
    Tuple[int, int, float, Dict]
        (start_index, end_index, best_r2, search_results)
    """
    n_points = len(time)

    # Step 0: Find where real growth begins (trim noisy start)
    if trim_noisy_start:
        start_idx, start_metadata = find_growth_start(time, od600)
    else:
        start_idx = 0
        start_metadata = {}

    # Step 1: Find biologically-motivated truncation point (stationary phase start)
    bio_truncation_idx, bio_metadata = find_stationary_phase_start(time, od600)

    # Step 2: Define search window around the biological truncation point
    # Search from 70% of bio_truncation to 120% of bio_truncation (or max OD + buffer)
    max_od_idx = bio_metadata['max_od_idx']

    search_start = max(start_idx + min_points, int(bio_truncation_idx * 0.70))
    search_end = min(
        n_points - 1,
        max(
            int(bio_truncation_idx * 1.20),
            max_od_idx + int(n_points * 0.05)  # At least to max OD + 5%
        )
    )

    # Ensure valid search range
    if search_end <= search_start:
        search_end = min(n_points - 1, search_start + 20)

    best_r2 = float('-inf')
    best_end_idx = bio_truncation_idx  # Default to biological estimate
    search_results = []

    # Search in steps to be efficient
    step = max(1, (search_end - search_start) // 30)

    for end_idx in range(search_start, search_end, step):
        # Use trimmed data (from start_idx to end_idx)
        t_trunc = time[start_idx:end_idx]
        od_trunc = od600[start_idx:end_idx]

        if len(t_trunc) < min_points:
            continue

        try:
            # Quick fit attempt
            a_init = np.max(od_trunc)
            diff = np.diff(od_trunc)
            dt = np.diff(t_trunc)
            growth_rates = diff / dt
            mu_init = max(np.max(growth_rates), 0.01)
            lambda_init = t_trunc[np.argmax(growth_rates)] if len(growth_rates) > 0 else t_trunc[0]

            popt, _ = curve_fit(
                gompertz_model,
                t_trunc, od_trunc,
                p0=[a_init, mu_init, lambda_init],
                bounds=([0.01, 0.001, 0], [3*a_init, 10*mu_init+1, t_trunc[-1]]),
                maxfev=1000
            )

            predicted = gompertz_model(t_trunc, *popt)
            ss_res = np.sum((od_trunc - predicted)**2)
            ss_tot = np.sum((od_trunc - np.mean(od_trunc))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            search_results.append({
                'idx': end_idx,
                'time': time[end_idx-1] if end_idx > 0 else time[0],
                'r2': r2,
                'a': popt[0],
                'mu': popt[1],
                'lambda': popt[2]
            })

            # Update best if this is better OR similar R² but closer to bio estimate
            # This balances fit quality with biological relevance
            dist_to_bio = abs(end_idx - bio_truncation_idx)
            best_dist_to_bio = abs(best_end_idx - bio_truncation_idx)

            if r2 > best_r2 + 0.005:
                # Clearly better R² - use it
                best_r2 = r2
                best_end_idx = end_idx
            elif r2 >= best_r2 - 0.005:
                # Similar R² - prefer point closer to biological estimate
                if dist_to_bio < best_dist_to_bio:
                    best_r2 = r2
                    best_end_idx = end_idx

        except Exception:
            continue

    # Fine-tune around the best point (search every point in a window)
    fine_start = max(search_start, best_end_idx - 10 * step)
    fine_end = min(search_end, best_end_idx + 10 * step)

    for end_idx in range(fine_start, fine_end):
        if any(r['idx'] == end_idx for r in search_results):
            continue  # Already tested

        t_trunc = time[start_idx:end_idx]
        od_trunc = od600[start_idx:end_idx]

        if len(t_trunc) < min_points:
            continue

        try:
            a_init = np.max(od_trunc)
            diff = np.diff(od_trunc)
            dt = np.diff(t_trunc)
            growth_rates = diff / dt
            mu_init = max(np.max(growth_rates), 0.01)
            lambda_init = t_trunc[np.argmax(growth_rates)] if len(growth_rates) > 0 else t_trunc[0]

            popt, _ = curve_fit(
                gompertz_model,
                t_trunc, od_trunc,
                p0=[a_init, mu_init, lambda_init],
                bounds=([0.01, 0.001, 0], [3*a_init, 10*mu_init+1, t_trunc[-1]]),
                maxfev=1000
            )

            predicted = gompertz_model(t_trunc, *popt)
            ss_res = np.sum((od_trunc - predicted)**2)
            ss_tot = np.sum((od_trunc - np.mean(od_trunc))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            # Same logic as above
            dist_to_bio = abs(end_idx - bio_truncation_idx)
            best_dist_to_bio = abs(best_end_idx - bio_truncation_idx)

            if r2 > best_r2 + 0.005:
                best_r2 = r2
                best_end_idx = end_idx
            elif r2 >= best_r2 - 0.005 and dist_to_bio < best_dist_to_bio:
                best_r2 = r2
                best_end_idx = end_idx

        except Exception:
            continue

    return start_idx, best_end_idx, best_r2, {
        'search_points': search_results,
        'bio_truncation_idx': bio_truncation_idx,
        'bio_metadata': bio_metadata,
        'start_metadata': start_metadata
    }


def truncate_at_max(
    time: np.ndarray,
    od600: np.ndarray,
    smoothing_window: int = 5,
    buffer_points: int = 3,
    use_first_peak: bool = True,
    use_adaptive: bool = False
) -> TruncationResult:
    """
    Truncate growth curve at maximum OD600 value.

    Uses rolling average smoothing to find the peak, avoiding noise spikes.

    By default, finds the FIRST local maximum (end of primary growth phase),
    which is more appropriate for Gompertz fitting of bacterial growth curves.

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    smoothing_window : int
        Window size for rolling average (default: 5)
    buffer_points : int
        Number of points to keep after max for fit stability (default: 3)
    use_first_peak : bool
        If True, truncate at first local maximum (recommended for Gompertz).
        If False, truncate at global maximum. (default: True)
    use_adaptive : bool
        If True, find optimal truncation point that maximizes R².
        This overrides use_first_peak. (default: False)

    Returns
    -------
    TruncationResult
        Contains truncated data and metadata
    """
    # Apply rolling average smoothing
    smoothed_od = pd.Series(od600).rolling(
        window=smoothing_window,
        center=True,
        min_periods=1
    ).mean().values

    # Default start index is 0 (no trimming)
    start_idx = 0

    # Check if curve is incomplete (still rising at experiment end)
    is_incomplete, incomplete_meta = detect_incomplete_curve(time, od600)

    if is_incomplete and not use_adaptive:
        # Curve never reached stationary phase — use all data (no truncation)
        # This prevents false negatives on truncation_challenge-type curves
        truncation_idx = len(od600) - 1
        actual_max_idx = np.argmax(od600)
    elif use_adaptive:
        # Find optimal truncation point that maximizes R² (also trims noisy start)
        start_idx, end_idx, best_r2, _ = find_optimal_truncation(time, od600)
        truncation_idx = end_idx
        actual_max_idx = np.argmax(od600[start_idx:truncation_idx]) + start_idx if truncation_idx > start_idx else start_idx
    elif use_first_peak:
        # Find FIRST local maximum (scientifically appropriate for Gompertz)
        max_idx_smoothed = find_first_local_maximum(od600, smoothed_od)

        # Find actual maximum within a small window around the detected peak
        search_start = max(0, max_idx_smoothed - smoothing_window)
        search_end = min(len(od600), max_idx_smoothed + smoothing_window + 1)
        local_max_offset = np.argmax(od600[search_start:search_end])
        actual_max_idx = search_start + local_max_offset

        # Calculate truncation index with buffer
        truncation_idx = min(actual_max_idx + buffer_points, len(od600) - 1)
    else:
        # Find global maximum (original behavior)
        max_idx_smoothed = np.argmax(smoothed_od)

        # Find actual maximum within a small window around the detected peak
        search_start = max(0, max_idx_smoothed - smoothing_window)
        search_end = min(len(od600), max_idx_smoothed + smoothing_window + 1)
        local_max_offset = np.argmax(od600[search_start:search_end])
        actual_max_idx = search_start + local_max_offset

        # Calculate truncation index with buffer
        truncation_idx = min(actual_max_idx + buffer_points, len(od600) - 1)

    # Create truncated arrays (from start_idx to truncation_idx)
    time_truncated = time[start_idx:truncation_idx + 1].copy()
    od_truncated = od600[start_idx:truncation_idx + 1].copy()

    return TruncationResult(
        truncation_index=truncation_idx,
        truncation_time=time[truncation_idx],
        max_od=od600[actual_max_idx],
        time_truncated=time_truncated,
        od_truncated=od_truncated,
        time_original=time.copy(),
        od_original=od600.copy(),
        smoothed_od=smoothed_od,
        start_index=start_idx  # New field for start trimming
    )


def validate_truncation(result: TruncationResult, min_points: int = 20) -> Tuple[bool, str]:
    """
    Validate that truncation result is usable.

    Parameters
    ----------
    result : TruncationResult
        Output from truncate_at_max
    min_points : int
        Minimum number of points required for fitting

    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason)
    """
    if len(result.od_truncated) < min_points:
        return False, f"Too few points after truncation: {len(result.od_truncated)} < {min_points}"

    # Check if max is too early (might indicate contamination or error)
    total_duration = result.time_original[-1] - result.time_original[0]
    time_to_max_ratio = result.truncation_time / total_duration if total_duration > 0 else 0

    if time_to_max_ratio < 0.05:  # Max within first 5% of experiment
        return False, f"Maximum too early in experiment: {time_to_max_ratio:.1%} of total duration"

    return True, "Truncation valid"


# =============================================================================
# Gompertz Model Fitting
# =============================================================================

def gompertz_model(t: np.ndarray, a: float, mu: float, lam: float) -> np.ndarray:
    """
    Modified Gompertz growth model.

    y(t) = a * exp(-exp((mu * e / a) * (lambda - t) + 1))

    Parameters
    ----------
    t : np.ndarray
        Time values
    a : float
        Maximum population (asymptotic value)
    mu : float
        Maximum specific growth rate
    lam : float
        Lag phase duration

    Returns
    -------
    np.ndarray
        Predicted OD600 values
    """
    return a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1))


def fit_gompertz(
    time: np.ndarray,
    od600: np.ndarray,
    max_iterations: int = 5000
) -> FitResult:
    """
    Fit Gompertz model to growth curve data.

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    max_iterations : int
        Maximum iterations for curve_fit

    Returns
    -------
    FitResult
        Contains optimized parameters, errors, and fit quality metrics
    """
    try:
        # Initial parameter estimates
        a_init = np.max(od600)

        # Estimate mu from maximum growth region
        diff = np.diff(od600)
        dt = np.diff(time)
        growth_rates = diff / dt
        mu_init = np.max(growth_rates) if len(growth_rates) > 0 else 0.1
        mu_init = max(mu_init, 0.01)  # Ensure positive

        # Estimate lambda (lag phase) - find where growth accelerates
        max_growth_idx = np.argmax(growth_rates)
        lambda_init = time[max_growth_idx] if max_growth_idx > 0 else time[0]

        p0 = [a_init, mu_init, lambda_init]

        # Parameter bounds
        bounds = (
            [0.01, 0.001, 0],                            # Lower bounds
            [3 * a_init, 10 * mu_init + 1, time[-1]]    # Upper bounds
        )

        # Fit the model
        popt, pcov = curve_fit(
            gompertz_model,
            time,
            od600,
            p0=p0,
            bounds=bounds,
            maxfev=max_iterations
        )

        a_opt, mu_opt, lambda_opt = popt

        # Calculate standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        a_err, mu_err, lambda_err = perr

        # Calculate fit quality metrics
        predicted = gompertz_model(time, *popt)
        residuals = od600 - predicted

        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((od600 - np.mean(od600))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return FitResult(
            success=True,
            a_opt=a_opt,
            mu_opt=mu_opt,
            lambda_opt=lambda_opt,
            a_err=a_err,
            mu_err=mu_err,
            lambda_err=lambda_err,
            mae=mae,
            mse=mse,
            rmse=rmse,
            r_squared=r_squared,
            predicted=predicted,
            residuals=residuals,
            error_message=""
        )

    except Exception as e:
        return FitResult(
            success=False,
            a_opt=0, mu_opt=0, lambda_opt=0,
            a_err=0, mu_err=0, lambda_err=0,
            mae=0, mse=0, rmse=0, r_squared=0,
            predicted=np.array([]),
            residuals=np.array([]),
            error_message=str(e)
        )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_truncation_comparison(
    truncation_result: TruncationResult,
    fit_result: FitResult,
    strain_name: str,
    output_dir: str
) -> str:
    """
    Generate before/after visualization of truncation and fitting.

    Creates a 2x2 subplot:
    - Top left: Original data with truncation point marked
    - Top right: Truncated data with Gompertz fit
    - Bottom left: Residuals plot
    - Bottom right: Fit quality metrics text box

    Parameters
    ----------
    truncation_result : TruncationResult
        Output from truncate_at_max
    fit_result : FitResult
        Output from fit_gompertz
    strain_name : str
        Name of the strain for title
    output_dir : str
        Directory to save output files

    Returns
    -------
    str
        Path to saved figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- Top Left: Original Data with Truncation Point ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(truncation_result.time_original, truncation_result.od_original,
             'b-', alpha=0.5, linewidth=1, label='Original data')
    ax1.plot(truncation_result.time_original, truncation_result.smoothed_od,
             'g--', alpha=0.7, linewidth=1.5, label='Smoothed')
    ax1.axvline(x=truncation_result.truncation_time, color='r', linestyle='--',
                linewidth=2, label=f'Truncation (t={truncation_result.truncation_time:.1f}h)')
    ax1.scatter([truncation_result.truncation_time], [truncation_result.max_od],
                color='r', s=100, zorder=5, marker='*',
                label=f'Max OD={truncation_result.max_od:.3f}')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('OD600')
    ax1.set_title(f'{strain_name} - Original Data')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Top Right: Truncated Data with Fit ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(truncation_result.time_truncated, truncation_result.od_truncated,
                c='blue', s=10, alpha=0.5, label='Truncated data')
    if fit_result.success:
        # Generate smooth fit curve for plotting
        t_smooth = np.linspace(truncation_result.time_truncated[0],
                               truncation_result.time_truncated[-1], 200)
        od_fit_smooth = gompertz_model(t_smooth, fit_result.a_opt,
                                        fit_result.mu_opt, fit_result.lambda_opt)
        ax2.plot(t_smooth, od_fit_smooth, 'r-', linewidth=2, label='Gompertz fit')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('OD600')
    ax2.set_title(f'{strain_name} - Truncated + Fit')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Bottom Left: Residuals ---
    ax3 = fig.add_subplot(gs[1, 0])
    if fit_result.success and len(fit_result.residuals) > 0:
        ax3.scatter(truncation_result.time_truncated, fit_result.residuals,
                    c='blue', s=10, alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals Analysis')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Fit Failed', ha='center', va='center', fontsize=14)
        ax3.set_title('Residuals Analysis')

    # --- Bottom Right: Metrics Text Box ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    if fit_result.success:
        metrics_text = f"""Gompertz Fit Parameters
========================
A (max OD600):    {fit_result.a_opt:.4f} +/- {fit_result.a_err:.4f}
mu (growth rate): {fit_result.mu_opt:.4f} +/- {fit_result.mu_err:.4f} OD/h
lambda (lag):     {fit_result.lambda_opt:.2f} +/- {fit_result.lambda_err:.2f} h

Fit Quality Metrics
====================
R-squared:        {fit_result.r_squared:.4f}
RMSE:             {fit_result.rmse:.4f}
MAE:              {fit_result.mae:.4f}

Truncation Info
================
Points used:      {len(truncation_result.od_truncated)}
Truncation time:  {truncation_result.truncation_time:.1f} h
Max OD600:        {truncation_result.max_od:.4f}"""
    else:
        metrics_text = f"""Fit Failed
===========
Error: {fit_result.error_message}

Truncation Info
================
Points used:      {len(truncation_result.od_truncated)}
Truncation time:  {truncation_result.truncation_time:.1f} h
Max OD600:        {truncation_result.max_od:.4f}"""

    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Growth Curve Analysis: {strain_name}', fontsize=14, fontweight='bold')

    # Save figure
    output_path = f"{output_dir}/{strain_name}_truncation_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_bad_curve(
    time: np.ndarray,
    od600: np.ndarray,
    classification: 'ClassificationResult',
    strain_name: str,
    output_dir: str
) -> str:
    """
    Generate visualization for curves classified as "bad" (not worth fitting).

    This helps validate that the classification is correct.

    Parameters
    ----------
    time : np.ndarray
        Time values in hours
    od600 : np.ndarray
        OD600 absorbance values
    classification : ClassificationResult
        Classification result with metrics
    strain_name : str
        Name of the strain for title
    output_dir : str
        Directory to save output files

    Returns
    -------
    str
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Raw growth curve ---
    ax1 = axes[0]
    ax1.plot(time, od600, 'b-', linewidth=1, alpha=0.7, label='OD600')
    ax1.scatter(time, od600, c='blue', s=5, alpha=0.3)
    ax1.axhline(y=classification.metrics['max_od'], color='r', linestyle='--',
                alpha=0.5, label=f"Max OD = {classification.metrics['max_od']:.3f}")
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('OD600')
    ax1.set_title(f'{strain_name} - Raw Data')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right: Classification info ---
    ax2 = axes[1]
    ax2.axis('off')

    # Create text with classification details
    metrics = classification.metrics
    reason_lines = classification.reason.replace("BAD: ", "").split("; ")
    reason_text = "\n".join([f"  - {r}" for r in reason_lines])

    info_text = f"""CLASSIFICATION: BAD (Not suitable for Gompertz fitting)

Reasons:
{reason_text}

Metrics:
  Initial OD:     {metrics['initial_od']:.4f}
  Maximum OD:     {metrics['max_od']:.4f}
  Delta OD:       {metrics['delta_od']:.4f}
  SNR:            {metrics['snr']:.1f}
  Time to max:    {metrics['time_to_max']:.1f} h

Thresholds:
  Min delta OD:   {CLASSIFICATION_THRESHOLDS['min_delta_od']}
  Min max OD:     {CLASSIFICATION_THRESHOLDS['min_max_od']}
  Min SNR:        {CLASSIFICATION_THRESHOLDS['min_snr']}
  Max initial OD: {CLASSIFICATION_THRESHOLDS['max_initial_od']}"""

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'BAD Classification: {strain_name}', fontsize=14,
                 fontweight='bold', color='red')
    plt.tight_layout()

    # Save figure
    output_path = f"{output_dir}/{strain_name}_BAD_classification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_bad_curve_with_fit(
    time: np.ndarray,
    od600: np.ndarray,
    truncation: 'TruncationResult',
    fit: 'FitResult',
    classification: 'ClassificationResult',
    strain_name: str,
    output_dir: str
) -> str:
    """
    Generate visualization for curves that were fit but classified as bad.

    Shows the attempted fit so users can see WHY it failed.

    Parameters
    ----------
    time : np.ndarray
        Original time values
    od600 : np.ndarray
        Original OD600 values
    truncation : TruncationResult
        Truncation result
    fit : FitResult
        Fit result (may have failed or poor quality)
    classification : ClassificationResult
        Classification result with fit quality metrics
    strain_name : str
        Name of the strain
    output_dir : str
        Directory to save output

    Returns
    -------
    str
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Left: Original data with truncation ---
    ax1 = axes[0]
    ax1.plot(time, od600, 'b-', linewidth=1, alpha=0.7, label='Original')
    ax1.axvline(x=truncation.truncation_time, color='r', linestyle='--',
                alpha=0.7, label=f'Truncation @ {truncation.truncation_time:.1f}h')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('OD600')
    ax1.set_title(f'{strain_name} - Original Data')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Middle: Truncated data with fit attempt ---
    ax2 = axes[1]
    ax2.scatter(truncation.time_truncated, truncation.od_truncated,
                c='blue', s=15, alpha=0.5, label='Data')
    if fit.success and len(fit.predicted) > 0:
        # Plot the fit curve
        t_smooth = np.linspace(truncation.time_truncated[0],
                               truncation.time_truncated[-1], 200)
        od_fit = gompertz_model(t_smooth, fit.a_opt, fit.mu_opt, fit.lambda_opt)
        ax2.plot(t_smooth, od_fit, 'r-', linewidth=2, alpha=0.7,
                 label=f'Gompertz (R²={fit.r_squared:.3f})')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('OD600')
    ax2.set_title('Truncated + Fit Attempt')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Right: Classification details ---
    ax3 = axes[2]
    ax3.axis('off')

    metrics = classification.metrics
    reason_text = classification.reason.replace("BAD: ", "")

    if fit.success:
        info_text = f"""CLASSIFICATION: BAD (Fit quality insufficient)

Reason: {reason_text}

Fit Results:
  A (max OD):     {fit.a_opt:.4f} ± {fit.a_err:.4f} ({metrics.get('a_err_pct', 0):.1f}%)
  mu (growth):    {fit.mu_opt:.4f} ± {fit.mu_err:.4f} ({metrics.get('mu_err_pct', 0):.1f}%)
  lambda (lag):   {fit.lambda_opt:.2f} ± {fit.lambda_err:.2f}

Quality Metrics:
  R²:             {fit.r_squared:.4f}
  RMSE:           {fit.rmse:.4f}

Thresholds:
  Min R²:         {FIT_QUALITY_THRESHOLDS['min_r_squared']}
  Max param err:  {FIT_QUALITY_THRESHOLDS['max_param_error_pct']}%

Data Info:
  Delta OD:       {metrics.get('delta_od', 0):.4f}
  Max OD:         {metrics.get('max_od', 0):.4f}"""
    else:
        info_text = f"""CLASSIFICATION: BAD (Fit failed)

Reason: {reason_text}

Error: {fit.error_message}

Data Info:
  Delta OD:       {metrics.get('delta_od', 0):.4f}
  Max OD:         {metrics.get('max_od', 0):.4f}"""

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'BAD Classification (Fit-Based): {strain_name}', fontsize=14,
                 fontweight='bold', color='red')
    plt.tight_layout()

    # Save figure
    output_path = f"{output_dir}/{strain_name}_BAD_fit_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_classification_summary(
    results: Dict[str, ClassificationResult],
    output_dir: str
) -> str:
    """
    Generate summary visualization of all classification results.

    Parameters
    ----------
    results : Dict[str, ClassificationResult]
        Dictionary mapping strain names to classification results
    output_dir : str
        Directory to save output

    Returns
    -------
    str
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract metrics for plotting (use .get() to handle both classification modes)
    strains = list(results.keys())
    delta_ods = [r.metrics.get('delta_od', 0) or 0 for r in results.values()]
    max_ods = [r.metrics.get('max_od', 0) or 0 for r in results.values()]
    # For fit-based, show R² instead of SNR
    r_squared = [r.metrics.get('r_squared') for r in results.values()]
    snrs = [r.metrics.get('snr') for r in results.values()]
    is_good = [r.is_good for r in results.values()]
    colors = ['green' if g else 'red' for g in is_good]

    # Determine if we're using fit-based classification
    use_fit_based = r_squared[0] is not None if r_squared else False

    # Delta OD bar chart
    ax1 = axes[0, 0]
    ax1.bar(range(len(strains)), delta_ods, color=colors, alpha=0.7)
    ax1.axhline(y=CLASSIFICATION_THRESHOLDS['min_delta_od'], color='orange',
                linestyle='--', linewidth=2, label=f"Threshold ({CLASSIFICATION_THRESHOLDS['min_delta_od']})")
    ax1.set_xticks(range(len(strains)))
    ax1.set_xticklabels(strains, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Delta OD600')
    ax1.set_title('OD600 Change by Strain')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Max OD bar chart
    ax2 = axes[0, 1]
    ax2.bar(range(len(strains)), max_ods, color=colors, alpha=0.7)
    ax2.axhline(y=CLASSIFICATION_THRESHOLDS['min_max_od'], color='orange',
                linestyle='--', linewidth=2, label=f"Threshold ({CLASSIFICATION_THRESHOLDS['min_max_od']})")
    ax2.set_xticks(range(len(strains)))
    ax2.set_xticklabels(strains, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Maximum OD600')
    ax2.set_title('Maximum OD600 by Strain')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Third panel: R² (fit-based) or SNR (OD-based)
    ax3 = axes[1, 0]
    if use_fit_based:
        # Show R² for fit-based classification
        r2_values = [max(0, r or 0) for r in r_squared]  # Clamp negative R² to 0 for display
        ax3.bar(range(len(strains)), r2_values, color=colors, alpha=0.7)
        ax3.axhline(y=FIT_QUALITY_THRESHOLDS['min_r_squared'], color='orange',
                    linestyle='--', linewidth=2, label=f"Threshold ({FIT_QUALITY_THRESHOLDS['min_r_squared']})")
        ax3.set_ylabel('R² (Fit Quality)')
        ax3.set_title('Gompertz Fit R² by Strain')
        ax3.set_ylim(0, 1.05)
    else:
        # Show SNR for OD-based classification
        snrs_capped = [min(s or 0, 100) for s in snrs]
        ax3.bar(range(len(strains)), snrs_capped, color=colors, alpha=0.7)
        ax3.axhline(y=CLASSIFICATION_THRESHOLDS['min_snr'], color='orange',
                    linestyle='--', linewidth=2, label=f"Threshold ({CLASSIFICATION_THRESHOLDS['min_snr']})")
        ax3.set_ylabel('Signal-to-Noise Ratio')
        ax3.set_title('SNR by Strain')
    ax3.set_xticks(range(len(strains)))
    ax3.set_xticklabels(strains, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Pie chart of good vs bad
    ax4 = axes[1, 1]
    good_count = sum(is_good)
    bad_count = len(is_good) - good_count
    if good_count > 0 or bad_count > 0:
        ax4.pie([good_count, bad_count], labels=['Good', 'Bad'],
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Classification Summary\n({good_count} good, {bad_count} bad)')

    plt.tight_layout()
    output_path = f"{output_dir}/classification_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def process_growth_curves(
    data_dir: str,
    output_dir: str,
    classification_thresholds: Optional[Dict] = None,
    truncation_params: Optional[Dict] = None,
    fit_quality_thresholds: Optional[Dict] = None,
    use_fit_based_classification: bool = True,
    verbose: bool = True
) -> Dict[str, ProcessingResult]:
    """
    Main pipeline function to process all growth curves in a directory.

    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files
    output_dir : str
        Path for output files and plots
    classification_thresholds : Optional[Dict]
        Custom classification thresholds (for OD-based classification)
    truncation_params : Optional[Dict]
        Custom truncation parameters
    fit_quality_thresholds : Optional[Dict]
        Custom fit quality thresholds (for fit-based classification)
    use_fit_based_classification : bool
        If True (default), classify based on fit quality (R², parameter errors).
        If False, use OD-threshold-based classification.
    verbose : bool
        Print progress messages

    Returns
    -------
    Dict[str, ProcessingResult]
        Results for each processed strain
    """
    # Use defaults if not provided
    if classification_thresholds is None:
        classification_thresholds = CLASSIFICATION_THRESHOLDS
    if truncation_params is None:
        truncation_params = TRUNCATION_PARAMS
    if fit_quality_thresholds is None:
        fit_quality_thresholds = FIT_QUALITY_THRESHOLDS

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    results = {}

    # Find all CSV files
    csv_files = list(Path(data_dir).glob("*_DATA.csv"))

    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for csv_file in csv_files:
        if verbose:
            print(f"\nProcessing: {csv_file.name}")

        # Load data
        df = pd.read_csv(csv_file)

        # Extract time column
        time_cols = [c for c in df.columns if 'TIME' in c.upper()]
        if not time_cols:
            print(f"  Warning: No TIME column found in {csv_file.name}, skipping")
            continue

        time_col = time_cols[0]
        time = df[time_col].values

        # Process each strain column
        strain_cols = [c for c in df.columns if c != time_col]

        for strain_col in strain_cols:
            strain_name = strain_col.replace('_blanked', '').replace('_', '-')
            od600 = df[strain_col].values

            # Clean data: remove NaN
            valid_mask = ~np.isnan(od600)
            time_clean = time[valid_mask]
            od600_clean = od600[valid_mask]

            if len(od600_clean) < fit_quality_thresholds.get('min_points_for_fit', 15):
                if verbose:
                    print(f"  {strain_name}: Insufficient data points, skipping")
                continue

            # =================================================================
            # FIT-FIRST APPROACH (New, more rigorous)
            # =================================================================
            if use_fit_based_classification:

                # Pre-fit noise filter: skip fitting if signal is buried in noise
                n_baseline = min(15, len(od600_clean) // 5)
                noise_std = float(np.std(od600_clean[:n_baseline])) if n_baseline > 2 else 1e-8
                signal_range = float(np.max(od600_clean) - np.mean(od600_clean[:min(5, len(od600_clean))]))
                pre_fit_snr = signal_range / max(noise_std, 1e-8)

                if pre_fit_snr < 5.0 and signal_range < 0.5:
                    # Signal is buried in noise AND growth is small -- skip fitting, classify BAD
                    # (If signal_range >= 0.5 we let the fitter decide; large growth overrides low SNR)
                    # Raised from 3.0 to 5.0 to catch high_noise and borderline_noise FPs
                    classification = ClassificationResult(
                        is_good=False,
                        reason=f"BAD: Pre-fit SNR too low ({pre_fit_snr:.1f} < 3.0)",
                        metrics={'snr': pre_fit_snr, 'delta_od': signal_range,
                                 'max_od': float(np.max(od600_clean))}
                    )
                    truncation = TruncationResult(
                        truncation_index=len(od600_clean)-1,
                        truncation_time=float(time_clean[-1]),
                        max_od=float(np.max(od600_clean)),
                        time_truncated=time_clean, od_truncated=od600_clean,
                        time_original=time_clean, od_original=od600_clean,
                        smoothed_od=od600_clean
                    )
                    fit = FitResult(
                        success=False, a_opt=0, mu_opt=0, lambda_opt=0,
                        a_err=0, mu_err=0, lambda_err=0,
                        mae=0, mse=0, rmse=0, r_squared=0,
                        predicted=np.array([]), residuals=np.array([]),
                        error_message=f"Skipped: pre-fit SNR={pre_fit_snr:.1f}"
                    )

                    if verbose:
                        print(f"  {strain_name}: BAD (pre-fit SNR={pre_fit_snr:.1f})")

                    plot_bad_curve_with_fit(
                        time_clean, od600_clean,
                        truncation, fit,
                        classification, strain_name,
                        plots_dir
                    )

                    results[strain_name] = ProcessingResult(
                        strain_name=strain_name,
                        classification=classification,
                        truncation=truncation,
                        fit=fit
                    )
                    continue

                # Step 1: Truncate (using specified method)
                truncation = truncate_at_max(
                    time_clean, od600_clean,
                    smoothing_window=truncation_params['smoothing_window'],
                    buffer_points=truncation_params['buffer_points'],
                    use_first_peak=truncation_params.get('use_first_peak', True),
                    use_adaptive=truncation_params.get('use_adaptive', False)
                )

                # Step 2: Attempt Gompertz fit
                fit = fit_gompertz(
                    truncation.time_truncated,
                    truncation.od_truncated
                )

                # Step 2b: If Gompertz fit is poor, retry with adaptive truncation
                if fit.r_squared < 0.90 and not truncation_params.get('use_adaptive', False):
                    adaptive_trunc = truncate_at_max(
                        time_clean, od600_clean,
                        smoothing_window=truncation_params['smoothing_window'],
                        buffer_points=truncation_params['buffer_points'],
                        use_first_peak=False,
                        use_adaptive=True
                    )
                    adaptive_fit = fit_gompertz(
                        adaptive_trunc.time_truncated,
                        adaptive_trunc.od_truncated
                    )
                    if adaptive_fit.r_squared > fit.r_squared:
                        truncation = adaptive_trunc
                        fit = adaptive_fit

                # Step 3: Classify based on fit quality (with secondary quality gates)
                classification = classify_by_fit_quality(
                    fit, fit_quality_thresholds,
                    time=time_clean, od600=od600_clean
                )

                # Add basic OD metrics to classification for reference
                if 'delta_od' not in classification.metrics:
                    classification.metrics['delta_od'] = np.max(od600_clean) - np.mean(od600_clean[:5])
                classification.metrics['max_od'] = np.max(od600_clean)

                if verbose:
                    status = "GOOD" if classification.is_good else "BAD"
                    r2 = fit.r_squared if fit.success else 0
                    print(f"  {strain_name}: {status} (R²={r2:.3f}, A={fit.a_opt:.3f})")
                    if not classification.is_good:
                        print(f"    Reason: {classification.reason}")

                # Step 4: Generate visualization
                if classification.is_good:
                    plot_truncation_comparison(
                        truncation, fit, strain_name,
                        plots_dir
                    )
                else:
                    # Plot bad curve with fit attempt shown
                    plot_bad_curve_with_fit(
                        time_clean, od600_clean,
                        truncation, fit,
                        classification, strain_name,
                        plots_dir
                    )

            # =================================================================
            # OD-THRESHOLD APPROACH (Original)
            # =================================================================
            else:
                # Step 1: Classify by OD thresholds
                classification = classify_growth_curve(
                    time_clean, od600_clean,
                    thresholds=classification_thresholds
                )

                if verbose:
                    status = "GOOD" if classification.is_good else "BAD"
                    print(f"  {strain_name}: {status} (delta={classification.metrics['delta_od']:.3f}, max={classification.metrics['max_od']:.3f})")

                truncation = None
                fit = None

                if classification.is_good:
                    # Step 2: Truncate
                    truncation = truncate_at_max(
                        time_clean, od600_clean,
                        smoothing_window=truncation_params['smoothing_window'],
                        buffer_points=truncation_params['buffer_points'],
                        use_first_peak=truncation_params.get('use_first_peak', True),
                        use_adaptive=truncation_params.get('use_adaptive', False)
                    )

                    # Validate truncation
                    is_valid, reason = validate_truncation(
                        truncation,
                        min_points=truncation_params['min_points_for_fit']
                    )

                    if is_valid:
                        # Step 3: Fit Gompertz model
                        fit = fit_gompertz(
                            truncation.time_truncated,
                            truncation.od_truncated
                        )

                        if verbose:
                            if fit.success:
                                print(f"    Fit: a={fit.a_opt:.3f}, mu={fit.mu_opt:.3f}, lambda={fit.lambda_opt:.2f}, R2={fit.r_squared:.3f}")
                            else:
                                print(f"    Fit failed: {fit.error_message}")

                        # Step 4: Generate visualization
                        plot_truncation_comparison(
                            truncation, fit, strain_name,
                            plots_dir
                        )
                    else:
                        if verbose:
                            print(f"    Truncation invalid: {reason}")

                # Generate visualization for BAD curves too (for validation)
                if not classification.is_good:
                    plot_bad_curve(
                        time_clean, od600_clean,
                        classification, strain_name,
                        plots_dir
                    )

            results[strain_name] = ProcessingResult(
                strain_name=strain_name,
                classification=classification,
                truncation=truncation,
                fit=fit
            )

    # Generate summary visualization
    if results:
        classifications = {k: v.classification for k, v in results.items()}
        plot_classification_summary(classifications, output_dir)

    # Export results to CSV
    export_results_to_csv(results, output_dir)

    return results


def export_results_to_csv(results: Dict[str, ProcessingResult], output_dir: str) -> str:
    """
    Export all results to a summary CSV file.

    Parameters
    ----------
    results : Dict[str, ProcessingResult]
        Processing results for all strains
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to exported CSV
    """
    rows = []
    for strain_name, result in results.items():
        metrics = result.classification.metrics
        row = {
            'strain': strain_name,
            'is_good': result.classification.is_good,
            'classification_reason': result.classification.reason,
            # Use .get() for metrics that may not exist in fit-based classification
            'delta_od': metrics.get('delta_od'),
            'max_od': metrics.get('max_od'),
            'initial_od': metrics.get('initial_od'),
            'snr': metrics.get('snr'),
            'time_to_max': metrics.get('time_to_max'),
            # Add fit quality metrics (may not exist in OD-based)
            'r_squared': metrics.get('r_squared'),
            'a_err_pct': metrics.get('a_err_pct'),
            'mu_err_pct': metrics.get('mu_err_pct'),
        }

        if result.truncation:
            row.update({
                'truncation_time': result.truncation.truncation_time,
                'points_used': len(result.truncation.od_truncated),
            })
        else:
            row.update({
                'truncation_time': None,
                'points_used': None,
            })

        if result.fit and result.fit.success:
            row.update({
                'gompertz_a': result.fit.a_opt,
                'gompertz_a_err': result.fit.a_err,
                'gompertz_mu': result.fit.mu_opt,
                'gompertz_mu_err': result.fit.mu_err,
                'gompertz_lambda': result.fit.lambda_opt,
                'gompertz_lambda_err': result.fit.lambda_err,
                'fit_r_squared': result.fit.r_squared,
                'fit_rmse': result.fit.rmse,
                'fit_mae': result.fit.mae,
            })
        else:
            row.update({
                'gompertz_a': None,
                'gompertz_a_err': None,
                'gompertz_mu': None,
                'gompertz_mu_err': None,
                'gompertz_lambda': None,
                'gompertz_lambda_err': None,
                'fit_r_squared': None,
                'fit_rmse': None,
                'fit_mae': None,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = f"{output_dir}/processing_results.csv"
    df.to_csv(output_path, index=False)

    return output_path


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process bacterial growth curves: classify, truncate at max, and fit Gompertz model.'
    )
    parser.add_argument(
        'data_dir',
        help='Directory containing CSV data files'
    )
    parser.add_argument(
        '-o', '--output',
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    parser.add_argument(
        '--min-delta-od',
        type=float,
        default=0.3,
        help='Minimum OD600 change for good classification (default: 0.3)'
    )
    parser.add_argument(
        '--min-max-od',
        type=float,
        default=0.5,
        help='Minimum peak OD600 for good classification (default: 0.5)'
    )
    parser.add_argument(
        '--min-snr',
        type=float,
        default=5.0,
        help='Minimum signal-to-noise ratio (default: 5.0)'
    )
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=5,
        help='Rolling average window for peak detection (default: 5)'
    )
    parser.add_argument(
        '--buffer-points',
        type=int,
        default=3,
        help='Points to keep after max for fit stability (default: 3)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--global-max',
        action='store_true',
        help='Truncate at global maximum instead of first local maximum (default: first peak)'
    )
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Use adaptive truncation that optimizes for best R² (overrides --global-max)'
    )
    parser.add_argument(
        '--od-based',
        action='store_true',
        help='Use OD-threshold-based classification instead of fit-quality-based (default: fit-based)'
    )
    parser.add_argument(
        '--min-r2',
        type=float,
        default=0.95,
        help='Minimum R² for good fit classification (default: 0.95)'
    )
    parser.add_argument(
        '--max-param-error',
        type=float,
        default=20.0,
        help='Maximum parameter error %% for good fit (default: 20.0)'
    )

    args = parser.parse_args()

    # Build threshold dict for OD-based classification
    thresholds = {
        'min_delta_od': args.min_delta_od,
        'min_max_od': args.min_max_od,
        'min_snr': args.min_snr,
        'max_initial_od': 0.15,
    }

    # Build threshold dict for fit-based classification
    fit_thresholds = {
        'min_r_squared': args.min_r2,
        'max_param_error_pct': args.max_param_error,
        'min_points_for_fit': 15,
    }

    truncation_params = {
        'smoothing_window': args.smoothing_window,
        'buffer_points': args.buffer_points,
        'min_points_for_fit': 20,
        'use_first_peak': not args.global_max,  # Default: use first peak
        'use_adaptive': args.adaptive,  # Adaptive overrides other truncation methods
    }

    # Run pipeline
    use_fit_based = not args.od_based
    if use_fit_based:
        print("Using FIT-BASED classification (R², parameter errors)")
    else:
        print("Using OD-THRESHOLD classification (delta_od, max_od, snr)")

    if args.adaptive:
        print("Using ADAPTIVE truncation (optimizes R²)")
    elif args.global_max:
        print("Using GLOBAL MAX truncation")
    else:
        print("Using FIRST PEAK truncation")

    results = process_growth_curves(
        data_dir=args.data_dir,
        output_dir=args.output,
        classification_thresholds=thresholds,
        truncation_params=truncation_params,
        fit_quality_thresholds=fit_thresholds,
        use_fit_based_classification=use_fit_based,
        verbose=not args.quiet
    )

    # Print summary
    good_count = sum(1 for r in results.values() if r.classification.is_good)
    fit_count = sum(1 for r in results.values() if r.fit and r.fit.success)

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total curves processed: {len(results)}")
    print(f"Good curves: {good_count}")
    print(f"Bad curves: {len(results) - good_count}")
    print(f"Successful fits: {fit_count}")
    print(f"\nResults saved to: {args.output}/")
    print(f"  - processing_results.csv")
    print(f"  - classification_summary.png")
    print(f"  - plots/ (individual strain plots)")


if __name__ == "__main__":
    main()
