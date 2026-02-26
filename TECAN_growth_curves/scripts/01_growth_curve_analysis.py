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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass
class TruncationCandidate:
    """A single evaluated truncation endpoint."""
    end_idx: int
    end_time: float
    n_points: int
    cv_score: float        # mean CV R² (higher = better)
    cv_std: float          # std of CV R² across folds
    raw_r2: float          # full-data R² for reference
    model_name: str
    params: Dict[str, float]


@dataclass
class TruncationLandscape:
    """Rich result from find_optimal_truncation_v2()."""
    best_start_idx: int
    best_end_idx: int
    best_end_time: float
    best_cv_score: float
    best_model: str
    candidates: List[TruncationCandidate]
    confidence: float      # derived from landscape shape
    bio_estimate_idx: int  # biological estimate for reference


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
    od600: Optional[np.ndarray] = None,
    is_incomplete: bool = False
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
    # The R² bypass skips SNR/delta-OD-CI checks when the fit is good
    # enough that the model itself proves real growth. Kept at 0.98 to
    # avoid reclassifying strains with small but real growth signals
    # (e.g. H2O controls, pesticide-only with low delta-OD).
    # Added: residual autocorrelation check and monotonicity gate (these
    # target noise/non-growth specifically without hurting real low-OD growth).
    # =====================================================================
    if od600 is not None and len(od600) > 10:
        n_baseline = min(10, len(od600) // 5)
        baseline = od600[:n_baseline]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline)) if float(np.std(baseline)) > 1e-8 else 1e-8
        max_od = float(np.max(od600))
        delta_od = max_od - baseline_mean

        snr = (max_od - baseline_mean) / baseline_std

        # Bypass noise-based gates when fit quality is excellent
        excellent_threshold = thresholds.get('excellent_r2_threshold', 0.98)
        fit_is_excellent = fit_result.r_squared >= excellent_threshold

        # Always compute delta-OD CI (needed for ML features even when bypassed)
        delta_od_ci_lower = delta_od - 2 * baseline_std

        if not fit_is_excellent:
            # SNR filter: signal must meaningfully exceed noise floor
            min_snr = thresholds.get('min_snr', 5.0)
            if snr < min_snr:
                reasons.append(f"Low SNR: {snr:.1f} < {min_snr}")

            # Delta-OD confidence interval: growth must be statistically significant
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
            if lag1_autocorr > max_autocorr and not fit_is_excellent and not is_incomplete:
                reasons.append(
                    f"High residual autocorrelation: {lag1_autocorr:.2f} > {max_autocorr}"
                )
            metrics['residual_autocorr'] = lag1_autocorr

        metrics['snr'] = snr
        metrics['delta_od'] = delta_od
        metrics['baseline_std'] = baseline_std
        metrics['monotone_fraction'] = monotone_frac
        metrics['delta_od_ci_lower'] = delta_od_ci_lower

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


# =============================================================================
# Monte Carlo Cross-Validated Truncation Optimization
# =============================================================================

def mccv_score(
    time: np.ndarray,
    od600: np.ndarray,
    model_func=None,
    p0_func=None,
    n_folds: int = 20,
    holdout_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """
    Compute Monte Carlo Cross-Validation score for a data segment.

    Randomly holds out a fraction of data points, fits the model on the rest,
    and evaluates prediction error on the holdout. Repeats n_folds times.

    Parameters
    ----------
    time, od600 : np.ndarray
        Data segment to evaluate
    model_func : callable
        Growth model function (default: gompertz_model)
    p0_func : callable
        Function that returns (p0, bounds) given (time, od) for the model
    n_folds : int
        Number of random splits (default: 20)
    holdout_fraction : float
        Fraction of data to hold out per fold (default: 0.2)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[float, float, float, Optional[np.ndarray]]
        (mean_cv_r2, std_cv_r2, raw_r2, best_params)
    """
    if model_func is None:
        model_func = gompertz_model
    if p0_func is None:
        p0_func = _gompertz_p0

    n = len(time)
    n_holdout = max(2, int(n * holdout_fraction))
    n_train = n - n_holdout

    if n_train < 10:
        return -1.0, 1.0, 0.0, None

    rng = np.random.default_rng(seed)
    cv_r2s = []
    best_full_params = None

    # Compute raw R² on full data (for reference)
    try:
        p0, bounds = p0_func(time, od600)
        popt_full, _ = curve_fit(model_func, time, od600, p0=p0,
                                 bounds=bounds, maxfev=1000)
        pred_full = model_func(time, *popt_full)
        ss_res_full = np.sum((od600 - pred_full) ** 2)
        ss_tot_full = np.sum((od600 - np.mean(od600)) ** 2)
        raw_r2 = 1 - ss_res_full / ss_tot_full if ss_tot_full > 0 else 0.0
        best_full_params = popt_full
    except Exception:
        raw_r2 = 0.0

    for _ in range(n_folds):
        # Random 80/20 split (preserving time ordering in both sets)
        indices = np.arange(n)
        holdout_idx = np.sort(rng.choice(indices, size=n_holdout, replace=False))
        train_mask = np.ones(n, dtype=bool)
        train_mask[holdout_idx] = False

        t_train = time[train_mask]
        od_train = od600[train_mask]
        t_test = time[holdout_idx]
        od_test = od600[holdout_idx]

        try:
            p0, bounds = p0_func(t_train, od_train)
            popt, _ = curve_fit(model_func, t_train, od_train, p0=p0,
                                bounds=bounds, maxfev=1000)

            # Predict on holdout
            pred_test = model_func(t_test, *popt)
            if np.any(np.isnan(pred_test)) or np.any(np.isinf(pred_test)):
                continue

            # Compute test R² (relative to test set mean)
            ss_res = np.sum((od_test - pred_test) ** 2)
            ss_tot = np.sum((od_test - np.mean(od_test)) ** 2)
            test_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            cv_r2s.append(test_r2)
        except Exception:
            continue

    if len(cv_r2s) < 3:
        return -1.0, 1.0, raw_r2, best_full_params

    return float(np.mean(cv_r2s)), float(np.std(cv_r2s)), raw_r2, best_full_params


def _gompertz_p0(time, od600):
    """Return initial guesses and bounds for Gompertz curve_fit."""
    a_init = max(float(np.max(od600)), 0.05)
    diff = np.diff(od600)
    dt = np.diff(time)
    rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(rates)), 0.01)
    max_rate_idx = int(np.argmax(rates))
    lam_init = float(time[max_rate_idx]) if max_rate_idx > 0 else float(time[0])
    p0 = [a_init, mu_init, lam_init]
    bounds = ([0.01, 0.001, 0], [3 * a_init, 10 * mu_init + 1, float(time[-1])])
    return p0, bounds


def _logistic_p0(time, od600):
    """Return initial guesses and bounds for logistic curve_fit."""
    a_init = max(float(np.max(od600)), 0.05)
    diff = np.diff(od600)
    dt = np.diff(time)
    rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(rates)), 0.01)
    max_rate_idx = int(np.argmax(rates))
    lam_init = float(time[max_rate_idx]) if max_rate_idx > 0 else float(time[0])
    k_init = 4 * mu_init / (np.e * max(a_init, 0.01))
    t_mid_init = lam_init + a_init / (mu_init * np.e + 1e-6)
    p0 = [a_init, k_init, t_mid_init]
    bounds = ([0.01, 0.001, 0], [3 * a_init, 20, float(time[-1]) * 1.5])
    return p0, bounds


def _baranyi_p0(time, od600):
    """Return initial guesses and bounds for Baranyi curve_fit."""
    a_init = max(float(np.max(od600)), 0.05)
    y0_init = max(float(np.mean(od600[:min(5, len(od600))])), 0.001)
    diff = np.diff(od600)
    dt = np.diff(time)
    rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(rates)), 0.01)
    max_rate_idx = int(np.argmax(rates))
    lam_init = float(time[max_rate_idx]) if max_rate_idx > 0 else float(time[0])
    h0_init = mu_init * max(lam_init, 0.1)
    p0 = [y0_init, a_init, mu_init, h0_init]
    bounds = ([1e-6, 0.01, 0.001, 0.001],
              [a_init + 0.01, 3 * a_init, 10 * mu_init + 1, 200])
    return p0, bounds


def _richards_p0(time, od600):
    """Return initial guesses and bounds for Richards curve_fit."""
    a_init = max(float(np.max(od600)), 0.05)
    diff = np.diff(od600)
    dt = np.diff(time)
    rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(rates)), 0.01)
    max_rate_idx = int(np.argmax(rates))
    lam_init = float(time[max_rate_idx]) if max_rate_idx > 0 else float(time[0])
    k_init = mu_init * np.e / max(a_init, 0.01) * 1.5
    t_mid_init = lam_init + a_init / (mu_init * np.e + 1e-6)
    p0 = [a_init, k_init, t_mid_init, 0.5]
    bounds = ([0.01, 0.001, 0, 0.01],
              [3 * a_init, 20, float(time[-1]) * 1.5, 5.0])
    return p0, bounds


# Model registry for MCCV multi-model evaluation
_MCCV_MODELS = None  # Lazy-initialized after model functions are defined


def _get_mccv_models():
    """Get model registry (lazy init to avoid forward-reference issues)."""
    global _MCCV_MODELS
    if _MCCV_MODELS is None:
        _MCCV_MODELS = {
            'gompertz': (gompertz_model, _gompertz_p0),
            'logistic': (logistic_model, _logistic_p0),
            'baranyi': (baranyi_model, _baranyi_p0),
            'richards': (richards_model, _richards_p0),
        }
    return _MCCV_MODELS


def find_optimal_truncation_v2(
    time: np.ndarray,
    od600: np.ndarray,
    min_points: int = 30,
    trim_noisy_start: bool = True,
    n_coarse: int = 50,
    n_cv_folds: int = 20,
    cv_holdout: float = 0.2,
    n_fine_top: int = 3,
    fine_window: int = 10,
    try_multi_model: bool = False
) -> TruncationLandscape:
    """
    Find optimal truncation point via Monte Carlo Cross-Validation.

    Scans the FULL range of possible endpoints, scoring each by how well
    a Gompertz fit generalizes to held-out data points. This naturally
    prevents overfitting on short segments and penalizes including noisy
    post-growth data.

    Parameters
    ----------
    time, od600 : np.ndarray
        Full time and OD600 arrays
    min_points : int
        Minimum data points for fitting (default: 30)
    trim_noisy_start : bool
        Find and trim noisy baseline (default: True)
    n_coarse : int
        Number of coarse-scan candidates (default: 50)
    n_cv_folds : int
        CV folds per candidate (default: 20)
    cv_holdout : float
        Fraction held out per fold (default: 0.2)
    n_fine_top : int
        Number of top candidates to fine-tune (default: 3)
    fine_window : int
        Points ±around each top candidate for fine-tuning (default: 10)
    try_multi_model : bool
        Also try logistic/baranyi/richards at top points (default: False)

    Returns
    -------
    TruncationLandscape
        Rich result with best truncation, candidates, confidence, etc.
    """
    n_points = len(time)

    # Find growth start
    if trim_noisy_start:
        start_idx, _ = find_growth_start(time, od600)
    else:
        start_idx = 0

    # Early exit for flat / no-growth curves (skip expensive MCCV)
    signal_range = float(np.max(od600) - np.min(od600))
    if signal_range < 0.05:
        return TruncationLandscape(
            best_start_idx=start_idx, best_end_idx=n_points - 1,
            best_end_time=float(time[-1]), best_cv_score=0.0,
            best_model='gompertz', candidates=[], confidence=0.0,
            bio_estimate_idx=n_points - 1
        )

    # Biological estimate (for reference and as a candidate)
    bio_idx, bio_meta = find_stationary_phase_start(time, od600)
    max_od_idx = bio_meta['max_od_idx']

    # Find inflection point (peak growth rate) — truncation is always after this
    smoothed = np.convolve(od600, np.ones(5) / 5, mode='same')
    growth_rates = np.diff(smoothed) / np.maximum(np.diff(time), 1e-6)
    inflection_idx = int(np.argmax(growth_rates))  # already full-array index

    # ---- Phase 1: Coarse scan from inflection through end ----
    # Truncation point is always after peak growth rate, so start scanning there
    # but ensure we keep at least min_points from start_idx
    scan_start = max(start_idx + min_points, inflection_idx)
    scan_end = n_points - 1

    if scan_end <= scan_start:
        # Not enough data — return full range
        return TruncationLandscape(
            best_start_idx=start_idx, best_end_idx=n_points - 1,
            best_end_time=float(time[-1]), best_cv_score=0.0,
            best_model='gompertz', candidates=[], confidence=0.0,
            bio_estimate_idx=bio_idx
        )

    # Generate uniformly spaced candidates
    n_actual = min(n_coarse, scan_end - scan_start)
    step = max(1, (scan_end - scan_start) // n_actual)
    candidate_indices = list(range(scan_start, scan_end + 1, step))

    # Always include biological estimate and max OD
    for special_idx in [bio_idx, max_od_idx]:
        if scan_start <= special_idx <= scan_end and special_idx not in candidate_indices:
            candidate_indices.append(special_idx)
    candidate_indices = sorted(set(candidate_indices))

    # Pre-filter: single fast fit to skip obviously bad candidates
    viable_candidates = []
    for end_idx in candidate_indices:
        if end_idx + 1 - start_idx < min_points:
            continue
        t_seg = time[start_idx:end_idx + 1]
        od_seg = od600[start_idx:end_idx + 1]
        try:
            p0, bounds = _gompertz_p0(t_seg, od_seg)
            popt, _ = curve_fit(gompertz_model, t_seg, od_seg, p0=p0,
                                bounds=bounds, maxfev=500)
            pred = gompertz_model(t_seg, *popt)
            ss_res = np.sum((od_seg - pred) ** 2)
            ss_tot = np.sum((od_seg - np.mean(od_seg)) ** 2)
            quick_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            quick_r2 = 0.0
        if quick_r2 > 0.3 or end_idx in (bio_idx, max_od_idx):
            viable_candidates.append(end_idx)

    # Score viable candidates in parallel (curve_fit releases GIL for BLAS)
    n_workers = min(os.cpu_count() or 4, 8)
    evaluated = {}  # idx -> TruncationCandidate

    def _make_gompertz_candidate(end_idx, cv_mean, cv_std, raw_r2, params):
        p_dict = {}
        if params is not None:
            p_dict = {'a': float(params[0]), 'mu': float(params[1]),
                      'lambda': float(params[2])}
        return TruncationCandidate(
            end_idx=end_idx, end_time=float(time[end_idx]),
            n_points=end_idx + 1 - start_idx, cv_score=cv_mean,
            cv_std=cv_std, raw_r2=raw_r2, model_name='gompertz',
            params=p_dict
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # ---- Phase 1: Coarse scan (parallel) ----
        phase1_futures = {}
        for end_idx in viable_candidates:
            t_seg = time[start_idx:end_idx + 1]
            od_seg = od600[start_idx:end_idx + 1]
            f = pool.submit(mccv_score, t_seg, od_seg,
                            n_folds=n_cv_folds,
                            holdout_fraction=cv_holdout,
                            seed=42 + end_idx)
            phase1_futures[f] = end_idx

        for f in as_completed(phase1_futures):
            end_idx = phase1_futures[f]
            cv_mean, cv_std, raw_r2, params = f.result()
            evaluated[end_idx] = _make_gompertz_candidate(
                end_idx, cv_mean, cv_std, raw_r2, params)

        if not evaluated:
            return TruncationLandscape(
                best_start_idx=start_idx, best_end_idx=n_points - 1,
                best_end_time=float(time[-1]), best_cv_score=0.0,
                best_model='gompertz', candidates=[], confidence=0.0,
                bio_estimate_idx=bio_idx
            )

        # ---- Phase 2: Fine-tune around top candidates (parallel) ----
        sorted_candidates = sorted(evaluated.values(),
                                   key=lambda c: c.cv_score, reverse=True)
        top_indices = [c.end_idx for c in sorted_candidates[:n_fine_top]]

        phase2_futures = {}
        for center_idx in top_indices:
            fine_start = max(scan_start, center_idx - fine_window)
            fine_end = min(scan_end, center_idx + fine_window)
            for end_idx in range(fine_start, fine_end + 1):
                if end_idx in evaluated or end_idx in phase2_futures.values():
                    continue
                if end_idx + 1 - start_idx < min_points:
                    continue
                t_seg = time[start_idx:end_idx + 1]
                od_seg = od600[start_idx:end_idx + 1]
                f = pool.submit(mccv_score, t_seg, od_seg,
                                n_folds=n_cv_folds,
                                holdout_fraction=cv_holdout,
                                seed=42 + end_idx)
                phase2_futures[f] = end_idx

        for f in as_completed(phase2_futures):
            end_idx = phase2_futures[f]
            cv_mean, cv_std, raw_r2, params = f.result()
            evaluated[end_idx] = _make_gompertz_candidate(
                end_idx, cv_mean, cv_std, raw_r2, params)

        # ---- Phase 3: Multi-model at top candidates (parallel, optional) ----
        if try_multi_model:
            models = _get_mccv_models()
            sorted_candidates = sorted(evaluated.values(),
                                       key=lambda c: c.cv_score, reverse=True)
            top_for_mm = [c.end_idx for c in sorted_candidates[:3]]

            phase3_futures = {}
            for end_idx in top_for_mm:
                t_seg = time[start_idx:end_idx + 1]
                od_seg = od600[start_idx:end_idx + 1]
                for model_name, (model_func, p0_func) in models.items():
                    if model_name == 'gompertz':
                        continue
                    f = pool.submit(mccv_score, t_seg, od_seg,
                                    model_func=model_func, p0_func=p0_func,
                                    n_folds=n_cv_folds,
                                    holdout_fraction=cv_holdout,
                                    seed=42 + end_idx)
                    phase3_futures[f] = (end_idx, model_name)

            for f in as_completed(phase3_futures):
                end_idx, model_name = phase3_futures[f]
                cv_mean, cv_std, raw_r2, params = f.result()
                if cv_mean > evaluated[end_idx].cv_score:
                    p_dict = {}
                    if params is not None:
                        if model_name in ('logistic', 'richards'):
                            p_dict = {'a': float(params[0]), 'mu': float(params[1]),
                                      'lambda': float(params[2])}
                        elif model_name == 'baranyi':
                            p_dict = {'a': float(params[1]), 'mu': float(params[2]),
                                      'lambda': float(params[3] / max(params[2], 1e-6))}
                    evaluated[end_idx] = TruncationCandidate(
                        end_idx=end_idx, end_time=float(time[end_idx]),
                        n_points=end_idx + 1 - start_idx, cv_score=cv_mean,
                        cv_std=cv_std, raw_r2=raw_r2, model_name=model_name,
                        params=p_dict
                    )

    # ---- Find best and compute confidence ----
    all_candidates = sorted(evaluated.values(), key=lambda c: c.cv_score,
                            reverse=True)
    best = all_candidates[0]

    # Confidence: how peaked is the landscape?
    scores = np.array([c.cv_score for c in all_candidates if c.cv_score > -0.5])
    if len(scores) >= 3:
        best_score = scores[0]
        score_range = best_score - np.min(scores)
        if score_range > 1e-6:
            near_best = np.sum(scores >= 0.95 * best_score) / len(scores)
            confidence = float(np.clip(1.0 - near_best / 0.5, 0.0, 1.0))
        else:
            confidence = 0.0
    else:
        confidence = 0.5

    # ---- Optimize start point (parallel) ----
    if trim_noisy_start:
        start_candidates = sorted(set([
            0,
            max(0, start_idx - 5),
            start_idx,
            min(start_idx + 5, best.end_idx - min_points),
            min(start_idx + 10, best.end_idx - min_points),
        ]))
        start_candidates = [s for s in start_candidates
                            if 0 <= s and best.end_idx - s >= min_points]
        best_start = start_idx
        best_start_cv = best.cv_score

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            start_futures = {}
            for s in start_candidates:
                if s == start_idx:
                    continue
                t_seg = time[s:best.end_idx + 1]
                od_seg = od600[s:best.end_idx + 1]
                f = pool.submit(mccv_score, t_seg, od_seg,
                                n_folds=n_cv_folds,
                                holdout_fraction=cv_holdout,
                                seed=42 + s)
                start_futures[f] = s

            for f in as_completed(start_futures):
                s = start_futures[f]
                cv_mean, _, _, _ = f.result()
                if cv_mean > best_start_cv:
                    best_start = s
                best_start_cv = cv_mean
    else:
        best_start = 0

    return TruncationLandscape(
        best_start_idx=best_start,
        best_end_idx=best.end_idx,
        best_end_time=best.end_time,
        best_cv_score=best.cv_score,
        best_model=best.model_name,
        candidates=all_candidates,
        confidence=confidence,
        bio_estimate_idx=bio_idx
    )


def _UNUSED_find_optimal_truncation_v1(
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

    if is_incomplete:
        # Curve never reached stationary phase — use all data (no truncation)
        # This prevents false negatives on truncation_challenge-type curves
        truncation_idx = len(od600) - 1
        actual_max_idx = np.argmax(od600)
    elif use_adaptive:
        # Monte Carlo Cross-Validated truncation: scan full range, score by
        # out-of-sample prediction quality (prevents overfitting on short segments)
        landscape = find_optimal_truncation_v2(time, od600)
        start_idx = landscape.best_start_idx
        truncation_idx = landscape.best_end_idx
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


def logistic_model(t: np.ndarray, A: float, k: float, t_mid: float) -> np.ndarray:
    """
    Logistic growth model: y(t) = A / (1 + exp(-k*(t - t_mid)))

    Parameters
    ----------
    t : np.ndarray
        Time values
    A : float
        Maximum population (carrying capacity)
    k : float
        Growth rate coefficient
    t_mid : float
        Midpoint time (inflection point)
    """
    return A / (1 + np.exp(-k * (t - t_mid)))


def baranyi_model(t: np.ndarray, y0: float, y_max: float, mu_max: float, h0: float) -> np.ndarray:
    """
    Baranyi-Roberts growth model with lag adjustment function.

    Parameters
    ----------
    t : np.ndarray
        Time values
    y0 : float
        Initial population
    y_max : float
        Maximum population
    mu_max : float
        Maximum specific growth rate
    h0 : float
        Physiological state parameter (controls lag duration)
    """
    with np.errstate(over='ignore', invalid='ignore'):
        term1 = np.exp(-mu_max * t)
        term2 = np.exp(-h0)
        term3 = np.exp(-mu_max * t - h0)
        inner = np.clip(term1 + term2 - term3, 1e-10, None)
        A_t = t + (1.0 / mu_max) * np.log(inner)
        growth_term = mu_max * A_t
        saturation = np.log(1 + (np.exp(np.clip(growth_term, -50, 50)) - 1) /
                           np.exp(np.clip(y_max - y0, -50, 50)))
        y = y0 + growth_term - saturation
    return np.clip(np.nan_to_num(y, nan=y0, posinf=y_max), y0, y_max)


def richards_model(t: np.ndarray, A: float, k: float, t_mid: float, nu: float) -> np.ndarray:
    """
    Richards growth model (generalised logistic).

    y(t) = A * (1 + nu*exp(-k*(t-t_mid)))^(-1/nu)

    Parameters
    ----------
    t : np.ndarray
        Time values
    A : float
        Maximum population (carrying capacity)
    k : float
        Growth rate coefficient
    t_mid : float
        Midpoint time
    nu : float
        Shape parameter (asymmetry)
    """
    if abs(nu) < 1e-6:
        return gompertz_model(t, A, k * A / np.e, t_mid - A / (k * np.e))
    with np.errstate(over='ignore', invalid='ignore'):
        base = np.clip(1 + nu * np.exp(-k * (t - t_mid)), 1e-10, None)
        y = A * np.power(base, -1.0 / nu)
    return np.nan_to_num(y, nan=0.0, posinf=A)


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
        dt = np.maximum(np.diff(time), 1e-6)
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
# Multi-Model Fallback
# =============================================================================

# Map of alternative models: name -> (function, param_names, n_params)
ALTERNATIVE_MODELS = {
    'logistic': (logistic_model, ['A', 'k', 't_mid'], 3),
    'baranyi': (baranyi_model, ['y0', 'y_max', 'mu_max', 'h0'], 4),
    'richards': (richards_model, ['A', 'k', 't_mid', 'nu'], 4),
}


def try_alternative_models(
    time: np.ndarray,
    od600: np.ndarray,
    gompertz_r2: float = 0.0,
    max_iterations: int = 5000
) -> Tuple[Optional[FitResult], str]:
    """
    Try fitting alternative growth models when Gompertz fit is poor.

    Uses lightweight curve_fit (not full MLE) for speed. Returns the best
    alternative fit (if it beats the Gompertz R²) along with the model name.

    Parameters
    ----------
    time : np.ndarray
        Time values
    od600 : np.ndarray
        OD600 values (truncated)
    gompertz_r2 : float
        R² from the Gompertz fit (alternative must beat this)
    max_iterations : int
        Max iterations for curve_fit

    Returns
    -------
    Tuple[Optional[FitResult], str]
        (best_fit_result, model_name) or (None, 'gompertz') if no improvement
    """
    import warnings as _warnings

    a_init = float(np.max(od600))
    a_init = max(a_init, 0.05)

    diff = np.diff(od600)
    dt = np.diff(time)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        growth_rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(growth_rates)), 0.01)
    max_growth_idx = int(np.argmax(growth_rates))
    lambda_init = float(time[max_growth_idx]) if max_growth_idx > 0 else float(time[0])
    t_max = float(time[-1])
    y0_init = float(np.mean(od600[:min(5, len(od600))]))

    # Define initial guesses and bounds for each model
    model_configs = {
        'logistic': {
            'p0': [a_init, 4 * mu_init / (np.e * max(a_init, 0.01)),
                   lambda_init + a_init / (mu_init * np.e + 1e-6)],
            'bounds': ([0.01, 0.001, 0], [3 * a_init, 20, t_max * 1.5]),
        },
        'baranyi': {
            'p0': [max(y0_init, 0.001), a_init, mu_init,
                   mu_init * max(lambda_init, 0.1)],
            'bounds': ([1e-6, 0.01, 0.001, 0.001],
                       [a_init + 0.01, 3 * a_init, 10 * mu_init + 1, 200]),
        },
        'richards': {
            'p0': [a_init,
                   mu_init * np.e / max(a_init, 0.01) * 1.5,
                   lambda_init + a_init / (mu_init * np.e + 1e-6),
                   0.5],
            'bounds': ([0.01, 0.001, 0, 0.01],
                       [3 * a_init, 20, t_max * 1.5, 5.0]),
        },
    }

    best_fit = None
    best_model = 'gompertz'
    best_r2 = gompertz_r2

    for model_name, (model_func, param_names, n_params) in ALTERNATIVE_MODELS.items():
        cfg = model_configs[model_name]
        try:
            popt, pcov = curve_fit(
                model_func,
                time, od600,
                p0=cfg['p0'],
                bounds=cfg['bounds'],
                maxfev=max_iterations
            )

            predicted = model_func(time, *popt)
            if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                continue

            residuals = od600 - predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((od600 - np.mean(od600)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r2 > best_r2:
                perr = np.sqrt(np.diag(pcov))

                # Map params to Gompertz-equivalent fields for FitResult
                # Each model parameterizes growth rate differently;
                # convert to Gompertz mu = max absolute growth rate
                # (dy/dt at inflection) so parameters are comparable.
                if model_name == 'logistic':
                    # Logistic: y = A/(1+exp(-k*(t-t_mid)))
                    # Max slope = A*k/4 at t=t_mid
                    # Gompertz mu = A*mu_e/A*(e) => mu = max_slope
                    A_raw, k_raw, tmid_raw = popt[0], popt[1], popt[2]
                    a_opt = A_raw
                    mu_opt = A_raw * k_raw / 4.0  # max absolute growth rate
                    lambda_opt = tmid_raw - 2.0 / max(k_raw, 1e-6)  # approx lag
                    a_err = perr[0]
                    mu_err = perr[1] * A_raw / 4.0  # scale error
                    lambda_err = perr[2]
                elif model_name == 'baranyi':
                    # Baranyi: mu_max = max specific growth rate (1/t units)
                    # Gompertz mu = A * mu_max / e (max absolute rate)
                    y0_raw, ymax_raw, mumax_raw, h0_raw = popt
                    a_opt = ymax_raw
                    mu_opt = ymax_raw * mumax_raw / np.e  # convert to Gompertz mu
                    lambda_opt = h0_raw / max(mumax_raw, 1e-6)  # h0/mu_max ≈ lag
                    a_err = perr[1]
                    mu_err = perr[2] * ymax_raw / np.e  # scale error
                    lambda_err = perr[3]
                elif model_name == 'richards':
                    # Richards: y = A*(1+nu*exp(-k*(t-t_mid)))^(-1/nu)
                    # Max slope = A*k*nu^(1/(1+nu))/(1+nu)
                    A_raw, k_raw, tmid_raw, nu_raw = popt
                    a_opt = A_raw
                    mu_opt = A_raw * k_raw * nu_raw**(1.0/(1.0+nu_raw)) / (1.0+nu_raw)
                    lambda_opt = tmid_raw - np.log(1.0 + nu_raw) / max(k_raw, 1e-6)
                    a_err = perr[0]
                    mu_err = perr[1] * A_raw * nu_raw**(1.0/(1.0+nu_raw)) / (1.0+nu_raw)
                    lambda_err = perr[2]
                else:
                    continue

                mae = float(np.mean(np.abs(residuals)))
                mse = float(np.mean(residuals ** 2))
                rmse = float(np.sqrt(mse))

                best_fit = FitResult(
                    success=True,
                    a_opt=float(a_opt), mu_opt=float(mu_opt),
                    lambda_opt=float(lambda_opt),
                    a_err=float(a_err), mu_err=float(mu_err),
                    lambda_err=float(lambda_err),
                    mae=mae, mse=mse, rmse=rmse,
                    r_squared=float(r2),
                    predicted=predicted, residuals=residuals,
                    error_message=""
                )
                best_r2 = r2
                best_model = model_name

        except Exception:
            continue

    return best_fit, best_model


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
    use_ml_classifier: bool = False,
    ml_classifier_config: Optional[Dict] = None,
    verbose: bool = True,
    no_plots: bool = False
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

    # Initialize ML classifiers if requested
    prefit_gate = None
    postfit_clf = None
    if use_ml_classifier:
        try:
            from ml_classifier import PreFitGate, PostFitClassifier
            cfg = ml_classifier_config or {}
            prefit_cfg = cfg.get('prefit_gate', {})
            postfit_cfg = cfg.get('postfit_classifier', {})

            prefit_gate = PreFitGate(
                model_path=prefit_cfg.get('model_path'),
                reject_threshold=prefit_cfg.get('reject_threshold', 0.2),
            )
            postfit_clf = PostFitClassifier(
                model_path=postfit_cfg.get('model_path'),
                feature_config_path=cfg.get('feature_config_path'),
                good_threshold=postfit_cfg.get('good_threshold', 0.7),
                bad_threshold=postfit_cfg.get('bad_threshold', 0.3),
            )
            if verbose:
                gate_status = "active" if prefit_gate.model else "disabled (no model)"
                clf_status = "active" if postfit_clf.model else "disabled (no model)"
                print(f"ML classifier: pre-fit gate {gate_status}, post-fit {clf_status}")
        except Exception as e:
            fallback = (ml_classifier_config or {}).get('fallback_to_rules', True)
            if fallback:
                if verbose:
                    print(f"ML classifier unavailable ({e}), falling back to rule-based")
            else:
                raise

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

                snr_thresh = fit_quality_thresholds.get('pre_fit_snr_threshold', 3.0)
                od_override = fit_quality_thresholds.get('pre_fit_delta_od_override', 0.5)
                if pre_fit_snr < snr_thresh and signal_range < od_override:
                    # Signal is buried in noise AND growth is small -- skip fitting, classify BAD
                    # (If signal_range >= override we let the fitter decide; large growth overrides low SNR)
                    classification = ClassificationResult(
                        is_good=False,
                        reason="BAD: Pre-fit SNR too low ({:.1f} < {:.1f})".format(pre_fit_snr, snr_thresh),
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

                    if not no_plots:
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

                # ML pre-fit gate: reject obvious junk before expensive fitting
                if prefit_gate is not None and prefit_gate.should_skip(time_clean, od600_clean, strain_name=strain_name):
                    classification = ClassificationResult(
                        is_good=False,
                        reason="BAD: Rejected by ML pre-fit gate",
                        metrics={'delta_od': signal_range, 'max_od': float(np.max(od600_clean)),
                                 'snr': pre_fit_snr, 'ml_classification': 'BAD',
                                 'p_good': 0.0}
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
                        error_message="Skipped: ML pre-fit gate rejected"
                    )
                    if verbose:
                        print(f"  {strain_name}: BAD (ML pre-fit gate)")
                    if not no_plots:
                        plot_bad_curve_with_fit(
                            time_clean, od600_clean, truncation, fit,
                            classification, strain_name, plots_dir
                        )
                    results[strain_name] = ProcessingResult(
                        strain_name=strain_name,
                        classification=classification,
                        truncation=truncation,
                        fit=fit
                    )
                    continue

                # Check if curve is incomplete (still rising at experiment end)
                is_incomplete, _ = detect_incomplete_curve(time_clean, od600_clean)

                # Step 1: Truncate (using specified method)
                truncation = truncate_at_max(
                    time_clean, od600_clean,
                    smoothing_window=truncation_params['smoothing_window'],
                    buffer_points=truncation_params['buffer_points'],
                    use_first_peak=truncation_params.get('use_first_peak', True),
                    use_adaptive=truncation_params.get('use_adaptive', False)
                )

                # Validate truncation before fitting
                trunc_valid, trunc_reason = validate_truncation(
                    truncation,
                    min_points=truncation_params['min_points_for_fit']
                )

                if not trunc_valid:
                    if verbose:
                        print(f"    Truncation invalid: {trunc_reason}")
                    classification = ClassificationResult(
                        is_good=False,
                        reason=f"BAD: {trunc_reason}",
                        metrics={'delta_od': float(np.max(od600_clean) - np.mean(od600_clean[:5])),
                                 'max_od': float(np.max(od600_clean))}
                    )
                    fit = FitResult(
                        success=False, a_opt=0, mu_opt=0, lambda_opt=0,
                        a_err=0, mu_err=0, lambda_err=0,
                        mae=0, mse=0, rmse=0, r_squared=0,
                        predicted=np.array([]), residuals=np.array([]),
                        error_message=trunc_reason
                    )
                    results[strain_name] = ProcessingResult(
                        strain_name=strain_name,
                        classification=classification,
                        truncation=truncation,
                        fit=fit
                    )
                    continue

                # Step 2: Attempt Gompertz fit
                fit = fit_gompertz(
                    truncation.time_truncated,
                    truncation.od_truncated
                )

                # Step 2b: Multi-model fallback (disabled by default — net negative
                # on validation: rescues 6 good curves but creates 14 false positives
                # because Baranyi is flexible enough to fit contamination artifacts)
                best_model_name = 'gompertz'
                use_multi_model = truncation_params.get('try_multi_model', False)

                if use_multi_model and fit.r_squared < fit_quality_thresholds.get('min_r_squared', 0.95):
                    gompertz_r2_val = fit.r_squared
                    alt_fit, alt_model = try_alternative_models(
                        truncation.time_truncated,
                        truncation.od_truncated,
                        gompertz_r2=gompertz_r2_val
                    )
                    if alt_fit is not None:
                        fit = alt_fit
                        best_model_name = alt_model
                        if verbose:
                            print(f"    Multi-model: {alt_model} R²={alt_fit.r_squared:.3f} "
                                  f"(Gompertz was {gompertz_r2_val:.3f})")

                # Step 3: Classify based on fit quality (with secondary quality gates)
                classification = classify_by_fit_quality(
                    fit, fit_quality_thresholds,
                    time=time_clean, od600=od600_clean,
                    is_incomplete=is_incomplete
                )

                # Add basic OD metrics to classification for reference
                if 'delta_od' not in classification.metrics:
                    classification.metrics['delta_od'] = np.max(od600_clean) - np.mean(od600_clean[:5])
                classification.metrics['max_od'] = np.max(od600_clean)
                classification.metrics['best_model'] = best_model_name

                # ML post-fit classifier: override rule-based with probability
                if postfit_clf is not None:
                    ml_result = postfit_clf.classify(
                        fit_result=fit,
                        classification_metrics=classification.metrics,
                        truncation_time=truncation.truncation_time,
                        points_used=len(truncation.od_truncated),
                        strain_name=strain_name,
                    )
                    if ml_result is not None:
                        classification.metrics['p_good'] = ml_result['p_good']
                        classification.metrics['ml_classification'] = ml_result['ml_classification']
                        classification.metrics['rule_based_result'] = classification.is_good
                        classification.metrics['rule_based_reason'] = classification.reason
                        classification = ClassificationResult(
                            is_good=ml_result['is_good'],
                            reason=ml_result['reason'],
                            metrics=classification.metrics,
                        )

                if verbose:
                    status = "GOOD" if classification.is_good else "BAD"
                    r2 = fit.r_squared if fit.success else 0
                    model_tag = f" [{best_model_name}]" if best_model_name != 'gompertz' else ""
                    print(f"  {strain_name}: {status} (R²={r2:.3f}, A={fit.a_opt:.3f}){model_tag}")
                    if not classification.is_good:
                        print(f"    Reason: {classification.reason}")

                # Step 4: Generate visualization
                if not no_plots:
                    if classification.is_good:
                        plot_truncation_comparison(
                            truncation, fit, strain_name,
                            plots_dir
                        )
                    else:
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
                        if not no_plots:
                            plot_truncation_comparison(
                                truncation, fit, strain_name,
                                plots_dir
                            )
                    else:
                        if verbose:
                            print(f"    Truncation invalid: {reason}")

                # Generate visualization for BAD curves too (for validation)
                if not no_plots and not classification.is_good:
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
    if results and not no_plots:
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
            'best_model': metrics.get('best_model', 'gompertz'),
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

        # Secondary quality gate features (for ML classifier training)
        row.update({
            'baseline_std': metrics.get('baseline_std'),
            'monotone_fraction': metrics.get('monotone_fraction'),
            'residual_autocorr': metrics.get('residual_autocorr'),
            'delta_od_ci_lower': metrics.get('delta_od_ci_lower'),
            'p_good': metrics.get('p_good'),
            'ml_classification': metrics.get('ml_classification'),
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
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating per-curve plots (faster for batch/validation runs)'
    )
    parser.add_argument(
        '--ml-classify',
        action='store_true',
        help='Use pre-trained ML classifier (requires models/ directory with trained models)'
    )

    args = parser.parse_args()

    # Load config.yaml for ML classifier settings
    config = {}
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Merge config.yaml values with CLI args (CLI overrides config)
    cfg_class = config.get('classification', {})
    cfg_gates = config.get('quality_gates', {})
    cfg_trunc = config.get('truncation', {})
    cfg_od = config.get('od_thresholds', {})

    # Build threshold dict for OD-based classification
    thresholds = {
        'min_delta_od': args.min_delta_od,
        'min_max_od': args.min_max_od,
        'min_snr': args.min_snr,
        'max_initial_od': cfg_od.get('max_initial_od', 0.15),
    }

    # Build threshold dict for fit-based classification
    # Includes secondary quality gates from config.yaml
    fit_thresholds = {
        'min_r_squared': args.min_r2,
        'max_param_error_pct': args.max_param_error,
        'min_points_for_fit': cfg_class.get('min_points_for_fit', 15),
        # Secondary quality gates (from config.yaml quality_gates section)
        'min_snr': cfg_gates.get('min_snr', 5.0),
        'min_delta_od_ci': cfg_gates.get('min_delta_od_ci', 0.1),
        'min_absolute_delta_od': cfg_gates.get('min_absolute_delta_od', 0.15),
        'pre_fit_snr_threshold': cfg_gates.get('pre_fit_snr_threshold', 3.0),
        'pre_fit_delta_od_override': cfg_gates.get('pre_fit_delta_od_override', 0.5),
        'excellent_r2_threshold': cfg_gates.get('excellent_r2_threshold', 0.98),
        'min_monotone_fraction': cfg_gates.get('min_monotone_fraction', 0.55),
        'max_residual_autocorr': cfg_gates.get('max_residual_autocorr', 0.7),
    }

    truncation_params = {
        'smoothing_window': args.smoothing_window,
        'buffer_points': args.buffer_points,
        'min_points_for_fit': cfg_trunc.get('min_points_for_fit', 20),
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

    # Load ML classifier config if requested
    ml_config = None
    if args.ml_classify:
        print("Using ML CLASSIFIER")
        ml_config = config.get('ml_classifier', {}) if config else {}

    results = process_growth_curves(
        data_dir=args.data_dir,
        output_dir=args.output,
        classification_thresholds=thresholds,
        truncation_params=truncation_params,
        fit_quality_thresholds=fit_thresholds,
        use_fit_based_classification=use_fit_based,
        use_ml_classifier=args.ml_classify,
        ml_classifier_config=ml_config,
        verbose=not args.quiet,
        no_plots=args.no_plots
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
