#!/usr/bin/env python3
"""
Advanced Statistical Fitting for Growth Curve Analysis

Upgrades every pipeline stage with modern statistical methods:
  1. GP Truncation      — Gaussian Process growth phase detection
  2. Bayesian Gompertz  — Hierarchical HMC with partial pooling
  3. Bayesian Haldane   — Hierarchical Ki by pesticide (ODE + DEMetropolisZ)
  4. Bootstrap CIs      — Lightweight nonparametric uncertainty
  5. Bayesian Class.    — P(good) replaces hard thresholds
  6. Model Comparison   — WAIC/LOO replaces AIC

Usage:
    python 06_advanced_fitting.py                    # run everything
    python 06_advanced_fitting.py --gp-only          # GP truncation only
    python 06_advanced_fitting.py --bootstrap-only   # bootstrap CIs only
    python 06_advanced_fitting.py --gompertz-only    # Bayesian Gompertz only
    python 06_advanced_fitting.py --haldane-only     # Bayesian Haldane only
    python 06_advanced_fitting.py --no-haldane       # skip slow Haldane ODE

BIO380SP25 — Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
import yaml

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Optional: ruptures for changepoint detection
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

# Import truncation methods from 01_growth_curve_analysis.py
import importlib.util as _ilu
_gca_path = Path(__file__).parent / '01_growth_curve_analysis.py'
_gca = None
if _gca_path.exists():
    try:
        _gca_spec = _ilu.spec_from_file_location("growth_curve_analysis", str(_gca_path))
        _gca = _ilu.module_from_spec(_gca_spec)
        _gca_spec.loader.exec_module(_gca)
    except Exception:
        _gca = None

# Import shared Haldane functions from 05 to avoid code duplication
_h05_path = Path(__file__).parent / '05_haldane_analysis.py'
_h05 = None
if _h05_path.exists():
    try:
        _h05_spec = _ilu.spec_from_file_location("haldane_analysis", str(_h05_path))
        _h05 = _ilu.module_from_spec(_h05_spec)
        _h05_spec.loader.exec_module(_h05)
    except Exception:
        _h05 = None


@dataclass
class EnsembleTruncationResult:
    """Result of ensemble truncation across multiple methods."""
    consensus_idx: int
    consensus_time: float
    consensus_confidence: float
    method_results: Dict[str, Dict[str, Any]]
    disagreement_hours: float
    flagged_for_review: bool
    method_used: str
    n_methods_succeeded: int


# =============================================================================
# SECTION 0: Config & Data Loading
# =============================================================================

def load_config(config_path=None):
    """Load pipeline configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Re-use shared functions from 05_haldane_analysis.py (single source of truth)
if _h05 is not None:
    haldane_ode = _h05.haldane_ode
    solve_haldane = _h05.solve_haldane
    identify_pesticide_strains = _h05.identify_pesticide_strains
    extract_pesticide_name = _h05.extract_pesticide_name
else:
    # Fallback definitions if 05 import failed
    def haldane_ode(t, y, mu_max, Ks, Ki, X_max, q):
        """Haldane/Andrews ODE system."""
        X, S = y
        S = max(S, 0)
        X = max(X, 0)
        if S < 1e-10:
            mu_S = 0.0
        else:
            mu_S = mu_max * S / (Ks + S + S**2 / Ki)
        dXdt = mu_S * X * (1 - X / X_max)
        dSdt = -q * mu_S * X
        return [dXdt, dSdt]

    def solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0):
        """Solve Haldane ODE. Returns (X(t), S(t))."""
        try:
            sol = solve_ivp(
                haldane_ode, [time[0], time[-1]], [X0, S0],
                args=(mu_max, Ks, Ki, X_max, q),
                t_eval=time, method='RK45', max_step=0.5,
                rtol=1e-8, atol=1e-10
            )
            if sol.success:
                return sol.y[0], sol.y[1]
        except Exception:
            pass
        return np.full_like(time, X0, dtype=float), np.full_like(time, S0, dtype=float)

    def identify_pesticide_strains(df):
        """Filter to pesticide+nutrient strains (contain 'ANDLB' or 'ANDG')."""
        mask = (
            df['strain'].str.contains('ANDLB', case=False, na=False) |
            df['strain'].str.contains('ANDG', case=False, na=False)
        )
        return df[mask].copy()

    def extract_pesticide_name(strain):
        """Extract pesticide from strain like 'BifenthrinANDLB-BIF2' or 'DiazinnonANDG-DIAZ1'."""
        s = strain.upper()
        if 'ANDLB' in s:
            return s.split('ANDLB')[0]
        if 'ANDG' in s:
            return s.split('ANDG')[0]
        return 'UNKNOWN'


def gompertz_model(t, A, mu, lam):
    """Modified Gompertz growth model."""
    return A * np.exp(-np.exp((mu * np.e / A) * (lam - t) + 1))


def load_raw_data(data_dir, strain_name, group):
    """Load raw time-series for a strain. Returns (time, od) or (None, None)."""
    group_dirs = {
        'Group1': 'Group1/Group_1_DATA',
        'Group2': 'Group 2/Group_2_DATA',
        'Group3': 'Group 3/Group_3_DATA',
        'Group4': 'Group 4/Group4_DATA',
        'Group5': 'Group5/Group5_DATA_processed',
        'Group6': 'Group6/Group6_DATA_processed',
    }
    group_path = data_dir / group_dirs.get(group, group)
    if not group_path.exists():
        return None, None

    for csv_path in group_path.glob("*_DATA.csv"):
        try:
            df = pd.read_csv(csv_path)
            time_col = [c for c in df.columns if 'TIME' in c.upper()]
            if not time_col:
                continue
            time = df[time_col[0]].values
            for col in df.columns:
                col_norm = col.replace('_blanked', '').replace('_', '-').upper()
                strain_norm = strain_name.replace('_', '-').upper()
                if col_norm == strain_norm:
                    return time, df[col].values
                # Partial match
                parts = strain_name.upper().split('-')
                col_upper = col.upper().replace('_BLANKED', '')
                if len(parts) >= 2:
                    prefix = parts[0].replace('-', '')
                    suffix = parts[-1]
                    if prefix in col_upper and suffix in col_upper:
                        return time, df[col].values
        except Exception:
            continue
    return None, None


def thin_strain_data(strain_data, max_points=50):
    """
    Thin each strain's time series to at most max_points via uniform subsampling.

    Always keeps the first and last points. Remaining points are evenly spaced.
    This reduces Bayesian model compilation and sampling time dramatically.

    Args:
        strain_data: list of (time_array, od_array) tuples
        max_points: maximum number of points per strain

    Returns:
        list of thinned (time_array, od_array) tuples
    """
    thinned = []
    for t, od in strain_data:
        if len(t) <= max_points:
            thinned.append((t, od))
        else:
            # Always keep first and last; uniformly space the rest
            idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
            idx = np.unique(idx)  # remove duplicates from rounding
            thinned.append((t[idx], od[idx]))
    return thinned


# =============================================================================
# SECTION 1: GP-Based Truncation
# =============================================================================

def fit_gp(time, od, config=None):
    """
    Fit a Gaussian Process to growth curve data.

    Returns fitted GP model and dense prediction grid.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    cfg = (config or {}).get('advanced', {}).get('gp', {})
    length_scale = cfg.get('kernel_length_scale', 2.0)
    noise_level = cfg.get('kernel_noise', 0.01)
    n_restarts = cfg.get('n_restarts', 5)
    min_ls = cfg.get('min_length_scale', 0.5)

    kernel = (
        ConstantKernel(1.0, (0.01, 10.0))
        * RBF(length_scale=length_scale, length_scale_bounds=(min_ls, 20.0))
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 0.5))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        alpha=1e-6
    )

    # Reshape for sklearn
    T = time.reshape(-1, 1)
    gp.fit(T, od)

    # Dense prediction grid
    t_dense = np.linspace(time[0], time[-1], 500).reshape(-1, 1)
    mu, sigma = gp.predict(t_dense, return_std=True)

    return gp, t_dense.ravel(), mu, sigma


def gp_find_phases(t_dense, mu_dense, config=None):
    """
    Identify growth phases from GP derivative.

    Returns dict with phase boundaries:
      - lag_end: end of lag phase
      - exp_peak: peak exponential growth
      - stat_start: start of stationary phase (TRUNCATION POINT)
      - death_start: start of death phase (if present)
    """
    cfg = (config or {}).get('advanced', {}).get('gp', {})
    deriv_thresh_cfg = cfg.get('derivative_threshold', 0.01)

    # Compute derivative via finite differences on GP mean
    dt = np.diff(t_dense)
    dmu = np.diff(mu_dense)
    deriv = dmu / dt
    t_deriv = (t_dense[:-1] + t_dense[1:]) / 2

    # Normalize derivative threshold relative to the max derivative.
    # The config value (default 0.01) is treated as a fraction of max dOD/dt
    # when normalize_derivative is True, making it scale-independent across
    # strains with different OD ranges.
    normalize = cfg.get('normalize_derivative', True)
    max_deriv_val = float(np.max(deriv)) if len(deriv) > 0 else 1.0
    if normalize and max_deriv_val > 1e-8:
        deriv_thresh = deriv_thresh_cfg * max_deriv_val
    else:
        deriv_thresh = deriv_thresh_cfg

    phases = {
        'lag_end': None,
        'exp_peak': None,
        'stat_start': None,
        'death_start': None,
        'derivative': deriv,
        't_derivative': t_deriv,
    }

    # Find where derivative first exceeds threshold (end of lag)
    above_thresh = np.where(deriv > deriv_thresh)[0]
    if len(above_thresh) > 0:
        phases['lag_end'] = t_deriv[above_thresh[0]]

    # Find peak derivative (max exponential growth)
    peak_idx = np.argmax(deriv)
    phases['exp_peak'] = t_deriv[peak_idx]

    # Find where derivative drops back to ~0 after peak (stationary onset)
    # Primary: first zero-crossing of derivative after the peak
    # Improved: also check for derivative dropping below 5% of max (more
    # robust than exact zero-crossing which may never occur for noisy data)
    post_peak = deriv[peak_idx:]
    zero_crossings = np.where(
        (post_peak[:-1] > deriv_thresh) & (post_peak[1:] <= deriv_thresh)
    )[0]
    if len(zero_crossings) > 0:
        stat_idx = peak_idx + zero_crossings[0]
        phases['stat_start'] = t_deriv[stat_idx]
    else:
        # Fallback: use point where derivative drops to 5% of max
        # (lowered from 10% for better sensitivity on incomplete curves)
        fallback_frac = cfg.get('stationary_fallback_fraction', 0.05)
        if max_deriv_val > 1e-8:
            below_threshold = np.where(
                deriv[peak_idx:] < fallback_frac * max_deriv_val
            )[0]
            if len(below_threshold) > 0:
                stat_idx = peak_idx + below_threshold[0]
                phases['stat_start'] = t_deriv[stat_idx]

    # Find death phase (derivative goes negative after stationary)
    neg_thresh = -deriv_thresh  # symmetric with positive threshold
    neg_crossings = np.where(deriv < neg_thresh)[0]
    if len(neg_crossings) > 0:
        # Only count if after the peak
        post_peak_neg = neg_crossings[neg_crossings > peak_idx]
        if len(post_peak_neg) > 0:
            phases['death_start'] = t_deriv[post_peak_neg[0]]

    return phases


def gp_truncate(time, od, config=None):
    """
    GP-based truncation: fit GP, find stationary onset, return truncation index.

    Returns:
        truncation_idx: index into original time array
        confidence: GP variance at truncation point (lower = more confident)
        phases: phase boundary dict
        gp_data: (t_dense, mu, sigma) for plotting
    """
    gp, t_dense, mu, sigma = fit_gp(time, od, config)
    phases = gp_find_phases(t_dense, mu, config)

    # Truncation point = stationary onset
    trunc_time = phases.get('stat_start')

    if trunc_time is None:
        # Fallback: use time of GP maximum
        trunc_time = t_dense[np.argmax(mu)]

    # Map back to nearest index in original time array
    truncation_idx = int(np.argmin(np.abs(time - trunc_time)))
    # Ensure minimum points
    truncation_idx = max(truncation_idx, 20)
    truncation_idx = min(truncation_idx, len(time) - 1)

    # Confidence from GP variance at truncation point, normalized by max OD
    # so it's comparable across strains with different OD ranges
    t_trunc_arr = np.array([[trunc_time]])
    _, std_at_trunc = gp.predict(t_trunc_arr, return_std=True)
    max_od = float(np.max(od)) if len(od) > 0 else 1.0
    # Confidence = 1 - relative_uncertainty (clamped to [0, 1])
    confidence = max(0.0, min(1.0, 1.0 - float(std_at_trunc[0]) / max(max_od, 1e-6)))

    return truncation_idx, confidence, phases, (t_dense, mu, sigma)


def plot_gp_truncation(time, od, truncation_idx, phases, gp_data,
                       strain_name, output_path):
    """Plot GP fit with derivative and phase markers."""
    t_dense, mu, sigma = gp_data

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                             sharex=True)

    # Top: GP fit + data + truncation
    ax = axes[0]
    ax.scatter(time, od, s=8, alpha=0.3, color='gray', label='Data', zorder=1)
    ax.plot(t_dense, mu, 'b-', linewidth=2, label='GP mean', zorder=3)
    ax.fill_between(t_dense, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.15, color='blue', label='95% CI')
    ax.axvline(time[truncation_idx], color='red', linestyle='--',
               linewidth=2, label=f'Truncation (t={time[truncation_idx]:.1f}h)')

    # Phase markers
    for phase, label, color in [
        ('lag_end', 'Lag end', 'green'),
        ('exp_peak', 'Max growth', 'orange'),
        ('death_start', 'Death onset', 'purple'),
    ]:
        t_phase = phases.get(phase)
        if t_phase is not None:
            ax.axvline(t_phase, color=color, linestyle=':', alpha=0.6, label=label)

    ax.set_ylabel('OD600', fontsize=11)
    ax.set_title(f'{strain_name}: GP-Based Growth Phase Detection', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Bottom: derivative
    ax2 = axes[1]
    t_deriv = phases.get('t_derivative', [])
    deriv = phases.get('derivative', [])
    if len(t_deriv) > 0:
        ax2.plot(t_deriv, deriv, 'k-', linewidth=1)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.fill_between(t_deriv, 0, deriv, where=np.array(deriv) > 0,
                         alpha=0.2, color='green', label='Growth')
        ax2.fill_between(t_deriv, 0, deriv, where=np.array(deriv) < 0,
                         alpha=0.2, color='red', label='Decline')
        ax2.axvline(time[truncation_idx], color='red', linestyle='--', linewidth=2)

    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('dOD/dt', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 1.5: Ensemble Truncation
# =============================================================================

def weighted_median(values, weights):
    """Compute weighted median of values with given weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_wts = weights[sorted_idx]
    cumulative = np.cumsum(sorted_wts)
    midpoint = cumulative[-1] / 2.0
    idx = np.searchsorted(cumulative, midpoint)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def changepoint_truncate(time, od, config=None):
    """
    Changepoint-based truncation using ruptures PELT with RBF cost.

    Detects regime transitions (lag -> exponential -> stationary).
    Truncation = first changepoint where the next segment is near max OD
    and has near-zero slope (stationary phase onset).

    Returns:
        truncation_idx: int, index into original time array
        confidence: float (0-1), segmentation quality score
        changepoints: list of int, all detected changepoint indices
        n_changepoints: int
    """
    if not HAS_RUPTURES:
        return None, 0.0, [], 0

    cfg = (config or {}).get('advanced', {}).get('ensemble', {}).get('changepoint', {})
    model = cfg.get('model', 'rbf')
    min_size = cfg.get('min_size') or max(10, len(od) // 10)
    penalty = cfg.get('penalty') or 3.0 * np.log(len(od))

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(od)
        result = algo.predict(pen=penalty)
        # ruptures returns breakpoint indices (1-indexed end of segment), last is always len(signal)
        changepoints = [cp for cp in result if cp < len(od)]

        if len(changepoints) == 0:
            return int(np.argmax(od)), 0.0, [], 0

        # Find the stationary onset: first changepoint where the next segment
        # has mean OD >= 90% of max AND low slope
        max_od = np.max(od)
        best_cp = None
        for i, cp in enumerate(changepoints):
            # Look at the segment starting at this changepoint
            next_end = changepoints[i + 1] if i + 1 < len(changepoints) else len(od)
            seg = od[cp:next_end]
            if len(seg) < 3:
                continue
            seg_mean = np.mean(seg)
            seg_slope = (seg[-1] - seg[0]) / (time[min(next_end - 1, len(time) - 1)] - time[cp]) if next_end > cp + 1 else 0
            if seg_mean >= 0.85 * max_od and abs(seg_slope) < 0.05:
                best_cp = cp
                break

        if best_cp is None:
            # Fallback: changepoint closest to and before the OD maximum
            max_idx = int(np.argmax(od))
            before_max = [cp for cp in changepoints if cp <= max_idx + 5]
            if before_max:
                best_cp = before_max[-1]
            else:
                best_cp = changepoints[0]

        # Confidence: based on how well segmentation explains variance
        total_var = np.var(od) * len(od)
        if total_var > 0:
            # Compute within-segment variance
            segs = [0] + changepoints + ([len(od)] if changepoints[-1] != len(od) else [])
            within_var = 0
            for j in range(len(segs) - 1):
                seg = od[segs[j]:segs[j + 1]]
                if len(seg) > 1:
                    within_var += np.var(seg) * len(seg)
            confidence = max(0.0, min(1.0, 1.0 - within_var / total_var))
        else:
            confidence = 0.0

        return int(best_cp), confidence, changepoints, len(changepoints)

    except Exception:
        return None, 0.0, [], 0


def first_peak_truncate_wrapper(time, od, config=None):
    """Wrapper around 01_gca.truncate_at_max(use_first_peak=True)."""
    if _gca is None:
        return None, 0.0, {}
    try:
        result = _gca.truncate_at_max(time, od, use_first_peak=True)
        idx = result.truncation_index if hasattr(result, 'truncation_index') else result
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        idx = int(idx)
        # Confidence: derivative sharpness at the truncation point
        smoothed = np.convolve(od, np.ones(5)/5, mode='same')
        deriv = np.diff(smoothed)
        if idx > 0 and idx < len(deriv):
            # How clearly the derivative crosses zero at this peak
            max_deriv = np.max(np.abs(deriv)) if len(deriv) > 0 else 1
            confidence = min(1.0, abs(deriv[min(idx, len(deriv)-1)]) / max(max_deriv, 1e-6))
            confidence = 1.0 - confidence  # near zero derivative AT peak = high confidence
        else:
            confidence = 0.5
        return idx, confidence, {'source': 'first_peak'}
    except Exception:
        return None, 0.0, {}


def stationary_phase_truncate_wrapper(time, od, config=None):
    """Wrapper around 01_gca.find_stationary_phase_start()."""
    if _gca is None:
        return None, 0.0, {}
    try:
        result = _gca.find_stationary_phase_start(time, od)
        if isinstance(result, tuple):
            idx = int(result[0])
            meta = result[1] if len(result) > 1 else {}
        else:
            idx = int(result)
            meta = {}
        # Confidence: agreement between the two internal criteria
        # Lower disagreement = higher confidence
        r90 = meta.get('reached_90_idx', idx)
        rp = meta.get('rate_plateau_idx', idx)
        spread = abs(r90 - rp) / max(len(time), 1)
        confidence = max(0.0, 1.0 - spread)
        return idx, confidence, meta
    except Exception:
        return None, 0.0, {}


def adaptive_r2_truncate_wrapper(time, od, config=None):
    """Wrapper around 01_gca.find_optimal_truncation_v2() (MCCV-based)."""
    if _gca is None:
        return None, 0.0, {}
    try:
        if hasattr(_gca, 'find_optimal_truncation_v2'):
            landscape = _gca.find_optimal_truncation_v2(time, od)
            idx = int(landscape.best_end_idx)
            confidence = float(landscape.confidence)
            return idx, confidence, {
                'best_cv_score': landscape.best_cv_score,
                'best_model': landscape.best_model,
                'n_candidates': len(landscape.candidates),
            }
        else:
            # Fallback to v1
            result = _gca.find_optimal_truncation(time, od)
            if isinstance(result, tuple) and len(result) >= 3:
                idx = int(result[1])
                confidence = float(result[2])
            else:
                idx = int(result)
                confidence = 0.5
            return idx, confidence, {'best_r2': confidence}
    except Exception:
        return None, 0.0, {}


def gp_truncate_wrapper(time, od, config=None):
    """Wrapper around the existing gp_truncate() function."""
    try:
        trunc_idx, gp_conf, phases, gp_data = gp_truncate(time, od, config)
        # gp_conf is already 1.0 - relative_uncertainty (higher = better)
        confidence = max(0.0, min(1.0, gp_conf))
        return trunc_idx, confidence, {'phases': phases}
    except Exception:
        return None, 0.0, {}


def ensemble_truncate(time, od, config=None):
    """
    Run all enabled truncation methods and produce a consensus via weighted median.

    Returns EnsembleTruncationResult with consensus + per-method details.
    """
    cfg = (config or {}).get('advanced', {}).get('ensemble', {})
    method_cfg = cfg.get('methods', {})
    consensus_method = cfg.get('consensus_method', 'weighted_median')
    max_disagree = cfg.get('max_disagreement_hours', 4.0)
    min_methods = cfg.get('min_methods_required', 3)

    methods = []
    if method_cfg.get('first_peak', True):
        methods.append(('first_peak', first_peak_truncate_wrapper))
    if method_cfg.get('stationary_phase', True):
        methods.append(('stationary_phase', stationary_phase_truncate_wrapper))
    if method_cfg.get('adaptive_r2', True):
        methods.append(('adaptive_r2', adaptive_r2_truncate_wrapper))
    if method_cfg.get('gp_derivative', True):
        methods.append(('gp_derivative', gp_truncate_wrapper))
    if method_cfg.get('changepoint', True):
        methods.append(('changepoint', changepoint_truncate))

    results = {}
    for name, func in methods:
        try:
            out = func(time, od, config)
            if out is None:
                continue
            if name == 'changepoint':
                idx, conf, cps, n_cp = out
                meta = {'changepoints': cps, 'n_changepoints': n_cp}
            else:
                idx, conf, meta = out
            if idx is not None and 0 <= idx < len(time):
                results[name] = {
                    'idx': int(idx),
                    'time': float(time[idx]),
                    'confidence': float(conf),
                    'metadata': meta,
                }
        except Exception:
            continue

    n_succeeded = len(results)

    if n_succeeded == 0:
        # Complete failure: use global max as last resort
        fallback_idx = int(np.argmax(od))
        return EnsembleTruncationResult(
            consensus_idx=fallback_idx,
            consensus_time=float(time[fallback_idx]),
            consensus_confidence=0.0,
            method_results={},
            disagreement_hours=0.0,
            flagged_for_review=True,
            method_used='fallback_max',
            n_methods_succeeded=0,
        )

    times = np.array([r['time'] for r in results.values()])
    confs = np.array([r['confidence'] for r in results.values()])
    method_names = list(results.keys())
    # Ensure minimum confidence weight
    confs = np.maximum(confs, 0.01)

    # MAD-based outlier detection: downweight methods that disagree with the
    # majority. This prevents a single bad method (e.g. changepoint) from
    # pulling the consensus. A method is an outlier if its time is > 3 MADs
    # from the median of all method times.
    mad_threshold = cfg.get('mad_outlier_threshold', 3.0)
    if n_succeeded >= 3:
        med_time = float(np.median(times))
        abs_devs = np.abs(times - med_time)
        mad = float(np.median(abs_devs)) if len(abs_devs) > 0 else 0.0
        # MAD = 0 means perfect agreement (all at median); skip outlier check
        if mad > 1e-6:
            for i, (name, t_val) in enumerate(zip(method_names, times)):
                deviation = abs_devs[i] / mad
                if deviation > mad_threshold:
                    # Outlier: reduce confidence to 10% of original
                    original_conf = confs[i]
                    confs[i] = original_conf * 0.1
                    results[name]['outlier'] = True
                    results[name]['mad_deviation'] = float(deviation)

    # Compute consensus
    if consensus_method == 'weighted_median' and n_succeeded >= 2:
        consensus_time = weighted_median(times, confs)
    elif consensus_method == 'weighted_mean' and n_succeeded >= 2:
        consensus_time = float(np.average(times, weights=confs))
    else:
        consensus_time = float(np.median(times))

    # Map consensus time to nearest index
    consensus_idx = int(np.argmin(np.abs(time - consensus_time)))
    consensus_idx = max(consensus_idx, 20)
    consensus_idx = min(consensus_idx, len(time) - 1)
    consensus_time = float(time[consensus_idx])

    # Disagreement metrics
    spread = float(np.max(times) - np.min(times))
    if n_succeeded >= 4:
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
    else:
        iqr = spread
    total_duration = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    consensus_confidence = max(0.0, 1.0 - iqr / max(total_duration, 1e-6))

    # Flagging
    flagged = spread > max_disagree or n_succeeded < min_methods

    return EnsembleTruncationResult(
        consensus_idx=consensus_idx,
        consensus_time=consensus_time,
        consensus_confidence=consensus_confidence,
        method_results=results,
        disagreement_hours=spread,
        flagged_for_review=flagged,
        method_used=consensus_method,
        n_methods_succeeded=n_succeeded,
    )


def plot_ensemble_truncation(time, od, ensemble_result, strain_name, output_path):
    """Diagnostic plot: all methods overlaid + consensus."""
    METHOD_COLORS = {
        'first_peak': '#2ecc71',       # green
        'stationary_phase': '#3498db',  # blue
        'adaptive_r2': '#e67e22',       # orange
        'gp_derivative': '#e74c3c',     # red
        'changepoint': '#9b59b6',       # purple
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[4, 1],
                             gridspec_kw={'hspace': 0.08})

    # Top: OD data + truncation lines
    ax = axes[0]
    ax.scatter(time, od, s=8, alpha=0.3, color='gray', label='Data', zorder=1)

    for name, data in ensemble_result.method_results.items():
        color = METHOD_COLORS.get(name, 'gray')
        t_val = data['time']
        conf = data['confidence']
        ax.axvline(t_val, color=color, linestyle=':', linewidth=1.5,
                   alpha=max(0.3, conf),
                   label=f'{name} (t={t_val:.1f}h, c={conf:.2f})')

    # Consensus line
    ax.axvline(ensemble_result.consensus_time, color='black', linestyle='--',
               linewidth=2.5, label=f'Consensus (t={ensemble_result.consensus_time:.1f}h)')

    flag_str = " [FLAGGED]" if ensemble_result.flagged_for_review else ""
    ax.set_title(f'{strain_name}: Ensemble Truncation '
                 f'(conf={ensemble_result.consensus_confidence:.2f}, '
                 f'{ensemble_result.n_methods_succeeded}/5 methods){flag_str}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('OD600', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels([])

    # Bottom: bar chart of method truncation times
    ax2 = axes[1]
    names = list(ensemble_result.method_results.keys())
    t_vals = [ensemble_result.method_results[n]['time'] for n in names]
    confs = [ensemble_result.method_results[n]['confidence'] for n in names]
    colors = [METHOD_COLORS.get(n, 'gray') for n in names]
    y_pos = range(len(names))

    bars = ax2.barh(y_pos, t_vals, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=0.5, height=0.6)
    # Set per-bar alpha based on method confidence
    for bar, c in zip(bars, confs):
        bar.set_alpha(max(0.3, c))
    ax2.axvline(ensemble_result.consensus_time, color='black', linestyle='--', linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([n.replace('_', ' ').title() for n in names], fontsize=8)
    ax2.set_xlabel('Truncation Time (hours)', fontsize=10)
    ax2.set_xlim(ax.get_xlim())
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 2: Bayesian Gompertz with HMC (NUTS)
# =============================================================================

def build_gompertz_model(strain_data, strain_names, pesticide_indices,
                         n_pesticides, mle_estimates=None, config=None,
                         genomic_priors=None):
    """
    Build hierarchical Bayesian Gompertz model in PyMC.

    Uses vectorized likelihood with padded arrays to avoid creating
    N separate likelihood nodes (which overwhelms PyTensor's optimizer).

    Args:
        strain_data: list of (time_array, od_array) per strain
        strain_names: list of strain name strings
        pesticide_indices: int array mapping each strain -> pesticide group
        n_pesticides: number of pesticide groups
        mle_estimates: dict of MLE estimates for initialization
        config: pipeline config dict
        genomic_priors: optional DataFrame with per-strain genomic prior shifts.
            Expected columns: strain_id, mu_prior_mean, mu_prior_sigma
            (and optionally lam_prior_mean/sigma, a_prior_mean/sigma).
            When provided, adds per-strain shifts to the non-centered
            parameterization based on genomic predictions.

    Returns:
        PyMC model, (time_padded, od_padded, mask, strain_idx_per_obs)
    """
    import pymc as pm
    import pytensor.tensor as pt

    cfg = (config or {}).get('advanced', {}).get('priors', {})
    mu_A_prior = cfg.get('mu_A', [1.3, 0.5])
    mu_mu_prior = cfg.get('mu_mu', [0.25, 0.15])
    mu_lam_prior = cfg.get('mu_lam', [2.0, 2.0])
    sigma_obs_prior = cfg.get('sigma_obs', 0.05)

    n_strains = len(strain_data)

    # Vectorize: concatenate all observations into flat arrays with strain indices
    all_times = []
    all_ods = []
    all_strain_idx = []
    for i, (t, od) in enumerate(strain_data):
        all_times.append(t)
        all_ods.append(od)
        all_strain_idx.append(np.full(len(t), i, dtype=int))

    time_flat = np.concatenate(all_times)
    od_flat = np.concatenate(all_ods)
    strain_idx_flat = np.concatenate(all_strain_idx)

    with pm.Model() as model:
        # === Population hyperpriors ===
        mu_A = pm.Normal('mu_A', mu=mu_A_prior[0], sigma=mu_A_prior[1])
        sigma_A = pm.HalfNormal('sigma_A', sigma=0.3)

        mu_mu_pop = pm.Normal('mu_mu', mu=mu_mu_prior[0], sigma=mu_mu_prior[1])
        sigma_mu_pop = pm.HalfNormal('sigma_mu', sigma=0.1)

        mu_lam = pm.Normal('mu_lam', mu=mu_lam_prior[0], sigma=mu_lam_prior[1])
        sigma_lam = pm.HalfNormal('sigma_lam', sigma=2.0)

        # === Pesticide-group level (non-centered) ===
        A_group_offset = pm.Normal('A_group_offset', mu=0, sigma=1,
                                   shape=n_pesticides)
        A_group = pm.Deterministic('A_group', mu_A + sigma_A * A_group_offset)

        mu_group_offset = pm.Normal('mu_group_offset', mu=0, sigma=1,
                                    shape=n_pesticides)
        mu_group = pm.Deterministic('mu_group',
                                    mu_mu_pop + sigma_mu_pop * mu_group_offset)

        lam_group_offset = pm.Normal('lam_group_offset', mu=0, sigma=1,
                                     shape=n_pesticides)
        lam_group = pm.Deterministic('lam_group',
                                     mu_lam + sigma_lam * lam_group_offset)

        # === Strain level (non-centered) ===
        sigma_A_strain = pm.HalfNormal('sigma_A_strain', sigma=0.15)
        sigma_mu_strain = pm.HalfNormal('sigma_mu_strain', sigma=0.05)
        sigma_lam_strain = pm.HalfNormal('sigma_lam_strain', sigma=1.0)

        # === Genomic prior shifts (fixed data, not random variables) ===
        # When genomic priors are available, shift each strain's mean
        # based on genotype-to-phenotype predictions
        genomic_mu_shift = np.zeros(n_strains)
        genomic_lam_shift = np.zeros(n_strains)
        genomic_A_shift = np.zeros(n_strains)

        if genomic_priors is not None and len(genomic_priors) > 0:
            from genomic_features import resolve_strain_id
            for i, name in enumerate(strain_names):
                bio_id = resolve_strain_id(name)
                match = genomic_priors[genomic_priors['strain_id'] == bio_id]
                if len(match) > 0:
                    row = match.iloc[0]
                    # Shift = genomic prediction - population prior mean
                    # This moves the strain's prior center toward the genomic prediction
                    if 'mu_prior_mean' in row and not np.isnan(row['mu_prior_mean']):
                        genomic_mu_shift[i] = row['mu_prior_mean'] - mu_mu_prior[0]
                    if 'lambda_prior_mean' in row and not np.isnan(row['lambda_prior_mean']):
                        genomic_lam_shift[i] = row['lambda_prior_mean'] - mu_lam_prior[0]
                    if 'a_prior_mean' in row and not np.isnan(row['a_prior_mean']):
                        genomic_A_shift[i] = row['a_prior_mean'] - mu_A_prior[0]

            n_shifted = np.sum(genomic_mu_shift != 0)
            if n_shifted > 0:
                logger.info(f"  Applied genomic prior shifts to {n_shifted}/{n_strains} strains")

        A_offset = pm.Normal('A_offset', mu=0, sigma=1, shape=n_strains)
        A_strain = pm.Deterministic(
            'A_strain',
            A_group[pesticide_indices] + genomic_A_shift + sigma_A_strain * A_offset
        )

        mu_offset = pm.Normal('mu_offset', mu=0, sigma=1, shape=n_strains)
        mu_strain = pm.Deterministic(
            'mu_strain',
            pm.math.abs(mu_group[pesticide_indices] + genomic_mu_shift + sigma_mu_strain * mu_offset)
        )

        lam_offset = pm.Normal('lam_offset', mu=0, sigma=1, shape=n_strains)
        lam_strain = pm.Deterministic(
            'lam_strain',
            lam_group[pesticide_indices] + genomic_lam_shift + sigma_lam_strain * lam_offset
        )

        # === Observation noise ===
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=sigma_obs_prior)

        # === Vectorized likelihood (single node for all observations) ===
        # Index into strain params for each observation
        A_obs = A_strain[strain_idx_flat]
        mu_obs = mu_strain[strain_idx_flat]
        lam_obs = lam_strain[strain_idx_flat]

        # Gompertz prediction (vectorized over all observations)
        pred_flat = A_obs * pm.math.exp(
            -pm.math.exp(
                (mu_obs * np.e / A_obs) * (lam_obs - time_flat) + 1
            )
        )

        pm.Normal('obs', mu=pred_flat, sigma=sigma_obs, observed=od_flat)

    return model


def fit_bayesian_gompertz(model, config=None):
    """Sample from the Bayesian Gompertz model."""
    import pymc as pm

    cfg = (config or {}).get('advanced', {}).get('bayesian', {}).get('gompertz', {})
    draws = cfg.get('draws', 2000)
    tune = cfg.get('tune', 1000)
    chains = cfg.get('chains', 4)
    target_accept = cfg.get('target_accept', 0.90)

    with model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
        )
    return trace


# =============================================================================
# SECTION 3: Bayesian Haldane with Hierarchical Ki
# =============================================================================

def build_haldane_model(strain_data, strain_names, pesticide_indices,
                        n_pesticides, mle_estimates=None, config=None):
    """
    Build hierarchical Bayesian Haldane model in PyMC.

    Uses custom pytensor Op wrapping scipy ODE solver.
    Ki is partially pooled by pesticide group.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
    from pytensor.graph.basic import Apply

    cfg = (config or {}).get('advanced', {}).get('priors', {})
    n_strains = len(strain_data)

    # Custom Op for Haldane ODE
    class HaldaneSolveOp(Op):
        """Wraps scipy solve_ivp for use in PyMC."""

        def __init__(self, time_points):
            self.time_points = np.asarray(time_points, dtype='float64')

        def make_node(self, params):
            params = pt.as_tensor_variable(params)
            return Apply(self, [params],
                         [pt.dvector()])

        def perform(self, node, inputs, output_storage):
            p = inputs[0]
            mu_max, Ks, Ki, X_max, q, X0, S0 = p
            X_pred, _ = solve_haldane(
                self.time_points, mu_max, Ks, Ki, X_max, q, X0, S0
            )
            output_storage[0][0] = np.asarray(X_pred, dtype='float64')

    with pm.Model() as model:
        # === Population hyperpriors for Ki ===
        mu_log_Ki_val = cfg.get('mu_log_Ki', [2.0, 1.0])
        mu_log_Ki = pm.Normal('mu_log_Ki', mu=mu_log_Ki_val[0],
                              sigma=mu_log_Ki_val[1])
        sigma_log_Ki = pm.HalfNormal('sigma_log_Ki',
                                     sigma=cfg.get('sigma_log_Ki', 1.0))

        # === Pesticide-level Ki (non-centered) ===
        log_Ki_offset = pm.Normal('log_Ki_offset', mu=0, sigma=1,
                                  shape=n_pesticides)
        log_Ki_pest = pm.Deterministic(
            'log_Ki_pest', mu_log_Ki + sigma_log_Ki * log_Ki_offset
        )
        Ki_pest = pm.Deterministic('Ki_pest', pm.math.exp(log_Ki_pest))

        # === Strain-level Ki (within-pesticide variation) ===
        sigma_Ki_within = pm.HalfNormal('sigma_Ki_within', sigma=0.5)
        log_Ki_strain_offset = pm.Normal('log_Ki_strain_offset', mu=0, sigma=1,
                                         shape=n_strains)
        log_Ki_strain = pm.Deterministic(
            'log_Ki_strain',
            log_Ki_pest[pesticide_indices] + sigma_Ki_within * log_Ki_strain_offset
        )
        Ki_strain = pm.Deterministic('Ki_strain', pm.math.exp(log_Ki_strain))

        # === Strain-level other params ===
        mu_max_strain = pm.LogNormal('mu_max_strain', mu=0.0, sigma=1.0,
                                     shape=n_strains)
        Ks_strain = pm.LogNormal('Ks_strain', mu=-1.0, sigma=1.5,
                                  shape=n_strains)

        mu_Xmax = pm.Normal('mu_Xmax', mu=1.5, sigma=0.3)
        sigma_Xmax = pm.HalfNormal('sigma_Xmax', sigma=0.2)
        X_max_offset = pm.Normal('X_max_offset', mu=0, sigma=1, shape=n_strains)
        X_max_strain = pm.Deterministic(
            'X_max_strain',
            pm.math.abs(mu_Xmax + sigma_Xmax * X_max_offset) + 0.1
        )

        q_strain = pm.LogNormal('q_strain', mu=-1.0, sigma=1.0,
                                 shape=n_strains)
        X0_strain = pm.LogNormal('X0_strain', mu=-4.0, sigma=1.0,
                                  shape=n_strains)

        S0 = 1.0  # FIXED

        # === Observation noise ===
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.05)

        # === Likelihood (per strain via custom Op) ===
        for i, (t, od) in enumerate(strain_data):
            ode_op = HaldaneSolveOp(t)
            params_vec = pt.stack([
                mu_max_strain[i], Ks_strain[i], Ki_strain[i],
                X_max_strain[i], q_strain[i], X0_strain[i],
                pt.constant(S0)
            ])
            X_pred = ode_op(params_vec)
            pm.Normal(f'obs_{i}', mu=X_pred, sigma=sigma_obs, observed=od)

    return model


def fit_bayesian_haldane(model, config=None):
    """Sample from Bayesian Haldane using DEMetropolisZ (gradient-free)."""
    import pymc as pm

    cfg = (config or {}).get('advanced', {}).get('bayesian', {}).get('haldane', {})
    draws = cfg.get('draws', 5000)
    tune = cfg.get('tune', 3000)
    chains = cfg.get('chains', 4)
    sampler = cfg.get('sampler', 'DEMetropolisZ')

    with model:
        if sampler == 'DEMetropolisZ':
            step = pm.DEMetropolisZ()
        elif sampler == 'Slice':
            step = pm.Slice()
        else:
            step = None  # NUTS

        kwargs = dict(
            draws=draws, tune=tune, chains=chains,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
        )
        if step is not None:
            kwargs['step'] = step

        trace = pm.sample(**kwargs)
    return trace


# =============================================================================
# SECTION 4: Bootstrap Uncertainty
# =============================================================================

def bootstrap_gompertz(time, od, n_resamples=1000, ci_level=0.95):
    """
    Bootstrap CIs on Gompertz parameters via residual resampling.

    Returns:
        dict with keys: A_ci, mu_ci, lam_ci, r2_ci, p_good, params_all
    """
    # Initial fit
    try:
        a_init = np.max(od)
        diff = np.diff(od)
        dt = np.diff(time)
        rates = diff / np.maximum(dt, 1e-6)
        mu_init = max(np.max(rates), 0.01)
        lam_init = time[np.argmax(rates)] if len(rates) > 0 else time[0]

        popt, _ = curve_fit(
            gompertz_model, time, od,
            p0=[a_init, mu_init, lam_init],
            bounds=([0.01, 0.001, 0], [3*a_init, 10*mu_init+1, time[-1]]),
            maxfev=3000
        )
    except Exception:
        return None

    fitted = gompertz_model(time, *popt)
    residuals = od - fitted

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_params = []
    boot_r2 = []
    boot_good = 0

    for _ in range(n_resamples):
        # Resample residuals
        boot_resid = rng.choice(residuals, size=len(residuals), replace=True)
        boot_y = fitted + boot_resid

        try:
            bp, _ = curve_fit(
                gompertz_model, time, boot_y,
                p0=popt,
                bounds=([0.01, 0.001, 0], [3*popt[0]+0.5, 10*popt[1]+1, time[-1]]),
                maxfev=2000
            )
            boot_params.append(bp)

            pred = gompertz_model(time, *bp)
            ss_res = np.sum((boot_y - pred)**2)
            ss_tot = np.sum((boot_y - np.mean(boot_y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            boot_r2.append(r2)

            if r2 >= 0.95:
                boot_good += 1
        except Exception:
            continue

    if len(boot_params) < 10:
        return None

    boot_params = np.array(boot_params)
    boot_r2 = np.array(boot_r2)
    alpha = (1 - ci_level) / 2

    return {
        'A_mean': np.mean(boot_params[:, 0]),
        'A_ci': (np.quantile(boot_params[:, 0], alpha),
                 np.quantile(boot_params[:, 0], 1-alpha)),
        'mu_mean': np.mean(boot_params[:, 1]),
        'mu_ci': (np.quantile(boot_params[:, 1], alpha),
                  np.quantile(boot_params[:, 1], 1-alpha)),
        'lam_mean': np.mean(boot_params[:, 2]),
        'lam_ci': (np.quantile(boot_params[:, 2], alpha),
                   np.quantile(boot_params[:, 2], 1-alpha)),
        'r2_mean': np.mean(boot_r2),
        'r2_ci': (np.quantile(boot_r2, alpha), np.quantile(boot_r2, 1-alpha)),
        'p_good': boot_good / len(boot_params),
        'n_successful': len(boot_params),
        'mle_params': popt,
    }


# =============================================================================
# SECTION 5: Bayesian Classification
# =============================================================================

def classify_bayesian(trace, strain_idx, time, od, config=None):
    """
    Compute P(good) from Bayesian Gompertz posterior.

    For each posterior draw, compute R² and check if it passes
    classification thresholds. P(good) = fraction of draws passing.
    """
    import arviz as az

    cfg = (config or {}).get('advanced', {}).get('classification', {})
    p_high = cfg.get('p_good_high', 0.90)
    p_low = cfg.get('p_good_low', 0.50)

    # Extract posterior samples for this strain
    A_samples = trace.posterior[f'A_strain'].values[:, :, strain_idx].ravel()
    mu_samples = trace.posterior[f'mu_strain'].values[:, :, strain_idx].ravel()
    lam_samples = trace.posterior[f'lam_strain'].values[:, :, strain_idx].ravel()

    n_draws = len(A_samples)
    n_good = 0

    for j in range(n_draws):
        pred = gompertz_model(time, A_samples[j], mu_samples[j], lam_samples[j])
        ss_res = np.sum((od - pred)**2)
        ss_tot = np.sum((od - np.mean(od))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 >= 0.95:
            n_good += 1

    p_good = n_good / n_draws

    if p_good >= p_high:
        label = 'GOOD (high confidence)'
    elif p_good >= p_low:
        label = 'GOOD (moderate confidence)'
    else:
        label = 'BAD'

    borderline = cfg.get('borderline_range', [0.30, 0.70])
    is_borderline = borderline[0] <= p_good <= borderline[1]

    return {
        'p_good': p_good,
        'label': label,
        'is_borderline': is_borderline,
    }


# =============================================================================
# SECTION 6: Model Comparison (WAIC/LOO)
# =============================================================================

def compare_models(traces_dict):
    """
    Compare models using WAIC and LOO-CV via ArviZ.

    NOTE: Models must be fitted to the SAME observed data for valid comparison.
    Gompertz (all strains) vs Haldane (pesticide subset) is NOT a valid comparison
    because they have different likelihood dimensions.

    Args:
        traces_dict: {'Gompertz': trace1, 'Haldane': trace2}

    Returns:
        DataFrame with comparison metrics
    """
    import arviz as az

    try:
        loo_comp = az.compare(traces_dict, ic='loo', scale='log')
    except Exception as e:
        print(f"  LOO comparison failed (models may have different observed data): {e}")
        loo_comp = None

    try:
        waic_comp = az.compare(traces_dict, ic='waic', scale='log')
    except Exception as e:
        print(f"  WAIC comparison failed (models may have different observed data): {e}")
        waic_comp = None

    return loo_comp, waic_comp


# =============================================================================
# SECTION 7: Plotting
# =============================================================================

def plot_forest_ki(trace, pesticide_names, output_path):
    """Forest plot of Ki posteriors by pesticide."""
    import arviz as az

    ki_samples = trace.posterior['Ki_pest'].values  # (chains, draws, n_pest)
    n_pest = len(pesticide_names)

    # Compute stats and sort by median
    medians = []
    for i in range(n_pest):
        medians.append(np.median(ki_samples[:, :, i].ravel()))
    sort_idx = np.argsort(medians)

    fig, ax = plt.subplots(figsize=(10, max(4, n_pest * 0.8)))

    for rank, i in enumerate(sort_idx):
        samples = ki_samples[:, :, i].ravel()
        median = np.median(samples)
        hdi_50 = az.hdi(samples, hdi_prob=0.50)
        hdi_95 = az.hdi(samples, hdi_prob=0.95)

        ax.plot(hdi_95, [rank, rank], 'k-', linewidth=1)
        ax.plot(hdi_50, [rank, rank], 'k-', linewidth=3)
        ax.plot(median, rank, 'ko', markersize=8)

    ax.set_yticks(range(n_pest))
    ax.set_yticklabels([pesticide_names[i] for i in sort_idx])
    ax.set_xlabel('Ki (Inhibition Constant)', fontsize=12)
    ax.set_title('Substrate Inhibition by Pesticide\n'
                 '(lower Ki = stronger inhibition)\n'
                 'Thick bar = 50% HDI, Thin bar = 95% HDI',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_posterior_predictive(trace, strain_idx, time, od, strain_name,
                              model_type, output_path):
    """Overlay posterior draws on data."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(time, od, s=10, alpha=0.4, color='gray', label='Data', zorder=1)

    if model_type == 'gompertz':
        A_s = trace.posterior['A_strain'].values[:, :, strain_idx].ravel()
        mu_s = trace.posterior['mu_strain'].values[:, :, strain_idx].ravel()
        lam_s = trace.posterior['lam_strain'].values[:, :, strain_idx].ravel()

        rng = np.random.default_rng(42)
        n_draws = min(100, len(A_s))
        idx = rng.choice(len(A_s), n_draws, replace=False)

        all_preds = []
        for j in idx:
            pred = gompertz_model(time, A_s[j], mu_s[j], lam_s[j])
            ax.plot(time, pred, 'r-', alpha=0.03, zorder=2)
            all_preds.append(pred)

        all_preds = np.array(all_preds)
        mean_pred = np.mean(all_preds, axis=0)
        ax.plot(time, mean_pred, 'r-', linewidth=2, label='Posterior mean', zorder=3)

        lo = np.percentile(all_preds, 2.5, axis=0)
        hi = np.percentile(all_preds, 97.5, axis=0)
        ax.fill_between(time, lo, hi, alpha=0.15, color='red',
                        label='95% credible interval')

    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('OD600', fontsize=11)
    ax.set_title(f'{strain_name}: {model_type.title()} Posterior Predictive Check',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trace_diagnostics(trace, params, output_path):
    """Plot trace and rank plots for key parameters."""
    import arviz as az

    fig = az.plot_trace(trace, var_names=params, compact=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_bootstrap_summary(bootstrap_df, output_path):
    """Plot bootstrap P(good) and CI widths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P(good) distribution
    ax = axes[0]
    pgood = bootstrap_df['p_good'].dropna()
    ax.hist(pgood, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0.90, color='green', linestyle='--', label='High confidence')
    ax.axvline(0.50, color='orange', linestyle='--', label='Moderate confidence')
    ax.set_xlabel('P(good)', fontsize=11)
    ax.set_ylabel('Number of strains', fontsize=11)
    ax.set_title('Bootstrap Classification Confidence', fontsize=13,
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: mu CI width
    ax2 = axes[1]
    if 'mu_ci_width' in bootstrap_df.columns:
        ci_w = bootstrap_df['mu_ci_width'].dropna()
        ax2.hist(ci_w, bins=20, color='coral', edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Growth Rate (mu) 95% CI Width', fontsize=11)
        ax2.set_ylabel('Number of strains', fontsize=11)
        ax2.set_title('Parameter Uncertainty', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(loo_comp, output_path):
    """Plot model comparison results."""
    if loo_comp is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    models = loo_comp.index.tolist()
    elpd = loo_comp['elpd_loo'].values
    se = loo_comp['se'].values

    ax.barh(models, elpd, xerr=se, capsize=5, color=['steelblue', 'coral'],
            alpha=0.8)
    ax.set_xlabel('ELPD (LOO)', fontsize=11)
    ax.set_title('Model Comparison (higher = better fit)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 8: Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced statistical fitting for growth curve analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods applied:
  GP truncation       Gaussian Process growth phase detection
  Bayesian Gompertz   Hierarchical HMC (NUTS) with partial pooling
  Bayesian Haldane    Hierarchical Ki by pesticide (ODE + DEMetropolisZ)
  Bootstrap CIs       Nonparametric residual resampling
  Bayesian class.     P(good) replaces hard thresholds
  Model comparison    WAIC/LOO replaces AIC
        """)

    parser.add_argument('--results-dir', '-r', default=None)
    parser.add_argument('--data-dir', '-d', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--config', default=None)

    # Phase selectors
    parser.add_argument('--gp-only', action='store_true',
                        help='Only run GP truncation')
    parser.add_argument('--bootstrap-only', action='store_true',
                        help='Only run bootstrap CIs')
    parser.add_argument('--gompertz-only', action='store_true',
                        help='Only run Bayesian Gompertz')
    parser.add_argument('--haldane-only', action='store_true',
                        help='Only run Bayesian Haldane')
    parser.add_argument('--no-haldane', action='store_true',
                        help='Run everything except Haldane (skip slow ODE)')
    parser.add_argument('--ensemble-truncation', action='store_true',
                        help='Run ensemble truncation and feed consensus into downstream')
    parser.add_argument('--ensemble-only', action='store_true',
                        help='Only run ensemble truncation (no Bayesian/bootstrap)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Skip ensemble truncation')
    parser.add_argument('--no-genomic', action='store_true',
                        help='Skip genomic prior integration even if data exists')

    # Sampling overrides
    parser.add_argument('--chains', type=int, default=None)
    parser.add_argument('--draws', type=int, default=None)
    parser.add_argument('--tune', type=int, default=None)
    parser.add_argument('--sampler', choices=['nuts', 'DEMetropolisZ', 'Slice'],
                        default=None)

    parser.add_argument('--max-strains', type=int, default=None,
                        help='Limit number of strains for Bayesian models (faster)')
    parser.add_argument('--thin', type=int, default=50,
                        help='Thin data to N points per strain for Bayesian (default: 50)')
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    config = load_config(Path(args.config) if args.config else None)

    # Apply CLI overrides to config
    if args.chains:
        config.setdefault('advanced', {}).setdefault('bayesian', {}) \
            .setdefault('gompertz', {})['chains'] = args.chains
        config['advanced']['bayesian'].setdefault('haldane', {})['chains'] = args.chains
    if args.draws:
        config.setdefault('advanced', {}).setdefault('bayesian', {}) \
            .setdefault('gompertz', {})['draws'] = args.draws
        config['advanced']['bayesian'].setdefault('haldane', {})['draws'] = args.draws
    if args.tune:
        config.setdefault('advanced', {}).setdefault('bayesian', {}) \
            .setdefault('gompertz', {})['tune'] = args.tune
        config['advanced']['bayesian'].setdefault('haldane', {})['tune'] = args.tune
    if args.sampler:
        config.setdefault('advanced', {}).setdefault('bayesian', {}) \
            .setdefault('haldane', {})['sampler'] = args.sampler

    results_dir = Path(args.results_dir) if args.results_dir else base_dir / 'results' / 'tables'
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'data' / 'raw'

    adv_cfg = config.get('advanced', {})
    output_dir = Path(args.output) if args.output else base_dir / adv_cfg.get(
        'output_dir', 'results/tables/Advanced_Analysis')

    # Determine what to run
    any_specific = (args.gp_only or args.bootstrap_only or
                    args.gompertz_only or args.haldane_only or
                    args.ensemble_only)
    run_gp = args.gp_only or args.ensemble_only or (not any_specific)
    run_ensemble = (args.ensemble_truncation or args.ensemble_only or
                    (not any_specific and config.get('advanced', {}).get('ensemble', {}).get('enabled', False)))
    if args.no_ensemble:
        run_ensemble = False
    run_bootstrap = args.bootstrap_only or (not any_specific and not args.ensemble_only)
    run_gompertz = args.gompertz_only or (not any_specific and not args.ensemble_only)
    run_haldane = args.haldane_only or (not any_specific and not args.no_haldane and not args.ensemble_only)

    # Load pipeline results
    results_csv = results_dir / 'all_groups_results.csv'
    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found. Run main pipeline first.")
        sys.exit(1)

    all_results = pd.read_csv(results_csv)
    good_results = all_results[all_results['is_good'] == True].copy()
    pest_results = identify_pesticide_strains(good_results)

    print(f"Loaded {len(all_results)} total strains, "
          f"{len(good_results)} good fits, "
          f"{len(pest_results)} pesticide+LB strains")

    # Build pesticide index mapping
    pest_results['pesticide'] = pest_results['strain'].apply(extract_pesticide_name)
    pesticide_list = sorted(pest_results['pesticide'].unique())
    pest_to_idx = {p: i for i, p in enumerate(pesticide_list)}
    n_pesticides = len(pesticide_list)

    print(f"Pesticides ({n_pesticides}): {', '.join(pesticide_list)}")

    # Load raw data for each strain
    strain_data_good = []  # for Gompertz (all good strains)
    strain_names_good = []
    pest_indices_good = []

    strain_data_haldane = []  # for Haldane (pesticide+LB only)
    strain_names_haldane = []
    pest_indices_haldane = []

    # For good strains, assign pesticide index (or a "control" group)
    all_good = good_results.copy()
    all_good['pesticide'] = all_good['strain'].apply(extract_pesticide_name)
    # Add control groups
    control_idx = n_pesticides  # extra group for non-pesticide strains
    extended_pest_list = pesticide_list + ['CONTROL']

    for _, row in all_good.iterrows():
        t, od = load_raw_data(data_dir, row['strain'], row['group'])
        if t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(od)
        t, od = t[mask].astype(float), od[mask].astype(float)
        if len(t) < 20:
            continue
        strain_data_good.append((t, od))
        strain_names_good.append(row['strain'])
        pidx = pest_to_idx.get(row['pesticide'], control_idx)
        pest_indices_good.append(pidx)

    for _, row in pest_results.iterrows():
        t, od = load_raw_data(data_dir, row['strain'], row['group'])
        if t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(od)
        t, od = t[mask].astype(float), od[mask].astype(float)
        if len(t) < 20:
            continue
        strain_data_haldane.append((t, od))
        strain_names_haldane.append(row['strain'])
        pest_indices_haldane.append(pest_to_idx[row['pesticide']])

    pest_indices_good = np.array(pest_indices_good, dtype=int)
    pest_indices_haldane = np.array(pest_indices_haldane, dtype=int)
    n_groups_good = n_pesticides + 1  # pesticides + control

    print(f"Loaded raw data: {len(strain_data_good)} good strains, "
          f"{len(strain_data_haldane)} Haldane targets")

    # Prepare thinned/limited copies for Bayesian models (GP + bootstrap use full data)
    bayes_data_good = list(strain_data_good)
    bayes_names_good = list(strain_names_good)
    bayes_pest_idx_good = pest_indices_good.copy()

    bayes_data_haldane = list(strain_data_haldane)
    bayes_names_haldane = list(strain_names_haldane)
    bayes_pest_idx_haldane = pest_indices_haldane.copy()

    # Apply --max-strains limit
    if args.max_strains is not None:
        if len(bayes_data_good) > args.max_strains:
            print(f"  Limiting Bayesian Gompertz to {args.max_strains} strains "
                  f"(from {len(bayes_data_good)})")
            bayes_data_good = bayes_data_good[:args.max_strains]
            bayes_names_good = bayes_names_good[:args.max_strains]
            bayes_pest_idx_good = bayes_pest_idx_good[:args.max_strains]
        if len(bayes_data_haldane) > args.max_strains:
            print(f"  Limiting Bayesian Haldane to {args.max_strains} strains "
                  f"(from {len(bayes_data_haldane)})")
            bayes_data_haldane = bayes_data_haldane[:args.max_strains]
            bayes_names_haldane = bayes_names_haldane[:args.max_strains]
            bayes_pest_idx_haldane = bayes_pest_idx_haldane[:args.max_strains]

    # Apply --thin to subsample time series for Bayesian models
    if args.thin is not None and args.thin > 0:
        orig_pts = sum(len(t) for t, _ in bayes_data_good)
        bayes_data_good = thin_strain_data(bayes_data_good, max_points=args.thin)
        new_pts = sum(len(t) for t, _ in bayes_data_good)
        print(f"  Thinned Gompertz data: {orig_pts} -> {new_pts} total observations "
              f"(max {args.thin} pts/strain)")

        orig_pts_h = sum(len(t) for t, _ in bayes_data_haldane)
        bayes_data_haldane = thin_strain_data(bayes_data_haldane, max_points=args.thin)
        new_pts_h = sum(len(t) for t, _ in bayes_data_haldane)
        print(f"  Thinned Haldane data: {orig_pts_h} -> {new_pts_h} total observations "
              f"(max {args.thin} pts/strain)")

    # Remap pesticide indices to only include groups present in the limited data
    if args.max_strains is not None:
        # Recompute n_groups for Gompertz (may have fewer groups after limiting)
        unique_pest_good = np.unique(bayes_pest_idx_good)
        n_groups_good_bayes = len(unique_pest_good)
        # Build remapping: old index -> contiguous 0..n_groups-1
        pest_remap_good = {old: new for new, old in enumerate(unique_pest_good)}
        bayes_pest_idx_good = np.array([pest_remap_good[p] for p in bayes_pest_idx_good])
        extended_pest_list_bayes = [extended_pest_list[i] for i in unique_pest_good]

        unique_pest_hald = np.unique(bayes_pest_idx_haldane)
        n_pesticides_bayes = len(unique_pest_hald)
        pest_remap_hald = {old: new for new, old in enumerate(unique_pest_hald)}
        bayes_pest_idx_haldane = np.array([pest_remap_hald[p] for p in bayes_pest_idx_haldane])
        pesticide_list_bayes = [pesticide_list[i] for i in unique_pest_hald if i < len(pesticide_list)]
    else:
        n_groups_good_bayes = n_groups_good
        extended_pest_list_bayes = extended_pest_list
        n_pesticides_bayes = n_pesticides
        pesticide_list_bayes = pesticide_list

    # =========================================================================
    # Run phases
    # =========================================================================

    # --- GP Truncation ---
    if run_gp:
        print(f"\n{'='*60}")
        print("  PHASE 1: GP-BASED TRUNCATION")
        print(f"{'='*60}\n")

        gp_dir = output_dir / 'gp_truncation'
        gp_plots = gp_dir / 'plots'
        gp_dir.mkdir(parents=True, exist_ok=True)
        gp_plots.mkdir(exist_ok=True)

        gp_rows = []
        for i, (t, od) in enumerate(strain_data_good):
            name = strain_names_good[i]
            if not args.quiet:
                print(f"  [{i+1}/{len(strain_data_good)}] {name}...", end=' ')
            try:
                trunc_idx, conf, phases, gp_data = gp_truncate(t, od, config)
                gp_rows.append({
                    'strain': name,
                    'truncation_idx': trunc_idx,
                    'truncation_time': t[trunc_idx],
                    'confidence': conf,
                    'lag_end': phases.get('lag_end'),
                    'exp_peak': phases.get('exp_peak'),
                    'stat_start': phases.get('stat_start'),
                    'death_start': phases.get('death_start'),
                })
                safe = name.replace('/', '_').replace(' ', '_')
                plot_gp_truncation(t, od, trunc_idx, phases, gp_data, name,
                                   str(gp_plots / f'{safe}_gp_truncation.png'))
                if not args.quiet:
                    print(f"trunc at {t[trunc_idx]:.1f}h (conf={conf:.3f})")
            except Exception as e:
                if not args.quiet:
                    print(f"FAILED ({e})")
                gp_rows.append({'strain': name, 'truncation_idx': None})

        gp_df = pd.DataFrame(gp_rows)
        gp_df.to_csv(gp_dir / 'gp_truncation_results.csv', index=False)
        print(f"\nSaved GP results to {gp_dir / 'gp_truncation_results.csv'}")
        n_ok = gp_df['truncation_idx'].notna().sum()
        print(f"  {n_ok}/{len(gp_rows)} strains successfully truncated")

    # --- Ensemble Truncation ---
    ensemble_results = {}
    strain_data_truncated = list(strain_data_good)  # default: untruncated

    if run_ensemble:
        print(f"\n{'='*60}")
        print("  PHASE 1.5: ENSEMBLE TRUNCATION")
        print(f"{'='*60}\n")

        ens_dir = output_dir / 'ensemble_truncation'
        ens_plots = ens_dir / 'plots'
        ens_dir.mkdir(parents=True, exist_ok=True)
        ens_plots.mkdir(exist_ok=True)

        ens_rows = []
        strain_data_truncated = []

        for i, (t, od) in enumerate(strain_data_good):
            name = strain_names_good[i]
            if not args.quiet:
                print(f"  [{i+1}/{len(strain_data_good)}] {name}...", end=' ')

            try:
                result = ensemble_truncate(t, od, config)
                ensemble_results[name] = result

                # Build truncated data
                cidx = result.consensus_idx
                strain_data_truncated.append((t[:cidx+1], od[:cidx+1]))

                # Build CSV row
                row = {
                    'strain': name,
                    'consensus_idx': result.consensus_idx,
                    'consensus_time': result.consensus_time,
                    'consensus_confidence': result.consensus_confidence,
                    'n_methods_succeeded': result.n_methods_succeeded,
                    'disagreement_hours': result.disagreement_hours,
                    'flagged_for_review': result.flagged_for_review,
                    'consensus_method': result.method_used,
                }
                for method_name, mdata in result.method_results.items():
                    row[f'{method_name}_idx'] = mdata.get('idx')
                    row[f'{method_name}_time'] = mdata.get('time')
                    row[f'{method_name}_confidence'] = mdata.get('confidence')
                ens_rows.append(row)

                # Plot
                ens_cfg = config.get('advanced', {}).get('ensemble', {})
                if ens_cfg.get('output_plots', True):
                    safe = name.replace('/', '_').replace(' ', '_')
                    plot_ensemble_truncation(
                        t, od, result, name,
                        str(ens_plots / f'{safe}_ensemble_truncation.png')
                    )

                if not args.quiet:
                    flag = " [FLAGGED]" if result.flagged_for_review else ""
                    print(f"consensus at {result.consensus_time:.1f}h "
                          f"(conf={result.consensus_confidence:.2f}, "
                          f"{result.n_methods_succeeded}/5 methods){flag}")
            except Exception as e:
                if not args.quiet:
                    print(f"FAILED ({e})")
                strain_data_truncated.append((t, od))  # fallback
                ens_rows.append({'strain': name, 'consensus_idx': None})

        # Save CSV outputs
        ens_df = pd.DataFrame(ens_rows)
        ens_df.to_csv(ens_dir / 'ensemble_truncation_results.csv', index=False)

        flagged = ens_df[ens_df.get('flagged_for_review', pd.Series(dtype=bool)) == True]
        if len(flagged) > 0:
            flagged.to_csv(ens_dir / 'flagged_for_review.csv', index=False)

        n_ok = ens_df['consensus_idx'].notna().sum()
        n_flagged = len(flagged)
        mean_disagree = ens_df['disagreement_hours'].mean() if 'disagreement_hours' in ens_df else 0
        print(f"\nEnsemble truncation: {n_ok}/{len(ens_rows)} strains")
        print(f"  Flagged for review: {n_flagged}")
        print(f"  Mean disagreement: {mean_disagree:.2f} hours")
        print(f"  Results saved to {ens_dir}")

    # --- Re-wire Bayesian data with ensemble-truncated data ---
    if run_ensemble and len(ensemble_results) > 0:
        # Rebuild bayes_data_good from truncated data
        bayes_data_good = list(strain_data_truncated)
        bayes_names_good = list(strain_names_good)
        bayes_pest_idx_good = pest_indices_good.copy()

        # Truncate Haldane strains using ensemble results (match by name)
        for i, name in enumerate(strain_names_haldane):
            if name in ensemble_results:
                cidx = ensemble_results[name].consensus_idx
                t_h, od_h = strain_data_haldane[i]
                # Map consensus time to the Haldane strain's time array
                # (they share same raw data, but just in case of slight differences)
                ens_time = ensemble_results[name].consensus_time
                haldane_idx = np.searchsorted(t_h, ens_time, side='right')
                haldane_idx = min(haldane_idx, len(t_h) - 1)
                bayes_data_haldane[i] = (t_h[:haldane_idx+1], od_h[:haldane_idx+1])
        bayes_names_haldane = list(strain_names_haldane)
        bayes_pest_idx_haldane = pest_indices_haldane.copy()

        # Re-apply --max-strains limit
        if args.max_strains is not None:
            if len(bayes_data_good) > args.max_strains:
                print(f"  Re-limiting Bayesian Gompertz to {args.max_strains} strains "
                      f"(from {len(bayes_data_good)})")
                bayes_data_good = bayes_data_good[:args.max_strains]
                bayes_names_good = bayes_names_good[:args.max_strains]
                bayes_pest_idx_good = bayes_pest_idx_good[:args.max_strains]
            if len(bayes_data_haldane) > args.max_strains:
                print(f"  Re-limiting Bayesian Haldane to {args.max_strains} strains "
                      f"(from {len(bayes_data_haldane)})")
                bayes_data_haldane = bayes_data_haldane[:args.max_strains]
                bayes_names_haldane = bayes_names_haldane[:args.max_strains]
                bayes_pest_idx_haldane = bayes_pest_idx_haldane[:args.max_strains]

        # Re-apply --thin
        if args.thin is not None and args.thin > 0:
            orig_pts = sum(len(t) for t, _ in bayes_data_good)
            bayes_data_good = thin_strain_data(bayes_data_good, max_points=args.thin)
            new_pts = sum(len(t) for t, _ in bayes_data_good)
            print(f"  Re-thinned Gompertz data: {orig_pts} -> {new_pts} total observations")

            orig_pts_h = sum(len(t) for t, _ in bayes_data_haldane)
            bayes_data_haldane = thin_strain_data(bayes_data_haldane, max_points=args.thin)
            new_pts_h = sum(len(t) for t, _ in bayes_data_haldane)
            print(f"  Re-thinned Haldane data: {orig_pts_h} -> {new_pts_h} total observations")

        # Re-remap pesticide indices
        if args.max_strains is not None:
            unique_pest_good = np.unique(bayes_pest_idx_good)
            n_groups_good_bayes = len(unique_pest_good)
            pest_remap_good = {old: new for new, old in enumerate(unique_pest_good)}
            bayes_pest_idx_good = np.array([pest_remap_good[p] for p in bayes_pest_idx_good])
            extended_pest_list_bayes = [extended_pest_list[i] for i in unique_pest_good]

            unique_pest_hald = np.unique(bayes_pest_idx_haldane)
            n_pesticides_bayes = len(unique_pest_hald)
            pest_remap_hald = {old: new for new, old in enumerate(unique_pest_hald)}
            bayes_pest_idx_haldane = np.array([pest_remap_hald[p] for p in bayes_pest_idx_haldane])
            pesticide_list_bayes = [pesticide_list[i] for i in unique_pest_hald if i < len(pesticide_list)]

        print(f"  Bayesian models will use ensemble-truncated data "
              f"({sum(len(t) for t, _ in bayes_data_good)} Gompertz obs, "
              f"{sum(len(t) for t, _ in bayes_data_haldane)} Haldane obs)")

    # --- Bootstrap CIs ---
    if run_bootstrap:
        print(f"\n{'='*60}")
        print("  PHASE 4: BOOTSTRAP UNCERTAINTY")
        print(f"{'='*60}\n")

        boot_dir = output_dir / 'bootstrap'
        boot_dir.mkdir(parents=True, exist_ok=True)

        boot_cfg = adv_cfg.get('bootstrap', {})
        n_resamples = boot_cfg.get('n_resamples', 1000)
        ci_level = boot_cfg.get('ci_level', 0.95)

        boot_data = strain_data_truncated if run_ensemble else strain_data_good
        boot_rows = []
        for i, (t, od) in enumerate(boot_data):
            name = strain_names_good[i]
            if not args.quiet:
                print(f"  [{i+1}/{len(boot_data)}] {name}...", end=' ')

            result = bootstrap_gompertz(t, od, n_resamples=n_resamples,
                                         ci_level=ci_level)
            if result is not None:
                boot_rows.append({
                    'strain': name,
                    'A_mean': result['A_mean'],
                    'A_ci_low': result['A_ci'][0],
                    'A_ci_high': result['A_ci'][1],
                    'mu_mean': result['mu_mean'],
                    'mu_ci_low': result['mu_ci'][0],
                    'mu_ci_high': result['mu_ci'][1],
                    'mu_ci_width': result['mu_ci'][1] - result['mu_ci'][0],
                    'lam_mean': result['lam_mean'],
                    'lam_ci_low': result['lam_ci'][0],
                    'lam_ci_high': result['lam_ci'][1],
                    'r2_mean': result['r2_mean'],
                    'r2_ci_low': result['r2_ci'][0],
                    'r2_ci_high': result['r2_ci'][1],
                    'p_good': result['p_good'],
                    'n_successful': result['n_successful'],
                })
                if not args.quiet:
                    print(f"mu={result['mu_mean']:.3f} "
                          f"[{result['mu_ci'][0]:.3f}, {result['mu_ci'][1]:.3f}] "
                          f"P(good)={result['p_good']:.2f}")
            else:
                boot_rows.append({'strain': name, 'p_good': None})
                if not args.quiet:
                    print("FAILED")

        boot_df = pd.DataFrame(boot_rows)
        boot_df.to_csv(boot_dir / 'bootstrap_ci.csv', index=False)

        # Classification table
        class_rows = []
        cls_cfg = adv_cfg.get('classification', {})
        for _, row in boot_df.iterrows():
            pg = row.get('p_good')
            if pg is None:
                label = 'FAILED'
                borderline = False
            elif pg >= cls_cfg.get('p_good_high', 0.90):
                label = 'GOOD (high confidence)'
                borderline = False
            elif pg >= cls_cfg.get('p_good_low', 0.50):
                label = 'GOOD (moderate confidence)'
                borderline = True
            else:
                label = 'BAD'
                borderline = pg is not None and pg >= 0.30
            class_rows.append({
                'strain': row['strain'],
                'p_good': pg,
                'label': label,
                'is_borderline': borderline,
            })
        class_df = pd.DataFrame(class_rows)
        class_df.to_csv(boot_dir / 'bootstrap_classification.csv', index=False)

        plot_bootstrap_summary(boot_df, boot_dir / 'bootstrap_summary.png')

        n_ok = boot_df['p_good'].notna().sum()
        n_high = (boot_df['p_good'] >= 0.90).sum()
        n_border = ((boot_df['p_good'] >= 0.30) & (boot_df['p_good'] <= 0.70)).sum()
        print(f"\nBootstrap results: {n_ok} strains analyzed")
        print(f"  High confidence GOOD: {n_high}")
        print(f"  Borderline: {n_border}")
        print(f"Saved to {boot_dir}")

    # --- Bayesian Gompertz ---
    gompertz_trace = None
    if run_gompertz:
        print(f"\n{'='*60}")
        print("  PHASE 2: BAYESIAN GOMPERTZ (NUTS)")
        print(f"{'='*60}\n")

        gomp_dir = output_dir / 'bayesian_gompertz'
        gomp_plots = gomp_dir / 'plots'
        gomp_dir.mkdir(parents=True, exist_ok=True)
        gomp_plots.mkdir(exist_ok=True)

        print(f"Building hierarchical model: {len(bayes_data_good)} strains, "
              f"{n_groups_good_bayes} groups "
              f"({sum(len(t) for t, _ in bayes_data_good)} total obs)...")

        # Load genomic priors if available
        genomic_priors = None
        genomic_priors_path = output_dir.parent / 'Genomic_Analysis' / 'genomic_priors.csv'
        if genomic_priors_path.exists() and not args.no_genomic:
            try:
                genomic_priors = pd.read_csv(genomic_priors_path)
                print(f"  Loaded genomic priors from {genomic_priors_path} "
                      f"({len(genomic_priors)} strains)")
            except Exception as e:
                print(f"  Warning: could not load genomic priors: {e}")

        try:
            gomp_model = build_gompertz_model(
                bayes_data_good, bayes_names_good,
                bayes_pest_idx_good, n_groups_good_bayes,
                config=config, genomic_priors=genomic_priors
            )
            print("Model built. Sampling...")
            gompertz_trace = fit_bayesian_gompertz(gomp_model, config)

            # Save trace
            import arviz as az
            gompertz_trace.to_netcdf(str(gomp_dir / 'gompertz_trace.nc'))

            # Extract posterior summaries
            summary_rows = []
            for i, name in enumerate(bayes_names_good):
                A_s = gompertz_trace.posterior['A_strain'].values[:, :, i].ravel()
                mu_s = gompertz_trace.posterior['mu_strain'].values[:, :, i].ravel()
                lam_s = gompertz_trace.posterior['lam_strain'].values[:, :, i].ravel()

                A_hdi = az.hdi(A_s, hdi_prob=0.95)
                mu_hdi = az.hdi(mu_s, hdi_prob=0.95)
                lam_hdi = az.hdi(lam_s, hdi_prob=0.95)

                summary_rows.append({
                    'strain': name,
                    'A_mean': np.mean(A_s), 'A_median': np.median(A_s),
                    'A_hdi_low': A_hdi[0], 'A_hdi_high': A_hdi[1],
                    'mu_mean': np.mean(mu_s), 'mu_median': np.median(mu_s),
                    'mu_hdi_low': mu_hdi[0], 'mu_hdi_high': mu_hdi[1],
                    'lam_mean': np.mean(lam_s), 'lam_median': np.median(lam_s),
                    'lam_hdi_low': lam_hdi[0], 'lam_hdi_high': lam_hdi[1],
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(gomp_dir / 'gompertz_posterior_summary.csv', index=False)

            # Group-level summaries
            group_rows = []
            for j, pname in enumerate(extended_pest_list_bayes):
                A_g = gompertz_trace.posterior['A_group'].values[:, :, j].ravel()
                mu_g = gompertz_trace.posterior['mu_group'].values[:, :, j].ravel()
                lam_g = gompertz_trace.posterior['lam_group'].values[:, :, j].ravel()
                group_rows.append({
                    'group': pname,
                    'A_group_mean': np.mean(A_g),
                    'A_group_hdi_low': az.hdi(A_g, hdi_prob=0.95)[0],
                    'A_group_hdi_high': az.hdi(A_g, hdi_prob=0.95)[1],
                    'mu_group_mean': np.mean(mu_g),
                    'mu_group_hdi_low': az.hdi(mu_g, hdi_prob=0.95)[0],
                    'mu_group_hdi_high': az.hdi(mu_g, hdi_prob=0.95)[1],
                    'lam_group_mean': np.mean(lam_g),
                    'lam_group_hdi_low': az.hdi(lam_g, hdi_prob=0.95)[0],
                    'lam_group_hdi_high': az.hdi(lam_g, hdi_prob=0.95)[1],
                })
            group_df = pd.DataFrame(group_rows)
            group_df.to_csv(gomp_dir / 'gompertz_group_summary.csv', index=False)

            # Convergence diagnostics with threshold checks
            rhat = az.rhat(gompertz_trace)
            ess = az.ess(gompertz_trace)
            conv_cfg = config.get('advanced', {}).get('convergence', {})
            max_rhat = conv_cfg.get('max_rhat', 1.05)
            min_ess = conv_cfg.get('min_ess_bulk', 400)
            conv_data = {}
            convergence_warnings = []
            for var in ['mu_A', 'sigma_A', 'mu_mu', 'sigma_mu', 'sigma_obs']:
                try:
                    rhat_val = float(rhat[var].values)
                    ess_val = float(ess[var].values)
                    conv_data[f'{var}_rhat'] = rhat_val
                    conv_data[f'{var}_ess'] = ess_val
                    if rhat_val > max_rhat:
                        convergence_warnings.append(f"{var}: R-hat={rhat_val:.3f} > {max_rhat}")
                    if ess_val < min_ess:
                        convergence_warnings.append(f"{var}: ESS={ess_val:.0f} < {min_ess}")
                except Exception:
                    pass
            if convergence_warnings:
                import warnings
                warnings.warn("Bayesian Gompertz convergence issues:\n  " + "\n  ".join(convergence_warnings))
            pd.DataFrame([conv_data]).to_csv(gomp_dir / 'gompertz_convergence.csv',
                                              index=False)

            # Trace diagnostics plot
            try:
                plot_trace_diagnostics(
                    gompertz_trace,
                    ['mu_A', 'sigma_A', 'mu_mu', 'sigma_obs'],
                    str(gomp_dir / 'plots' / 'gompertz_trace_diag.png')
                )
            except Exception:
                pass

            # Posterior predictive plots (first 5 strains)
            for i in range(min(5, len(bayes_data_good))):
                try:
                    t, od = bayes_data_good[i]
                    safe = bayes_names_good[i].replace('/', '_').replace(' ', '_')
                    plot_posterior_predictive(
                        gompertz_trace, i, t, od, bayes_names_good[i],
                        'gompertz',
                        str(gomp_plots / f'{safe}_gompertz_ppc.png')
                    )
                except Exception:
                    pass

            # Bayesian classification
            print("\nComputing Bayesian classification...")
            bayes_class_dir = output_dir / 'classification'
            bayes_class_dir.mkdir(parents=True, exist_ok=True)

            bayes_class_rows = []
            for i, (t, od) in enumerate(bayes_data_good):
                cls = classify_bayesian(gompertz_trace, i, t, od, config)
                bayes_class_rows.append({
                    'strain': bayes_names_good[i],
                    'p_good': cls['p_good'],
                    'label': cls['label'],
                    'is_borderline': cls['is_borderline'],
                })
            bayes_class_df = pd.DataFrame(bayes_class_rows)
            bayes_class_df.to_csv(bayes_class_dir / 'bayesian_classification.csv',
                                   index=False)

            borderline = bayes_class_df[bayes_class_df['is_borderline']]
            if len(borderline) > 0:
                borderline.to_csv(bayes_class_dir / 'borderline_strains.csv',
                                   index=False)

            print(f"\nBayesian Gompertz complete!")
            print(f"  Posterior summaries: {gomp_dir / 'gompertz_posterior_summary.csv'}")
            n_high = (bayes_class_df['p_good'] >= 0.90).sum()
            n_border = bayes_class_df['is_borderline'].sum()
            print(f"  Classification: {n_high} high confidence, {n_border} borderline")

        except Exception as e:
            print(f"\nERROR in Bayesian Gompertz: {e}")
            import traceback
            traceback.print_exc()

    # --- Bayesian Haldane ---
    haldane_trace = None
    if run_haldane and len(bayes_data_haldane) > 0:
        print(f"\n{'='*60}")
        print("  PHASE 3: BAYESIAN HALDANE (DEMetropolisZ)")
        print(f"{'='*60}\n")

        hald_dir = output_dir / 'bayesian_haldane'
        hald_plots = hald_dir / 'plots'
        hald_dir.mkdir(parents=True, exist_ok=True)
        hald_plots.mkdir(exist_ok=True)

        print(f"Building hierarchical Haldane model: "
              f"{len(bayes_data_haldane)} strains, {n_pesticides_bayes} pesticides "
              f"({sum(len(t) for t, _ in bayes_data_haldane)} total obs)...")

        try:
            hald_model = build_haldane_model(
                bayes_data_haldane, bayes_names_haldane,
                bayes_pest_idx_haldane, n_pesticides_bayes,
                config=config
            )
            print("Model built. Sampling (this may take a while)...")
            haldane_trace = fit_bayesian_haldane(hald_model, config)

            import arviz as az
            haldane_trace.to_netcdf(str(hald_dir / 'haldane_trace.nc'))

            # Per-strain posterior summary
            h_rows = []
            for i, name in enumerate(bayes_names_haldane):
                Ki_s = haldane_trace.posterior['Ki_strain'].values[:, :, i].ravel()
                mu_s = haldane_trace.posterior['mu_max_strain'].values[:, :, i].ravel()

                Ki_hdi = az.hdi(Ki_s, hdi_prob=0.95)
                mu_hdi = az.hdi(mu_s, hdi_prob=0.95)

                h_rows.append({
                    'strain': name,
                    'pesticide': extract_pesticide_name(name),
                    'Ki_mean': np.mean(Ki_s), 'Ki_median': np.median(Ki_s),
                    'Ki_hdi_low': Ki_hdi[0], 'Ki_hdi_high': Ki_hdi[1],
                    'mu_max_mean': np.mean(mu_s), 'mu_max_median': np.median(mu_s),
                    'mu_max_hdi_low': mu_hdi[0], 'mu_max_hdi_high': mu_hdi[1],
                })

            h_df = pd.DataFrame(h_rows)
            h_df.to_csv(hald_dir / 'haldane_posterior_summary.csv', index=False)

            # Per-pesticide Ki summary
            ki_rows = []
            for j, pname in enumerate(pesticide_list_bayes):
                Ki_p = haldane_trace.posterior['Ki_pest'].values[:, :, j].ravel()
                Ki_hdi = az.hdi(Ki_p, hdi_prob=0.95)
                ki_rows.append({
                    'pesticide': pname,
                    'Ki_mean': np.mean(Ki_p),
                    'Ki_median': np.median(Ki_p),
                    'Ki_hdi_low': Ki_hdi[0],
                    'Ki_hdi_high': Ki_hdi[1],
                    'Ki_sd': np.std(Ki_p),
                    'P_Ki_lt_5': np.mean(Ki_p < 5),
                    'P_Ki_lt_10': np.mean(Ki_p < 10),
                    'n_strains': np.sum(bayes_pest_idx_haldane == j),
                })
            ki_df = pd.DataFrame(ki_rows)
            ki_df.to_csv(hald_dir / 'haldane_Ki_by_pesticide.csv', index=False)

            # Forest plot
            plot_forest_ki(haldane_trace, pesticide_list_bayes,
                          str(hald_plots / 'haldane_forest_Ki.png'))

            print(f"\nBayesian Haldane complete!")
            print(f"  Ki by pesticide:")
            for _, r in ki_df.iterrows():
                print(f"    {r['pesticide']:<25s} Ki={r['Ki_median']:.1f} "
                      f"[{r['Ki_hdi_low']:.1f}, {r['Ki_hdi_high']:.1f}] "
                      f"P(Ki<10)={r['P_Ki_lt_10']:.2f}")

        except Exception as e:
            print(f"\nERROR in Bayesian Haldane: {e}")
            import traceback
            traceback.print_exc()

    # --- Model Comparison ---
    if gompertz_trace is not None and haldane_trace is not None:
        print(f"\n{'='*60}")
        print("  PHASE 6: MODEL COMPARISON (WAIC/LOO)")
        print(f"{'='*60}\n")

        try:
            loo_comp, waic_comp = compare_models({
                'Gompertz': gompertz_trace,
                'Haldane': haldane_trace,
            })

            if loo_comp is not None:
                loo_comp.to_csv(output_dir / 'model_comparison.csv')
                plot_model_comparison(loo_comp,
                                     str(output_dir / 'model_comparison_plot.png'))
                print("LOO Comparison:")
                print(loo_comp.to_string())
        except Exception as e:
            print(f"Model comparison error: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print("  ADVANCED ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
