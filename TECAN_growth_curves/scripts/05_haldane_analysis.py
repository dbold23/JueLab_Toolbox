#!/usr/bin/env python3
"""
Haldane Feedback Inhibition Analysis

Fits the Haldane/Andrews substrate inhibition model to pesticide treatment
growth curves and compares with Gompertz fits from the main pipeline.

The Haldane model is mechanistic:
    dX/dt = mu(S) * X * (1 - X/X_max)
    dS/dt = -q * mu(S) * X
    mu(S) = mu_max * S / (Ks + S + S^2/Ki)

Key outputs:
    - haldane_comparison.csv: Per-strain Gompertz vs Haldane comparison
    - haldane_summary.csv: Per-pesticide summary with Ki rankings
    - plots/: Dual-panel comparison plots (biomass + substrate depletion)
    - haldane_overview.png: Summary dashboard

Usage:
    python 05_haldane_analysis.py [--results-dir DIR] [--output DIR]

BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# The package next to scripts/ holds the shared helpers (curve lookup) and the
# loader for the validated core in 01_growth_curve_analysis.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tecan_growth.legacy_data import find_curve  # noqa: E402
from tecan_growth.legacy import growth_curve_analysis as _load_gca  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)


def _gca():
    """The validated core (01_growth_curve_analysis.py), loaded once on demand."""
    return _load_gca()


# =============================================================================
# Haldane Model (self-contained, no dependency on 03_mle_model_fitting.py)
# =============================================================================

def haldane_ode(t, y, mu_max, Ks, Ki, X_max, q):
    """
    Coupled ODE: biomass growth + substrate depletion with Haldane kinetics.

    dX/dt = mu(S) * X * (1 - X/X_max)
    dS/dt = -q * mu(S) * X

    where mu(S) = mu_max * S / (Ks + S + S^2/Ki)
    """
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


def solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0, return_status=False,
                  rtol=1e-8, atol=1e-10, max_step=0.5):
    """
    Solve the Haldane ODE system. Returns X(t), S(t).

    When the integrator fails, flat curves at the initial conditions are
    returned. That fallback is indistinguishable from a genuinely flat
    solution, which is how a failed solve used to be reported as a successful
    fit; pass ``return_status=True`` to get a third value that is False
    whenever the fallback is what you are holding.

    The solver accuracy is exposed so a fitting loop can run at a tolerance
    suited to OD data (see FIT_SOLVER_OPTIONS) instead of paying for eight
    digits on every one of thousands of trial solutions. The defaults are
    unchanged, so every existing caller integrates exactly as before.
    """
    try:
        sol = solve_ivp(
            haldane_ode,
            [time[0], time[-1]],
            [X0, S0],
            args=(mu_max, Ks, Ki, X_max, q),
            t_eval=time,
            method='RK45',
            max_step=max_step,
            rtol=rtol,
            atol=atol
        )
        if sol.success and sol.y.shape[1] == len(time):
            return (sol.y[0], sol.y[1], True) if return_status else (sol.y[0], sol.y[1])
    except Exception as e:
        import warnings
        warnings.warn(f"Haldane ODE solver failed: {e}")
    flat = (np.full_like(time, X0, dtype=float), np.full_like(time, S0, dtype=float))
    return flat + (False,) if return_status else flat


# Multi-start values for Ki. L-BFGS-B on this objective barely moves Ki away
# from where it starts, so a single start reports its own initial guess.
KI_MULTISTART = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

# ODE accuracy used while searching (the reported fit is re-solved at the
# default tolerances). 1e-6 relative is four orders of magnitude finer than the
# OD noise these curves carry and integrates about three times faster, which is
# what makes a multi-start plus a profile affordable.
FIT_SOLVER_OPTIONS = {'rtol': 1e-6, 'atol': 1e-9, 'max_step': np.inf}

# Added to the SSE when the ODE solver fell back to a flat curve, so a failed
# solve can never win the optimisation.
SOLVER_FAILURE_PENALTY = 1e6

# Ki counts as identified only if the SSE moves by at least this much (percent)
# across the profile over 0.1x-10x the fitted value.
KI_PROFILE_MIN_SSE_CHANGE_PCT = 1.0


def fit_haldane(time, od, S0=1.0, gompertz_params=None,
                ki_starts=KI_MULTISTART, profile_ki=True):
    """
    Fit the Haldane model to OD data by least squares, with S0 held FIXED.

    S0 used to be fitted alongside the other six parameters, which makes Ki
    unidentifiable: the curve only ever samples mu(S) along one trajectory, so
    a change in Ki can be absorbed by rescaling S0 (with Ks and q). Every
    strain came back with Ki within a hair of the initial guess of 10.0 and the
    per-pesticide "inhibition ranking" ranked nothing. S0 is the experiment's
    dosing unit, known up to the arbitrary scale the whole model is written in,
    so it is fixed at the value passed in (--s0) and six parameters are fitted.

    The search itself runs on log(parameter) — with a linear Ki the optimiser's
    finite-difference step is far too small to move a parameter of size 10, so
    Ki never left its starting value at all (see the comment on log_bounds).

    Ki is still only weakly constrained by some curves, so the fit is repeated
    from several Ki starting values (best SSE wins) and the winner is profiled:
    Ki is pinned across 0.1x-10x its fitted value with everything else
    re-optimised. If the SSE never moves by 1%, the data cannot distinguish
    those values -- ``Ki_identified`` is False and ``Ki`` is NaN, because
    reporting the number the optimiser happened to stop at would be inventing
    a measurement. ``Ki_fitted`` keeps that number for the audit trail.

    Args:
        time: Time array (hours)
        od: OD600 values
        S0: Initial substrate concentration (arbitrary units), held fixed
        gompertz_params: Dict with 'mu' (and optionally 'a', 'lambda') used
            only to seed mu_max
        ki_starts: Ki values to start the optimiser from
        profile_ki: Run the identifiability profile (set False to skip its cost)

    Returns:
        Dict with fit results. 'success' is False when the fit failed *or* when
        the ODE solver fell back to a flat curve at the optimum.
    """
    time = np.asarray(time, dtype=float)
    od = np.asarray(od, dtype=float)
    S0 = float(S0)

    X0_init = max(float(np.mean(od[:3])), 0.001)
    X_max_init = float(np.max(od)) * 1.1

    if gompertz_params:
        mu_init = gompertz_params.get('mu', 0.2) * 2.0
    else:
        diff = np.diff(od)
        dt = np.diff(time)
        mu_init = max(float(np.max(diff / np.maximum(dt, 1e-6))), 0.01) * 2.0

    # Free parameters: mu_max, Ks, Ki, X_max, q, X0  (S0 is fixed)
    bounds = [
        (0.001, 5.0),    # mu_max
        (0.001, 10.0),   # Ks
        (0.1, 1000.0),   # Ki
        (0.05, 5.0),     # X_max
        (0.001, 10.0),   # q
        (1e-6, 0.5),     # X0
    ]
    KI = 2  # index of Ki in the parameter vector

    # Every parameter is positive and they span very different scales (Ki alone
    # runs 0.1-1000), so the search runs on log(parameter). L-BFGS-B takes its
    # finite differences with an ABSOLUTE step of ~1e-8: on a linear Ki of 10
    # that is a relative step of 1e-9, the gradient underflows to zero and the
    # optimiser leaves Ki exactly where it started -- which is why every strain
    # reported Ki = 10.0, the initial guess. In log space the same step is a
    # 1e-8 relative move for every parameter.
    log_bounds = [(float(np.log(lo)), float(np.log(hi))) for lo, hi in bounds]

    def _clip_log(params):
        return np.array([
            float(np.clip(np.log(v), lo, hi))
            for v, (lo, hi) in zip(params, log_bounds)
        ])

    def _predict(params, **solver_options):
        mu_max, Ks, Ki, X_max, q, X0 = params
        return solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0,
                             return_status=True, **solver_options)

    def objective(log_params):
        params = np.exp(np.asarray(log_params, dtype=float))
        X_pred, _S_pred, solved = _predict(params, **FIT_SOLVER_OPTIONS)
        sse = float(np.sum((od - X_pred) ** 2))
        return sse if solved else sse + SOLVER_FAILURE_PENALTY

    def _minimise(x0, maxiter=5000):
        return minimize(objective, _clip_log(x0), method='L-BFGS-B',
                        bounds=log_bounds, options={'maxiter': maxiter})

    try:
        # ---- Multi-start over Ki -------------------------------------------
        best = None
        for ki0 in ki_starts:
            x0 = [mu_init, 0.1, float(ki0), X_max_init, 0.1, X0_init]
            try:
                result = _minimise(x0)
            except Exception:
                continue
            if not np.isfinite(result.fun):
                continue
            if best is None or result.fun < best.fun:
                best = result

        if best is None:
            return {'success': False, 'r_squared': float('nan'),
                    'aic': np.inf, 'bic': np.inf,
                    'Ki_identified': False,
                    'error': 'all Ki starts failed'}

        best_params = np.exp(best.x)
        mu_max, Ks, Ki, X_max, q, X0_fit = best_params
        X_pred, S_pred, solved = _predict(best_params)

        if not solved:
            # The ODE solver fell back to a flat curve at the optimum: there is
            # no fitted model here, however small the residuals happen to be.
            return {'success': False, 'r_squared': float('nan'),
                    'aic': np.inf, 'bic': np.inf,
                    'Ki_identified': False,
                    'error': 'ODE solver failed at the optimum'}

        ki_centre = float(Ki)  # the Ki the profile below is centred on

        # ---- Profile likelihood over Ki ------------------------------------
        found_better_on_profile = False
        # Ki is pinned at a grid of values spanning 0.1x-10x the fitted one and
        # everything else is re-optimised. The SSE has to climb on BOTH sides:
        # a profile that only rises towards small Ki means the data set a lower
        # bound on the inhibition constant and nothing more, which is not a
        # measurement of Ki however confidently the optimiser stopped there.
        ki_identified = False
        profile_range_pct = float('nan')
        profile = []
        if profile_ki:
            free_bounds = [b for i, b in enumerate(log_bounds) if i != KI]
            free_best = np.delete(best.x, KI)
            lo = max(bounds[KI][0], 0.1 * ki_centre)
            hi = min(bounds[KI][1], 10.0 * ki_centre)

            def _profile_point(ki_fixed, warm_start):
                """Re-optimise everything but Ki, warm-started. Returns (sse, x)."""
                log_ki = float(np.log(ki_fixed))

                def objective_at_ki(free):
                    return objective(np.insert(free, KI, log_ki))

                try:
                    res = minimize(objective_at_ki, warm_start, method='L-BFGS-B',
                                   bounds=free_bounds, options={'maxiter': 500})
                except Exception:
                    return None, warm_start
                if not np.isfinite(res.fun) or res.fun >= SOLVER_FAILURE_PENALTY:
                    return None, warm_start
                return float(res.fun), res.x

            if hi > lo:
                # Walk outwards from the optimum in both directions, warm
                # starting each step from the neighbour that was just solved.
                # Re-optimising every point from the optimum instead makes far
                # points look worse than they are and turns a flat profile into
                # a false claim of identifiability.
                grid = list(np.geomspace(lo, hi, 5))
                downhill = sorted([k for k in grid if k < ki_centre], reverse=True)
                uphill = sorted([k for k in grid if k > ki_centre])
                for direction in (downhill, uphill):
                    warm = free_best
                    for ki_fixed in direction:
                        sse_here, warm = _profile_point(ki_fixed, warm)
                        if sse_here is not None:
                            profile.append((float(ki_fixed), sse_here))
                            # A profile point that beats the multi-start
                            # optimum means the search stopped in a local
                            # minimum; take the better fit.
                            if sse_here < float(best.fun):
                                best_params = np.exp(np.insert(warm, KI, np.log(ki_fixed)))
                                best.x = np.log(best_params)
                                best.fun = sse_here
                                found_better_on_profile = True

            if profile:
                # Reference: the best SSE seen anywhere on the profile, so a
                # profile point that beats the multi-start optimum (a local
                # minimum) does not make Ki look identified.
                # Compared against the search-tolerance SSE at the optimum,
                # so profile and optimum are measured on the same solver.
                sse_ref = min(float(best.fun), min(s for _, s in profile))
                if sse_ref > 0:
                    rises = {
                        ki_fixed: (sse - sse_ref) / sse_ref * 100.0
                        for ki_fixed, sse in profile
                    }
                    below = [r for ki_fixed, r in rises.items() if ki_fixed < ki_centre * 0.999]
                    above = [r for ki_fixed, r in rises.items() if ki_fixed > ki_centre * 1.001]
                    profile_range_pct = float(max(rises.values()))
                    ki_identified = bool(
                        below and above
                        and max(below) >= KI_PROFILE_MIN_SSE_CHANGE_PCT
                        and max(above) >= KI_PROFILE_MIN_SSE_CHANGE_PCT
                        # The profile is only evidence about the point it is
                        # centred on; if it found a better fit elsewhere, that
                        # centre was wrong and nothing here is established.
                        and not found_better_on_profile
                    )
                else:
                    profile_range_pct = 0.0

        # The reported fit is whichever optimum survived (the multi-start one,
        # or a better one the profile walked into), re-solved at the default
        # solver accuracy.
        mu_max, Ks, Ki, X_max, q, X0_fit = best_params
        X_pred, S_pred, solved = _predict(best_params)
        if not solved:
            return {'success': False, 'r_squared': float('nan'),
                    'aic': np.inf, 'bic': np.inf,
                    'Ki_identified': False,
                    'error': 'ODE solver failed at the optimum'}

        residuals = od - X_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((od - np.mean(od)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        n = len(od)
        k = len(bounds)  # free parameters (S0 is fixed, so it does not count)
        aic = n * np.log(ss_res / n) + 2 * k
        # Small-sample correction (AICc) -- important for k=6 with moderate n
        if n - k - 1 > 0:
            aic += 2 * k * (k + 1) / (n - k - 1)
        bic = n * np.log(ss_res / n) + k * np.log(n)

        # Parameters the optimiser pushed onto a bound are not estimates: the
        # audit found mu_max pinned at 5.0 for 15 of 22 strains, which is the
        # model telling you it cannot describe the curve.
        names = ['mu_max', 'Ks', 'Ki', 'X_max', 'q', 'X0']
        at_bounds = [
            name for name, value, (lo, hi) in zip(names, best_params, bounds)
            if np.isclose(value, lo, rtol=1e-6, atol=1e-12)
            or np.isclose(value, hi, rtol=1e-6, atol=1e-12)
        ]

        return {
            'success': True,
            'params_at_bounds': at_bounds,
            'mu_max': mu_max, 'Ks': Ks,
            'Ki': Ki if ki_identified else float('nan'),
            'Ki_fitted': Ki,
            'Ki_identified': bool(ki_identified),
            'Ki_profile_sse_range_pct': profile_range_pct,
            'Ki_profile': profile,
            'X_max': X_max, 'q': q, 'X0': X0_fit, 'S0': S0,
            'r_squared': r_squared,
            'aic': aic, 'bic': bic,
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'sse': ss_res,
            'n_points': n,
            'predicted_od': X_pred,
            'predicted_substrate': S_pred,
        }
    except Exception as e:
        import warnings
        warnings.warn(f"Haldane fitting failed: {e}")

    return {'success': False, 'r_squared': float('nan'), 'aic': np.inf,
            'bic': np.inf, 'Ki_identified': False}


# =============================================================================
# Gompertz Model (for AIC-fair comparison)
# =============================================================================

def gompertz_model(t, A, mu, lambda_):
    """Modified Gompertz growth model."""
    return A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))


def _gompertz_sse(time, od, params):
    """Sum of squared residuals of a Gompertz parameter set on this window."""
    pred = gompertz_model(time, params['a'], params['mu'], params['lambda'])
    if not np.all(np.isfinite(pred)):
        return float('inf')
    return float(np.sum((od - pred) ** 2))


def gompertz_on_window(time, od, fallback_params=None):
    """
    Fit the Gompertz model on the window the Haldane fit is scored on.

    The AICc comparison used to score the parameters 01 fitted on its
    *truncated* growth window against the *full* curve, while Haldane was
    refitted to that full curve. On any curve with a plateau or a decline the
    Gompertz R² then went negative and "Haldane preferred" was a foregone
    conclusion — the mechanistic model had simply been allowed to see the data
    the other one was denied. Both models are fitted to the same points now.

    The refit is compared against the pipeline's own parameters on this same
    window and the lower-SSE one wins, so the comparison always uses the best
    Gompertz description of the window that either route found (curve_fit can
    land in a poor local optimum on a curve with a long decline).

    Returns ``(params, source)`` where source is 'refit' or 'pipeline'.
    """
    candidates = []
    try:
        fit = _gca().fit_gompertz(time, od)
        if fit.success and np.isfinite(fit.r_squared):
            candidates.append(('refit', {
                'a': float(fit.a_opt),
                'mu': float(fit.mu_opt),
                'lambda': float(fit.lambda_opt),
            }))
    except Exception as e:
        warnings.warn(f"Gompertz refit on the comparison window failed: {e}")

    if fallback_params:
        candidates.append(('pipeline', {
            'a': float(fallback_params.get('a', 1.0)),
            'mu': float(fallback_params.get('mu', 0.15)),
            'lambda': float(fallback_params.get('lambda', 3.0)),
        }))

    if not candidates:
        return {'a': 1.0, 'mu': 0.15, 'lambda': 3.0}, 'default'

    source, params = min(candidates, key=lambda c: _gompertz_sse(time, od, c[1]))
    return params, source


def gompertz_aic(time, od, A, mu, lambda_):
    """Compute AIC for Gompertz fit using same n."""
    pred = gompertz_model(time, A, mu, lambda_)
    residuals = od - pred
    ss_res = np.sum(residuals**2)
    n = len(od)
    k = 3  # Gompertz has 3 parameters
    aic = n * np.log(ss_res / n) + 2 * k
    # Small-sample correction (AICc) — consistent with Haldane
    if n - k - 1 > 0:
        aic += 2 * k * (k + 1) / (n - k - 1)
    bic = n * np.log(ss_res / n) + k * np.log(n)
    r_squared = 1 - ss_res / np.sum((od - np.mean(od))**2) if np.sum((od - np.mean(od))**2) > 0 else 0
    return aic, bic, r_squared, np.sqrt(np.mean(residuals**2))


# =============================================================================
# Data Loading
# =============================================================================

def identify_pesticide_strains(results_df):
    """
    Identify which strains are pesticide treatment curves.

    Pesticide treatment strains have format: [PESTICIDE]ANDLB-[STRAIN_ID]
    Controls: LB-*, H2O-*, [PESTICIDE]-* (no LB)

    Returns:
        DataFrame filtered to pesticide+LB treatment strains
    """
    pesticide_mask = results_df['strain'].str.contains('ANDLB', case=False, na=False)
    return results_df[pesticide_mask].copy()


def extract_pesticide_name(strain):
    """Extract pesticide name from strain like 'BifenthrinANDLB-BIF2'."""
    if 'ANDLB' in strain.upper():
        return strain.upper().split('ANDLB')[0]
    return 'UNKNOWN'


#: Where each group's processed *_DATA.csv files live, relative to data/raw.
#: Only used when config.yaml has no ``groups`` section.
FALLBACK_GROUP_DIRS = {
    'Group1': 'Group1/Group_1_DATA',
    'Group2': 'Group 2/Group_2_DATA',
    'Group3': 'Group 3/Group_3_DATA',
    'Group4': 'Group 4/Group4_DATA',
}


def resolve_group_dir(data_dir, group, config=None):
    """
    Directory holding a group's processed ``*_DATA.csv`` files.

    Read from ``config.yaml`` (``groups.<Group>.data_dir``) so that renaming a
    group's folder in one place is enough; ``FALLBACK_GROUP_DIRS`` is used only
    when the config has no ``groups`` section. A ``data_dir`` that sits under
    ``paths.raw_data`` is re-anchored on the ``data_dir`` argument, so
    ``--data-dir`` keeps working on a copy of the tree.
    """
    if config is None:
        import yaml
        cfg_path = Path(__file__).resolve().parent / 'config.yaml'
        try:
            with open(cfg_path) as f:
                config = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            config = {}

    entry = ((config.get('groups') or {}).get(group) or {})
    configured = entry.get('data_dir')
    if not configured:
        return Path(data_dir) / FALLBACK_GROUP_DIRS.get(group, group)

    configured = Path(configured)
    raw_base = Path((config.get('paths') or {}).get('raw_data', 'data/raw'))
    try:
        return Path(data_dir) / configured.relative_to(raw_base)
    except ValueError:
        return configured if configured.is_absolute() else _REPO_ROOT / configured


def load_raw_data(data_dir, strain_name, group):
    """
    Load raw time-series data for a specific strain.

    Args:
        data_dir: Base data directory (e.g., data/raw)
        strain_name: e.g., 'BifenthrinANDLB-BIF2'
        group: e.g., 'Group1'

    Returns:
        (time, od) arrays or (None, None) if not found
    """
    group_path = resolve_group_dir(data_dir, group)
    if not group_path.exists():
        return None, None

    # Exact, index-based column lookup (see tecan_growth.legacy_data): the old
    # glob-order scan with a substring "partial match" loaded 33 of 92 strains
    # from another strain's column.
    time, od, _csv_path, _column = find_curve(group_path, strain_name)
    return time, od


# =============================================================================
# Plotting
# =============================================================================

def plot_haldane_comparison(
    time, od, haldane_result, gompertz_params, strain_name, output_path
):
    """
    Create dual-panel comparison plot: Gompertz vs Haldane.

    Left: Biomass fits (data + both model predictions)
    Right: Haldane substrate depletion prediction
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Biomass comparison
    ax = axes[0]
    ax.scatter(time, od, s=10, alpha=0.4, color='gray', label='Data', zorder=1)

    # Gompertz fit
    if gompertz_params:
        gomp_pred = gompertz_model(
            time, gompertz_params['a'], gompertz_params['mu'], gompertz_params['lambda']
        )
        ax.plot(time, gomp_pred, 'b-', linewidth=2, label='Gompertz', zorder=2)

    # Haldane fit
    if haldane_result['success']:
        ax.plot(time, haldane_result['predicted_od'], 'r--', linewidth=2,
                label='Haldane', zorder=3)

    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('OD600 (Biomass)', fontsize=11)
    ax.set_title(f'{strain_name}: Model Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add R²/AIC annotation
    info_parts = []
    if gompertz_params:
        info_parts.append(f"Gompertz: R²={gompertz_params.get('r2', 0):.4f}")
    if haldane_result['success']:
        info_parts.append(f"Haldane:  R²={haldane_result['r_squared']:.4f}")
    if info_parts:
        ax.text(0.02, 0.98, '\n'.join(info_parts), transform=ax.transAxes,
                verticalalignment='top', fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: Substrate depletion
    ax = axes[1]
    if haldane_result['success']:
        ax.plot(time, haldane_result['predicted_substrate'], 'b-', linewidth=2)
        ax.fill_between(time, 0, haldane_result['predicted_substrate'], alpha=0.1, color='blue')
        ax.set_xlabel('Time (hours)', fontsize=11)
        ax.set_ylabel('Substrate Concentration (a.u.)', fontsize=11)

        Ki = haldane_result['Ki']
        Ks = haldane_result['Ks']
        mu_max = haldane_result['mu_max']
        q = haldane_result['q']

        ax.set_title(
            f'Predicted Pesticide Depletion\n'
            f'Ki={Ki:.2f} (inhibition constant)',
            fontsize=12, fontweight='bold'
        )

        param_text = (
            f"mu_max = {mu_max:.3f} h⁻¹\n"
            f"Ks     = {Ks:.3f}\n"
            f"Ki     = {Ki:.1f}\n"
            f"q      = {q:.3f}\n"
            f"S0     = {haldane_result['S0']:.2f}"
        )
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Haldane fit failed', ha='center', va='center',
                fontsize=14, color='red')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overview_dashboard(comparison_df, output_path):
    """
    Create summary dashboard showing key Haldane results.

    4-panel figure:
    1. Ki by pesticide (lower = more inhibitory)
    2. Gompertz vs Haldane R² scatter
    3. AIC comparison (which model is preferred)
    4. Substrate depletion rate by pesticide
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Filter to successful Haldane fits
    df = comparison_df[comparison_df['haldane_success']].copy()
    if len(df) == 0:
        fig.text(0.5, 0.5, 'No successful Haldane fits', ha='center', va='center', fontsize=16)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    df['pesticide'] = df['strain'].apply(extract_pesticide_name)

    # Panel 1: Ki by pesticide (boxplot) — only strains whose Ki is identified
    ax1 = fig.add_subplot(gs[0, 0])
    ki_df = df[df['haldane_Ki'].notna()]
    pesticides = ki_df.groupby('pesticide')['haldane_Ki'].median().sort_values().index.tolist()
    ki_data = [ki_df[ki_df['pesticide'] == p]['haldane_Ki'].values for p in pesticides]
    if ki_data:
        bp = ax1.boxplot(ki_data, tick_labels=pesticides, patch_artist=True, vert=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(pesticides)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        ax1.text(0.5, 0.5, 'Ki not identified for any strain',
                 ha='center', va='center', fontsize=12, color='red')
    ax1.set_ylabel('Ki (Inhibition Constant)', fontsize=11)
    ax1.set_title(
        f'Substrate Inhibition by Pesticide\n'
        f'(lower Ki = stronger inhibition; {len(ki_df)}/{len(df)} strains identified)',
        fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Gompertz vs Haldane R²
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['gompertz_r2'], df['haldane_r2'], s=40, alpha=0.7,
                c=df['pesticide'].astype('category').cat.codes, cmap='Set1')
    lims = [min(df['gompertz_r2'].min(), df['haldane_r2'].min()) - 0.02, 1.01]
    ax2.plot(lims, lims, 'k--', alpha=0.3, label='Equal fit')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Gompertz R²', fontsize=11)
    ax2.set_ylabel('Haldane R²', fontsize=11)
    ax2.set_title('Model Fit Quality Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: AIC preference (bar chart)
    ax3 = fig.add_subplot(gs[1, 0])
    df['preferred_model'] = df.apply(
        lambda r: 'Haldane' if r['haldane_aic'] < r['gompertz_aic'] else 'Gompertz', axis=1
    )
    pref_counts = df.groupby('pesticide')['preferred_model'].value_counts().unstack(fill_value=0)
    if 'Haldane' not in pref_counts.columns:
        pref_counts['Haldane'] = 0
    if 'Gompertz' not in pref_counts.columns:
        pref_counts['Gompertz'] = 0
    pref_counts = pref_counts[['Gompertz', 'Haldane']]
    pref_counts.plot(kind='bar', ax=ax3, color=['#4C72B0', '#DD8452'], width=0.7)
    ax3.set_ylabel('Number of Strains', fontsize=11)
    ax3.set_title('Preferred Model by AIC\n(per pesticide)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Preferred')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Effective growth rate under inhibition
    ax4 = fig.add_subplot(gs[1, 1])
    # mu_effective = mu_max * S0 / (Ks + S0 + S0^2/Ki)
    # Uses the fitted Ki (not the NaN-when-unidentified one), because mu at S0
    # is a property of the fitted curve even where Ki alone is not pinned down.
    df['mu_effective'] = df.apply(
        lambda r: r['haldane_mu_max'] * r['haldane_S0'] / (
            r['haldane_Ks'] + r['haldane_S0'] + r['haldane_S0']**2 / r['haldane_Ki_fitted']
        ) if r.get('haldane_Ki_fitted', float('nan')) > 0 else 0,
        axis=1
    )
    pest_mu = df.groupby('pesticide')['mu_effective'].agg(['mean', 'std']).sort_values('mean')
    ax4.barh(pest_mu.index, pest_mu['mean'], xerr=pest_mu['std'],
             color=plt.cm.viridis(np.linspace(0.3, 0.8, len(pest_mu))),
             capsize=3, alpha=0.8)
    ax4.set_xlabel('Effective Growth Rate (h⁻¹)', fontsize=11)
    ax4.set_title('Growth Rate Under Inhibition\n(higher = less inhibited)',
                   fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Haldane Feedback Inhibition Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Haldane feedback inhibition analysis for pesticide growth curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analyzes pesticide treatment growth curves using the Haldane/Andrews substrate
inhibition model. Compares mechanistic Haldane fits with phenomenological
Gompertz fits using AIC.

Key outputs:
    haldane_comparison.csv  - Per-strain model comparison
    haldane_summary.csv     - Per-pesticide Ki rankings
    haldane_overview.png    - Summary dashboard
    plots/                  - Individual strain comparison plots
        """)
    parser.add_argument(
        '--results-dir', '-r',
        default=None,
        help='Path to results/tables directory (default: auto-detect)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default=None,
        help='Path to data/raw directory for loading raw time series'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: results/tables/Haldane_Analysis)'
    )
    parser.add_argument(
        '--s0', type=float, default=1.0,
        help='Initial substrate concentration in arbitrary units (default: 1.0)'
    )
    parser.add_argument(
        '--all-strains', action='store_true',
        help='Fit all strains, not just pesticide+LB treatments'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress detailed progress output'
    )

    args = parser.parse_args()

    # Auto-detect paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = base_dir / "results" / "tables"

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = base_dir / "data" / "raw"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_dir / "Haldane_Analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load pipeline results
    results_csv = results_dir / "all_groups_results.csv"
    if not results_csv.exists():
        print(f"ERROR: Pipeline results not found at {results_csv}")
        print("Run the main pipeline (01_growth_curve_analysis.py) first.")
        sys.exit(1)

    all_results = pd.read_csv(results_csv)
    print(f"Loaded {len(all_results)} strains from pipeline results")

    # Identify pesticide treatment strains
    if args.all_strains:
        target_strains = all_results[all_results['is_good'] == True].copy()
        print(f"  Analyzing ALL {len(target_strains)} good-fit strains")
    else:
        target_strains = identify_pesticide_strains(all_results)
        # Only analyze strains with good Gompertz fits
        target_strains = target_strains[target_strains['is_good'] == True].copy()
        print(f"  Found {len(target_strains)} pesticide+LB strains with good fits")

    if len(target_strains) == 0:
        print("No target strains found. Check pipeline results.")
        sys.exit(1)

    # Analyze each strain
    print(f"\n{'='*60}")
    print(f"  HALDANE FEEDBACK INHIBITION ANALYSIS")
    print(f"  S0 = {args.s0} (arbitrary units)")
    print(f"{'='*60}\n")

    comparison_rows = []
    n_total = len(target_strains)

    for i, (_, row) in enumerate(target_strains.iterrows()):
        strain = row['strain']
        group = row['group']

        if not args.quiet:
            print(f"  [{i+1}/{n_total}] {strain}...", end=' ')

        # Load raw time-series data
        time, od = load_raw_data(data_dir, strain, group)

        if time is None:
            if not args.quiet:
                print("SKIPPED (raw data not found)")
            continue

        # Clean data: remove NaN, ensure numeric
        mask = np.isfinite(time) & np.isfinite(od)
        time = time[mask].astype(float)
        od = od[mask].astype(float)

        if len(time) < 20:
            if not args.quiet:
                print("SKIPPED (too few points)")
            continue

        # Gompertz parameters as the pipeline fitted them (on its truncated
        # window) — kept for the audit trail, not used for the AICc comparison
        pipeline_gompertz = {
            'a': row.get('gompertz_a', 1.0),
            'mu': row.get('gompertz_mu', 0.15),
            'lambda': row.get('gompertz_lambda', 3.0),
            'r2': row.get('fit_r_squared', row.get('r_squared', 0)),
        }

        # Refit Gompertz on exactly the window the Haldane fit sees, then score
        # both models on those same points.
        gompertz_params, gomp_params_source = gompertz_on_window(time, od, pipeline_gompertz)
        gomp_aic, gomp_bic, gomp_r2_full, gomp_rmse = gompertz_aic(
            time, od, gompertz_params['a'], gompertz_params['mu'], gompertz_params['lambda']
        )
        gompertz_params['r2'] = gomp_r2_full

        # Fit Haldane model
        haldane = fit_haldane(
            time, od, S0=args.s0,
            gompertz_params={'mu': gompertz_params['mu']}
        )

        # Determine preferred model
        if haldane['success']:
            delta_aic = gomp_aic - haldane['aic']  # positive = Haldane preferred
            preferred = 'Haldane' if delta_aic > 2 else ('Gompertz' if delta_aic < -2 else 'Equivalent')
        else:
            delta_aic = float('nan')
            preferred = 'Gompertz'

        # Collect results
        comp_row = {
            'strain': strain,
            'group': group,
            'pesticide': extract_pesticide_name(strain),
            # Gompertz
            'gompertz_a': gompertz_params['a'],
            'gompertz_mu': gompertz_params['mu'],
            'gompertz_lambda': gompertz_params['lambda'],
            'gompertz_r2': gomp_r2_full,
            'gompertz_aic': gomp_aic,
            'gompertz_bic': gomp_bic,
            'gompertz_rmse': gomp_rmse,
            'gompertz_params_source': gomp_params_source,
            # Gompertz as the pipeline fitted it (truncated window)
            'pipeline_gompertz_a': pipeline_gompertz['a'],
            'pipeline_gompertz_mu': pipeline_gompertz['mu'],
            'pipeline_gompertz_lambda': pipeline_gompertz['lambda'],
            'pipeline_gompertz_r2': pipeline_gompertz['r2'],
            # Haldane
            'haldane_success': haldane['success'],
            'haldane_r2': haldane.get('r_squared', float('nan')),
            'haldane_aic': haldane.get('aic', float('nan')),
            'haldane_bic': haldane.get('bic', float('nan')),
            'haldane_rmse': haldane.get('rmse', float('nan')),
            'haldane_mu_max': haldane.get('mu_max', float('nan')),
            'haldane_Ks': haldane.get('Ks', float('nan')),
            'haldane_Ki': haldane.get('Ki', float('nan')),
            'haldane_Ki_fitted': haldane.get('Ki_fitted', float('nan')),
            'haldane_Ki_identified': haldane.get('Ki_identified', False),
            'haldane_Ki_profile_sse_range_pct': haldane.get(
                'Ki_profile_sse_range_pct', float('nan')),
            'haldane_X_max': haldane.get('X_max', float('nan')),
            'haldane_q': haldane.get('q', float('nan')),
            'haldane_S0': haldane.get('S0', float('nan')),
            # Comparison — both models fitted on exactly these points
            'comparison_n_points': int(len(time)),
            'comparison_t_start': float(time[0]),
            'comparison_t_end': float(time[-1]),
            'delta_aic': delta_aic,
            'preferred_model': preferred,
        }
        comparison_rows.append(comp_row)

        if not args.quiet:
            if haldane['success']:
                ki_txt = (f"Ki={haldane['Ki']:.2f}" if haldane.get('Ki_identified')
                          else f"Ki=unidentified (best {haldane.get('Ki_fitted', float('nan')):.2f})")
                print(f"Haldane R²={haldane['r_squared']:.4f} {ki_txt} "
                      f"| Gompertz R²={gomp_r2_full:.4f} | {preferred} (ΔAIC={delta_aic:+.1f})")
            else:
                print(f"Haldane FAILED | Gompertz R²={gomp_r2_full:.4f}")

        # Generate comparison plot
        safe_name = strain.replace('/', '_').replace(' ', '_')
        plot_haldane_comparison(
            time, od, haldane, gompertz_params, strain,
            str(plots_dir / f'{safe_name}_haldane_vs_gompertz.png')
        )

    # Save comparison results
    if not comparison_rows:
        print("\nNo strains were successfully analyzed!")
        sys.exit(1)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / 'haldane_comparison.csv', index=False)
    print(f"\nSaved comparison to {output_dir / 'haldane_comparison.csv'}")

    # Generate summary by pesticide
    haldane_ok = comparison_df[comparison_df['haldane_success']].copy()

    if len(haldane_ok) > 0:
        # Ki statistics cover only the strains whose Ki is identified: the
        # unidentified ones carry NaN, which pandas skips in the aggregates,
        # and n_Ki_identified says how many strains each Ki number rests on.
        summary = haldane_ok.groupby('pesticide').agg({
            'haldane_Ki': ['median', 'mean', 'std', 'count'],
            'haldane_r2': ['count', 'median', 'mean'],
            'haldane_mu_max': ['median', 'mean'],
            'haldane_Ks': ['median', 'mean'],
            'gompertz_r2': ['median', 'mean'],
            'preferred_model': lambda x: (x == 'Haldane').sum(),
        }).reset_index()

        # Flatten column names
        summary.columns = [
            'pesticide',
            'Ki_median', 'Ki_mean', 'Ki_std', 'n_Ki_identified',
            'n_strains', 'haldane_r2_median', 'haldane_r2_mean',
            'mu_max_median', 'mu_max_mean',
            'Ks_median', 'Ks_mean',
            'gompertz_r2_median', 'gompertz_r2_mean',
            'n_haldane_preferred',
        ]
        summary = summary.sort_values('Ki_median', na_position='last')
        summary.to_csv(output_dir / 'haldane_summary.csv', index=False)
        print(f"Saved summary to {output_dir / 'haldane_summary.csv'}")

    # Generate overview dashboard
    plot_overview_dashboard(comparison_df, output_dir / 'haldane_overview.png')
    print(f"Saved overview to {output_dir / 'haldane_overview.png'}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  HALDANE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total strains analyzed: {len(comparison_df)}")
    print(f"  Successful Haldane fits: {len(haldane_ok)}")
    if len(haldane_ok) > 0:
        n_haldane_pref = (comparison_df['preferred_model'] == 'Haldane').sum()
        n_gomp_pref = (comparison_df['preferred_model'] == 'Gompertz').sum()
        n_equiv = (comparison_df['preferred_model'] == 'Equivalent').sum()
        print(f"  Preferred model (by AIC):")
        print(f"    Haldane:    {n_haldane_pref}")
        print(f"    Gompertz:   {n_gomp_pref}")
        print(f"    Equivalent: {n_equiv}")

        n_ki = int(comparison_df.get('haldane_Ki_identified', pd.Series(dtype=bool)).sum())
        print(f"  Strains with an identified Ki: {n_ki}/{len(haldane_ok)}")

        print(f"\n  Inhibition Ranking (lower Ki = stronger inhibition):")
        for _, srow in summary.iterrows():
            n_id = int(srow['n_Ki_identified'])
            ki_txt = f"Ki={srow['Ki_median']:.2f}" if n_id else "Ki=unidentified"
            print(f"    {srow['pesticide']:<30s} {ki_txt} "
                  f"(n={n_id}/{int(srow['n_strains'])} identified)")

        # Only rank pesticides whose Ki the data actually pinned down —
        # ordering NaNs would be ranking the optimiser's starting points.
        ranked = summary[summary['Ki_median'].notna()]
        print(f"\n  Biological Interpretation:")
        if len(ranked) >= 2:
            print(f"    Most inhibitory pesticide:  {ranked.iloc[0]['pesticide']} (lowest Ki)")
            print(f"    Least inhibitory pesticide: {ranked.iloc[-1]['pesticide']} (highest Ki)")
        else:
            print("    Not enough identified Ki values to rank pesticides.")
    print(f"{'='*60}")

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
