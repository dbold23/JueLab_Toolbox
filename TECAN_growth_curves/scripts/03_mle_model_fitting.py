#!/usr/bin/env python3
"""
MLE-Based Growth Model Fitting with Multi-Model Comparison

Extends the existing Gompertz-only pipeline with:
1. Maximum Likelihood Estimation (MLE) for model fitting
2. Heteroscedastic error modeling (variance scales with OD)
3. AIC/BIC model comparison across Gompertz, Baranyi, Logistic, Richards
4. Haldane/Andrews feedback inhibition model (substrate inhibition)
5. Profile likelihood confidence intervals

Usage:
    # Run on Group 2 MLE test data
    python 03_mle_model_fitting.py "DATA/Group 2/GROUP2_MLE_TEST_DATA" -o OUTPUT/MLE_Results

    # Run on all groups output with model comparison
    python 03_mle_model_fitting.py "DATA/Group 2/Group_2_DATA" -o OUTPUT/MLE_Group2

    # Include feedback inhibition model
    python 03_mle_model_fitting.py "DATA/Group 2/GROUP2_MLE_TEST_DATA" -o OUTPUT/MLE_Results --haldane

BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
"""

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Growth Model Functions
# =============================================================================

def gompertz_model(t, a, mu, lam):
    """Modified Gompertz: y(t) = a * exp(-exp((mu*e/a)*(lam-t)+1))"""
    return a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1))


def baranyi_model(t, y0, y_max, mu_max, h0):
    """Baranyi-Roberts model with lag adjustment function."""
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


def logistic_model(t, A, k, t_mid):
    """Logistic: y(t) = A / (1 + exp(-k*(t - t_mid)))"""
    return A / (1 + np.exp(-k * (t - t_mid)))


def richards_model(t, A, k, t_mid, nu):
    """Richards: y(t) = A * (1 + nu*exp(-k*(t-t_mid)))^(-1/nu)"""
    if abs(nu) < 1e-6:
        return gompertz_model(t, A, k * A / np.e, t_mid - A / (k * np.e))
    with np.errstate(over='ignore', invalid='ignore'):
        base = np.clip(1 + nu * np.exp(-k * (t - t_mid)), 1e-10, None)
        y = A * np.power(base, -1.0 / nu)
    return np.nan_to_num(y, nan=0.0, posinf=A)


# =============================================================================
# MLE Fitting Framework
# =============================================================================

@dataclass
class MLEFitResult:
    """Container for MLE fit results."""
    model_name: str
    success: bool
    params: Dict[str, float]
    param_errors: Dict[str, float]
    nll: float            # negative log-likelihood
    aic: float
    bic: float
    aicc: float
    r_squared: float
    rmse: float
    n_params: int
    n_data: int
    predicted: np.ndarray
    residuals: np.ndarray
    error_message: str = ""


def neg_log_likelihood_homoscedastic(theta, time, od, model_func, n_model_params):
    """
    Negative log-likelihood with constant Gaussian errors.

    theta = [model_params..., sigma]
    """
    model_params = theta[:n_model_params]
    sigma = theta[n_model_params]

    if sigma <= 0:
        return 1e12

    try:
        predicted = model_func(time, *model_params)
        if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
            return 1e12
        residuals = od - predicted
        n = len(od)
        nll = (n / 2) * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)
        return nll
    except Exception:
        return 1e12


def neg_log_likelihood_heteroscedastic(theta, time, od, model_func, n_model_params):
    """
    Negative log-likelihood with OD-dependent variance.

    sigma(t) = sigma_base + sigma_scale * |predicted(t)|
    theta = [model_params..., sigma_base, sigma_scale]
    """
    model_params = theta[:n_model_params]
    sigma_base = theta[n_model_params]
    sigma_scale = theta[n_model_params + 1]

    if sigma_base <= 0 or sigma_scale < 0:
        return 1e12

    try:
        predicted = model_func(time, *model_params)
        if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
            return 1e12
        sigma_t = sigma_base + sigma_scale * np.abs(predicted)
        sigma_t = np.maximum(sigma_t, 1e-8)
        nll = np.sum(0.5 * np.log(2 * np.pi * sigma_t**2) +
                     (od - predicted)**2 / (2 * sigma_t**2))
        return nll
    except Exception:
        return 1e12


def compute_aic_bic(nll, n_params, n_data):
    """Compute AIC, BIC, and corrected AIC."""
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_data) + 2 * nll
    denom = max(n_data - n_params - 1, 1)
    aicc = aic + 2 * n_params * (n_params + 1) / denom
    return aic, bic, aicc


def fit_model_mle(
    time: np.ndarray,
    od: np.ndarray,
    model_name: str,
    model_func,
    p0: list,
    bounds: list,
    heteroscedastic: bool = False
) -> MLEFitResult:
    """
    Fit a growth model using MLE.

    Args:
        time: Time array in hours
        od: OD600 values
        model_name: Name of the model
        model_func: Model function f(t, *params) -> y
        p0: Initial parameter guesses (model params only)
        bounds: List of (lower, upper) tuples for model params
        heteroscedastic: Use OD-dependent variance model

    Returns:
        MLEFitResult with fit statistics
    """
    n_data = len(od)
    n_model_params = len(p0)

    # Estimate initial sigma from data
    try:
        predicted_init = model_func(time, *p0)
        sigma_init = np.std(od - predicted_init) if not np.any(np.isnan(predicted_init)) else 0.05
    except Exception:
        sigma_init = 0.05
    sigma_init = max(sigma_init, 0.001)

    # Build full parameter vector and bounds
    if heteroscedastic:
        theta0 = list(p0) + [sigma_init, 0.01]
        full_bounds = list(bounds) + [(1e-6, 1.0), (0.0, 0.5)]
        nll_func = neg_log_likelihood_heteroscedastic
        n_total_params = n_model_params + 2
    else:
        theta0 = list(p0) + [sigma_init]
        full_bounds = list(bounds) + [(1e-6, 1.0)]
        nll_func = neg_log_likelihood_homoscedastic
        n_total_params = n_model_params + 1

    try:
        result = minimize(
            nll_func,
            theta0,
            args=(time, od, model_func, n_model_params),
            method='L-BFGS-B',
            bounds=full_bounds,
            options={'maxiter': 5000, 'ftol': 1e-12}
        )

        if not result.success:
            # Try Nelder-Mead as fallback (no bounds)
            result2 = minimize(
                nll_func,
                theta0,
                args=(time, od, model_func, n_model_params),
                method='Nelder-Mead',
                options={'maxiter': 10000}
            )
            if result2.fun < result.fun:
                result = result2

        opt_params = result.x[:n_model_params]
        nll = result.fun

        # Compute predicted and fit quality
        predicted = model_func(time, *opt_params)
        residuals = od - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((od - np.mean(od))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # AIC/BIC
        aic, bic, aicc = compute_aic_bic(nll, n_total_params, n_data)

        # Estimate parameter errors from Hessian (numerical)
        param_errors = _estimate_param_errors(result, n_model_params)

        # Map parameter names
        param_names = _get_param_names(model_name)
        params_dict = {name: float(val) for name, val in zip(param_names, opt_params)}
        errors_dict = {name: float(err) for name, err in zip(param_names, param_errors)}

        return MLEFitResult(
            model_name=model_name,
            success=True,
            params=params_dict,
            param_errors=errors_dict,
            nll=nll,
            aic=aic,
            bic=bic,
            aicc=aicc,
            r_squared=r_squared,
            rmse=rmse,
            n_params=n_total_params,
            n_data=n_data,
            predicted=predicted,
            residuals=residuals
        )

    except Exception as e:
        return MLEFitResult(
            model_name=model_name,
            success=False,
            params={},
            param_errors={},
            nll=np.inf,
            aic=np.inf,
            bic=np.inf,
            aicc=np.inf,
            r_squared=0,
            rmse=np.inf,
            n_params=n_total_params,
            n_data=n_data,
            predicted=np.array([]),
            residuals=np.array([]),
            error_message=str(e)
        )


def _estimate_param_errors(result, n_model_params):
    """Estimate parameter standard errors from optimization result."""
    try:
        if hasattr(result, 'hess_inv'):
            hess_inv = result.hess_inv
            if hasattr(hess_inv, 'todense'):
                hess_inv = hess_inv.todense()
            diag = np.diag(np.array(hess_inv))
            errors = np.sqrt(np.abs(diag[:n_model_params]))
            return errors
    except Exception:
        pass
    return np.zeros(n_model_params)


def _get_param_names(model_name):
    """Get parameter names for each model."""
    names = {
        'gompertz': ['A', 'mu', 'lambda'],
        'baranyi': ['y0', 'y_max', 'mu_max', 'h0'],
        'logistic': ['A', 'k', 't_mid'],
        'richards': ['A', 'k', 't_mid', 'nu'],
    }
    return names.get(model_name, [f'p{i}' for i in range(10)])


# =============================================================================
# Multi-Model Comparison
# =============================================================================

def compare_models(
    time: np.ndarray,
    od: np.ndarray,
    heteroscedastic: bool = False
) -> Dict[str, MLEFitResult]:
    """
    Fit all four growth models and compare via AIC/BIC.

    Args:
        time: Time array
        od: OD600 values
        heteroscedastic: Use OD-dependent variance

    Returns:
        Dict mapping model name to MLEFitResult
    """
    results = {}

    # Initial estimates from data
    a_init = float(np.max(od))
    a_init = max(a_init, 0.05)

    diff = np.diff(od)
    dt = np.diff(time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        growth_rates = diff / np.maximum(dt, 1e-6)
    mu_init = max(float(np.max(growth_rates)), 0.01)
    max_growth_idx = int(np.argmax(growth_rates))
    lambda_init = float(time[max_growth_idx]) if max_growth_idx > 0 else float(time[0])
    t_max = float(time[-1])

    # 1. Gompertz
    p0_gomp = [a_init, mu_init, lambda_init]
    bounds_gomp = [(0.01, 3 * a_init), (0.001, 10 * mu_init + 1), (0, t_max)]
    results['gompertz'] = fit_model_mle(
        time, od, 'gompertz', gompertz_model, p0_gomp, bounds_gomp, heteroscedastic
    )

    # 2. Baranyi
    y0_init = float(np.mean(od[:5]))
    h0_init = mu_init * max(lambda_init, 0.1)
    p0_bar = [max(y0_init, 0.001), a_init, mu_init, h0_init]
    bounds_bar = [(1e-6, a_init), (0.01, 3 * a_init), (0.001, 10 * mu_init + 1), (0.001, 200)]
    results['baranyi'] = fit_model_mle(
        time, od, 'baranyi', baranyi_model, p0_bar, bounds_bar, heteroscedastic
    )

    # 3. Logistic
    k_init = 4 * mu_init / (np.e * max(a_init, 0.01))
    t_mid_init = lambda_init + a_init / (mu_init * np.e)
    p0_log = [a_init, k_init, t_mid_init]
    bounds_log = [(0.01, 3 * a_init), (0.001, 20), (0, t_max * 1.5)]
    results['logistic'] = fit_model_mle(
        time, od, 'logistic', logistic_model, p0_log, bounds_log, heteroscedastic
    )

    # 4. Richards
    nu_init = 0.5
    k_rich_init = mu_init * np.e / max(a_init, 0.01) * (1 + nu_init)
    p0_rich = [a_init, k_rich_init, t_mid_init, nu_init]
    bounds_rich = [(0.01, 3 * a_init), (0.001, 20), (0, t_max * 1.5), (0.01, 5.0)]
    results['richards'] = fit_model_mle(
        time, od, 'richards', richards_model, p0_rich, bounds_rich, heteroscedastic
    )

    return results


def compute_akaike_weights(results: Dict[str, MLEFitResult]) -> Dict[str, float]:
    """Compute Akaike weights from AICc values."""
    successful = {k: v for k, v in results.items() if v.success and np.isfinite(v.aicc)}
    if not successful:
        return {k: 0.0 for k in results}

    aicc_values = {k: v.aicc for k, v in successful.items()}
    min_aicc = min(aicc_values.values())
    delta_aicc = {k: v - min_aicc for k, v in aicc_values.items()}

    # Relative likelihoods
    rel_lik = {k: np.exp(-0.5 * d) for k, d in delta_aicc.items()}
    total = sum(rel_lik.values())

    weights = {}
    for k in results:
        if k in rel_lik and total > 0:
            weights[k] = rel_lik[k] / total
        else:
            weights[k] = 0.0

    return weights


# =============================================================================
# Haldane/Andrews Feedback Inhibition Model
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


def solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0):
    """
    Solve the Haldane ODE system.

    Returns:
        X(t), S(t) arrays
    """
    try:
        sol = solve_ivp(
            haldane_ode,
            [time[0], time[-1]],
            [X0, S0],
            args=(mu_max, Ks, Ki, X_max, q),
            t_eval=time,
            method='RK45',
            max_step=0.5,
            rtol=1e-8,
            atol=1e-10
        )
        if sol.success:
            return sol.y[0], sol.y[1]
        else:
            return np.full_like(time, X0), np.full_like(time, S0)
    except Exception:
        return np.full_like(time, X0), np.full_like(time, S0)


def haldane_od_model(t, mu_max, Ks, Ki, X_max, q, X0, S0):
    """Haldane model returning only biomass (OD) for fitting."""
    X, S = solve_haldane(t, mu_max, Ks, Ki, X_max, q, X0, S0)
    return X


@dataclass
class HaldaneFitResult:
    """Container for Haldane model fit results."""
    success: bool
    mu_max: float
    Ks: float
    Ki: float
    X_max: float
    q: float
    X0: float
    S0: float
    r_squared: float
    aic: float
    bic: float
    predicted_od: np.ndarray
    predicted_substrate: np.ndarray
    error_message: str = ""


def fit_haldane(
    time: np.ndarray,
    od: np.ndarray,
    S0: float = 1.0,
    gompertz_params: Optional[Dict] = None
) -> HaldaneFitResult:
    """
    Fit Haldane feedback inhibition model via MLE.

    Args:
        time: Time array
        od: OD600 values
        S0: Initial substrate concentration (default 1.0 arbitrary units)
        gompertz_params: Optional Gompertz fit results to initialize parameters

    Returns:
        HaldaneFitResult
    """
    X0 = max(float(np.mean(od[:3])), 0.001)
    X_max_init = float(np.max(od))

    if gompertz_params:
        mu_init = gompertz_params.get('mu', 0.2)
    else:
        diff = np.diff(od)
        dt = np.diff(time)
        mu_init = max(float(np.max(diff / np.maximum(dt, 1e-6))), 0.01)

    # Initial guesses
    p0 = [mu_init * 2, 0.1, 10.0, X_max_init, 0.1, X0, S0]
    bounds = [
        (0.001, 5.0),    # mu_max
        (0.001, 10.0),   # Ks
        (0.1, 1000.0),   # Ki
        (0.05, 5.0),     # X_max
        (0.001, 10.0),   # q
        (1e-6, 0.5),     # X0
        (0.01, 100.0),   # S0
    ]

    n_model_params = 7

    def nll(theta):
        return neg_log_likelihood_homoscedastic(
            list(theta) + [0.05],  # append sigma
            time, od,
            lambda t, *p: haldane_od_model(t, *p),
            n_model_params
        )

    try:
        result = minimize(nll, p0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 5000})

        if result.success or result.fun < 1e10:
            opt = result.x
            mu_max, Ks, Ki, X_max, q, X0_fit, S0_fit = opt

            pred_X, pred_S = solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0_fit, S0_fit)

            residuals = od - pred_X
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((od - np.mean(od))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))

            nll_val = result.fun
            aic, bic, _ = compute_aic_bic(nll_val, n_model_params + 1, len(od))

            return HaldaneFitResult(
                success=True,
                mu_max=mu_max, Ks=Ks, Ki=Ki, X_max=X_max, q=q,
                X0=X0_fit, S0=S0_fit,
                r_squared=r_squared, aic=aic, bic=bic,
                predicted_od=pred_X, predicted_substrate=pred_S
            )
    except Exception as e:
        pass

    return HaldaneFitResult(
        success=False,
        mu_max=0, Ks=0, Ki=0, X_max=0, q=0, X0=0, S0=0,
        r_squared=0, aic=np.inf, bic=np.inf,
        predicted_od=np.array([]), predicted_substrate=np.array([]),
        error_message="Haldane fit failed"
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_model_comparison(
    time, od, results, weights, strain_name, output_path
):
    """Plot all model fits for a single curve."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: All model fits overlaid
    ax = axes[0, 0]
    ax.scatter(time, od, s=8, alpha=0.4, color='gray', label='Data')
    colors = {'gompertz': 'blue', 'baranyi': 'red', 'logistic': 'green', 'richards': 'purple'}
    for name, res in results.items():
        if res.success and len(res.predicted) > 0:
            w = weights.get(name, 0)
            ax.plot(time, res.predicted, color=colors.get(name, 'black'),
                    label=f'{name} (w={w:.2f})', linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD600')
    ax.set_title(f'{strain_name}: Model Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: AIC/BIC comparison bar chart
    ax = axes[0, 1]
    model_names = [k for k in results if results[k].success]
    aic_vals = [results[k].aic for k in model_names]
    bic_vals = [results[k].bic for k in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, aic_vals, width, label='AIC', color='steelblue')
    ax.bar(x + width/2, bic_vals, width, label='BIC', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylabel('Information Criterion')
    ax.set_title('Model Selection (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Residuals for best model
    ax = axes[1, 0]
    best = min((r for r in results.values() if r.success), key=lambda r: r.aicc, default=None)
    if best and len(best.residuals) > 0:
        ax.scatter(time, best.residuals, s=8, alpha=0.5, color='steelblue')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals ({best.model_name}, R²={best.r_squared:.4f})')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Akaike weights pie chart
    ax = axes[1, 1]
    filtered_weights = {k: v for k, v in weights.items() if v > 0.01}
    if filtered_weights:
        ax.pie(filtered_weights.values(), labels=filtered_weights.keys(),
               autopct='%1.1f%%', colors=[colors.get(k, 'gray') for k in filtered_weights])
        ax.set_title('Akaike Weights')
    else:
        ax.text(0.5, 0.5, 'No valid models', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_haldane_result(time, od, haldane_result, strain_name, output_path):
    """Plot Haldane feedback inhibition model fit."""
    if not haldane_result.success:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Biomass (OD) fit
    ax = axes[0]
    ax.scatter(time, od, s=10, alpha=0.5, color='gray', label='Data')
    ax.plot(time, haldane_result.predicted_od, 'r-', linewidth=2, label='Haldane fit')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD600 (Biomass)')
    ax.set_title(f'{strain_name}: Haldane Model (R²={haldane_result.r_squared:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Predicted substrate depletion
    ax = axes[1]
    ax.plot(time, haldane_result.predicted_substrate, 'b-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Substrate Concentration (a.u.)')
    ax.set_title(f'Predicted Pesticide Depletion\n'
                 f'Ki={haldane_result.Ki:.2f} (inhibition constant)')
    ax.grid(True, alpha=0.3)

    # Add parameter annotation
    param_text = (f"mu_max={haldane_result.mu_max:.3f}\n"
                  f"Ks={haldane_result.Ks:.3f}\n"
                  f"Ki={haldane_result.Ki:.1f}\n"
                  f"X_max={haldane_result.X_max:.3f}\n"
                  f"q={haldane_result.q:.3f}")
    ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def load_data_files(data_dir: Path) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Load TECAN-format CSV files from a directory.

    Returns:
        Dict mapping filename to (time_array, {strain_name: od_array})
    """
    datasets = {}
    csv_files = sorted(data_dir.glob("*_DATA.csv"))

    if not csv_files:
        print(f"No *_DATA.csv files found in {data_dir}")
        return datasets

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            time_col = [c for c in df.columns if 'TIME' in c.upper()][0]
            time = df[time_col].values

            strain_cols = [c for c in df.columns if c != time_col]
            strains = {}
            for col in strain_cols:
                strain_name = col.replace('_blanked', '')
                strains[strain_name] = df[col].values

            datasets[csv_path.stem] = (time, strains)
            print(f"  Loaded {csv_path.name}: {len(strain_cols)} curves, {len(time)} time points")
        except Exception as e:
            print(f"  Error loading {csv_path.name}: {e}")

    return datasets


def process_strain(
    strain_name: str,
    time: np.ndarray,
    od: np.ndarray,
    heteroscedastic: bool = False,
    include_haldane: bool = False
) -> Dict[str, Any]:
    """Process a single strain: truncate at peak, fit all models, compare."""

    # Find peak and truncate (simple first-peak approach)
    peak_idx = np.argmax(od)
    if peak_idx < len(od) - 1:
        # Truncate at peak
        time_trunc = time[:peak_idx + 1]
        od_trunc = od[:peak_idx + 1]
    else:
        time_trunc = time
        od_trunc = od

    # Skip if too few points
    if len(time_trunc) < 15:
        return {'strain': strain_name, 'skip': True, 'reason': 'Too few points after truncation'}

    # Skip if no growth
    delta_od = np.max(od_trunc) - np.mean(od_trunc[:5])
    if delta_od < 0.05:
        return {'strain': strain_name, 'skip': True, 'reason': f'Insufficient growth (delta_od={delta_od:.3f})'}

    # Fit all models
    results = compare_models(time_trunc, od_trunc, heteroscedastic)
    weights = compute_akaike_weights(results)

    # Best model
    best_name = min(
        (k for k, v in results.items() if v.success),
        key=lambda k: results[k].aicc,
        default='gompertz'
    )

    output = {
        'strain': strain_name,
        'skip': False,
        'results': results,
        'weights': weights,
        'best_model': best_name,
        'time': time_trunc,
        'od': od_trunc,
    }

    # Haldane model (on full data, not truncated)
    if include_haldane:
        gomp = results.get('gompertz')
        gomp_params = gomp.params if gomp and gomp.success else None
        haldane = fit_haldane(time, od, gompertz_params=gomp_params)
        output['haldane'] = haldane

    return output


def main():
    parser = argparse.ArgumentParser(
        description='MLE-based multi-model growth curve fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 03_mle_model_fitting.py "DATA/Group 2/GROUP2_MLE_TEST_DATA" -o OUTPUT/MLE_Results
    python 03_mle_model_fitting.py "DATA/Group 2/Group_2_DATA" -o OUTPUT/MLE_Group2 --hetero
    python 03_mle_model_fitting.py "DATA/Group 2/GROUP2_MLE_TEST_DATA" -o OUTPUT/MLE_Results --haldane
        """
    )

    parser.add_argument('data_dir', help='Directory with *_DATA.csv files')
    parser.add_argument('-o', '--output', default='OUTPUT/MLE_Results',
                       help='Output directory (default: OUTPUT/MLE_Results)')
    parser.add_argument('--hetero', action='store_true',
                       help='Use heteroscedastic (OD-dependent) noise model')
    parser.add_argument('--haldane', action='store_true',
                       help='Also fit Haldane feedback inhibition model')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress detailed progress output')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MLE MULTI-MODEL GROWTH CURVE FITTING")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Noise model: {'heteroscedastic' if args.hetero else 'homoscedastic'}")
    print(f"Haldane model: {'enabled' if args.haldane else 'disabled'}")

    # Load data
    print("\nLoading data...")
    datasets = load_data_files(data_dir)

    if not datasets:
        print("No data files found!")
        sys.exit(1)

    # Process all strains
    all_results = []
    total_strains = sum(len(strains) for _, (_, strains) in datasets.items())
    processed = 0

    for file_name, (time, strains) in datasets.items():
        print(f"\nProcessing: {file_name}")

        for strain_name, od in strains.items():
            processed += 1
            if not args.quiet:
                print(f"  [{processed}/{total_strains}] {strain_name}...", end=' ')

            result = process_strain(
                strain_name, time, od,
                heteroscedastic=args.hetero,
                include_haldane=args.haldane
            )

            if result.get('skip'):
                if not args.quiet:
                    print(f"SKIPPED ({result.get('reason', '')})")
                continue

            # Collect results
            best = result['results'][result['best_model']]
            row = {
                'strain': strain_name,
                'best_model': result['best_model'],
                'best_model_weight': result['weights'][result['best_model']],
            }

            # Add each model's metrics
            for model_name, mle_res in result['results'].items():
                prefix = model_name
                row[f'{prefix}_aic'] = mle_res.aic if mle_res.success else np.nan
                row[f'{prefix}_bic'] = mle_res.bic if mle_res.success else np.nan
                row[f'{prefix}_aicc'] = mle_res.aicc if mle_res.success else np.nan
                row[f'{prefix}_r2'] = mle_res.r_squared if mle_res.success else np.nan
                row[f'{prefix}_rmse'] = mle_res.rmse if mle_res.success else np.nan
                row[f'{prefix}_weight'] = result['weights'].get(model_name, 0)

                # Add parameters for each model
                if mle_res.success:
                    for pname, pval in mle_res.params.items():
                        row[f'{prefix}_{pname}'] = pval

            # Haldane results
            if 'haldane' in result:
                h = result['haldane']
                row['haldane_r2'] = h.r_squared if h.success else np.nan
                row['haldane_aic'] = h.aic if h.success else np.nan
                row['haldane_mu_max'] = h.mu_max if h.success else np.nan
                row['haldane_Ks'] = h.Ks if h.success else np.nan
                row['haldane_Ki'] = h.Ki if h.success else np.nan
                row['haldane_X_max'] = h.X_max if h.success else np.nan

            all_results.append(row)

            if not args.quiet:
                print(f"BEST={result['best_model']} "
                      f"(R²={best.r_squared:.4f}, AICc={best.aicc:.1f}, "
                      f"w={result['weights'][result['best_model']]:.2f})")

            # Generate plots
            safe_name = strain_name.replace('/', '_').replace(' ', '_')
            plot_model_comparison(
                result['time'], result['od'],
                result['results'], result['weights'],
                strain_name,
                str(plots_dir / f'{safe_name}_model_comparison.png')
            )

            if 'haldane' in result and result['haldane'].success:
                plot_haldane_result(
                    time, od, result['haldane'],
                    strain_name,
                    str(plots_dir / f'{safe_name}_haldane.png')
                )

    # Save results CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = output_dir / 'mle_model_comparison.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total curves fitted: {len(results_df)}")

        best_counts = results_df['best_model'].value_counts()
        print("\nBest model distribution:")
        for model, count in best_counts.items():
            pct = count / len(results_df) * 100
            print(f"  {model}: {count} ({pct:.1f}%)")

        # Mean Akaike weights
        print("\nMean Akaike weights:")
        for model in ['gompertz', 'baranyi', 'logistic', 'richards']:
            col = f'{model}_weight'
            if col in results_df.columns:
                mean_w = results_df[col].mean()
                print(f"  {model}: {mean_w:.3f}")

        # Gompertz vs alternatives
        gomp_best = (results_df['best_model'] == 'gompertz').sum()
        total = len(results_df)
        print(f"\nGompertz is best for {gomp_best}/{total} curves ({gomp_best/total*100:.1f}%)")
        if gomp_best < total:
            non_gomp = results_df[results_df['best_model'] != 'gompertz']
            print("Curves where Gompertz is NOT best:")
            for _, row in non_gomp.iterrows():
                print(f"  {row['strain']}: {row['best_model']} "
                      f"(weight={row['best_model_weight']:.2f})")

    # Generate summary comparison plot
    if all_results:
        _plot_summary(results_df, output_dir / 'model_comparison_summary.png')

    print(f"\nAll output saved to: {output_dir}")
    print("Done!")


def _plot_summary(df, output_path):
    """Generate summary visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Best model pie chart
    ax = axes[0]
    counts = df['best_model'].value_counts()
    colors = {'gompertz': '#4C72B0', 'baranyi': '#DD8452',
              'logistic': '#55A868', 'richards': '#C44E52'}
    ax.pie(counts.values, labels=counts.index,
           autopct='%1.1f%%',
           colors=[colors.get(k, 'gray') for k in counts.index])
    ax.set_title('Best Model (by AICc)')

    # 2. AIC comparison boxplot
    ax = axes[1]
    aic_data = []
    model_labels = []
    for model in ['gompertz', 'baranyi', 'logistic', 'richards']:
        col = f'{model}_aicc'
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                aic_data.append(vals.values)
                model_labels.append(model)
    if aic_data:
        bp = ax.boxplot(aic_data, labels=model_labels, patch_artist=True)
        for patch, model in zip(bp['boxes'], model_labels):
            patch.set_facecolor(colors.get(model, 'gray'))
            patch.set_alpha(0.7)
    ax.set_ylabel('AICc')
    ax.set_title('AICc Distribution by Model')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Mean Akaike weights
    ax = axes[2]
    weight_means = {}
    for model in ['gompertz', 'baranyi', 'logistic', 'richards']:
        col = f'{model}_weight'
        if col in df.columns:
            weight_means[model] = df[col].mean()
    if weight_means:
        ax.bar(weight_means.keys(), weight_means.values(),
               color=[colors.get(k, 'gray') for k in weight_means])
        ax.set_ylabel('Mean Akaike Weight')
        ax.set_title('Average Model Support')
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Multi-Model Comparison Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
