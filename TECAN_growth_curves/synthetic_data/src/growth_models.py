"""
Growth Models for Bacterial Growth Curve Generation

Implements multiple bacterial growth models:
- Gompertz (modified) - matches existing analysis pipeline
- Baranyi-Roberts - better stationary phase fit
- Logistic - symmetric S-curve
- Richards - flexible shape parameter
- Death phase extension - models decline after plateau

References:
- Zwietering et al. (1990) - Gompertz model
- Baranyi & Roberts (1994) - Baranyi model
- Richards (1959) - Richards growth function
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """Container for growth model parameters with metadata."""
    model_type: str
    params: dict
    description: str = ""


class GompertzModel:
    """
    Modified Gompertz growth model (matches existing TECAN analysis pipeline).

    Equation:
        y(t) = A * exp(-exp((mu * e / A) * (lambda - t) + 1))

    Parameters:
        A (float): Maximum OD600 (asymptotic value)
        mu (float): Maximum specific growth rate (OD/hour)
        lambda_ (float): Lag phase duration (hours)

    This is the standard model used in the existing analysis pipeline.
    """

    name = "gompertz"

    @staticmethod
    def compute(t: np.ndarray, A: float, mu: float, lambda_: float) -> np.ndarray:
        """
        Compute OD600 values for given time points.

        Args:
            t: Time values in hours
            A: Maximum OD600 (asymptotic value)
            mu: Maximum specific growth rate (OD/hour)
            lambda_: Lag phase duration (hours)

        Returns:
            OD600 values at each time point
        """
        return A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))

    @staticmethod
    def derivative(t: np.ndarray, A: float, mu: float, lambda_: float) -> np.ndarray:
        """
        Compute instantaneous growth rate (dOD/dt).

        Useful for identifying inflection points and growth phases.
        """
        inner = (mu * np.e / A) * (lambda_ - t) + 1
        return A * np.exp(-np.exp(inner)) * np.exp(inner) * (mu * np.e / A)

    @staticmethod
    def inflection_point(A: float, mu: float, lambda_: float) -> Tuple[float, float]:
        """
        Calculate the inflection point (time, OD) where growth rate is maximum.

        Returns:
            (time_inflection, od_inflection)
        """
        t_inf = lambda_ + A / (mu * np.e)
        od_inf = A / np.e
        return t_inf, od_inf

    @staticmethod
    def time_to_reach(target_od: float, A: float, mu: float, lambda_: float) -> float:
        """Calculate time to reach a target OD600 value."""
        if target_od >= A:
            return np.inf
        return lambda_ - (A / (mu * np.e)) * (np.log(-np.log(target_od / A)) - 1)


class BaranyiModel:
    """
    Baranyi-Roberts growth model - better fit for stationary phase.

    This model explicitly accounts for the physiological state of cells
    during the lag phase. Often provides better fits than Gompertz,
    especially for the transition to stationary phase.

    Simplified Baranyi equation:
        y(t) = y0 + mu_max * A(t) - ln(1 + (exp(mu_max * A(t)) - 1) / exp(y_max - y0))

    Where:
        A(t) = t + (1/mu_max) * ln(exp(-mu_max*t) + exp(-h0) - exp(-mu_max*t - h0))

    Parameters:
        y0 (float): Initial OD600 (or log10(N0))
        y_max (float): Maximum OD600 (or log10(Nmax))
        mu_max (float): Maximum specific growth rate
        h0 (float): Dimensionless parameter = mu_max * lag
    """

    name = "baranyi"

    @staticmethod
    def _adjustment_function(t: np.ndarray, mu_max: float, h0: float) -> np.ndarray:
        """
        Compute the adjustment function A(t) that models lag phase.

        A(t) represents the "effective time" accounting for the initial
        physiological state of the cells.
        """
        # Prevent numerical overflow
        with np.errstate(over='ignore', invalid='ignore'):
            term1 = np.exp(-mu_max * t)
            term2 = np.exp(-h0)
            term3 = np.exp(-mu_max * t - h0)

            inner = term1 + term2 - term3
            # Clip to avoid log of negative numbers
            inner = np.clip(inner, 1e-10, None)

            A_t = t + (1.0 / mu_max) * np.log(inner)

        return np.nan_to_num(A_t, nan=0.0, posinf=t[-1] if len(t) > 0 else 100)

    @staticmethod
    def compute(t: np.ndarray, y0: float, y_max: float, mu_max: float, h0: float) -> np.ndarray:
        """
        Compute OD600 values using Baranyi model.

        Args:
            t: Time values in hours
            y0: Initial OD600
            y_max: Maximum OD600
            mu_max: Maximum specific growth rate
            h0: Dimensionless lag parameter (h0 = mu_max * lag_time)

        Returns:
            OD600 values at each time point
        """
        A_t = BaranyiModel._adjustment_function(t, mu_max, h0)

        with np.errstate(over='ignore', invalid='ignore'):
            growth_term = mu_max * A_t
            saturation = np.log(1 + (np.exp(growth_term) - 1) / np.exp(y_max - y0))
            y = y0 + growth_term - saturation

        return np.clip(np.nan_to_num(y, nan=y0, posinf=y_max), y0, y_max)

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             initial_od: float = 0.01) -> dict:
        """
        Convert Gompertz parameters to approximate Baranyi parameters.

        This allows generating curves with Baranyi model using parameters
        extracted from Gompertz fits.

        Args:
            A: Gompertz A parameter (max OD)
            mu: Gompertz mu parameter (growth rate)
            lambda_: Gompertz lambda parameter (lag time)
            initial_od: Initial OD600 value

        Returns:
            dict with Baranyi parameters: y0, y_max, mu_max, h0
        """
        return {
            'y0': initial_od,
            'y_max': A,
            'mu_max': mu,
            'h0': mu * lambda_  # h0 = mu_max * lag
        }


class LogisticModel:
    """
    Logistic growth model - symmetric S-curve.

    Equation:
        y(t) = A / (1 + exp(-k * (t - t_mid)))

    Or with initial OD:
        y(t) = y0 + (A - y0) / (1 + exp(-k * (t - t_mid)))

    Parameters:
        A (float): Maximum OD600 (carrying capacity)
        k (float): Growth rate constant
        t_mid (float): Time at inflection point (half-maximum)
        y0 (float): Initial OD600 (optional baseline)

    The logistic model produces a symmetric curve around the inflection point,
    unlike Gompertz which is asymmetric.
    """

    name = "logistic"

    @staticmethod
    def compute(t: np.ndarray, A: float, k: float, t_mid: float,
                y0: float = 0.0) -> np.ndarray:
        """
        Compute OD600 values using logistic model.

        Args:
            t: Time values in hours
            A: Maximum OD600 (carrying capacity)
            k: Growth rate constant
            t_mid: Time at inflection point
            y0: Initial/baseline OD600

        Returns:
            OD600 values at each time point
        """
        return y0 + (A - y0) / (1 + np.exp(-k * (t - t_mid)))

    @staticmethod
    def derivative(t: np.ndarray, A: float, k: float, t_mid: float,
                   y0: float = 0.0) -> np.ndarray:
        """Compute instantaneous growth rate (dOD/dt)."""
        exp_term = np.exp(-k * (t - t_mid))
        return k * (A - y0) * exp_term / (1 + exp_term)**2

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float) -> dict:
        """
        Convert Gompertz parameters to approximate Logistic parameters.

        Args:
            A: Gompertz A (max OD)
            mu: Gompertz mu (growth rate)
            lambda_: Gompertz lambda (lag time)

        Returns:
            dict with Logistic parameters: A, k, t_mid, y0
        """
        # For logistic: max growth rate = k * A / 4
        # For Gompertz: max growth rate = mu / e
        # So: k = 4 * mu / (e * A)
        k = 4 * mu / (np.e * A)

        # Inflection point for Gompertz is at t = lambda + A/(mu*e)
        # For logistic, inflection is at t_mid
        t_mid = lambda_ + A / (mu * np.e)

        return {
            'A': A,
            'k': k,
            't_mid': t_mid,
            'y0': 0.0
        }


class RichardsModel:
    """
    Richards growth model - flexible asymmetric sigmoid.

    Equation:
        y(t) = A * (1 + nu * exp(1 + nu) * exp(-k*(t - t_mid)))^(-1/nu)

    Or simplified:
        y(t) = A / (1 + exp(-k*(t - t_mid)))^(1/nu)

    Parameters:
        A (float): Maximum OD600
        k (float): Growth rate constant
        t_mid (float): Inflection time
        nu (float): Shape parameter
            - nu = 1 gives logistic model
            - nu -> 0 approaches Gompertz model
            - nu > 1 gives slower approach to plateau

    The Richards model is a generalization that includes both Gompertz
    and Logistic as special cases.
    """

    name = "richards"

    @staticmethod
    def compute(t: np.ndarray, A: float, k: float, t_mid: float,
                nu: float = 1.0, y0: float = 0.0) -> np.ndarray:
        """
        Compute OD600 values using Richards model.

        Args:
            t: Time values in hours
            A: Maximum OD600
            k: Growth rate constant
            t_mid: Time at inflection point
            nu: Shape parameter (1 = logistic, smaller = more Gompertz-like)
            y0: Initial/baseline OD600

        Returns:
            OD600 values at each time point
        """
        if abs(nu) < 1e-6:
            # nu ≈ 0: use Gompertz approximation
            return GompertzModel.compute(t, A, k * A / np.e, t_mid - A / (k * np.e))

        with np.errstate(over='ignore', invalid='ignore'):
            base = 1 + nu * np.exp(-k * (t - t_mid))
            # Clip to avoid negative bases with fractional exponents
            base = np.clip(base, 1e-10, None)
            y = y0 + (A - y0) * np.power(base, -1.0 / nu)

        return np.nan_to_num(y, nan=y0, posinf=A)

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             nu: float = 0.5) -> dict:
        """
        Convert Gompertz parameters to Richards parameters.

        Args:
            A: Gompertz A (max OD)
            mu: Gompertz mu (growth rate)
            lambda_: Gompertz lambda (lag time)
            nu: Desired shape parameter

        Returns:
            dict with Richards parameters
        """
        # Approximate conversion
        k = mu * np.e / A * (1 + nu)
        t_mid = lambda_ + A / (mu * np.e)

        return {
            'A': A,
            'k': k,
            't_mid': t_mid,
            'nu': nu,
            'y0': 0.0
        }


class HaldaneModel:
    """
    Haldane/Andrews substrate inhibition growth model.

    Coupled ODE system:
        dX/dt = mu(S) * X * (1 - X/X_max)
        dS/dt = -q * mu(S) * X

    where mu(S) = mu_max * S / (Ks + S + S^2/Ki)

    Parameters:
        mu_max (float): Maximum specific growth rate (per hour)
        Ks (float): Half-saturation constant
        Ki (float): Substrate inhibition constant
        X_max (float): Maximum biomass (carrying capacity, OD units)
        q (float): Substrate consumption rate per unit biomass growth
        X0 (float): Initial biomass (OD600)
        S0 (float): Initial substrate concentration

    The Haldane model is mechanistic: it predicts both biomass growth and
    substrate depletion, and captures feedback inhibition at high substrate
    concentrations. At low S, growth is Monod-like; at high S, growth is
    inhibited (mu decreases).

    References:
        - Andrews (1968) - A mathematical model for the continuous culture
          of microorganisms utilizing inhibitory substrates
        - Haldane (1930) - Enzymes
    """

    name = "haldane"

    @staticmethod
    def _ode(t, y, mu_max, Ks, Ki, X_max, q):
        """Haldane ODE system."""
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

    @staticmethod
    def compute(t: np.ndarray, mu_max: float, Ks: float, Ki: float,
                X_max: float, q: float, X0: float, S0: float) -> np.ndarray:
        """
        Compute biomass OD600 over time using Haldane kinetics.

        Args:
            t: Time values in hours
            mu_max: Maximum specific growth rate
            Ks: Half-saturation constant
            Ki: Substrate inhibition constant
            X_max: Carrying capacity (max OD)
            q: Substrate consumption coefficient
            X0: Initial biomass OD
            S0: Initial substrate concentration

        Returns:
            OD600 (biomass) values at each time point
        """
        from scipy.integrate import solve_ivp

        try:
            sol = solve_ivp(
                HaldaneModel._ode,
                [t[0], t[-1]],
                [X0, S0],
                args=(mu_max, Ks, Ki, X_max, q),
                t_eval=t,
                method='RK45',
                max_step=0.5,
                rtol=1e-8,
                atol=1e-10
            )
            if sol.success:
                return sol.y[0]  # Biomass (X)
            else:
                return np.full_like(t, X0, dtype=float)
        except Exception:
            return np.full_like(t, X0, dtype=float)

    @staticmethod
    def compute_full(t: np.ndarray, mu_max: float, Ks: float, Ki: float,
                     X_max: float, q: float, X0: float, S0: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both biomass and substrate over time.

        Returns:
            (X(t), S(t)) -- biomass and substrate arrays
        """
        from scipy.integrate import solve_ivp

        try:
            sol = solve_ivp(
                HaldaneModel._ode,
                [t[0], t[-1]],
                [X0, S0],
                args=(mu_max, Ks, Ki, X_max, q),
                t_eval=t,
                method='RK45',
                max_step=0.5,
                rtol=1e-8,
                atol=1e-10
            )
            if sol.success:
                return sol.y[0], sol.y[1]
            else:
                return np.full_like(t, X0, dtype=float), np.full_like(t, S0, dtype=float)
        except Exception:
            return np.full_like(t, X0, dtype=float), np.full_like(t, S0, dtype=float)

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             S0: float = 1.0, Ki: float = 10.0) -> dict:
        """
        Convert Gompertz parameters to approximate Haldane parameters.

        This provides a reasonable starting point. The Gompertz mu is the
        maximum slope of the OD curve; the Haldane mu_max is the intrinsic
        maximum specific growth rate (generally higher).

        Args:
            A: Gompertz A (max OD)
            mu: Gompertz mu (growth rate)
            lambda_: Gompertz lambda (lag time)
            S0: Initial substrate concentration
            Ki: Substrate inhibition constant

        Returns:
            dict with Haldane parameters
        """
        return {
            'mu_max': mu * 2.5,     # Haldane mu_max > apparent growth rate
            'Ks': 0.1,              # Half-saturation (typical)
            'Ki': Ki,               # Inhibition constant
            'X_max': A,             # Carrying capacity ≈ Gompertz A
            'q': 0.1,              # Substrate consumption coefficient
            'X0': 0.01,            # Initial biomass
            'S0': S0,              # Initial substrate
        }


class DeathPhaseExtension:
    """
    Extends any growth model with a death/decline phase.

    After the plateau time (t_death), OD decreases exponentially:
        y_death(t) = y_plateau * exp(-k_death * (t - t_death))

    This models cell lysis and OD decline observed in extended experiments.

    Parameters:
        base_model: Any growth model class
        t_death (float): Time when death phase begins (hours)
        k_death (float): Death rate constant (per hour)
        decline_fraction (float): How much OD drops (0-1), e.g., 0.3 = 30% decline
    """

    name = "death_phase"

    def __init__(self, base_model):
        """
        Initialize with a base growth model.

        Args:
            base_model: Growth model class (GompertzModel, BaranyiModel, etc.)
        """
        self.base_model = base_model

    def compute(self, t: np.ndarray, t_death: float, k_death: float,
                **base_params) -> np.ndarray:
        """
        Compute OD600 with growth followed by death phase.

        Args:
            t: Time values in hours
            t_death: Time when death phase begins
            k_death: Death rate constant
            **base_params: Parameters for the base growth model

        Returns:
            OD600 values including death phase decline
        """
        # Compute base growth curve
        y_growth = self.base_model.compute(t, **base_params)

        # Get OD at death phase start
        y_at_death = self.base_model.compute(np.array([t_death]), **base_params)[0]

        # Apply death phase decay after t_death
        y = y_growth.copy()
        death_mask = t > t_death
        y[death_mask] = y_at_death * np.exp(-k_death * (t[death_mask] - t_death))

        return y

    @staticmethod
    def compute_with_model(t: np.ndarray, model_type: str, t_death: float,
                           k_death: float, **params) -> np.ndarray:
        """
        Convenience method to compute with any model type.

        Args:
            t: Time values
            model_type: 'gompertz', 'baranyi', 'logistic', or 'richards'
            t_death: Death phase start time
            k_death: Death rate constant
            **params: Model-specific parameters
        """
        models = {
            'gompertz': GompertzModel,
            'baranyi': BaranyiModel,
            'logistic': LogisticModel,
            'richards': RichardsModel,
            'haldane': HaldaneModel
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        extension = DeathPhaseExtension(models[model_type])
        return extension.compute(t, t_death, k_death, **params)


# =============================================================================
# Utility Functions
# =============================================================================

def get_model(model_type: str):
    """
    Get a growth model class by name.

    Args:
        model_type: 'gompertz', 'baranyi', 'logistic', or 'richards'

    Returns:
        Model class
    """
    models = {
        'gompertz': GompertzModel,
        'baranyi': BaranyiModel,
        'logistic': LogisticModel,
        'richards': RichardsModel,
        'haldane': HaldaneModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")

    return models[model_type]


def convert_parameters(from_model: str, to_model: str, **params) -> dict:
    """
    Convert parameters between different growth models.

    Currently supports conversion FROM Gompertz to other models.

    Args:
        from_model: Source model type
        to_model: Target model type
        **params: Parameters of source model

    Returns:
        dict of parameters for target model
    """
    if from_model != 'gompertz':
        raise NotImplementedError("Currently only supports conversion from Gompertz")

    converters = {
        'baranyi': BaranyiModel.from_gompertz_params,
        'logistic': LogisticModel.from_gompertz_params,
        'richards': RichardsModel.from_gompertz_params,
        'haldane': HaldaneModel.from_gompertz_params
    }

    if to_model == 'gompertz':
        return params  # No conversion needed

    if to_model not in converters:
        raise ValueError(f"Unknown target model: {to_model}")

    return converters[to_model](**params)


def generate_curve(t: np.ndarray, model_type: str,
                   include_death_phase: bool = False,
                   t_death: Optional[float] = None,
                   k_death: float = 0.02,
                   **params) -> np.ndarray:
    """
    Generate a growth curve using specified model.

    Convenience function that handles model selection and optional death phase.

    Args:
        t: Time values in hours
        model_type: 'gompertz', 'baranyi', 'logistic', or 'richards'
        include_death_phase: Whether to add death/decline phase
        t_death: Time when death phase begins (auto-calculated if None)
        k_death: Death rate constant
        **params: Model-specific parameters

    Returns:
        OD600 values at each time point
    """
    model = get_model(model_type)

    if include_death_phase:
        # Auto-calculate t_death if not provided (90% of max time)
        if t_death is None:
            t_death = t[-1] * 0.7

        return DeathPhaseExtension.compute_with_model(
            t, model_type, t_death, k_death, **params
        )
    else:
        return model.compute(t, **params)


# =============================================================================
# Testing / Demonstration
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate time array (0 to 50 hours, 0.25h resolution)
    t = np.arange(0, 50, 0.25)

    # Standard Gompertz parameters
    A, mu, lambda_ = 1.5, 0.2, 3.0

    # Generate curves with different models
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Compare all models
    ax1 = axes[0, 0]
    y_gompertz = GompertzModel.compute(t, A, mu, lambda_)

    baranyi_params = BaranyiModel.from_gompertz_params(A, mu, lambda_)
    y_baranyi = BaranyiModel.compute(t, **baranyi_params)

    logistic_params = LogisticModel.from_gompertz_params(A, mu, lambda_)
    y_logistic = LogisticModel.compute(t, **logistic_params)

    richards_params = RichardsModel.from_gompertz_params(A, mu, lambda_)
    y_richards = RichardsModel.compute(t, **richards_params)

    ax1.plot(t, y_gompertz, 'b-', label='Gompertz', linewidth=2)
    ax1.plot(t, y_baranyi, 'r--', label='Baranyi', linewidth=2)
    ax1.plot(t, y_logistic, 'g-.', label='Logistic', linewidth=2)
    ax1.plot(t, y_richards, 'm:', label='Richards (nu=0.5)', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('OD600')
    ax1.set_title('Growth Model Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Death phase extension
    ax2 = axes[0, 1]
    y_no_death = GompertzModel.compute(t, A, mu, lambda_)
    y_death = DeathPhaseExtension.compute_with_model(
        t, 'gompertz', t_death=30, k_death=0.03, A=A, mu=mu, lambda_=lambda_
    )

    ax2.plot(t, y_no_death, 'b-', label='Without death phase', linewidth=2)
    ax2.plot(t, y_death, 'r-', label='With death phase', linewidth=2)
    ax2.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='Death phase start')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('OD600')
    ax2.set_title('Death Phase Extension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Richards model with different nu values
    ax3 = axes[1, 0]
    for nu in [0.2, 0.5, 1.0, 2.0]:
        params = RichardsModel.from_gompertz_params(A, mu, lambda_, nu=nu)
        y = RichardsModel.compute(t, **params)
        ax3.plot(t, y, label=f'nu={nu}', linewidth=2)

    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('OD600')
    ax3.set_title('Richards Model - Shape Parameter Effect')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Growth rate comparison
    ax4 = axes[1, 1]
    dy_gompertz = GompertzModel.derivative(t, A, mu, lambda_)
    dy_logistic = LogisticModel.derivative(t, **logistic_params)

    ax4.plot(t, dy_gompertz, 'b-', label='Gompertz', linewidth=2)
    ax4.plot(t, dy_logistic, 'g--', label='Logistic', linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Growth Rate (dOD/dt)')
    ax4.set_title('Instantaneous Growth Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('growth_models_demo.png', dpi=150)
    plt.show()

    print("Demo complete! Saved to growth_models_demo.png")
