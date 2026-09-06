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


# =============================================================================
# Shape measurement -- ONE convention shared by the converters, the truth
# table (WP3) and the tests.
# =============================================================================

#: Grid used to measure a converted curve when the caller gives none.
#: 0-100 h at 0.25 h is the grid every shipped scenario is generated on
#: (ground_truth.csv: duration_hours 99.75, n_points 400).
DEFAULT_GRID = np.arange(0.0, 100.0, 0.25)

#: Relative tolerance used by the ``realised`` flag returned by every
#: ``from_gompertz_params``.
REALISATION_TOL = 0.02

#: Metadata keys that ``from_gompertz_params`` adds to the parameter dict.
#: Every ``compute`` accepts and ignores them so that the returned dict can
#: still be splatted straight into ``Model.compute(t, **params)``.
META_KEYS = ('realised', 'achieved')


def _reject_unknown(meta: dict) -> None:
    """Allow only the documented metadata keys to reach a ``compute`` call."""
    unknown = sorted(k for k in meta if k not in META_KEYS)
    if unknown:
        raise TypeError(
            f"compute() got unexpected keyword argument(s): {', '.join(unknown)}"
        )


def _as_grid(t: Optional[np.ndarray]) -> np.ndarray:
    """Return the caller's time grid, defaulting to DEFAULT_GRID."""
    if t is None:
        return DEFAULT_GRID
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError("time grid must be a 1-D array with at least 3 points")
    return t


def measure_shape(y: np.ndarray, t: np.ndarray) -> dict:
    """
    Measure the shape of a curve with the pipeline-facing convention.

    This is the single definition used by the converters below, by the
    ground-truth writer and by the tests, so that "the curve has mu = x"
    means exactly one thing everywhere.

    Convention:
        max_slope   = max(np.gradient(y, t))
        tangent_lag = t[i] - (y[i] - y[0]) / max_slope   at i = argmax
        plateau     = max(y)
        y0          = y[0]

    Note that ``tangent_lag`` subtracts the *baseline* y[0]. For a model whose
    curve does not start at ~0 (a logistic with a short lag starts at up to
    ~6 % of A) this is larger than the lag the converter targeted by exactly
    y0 / max_slope; ``tangent_lag - y0 / max_slope`` is the baseline-free
    tangent lag the converters solve for. Both numbers are honest, they just
    answer different questions, so both are recoverable from this dict.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.shape != t.shape:
        raise ValueError("y and t must have the same shape")

    d = np.gradient(y, t)
    i = int(np.nanargmax(d))
    slope = float(d[i])
    y0 = float(y[0])
    if not np.isfinite(slope) or slope <= 0:
        lag = float('nan')
    else:
        lag = float(t[i] - (y[i] - y0) / slope)

    return {
        'max_slope': slope,
        'tangent_lag': lag,
        'plateau': float(np.nanmax(y)),
        'y0': y0,
    }


def compute_od(model, t: np.ndarray, **params) -> np.ndarray:
    """
    Evaluate any model class on ``t`` in OD space.

    Every ``Model.compute`` in this module returns OD600, Baranyi included
    (its *parameters* are log space; its output is not -- see
    :class:`BaranyiModel`), so this is a thin, uniform wrapper. It stays as
    the single place model dispatch happens, and as the name to call when a
    caller wants OD stated rather than assumed.
    """
    return model.compute(t, **params)


def curve_from_params(model_name: str, params: dict,
                      t: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Evaluate a model on ``t`` from a parameter dict, ignoring metadata keys.

    The result is OD600 for every model (Baranyi's parameters are log space,
    its curve is not).
    """
    t = _as_grid(t)
    kwargs = {k: v for k, v in params.items() if k not in META_KEYS}
    return compute_od(get_model(model_name), t, **kwargs)


def shape_of(model_name: str, params: dict,
             t: Optional[np.ndarray] = None) -> dict:
    """
    Measure {max_slope, tangent_lag, plateau, y0} for a model + parameter dict.

    Args:
        model_name: 'gompertz', 'baranyi', 'logistic', 'richards' or 'haldane'
        params: the dict returned by ``from_gompertz_params`` (metadata keys
            are ignored) or any explicit parameter dict for that model
        t: time grid; DEFAULT_GRID (0-100 h at 0.25 h) when None

    Returns:
        dict as documented on :func:`measure_shape`
    """
    t = _as_grid(t)
    return measure_shape(curve_from_params(model_name, params, t), t)


def _achieved_from_shape(shape: dict) -> dict:
    """The (A, mu, lambda) a curve actually has, as WP3 records it."""
    return {
        'A': shape['plateau'],
        'mu': shape['max_slope'],
        'lambda': shape['tangent_lag'],
    }


#: Floor on the absolute lag tolerance, in hours. The tangent lag measured
#: with np.gradient on a 0.25 h grid carries ~0.01 h of discretisation error
#: (the max-slope estimate is ~0.25 % low, which moves the tangent intercept),
#: so 0.05 h is comfortably above the noise and still far below one sample.
LAG_TOL_FLOOR_H = 0.05


def _lag_tolerance(lambda_: float, t: np.ndarray) -> float:
    """Absolute lag tolerance: 2 %, never tighter than the measurement noise."""
    return max(REALISATION_TOL * abs(float(lambda_)), LAG_TOL_FLOOR_H)


def _realised(A: float, mu: float, lambda_: float, shape: dict,
              t: np.ndarray, solver_ok: bool = True) -> bool:
    """
    Did the converter hit the shape it was asked for?

    True when, on the caller's grid, the plateau is within REALISATION_TOL of
    A, the maximum slope within REALISATION_TOL of mu, and the *baseline-free*
    tangent lag (``tangent_lag - y0 / max_slope``, the quantity the closed
    forms and the root-finders below actually solve) within REALISATION_TOL of
    lambda_ or LAG_TOL_FLOOR_H, whichever is looser.

    ``solver_ok=False`` (a bracket that did not exist, a root-finder that did
    not converge, a capped mu_max) forces False regardless of the measurement.
    """
    if not solver_ok:
        return False
    slope = shape['max_slope']
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(shape['tangent_lag']):
        return False
    if abs(shape['plateau'] - A) > REALISATION_TOL * abs(A):
        return False
    if abs(slope - mu) > REALISATION_TOL * abs(mu):
        return False
    lag_free = shape['tangent_lag'] - shape['y0'] / slope
    return bool(abs(lag_free - lambda_) <= _lag_tolerance(lambda_, t))


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
    def compute(t: np.ndarray, A: float, mu: float, lambda_: float,
                **_meta) -> np.ndarray:
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
        _reject_unknown(_meta)
        return A * np.exp(-np.exp((mu * np.e / A) * (lambda_ - t) + 1))

    @staticmethod
    def derivative(t: np.ndarray, A: float, mu: float, lambda_: float,
                   **_meta) -> np.ndarray:
        """
        Compute instantaneous growth rate (dOD/dt).

        Useful for identifying inflection points and growth phases.
        """
        _reject_unknown(_meta)
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

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             t: Optional[np.ndarray] = None) -> dict:
        """
        Identity converter, present so every model has the same interface.

        The Zwietering parameterisation used above already has max slope
        exactly mu and tangent-intercept lag exactly lambda_, so nothing is
        converted; the shape is still *measured* on the caller's grid so that
        callers get the same ``realised``/``achieved`` contract as for the
        other models (a curve that has not plateaued inside the run, e.g. a
        very slow evaporation-drift curve, honestly reports realised=False).

        Args:
            A: Maximum OD600
            mu: Maximum slope (OD/h)
            lambda_: Lag time (h)
            t: Grid the shape is measured on (DEFAULT_GRID when None)

        Returns:
            dict with A, mu, lambda_, plus 'realised' and 'achieved'
        """
        t = _as_grid(t)
        params = {'A': A, 'mu': mu, 'lambda_': lambda_}
        shape = measure_shape(GompertzModel.compute(t, **params), t)
        params['realised'] = _realised(A, mu, lambda_, shape, t)
        params['achieved'] = _achieved_from_shape(shape)
        return params


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
        y0 (float): Initial value in LOG space, ln(OD600)
        y_max (float): Asymptotic value in LOG space, ln(OD600)
        mu_max (float): Maximum specific growth rate (per hour, log space)
        h0 (float): Dimensionless parameter = mu_max * lag

    Spaces (read this before calling anything):
        The *parameters* are log space -- that is what the Baranyi-Roberts
        model is, and :meth:`from_gompertz_params` returns ln(OD) for y0 and
        y_max. The *output of* :meth:`compute` is OD600, exactly like
        GompertzModel.compute, LogisticModel.compute and
        RichardsModel.compute, so that ``Model.compute(t, **params)`` means
        the same thing for every model in this module. The raw log-space
        equation is :meth:`compute_log`; :meth:`compute_od` is a spelled-out
        alias of :meth:`compute` for call sites that want to say which space
        they are in.
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
    def compute_log(t: np.ndarray, y0: float, y_max: float, mu_max: float,
                    h0: float, **_meta) -> np.ndarray:
        """
        Evaluate the Baranyi-Roberts equation itself, in LOG space.

        y is ln(cell concentration), so y0 and y_max are ln(OD). Feeding raw
        OD in as y0/y_max makes the saturation term
        ln(1 + (exp(mu*A_t)-1)/exp(y_max-y0)) act on a span of ~1.2 instead of
        ~5 log units, which saturates the curve from the first step; that bug
        cost the shipped baranyi_generated curves 61 % of their growth rate.

        Callers who want a growth curve want :meth:`compute`, which is this
        exponentiated; this method exists for anyone working in log space
        (a log-space fit, a diagnostic plot of ln(OD)).

        Args:
            t: Time values in hours
            y0: Initial value, ln(OD)
            y_max: Asymptotic value, ln(OD)
            mu_max: Maximum specific growth rate (per hour, in log space)
            h0: Dimensionless lag parameter (h0 = mu_max * lag_time)

        Returns:
            ln(OD) values at each time point
        """
        _reject_unknown(_meta)
        A_t = BaranyiModel._adjustment_function(t, mu_max, h0)

        with np.errstate(over='ignore', invalid='ignore'):
            growth_term = mu_max * A_t
            saturation = np.log(1 + (np.exp(growth_term) - 1) / np.exp(y_max - y0))
            y = y0 + growth_term - saturation

        return np.clip(np.nan_to_num(y, nan=y0, posinf=y_max), y0, y_max)

    @staticmethod
    def compute(t: np.ndarray, y0: float, y_max: float, mu_max: float,
                h0: float, **_meta) -> np.ndarray:
        """
        Compute the Baranyi-Roberts growth curve in OD600: exp(compute_log(...)).

        The parameters are log space (y0 = ln(OD0), y_max = ln(A)) because the
        model is; the returned curve is OD, like every other ``compute`` in
        this module. That symmetry is the point: ``Model.compute(t, **params)``
        with the dict ``Model.from_gompertz_params`` returned is the call every
        generator and plotter in this repo makes, and it must mean "the growth
        curve" for all five models. When it meant ln(OD) for Baranyi alone, the
        generator produced curves peaking at 0.26 instead of 1.25 and the
        model-comparison figure plotted one line spanning -4.6..0.18 against
        three spanning 0..1.2, with nothing raising.

        Args:
            t: Time values in hours
            y0: Initial value, ln(OD)
            y_max: Asymptotic value, ln(OD)
            mu_max: Maximum specific growth rate (per hour, in log space)
            h0: Dimensionless lag parameter (h0 = mu_max * lag_time)

        Returns:
            OD600 values at each time point
        """
        _reject_unknown(_meta)
        return np.exp(BaranyiModel.compute_log(t, y0, y_max, mu_max, h0))

    @staticmethod
    def compute_od(t: np.ndarray, y0: float, y_max: float, mu_max: float,
                   h0: float, **_meta) -> np.ndarray:
        """
        The Baranyi curve in OD space -- an explicit alias of :meth:`compute`.

        Kept as a name a call site can use to say out loud which space it
        wants, and so that code written against the intermediate contract
        (where ``compute`` was log space and ``compute_od`` the OD entry
        point) keeps returning OD.
        """
        _reject_unknown(_meta)
        return BaranyiModel.compute(t, y0, y_max, mu_max, h0)

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             initial_od: float = 0.01,
                             t: Optional[np.ndarray] = None) -> dict:
        """
        Solve Baranyi parameters that realise a Gompertz (A, mu, lambda).

        The returned parameters are LOG SPACE (y0 = ln(OD0), y_max = ln(A)).
        Evaluating them with ``BaranyiModel.compute(t, **params)`` gives OD600
        -- the same contract as the other four models -- and
        ``compute_log(t, **params)`` gives ln(OD).

        mu_max is solved by brentq over [0.5*mu, 30*mu] so that the OD-space
        maximum slope max(np.gradient(exp(y), t)) equals mu on the caller's
        grid. h0 is then solved by brentq over [1e-9, 50] so that the
        baseline-free tangent lag equals lambda_.

        A Baranyi curve rising from OD0 to A at slope mu has an intrinsic
        minimum lag (the time the exponential phase itself takes), reached at
        h0 -> 0. For 9 of the 15 shipped baranyi_generated triples the
        requested lambda is below that floor (measured here: 2.61-6.12 h;
        CURVE0390 requests 2.45 h against a floor of 3.91 h). It is NOT
        shipped: h0 stays at the floor, ``realised`` is False and ``achieved``
        carries the lag the curve really has, which the caller records as
        truth. Nothing is raised.

        Args:
            A: Gompertz A parameter (max OD)
            mu: Gompertz mu parameter (max slope, OD/h)
            lambda_: Gompertz lambda parameter (lag time, h)
            initial_od: Initial OD600 (floored at 0.01 so ln is well behaved)
            t: Grid the shape is solved and measured on (DEFAULT_GRID if None)

        Returns:
            dict with y0, y_max, mu_max, h0, plus 'realised' and 'achieved'
        """
        from scipy.optimize import brentq

        t = _as_grid(t)
        y0 = float(np.log(max(float(initial_od), 0.01)))
        y_max = float(np.log(A))

        def shape(mu_max: float, h0: float) -> dict:
            od = BaranyiModel.compute_od(t, y0, y_max, mu_max, h0)
            return measure_shape(od, t)

        def lag_free(mu_max: float, h0: float) -> float:
            s = shape(mu_max, h0)
            if not np.isfinite(s['max_slope']) or s['max_slope'] <= 0:
                return float('nan')
            return s['tangent_lag'] - s['y0'] / s['max_slope']

        solver_ok = True
        mu_max = max(float(mu), 1e-6)
        h0 = 1e-9

        if y_max <= y0 or mu <= 0:
            # Degenerate: no headroom above the inoculum, or no growth asked
            # for at all -- there is no bracket to search.
            solver_ok = False
        else:
            # mu_max and h0 are almost independent (h0 only translates the
            # curve in time), so two passes are plenty.
            for _ in range(2):
                try:
                    mu_max = brentq(
                        lambda m: shape(m, h0)['max_slope'] - mu,
                        0.5 * mu, 30.0 * mu, xtol=1e-12, rtol=1e-13, maxiter=200)
                except (ValueError, RuntimeError):
                    solver_ok = False
                    break

                floor = lag_free(mu_max, 1e-9)
                if not np.isfinite(floor) or lambda_ <= floor:
                    # Below the achievable floor: keep the floor, say so.
                    h0 = 1e-9
                    solver_ok = False
                    break
                try:
                    h0 = brentq(
                        lambda h: lag_free(mu_max, h) - lambda_,
                        1e-9, 50.0, xtol=1e-10, rtol=1e-12, maxiter=200)
                except (ValueError, RuntimeError):
                    solver_ok = False
                    break

        params = {'y0': y0, 'y_max': y_max, 'mu_max': float(mu_max),
                  'h0': float(h0)}
        final = shape(params['mu_max'], params['h0'])
        params['realised'] = _realised(A, mu, lambda_, final, t, solver_ok)
        params['achieved'] = _achieved_from_shape(final)
        return params

    @staticmethod
    def lag_floor(A: float, mu: float, initial_od: float = 0.01,
                  t: Optional[np.ndarray] = None) -> float:
        """
        Smallest baseline-free tangent lag a Baranyi curve of this size can have.

        This is the lag at h0 -> 0 (no physiological lag at all): the
        exponential phase from ln(OD0) to ln(A) at the solved mu_max still
        takes time, and the tangent drawn at the point of maximum OD slope
        crosses the baseline that many hours after t = 0.
        """
        from scipy.optimize import brentq

        t = _as_grid(t)
        y0 = float(np.log(max(float(initial_od), 0.01)))
        y_max = float(np.log(A))
        if y_max <= y0 or mu <= 0:
            return float('nan')

        def slope(m: float) -> float:
            return measure_shape(
                BaranyiModel.compute_od(t, y0, y_max, m, 1e-9), t)['max_slope']

        try:
            mu_max = brentq(lambda m: slope(m) - mu, 0.5 * mu, 30.0 * mu,
                            xtol=1e-12, rtol=1e-13, maxiter=200)
        except (ValueError, RuntimeError):
            return float('nan')

        s = measure_shape(BaranyiModel.compute_od(t, y0, y_max, mu_max, 1e-9), t)
        return float(s['tangent_lag'] - s['y0'] / s['max_slope'])


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
                y0: float = 0.0, **_meta) -> np.ndarray:
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
        _reject_unknown(_meta)
        return y0 + (A - y0) / (1 + np.exp(-k * (t - t_mid)))

    @staticmethod
    def derivative(t: np.ndarray, A: float, k: float, t_mid: float,
                   y0: float = 0.0, **_meta) -> np.ndarray:
        """Compute instantaneous growth rate (dOD/dt)."""
        _reject_unknown(_meta)
        exp_term = np.exp(-k * (t - t_mid))
        return k * (A - y0) * exp_term / (1 + exp_term)**2

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             t: Optional[np.ndarray] = None) -> dict:
        """
        Convert Gompertz parameters to the Logistic parameters that realise them.

        Args:
            A: Gompertz A (max OD)
            mu: Gompertz mu (maximum slope, OD/h)
            lambda_: Gompertz lambda (lag time, h)
            t: Grid the shape is measured on (DEFAULT_GRID when None)

        Returns:
            dict with Logistic parameters A, k, t_mid, y0, plus 'realised'
            and 'achieved'

        Maths:
            Logistic max slope is k*A/4, at t = t_mid where y = A/2.
            In THIS repo's modified Gompertz (see GompertzModel.compute) the
            max slope is exactly mu -- not mu/e. The old code divided by e
            twice over (k = 4*mu/(e*A) and t_mid = lambda + A/(mu*e)), which
            produced a curve with 0.368*mu of slope and essentially no lag,
            and, because t_mid was pushed out to the Gompertz inflection while
            y0 stayed 0, a start at 14-20 % of A.

                k     = 4*mu/A                      (max slope = k*A/4 = mu)
                t_mid = lambda_ + A/(2*mu)          (tangent lag = lambda_)

            y0 stays 0.0: the logistic's own start, y(0) = A/(1+exp(k*t_mid)),
            is the honest consequence of asking a symmetric sigmoid for a
            short lag (up to ~5.7 % of A at the corner of the shipped range
            A 1.0-1.5, mu 0.15-0.25, lambda 2-6 h), not an injected offset.
            That same y(0) is why ``achieved['lambda']`` -- which subtracts
            the baseline, per measure_shape -- exceeds lambda_ by y(0)/mu.
        """
        t = _as_grid(t)
        k = 4.0 * mu / A
        t_mid = lambda_ + A / (2.0 * mu)

        params = {'A': A, 'k': k, 't_mid': t_mid, 'y0': 0.0}
        shape = measure_shape(LogisticModel.compute(t, **params), t)
        params['realised'] = _realised(A, mu, lambda_, shape, t)
        params['achieved'] = _achieved_from_shape(shape)
        return params


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
                nu: float = 1.0, y0: float = 0.0, **_meta) -> np.ndarray:
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
        _reject_unknown(_meta)
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
                             nu: float = 0.5,
                             t: Optional[np.ndarray] = None) -> dict:
        """
        Convert Gompertz parameters to the Richards parameters that realise them.

        Args:
            A: Gompertz A (max OD)
            mu: Gompertz mu (maximum slope, OD/h)
            lambda_: Gompertz lambda (lag time, h)
            nu: Shape parameter (1 = logistic, -> 0 = Gompertz)
            t: Grid the shape is measured on (DEFAULT_GRID when None)

        Returns:
            dict with Richards parameters A, k, t_mid, nu, y0, plus 'realised'
            and 'achieved'

        Maths:
            For y = A*(1 + nu*exp(-k*(t-t_mid)))**(-1/nu) the derivative is
            maximised where nu*exp(-k*(t-t_mid)) = nu, i.e. at t = t_mid, and
            there dy/dt = A*k*(1+nu)**(-1-1/nu) and y = A*(1+nu)**(-1/nu).
            Requiring max slope = mu and tangent lag = lambda_ gives

                k     = mu*(1+nu)**(1 + 1/nu)/A
                t_mid = lambda_ + A*(1+nu)**(-1/nu)/mu

            The old k = mu*e*(1+nu)/A inflated the slope by e*(1+nu)**(-1/nu)
            (1.208x at nu=0.5, the shipped value), and the exponent 1/nu that
            an earlier draft of the fix proposed *deflates* it instead
            (0.83/0.67/0.50/0.33x at nu=0.2/0.5/1.0/2.0). At nu=1 this form
            reduces to the corrected LogisticModel exactly, as it must.
        """
        t = _as_grid(t)
        k = mu * (1.0 + nu) ** (1.0 + 1.0 / nu) / A
        t_mid = lambda_ + A * (1.0 + nu) ** (-1.0 / nu) / mu

        params = {'A': A, 'k': k, 't_mid': t_mid, 'nu': nu, 'y0': 0.0}
        shape = measure_shape(RichardsModel.compute(t, **params), t)
        params['realised'] = _realised(A, mu, lambda_, shape, t)
        params['achieved'] = _achieved_from_shape(shape)
        return params


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
                X_max: float, q: float, X0: float, S0: float,
                **_meta) -> np.ndarray:
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
        _reject_unknown(_meta)
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
                     X_max: float, q: float, X0: float, S0: float,
                     **_meta) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both biomass and substrate over time.

        Returns:
            (X(t), S(t)) -- biomass and substrate arrays
        """
        _reject_unknown(_meta)
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

    #: Largest intrinsic growth rate a converted Haldane curve may use.
    #: 2.0 /h is a ~21 min doubling time -- already at the edge of what a
    #: bacterium does. The strong-inhibition corner (Ki ~ Ks, S0 = 1) would
    #: need mu_max ~ 6 /h to force the requested slope; rather than ship that,
    #: the cap binds and the converter reports realised=False.
    MU_MAX_CAP = 2.0

    @staticmethod
    def _biomass_fast(tau: np.ndarray, Ks: float, Ki: float, X_max: float,
                      q: float, X0: float, S0: float) -> np.ndarray:
        """
        Biomass at mu_max = 1 on the rescaled grid ``tau``, for the solver only.

        LSODA with rtol 1e-8 agrees with the public :meth:`compute`
        (RK45, max_step 0.5) to 8 significant figures on the maximum slope and
        is ~3.5x faster, which matters because the converter integrates this
        system a few dozen times per curve. The public path is left untouched
        so that every already-shipped Haldane curve is byte-identical.
        """
        from scipy.integrate import solve_ivp

        try:
            sol = solve_ivp(
                HaldaneModel._ode, [tau[0], tau[-1]], [X0, S0],
                args=(1.0, Ks, Ki, X_max, q), t_eval=tau,
                method='LSODA', rtol=1e-8, atol=1e-10)
            if sol.success:
                return sol.y[0]
        except Exception:
            pass
        return np.full_like(tau, X0, dtype=float)

    @staticmethod
    def _shape_at(t: np.ndarray, mu_max: float, Ks: float, Ki: float,
                  X_max: float, q: float, X0: float, S0: float) -> dict:
        """
        Shape of the Haldane biomass curve on grid ``t`` for a given mu_max.

        Uses the exact time-rescaling property of the system: mu_max appears
        as a common factor in both dX/dt and dS/dt, so X(t; mu_max) is
        X(mu_max*t; 1). Integrating once on tau = mu_max*t therefore gives the
        curve for any mu_max, which is what makes the nested solve below cheap
        (one ODE integration per candidate, not per iteration).
        """
        tau = np.asarray(t, dtype=float) * mu_max
        X = HaldaneModel._biomass_fast(tau, Ks, Ki, X_max, q, X0, S0)
        return measure_shape(X, t)

    @staticmethod
    def _solve_mu_max(t: np.ndarray, mu: float, Ks: float, Ki: float,
                      X_max: float, q: float, X0: float, S0: float,
                      cap: float, guess: Optional[float] = None
                      ) -> Tuple[float, dict, bool]:
        """
        Find mu_max giving max(dX/dt) == mu for this X0, honouring the cap.

        Fixed-point iteration on mu_max <- mu_max * mu / measured_slope. The
        measured slope is proportional to mu_max except for grid
        discretisation, so this converges in two or three passes; ``guess``
        (the previous X0's answer) usually makes it two.
        """
        m = float(np.clip(guess if guess else 2.5 * mu, 1e-4, cap))
        shape = HaldaneModel._shape_at(t, m, Ks, Ki, X_max, q, X0, S0)
        capped = False
        for _ in range(12):
            slope = shape['max_slope']
            if not np.isfinite(slope) or slope <= 0:
                return m, shape, False
            m_new = m * mu / slope
            if m_new > cap:
                m_new, capped = cap, True
            m_new = float(np.clip(m_new, 1e-4, cap))
            if abs(m_new - m) <= 1e-6 * max(m, 1e-3):
                break
            m = m_new
            shape = HaldaneModel._shape_at(t, m, Ks, Ki, X_max, q, X0, S0)
        return m, shape, not capped

    @staticmethod
    def from_gompertz_params(A: float, mu: float, lambda_: float,
                             S0: float = 1.0, Ki: float = 10.0,
                             Ks: float = 0.1, q: float = 0.1,
                             X0: Optional[float] = None,
                             t: Optional[np.ndarray] = None) -> dict:
        """
        Solve Haldane parameters that realise a Gompertz (A, mu, lambda).

        Ks, Ki, q and S0 are real arguments, not constants baked into the
        body: with them hard-coded, the three haldane_* scenarios differed
        only in the (A, mu, lambda) they sampled, i.e. they were one model
        sampled three ways rather than three inhibition regimes.

        X_max is A. X0 is found by brentq over log10(X0) in [1e-6, min(0.05,
        A/2)] so that the baseline-free tangent lag equals lambda_, with
        mu_max re-solved inside each evaluation so that max(dX/dt) equals mu.
        mu_max is capped at MU_MAX_CAP (2.0 /h); when the cap binds, or when
        the requested lag lies outside the range X0 can produce, the
        achievable shape is returned with realised=False and ``achieved``
        carrying what the curve really does.

        Args:
            A: Gompertz A (max OD) -- becomes X_max
            mu: Gompertz mu (maximum slope, OD/h)
            lambda_: Gompertz lambda (lag time, h) -- now actually used
            S0: Initial substrate concentration
            Ki: Substrate inhibition constant (Ki >> Ks: Monod-like;
                Ki ~ Ks: strongly inhibited)
            Ks: Half-saturation constant
            q: Substrate consumption per unit biomass growth
            X0: Initial biomass; solved for the requested lag when None
            t: Grid the shape is solved and measured on (DEFAULT_GRID if None)

        Returns:
            dict with mu_max, Ks, Ki, X_max, q, X0, S0, plus 'realised' and
            'achieved'
        """
        from scipy.optimize import brentq

        t = _as_grid(t)
        cap = HaldaneModel.MU_MAX_CAP
        X_max = float(A)

        last_mu_max = [None]

        def solve_for(x0: float) -> Tuple[float, dict, bool]:
            out = HaldaneModel._solve_mu_max(
                t, mu, Ks, Ki, X_max, q, x0, S0, cap, guess=last_mu_max[0])
            last_mu_max[0] = out[0]
            return out

        def lag_free(shape: dict) -> float:
            slope = shape['max_slope']
            if not np.isfinite(slope) or slope <= 0:
                return float('nan')
            return shape['tangent_lag'] - shape['y0'] / slope

        solver_ok = True

        if X0 is not None:
            x0 = float(X0)
            mu_max, shape, mu_ok = solve_for(x0)
            solver_ok = mu_ok
        else:
            lo, hi = 1e-6, min(0.05, 0.5 * X_max)
            if hi <= lo:
                lo, hi = 0.1 * X_max, 0.5 * X_max

            cache: dict = {}

            def lag_err(log_x0: float) -> float:
                x0 = float(10.0 ** log_x0)
                m, s, ok = solve_for(x0)
                cache[log_x0] = (x0, m, s, ok)
                return lag_free(s) - lambda_

            log_lo, log_hi = np.log10(lo), np.log10(hi)
            e_lo, e_hi = lag_err(log_lo), lag_err(log_hi)

            if not np.isfinite(e_lo) or not np.isfinite(e_hi):
                # Cannot even evaluate the endpoints: keep the larger inoculum.
                log_star, solver_ok = log_hi, False
            elif e_lo * e_hi > 0:
                # Requested lag is outside what any inoculum in range gives.
                log_star = log_lo if abs(e_lo) < abs(e_hi) else log_hi
                solver_ok = False
            else:
                try:
                    # xtol on log10(X0): d(lag)/d(log10 X0) is a few hours per
                    # decade, so 1e-3 of a decade is ~0.005 h of lag -- an
                    # order of magnitude inside LAG_TOL_FLOOR_H.
                    log_star = brentq(lag_err, log_lo, log_hi,
                                      xtol=1e-3, rtol=1e-10, maxiter=100)
                except (ValueError, RuntimeError):
                    log_star = log_hi
                    solver_ok = False

            if log_star not in cache:
                lag_err(log_star)
            x0, mu_max, shape, mu_ok = cache[log_star]
            solver_ok = solver_ok and mu_ok

        params = {
            'mu_max': float(mu_max),
            'Ks': float(Ks),
            'Ki': float(Ki),
            'X_max': X_max,
            'q': float(q),
            'X0': float(x0),
            'S0': float(S0),
        }
        # Re-measure through the public path so 'achieved' describes exactly
        # the curve a caller gets from HaldaneModel.compute(t, **params).
        final = measure_shape(HaldaneModel.compute(t, **params), t)
        params['realised'] = _realised(A, mu, lambda_, final, t, solver_ok)
        params['achieved'] = _achieved_from_shape(final)
        return params


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
        # Compute base growth curve (OD space; Baranyi is log-space natively)
        y_growth = compute_od(self.base_model, t, **base_params)

        # Get OD at death phase start
        y_at_death = compute_od(
            self.base_model, np.array([t_death], dtype=float), **base_params)[0]

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
        'gompertz': GompertzModel.from_gompertz_params,
        'baranyi': BaranyiModel.from_gompertz_params,
        'logistic': LogisticModel.from_gompertz_params,
        'richards': RichardsModel.from_gompertz_params,
        'haldane': HaldaneModel.from_gompertz_params
    }

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
        return compute_od(model, t, **params)


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

    baranyi_params = BaranyiModel.from_gompertz_params(A, mu, lambda_, t=t)
    # log-space parameters, OD out (compute_od == compute for this model)
    y_baranyi = BaranyiModel.compute_od(t, **baranyi_params)

    logistic_params = LogisticModel.from_gompertz_params(A, mu, lambda_, t=t)
    y_logistic = LogisticModel.compute(t, **logistic_params)

    richards_params = RichardsModel.from_gompertz_params(A, mu, lambda_, t=t)
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
        params = RichardsModel.from_gompertz_params(A, mu, lambda_, nu=nu, t=t)
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
