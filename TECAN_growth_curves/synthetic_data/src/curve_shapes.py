"""
Curve shape library (WP4) - the multi-phase, decline and recovery shapes the
synthetic suite has never had.

Why this module exists
----------------------
Until now every synthetic curve was a single monotone sigmoid (plus, for the
15 ``truncation_challenge`` curves, one hard-coded saturating exponential
bolted onto its plateau).  The 92 real conditions in ``data/raw/Group1..4``
are not like that:

  * 58.1 % of real growers (n=31 with dOD>0.5 and run>=90 h) are still
    gaining OD after t=50 h - median +0.074 OD, p90 +0.20, max +0.516
    (G4 PERMETHRINANDLB_CYPERM2, 77 % of its dOD after t=50 h; then
    G3 LB_LCY2 +0.322, G2 MALATHIONANDLB_MAL5 +0.299).  Synthetic: 17.3 %
    above 0.05 OD, 2.0 % above 0.20 OD, median +0.002.
  * 13/92 real conditions lose 15-50 % of their peak by the end of the run
    (G4 LB_PERM4 0.372, G2 LB_MAL4 0.263, G4 LB_PERM7 0.253, G4 LB_CYPERM2
    0.268, G2 LB_MAL8 0.224, G3 LAMBDACYHALOTHRINANDLB_LCY2 0.195), and the
    shape is a ~10 h sigmoid step, not an exponential decay: LB_PERM4 holds
    1.880 through ~53 h, its steepest 4 h window loses 0.507 of the 0.70 OD
    total starting at 57.5 h, and it settles at 1.180 by 70 h and stays flat
    to 96 h.
  * 6/44 real growers dip and recover at full magnitude (MALATHIONANDLB
    MAL1/MAL4/MAL5/MAL8/MAL10 and LB_PERM7).  G2 MALATHIONANDLB_MAL8: peak
    1.612 at 10 h, drawdown 0.502 OD by 27.5 h, recovery +0.297 OD by 100 h.
    Zero synthetic scenarios are designed to do this (the 6 synthetic curves
    that pass the same filter are noise artefacts in erratic/high_noise).
  * 28/92 real conditions have a zero-variance opening segment and 12/92 are
    identically zero somewhere, versus 0/480 synthetic; G4
    PERMETHRINANDLB_CYPERM2 is 26.4 % exact zeros, flat zero through ~30 h,
    then rises from ~40 h to 0.688 OD.

Every magnitude below is one of those measured numbers; each constant names
the condition it came from.  Nothing here is invented, and nothing here
touches a pipeline threshold: the truncation ground truth these shapes
publish (``t_primary_95``, ``t_second_start``, ``t_death``) is defined from
the written curve alone, in closed form, and never from PEAK_RATE_FRACTION or
any fraction-of-max-rate rule - otherwise a benchmark built on it would score
the pipeline against the pipeline.

Contract (WP4, Contract C)
--------------------------
    build_clean_curve(shape, t, params) -> (clean_od, shape_meta)

``shape_meta`` always carries

    t_primary_95      first time the FIRST phase alone reaches 0.95*A1
                      (closed form, see :func:`t95_gompertz`)
    t_second_start    onset of the second growth phase, NaN when there is none
    t_death           onset of the decline, NaN when there is none
    true_window_end_h end of the window over which the first-phase parameters
                      are identifiable (see :func:`true_window_end`)
    A1, mu1, lambda1  the first-phase parameters actually used
    shape             the registry name
    extras            a dict of shape-specific bookkeeping (realised loss,
                      clip offset, second-phase span, ...)

The returned curve is the *clean* (noiseless) OD **without** ``initial_od``:
the caller adds the baseline exactly as it does for the plain model branches.
The one shape that depends on its baseline is ``clipped_then_grows``, which
must be generated with ``initial_od = 0`` - it sets
``extras['requires_zero_baseline'] = True`` so the generator can assert it.

The correct-cut band for a truncation metric is
``[t_primary_95, t_second_start]`` (:func:`correct_cut_band`); for declining
shapes the upper edge is ``t_death``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import of the growth models.
#
# WP1 is rewriting growth_models.py in parallel; the modified-Gompertz
# signature GompertzModel.compute(t, A, mu, lambda_) is the stable part, so we
# resolve the class at call time.  Both import styles used in this repo are
# supported: `from src.curve_shapes import ...` (synthetic_data on sys.path,
# the scripts/tests convention) and a bare `import curve_shapes` (conftest.py
# puts synthetic_data/src on sys.path).
# ---------------------------------------------------------------------------


def _gompertz_model():
    """Return the GompertzModel class, imported lazily (see module note)."""
    try:                                                # package-relative
        from .growth_models import GompertzModel
    except ImportError:                                 # flat sys.path entry
        from growth_models import GompertzModel         # type: ignore
    return GompertzModel


def _gompertz(t: np.ndarray, A: float, mu: float, lambda_: float) -> np.ndarray:
    """GompertzModel.compute with underflow/overflow warnings silenced.

    A second phase whose lag sits 60 h into the run evaluates
    ``exp(exp(large))`` at t=0; the result underflows to exactly 0.0, which is
    the answer we want, so the warning is noise.
    """
    if A <= 0 or mu <= 0:
        return np.zeros_like(np.asarray(t, dtype=float))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        y = _gompertz_model().compute(np.asarray(t, dtype=float),
                                      float(A), float(mu), float(lambda_))
    return np.nan_to_num(np.asarray(y, dtype=float), nan=0.0,
                         posinf=float(A), neginf=0.0)


# ---------------------------------------------------------------------------
# Closed-form Gompertz geometry
#
# For the modified (Zwietering) Gompertz used everywhere in this repo,
#     y(t) = A * exp(-exp((mu*e/A)*(lambda - t) + 1))
# the time at which y = f*A is
#     t(f) = lambda + (A/(mu*e)) * (1 - ln(-ln f))
# so the whole geometry of a phase is fixed by (A, mu, lambda).
# ---------------------------------------------------------------------------

_F95_FACTOR = 1.0 - math.log(-math.log(0.95))    # 3.97024...
_F05_FACTOR = 1.0 - math.log(-math.log(0.05))    # -0.09755...
#: 5 % -> 95 % rise of a Gompertz phase, in units of A/(mu*e)
_SPAN_FACTOR = _F95_FACTOR - _F05_FACTOR         # 4.06779...


def t_at_fraction_of_A(fraction: float, A: float, mu: float,
                       lambda_: float) -> float:
    """Time at which a Gompertz phase reaches ``fraction`` of its amplitude."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if A <= 0 or mu <= 0:
        return float('nan')
    return float(lambda_ + (A / (mu * math.e)) * (1.0 - math.log(-math.log(fraction))))


def t95_gompertz(A: float, mu: float, lambda_: float) -> float:
    """First time the phase reaches 0.95*A - the primary-phase completion time.

    ``lambda + (A/(mu*e)) * (1 - ln(-ln 0.95))``, pipeline-independent by
    construction: it is a property of the curve that was written, not of any
    truncation rule.
    """
    return t_at_fraction_of_A(0.95, A, mu, lambda_)


def gompertz_mu_for_span(A: float, span_h: float) -> float:
    """The ``mu`` giving a phase whose 5 %->95 % rise takes ``span_h`` hours.

    Used to express a second phase as "a 40 h creep" rather than as a rate,
    because the real quantity that was measured is a duration (G3 LB_IMID3
    adds +0.45 OD over ~40 h after reaching 1.20 at 22 h).
    """
    if span_h <= 0:
        raise ValueError(f"span_h must be positive, got {span_h}")
    return float(_SPAN_FACTOR * A / (math.e * span_h))


def sigmoid_step(t: np.ndarray, t_mid: float, width_h: float) -> np.ndarray:
    """Logistic 0->1 step whose 5 %->95 % transition takes ``width_h`` hours.

    Calibrated against G4 LB_PERM4: with ``width_h = 10`` the steepest 4 h
    window carries 0.529 of the total step, against 0.507 measured on the real
    well - i.e. the real transition is ~10 h wide, which is what fixes the
    8-12 h band used by :data:`CRASH_WIDTH_H`.
    """
    if width_h <= 0:
        raise ValueError(f"width_h must be positive, got {width_h}")
    k = 2.0 * math.log(19.0) / width_h        # 5.8889 / width
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-k * (np.asarray(t, dtype=float) - t_mid)))


def sigmoid_step_onset(t_mid: float, width_h: float) -> float:
    """Time at which a :func:`sigmoid_step` has delivered 5 % of its change."""
    return float(t_mid - 0.5 * width_h)


# ---------------------------------------------------------------------------
# Calibration constants.  Every one cites the real condition it came from.
# ---------------------------------------------------------------------------

#: slow_secondary_drift second-phase amplitude, as a fraction of A1.
#: Real: post-t=50 h gain as a fraction of dOD has median 0.040 and p90 0.196
#: over the 31 real conditions with dOD>0.5 and run>=90 h.
DRIFT_A2_FRACTION = (0.04, 0.25)
#: 5 %->95 % duration of that second phase.  Real: G3 LB_IMID3 reaches 1.20 at
#: 22 h and adds +0.45 OD over the following ~40 h; G3 LB_LCY2 t50 15.5 h but
#: t95 86.5 h; G4 PERMETHRINANDLB_CYPERM2 puts 77 % of its dOD after t=50 h.
DRIFT_SPAN_H = (30.0, 80.0)
#: Gap between primary completion (t95) and the second phase's lag.  This is
#: the one number here that is NOT directly measurable on averaged real traces
#: (nothing marks where a creep "starts"), so it is the free parameter, chosen
#: so that the quantity that WAS measured is reproduced: with A1 in the real
#: GOOD amplitude band (real GOOD gompertz_a median 1.49, max 1.87; real max
#: OD 1.94) the realised post-t=50 h gain is >0.05 OD for 93 % and >0.20 OD for
#: 30 % of draws, against 58.1 % and 22.6 % over the 31 real conditions, with a
#: maximum of 0.44 OD against the real 0.516.  A scenario named
#: slow_secondary_drift is the drifting sub-population, so it sits at or above
#: the whole-plate rates; it never exceeds the real magnitude.  Real onsets do
#: span this range: G3 LB_IMID3 creeps straight off its 22 h plateau while
#: PERMETHRINANDLB_CISPERM1 has t50 = 78 h and LB_LCY2 t95 = 86.5 h.
DRIFT_GAP_H = (18.0, 55.0)

#: two_phase_gompertz defaults.  The values replaced here were hard-coded at
#: data_generator.py:337-345 (A2=0.5, mu2=0.15, gap_hours=10) and shared by all
#: 15 truncation_challenge curves; the old second phase was
#: A2*(1-exp(-mu2*(t-t0))), whose maximum slope is at its own onset.
DIAUXIC_A2 = (0.30, 0.70)
DIAUXIC_SPAN_H = (10.0, 25.0)
DIAUXIC_GAP_H = (5.0, 15.0)
#: diauxic_second_larger: the second phase dominates (WP5 samples A1 0.4-0.7).
DIAUXIC_LARGE_A2 = (0.80, 1.20)

#: partial_lysis: fraction of the plateau lost, and how gently.
#: Real: 13/92 conditions lose 15-50 % (G4 LB_PERM4 0.372, G4 LB_CYPERM2 0.268,
#: G4 LB_PERM7 0.253, G2 LB_MAL4 0.263, G2 LB_MAL8 0.224, G3 LCY2 0.195);
#: over the 48 real growers the loss distribution is median 0.005, p75 0.076,
#: p90 0.186, max 0.372, with 1/48 above 0.30.
LYSIS_LOSS_FRACTION = (0.20, 0.35)
LYSIS_ONSET_OFFSET_H = (5.0, 20.0)
LYSIS_WIDTH_H = (15.0, 30.0)

#: sudden_crash, measured from G4 LB_PERM4: mean 1.880 through ~53 h, steepest
#: 4 h window losing 0.507 of the 0.70 OD total from t=57.5 h, 1.180 by 70 h,
#: flat to 96 h.  Centre U(50,60) with width U(8,12) reproduces both the
#: 45-55 h hold and the ~10 h transition.
CRASH_LOSS_FRACTION = (0.25, 0.40)
CRASH_CENTRE_H = (50.0, 60.0)
CRASH_WIDTH_H = (8.0, 12.0)

#: dip_and_recover, measured from G2 MALATHIONANDLB_MAL8: peak 1.612 at 10 h,
#: drawdown 0.502 OD by 27.5 h (~17 h of decline), recovery +0.297 OD by 100 h
#: (~72 h of re-rise).  MAL1/MAL4/MAL5/MAL10 and G4 LB_PERM7 do the same at
#: 25-32 h.
DIP_PEAK_A = (1.40, 1.70)
DIP_DRAWDOWN_OD = (0.30, 0.50)
DIP_SPAN_H = (12.0, 18.0)
DIP_ONSET_OFFSET_H = (0.0, 3.0)
RECOVERY_OD = (0.20, 0.35)
RECOVERY_SPAN_H = (60.0, 80.0)
#: The recovery is sampled as a fraction of the drawdown, then clipped into
#: :data:`RECOVERY_OD`.  MAL8 recovers 0.297 of its 0.502 drawdown = 0.59, and
#: none of the six real dip-and-recover growers regains its pre-dip peak, so
#: tying the two keeps every draw a genuine excursion (a recovery larger than
#: the dip would make the curve monotone again and the shape meaningless).
RECOVERY_FRACTION_OF_DIP = (0.50, 0.75)

#: clipped_then_grows, measured from G4 PERMETHRINANDLB_CYPERM2: 26.4 % of its
#: points are exactly 0, it is flat zero through ~30 h and rises from ~40 h to
#: 0.688 OD by the end of the run.
CLIP_ZERO_UNTIL_H = (30.0, 40.0)
CLIP_LAG_AFTER_ZERO_H = (2.0, 8.0)
CLIP_FINAL_OD = (0.50, 0.80)
CLIP_SPAN_H = (40.0, 70.0)

#: death_by_target_loss.  Real ceiling for end-of-run loss over 48 growers is
#: 0.372 (G4 LB_PERM4) with only 1/48 above 0.30, so anything above 0.50 is
#: out of distribution and only the explicitly named *_extreme variant may ask
#: for it.  Today death_phase_severe realises a median loss of 0.979 and
#: death_phase_moderate 0.718-0.720.
DEATH_LOSS_FRACTION = (0.05, 0.20)
MAX_DEATH_LOSS_FRAC = 0.50
#: Decline onset is anchored to the curve's own t95, never to a fraction of
#: the run: today t_death is 39.9-69.8 h while t95 is 9.2-18.0 h (median
#: 12.2/12.8 h) and the median truncation time is 11.25 h, so the "severe
#: death phase" test never puts a declining point near the fit window.
DEATH_OFFSET_MODERATE_H = (2.0, 8.0)
DEATH_OFFSET_SEVERE_H = (0.5, 3.0)

#: Default first-phase parameters, used only when the caller supplies none.
DEFAULT_A = 1.20
DEFAULT_MU = 0.25
DEFAULT_LAMBDA = 2.0

#: Multiplier defining the identifiable window of a completed single phase.
#: Matches WP3's true_window_end_h = 1.3 * t95 for the plain shapes.
WINDOW_END_FACTOR = 1.3

SHAPE_META_KEYS = (
    't_primary_95', 't_second_start', 't_death', 'true_window_end_h',
    'A1', 'mu1', 'lambda1', 'shape', 'extras',
)


# ---------------------------------------------------------------------------
# Parameter plumbing
# ---------------------------------------------------------------------------


def _as_rng(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(rng)


def _resolve_rng(params: Dict[str, Any]) -> np.random.Generator:
    """Every shape draws its unspecified quantities from one explicit stream.

    ``params['rng']`` (a Generator) wins, then ``params['seed']`` / the
    generator's ``curve_seed``; with neither, a fresh unseeded Generator is
    used.  ``np.random.seed`` is never touched.
    """
    if params.get('rng') is not None:
        return _as_rng(params['rng'])
    for key in ('seed', 'curve_seed', 'shape_seed'):
        if params.get(key) is not None:
            return _as_rng(params[key])
    return np.random.default_rng()


def _draw(value: Any, default: Any, rng: np.random.Generator) -> float:
    """Resolve a scalar / (lo, hi) range / None into a float.

    Scenario parameters reach us already sampled (a float) from
    ScenarioSampler, but ranges are accepted too so that a caller - or a test
    drawing 200 curves - can hand the shape its own band.
    """
    v = default if value is None else value
    if v is None:
        return float('nan')
    if isinstance(v, (tuple, list, np.ndarray)):
        if len(v) != 2:
            raise ValueError(f"range must have exactly 2 entries, got {v!r}")
        lo, hi = float(v[0]), float(v[1])
        if hi < lo:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    return float(v)


def _first_present(params: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if params.get(k) is not None:
            return params[k]
    return None


def _phase1(params: Dict[str, Any], rng: np.random.Generator,
            a_default: Any = DEFAULT_A) -> Tuple[float, float, float]:
    """Pull the first-phase (A1, mu1, lambda1) out of a sampled params dict."""
    A1 = _draw(_first_present(params, 'A1', 'A'), a_default, rng)
    mu1 = _draw(_first_present(params, 'mu1', 'mu'), DEFAULT_MU, rng)
    lambda1 = _draw(_first_present(params, 'lambda1', 'lambda_', 'lam'),
                    DEFAULT_LAMBDA, rng)
    return A1, mu1, lambda1


def true_window_end(t_primary_95: float, boundary: float,
                    t_end: float) -> float:
    """End of the window over which the first-phase parameters are identifiable.

    ``1.3 * t_primary_95`` for a curve that simply plateaus (WP3's convention),
    truncated at the onset of whatever happens next (a second phase or a
    decline) and at the end of the run.
    """
    end = WINDOW_END_FACTOR * t_primary_95
    if boundary is not None and np.isfinite(boundary):
        end = min(end, float(boundary))
    if np.isfinite(t_end):
        end = min(end, float(t_end))
    return float(end)


def _meta(shape: str, t: np.ndarray, A1: float, mu1: float, lambda1: float,
          t_primary_95: float, t_second_start: float = float('nan'),
          t_death: float = float('nan'),
          extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    boundaries = [b for b in (t_second_start, t_death) if np.isfinite(b)]
    boundary = min(boundaries) if boundaries else float('nan')
    return {
        't_primary_95': float(t_primary_95),
        't_second_start': float(t_second_start),
        't_death': float(t_death),
        'true_window_end_h': true_window_end(t_primary_95, boundary,
                                             float(t[-1]) if len(t) else np.nan),
        'A1': float(A1),
        'mu1': float(mu1),
        'lambda1': float(lambda1),
        'shape': shape,
        'extras': dict(extras or {}),
    }


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def build_gompertz(t: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    """Plain single-phase modified Gompertz (registry passthrough).

    Present so that ``shape='gompertz'`` is a legal, meaningful value and every
    scenario can route through one code path.  No second phase, no decline:
    ``t_second_start`` and ``t_death`` are NaN and the identifiable window is
    the usual 1.3*t95.
    """
    rng = _resolve_rng(params)
    A1, mu1, lambda1 = _phase1(params, rng)
    od = _gompertz(t, A1, mu1, lambda1)
    return od, _meta('gompertz', t, A1, mu1, lambda1, t95_gompertz(A1, mu1, lambda1))


def _two_phase(t: np.ndarray, params: Dict[str, Any], shape_name: str,
               a2_default: Any, span_default: Any, gap_default: Any,
               a2_is_fraction: bool = False) -> Tuple[np.ndarray, Dict]:
    """Shared builder for the three two-phase shapes.

    The second phase is a *second Gompertz* whose lag is ``t_second_start``:

        od = Gompertz(t; A1, mu1, lambda1) + Gompertz(t; A2, mu2, t_second_start)

    replacing the saturating exponential ``A2*(1-exp(-mu2*(t-t0)))`` at
    data_generator.py:387-389, whose maximum slope is at its own onset - an
    abrupt kink rather than a lag/exponential/plateau second phase.  The new
    form has its maximum slope at ``t_second_start + A2/(mu2*e)``, strictly
    after the onset, which is what a diauxic shift actually looks like and
    what the first-peak truncation rule was calibrated on.

    ``t_second_start`` is the second phase's tangent-intercept lag, i.e. the
    pipeline-independent onset of phase 2; it defaults to
    ``t_primary_95 + gap_hours`` so the gap is measured from the honest
    completion time of phase 1 rather than from the Gompertz inflection.
    """
    rng = _resolve_rng(params)
    A1, mu1, lambda1 = _phase1(params, rng)
    t_primary_95 = t95_gompertz(A1, mu1, lambda1)

    A2 = _draw(params.get('A2'), a2_default, rng)
    if a2_is_fraction and params.get('A2') is None:
        A2 = A2 * A1
    A2 = max(A2, 0.0)

    gap_hours = _draw(params.get('gap_hours'), gap_default, rng)
    lambda2 = params.get('lambda2')
    if lambda2 is not None:
        t_second_start = _draw(lambda2, None, rng)
    else:
        t_second_start = t_primary_95 + gap_hours

    if params.get('mu2') is not None:
        mu2 = _draw(params.get('mu2'), None, rng)
        span_h = _SPAN_FACTOR * A2 / (math.e * mu2) if mu2 > 0 else float('nan')
    else:
        span_h = _draw(None, span_default, rng)
        mu2 = gompertz_mu_for_span(A2, span_h) if A2 > 0 else 0.0

    od1 = _gompertz(t, A1, mu1, lambda1)
    od2 = _gompertz(t, A2, mu2, t_second_start)
    od = od1 + od2

    extras = {
        'A2': float(A2),
        'mu2': float(mu2),
        'lambda2': float(t_second_start),
        'gap_hours': float(gap_hours),
        'second_phase_span_h': float(span_h),
        'second_phase_max_slope_time': float(t_second_start + A2 / (mu2 * math.e))
                                       if mu2 > 0 else float('nan'),
        'post_t50_gain_od': float(od[-1] - od[np.searchsorted(t, 50.0)])
                            if len(t) and t[-1] > 50.0 else float('nan'),
    }
    return od, _meta(shape_name, t, A1, mu1, lambda1, t_primary_95,
                     t_second_start=t_second_start, extras=extras)


def build_two_phase_gompertz(t: np.ndarray, params: Dict[str, Any]):
    """Classic diauxic shift: a full sigmoid, a pause, then a smaller sigmoid.

    Replaces the hard-coded second phase (A2=0.5, mu2=0.15, gap_hours=10,
    identical across all 15 truncation_challenge curves) with sampled
    parameters and a real lag/exponential/plateau second phase.
    """
    return _two_phase(t, params, 'two_phase_gompertz',
                      DIAUXIC_A2, DIAUXIC_SPAN_H, DIAUXIC_GAP_H)


def build_diauxic_second_larger(t: np.ndarray, params: Dict[str, Any]):
    """Diauxic shift whose SECOND phase is the larger one.

    The suite has only ever contained second phases smaller than the first
    (median post-primary gain 0.40 of dOD, always A2 < A1), so the first-peak
    rule was never asked to keep the smaller, earlier plateau.  WP5 samples
    A1 0.4-0.7 against A2 0.8-1.2.
    """
    return _two_phase(t, params, 'diauxic_second_larger',
                      DIAUXIC_LARGE_A2, DIAUXIC_SPAN_H, DIAUXIC_GAP_H)


def build_slow_secondary_drift(t: np.ndarray, params: Dict[str, Any]):
    """Plateau, then a slow 30-80 h creep - the dominant real grower shape.

    G3 LB_IMID3 reaches 1.20 at 22 h and then adds +0.45 OD over ~40 h;
    G3 LB_LCY2 has t50 15.5 h but t95 86.5 h.  Amplitude is sampled as a
    fraction of A1 in [0.04, 0.25], from the measured distribution of
    post-t=50 h gain as a fraction of dOD (median 0.040, p90 0.196 over the 31
    real conditions with dOD>0.5 and run>=90 h).

    This is the shape that drove the human PEAK_RATE_FRACTION calibration
    ("10 % left curves with a slow secondary drift uncut"), and the benchmark
    has never contained one.

    Note for the scenario that uses it: the second phase is a fraction of A1,
    so the real post-t=50 h gain rates are reproduced only when A1 is sampled
    in the real grower band (the measurement is conditioned on dOD>0.5).  At
    A1 ~ U(1.0, 1.9) the realised rates are 93 % / 30 % above 0.05 / 0.20 OD
    against the real 58.1 % / 22.6 %; at A1 ~ U(0.8, 1.5) they fall to
    87 % / 13 % and the >0.20 acceptance target is missed.
    """
    return _two_phase(t, params, 'slow_secondary_drift',
                      DRIFT_A2_FRACTION, DRIFT_SPAN_H, DRIFT_GAP_H,
                      a2_is_fraction=True)


def _decline_shape(t: np.ndarray, params: Dict[str, Any], shape_name: str,
                   loss_default: Any, anchor_default: Any, width_default: Any,
                   absolute_centre: bool) -> Tuple[np.ndarray, Dict]:
    """Shared builder for partial_lysis and sudden_crash.

    Growth to A1, then a *sigmoid step* down to a flat tail - the measured
    shape of real decline (G4 LB_PERM4: 1.880 through ~53 h, transition
    ~57-68 h, 1.180 thereafter and flat to 96 h), not the monotone
    exponential decay of data_generator.py:147-158.
    """
    rng = _resolve_rng(params)
    A1, mu1, lambda1 = _phase1(params, rng)
    t_primary_95 = t95_gompertz(A1, mu1, lambda1)

    loss_frac = _draw(_first_present(params, 'death_loss_frac', 'loss_frac'),
                      loss_default, rng)
    loss_frac = float(np.clip(loss_frac, 0.0, 0.95))
    width_h = _draw(params.get('decline_width_h'), width_default, rng)

    # sudden_crash is anchored in wall-clock time (its measured anchor is
    # "45-55 h into a ~96 h run"), the others to the curve's own t95 - but an
    # explicit t_death_offset_h from the scenario always wins, so a caller that
    # anchors every declining scenario to t95 gets what it asked for.
    explicit_offset = _first_present(params, 't_death_offset_h',
                                     'decline_offset_h')
    if absolute_centre and explicit_offset is None:
        t_mid = _draw(params.get('decline_centre_h'), anchor_default, rng)
        t_onset = sigmoid_step_onset(t_mid, width_h)
    else:
        offset = _draw(explicit_offset,
                       None if absolute_centre else anchor_default, rng)
        t_onset = t_primary_95 + offset
        t_mid = t_onset + 0.5 * width_h

    od1 = _gompertz(t, A1, mu1, lambda1)
    if not (np.isfinite(t_mid) and A1 > 0):
        # No growth phase means nothing to decline from (a flat scenario routed
        # through a declining shape).  Return the curve as written and say so,
        # rather than anchoring a step to a NaN.
        return od1, _meta(shape_name, t, A1, mu1, lambda1, t_primary_95,
                          extras={'degenerate_no_growth': True})
    step = sigmoid_step(t, t_mid, width_h)
    od = np.maximum(od1 - loss_frac * A1 * step, 0.0)

    peak = float(np.max(od)) if len(od) else float('nan')
    realised_loss = float((peak - od[-1]) / peak) if peak > 0 else float('nan')
    extras = {
        'requested_loss_frac': float(loss_frac),
        'realised_loss_frac': realised_loss,
        'decline_width_h': float(width_h),
        'decline_centre_h': float(t_mid),
        'tail_od': float(od[-1]) if len(od) else float('nan'),
    }
    return od, _meta(shape_name, t, A1, mu1, lambda1, t_primary_95,
                     t_death=t_onset, extras=extras)


def build_partial_lysis(t: np.ndarray, params: Dict[str, Any]):
    """Plateau, then a gentle 20-35 % loss to a flat tail.

    The real middle of the decline distribution: 13/92 conditions lose 15-50 %
    (G4 LB_CYPERM2 0.268, G4 LB_PERM7 0.253, G2 LB_MAL4 0.263, G2 LB_MAL8
    0.224, G3 LCY2 0.195), against a synthetic suite whose only declining
    scenarios crash to near zero (median loss 0.72 and 0.98).
    """
    return _decline_shape(t, params, 'partial_lysis', LYSIS_LOSS_FRACTION,
                          LYSIS_ONSET_OFFSET_H, LYSIS_WIDTH_H,
                          absolute_centre=False)


def build_sudden_crash(t: np.ndarray, params: Dict[str, Any]):
    """Hold ~45-55 h, then a ~10 h sigmoid step down of 25-40 %, then flat.

    Measured from G4 LB_PERM4: mean 1.880 through ~53 h, steepest 4 h window
    losing 0.507 of the 0.70 OD total from t=57.5 h, 1.180 by 70 h, flat to
    96 h - i.e. a ~10 h transition, not a 4 h cliff.  The onset returned as
    ``t_death`` is ``centre - width/2``, so a centre of U(50,60) with a width
    of U(8,12) reproduces the 45-55 h hold.
    """
    return _decline_shape(t, params, 'sudden_crash', CRASH_LOSS_FRACTION,
                          CRASH_CENTRE_H, CRASH_WIDTH_H, absolute_centre=True)


def build_dip_and_recover(t: np.ndarray, params: Dict[str, Any]):
    """Peak, a 0.3-0.5 OD drawdown over ~15 h, then a 0.2-0.35 OD re-rise.

    Measured from G2 MALATHIONANDLB_MAL8: peak 1.612 at 10 h, drawdown
    0.502 OD by 27.5 h, recovery +0.297 OD by 100 h.  Six real growers do this
    at full magnitude (MAL1/MAL4/MAL5/MAL8/MAL10 and G4 LB_PERM7); the
    generator has no mechanism for it at all - the death branch is a monotone
    decay and the diauxic branch has no dip - so the 6 synthetic curves that
    pass the same filter are noise artefacts in erratic/high_noise.

    ``t_death`` is the onset of the drawdown and ``t_second_start`` the lag of
    the recovery phase, so both edges of the excursion are scoreable.
    """
    rng = _resolve_rng(params)
    A1, mu1, lambda1 = _phase1(params, rng, a_default=DIP_PEAK_A)
    t_primary_95 = t95_gompertz(A1, mu1, lambda1)

    dip_od = _draw(params.get('dip_od'), DIP_DRAWDOWN_OD, rng)
    dip_span = _draw(params.get('dip_span_h'), DIP_SPAN_H, rng)
    onset_offset = _draw(_first_present(params, 't_death_offset_h',
                                        'dip_offset_h'), DIP_ONSET_OFFSET_H, rng)
    if params.get('recovery_od') is not None:
        rec_od = _draw(params.get('recovery_od'), None, rng)
    else:
        rec_od = float(np.clip(
            dip_od * _draw(None, RECOVERY_FRACTION_OF_DIP, rng),
            min(RECOVERY_OD), max(RECOVERY_OD)))
    rec_span = _draw(params.get('recovery_span_h'), RECOVERY_SPAN_H, rng)

    t_dip_start = t_primary_95 + onset_offset
    t_dip_mid = t_dip_start + 0.5 * dip_span
    t_recovery_start = t_dip_start + dip_span

    od1 = _gompertz(t, A1, mu1, lambda1)
    if not (np.isfinite(t_dip_start) and A1 > 0):
        return od1, _meta('dip_and_recover', t, A1, mu1, lambda1, t_primary_95,
                          extras={'degenerate_no_growth': True})
    dip = dip_od * sigmoid_step(t, t_dip_mid, dip_span)
    mu_rec = gompertz_mu_for_span(rec_od, rec_span) if rec_od > 0 else 0.0
    recovery = _gompertz(t, rec_od, mu_rec, t_recovery_start)
    od = np.maximum(od1 - dip + recovery, 0.0)

    # Measure the excursion the way the real traces were measured: the peak is
    # the pre-dip maximum, the trough the minimum between dip onset and the
    # recovery, and the recovery is what is regained by the last point.
    pre = t <= t_dip_start
    peak = float(np.max(od[pre])) if pre.any() else float(np.max(od))
    after = t >= t_dip_start
    trough = float(np.min(od[after])) if after.any() else float(np.min(od))
    extras = {
        'dip_od': float(dip_od),
        'dip_span_h': float(dip_span),
        'recovery_od': float(rec_od),
        'recovery_span_h': float(rec_span),
        'mu_recovery': float(mu_rec),
        'realised_drawdown_od': float(peak - trough),
        'realised_recovery_od': float(od[-1] - trough) if len(od) else float('nan'),
    }
    return od, _meta('dip_and_recover', t, A1, mu1, lambda1, t_primary_95,
                     t_second_start=t_recovery_start, t_death=t_dip_start,
                     extras=extras)


def build_clipped_then_grows(t: np.ndarray, params: Dict[str, Any]):
    """Identically zero for ~30-40 h, then growth to 0.5-0.8 OD.

    Reproduces G4 PERMETHRINANDLB_CYPERM2 (26.4 % of points exactly 0, flat
    zero through ~30 h, rising from ~40 h to 0.688 OD).  The mechanism is
    blank over-subtraction: the well's signal minus a blank slightly larger
    than its baseline goes negative and is clipped at zero, which is why
    28/92 real conditions have a zero-variance opening segment and 12/92 are
    identically zero - against 0/480 synthetic, so the noise_floor=0.002
    substitution and the baseline_clipped path are code the benchmark has
    never entered.

    Implemented exactly that way: a Gompertz minus the constant offset that
    makes it cross zero at ``t_zero_until``, clipped at 0.  The curve must be
    generated with ``initial_od = 0`` or the zeros are destroyed; ``extras``
    flags that.
    """
    rng = _resolve_rng(params)
    t_zero_until = _draw(params.get('zero_until_h'), CLIP_ZERO_UNTIL_H, rng)
    lag_after = _draw(params.get('clip_lag_after_zero_h'),
                      CLIP_LAG_AFTER_ZERO_H, rng)
    final_od = _draw(_first_present(params, 'A1', 'A'), CLIP_FINAL_OD, rng)
    span_h = _draw(params.get('growth_span_h'), CLIP_SPAN_H, rng)

    lambda1 = t_zero_until + lag_after
    # The Gompertz shape is amplitude-invariant once the span is fixed
    # (mu enters only as mu/A), so build the unit curve once and scale it so
    # that the VISIBLE curve ends the run at exactly final_od - the quantity
    # the real trace reports (CYPERM2 ends at 0.688 while still rising).
    mu_unit = gompertz_mu_for_span(1.0, span_h)
    unit = _gompertz(t, 1.0, mu_unit, lambda1)
    unit0 = float(_gompertz(np.array([t_zero_until]), 1.0, mu_unit, lambda1)[0])
    visible_unit_end = float(unit[-1] - unit0)
    A_total = final_od / visible_unit_end if visible_unit_end > 1e-9 else final_od
    mu1 = gompertz_mu_for_span(A_total, span_h)
    offset = A_total * unit0

    od = np.maximum(A_total * unit - offset, 0.0)

    A1_visible = float(np.max(od)) if len(od) else float('nan')
    t_primary_95 = t95_gompertz(A_total, mu1, lambda1)
    extras = {
        'requires_zero_baseline': True,
        'zero_until_h': float(t_zero_until),
        'clip_offset_od': float(offset),
        'A_total': float(A_total),
        'growth_span_h': float(span_h),
        'n_zero_points': int(np.sum(od == 0.0)),
        'frac_zero_points': float(np.mean(od == 0.0)) if len(od) else float('nan'),
        'final_od': A1_visible,
        # A curve still rising at its last point does not hide it: this is the
        # 'A not identifiable yet' case (25 % of real growers, 1.7 % synthetic).
        'frac_of_A_reached': float(A1_visible / (A_total - offset))
                             if A_total > offset else float('nan'),
        'complete_within_run': bool(t_primary_95 <= float(t[-1])),
    }
    return od, _meta('clipped_then_grows', t, A1_visible, mu1, lambda1,
                     t_primary_95, extras=extras)


def build_death_by_target_loss(t: np.ndarray, params: Dict[str, Any]):
    """Growth, then an exponential decline parameterised by the loss it ends on.

    Two fixes to the way decline has been generated:

    * the *rate* is solved from the loss the run should end with,
      ``k_death = -ln(1 - loss) / (T_end - t_death)``, so a scenario asking for
      a 5-20 % loss gets exactly that.  Real ceiling over 48 growers is 0.372
      (G4 LB_PERM4) with 1/48 above 0.30; today death_phase_severe realises a
      median loss of 0.979 and death_phase_moderate 0.718, which nothing in
      the 92 real conditions does.  Any request above
      :data:`MAX_DEATH_LOSS_FRAC` (0.50) is clamped and warned about unless
      the caller explicitly names itself extreme (``allow_extreme_loss=True``
      or a scenario name containing "extreme"), in which case it is written as
      asked and flagged out-of-distribution in ``extras``.
    * the *onset* is anchored to the curve's own t95 - ``t95 + U(2,8) h``
      (moderate) or ``t95 + U(0.5,3) h`` (severe) - not to a fraction of the
      run.  Today t_death is 39.9-69.8 h while t95 is 9.2-18.0 h and the
      median truncation time is 11.25 h, so a 25-35 h flat plateau separates
      the decline from the fit window and A is recovered to 2.0-2.6 %.
    """
    rng = _resolve_rng(params)
    A1, mu1, lambda1 = _phase1(params, rng)
    t_primary_95 = t95_gompertz(A1, mu1, lambda1)
    t_end = float(t[-1]) if len(t) else float('nan')

    offset = _draw(params.get('t_death_offset_h'), DEATH_OFFSET_MODERATE_H, rng)
    t_death = t_primary_95 + offset

    requested_loss = _draw(params.get('death_loss_frac'),
                           DEATH_LOSS_FRACTION, rng)
    scenario = str(params.get('scenario_name') or params.get('scenario') or '')
    allow_extreme = bool(params.get('allow_extreme_loss', False)) or \
        'extreme' in scenario.lower()
    loss = float(np.clip(requested_loss, 0.0, 0.999))
    if loss > MAX_DEATH_LOSS_FRAC and not allow_extreme:
        warnings.warn(
            f"death_by_target_loss: requested end-of-run loss {loss:.3f} "
            f"exceeds the real-plate ceiling {MAX_DEATH_LOSS_FRAC:.2f} "
            f"(real max 0.372, G4 LB_PERM4); clamping. Name the scenario "
            f"*_extreme or pass allow_extreme_loss=True to ask for it "
            f"deliberately and have it reported as out-of-distribution.",
            RuntimeWarning, stacklevel=2)
        loss = MAX_DEATH_LOSS_FRAC

    od = _gompertz(t, A1, mu1, lambda1)
    if not (np.isfinite(t_death) and A1 > 0):
        return od, _meta('death_by_target_loss', t, A1, mu1, lambda1,
                         t_primary_95, extras={'degenerate_no_growth': True})
    if np.isfinite(t_end) and t_end > t_death:
        k_death = -math.log(max(1.0 - loss, 1e-12)) / (t_end - t_death)
        idx = np.searchsorted(np.asarray(t, dtype=float), t_death)
        if idx < len(t):
            od_at_death = float(np.interp(t_death, t, od))
            mask = np.asarray(t, dtype=float) > t_death
            od = od.copy()
            od[mask] = od_at_death * np.exp(-k_death * (t[mask] - t_death))
    else:
        k_death = float('nan')

    peak = float(np.max(od)) if len(od) else float('nan')
    realised_loss = float((peak - od[-1]) / peak) if peak > 0 else float('nan')
    extras = {
        'requested_loss_frac': float(requested_loss),
        'applied_loss_frac': float(loss),
        'realised_loss_frac': realised_loss,
        'k_death': float(k_death),
        't_death_offset_h': float(offset),
        'out_of_distribution': bool(loss > MAX_DEATH_LOSS_FRAC),
        'allow_extreme_loss': allow_extreme,
    }
    return od, _meta('death_by_target_loss', t, A1, mu1, lambda1,
                     t_primary_95, t_death=t_death, extras=extras)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SHAPES: Dict[str, Callable[[np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict]]] = {
    'gompertz': build_gompertz,
    'two_phase_gompertz': build_two_phase_gompertz,
    'diauxic_second_larger': build_diauxic_second_larger,
    'slow_secondary_drift': build_slow_secondary_drift,
    'partial_lysis': build_partial_lysis,
    'sudden_crash': build_sudden_crash,
    'dip_and_recover': build_dip_and_recover,
    'clipped_then_grows': build_clipped_then_grows,
    'death_by_target_loss': build_death_by_target_loss,
}

#: Shapes whose second phase / decline gives them a finite boundary after
#: t_primary_95.  gompertz and clipped_then_grows are single-phase by
#: construction and report NaN for both boundaries.
SHAPES_WITH_BOUNDARY = tuple(
    name for name in SHAPES if name not in ('gompertz', 'clipped_then_grows'))


def build_clean_curve(shape: str, t: np.ndarray,
                      params: Optional[Dict[str, Any]] = None
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the clean (noiseless) curve for ``shape`` on grid ``t``.

    Args:
        shape: a key of :data:`SHAPES`.
        t: time grid in hours (ascending).
        params: the sampled scenario parameters.  Recognised keys are the
            Contract A fields (A/mu/lambda_, A2, mu2, lambda2, gap_hours,
            death_loss_frac, t_death_offset_h, ...) plus ``rng``/``seed`` for
            the shape's own draws.  Ranges ``(lo, hi)`` are accepted anywhere a
            scalar is, and are sampled from the resolved stream.

    Returns:
        ``(clean_od, shape_meta)``.  ``clean_od`` excludes ``initial_od``; the
        caller adds the baseline exactly as it does for the model branches.
    """
    if shape not in SHAPES:
        raise ValueError(
            f"Unknown curve shape {shape!r}; known shapes: "
            f"{', '.join(sorted(SHAPES))}")
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a non-empty 1-D array of hours")
    od, meta = SHAPES[shape](t, dict(params or {}))
    od = np.asarray(od, dtype=float)
    if od.shape != t.shape:
        raise AssertionError(
            f"shape {shape!r} returned {od.shape} values for {t.shape} times")
    return od, meta


def correct_cut_band(meta: Dict[str, Any]) -> Tuple[float, float]:
    """The band in which a truncation must land for this curve, in hours.

    ``[t_primary_95, t_second_start]`` for a multi-phase curve, with the upper
    edge falling back to ``t_death`` for a declining one and to
    ``true_window_end_h`` for a curve that simply plateaus.  Defined entirely
    from the written curve: no PEAK_RATE_FRACTION, no fraction-of-max-rate
    rule, nothing the pipeline could be graded against itself with.
    """
    lo = float(meta['t_primary_95'])
    edges = [float(meta[k]) for k in ('t_second_start', 't_death')
             if np.isfinite(meta.get(k, float('nan')))]
    hi = min(edges) if edges else float(meta['true_window_end_h'])
    return lo, hi


__all__ = [
    'SHAPES', 'SHAPES_WITH_BOUNDARY', 'SHAPE_META_KEYS',
    'build_clean_curve', 'correct_cut_band',
    't_at_fraction_of_A', 't95_gompertz', 'gompertz_mu_for_span',
    'sigmoid_step', 'sigmoid_step_onset', 'true_window_end',
    'build_gompertz', 'build_two_phase_gompertz', 'build_diauxic_second_larger',
    'build_slow_secondary_drift', 'build_partial_lysis', 'build_sudden_crash',
    'build_dip_and_recover', 'build_clipped_then_grows',
    'build_death_by_target_loss',
    'DRIFT_A2_FRACTION', 'DRIFT_SPAN_H', 'DRIFT_GAP_H',
    'DIAUXIC_A2', 'DIAUXIC_LARGE_A2', 'DIAUXIC_SPAN_H', 'DIAUXIC_GAP_H',
    'LYSIS_LOSS_FRACTION', 'LYSIS_ONSET_OFFSET_H', 'LYSIS_WIDTH_H',
    'CRASH_LOSS_FRACTION', 'CRASH_CENTRE_H', 'CRASH_WIDTH_H',
    'DIP_PEAK_A', 'DIP_DRAWDOWN_OD', 'DIP_SPAN_H', 'DIP_ONSET_OFFSET_H',
    'RECOVERY_OD', 'RECOVERY_SPAN_H',
    'CLIP_ZERO_UNTIL_H', 'CLIP_FINAL_OD', 'CLIP_SPAN_H',
    'DEATH_LOSS_FRACTION', 'MAX_DEATH_LOSS_FRAC',
    'DEATH_OFFSET_MODERATE_H', 'DEATH_OFFSET_SEVERE_H',
    'WINDOW_END_FACTOR',
]
