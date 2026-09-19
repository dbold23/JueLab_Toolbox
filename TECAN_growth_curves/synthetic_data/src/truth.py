"""
Ground-truth schema and the measurements behind it (WP3).

Why this module exists
----------------------
Until now "ground truth" was the *request* the sampler made, and the label was
a whole-trace statistic:

  * ``true_A/true_mu/true_lambda`` were the numbers handed to a converter, not
    numbers measured on the curve that was written.  ``baranyi_generated``
    shipped ``true_mu`` at 2.3x the real maximum slope of its own curve
    (measured slope 0.066-0.099 against a recorded 0.173-0.209 on
    CURVE0390-0393) and all 15 curves still scored as classification
    successes, because a converter bug is indistinguishable from a pipeline
    recovery failure once the truth table is fiction.
  * ``expected_class`` for the three "noisy BAD" scenarios came from
    ``target_r_squared``, realised as R^2 of the noisy array against the clean
    array over the whole run.  That is not a property of the curve: the same
    curve at the same sigma scores 0.977 over 20 h and 0.880 over 200 h, so a
    plate that runs longer relabels GOOD as BAD.  An oracle Gompertz fit over
    the curve's own growth window recovers A and mu to a median 1.4-2.9 % and
    5.3 % on those same 45 curves - i.e. they are recoverable, and the label
    said otherwise.

So this module owns two things and nothing else:

  1. :data:`GROUND_TRUTH_COLUMNS` - the ordered v2 schema (interface contract
     B), plus :data:`TRUTH_SCHEMA_VERSION` and :data:`LABEL_RULE_VERSION`.
     ``output_formatter.ground_truth_columns()`` imports it lazily and it wins
     over the v1 fallback as soon as this file exists.
  2. The three measurement functions the generator uses to fill it:
     :func:`measure_shape` (descriptive truth from the noiseless curve),
     :func:`oracle_recovery` (is the noisy curve still identifiable?) and
     :func:`apply_label_rule` (the scoped GOOD/BAD rule).

Conventions that the rest of the suite depends on
-------------------------------------------------
* **Truth describes the curve that was written.**  ``true_A``, ``true_mu`` and
  ``true_lambda`` are measured on the *noiseless* curve on its own grid, with
  the same estimator the pipeline-facing convention uses
  (``growth_models.measure_shape``): ``max(np.gradient(y, t))`` and the tangent
  intercept at that point.  The sampler's request survives as ``requested_*``.
  A measured column never carries a sentinel: ``true_A`` is the amplitude of
  the array that was handed over, so a random walk that rises 0.42 OD reports
  0.42 (it used to report 0.0), and where a quantity does not exist - the rate
  and lag of a trace with no growth phase - the answer is NaN, which every
  consumer already skips, rather than a number that reads as a measurement.
* **The measurement window is the primary phase.**  A curve with a second
  phase or a decline is measured over ``t <= boundary``, where the boundary is
  the onset of whatever happens next (``t_second_start`` / ``t_death``).
  Otherwise a diauxic curve whose second phase is steeper would ship the
  second phase's slope as the truth of the first.
* **expected_class stays binary.**  Difficulty lives in ``scoring_mode``;
  three consumers coerce "not GOOD" to BAD, so a third class would silently be
  scored and trained as "the pipeline should have rejected this".
* **The label rule is scoped.**  It is applied only to scenarios that opt in
  with ``label_rule='oracle'``.  A blanket rule would flip ``contamination``
  (dOD 0.30-0.52, parameters perfectly recoverable) to GOOD and destroy a real
  BAD case: that scenario is BAD because of *what the well contains*, which no
  fit statistic can see.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Version of the ground-truth schema written by the generator.
TRUTH_SCHEMA_VERSION = '2'

#: Version of the label rule in force (see :func:`apply_label_rule`).  Bumping
#: this is how a label change becomes traceable to a model version.
LABEL_RULE_VERSION = '1'

#: The ordered v2 ground-truth columns (interface contract B).  Order is part
#: of the contract: ``output_formatter.create_ground_truth_csv`` writes exactly
#: this list, in this order.
GROUND_TRUTH_COLUMNS: Tuple[str, ...] = (
    # identity and label
    'curve_id', 'strain_name', 'scenario', 'expected_class',
    'label_rule', 'label_rule_version', 'scoring_mode', 'lab_band',
    # what was generated
    'model_type', 'shape',
    'requested_A', 'requested_mu', 'requested_lambda',
    # what was actually written
    'true_A', 'true_mu', 'true_lambda',
    'mu_realisation_ratio', 'lambda_realisation_error', 'shape_solved',
    'initial_od',
    # noise provenance
    'noise_level', 'noise_sources',
    'target_r_squared', 'actual_r_squared',
    'artefact_free_r_squared', 'artefact_free_rmse', 'rmse',
    # trace summary
    'max_od', 'delta_od', 'n_points', 'time_resolution', 'duration_hours',
    'seed',
    # truncation / identifiability ground truth
    'true_window_end_h', 't_primary_95', 't_second_start', 't_death',
    'oracle_A_relerr', 'oracle_mu_relerr', 'oracle_fit_ok',
    'true_frac_of_A_reached',
    # replicates and artefacts
    'replicate_A_cv', 'replicate_mu_cv', 'outlier_well',
    'spike_indices', 'drift_od_per_h', 'twin_curve_id',
    'metrics_defined',
)

#: Declared *annotation* columns and the value used when a caller has no
#: concept of them.  Deliberately contains no measurement: a generator that
#: forgets to compute ``true_A`` or ``oracle_A_relerr`` must still get a
#: :class:`output_formatter.GroundTruthSchemaError`, which is why every
#: measured column is absent from this table.
GROUND_TRUTH_FILL_DEFAULTS: Dict[str, Any] = {
    'target_r_squared': float('nan'),   # a *requested* R^2; nan == none requested
    'label_rule': 'scenario',
    'label_rule_version': LABEL_RULE_VERSION,
    'scoring_mode': 'standard',
    'lab_band': '',
    'shape': '',
    'noise_sources': '',
    'replicate_A_cv': 0.0,
    'replicate_mu_cv': 0.0,
    'outlier_well': False,
    'spike_indices': '',
    'drift_od_per_h': 0.0,
    'twin_curve_id': '',
}

#: The scoring policies a scenario may declare.  ``expected_class`` stays
#: binary; this column is where "hard for a reason" is recorded.
SCORING_MODES = (
    'standard',
    'a_not_identifiable',     # the run ends before the asymptote is pinned
    'preprocessing_blind',    # the benchmark bypasses a pipeline stage
    'model_mismatch',         # generated by a non-Gompertz model
)

#: The label rules a scenario may declare.
LABEL_RULES = ('scenario', 'oracle')

# ---------------------------------------------------------------------------
# Label-rule constants
# ---------------------------------------------------------------------------

#: Minimum amplitude of the *truth curve* for an oracle-labelled GOOD.  Equal
#: to the pipeline's own min_absolute_delta_od (0.15 OD) - the rule asks "is
#: there enough signal to be worth calling growth", not "did the pipeline like
#: it".  Nothing here changes a pipeline threshold; this is a property of the
#: written curve.
LABEL_MIN_DELTA_OD = 0.15

#: Maximum relative error of the oracle fit for an oracle-labelled GOOD.
LABEL_MAX_ORACLE_RELERR = 0.20

#: ``true_window_end_h = WINDOW_END_FACTOR * t_primary_95`` for a curve that
#: simply plateaus (the same constant curve_shapes.py uses, by construction).
WINDOW_END_FACTOR = 1.3

#: Fraction of the asymptote that defines "the primary phase is complete".
PRIMARY_COMPLETION_FRACTION = 0.95

#: ``curve_fit`` iteration budget for the oracle fit.
ORACLE_MAXFEV = 20000

#: Fewest points an oracle fit is attempted on (4 free parameters).
ORACLE_MIN_POINTS = 8

#: Smallest truth amplitude for which the *relative* oracle errors are a
#: measurement rather than a division by the noise.  Equal to the noise floor
#: the pipeline itself substitutes when a well has no measurable variance
#: (0.002 OD, scripts/01_growth_curve_analysis.py:308-311); it is quoted here,
#: not changed, and nothing in scripts/ or tecan_growth/ reads this constant.
#:
#: This is the conditioning guard, and it is the reason the suite has a
#: reproducibility problem without it: ``evaporation_drift`` writes curves whose
#: whole rise is 0.0008 OD, the four-parameter Gompertz is then unidentifiable
#: on them, and dividing an unidentifiable fit by an amplitude 2.5x below the
#: noise floor published ``oracle_A_relerr = 3.5e5``.  A single-ulp change in
#: ONE of the 400 samples of that curve moved the published number by 0.3 %
#: (91.150 -> 90.881 on CURVE0690 at seed 42), which is how two regenerations at
#: the same seed produced a ground_truth.csv with a different sha256 while every
#: hard check (true_A/true_mu/true_lambda to 1e-9) stayed green.
ORACLE_MIN_TRUE_A = 0.002

#: Resolution the oracle reads the noisy curve at, in OD.  The fit is seeded at
#: the truth and run on ``round(noisy, 6)`` rather than on the raw float array.
#:
#: This is a reproducibility guard with a physical reading.  1e-6 OD is 460x
#: below the quietest noise this instrument has ever produced (robust HF sigma
#: 0.00046 OD in the 0.00-0.05 band over 92 real conditions, M06; p10 over all
#: 96 raw wells of Group 3 is 0.00073) and 2000x below the pipeline's own noise
#: floor, so no measurement changes - but a difference in the last bits of a
#: float, which belongs to the numpy/BLAS build and not to the curve, can no
#: longer reach the optimiser.  That matters because a least-squares fit is the
#: one step in this module that AMPLIFIES such a difference instead of passing
#: it through: one ulp in one of 400 samples of CURVE0690 moved its published
#: relative error by 0.3 %, and the same signature (340076.71 vs 346281.25 on an
#: evaporation_drift curve) is what made two regenerations at the same seed
#: write a ground_truth.csv with a different sha256 while all 19 DATA/*.csv were
#: byte-identical.
ORACLE_INPUT_QUANTUM_OD = 1e-6

#: Decimal places of :data:`ORACLE_INPUT_QUANTUM_OD`, for ``np.round``.
ORACLE_INPUT_DECIMALS = 6

#: How far outside the fitted window the fitted lag may land before the fit is
#: called non-converged, as a multiple of the window length.  A Gompertz fitted
#: to a 100 h window that reports ``lambda = 3098 h`` is not a recovery of that
#: curve, it is the optimiser walking off into a flat direction.
ORACLE_LAMBDA_SLACK = 1.0

#: Relative errors are reported saturated at this value.  Past it the fit is a
#: failure rather than a measurement, and saturating keeps the published number
#: from carrying digits that a different BLAS build would move; the label rule
#: (:data:`LABEL_MAX_ORACLE_RELERR` = 0.20) reads the same on either side of it.
ORACLE_RELERR_CAP = 10.0

#: Published precision of the two oracle relative errors, in decimal places.
#: They are the output of a nonlinear least-squares fit, so their last bits are
#: a property of the numpy/BLAS build rather than of the curve; 1e-9 is ~7
#: orders of magnitude below the label threshold they feed and ~7 above the
#: ulp-level jitter it absorbs.
ORACLE_RELERR_DECIMALS = 9

#: Smallest rise (OD) that counts as a growth *phase* rather than as numerical
#: dust.  Calibrated on the real plates, not invented: the robust high-frequency
#: noise sigma of a real well in the 0.00-0.05 OD band is 0.00046 OD (M06, 92
#: real conditions), so a curve whose entire rise is below 1e-4 OD - four times
#: smaller than the quietest noise ever measured on this instrument - never
#: leaves the baseline.  Such a curve has an amplitude (measured, and usually
#: ~0) but no rate and no lag, so ``true_mu``/``true_lambda`` are NaN rather
#: than the slope of a rounding error: ``haldane_total_inhibition`` writes a
#: curve that rises 1.2e-6 OD over 100 h and used to ship
#: ``true_lambda = 30.9 h`` for it.  This is a property of the truth curve and
#: is NOT a pipeline gate - no threshold in scripts/ or tecan_growth/ reads it.
MIN_RESOLVABLE_AMPLITUDE_OD = 1e-4


# ---------------------------------------------------------------------------
# Descriptive truth
# ---------------------------------------------------------------------------

def _as_arrays(y: Sequence[float], t: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.shape != t.shape:
        raise ValueError(f"y {y.shape} and t {t.shape} must have the same shape")
    if y.ndim != 1 or y.size < 3:
        raise ValueError("need a 1-D curve with at least 3 points")
    return y, t


def measurement_boundary(t: Sequence[float],
                         t_second_start: Optional[float] = None,
                         t_death: Optional[float] = None) -> float:
    """The end of the window ``true_*`` is measured over.

    The onset of whatever happens after the primary phase - a second growth
    phase or a decline - whichever comes first, and the end of the run when
    there is neither.  Measuring past that boundary would ship the second
    phase's slope (``diauxic_second_larger`` is explicitly steeper than its
    first phase) or the decline's minimum as the truth of the first phase.
    """
    t = np.asarray(t, dtype=float)
    edges = [float(b) for b in (t_second_start, t_death)
             if b is not None and np.isfinite(b)]
    end = float(t[-1])
    if edges:
        end = min(end, min(edges))
    return end


def measure_shape(clean_od: Sequence[float],
                  t: Sequence[float],
                  t_death: Optional[float] = None,
                  t_second_start: Optional[float] = None,
                  has_growth_process: bool = True) -> Dict[str, Any]:
    """
    Measure the truth of a noiseless curve: what was actually written.

    ``true_A``     ``max(clean) - clean[0]`` over the measurement window
    ``true_mu``    ``max(np.gradient(clean, t))`` over the same window
    ``true_lambda`` ``t[i] - (clean[i] - clean[0]) / true_mu`` at ``i`` = the
                   argmax of the gradient (the tangent-intercept lag, i.e. the
                   pipeline-facing convention of
                   ``growth_models.measure_shape``)

    The window is ``t <= measurement_boundary(t, t_second_start, t_death)``.

    **``true_A`` is always a measurement of the array it was handed.**  It is
    never a sentinel: a trace that rises 0.42 OD reports 0.42, whatever process
    produced it.  ``erratic`` used to ship ``true_A = 0.0`` for random walks
    whose own array rises up to 0.42 OD, which is exactly the contradiction
    between the truth table and the shipped curve this module exists to make
    impossible.

    ``true_mu`` and ``true_lambda``, by contrast, describe a *growth phase*, so
    they are NaN - never 0.0, never the slope of a rounding error - in the two
    cases where no growth phase exists:

    * ``has_growth_process=False``: nothing generated this trace (a random
      walk is its own "clean" array).  Its maximum finite difference is a
      property of the noise, not a growth rate, and the tangent intercept of
      that difference is not a lag.
    * the measured amplitude is below :data:`MIN_RESOLVABLE_AMPLITUDE_OD`, or
      the curve never rises at all.  ``haldane_total_inhibition`` writes a
      curve that rises 1.2e-6 OD and used to report a 30.9 h lag for it.

    Args:
        clean_od: the noiseless curve, INCLUDING its baseline offset.
        t: the time grid in hours.
        t_death: onset of a decline, if the curve has one.
        t_second_start: onset of a second growth phase, if the curve has one.
        has_growth_process: False when the array is not the realisation of a
            growth model (``pattern='random_walk'``).

    Returns:
        dict with ``true_A``, ``true_mu``, ``true_lambda``, the index and time
        of the maximum slope, the window end actually used, ``t_primary_95``
        (the first time the curve reaches 95 % of its own amplitude - a
        property of the curve, not of any truncation rule),
        ``has_growth_phase`` and ``has_growth_process``.  Nothing raises.
    """
    y, tt = _as_arrays(clean_od, t)
    end = measurement_boundary(tt, t_second_start, t_death)
    mask = tt <= end + 1e-12
    if mask.sum() < 3:                                     # pragma: no cover
        mask = np.ones_like(tt, dtype=bool)
    yw, tw = y[mask], tt[mask]

    y0 = float(yw[0])
    # The amplitude of the array as written.  Clipped at zero only for a curve
    # whose maximum IS its first point (a monotone decline has no amplitude to
    # report); it is never replaced by a placeholder.
    amplitude = float(np.nanmax(yw) - y0)
    true_A = max(amplitude, 0.0) if np.isfinite(amplitude) else float('nan')

    d = np.gradient(yw, tw)
    i = int(np.nanargmax(d))
    slope = float(d[i])

    has_growth_phase = bool(
        has_growth_process
        and np.isfinite(slope) and slope > 0
        and np.isfinite(true_A) and true_A >= MIN_RESOLVABLE_AMPLITUDE_OD
    )

    if not has_growth_phase:
        return {
            'true_A': true_A,
            'true_mu': float('nan'),
            'true_lambda': float('nan'),
            'i_max_slope': i,
            't_at_max_slope': float(tw[i]),
            'measured_to_h': float(end),
            't_primary_95': float('nan'),
            'y0': y0,
            'has_growth_phase': False,
            'has_growth_process': bool(has_growth_process),
        }

    true_lambda = float(tw[i] - (yw[i] - y0) / slope)
    return {
        'true_A': true_A,
        'true_mu': slope,
        'true_lambda': true_lambda,
        'i_max_slope': i,
        't_at_max_slope': float(tw[i]),
        'measured_to_h': float(end),
        't_primary_95': t_at_fraction(yw, tw, PRIMARY_COMPLETION_FRACTION),
        'y0': y0,
        'has_growth_phase': True,
        'has_growth_process': True,
    }


def t_at_fraction(clean_od: Sequence[float], t: Sequence[float],
                  fraction: float = PRIMARY_COMPLETION_FRACTION) -> float:
    """First time the curve reaches ``fraction`` of its own amplitude.

    Linear interpolation between the bracketing samples, so the answer does not
    jump by a whole sampling interval when the grid changes.  NaN for a curve
    with no amplitude.
    """
    y, tt = _as_arrays(clean_od, t)
    y0 = float(y[0])
    amp = float(np.nanmax(y) - y0)
    if not np.isfinite(amp) or amp <= 0:
        return float('nan')
    target = y0 + fraction * amp
    idx = np.nonzero(y >= target)[0]
    if idx.size == 0:                                      # pragma: no cover
        return float('nan')
    j = int(idx[0])
    if j == 0:
        return float(tt[0])
    y_lo, y_hi = float(y[j - 1]), float(y[j])
    if y_hi == y_lo:                                       # pragma: no cover
        return float(tt[j])
    frac = (target - y_lo) / (y_hi - y_lo)
    return float(tt[j - 1] + frac * (tt[j] - tt[j - 1]))


def true_window_end(t_primary_95: float,
                    boundary: Optional[float] = None,
                    t_end: Optional[float] = None,
                    factor: float = WINDOW_END_FACTOR) -> float:
    """
    End of the window over which the primary-phase parameters are identifiable.

    ``factor * t_primary_95`` for a curve that simply plateaus, truncated at the
    onset of whatever happens next and at the end of the run.  Identical to
    ``curve_shapes.true_window_end`` by construction - the two modules must
    agree, because a shape supplies this number and a plain model curve has it
    computed here.
    """
    if not np.isfinite(t_primary_95):
        return float(t_end) if t_end is not None and np.isfinite(t_end) else float('nan')
    end = float(factor) * float(t_primary_95)
    if boundary is not None and np.isfinite(boundary):
        end = min(end, float(boundary))
    if t_end is not None and np.isfinite(t_end):
        end = min(end, float(t_end))
    return float(end)


# ---------------------------------------------------------------------------
# Oracle recovery
# ---------------------------------------------------------------------------

def gompertz_with_offset(t: np.ndarray, A: float, mu: float, lam: float,
                         c: float) -> np.ndarray:
    """The pipeline's modified Gompertz plus a free baseline offset.

    Four parameters, because a real blanked well does not start at exactly 0
    and the oracle must not be penalised for a baseline the pipeline itself
    fits out.
    """
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return c + A * np.exp(-np.exp((mu * math.e / A) * (lam - t) + 1.0))


def oracle_fit_is_identifiable(popt: Sequence[float],
                               pcov: Any,
                               t_window: Sequence[float]) -> bool:
    """
    Did the oracle fit converge on a growth curve, or walk off a flat direction?

    ``curve_fit`` always returns a point; it does not always return a *fit*.
    When the four-parameter Gompertz is unidentifiable on the window - which is
    what a curve that never leaves its baseline gives it - MINPACK stops
    somewhere in a flat valley and reports an infinite covariance.  The point it
    stops at is chaotic in the input: on CURVE0690 (``evaporation_drift``, rise
    0.0008 OD) one ulp in one of 400 samples moved it from A=0.07262,
    lambda=3098 h to A=0.07241, lambda=3104 h.  Publishing a relative error
    built from that point is what let two regenerations at the same seed differ.

    So a fit counts only when all four of these hold, none of which is a
    tolerance on how *good* the recovery is - a wrong-but-identified fit is a
    real measurement and stays one:

      1. the covariance is finite, i.e. the parameters are locally identified;
      2. ``A > 0`` - it is a growth curve, not a mirror image of one;
      3. ``mu > 0`` - likewise;
      4. the fitted lag lands within one window length of the fitted window,
         rather than hundreds of hours past the end of the run.

    Args:
        popt: the fitted ``(A, mu, lambda, c)``.
        pcov: the covariance ``curve_fit`` returned alongside it.
        t_window: the times the fit actually saw.

    Returns:
        True when the fit is a usable measurement.
    """
    p = np.asarray(popt, dtype=float)
    tw = np.asarray(t_window, dtype=float)
    if p.size < 3 or tw.size == 0 or not np.all(np.isfinite(p)):
        return False

    cov = np.asarray(pcov, dtype=float)
    if cov.size == 0 or not np.all(np.isfinite(cov)):
        return False

    A_fit, mu_fit, lam_fit = float(p[0]), float(p[1]), float(p[2])
    if not (A_fit > 0.0 and mu_fit > 0.0):
        return False

    t0, t1 = float(np.min(tw)), float(np.max(tw))
    slack = ORACLE_LAMBDA_SLACK * max(t1 - t0, 1e-12)
    return (t0 - slack) <= lam_fit <= (t1 + slack)


def oracle_recovery(noisy: Sequence[float],
                    clean: Sequence[float],
                    t: Sequence[float],
                    true_params: Mapping[str, float],
                    window_end_h: Optional[float] = None) -> Dict[str, Any]:
    """
    Can the parameters still be recovered from THIS noisy curve?

    Fits :func:`gompertz_with_offset` to the *noisy* curve over the curve's own
    growth window (``t <= window_end_h``, by default
    ``1.3 * t_primary_95`` of the clean curve), seeded at the truth, and
    reports the relative error in A and mu.  This is the statistic the label
    rule uses, and it is a property of the curve rather than of the run length:
    the whole-trace R^2 it replaces moves from 0.977 to 0.880 on one unchanged
    curve simply by running the plate ten times longer.

    Args:
        noisy: the curve as emitted (post-noise, post-artefact, post-clip).
        clean: the noiseless curve, used only to locate the window.
        t: the time grid.
        true_params: ``{'A','mu','lambda_','initial_od'}`` - the measured truth
            of the clean curve, used as the seed and as the reference the
            errors are relative to.
        window_end_h: window end, computed from ``clean`` when None.

    Returns:
        ``{'true_window_end_h', 't_primary_95', 'oracle_A_relerr',
        'oracle_mu_relerr', 'oracle_fit_ok'}``.  All three measurements are NaN
        and ``oracle_fit_ok`` is False when the curve has no growth phase to
        recover - ``true_A <= 0`` or a non-finite ``true_mu``, which is what
        ``erratic`` (a random walk: an amplitude but no rate) and
        ``haldane_total_inhibition`` (no rise at all) report, and which would
        otherwise divide by zero - when the window holds too few points, or
        when the fit does not converge.  Nothing raises.

        Two further conditions make the numbers reproducible rather than
        merely finite.  ``true_A < ORACLE_MIN_TRUE_A`` (0.002 OD, the plate's
        own noise floor) is reported as non-converged, because a relative error
        divided by an amplitude below the noise is not a measurement; and a fit
        that :func:`oracle_fit_is_identifiable` rejects - infinite covariance,
        a negative A or mu, a lag hundreds of hours past the run - is reported
        as non-converged too.  What survives is saturated at
        :data:`ORACLE_RELERR_CAP` and rounded to
        :data:`ORACLE_RELERR_DECIMALS`.  Without this, one ulp in one sample of
        CURVE0690 moved a published ``oracle_A_relerr`` by 0.3 %, and two
        regenerations at the same seed wrote a differently-hashed
        ground_truth.csv with every hard check green.
    """
    y = np.asarray(noisy, dtype=float)
    c = np.asarray(clean, dtype=float)
    tt = np.asarray(t, dtype=float)

    t95 = t_at_fraction(c, tt) if c.size == tt.size else float('nan')
    end = (float(window_end_h) if window_end_h is not None and np.isfinite(window_end_h)
           else true_window_end(t95, t_end=float(tt[-1]) if tt.size else float('nan')))

    out: Dict[str, Any] = {
        'true_window_end_h': float(end) if np.isfinite(end) else float('nan'),
        't_primary_95': float(t95),
        'oracle_A_relerr': float('nan'),
        'oracle_mu_relerr': float('nan'),
        'oracle_fit_ok': False,
    }

    true_A = float(true_params.get('A', float('nan')))
    true_mu = float(true_params.get('mu', float('nan')))
    if not np.isfinite(true_A) or true_A <= 0 or not np.isfinite(end):
        return out
    if not np.isfinite(true_mu) or true_mu <= 0:
        return out
    if true_A < ORACLE_MIN_TRUE_A:
        # A relative error whose denominator is below the plate's own noise
        # floor is not a measurement of anything: see ORACLE_MIN_TRUE_A.
        return out

    mask = tt <= end + 1e-12
    if int(mask.sum()) < ORACLE_MIN_POINTS:
        return out

    p0 = [true_A, true_mu,
          float(true_params.get('lambda_', 0.0) or 0.0),
          float(true_params.get('initial_od', 0.0) or 0.0)]
    if not all(np.isfinite(p0)):                           # pragma: no cover
        return out

    try:
        from scipy.optimize import curve_fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, pcov = curve_fit(gompertz_with_offset, tt[mask],
                                   np.round(y[mask], ORACLE_INPUT_DECIMALS),
                                   p0=p0, maxfev=ORACLE_MAXFEV)
    except Exception:            # RuntimeError, ValueError, TypeError, LinAlgError
        return out

    if not oracle_fit_is_identifiable(popt, pcov, tt[mask]):
        return out

    A_fit, mu_fit = float(popt[0]), float(popt[1])
    out['oracle_A_relerr'] = _publish_relerr(abs(A_fit - true_A) / true_A)
    out['oracle_mu_relerr'] = _publish_relerr(abs(mu_fit - true_mu) / true_mu)
    out['oracle_fit_ok'] = True
    return out


def _publish_relerr(value: float) -> float:
    """Saturate and round one oracle relative error to its published form.

    Two steps, both about reproducibility rather than about the number:
    saturation at :data:`ORACLE_RELERR_CAP` so a fit that is off by orders of
    magnitude publishes one stable value instead of a chaotic one, and rounding
    to :data:`ORACLE_RELERR_DECIMALS` so the last bits of a least-squares
    solution - which belong to the numpy/BLAS build, not to the curve - cannot
    change the sha256 of ground_truth.csv.  Both sit far above the label rule's
    0.20, so no label can turn on either.
    """
    if not np.isfinite(value):                             # pragma: no cover
        return float('nan')
    return float(round(min(float(value), ORACLE_RELERR_CAP), ORACLE_RELERR_DECIMALS))


# ---------------------------------------------------------------------------
# Label rule
# ---------------------------------------------------------------------------

def _cfg_get(scenario_cfg: Any, key: str, default: Any) -> Any:
    """Read a field from a ScenarioConfig, a sampled params dict, or None.

    Contract A's fields land on ``ScenarioConfig`` in a later stage; until then
    the generator carries them in the sampled params dict.  Both spellings are
    accepted here so the rule is the same object in both worlds.
    """
    if scenario_cfg is None:
        return default
    if isinstance(scenario_cfg, Mapping):
        value = scenario_cfg.get(key, default)
    else:
        value = getattr(scenario_cfg, key, default)
    return default if value is None else value


def apply_label_rule(scenario_cfg: Any,
                     metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Decide ``expected_class`` for one curve, scoped to the scenario's rule.

    ``label_rule='oracle'``  GOOD iff the truth curve has at least
        :data:`LABEL_MIN_DELTA_OD` of amplitude AND the oracle fit succeeded
        AND both oracle relative errors are <= :data:`LABEL_MAX_ORACLE_RELERR`.
    ``label_rule='scenario'`` (the default) the scenario's own
        ``expected_class``, untouched.  Applying the oracle rule to every
        scenario would relabel ``contamination`` GOOD - its parameters are
        perfectly recoverable; it is BAD because of what is in the well.

    Args:
        scenario_cfg: ScenarioConfig, sampled params dict, or None.
        metrics: needs ``true_delta_od`` (the amplitude of the TRUTH curve, not
            a whole-trace statistic of the noisy one), ``oracle_A_relerr``,
            ``oracle_mu_relerr`` and ``oracle_fit_ok``.

    Returns:
        ``{'expected_class', 'label_rule', 'label_rule_version',
        'label_reason'}``.
    """
    rule = str(_cfg_get(scenario_cfg, 'label_rule', 'scenario'))
    if rule not in LABEL_RULES:
        raise ValueError(f"unknown label_rule {rule!r}; expected one of {LABEL_RULES}")

    scenario_class = str(_cfg_get(scenario_cfg, 'expected_class', 'BAD')).upper()

    if rule != 'oracle':
        return {
            'expected_class': scenario_class,
            'label_rule': rule,
            'label_rule_version': LABEL_RULE_VERSION,
            'label_reason': 'scenario',
        }

    delta = float(metrics.get('true_delta_od', float('nan')))
    a_err = float(metrics.get('oracle_A_relerr', float('nan')))
    mu_err = float(metrics.get('oracle_mu_relerr', float('nan')))
    fit_ok = bool(metrics.get('oracle_fit_ok', False))

    reasons = []
    if not (np.isfinite(delta) and delta >= LABEL_MIN_DELTA_OD):
        reasons.append('delta_od')
    if not fit_ok:
        reasons.append('oracle_fit')
    else:
        if not (np.isfinite(a_err) and a_err <= LABEL_MAX_ORACLE_RELERR):
            reasons.append('oracle_A')
        if not (np.isfinite(mu_err) and mu_err <= LABEL_MAX_ORACLE_RELERR):
            reasons.append('oracle_mu')

    return {
        'expected_class': 'GOOD' if not reasons else 'BAD',
        'label_rule': 'oracle',
        'label_rule_version': LABEL_RULE_VERSION,
        'label_reason': 'oracle_pass' if not reasons else '+'.join(reasons),
    }


def assert_scenario_labels_untouched(rows: Sequence[Mapping[str, Any]],
                                     scenario_class: Mapping[str, str]) -> None:
    """
    Generation-time guard: a ``label_rule='scenario'`` row keeps its label.

    Raises ``AssertionError`` naming every offending curve.  This is the check
    that makes "no scenario was quietly relabelled to raise the score"
    verifiable rather than asserted in prose.
    """
    bad = []
    for row in rows:
        if str(row.get('label_rule', 'scenario')) != 'scenario':
            continue
        expected = scenario_class.get(str(row.get('scenario')))
        if expected is None:
            continue
        if str(row.get('expected_class')) != str(expected):
            bad.append(f"{row.get('curve_id')} ({row.get('scenario')}): "
                       f"{row.get('expected_class')} != {expected}")
    if bad:
        raise AssertionError(
            "label_rule='scenario' rows were relabelled: " + '; '.join(bad))


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def empty_row() -> Dict[str, Any]:
    """A row with every declared column present and neutral.

    Measurements default to NaN (never 0.0 or -1.0, which read as real values);
    annotations take their declared fill default.
    """
    row: Dict[str, Any] = {}
    for column in GROUND_TRUTH_COLUMNS:
        if column in GROUND_TRUTH_FILL_DEFAULTS:
            row[column] = GROUND_TRUTH_FILL_DEFAULTS[column]
        else:
            row[column] = float('nan')
    row['label_rule_version'] = LABEL_RULE_VERSION
    row['metrics_defined'] = False
    row['oracle_fit_ok'] = False
    row['shape_solved'] = False
    return row


def strain_name(curve_id: int) -> str:
    """``CURVE0007`` - the join key pipeline_validator.py:177 and
    09_train_classifier.py:356 both parse.  Never change the format."""
    return f"CURVE{int(curve_id):04d}"


def format_spike_indices(indices: Sequence[int]) -> str:
    """Spike sample indices as a ';'-joined string (empty when there are none)."""
    return ';'.join(str(int(i)) for i in np.asarray(list(indices), dtype=int).ravel())


__all__ = [
    'GROUND_TRUTH_COLUMNS', 'GROUND_TRUTH_FILL_DEFAULTS',
    'TRUTH_SCHEMA_VERSION', 'LABEL_RULE_VERSION',
    'SCORING_MODES', 'LABEL_RULES',
    'LABEL_MIN_DELTA_OD', 'LABEL_MAX_ORACLE_RELERR',
    'WINDOW_END_FACTOR', 'PRIMARY_COMPLETION_FRACTION',
    'MIN_RESOLVABLE_AMPLITUDE_OD',
    'ORACLE_MIN_TRUE_A', 'ORACLE_LAMBDA_SLACK',
    'ORACLE_INPUT_QUANTUM_OD', 'ORACLE_INPUT_DECIMALS',
    'ORACLE_RELERR_CAP', 'ORACLE_RELERR_DECIMALS',
    'measure_shape', 'measurement_boundary', 't_at_fraction', 'true_window_end',
    'oracle_recovery', 'gompertz_with_offset', 'oracle_fit_is_identifiable',
    'apply_label_rule', 'assert_scenario_labels_untouched',
    'empty_row', 'strain_name', 'format_spike_indices',
]
