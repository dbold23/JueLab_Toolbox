"""
Synthetic Growth Curve Data Generator (WP3: generation core and truth schema)

What changed and why
--------------------
The generator used to write a truth table describing what it *asked* for and a
label derived from a whole-trace statistic. Both are fixed here:

* **Descriptive truth.** Every curve now measures ``true_A``/``true_mu``/
  ``true_lambda`` on its own noiseless curve (:mod:`truth`), over the primary
  phase's own window, and keeps the sampler's request as ``requested_*`` with
  ``mu_realisation_ratio`` and ``lambda_realisation_error`` alongside. A
  converter that does not realise the shape it was asked for can no longer
  masquerade as a pipeline recovery failure - it shows up as
  ``shape_solved=False`` and a ratio away from 1.
* **Oracle columns and a scoped label rule.** Where the clean and noisy curves
  are both in scope the generator fits a 4-parameter Gompertz to the NOISY
  curve over ``t <= true_window_end_h``, seeded at the truth, and records the
  relative error in A and mu. Scenarios that opt in with ``label_rule='oracle'``
  are labelled from that; everything else keeps its scenario label, and a
  generation-time assertion proves no ``label_rule='scenario'`` row was
  touched.
* **Content-addressed seeding.** Each curve's seed is
  ``blake2b(f"{seed}:{scenario}:{i}")``, with separate streams for the sampler,
  the noise, the blank, the spikes and the drift. Adding a scenario or changing
  ``--n-per-scenario`` no longer rewrites every other curve, and ``--seed 0`` is
  seeded (``if seed`` was false for 0, so seed 0 silently produced unseeded
  data: two runs did not match).
* **Real failure modes.** A blank stage that is ON by default - every real
  column is blank-subtracted, so subtracting a :class:`BlankModel` mean and
  clipping at zero is not an opt-in decoration; it is what finally enters the
  clipped-baseline path the pipeline has for 30.4 % of real conditions (a
  scenario keeps its raw baseline with ``blank_offset_od=BLANK_OFF``, and a
  declared baseline above ``BLANK_DEFAULT_MAX_BASELINE_OD`` keeps it without
  asking) - replicates with 76 % common-mode noise
  and biological spread, one-sided decaying spikes and signed drift, and
  additive preset+target-RMSE noise instead of two exclusive branches.
* **No sentinel metrics.** ``actual_r_squared`` was hard-coded to 0.0 for flat
  curves and -1.0 for random walks, and ``duration_hours`` read a key that
  never existed (so a 100 h run reported 99.75). Metrics are computed from the
  same noisy-vs-clean formula everywhere a clean reference exists and are NaN
  with ``metrics_defined=False`` where none does.

Interface note
--------------
Contract A's new ``ScenarioConfig`` fields land in a later stage. Every one of
them is read here out of the sampled parameters dict with ``.get()`` and the
contract default, so this module works both before and after that lands and no
existing scenario changes behaviour.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                                    # package-relative
    from .growth_models import (
        GompertzModel, BaranyiModel, LogisticModel, RichardsModel, HaldaneModel,
        DeathPhaseExtension, get_model, convert_parameters, generate_curve,
        shape_of,
    )
    from .noise_models import (
        GaussianNoise, ODDependentNoise, InstrumentNoise, RMSEBasedNoise,
        ColoredNoise, BlankModel, get_noise_model, get_rmse_noise,
        inject_spikes, apply_signed_drift, add_target_rmse_component,
        sample_drift_rate, NOISE_PRESETS,
    )
    from .curve_shapes import SHAPES, build_clean_curve
    from .curve_scenarios import (
        ScenarioConfig, ScenarioSampler, ALL_SCENARIOS,
        GOOD_GROWTH_SCENARIOS, BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS,
        get_scenario, get_comprehensive_test_config,
    )
    from . import truth as truth_module
except ImportError:                                     # flat sys.path entry
    from growth_models import (                         # type: ignore
        GompertzModel, BaranyiModel, LogisticModel, RichardsModel, HaldaneModel,
        DeathPhaseExtension, get_model, convert_parameters, generate_curve,
        shape_of,
    )
    from noise_models import (                          # type: ignore
        GaussianNoise, ODDependentNoise, InstrumentNoise, RMSEBasedNoise,
        ColoredNoise, BlankModel, get_noise_model, get_rmse_noise,
        inject_spikes, apply_signed_drift, add_target_rmse_component,
        sample_drift_rate, NOISE_PRESETS,
    )
    from curve_shapes import SHAPES, build_clean_curve  # type: ignore
    from curve_scenarios import (                       # type: ignore
        ScenarioConfig, ScenarioSampler, ALL_SCENARIOS,
        GOOD_GROWTH_SCENARIOS, BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS,
        get_scenario, get_comprehensive_test_config,
    )
    import truth as truth_module                        # type: ignore


# =============================================================================
# Calibration constants.  Every magnitude cites the real condition it came from.
# =============================================================================

#: Fraction of a well's high-frequency sigma that is COMMON to the three wells
#: of a condition.  Real: per-well sigma vs the sigma of their mean gives
#: ratios 0.95/0.85/0.84/0.90/0.94/0.75 over Group 3 IMID2/3/5/8, LCY01, LCY2
#: (median 0.848 over 24 Group 3 conditions) against 1/sqrt(3)=0.577 for
#: independent noise.  With sigma_shared = 0.76*sigma and sigma_indep =
#: sqrt(1-0.76^2)*sigma the mean of three wells has
#: sqrt(0.76^2 + (1-0.76^2)/3) = 0.848 of a single well's sigma - i.e. this one
#: number reproduces the measured averaging ratio exactly.
REPLICATE_SHARED_SIGMA_FRACTION = 0.76

#: Wells per condition on the real plates (GROUP3_KEY.csv gives 3 wells for 30
#: of 31 conditions).
DEFAULT_N_REPLICATES = 3

#: Biological spread between the replicates of one condition.  Real plateau CV
#: across the 3 wells of a condition is 1.9-21.5 %.
DEFAULT_REPLICATE_A_CV = 0.05
DEFAULT_REPLICATE_MU_CV = 0.10
DEFAULT_REPLICATE_LAMBDA_SD_H = 0.5

#: A divergent well: G3 LAMBDACYHALOTHRIN_LCY01's third well plateaus at 1.04
#: against 1.73/1.72, i.e. 0.60 of its siblings.
OUTLIER_WELL_A_SCALE = (0.50, 0.70)

#: Default spike height when a scenario asks for spikes without naming one.
#: Real bubble artefacts are 0.15-0.8 OD (largest single positive excursion vs
#: an 11-point rolling median: typical well +0.0058, p90 +0.0211, max +0.295 in
#: Group 3 H12; the visible bubbles the lab discards are 0.2-1.0 OD).
DEFAULT_SPIKE_OD = (0.15, 0.40)
DEFAULT_SPIKE_DECAY_POINTS = 3

# -- the blank stage -----------------------------------------------------------
#
# Every column the pipeline reads on a real plate is blank-subtracted, so the
# blank stage is ON by default here: a scenario that says nothing about it gets
# the calibrated draw below, and only a scenario that names its own band (or
# opts out with ``BLANK_OFF``) overrides that.  Leaving it opt-in is what kept
# the suite at 13.2 % clipped against the real 30.4 % - i.e. it kept the
# ``noise_floor=0.002`` substitution at scripts/01_growth_curve_analysis.py:308
# a path the benchmark barely entered (M09: real 28/92 conditions have a
# zero-variance opening segment, 12/92 are identically zero, synthetic 0/480).

#: Blank wells averaged into the subtracted blank.  A lab subtracts the mean of
#: the plate's blank wells - the real plates carry 18 per media (M09) - so the
#: reference is quieter than any single well and the subtraction does not hand
#: every trace sqrt(2) x a blank's noise and a second independent drift.
BLANK_REFERENCE_WELLS = 3

#: Sentinel meaning "draw the offset from the calibrated mixture below".  It is
#: the default value of ``blank_offset_od``; a scenario may instead name a
#: ``(lo, hi)`` band in OD, or :data:`BLANK_OFF` to keep its raw baseline.
BLANK_CALIBRATED = 'calibrated'

#: Explicit opt-out: no blank is subtracted and the curve keeps the baseline
#: the scenario asked for.
BLANK_OFF = 'none'

#: Offset of the SUBTRACTED blank above the well's own true baseline, in OD.
#: A positive offset over-subtracts and the trace clips at zero; a negative one
#: leaves the well sitting slightly high, which is the other half of the real
#: spread.  Every draw of both branches stays inside the measured per-media
#: across-well b0 spread of the 18 Group 3 blank wells (0.003-0.029 OD).  The
#: real plates are reproduced by a MIXTURE, not by a symmetric band (M09:
#: "choosing the blank mean above the true well baseline in ~30 % of curves
#: reproduces the observed clipping"): a symmetric +-0.015 uniform puts half
#: its positive draws under the blank's own wander, where they pin nothing, and
#: clips 24.7 % of the curves it is applied to instead of 30.4 %.  The
#: over-subtraction branch is the upper part of the measured spread for the
#: same reason - an over-subtraction smaller than the blank noise it rides on
#: cannot hold a whole opening window at zero.  Measured on the curves this
#: default is applied to: 36.5 % clipped, against the real 30.4 %.
BLANK_OVERSUBTRACTION_OD = (0.012, 0.029)
BLANK_UNDERSUBTRACTION_OD = (-0.015, 0.005)

#: Weight of the over-subtraction branch.  This is the one fitted number of the
#: stage - the magnitudes above are measured - and it is fitted to the single
#: number the stage exists to reproduce: 28/92 = 30.4 % of real conditions have
#: a zero-variance opening segment.  Measured on the shipped suite at --seed 42
#: (1139 curves, read off the written plate files by
#: tests/synthetic/test_truth.py::test_clipping_prevalence, not off a probe):
#:     over=(0.005,0.029): 0.70 -> 25.3 %   0.85 -> 26.9 %
#:     over=(0.012,0.029): 0.70 -> 26.5 %   0.85 -> 28.4 %
#: The suite sits below the 36.5 % this default delivers per curve because 33
#: scenarios still declare the older symmetric +-0.015 band themselves (24.7 %
#: clipped) and the elevated-baseline scenarios get no blank at all.  If those
#: scenarios ever move onto this default the suite measures 37.4 %, over the
#: band, and this weight should come back down to ~0.55 with the branch at
#: (0.005, 0.029).  It is a mixture weight, not a claim that 85 % of real
#: blanks over-subtract.
BLANK_OVERSUBTRACTION_PROB = 0.85

#: The offset is measured from the well's own true baseline (media blank plus
#: the cells already in it), so a blank-subtracted curve starts at ~0 whatever
#: baseline the scenario asked for.  That is right for a scenario whose
#: ``initial_od`` is an inoculum level, and wrong for one whose ``initial_od``
#: IS the post-blank baseline it is testing (``high_baseline_good`` at
#: 0.08-0.15, ``contamination`` at 0.2-0.5 - M09's point that initial_od must
#: stop separating the classes).  A default blank is therefore withheld from a
#: SCENARIO whose declared baseline FLOOR reaches this threshold - clear of the
#: real post-blank baseline IQR of growing conditions (0.019-0.052 OD) and
#: exactly the floor those two scenarios declare (0.08 and 0.2).  Reading the
#: declared floor rather than the sampled draw is deliberate: the statement is
#: about a scenario, and keying it on the draw blanked the low half of every
#: 0.02-0.10 scenario and not the high half (a cliff nobody declared, and 2.8
#: points of suite clipping prevalence).  Scenarios that name their own band
#: are never second-guessed.
BLANK_DEFAULT_MAX_BASELINE_OD = 0.08

#: What ``blank_offset_od`` resolves to when a scenario names nothing.  Kept
#: under its old name so a caller that passes it explicitly still gets the
#: default behaviour; it is now the mixture sentinel rather than a band.
DEFAULT_BLANK_OFFSET_OD = BLANK_CALIBRATED

#: Media name used when a scenario does not name one (BlankModel keys its
#: per-media baseline off this string).
DEFAULT_MEDIA = 'LB'

#: Truth-schema constants re-exported for callers that have the generator only.
TRUTH_SCHEMA_VERSION = truth_module.TRUTH_SCHEMA_VERSION
LABEL_RULE_VERSION = truth_module.LABEL_RULE_VERSION
GROUND_TRUTH_COLUMNS = truth_module.GROUND_TRUTH_COLUMNS

#: The per-curve random streams.  Fixed order: a stream's identity is its
#: position, so adding one at the END leaves every existing stream unchanged.
_STREAM_NAMES = (
    'model', 'shape', 'noise_shared', 'noise_wells', 'replicate',
    'spike', 'drift', 'blank', 'rmse', 'walk',
)


# =============================================================================
# Seeding
# =============================================================================

def curve_seed(base_seed: Optional[int], scenario: str, index: int,
               stream: str = 'curve') -> Optional[int]:
    """
    Content-addressed per-curve seed.

    ``int.from_bytes(blake2b(f"{seed}:{scenario}:{i}", 8), 'big') % 2**31`` for
    the curve stream, with a distinct payload for every other stream (the
    ScenarioSampler's parameters draw from ``stream='params'``).

    This is what makes the suite composable: the realisation of
    ``standard`` curve 3 depends on the base seed, the scenario name and the
    index only, so appending a scenario or changing ``--n-per-scenario`` cannot
    rewrite any other curve.  The old scheme was ``seed + curve_id`` with
    ``curve_id`` accumulating across scenarios in dict order.

    Returns None when ``base_seed`` is None (an explicitly unseeded run).
    """
    if base_seed is None:
        return None
    payload = f"{base_seed}:{scenario}:{int(index)}"
    if stream != 'curve':
        payload = f"{payload}:{stream}"
    digest = hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % 2 ** 31


def _streams(seed: Optional[int],
             fallback: Optional[np.random.Generator] = None
             ) -> Dict[str, np.random.Generator]:
    """Independent, reproducible sub-streams for one curve.

    Derived with :class:`numpy.random.SeedSequence.spawn`, so the noise stream
    is unaffected by how many draws the shape or blank stages made - which is
    what lets an artefact twin be generated with byte-identical noise.
    """
    if seed is None:
        source = fallback if fallback is not None else np.random.default_rng()
        seq = np.random.SeedSequence(int(source.integers(0, 2 ** 31)))
    else:
        seq = np.random.SeedSequence(int(seed))
    children = seq.spawn(len(_STREAM_NAMES))
    return {name: np.random.default_rng(child)
            for name, child in zip(_STREAM_NAMES, children)}


# =============================================================================
# Containers
# =============================================================================

@dataclass
class GeneratedCurve:
    """Container for a single generated curve with metadata.

    ``truth`` carries the measured ground truth of this curve (the v2 columns
    of :data:`truth.GROUND_TRUTH_COLUMNS` that are properties of the curve
    rather than of its position in a batch).
    """
    time: np.ndarray
    od600: np.ndarray
    clean_od600: np.ndarray  # Without noise
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    truth: Dict[str, Any] = field(default_factory=dict)
    replicates: Optional[List[np.ndarray]] = None


def _get(params: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Contract A field lookup: sampled params dict first, contract default else."""
    if not params:
        return default
    value = params.get(key, default)
    return default if value is None else value


def _draw(rng: np.random.Generator, value: Any, default: Any = None) -> Optional[float]:
    """Resolve a scalar / ``(lo, hi)`` range / None to a float."""
    v = default if value is None else value
    if v is None:
        return None
    if isinstance(v, (tuple, list, np.ndarray)):
        lo, hi = float(v[0]), float(v[1])
        if hi < lo:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    return float(v)


def _declared_baseline_floor(params: Optional[Dict[str, Any]],
                             initial_od: float) -> float:
    """
    The lowest baseline the curve's scenario can produce, in OD.

    The guard below is a statement about a SCENARIO ("this one's ``initial_od``
    is the post-blank baseline it exists to test"), so it reads the declared
    range and not the sampled draw: keying it on the draw would blank the
    low-baseline half of a scenario and not the high half, which is a per-curve
    cliff nobody declared.  Falls back to the sampled value for a curve
    generated outside any scenario.
    """
    name = (params or {}).get('scenario_name')
    if name:
        try:
            declared = getattr(get_scenario(str(name)), 'initial_od', None)
        except (KeyError, ValueError):
            declared = None
        if isinstance(declared, (tuple, list)) and len(declared) == 2:
            return float(min(declared))
        if isinstance(declared, (int, float)):
            return float(declared)
    return float(initial_od)


def _draw_blank_offset(rng: np.random.Generator, spec: Any,
                       baseline_floor: float, named: bool) -> Optional[float]:
    """
    Resolve ``blank_offset_od`` to the offset of the subtracted blank, in OD.

    ``spec`` is one of :data:`BLANK_CALIBRATED` (the default mixture),
    :data:`BLANK_OFF` / ``None`` / ``False`` (no blank at all), a ``(lo, hi)``
    band or a scalar.  ``named`` says whether the scenario asked for this
    explicitly: only the DEFAULT is withheld, and only from a scenario whose
    declared baseline floor reaches :data:`BLANK_DEFAULT_MAX_BASELINE_OD`,
    because there ``initial_od`` is the post-blank baseline the scenario exists
    to test and re-referencing it would silently delete the scenario.

    Returns ``None`` when no blank is to be subtracted.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in (BLANK_OFF, 'off', 'false'):
            return None
        if key != BLANK_CALIBRATED:
            raise ValueError(
                f"blank_offset_od must be a (lo, hi) band, a scalar, "
                f"{BLANK_CALIBRATED!r} or {BLANK_OFF!r}; got {spec!r}")
        if not named and float(baseline_floor) >= BLANK_DEFAULT_MAX_BASELINE_OD:
            return None
        band = (BLANK_OVERSUBTRACTION_OD
                if rng.random() < BLANK_OVERSUBTRACTION_PROB
                else BLANK_UNDERSUBTRACTION_OD)
        return _draw(rng, band)
    return _draw(rng, spec)


# =============================================================================
# Generator
# =============================================================================

class SyntheticGrowthCurveGenerator:
    """
    Main class for generating synthetic bacterial growth curves.

    Example usage:
        generator = SyntheticGrowthCurveGenerator()
        curve = generator.generate_single_curve(A=1.5, mu=0.2, lambda_=3.0)
        batch = generator.generate_from_scenario('standard', n=50)
        full_test = generator.generate_comprehensive_test_set()
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Default time parameters
        self.default_duration = 100.0
        self.default_resolution = 0.25

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_time_array(
        self,
        duration_hours: float = None,
        time_resolution: float = None
    ) -> np.ndarray:
        """Generate time array with specified parameters."""
        duration = duration_hours or self.default_duration
        resolution = time_resolution or self.default_resolution
        return np.arange(0, duration, resolution)

    # ------------------------------------------------------------------ clean
    def _model_curve(self, model_type: str, A: float, mu: float, lambda_: float,
                     initial_od: float, time: np.ndarray,
                     params: Optional[Dict[str, Any]] = None
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build one noiseless model curve (baseline NOT added) on ``time``.

        Uses each model's ``from_gompertz_params`` on the caller's own grid, so
        the returned ``realised``/``achieved`` describe this curve rather than a
        default grid, and passes the Contract A kinetic parameters (Ks, Ki, q,
        S0, nu) straight through to the constructors that now accept them.
        """
        extra: Dict[str, Any] = {}
        if model_type == 'gompertz':
            converted = GompertzModel.from_gompertz_params(A, mu, lambda_, t=time)
            model = GompertzModel
        elif model_type == 'baranyi':
            converted = BaranyiModel.from_gompertz_params(
                A, mu, lambda_, initial_od if initial_od > 0 else 0.01, t=time)
            model = BaranyiModel
        elif model_type == 'logistic':
            converted = LogisticModel.from_gompertz_params(A, mu, lambda_, t=time)
            model = LogisticModel
        elif model_type == 'richards':
            nu = _get(params, 'nu', 0.5)
            converted = RichardsModel.from_gompertz_params(A, mu, lambda_, nu=nu, t=time)
            model = RichardsModel
        elif model_type == 'haldane':
            for key, default in (('S0', 1.0), ('Ki', 10.0), ('Ks', 0.1), ('q', 0.1)):
                extra[key] = float(_get(params, key, default))
            converted = HaldaneModel.from_gompertz_params(
                A, mu, lambda_, t=time, **extra)
            model = HaldaneModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        kwargs = {k: v for k, v in converted.items() if k not in ('realised', 'achieved')}
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            clean = np.asarray(model.compute(time, **kwargs), dtype=float)
        info = {
            'realised': bool(converted.get('realised', False)),
            'achieved': dict(converted.get('achieved', {})),
            'model_params': kwargs,
            'kinetics': extra,
        }
        return clean, info

    def _shape_curve(self, shape: str, time: np.ndarray, A: float, mu: float,
                     lambda_: float, params: Optional[Dict[str, Any]],
                     rng: np.random.Generator
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dispatch into WP4's shape registry (Contract C)."""
        shape_params: Dict[str, Any] = {
            'A': A, 'mu': mu, 'lambda_': lambda_, 'rng': rng,
        }
        for key in ('A2', 'mu2', 'lambda2', 'gap_hours',
                    'death_loss_frac', 't_death_offset_h'):
            value = _get(params, key, None)
            if value is not None:
                shape_params[key] = value
        return build_clean_curve(shape, time, shape_params)

    # ------------------------------------------------------- single curve API
    def generate_single_curve(
        self,
        A: float = 1.0,
        mu: float = 0.15,
        lambda_: float = 3.0,
        model_type: str = 'gompertz',
        noise_level: str = 'medium',
        target_r_squared: Optional[float] = None,
        include_death_phase: bool = False,
        t_death: Optional[float] = None,
        k_death: float = 0.02,
        initial_od: float = 0.0,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        apply_artefacts: bool = True,
        scenario: str = 'custom',
    ) -> GeneratedCurve:
        """
        Generate a single synthetic growth curve.

        Args:
            A: Maximum OD600 (asymptotic value) - the REQUEST; what the curve
                actually realises is measured and stored as ``true_A``.
            mu: Maximum specific growth rate (OD/hour), requested
            lambda_: Lag phase duration (hours), requested
            model_type: 'gompertz', 'baranyi', 'logistic', 'richards', 'haldane'
            noise_level: 'very_low', 'low', 'medium', 'high', 'very_high'
            target_r_squared: when given, extra noise is added ON TOP of the
                preset until the full-trace R^2 against the clean curve equals
                it (the two paths used to be exclusive, so a 'very_high' preset
                recorded in ground truth was never applied)
            include_death_phase: Whether to add death/decline phase
            t_death: Time when death phase begins (auto if None)
            k_death: Death rate constant
            initial_od: Initial/baseline OD600
            duration_hours: Experiment duration in hours
            time_resolution: Time between measurements (hours)
            seed: Random seed for this specific curve
            params: the sampled scenario parameters.  Every Contract A field is
                read from here with its contract default: ``shape``,
                ``label_rule``, ``scoring_mode``, ``lab_band``, ``A2``/``mu2``/
                ``lambda2``/``gap_hours``, ``drift_od_per_h``, ``n_spikes``/
                ``spike_od``/``spike_decay_points``, ``blank_offset_od``,
                ``replicate_A_cv``/``replicate_mu_cv``/``replicate_lambda_sd``/
                ``outlier_well_prob``, ``Ks``/``Ki``/``q``/``S0``,
                ``death_loss_frac``/``t_death_offset_h``.
            apply_artefacts: False generates the artefact-free twin of this
                curve - same seed, same noise realisation, no spikes and no
                scenario drift.
            scenario: scenario name recorded in the truth row.

        Returns:
            GeneratedCurve with time, OD600, metadata and measured ``truth``.
        """
        params = dict(params or {})
        streams = _streams(seed, self.rng)
        time = self._get_time_array(duration_hours, time_resolution)

        shape = _get(params, 'shape', None)
        n_replicates = int(_get(params, 'n_replicates', 0)) or None

        replicate_A_cv = float(_get(params, 'replicate_A_cv', 0.0))
        replicate_mu_cv = float(_get(params, 'replicate_mu_cv', 0.0))
        replicate_lambda_sd = float(_get(params, 'replicate_lambda_sd', 0.0))
        outlier_well_prob = float(_get(params, 'outlier_well_prob', 0.0))
        wants_replicates = (replicate_A_cv > 0 or replicate_mu_cv > 0
                            or replicate_lambda_sd > 0 or outlier_well_prob > 0)
        if n_replicates is None:
            n_replicates = DEFAULT_N_REPLICATES if wants_replicates else 1

        # ---------------------------------------------------------- replicates
        wells: List[Dict[str, Any]] = []
        rep_rng = streams['replicate']
        outlier_wells: set = set()
        if n_replicates > 1 and outlier_well_prob > 0:
            if rep_rng.random() < outlier_well_prob:
                n_outlier = int(_get(params, 'n_outlier_wells', 1))
                n_outlier = max(1, min(n_outlier, n_replicates - 1))
                outlier_wells = {int(i) for i in rep_rng.choice(
                    n_replicates, size=n_outlier, replace=False)}
        outlier_scale = _get(params, 'outlier_well_A_scale', OUTLIER_WELL_A_SCALE)
        for w in range(n_replicates):
            if n_replicates == 1:
                A_w, mu_w, lambda_w = A, mu, lambda_
            else:
                A_w = A * float(rep_rng.lognormal(0.0, replicate_A_cv)) if replicate_A_cv > 0 else A
                mu_w = mu * float(rep_rng.lognormal(0.0, replicate_mu_cv)) if replicate_mu_cv > 0 else mu
                lambda_w = lambda_ + (float(rep_rng.normal(0.0, replicate_lambda_sd))
                                      if replicate_lambda_sd > 0 else 0.0)
                lambda_w = max(0.0, lambda_w)
            if w in outlier_wells:
                A_w = A_w * float(_draw(rep_rng, outlier_scale, OUTLIER_WELL_A_SCALE))
            wells.append({'A': A_w, 'mu': mu_w, 'lambda_': lambda_w,
                          'outlier': w in outlier_wells})

        # ------------------------------------------------------- clean curves
        shape_meta: Dict[str, Any] = {}
        realised_all = True
        achieved: Dict[str, float] = {}
        shape_seed = int(streams['shape'].integers(0, 2 ** 31))
        for w in wells:
            if shape:
                clean_w, meta_w = self._shape_curve(
                    shape, time, w['A'], w['mu'], w['lambda_'], params,
                    np.random.default_rng(shape_seed))
                if meta_w.get('extras', {}).get('requires_zero_baseline') and initial_od:
                    raise ValueError(
                        f"shape {shape!r} must be generated with initial_od=0, "
                        f"got {initial_od}")
                shape_meta = meta_w
                w['clean'] = clean_w
                # A shape "solved" its request when its first phase really is
                # the (A, mu, lambda) it was handed.  clipped_then_grows is
                # calibrated to G4 PERMETHRINANDLB_CYPERM2 and overrides them,
                # so it honestly reports False rather than claiming a shape it
                # did not build.
                w['realised'] = _phase1_matches(meta_w, w['A'], w['mu'], w['lambda_'])
                realised_all = realised_all and w['realised']
            else:
                clean_w, info = self._model_curve(
                    model_type, w['A'], w['mu'], w['lambda_'], initial_od,
                    time, params)
                w['clean'] = clean_w
                w['realised'] = info['realised']
                realised_all = realised_all and info['realised']
                achieved = info['achieved'] or achieved

        # baseline + death phase
        t_death_used = float('nan')
        for w in wells:
            clean_w = w['clean'] + initial_od
            if include_death_phase:
                td = t_death if t_death is not None else duration_hours * 0.7
                clean_w = _apply_exponential_death(clean_w, time, td, k_death)
                t_death_used = float(td)
            w['clean'] = clean_w

        if shape:
            t_death_used = float(shape_meta.get('t_death', float('nan')))
        t_second_start = float(shape_meta.get('t_second_start', float('nan'))) if shape else float('nan')

        clean = np.mean(np.array([w['clean'] for w in wells]), axis=0)

        # ------------------------------------------------------------- noise
        observed, noise_info = self._apply_noise_stack(
            wells, clean, time, params, streams,
            noise_level=noise_level,
            target_r_squared=target_r_squared,
            initial_od=initial_od,
            apply_artefacts=apply_artefacts,
        )

        # ------------------------------------------------------------- truth
        parameters = {
            'A': A, 'mu': mu, 'lambda_': lambda_,
            'model_type': model_type,
            'initial_od': initial_od,
            'include_death_phase': include_death_phase,
            't_death': t_death_used if include_death_phase else None,
            'k_death': k_death if include_death_phase else None,
            'shape': shape,
            'duration_hours': duration_hours,
            'time_resolution': time_resolution,
        }
        metadata = {
            'noise_level': noise_level,
            'target_r_squared': target_r_squared,
            'scenario': scenario,
            'seed': seed,
            'has_clean_reference': True,
        }
        metadata.update(noise_info)

        curve = GeneratedCurve(
            time=time, od600=observed, clean_od600=clean,
            parameters=parameters, metadata=metadata,
            replicates=[w['noisy'] for w in wells] if n_replicates > 1 else None,
        )
        self._finalize_metadata(curve, params=params,
                                t_death=t_death_used,
                                t_second_start=t_second_start,
                                shape_meta=shape_meta,
                                realised=realised_all,
                                n_replicates=n_replicates,
                                replicate_A_cv=replicate_A_cv,
                                replicate_mu_cv=replicate_mu_cv,
                                outlier_well=bool(outlier_wells))
        return curve

    # ------------------------------------------------------------ noise stack
    def _apply_noise_stack(
        self,
        wells: List[Dict[str, Any]],
        clean: np.ndarray,
        time: np.ndarray,
        params: Dict[str, Any],
        streams: Dict[str, np.random.Generator],
        noise_level: str,
        target_r_squared: Optional[float],
        initial_od: float,
        apply_artefacts: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Noise, artefacts and the blank stage, in the order the instrument does
        them.

        1. per-well instrument noise (coloured; ``REPLICATE_SHARED_SIGMA_FRACTION``
           of the high-frequency sigma plus the slow terms and the drift are
           COMMON to the wells of a condition, only the white part is
           independent), then the mean over wells;
        2. scenario artefacts - one-sided decaying spikes and a signed drift -
           applied to the well signal, skipped entirely for an artefact-free
           twin;
        3. the blank stage: subtract a :class:`BlankModel` trace whose mean sits
           ``blank_offset_od`` above the well's own true baseline, then clip at
           zero.  This is the only way the pipeline's ``noise_floor=0.002``
           substitution for a clipped baseline is ever exercised;
        4. the target-R^2 top-up, added to whatever the preset produced rather
           than replacing it.
        """
        n = time.size
        n_wells = len(wells)
        info: Dict[str, Any] = {
            'noise_sources': 'preset',
            'spike_indices': np.zeros(0, dtype=int),
            'drift_od_per_h': 0.0,
            'blank_offset_od': float('nan'),
            'n_replicates': n_wells,
        }

        scenario_drift = _draw(streams['drift'], _get(params, 'drift_od_per_h', None))
        blank_spec = _get(params, 'blank_offset_od', DEFAULT_BLANK_OFFSET_OD)
        blank_named = bool(params) and params.get('blank_offset_od') is not None
        blank_offset = _draw_blank_offset(
            streams['blank'], blank_spec,
            _declared_baseline_floor(params, initial_od), blank_named)
        media = str(_get(params, 'media', DEFAULT_MEDIA))

        # ---- blank the well sits on (raw wells are ~0.13 OD, not 0)
        blank_own = None
        blank_sub = None
        if blank_offset is not None:
            blank_model = BlankModel()
            blank_own = blank_model.sample_blank(media, time, rng=streams['blank'])
            # What a lab subtracts is the MEAN of the plate's blank wells (the
            # real plates carry 18 of them per media), not one well: averaging
            # BLANK_REFERENCE_WELLS of them is what keeps the subtraction from
            # adding sqrt(2) x a blank's own noise and its own drift to every
            # trace - with a single reference well the three-well mean of a
            # condition ends up NOISIER than its wells (ratio 1.05 against the
            # real 0.848), which is the opposite of what averaging means.
            blank_sub = np.mean(
                [blank_model.sample_blank(media, time, rng=streams['blank'])
                 for _ in range(BLANK_REFERENCE_WELLS)], axis=0)
            # The subtracted blank's MEAN sits ``blank_offset`` above the well's
            # own TRUE baseline - its media blank plus the cells already in it -
            # so the clipping prevalence is a property of the blank choice and
            # not of how high the scenario's initial_od happens to be.
            blank_sub = (blank_sub - float(np.mean(blank_sub))
                         + float(np.mean(blank_own)) + float(initial_od) + blank_offset)
            info['blank_offset_od'] = float(blank_offset)

        # ---- 1. instrument noise
        preset = get_noise_model(noise_level)
        # A scenario may override the preset's high-frequency sigma with a
        # single per-well sigma (the noise ladder's rungs are defined by the
        # realised oracle error at a stated per-well sigma, not by a nominal
        # R^2).  The override replaces both the baseline and the proportional
        # term so the stated number IS the sigma the fit sees.
        well_sigma = _draw(streams['noise_wells'], _get(params, 'well_sigma_od', None))
        if well_sigma is not None:
            preset.baseline_sigma = float(well_sigma)
            preset.proportional_sigma = 0.0
            info['well_sigma_od'] = float(well_sigma)
        if scenario_drift is not None:
            # the scenario owns the drift; do not double-count the preset's
            preset.drift_rate = 0.0
        drift_total = 0.0

        if n_wells == 1:
            base = wells[0]['clean'] if blank_own is None else wells[0]['clean'] + blank_own
            noisy = preset.apply(base, time=time, rng=streams['noise_wells'])
            drift_total += float(preset.last_drift_rate)
            wells[0]['noisy'] = noisy
            signal = noisy
        else:
            shared = self._shared_noise(clean, time, noise_level, streams['noise_shared'],
                                        sigma_override=well_sigma)
            drift_rate = preset.sample_drift_rate(streams['noise_shared'])
            drift_total += float(drift_rate)
            shared = shared + drift_rate * (time - time[0])
            spike_rng = streams['noise_wells']
            for w in wells:
                base = w['clean'] if blank_own is None else w['clean'] + blank_own
                indep = self._independent_noise(base, time, noise_level, spike_rng,
                                                sigma_override=well_sigma)
                noisy = base + shared + indep
                idx = np.nonzero(spike_rng.random(n) < preset.outlier_prob)[0] \
                    if preset.outlier_prob else np.zeros(0, dtype=int)
                if idx.size:
                    mags = preset.outlier_magnitude * spike_rng.uniform(0.6, 1.4, size=idx.size)
                    noisy = inject_spikes(noisy, idx, mags, preset.spike_decay_points)
                w['noisy'] = np.maximum(0.0, noisy)
            signal = np.mean(np.array([w['noisy'] for w in wells]), axis=0)

        # ---- 2. scenario artefacts
        n_spikes = int(_get(params, 'n_spikes', 0))
        if apply_artefacts and n_spikes > 0 and n:
            spike_rng = streams['spike']
            window = _get(params, 'spike_window_h', None)
            if window is not None:
                # Spikes inside a stated time window (hours): a bubble that
                # lands on the plateau is a different test from one that lands
                # in the exponential phase, where it can be taken for the peak.
                lo = int(np.searchsorted(time, float(window[0])))
                hi = int(np.searchsorted(time, float(window[1]), side='right'))
                lo = max(0, min(lo, n - 1))
                hi = max(lo + 1, min(hi, n))
            else:
                lo, hi = int(0.05 * n), n
            pool = np.arange(lo, hi)
            idx = np.sort(spike_rng.choice(pool, size=min(n_spikes, pool.size),
                                           replace=False))
            spike_range = _get(params, 'spike_od', DEFAULT_SPIKE_OD)
            mags = np.array([_draw(spike_rng, spike_range) for _ in idx], dtype=float)
            decay = int(_get(params, 'spike_decay_points', DEFAULT_SPIKE_DECAY_POINTS))
            signal = inject_spikes(signal, idx, mags, decay)
            info['spike_indices'] = idx

        if apply_artefacts and scenario_drift is not None:
            signal = apply_signed_drift(signal, time, scenario_drift)
            drift_total += float(scenario_drift)
            info['scenario_drift_applied'] = True
        info['drift_od_per_h'] = float(drift_total)

        # ---- 3. blank stage, then the single zero clip
        if blank_sub is not None:
            signal = signal - blank_sub
            # The per-well traces a lab hands you are blank-subtracted too, and
            # the same blank is subtracted from each: keeping the stored
            # replicates in the raw state while the condition mean is
            # blank-subtracted made the mean look NOISIER than its own wells
            # (ratio 1.05 against the real 0.848) - an artefact of comparing
            # two different processing states, not of the noise model.
            for well in wells:
                well['noisy'] = np.maximum(0.0, well['noisy'] - blank_sub)
        signal = np.maximum(0.0, signal)

        # ---- 4. target-R^2 top-up, additive to the preset
        if target_r_squared is not None:
            ss_tot = float(np.sum((clean - np.mean(clean)) ** 2))
            target_ss_res = (1.0 - float(target_r_squared)) * ss_tot
            component, _achieved_r2, added = add_target_rmse_component(
                clean, signal - clean, target_ss_res, rng=streams['rmse'])
            if added:
                signal = np.maximum(0.0, signal + component)
                info['noise_sources'] = 'preset+rmse'
            else:
                info['noise_sources'] = 'preset'

        return signal, info

    @staticmethod
    def _sigma_profile(od: np.ndarray, noise_level: str,
                       sigma_override: Optional[float] = None) -> np.ndarray:
        if sigma_override is not None:
            return np.full_like(np.asarray(od, dtype=float), float(sigma_override))
        preset = NOISE_PRESETS[noise_level]
        return (float(preset['baseline_sigma'])
                + float(preset['proportional_sigma']) * np.abs(od))

    def _shared_noise(self, clean: np.ndarray, time: np.ndarray,
                      noise_level: str, rng: np.random.Generator,
                      sigma_override: Optional[float] = None) -> np.ndarray:
        """The common-mode part of the three wells' noise.

        ``REPLICATE_SHARED_SIGMA_FRACTION`` of the high-frequency sigma plus the
        WHOLE slow structure: the OU wander and the drift are properties of the
        plate and the reader, not of a well.
        """
        sigma = REPLICATE_SHARED_SIGMA_FRACTION * self._sigma_profile(
            clean, noise_level, sigma_override)
        return ColoredNoise(sigma).sample(time, rng=rng)

    def _independent_noise(self, clean: np.ndarray, time: np.ndarray,
                           noise_level: str, rng: np.random.Generator,
                           sigma_override: Optional[float] = None) -> np.ndarray:
        """The per-well part: white/fast only, ``sqrt(1 - f^2)`` of sigma."""
        f = REPLICATE_SHARED_SIGMA_FRACTION
        sigma = np.sqrt(max(0.0, 1.0 - f * f)) * self._sigma_profile(
            clean, noise_level, sigma_override)
        return ColoredNoise(sigma, sd_slow=0.0).sample(time, rng=rng)

    # -------------------------------------------------------------- metadata
    def _finalize_metadata(
        self,
        curve: GeneratedCurve,
        params: Optional[Dict[str, Any]] = None,
        t_death: float = float('nan'),
        t_second_start: float = float('nan'),
        shape_meta: Optional[Dict[str, Any]] = None,
        realised: bool = True,
        n_replicates: int = 1,
        replicate_A_cv: float = 0.0,
        replicate_mu_cv: float = 0.0,
        outlier_well: bool = False,
        artefact_free: Optional[GeneratedCurve] = None,
    ) -> GeneratedCurve:
        """
        Fill the grid columns, the fit-free metrics and the measured truth.

        Used by every entry point (single curve, flat, random walk, diauxic) so
        that ``n_points``, ``time_resolution`` and ``duration_hours`` mean the
        same thing everywhere, and so that no metric is ever a sentinel:
        ``duration_hours = n_points * dt`` (a 100 h run reports 100.0, not the
        99.75 the old code produced by reading a key that never existed), and
        ``actual_r_squared``/``rmse`` are NaN with ``metrics_defined=False``
        wherever there is no clean reference to compute them against.
        """
        params = dict(params or {})
        shape_meta = dict(shape_meta or {})
        time = np.asarray(curve.time, dtype=float)
        od = np.asarray(curve.od600, dtype=float)
        clean = np.asarray(curve.clean_od600, dtype=float)

        n_points = int(time.size)
        dt = float(np.median(np.diff(time))) if n_points > 1 else float('nan')
        curve.metadata['n_points'] = n_points
        curve.metadata['time_resolution'] = dt
        curve.metadata['duration_hours'] = float(n_points * dt) if np.isfinite(dt) else float('nan')
        curve.metadata['max_od'] = float(np.max(od))
        curve.metadata['delta_od'] = float(np.max(od) - np.mean(od[:5]))

        has_clean = bool(curve.metadata.get('has_clean_reference', True))
        rmse, r2 = _fit_free_metrics(od, clean, has_clean)
        curve.metadata['rmse'] = rmse
        curve.metadata['actual_r_squared'] = r2
        curve.metadata['metrics_defined'] = bool(np.isfinite(rmse) and np.isfinite(r2))

        # ---- descriptive truth, measured on this curve's own clean version
        #
        # Measured on the array that was actually written, in every case.  A
        # random walk has no generating curve (its "clean" array IS the trace
        # it ships), so it declares has_growth_process=False: truth.measure_shape
        # still measures the amplitude that array reaches - erratic walks rise
        # up to 0.42 OD and used to ship true_A = 0.0, a sentinel in a measured
        # column contradicting the curve in the same row - and returns NaN for
        # the rate and the lag, which a walk does not have.  That NaN true_mu is
        # also what keeps the oracle columns NaN instead of dividing by zero.
        measured = truth_module.measure_shape(
            clean, time, t_death=t_death, t_second_start=t_second_start,
            has_growth_process=has_clean)

        requested_A = float(curve.parameters.get('A', float('nan')))
        requested_mu = float(curve.parameters.get('mu', float('nan')))
        requested_lambda = float(curve.parameters.get('lambda_', float('nan')))

        t_primary_95 = float(shape_meta.get('t_primary_95', measured.get('t_primary_95', float('nan'))))
        if shape_meta:
            window_end = float(shape_meta.get('true_window_end_h', float('nan')))
        else:
            boundary = truth_module.measurement_boundary(time, t_second_start, t_death)
            window_end = truth_module.true_window_end(
                t_primary_95, boundary=boundary, t_end=float(time[-1]))

        oracle = truth_module.oracle_recovery(
            od, clean, time,
            {'A': measured['true_A'], 'mu': measured['true_mu'],
             'lambda_': measured['true_lambda'],
             'initial_od': float(curve.parameters.get('initial_od', 0.0))},
            window_end_h=window_end,
        ) if has_clean else {
            'true_window_end_h': float('nan'), 't_primary_95': float('nan'),
            'oracle_A_relerr': float('nan'), 'oracle_mu_relerr': float('nan'),
            'oracle_fit_ok': False}

        label = truth_module.apply_label_rule(
            params if params else {'expected_class': curve.metadata.get('expected_class', 'BAD')},
            {'true_delta_od': measured['true_A'],
             'oracle_A_relerr': oracle['oracle_A_relerr'],
             'oracle_mu_relerr': oracle['oracle_mu_relerr'],
             'oracle_fit_ok': oracle['oracle_fit_ok']},
        )
        if not params:
            # A hand-built curve has no scenario, so it has no scenario label:
            # 'UNKNOWN' rather than a default that a consumer would score.
            label['expected_class'] = str(curve.metadata.get('expected_class', 'UNKNOWN'))

        mu_ratio = (measured['true_mu'] / requested_mu
                    if np.isfinite(measured['true_mu']) and requested_mu else float('nan'))
        lambda_err = (measured['true_lambda'] - requested_lambda
                      if np.isfinite(measured['true_lambda']) else float('nan'))
        frac_of_A = (measured['true_A'] / requested_A
                     if np.isfinite(measured['true_A']) and requested_A else float('nan'))

        af_r2 = float('nan')
        af_rmse = float('nan')
        if artefact_free is not None:
            af_rmse = artefact_free.metadata.get('rmse', float('nan'))
            af_r2 = artefact_free.metadata.get('actual_r_squared', float('nan'))
        elif not _has_artefacts(curve.metadata):
            af_rmse, af_r2 = rmse, r2

        curve.truth = {
            'scenario': curve.metadata.get('scenario', 'custom'),
            'expected_class': label['expected_class'],
            'label_rule': label['label_rule'],
            'label_rule_version': label['label_rule_version'],
            'label_reason': label['label_reason'],
            'scoring_mode': str(_get(params, 'scoring_mode', 'standard')),
            'lab_band': str(_get(params, 'lab_band', '') or ''),
            'model_type': curve.parameters.get('model_type', 'unknown'),
            'shape': str(curve.parameters.get('shape') or ''),
            'requested_A': requested_A,
            'requested_mu': requested_mu,
            'requested_lambda': requested_lambda,
            'true_A': measured['true_A'],
            'true_mu': measured['true_mu'],
            'true_lambda': measured['true_lambda'],
            'mu_realisation_ratio': mu_ratio,
            'lambda_realisation_error': lambda_err,
            'shape_solved': bool(realised),
            'initial_od': float(curve.parameters.get('initial_od', 0.0)),
            'noise_level': curve.metadata.get('noise_level', 'medium'),
            'noise_sources': curve.metadata.get('noise_sources', 'preset'),
            'target_r_squared': curve.metadata.get('target_r_squared', float('nan')),
            'actual_r_squared': r2,
            'artefact_free_r_squared': af_r2,
            'artefact_free_rmse': af_rmse,
            'rmse': rmse,
            'max_od': curve.metadata['max_od'],
            'delta_od': curve.metadata['delta_od'],
            'n_points': n_points,
            'time_resolution': dt,
            'duration_hours': curve.metadata['duration_hours'],
            'seed': curve.metadata.get('seed'),
            'true_window_end_h': oracle['true_window_end_h'],
            't_primary_95': t_primary_95,
            't_second_start': float(t_second_start),
            't_death': float(t_death),
            'oracle_A_relerr': oracle['oracle_A_relerr'],
            'oracle_mu_relerr': oracle['oracle_mu_relerr'],
            'oracle_fit_ok': bool(oracle['oracle_fit_ok']),
            'true_frac_of_A_reached': frac_of_A,
            'replicate_A_cv': float(replicate_A_cv),
            'replicate_mu_cv': float(replicate_mu_cv),
            'outlier_well': bool(outlier_well),
            'spike_indices': truth_module.format_spike_indices(
                curve.metadata.get('spike_indices', ())),
            'drift_od_per_h': float(curve.metadata.get('drift_od_per_h', 0.0)),
            'twin_curve_id': '',
            'metrics_defined': bool(curve.metadata['metrics_defined']),
            'n_replicates': int(n_replicates),
        }
        curve.metadata['expected_class'] = label['expected_class']
        if curve.truth['scoring_mode'] not in truth_module.SCORING_MODES:
            raise ValueError(
                f"unknown scoring_mode {curve.truth['scoring_mode']!r}; "
                f"expected one of {truth_module.SCORING_MODES}")
        return curve

    # -------------------------------------------------------- special curves
    def generate_flat_curve(
        self,
        initial_od: float = 0.01,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        noise_level: str = 'very_low',
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        apply_artefacts: bool = True,
        scenario: str = 'flat',
    ) -> GeneratedCurve:
        """
        Generate a flat curve (no growth, like an H2O control).

        The clean reference exists (it is the constant baseline), so ``rmse``
        is a real number; ``actual_r_squared`` is NaN with
        ``metrics_defined=False`` because the total sum of squares of a
        constant curve is exactly zero - it used to be hard-coded to 0.0, which
        reads as a real R^2 of zero.
        """
        params = dict(params or {})
        streams = _streams(seed, self.rng)
        time = self._get_time_array(duration_hours, time_resolution)
        clean = np.full_like(time, float(initial_od))

        wells = [{'A': 0.0, 'mu': 0.0, 'lambda_': 0.0, 'clean': clean, 'outlier': False}]
        observed, info = self._apply_noise_stack(
            wells, clean, time, params, streams,
            noise_level=noise_level, target_r_squared=None,
            initial_od=initial_od, apply_artefacts=apply_artefacts)

        parameters = {
            'A': 0.0, 'mu': 0.0, 'lambda_': 0.0,
            'model_type': 'flat', 'initial_od': initial_od,
            'include_death_phase': False, 'shape': None,
            'duration_hours': duration_hours, 'time_resolution': time_resolution,
        }
        metadata = {'noise_level': noise_level, 'pattern': 'flat',
                    'scenario': scenario, 'seed': seed,
                    'target_r_squared': None, 'has_clean_reference': True}
        metadata.update(info)

        curve = GeneratedCurve(time=time, od600=observed, clean_od600=clean,
                               parameters=parameters, metadata=metadata)
        return self._finalize_metadata(curve, params=params)

    def generate_random_walk_curve(
        self,
        initial_od: float = 0.1,
        walk_sigma: float = 0.02,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        apply_artefacts: bool = True,
        scenario: str = 'random_walk',
    ) -> GeneratedCurve:
        """
        Generate an erratic random walk curve (non-biological pattern).

        There is no generating curve behind a random walk, so ``rmse`` and
        ``actual_r_squared`` are NaN and ``metrics_defined`` is False.  The old
        code wrote -1.0, which every consumer reads as a real (terrible) R^2.
        """
        params = dict(params or {})
        streams = _streams(seed, self.rng)
        time = self._get_time_array(duration_hours, time_resolution)

        steps = streams['walk'].normal(0, walk_sigma, time.size)
        od = np.clip(initial_od + np.cumsum(steps), 0, 2.0)

        parameters = {
            'A': 0.0, 'mu': 0.0, 'lambda_': 0.0,
            'model_type': 'random_walk', 'initial_od': initial_od,
            'include_death_phase': False, 'shape': None,
            'duration_hours': duration_hours, 'time_resolution': time_resolution,
        }
        metadata = {'pattern': 'random_walk', 'walk_sigma': walk_sigma,
                    'noise_level': str(_get(params, 'noise_level', 'high')),
                    'noise_sources': 'random_walk',
                    'scenario': scenario, 'seed': seed,
                    'target_r_squared': None, 'has_clean_reference': False,
                    'spike_indices': np.zeros(0, dtype=int), 'drift_od_per_h': 0.0}

        curve = GeneratedCurve(time=time, od600=od, clean_od600=od.copy(),
                               parameters=parameters, metadata=metadata)
        return self._finalize_metadata(curve, params=params)

    def generate_diauxic_curve(
        self,
        A1: float = 0.8,
        mu1: float = 0.25,
        lambda1: float = 2.0,
        A2: float = 0.5,
        mu2: float = 0.15,
        gap_hours: float = 10.0,
        duration_hours: float = 100.0,
        time_resolution: float = 0.25,
        noise_level: str = 'medium',
        initial_od: float = 0.0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        apply_artefacts: bool = True,
        scenario: str = 'diauxic',
    ) -> GeneratedCurve:
        """
        Generate a diauxic-like curve with two growth phases.

        A clean reference exists here, so the metrics are real numbers - they
        used to be missing entirely, which is why ``truncation_challenge``
        shipped NaN ``actual_r_squared``/``rmse``/``delta_od``.  The truth of
        the FIRST phase is measured over ``t <= t_second_start``, so the second
        phase can never be reported as the first phase's amplitude.
        """
        params = dict(params or {})
        streams = _streams(seed, self.rng)
        time = self._get_time_array(duration_hours, time_resolution)

        od1 = GompertzModel.compute(time, A1, mu1, lambda1)
        t_first_plateau = lambda1 + A1 / (mu1 * np.e) + 5
        t_second_start = t_first_plateau + gap_hours
        od2 = np.zeros_like(time)
        mask = time > t_second_start
        od2[mask] = A2 * (1 - np.exp(-mu2 * (time[mask] - t_second_start)))
        clean = od1 + od2 + initial_od

        wells = [{'A': A1, 'mu': mu1, 'lambda_': lambda1, 'clean': clean,
                  'outlier': False}]
        observed, info = self._apply_noise_stack(
            wells, clean, time, params, streams,
            noise_level=noise_level, target_r_squared=None,
            initial_od=initial_od, apply_artefacts=apply_artefacts)

        parameters = {
            'A': A1, 'mu': mu1, 'lambda_': lambda1,
            'model_type': 'diauxic', 'initial_od': initial_od,
            'include_death_phase': False, 'shape': None,
            'A2': A2, 'mu2': mu2, 'gap_hours': gap_hours,
            'duration_hours': duration_hours, 'time_resolution': time_resolution,
        }
        metadata = {'pattern': 'diauxic', 'noise_level': noise_level,
                    'scenario': scenario, 'seed': seed,
                    'target_r_squared': None, 'has_clean_reference': True,
                    'first_plateau_time': float(t_first_plateau)}
        metadata.update(info)

        curve = GeneratedCurve(time=time, od600=observed, clean_od600=clean,
                               parameters=parameters, metadata=metadata)
        return self._finalize_metadata(curve, params=params,
                                       t_second_start=float(t_second_start))

    # ------------------------------------------------------------- scenarios
    def generate_from_scenario(
        self,
        scenario_name: str,
        n_curves: int = 10,
        seed: Optional[int] = None,
    ) -> List[GeneratedCurve]:
        """
        Generate curves according to a scenario specification.

        Seeding is content-addressed (see :func:`curve_seed`): curve ``i`` of
        this scenario depends on ``(seed, scenario_name, i)`` only, so the
        curves of one scenario are unaffected by which other scenarios are
        generated, in what order, or how many.

        A scenario that carries an explicit artefact (``n_spikes`` or
        ``drift_od_per_h``) additionally yields an artefact-free TWIN of every
        curve, generated from the same seed with the same noise realisation, so
        the validator can measure the shift the artefact caused rather than
        only the absolute error.  ``generate_twin`` in the scenario parameters
        overrides that default.

        Args:
            scenario_name: Name of scenario from curve_scenarios.py
            n_curves: Number of curves to generate (twins are additional)
            seed: Base random seed

        Returns:
            List of GeneratedCurve objects
        """
        scenario = get_scenario(scenario_name)

        curves: List[GeneratedCurve] = []
        for i in range(n_curves):
            sampler = ScenarioSampler(
                scenario, seed=curve_seed(seed, scenario_name, i, stream='params'))
            params = sampler.sample_parameters()
            params.setdefault('expected_class', scenario.expected_class)
            this_seed = curve_seed(seed, scenario_name, i)

            primary = self._curve_for_params(params, scenario_name, this_seed,
                                             apply_artefacts=True)
            primary.metadata['curve_index'] = i
            primary.metadata['is_twin'] = False
            curves.append(primary)

            if _wants_twin(params):
                twin = self._curve_for_params(params, scenario_name, this_seed,
                                              apply_artefacts=False)
                twin.metadata['curve_index'] = i
                twin.metadata['is_twin'] = True
                twin.truth['artefact_free_r_squared'] = twin.truth['actual_r_squared']
                twin.truth['artefact_free_rmse'] = twin.truth['rmse']
                primary.truth['artefact_free_r_squared'] = twin.truth['actual_r_squared']
                primary.truth['artefact_free_rmse'] = twin.truth['rmse']
                curves.append(twin)

        return curves

    def _curve_for_params(self, params: Dict[str, Any], scenario_name: str,
                          seed: Optional[int], apply_artefacts: bool = True
                          ) -> GeneratedCurve:
        """Dispatch one sampled parameter set to the right builder."""
        pattern = params.get('pattern')
        common = dict(seed=seed, params=params, apply_artefacts=apply_artefacts,
                      scenario=scenario_name)

        if pattern == 'flat':
            curve = self.generate_flat_curve(
                initial_od=params['initial_od'],
                duration_hours=params['duration_hours'],
                time_resolution=params['time_resolution'],
                noise_level=params['noise_level'], **common)
        elif pattern == 'random_walk':
            curve = self.generate_random_walk_curve(
                initial_od=params['initial_od'],
                duration_hours=params['duration_hours'],
                time_resolution=params['time_resolution'], **common)
        elif pattern == 'diauxic':
            rng = np.random.default_rng(curve_seed(seed, scenario_name, 0, 'diauxic')
                                        if seed is not None else None)
            curve = self.generate_diauxic_curve(
                A1=params['A'], mu1=params['mu'], lambda1=params['lambda_'],
                A2=_draw(rng, params.get('A2'), 0.5),
                mu2=_draw(rng, params.get('mu2'), 0.15),
                gap_hours=_draw(rng, params.get('gap_hours'), 10.0),
                initial_od=params['initial_od'],
                duration_hours=params['duration_hours'],
                time_resolution=params['time_resolution'],
                noise_level=params['noise_level'], **common)
        elif _wants_replicates(params):
            # A condition with replicate structure goes through the replicate
            # path: three wells with 76 % common-mode noise, averaged, with the
            # truth measured on the noiseless mean.  This is what every real
            # column the pipeline reads is.
            curve = self.generate_triplicate(
                A=params['A'], mu=params['mu'], lambda_=params['lambda_'],
                noise_level=params['noise_level'],
                n_replicates=int(_get(params, 'n_replicates', DEFAULT_N_REPLICATES)),
                model_type=params['model'],
                target_r_squared=params.get('target_r_squared'),
                include_death_phase=params['include_death_phase'],
                t_death=(params['duration_hours'] * params.get('t_death_fraction', 0.7)
                         if params['include_death_phase'] else None),
                k_death=params.get('k_death', 0.02),
                initial_od=params['initial_od'],
                duration_hours=params['duration_hours'],
                time_resolution=params['time_resolution'], **common)[4]
        else:
            curve = self.generate_single_curve(
                A=params['A'], mu=params['mu'], lambda_=params['lambda_'],
                model_type=params['model'],
                noise_level=params['noise_level'],
                target_r_squared=params.get('target_r_squared'),
                include_death_phase=params['include_death_phase'],
                t_death=(params['duration_hours'] * params.get('t_death_fraction', 0.7)
                         if params['include_death_phase'] else None),
                k_death=params.get('k_death', 0.02),
                initial_od=params['initial_od'],
                duration_hours=params['duration_hours'],
                time_resolution=params['time_resolution'], **common)

        curve.metadata['scenario'] = scenario_name
        curve.truth['scenario'] = scenario_name
        return curve

    # ------------------------------------------------------------ replicates
    def generate_triplicate(
        self,
        A: float = 1.0,
        mu: float = 0.15,
        lambda_: float = 3.0,
        replicate_variance: float = 0.05,
        noise_level: str = 'medium',
        seed: Optional[int] = None,
        n_replicates: int = DEFAULT_N_REPLICATES,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], GeneratedCurve]:
        """
        Generate biological replicates of one condition and their mean.

        This was dead code with no caller; it is now the replicate path the
        comprehensive generator uses whenever a scenario declares replicate
        variation, and it models what the real plates do: 76 % of a well's
        high-frequency sigma is COMMON to the three wells of the condition
        (measured mean/well ratio 0.848 against 0.577 for independent noise),
        the slow wander and the drift are entirely common, and only the white
        term is independent.  Biological spread between wells is A CV 0.05
        (lognormal), mu CV 0.10 and lambda sd 0.5 h; ``outlier_well_prob``
        turns one well into the G3 LCY01 case (0.50-0.70 of its siblings' A).

        Args:
            A, mu, lambda_: base parameters (the request)
            replicate_variance: default CV used for A when the params dict
                carries no explicit ``replicate_A_cv``
            noise_level: preset for every well
            seed: base seed for this condition
            n_replicates: wells per condition (3 on the real plates)
            params: sampled scenario parameters (Contract A fields)
            **kwargs: forwarded to :meth:`generate_single_curve`

        Returns:
            ``(time, od_mean, od_std, well_traces, mean_curve)`` - the last
            entry is the :class:`GeneratedCurve` for the averaged condition,
            whose truth is measured on the noiseless MEAN of the wells.
        """
        merged = dict(params or {})
        merged.setdefault('replicate_A_cv', replicate_variance)
        merged.setdefault('replicate_mu_cv', DEFAULT_REPLICATE_MU_CV)
        merged.setdefault('replicate_lambda_sd', DEFAULT_REPLICATE_LAMBDA_SD_H)
        merged['n_replicates'] = int(n_replicates)

        curve = self.generate_single_curve(
            A=A, mu=mu, lambda_=lambda_, noise_level=noise_level,
            seed=seed, params=merged, **kwargs)

        wells = curve.replicates or [curve.od600]
        stack = np.array(wells)
        return (curve.time, np.mean(stack, axis=0), np.std(stack, axis=0),
                list(wells), curve)

    # ------------------------------------------------------------ batch API
    def generate_comprehensive_test_set(
        self,
        config: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate comprehensive test data covering all scenarios.

        Every row carries the full v2 ground-truth schema
        (:data:`truth.GROUND_TRUTH_COLUMNS`) plus the time and OD vectors, so
        ``output_formatter.create_ground_truth_csv`` only has to select.

        Args:
            config: Dict mapping scenario name to number of curves.
                   If None, uses default configuration.
            seed: Random seed for reproducibility
            verbose: print per-scenario progress

        Returns:
            DataFrame with all generated curves and metadata
        """
        if config is None:
            config = get_comprehensive_test_config()

        if seed is not None:
            self.set_seed(seed)

        all_rows: List[Dict[str, Any]] = []
        curve_id = 0

        for scenario_name, n_curves in config.items():
            if verbose:
                print(f"Generating {n_curves} curves for scenario: {scenario_name}")

            curves = self.generate_from_scenario(scenario_name, n_curves=n_curves,
                                                 seed=seed)

            all_rows.extend(curve_rows(curves, start_id=curve_id))
            curve_id += len(curves)

        df = pd.DataFrame(all_rows)

        scenario_class = {name: cfg.expected_class for name, cfg in ALL_SCENARIOS.items()}
        truth_module.assert_scenario_labels_untouched(all_rows, scenario_class)

        if verbose:
            print(f"\nGenerated {len(df)} total curves:")
            print(f"  Expected GOOD: {len(df[df['expected_class'] == 'GOOD'])}")
            print(f"  Expected BAD: {len(df[df['expected_class'] == 'BAD'])}")

        return df

    def generate_batch_from_parameters(
        self,
        parameter_df: pd.DataFrame,
        noise_level: str = 'medium',
        seed: Optional[int] = None
    ) -> List[GeneratedCurve]:
        """
        Generate curves from a DataFrame of parameters.

        Useful for generating curves from extracted real data parameters.

        Args:
            parameter_df: DataFrame with columns 'A', 'mu', 'lambda'
            noise_level: Noise level for all curves
            seed: Random seed

        Returns:
            List of GeneratedCurve objects
        """
        curves = []
        for i, row in parameter_df.iterrows():
            curves.append(self.generate_single_curve(
                A=row['A'],
                mu=row['mu'],
                lambda_=row.get('lambda', row.get('lambda_', 3.0)),
                noise_level=noise_level,
                seed=curve_seed(seed, 'batch', int(i)),
            ))
        return curves


# =============================================================================
# Helpers
# =============================================================================

def _apply_exponential_death(clean: np.ndarray, time: np.ndarray,
                             t_death: float, k_death: float) -> np.ndarray:
    """The historic exponential decline after ``t_death`` (unchanged shape)."""
    out = np.array(clean, dtype=float, copy=True)
    idx = int(np.searchsorted(time, t_death))
    if idx < time.size:
        od_at_death = out[idx]
        mask = time > t_death
        out[mask] = od_at_death * np.exp(-k_death * (time[mask] - t_death))
    return out


#: Below this total sum of squares the clean curve carries no variation at all
#: and R^2 is undefined rather than large and negative.
_SS_TOT_FLOOR = 1e-12


def _fit_free_metrics(od: np.ndarray, clean: np.ndarray,
                      has_clean: bool) -> Tuple[float, float]:
    """``(rmse, r_squared)`` of the emitted curve against its clean reference.

    NaN for both where there is no clean reference, and NaN for R^2 where the
    clean curve is constant (a flat control has zero total sum of squares, so
    R^2 is undefined - not 0.0, and certainly not -1.0).
    """
    if not has_clean or od.shape != clean.shape or od.size == 0:
        return float('nan'), float('nan')
    residual = od - clean
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((clean - np.mean(clean)) ** 2))
    # A flat control's total sum of squares is zero up to the rounding of its
    # own mean (~1e-31 over 400 points), which would report R^2 = -6e30 rather
    # than "undefined".  Anything with real amplitude is many orders above.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _SS_TOT_FLOOR else float('nan')
    return rmse, float(r2)


def _phase1_matches(shape_meta: Dict[str, Any], A: float, mu: float,
                    lambda_: float, rel_tol: float = 0.02) -> bool:
    """Did a shape build the first phase it was handed? (see ``shape_solved``)"""
    try:
        A1 = float(shape_meta['A1'])
        mu1 = float(shape_meta['mu1'])
        lam1 = float(shape_meta['lambda1'])
    except (KeyError, TypeError, ValueError):              # pragma: no cover
        return False
    if not all(np.isfinite([A1, mu1, lam1])):              # pragma: no cover
        return False
    return (abs(A1 - A) <= rel_tol * abs(A)
            and abs(mu1 - mu) <= rel_tol * abs(mu)
            and abs(lam1 - lambda_) <= max(rel_tol * abs(lambda_), 0.05))


def _has_artefacts(metadata: Dict[str, Any]) -> bool:
    """True when a scenario artefact (spike or scenario drift) was applied."""
    idx = metadata.get('spike_indices', ())
    if len(np.asarray(idx, dtype=int).ravel()):
        return True
    return bool(metadata.get('scenario_drift_applied', False))


def _wants_replicates(params: Dict[str, Any]) -> bool:
    """Does this scenario declare replicate structure? (0 everywhere = one well)"""
    return (float(_get(params, 'replicate_A_cv', 0.0)) > 0
            or float(_get(params, 'replicate_mu_cv', 0.0)) > 0
            or float(_get(params, 'replicate_lambda_sd', 0.0)) > 0
            or float(_get(params, 'outlier_well_prob', 0.0)) > 0
            or int(_get(params, 'n_replicates', 1)) > 1)


def _wants_twin(params: Dict[str, Any]) -> bool:
    """Artefact scenarios get an artefact-free twin unless told otherwise."""
    has_artefact = (int(_get(params, 'n_spikes', 0)) > 0
                    or _get(params, 'drift_od_per_h', None) is not None)
    return bool(_get(params, 'generate_twin', has_artefact))


def curve_to_row(curve: GeneratedCurve, curve_id: int) -> Dict[str, Any]:
    """
    One generated curve as a full v2 ground-truth row (plus its series).

    Every column of :data:`truth.GROUND_TRUTH_COLUMNS` is present, so
    ``create_ground_truth_csv`` never has to derive or fill anything - which is
    the point of ``require_complete=True``.
    """
    row = truth_module.empty_row()
    row.update({k: v for k, v in curve.truth.items() if k in row})
    row['curve_id'] = int(curve_id)
    row['strain_name'] = truth_module.strain_name(curve_id)
    row['time'] = np.asarray(curve.time, dtype=float).tolist()
    row['od600'] = np.asarray(curve.od600, dtype=float).tolist()
    return row


def curve_rows(curves: Sequence[GeneratedCurve],
               start_id: int = 0) -> List[Dict[str, Any]]:
    """
    A batch of generated curves as full v2 ground-truth rows.

    ``curve_id`` runs ``start_id, start_id+1, ...`` in the order the generator
    yielded the curves, and an artefact twin (``metadata['is_twin']``) is
    cross-linked with the curve it follows through ``twin_curve_id``.  This is
    the one place that mapping lives, so every caller - the comprehensive set
    and the ``generate_synthetic.py`` CLI alike - emits the same schema and
    ``create_ground_truth_csv`` never has to reconstruct a measurement.
    """
    ids = [start_id + offset for offset in range(len(curves))]

    twin_partner: Dict[int, int] = {}
    for pos, curve in enumerate(curves):
        if pos and curve.metadata.get('is_twin'):
            twin_partner[ids[pos]] = ids[pos - 1]
            twin_partner[ids[pos - 1]] = ids[pos]

    rows: List[Dict[str, Any]] = []
    for pos, curve in enumerate(curves):
        row = curve_to_row(curve, ids[pos])
        partner = twin_partner.get(ids[pos])
        if partner is not None:
            row['twin_curve_id'] = truth_module.strain_name(partner)
        rows.append(row)
    return rows


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_quick_test_set(n_per_category: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small test set for quick validation.

    Args:
        n_per_category: Curves per scenario category
        seed: Random seed

    Returns:
        DataFrame with test curves
    """
    generator = SyntheticGrowthCurveGenerator(seed=seed)

    config = {
        'standard': n_per_category,
        'flat_no_growth': n_per_category,
        'high_noise': n_per_category,
        'death_phase_moderate': n_per_category,
        'borderline_r2_good': n_per_category,
        'borderline_r2_bad': n_per_category,
    }

    return generator.generate_comprehensive_test_set(config=config, seed=seed)


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":                                    # pragma: no cover
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Testing SyntheticGrowthCurveGenerator...")

    generator = SyntheticGrowthCurveGenerator(seed=42)

    print("\n1. Single curve generation:")
    curve = generator.generate_single_curve(A=1.5, mu=0.2, lambda_=3.0, seed=1)
    print(f"   {len(curve.time)} points, max OD {curve.metadata['max_od']:.3f}, "
          f"R2 {curve.metadata['actual_r_squared']:.4f}")
    print(f"   requested mu {curve.truth['requested_mu']:.4f} -> "
          f"true mu {curve.truth['true_mu']:.4f} "
          f"(ratio {curve.truth['mu_realisation_ratio']:.3f})")

    print("\n2. Scenario generation:")
    for scenario in ('standard', 'flat_no_growth', 'death_phase_moderate'):
        curves = generator.generate_from_scenario(scenario, n_curves=3, seed=42)
        print(f"   {scenario}: {len(curves)} curves, "
              f"expected={curves[0].truth['expected_class']}")

    print("\n3. Quick test set generation:")
    test_df = generate_quick_test_set(n_per_category=3, seed=42)
    print(f"   {len(test_df)} curves over {test_df['scenario'].nunique()} scenarios")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, scenario in zip(axes.flat, [
            'standard', 'flat_no_growth', 'high_noise',
            'death_phase_moderate', 'truncation_challenge', 'borderline_r2_bad']):
        c = generator.generate_from_scenario(scenario, n_curves=1, seed=42)[0]
        ax.plot(c.time, c.od600, 'b-', alpha=0.7, label='Noisy')
        ax.plot(c.time, c.clean_od600, 'r--', alpha=0.5, label='Clean')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('OD600')
        ax.set_title(f"{scenario} ({c.truth['expected_class']})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_generator_demo.png', dpi=150)
    print("\n4. Saved demo plot to data_generator_demo.png")
