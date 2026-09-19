"""
Curve scenarios (WP5): the scenario catalog, the suite composition and the lab
prior.

What a scenario is
------------------
A :class:`ScenarioConfig` is a *generative* description of one population of
curves: parameter ranges, the shape family, the artefacts, the noise, the
replicate structure - plus how the row is to be LABELLED (``label_rule``) and
how it is to be SCORED (``scoring_mode``).  ``expected_class`` stays binary
GOOD/BAD, because three consumers coerce "not GOOD" to BAD; difficulty lives in
``scoring_mode``, never in a third class.

What changed in this rebuild
----------------------------
* **Every generative knob is a field.**  The old config could only vary A, mu,
  lambda, the noise preset and a ``pattern`` string, so "artefact" scenarios
  contained no artefact: ``drift_positive`` and ``outlier_contaminated``
  differed from each other in name, description and tags ONLY (M21).  They now
  carry real, separately calibrated artefacts (``drift_od_per_h``,
  ``n_spikes``/``spike_od``/``spike_window_h``).
* **Labels by rule where the label is a noise question.**  ``high_noise``,
  ``borderline_noise``, ``borderline_r2_bad``, ``borderline_r2_good``, the
  ``noise_ladder`` rungs and the three ``haldane_*`` scenarios declare
  ``label_rule='oracle'``: their expected class is decided by whether the
  parameters are recoverable from the curve that was written (delta_od >= 0.15
  and oracle A/mu error <= 20 %), not by a whole-trace R^2 against the clean
  curve, which moves when the plate runs longer (M02).  Every other scenario
  keeps ``label_rule='scenario'`` and its hand-declared class, and the
  generator asserts that nothing relabelled it.
* **The real failure modes.**  Incomplete runs (25 % of real growers, 0.4 % of
  the old suite - M07), clipped and over-subtracted baselines (30 % of real
  conditions, 0/480 - M09), slow post-primary drift (58 % of real growers -
  M08), replicate disagreement (40 % of Group 4 conditions - M10), a real
  contamination boundary (M15/M16) and a noise ladder whose rungs are defined
  by realised oracle error rather than by nominal R^2.
* **Composition.**  ``n_default`` lives on the scenario, so the suite's
  composition is part of its definition instead of a pair of flags in the
  generator script.  ``lab_band`` names one of the six measured delta-OD bands
  in ``config/lab_prior.json``; ``--weighting lab`` re-allocates the curve
  budget to the lab's own band frequencies (M11).  The DEFAULT (uniform)
  composition is itself tuned to the lab's prevalence: 646/1139 = 56.7 % GOOD
  at seed 42 (the run in MANIFEST.json is authoritative), against 71.9 % in the
  shipped benchmark and 49-60 % over the 92 real conditions.  It is reached by trimming the over-weighted full-culture
  growers (``high_A``, ``fast_growth``, ``short_lag``, ``long_lag``,
  ``pesticide_lb_typical``, the model-mismatch trio and the second copies of
  the incomplete family) and by raising the dead- and trace-band BAD scenarios
  and the two probes that sit ON a measured acceptance boundary
  (``contamination_offset`` 15 -> 35, ``clipped_baseline`` 15 -> 30) - never by
  deleting a hard scenario.
* **Declared is not realised.**  :func:`audit_realisation` measures the
  generated rows against the fields that asked for them, and
  ``generate_comprehensive.py`` refuses to write a suite that fails it.  Six
  declarations were inert in the first pass (the four ``noise_ladder`` rungs
  all ran at one preset, both replicate scenarios produced ordinary growers,
  and the spike windows were ignored), which is exactly the failure mode a
  config-only catalog has.

Every magnitude below cites the real condition it was measured on.  Nothing
here changes a pipeline threshold, and no scenario was deleted, softened or
lifted out of the headline: ``sparse_data``, ``fast_growth`` and ``low_A`` stay
exactly as hard as they were.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Range = Optional[Tuple[float, float]]

#: Where the derived lab prior lives (written by scripts/derive_lab_prior.py).
LAB_PRIOR_PATH = Path(__file__).resolve().parent.parent / 'config' / 'lab_prior.json'


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario.

    Every field added in this rebuild is defaulted so that a scenario which
    does not mention it behaves exactly as it did before.
    """

    name: str
    description: str
    expected_class: str  # 'GOOD' or 'BAD'

    # Gompertz parameters (can be fixed or range)
    A: Tuple[float, float] = (1.0, 1.5)
    mu: Tuple[float, float] = (0.1, 0.3)
    lambda_: Tuple[float, float] = (2.0, 8.0)

    # Initial OD (for contamination scenarios)
    initial_od: Tuple[float, float] = (0.0, 0.02)

    # Noise configuration
    noise_level: str = 'medium'  # 'very_low', 'low', 'medium', 'high', 'very_high'
    target_r_squared: Optional[Tuple[float, float]] = None  # Override noise to target R²

    # Death phase (the historic exponential decline; the calibrated decline
    # shapes are `shape='death_by_target_loss'` and friends)
    include_death_phase: bool = False
    t_death_fraction: Tuple[float, float] = (0.6, 0.8)  # Fraction of experiment
    k_death: Tuple[float, float] = (0.01, 0.05)

    # Experiment parameters
    duration_hours: float = 100.0
    time_resolution: float = 0.25

    # Special patterns
    pattern: Optional[str] = None  # 'flat', 'random_walk', 'diauxic', etc.

    # True when the scenario's difficulty comes from an INSTRUMENT artefact
    # rather than from biology.  This is no longer a documented no-op: it is
    # derived in __post_init__ from the artefact fields the generator actually
    # consumes - spikes, a signed drift, or a per-well sigma override - so it
    # selects exactly the scenarios whose READER output carries an artefact.
    # The blank offset is deliberately NOT one of them: an over-subtracted
    # blank is a preprocessing choice, not something the instrument did.
    instrument_noise: bool = False

    # Model to use for generation
    model: str = 'gompertz'

    # Additional metadata
    tags: List[str] = field(default_factory=list)

    # ---------------------------------------------------- Contract A (WP5)
    #: Curves generated for this scenario by the comprehensive suite.
    n_default: int = 15
    #: 'scenario' keeps expected_class; 'oracle' derives it from the truth
    #: curve's own recoverability (truth.apply_label_rule).
    label_rule: str = 'scenario'
    #: Scoring policy: 'standard', 'a_not_identifiable', 'preprocessing_blind',
    #: 'model_mismatch'.
    scoring_mode: str = 'standard'
    #: Name of a band in config/lab_prior.json (used by --weighting lab).
    lab_band: Optional[str] = None
    #: Name in curve_shapes.SHAPES; None keeps the plain model branch.
    shape: Optional[str] = None

    # Second-phase parameters, passed to curve_shapes as RANGES (the shape
    # library draws them on the curve's own stream).
    A2: Range = None
    mu2: Range = None
    lambda2: Range = None
    gap_hours: Range = None

    # Instrument artefacts
    drift_od_per_h: Range = None
    n_spikes: Union[int, Tuple[int, int]] = 0
    spike_od: Tuple[float, float] = (0.15, 0.40)
    spike_decay_points: int = 3
    #: Time window (hours) the spikes are placed in; None spreads them over the
    #: whole trace after the first 5 % of points.
    spike_window_h: Range = None

    # Blank stage: offset of the SUBTRACTED blank above the well's own true
    # baseline.  A positive offset over-subtracts and the trace clips at zero.
    # ``None`` takes the generator's calibrated default mixture; :data:`NO_BLANK`
    # subtracts no blank at all (for a scenario whose ``initial_od`` already IS
    # a post-blank baseline).
    blank_offset_od: Union[Tuple[float, float], str, None] = None

    # Replicate structure (0 everywhere = a single well)
    replicate_A_cv: float = 0.0
    replicate_mu_cv: float = 0.0
    replicate_lambda_sd: float = 0.0
    outlier_well_prob: float = 0.0
    #: How many of the wells are outliers, and by what factor their A is
    #: scaled.  ``None`` keeps the generator's calibrated default
    #: (1 well at 0.50-0.70 of its siblings, G3 LCY01).
    n_outlier_wells: Optional[int] = None
    outlier_well_A_scale: Range = None
    #: Wells per condition; None = 3 when any replicate field is set, else 1.
    n_replicates: Optional[int] = None

    # Haldane kinetics
    Ks: Optional[float] = None
    Ki: Optional[float] = None
    q: Optional[float] = None
    S0: Optional[float] = None
    #: Richards shape parameter (scalar or range).
    nu: Union[float, Tuple[float, float], None] = None

    # Decline shapes
    death_loss_frac: Range = None
    t_death_offset_h: Range = None

    # ------------------------------------------------------------- extras
    #: Per-well high-frequency sigma (OD) overriding the preset's
    #: baseline/proportional sigma.  Used by the noise ladder, whose rungs are
    #: defined by realised oracle error, not by a nominal R^2.
    well_sigma_od: Range = None
    #: Solve mu so the curve reaches this fraction of A at its LAST point
    #: (see :meth:`ScenarioSampler.sample_parameters`).  This is how the
    #: incomplete-run family is specified: "the run ends at 60-85 % of A" is
    #: the property, and mu is whatever realises it.
    end_fraction_of_A: Range = None
    #: 't99_plus_buffer' sets duration per curve to the time the curve reaches
    #: 99 % of A plus ``plateau_buffer_points`` samples.
    duration_rule: Optional[str] = None
    plateau_buffer_points: int = 4
    #: Contamination as an OFFSET FRACTION of the final OD:
    #: ``initial_od = frac * A / (1 - frac)``.  The measured acceptance
    #: boundary is frac ~0.11-0.16, not ~40 %.
    initial_od_offset_fraction: Range = None
    #: Media token handed to noise_models.BlankModel.
    media: str = 'LB'
    #: Force (or suppress) the artefact-free twin.  None = the generator's
    #: default (a twin exactly when an artefact is declared).
    generate_twin: Optional[bool] = None

    def __post_init__(self) -> None:
        declared = bool(
            self.n_spikes if isinstance(self.n_spikes, int) else max(self.n_spikes)
        ) or self.drift_od_per_h is not None \
            or self.well_sigma_od is not None
        # `instrument_noise` is now derived, not declared: it is True exactly
        # when the scenario carries an instrument artefact.
        if declared:
            self.instrument_noise = True
        if self.label_rule not in ('scenario', 'oracle'):
            raise ValueError(f"{self.name}: unknown label_rule {self.label_rule!r}")
        if self.scoring_mode not in (
                'standard', 'a_not_identifiable', 'preprocessing_blind',
                'model_mismatch'):
            raise ValueError(f"{self.name}: unknown scoring_mode {self.scoring_mode!r}")
        if self.expected_class not in ('GOOD', 'BAD'):
            raise ValueError(f"{self.name}: expected_class must be GOOD or BAD")

    # ------------------------------------------------------------- helpers
    def generative_fingerprint(self) -> Tuple[Any, ...]:
        """Everything that affects the curves this scenario writes.

        Name, description and tags are excluded on purpose: two scenarios that
        agree on this tuple generate the same population and one of them is
        redundant (the M21 case, where ``drift_positive`` and
        ``outlier_contaminated`` were identical except for their names).
        """
        skip = {'name', 'description', 'tags', 'n_default', 'lab_band'}
        out = []
        for f in fields(self):
            if f.name in skip:
                continue
            value = getattr(self, f.name)
            if isinstance(value, list):
                value = tuple(value)
            out.append((f.name, value))
        return tuple(out)


# =============================================================================
# Good Growth Scenarios (Should classify as GOOD)
# =============================================================================

#: Baseline OD of a real growing well after blank subtraction: the IQR over the
#: 48 real growers is 0.019-0.066 OD with a maximum of 0.144, so a GOOD curve
#: starting at 0.000-0.020 (the old range for every GOOD scenario) made
#: ``initial_od`` a perfect shortcut for the label - every GOOD scenario was
#: below 0.020 and ``contamination`` was 0.238-0.483 (M15).
GROWER_INITIAL_OD = (0.02, 0.10)

#: Blank offset for the wells that go through the blank stage.  The subtracted
#: blank's mean sits this far above the well's own true baseline; positive
#: over-subtracts and the trace clips at zero.  The band is the measured
#: per-media across-well b0 spread of the 18 Group 3 blank wells (0.003-0.029
#: OD).  This is what puts the real 30.4 % clipped-baseline population into the
#: suite (0/480 before).
BLANK_OFFSET_OD = (-0.015, 0.015)

#: ``blank_offset_od`` sentinel meaning "subtract no blank at all"; it is
#: ``data_generator.BLANK_OFF`` (pinned by test_scenarios.py, and spelled as a
#: literal here because data_generator imports this module).  A scenario needs
#: it when its ``initial_od`` IS the post-blank baseline the scenario exists to
#: reproduce: the two dead-well drift scenarios are calibrated to G3 H2O_IMID3,
#: whose real blank-subtracted trace sits at 0.0431-0.0293 OD and contains no
#: zeros at all, so re-referencing them to a fresh blank clips the artefact
#: away (the -0.014 OD decline and the +0.08 OD ramp both vanish into the zero
#: floor).
NO_BLANK = 'none'

GOOD_GROWTH_SCENARIOS = {
    'standard': ScenarioConfig(
        name='standard',
        description='Typical good growth curve (LB media)',
        expected_class='GOOD',
        A=(0.8, 1.5),
        mu=(0.1, 0.3),
        lambda_=(2.0, 8.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=10,
        lab_band='full',
        tags=['typical', 'LB']
    ),

    'high_A': ScenarioConfig(
        name='high_A',
        description='Very high final OD (dense culture)',
        expected_class='GOOD',
        A=(1.5, 2.0),
        mu=(0.15, 0.35),
        lambda_=(1.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['high_density']
    ),

    'low_A': ScenarioConfig(
        name='low_A',
        description='Low but valid growth (still above thresholds)',
        expected_class='GOOD',
        # 22.8 % of audited real GOOD conditions have A < 0.5, so the band is
        # widened upward, not softened: the 0.15-0.35 draws that straddle the
        # 0.15 OD rise gate are still in it.
        A=(0.15, 0.5),
        mu=(0.05, 0.15),
        lambda_=(3.0, 10.0),
        noise_level='low',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='marginal',
        tags=['low_density', 'borderline']
    ),

    'fast_growth': ScenarioConfig(
        name='fast_growth',
        description='Rapid exponential phase (high mu)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.4, 0.75),
        lambda_=(0.5, 2.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['fast', 'high_growth_rate']
    ),

    'slow_growth': ScenarioConfig(
        name='slow_growth',
        description='Slow exponential phase (low mu)',
        expected_class='GOOD',
        A=(0.8, 1.2),
        mu=(0.02, 0.08),
        lambda_=(5.0, 15.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='low',
        blank_offset_od=BLANK_OFFSET_OD,
        duration_hours=120.0,  # Needs longer experiment
        lab_band='moderate',
        n_default=10,
        tags=['slow', 'low_growth_rate']
    ),

    'short_lag': ScenarioConfig(
        name='short_lag',
        description='Very short lag phase (pre-adapted cells)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        # 19.3 % of audited real GOOD conditions fit lambda < 0.5 h.
        lambda_=(0.0, 0.5),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['short_lag', 'pre_adapted']
    ),

    'long_lag': ScenarioConfig(
        name='long_lag',
        description='Extended lag phase (stressed cells)',
        expected_class='GOOD',
        A=(1.0, 1.4),
        mu=(0.1, 0.2),
        # Real maximum fitted lag is 76.7 h (10.5 % of audited GOOD are above
        # 20 h); the old (20, 50) never reached it.
        lambda_=(20.0, 77.0),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        duration_hours=120.0,
        lab_band='full',
        n_default=10,
        tags=['long_lag', 'stressed']
    ),

    'very_clean': ScenarioConfig(
        name='very_clean',
        description='Very clean data with minimal noise (R² > 0.99)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        noise_level='very_low',
        n_default=10,
        lab_band='full',
        tags=['clean', 'high_quality']
    ),

    'pesticide_lb_typical': ScenarioConfig(
        name='pesticide_lb_typical',
        description='Typical pesticide+LB growth (slightly lower A)',
        expected_class='GOOD',
        A=(1.2, 1.7),
        mu=(0.12, 0.22),
        lambda_=(2.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        media='PESTICIDEANDLB',
        lab_band='full',
        n_default=10,
        tags=['pesticide', 'treatment']
    ),

    # ---------------------------------------------------------- new (WP5)
    'high_baseline_good': ScenarioConfig(
        name='high_baseline_good',
        description='Good growth from a high but legitimate baseline (0.08-0.15 OD)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.12, 0.25),
        lambda_=(1.0, 5.0),
        # The real grower baseline distribution has a maximum of 0.144 OD, so
        # a high starting OD is not by itself contamination.  Without this
        # scenario `initial_od` alone separates GOOD from BAD perfectly.
        initial_od=(0.08, 0.15),
        noise_level='medium',
        n_default=10,
        lab_band='full',
        tags=['baseline', 'shortcut_defeat']
    ),
}


# =============================================================================
# Bad Curve Scenarios (Should classify as BAD)
# =============================================================================

BAD_CURVE_SCENARIOS = {
    'flat_no_growth': ScenarioConfig(
        name='flat_no_growth',
        description='H2O control - no growth at all',
        expected_class='BAD',
        A=(0.0, 0.0),  # No growth
        mu=(0.0, 0.0),
        lambda_=(0.0, 0.0),
        initial_od=(0.0, 0.02),
        noise_level='very_low',
        pattern='flat',
        blank_offset_od=BLANK_OFFSET_OD,
        # 31.5 % of real conditions are dead wells; they were 2.5 % of the old
        # suite, which is the single largest composition error (M11).
        n_default=45,
        media='H2O',
        lab_band='dead',
        tags=['negative_control', 'H2O', 'no_growth']
    ),

    'minimal_growth': ScenarioConfig(
        name='minimal_growth',
        description='Barely detectable growth (delta OD < 0.1)',
        expected_class='BAD',
        A=(0.03, 0.08),
        mu=(0.005, 0.02),
        lambda_=(5.0, 20.0),
        noise_level='low',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=40,
        lab_band='trace',
        tags=['minimal', 'borderline_bad']
    ),

    'contamination': ScenarioConfig(
        name='contamination',
        description='High initial OD indicating contamination (offset fraction 0.34-0.60)',
        expected_class='BAD',
        A=(0.3, 0.6),
        mu=(0.05, 0.15),
        lambda_=(0.0, 1.0),
        initial_od=(0.2, 0.5),  # High starting OD
        noise_level='medium',
        lab_band='marginal',
        n_default=25,
        tags=['contamination', 'high_initial_od']
    ),

    'high_noise': ScenarioConfig(
        name='high_noise',
        description='Valid growth pattern under the noisiest real well (label by oracle rule)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='very_high',
        target_r_squared=(0.70, 0.90),
        # A whole-trace R^2 is not a label: the SAME curve at the SAME sigma
        # scores 0.977 at 20 h and 0.880 at 200 h.  The class now comes from
        # whether A and mu are recoverable from the curve that was written.
        label_rule='oracle',
        lab_band='full',
        tags=['noisy', 'poor_fit', 'oracle_labelled']
    ),

    'borderline_noise': ScenarioConfig(
        name='borderline_noise',
        description='Growth with borderline noise (label by oracle rule)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        target_r_squared=(0.90, 0.94),
        label_rule='oracle',
        lab_band='full',
        tags=['noisy', 'borderline', 'oracle_labelled']
    ),

    'erratic': ScenarioConfig(
        name='erratic',
        description='Non-biological erratic pattern',
        expected_class='BAD',
        pattern='random_walk',
        initial_od=(0.05, 0.15),
        noise_level='high',
        lab_band='trace',
        n_default=25,
        tags=['erratic', 'non_biological']
    ),

    'pesticide_only': ScenarioConfig(
        name='pesticide_only',
        description='Pesticide without nutrients - no growth',
        expected_class='BAD',
        A=(0.02, 0.08),
        mu=(0.001, 0.01),
        lambda_=(10.0, 50.0),
        noise_level='low',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=35,
        media='PESTICIDE',
        lab_band='trace',
        tags=['pesticide', 'no_nutrients', 'inhibited']
    ),

    'fit_failure': ScenarioConfig(
        name='fit_failure',
        description='Curve that causes fit convergence failure',
        expected_class='BAD',
        A=(0.01, 0.02),  # Very low signal
        mu=(0.001, 0.005),
        lambda_=(0.0, 1.0),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=35,
        lab_band='dead',
        tags=['fit_failure', 'low_signal']
    ),

    # ---------------------------------------------------------- new (WP5)
    'contamination_offset': ScenarioConfig(
        name='contamination_offset',
        description='Contamination at the measured acceptance boundary (offset fraction 0.10-0.20)',
        expected_class='BAD',
        A=(0.9, 1.5),
        mu=(0.10, 0.25),
        lambda_=(0.0, 2.0),
        # The probe that matters: at A=1.2 the pipeline says GOOD for 10/10 at
        # offset fraction 0.08, 9/10 at 0.11, 3/10 at 0.14 and 0/10 at 0.20.
        # `contamination` sits at 0.34-0.60 and is already rejected 15/15, so
        # it measures nothing about where the boundary is.
        initial_od_offset_fraction=(0.10, 0.20),
        noise_level='medium',
        lab_band='full',
        n_default=35,
        tags=['contamination', 'boundary', 'high_initial_od']
    ),

    'clipped_baseline': ScenarioConfig(
        name='clipped_baseline',
        description='Dead well whose blank is over-subtracted: the opening segment is clipped at zero',
        expected_class='BAD',
        A=(0.0, 0.03),
        mu=(0.0, 0.005),
        lambda_=(0.0, 5.0),
        initial_od=(0.0, 0.02),
        noise_level='low',
        # Over-subtraction only: 28/92 real conditions have a zero-variance
        # opening segment, which is the branch where the pipeline substitutes
        # its 0.002 noise floor for the baseline sigma.
        blank_offset_od=(0.010, 0.030),
        media='H2O',
        lab_band='dead',
        n_default=30,
        tags=['clipped', 'blank', 'no_growth']
    ),

    'all_zero': ScenarioConfig(
        name='all_zero',
        description='Identically zero trace (12/92 real conditions)',
        expected_class='BAD',
        A=(0.0, 0.0),
        mu=(0.0, 0.0),
        lambda_=(0.0, 0.0),
        initial_od=(0.0, 0.0),
        noise_level='very_low',
        pattern='flat',
        # A blank well subtracted from an emptier well: the whole trace clips.
        blank_offset_od=(0.05, 0.10),
        n_default=18,
        media='H2O',
        lab_band='dead',
        tags=['all_zero', 'clipped', 'negative_control']
    ),

    'negative_drift': ScenarioConfig(
        name='negative_drift',
        description='Dead well drifting DOWN (G3 H2O_IMID3: 0.0431 -> 0.0293 over 99 h)',
        expected_class='BAD',
        A=(0.0, 0.005),
        mu=(0.0, 0.001),
        lambda_=(0.0, 1.0),
        initial_od=(0.035, 0.050),
        noise_level='low',
        # G3 H2O_IMID3's trace is already blank-subtracted and never touches
        # zero, so no second blank is taken off it.
        blank_offset_od=NO_BLANK,
        # (0.0293 - 0.0431)/99 h = -0.00014 OD/h.
        drift_od_per_h=(-0.00016, -0.00012),
        media='H2O',
        lab_band='dead',
        n_default=20,
        tags=['drift', 'artifact', 'no_growth']
    ),

    'replicate_outlier_high': ScenarioConfig(
        name='replicate_outlier_high',
        description='One well grows, two are dead (G4 PERMETHRINANDLB-CISPERM1 = [0.016, 1.063, 0.010])',
        expected_class='BAD',
        A=(0.9, 1.6),
        mu=(0.10, 0.20),
        lambda_=(2.0, 6.0),
        initial_od=(0.0, 0.02),
        noise_level='medium',
        n_replicates=3,
        outlier_well_prob=1.0,
        # TWO of the three wells are dead, reproducing [0.016, 1.063, 0.010]:
        # the condition MEAN is what the pipeline sees and it looks like a
        # modest grower at a third of the real amplitude.  40 % of Group 4 and
        # 29 % of Group 3 conditions have a relative well spread above 0.3.
        n_outlier_wells=2,
        outlier_well_A_scale=(0.005, 0.020),
        replicate_A_cv=0.05,
        replicate_mu_cv=0.10,
        replicate_lambda_sd=0.5,
        # The benchmark hands the pipeline the averaged condition, so the
        # disagreement between the wells is information it never sees.
        scoring_mode='preprocessing_blind',
        lab_band='marginal',
        n_default=20,
        tags=['replicates', 'outlier_well', 'preprocessing']
    ),

    'replicate_dropout': ScenarioConfig(
        name='replicate_dropout',
        description='One of three wells never inoculated: the condition mean is 2/3 of the truth',
        expected_class='BAD',
        A=(1.0, 1.6),
        mu=(0.12, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        n_replicates=3,
        outlier_well_prob=1.0,
        n_outlier_wells=1,
        outlier_well_A_scale=(0.0, 0.05),
        replicate_A_cv=0.05,
        replicate_mu_cv=0.10,
        replicate_lambda_sd=0.5,
        n_default=25,
        scoring_mode='preprocessing_blind',
        lab_band='full',
        tags=['replicates', 'dropout', 'preprocessing']
    ),

    'contaminated_blank': ScenarioConfig(
        name='contaminated_blank',
        description='The blank well itself grows, so every condition on the plate sags after subtraction',
        expected_class='BAD',
        A=(0.6, 1.0),
        mu=(0.10, 0.20),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        # A blank growing at 0.2-0.4 OD over 100 h subtracts a rising ramp from
        # every well: the condition loses amplitude as the run goes on and the
        # early points clip.
        drift_od_per_h=(-0.004, -0.002),
        blank_offset_od=(0.005, 0.020),
        n_replicates=3,
        replicate_A_cv=0.05,
        replicate_mu_cv=0.10,
        replicate_lambda_sd=0.5,
        scoring_mode='preprocessing_blind',
        # NO artefact-free twin.  Every other artefact scenario is BAD because
        # of what the artefact does to a well that was already dead
        # (evaporation_drift and negative_drift twins have delta_od 0.009-0.012
        # and are honestly BAD without it), so their twins keep the scenario's
        # label truthfully.  Here the base curve is a healthy grower and the
        # contamination is entirely in the blank that was subtracted: an
        # artefact-free twin of it IS a good curve, and shipping it with this
        # scenario's BAD label put 15 rows in the benchmark whose truth curve
        # does not support their label.  The twin is suppressed rather than
        # relabelled, because a row's class must come from its own curve.
        generate_twin=False,
        lab_band='moderate',
        n_default=30,
        tags=['blank', 'contamination', 'preprocessing', 'artifact']
    ),
}


# =============================================================================
# Edge Case Scenarios (Test boundary conditions)
# =============================================================================

EDGE_CASE_SCENARIOS = {
    'borderline_r2_good': ScenarioConfig(
        name='borderline_r2_good',
        description='Noise just below the R² 0.95 threshold (label by oracle rule)',
        expected_class='GOOD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        target_r_squared=(0.95, 0.97),
        label_rule='oracle',
        lab_band='full',
        tags=['borderline', 'threshold_test', 'oracle_labelled']
    ),

    'borderline_r2_bad': ScenarioConfig(
        name='borderline_r2_bad',
        description='Noise just above the R² 0.95 threshold (label by oracle rule)',
        expected_class='BAD',
        A=(0.8, 1.2),
        mu=(0.1, 0.2),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        target_r_squared=(0.93, 0.949),
        label_rule='oracle',
        lab_band='full',
        tags=['borderline', 'threshold_test', 'oracle_labelled']
    ),

    'truncation_challenge': ScenarioConfig(
        name='truncation_challenge',
        description='Diauxic-like second phase (sampled, not the old hard-coded one)',
        expected_class='GOOD',  # Should still fit the first phase
        A=(1.0, 1.3),
        mu=(0.15, 0.25),
        lambda_=(2.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        # Was `pattern='diauxic'` with A2=0.5, mu2=0.15, gap=10 hard-coded and
        # IDENTICAL on all 15 curves, and a second phase whose maximum slope
        # was at its own onset.  Now a real lag/exponential/plateau phase with
        # sampled amplitude and timing.
        shape='two_phase_gompertz',
        A2=(0.30, 0.70),
        gap_hours=(5.0, 15.0),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=12,
        tags=['truncation', 'diauxic', 'multi_peak']
    ),

    'death_phase_moderate': ScenarioConfig(
        name='death_phase_moderate',
        description='Moderate decline after the plateau (5-20 % of peak, anchored to t95)',
        expected_class='GOOD',  # Should truncate before death
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        # The old exponential decline realised a median loss of 0.72 of the
        # peak and started 25-35 h after the fit window, so the "death phase"
        # test never put a declining point near the fit.  Real end-of-run loss
        # over 48 growers is median 0.005, p90 0.186, max 0.372.
        shape='death_by_target_loss',
        death_loss_frac=(0.05, 0.20),
        t_death_offset_h=(2.0, 8.0),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=12,
        tags=['death_phase', 'decline']
    ),

    'death_phase_severe': ScenarioConfig(
        name='death_phase_severe',
        description='Severe decline (25-45 % of peak) beginning right after t95',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 4.0),
        initial_od=GROWER_INITIAL_OD,
        shape='death_by_target_loss',
        # 0.45 is the real ceiling: the worst measured loss is 0.372
        # (G4 LB_PERM4).
        death_loss_frac=(0.25, 0.45),
        t_death_offset_h=(0.5, 3.0),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=12,
        tags=['death_phase', 'severe_decline']
    ),

    'short_experiment': ScenarioConfig(
        name='short_experiment',
        description='Very short experiment (20 hours)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.2, 0.4),  # Fast growth to reach plateau
        lambda_=(1.0, 3.0),
        initial_od=GROWER_INITIAL_OD,
        duration_hours=20.0,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['short_duration', 'limited_data']
    ),

    'long_experiment': ScenarioConfig(
        name='long_experiment',
        description='Extended experiment (150 hours)',
        expected_class='GOOD',
        A=(1.0, 1.4),
        mu=(0.08, 0.15),
        lambda_=(5.0, 15.0),
        initial_od=GROWER_INITIAL_OD,
        duration_hours=150.0,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['long_duration', 'extended']
    ),

    'sparse_data': ScenarioConfig(
        name='sparse_data',
        description='Sparse time resolution (1 hour intervals)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        time_resolution=1.0,  # 1 hour instead of 0.25
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        # Unchanged on purpose: 10 of its curves miss the 20-point minimum
        # after truncation and it stays exactly that hard, in the headline.
        lab_band='full',
        tags=['sparse', 'low_resolution']
    ),

    'dense_data': ScenarioConfig(
        name='dense_data',
        description='Dense time resolution (5 min intervals)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        time_resolution=0.083,  # 5 minutes
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=10,
        tags=['dense', 'high_resolution']
    ),

    'baranyi_generated': ScenarioConfig(
        name='baranyi_generated',
        description='Baranyi model data fitted with Gompertz',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        model='baranyi',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        scoring_mode='model_mismatch',
        lab_band='full',
        n_default=12,
        tags=['model_mismatch', 'robustness']
    ),

    'logistic_generated': ScenarioConfig(
        name='logistic_generated',
        description='Logistic model data fitted with Gompertz',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        model='logistic',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        scoring_mode='model_mismatch',
        lab_band='full',
        n_default=12,
        tags=['model_mismatch', 'robustness']
    ),

    'richards_asymmetric': ScenarioConfig(
        name='richards_asymmetric',
        description='Highly asymmetric Richards curve',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        model='richards',
        nu=(0.3, 0.8),
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        scoring_mode='model_mismatch',
        lab_band='full',
        n_default=12,
        tags=['model_mismatch', 'asymmetric']
    ),

    'outlier_contaminated': ScenarioConfig(
        name='outlier_contaminated',
        description='Good curve with 3-8 real outlier spikes inside the growth window',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        # Was generatively IDENTICAL to drift_positive and contained 0.00
        # points beyond 5 robust MADs per curve - fewer than 'standard'.
        # Spike heights are the measured bubble range (per-well largest
        # positive excursion: typical +0.0058, p90 +0.0211, max +0.295 OD in
        # Group 3 H12; the visible bubbles the lab discards are 0.2-1.0 OD).
        n_spikes=(3, 8),
        spike_od=(0.15, 0.40),
        spike_window_h=(0.0, 25.0),
        n_default=10,
        lab_band='full',
        tags=['outliers', 'robust_fit', 'artifact']
    ),

    'drift_positive': ScenarioConfig(
        name='drift_positive',
        description='Good curve with positive baseline drift (0.0008-0.002 OD/h)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        # G3 H2O_IMID2 gains +0.194 OD over 99 h = 0.00196 OD/h.
        drift_od_per_h=(0.0008, 0.0020),
        n_default=10,
        lab_band='full',
        tags=['drift', 'systematic_error', 'artifact']
    ),

    'borderline_delta_od': ScenarioConfig(
        name='borderline_delta_od',
        description='Delta OD just above the 0.15 OD gate (0.16-0.24 realised)',
        expected_class='GOOD',
        # The shipped curves had true_A 0.320-0.380 against a config of
        # (0.17, 0.23) - a +0.150 offset on all 15 rows put the "borderline"
        # probe at 2.1-2.5x the gate, where it scored a meaningless 15/15.
        # The initial_od is kept at 0 so the realised rise IS the amplitude.
        A=(0.16, 0.24),
        mu=(0.08, 0.12),
        lambda_=(3.0, 8.0),
        initial_od=(0.0, 0.0),
        noise_level='low',
        lab_band='marginal',
        tags=['borderline', 'low_signal']
    ),

    'haldane_mild_inhibition': ScenarioConfig(
        name='haldane_mild_inhibition',
        description='Haldane, mild substrate inhibition (Ki=10 >> Ks=0.1, S0=1.0)',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.30),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        model='haldane',
        Ks=0.1, Ki=10.0, q=0.1, S0=1.0,
        noise_level='medium',
        label_rule='oracle',
        scoring_mode='model_mismatch',
        lab_band='full',
        n_default=12,
        tags=['haldane', 'inhibition', 'mild', 'oracle_labelled']
    ),

    'haldane_strong_inhibition': ScenarioConfig(
        name='haldane_strong_inhibition',
        description='Haldane, strong substrate inhibition (Ki=0.15 ~ Ks=0.1, S0=1.0)',
        expected_class='GOOD',
        A=(0.5, 1.0),
        mu=(0.10, 0.20),
        lambda_=(3.0, 8.0),
        initial_od=GROWER_INITIAL_OD,
        model='haldane',
        # All three scenarios used to get Ki=10, Ks=0.1, q=0.1, S0=1.0
        # regardless of what their descriptions claimed.
        Ks=0.1, Ki=0.15, q=0.1, S0=1.0,
        noise_level='medium',
        label_rule='oracle',
        scoring_mode='model_mismatch',
        lab_band='moderate',
        tags=['haldane', 'inhibition', 'strong', 'edge', 'oracle_labelled']
    ),

    'haldane_total_inhibition': ScenarioConfig(
        name='haldane_total_inhibition',
        description='Haldane at inhibitory substrate (Ki=0.02, S0=5.0): no growth phase',
        expected_class='BAD',
        A=(0.02, 0.05),
        mu=(0.01, 0.03),
        lambda_=(10.0, 50.0),
        model='haldane',
        Ks=0.1, Ki=0.02, q=0.1, S0=5.0,
        noise_level='low',
        label_rule='oracle',
        scoring_mode='model_mismatch',
        lab_band='dead',
        n_default=18,
        tags=['haldane', 'inhibition', 'total', 'no_growth', 'oracle_labelled']
    ),

    'evaporation_drift': ScenarioConfig(
        name='evaporation_drift',
        description='Dead edge well drifting up 0.0008-0.002 OD/h - the false positive its name promises',
        expected_class='BAD',
        # A dead well, not a growing one: the whole rise must come from the
        # drift, ending 0.08-0.20 OD above where it started.
        A=(0.0, 0.01),
        mu=(0.0, 0.002),
        lambda_=(0.0, 1.0),
        initial_od=(0.0, 0.02),
        model='gompertz',
        noise_level='low',
        # The whole rise is the evaporation ramp; an over-subtracted blank
        # would clip the first half of it into the zero floor and the scenario
        # would stop being the false positive its name promises.
        blank_offset_od=NO_BLANK,
        drift_od_per_h=(0.0008, 0.0020),
        media='H2O',
        lab_band='trace',
        n_default=20,
        tags=['artifact', 'evaporation', 'drift', 'edge_well']
    ),

    'bubble_spike': ScenarioConfig(
        name='bubble_spike',
        description='Growth curve with 1-2 large bubble artefacts (0.25-0.8 OD) in the exponential phase',
        expected_class='GOOD',
        A=(0.8, 1.5),
        mu=(0.15, 0.30),
        lambda_=(2.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        model='gompertz',
        noise_level='medium',
        n_spikes=(1, 2),
        spike_od=(0.25, 0.80),
        # Inside the exponential phase, where a spike can be mistaken for the
        # peak: lambda is 2-5 h and t95 is 10-18 h for these parameters.
        spike_window_h=(2.0, 15.0),
        lab_band='full',
        n_default=10,
        tags=['artifact', 'bubble', 'outlier', 'spike']
    ),

    # ---------------------------------------------------------- new (WP5)
    # ------------------------------------------------ the noise ladder
    # Four rungs whose difficulty is the REALISED oracle error, not a nominal
    # R^2.  Each curve is labelled by the oracle rule applied to its OWN
    # realised error, so a rung is a population, not a label.  The rungs
    # deliberately straddle - and avoid - the rule's own 20 % threshold:
    # targets are ~5 %, ~10 %, ~35 % and ~60 %.  Sigma is the per-well
    # high-frequency sigma; a scenario with three wells would see 0.848 of it
    # after averaging (the measured common-mode ratio), so these single-well
    # rungs ARE the sigma the fit sees.
    'noise_ladder_1': ScenarioConfig(
        name='noise_ladder_1',
        description='Noise ladder rung 1: per-well sigma 0.06 OD (target oracle error ~5 %)',
        expected_class='GOOD',
        A=(0.8, 1.5),
        mu=(0.10, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='very_high',
        well_sigma_od=(0.055, 0.065),
        label_rule='oracle',
        lab_band='full',
        tags=['noise_ladder', 'oracle_labelled']
    ),

    'noise_ladder_2': ScenarioConfig(
        name='noise_ladder_2',
        description='Noise ladder rung 2: per-well sigma 0.10 OD (target oracle error ~10 %)',
        expected_class='GOOD',
        A=(0.8, 1.5),
        mu=(0.10, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='very_high',
        well_sigma_od=(0.09, 0.11),
        label_rule='oracle',
        lab_band='full',
        tags=['noise_ladder', 'oracle_labelled']
    ),

    'noise_ladder_3': ScenarioConfig(
        name='noise_ladder_3',
        description='Noise ladder rung 3: per-well sigma 0.25 OD (target oracle error ~35 %, out of distribution)',
        expected_class='BAD',
        A=(0.8, 1.5),
        mu=(0.10, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='very_high',
        # 0.25 OD is ~10x the noisiest real well's plateau sigma (G3 H12,
        # 0.0255), so rungs 3 and 4 are out of the measured distribution and
        # are reported as such - they exist to place the rule's threshold
        # between two populations, not to claim realism.
        well_sigma_od=(0.24, 0.26),
        label_rule='oracle',
        lab_band='full',
        n_default=20,
        tags=['noise_ladder', 'oracle_labelled', 'out_of_distribution']
    ),

    'noise_ladder_4': ScenarioConfig(
        name='noise_ladder_4',
        description='Noise ladder rung 4: per-well sigma 0.35 OD (target oracle error ~60 %, out of distribution)',
        expected_class='BAD',
        A=(0.8, 1.5),
        mu=(0.10, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='very_high',
        well_sigma_od=(0.34, 0.36),
        label_rule='oracle',
        lab_band='full',
        n_default=20,
        tags=['noise_ladder', 'oracle_labelled', 'out_of_distribution']
    ),

    # --------------------------------------------- the incomplete family
    'incomplete_24h': ScenarioConfig(
        name='incomplete_24h',
        description='Group 1 geometry (24 h, 96 points): the run ends at 60-85 % of A',
        expected_class='GOOD',
        A=(0.8, 1.8),
        # mu is SOLVED from `end_fraction_of_A` (see ScenarioSampler): the
        # property being specified is "the run ends before the asymptote", and
        # mu is whatever realises it.  The realised band is ~0.03-0.11 OD/h.
        mu=(0.05, 0.20),
        lambda_=(2.0, 8.0),
        initial_od=GROWER_INITIAL_OD,
        end_fraction_of_A=(0.60, 0.85),
        duration_hours=24.0,
        time_resolution=0.25,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        # 24/92 real conditions are 96-point/23.75 h Group 1 runs and they
        # contain 6 of the 12 real incomplete curves.
        n_default=50,
        scoring_mode='a_not_identifiable',
        lab_band='moderate',
        tags=['incomplete', 'truncated_run', 'group1_geometry']
    ),

    'run_24h_plateaued': ScenarioConfig(
        name='run_24h_plateaued',
        description='Group 1 geometry that DOES plateau inside 24 h',
        expected_class='GOOD',
        A=(0.8, 1.8),
        mu=(0.25, 0.45),
        lambda_=(1.0, 3.0),
        initial_od=GROWER_INITIAL_OD,
        duration_hours=24.0,
        time_resolution=0.25,
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        # Without these the 24 h geometry would predict the label by itself.
        n_default=10,
        lab_band='full',
        tags=['truncated_run', 'group1_geometry', 'shortcut_defeat']
    ),

    'incomplete_growth_richards': ScenarioConfig(
        name='incomplete_growth_richards',
        description='100 h run still rising at the end, Richards-generated so A is genuinely non-identifiable',
        expected_class='GOOD',
        A=(1.3, 1.9),
        mu=(0.015, 0.035),
        lambda_=(2.0, 8.0),
        initial_od=GROWER_INITIAL_OD,
        end_fraction_of_A=(0.60, 0.85),
        model='richards',
        nu=(0.3, 0.8),
        noise_level='medium',
        n_default=15,
        scoring_mode='a_not_identifiable',
        lab_band='full',
        tags=['incomplete', 'model_mismatch']
    ),

    'incomplete_growth_baranyi': ScenarioConfig(
        name='incomplete_growth_baranyi',
        description='100 h run still rising at the end, Baranyi-generated so A is genuinely non-identifiable',
        expected_class='GOOD',
        A=(1.3, 1.9),
        mu=(0.015, 0.035),
        lambda_=(2.0, 8.0),
        initial_od=GROWER_INITIAL_OD,
        end_fraction_of_A=(0.60, 0.85),
        model='baranyi',
        noise_level='medium',
        n_default=15,
        scoring_mode='a_not_identifiable',
        lab_band='full',
        tags=['incomplete', 'model_mismatch']
    ),

    'late_growth_still_rising': ScenarioConfig(
        name='late_growth_still_rising',
        description='60 h flat, then growth (the G4 PERMETHRINANDLB_CISPERM1 shape, t50 = 78 h)',
        expected_class='GOOD',
        A=(0.4, 1.2),
        mu=(0.05, 0.12),
        lambda_=(50.0, 70.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        # Tests that a long flat stretch before growth survives the pre-fit
        # SNR / rise gate.
        lab_band='moderate',
        n_default=12,
        tags=['incomplete', 'late_growth', 'prefit_gate']
    ),

    'run_ends_at_plateau_edge': ScenarioConfig(
        name='run_ends_at_plateau_edge',
        description='Run stops 4 samples after t99, leaving the peak-finder no plateau window',
        expected_class='GOOD',
        A=(1.0, 1.6),
        mu=(0.12, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        # NOT lambda + A/(mu*e) + 2: that is the Gompertz INFLECTION, where the
        # curve sits at 37 % of A.  t99 is the honest "the primary phase is
        # over" time.
        duration_rule='t99_plus_buffer',
        plateau_buffer_points=4,
        time_resolution=0.25,
        noise_level='medium',
        n_default=10,
        lab_band='full',
        tags=['truncation', 'plateau_edge']
    ),

    # --------------------------------------------------- multi-phase
    'slow_secondary_drift': ScenarioConfig(
        name='slow_secondary_drift',
        description='Plateau, then a slow 30-80 h creep - the dominant real grower shape',
        expected_class='GOOD',
        # 58.1 % of real growers are still gaining OD after t=50 h (median
        # +0.074, p90 +0.20, max +0.516 OD).  This is the shape that drove the
        # human PEAK_RATE_FRACTION calibration and the suite has never had one.
        # A1 must be in the real grower band for the measured post-t50 rates.
        A=(1.0, 1.9),
        mu=(0.10, 0.25),
        lambda_=(1.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        shape='slow_secondary_drift',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=15,
        lab_band='full',
        tags=['truncation', 'secondary_phase', 'drift']
    ),

    'diauxic_second_larger': ScenarioConfig(
        name='diauxic_second_larger',
        description='Diauxic shift whose SECOND phase is the larger one (A1 0.4-0.7, A2 0.8-1.2)',
        expected_class='GOOD',
        # Every previous second phase was smaller than the first, so the
        # first-peak rule was never asked to keep the smaller, earlier plateau.
        A=(0.4, 0.7),
        mu=(0.10, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        shape='diauxic_second_larger',
        A2=(0.8, 1.2),
        gap_hours=(5.0, 15.0),
        noise_level='medium',
        lab_band='full',
        n_default=12,
        tags=['truncation', 'diauxic', 'multi_peak']
    ),

    'clipped_then_grows': ScenarioConfig(
        name='clipped_then_grows',
        description='Flat zero for 30-40 h, then real growth (G4 PERMETHRINANDLB_CYPERM2, 26.4 % exact zeros)',
        expected_class='GOOD',
        A=(0.5, 0.8),
        mu=(0.05, 0.15),
        lambda_=(2.0, 8.0),
        # The shape needs a zero baseline: it IS the baseline story.
        initial_od=(0.0, 0.0),
        shape='clipped_then_grows',
        noise_level='low',
        n_default=10,
        lab_band='moderate',
        tags=['clipped', 'late_growth', 'baseline']
    ),

    # ------------------------------------------------------ replicates
    'edge_well_evaporation': ScenarioConfig(
        name='edge_well_evaporation',
        description='One of three wells evaporating at +0.0015 OD/h; the condition mean carries a third of it',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.12, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        noise_level='medium',
        # The generator's drift is common-mode (it is a property of the plate),
        # so a +0.0015 OD/h ramp on ONE of three wells is applied to the
        # condition as +0.0005 OD/h.  The row records what was applied.
        drift_od_per_h=(0.0005, 0.0005),
        n_replicates=3,
        replicate_A_cv=0.05,
        replicate_mu_cv=0.10,
        replicate_lambda_sd=0.5,
        scoring_mode='preprocessing_blind',
        lab_band='full',
        n_default=12,
        tags=['replicates', 'evaporation', 'drift', 'edge_well', 'artifact']
    ),

    'high_autocorr_good': ScenarioConfig(
        name='high_autocorr_good',
        description='Model mismatch with structured residuals: autocorr 0.7-0.95 at R² 0.96-0.99, still GOOD',
        expected_class='GOOD',
        # Anchored to Group 3 LB-IMID5: R² 0.974, lag-1 residual autocorrelation
        # 0.942, delta_od 1.676 - the pipeline calls it BAD on the
        # autocorrelation gate while validation_audit.csv records the human
        # confirming it GOOD.  No synthetic GOOD curve has ever exceeded 0.7221.
        A=(1.2, 1.8),
        # Slow growth is what makes a MISMATCH residual autocorrelated at the
        # gate's own 1 h lag: the pipeline measures lag-1-hour autocorrelation,
        # so the systematic residual has to be slow compared with 1 h.  At
        # mu 0.15 the same mismatch only reaches 0.62.
        mu=(0.025, 0.05),
        lambda_=(1.0, 5.0),
        initial_od=GROWER_INITIAL_OD,
        model='richards',
        # nu far from the Gompertz shape is what makes the residuals
        # systematic; the range is tuned so the pipeline's own R² spans the
        # 0.98 excellent-fit exemption from both sides.  Measured over 15
        # draws: R² 0.966-0.996 (5/15 below 0.98), lag-1 h autocorrelation
        # 0.77-0.92 (15/15 inside 0.7-0.95), 3/15 accepted.  The autocorrelation
        # gate fires on 5 and the A standard-error gate on 12 - the same
        # systematic misfit that raises the autocorrelation widens the
        # asymptote's standard error, which is itself a finding.
        nu=(4.0, 14.0),
        noise_level='low',
        scoring_mode='model_mismatch',
        lab_band='full',
        n_default=12,
        tags=['autocorr', 'model_mismatch', 'gate_boundary']
    ),

    # --------------------------------------------------------- decline
    'death_phase_extreme': ScenarioConfig(
        name='death_phase_extreme',
        description='The historic exponential collapse (median 72-98 % of peak lost) - out of distribution',
        expected_class='GOOD',
        A=(1.0, 1.5),
        mu=(0.15, 0.25),
        lambda_=(2.0, 4.0),
        initial_od=GROWER_INITIAL_OD,
        # Kept, named and reported: the real ceiling for end-of-run loss over
        # 48 growers is 0.372, so this rate is out of distribution and says so
        # instead of being the definition of "death phase".
        include_death_phase=True,
        t_death_fraction=(0.4, 0.5),
        k_death=(0.05, 0.1),
        noise_level='medium',
        n_default=5,
        lab_band='full',
        tags=['death_phase', 'severe_decline', 'out_of_distribution']
    ),

    'partial_lysis': ScenarioConfig(
        name='partial_lysis',
        description='Plateau then a gentle 20-35 % sigmoid loss (G4 LB_PERM7 0.253, G2 LB_MAL4 0.263)',
        expected_class='GOOD',
        A=(1.2, 1.8),
        mu=(0.12, 0.25),
        lambda_=(2.0, 6.0),
        initial_od=GROWER_INITIAL_OD,
        shape='partial_lysis',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=12,
        tags=['decline', 'lysis']
    ),

    'sudden_crash': ScenarioConfig(
        name='sudden_crash',
        description='Hold, then a ~10 h collapse of 25-40 % (G4 LB_PERM4: 1.880 -> 1.180 across 57-68 h)',
        expected_class='GOOD',
        A=(1.5, 2.0),
        mu=(0.15, 0.30),
        lambda_=(1.0, 4.0),
        initial_od=GROWER_INITIAL_OD,
        shape='sudden_crash',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        n_default=10,
        lab_band='full',
        tags=['decline', 'crash']
    ),

    'dip_and_recover': ScenarioConfig(
        name='dip_and_recover',
        description='Peak, 0.3-0.5 OD drawdown, then partial recovery (G2 MALATHIONANDLB_MAL8)',
        expected_class='GOOD',
        # 6/44 real growers do this at full magnitude and zero synthetic
        # scenarios were designed to.
        A=(1.4, 1.7),
        mu=(0.15, 0.30),
        lambda_=(1.0, 4.0),
        initial_od=GROWER_INITIAL_OD,
        shape='dip_and_recover',
        noise_level='medium',
        blank_offset_od=BLANK_OFFSET_OD,
        lab_band='full',
        n_default=12,
        tags=['decline', 'recovery', 'non_monotone']
    ),
}


# =============================================================================
# Scenario Collections
# =============================================================================

ALL_SCENARIOS = {
    **GOOD_GROWTH_SCENARIOS,
    **BAD_CURVE_SCENARIOS,
    **EDGE_CASE_SCENARIOS
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get a scenario by name."""
    if name not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. "
                        f"Available: {list(ALL_SCENARIOS.keys())}")
    return ALL_SCENARIOS[name]


def get_scenarios_by_class(expected_class: str) -> Dict[str, ScenarioConfig]:
    """Get all scenarios with a given expected classification."""
    return {
        name: config for name, config in ALL_SCENARIOS.items()
        if config.expected_class == expected_class
    }


def get_scenarios_by_tag(tag: str) -> Dict[str, ScenarioConfig]:
    """Get all scenarios with a given tag."""
    return {
        name: config for name, config in ALL_SCENARIOS.items()
        if tag in config.tags
    }


def list_scenarios() -> None:
    """Print a summary of all available scenarios."""
    print("=" * 70)
    print("AVAILABLE SCENARIOS")
    print("=" * 70)

    for category_name, scenarios in [
        ("GOOD GROWTH SCENARIOS", GOOD_GROWTH_SCENARIOS),
        ("BAD CURVE SCENARIOS", BAD_CURVE_SCENARIOS),
        ("EDGE CASE SCENARIOS", EDGE_CASE_SCENARIOS)
    ]:
        print(f"\n{category_name}:")
        print("-" * 70)
        for name, config in scenarios.items():
            print(f"  {name:30s} [{config.expected_class}] - {config.description}")

    print("\n" + "=" * 70)
    print(f"Total scenarios: {len(ALL_SCENARIOS)}")
    print(f"  Good: {len(get_scenarios_by_class('GOOD'))}")
    print(f"  Bad: {len(get_scenarios_by_class('BAD'))}")


# =============================================================================
# Suite composition
# =============================================================================

def get_comprehensive_test_config(
    n_per_scenario: Optional[int] = None,
    scenarios: Optional[Dict[str, ScenarioConfig]] = None,
) -> Dict[str, int]:
    """
    Curves per scenario for the comprehensive suite.

    The count is the scenario's own ``n_default``, so the suite's composition
    is part of the catalog rather than a pair of ``--extra-good`` /
    ``--extra-borderline`` flags in the generator script that no reader could
    reconstruct from the definitions.

    Args:
        n_per_scenario: when given, overrides every ``n_default`` (uniform).
        scenarios: the catalog to compose (defaults to ALL_SCENARIOS).

    Returns:
        Dict mapping scenario name to number of curves to generate.
    """
    catalog = ALL_SCENARIOS if scenarios is None else scenarios
    if n_per_scenario is not None:
        return {name: int(n_per_scenario) for name in catalog}
    return {name: int(cfg.n_default) for name, cfg in catalog.items()}


def load_lab_prior(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Load the derived lab prior (``config/lab_prior.json``).

    The file is written by ``scripts/derive_lab_prior.py`` from
    ``data/raw/Group1..4`` and is never hand-edited: the band fractions are a
    measurement, so duplicating them as literals here would let the two drift
    apart.
    """
    p = Path(path) if path is not None else LAB_PRIOR_PATH
    with open(p) as fh:
        return json.load(fh)


def lab_band_names(prior: Optional[Dict[str, Any]] = None) -> List[str]:
    """Ordered band names of the lab prior."""
    prior = prior or load_lab_prior()
    return [b['name'] for b in prior['bands']]


def get_lab_weighted_config(
    total_curves: int,
    scenarios: Optional[Dict[str, ScenarioConfig]] = None,
    prior: Optional[Dict[str, Any]] = None,
    min_per_scenario: int = 5,
) -> Dict[str, int]:
    """
    Curves per scenario so that the SUITE's delta-OD composition matches the
    lab's own.

    The measured composition of the 92 real conditions is 6 bands of total
    rise; the old suite inverted it (the 0.6-1.2 OD band was 46 % of the
    benchmark and 2 % of the lab, the dead-well band 4 % against 32 %).  Each
    scenario declares which band it belongs to via ``lab_band``; the budget of
    a band is split evenly between its scenarios, with ``min_per_scenario`` as
    a floor so no scenario silently disappears from the headline.

    Args:
        total_curves: approximate size of the suite (the floor can push the
            realised total above it).
        scenarios: catalog to compose (defaults to ALL_SCENARIOS).
        prior: parsed lab prior (defaults to the derived file).
        min_per_scenario: never generate fewer than this per scenario.

    Returns:
        Dict mapping scenario name to number of curves.
    """
    catalog = ALL_SCENARIOS if scenarios is None else scenarios
    prior = prior or load_lab_prior()
    fractions = {b['name']: float(b['fraction']) for b in prior['bands']}

    by_band: Dict[str, List[str]] = {}
    for name, cfg in catalog.items():
        band = cfg.lab_band or 'unbanded'
        by_band.setdefault(band, []).append(name)

    unknown = set(by_band) - set(fractions) - {'unbanded'}
    if unknown:
        raise ValueError(f"scenarios reference unknown lab bands: {sorted(unknown)}")

    config: Dict[str, int] = {}
    for band, names in by_band.items():
        share = fractions.get(band, 0.0)
        budget = total_curves * share
        per = int(round(budget / len(names))) if names else 0
        for name in names:
            config[name] = max(min_per_scenario, per)
    return {name: config[name] for name in catalog}


# =============================================================================
# Declared vs realised: the suite may not ship a property it only declared
# =============================================================================

#: How far a realised aggregate may sit from the declared band before the
#: suite is refused.  Generous on purpose - this gate catches a field that is
#: NOT CONSUMED AT ALL (the failure it exists for: ``well_sigma_od``,
#: ``spike_window_h``, ``n_outlier_wells`` and ``outlier_well_A_scale`` were
#: emitted by the sampler and read by nobody, so four noise rungs all ran at
#: the same preset and 35 curves labelled BAD for two dead wells were ordinary
#: growers), not a magnitude that is 10 % off its target.
REALISATION_SIGMA_LO = 0.6      # realised residual sigma vs the declared floor
REALISATION_SIGMA_HI = 2.5      # ... and vs the declared ceiling
REALISATION_SCALE_TOL = 0.25    # relative tolerance on a realised amplitude ratio


def _as_rows(ground_truth: Any) -> List[Dict[str, Any]]:
    """Accept a DataFrame or an iterable of mappings without importing pandas."""
    if hasattr(ground_truth, 'to_dict'):
        return list(ground_truth.to_dict('records'))
    return [dict(r) for r in ground_truth]


def _f(row: Dict[str, Any], key: str, default: float = float('nan')) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _spike_indices(row: Dict[str, Any]) -> List[int]:
    raw = row.get('spike_indices', '')
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    return [int(float(v)) for v in str(raw).split(';') if v not in ('', 'nan')]


def is_twin_row(row: Mapping[str, Any], siblings: Sequence[Mapping[str, Any]]) -> bool:
    """
    True when ``row`` is the ARTEFACT-FREE twin of another row.

    The generator appends the twin immediately after the curve it mirrors, and
    ``twin_curve_id`` cross-references the pair by ``strain_name``, so the twin
    is the member of the pair whose partner was written first.  Callers need
    this because a twin carries the preset's own drift and would otherwise look
    like it carried the scenario's artefact.
    """
    partner = str(row.get('twin_curve_id') or '')
    if not partner:
        return False
    names = {str(r.get('strain_name')) for r in siblings}
    return partner in names and partner < str(row.get('strain_name'))


def _median(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float('nan')


def audit_realisation(ground_truth: Any,
                      catalog: Optional[Dict[str, ScenarioConfig]] = None
                      ) -> List[str]:
    """
    Check that every generative property a scenario DECLARES is visible in the
    curves that were written, and report the ones that are not.

    A scenario config is a request to the generator.  When the generator does
    not consume a field the request is silently dropped, and the suite then
    ships ground-truth rows whose labels are justified by a property the data
    does not have - which is the one thing this rebuild is not allowed to do
    ("labels come from the truth curve").  This function is the gate:
    ``generate_comprehensive.py`` refuses to write a suite that fails it.

    It deliberately measures the DATA (the ground-truth rows), never the config
    it came from, and it is aggregate rather than per-curve so noise cannot
    trip it.

    Args:
        ground_truth: the generated ground-truth table (DataFrame or rows).
        catalog: scenario definitions (defaults to ALL_SCENARIOS).

    Returns:
        A list of human-readable violations; empty when the suite is honest.
    """
    catalog = ALL_SCENARIOS if catalog is None else catalog
    rows = _as_rows(ground_truth)
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get('scenario')), []).append(row)

    problems: List[str] = []
    for name, cfg in catalog.items():
        srows = by_scenario.get(name)
        if not srows:
            continue

        primary = [r for r in srows if not is_twin_row(r, srows)]

        # ---- spikes
        n_spikes = cfg.n_spikes if isinstance(cfg.n_spikes, int) else max(cfg.n_spikes)
        if n_spikes > 0:
            spiked = [r for r in primary if _spike_indices(r)]
            if len(spiked) < 0.9 * len(primary):
                problems.append(
                    f"{name}: declares {n_spikes} spike(s) but only "
                    f"{len(spiked)}/{len(primary)} curves recorded one "
                    f"(data_generator must read 'n_spikes'/'spike_od')")
            if cfg.spike_window_h is not None and spiked:
                lo, hi = float(cfg.spike_window_h[0]), float(cfg.spike_window_h[1])
                outside = 0
                for r in spiked:
                    dt = _f(r, 'time_resolution', float('nan'))
                    if not np.isfinite(dt):
                        continue
                    times = [i * dt for i in _spike_indices(r)]
                    outside += sum(1 for t in times if t < lo - 1e-9 or t > hi + 1e-9)
                if outside:
                    problems.append(
                        f"{name}: declares spike_window_h=({lo}, {hi}) h but "
                        f"{outside} spike(s) landed outside it "
                        f"(data_generator must read 'spike_window_h')")

        # ---- a signed drift
        if cfg.drift_od_per_h is not None:
            floor = 0.5 * min(abs(v) for v in cfg.drift_od_per_h)
            drifted = [r for r in primary if abs(_f(r, 'drift_od_per_h', 0.0)) >= floor]
            if len(drifted) < 0.9 * len(primary):
                problems.append(
                    f"{name}: declares drift_od_per_h={cfg.drift_od_per_h} but only "
                    f"{len(drifted)}/{len(primary)} curves carry it")

        # ---- a per-well sigma override (the noise ladder's rungs)
        if cfg.well_sigma_od is not None:
            lo, hi = float(cfg.well_sigma_od[0]), float(cfg.well_sigma_od[1])
            realised = _median([_f(r, 'rmse') for r in primary])
            if not (REALISATION_SIGMA_LO * lo <= realised <= REALISATION_SIGMA_HI * hi):
                problems.append(
                    f"{name}: declares well_sigma_od=({lo}, {hi}) OD but the "
                    f"realised residual sigma is {realised:.4f} "
                    f"(data_generator must read 'well_sigma_od')")

        # ---- dead / divergent replicate wells
        if cfg.outlier_well_A_scale is not None and cfg.outlier_well_prob > 0:
            n_rep = int(cfg.n_replicates or 3)
            n_out = int(cfg.n_outlier_wells or 1)
            scale = 0.5 * (float(cfg.outlier_well_A_scale[0])
                           + float(cfg.outlier_well_A_scale[1]))
            expected = ((n_rep - n_out) + n_out * scale) / n_rep
            realised = _median([_f(r, 'true_frac_of_A_reached') for r in primary])
            if not np.isfinite(realised) or abs(realised - expected) > max(
                    REALISATION_SCALE_TOL * expected, 0.05):
                problems.append(
                    f"{name}: declares {n_out} of {n_rep} wells at "
                    f"{cfg.outlier_well_A_scale} of their siblings, so the "
                    f"condition mean should reach {expected:.2f} of a well; the "
                    f"curves reach {realised:.2f} (data_generator must read "
                    f"'n_outlier_wells'/'outlier_well_A_scale')")

        # ---- the shape library
        if cfg.shape is not None:
            wrong = [r for r in primary if str(r.get('shape')) != cfg.shape]
            if wrong:
                problems.append(
                    f"{name}: declares shape={cfg.shape!r} but {len(wrong)} rows "
                    f"record a different one")

    return problems


# =============================================================================
# Scenario Generator Helper
# =============================================================================

def _e() -> float:
    return float(np.e)


class ScenarioSampler:
    """Sample parameters from a scenario configuration.

    Two of the sampled quantities are SOLVED rather than drawn, because the
    property the scenario declares is not a parameter:

    * ``end_fraction_of_A`` - "the run ends at 60-85 % of A".  The modified
      Gompertz reaches ``f`` of its asymptote at
      ``t = lambda + A*(1 - ln(-ln f))/(mu*e)``, so for a fixed run length the
      rate that realises the requested fraction is
      ``mu = A*(1 - ln(-ln f))/(e*(T - lambda))``.  Sampling mu directly cannot
      express it: with the plan's (0.10, 0.20) band in a 24 h run only 3 % of
      draws end below 90 % of A.
    * ``duration_rule='t99_plus_buffer'`` - "the run stops just after the
      primary phase", i.e. ``t99 + buffer*dt``.

    Both are recorded in the sampled parameters, and the ground truth measures
    what the curve actually did, so neither can flatter the row.
    """

    def __init__(self, scenario: ScenarioConfig, seed: Optional[int] = None):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- helpers
    def _sample_range(self, range_tuple: Tuple[float, float]) -> float:
        """Sample uniformly from a range."""
        low, high = range_tuple
        if low == high:
            return low
        return self.rng.uniform(low, high)

    def _maybe(self, value: Any) -> Optional[float]:
        """Sample a scalar / range / None to a float (None stays None)."""
        if value is None:
            return None
        if isinstance(value, (tuple, list, np.ndarray)):
            return float(self._sample_range((float(value[0]), float(value[1]))))
        return float(value)

    def _maybe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (tuple, list, np.ndarray)):
            lo, hi = int(value[0]), int(value[1])
            return int(self.rng.integers(lo, hi + 1))
        return int(value)

    @staticmethod
    def _gompertz_time_at_fraction(f: float, A: float, mu: float,
                                   lambda_: float) -> float:
        """Time the modified Gompertz reaches fraction ``f`` of ``A``."""
        return lambda_ + A * (1.0 - float(np.log(-np.log(f)))) / (mu * _e())

    @staticmethod
    def _gompertz_mu_for_fraction(f: float, A: float, lambda_: float,
                                  t_end: float) -> float:
        """Rate that puts the curve at fraction ``f`` of ``A`` at ``t_end``."""
        span = max(t_end - lambda_, 1e-6)
        return float(A * (1.0 - float(np.log(-np.log(f)))) / (_e() * span))

    # -------------------------------------------------------------- public
    def sample_parameters(self) -> Dict[str, Any]:
        """Sample a single set of parameters from the scenario."""
        s = self.scenario

        A = self._sample_range(s.A)
        mu = self._sample_range(s.mu)
        lambda_ = self._sample_range(s.lambda_)
        initial_od = self._sample_range(s.initial_od)
        duration_hours = float(s.duration_hours)
        time_resolution = float(s.time_resolution)

        end_fraction = None
        if s.end_fraction_of_A is not None and A > 0:
            end_fraction = self._sample_range(s.end_fraction_of_A)
            # The last sample of np.arange(0, duration, dt).
            t_end = duration_hours - time_resolution
            mu = self._gompertz_mu_for_fraction(end_fraction, A, lambda_, t_end)

        if s.duration_rule == 't99_plus_buffer' and A > 0 and mu > 0:
            t99 = self._gompertz_time_at_fraction(0.99, A, mu, lambda_)
            duration_hours = float(t99 + s.plateau_buffer_points * time_resolution
                                   + time_resolution)

        if s.initial_od_offset_fraction is not None:
            frac = self._sample_range(s.initial_od_offset_fraction)
            # initial_od / (initial_od + A) == frac
            initial_od = float(frac * A / (1.0 - frac))
        else:
            frac = None

        params: Dict[str, Any] = {
            'A': A,
            'mu': mu,
            'lambda_': lambda_,
            'initial_od': initial_od,
            'model': s.model,
            'noise_level': s.noise_level,
            'duration_hours': duration_hours,
            'time_resolution': time_resolution,
            'include_death_phase': s.include_death_phase,
            'pattern': s.pattern,
            'instrument_noise': s.instrument_noise,
            'expected_class': s.expected_class,
            'scenario_name': s.name,
            # ---- Contract A
            'label_rule': s.label_rule,
            'scoring_mode': s.scoring_mode,
            'lab_band': s.lab_band,
            'shape': s.shape,
            # Second-phase / decline parameters stay RANGES: curve_shapes draws
            # them on the curve's own stream (Contract C).
            'A2': s.A2,
            'mu2': s.mu2,
            'lambda2': s.lambda2,
            'gap_hours': s.gap_hours,
            'death_loss_frac': s.death_loss_frac,
            't_death_offset_h': s.t_death_offset_h,
            'drift_od_per_h': s.drift_od_per_h,
            'blank_offset_od': s.blank_offset_od,
            'spike_od': s.spike_od,
            'spike_window_h': s.spike_window_h,
            'well_sigma_od': s.well_sigma_od,
            'n_spikes': self._maybe_int(s.n_spikes) or 0,
            'spike_decay_points': int(s.spike_decay_points),
            'replicate_A_cv': float(s.replicate_A_cv),
            'replicate_mu_cv': float(s.replicate_mu_cv),
            'replicate_lambda_sd': float(s.replicate_lambda_sd),
            'outlier_well_prob': float(s.outlier_well_prob),
            'outlier_well_A_scale': s.outlier_well_A_scale,
            'media': s.media,
            # ---- provenance of the two solved quantities
            'end_fraction_of_A': end_fraction,
            'initial_od_offset_fraction': frac,
        }

        if s.n_outlier_wells is not None:
            params['n_outlier_wells'] = int(s.n_outlier_wells)
        if s.n_replicates is not None:
            params['n_replicates'] = int(s.n_replicates)
        if s.generate_twin is not None:
            params['generate_twin'] = bool(s.generate_twin)
        for key in ('Ks', 'Ki', 'q', 'S0'):
            value = getattr(s, key)
            if value is not None:
                params[key] = float(value)
        if s.nu is not None:
            params['nu'] = self._maybe(s.nu)

        if s.include_death_phase:
            params['t_death_fraction'] = self._sample_range(s.t_death_fraction)
            params['k_death'] = self._sample_range(s.k_death)

        if s.target_r_squared:
            params['target_r_squared'] = self._sample_range(s.target_r_squared)

        return params

    def sample_batch(self, n: int) -> List[Dict[str, Any]]:
        """Sample n parameter sets from the scenario."""
        return [self.sample_parameters() for _ in range(n)]


__all__ = [
    'ScenarioConfig', 'ScenarioSampler',
    'GOOD_GROWTH_SCENARIOS', 'BAD_CURVE_SCENARIOS', 'EDGE_CASE_SCENARIOS',
    'ALL_SCENARIOS', 'GROWER_INITIAL_OD', 'BLANK_OFFSET_OD', 'NO_BLANK',
    'get_scenario', 'get_scenarios_by_class', 'get_scenarios_by_tag',
    'list_scenarios', 'get_comprehensive_test_config',
    'get_lab_weighted_config', 'load_lab_prior', 'lab_band_names',
    'audit_realisation', 'is_twin_row', 'LAB_PRIOR_PATH',
]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    list_scenarios()

    print("\n\nExample: Sampling from 'standard' scenario:")
    sampler = ScenarioSampler(get_scenario('standard'), seed=42)
    for i in range(3):
        params = sampler.sample_parameters()
        print(f"  Sample {i+1}: A={params['A']:.3f}, mu={params['mu']:.3f}, "
              f"lambda={params['lambda_']:.2f}")

    print("\n\nComprehensive test config:")
    config = get_comprehensive_test_config()
    total = sum(config.values())
    print(f"Total curves to generate: {total}")
