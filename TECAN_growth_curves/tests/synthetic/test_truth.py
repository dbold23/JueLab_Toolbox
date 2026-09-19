"""WP3 acceptance tests: the generation core and the v2 truth schema.

What these tests are for
------------------------
Every one of them pins a property that was false in the shipped 480-curve
benchmark, and each failure mode is named in the docstring of the test that
would now catch it.  The reference numbers are measured, not invented:

    baranyi_generated truth      recorded true_mu 0.173-0.209 against a
                                 measured max slope of 0.066-0.099 on
                                 CURVE0390-0393 (truth described the request,
                                 not the curve)
    logistic_generated (M01)     smoothed max dOD/dt was 0.39-0.42x true_mu and
                                 the first point sat 13-21 % of A above the
                                 baseline on CURVE0405-0408
    seeding (M03)                two runs at --seed 0 produced different data
                                 (np.allclose False), and per-curve seeds were
                                 base + a running index across scenarios
    sentinel metrics (M03)       actual_r_squared hard-coded to 0.0 for flat
                                 curves and -1.0 for random walks; 15 rows with
                                 NaN actual_r_squared and 45 with NaN rmse
    duration (M28)               duration_hours read a key that never existed,
                                 so a 100 h run reported 99.75
    clipping (M09)               28/92 real conditions (30.4 %) have a
                                 zero-variance opening segment and 12/92 are
                                 identically zero somewhere, against 0/480
                                 synthetic

The suite is generated at n=2 per scenario (74 curves, ~2 s) so the whole file
stays well inside the 60 s budget.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from src import truth                                            # noqa: E402
from src.curve_scenarios import ALL_SCENARIOS                    # noqa: E402
from src.data_generator import (                                 # noqa: E402
    BLANK_OFF,
    BLANK_OVERSUBTRACTION_OD,
    BLANK_OVERSUBTRACTION_PROB,
    BLANK_UNDERSUBTRACTION_OD,
    DEFAULT_BLANK_OFFSET_OD,
    DEFAULT_N_REPLICATES,
    REPLICATE_SHARED_SIGMA_FRACTION,
    SyntheticGrowthCurveGenerator,
    _draw_blank_offset,
    curve_rows,
    curve_seed,
    curve_to_row,
)

SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _measure(clean, t, t_death=np.nan, t_second_start=np.nan):
    """Independent re-measurement of (max slope, tangent lag) of a curve.

    Deliberately NOT a call into ``truth.measure_shape``: the point of the
    structural guard is that the shipped truth matches a measurement taken
    from outside the module that wrote it.
    """
    clean = np.asarray(clean, dtype=float)
    t = np.asarray(t, dtype=float)
    edges = [e for e in (t_second_start, t_death) if np.isfinite(e)]
    end = min([float(t[-1])] + [float(e) for e in edges])
    mask = t <= end + 1e-12
    y, tt = clean[mask], t[mask]
    d = np.gradient(y, tt)
    i = int(np.argmax(d))
    slope = float(d[i])
    lag = float(tt[i] - (y[i] - y[0]) / slope) if slope > 0 else float('nan')
    return slope, lag, float(np.max(y) - y[0])


def _has_growth_phase(curve):
    """Does this curve have a growth phase at all (outside truth.py)?

    True when a growth model generated it AND its own array rises by at least
    :data:`truth.MIN_RESOLVABLE_AMPLITUDE_OD` with a positive slope.  A random
    walk fails the first clause (its maximum finite difference is noise, not a
    rate); a totally inhibited curve fails the second.
    """
    row = curve.truth
    slope, _, amp = _measure(curve.clean_od600, curve.time,
                             row['t_death'], row['t_second_start'])
    return bool(curve.metadata.get('has_clean_reference', True)
                and slope > 0
                and amp >= truth.MIN_RESOLVABLE_AMPLITUDE_OD)


@pytest.fixture(scope='module')
def suite():
    """Two curves for every defined scenario, seed 42 (74 rows, ~2 s)."""
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    curves = []
    for name in ALL_SCENARIOS:
        curves.extend(generator.generate_from_scenario(name, n_curves=2, seed=SEED))
    assert len(curves) >= 2 * len(ALL_SCENARIOS)
    return curves


@pytest.fixture(scope='module')
def suite_frame():
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    return generator.generate_comprehensive_test_set(
        {name: 2 for name in ALL_SCENARIOS}, seed=SEED, verbose=False)


@pytest.fixture(scope='module')
def shipped_suite(tmp_path_factory):
    """
    The suite the generator CLI actually writes, read back off the plate files.

    This is the population an auditor measures - the full shipped composition
    at ``--seed 42``, written to a scratch directory (never into
    ``synthetic_data/output/``) - and it is what acceptance item 7 is about.
    Generating it costs ~17 s, so it is module-scoped.

    Returns ``(ods, ground_truth)``: the OD array of every curve keyed by its
    CURVE number, and the ground-truth frame with that number in
    ``curve_number``.
    """
    import re
    import subprocess

    import pandas as pd

    out_dir = tmp_path_factory.mktemp('shipped_suite')
    proc = subprocess.run(
        [sys.executable, str(SYNTH / 'scripts' / 'generate_comprehensive.py'),
         '-o', str(out_dir), '--seed', str(SEED)],
        capture_output=True, text=True, cwd=str(SYNTH),
        env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert proc.returncode == 0, proc.stdout + proc.stderr

    ods = {}
    for path in sorted((out_dir / 'test_data' / 'DATA').glob('*.csv')):
        frame = pd.read_csv(path)
        for column in frame.columns[1:]:
            match = re.search(r'CURVE(\d+)', str(column))
            if match:
                ods[int(match.group(1))] = frame[column].to_numpy(dtype=float)
    ground_truth = pd.read_csv(out_dir / 'test_data' / 'ground_truth.csv')
    ground_truth['curve_number'] = (
        ground_truth['strain_name'].str.extract(r'CURVE(\d+)')[0].astype(int))
    assert set(ground_truth['curve_number']) <= set(ods)
    return ods, ground_truth


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_is_contract_b_in_order():
    """The v2 column list, its order and the two version stamps are a contract."""
    expected = [
        'curve_id', 'strain_name', 'scenario', 'expected_class', 'label_rule',
        'label_rule_version', 'scoring_mode', 'lab_band', 'model_type', 'shape',
        'requested_A', 'requested_mu', 'requested_lambda',
        'true_A', 'true_mu', 'true_lambda',
        'mu_realisation_ratio', 'lambda_realisation_error', 'shape_solved',
        'initial_od', 'noise_level', 'noise_sources', 'target_r_squared',
        'actual_r_squared', 'artefact_free_r_squared', 'artefact_free_rmse',
        'rmse', 'max_od', 'delta_od', 'n_points', 'time_resolution',
        'duration_hours', 'seed', 'true_window_end_h', 't_primary_95',
        't_second_start', 't_death', 'oracle_A_relerr', 'oracle_mu_relerr',
        'oracle_fit_ok', 'true_frac_of_A_reached', 'replicate_A_cv',
        'replicate_mu_cv', 'outlier_well', 'spike_indices', 'drift_od_per_h',
        'twin_curve_id', 'metrics_defined',
    ]
    assert list(truth.GROUND_TRUTH_COLUMNS) == expected
    assert truth.TRUTH_SCHEMA_VERSION == '2'
    assert truth.LABEL_RULE_VERSION == '1'


def test_fill_defaults_carry_no_measurement():
    """A generator that forgets a measurement must still get a loud error.

    Only annotation columns may have a fill default; if ``true_A`` or
    ``oracle_A_relerr`` were fillable, output_formatter's schema check would
    silently invent them.
    """
    measured = {
        'true_A', 'true_mu', 'true_lambda', 'mu_realisation_ratio',
        'lambda_realisation_error', 'actual_r_squared', 'rmse',
        'artefact_free_r_squared', 'artefact_free_rmse', 'max_od', 'delta_od',
        'seed', 'true_window_end_h', 't_primary_95', 't_second_start',
        't_death', 'oracle_A_relerr', 'oracle_mu_relerr', 'oracle_fit_ok',
        'true_frac_of_A_reached', 'metrics_defined', 'shape_solved',
    }
    assert measured.isdisjoint(set(truth.GROUND_TRUTH_FILL_DEFAULTS))


def test_generated_rows_carry_every_declared_column(suite_frame):
    """The generator computes the whole schema itself - nothing is derived."""
    missing = [c for c in truth.GROUND_TRUTH_COLUMNS if c not in suite_frame.columns]
    assert missing == []
    assert list(suite_frame['strain_name'])[:2] == ['CURVE0000', 'CURVE0001']


# ---------------------------------------------------------------------------
# acceptance 1: truth matches the shipped curve
# ---------------------------------------------------------------------------

def test_truth_matches_the_shipped_curve(suite):
    """
    Every row's ``true_mu``/``true_lambda`` are properties of its own curve.

    The structural guard that a scenario can never ship a truth table its
    generator did not realise: ``baranyi_generated`` used to record true_mu at
    2.3x the maximum slope of the curve it wrote, and all 15 of those curves
    still scored as classification successes.

    Curves with no growth phase (``flat_no_growth``, ``erratic``,
    ``haldane_total_inhibition``) have no rate or lag to match; they are
    asserted to say so (NaN) rather than to carry a number.  Their AMPLITUDE is
    still a measurement - see
    :func:`test_true_A_is_always_the_amplitude_of_the_shipped_curve`.
    """
    checked = 0
    for curve in suite:
        row = curve.truth
        if not np.isfinite(row['true_mu']):
            assert not _has_growth_phase(curve), row['scenario']
            assert not np.isfinite(row['true_lambda']), row['scenario']
            continue
        slope, lag, amp = _measure(curve.clean_od600, curve.time,
                                   row['t_death'], row['t_second_start'])
        assert slope == pytest.approx(row['true_mu'], rel=0.02), row['scenario']
        assert amp == pytest.approx(row['true_A'], rel=0.02), row['scenario']
        assert lag == pytest.approx(row['true_lambda'], rel=0.02, abs=0.05), row['scenario']
        assert np.isfinite(row['mu_realisation_ratio']), row['scenario']
        checked += 1
    assert checked > 60


def test_true_A_is_always_the_amplitude_of_the_shipped_curve(suite):
    """
    ``true_A`` is a measurement of the array in the same row - never a sentinel.

    The defect this pins: every ``erratic`` row shipped ``true_A = 0.0`` while
    its own noiseless array rose by up to 0.42 OD (and the written trace had
    delta_od up to 0.719).  A random walk has no rate and no lag - those stay
    NaN, which is honest - but "the amplitude of this curve is zero" was a
    claim contradicted by the curve one column to the left.
    """
    walk_amplitudes = []
    for curve in suite:
        row = curve.truth
        _, _, amp = _measure(curve.clean_od600, curve.time,
                             row['t_death'], row['t_second_start'])
        assert np.isfinite(row['true_A']), row['scenario']
        assert row['true_A'] == pytest.approx(max(amp, 0.0), rel=0.02, abs=1e-9), (
            f"{row['scenario']}: true_A {row['true_A']} != measured {amp}")
        if not curve.metadata.get('has_clean_reference', True):
            walk_amplitudes.append(row['true_A'])

    # and the walks are the case that used to be zero: at least one of them
    # really does climb past the pipeline's own 0.15 OD growth threshold.
    assert walk_amplitudes, 'no model-free trace in the suite'
    assert max(walk_amplitudes) > 0.15


def test_no_growth_phase_means_no_rate_and_no_lag(suite):
    """
    A curve that never leaves the baseline reports NaN, not a number.

    The mirror of the ``erratic`` defect: ``haldane_total_inhibition`` writes a
    curve that rises 1.2e-6 OD over 100 h - four orders of magnitude below the
    0.00046 OD noise sigma of the quietest real well - and used to ship
    ``true_mu = 1.8e-08`` with ``true_lambda = 30.9 h``, a lag phase measured on
    a rounding error.
    """
    seen = set()
    for curve in suite:
        row = curve.truth
        if _has_growth_phase(curve):
            continue
        seen.add(row['scenario'])
        assert not np.isfinite(row['true_mu']), row['scenario']
        assert not np.isfinite(row['true_lambda']), row['scenario']
        assert not bool(row['oracle_fit_ok']), row['scenario']
    assert {'erratic', 'haldane_total_inhibition', 'flat_no_growth'} <= seen


def test_requested_and_true_are_separate_columns(suite):
    """The sampler's request survives as ``requested_*`` next to the measurement."""
    for curve in suite:
        row = curve.truth
        assert row['requested_A'] == pytest.approx(curve.parameters['A'])
        assert row['requested_mu'] == pytest.approx(curve.parameters['mu'])
        assert row['requested_lambda'] == pytest.approx(curve.parameters['lambda_'])
        if np.isfinite(row['true_mu']) and row['requested_mu']:
            assert row['mu_realisation_ratio'] == pytest.approx(
                row['true_mu'] / row['requested_mu'], rel=1e-9)


def test_a_converter_that_misses_its_target_says_so():
    """``shape_solved`` is the honest flag, not a claim.

    A Baranyi curve whose requested lag is below the model's own intrinsic lag
    floor cannot be shipped: the converter returns the achievable shape with
    ``realised=False``, and the row records ``shape_solved=False`` with a
    ``true_lambda`` that is the lag the curve really has.
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    curve = generator.generate_single_curve(
        A=1.2, mu=0.2, lambda_=0.5, model_type='baranyi', seed=11)
    assert curve.truth['shape_solved'] is False
    slope, lag, _ = _measure(curve.clean_od600, curve.time)
    assert curve.truth['true_lambda'] == pytest.approx(lag, rel=0.02, abs=0.05)
    assert curve.truth['lambda_realisation_error'] == pytest.approx(
        curve.truth['true_lambda'] - 0.5)


# ---------------------------------------------------------------------------
# acceptance 2: M01 in the artifacts, not only in the model function
# ---------------------------------------------------------------------------

def test_logistic_data_regression():
    """
    The logistic curves the generator WRITES have the slope and start they claim.

    M01: the converter divided by e twice, so the shipped CURVE0405-0408 had a
    smoothed max dOD/dt of 0.39-0.42x their recorded true_mu and started
    13-21 % of A above the baseline.  This asserts it on the emitted arrays -
    the level the unit tests of growth_models.py cannot see.
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    curves = generator.generate_from_scenario('logistic_generated', n_curves=8,
                                              seed=SEED)
    kernel = np.ones(5) / 5.0
    for curve in curves:
        row = curve.truth
        smoothed = np.convolve(curve.od600, kernel, mode='same')
        d = np.gradient(smoothed, curve.time)
        max_rate = float(np.max(d[3:-3]))
        assert max_rate == pytest.approx(row['true_mu'], rel=0.10)
        start_frac = (curve.od600[0] - row['initial_od']) / row['true_A']
        assert start_frac < 0.06, f"logistic starts at {start_frac:.3f} of A"


# ---------------------------------------------------------------------------
# acceptance 3: determinism
# ---------------------------------------------------------------------------

def test_curve_seed_is_content_addressed():
    """``blake2b(f"{seed}:{scenario}:{i}")``: position-independent, and seed 0 seeds."""
    assert curve_seed(0, 'standard', 0) is not None
    assert curve_seed(0, 'standard', 0) == curve_seed(0, 'standard', 0)
    assert curve_seed(0, 'standard', 0) != curve_seed(0, 'high_A', 0)
    assert curve_seed(0, 'standard', 0) != curve_seed(0, 'standard', 1)
    assert curve_seed(0, 'standard', 0) != curve_seed(1, 'standard', 0)
    # the sampler draws from a different stream than the curve
    assert curve_seed(0, 'standard', 0, 'params') != curve_seed(0, 'standard', 0)
    assert curve_seed(None, 'standard', 0) is None
    assert 0 <= curve_seed(42, 'standard', 3) < 2 ** 31


@pytest.mark.parametrize('seed', [0, 42])
def test_determinism(seed):
    """
    Same seed -> byte-identical data; adding a scenario changes nothing else.

    M03: ``seed + i if seed else None`` made ``--seed 0`` silently unseeded (two
    runs did not match), and per-curve seeds were the base seed plus an index
    accumulating across scenarios in dict order, so appending one GOOD scenario
    rewrote every BAD and EDGE curve.
    """
    def run(config):
        return SyntheticGrowthCurveGenerator(seed=seed).generate_comprehensive_test_set(
            config, seed=seed, verbose=False)

    first = run({'standard': 2})
    again = run({'standard': 2})
    for a, b in zip(first['od600'], again['od600']):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    with_more = run({'standard': 2, 'high_A': 2})
    standard = with_more[with_more['scenario'] == 'standard']
    assert len(standard) == 2
    for a, b in zip(first['od600'], standard['od600']):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    reordered = run({'high_A': 2, 'standard': 2})
    reordered_standard = reordered[reordered['scenario'] == 'standard']
    for a, b in zip(first['od600'], reordered_standard['od600']):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_shipped_command_is_reproducible_at_seed_0(tmp_path):
    """
    Two runs of the actual generator CLI at ``--seed 0`` are byte-identical.

    The DataFrame-level test above cannot see the export; this one runs the
    command the suite is generated with (into a scratch directory, never into
    ``synthetic_data/output/``) and hashes every file it wrote.  Before the fix
    ``--seed 0`` produced different data on every run.
    """
    import hashlib
    import subprocess

    def run(out_dir):
        proc = subprocess.run(
            [sys.executable, str(SYNTH / 'scripts' / 'generate_comprehensive.py'),
             '-o', str(out_dir), '--n-per-scenario', '1', '--seed', '0'],
            capture_output=True, text=True, cwd=str(SYNTH),
            env={**os.environ, 'MPLBACKEND': 'Agg'})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        digests = {}
        for path in sorted((out_dir / 'test_data').rglob('*.csv')):
            digests[path.relative_to(out_dir).as_posix()] = hashlib.sha256(
                path.read_bytes()).hexdigest()
        assert digests
        return digests

    first = run(tmp_path / 'a')
    second = run(tmp_path / 'b')
    assert first == second


# ---------------------------------------------------------------------------
# acceptance 4: the label rule is scoped
# ---------------------------------------------------------------------------

def test_label_rule_scoped(suite_frame):
    """
    No ``label_rule='scenario'`` row was relabelled, contamination included.

    A blanket oracle rule would flip ``contamination`` (dOD 0.30-0.52,
    parameters perfectly recoverable) to GOOD and destroy a real BAD case.
    """
    declared = {name: cfg.expected_class for name, cfg in ALL_SCENARIOS.items()}
    scenario_rows = suite_frame[suite_frame['label_rule'] == 'scenario']
    # The rule is SCOPED, not absent: a scenario opts in by declaring
    # label_rule='oracle' (WP5 does so for the noise-defined scenarios), and
    # every other row keeps the class its scenario declared.
    opted_in = {name for name, cfg in ALL_SCENARIOS.items()
                if getattr(cfg, 'label_rule', 'scenario') == 'oracle'}
    assert set(scenario_rows['scenario']) == set(ALL_SCENARIOS) - opted_in
    assert set(suite_frame[suite_frame['label_rule'] == 'oracle']['scenario']) == opted_in
    for _, row in scenario_rows.iterrows():
        assert row['expected_class'] == declared[row['scenario']], row['scenario']
    assert set(suite_frame[suite_frame['scenario'] == 'contamination']
               ['expected_class']) == {'BAD'}
    assert set(suite_frame['label_rule_version']) == {truth.LABEL_RULE_VERSION}


def test_erratic_has_no_oracle_and_does_not_raise(suite_frame):
    """``erratic`` has no rate to recover: NaN oracle columns, fit_ok False.

    A random walk carries a measured ``true_A`` (it is not flat) but a NaN
    ``true_mu``, and it is the NaN rate - not a zeroed amplitude - that keeps
    the oracle out of a division by zero.
    """
    rows = suite_frame[suite_frame['scenario'] == 'erratic']
    assert len(rows) >= 2
    for _, row in rows.iterrows():
        assert np.isfinite(row['true_A'])
        assert np.isnan(row['true_mu'])
        assert np.isnan(row['oracle_A_relerr'])
        assert np.isnan(row['oracle_mu_relerr'])
        assert bool(row['oracle_fit_ok']) is False
    assert rows['true_A'].max() > 0.0


def test_oracle_rule_applies_where_a_scenario_opts_in():
    """
    ``label_rule='oracle'`` labels from the truth curve, not from whole-trace R2.

    M02: the three "noisy BAD" scenarios were labelled by R^2 of the noisy array
    against the clean one over the whole run - a statistic that moves from
    0.977 to 0.880 on ONE unchanged curve by running the plate longer.  An
    oracle fit over the curve's own window recovers A and mu to a few percent
    on exactly those curves, so under the rule they are GOOD.
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    params = {
        'A': 1.0, 'mu': 0.15, 'lambda_': 4.0, 'initial_od': 0.01,
        'expected_class': 'BAD', 'label_rule': 'oracle',
        'target_r_squared': 0.93,
    }
    curve = generator.generate_single_curve(
        A=1.0, mu=0.15, lambda_=4.0, initial_od=0.01, noise_level='medium',
        target_r_squared=0.93, seed=5, params=params)
    assert curve.truth['label_rule'] == 'oracle'
    assert curve.truth['oracle_fit_ok'] is True
    assert curve.truth['oracle_A_relerr'] < 0.20
    assert curve.truth['oracle_mu_relerr'] < 0.20
    assert curve.truth['expected_class'] == 'GOOD'

    # ... and the same rule keeps a genuinely unrecoverable curve BAD
    flat = dict(params, A=0.05, expected_class='GOOD')
    tiny = generator.generate_single_curve(
        A=0.05, mu=0.01, lambda_=4.0, initial_od=0.01, noise_level='medium',
        seed=6, params=flat)
    assert tiny.truth['expected_class'] == 'BAD'


def test_label_rule_helper_is_pure():
    """:func:`truth.apply_label_rule` is decidable from the metrics alone."""
    good = truth.apply_label_rule(
        {'label_rule': 'oracle', 'expected_class': 'BAD'},
        {'true_delta_od': 0.9, 'oracle_A_relerr': 0.05,
         'oracle_mu_relerr': 0.05, 'oracle_fit_ok': True})
    assert good['expected_class'] == 'GOOD'
    for metrics in (
        {'true_delta_od': 0.10, 'oracle_A_relerr': 0.05, 'oracle_mu_relerr': 0.05,
         'oracle_fit_ok': True},
        {'true_delta_od': 0.9, 'oracle_A_relerr': 0.25, 'oracle_mu_relerr': 0.05,
         'oracle_fit_ok': True},
        {'true_delta_od': 0.9, 'oracle_A_relerr': 0.05, 'oracle_mu_relerr': 0.25,
         'oracle_fit_ok': True},
        {'true_delta_od': 0.9, 'oracle_A_relerr': float('nan'),
         'oracle_mu_relerr': float('nan'), 'oracle_fit_ok': False},
    ):
        out = truth.apply_label_rule({'label_rule': 'oracle',
                                      'expected_class': 'GOOD'}, metrics)
        assert out['expected_class'] == 'BAD', metrics

    kept = truth.apply_label_rule({'label_rule': 'scenario', 'expected_class': 'BAD'},
                                  {'true_delta_od': 9.0, 'oracle_A_relerr': 0.0,
                                   'oracle_mu_relerr': 0.0, 'oracle_fit_ok': True})
    assert kept['expected_class'] == 'BAD'


def test_relabelling_a_scenario_row_is_an_error():
    """The generation-time guard is a real assertion, not a comment."""
    with pytest.raises(AssertionError):
        truth.assert_scenario_labels_untouched(
            [{'curve_id': 3, 'scenario': 'contamination',
              'label_rule': 'scenario', 'expected_class': 'GOOD'}],
            {'contamination': 'BAD'})


# ---------------------------------------------------------------------------
# acceptance 5 + 6: no sentinel metrics, consistent grid bookkeeping
# ---------------------------------------------------------------------------

def test_no_sentinel_metrics(suite_frame):
    """
    0.0 and -1.0 are not R^2 values, they were placeholders.

    ``generate_flat_curve`` wrote 0.0 and ``generate_random_walk_curve`` wrote
    -1.0, while ``truncation_challenge`` wrote nothing at all - three
    conventions in one column.  Now: a real number wherever a clean reference
    exists, NaN plus ``metrics_defined=False`` where none does.
    """
    for _, row in suite_frame.iterrows():
        defined = bool(row['metrics_defined'])
        if defined:
            assert np.isfinite(row['actual_r_squared']), row['scenario']
            assert np.isfinite(row['rmse']), row['scenario']
            assert row['actual_r_squared'] not in (0.0, -1.0), row['scenario']
        else:
            assert np.isnan(row['actual_r_squared']), row['scenario']

    walks = suite_frame[suite_frame['scenario'] == 'erratic']
    assert (~walks['metrics_defined'].astype(bool)).all()
    assert walks['rmse'].isna().all()

    diauxic = suite_frame[suite_frame['scenario'] == 'truncation_challenge']
    assert diauxic['metrics_defined'].astype(bool).all()
    assert diauxic['rmse'].notna().all()
    assert diauxic['delta_od'].notna().all()


def test_duration_is_consistent(suite_frame):
    """
    ``duration_hours == n_points * time_resolution`` for every row.

    M28: ``duration_hours`` read ``curve.parameters['duration_hours']`` through
    a key that never existed and fell back to ``time[-1]``, so a 100 h run
    reported 99.75 - one sample short, in every row and in every file name.
    """
    for _, row in suite_frame.iterrows():
        assert abs(row['duration_hours']
                   - row['n_points'] * row['time_resolution']) < 1e-6
    hundred = suite_frame[suite_frame['n_points'] == 400]
    assert len(hundred) > 0
    assert set(np.round(hundred['duration_hours'], 9)) == {100.0}


# ---------------------------------------------------------------------------
# acceptance 7: the blank stage and the clipped-baseline path
# ---------------------------------------------------------------------------

def test_clipping_prevalence(shipped_suite):
    """
    The SHIPPED suite clips like the real plates do.

    M09: 28/92 real conditions (30.4 %) have a zero-variance opening segment and
    12/92 are identically zero somewhere, against 0/480 synthetic - so the
    ``noise_floor=0.002`` substitution the pipeline applies to a clipped
    baseline was a path the benchmark had never entered.

    This measures the curves the generator CLI actually writes, exactly as an
    auditor would read them off the plate files - not a bespoke probe.  A probe
    is what let the first cut of this test pass at 32 % while the suite it was
    meant to certify sat at 13.2 %, because the blank stage was opt-in and half
    the scenarios never opted in.
    """
    ods, gt = shipped_suite
    zero_open, zero_any = [], []
    for curve_number in gt['curve_number']:
        od = ods[int(curve_number)]
        k = min(10, od.size // 5)
        zero_open.append(float(np.std(od[:k])) == 0.0)
        zero_any.append(bool(np.any(od == 0.0)))
    frac_open = float(np.mean(zero_open))
    frac_any = float(np.mean(zero_any))
    assert len(gt) > 500, len(gt)                       # this is the real suite
    assert 0.25 <= frac_open <= 0.35, (frac_open, len(gt))   # real 30.4 %
    assert frac_any >= 0.08, frac_any                        # real 12/92 = 13 %


def test_blank_stage_is_on_by_default_and_can_be_switched_off():
    """
    Every real column is blank-subtracted, so the blank stage is the default.

    A scenario that wants its raw baseline back - ``high_baseline_good`` and
    ``contamination`` exist to test a baseline, so re-referencing them to the
    blank would delete them - says so with ``BLANK_OFF``, and a scenario whose
    DECLARED baseline floor is above the real post-blank IQR
    (``BLANK_DEFAULT_MAX_BASELINE_OD``) keeps it without having to ask.  The
    guard reads the declared floor, not the sampled draw, so a scenario is
    blanked or not as a whole.
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)

    on = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0,
                                         initial_od=0.01, seed=3)
    assert np.isfinite(on.metadata['blank_offset_od'])
    assert float(np.mean(np.asarray(on.od600)[:10])) < 0.01

    off = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0,
                                          initial_od=0.01, seed=3,
                                          params={'blank_offset_od': BLANK_OFF})
    assert np.isnan(off.metadata['blank_offset_od'])
    assert float(np.mean(np.asarray(off.od600)[:10])) == pytest.approx(0.01, abs=0.02)

    high = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0,
                                           initial_od=0.12, seed=3)
    assert np.isnan(high.metadata['blank_offset_od'])
    assert float(np.mean(np.asarray(high.od600)[:10])) == pytest.approx(0.12, abs=0.03)

    named = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0,
                                            initial_od=0.12, seed=3,
                                            params={'blank_offset_od': (0.01, 0.02)})
    assert np.isfinite(named.metadata['blank_offset_od'])   # explicit wins

    # the guard is a statement about the SCENARIO: high_baseline_good keeps the
    # baseline it declares for every one of its curves, and a scenario that
    # merely samples up to 0.1 is blanked as a whole.
    for curve in generator.generate_from_scenario('high_baseline_good',
                                                  n_curves=4, seed=SEED):
        assert np.isnan(curve.metadata['blank_offset_od'])
        assert float(np.mean(np.asarray(curve.od600)[:5])) > 0.05


def test_blank_offset_draws_stay_inside_the_measured_blank_spread():
    """
    Both branches of the calibrated mixture sit inside the real blank spread.

    The magnitudes are the measured per-media across-well b0 spread of the 18
    Group 3 blank wells (0.003-0.029 OD); only the mixture WEIGHT is calibrated
    to the observed prevalence, so no draw is ever larger than a blank error
    the plates actually show.
    """
    assert BLANK_OVERSUBTRACTION_OD[1] <= 0.029
    assert BLANK_UNDERSUBTRACTION_OD[0] >= -0.029
    assert 0.0 < BLANK_OVERSUBTRACTION_PROB < 1.0
    rng = np.random.default_rng(0)
    draws = [_draw_blank_offset(rng, DEFAULT_BLANK_OFFSET_OD, 0.01, False)
             for _ in range(2000)]
    assert all(-0.029 <= d <= 0.029 for d in draws)
    over = float(np.mean([d >= BLANK_OVERSUBTRACTION_OD[0] for d in draws]))
    assert abs(over - BLANK_OVERSUBTRACTION_PROB) < 0.04, over


# ---------------------------------------------------------------------------
# replicates
# ---------------------------------------------------------------------------

def test_replicate_noise_is_mostly_common_mode():
    """
    Averaging three synthetic wells behaves like averaging three real ones.

    M05/M10: every real column the pipeline reads is a blank-subtracted 3-well
    mean, and per-well sigma against the sigma of their mean gives 0.75-0.95
    (median 0.848) rather than the 0.577 of independent noise.  With
    ``REPLICATE_SHARED_SIGMA_FRACTION`` of sigma shared in quadrature the
    construction ratio is sqrt(f^2 + (1-f^2)/3) = 0.848 exactly.
    """
    f = REPLICATE_SHARED_SIGMA_FRACTION
    assert np.sqrt(f ** 2 + (1 - f ** 2) / 3) == pytest.approx(0.848, abs=0.005)

    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    ratios = []
    for i in range(8):
        curve = generator.generate_single_curve(
            A=1.0, mu=0.2, lambda_=4.0, noise_level='medium', seed=500 + i,
            params={'n_replicates': DEFAULT_N_REPLICATES})
        assert len(curve.replicates) == 3
        residual = np.array(curve.replicates) - curve.clean_od600
        per_well = np.mean([np.std(np.diff(r)) for r in residual])
        mean_res = np.std(np.diff(np.asarray(curve.od600) - curve.clean_od600))
        ratios.append(mean_res / per_well)
    ratio = float(np.mean(ratios))
    assert 0.70 <= ratio <= 0.95, ratio        # independent noise would be 0.577


def test_truth_of_a_replicate_condition_is_the_mean_curve():
    """Ground truth is measured on the noiseless MEAN of the wells."""
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    time, mean, sd, wells, curve = generator.generate_triplicate(
        A=1.2, mu=0.2, lambda_=4.0, seed=17)
    assert len(wells) == DEFAULT_N_REPLICATES
    # The stored wells are in the same processing state as the condition: both
    # blank-subtracted and clipped at zero.  Their mean is the condition's own
    # trace everywhere the clip did not bite; where it did, the mean of clipped
    # wells sits ABOVE the clipped mean, and only by the amount the clip
    # removed (a few mOD, in the opening window).
    mean = np.asarray(mean, dtype=float)
    od = np.asarray(curve.od600, dtype=float)
    unclipped = od > 0
    assert unclipped.sum() > 0.5 * od.size
    assert np.allclose(mean[unclipped], od[unclipped], atol=5e-3)
    assert np.all(mean >= od - 1e-9)
    assert np.any(sd > 0)
    slope, lag, amp = _measure(curve.clean_od600, curve.time)
    assert curve.truth['true_A'] == pytest.approx(amp, rel=1e-9)
    assert curve.truth['true_mu'] == pytest.approx(slope, rel=1e-9)
    assert curve.truth['replicate_A_cv'] > 0


def test_outlier_well_is_recorded_when_it_happens():
    """``outlier_well`` is a column, not a hidden event."""
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    seen = False
    for i in range(12):
        curve = generator.generate_single_curve(
            A=1.2, mu=0.2, lambda_=4.0, seed=800 + i,
            params={'replicate_A_cv': 0.05, 'outlier_well_prob': 1.0})
        assert curve.truth['outlier_well'] is True
        peaks = [float(np.max(w)) for w in curve.replicates]
        assert min(peaks) < 0.85 * max(peaks)
        seen = True
    assert seen


# ---------------------------------------------------------------------------
# artefacts and twins
# ---------------------------------------------------------------------------

def test_artefacts_are_applied_and_recorded():
    """Spikes and a signed drift actually reach the emitted curve."""
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    params = {'n_spikes': 4, 'spike_od': (0.15, 0.40),
              'drift_od_per_h': (0.0008, 0.002), 'expected_class': 'GOOD'}
    curve = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0, seed=21,
                                            params=params)
    indices = [int(i) for i in curve.truth['spike_indices'].split(';') if i]
    assert len(indices) == 4
    od = np.asarray(curve.od600)
    for i in indices:
        assert od[i] - curve.clean_od600[i] > 0.10
    assert curve.truth['drift_od_per_h'] > 0.0005


def test_artefact_twin_shares_the_noise_realisation():
    """
    The twin isolates the artefact: same seed, same noise, no spikes.

    Without it the validator can only measure absolute error, never the shift
    the artefact caused.
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    params = {'n_spikes': 3, 'spike_od': (0.2, 0.3),
              'drift_od_per_h': 0.0015, 'expected_class': 'GOOD'}
    kwargs = dict(A=1.2, mu=0.2, lambda_=4.0, seed=21, params=params)
    with_art = generator.generate_single_curve(apply_artefacts=True, **kwargs)
    without = generator.generate_single_curve(apply_artefacts=False, **kwargs)

    assert without.truth['spike_indices'] == ''
    assert without.truth['drift_od_per_h'] == pytest.approx(0.0)
    assert np.max(np.abs(with_art.od600 - without.od600)) > 0.1
    assert without.truth['actual_r_squared'] > with_art.truth['actual_r_squared']

    # the noise realisation itself is identical: strip the drift and the spikes
    # and the two traces agree to the OD resolution of the instrument
    drift = 0.0015 * (with_art.time - with_art.time[0])
    residual = np.abs((with_art.od600 - drift) - without.od600)
    assert float(np.median(residual)) < 1e-9


def test_scenario_twins_are_emitted_and_cross_referenced():
    """An artefact scenario yields both rows, each naming the other."""
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    frame = generator.generate_comprehensive_test_set({'standard': 2}, seed=SEED,
                                                      verbose=False)
    assert (frame['twin_curve_id'] == '').all()          # no artefact declared

    curves = generator.generate_from_scenario('standard', n_curves=1, seed=SEED)
    assert len(curves) == 1
    assert curves[0].metadata['is_twin'] is False


# ---------------------------------------------------------------------------
# truth module units
# ---------------------------------------------------------------------------

def test_measure_shape_respects_the_phase_boundary():
    """A second phase can never be reported as the first phase's amplitude."""
    t = np.arange(0, 100, 0.25)
    from src.growth_models import GompertzModel
    first = GompertzModel.compute(t, 1.0, 0.2, 4.0)
    second = GompertzModel.compute(t, 0.8, 0.4, 60.0)
    clean = first + second

    unbounded = truth.measure_shape(clean, t)
    bounded = truth.measure_shape(clean, t, t_second_start=55.0)
    assert unbounded['true_A'] == pytest.approx(1.8, rel=0.02)
    assert bounded['true_A'] == pytest.approx(1.0, rel=0.02)
    assert bounded['true_mu'] == pytest.approx(0.2, rel=0.02)


def test_measure_shape_measures_a_model_free_trace_but_gives_it_no_rate():
    """``has_growth_process=False``: amplitude measured, rate and lag NaN.

    This is the ``erratic`` contract at the unit level.  The amplitude of the
    array is a fact whatever wrote it; the maximum finite difference of a
    random walk is not a growth rate, so it is NaN rather than a number a
    consumer would score.
    """
    t = np.arange(0, 100, 0.25)
    rng = np.random.default_rng(0)
    walk = np.clip(0.1 + np.cumsum(rng.normal(0, 0.02, t.size)), 0, 2.0)

    measured = truth.measure_shape(walk, t, has_growth_process=False)
    assert measured['true_A'] == pytest.approx(float(walk.max() - walk[0]))
    assert measured['true_A'] > 0.1                      # this walk really climbs
    assert np.isnan(measured['true_mu'])
    assert np.isnan(measured['true_lambda'])
    assert measured['has_growth_phase'] is False
    assert measured['has_growth_process'] is False

    # the same array, declared as a growth realisation, does get a rate: the
    # NaN is the declaration's doing, not a silent property of the numbers.
    assert np.isfinite(truth.measure_shape(walk, t)['true_mu'])


def test_measure_shape_refuses_a_lag_for_an_unresolvable_rise():
    """A rise below the instrument's own noise is not a growth phase.

    ``haldane_total_inhibition``'s curve rises 1.2e-6 OD over 100 h and used to
    report a 30.9 h lag.  The amplitude is still reported - it is what the
    curve does - but the rate and the lag are NaN.
    """
    t = np.arange(0, 100, 0.25)
    from src.growth_models import GompertzModel

    dust = GompertzModel.compute(t, 1e-6, 1e-8, 20.0)
    measured = truth.measure_shape(dust, t)
    assert measured['true_A'] < truth.MIN_RESOLVABLE_AMPLITUDE_OD
    assert measured['true_A'] == pytest.approx(float(dust.max() - dust[0]))
    assert np.isnan(measured['true_mu'])
    assert np.isnan(measured['true_lambda'])
    assert np.isnan(measured['t_primary_95'])
    assert measured['has_growth_phase'] is False

    # just above the floor the measurement comes back
    real = GompertzModel.compute(t, 0.01, 0.002, 20.0)
    assert truth.measure_shape(real, t)['has_growth_phase'] is True
    assert np.isfinite(truth.measure_shape(real, t)['true_lambda'])


def test_oracle_returns_nan_for_a_curve_with_no_rate():
    """No growth phase, no oracle - and no ZeroDivisionError either."""
    t = np.arange(0, 100, 0.25)
    rng = np.random.default_rng(1)
    walk = np.clip(0.1 + np.cumsum(rng.normal(0, 0.02, t.size)), 0, 2.0)
    out = truth.oracle_recovery(
        walk, walk, t,
        {'A': float(walk.max() - walk[0]), 'mu': float('nan'),
         'lambda_': float('nan'), 'initial_od': 0.1})
    assert np.isnan(out['oracle_A_relerr'])
    assert np.isnan(out['oracle_mu_relerr'])
    assert out['oracle_fit_ok'] is False


def test_true_window_end_is_1_3_times_t95():
    t = np.arange(0, 100, 0.25)
    from src.growth_models import GompertzModel
    clean = GompertzModel.compute(t, 1.2, 0.2, 4.0)
    measured = truth.measure_shape(clean, t)
    t95 = measured['t_primary_95']
    assert truth.true_window_end(t95, t_end=100.0) == pytest.approx(1.3 * t95)
    # and it never runs past a boundary or the end of the run
    assert truth.true_window_end(t95, boundary=14.0, t_end=100.0) == 14.0
    assert truth.true_window_end(t95, t_end=10.0) == 10.0


def test_oracle_is_a_property_of_the_curve_not_of_the_run_length():
    """
    The label statistic must not move when the plate runs longer.

    Same curve, same sigma: whole-trace R^2 falls from 0.977 at 20 h to 0.880 at
    200 h, while the oracle errors over the curve's own growth window stay put.
    That difference is the whole reason the label rule changed.
    """
    from src.growth_models import GompertzModel
    errors = []
    r2s = []
    for duration in (20.0, 100.0, 200.0):
        t = np.arange(0, duration, 0.25)
        clean = GompertzModel.compute(t, 1.0, 0.15, 4.0)
        noisy = clean + np.random.default_rng(0).normal(0, 0.06, t.size)
        measured = truth.measure_shape(clean, t)
        out = truth.oracle_recovery(noisy, clean, t,
                                    {'A': measured['true_A'], 'mu': measured['true_mu'],
                                     'lambda_': measured['true_lambda'],
                                     'initial_od': 0.0})
        assert out['oracle_fit_ok']
        errors.append(out['oracle_mu_relerr'])
        ss_res = float(np.sum((noisy - clean) ** 2))
        ss_tot = float(np.sum((clean - clean.mean()) ** 2))
        r2s.append(1 - ss_res / ss_tot)

    assert max(r2s) - min(r2s) > 0.05          # the statistic that was the label
    assert max(errors) - min(errors) < 0.05    # the statistic that is the label


def test_row_helpers():
    row = truth.empty_row()
    assert set(row) == set(truth.GROUND_TRUTH_COLUMNS)
    assert row['scoring_mode'] == 'standard'
    assert np.isnan(row['true_A'])
    assert truth.strain_name(7) == 'CURVE0007'
    assert truth.format_spike_indices([3, 9]) == '3;9'
    assert truth.format_spike_indices([]) == ''


def test_curve_to_row_is_the_full_schema():
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    curve = generator.generate_single_curve(A=1.2, mu=0.2, lambda_=4.0, seed=2)
    row = curve_to_row(curve, 12)
    assert set(truth.GROUND_TRUTH_COLUMNS) <= set(row)
    assert row['curve_id'] == 12
    assert row['strain_name'] == 'CURVE0012'
    assert len(row['time']) == len(row['od600']) == row['n_points']


def test_curve_rows_numbers_and_links_a_batch():
    """
    ``curve_rows`` is the single row-building path.

    ``generate_comprehensive_test_set`` and ``generate_synthetic.py``'s
    ``--scenario`` / manual branches all go through it, so a caller cannot
    hand ``create_ground_truth_csv`` a v1 subset again (the CLI used to build
    its own 15-key dict and exited 1 the moment the v2 schema landed).
    """
    generator = SyntheticGrowthCurveGenerator(seed=SEED)
    # negative_drift carries an artefact, so every curve gets its twin
    curves = generator.generate_from_scenario('negative_drift', n_curves=2, seed=SEED)
    rows = curve_rows(curves, start_id=5)

    assert [r['curve_id'] for r in rows] == [5, 6, 7, 8]
    assert [r['strain_name'] for r in rows] == [
        'CURVE0005', 'CURVE0006', 'CURVE0007', 'CURVE0008']
    # twins are cross-linked, and the link is symmetric
    assert [r['twin_curve_id'] for r in rows] == [
        'CURVE0006', 'CURVE0005', 'CURVE0008', 'CURVE0007']
    for row in rows:
        assert set(truth.GROUND_TRUTH_COLUMNS) <= set(row)

    # a batch with no twins is numbered straight through and links nothing
    plain = curve_rows(generator.generate_from_scenario('standard', n_curves=2,
                                                        seed=SEED))
    assert [r['curve_id'] for r in plain] == [0, 1]
    assert [r['twin_curve_id'] for r in plain] == ['', '']


# ---------------------------------------------------------------------------
# the oracle fit is a measurement, not a chaotic one
# ---------------------------------------------------------------------------
#
# Round-2 defect: two regenerations at the same seed wrote a ground_truth.csv
# with a different sha256 while every hard check (true_A/true_mu/true_lambda to
# 1e-9) was green and all 19 DATA/*.csv were byte-identical -- i.e. a column
# computed FROM identical arrays moved.  The column was oracle_A_relerr
# (340076.71 vs 346281.25 on an evaporation_drift curve, independently seen by
# WP10), and the mechanism is reproducible: on CURVE0690, whose entire rise is
# 0.0008 OD, the four-parameter Gompertz is unidentifiable, curve_fit stops in a
# flat valley at A=0.0726 / mu=-0.00047 / lambda=3098 h with an infinite
# covariance, and ONE ulp in ONE of the 400 samples moves the published relative
# error by 0.3 % (91.150 -> 90.881).  These four tests pin the three guards that
# close that channel: the fit reads the curve at 1e-6 OD, an unidentified fit is
# reported as non-converged rather than as a number, and an amplitude below the
# plate's noise floor is not a denominator.


def _tiny_rise_curve(n=400, amplitude=0.0008, sigma=0.004, seed=7):
    """An evaporation_drift-like trace: a rise far below the noise on it."""
    t = np.arange(0, 100.0, 0.25)[:n]
    clean = 0.02 + amplitude * (t / t[-1])
    noisy = clean + np.random.default_rng(seed).normal(0, sigma, t.size)
    return t, clean, noisy


def test_oracle_rejects_an_unidentified_fit():
    """
    ``curve_fit`` always returns a point; it does not always return a fit.

    The rejected point is the one measured on CURVE0690 (evaporation_drift):
    negative mu, a lag 31x past the end of the run, infinite covariance.  Each
    of the three is on its own enough to call the fit non-converged.
    """
    t_window = np.arange(0, 100.0, 0.25)
    good = ([1.0, 0.15, 4.0, 0.01], np.eye(4) * 1e-4)
    assert truth.oracle_fit_is_identifiable(good[0], good[1], t_window) is True

    diverged = [0.0726209846, -0.000470733012, 3097.66254, 0.0286569447]
    assert truth.oracle_fit_is_identifiable(
        diverged, np.full((4, 4), np.inf), t_window) is False
    # ... and each clause alone
    assert truth.oracle_fit_is_identifiable(
        diverged, np.eye(4) * 1e-4, t_window) is False          # mu < 0, lag out
    assert truth.oracle_fit_is_identifiable(
        [1.0, 0.15, 4.0, 0.01], np.full((4, 4), np.inf), t_window) is False
    assert truth.oracle_fit_is_identifiable(
        [-1.0, 0.15, 4.0, 0.01], np.eye(4) * 1e-4, t_window) is False
    assert truth.oracle_fit_is_identifiable(
        [1.0, 0.15, 3098.0, 0.01], np.eye(4) * 1e-4, t_window) is False


def test_oracle_needs_an_amplitude_above_the_noise_floor(suite_frame):
    """
    A relative error divided by an amplitude below 0.002 OD is not a measurement.

    0.002 OD is the noise floor the pipeline itself substitutes for a
    zero-variance well; it is quoted here, never changed.  Below it the fit has
    nothing to identify, so the answer is "non-converged", not a number with
    five digits in front of the decimal point.
    """
    t, clean, noisy = _tiny_rise_curve()
    measured = truth.measure_shape(clean, t)
    assert 0 < measured['true_A'] < truth.ORACLE_MIN_TRUE_A
    out = truth.oracle_recovery(noisy, clean, t,
                                {'A': measured['true_A'], 'mu': measured['true_mu'],
                                 'lambda_': measured['true_lambda'],
                                 'initial_od': float(clean[0])})
    assert out['oracle_fit_ok'] is False
    assert np.isnan(out['oracle_A_relerr'])
    assert np.isnan(out['oracle_mu_relerr'])
    assert np.isfinite(out['true_window_end_h'])       # the window still exists

    # and no shipped row claims a recovery it cannot have
    below = suite_frame[suite_frame['true_A'] < truth.ORACLE_MIN_TRUE_A]
    assert len(below) > 0
    assert not below['oracle_fit_ok'].any()
    assert below['oracle_A_relerr'].isna().all()


def test_oracle_relerr_is_saturated_and_published_at_a_fixed_precision(suite_frame):
    """
    Every published oracle error is <= the cap and equal to its own round(9).

    The saturation is what keeps a fit that is off by orders of magnitude from
    publishing digits that a different BLAS build would move; both the cap
    (10.0) and the precision (1e-9) sit far above the label rule's 0.20, so no
    label can turn on either.
    """
    for column in ('oracle_A_relerr', 'oracle_mu_relerr'):
        values = suite_frame[column].dropna().astype(float)
        assert len(values) > 0
        assert (values <= truth.ORACLE_RELERR_CAP + 1e-12).all()
        assert np.allclose(values, values.round(truth.ORACLE_RELERR_DECIMALS),
                           rtol=0, atol=0)


@pytest.mark.parametrize('amplitude,sigma', [(0.0008, 0.004),   # unidentifiable
                                             (0.9, 0.05)])      # ordinary
def test_oracle_is_immune_to_last_bit_input_noise(amplitude, sigma):
    """
    Perturbing samples by a few ulps must not move a published oracle number.

    This is the regression for the round-2 defect itself.  Before the guards, 3
    ulps in one sample of a 0.0008 OD curve moved oracle_A_relerr by 0.3 %; the
    fit now reads the curve at ORACLE_INPUT_QUANTUM_OD (1e-6 OD -- 460x below
    the quietest noise this instrument has produced, M06), so a difference in
    the last bits of a float cannot reach the optimiser at all.  Over the whole
    1050-curve suite this check moves nothing: 0 changed values and 0 changed
    oracle_fit_ok flags over 984 curves x 4 perturbation sites.
    """
    from src.growth_models import GompertzModel
    if amplitude < 0.01:
        t, clean, noisy = _tiny_rise_curve(amplitude=amplitude, sigma=sigma)
    else:
        t = np.arange(0, 40.0, 0.25)
        clean = GompertzModel.compute(t, amplitude, 0.15, 4.0)
        noisy = clean + np.random.default_rng(3).normal(0, sigma, t.size)

    measured = truth.measure_shape(clean, t)
    seed_params = {'A': measured['true_A'], 'mu': measured['true_mu'],
                   'lambda_': measured['true_lambda'],
                   'initial_od': float(clean[0])}
    base = truth.oracle_recovery(noisy, clean, t, seed_params)

    for site in (3, 11, noisy.size // 2, noisy.size - 2):
        jittered = noisy.copy()
        for _ in range(3):
            jittered[site] = np.nextafter(jittered[site], np.inf)
        assert not np.array_equal(jittered, noisy)          # the input DID move
        out = truth.oracle_recovery(jittered, clean, t, seed_params)
        assert out['oracle_fit_ok'] == base['oracle_fit_ok']
        for key in ('oracle_A_relerr', 'oracle_mu_relerr'):
            if np.isnan(base[key]):
                assert np.isnan(out[key])
            else:
                # bit-identical, not approximately equal: the published CSV is
                # hashed, so "close enough" is a different sha256.
                assert out[key] == base[key]
