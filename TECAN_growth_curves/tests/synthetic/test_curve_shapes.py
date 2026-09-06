"""WP4 acceptance tests for synthetic_data/src/curve_shapes.py.

Every threshold asserted here is a measured real-plate number; the module
docstring of ``curve_shapes.py`` names the condition each one came from.
Reference values used below:

    post-t=50 h gain, 31 real conditions      >0.05 OD 58.1 %, >0.10 OD 41.9 %,
    with dOD>0.5 and run>=90 h                >0.20 OD 22.6 %, median +0.0737,
                                              max +0.516 (G4 PERMETHRINANDLB_CYPERM2)
    end-of-run loss, 48 real growers          median 0.005, p75 0.076, p90 0.186,
                                              max 0.372 (G4 LB_PERM4), 1/48 >0.30
    G4 LB_PERM4 crash                         1.880 through ~53 h, steepest 4 h
                                              window -0.507 of the 0.70 OD total
                                              from 57.5 h, 1.180 by 70 h, flat to 96 h
    G2 MALATHIONANDLB_MAL8 excursion          peak 1.612 at 10 h, drawdown 0.502 OD
                                              by 27.5 h, recovery +0.297 OD by 100 h
    G4 PERMETHRINANDLB_CYPERM2 clipping       26.4 % exact zeros, zero through ~30 h,
                                              rising from ~40 h to 0.688 OD

A note on one acceptance item.  "Every registered shape returns finite
t_primary_95 and either t_second_start or t_death, and t_primary_95 < that
boundary" cannot hold for two of the nine registered shapes: ``gompertz`` (the
passthrough) and ``clipped_then_grows`` are single-phase by construction and
have neither a second phase nor a decline.  Inventing a boundary for them
would put a number into ground truth that nothing in the curve corresponds to,
so they report NaN for both and the test asserts the honest invariant instead:
a finite t_primary_95, NaN boundaries, and a true_window_end_h that is exactly
min(1.3*t_primary_95, end of run) - the last clause because a clipped curve
that is still rising at its final point genuinely has no completed primary
phase inside the run.  The strong form is asserted for the other seven shapes.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from src.curve_shapes import (                                   # noqa: E402
    CRASH_CENTRE_H,
    CRASH_LOSS_FRACTION,
    CRASH_WIDTH_H,
    DEATH_OFFSET_MODERATE_H,
    DEATH_OFFSET_SEVERE_H,
    DIP_DRAWDOWN_OD,
    LYSIS_LOSS_FRACTION,
    MAX_DEATH_LOSS_FRAC,
    SHAPES,
    SHAPES_WITH_BOUNDARY,
    SHAPE_META_KEYS,
    build_clean_curve,
    correct_cut_band,
    gompertz_mu_for_span,
    sigmoid_step,
    t95_gompertz,
    t_at_fraction_of_A,
)

#: 100 h at the lab's only real sampling rate (0.25 h), the geometry of
#: 390 of the 480 shipped synthetic curves.
T = np.arange(0, 100, 0.25)

#: The real GOOD amplitude band: real GOOD gompertz_a has median 1.49 and
#: max 1.87, and the highest real condition is 1.94 (G3 IMIDACLOPRIDANDLB_IMID3).
REAL_A_BAND = (1.0, 1.9)

REGISTERED = (
    'gompertz',
    'two_phase_gompertz',
    'diauxic_second_larger',
    'slow_secondary_drift',
    'partial_lysis',
    'sudden_crash',
    'dip_and_recover',
    'clipped_then_grows',
    'death_by_target_loss',
)


def _params(**kw):
    """Typical grower parameters plus a seeded stream for the shape's draws."""
    p = dict(A=1.3, mu=0.25, lambda_=3.0, rng=np.random.default_rng(kw.pop('seed', 0)))
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_exactly_the_contracted_shapes():
    assert set(SHAPES) == set(REGISTERED)


def test_unknown_shape_raises_and_names_the_alternatives():
    with pytest.raises(ValueError) as exc:
        build_clean_curve('exponential_kink', T, _params())
    assert 'two_phase_gompertz' in str(exc.value)


@pytest.mark.parametrize('shape', REGISTERED)
def test_curve_is_finite_non_negative_and_on_the_caller_grid(shape):
    od, meta = build_clean_curve(shape, T, _params(seed=3))
    assert od.shape == T.shape
    assert np.all(np.isfinite(od))
    assert od.min() >= 0.0
    assert set(SHAPE_META_KEYS) <= set(meta)


@pytest.mark.parametrize('shape', REGISTERED)
def test_same_seed_gives_the_same_curve_and_params_are_not_mutated(shape):
    p1 = _params(seed=11)
    p2 = _params(seed=11)
    before = {k: v for k, v in p1.items() if k != 'rng'}
    od1, m1 = build_clean_curve(shape, T, p1)
    od2, m2 = build_clean_curve(shape, T, p2)
    assert np.array_equal(od1, od2)
    assert m1['t_primary_95'] == m2['t_primary_95']
    assert {k: v for k, v in p1.items() if k != 'rng'} == before


def test_module_never_touches_the_global_numpy_seed():
    """WP2/WP3 rule, restated for this module: one explicit stream per call."""
    text = (SYNTH / 'src' / 'curve_shapes.py').read_text()
    code = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    assert not re.search(r'(np|numpy)\.random\.seed\s*\(', code)
    assert not re.search(r'(np|numpy)\.random\.(uniform|normal|choice)\s*\(', code)


# ---------------------------------------------------------------------------
# Acceptance 1: shape_meta completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES_WITH_BOUNDARY)
def test_shape_meta_is_complete(shape):
    """Finite t_primary_95, a finite boundary after it, and a scoreable band."""
    for seed in range(25):
        od, meta = build_clean_curve(shape, T, _params(seed=seed))
        t95 = meta['t_primary_95']
        assert np.isfinite(t95), f'{shape} has no primary completion time'
        boundaries = [meta['t_second_start'], meta['t_death']]
        finite = [b for b in boundaries if np.isfinite(b)]
        assert finite, f'{shape} declares neither t_second_start nor t_death'
        assert t95 < min(finite), (
            f'{shape}: primary completion {t95:.2f} h is not before its '
            f'boundary {min(finite):.2f} h')
        lo, hi = correct_cut_band(meta)
        assert lo == t95 and hi == min(finite) and lo < hi
        assert t95 < meta['true_window_end_h'] <= min(finite)


@pytest.mark.parametrize('shape', ['gompertz', 'clipped_then_grows'])
def test_single_phase_shapes_declare_no_boundary(shape):
    """The honest form of acceptance 1 for the two single-phase shapes."""
    od, meta = build_clean_curve(shape, T, _params(seed=5))
    assert np.isfinite(meta['t_primary_95'])
    assert np.isnan(meta['t_second_start'])
    assert np.isnan(meta['t_death'])
    assert meta['true_window_end_h'] == pytest.approx(
        min(1.3 * meta['t_primary_95'], float(T[-1])))
    if meta['t_primary_95'] <= T[-1]:
        assert meta['true_window_end_h'] > meta['t_primary_95']
    lo, hi = correct_cut_band(meta)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_t_primary_95_is_the_closed_form_and_matches_the_written_curve():
    """t95 is a property of the curve, not of any truncation rule."""
    A, mu, lam = 1.3, 0.25, 3.0
    od, meta = build_clean_curve('gompertz', T, dict(A=A, mu=mu, lambda_=lam))
    closed = lam + (A / (mu * np.e)) * (1.0 - np.log(-np.log(0.95)))
    assert meta['t_primary_95'] == pytest.approx(closed, rel=1e-12)
    # ... and the curve really is at 95 % of A there (linear interpolation on
    # the 0.25 h lab grid costs ~1e-4 OD on this curvature).
    assert np.interp(closed, T, od) == pytest.approx(0.95 * A, abs=5e-4)
    assert t95_gompertz(A, mu, lam) == pytest.approx(closed, rel=1e-12)
    assert t_at_fraction_of_A(0.5, A, mu, lam) < closed


def test_window_end_is_clipped_by_the_second_phase():
    """A curve that starts phase 2 early loses window, not honesty."""
    od, meta = build_clean_curve(
        'two_phase_gompertz', T,
        dict(A=1.2, mu=0.25, lambda_=2.0, A2=0.5, gap_hours=1.0,
             rng=np.random.default_rng(0)))
    assert meta['true_window_end_h'] == pytest.approx(meta['t_second_start'])
    assert meta['true_window_end_h'] < 1.3 * meta['t_primary_95']


# ---------------------------------------------------------------------------
# Acceptance 2: the second phase is a real sigmoid, not a kink
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shape', ['two_phase_gompertz', 'diauxic_second_larger',
                                   'slow_secondary_drift'])
def test_second_phase_is_smooth(shape):
    """Max slope inside phase 2 falls strictly after its onset.

    The replaced form, ``od2 = A2*(1 - exp(-mu2*(t - t0)))``
    (data_generator.py:389), has its maximum slope exactly at t0 - an abrupt
    kink.  A second Gompertz peaks at ``t_second_start + A2/(mu2*e)``.
    """
    for seed in range(60):
        od, meta = build_clean_curve(
            shape, T, _params(seed=seed, A=(0.5, 1.6), mu=(0.15, 0.35),
                              lambda_=(1.0, 5.0)))
        ts = meta['t_second_start']
        slope = np.gradient(od, T)
        after = T >= ts
        t_peak = T[after][int(np.argmax(slope[after]))]
        assert t_peak > ts + 0.25, (
            f'{shape} seed {seed}: max slope of phase 2 at {t_peak:.2f} h, '
            f'onset {ts:.2f} h - that is a kink, not a sigmoid')
        assert meta['extras']['second_phase_max_slope_time'] > ts


def test_old_exponential_second_phase_would_fail_this_test():
    """Pin the defect the new form fixes, so the test cannot silently pass."""
    t0 = 25.0
    od2 = np.where(T > t0, 0.5 * (1 - np.exp(-0.15 * (T - t0))), 0.0)
    slope = np.gradient(od2, T)
    after = T >= t0
    t_peak = T[after][int(np.argmax(slope[after]))]
    assert t_peak <= t0 + 0.25


def test_second_phase_amplitude_and_onset_come_from_the_caller():
    """Contract A: A2/mu2/lambda2/gap_hours are no longer hard-coded."""
    od_a, meta_a = build_clean_curve(
        'two_phase_gompertz', T,
        dict(A=1.0, mu=0.25, lambda_=2.0, A2=0.4, gap_hours=8.0,
             rng=np.random.default_rng(0)))
    od_b, meta_b = build_clean_curve(
        'two_phase_gompertz', T,
        dict(A=1.0, mu=0.25, lambda_=2.0, A2=0.9, gap_hours=20.0,
             rng=np.random.default_rng(0)))
    assert meta_a['extras']['A2'] == pytest.approx(0.4)
    assert meta_b['extras']['A2'] == pytest.approx(0.9)
    assert meta_b['t_second_start'] > meta_a['t_second_start'] + 11.0
    assert od_b[-1] - od_a[-1] == pytest.approx(0.5, abs=0.02)
    # an explicit lambda2 overrides the gap entirely
    _, meta_c = build_clean_curve(
        'two_phase_gompertz', T,
        dict(A=1.0, mu=0.25, lambda_=2.0, A2=0.4, gap_hours=8.0, lambda2=40.0,
             rng=np.random.default_rng(0)))
    assert meta_c['t_second_start'] == pytest.approx(40.0)


def test_diauxic_second_larger_really_is_larger():
    """WP5 samples A1 0.4-0.7 against A2 0.8-1.2; the shape must honour it."""
    for seed in range(20):
        od, meta = build_clean_curve(
            'diauxic_second_larger', T,
            _params(seed=seed, A=(0.4, 0.7), mu=(0.15, 0.3), lambda_=(1.0, 4.0)))
        assert meta['extras']['A2'] > meta['A1']
        assert od[-1] > 2 * meta['A1']


# ---------------------------------------------------------------------------
# Acceptance 3: the drift statistics match the real plates
# ---------------------------------------------------------------------------


def test_drift_matches_real_stats():
    """Real: 58.1 % of growers gain >0.05 OD after t=50 h, 22.6 % gain >0.20.

    Synthetic today: 17.3 % and 2.0 %, median +0.002 against a real +0.074.
    """
    rng = np.random.default_rng(20240906)
    gains = []
    for _ in range(200):
        od, meta = build_clean_curve(
            'slow_secondary_drift', T,
            dict(A=REAL_A_BAND, mu=(0.15, 0.35), lambda_=(1.0, 5.0), rng=rng))
        gains.append(od[-1] - od[np.searchsorted(T, 50.0)])
    gains = np.asarray(gains)

    assert (gains > 0.05).mean() >= 0.50, (gains > 0.05).mean()
    assert (gains > 0.20).mean() >= 0.20, (gains > 0.20).mean()
    # ... and never harder than the real plates ever were.
    assert gains.max() <= 0.516, f'max drift {gains.max():.3f} > real max 0.516'
    assert np.median(gains) >= 0.0737 * 0.5


def test_drift_amplitude_stays_inside_the_measured_band():
    """A2/A1 in [0.04, 0.25]: the measured post-t=50 gain fraction of dOD."""
    rng = np.random.default_rng(4)
    for _ in range(100):
        od, meta = build_clean_curve(
            'slow_secondary_drift', T,
            dict(A=REAL_A_BAND, mu=(0.15, 0.35), lambda_=(1.0, 5.0), rng=rng))
        frac = meta['extras']['A2'] / meta['A1']
        assert 0.04 - 1e-9 <= frac <= 0.25 + 1e-9, frac
        assert 30.0 <= meta['extras']['second_phase_span_h'] <= 80.0


def test_drift_second_phase_is_slower_than_the_first():
    """It is a creep, not a second sigmoid: G3 LB_IMID3 adds 0.45 OD over 40 h."""
    rng = np.random.default_rng(6)
    for _ in range(30):
        od, meta = build_clean_curve(
            'slow_secondary_drift', T,
            dict(A=REAL_A_BAND, mu=(0.15, 0.35), lambda_=(1.0, 5.0), rng=rng))
        assert meta['extras']['mu2'] < 0.1 * meta['mu1']


# ---------------------------------------------------------------------------
# Acceptance 4: decline magnitude is bounded by what the plates show
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('band,offset', [
    ((0.05, 0.20), DEATH_OFFSET_MODERATE_H),
    ((0.25, 0.45), DEATH_OFFSET_SEVERE_H),
])
def test_decline_is_bounded(band, offset):
    """Realised end-of-run loss lands in the requested band, always <= 0.50.

    Real ceiling over 48 growers is 0.372 (G4 LB_PERM4) with 1/48 above 0.30;
    today death_phase_severe realises a median loss of 0.979 and
    death_phase_moderate 0.718.
    """
    rng = np.random.default_rng(2024)
    losses = []
    for _ in range(200):
        od, meta = build_clean_curve(
            'death_by_target_loss', T,
            dict(A=(1.0, 1.5), mu=(0.15, 0.25), lambda_=(2.0, 5.0),
                 death_loss_frac=band, t_death_offset_h=offset, rng=rng))
        e = meta['extras']
        assert e['realised_loss_frac'] == pytest.approx(
            e['requested_loss_frac'], abs=5e-3)
        assert not e['out_of_distribution']
        losses.append(e['realised_loss_frac'])
    losses = np.asarray(losses)
    assert losses.min() >= band[0] - 5e-3
    assert losses.max() <= band[1] + 5e-3
    assert losses.max() <= MAX_DEATH_LOSS_FRAC


def test_decline_above_the_real_ceiling_is_clamped_unless_named_extreme():
    p = dict(A=1.2, mu=0.2, lambda_=3.0, death_loss_frac=0.98,
             rng=np.random.default_rng(0))
    with pytest.warns(RuntimeWarning, match='0.372'):
        od, meta = build_clean_curve('death_by_target_loss', T, dict(p))
    assert meta['extras']['realised_loss_frac'] == pytest.approx(0.50, abs=5e-3)

    named = dict(p, scenario_name='death_phase_extreme',
                 rng=np.random.default_rng(0))
    od_x, meta_x = build_clean_curve('death_by_target_loss', T, named)
    assert meta_x['extras']['realised_loss_frac'] == pytest.approx(0.98, abs=5e-3)
    assert meta_x['extras']['out_of_distribution'] is True


def test_death_onset_is_anchored_to_t95_not_to_the_run_length():
    """Today t_death is 39.9-69.8 h while t95 is 9.2-18.0 h and the median
    truncation time is 11.25 h, so the decline never touches the fit window."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        od, meta = build_clean_curve(
            'death_by_target_loss', T,
            dict(A=(1.0, 1.5), mu=(0.15, 0.25), lambda_=(2.0, 5.0),
                 t_death_offset_h=DEATH_OFFSET_SEVERE_H,
                 death_loss_frac=(0.25, 0.45), rng=rng))
        gap = meta['t_death'] - meta['t_primary_95']
        assert 0.5 - 1e-9 <= gap <= 3.0 + 1e-9
        assert meta['t_death'] < 25.0        # never a 40-70 h afterthought


@pytest.mark.parametrize('shape,band', [
    ('partial_lysis', LYSIS_LOSS_FRACTION),
    ('sudden_crash', CRASH_LOSS_FRACTION),
])
def test_step_declines_stay_inside_the_real_loss_range(shape, band):
    """13/92 real conditions lose 15-50 % of their peak; none loses more."""
    rng = np.random.default_rng(8)
    losses = []
    for _ in range(100):
        od, meta = build_clean_curve(
            shape, T, dict(A=(1.0, 1.9), mu=(0.15, 0.35), lambda_=(1.0, 5.0),
                           rng=rng))
        losses.append(meta['extras']['realised_loss_frac'])
    losses = np.asarray(losses)
    assert losses.min() >= 0.15
    assert losses.max() <= 0.50
    assert band[0] - 0.05 <= np.median(losses) <= band[1] + 0.05


def test_sudden_crash_reproduces_LB_PERM4():
    """G4 LB_PERM4: 1.880 through ~53 h, then ~10 h of transition to 1.180.

    The steepest 4 h window carries 0.507 of the 0.70 OD total; a sigmoid step
    of width 8-12 h carries 0.44-0.61 of its total over any 4 h at the centre,
    which is what fixes the width band (a 4 h cliff would carry ~1.0).
    """
    od, meta = build_clean_curve(
        'sudden_crash', T,
        dict(A=1.88, mu=0.3, lambda_=3.0, rng=np.random.default_rng(0)))
    e = meta['extras']
    assert CRASH_CENTRE_H[0] <= e['decline_centre_h'] <= CRASH_CENTRE_H[1]
    assert CRASH_WIDTH_H[0] <= e['decline_width_h'] <= CRASH_WIDTH_H[1]
    assert 45.0 <= meta['t_death'] <= 56.0          # the measured 45-55 h hold

    # flat before the crash and flat after it, like the real well
    pre = od[(T >= 30) & (T <= 40)]
    post = od[(T >= 85) & (T <= 96)]
    assert pre.max() - pre.min() < 1e-3
    assert post.max() - post.min() < 1e-3

    # the steepest 4 h window carries roughly half the loss, not all of it
    drop = pre.mean() - post.mean()
    k = 16                                           # 4 h at 0.25 h resolution
    window_drop = (od[:-k] - od[k:]).max()
    assert 0.40 <= window_drop / drop <= 0.65, window_drop / drop

    # and the sigmoid helper itself is calibrated: 0.507 measured, 0.529 at 10 h
    s = sigmoid_step(T, 50.0, 10.0)
    assert (s[T == 52.0][0] - s[T == 48.0][0]) == pytest.approx(0.53, abs=0.02)


# ---------------------------------------------------------------------------
# Acceptance 5: dip and recover
# ---------------------------------------------------------------------------


def test_dip_and_recover_reproduces_MAL8():
    """G2 MALATHIONANDLB_MAL8: peak 1.612 at 10 h, -0.502 OD by 27.5 h,
    +0.297 OD back by 100 h.  Six real growers do this; the generator has had
    no mechanism for it at all."""
    od, meta = build_clean_curve(
        'dip_and_recover', T,
        dict(A=1.61, mu=0.35, lambda_=2.0, rng=np.random.default_rng(0)))

    i_peak = int(np.argmax(od[T <= meta['t_death']]))
    peak = od[i_peak]
    trough_idx = i_peak + int(np.argmin(od[i_peak:]))
    trough = od[trough_idx]

    assert peak - trough >= 0.2, f'drawdown {peak - trough:.3f} OD'
    assert od[-1] - trough >= 0.15, f'recovery {od[-1] - trough:.3f} OD'
    assert od[-1] < peak                    # none of the six regains its peak
    assert np.isfinite(meta['t_death']) and np.isfinite(meta['t_second_start'])
    assert meta['t_primary_95'] <= meta['t_death'] < meta['t_second_start']
    assert T[trough_idx] < meta['t_second_start'] + 5.0


def test_dip_and_recover_holds_over_many_draws():
    rng = np.random.default_rng(17)
    draw, rec = [], []
    for _ in range(200):
        od, meta = build_clean_curve(
            'dip_and_recover', T,
            dict(A=(1.4, 1.7), mu=(0.25, 0.45), lambda_=(1.0, 4.0), rng=rng))
        e = meta['extras']
        draw.append(e['realised_drawdown_od'])
        rec.append(e['realised_recovery_od'])
        assert e['recovery_od'] < e['dip_od']
    draw = np.asarray(draw)
    rec = np.asarray(rec)
    assert draw.min() >= 0.15
    assert draw.max() <= 0.502 + 1e-9          # the real MAL8 drawdown
    assert rec.min() >= 0.10
    assert rec.max() <= 0.35 + 1e-9
    assert (draw >= 0.2).mean() >= 0.9
    assert DIP_DRAWDOWN_OD == (0.30, 0.50)


# ---------------------------------------------------------------------------
# clipped_then_grows
# ---------------------------------------------------------------------------


def test_clipped_then_grows_reproduces_CYPERM2():
    """G4 PERMETHRINANDLB_CYPERM2: 26.4 % exact zeros, zero through ~30 h,
    rising from ~40 h to 0.688 OD - a shape 0/480 synthetic curves have."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        od, meta = build_clean_curve('clipped_then_grows', T, dict(rng=rng))
        e = meta['extras']
        assert e['requires_zero_baseline'] is True
        zero_until = e['zero_until_h']
        assert 30.0 <= zero_until <= 40.0
        assert np.all(od[T <= zero_until] == 0.0)      # identically zero
        assert e['frac_zero_points'] >= 0.25           # real 26.4 %
        assert 0.50 - 1e-9 <= od[-1] <= 0.80 + 1e-9    # real 0.688
        assert np.std(od[:10]) == 0.0                  # the clipped-baseline path
        assert od[T > zero_until + 20.0][0] > 0.0      # and it does grow


def test_clipped_then_grows_amplitude_is_the_visible_one():
    """A1 describes the curve that was written, not the one that was asked for.

    The requested amplitude is the OD the VISIBLE (post-clip) trace ends the
    run on - 0.688 for G4 PERMETHRINANDLB_CYPERM2 - so ground truth can never
    claim an asymptote the run did not reach.
    """
    od, meta = build_clean_curve(
        'clipped_then_grows', T,
        dict(A=0.688, zero_until_h=30.0, rng=np.random.default_rng(1)))
    assert od[-1] == pytest.approx(0.688, rel=1e-6)
    assert meta['A1'] == pytest.approx(0.688, rel=1e-6)
    assert meta['extras']['A_total'] >= 0.688      # the unreached asymptote
    assert 0.0 < meta['extras']['frac_of_A_reached'] <= 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_gompertz_mu_for_span_delivers_the_requested_duration():
    for A, span in ((0.3, 40.0), (1.2, 12.0), (0.05, 75.0)):
        mu = gompertz_mu_for_span(A, span)
        t05 = t_at_fraction_of_A(0.05, A, mu, 10.0)
        t95 = t_at_fraction_of_A(0.95, A, mu, 10.0)
        assert t95 - t05 == pytest.approx(span, rel=1e-9)


def test_sigmoid_step_width_is_the_5_to_95_percent_time():
    s_lo = sigmoid_step(np.array([45.0]), 50.0, 10.0)[0]
    s_hi = sigmoid_step(np.array([55.0]), 50.0, 10.0)[0]
    assert s_lo == pytest.approx(0.05, abs=1e-9)
    assert s_hi == pytest.approx(0.95, abs=1e-9)


def test_degenerate_no_growth_is_reported_not_NaN():
    """A flat scenario routed through a declining shape must not write NaNs.

    flat_no_growth has A=0.0: there is no plateau to decline from, so the shape
    returns the curve as written and says so, rather than anchoring a step to a
    NaN t95 and filling ground truth with NaN OD values.
    """
    for shape in ('partial_lysis', 'sudden_crash', 'dip_and_recover',
                  'death_by_target_loss'):
        od, meta = build_clean_curve(
            shape, T, dict(A=0.0, mu=0.0, lambda_=0.0,
                           rng=np.random.default_rng(0)))
        assert np.all(np.isfinite(od))
        assert np.all(od == 0.0)
        assert meta['extras'].get('degenerate_no_growth') is True
        assert np.isnan(meta['t_death'])


def test_sudden_crash_accepts_a_t95_anchor_when_the_scenario_gives_one():
    """A scenario that anchors every decline to t95 gets that for crash too."""
    od, meta = build_clean_curve(
        'sudden_crash', T,
        dict(A=1.5, mu=0.25, lambda_=2.0, t_death_offset_h=4.0,
             rng=np.random.default_rng(0)))
    assert meta['t_death'] == pytest.approx(meta['t_primary_95'] + 4.0)
    assert 0.25 <= meta['extras']['realised_loss_frac'] <= 0.40


def test_accepts_a_sampled_scenario_parameter_dict():
    """WP3 dispatches the sampler's dict straight in; unknown keys are ignored."""
    sampled = {
        'A': 1.25, 'mu': 0.22, 'lambda_': 3.4, 'initial_od': 0.03,
        'model_type': 'gompertz', 'noise_level': 'medium',
        'include_death_phase': True, 'pattern': None,
        'expected_class': 'GOOD', 'scenario_name': 'death_phase_moderate',
        'death_loss_frac': 0.12, 't_death_offset_h': 5.0,
        'instrument_noise': False, 'seed': 4242,
    }
    od, meta = build_clean_curve('death_by_target_loss', T, sampled)
    assert meta['extras']['realised_loss_frac'] == pytest.approx(0.12, abs=5e-3)
    assert meta['t_death'] == pytest.approx(meta['t_primary_95'] + 5.0)
    # the baseline is NOT added here - the caller does that, as for every model
    assert od[0] < 0.01


def test_short_run_geometry_is_supported():
    """incomplete_24h uses the literal Group 1 geometry: 96 points, 23.75 h."""
    t24 = np.arange(0, 24, 0.25)
    for shape in REGISTERED:
        od, meta = build_clean_curve(shape, t24, _params(seed=1))
        assert od.shape == t24.shape
        assert np.all(np.isfinite(od))
        assert meta['true_window_end_h'] <= t24[-1] or np.isnan(
            meta['true_window_end_h'])
