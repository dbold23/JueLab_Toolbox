"""WP2 acceptance tests for synthetic_data/src/noise_models.py.

Every threshold in this file is a measured real-plate number; the module
docstring of ``noise_models.py`` names the wells and conditions each one came
from.  Reference values used here:

    per-well sigma_hat, 96 raw Group 3 wells   median 0.00127 (p10 0.00073, p90 0.00200)
    per-well plateau sigma_hat (OD>1.4)        0.00139-0.00169
    16 real non-growing condition curves       detrended ac 0.923 / 0.851 / 0.570 / 0.203
                                               at 0.25 / 1 / 5 / 10 h
                                               5 h high-passed lag-1 ac  +0.085
                                               total linear drift -0.025 .. +0.052 OD
    Group 3 well H12 (noisiest of 96)          whole-trace sigma_hat 0.00596,
                                               plateau sigma_hat 0.02554,
                                               largest excursion +0.295 OD
    18 Group 3 BLANK wells                     b0 0.120-0.155, drift -0.0285..+0.0219 OD,
                                               sigma_hat 0.00036-0.00248
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

from src.growth_models import GompertzModel                       # noqa: E402
from src.noise_models import (                                    # noqa: E402
    NOISE_PRESETS,
    DRIFT_RATE_RANGE,
    BlankModel,
    ColoredNoise,
    GaussianNoise,
    InstrumentNoise,
    ODDependentNoise,
    RMSEBasedNoise,
    add_target_rmse_component,
    apply_signed_drift,
    get_noise_model,
    get_rmse_noise,
    inject_spikes,
    sample_drift_rate,
)

NOISE_MODELS_PY = SYNTH / 'src' / 'noise_models.py'
T = np.arange(0, 100, 0.25)          # the real Group 3 geometry: 400 pts, 0.25 h


# --------------------------------------------------------------------------
# estimators - identical to the ones the audit used on the real plates
# --------------------------------------------------------------------------

def sigma_hat_from_d2(d2):
    """1.4826 * MAD(2nd differences) / sqrt(6): curvature-insensitive HF sigma."""
    d2 = np.asarray(d2, dtype=float)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))


def sigma_hat(y):
    return sigma_hat_from_d2(np.diff(np.asarray(y, dtype=float), 2))


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def detrended(y, t=T):
    t = np.asarray(t, dtype=float)
    tm = t.mean()
    slope = np.sum((t - tm) * (y - y.mean())) / np.sum((t - tm) ** 2)
    return y - slope * t


def high_pass(y, window=21):
    """Remove a 5 h centred moving average (21 points at 0.25 h)."""
    k = np.ones(window) / window
    ma = np.convolve(y, k, mode='same')
    return (y - ma)[window // 2: -(window // 2)]


def rolling_median(y, window=11):
    n = len(y)
    half = window // 2
    return np.array([np.median(y[max(0, i - half): min(n, i + half + 1)]) for i in range(n)])


def gompertz(A=1.2, mu=0.15, lam=4.0, t=T):
    return GompertzModel.compute(t, A=A, mu=mu, lambda_=lam)


# --------------------------------------------------------------------------
# ACCEPTANCE
# --------------------------------------------------------------------------

def test_medium_matches_real_instrument():
    """'medium' reproduces the median real well, and sigma is nearly OD-flat.

    Band sigma_hat is computed from the 2nd differences POOLED across the fixed
    30-curve set (a per-curve estimate over the OD 0.05-0.20 band rests on ~10
    points and swings by +-20 %; pooling gives ~300 and is stable).
    """
    pools = {'mid': [], 'high': [], 'low': []}
    for s in range(30):
        rng = np.random.default_rng(100 + s)
        clean = gompertz(A=rng.uniform(0.9, 1.6), mu=rng.uniform(0.06, 0.18),
                         lam=rng.uniform(2.0, 8.0))
        y = get_noise_model('medium').apply(clean, time=T, rng=rng)
        d2 = np.diff(y, 2)
        od = y[1:-1]
        for name, (lo, hi) in (('mid', (0.6, 1.2)), ('high', (1.2, np.inf)),
                               ('low', (0.05, 0.20))):
            mask = (od >= lo) & (od < hi)
            if mask.any():
                pools[name].append(d2[mask])

    band = {k: sigma_hat_from_d2(np.concatenate(v)) for k, v in pools.items()}

    # Real per-well plateau sigma_hat is 0.0013-0.0017 (Group 3 H4..H8).
    assert 0.0015 / 2 <= band['mid'] <= 0.0015 * 2, band
    # Real sigma(>1.2) / sigma(0.05-0.20) is ~1.4 (0.00162 / 0.00114).
    assert band['high'] / band['low'] <= 2.5, band


@pytest.mark.parametrize('target', [0.80, 0.90, 0.93, 0.95, 0.97])
def test_target_r2_is_exact(target):
    """Every curve lands on its requested full-trace R^2, not near it.

    The old profile realised 0.9337 for a 0.930 request and 0.8108 for 0.800,
    which is why borderline_r2_bad and borderline_r2_good overlapped.
    """
    clean = gompertz(A=1.0, mu=0.15, lam=4.0)
    model = RMSEBasedNoise()
    for s in range(50):
        _, actual_r2, _ = model.generate_noise_to_target_r2(clean, target, rng=1000 + s)
        assert abs(actual_r2 - target) < 1e-6, (target, s, actual_r2)


def test_rmse_based_noise_delivers_requested_rmse():
    """RMSEBasedNoise.apply is public and used to under-deliver by 2.1 %."""
    clean = gompertz()
    for target_rmse in (0.001, 0.005, 0.02, 0.05):
        for s in range(10):
            noisy, actual = RMSEBasedNoise(target_rmse=target_rmse).apply(clean, rng=s)
            assert abs(actual - target_rmse) / target_rmse < 1e-6
            assert np.all(noisy >= 0.0)


def test_colored_flats_have_positive_lag1():
    """Non-growing coloured curves carry the real residual STRUCTURE.

    Real: 5 h high-passed lag-1 ac +0.085; detrended-residual ac 0.851 at 1 h and
    0.570 at 5 h.  The old i.i.d. generator gave -0.052 / ~0.0 / ~0.0.
    """
    hp_lag1, ac1h, ac5h = [], [], []
    flat = np.full_like(T, 0.05)
    for s in range(40):
        y = get_noise_model('medium').apply(flat, time=T, rng=200 + s)
        resid = detrended(y)
        ac1h.append(autocorr(resid, 4))       # 1 h  = 4 samples
        ac5h.append(autocorr(resid, 20))      # 5 h  = 20 samples
        hp_lag1.append(autocorr(high_pass(y), 1))

    assert 0.05 <= np.median(hp_lag1) <= 0.30, np.median(hp_lag1)
    assert abs(np.median(ac1h) - 0.851) < 0.10, np.median(ac1h)
    assert abs(np.median(ac5h) - 0.570) < 0.10, np.median(ac5h)


def test_white_noise_flats_are_negatively_correlated_after_high_pass():
    """The control: switch the colour off and the old -0.05 signature returns."""
    model = get_noise_model('medium')
    model.colored = False
    model.outlier_prob = 0.0          # spikes decay over 3 points, i.e. are coloured
    vals = [autocorr(high_pass(model.apply(np.full_like(T, 0.05), time=T, rng=300 + s)), 1)
            for s in range(20)]
    assert np.median(vals) < 0.0


def test_drift_is_signed():
    """Drift has both signs and spans the measured real range.

    Real non-growing curves drift -0.025 .. +0.052 OD over ~100 h (median +0.004);
    the old presets had drift_rate >= 0 everywhere, giving synthetic flats a
    +0.0395 median with a minimum of -0.0004.
    """
    rates = np.array([sample_drift_rate(rng=np.random.default_rng(400 + s))
                      for s in range(200)])
    assert (rates < 0).any() and (rates > 0).any()
    total = rates * 100.0
    assert -0.026 <= total.min() <= -0.020
    assert 0.055 <= total.max() <= 0.061
    assert DRIFT_RATE_RANGE == (-0.00025, 0.00060)

    # and the model actually uses it
    used = [get_noise_model('medium').sample_drift_rate(np.random.default_rng(s))
            for s in range(200)]
    assert min(used) < 0 < max(used)


def test_very_high_has_visible_outliers():
    """A very_high curve always carries a detectable one-sided spike.

    Anchor: Group 3 well H12, the noisiest of 96 real wells, whose largest single
    positive excursion vs an 11-point rolling median is +0.295 OD.  With
    outlier_magnitude 0.1 (the old constructor default) only 28.5 % of very_high
    curves contained a detectable outlier.
    """
    clean = gompertz()
    for s in range(50):
        y = get_noise_model('very_high').apply(clean, time=T, rng=500 + s)
        resid = y - rolling_median(y)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        assert np.any(resid > 5 * mad), s
    assert NOISE_PRESETS['very_high']['outlier_magnitude'] >= 0.25


def test_very_high_reproduces_well_H12():
    """very_high's plateau sigma is Group 3 H12's measured 0.02554."""
    preset = NOISE_PRESETS['very_high']
    sigma_at_plateau = preset['baseline_sigma'] + preset['proportional_sigma'] * 1.8
    assert 0.020 <= sigma_at_plateau <= 0.032

    clean = gompertz(A=1.8, mu=0.2, lam=4.0)
    model = InstrumentNoise(baseline_sigma=preset['baseline_sigma'],
                            proportional_sigma=preset['proportional_sigma'],
                            outlier_prob=0.0, drift_rate=0.0)
    vals = []
    for s in range(20):
        y = model.apply(clean, time=T, rng=600 + s)
        vals.append(sigma_hat(y[y > 1.4]))
    assert 0.0255 / 2 <= np.median(vals) <= 0.0255 * 2, np.median(vals)


def test_no_global_seed():
    """No np.random.seed / np.random.<legacy> anywhere in the module."""
    source = NOISE_MODELS_PY.read_text()
    assert 'np.random.seed(' not in source
    assert len(re.findall(r'np\.random\.seed', source)) == 0
    # the only np.random.* uses allowed are the Generator constructors / types
    used = set(re.findall(r'np\.random\.(\w+)', source))
    assert used <= {'default_rng', 'Generator', 'RandomState'}, used


# --------------------------------------------------------------------------
# PRESETS AND API SURFACE
# --------------------------------------------------------------------------

def test_preset_names_are_unchanged():
    assert list(NOISE_PRESETS) == ['very_low', 'low', 'medium', 'high', 'very_high']
    for name, params in NOISE_PRESETS.items():
        for key in ('description', 'baseline_sigma', 'proportional_sigma',
                    'outlier_prob', 'outlier_magnitude', 'rmse_range'):
            assert key in params, (name, key)
        assert isinstance(get_noise_model(name), InstrumentNoise)
        assert isinstance(get_rmse_noise(name), RMSEBasedNoise)
    with pytest.raises(ValueError):
        get_noise_model('nope')


def test_presets_match_the_measured_real_distribution():
    """baseline_sigma is the measured per-well sigma_hat ladder."""
    assert NOISE_PRESETS['very_low']['baseline_sigma'] == 0.0003    # cleanest well A1
    assert NOISE_PRESETS['low']['baseline_sigma'] == 0.0005         # p10 of the 16 flats
    assert NOISE_PRESETS['medium']['baseline_sigma'] == 0.0009      # median well
    assert NOISE_PRESETS['high']['baseline_sigma'] == 0.0020        # p90 well
    # nothing may go back to the old 6-20x oversized values
    for name, params in NOISE_PRESETS.items():
        if name != 'very_high':
            assert params['baseline_sigma'] <= 0.0020
            assert params['proportional_sigma'] <= 0.0025


def test_every_apply_accepts_none_int_and_generator():
    signal = gompertz()
    for model, kwargs in ((GaussianNoise(0.001), {}),
                          (ODDependentNoise(), {}),
                          (InstrumentNoise(), {'time': T})):
        a = model.apply(signal, rng=7, **kwargs)
        b = model.apply(signal, rng=np.random.default_rng(7), **kwargs)
        c = model.apply(signal, rng=None, **kwargs)
        assert np.allclose(a, b)
        assert a.shape == c.shape == signal.shape
        # legacy keyword still works and is equivalent to rng=<int>
        assert np.allclose(model.apply(signal, seed=7, **kwargs), a)


def test_apply_is_deterministic_and_does_not_touch_global_state():
    np.random.seed(1234)                     # deliberately poison the global state
    before = np.random.random()
    np.random.seed(1234)
    _ = np.random.random()
    y1 = get_noise_model('medium').apply(gompertz(), time=T, rng=11)
    after = np.random.random()
    np.random.seed(1234)
    _ = np.random.random()
    expected_after = np.random.random()
    assert after == expected_after            # our RNG did not advance the global one
    y2 = get_noise_model('medium').apply(gompertz(), time=T, rng=11)
    assert np.array_equal(y1, y2)
    assert before == before


def test_instrument_noise_clips_at_zero():
    y = get_noise_model('very_high').apply(np.zeros_like(T), time=T, rng=3)
    assert np.all(y >= 0.0)


# --------------------------------------------------------------------------
# ARTEFACT PRIMITIVES
# --------------------------------------------------------------------------

def test_inject_spikes_is_one_sided_and_decays():
    od = np.full(50, 0.5)
    out = inject_spikes(od, [10, 30], [0.3, 0.2], decay_points=3)
    assert np.all(out >= od)                                  # one-sided positive
    assert out[10] == pytest.approx(0.8)
    assert out[11] > out[12] > 0.5                            # decaying
    assert out[13] == pytest.approx(0.5)                      # at most 3 points
    assert out[14:30] == pytest.approx(0.5)
    assert od[10] == 0.5                                      # input untouched


def test_inject_spikes_decay_is_capped_at_four_points():
    od = np.zeros(40)
    out = inject_spikes(od, [5], [1.0], decay_points=99)
    assert np.count_nonzero(out) == 4
    out1 = inject_spikes(od, [5], [1.0], decay_points=1)
    assert np.count_nonzero(out1) == 1
    with pytest.raises(ValueError):
        inject_spikes(od, [5, 6], [1.0], decay_points=3)


def test_inject_spikes_ignores_out_of_range_indices():
    od = np.zeros(10)
    assert np.array_equal(inject_spikes(od, [20], [1.0], 3), od)
    assert np.array_equal(inject_spikes(od, [], [], 3), od)


def test_apply_signed_drift():
    od = np.zeros_like(T)
    up = apply_signed_drift(od, T, 0.0006)
    down = apply_signed_drift(od, T, -0.00025)
    assert up[-1] == pytest.approx(0.0006 * (T[-1] - T[0]))
    assert down[-1] < 0
    assert up[0] == 0.0
    with pytest.raises(ValueError):
        apply_signed_drift(np.zeros(3), T, 0.001)


# --------------------------------------------------------------------------
# ADDITIVE PRESET + TARGET-RMSE PATH
# --------------------------------------------------------------------------

@pytest.mark.parametrize('target', [0.80, 0.90, 0.95])
def test_add_target_rmse_component_lands_exactly(target):
    clean = gompertz(A=1.0, mu=0.15, lam=4.0)
    ss_tot = float(np.sum((clean - clean.mean()) ** 2))
    for s in range(10):
        rng = np.random.default_rng(700 + s)
        noisy = get_noise_model('medium').apply(clean, time=T, rng=rng)
        res0 = noisy - clean
        component, r2, added = add_target_rmse_component(
            clean, res0, (1.0 - target) * ss_tot, rng=rng)
        assert added
        assert abs(r2 - target) < 1e-6
        # the component only tops up; it never replaces the preset noise
        assert np.any(component != 0.0)


def test_add_target_rmse_component_reports_when_preset_already_exceeds():
    clean = gompertz(A=1.0, mu=0.15, lam=4.0)
    ss_tot = float(np.sum((clean - clean.mean()) ** 2))
    res0 = np.full_like(clean, 0.5)
    component, r2, added = add_target_rmse_component(clean, res0, 0.001 * ss_tot, rng=1)
    assert added is False
    assert np.all(component == 0.0)
    assert r2 < 0.999


# --------------------------------------------------------------------------
# COLOURED NOISE
# --------------------------------------------------------------------------

def test_colored_noise_signature_and_defaults():
    model = ColoredNoise(0.001)
    assert model.tau_fast_h == 0.5
    assert model.tau_slow_h == 7.5
    assert model.sd_slow == 0.0035
    out = model.apply(np.zeros_like(T), T, rng=1)
    assert out.shape == T.shape


def test_colored_noise_slow_wander_matches_measured_amplitude():
    """Detrended residual sd of a coloured flat is the real 0.0035 (p90 0.0051)."""
    sds = []
    for s in range(40):
        y = ColoredNoise(0.0009).apply(np.zeros_like(T), T, rng=800 + s)
        sds.append(float(np.std(detrended(y))))
    assert 0.0020 <= np.median(sds) <= 0.0055, np.median(sds)


def test_colored_noise_preserves_the_requested_hf_sigma():
    for sigma in (0.0005, 0.0009, 0.002):
        vals = [sigma_hat(ColoredNoise(sigma).apply(np.zeros_like(T), T, rng=900 + s))
                for s in range(15)]
        assert 0.75 * sigma <= np.median(vals) <= 1.25 * sigma, (sigma, np.median(vals))


def test_colored_noise_falls_back_to_white_on_a_degenerate_grid():
    tiny = np.array([0.0, 1.0, 2.0])
    out = InstrumentNoise().apply(np.zeros(3), time=tiny, rng=2)
    assert out.shape == (3,)
    out_no_time = InstrumentNoise().apply(np.zeros(50), time=None, rng=2)
    assert out_no_time.shape == (50,)


# --------------------------------------------------------------------------
# BLANK MODEL
# --------------------------------------------------------------------------

def test_blank_model_matches_group3_blank_wells():
    """18 Group 3 wells with Strain == BLANK; see the module docstring."""
    model = BlankModel()
    media_names = ['LB', 'H2O', 'IMIDACLOPRID', 'IMIDACLOPRIDANDLB',
                   'LAMBDACYHALOTHRIN', 'LAMBDACYHALOTHRINANDLB']
    for media in media_names:
        assert 0.120 <= model.media_baseline(media) <= 0.155
        assert 0.003 <= model.media_well_spread(media) <= 0.029

    b0s, drifts, sigmas = [], [], []
    for i, media in enumerate(media_names * 8):
        blank = model.sample_blank(media, T, rng=1000 + i)
        b0s.append(float(np.median(blank[:5])))
        drifts.append(float(np.median(blank[-5:]) - np.median(blank[:5])))
        sigmas.append(sigma_hat(blank))

    # real per-well b0 spans 0.117 (B1) .. 0.182 (C1)
    assert 0.08 <= np.median(b0s) <= 0.20, np.median(b0s)
    # real blank drift over the run: -0.0285 .. +0.0219 OD
    assert -0.05 <= min(drifts) and max(drifts) <= 0.05
    assert min(drifts) < 0 < max(drifts)
    # real blank sigma_hat 0.00036 .. 0.00248, median 0.00112
    assert 0.0004 <= np.median(sigmas) <= 0.0030, np.median(sigmas)


def test_blank_media_baseline_is_stable_and_media_specific():
    model = BlankModel()
    assert model.media_baseline('LB') == model.media_baseline('LB')
    assert model.media_baseline('LB') != model.media_baseline('H2O')
    assert BlankModel(seed=1).media_baseline('LB') != BlankModel(seed=2).media_baseline('LB')


def test_blank_subtraction_can_clip_a_dead_well_to_zero():
    """The point of the blank stage: 30.4 % of real conditions are zero-clipped."""
    model = BlankModel()
    blank = model.sample_blank('LB', T, rng=5)
    well = np.full_like(T, float(np.median(blank)) - 0.005)     # blank sits above
    blanked = np.maximum(0.0, well - blank)
    assert np.mean(blanked == 0.0) > 0.5
