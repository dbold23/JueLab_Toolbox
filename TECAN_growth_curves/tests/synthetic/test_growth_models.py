"""
Shape round-trip tests for the synthetic growth-model converters (WP1).

Every ``from_gompertz_params`` in synthetic_data/src/growth_models.py exists
for one reason: to turn a sampled (A, mu, lambda) into parameters of another
model that *has* that shape, so the ground-truth table describes the curve
that was actually written. Before this package none of the four converters
did that:

    logistic  max slope 0.368*mu (a spurious 1/e), no lag, y(0) at 14-20 % of A
    baranyi   raw OD fed into a log-space model: 0.39*mu, lag ~0.3 h of 4 h
    richards  max slope 1.208*mu at the shipped nu=0.5
    haldane   Ks/Ki/q/S0 hard-coded, lambda_ accepted and never used

The tests below pin the three measurable properties of a converted curve --
plateau, maximum slope and tangent lag -- against the request, using the one
measurement convention every consumer shares (``growth_models.measure_shape``:
max of np.gradient, tangent intercept at the argmax).

A note on the lag, because two honest numbers differ here. measure_shape
subtracts the baseline y[0], as the ground-truth writer does:

    tangent_lag = t[i] - (y[i] - y[0]) / max_slope

The converters solve for the *baseline-free* tangent lag, t[i] - y[i]/slope,
which is what "lag" means for a curve that starts at zero. The two differ by
exactly y[0]/max_slope. For Gompertz and Baranyi that is a rounding error, but
a symmetric sigmoid asked for a 2 h lag genuinely starts at ~5 % of A, so a
logistic reports a tangent_lag up to ~0.3 h above the lambda it was given.
That gap is a property of the logistic, not a bug to be tuned away: closing it
would mean moving t_mid earlier and starting the curve even higher. Both
numbers are asserted below -- the baseline-free lag against the request, and
the reported lag against the request plus exactly y[0]/slope.

A note on spaces, because it cost a round. Baranyi's *parameters* are log
space -- y0 = ln(OD0), y_max = ln(A) -- which is what the model is and what
the converter returns. Its ``compute`` nevertheless returns OD600, like every
other model here, because ``Model.compute(t, **from_gompertz_params(...))`` is
the call every generator and plotter in the repo makes and it has to mean the
same thing five times. When it did not, a whole model's curves came out as
ln(OD) with a green test suite; ``test_every_model_compute_returns_od`` and
``test_generator_baranyi_curves_are_od`` are there so that cannot recur.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
sys.path.insert(0, str(SYNTH))

from src.growth_models import (  # noqa: E402
    DEFAULT_GRID,
    BaranyiModel,
    GompertzModel,
    HaldaneModel,
    LogisticModel,
    RichardsModel,
    curve_from_params,
    measure_shape,
    shape_of,
)

# The range every shipped model-mismatch scenario samples from
# (curve_scenarios.py: baranyi_generated / logistic_generated /
# richards_asymmetric all use A 1.0-1.5, mu 0.15-0.25, lambda 2.0-6.0).
A_RANGE = (1.0, 1.5)
MU_RANGE = (0.15, 0.25)
LAMBDA_RANGE = (2.0, 6.0)

TOL = 0.02  # the 2 % the acceptance criteria are written against
N_DRAWS = 100


def draws(n=N_DRAWS, seed=20240601):
    """n reproducible (A, mu, lambda) triples from the shipped scenario range."""
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(*A_RANGE, n),
                    rng.uniform(*MU_RANGE, n),
                    rng.uniform(*LAMBDA_RANGE, n)))


def convert(model_name, A, mu, lambda_, **kw):
    """Call the converter for `model_name` and return (params, shape)."""
    converters = {
        'gompertz': GompertzModel.from_gompertz_params,
        'logistic': LogisticModel.from_gompertz_params,
        'richards': RichardsModel.from_gompertz_params,
        'baranyi': BaranyiModel.from_gompertz_params,
        'haldane': HaldaneModel.from_gompertz_params,
    }
    params = converters[model_name](A, mu, lambda_, **kw)
    return params, shape_of(model_name, params, DEFAULT_GRID)


def baseline_free_lag(shape):
    """t[i] - y[i]/slope: the lag of the curve ignoring where it starts."""
    return shape['tangent_lag'] - shape['y0'] / shape['max_slope']


#: Absolute floor on the lag tolerance, matching growth_models.LAG_TOL_FLOOR_H.
#: Measuring the tangent intercept from np.gradient on a 0.25 h grid carries up
#: to ~0.02 h of discretisation error (max over 100 draws, all models), and 2 %
#: of a 2 h lag is only 0.04 h, so a bare 2 % would be asserting the grid rather
#: than the converter. Measured, not tuned: raising it would need re-measuring.
LAG_TOL_FLOOR_H = 0.05


def lag_close(measured, requested):
    """Lag agreement: 2 %, never tighter than the grid's own resolution."""
    return abs(measured - requested) <= max(TOL * abs(requested), LAG_TOL_FLOOR_H)


# ---------------------------------------------------------------------------
# The headline round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_name,kwargs', [
    ('gompertz', {}),
    ('logistic', {}),
    ('richards', {'nu': 0.2}),
    ('richards', {'nu': 0.5}),
    ('richards', {'nu': 1.0}),
    ('richards', {'nu': 2.0}),
    ('baranyi', {}),
])
def test_shape_roundtrip(model_name, kwargs):
    """
    A converted curve has the plateau, slope and lag it was asked for.

    100 draws on t = arange(0, 100, 0.25). Plateau and maximum slope are
    asserted within 2 % for every model; the baseline-free tangent lag within
    2 % for every model (Baranyi only where realised=True -- see
    test_baranyi_lag_floor_is_reported for the curves whose request is below
    the model's own floor).
    """
    n_realised = 0
    for A, mu, lambda_ in draws():
        params, shape = convert(model_name, A, mu, lambda_, **kwargs)
        label = f'{model_name}{kwargs} A={A:.3f} mu={mu:.3f} lam={lambda_:.3f}'

        assert shape['plateau'] == pytest.approx(A, rel=TOL), f'plateau {label}'
        assert shape['max_slope'] == pytest.approx(mu, rel=TOL), f'slope {label}'

        if model_name == 'baranyi' and not params['realised']:
            # Lag below the achievable floor; the round trip cannot hold and
            # the converter says so rather than shipping the request.
            continue
        n_realised += 1
        assert lag_close(baseline_free_lag(shape), lambda_), f'lag {label}'
        assert params['realised'] is True, f'realised {label}'

    assert n_realised > 0


@pytest.mark.parametrize('model_name,kwargs', [
    ('gompertz', {}),
])
def test_reported_tangent_lag_matches_request_when_the_curve_starts_at_zero(
        model_name, kwargs):
    """
    For the reference model, whose y(0) is negligible, the two lag
    conventions coincide.

    This is the stricter form of the round trip: the lag as the ground-truth
    writer measures it (baseline subtracted) is itself within 2 % of lambda_.
    Only Gompertz qualifies over the whole shipped range -- even Richards at
    nu=0.2 opens at up to 1.3 % of A for a 2 h lag, which is the subject of
    test_reported_lag_exceeds_the_request_by_exactly_the_baseline_offset.
    """
    for A, mu, lambda_ in draws(n=25):
        _, shape = convert(model_name, A, mu, lambda_, **kwargs)
        assert shape['y0'] < 0.01 * A
        assert lag_close(shape['tangent_lag'], lambda_)


@pytest.mark.parametrize('model_name,kwargs', [
    ('logistic', {}),
    ('richards', {'nu': 0.2}),
    ('richards', {'nu': 0.5}),
    ('richards', {'nu': 1.0}),
    ('richards', {'nu': 2.0}),
])
def test_reported_lag_exceeds_the_request_by_exactly_the_baseline_offset(
        model_name, kwargs):
    """
    The reported-versus-requested lag gap is y(0)/slope and nothing else.

    A symmetric (or flatter-topped) sigmoid with a short lag starts
    measurably above zero, so the baseline-subtracting tangent lag sits above
    lambda_. Pinning the gap to the analytic offset means a future regression
    that moves the lag for any *other* reason fails here.
    """
    seen_gap = 0.0
    for A, mu, lambda_ in draws(n=25):
        _, shape = convert(model_name, A, mu, lambda_, **kwargs)
        gap = shape['tangent_lag'] - lambda_
        offset = shape['y0'] / shape['max_slope']
        assert gap == pytest.approx(offset, abs=0.03)
        seen_gap = max(seen_gap, gap)
    assert seen_gap > 0.0


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------

def test_logistic_start_is_analytic():
    """
    y(0) is the logistic's own value, and it stays under 6 % of A.

    The pre-fix converter left t_mid at the Gompertz inflection while keeping
    y0 = 0, so the curve started at 14-20 % of A -- the shipped CURVE0405
    opens at OD 0.270 against A = 1.35 -- which reads as a contaminated well
    and is what the pipeline's own gates were reacting to. With
    t_mid = lambda_ + A/(2*mu) the start is the honest logistic value.

    The bound is 6 %, not 2 %: over the scenario's own range the worst corner
    (A=1.5, mu=0.15, lambda=2.0) analytically gives y(0)/A = 1/(1+exp(2.8))
    = 0.057, and a quarter of random draws exceed 2 %. Asserting 2 % would be
    asserting something false about the logistic model.
    """
    rng = np.random.default_rng(11)
    n = 2000
    worst = 0.0
    for A, mu, lambda_ in zip(rng.uniform(*A_RANGE, n),
                              rng.uniform(*MU_RANGE, n),
                              rng.uniform(*LAMBDA_RANGE, n)):
        p = LogisticModel.from_gompertz_params(A, mu, lambda_)
        y0 = float(LogisticModel.compute(np.array([0.0]), **p)[0])
        analytic = A / (1.0 + np.exp(p['k'] * p['t_mid']))
        assert y0 == pytest.approx(analytic, rel=1e-9)
        assert y0 <= 0.06 * A
        worst = max(worst, y0 / A)

    # The worst corner of the range, evaluated exactly.
    p = LogisticModel.from_gompertz_params(1.5, 0.15, 2.0)
    corner = 1.5 / (1.0 + np.exp(p['k'] * p['t_mid'])) / 1.5
    assert 0.05 < corner <= 0.06
    assert worst <= corner + 1e-12


def test_logistic_conversion_is_no_longer_off_by_e():
    """The 1/e that halved logistic_generated's growth rate is gone."""
    A, mu, lambda_ = 1.2, 0.20, 4.0
    p = LogisticModel.from_gompertz_params(A, mu, lambda_)
    assert p['k'] == pytest.approx(4.0 * mu / A)
    assert p['t_mid'] == pytest.approx(lambda_ + A / (2.0 * mu))
    assert p['y0'] == 0.0

    # ...and the realised curve now matches the Gompertz it was converted from.
    logistic = shape_of('logistic', p, DEFAULT_GRID)
    gompertz = measure_shape(
        GompertzModel.compute(DEFAULT_GRID, A, mu, lambda_), DEFAULT_GRID)
    assert logistic['max_slope'] / gompertz['max_slope'] == pytest.approx(1.0, rel=1e-3)


def test_richards_nu1_equals_logistic():
    """
    Richards at nu=1 IS the logistic, so the two converters must agree.

    They used to disagree by e**2/2 = 3.69 in slope: Richards handed back
    mu*e/2 and the logistic mu/e.
    """
    for A, mu, lambda_ in draws(n=25):
        pr = RichardsModel.from_gompertz_params(A, mu, lambda_, nu=1.0)
        pl = LogisticModel.from_gompertz_params(A, mu, lambda_)
        assert pr['k'] == pytest.approx(pl['k'], rel=1e-12)
        assert pr['t_mid'] == pytest.approx(pl['t_mid'], rel=1e-12)
        yr = curve_from_params('richards', pr, DEFAULT_GRID)
        yl = curve_from_params('logistic', pl, DEFAULT_GRID)
        assert np.max(np.abs(yr - yl)) < 1e-6


def test_richards_slope_is_flat_across_nu():
    """
    k = mu*(1+nu)**(1+1/nu)/A holds for every nu, both wrong forms do not.

    The shipped form (k = mu*e*(1+nu)/A) inflates the slope by
    e*(1+nu)**(-1/nu) -- 1.208x at nu=0.5, which is exactly the 1.130 median
    mu ratio richards_asymmetric shows against the pipeline. The exponent
    1/nu proposed in an early draft deflates it instead
    (0.83/0.67/0.50/0.33x at nu=0.2/0.5/1.0/2.0).
    """
    A, mu, lambda_ = 1.2, 0.20, 4.0
    for nu in (0.2, 0.5, 1.0, 2.0):
        p = RichardsModel.from_gompertz_params(A, mu, lambda_, nu=nu)
        assert p['k'] == pytest.approx(mu * (1 + nu) ** (1 + 1 / nu) / A)
        s = shape_of('richards', p, DEFAULT_GRID)
        assert s['max_slope'] / mu == pytest.approx(1.0, abs=0.005)

        wrong_old = mu * np.e / A * (1 + nu)
        assert not np.isclose(p['k'], wrong_old, rtol=1e-3)


# ---------------------------------------------------------------------------
# Baranyi
# ---------------------------------------------------------------------------

# The 15 shipped baranyi_generated triples, copied from the committed
# synthetic_data/output/comprehensive_test/test_data/ground_truth.csv
# (curve ids 390-404): (A, mu, lambda, initial_od). Hard-coded so this test
# keeps pinning the audit's evidence after the benchmark is regenerated.
SHIPPED_BARANYI = [
    (1.329156, 0.208940, 2.450151, 0.015058),
    (1.128689, 0.191888, 2.778918, 0.013861),
    (1.261116, 0.172547, 5.772908, 0.004942),
    (1.371012, 0.186846, 5.933585, 0.005839),
    (1.265872, 0.217325, 3.744767, 0.017051),
    (1.446935, 0.243462, 4.247187, 0.002606),
    (1.432883, 0.178386, 4.538444, 0.004774),
    (1.114931, 0.228899, 3.148086, 0.017153),
    (1.466065, 0.243470, 4.272212, 0.004633),
    (1.489780, 0.184345, 4.685488, 0.019135),
    (1.196651, 0.151404, 3.819223, 0.015371),
    (1.141847, 0.167337, 5.479232, 0.007176),
    (1.055440, 0.195715, 2.773334, 0.012640),
    (1.415111, 0.169948, 3.631539, 0.002422),
    (1.250519, 0.221183, 5.609094, 0.004211),
]


def test_baranyi_lag_floor_is_reported():
    """
    Below-floor lag requests are reported, never silently shipped.

    A Baranyi curve rising from OD0 to A at maximum slope mu spends a fixed
    amount of time in exponential phase, so its tangent lag has a floor even
    with h0 = 0. For 9 of the 15 shipped triples the requested lambda is under
    that floor (CURVE0390 asks for 2.45 h against a floor of 3.9 h). The
    converter must return realised=False for exactly those, return the shape
    the curve really has in ``achieved``, and raise nothing.
    """
    n_below = 0
    for A, mu, lambda_, initial_od in SHIPPED_BARANYI:
        floor = BaranyiModel.lag_floor(A, mu, initial_od)
        assert np.isfinite(floor) and floor > 0
        p = BaranyiModel.from_gompertz_params(A, mu, lambda_, initial_od)
        below = lambda_ < floor
        n_below += bool(below)

        assert p['realised'] is (not below), (
            f'A={A} mu={mu} lam={lambda_} floor={floor:.3f}')
        # Truth is what the curve does, whichever way the flag went.
        shape = shape_of('baranyi', p, DEFAULT_GRID)
        assert p['achieved']['mu'] == pytest.approx(shape['max_slope'])
        assert p['achieved']['lambda'] == pytest.approx(shape['tangent_lag'])
        assert p['achieved']['A'] == pytest.approx(shape['plateau'])
        # Slope and plateau are realised even when the lag is not.
        assert shape['max_slope'] == pytest.approx(mu, rel=TOL)
        assert shape['plateau'] == pytest.approx(A, rel=TOL)
        if below:
            assert baseline_free_lag(shape) == pytest.approx(floor, abs=0.05)
            assert baseline_free_lag(shape) > lambda_

    assert n_below == 9, (
        'the audit measured 9 of the 15 shipped baranyi triples below the '
        'lag floor; if the model changed, re-measure before editing this')


def test_baranyi_parameters_are_log_space():
    """
    y0/y_max are ln(OD); compute() and compute_od() both return OD.

    The parameters are log space because the Baranyi-Roberts model is. The
    curve is not: ``compute`` returns OD600 like every other model in the
    module, and ``compute_log`` is the raw log-space equation.
    """
    A, mu, lambda_, od0 = 1.2, 0.20, 5.0, 0.01
    p = BaranyiModel.from_gompertz_params(A, mu, lambda_, od0)
    assert p['y0'] == pytest.approx(np.log(0.01))
    assert p['y_max'] == pytest.approx(np.log(A))

    log_y = BaranyiModel.compute_log(DEFAULT_GRID, **p)
    od = BaranyiModel.compute(DEFAULT_GRID, **p)
    assert np.allclose(od, np.exp(log_y))
    assert np.allclose(BaranyiModel.compute_od(DEFAULT_GRID, **p), od)
    assert od[0] == pytest.approx(0.01, rel=1e-6)
    assert od.max() == pytest.approx(A, rel=1e-6)
    # The log-space curve is the thing the OD curve is the exponential of,
    # and it is emphatically not an OD trace.
    assert log_y.min() == pytest.approx(np.log(0.01))
    assert log_y.max() == pytest.approx(np.log(A))


def test_every_model_compute_returns_od():
    """
    ``Model.compute(t, **from_gompertz_params(...))`` is OD for ALL five models.

    This is the call every generator and plotter in the repo makes
    (data_generator.py:128-140, visualization/synthetic_plotter.py:121-127),
    splatting the converter's dict straight into ``compute``. When Baranyi
    alone answered it in log space, the generator wrote curves peaking at 0.26
    instead of 1.25 and the model-comparison figure plotted one line spanning
    -4.6..0.18 against three spanning 0..1.2 -- with nothing raising and the
    whole suite green. This test is the guard: every model's ``compute``
    returns a curve that starts low, ends near A, and never goes negative.
    """
    A, mu, lambda_ = 1.2, 0.20, 4.0
    t = DEFAULT_GRID
    calls = {
        'gompertz': (GompertzModel, GompertzModel.from_gompertz_params(A, mu, lambda_)),
        'baranyi': (BaranyiModel, BaranyiModel.from_gompertz_params(A, mu, lambda_)),
        'logistic': (LogisticModel, LogisticModel.from_gompertz_params(A, mu, lambda_)),
        'richards': (RichardsModel, RichardsModel.from_gompertz_params(A, mu, lambda_)),
        'haldane': (HaldaneModel, HaldaneModel.from_gompertz_params(A, mu, lambda_)),
    }
    for name, (model, params) in calls.items():
        y = model.compute(t, **params)
        assert np.all(np.isfinite(y)), name
        assert y.min() >= 0.0, f'{name} returned negative OD (log space?)'
        assert y.min() <= 0.1 * A, name
        assert y.max() == pytest.approx(A, rel=0.02), name


def test_generator_baranyi_curves_are_od():
    """
    End-to-end: the shipped generator's baranyi curves live in OD, not ln(OD).

    Guards the regression the unit tests above cannot see, because it lives at
    the seam between this module and data_generator.py: a baranyi curve must
    peak near A like the other three models, not near ln(A) + baseline.
    data_generator.py belongs to another work package, so if its entry point
    changes shape this skips rather than failing on someone else's rename.
    """
    import inspect

    try:
        from src.data_generator import SyntheticGrowthCurveGenerator
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f'data_generator unavailable: {exc}')

    gen = SyntheticGrowthCurveGenerator(seed=42)
    fn = getattr(gen, 'generate_single_curve', None)
    if fn is None:                                            # pragma: no cover
        pytest.skip('generate_single_curve has been renamed')
    kwargs = dict(A=1.2, mu=0.2, lambda_=4.0, noise_level='medium',
                  duration_hours=100.0, time_resolution=0.25, initial_od=0.05)
    sig = inspect.signature(fn).parameters
    if not ({'model_type'} | set(kwargs)).issubset(sig):      # pragma: no cover
        pytest.skip('generate_single_curve signature has changed')

    peaks = {}
    for model_type in ('gompertz', 'baranyi', 'logistic', 'richards'):
        curve = fn(model_type=model_type, **kwargs)
        od = np.asarray(curve.od600, dtype=float)
        assert od.min() >= 0.0, model_type
        peaks[model_type] = float(od.max())

    # ln(1.2) + 0.05 = 0.232: the failure mode is an order of magnitude away
    # from A, so a loose band on A separates them without pinning noise.
    for model_type, peak in peaks.items():
        assert 1.1 < peak < 1.45, f'{model_type} peaks at {peak:.4f}, not near A=1.2'


# ---------------------------------------------------------------------------
# Haldane
# ---------------------------------------------------------------------------

def test_haldane_ki_changes_shape():
    """
    Ki is a real input now, so the haldane_* scenarios are three regimes.

    With Ks, q, S0 and the requested (A, mu, lambda) all fixed, a strongly
    inhibited culture (Ki = 0.15, i.e. Ki ~ Ks) cannot be made to grow like a
    Monod-like one (Ki = 10): forcing the requested slope would need an
    intrinsic mu_max near 6 /h, which is not a bacterium, so the cap binds,
    the converter reports realised=False and the curve is visibly different.
    Before this package Ki reached the model only as a default and all three
    haldane scenarios were one model sampled three ways.
    """
    A, mu, lambda_ = 1.2, 0.20, 4.0
    p_mild = HaldaneModel.from_gompertz_params(A, mu, lambda_, Ki=10.0)
    p_strong = HaldaneModel.from_gompertz_params(A, mu, lambda_, Ki=0.15)

    assert p_mild['Ki'] == 10.0 and p_strong['Ki'] == 0.15
    assert p_mild['realised'] is True
    assert p_strong['realised'] is False
    assert p_strong['mu_max'] == pytest.approx(HaldaneModel.MU_MAX_CAP)

    y_mild = curve_from_params('haldane', p_mild, DEFAULT_GRID)
    y_strong = curve_from_params('haldane', p_strong, DEFAULT_GRID)
    diff = y_mild - y_strong
    assert np.max(np.abs(diff)) > 0.3
    assert np.ptp(diff) > 0.3

    # And the model itself responds to Ki with everything else held fixed.
    kw = dict(mu_max=0.5, Ks=0.1, X_max=1.2, q=0.1, X0=0.01, S0=1.0)
    y_hi = HaldaneModel.compute(DEFAULT_GRID, Ki=10.0, **kw)
    y_lo = HaldaneModel.compute(DEFAULT_GRID, Ki=0.15, **kw)
    assert np.max(np.abs(y_hi - y_lo)) > 0.3


def test_haldane_uses_lambda_and_the_supplied_kinetics():
    """
    lambda_ is honoured (it used to be accepted and ignored), and Ks/q/S0 land.

    X0 is solved for the requested lag instead of being pinned at 0.01, and
    mu_max for the requested slope instead of being mu*2.5.
    """
    for A, mu, lambda_ in [(1.2, 0.20, 4.0), (1.45, 0.28, 2.5), (0.75, 0.15, 5.0)]:
        p = HaldaneModel.from_gompertz_params(
            A, mu, lambda_, S0=1.0, Ki=10.0, Ks=0.12, q=0.09)
        assert p['Ks'] == 0.12 and p['q'] == 0.09 and p['S0'] == 1.0
        assert p['X_max'] == A
        assert 0 < p['X0'] < A
        assert p['realised'] is True
        s = shape_of('haldane', p, DEFAULT_GRID)
        assert s['plateau'] == pytest.approx(A, rel=TOL)
        assert s['max_slope'] == pytest.approx(mu, rel=TOL)
        assert lag_close(baseline_free_lag(s), lambda_)

    # Two different lags must give two different X0 values.
    p_short = HaldaneModel.from_gompertz_params(1.2, 0.20, 2.5)
    p_long = HaldaneModel.from_gompertz_params(1.2, 0.20, 6.0)
    assert p_short['X0'] > p_long['X0']


def test_haldane_default_call_still_serves_the_shipped_scenarios():
    """
    The three haldane_* scenarios call this with (A, mu, lambda) only.

    Their call site must keep working and keep returning every key
    HaldaneModel.compute needs, with X_max = A (pinned by
    tests/test_haldane.py::test_from_gompertz_params).
    """
    p = HaldaneModel.from_gompertz_params(A=1.2, mu=0.2, lambda_=4.0)
    for key in ('mu_max', 'Ks', 'Ki', 'X_max', 'q', 'X0', 'S0'):
        assert key in p and np.isfinite(p[key])
    assert p['X_max'] == 1.2
    assert p['X0'] > 0 and p['mu_max'] > 0 and p['Ks'] > 0 and p['Ki'] > 0
    od = HaldaneModel.compute(DEFAULT_GRID, **p)
    assert np.all(np.diff(od) >= -1e-8)


# ---------------------------------------------------------------------------
# The shared interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_name,kwargs', [
    ('gompertz', {}),
    ('logistic', {}),
    ('richards', {'nu': 0.5}),
    ('baranyi', {}),
    ('haldane', {}),
])
def test_every_converter_returns_realised_and_achieved(model_name, kwargs):
    """The interface WP3 codes against: params + realised + achieved."""
    params, shape = convert(model_name, 1.2, 0.2, 4.0, **kwargs)
    assert isinstance(params['realised'], bool)
    assert set(params['achieved']) == {'A', 'mu', 'lambda'}
    assert params['achieved']['A'] == pytest.approx(shape['plateau'])
    assert params['achieved']['mu'] == pytest.approx(shape['max_slope'])
    assert params['achieved']['lambda'] == pytest.approx(shape['tangent_lag'])
    for v in params['achieved'].values():
        assert np.isfinite(v)


@pytest.mark.parametrize('model_name,kwargs', [
    ('gompertz', {}),
    ('logistic', {}),
    ('richards', {'nu': 0.5}),
    ('baranyi', {}),
    ('haldane', {}),
])
def test_converted_params_splat_straight_into_compute(model_name, kwargs):
    """
    Model.compute(t, **from_gompertz_params(...)) still works, in OD, for all five.

    The metadata keys are accepted and ignored by compute so that existing
    call sites keep working; anything else is still a TypeError. Baranyi is in
    this list too: its parameters are log space but its ``compute`` answers in
    OD, so the splat means the same thing for every model.
    """
    params, shape = convert(model_name, 1.2, 0.2, 4.0, **kwargs)
    model = {'gompertz': GompertzModel, 'logistic': LogisticModel,
             'richards': RichardsModel, 'baranyi': BaranyiModel,
             'haldane': HaldaneModel}[model_name]
    y = model.compute(DEFAULT_GRID, **params)
    assert y.shape == DEFAULT_GRID.shape and np.all(np.isfinite(y))
    # Same curve shape_of measured, i.e. OD -- not a log-space trace.
    assert float(np.max(y)) == pytest.approx(shape['plateau'])
    with pytest.raises(TypeError):
        model.compute(DEFAULT_GRID, typo_key=1, **params)


def test_measure_shape_is_the_documented_convention():
    """max of np.gradient, tangent intercept at the argmax, max as plateau."""
    t = DEFAULT_GRID
    y = GompertzModel.compute(t, 1.2, 0.2, 4.0)
    s = measure_shape(y, t)
    d = np.gradient(y, t)
    i = int(np.argmax(d))
    assert s['max_slope'] == pytest.approx(d[i])
    assert s['tangent_lag'] == pytest.approx(t[i] - (y[i] - y[0]) / d[i])
    assert s['plateau'] == pytest.approx(np.max(y))
    assert s['y0'] == pytest.approx(y[0])
    assert set(s) == {'max_slope', 'tangent_lag', 'plateau', 'y0'}


def test_shape_of_defaults_to_the_generation_grid():
    """shape_of with t=None measures on arange(0, 100, 0.25)."""
    assert DEFAULT_GRID[0] == 0.0
    assert len(DEFAULT_GRID) == 400
    assert DEFAULT_GRID[-1] == pytest.approx(99.75)
    p = LogisticModel.from_gompertz_params(1.2, 0.2, 4.0)
    assert shape_of('logistic', p) == shape_of('logistic', p, DEFAULT_GRID)


def test_gompertz_passthrough_is_exact():
    """The reference model round-trips its own parameters."""
    for A, mu, lambda_ in draws(n=25):
        p = GompertzModel.from_gompertz_params(A, mu, lambda_)
        assert (p['A'], p['mu'], p['lambda_']) == (A, mu, lambda_)
        assert p['realised'] is True
        s = shape_of('gompertz', p, DEFAULT_GRID)
        assert s['max_slope'] == pytest.approx(mu, rel=0.01)
        assert lag_close(s['tangent_lag'], lambda_)
        assert s['plateau'] == pytest.approx(A, rel=1e-6)
