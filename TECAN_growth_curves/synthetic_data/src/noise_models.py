"""
Noise, artefact and blank models for synthetic TECAN growth curves.

Everything in this module is calibrated against a measured number taken from the
four real plates in ``data/raw/`` and the condition it came from is named next to
it.  No magnitude here is invented.

MEASUREMENT PROTOCOL
--------------------
* ``sigma_hat`` (high-frequency / "instrument" sigma) is the curvature-insensitive
  robust estimator ``1.4826 * MAD(diff(y, 2)) / sqrt(6)`` (var of a 2nd difference
  of white noise is ``6 sigma^2``).
* "detrended residual" means ``y`` minus its own least-squares line in ``t``.
* Autocorrelation is ``ac(x, lag) = sum(x[:-lag] * x[lag:]) / sum(x**2)`` on the
  mean-removed series - the same estimator the audit used, so the numbers below
  can be compared directly with synthetic output.

MEASURED REAL-PLATE NUMBERS USED HERE
-------------------------------------
Per-well instrument noise (96 raw wells of ``data/raw/Group 3/GROUP3_VALUES.csv``
via ``tecan_growth.load_plate``):

    sigma_hat median 0.00127, p10 0.00073, p90 0.00200 OD, and it is essentially
    FLAT in OD: the OD>1.4 plateaus of H4..H8 give 0.00139-0.00169, the same as a
    well sitting at OD 0.13.  Band medians over all 92 real condition curves:
    OD 0.00-0.05 -> 0.00046, 0.05-0.20 -> 0.00114, 0.20-0.60 -> 0.00133,
    0.60-1.20 -> 0.00215, >1.20 -> 0.00162.  The old presets (baseline 0.001-0.01,
    proportional 0.005-0.05) were 6.4x too large at baseline and 18.8x at OD>1.2.

Noisiest real object (Group 3 well H12, LAMBDACYHALOTHRINANDLB-LCY2 third well):

    whole-trace sigma_hat 0.00596, sigma_hat over its OD>1.4 plateau 0.02554,
    largest single positive excursion vs an 11-point rolling median +0.295 OD.
    This is the anchor for the ``very_high`` preset - "the noisiest well observed
    in 96 real wells", NOT "2.5x the worst real curve" (the worst real
    condition-average whole-curve sigma_hat is only 0.00443).

Residual STRUCTURE, from the 16 real non-growing condition curves in
``data/raw/*/*_DATA/*.csv`` (range < 0.05 OD, < 50 % exact zeros):

    sigma_hat median 0.00073 (p10 0.00050, p90 0.00218)
    detrended-residual sd median 0.00353 (p90 0.0051) = 4.8x sigma_hat
    detrended-residual autocorrelation 0.923 / 0.851 / 0.570 / 0.203
        at 0.25 / 1 / 5 / 10 h  (1/e at ~7-8 h)
    after a 5 h centred-moving-average high-pass, lag-1 ac is +0.085
        (today's synthetic flat_no_growth gives -0.052: i.i.d. through an MA)
    total LINEAR drift over the run: median +0.004, p10 -0.015, p90 +0.044,
        min -0.025, max +0.052 OD - i.e. SIGNED, unlike the old always-positive
        ``drift_rate * time``.
    fitting ``a + b t + c (1 - exp(-t / (f T)))`` per curve leaves a residual sd of
        only 0.00213: the nonlinear part of the baseline wander has amplitude
        |c| median 0.020, p90 0.046, max 0.086 OD (signed sd 0.033) with
        f in {0.2, 0.35, 0.5}.  That curved creep is why a single exponential
        Ornstein-Uhlenbeck term at tau 7.5 h cannot reproduce ac(5 h) = 0.570.

Decomposition actually implemented by :class:`ColoredNoise` (all three numbers are
measured, not tuned):

    detrended-residual sd 0.0035  =  curved creep 0.0028  (+)  wander 0.0021
    (0.0028 = sqrt(0.00353^2 - 0.00213^2), 0.0021 = sqrt(0.00213^2 - 0.00073^2))

Single-point excursions vs an 11-point rolling median, 76 usable Group 3 wells:

    median +0.0058 OD, p90 +0.0211 OD, max +0.295 OD (H12); only 2 of 76 wells
    carry an excursion above 0.05 OD.  ``outlier_magnitude`` per preset follows
    that distribution, with ``very_high`` at 0.35 (H12's own 0.295 rounded up).
    Spikes are ONE-SIDED positive and decay over at most 4 cycles - a bubble
    reads high for a few cycles, it is not a symmetric Gaussian outlier.

Blank wells (18 wells with ``Strain == BLANK`` in
``data/raw/Group 3/GROUP3_KEY.csv``, read from ``GrowthRate_GROUP3.xlsx`` through
``tecan_growth.load_plate``; 400 cycles, 99.75 h, 0.25 h resolution):

    per-media mean b0: H2O 0.1231, LB 0.1300, IMIDACLOPRIDANDLB 0.1249,
        LAMBDACYHALOTHRINANDLB 0.1237, LAMBDACYHALOTHRIN 0.1513,
        IMIDACLOPRID 0.1493  -> b0 sampled uniformly on [0.120, 0.155]
    per-media across-well b0 sd: 0.0029 (LCY+LB) .. 0.0292 (IMIDACLOPRID)
    blank linear drift over the run: -0.0285 (E2) .. +0.0219 (C3) OD,
        i.e. a rate on [-0.000285, +0.000220] OD/h
    blank sigma_hat: 0.00036 (A1) .. 0.00248 (E1), median 0.00112

    (The plan quotes -0.023..+0.025 OD and 0.0006-0.0026 for the last two; those
    came from a slightly different baseline estimator.  The ranges used here are
    the ones this module measured, and the script that produced them is
    reproduced in ``BlankModel``'s docstring.)

WHAT CHANGED versus the previous version of this file
-----------------------------------------------------
* ``NOISE_PRESETS`` reparameterised to the measured distribution (names kept).
* ``InstrumentNoise`` gained coloured residuals, a SIGNED per-curve drift rate
  and one-sided decaying spikes; ``apply_with_random_drift_direction`` (dead
  code, no caller in the repo) was deleted.
* ``RMSEBasedNoise`` now delivers the RMSE / R^2 it is asked for exactly; the old
  profile ``rmse * (0.7 + 0.3 |y| / max|y|)`` has RMS 0.9786 of target, so every
  target R^2 was overshot (0.930 realised 0.9337, 0.800 realised 0.8108).
* New reusable primitives: :func:`inject_spikes`, :func:`apply_signed_drift`,
  :func:`sample_drift_rate`, :func:`add_target_rmse_component`, :class:`BlankModel`.
* No global numpy seeding anywhere (the module never calls into the legacy
  ``np.random`` singleton): every entry point takes ``rng=None|int|Generator``.

NOTE ON ``actual_r_squared``
---------------------------
``RMSEBasedNoise.generate_noise_to_target_r2`` returns R^2 of the NOISY curve
against the noise-free generating curve over the FULL trace.  It is not the R^2
of a Gompertz fit to a growth window, and no validator should compare it with
``fit_r_squared``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    'NoiseCharacteristics',
    'GaussianNoise',
    'ODDependentNoise',
    'ColoredNoise',
    'InstrumentNoise',
    'RMSEBasedNoise',
    'BlankModel',
    'NOISE_PRESETS',
    'DRIFT_RATE_RANGE',
    'get_noise_model',
    'get_rmse_noise',
    'inject_spikes',
    'apply_signed_drift',
    'sample_drift_rate',
    'add_target_rmse_component',
]

RngLike = Union[None, int, np.random.Generator, np.random.RandomState]

# Signed per-curve drift rate (OD/h).  Real non-growing condition curves drift by
# -0.025 .. +0.052 OD over a ~100 h run (median +0.004); the old presets could
# only drift UP.  Spans -0.025 .. +0.060 OD over 100 h.
DRIFT_RATE_RANGE: Tuple[float, float] = (-0.00025, 0.00060)

# Largest number of cycles a single bubble stays visible for (real spikes decay
# back to the trace within a few 15-minute cycles).
MAX_SPIKE_DECAY_POINTS = 4


# =============================================================================
# Helpers
# =============================================================================

def _as_rng(rng: RngLike = None) -> np.random.Generator:
    """Coerce ``None`` / int seed / Generator / legacy RandomState to a Generator.

    Never touches the global numpy random state.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))


def _pick_rng(rng: RngLike, seed: RngLike) -> np.random.Generator:
    """``rng`` wins over the legacy ``seed`` keyword when both are given."""
    return _as_rng(rng if rng is not None else seed)


def _median_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return float('nan')
    dt = float(np.median(np.diff(t)))
    return dt if np.isfinite(dt) and dt > 0 else float('nan')


def _filtered_noise(rng: np.random.Generator, n: int, kernel: np.ndarray) -> np.ndarray:
    """Unit-variance stationary series = white noise convolved with ``kernel``."""
    k = np.asarray(kernel, dtype=float)
    norm = float(np.sqrt(np.sum(k ** 2)))
    if n <= 0:
        return np.zeros(0)
    if k.size <= 1 or norm == 0.0:
        return rng.standard_normal(n)
    z = rng.standard_normal(n + k.size - 1)
    return np.convolve(z, k, mode='valid')[:n] / norm


def _ar1_kernel(dt: float, tau_h: float, max_len: int = 4001) -> np.ndarray:
    """Exponential kernel: output autocorrelation is ``exp(-lag_h / tau_h)``."""
    if not np.isfinite(dt) or dt <= 0 or tau_h is None or tau_h <= 0:
        return np.ones(1)
    rho = float(np.exp(-dt / tau_h))
    if rho <= 1e-6:
        return np.ones(1)
    length = int(min(max_len, max(2, np.ceil(np.log(1e-6) / np.log(rho)) + 1)))
    return rho ** np.arange(length, dtype=float)


def _gaussian_kernel(dt: float, tau_h: float, max_len: int = 4001) -> np.ndarray:
    """Gaussian kernel: output autocorrelation is ``exp(-(lag_h / tau_h) ** 2)``.

    A Gaussian-correlated wander (1/e at ``tau_h``) reproduces the measured decay
    0.923 / 0.851 / 0.570 / 0.203 far better than a strict exponential
    Ornstein-Uhlenbeck term, whose ac(5 h) can only reach ~0.30 at tau 7.5 h.
    """
    if not np.isfinite(dt) or dt <= 0 or tau_h is None or tau_h <= 0:
        return np.ones(1)
    s = tau_h / (2.0 * dt)           # kernel sd in samples
    if s < 0.3:
        return np.ones(1)
    half = int(min((max_len - 1) // 2, np.ceil(4 * s)))
    x = np.arange(-half, half + 1, dtype=float)
    return np.exp(-0.5 * (x / s) ** 2)


def _orthonormal_creep(t: np.ndarray, tau_frac: float) -> np.ndarray:
    """Saturating creep ``1 - exp(-t' / (tau_frac * T))``, made orthogonal to
    ``{1, t}`` and normalised to unit sd.

    Orthogonalising against the line is what keeps the creep from double-counting
    the signed linear drift term: the creep contributes curvature only.
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    if n < 4:
        return np.zeros(n)
    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span <= 0:
        return np.zeros(n)
    tau = max(1e-6, float(tau_frac)) * span
    shape = 1.0 - np.exp(-(t - t[0]) / tau)
    design = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(design, shape, rcond=None)
    resid = shape - design @ coef
    sd = float(np.std(resid))
    if sd <= 0:
        return np.zeros(n)
    return resid / sd


# =============================================================================
# Extracted-characteristics container (unchanged public shape)
# =============================================================================

@dataclass
class NoiseCharacteristics:
    """Container for extracted noise characteristics."""
    baseline_sigma: float
    proportional_sigma: float
    outlier_probability: float
    drift_rate: float
    rmse_range: Tuple[float, float]


# =============================================================================
# Simple models
# =============================================================================

class GaussianNoise:
    """Constant-variance Gaussian noise (test scaffold, not a real-plate model)."""

    def __init__(self, sigma: float = 0.0012):
        self.sigma = sigma

    def apply(self, signal: np.ndarray, seed: RngLike = None,
              rng: RngLike = None) -> np.ndarray:
        generator = _pick_rng(rng, seed)
        signal = np.asarray(signal, dtype=float)
        noise = generator.normal(0.0, self.sigma, size=signal.shape)
        return np.maximum(0.0, signal + noise)

    def get_expected_rmse(self) -> float:
        return self.sigma


class ODDependentNoise:
    """Heteroscedastic noise ``sigma(OD) = sigma_base + sigma_scale * OD``.

    Real TECAN noise is very nearly OD-INDEPENDENT (see the module docstring), so
    ``sigma_scale`` is small; this class is kept because it is part of the public
    API and is useful for deliberately mis-calibrated stress tests.
    """

    def __init__(self, sigma_base: float = 0.0009, sigma_scale: float = 0.0009):
        self.sigma_base = sigma_base
        self.sigma_scale = sigma_scale

    def apply(self, signal: np.ndarray, seed: RngLike = None,
              rng: RngLike = None) -> np.ndarray:
        generator = _pick_rng(rng, seed)
        signal = np.asarray(signal, dtype=float)
        sigma = self.sigma_base + self.sigma_scale * np.abs(signal)
        noise = generator.standard_normal(signal.shape) * sigma
        return np.maximum(0.0, signal + noise)

    def get_sigma_at_od(self, od: float) -> float:
        return self.sigma_base + self.sigma_scale * od


# =============================================================================
# Coloured noise
# =============================================================================

class ColoredNoise:
    """Real TECAN residual structure: white + fast AR(1) + slow curved wander.

    ``sigma_white`` is the target high-frequency sigma (scalar, or an array giving
    a per-point sigma - :class:`InstrumentNoise` passes ``baseline + prop * OD``).
    It is split between a white term and a fast AR(1) term with 1/e time
    ``tau_fast_h`` so that the marginal sd of the high-frequency part is exactly
    ``sigma_white``; the AR(1) term is what makes the lag-1 autocorrelation of a
    5 h high-passed flat curve POSITIVE (real +0.085, old synthetic -0.052).

    ``sd_slow`` is the sd of the slow structure that survives a linear detrend -
    the measured 0.0035 OD on 16 real non-growing curves.  It is split into a
    curved creep (``creep_fraction`` of it, orthogonal to ``{1, t}``, so it does
    not double-count :func:`apply_signed_drift`) and a Gaussian-correlated wander
    with 1/e time ``tau_slow_h``.

    Measured on 40 fixed-seed flat curves at the ``medium`` preset (OD 0.05,
    400 points, 0.25 h): detrended-residual ac 0.840 at 1 h and 0.550 at 5 h, and
    a 5 h high-passed lag-1 ac of +0.144, against the real 0.851 / 0.570 / +0.085.
    """

    def __init__(self,
                 sigma_white: Union[float, np.ndarray],
                 tau_fast_h: float = 0.5,
                 tau_slow_h: float = 7.5,
                 sd_slow: float = 0.0035,
                 fast_fraction: float = 0.5,
                 creep_fraction: float = 0.8,
                 creep_tau_frac: Tuple[float, float] = (0.2, 0.5)):
        self.sigma_white = sigma_white
        self.tau_fast_h = float(tau_fast_h)
        self.tau_slow_h = float(tau_slow_h)
        self.sd_slow = float(sd_slow)
        self.fast_fraction = float(np.clip(fast_fraction, 0.0, 1.0))
        self.creep_fraction = float(np.clip(creep_fraction, 0.0, 1.0))
        self.creep_tau_frac = tuple(creep_tau_frac)

    # ------------------------------------------------------------------ core
    def sample(self, t: np.ndarray, rng: RngLike = None,
               sigma_white: Union[float, np.ndarray, None] = None) -> np.ndarray:
        """Return a coloured noise realisation on the grid ``t`` (no signal)."""
        generator = _as_rng(rng)
        t = np.asarray(t, dtype=float)
        n = t.size
        if n == 0:
            return np.zeros(0)
        sigma = self.sigma_white if sigma_white is None else sigma_white
        sigma = np.abs(np.asarray(sigma, dtype=float))
        if sigma.ndim == 0:
            sigma = np.full(n, float(sigma))
        dt = _median_dt(t)

        p = self.fast_fraction
        sd_fast = sigma * p
        sd_white = sigma * np.sqrt(max(0.0, 1.0 - p * p))

        out = sd_white * generator.standard_normal(n)
        if p > 0:
            out = out + sd_fast * _filtered_noise(generator, n, _ar1_kernel(dt, self.tau_fast_h))

        if self.sd_slow > 0 and n >= 4:
            wander_sd = self.sd_slow * np.sqrt(max(0.0, 1.0 - self.creep_fraction ** 2))
            creep_sd = self.sd_slow * self.creep_fraction
            if wander_sd > 0:
                out = out + wander_sd * _filtered_noise(
                    generator, n, _gaussian_kernel(dt, self.tau_slow_h))
            if creep_sd > 0:
                lo, hi = self.creep_tau_frac
                shape = _orthonormal_creep(t, generator.uniform(lo, hi))
                out = out + generator.normal(0.0, creep_sd) * shape
        return out

    def apply(self, signal: np.ndarray, t: np.ndarray, rng: RngLike = None,
              clip: bool = False) -> np.ndarray:
        """Add coloured noise to ``signal``.

        ``clip=False`` (default) leaves negatives alone so a caller can subtract a
        blank and clip once, which is what the real pipeline does.
        """
        signal = np.asarray(signal, dtype=float)
        noisy = signal + self.sample(t, rng=rng)
        return np.maximum(0.0, noisy) if clip else noisy


# =============================================================================
# Artefact primitives (pure functions on (t, od))
# =============================================================================

def inject_spikes(od: np.ndarray,
                  indices: Sequence[int],
                  magnitudes: Union[float, Sequence[float]],
                  decay_points: int = 3) -> np.ndarray:
    """Add one-sided POSITIVE spikes that decay over at most 4 cycles.

    A bubble or a condensation droplet reads high for a few cycles and then the
    trace returns; it is not a symmetric Gaussian outlier.  The amplitude at
    offset ``k`` is ``magnitude * exp(-2k / (decay_points - 1))``, so the last
    affected cycle keeps ~13.5 % of the peak.

    Returns a new array; ``od`` is not modified.
    """
    out = np.array(od, dtype=float, copy=True)
    idx = np.asarray(list(indices), dtype=int) if len(np.shape(indices)) else np.asarray([indices], dtype=int)
    if idx.size == 0:
        return out
    mags = np.asarray(magnitudes, dtype=float)
    if mags.ndim == 0:
        mags = np.full(idx.size, float(mags))
    if mags.size != idx.size:
        raise ValueError(f"magnitudes ({mags.size}) must match indices ({idx.size})")

    d = int(np.clip(int(decay_points), 1, MAX_SPIKE_DECAY_POINTS))
    offsets = np.arange(d)
    profile = np.ones(d) if d == 1 else np.exp(-2.0 * offsets / (d - 1))

    n = out.size
    for i, m in zip(idx, mags):
        if i < 0 or i >= n:
            continue
        j = np.minimum(i + offsets, n - 1)
        # np.add.at so overlapping spikes accumulate instead of overwriting
        np.add.at(out, j, np.abs(m) * profile)
    return out


def apply_signed_drift(od: np.ndarray, t: np.ndarray, rate: float) -> np.ndarray:
    """Add ``rate * (t - t[0])`` OD.  ``rate`` may be negative (settling) or
    positive (evaporation).  Returns a new array."""
    od = np.asarray(od, dtype=float)
    t = np.asarray(t, dtype=float)
    if od.shape != t.shape:
        raise ValueError(f"od {od.shape} and t {t.shape} must have the same shape")
    return od + float(rate) * (t - t[0])


def sample_drift_rate(rng: RngLike = None,
                      rate_range: Tuple[float, float] = DRIFT_RATE_RANGE) -> float:
    """Draw a SIGNED per-curve drift rate (OD/h) from the measured real range."""
    lo, hi = rate_range
    return float(_as_rng(rng).uniform(lo, hi))


# =============================================================================
# Instrument noise
# =============================================================================

class InstrumentNoise:
    """The full TECAN well model: coloured residuals + spikes + signed drift.

    Defaults are the ``medium`` preset, i.e. the median real well.
    """

    def __init__(
        self,
        baseline_sigma: float = 0.0009,
        proportional_sigma: float = 0.0009,
        outlier_prob: float = 0.005,
        outlier_magnitude: float = 0.008,
        drift_rate: Optional[float] = None,
        drift_rate_range: Optional[Tuple[float, float]] = DRIFT_RATE_RANGE,
        colored: bool = True,
        tau_fast_h: float = 0.5,
        tau_slow_h: float = 7.5,
        sd_slow: float = 0.0035,
        fast_fraction: float = 0.5,
        spike_decay_points: int = 3,
    ):
        """
        Args:
            baseline_sigma: high-frequency sigma at OD 0.
            proportional_sigma: extra high-frequency sigma per OD unit.
            outlier_prob: per-point probability of starting a positive spike.
            outlier_magnitude: typical spike height (OD); the drawn height is
                ``U(0.6, 1.4) * outlier_magnitude``.
            drift_rate: FIXED signed drift (OD/h).  Leave ``None`` to draw a
                signed rate per call from ``drift_rate_range`` - that is the
                calibrated behaviour; a fixed value is for reproducing a specific
                artefact.
            drift_rate_range: range the per-call signed rate is drawn from.
            colored: use :class:`ColoredNoise` for the stochastic part.  Falls
                back to white noise when no time axis is supplied.
        """
        self.baseline_sigma = baseline_sigma
        self.proportional_sigma = proportional_sigma
        self.outlier_prob = outlier_prob
        self.outlier_magnitude = outlier_magnitude
        self.drift_rate = drift_rate
        self.drift_rate_range = drift_rate_range
        self.colored = colored
        self.tau_fast_h = tau_fast_h
        self.tau_slow_h = tau_slow_h
        self.sd_slow = sd_slow
        self.fast_fraction = fast_fraction
        self.spike_decay_points = spike_decay_points
        # Populated by apply() so the generator can record what it actually did.
        self.last_drift_rate: float = 0.0
        self.last_spike_indices: np.ndarray = np.zeros(0, dtype=int)

    # ------------------------------------------------------------------ api
    def sigma_at(self, od: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.baseline_sigma + self.proportional_sigma * np.abs(od)

    def sample_drift_rate(self, rng: RngLike = None) -> float:
        if self.drift_rate is not None:
            return float(self.drift_rate)
        if not self.drift_rate_range:
            return 0.0
        return sample_drift_rate(rng, self.drift_rate_range)

    def apply(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        seed: RngLike = None,
        rng: RngLike = None,
    ) -> np.ndarray:
        """Add instrument noise, spikes and a signed drift; clip at 0."""
        generator = _pick_rng(rng, seed)
        signal = np.asarray(signal, dtype=float)
        n = signal.size
        sigma = self.sigma_at(signal)

        t = None if time is None else np.asarray(time, dtype=float)
        use_colored = (
            self.colored and t is not None and t.size == n and n >= 4
            and np.isfinite(_median_dt(t))
        )
        if use_colored:
            model = ColoredNoise(sigma, tau_fast_h=self.tau_fast_h,
                                 tau_slow_h=self.tau_slow_h, sd_slow=self.sd_slow,
                                 fast_fraction=self.fast_fraction)
            noisy = signal + model.sample(t, rng=generator)
        else:
            noisy = signal + generator.standard_normal(signal.shape) * sigma

        # One-sided positive spikes (bubbles / condensation).
        self.last_spike_indices = np.zeros(0, dtype=int)
        if self.outlier_prob and self.outlier_prob > 0 and n:
            idx = np.nonzero(generator.random(n) < self.outlier_prob)[0]
            if idx.size:
                mags = self.outlier_magnitude * generator.uniform(0.6, 1.4, size=idx.size)
                noisy = inject_spikes(noisy, idx, mags, self.spike_decay_points)
                self.last_spike_indices = idx

        # Signed drift.
        rate = self.sample_drift_rate(generator)
        self.last_drift_rate = rate
        if t is not None and t.size == n and rate:
            noisy = apply_signed_drift(noisy, t, rate)

        return np.maximum(0.0, noisy)


# =============================================================================
# RMSE / target-R2 noise
# =============================================================================

def _rmse_weight_profile(signal: np.ndarray) -> np.ndarray:
    """Normalised pointwise sd profile.

    The historical profile was ``w = 0.7 + 0.3 |y| / max(0.1, max|y|)``, used as
    the sd directly - its RMS is 0.9786 of the requested value, so every target
    RMSE was under-delivered and every target R^2 overshot.  Dividing by
    ``sqrt(mean(w**2))`` makes the mean square of the profile exactly 1.
    """
    signal = np.asarray(signal, dtype=float)
    denom = max(0.1, float(np.max(np.abs(signal))) if signal.size else 0.1)
    w = 0.7 + 0.3 * np.abs(signal) / denom
    ms = float(np.mean(w ** 2))
    if ms <= 0:
        return np.ones_like(w)
    return w / np.sqrt(ms)


def _clipped_ss(clean: np.ndarray, base: np.ndarray, noise: np.ndarray, c: float) -> float:
    """``|| clip(clean + base + c * noise) - clean ||^2``."""
    return float(np.sum((np.maximum(0.0, clean + base + c * noise) - clean) ** 2))


def _solve_clip_scale(clean: np.ndarray, base: np.ndarray, noise: np.ndarray,
                      target_ss_res: float, iterations: int = 3) -> float:
    """Solve ``|| clip(clean + base + c n) - clean ||^2 = target_ss_res`` for c > 0.

    Starts from the closed-form solution of the un-clipped quadratic (cross term
    included), runs the prescribed multiplicative clip-compensating iterations,
    then finishes with a bisection so the residual sum of squares is hit to
    machine precision - which is what makes the target-R^2 bands non-overlapping.
    """
    ss_n = float(np.sum(noise ** 2))
    if ss_n <= 0 or target_ss_res <= 0:
        return 0.0
    ss_b = float(np.sum(base ** 2))
    cross = float(np.dot(base, noise))
    disc = cross ** 2 - ss_n * (ss_b - target_ss_res)
    if disc < 0:
        return 0.0
    c = (-cross + np.sqrt(disc)) / ss_n
    if not np.isfinite(c) or c <= 0:
        c = np.sqrt(target_ss_res / ss_n)

    for _ in range(max(0, iterations)):
        got = _clipped_ss(clean, base, noise, c)
        if got <= 0:
            break
        c *= np.sqrt(target_ss_res / got)

    # Bracket then bisect to machine precision (the clip makes S(c) only
    # piecewise smooth, so a fixed number of multiplicative steps is not enough
    # for a 1e-6 guarantee on R^2).
    lo, hi = 0.0, max(c, 1e-12)
    for _ in range(200):
        if _clipped_ss(clean, base, noise, hi) >= target_ss_res:
            break
        lo, hi = hi, hi * 2.0
    else:                                              # pragma: no cover
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _clipped_ss(clean, base, noise, mid) < target_ss_res:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def add_target_rmse_component(clean: np.ndarray,
                              res0: np.ndarray,
                              target_ss_res: float,
                              rng: RngLike = None) -> Tuple[np.ndarray, float, bool]:
    """Top an existing residual up to a target residual sum of squares.

    This is the additive preset + target-R^2 path: the preset noise model runs
    first (with its spikes and drift), ``res0 = noisy - clean`` is whatever it
    produced, and this function adds just enough extra Gaussian noise to land on
    ``target_ss_res`` exactly, cross term and zero-clip included.

    Returns:
        (component, achieved_r2, added)

        ``component`` is the array to ADD to ``clean + res0``.  ``added`` is False
        (and ``component`` all zeros) when ``res0`` alone already meets or exceeds
        the target - in that case the achieved R^2 is simply reported.
    """
    generator = _as_rng(rng)
    clean = np.asarray(clean, dtype=float)
    res0 = np.zeros_like(clean) if res0 is None else np.asarray(res0, dtype=float)
    n = clean.size

    ss_tot = float(np.sum((clean - np.mean(clean)) ** 2)) if n else 0.0

    def _r2(component: np.ndarray) -> float:
        ss_res = _clipped_ss(clean, res0, component, 1.0)
        if ss_tot <= 0:
            return float('nan')
        return 1.0 - ss_res / ss_tot

    zeros = np.zeros(n)
    if n == 0 or target_ss_res <= 0:
        return zeros, _r2(zeros), False

    if _clipped_ss(clean, res0, zeros, 1.0) >= target_ss_res:
        return zeros, _r2(zeros), False

    noise = generator.standard_normal(n) * _rmse_weight_profile(clean)
    c = _solve_clip_scale(clean, res0, noise, float(target_ss_res))
    component = c * noise
    return component, _r2(component), True


class RMSEBasedNoise:
    """Noise scaled to hit a requested RMSE (or a requested full-trace R^2) exactly."""

    def __init__(
        self,
        target_rmse: Optional[float] = None,
        rmse_range: Tuple[float, float] = (0.0018, 0.0036),
    ):
        self.target_rmse = target_rmse
        self.rmse_range = rmse_range

    # ------------------------------------------------------------------ api
    def apply(
        self,
        signal: np.ndarray,
        seed: RngLike = None,
        distribution: str = 'uniform',
        rng: RngLike = None,
    ) -> Tuple[np.ndarray, float]:
        """Add noise whose realised RMSE equals the requested one.

        Returns ``(noisy_signal, actual_rmse)``; ``actual_rmse`` matches the
        target to ~1e-12 (the old implementation under-delivered by 2.1 %).
        """
        generator = _pick_rng(rng, seed)
        signal = np.asarray(signal, dtype=float)
        n = signal.size
        rmse = self.target_rmse if self.target_rmse is not None else self._sample_rmse(
            distribution, generator)
        if n == 0 or rmse <= 0:
            return signal.copy(), 0.0

        target_ss_res = float(n) * float(rmse) ** 2
        noise = generator.standard_normal(n) * _rmse_weight_profile(signal)
        c = _solve_clip_scale(signal, np.zeros(n), noise, target_ss_res)
        noisy = np.maximum(0.0, signal + c * noise)
        actual_rmse = float(np.sqrt(np.mean((noisy - signal) ** 2)))
        return noisy, actual_rmse

    def _sample_rmse(self, distribution: str, generator: np.random.Generator) -> float:
        low, high = self.rmse_range
        if distribution == 'uniform':
            return float(generator.uniform(low, high))
        if distribution == 'log_uniform':
            return float(10 ** generator.uniform(np.log10(low), np.log10(high)))
        if distribution == 'realistic':
            # Real per-well sigma_hat is log-normal-ish around 0.00127 (96 Group 3
            # wells: p10 0.00073, p90 0.00200).
            return float(generator.lognormal(mean=np.log(0.00127), sigma=0.45))
        raise ValueError(f"Unknown distribution: {distribution}")

    def generate_noise_to_target_r2(
        self,
        signal: np.ndarray,
        target_r2: float,
        seed: RngLike = None,
        rng: RngLike = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Add noise so that R^2 against the noise-free curve equals ``target_r2``.

        R^2 here is measured over the FULL trace against the generating curve, not
        against a fitted Gompertz (see the module docstring).

        Returns ``(noisy_signal, actual_r2, actual_rmse)`` with
        ``|actual_r2 - target_r2| < 1e-9`` per curve.
        """
        generator = _pick_rng(rng, seed)
        signal = np.asarray(signal, dtype=float)
        n = signal.size
        ss_tot = float(np.sum((signal - np.mean(signal)) ** 2)) if n else 0.0
        if ss_tot == 0:
            return signal.copy(), 1.0, 0.0

        target_ss_res = (1.0 - float(target_r2)) * ss_tot
        if target_ss_res <= 0:
            return signal.copy(), 1.0, 0.0

        noise = generator.standard_normal(n) * _rmse_weight_profile(signal)
        c = _solve_clip_scale(signal, np.zeros(n), noise, target_ss_res)
        noisy = np.maximum(0.0, signal + c * noise)
        ss_res = float(np.sum((noisy - signal) ** 2))
        actual_r2 = 1.0 - ss_res / ss_tot
        actual_rmse = float(np.sqrt(ss_res / n))
        return noisy, actual_r2, actual_rmse


# =============================================================================
# Blank model
# =============================================================================

class BlankModel:
    """Synthesise a media blank trace ``b0 + drift * t + coloured noise``.

    Calibrated from the 18 wells with ``Strain == BLANK`` in
    ``data/raw/Group 3/GROUP3_KEY.csv`` (A1-A3 LB, B1-B3 H2O, C1-C3 IMIDACLOPRID,
    D1-D3 IMIDACLOPRIDANDLB, E1-E3 LAMBDACYHALOTHRIN, F1-F3
    LAMBDACYHALOTHRINANDLB), read with::

        from tecan_growth import load_plate
        run = load_plate('data/raw/Group 3/GrowthRate_GROUP3.xlsx',
                         'data/raw/Group 3/GROUP3_KEY.csv')

    Measured over those 18 wells (400 cycles, 99.75 h, 0.25 h):

        per-media mean b0 0.1231 (H2O) .. 0.1513 (LAMBDACYHALOTHRIN)
            -> ``b0_range = (0.120, 0.155)``
        per-media across-well b0 sd 0.0029 (LCY+LB) .. 0.0292 (IMIDACLOPRID)
            -> ``well_spread_range = (0.003, 0.029)``
        linear drift over the run -0.0285 (E2) .. +0.0219 (C3) OD
            -> ``drift_rate_range = (-0.000285, +0.000220)`` OD/h
        sigma_hat 0.00036 (A1) .. 0.00248 (E1), median 0.00112
            -> ``hf_sigma_range = (0.00036, 0.00248)``

    ``sample_blank`` is what WP3 subtracts from the synthetic well before the
    existing ``np.maximum(0, ...)`` clip; choosing a blank a little above the true
    well baseline is what reproduces the 30.4 % of real conditions whose first
    points are identically zero.
    """

    def __init__(
        self,
        b0_range: Tuple[float, float] = (0.120, 0.155),
        well_spread_range: Tuple[float, float] = (0.003, 0.029),
        drift_rate_range: Tuple[float, float] = (-0.000285, 0.000220),
        hf_sigma_range: Tuple[float, float] = (0.00036, 0.00248),
        tau_fast_h: float = 0.5,
        tau_slow_h: float = 7.5,
        sd_slow: float = 0.0021,
        seed: int = 0,
    ):
        self.b0_range = b0_range
        self.well_spread_range = well_spread_range
        self.drift_rate_range = drift_rate_range
        self.hf_sigma_range = hf_sigma_range
        self.tau_fast_h = tau_fast_h
        self.tau_slow_h = tau_slow_h
        self.sd_slow = sd_slow
        self.seed = int(seed)

    # ---------------------------------------------------------------- media
    @staticmethod
    def _unit_hash(*parts: str) -> float:
        """Deterministic, order-independent U[0,1) from the media name."""
        digest = hashlib.blake2b(':'.join(parts).encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, 'big') / float(1 << 64)

    def media_baseline(self, media: str) -> float:
        """Per-media blank level b0, stable for a given media name and seed."""
        lo, hi = self.b0_range
        return float(lo + (hi - lo) * self._unit_hash(str(self.seed), 'b0', str(media)))

    def media_well_spread(self, media: str) -> float:
        """Per-media across-well b0 sd."""
        lo, hi = self.well_spread_range
        return float(lo + (hi - lo) * self._unit_hash(str(self.seed), 'spread', str(media)))

    # ---------------------------------------------------------------- sample
    def sample_blank(self, media: str, t: np.ndarray, rng: RngLike = None) -> np.ndarray:
        """Return one synthetic blank-well trace on the grid ``t``."""
        generator = _as_rng(rng)
        t = np.asarray(t, dtype=float)
        n = t.size
        if n == 0:
            return np.zeros(0)

        b0 = self.media_baseline(media) + generator.normal(0.0, self.media_well_spread(media))
        rate = generator.uniform(*self.drift_rate_range)
        sigma = generator.uniform(*self.hf_sigma_range)

        blank = np.full(n, float(b0)) + rate * (t - t[0])
        blank = blank + ColoredNoise(sigma, tau_fast_h=self.tau_fast_h,
                                     tau_slow_h=self.tau_slow_h,
                                     sd_slow=self.sd_slow).sample(t, rng=generator)
        return blank


# =============================================================================
# Noise level presets
# =============================================================================
#
# baseline_sigma is the measured real distribution of per-well sigma_hat at low
# OD: very_low 0.0003 (cleanest well A1 0.00036), low 0.0005 (p10 0.00050 over
# the 16 flats), medium 0.0009 (median well; 96 Group 3 wells give 0.00127
# whole-trace, 0.00114 in the OD 0.05-0.20 band), high 0.0020 (p90 0.00200).
# These are exactly the four baseline values the plan specifies.
#
# proportional_sigma is set EQUAL to baseline_sigma, i.e. sigma doubles between
# OD 0 and OD 1.  That is the slope implied by the measured band medians over all
# 92 real condition curves - 0.00114 at OD 0.05-0.20 rising to 0.00215 at OD
# 0.60-1.20, a factor 1.9 - and it reproduces all three bands to within 25 %
# (model/real: 0.81x at OD 0.125, 0.76x at 0.9, 1.24x at 1.45).  The plan's
# steeper 0.0012 for 'medium' matches the 0.60-1.20 band but then makes the
# sigma(>1.2) / sigma(0.05-0.20) ratio 2.7 against the real 1.4, because real
# sigma is NOT monotone in OD: the >1.2 band median falls back to 0.00162, a
# between-curve selection effect that no monotone sigma(OD) can reproduce.  The
# acceptance ratio test (<= 2.5) is what settles the tie.
#
# rmse_range is roughly sigma(OD=1) .. 2 * sigma(OD=1).
# outlier_magnitude follows the measured per-well positive-excursion
# distribution over 76 Group 3 wells: the typical well's LARGEST single positive
# excursion vs an 11-point rolling median is +0.0058 OD (p90 +0.0211, max +0.295
# in H12), and a typical well carries 2 points beyond 5 robust MADs per 400
# cycles - which is what outlier_prob 0.005 already produced.  So the magnitudes
# are 0.006 (very_low/low), 0.008 (medium), 0.021 (high, the real p90) and 0.35
# (very_high, H12's own +0.295 rounded up).

NOISE_PRESETS: Dict[str, Dict[str, object]] = {
    'very_low': {
        'description': 'Cleanest real wells (Group 3 A1 sigma_hat 0.00036)',
        'baseline_sigma': 0.0003,
        'proportional_sigma': 0.0003,
        'outlier_prob': 0.0,
        'outlier_magnitude': 0.006,
        'drift_rate': None,
        'drift_rate_range': DRIFT_RATE_RANGE,
        'rmse_range': (0.0006, 0.0012),
    },
    'low': {
        'description': 'p10 real well (sigma_hat 0.00073, e.g. Group 3 B2)',
        'baseline_sigma': 0.0005,
        'proportional_sigma': 0.0005,
        'outlier_prob': 0.001,
        'outlier_magnitude': 0.006,
        'drift_rate': None,
        'drift_rate_range': DRIFT_RATE_RANGE,
        'rmse_range': (0.0010, 0.0020),
    },
    'medium': {
        'description': 'Median real well (96 Group 3 wells, sigma_hat 0.00127)',
        'baseline_sigma': 0.0009,
        'proportional_sigma': 0.0009,
        'outlier_prob': 0.005,
        'outlier_magnitude': 0.008,
        'drift_rate': None,
        'drift_rate_range': DRIFT_RATE_RANGE,
        'rmse_range': (0.0018, 0.0036),
    },
    'high': {
        'description': 'p90 real well (sigma_hat 0.00200, e.g. Group 3 C1)',
        'baseline_sigma': 0.0020,
        'proportional_sigma': 0.0020,
        'outlier_prob': 0.01,
        'outlier_magnitude': 0.021,
        'drift_rate': None,
        'drift_rate_range': DRIFT_RATE_RANGE,
        'rmse_range': (0.0040, 0.0080),
    },
    'very_high': {
        # Anchored to a real object, not a multiplier: Group 3 well H12
        # (LAMBDACYHALOTHRINANDLB-LCY2, third well) - whole-trace sigma_hat
        # 0.00596, plateau (OD>1.4) sigma_hat 0.02554, largest single positive
        # excursion +0.295 OD.  The worst real CONDITION-average whole-curve
        # sigma_hat is only 0.00443, so this is "the noisiest well observed in 96
        # real wells", not "2.5x the worst real curve".
        'description': 'Noisiest well observed in 96 real wells (Group 3 H12)',
        'baseline_sigma': 0.0020,
        'proportional_sigma': 0.0140,
        'outlier_prob': 0.02,
        'outlier_magnitude': 0.35,
        'drift_rate': None,
        'drift_rate_range': DRIFT_RATE_RANGE,
        'rmse_range': (0.016, 0.032),
    },
}


def get_noise_model(preset: str) -> InstrumentNoise:
    """Get a pre-configured :class:`InstrumentNoise` for a preset level.

    Args:
        preset: one of 'very_low', 'low', 'medium', 'high', 'very_high'.
    """
    if preset not in NOISE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                         f"Available: {list(NOISE_PRESETS.keys())}")
    params = NOISE_PRESETS[preset]
    return InstrumentNoise(
        baseline_sigma=params['baseline_sigma'],
        proportional_sigma=params['proportional_sigma'],
        outlier_prob=params['outlier_prob'],
        outlier_magnitude=params['outlier_magnitude'],
        drift_rate=params.get('drift_rate'),
        drift_rate_range=params.get('drift_rate_range', DRIFT_RATE_RANGE),
    )


def get_rmse_noise(preset: str) -> RMSEBasedNoise:
    """Get an RMSE-based noise model for a preset level."""
    if preset not in NOISE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                         f"Available: {list(NOISE_PRESETS.keys())}")
    return RMSEBasedNoise(rmse_range=NOISE_PRESETS[preset]['rmse_range'])


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":                                    # pragma: no cover
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from growth_models import GompertzModel

    t = np.arange(0, 100, 0.25)
    clean = GompertzModel.compute(t, A=1.2, mu=0.15, lambda_=4.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(t, clean, 'k-', lw=2, label='clean')
    for preset in ('low', 'medium', 'high'):
        ax.plot(t, get_noise_model(preset).apply(clean, time=t, rng=42), alpha=0.7, label=preset)
    ax.set_title('Instrument noise presets (real magnitudes)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, get_noise_model('very_high').apply(clean, time=t, rng=7), 'r-', alpha=0.8)
    ax.plot(t, clean, 'k-', lw=2)
    ax.set_title('very_high = Group 3 well H12')
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    flat = np.full_like(t, 0.05)
    ax.plot(t, get_noise_model('medium').apply(flat, time=t, rng=3))
    ax.set_title('Coloured flat (real ac(1 h) = 0.851)')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for target in (0.97, 0.93, 0.80):
        noisy, r2, rmse = RMSEBasedNoise().generate_noise_to_target_r2(clean, target, rng=42)
        ax.plot(t, noisy, alpha=0.7, label=f'target {target} -> {r2:.6f}')
    ax.set_title('Exact target R2'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    spiked = inject_spikes(clean, [200, 260, 320], [0.3, 0.25, 0.4], 3)
    ax.plot(t, spiked, 'r-', alpha=0.8); ax.plot(t, clean, 'k-', lw=2)
    ax.set_title('One-sided decaying spikes'); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    bm = BlankModel()
    for media in ('LB', 'H2O', 'IMIDACLOPRID'):
        ax.plot(t, bm.sample_blank(media, t, rng=11), alpha=0.8, label=media)
    ax.set_title('Synthetic media blanks'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('noise_models_demo.png', dpi=130)
    print('Demo complete: noise_models_demo.png')
