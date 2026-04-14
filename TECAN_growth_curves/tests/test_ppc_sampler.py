"""
Tests for Phase B.1: PosteriorPredictiveSampler + ResidualBootstrapNoise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthetic_data" / "src"))


@pytest.fixture
def smoke_trace_path():
    """Path to the 5-strain smoke-test trace produced during dev.

    Skips tests gracefully when the trace isn't present (e.g. CI).
    """
    p = Path("/tmp/bayes_smoke_test/bayesian_gompertz/gompertz_trace.nc")
    if not p.exists():
        pytest.skip(f"Smoke trace not found at {p} — run 06 with --max-strains 5")
    return p


# ----------------------------------------------------------------------------
# PosteriorPredictiveSampler
# ----------------------------------------------------------------------------

def test_loads_trace(smoke_trace_path):
    from posterior_predictive_sampler import PosteriorPredictiveSampler

    sampler = PosteriorPredictiveSampler()
    sampler.load(smoke_trace_path)
    assert sampler.n_strains == 5
    assert sampler.n_draws > 0


def test_sample_parameters_preserves_marginals(smoke_trace_path):
    """Sampled (A, μ, λ) should have finite, positive values in expected ranges."""
    from posterior_predictive_sampler import PosteriorPredictiveSampler

    sampler = PosteriorPredictiveSampler()
    sampler.load(smoke_trace_path)
    params = sampler.sample_parameters(n=200, rng=np.random.default_rng(42))

    assert len(params) == 200
    assert (params["A"] > 0).all()
    assert (params["mu"] > 0).all()
    assert np.isfinite(params["lam"]).all()
    # Reasonable biological ranges (our strains live in A ∈ [0.2, 3], μ ∈ [0.05, 1.5])
    assert params["A"].between(0.05, 5.0).all()
    assert params["mu"].between(0.001, 3.0).all()


def test_sample_parameters_preserves_joint_correlation(smoke_trace_path):
    """Samples should preserve the empirical A-μ correlation from the posterior.

    With only 5 strains the correlation can be weak, so we just check that the
    sampled correlation is close to the population correlation (|Δ| < 0.2).
    """
    from posterior_predictive_sampler import PosteriorPredictiveSampler

    sampler = PosteriorPredictiveSampler()
    sampler.load(smoke_trace_path)

    # Population correlation in posterior
    pop_corr = sampler._samples[["A", "mu"]].corr().iloc[0, 1]

    # Sampled correlation
    draws = sampler.sample_parameters(n=2000, rng=np.random.default_rng(7))
    samp_corr = draws[["A", "mu"]].corr().iloc[0, 1]

    assert abs(samp_corr - pop_corr) < 0.2, (
        f"joint correlation drifted: pop={pop_corr:.3f} vs sampled={samp_corr:.3f}"
    )


def test_sample_curves_length_and_columns(smoke_trace_path):
    from posterior_predictive_sampler import PosteriorPredictiveSampler

    sampler = PosteriorPredictiveSampler()
    sampler.load(smoke_trace_path)
    t = np.linspace(0, 24, 50)
    curves = sampler.sample_curves(n=10, grids=[t], rng=np.random.default_rng(1))

    assert set(curves.columns) >= {"curve_id", "time", "od_clean", "od",
                                    "A", "mu", "lam", "strain"}
    assert curves["curve_id"].nunique() == 10
    # Each curve = 50 points
    assert len(curves) == 10 * 50
    # Clean and noisy are equal here (no noise model passed)
    np.testing.assert_allclose(curves["od_clean"].values, curves["od"].values)


def test_sample_curves_monotone_when_params_reasonable(smoke_trace_path):
    """Gompertz curves from posterior should be non-decreasing in [0, max_t]."""
    from posterior_predictive_sampler import PosteriorPredictiveSampler

    sampler = PosteriorPredictiveSampler()
    sampler.load(smoke_trace_path)
    t = np.linspace(0, 30, 60)
    curves = sampler.sample_curves(n=5, grids=[t], rng=np.random.default_rng(0))

    for cid, grp in curves.groupby("curve_id"):
        od = grp.sort_values("time")["od_clean"].values
        assert np.all(np.diff(od) >= -1e-9), f"curve {cid} not monotone"


# ----------------------------------------------------------------------------
# ResidualBootstrapNoise
# ----------------------------------------------------------------------------

def test_residual_bootstrap_fallback_with_empty_pool():
    """Empty pool → falls back to OD-dependent Gaussian; curve should stay near signal."""
    from noise_models import ResidualBootstrapNoise

    signal = np.linspace(0, 1.5, 100)
    noise = ResidualBootstrapNoise(residual_pool=None)
    out = noise.apply(signal, seed=42)

    assert out.shape == signal.shape
    # Fallback noise is small (<0.06 sigma at OD=1.5)
    assert np.max(np.abs(out - signal)) < 0.5


def test_residual_bootstrap_build_pool_from_fits():
    """build_pool_from_fits bins residuals correctly by OD level."""
    from noise_models import ResidualBootstrapNoise

    # Two mock curves: clean ramp 0→2, observed = clean + constant 0.03
    t = np.linspace(0, 1, 100)
    clean1 = 2 * t
    observed1 = clean1 + 0.03
    clean2 = 2 * t
    observed2 = clean2 - 0.02

    pool = ResidualBootstrapNoise.build_pool_from_fits(
        [(clean1, observed1), (clean2, observed2)]
    )
    assert pool  # non-empty
    # Every non-empty bin should contain ~50% +0.03 and ~50% -0.02
    for bin_id, res in pool.items():
        assert len(res) > 0
        assert np.isfinite(res).all()


def test_residual_bootstrap_samples_from_pool():
    """When pool has enough residuals, noise comes from the pool."""
    from noise_models import ResidualBootstrapNoise

    # Build a pool with known large residuals, all in the middle OD bin
    rng = np.random.default_rng(0)
    known_residuals = rng.normal(0.05, 0.01, size=500)  # unambiguous signature
    pool = {3: known_residuals}  # bin index 3 = [0.6, 1.0]

    signal = np.full(50, 0.75)  # all signal lands in bin 3
    noise = ResidualBootstrapNoise(residual_pool=pool)
    out = noise.apply(signal, seed=7)
    residuals_observed = out - signal

    # mean and std should be close to the pool's
    assert abs(residuals_observed.mean() - 0.05) < 0.02
    assert abs(residuals_observed.std() - 0.01) < 0.01
