"""
Tests for advanced statistical fitting methods (06_advanced_fitting.py).

Covers: GP truncation, bootstrap CIs, Bayesian model building,
Bayesian classification, and model comparison.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---- Dynamic import of 06_advanced_fitting.py ----
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

spec = importlib.util.spec_from_file_location(
    "advanced_fitting",
    str(SCRIPTS_DIR / "06_advanced_fitting.py")
)
adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adv)


# ---- Fixtures ----

@pytest.fixture
def standard_time():
    """100-point time array over 24 hours."""
    return np.linspace(0, 24, 100)


@pytest.fixture
def good_gompertz_curve(standard_time):
    """Clean Gompertz curve: A=1.2, mu=0.2, lambda=4.0."""
    np.random.seed(42)
    od = adv.gompertz_model(standard_time, 1.2, 0.2, 4.0)
    od += np.random.normal(0, 0.01, len(od))
    return standard_time, od


@pytest.fixture
def flat_curve(standard_time):
    """Flat noisy curve (no growth)."""
    np.random.seed(42)
    return standard_time, 0.05 + np.random.normal(0, 0.01, len(standard_time))


@pytest.fixture
def haldane_curve():
    """Synthetic Haldane curve: mild inhibition."""
    t = np.linspace(0, 20, 100)
    X, S = adv.solve_haldane(t, mu_max=0.5, Ks=0.2, Ki=10.0,
                              X_max=1.5, q=0.1, X0=0.01, S0=1.0)
    np.random.seed(42)
    X += np.random.normal(0, 0.005, len(X))
    return t, X


# =============================================================================
# GP Truncation Tests
# =============================================================================

class TestGPTruncation:

    def test_gp_fit_clean_sigmoid(self, good_gompertz_curve):
        """GP should fit a clean Gompertz curve with high quality."""
        t, od = good_gompertz_curve
        gp, t_dense, mu, sigma = adv.fit_gp(t, od)
        # GP predictions at training points should be close
        mu_at_data, _ = gp.predict(t.reshape(-1, 1), return_std=True)
        residuals = od - mu_at_data
        r2 = 1 - np.sum(residuals**2) / np.sum((od - np.mean(od))**2)
        assert r2 > 0.99, f"GP R² = {r2:.4f}, expected > 0.99"

    def test_gp_truncation_finds_stationary(self, good_gompertz_curve):
        """GP truncation should find stationary onset near expected region."""
        t, od = good_gompertz_curve
        trunc_idx, conf, phases, _ = adv.gp_truncate(t, od)
        # For A=1.2, mu=0.2, lambda=4.0 the curve plateaus around t≈16-20h
        trunc_time = t[trunc_idx]
        assert 10.0 < trunc_time < 24.0, \
            f"Truncation at {trunc_time:.1f}h, expected 10-24h"

    def test_gp_identifies_phases(self, good_gompertz_curve):
        """GP should identify lag, exponential, and stationary phases."""
        t, od = good_gompertz_curve
        _, _, mu, _ = adv.fit_gp(t, od)
        t_dense = np.linspace(t[0], t[-1], 500)
        phases = adv.gp_find_phases(t_dense, mu)

        assert phases['lag_end'] is not None, "Should identify lag end"
        assert phases['exp_peak'] is not None, "Should identify exponential peak"
        # Lag should end before peak growth
        if phases['lag_end'] is not None and phases['exp_peak'] is not None:
            assert phases['lag_end'] <= phases['exp_peak']

    def test_gp_handles_sparse_data(self):
        """GP should work on sparse data (20 points)."""
        t = np.linspace(0, 24, 20)
        np.random.seed(42)
        od = adv.gompertz_model(t, 1.0, 0.3, 3.0) + np.random.normal(0, 0.02, 20)
        trunc_idx, conf, phases, _ = adv.gp_truncate(t, od)
        assert trunc_idx >= 5, "Should find a valid truncation point"


# =============================================================================
# Bootstrap Tests
# =============================================================================

class TestBootstrap:

    def test_bootstrap_ci_contains_true_params(self, good_gompertz_curve):
        """Bootstrap 95% CI should contain the true parameters."""
        t, od = good_gompertz_curve
        result = adv.bootstrap_gompertz(t, od, n_resamples=200, ci_level=0.95)
        assert result is not None, "Bootstrap should succeed"

        # True params: A=1.2, mu=0.2, lam=4.0
        assert result['A_ci'][0] < 1.2 < result['A_ci'][1], \
            f"A CI {result['A_ci']} should contain 1.2"
        assert result['mu_ci'][0] < 0.2 < result['mu_ci'][1], \
            f"mu CI {result['mu_ci']} should contain 0.2"
        assert result['lam_ci'][0] < 4.0 < result['lam_ci'][1], \
            f"lam CI {result['lam_ci']} should contain 4.0"

    def test_bootstrap_pgood_high_for_good_curve(self, good_gompertz_curve):
        """P(good) should be high for a clean growth curve."""
        t, od = good_gompertz_curve
        result = adv.bootstrap_gompertz(t, od, n_resamples=200)
        assert result is not None
        assert result['p_good'] >= 0.90, \
            f"P(good) = {result['p_good']:.2f}, expected >= 0.90"

    def test_bootstrap_pgood_low_for_flat_curve(self, flat_curve):
        """P(good) should be low for a flat/noisy curve."""
        t, od = flat_curve
        result = adv.bootstrap_gompertz(t, od, n_resamples=200)
        # Flat curve may fail fitting entirely (result=None) or have low P(good)
        if result is not None:
            assert result['p_good'] < 0.50, \
                f"P(good) = {result['p_good']:.2f}, expected < 0.50 for flat curve"


# =============================================================================
# Haldane ODE Tests
# =============================================================================

class TestHaldaneODE:

    def test_solve_haldane_produces_growth(self, haldane_curve):
        """Haldane ODE should produce increasing biomass."""
        t, od = haldane_curve
        X, S = adv.solve_haldane(t, 0.5, 0.2, 10.0, 1.5, 0.1, 0.01, 1.0)
        assert X[-1] > X[0], "Biomass should increase"
        assert S[-1] < S[0], "Substrate should decrease"

    def test_solve_haldane_substrate_depletion(self):
        """Substrate should deplete over time."""
        t = np.linspace(0, 50, 200)
        X, S = adv.solve_haldane(t, 0.5, 0.2, 10.0, 1.5, 0.5, 0.01, 1.0)
        assert S[-1] < 0.5 * S[0], "Substrate should deplete significantly"

    def test_strong_inhibition_slows_growth(self):
        """Low Ki should produce slower growth than high Ki."""
        t = np.linspace(0, 30, 100)
        X_weak, _ = adv.solve_haldane(t, 0.5, 0.2, 100.0, 1.5, 0.1, 0.01, 1.0)
        X_strong, _ = adv.solve_haldane(t, 0.5, 0.2, 0.5, 1.5, 0.1, 0.01, 1.0)
        # Weak inhibition (high Ki) should reach higher biomass
        assert np.max(X_weak) > np.max(X_strong), \
            "Weak inhibition should allow more growth"


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestDataLoading:

    def test_extract_pesticide_name(self):
        assert adv.extract_pesticide_name('BifenthrinANDLB-BIF2') == 'BIFENTHRIN'
        assert adv.extract_pesticide_name('MALATHIONANDLB-MAL8') == 'MALATHION'
        assert adv.extract_pesticide_name('LB-BIF2') == 'UNKNOWN'

    def test_identify_pesticide_strains(self):
        df = pd.DataFrame({
            'strain': ['BifenthrinANDLB-BIF2', 'LB-BIF2', 'MALATHIONANDLB-MAL8'],
        })
        result = adv.identify_pesticide_strains(df)
        assert len(result) == 2
        assert 'LB-BIF2' not in result['strain'].values

    def test_load_config(self):
        config = adv.load_config()
        # May or may not find config.yaml depending on cwd
        assert isinstance(config, dict)


# =============================================================================
# Integration Tests (require pipeline outputs)
# =============================================================================

class TestAdvancedOutputs:

    @pytest.fixture
    def results_dir(self):
        return Path(__file__).parent.parent / "results" / "tables"

    def test_gp_output_exists(self, results_dir):
        path = results_dir / "Advanced_Analysis" / "gp_truncation"
        if not path.exists():
            pytest.skip("GP truncation not yet run")
        csv = path / "gp_truncation_results.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert len(df) > 0
        assert 'truncation_time' in df.columns

    def test_bootstrap_output_exists(self, results_dir):
        path = results_dir / "Advanced_Analysis" / "bootstrap"
        if not path.exists():
            pytest.skip("Bootstrap not yet run")
        csv = path / "bootstrap_ci.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert len(df) > 0
        assert 'p_good' in df.columns
        assert 'mu_ci_low' in df.columns


class TestThinning:

    def test_thin_preserves_short_data(self):
        """Data shorter than max_points should be unchanged."""
        data = [(np.linspace(0, 10, 30), np.random.rand(30))]
        result = adv.thin_strain_data(data, max_points=50)
        assert len(result[0][0]) == 30

    def test_thin_reduces_long_data(self):
        """Data longer than max_points should be reduced."""
        t = np.linspace(0, 24, 200)
        od = np.random.rand(200)
        result = adv.thin_strain_data([(t, od)], max_points=50)
        assert len(result[0][0]) <= 50

    def test_thin_preserves_endpoints(self):
        """First and last points should be kept after thinning."""
        t = np.linspace(0, 24, 200)
        od = np.random.rand(200)
        result = adv.thin_strain_data([(t, od)], max_points=50)
        assert result[0][0][0] == t[0], "First point should be preserved"
        assert result[0][0][-1] == t[-1], "Last point should be preserved"
