"""
Tests for Haldane/Andrews substrate inhibition model.
"""
import numpy as np
import pytest

from growth_models import HaldaneModel


class TestHaldaneModel:
    """Test the Haldane model ODE solver."""

    def test_mild_inhibition_reaches_carrying_capacity(self, standard_time):
        """With mild inhibition (high Ki), biomass should approach X_max."""
        X = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=10.0,
            X_max=1.2, q=0.1, X0=0.01, S0=1.0
        )
        assert X[-1] > 1.0  # should reach near X_max=1.2
        assert X[-1] <= 1.25

    def test_total_inhibition_prevents_growth(self, standard_time):
        """Very high substrate + low Ki should prevent growth."""
        X = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=0.05,
            X_max=1.0, q=0.1, X0=0.01, S0=5.0
        )
        assert X[-1] < 0.1  # minimal growth due to inhibition

    def test_substrate_depletes(self, standard_time):
        """Substrate should decrease as bacteria grow."""
        X, S = HaldaneModel.compute_full(
            standard_time, mu_max=0.5, Ks=0.1, Ki=10.0,
            X_max=1.2, q=0.5, X0=0.01, S0=1.0
        )
        assert S[-1] < S[0]  # substrate consumed
        assert S[-1] >= 0     # can't go negative

    def test_biomass_monotonic(self, standard_time):
        """Biomass should be monotonically non-decreasing (no death phase in Haldane)."""
        X = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=10.0,
            X_max=1.2, q=0.1, X0=0.01, S0=1.0
        )
        assert np.all(np.diff(X) >= -1e-8)

    def test_from_gompertz_params(self):
        """Conversion from Gompertz params should produce valid Haldane params."""
        params = HaldaneModel.from_gompertz_params(A=1.2, mu=0.2, lambda_=4.0)
        assert params['mu_max'] > 0
        assert params['Ks'] > 0
        assert params['Ki'] > 0
        assert params['X_max'] == 1.2  # should match Gompertz A
        assert params['X0'] > 0
        assert params['S0'] > 0

    def test_inhibition_constant_effect(self, standard_time):
        """Lower Ki should produce less growth (more inhibition)."""
        X_high_ki = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=100.0,
            X_max=1.2, q=0.1, X0=0.01, S0=1.0
        )
        X_low_ki = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=0.5,
            X_max=1.2, q=0.1, X0=0.01, S0=1.0
        )
        # At midpoint, high Ki should have grown more
        mid = len(standard_time) // 2
        assert X_high_ki[mid] > X_low_ki[mid]


class TestHaldaneFit:
    """Test the Haldane fitting function from 05_haldane_analysis.py."""

    def test_fit_on_haldane_generated_data(self, standard_time):
        """Fitting Haldane data with Haldane model should give good R²."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "haldane_analysis",
            str(Path(__file__).parent.parent / "scripts" / "05_haldane_analysis.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Generate clean Haldane data
        X_true = HaldaneModel.compute(
            standard_time, mu_max=0.5, Ks=0.1, Ki=10.0,
            X_max=1.2, q=0.1, X0=0.01, S0=1.0
        )
        rng = np.random.default_rng(42)
        X_noisy = X_true + rng.normal(0, 0.01, len(X_true))

        result = mod.fit_haldane(standard_time, X_noisy, S0=1.0)
        assert result['success'] is True
        assert result['r_squared'] > 0.95
