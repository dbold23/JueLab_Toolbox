"""
Tests for Gompertz model fitting and related functions.
"""
import numpy as np
import pytest

from growth_models import GompertzModel


class TestGompertzModel:
    """Test the Gompertz model computation."""

    def test_compute_basic_shape(self, standard_time):
        """Gompertz output should be monotonically increasing (no death phase)."""
        od = GompertzModel.compute(standard_time, A=1.5, mu=0.2, lambda_=3.0)
        assert od[0] < od[-1]
        assert np.all(np.diff(od) >= -1e-10)  # monotonically increasing

    def test_asymptote_approaches_A(self, standard_time):
        """Final OD should approach A for long experiments."""
        A = 1.5
        od = GompertzModel.compute(standard_time, A=A, mu=0.2, lambda_=3.0)
        assert abs(od[-1] - A) < 0.05  # within 5% of A

    def test_zero_growth_rate(self, standard_time):
        """mu~0 should produce a nearly flat curve (Gompertz lower asymptote = A*exp(-e))."""
        od = GompertzModel.compute(standard_time, A=1.0, mu=1e-10, lambda_=3.0)
        # With mu~0, curve stays at its lower asymptote ~A*exp(-exp(1)) ≈ 0.066
        assert np.ptp(od) < 0.01  # no change over time (flat)

    def test_derivative_peaks_near_inflection(self, standard_time):
        """Growth rate should peak near the inflection point."""
        A, mu, lambda_ = 1.5, 0.2, 5.0
        dy = GompertzModel.derivative(standard_time, A, mu, lambda_)
        t_inf_expected, _ = GompertzModel.inflection_point(A, mu, lambda_)
        t_peak = standard_time[np.argmax(dy)]
        assert abs(t_peak - t_inf_expected) < 1.0  # within 1 hour

    def test_parameter_recovery(self, standard_time):
        """Fitting clean Gompertz data should recover original parameters."""
        from scipy.optimize import curve_fit

        A_true, mu_true, lam_true = 1.2, 0.2, 4.0
        clean = GompertzModel.compute(standard_time, A_true, mu_true, lam_true)

        # Add very light noise
        rng = np.random.default_rng(42)
        noisy = clean + rng.normal(0, 0.005, len(clean))

        def gompertz_func(t, A, mu, lam):
            return GompertzModel.compute(t, A, mu, lam)

        popt, _ = curve_fit(gompertz_func, standard_time, noisy, p0=[1.0, 0.15, 3.0])
        assert abs(popt[0] - A_true) < 0.1   # A within 0.1
        assert abs(popt[1] - mu_true) < 0.05  # mu within 0.05
        assert abs(popt[2] - lam_true) < 1.0  # lambda within 1 hour


class TestGompertzFit:
    """Test the pipeline's Gompertz fitting function (fit_gompertz)."""

    @pytest.fixture(scope="class")
    def analysis(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "analysis",
            str(Path(__file__).parent.parent / "scripts" / "01_growth_curve_analysis.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_fit_good_curve_succeeds(self, analysis, good_growth_curve):
        """fit_gompertz should succeed and recover parameters on a clean growth curve."""
        time, noisy, clean = good_growth_curve
        result = analysis.fit_gompertz(time, noisy)
        assert result.success
        assert result.r_squared > 0.95
        assert result.a_opt > 0.5

    def test_fit_good_curve_parameter_recovery(self, analysis, good_growth_curve):
        """Fitted parameters should be close to ground truth."""
        time, noisy, clean = good_growth_curve
        result = analysis.fit_gompertz(time, noisy)
        # The fixture uses A=1.5, mu=0.2, lambda=3.0 (with noise)
        assert abs(result.a_opt - 1.5) < 0.5
        assert abs(result.mu_opt - 0.2) < 0.15

    def test_fit_flat_curve_poor_r2(self, analysis, flat_curve):
        """fit_gompertz on flat data should produce poor R² or fail."""
        time, od = flat_curve
        result = analysis.fit_gompertz(time, od)
        # Either fails or has terrible R²
        if result.success:
            assert result.r_squared < 0.5

    def test_fit_constant_data(self, analysis):
        """Fitting constant OD data should not crash."""
        time = np.linspace(0, 24, 100)
        od = np.full_like(time, 0.1)
        result = analysis.fit_gompertz(time, od)
        # Should not raise; may succeed with degenerate fit or fail gracefully
        assert hasattr(result, 'success')

    def test_fit_very_short_array(self, analysis):
        """Fitting with <5 points should not crash."""
        time = np.array([0, 1, 2])
        od = np.array([0.1, 0.2, 0.3])
        result = analysis.fit_gompertz(time, od)
        assert hasattr(result, 'success')

    def test_fit_duplicate_timestamps(self, analysis):
        """Duplicate timestamps should not cause division by zero."""
        time = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        od = np.array([0.1, 0.15, 0.15, 0.3, 0.5, 0.7, 0.85, 0.9, 0.92, 0.93, 0.93, 0.93])
        result = analysis.fit_gompertz(time, od)
        # Should not crash (was the C1 bug); may or may not converge
        assert hasattr(result, 'success')
