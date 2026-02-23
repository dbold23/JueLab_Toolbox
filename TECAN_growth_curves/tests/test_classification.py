"""
Tests for growth curve classification logic.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the classification function
from importlib import import_module


def _import_analysis():
    """Dynamically import 01_growth_curve_analysis without running __main__."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analysis",
        str(Path(__file__).parent.parent / "scripts" / "01_growth_curve_analysis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analysis():
    return _import_analysis()


class TestClassifyByFitQuality:
    """Test the classify_by_fit_quality function."""

    def _make_fit_result(self, analysis, **overrides):
        """Create a FitResult with sensible defaults."""
        defaults = dict(
            success=True,
            a_opt=1.2, mu_opt=0.2, lambda_opt=4.0,
            a_err=0.05, mu_err=0.01, lambda_err=0.5,
            mae=0.01, mse=0.0001, rmse=0.01,
            r_squared=0.98,
            predicted=np.zeros(100),
            residuals=np.zeros(100),
            error_message=""
        )
        defaults.update(overrides)
        return analysis.FitResult(**defaults)

    def test_good_fit_classified_good(self, analysis):
        """High R², low errors -> GOOD."""
        fit = self._make_fit_result(analysis, r_squared=0.98, a_err=0.05, mu_err=0.01)
        result = analysis.classify_by_fit_quality(fit)
        assert result.is_good is True
        assert "GOOD" in result.reason

    def test_low_r2_classified_bad(self, analysis):
        """Low R² -> BAD."""
        fit = self._make_fit_result(analysis, r_squared=0.85)
        result = analysis.classify_by_fit_quality(fit)
        assert result.is_good is False
        assert "R²" in result.reason

    def test_high_param_error_classified_bad(self, analysis):
        """Large parameter error -> BAD."""
        fit = self._make_fit_result(analysis, a_err=0.5)  # 41.7% of a_opt=1.2
        result = analysis.classify_by_fit_quality(fit)
        assert result.is_good is False
        assert "error" in result.reason.lower()

    def test_failed_fit_classified_bad(self, analysis):
        """Failed fit -> BAD."""
        fit = self._make_fit_result(analysis, success=False, r_squared=0)
        result = analysis.classify_by_fit_quality(fit)
        assert result.is_good is False

    def test_secondary_gates_with_raw_data(self, analysis):
        """Low delta-OD should be caught by secondary gates."""
        fit = self._make_fit_result(analysis, r_squared=0.96, a_opt=0.05)
        # Create raw data with no real growth
        od = np.full(100, 0.01) + np.random.default_rng(42).normal(0, 0.003, 100)
        time = np.linspace(0, 50, 100)
        result = analysis.classify_by_fit_quality(fit, time=time, od600=od)
        assert result.is_good is False

    def test_excellent_fit_bypasses_snr_gate(self, analysis):
        """R² >= 0.98 should bypass SNR-based rejection."""
        fit = self._make_fit_result(analysis, r_squared=0.99, a_opt=1.2)
        # Noisy baseline but real growth
        rng = np.random.default_rng(42)
        time = np.linspace(0, 50, 200)
        from growth_models import GompertzModel
        clean = GompertzModel.compute(time, 1.2, 0.2, 4.0)
        od = clean + rng.normal(0, 0.1, len(clean))  # Heavy noise
        result = analysis.classify_by_fit_quality(fit, time=time, od600=od)
        # Should still be GOOD because R² >= 0.98 bypasses SNR gate
        assert result.is_good is True
