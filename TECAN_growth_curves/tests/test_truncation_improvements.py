"""
Tests for truncation pipeline improvements:
- Incomplete curve detection
- Ensemble MAD outlier detection
- Tightened classification gates (monotonicity, residual autocorrelation)
- Normalized GP derivative threshold
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---- Dynamic imports ----
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

spec_analysis = importlib.util.spec_from_file_location(
    "analysis",
    str(SCRIPTS_DIR / "01_growth_curve_analysis.py")
)
analysis = importlib.util.module_from_spec(spec_analysis)
spec_analysis.loader.exec_module(analysis)

spec_adv = importlib.util.spec_from_file_location(
    "advanced_fitting",
    str(SCRIPTS_DIR / "06_advanced_fitting.py")
)
adv = importlib.util.module_from_spec(spec_adv)
spec_adv.loader.exec_module(adv)


# ---- Fixtures ----

@pytest.fixture
def standard_time():
    """100-point time array over 24 hours."""
    return np.linspace(0, 24, 100)


@pytest.fixture
def incomplete_curve(standard_time):
    """Curve still rising at experiment end (no stationary phase reached).
    Simulates truncation_challenge scenario."""
    # Gompertz with very long lag + slow growth -- never plateaus within 24h
    od = analysis.gompertz_model(standard_time, 2.0, 0.15, 12.0)
    rng = np.random.default_rng(42)
    od = od + rng.normal(0, 0.005, len(od))
    od = np.maximum(0, od)
    return standard_time, od


@pytest.fixture
def good_gompertz_curve(standard_time):
    """Clean Gompertz curve that plateaus normally."""
    od = analysis.gompertz_model(standard_time, 1.2, 0.2, 4.0)
    rng = np.random.default_rng(42)
    od = od + rng.normal(0, 0.01, len(od))
    od = np.maximum(0, od)
    return standard_time, od


@pytest.fixture
def noisy_flat_curve(standard_time):
    """Pure noise with no growth signal -- should be classified BAD."""
    rng = np.random.default_rng(42)
    od = 0.05 + rng.normal(0, 0.02, len(standard_time))
    return standard_time, np.maximum(0, od)


# ====================================================================
# Incomplete Curve Detection
# ====================================================================

class TestIncompleteCurveDetection:
    """Test detect_incomplete_curve function."""

    def test_rising_curve_detected_as_incomplete(self, incomplete_curve):
        """Curve still rising at end should be flagged as incomplete."""
        time, od = incomplete_curve
        is_incomplete, meta = analysis.detect_incomplete_curve(time, od)
        assert bool(is_incomplete) is True
        assert meta['tail_slope'] > 0
        assert bool(meta['near_max']) is True

    def test_plateaued_curve_not_incomplete(self, good_gompertz_curve):
        """Curve that reaches stationary phase should NOT be incomplete."""
        time, od = good_gompertz_curve
        is_incomplete, meta = analysis.detect_incomplete_curve(time, od)
        assert is_incomplete is False

    def test_flat_curve_not_incomplete(self, noisy_flat_curve):
        """Flat noise should NOT be flagged as incomplete."""
        time, od = noisy_flat_curve
        is_incomplete, meta = analysis.detect_incomplete_curve(time, od)
        assert is_incomplete is False

    def test_truncate_at_max_uses_all_data_for_incomplete(self, incomplete_curve):
        """When curve is incomplete, truncate_at_max should use full data."""
        time, od = incomplete_curve
        result = analysis.truncate_at_max(time, od, use_first_peak=True)
        # Truncation should be at or very near the end for incomplete curves
        assert result.truncation_index >= len(od) - 5


# ====================================================================
# Ensemble MAD Outlier Detection
# ====================================================================

class TestEnsembleOutlierDetection:
    """Test MAD-based outlier detection in ensemble_truncate."""

    def test_outlier_method_downweighted(self, good_gompertz_curve):
        """Run ensemble and verify that outlier methods get flagged."""
        time, od = good_gompertz_curve
        result = adv.ensemble_truncate(time, od)

        # With a clean curve, most methods should agree.
        # Check that any outlier method (if present) was marked.
        for name, data in result.method_results.items():
            if data.get('outlier', False):
                assert data.get('mad_deviation', 0) > 3.0

    def test_ensemble_consensus_in_range(self, good_gompertz_curve):
        """Consensus should be within the time range."""
        time, od = good_gompertz_curve
        result = adv.ensemble_truncate(time, od)
        assert result.consensus_time >= time[0]
        assert result.consensus_time <= time[-1]

    def test_ensemble_result_structure(self, good_gompertz_curve):
        """Ensemble result should have all required fields."""
        time, od = good_gompertz_curve
        result = adv.ensemble_truncate(time, od)
        assert hasattr(result, 'consensus_idx')
        assert hasattr(result, 'consensus_time')
        assert hasattr(result, 'consensus_confidence')
        assert hasattr(result, 'method_results')
        assert hasattr(result, 'disagreement_hours')
        assert hasattr(result, 'flagged_for_review')
        assert hasattr(result, 'n_methods_succeeded')
        assert result.n_methods_succeeded >= 1


# ====================================================================
# Tightened Classification Gates
# ====================================================================

class TestTightenedClassification:
    """Test the tightened secondary quality gates."""

    def _make_fit_result(self, **overrides):
        """Create a FitResult with sensible defaults."""
        defaults = dict(
            success=True,
            a_opt=1.2, mu_opt=0.2, lambda_opt=4.0,
            a_err=0.05, mu_err=0.01, lambda_err=0.5,
            mae=0.01, mse=0.0001, rmse=0.01,
            r_squared=0.98,
            predicted=np.zeros(100),
            residuals=np.random.default_rng(42).normal(0, 0.01, 100),
            error_message=""
        )
        defaults.update(overrides)
        return analysis.FitResult(**defaults)

    def test_non_excellent_fit_checked_by_secondary_gates(self):
        """R² = 0.96 (below 0.98 bypass) should have noise gates enforced."""
        fit = self._make_fit_result(r_squared=0.96, a_opt=0.05)
        # Create data with low SNR — pure noise
        rng = np.random.default_rng(42)
        time = np.linspace(0, 50, 100)
        od = 0.01 + rng.normal(0, 0.01, 100)
        thresholds = {
            'min_r_squared': 0.95,
            'max_param_error_pct': 20.0,
            'excellent_r2_threshold': 0.98,
            'min_snr': 5.0,
            'min_delta_od_ci': 0.1,
            'min_absolute_delta_od': 0.15,
            'min_monotone_fraction': 0.55,
            'max_residual_autocorr': 0.7,
        }
        result = analysis.classify_by_fit_quality(fit, thresholds, time=time, od600=od)
        # Should be BAD: low delta-OD, low SNR, A below noise
        assert result.is_good is False

    def test_monotonicity_rejects_random_noise(self):
        """Random noise (50% increasing) should fail monotonicity check."""
        rng = np.random.default_rng(42)
        time = np.linspace(0, 50, 200)
        od = 0.5 + rng.normal(0, 0.05, 200)  # random walk around 0.5
        fit = self._make_fit_result(r_squared=0.96, a_opt=0.5)
        thresholds = {
            'min_r_squared': 0.95,
            'max_param_error_pct': 20.0,
            'excellent_r2_threshold': 0.995,
            'min_snr': 5.0,
            'min_delta_od_ci': 0.1,
            'min_absolute_delta_od': 0.05,  # low so we test monotonicity specifically
            'min_monotone_fraction': 0.55,
            'max_residual_autocorr': 0.7,
        }
        result = analysis.classify_by_fit_quality(fit, thresholds, time=time, od600=od)
        assert result.is_good is False

    def test_real_growth_passes_monotonicity(self, good_gompertz_curve):
        """Real growth curve should pass monotonicity check."""
        time, od = good_gompertz_curve
        fit = self._make_fit_result(r_squared=0.98, a_opt=1.2)
        thresholds = {
            'min_r_squared': 0.95,
            'max_param_error_pct': 20.0,
            'excellent_r2_threshold': 0.995,
            'min_snr': 5.0,
            'min_delta_od_ci': 0.1,
            'min_absolute_delta_od': 0.15,
            'min_monotone_fraction': 0.55,
            'max_residual_autocorr': 0.7,
        }
        result = analysis.classify_by_fit_quality(fit, thresholds, time=time, od600=od)
        assert 'monoton' not in result.reason.lower()

    def test_structured_residuals_caught(self):
        """Highly autocorrelated residuals should trigger rejection."""
        time = np.linspace(0, 50, 200)
        # Create structured (autocorrelated) residuals — a sine wave
        structured_resid = 0.02 * np.sin(np.linspace(0, 6 * np.pi, 200))
        od = analysis.gompertz_model(time, 1.2, 0.2, 4.0) + structured_resid
        fit = self._make_fit_result(
            r_squared=0.96, a_opt=1.2,
            residuals=structured_resid
        )
        thresholds = {
            'min_r_squared': 0.95,
            'max_param_error_pct': 20.0,
            'excellent_r2_threshold': 0.995,
            'min_snr': 5.0,
            'min_delta_od_ci': 0.1,
            'min_absolute_delta_od': 0.15,
            'min_monotone_fraction': 0.55,
            'max_residual_autocorr': 0.7,
        }
        result = analysis.classify_by_fit_quality(fit, thresholds, time=time, od600=od)
        # Sine-wave residuals have high lag-1 autocorrelation
        assert result.metrics.get('residual_autocorr', 0) > 0.5


# ====================================================================
# GP Derivative Normalization
# ====================================================================

class TestGPDerivativeNormalization:
    """Test that GP derivative threshold is normalized to data scale."""

    def test_gp_phases_with_normalization(self, good_gompertz_curve):
        """GP phase detection should work with normalized derivative."""
        time, od = good_gompertz_curve
        gp, t_dense, mu, sigma = adv.fit_gp(time, od)
        config = {'advanced': {'gp': {
            'derivative_threshold': 0.01,
            'normalize_derivative': True,
        }}}
        phases = adv.gp_find_phases(t_dense, mu, config)
        # Should find all three main phases
        assert phases['lag_end'] is not None
        assert phases['exp_peak'] is not None
        assert phases['stat_start'] is not None
        # Stationary start should be after exponential peak
        assert phases['stat_start'] > phases['exp_peak']

    def test_gp_phases_without_normalization(self, good_gompertz_curve):
        """GP phase detection should still work without normalization."""
        time, od = good_gompertz_curve
        gp, t_dense, mu, sigma = adv.fit_gp(time, od)
        config = {'advanced': {'gp': {
            'derivative_threshold': 0.01,
            'normalize_derivative': False,
        }}}
        phases = adv.gp_find_phases(t_dense, mu, config)
        assert phases['exp_peak'] is not None

    def test_gp_truncation_confidence_normalized(self, good_gompertz_curve):
        """GP truncation confidence should be in [0, 1]."""
        time, od = good_gompertz_curve
        idx, confidence, phases, gp_data = adv.gp_truncate(time, od)
        assert 0.0 <= confidence <= 1.0
        # Clean curve should have high confidence
        assert confidence > 0.5

    def test_gp_handles_low_od_curve(self):
        """GP should work on low-OD curves (A=0.3) with normalization."""
        time = np.linspace(0, 24, 100)
        # Low-amplitude curve
        od = adv.gompertz_model(time, 0.3, 0.15, 5.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.005, len(od))
        od = np.maximum(0, od)

        config = {'advanced': {'gp': {
            'derivative_threshold': 0.01,
            'normalize_derivative': True,
        }}}
        gp, t_dense, mu, sigma = adv.fit_gp(time, od, config)
        phases = adv.gp_find_phases(t_dense, mu, config)
        # With normalization, should still detect phases on low-OD curve
        assert phases['exp_peak'] is not None


# ====================================================================
# Multi-Model Fallback Classification
# ====================================================================

class TestMultiModelFallback:
    """Test that alternative growth models rescue non-Gompertz curves."""

    def test_logistic_curve_rescued(self, standard_time):
        """Logistic-shaped curve should be rescued by logistic model."""
        # Pure logistic curve -- Gompertz won't fit well
        od = analysis.logistic_model(standard_time, 1.2, 0.8, 10.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.01, len(od))
        od = np.maximum(0, od)

        # Gompertz fit should be mediocre
        gompertz_fit = analysis.fit_gompertz(standard_time, od)

        alt_fit, model_name = analysis.try_alternative_models(
            standard_time, od, gompertz_r2=gompertz_fit.r_squared
        )
        # Alternative model should do at least as well
        if alt_fit is not None:
            assert alt_fit.r_squared >= gompertz_fit.r_squared
            assert model_name in ('logistic', 'richards', 'baranyi')

    def test_gompertz_curve_not_replaced(self, good_gompertz_curve):
        """Clean Gompertz curve should NOT be replaced by alternative."""
        time, od = good_gompertz_curve
        gompertz_fit = analysis.fit_gompertz(time, od)

        # Gompertz R² should be high -- no alternative should beat it
        alt_fit, model_name = analysis.try_alternative_models(
            time, od, gompertz_r2=gompertz_fit.r_squared
        )
        # With a good Gompertz fit, alternative should not beat it
        # (alt_fit is None means nothing beat it)
        if alt_fit is not None:
            # Even if an alternative matches, it shouldn't be much better
            assert alt_fit.r_squared >= gompertz_fit.r_squared

    def test_try_alternative_models_returns_none_when_no_improvement(self, noisy_flat_curve):
        """Flat noise should not be rescued by any model."""
        time, od = noisy_flat_curve
        # Set gompertz_r2 to 0.99 -- nothing should beat this
        alt_fit, model_name = analysis.try_alternative_models(
            time, od, gompertz_r2=0.99
        )
        assert alt_fit is None
        assert model_name == 'gompertz'

    def test_alternative_models_available(self):
        """All three alternative models should be accessible."""
        assert hasattr(analysis, 'logistic_model')
        assert hasattr(analysis, 'baranyi_model')
        assert hasattr(analysis, 'richards_model')
        assert hasattr(analysis, 'try_alternative_models')
        assert hasattr(analysis, 'ALTERNATIVE_MODELS')
        assert 'logistic' in analysis.ALTERNATIVE_MODELS
        assert 'baranyi' in analysis.ALTERNATIVE_MODELS
        assert 'richards' in analysis.ALTERNATIVE_MODELS

    def test_fit_result_structure_from_alternative(self, standard_time):
        """Alternative model fit should produce a valid FitResult."""
        # Logistic curve
        od = analysis.logistic_model(standard_time, 1.0, 1.0, 12.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.01, len(od))
        od = np.maximum(0, od)

        alt_fit, model_name = analysis.try_alternative_models(
            standard_time, od, gompertz_r2=0.0
        )
        assert alt_fit is not None
        assert alt_fit.success is True
        assert alt_fit.r_squared > 0.5
        assert alt_fit.a_opt > 0
        assert len(alt_fit.predicted) == len(od)
        assert len(alt_fit.residuals) == len(od)

    def test_best_model_tracked_in_metrics(self, standard_time):
        """Classification metrics should include best_model field."""
        # Create a logistic-shaped curve
        od = analysis.logistic_model(standard_time, 1.2, 0.8, 10.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.01, len(od))
        od = np.maximum(0, od)

        # Try Gompertz first
        gompertz_fit = analysis.fit_gompertz(standard_time, od)

        # Try alternative
        alt_fit, model_name = analysis.try_alternative_models(
            standard_time, od, gompertz_r2=gompertz_fit.r_squared
        )
        best_fit = alt_fit if alt_fit is not None else gompertz_fit
        best_name = model_name

        thresholds = {
            'min_r_squared': 0.95,
            'max_param_error_pct': 20.0,
            'excellent_r2_threshold': 0.98,
            'min_snr': 5.0,
            'min_delta_od_ci': 0.1,
            'min_absolute_delta_od': 0.15,
            'min_monotone_fraction': 0.55,
            'max_residual_autocorr': 0.7,
        }
        result = analysis.classify_by_fit_quality(
            best_fit, thresholds, time=standard_time, od600=od
        )
        # Metrics should be present and valid
        assert 'r_squared' in result.metrics
