"""Tests for ML classifier module."""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from ml_classifier import (
    extract_prefit_features,
    extract_postfit_features,
    extract_metadata_features,
    compute_derived_features,
    PreFitGate,
    PostFitClassifier,
    PREFIT_FEATURES,
    METADATA_FEATURES,
    ALL_POSTFIT_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_curve():
    """Clean Gompertz sigmoid."""
    t = np.linspace(0, 24, 100)
    A, mu, lam = 1.2, 0.3, 3.0
    od = A * np.exp(-np.exp((mu * np.e / A) * (lam - t) + 1))
    od += np.random.default_rng(42).normal(0, 0.01, len(od))
    return t, od


@pytest.fixture
def flat_curve():
    """Flat noise — no growth."""
    t = np.linspace(0, 24, 100)
    od = 0.05 + np.random.default_rng(42).normal(0, 0.01, len(t))
    return t, od


@pytest.fixture
def mock_fit_result():
    """Minimal fit result dict simulating FitResult attributes."""
    class FitResult:
        success = True
        r_squared = 0.98
        rmse = 0.02
        mae = 0.015
        a_opt = 1.2
        mu_opt = 0.3
        lambda_opt = 3.0
        a_err = 0.01
        mu_err = 0.005
        lambda_err = 0.1
        residuals = np.random.default_rng(42).normal(0, 0.02, 50)
    return FitResult()


@pytest.fixture
def mock_metrics():
    """Classification metrics dict as produced by classify_by_fit_quality."""
    return {
        'r_squared': 0.98,
        'a_err_pct': 1.5,
        'mu_err_pct': 3.0,
        'snr': 15.0,
        'delta_od': 1.1,
        'max_od': 1.3,
        'baseline_std': 0.01,
        'monotone_fraction': 0.85,
        'residual_autocorr': 0.2,
        'delta_od_ci_lower': 1.08,
    }


# ---------------------------------------------------------------------------
# Pre-fit feature extraction
# ---------------------------------------------------------------------------

class TestPreFitFeatures:
    def test_all_features_present(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od)
        for f in PREFIT_FEATURES:
            assert f in features, f"Missing feature: {f}"

    def test_good_curve_has_high_snr(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od)
        assert features['raw_snr'] > 5.0

    def test_flat_curve_has_low_delta_od(self, flat_curve):
        t, od = flat_curve
        features = extract_prefit_features(t, od)
        assert features['raw_delta_od'] < 0.2

    def test_monotonicity_high_for_growth(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od)
        assert features['raw_monotone_fraction'] > 0.5

    def test_time_span_correct(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od)
        assert abs(features['time_span'] - 24.0) < 0.5

    def test_n_points_correct(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od)
        assert features['n_points'] == 100


# ---------------------------------------------------------------------------
# Post-fit feature extraction
# ---------------------------------------------------------------------------

class TestPostFitFeatures:
    def test_all_features_present(self, mock_fit_result, mock_metrics):
        features = extract_postfit_features(mock_fit_result, mock_metrics, 12.0, 50)
        # Metadata features (is_control, concentration_numeric) are added by
        # the classifier from strain_name, not by extract_postfit_features
        non_meta = [f for f in ALL_POSTFIT_FEATURES if f not in METADATA_FEATURES]
        for f in non_meta:
            assert f in features, f"Missing feature: {f}"

    def test_derived_features_computed(self, mock_fit_result, mock_metrics):
        features = extract_postfit_features(mock_fit_result, mock_metrics, 12.0, 50)
        assert not np.isnan(features['mu_over_a'])
        assert abs(features['mu_over_a'] - 0.3/1.2) < 0.01

    def test_handles_failed_fit(self, mock_metrics):
        class FailedFit:
            success = False
        features = extract_postfit_features(FailedFit(), mock_metrics, None, None)
        assert np.isnan(features['fit_r_squared'])
        assert np.isnan(features['gompertz_a'])

    def test_zero_division_produces_nan(self, mock_fit_result, mock_metrics):
        mock_metrics['delta_od'] = 0.0
        features = extract_postfit_features(mock_fit_result, mock_metrics, 0.0, 50)
        assert np.isnan(features['rmse_over_delta_od'])
        assert np.isnan(features['points_per_hour'])


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    def test_basic_ratios(self):
        features = {
            'gompertz_a': 1.0, 'gompertz_mu': 0.5, 'gompertz_lambda': 3.0,
            'truncation_time': 12.0, 'fit_rmse': 0.02, 'delta_od': 1.0,
            'a_err_pct': 2.0, 'mu_err_pct': 5.0, 'points_used': 60.0,
        }
        derived = compute_derived_features(features)
        assert abs(derived['mu_over_a'] - 0.5) < 0.001
        assert abs(derived['lambda_over_trunc_time'] - 0.25) < 0.001
        assert abs(derived['rmse_over_delta_od'] - 0.02) < 0.001
        assert abs(derived['err_product'] - 10.0) < 0.001
        assert abs(derived['points_per_hour'] - 5.0) < 0.001

    def test_nan_propagation(self):
        features = {
            'gompertz_a': float('nan'), 'gompertz_mu': 0.5,
            'gompertz_lambda': 3.0, 'truncation_time': 12.0,
            'fit_rmse': 0.02, 'delta_od': 1.0,
            'a_err_pct': float('nan'), 'mu_err_pct': 5.0,
            'points_used': 60.0,
        }
        derived = compute_derived_features(features)
        assert np.isnan(derived['mu_over_a'])
        assert np.isnan(derived['err_product'])


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------

class TestMetadataFeatures:
    def test_water_control_detected(self):
        meta = extract_metadata_features('H2O-MAL10')
        assert meta['is_control'] == 1.0
        assert meta['concentration_numeric'] == 0.0  # controls have 0 pesticide

    def test_lb_control_detected(self):
        meta = extract_metadata_features('LB-CISPERM1')
        assert meta['is_control'] == 1.0
        assert meta['concentration_numeric'] == 0.0  # controls have 0 pesticide

    def test_treatment_has_real_concentration(self):
        meta = extract_metadata_features('Bifenthrin-BIF3')
        assert meta['is_control'] == 0.0
        assert meta['concentration_numeric'] == 50.0  # bifenthrin = 50 mg/L

    def test_malathion_concentration(self):
        meta = extract_metadata_features('MALATHIONANDLB-MAL10')
        assert meta['is_control'] == 0.0
        assert meta['concentration_numeric'] == 50.0  # malathion = 50 mg/L

    def test_imidacloprid_concentration(self):
        meta = extract_metadata_features('IMIDACLOPRID-IMID2')
        assert meta['is_control'] == 0.0
        assert meta['concentration_numeric'] == 20.0  # imidacloprid = 20 mg/L

    def test_permethrin_concentration(self):
        meta = extract_metadata_features('PERMETHRINANDLB-PERM4')
        assert meta['is_control'] == 0.0
        assert meta['concentration_numeric'] == 100.0  # permethrin = 100 mg/L

    def test_synthetic_name_not_control(self):
        meta = extract_metadata_features('CURVE0005')
        assert meta['is_control'] == 0.0

    def test_none_returns_nan(self):
        meta = extract_metadata_features(None)
        assert np.isnan(meta['is_control'])
        assert np.isnan(meta['concentration_numeric'])

    def test_prefit_includes_metadata(self, good_curve):
        t, od = good_curve
        features = extract_prefit_features(t, od, strain_name='H2O-TEST5')
        assert features['is_control'] == 1.0
        assert features['concentration_numeric'] == 0.0  # H2O control = 0 pesticide


# ---------------------------------------------------------------------------
# PreFitGate (without model)
# ---------------------------------------------------------------------------

class TestPreFitGateNoModel:
    def test_no_model_passes_everything(self, good_curve):
        gate = PreFitGate(model_path='/nonexistent/path.joblib')
        t, od = good_curve
        assert gate.should_skip(t, od) is False

    def test_no_model_passes_flat(self, flat_curve):
        gate = PreFitGate(model_path='/nonexistent/path.joblib')
        t, od = flat_curve
        assert gate.should_skip(t, od) is False


# ---------------------------------------------------------------------------
# PostFitClassifier (without model)
# ---------------------------------------------------------------------------

class TestPostFitClassifierNoModel:
    def test_no_model_returns_none(self, mock_fit_result, mock_metrics):
        clf = PostFitClassifier(model_path='/nonexistent/path.joblib')
        result = clf.classify(mock_fit_result, mock_metrics, 12.0, 50)
        assert result is None

    def test_threshold_attributes(self):
        clf = PostFitClassifier(
            model_path='/nonexistent/path.joblib',
            good_threshold=0.8,
            bad_threshold=0.2,
        )
        assert clf.good_threshold == 0.8
        assert clf.bad_threshold == 0.2
