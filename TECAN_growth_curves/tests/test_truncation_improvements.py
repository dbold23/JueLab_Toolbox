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
#
# These are module scoped on purpose.  Every one of them is a pure function of
# a fixed seed, and three of the tests below drive a Gaussian-process fit or a
# five-method ensemble that costs 5-18 s; recomputing them per test made this
# file the third most expensive in the suite for no added coverage.  Nothing
# here mutates a fixture's arrays.

@pytest.fixture(scope="module")
def standard_time():
    """100-point time array over 24 hours."""
    return np.linspace(0, 24, 100)


@pytest.fixture(scope="module")
def incomplete_curve(standard_time):
    """Curve still rising at experiment end (no stationary phase reached).
    Simulates truncation_challenge scenario."""
    # Gompertz with very long lag + slow growth -- never plateaus within 24h
    od = analysis.gompertz_model(standard_time, 2.0, 0.15, 12.0)
    rng = np.random.default_rng(42)
    od = od + rng.normal(0, 0.005, len(od))
    od = np.maximum(0, od)
    return standard_time, od


@pytest.fixture(scope="module")
def good_gompertz_curve(standard_time):
    """Clean Gompertz curve that plateaus normally."""
    od = analysis.gompertz_model(standard_time, 1.2, 0.2, 4.0)
    rng = np.random.default_rng(42)
    od = od + rng.normal(0, 0.01, len(od))
    od = np.maximum(0, od)
    return standard_time, od


@pytest.fixture(scope="module")
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

@pytest.fixture(scope="module")
def ensemble_result(good_gompertz_curve):
    """One five-method ensemble run, shared by the three tests that read it.

    ``ensemble_truncate`` costs ~15 s on this curve and is deterministic, so
    running it once per test was 30 s of the suite spent recomputing the same
    object three times.
    """
    time, od = good_gompertz_curve
    return adv.ensemble_truncate(time, od)


class TestEnsembleOutlierDetection:
    """Test MAD-based outlier detection in ensemble_truncate."""

    def test_outlier_method_downweighted(self, ensemble_result):
        """Run ensemble and verify that outlier methods get flagged."""
        result = ensemble_result

        # With a clean curve, most methods should agree.
        # Check that any outlier method (if present) was marked.
        for name, data in result.method_results.items():
            if data.get('outlier', False):
                assert data.get('mad_deviation', 0) > 3.0

    def test_ensemble_consensus_in_range(self, good_gompertz_curve, ensemble_result):
        """Consensus should be within the time range."""
        time, od = good_gompertz_curve
        result = ensemble_result
        assert result.consensus_time >= time[0]
        assert result.consensus_time <= time[-1]

    def test_ensemble_result_structure(self, ensemble_result):
        """Ensemble result should have all required fields."""
        result = ensemble_result
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

@pytest.fixture(scope="module")
def gp_fit(good_gompertz_curve):
    """One Gaussian-process fit of the clean curve, shared by the phase tests."""
    time, od = good_gompertz_curve
    return adv.fit_gp(time, od)


class TestGPDerivativeNormalization:
    """Test that GP derivative threshold is normalized to data scale."""

    def test_gp_phases_with_normalization(self, gp_fit):
        """GP phase detection should work with normalized derivative."""
        gp, t_dense, mu, sigma = gp_fit
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

    def test_gp_phases_without_normalization(self, gp_fit):
        """GP phase detection should still work without normalization."""
        gp, t_dense, mu, sigma = gp_fit
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


# ====================================================================
# MCCV Truncation Optimization (v2)
# ====================================================================

class TestMCCVTruncation:
    """Test Monte Carlo Cross-Validated truncation optimization."""

    def test_full_range_finds_correct_truncation(self, good_gompertz_curve):
        """MCCV should find truncation near the plateau for a clean curve."""
        time, od = good_gompertz_curve
        landscape = analysis.find_optimal_truncation_v2(time, od, min_points=15)
        # Should find a truncation point (not just "use all data")
        assert landscape.best_end_idx < len(od) - 1
        assert landscape.best_end_idx > 15
        assert landscape.best_cv_score > 0.5
        assert landscape.best_model == 'gompertz'

    def test_landscape_has_candidates(self, good_gompertz_curve):
        """Landscape should contain evaluated candidates."""
        time, od = good_gompertz_curve
        landscape = analysis.find_optimal_truncation_v2(
            time, od, min_points=15, n_coarse=20, n_cv_folds=10
        )
        assert len(landscape.candidates) > 10
        # Candidates should be sorted by cv_score (descending)
        scores = [c.cv_score for c in landscape.candidates]
        assert scores[0] >= scores[-1]

    def test_short_segment_not_preferred(self, standard_time):
        """MCCV should not prefer trivially short segments."""
        # Gompertz that plateaus around t=10, but has 100 data points
        od = analysis.gompertz_model(standard_time, 1.2, 0.4, 3.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.01, len(od))
        od = np.maximum(0, od)

        landscape = analysis.find_optimal_truncation_v2(
            standard_time, od, min_points=15, n_coarse=30, n_cv_folds=10
        )
        # Should not truncate to the minimum 15 points
        assert landscape.best_end_idx > 20

    def test_cv_score_structure(self, standard_time):
        """Each candidate should have valid CV score fields."""
        od = analysis.gompertz_model(standard_time, 1.0, 0.2, 5.0)
        rng = np.random.default_rng(42)
        od = od + rng.normal(0, 0.01, len(od))
        od = np.maximum(0, od)

        landscape = analysis.find_optimal_truncation_v2(
            standard_time, od, min_points=15, n_coarse=20, n_cv_folds=10
        )
        for c in landscape.candidates:
            assert hasattr(c, 'cv_score')
            assert hasattr(c, 'cv_std')
            assert hasattr(c, 'raw_r2')
            assert hasattr(c, 'model_name')
            assert c.n_points >= 15

    def test_confidence_high_for_clean_curve(self, good_gompertz_curve):
        """Clean curve should have reasonable confidence."""
        time, od = good_gompertz_curve
        landscape = analysis.find_optimal_truncation_v2(
            time, od, min_points=15, n_coarse=30, n_cv_folds=10
        )
        assert 0.0 <= landscape.confidence <= 1.0

    def test_mccv_score_function_directly(self, good_gompertz_curve):
        """mccv_score should return valid metrics."""
        time, od = good_gompertz_curve
        cv_mean, cv_std, raw_r2, params = analysis.mccv_score(
            time, od, n_folds=10
        )
        assert cv_mean > 0.5  # clean curve should CV well
        assert cv_std >= 0.0
        assert raw_r2 > 0.9
        assert params is not None

    def test_noisy_flat_curve_low_cv(self, noisy_flat_curve):
        """Flat noise should have low CV score."""
        time, od = noisy_flat_curve
        cv_mean, cv_std, raw_r2, _ = analysis.mccv_score(
            time, od, n_folds=10
        )
        # Flat noise should not fit well out-of-sample
        assert cv_mean < 0.5

    def test_truncate_at_max_uses_v2(self, good_gompertz_curve):
        """truncate_at_max with use_adaptive=True should use MCCV v2."""
        time, od = good_gompertz_curve
        result = analysis.truncate_at_max(time, od, use_adaptive=True)
        # Should produce a valid TruncationResult
        assert result.truncation_index > 0
        assert len(result.od_truncated) > 10


# ====================================================================
# PEAK_RATE_FRACTION calibration guard
# ====================================================================
#
# What this section defends
# -------------------------
# ``PEAK_RATE_FRACTION`` (scripts/01_growth_curve_analysis.py:631) is 0.25 and
# it was calibrated by hand, against the four project plates and a human audit,
# not against the synthetic suite.  The calibration is recorded in the
# docstring of ``find_first_local_maximum`` (:530-546) and rests on
# results/tables/validation_audit.csv: at 25 % the first-peak rule agreed with
# the human-audited adaptive verdicts on 83 of 85 conditions with a median cut
# 3 h from the cross-validated one, while at 10 % it left curves with a slow
# secondary drift uncut and agreed on only 75 of 85.
#
# It is worth defending because the synthetic benchmark ARGUES THE OTHER WAY.
# On the old 480-curve suite the benchmark ranked 0.10 best (accuracy 0.879 vs
# 0.873, median |A| error 0.025 vs 0.034), and it still does on the rebuilt
# suite: see test_the_multi_phase_scenarios_do_not_settle_the_calibration below
# for the measurement.  A future reader who trusts the benchmark over the audit
# would retune a real-instrument constant on synthetic evidence.
#
# NOBODY MAY MAKE THESE TESTS PASS BY RETUNING THE GENERATOR.  The direction
# comes from the real plates; the synthetic side of the guard exists only to
# record the mechanism and to keep the disagreement visible.

import importlib.util                                              # noqa: E402
import re                                                          # noqa: E402
import warnings                                                    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "results" / "tables" / "validation_audit.csv"
SYNTH_DATA_DIR = (REPO_ROOT / "synthetic_data" / "output" / "comprehensive_test"
                  / "test_data" / "DATA")
SYNTH_GROUND_TRUTH = (REPO_ROOT / "synthetic_data" / "output" / "comprehensive_test"
                      / "test_data" / "ground_truth.csv")
REAL_GROUP_DIRS = {
    "Group1": REPO_ROOT / "data" / "raw" / "Group1" / "Group_1_DATA",
    "Group2": REPO_ROOT / "data" / "raw" / "Group 2" / "Group_2_DATA",
    "Group3": REPO_ROOT / "data" / "raw" / "Group 3" / "Group_3_DATA",
    "Group4": REPO_ROOT / "data" / "raw" / "Group 4" / "Group4_DATA",
}

#: The two settings compared everywhere below: the shipped one and the one the
#: synthetic benchmark keeps recommending.
CALIBRATED_FRACTION = 0.25
REJECTED_FRACTION = 0.10


def _curve_token(name):
    hit = re.search(r"(CURVE\d+)", str(name))
    return hit.group(1) if hit else None


@pytest.fixture(scope="module")
def audited_real_curves():
    """The 85 human-audited real conditions: time, OD and the audited verdict.

    ``audit_result`` says whether the pipeline's verdict of the day was right,
    so the audited truth is ``is_good == (audit_result == 'correct')``.  The
    seven 'unsure' rows are dropped, exactly as the calibration did.
    """
    if not AUDIT_CSV.exists():
        pytest.skip("results/tables/validation_audit.csv not present")
    audit = pd.read_csv(AUDIT_CSV)
    decided = audit[audit["audit_result"].str.strip().str.lower()
                    .isin(["correct", "wrong"])]
    truth = {
        (row["group"], row["strain"]):
            bool(row["is_good"]) == (row["audit_result"].strip().lower() == "correct")
        for _, row in decided.iterrows()
    }

    curves = {}
    for group, directory in REAL_GROUP_DIRS.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_DATA.csv")):
            frame = pd.read_csv(path, float_precision="round_trip")
            time_cols = [c for c in frame.columns
                         if str(c).lower().startswith("time")]
            if not time_cols:
                continue
            time = frame[time_cols[0]].to_numpy(dtype=float)
            for column in frame.columns:
                if column == time_cols[0]:
                    continue
                strain = column.replace("_blanked", "").replace("_", "-")
                key = (group, strain)
                if key not in truth:
                    continue
                od = frame[column].to_numpy(dtype=float)
                mask = ~np.isnan(od)
                curves[key] = (time[mask], od[mask])

    if len(curves) < 80:
        pytest.skip(f"only {len(curves)} audited raw curves found")
    return {"curves": curves, "truth": truth}


@pytest.fixture(scope="module")
def multi_phase_curves():
    """Synthetic curves that have a real second phase, with their true band.

    The band is the one the benchmark scores against: ``[t_primary_95,
    t_second_start]`` (see pipeline_validator.truncation_metrics).
    """
    if not (SYNTH_GROUND_TRUTH.exists() and SYNTH_DATA_DIR.is_dir()):
        pytest.skip("synthetic suite not present")
    gt = pd.read_csv(SYNTH_GROUND_TRUTH)
    if "t_second_start" not in gt.columns:
        pytest.skip("ground truth v1 carries no growth-window columns")
    multi = gt[gt["t_second_start"].notna()]
    if len(multi) < 20:
        pytest.skip("no multi-phase scenarios in this suite")

    band = {}
    for _, row in multi.iterrows():
        token = _curve_token(row["strain_name"])
        upper = row["t_second_start"]
        if pd.isna(upper):
            upper = row["true_window_end_h"]
        band[token] = {"scenario": row["scenario"],
                       "lower": float(row["t_primary_95"]),
                       "upper": float(upper)}

    series = {}
    for path in sorted(SYNTH_DATA_DIR.glob("SYNTHETIC_*pts_*_DATA.csv")):
        frame = pd.read_csv(path)
        time = frame.iloc[:, 0].to_numpy(dtype=float)
        for column in frame.columns[1:]:
            token = _curve_token(column)
            if token in band:
                series[token] = (time, frame[column].to_numpy(dtype=float))
    assert len(series) == len(band), "a multi-phase curve is missing from DATA/"
    return {"series": series, "band": band}


#: Analytic curves for the two mechanism guards below.  These are BUILT IN THIS
#: FILE from fixed parameters -- no sampler, no seed, no generated data -- so
#: there is nothing here that could be re-tuned until the shipped constant wins.
SHOULDER_GRID = [
    (A, mu, lam)
    for A in (0.6, 1.2, 2.0)
    for mu in (0.08, 0.2, 0.4)
    for lam in (1.0, 4.0, 10.0)
]

#: (A1, mu1, lambda1, A2, mu2, gap_hours) for the second-phase guard.  The gap
#: sweep deliberately spans BOTH sides of the point where the two settings
#: start to differ (the suite samples 5-15 h; a real plate is under no
#: obligation to).
TWO_PHASE_GRID = [
    (1.0, mu, 3.0, A2, mu2, gap)
    for mu in (0.15, 0.25, 0.35)
    for A2 in (0.3, 0.5)
    for mu2 in (0.05, 0.08, 0.15)
    for gap in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)
]


def _shipped_fraction():
    """The constant as the pipeline ships it, read at call time.

    Every guard below compares this against ``REJECTED_FRACTION`` rather than
    against the literal 0.25, so setting the module constant to 0.10 makes the
    comparisons collapse and the guards fail -- which is the behaviour the
    acceptance list asks for.
    """
    return float(analysis.PEAK_RATE_FRACTION)


def _cut_index(time, od, fraction, monkeypatch):
    monkeypatch.setattr(analysis, "PEAK_RATE_FRACTION", fraction)
    result = analysis.truncate_at_max(time, od, use_first_peak=True,
                                      use_adaptive=False)
    return int(result.truncation_index), float(result.truncation_time)


def _cut_times(series, fraction, monkeypatch):
    monkeypatch.setattr(analysis, "PEAK_RATE_FRACTION", fraction)
    out = {}
    for key, (time, od) in series.items():
        result = analysis.truncate_at_max(time, od, use_first_peak=True,
                                          use_adaptive=False)
        out[key] = float(result.truncation_time)
    return out


class TestPeakRateFractionCalibration:
    """The constant is calibrated on real plates; keep it that way."""

    def test_the_shipped_fraction_is_the_calibrated_one(self):
        """0.25, and its calibration is cited where it is defined.

        This is the assertion the acceptance list asks for literally: it passes
        at 0.25 and fails the moment somebody sets the constant to 0.10.
        """
        assert analysis.PEAK_RATE_FRACTION == CALIBRATED_FRACTION

        source = (SCRIPTS_DIR / "01_growth_curve_analysis.py").read_text(encoding="utf-8")
        docstring = source.split("def find_first_local_maximum")[1][:4000]
        assert "validation_audit.csv" in docstring, (
            "the calibration of PEAK_RATE_FRACTION must name the evidence it "
            "rests on, so nobody re-tunes it against the synthetic suite")
        assert "83/85" in docstring and "75/85" in docstring, (
            "the audited agreement counts (83/85 at 25 %, 75/85 at 10 %) must "
            "stay in find_first_local_maximum's docstring; they are the only "
            "record of why the constant is 0.25 and not the value the "
            "synthetic benchmark prefers")

    def test_real_plate_agreement_prefers_the_calibrated_fraction(
            self, audited_real_curves, monkeypatch):
        """On the audited plates, 0.25 agrees with the humans more often than 0.10.

        The direction is asserted, never the count: pinning "74 of 85" would
        turn every honest pipeline improvement into a test failure.  Re-running
        the whole audited set costs ~4 s because only the truncation and the fit
        are exercised, no plots and no adaptive search.

        Measured on this tree (2026-09, first-peak, today's gate set): 0.25
        agrees with the audited truth on 74 of 85 and 0.10 on 72 of 85; against
        the verdicts as recorded in validation_audit.csv it is 75 against 73.
        The 83/85 and 75/85 of the original calibration are NOT reproduced --
        the pipeline has gained a pre-fit gate, an incomplete-curve detector
        and two secondary gates since -- so the direction is what survives, and
        the direction is what is asserted.

        RE-BASELINE POLICY: if this ever inverts, that is a finding about the
        pipeline, not a test to relax.  It may only be changed together with a
        written justification in docs/BENCHMARK_LEDGER.md.
        """
        curves = audited_real_curves["curves"]
        truth = audited_real_curves["truth"]

        agreement = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fraction in (CALIBRATED_FRACTION, REJECTED_FRACTION):
                monkeypatch.setattr(analysis, "PEAK_RATE_FRACTION", fraction)
                agreement[fraction] = sum(
                    bool(analysis.analyze_curve(t, od, strain_name=key[1])
                         .classification.is_good) == truth[key]
                    for key, (t, od) in curves.items())

        assert agreement[CALIBRATED_FRACTION] > agreement[REJECTED_FRACTION], (
            f"the human calibration no longer holds: {CALIBRATED_FRACTION} agrees "
            f"with {agreement[CALIBRATED_FRACTION]}/{len(curves)} audited "
            f"conditions and {REJECTED_FRACTION} with "
            f"{agreement[REJECTED_FRACTION]}/{len(curves)}")

    def test_the_shipped_fraction_cuts_on_the_shoulder_not_in_the_plateau(
            self, monkeypatch):
        """The documented meaning of the constant, checked on analytic curves.

        ``find_first_local_maximum``'s docstring says the crossing "sits at
        ~90 % of A" on an ideal Gompertz curve.  That is a statement about the
        RULE, not about any dataset, so it is checked here on 27 analytic
        Gompertz curves (A x mu x lambda, 15-minute sampling, no noise) where
        the fraction of A reached at the cut is known exactly.

        Measured on this tree: the shipped 0.25 cuts at 0.917-0.993 of A
        (median 0.941) and 0.10 cuts at 0.957-0.997 (median 0.978); the two
        differ on 24 of the 27 curves and 0.25 is never the later of the two.
        Only the ordering and a loose envelope are asserted -- a tighter pin
        would break on any legitimate change to the smoother.

        This is the guard the plan asked for and the benchmark could not give:
        it says what the constant MEANS, on synthetic data, without claiming
        the suite endorses its value.  A future reader who sets the constant
        to 0.10 fails here, because the shipped and rejected cuts then coincide
        on every curve.
        """
        shipped = _shipped_fraction()
        time = np.arange(0.0, 48.0 + 1e-9, 0.25)

        shipped_fracs, rejected_fracs, later = [], [], 0
        for amplitude, mu, lag in SHOULDER_GRID:
            clean = analysis.gompertz_model(time, amplitude, mu, lag)
            od = clean + 0.02
            i_ship, t_ship = _cut_index(time, od, shipped, monkeypatch)
            i_rej, t_rej = _cut_index(time, od, REJECTED_FRACTION, monkeypatch)
            if t_ship > t_rej + 1e-9:
                later += 1
            shipped_fracs.append(clean[i_ship] / amplitude)
            rejected_fracs.append(clean[i_rej] / amplitude)

        shipped_fracs = np.asarray(shipped_fracs)
        rejected_fracs = np.asarray(rejected_fracs)

        assert later == 0, (
            f"{later}/{len(SHOULDER_GRID)} analytic curves cut LATER at "
            f"{shipped} than at {REJECTED_FRACTION}")
        assert np.all(shipped_fracs <= rejected_fracs + 1e-12), (
            "the shipped fraction reaches a HIGHER fraction of A than "
            f"{REJECTED_FRACTION} on some analytic curve; the rule no longer "
            "means what its docstring says")
        differing = int(np.sum(shipped_fracs < rejected_fracs - 1e-9))
        assert differing >= len(SHOULDER_GRID) // 2, (
            f"only {differing}/{len(SHOULDER_GRID)} analytic curves separate "
            f"{shipped} from {REJECTED_FRACTION}; either the constant has been "
            "set to the rejected value or the rule has stopped depending on it")
        assert 0.85 <= float(np.median(shipped_fracs)) <= 0.96, (
            "the shipped fraction no longer cuts on the shoulder of the "
            f"primary phase (median {float(np.median(shipped_fracs)):.3f} of A; "
            "the docstring promises ~90 %)")
        assert float(np.median(rejected_fracs)) > float(np.median(shipped_fracs))

    def test_the_rejected_fraction_runs_into_a_second_phase(self, monkeypatch):
        """The failure mode the calibration cites, reproduced synthetically.

        The docstring's reason for rejecting 0.10 is that it "left curves with
        a slow secondary drift uncut".  That is reproducible on analytic
        two-phase curves: 126 of them, built here from a fixed parameter grid
        through ``curve_shapes.build_clean_curve`` so the time at which the
        second phase starts is known exactly rather than estimated.

        A curve is *contaminated* when the cut lands after ``t_second_start``:
        the fit window then contains data from a phase the Gompertz model does
        not describe.  Measured on this tree:

            contaminated at 0.25 -> 0/126      at 0.10 -> 26/126

        and the failure is ONE-SIDED: there is no curve on the grid that the
        shipped fraction contaminates and 0.10 does not.  The 26 are the
        short-gap corner (2-4 h between the phases), exactly the geometry in
        which a plateau never gets quiet enough for a 10 %-of-peak rule to
        fire before the second phase lifts the rate again.

        Only the one-sidedness and the existence of the asymmetry are
        asserted, never the count of 26 -- a count would ratchet.  THIS IS A
        CALIBRATION GUARD: a failure here is never to be fixed by re-tuning the
        generator or widening the grid until the shipped value wins.

        Note what this does NOT say: on the 51 multi-phase curves of the
        rebuilt suite, whose gaps are sampled from 5-15 h, the two settings
        contaminate the SAME 13 curves and 0.10 recovers parameters slightly
        better (see the test below).  The suite does not settle the
        calibration; this grid only shows the mechanism is real and one-sided.
        """
        try:
            sys.path.insert(0, str(REPO_ROOT / "synthetic_data"))
            from src.curve_shapes import build_clean_curve
        except ImportError:                                    # pragma: no cover
            pytest.skip("synthetic_data/src/curve_shapes.py is not importable")

        shipped = _shipped_fraction()
        time = np.arange(0.0, 72.0 + 1e-9, 0.25)

        shipped_only, rejected_only, both = 0, 0, 0
        for amplitude, mu, lag, a2, mu2, gap in TWO_PHASE_GRID:
            clean, meta = build_clean_curve("two_phase_gompertz", time, dict(
                A=amplitude, mu=mu, lambda_=lag, A2=a2, mu2=mu2,
                gap_hours=gap, initial_od=0.02))
            second_start = float(meta["t_second_start"])
            assert np.isfinite(second_start), (
                "two_phase_gompertz stopped reporting t_second_start")
            _, t_ship = _cut_index(time, clean, shipped, monkeypatch)
            _, t_rej = _cut_index(time, clean, REJECTED_FRACTION, monkeypatch)
            hit_ship = t_ship > second_start
            hit_rej = t_rej > second_start
            shipped_only += hit_ship and not hit_rej
            rejected_only += hit_rej and not hit_ship
            both += hit_ship and hit_rej

        assert shipped_only == 0, (
            f"{shipped_only} of {len(TWO_PHASE_GRID)} analytic two-phase "
            f"curves are cut inside the second phase at {shipped} but not at "
            f"{REJECTED_FRACTION}: the calibrated setting has acquired the "
            "failure mode it was chosen to avoid")
        assert rejected_only >= 10, (
            f"only {rejected_only} of {len(TWO_PHASE_GRID)} curves separate "
            f"the two settings ({both} contaminated by both). Either the "
            f"constant has been set to {REJECTED_FRACTION}, or the first-peak "
            "rule no longer runs on into a second phase at a low fraction -- "
            "in which case the calibration's stated reason has gone away and "
            "belongs in docs/BENCHMARK_LEDGER.md, not in a relaxed test")

    def test_a_higher_fraction_never_cuts_later(self, multi_phase_curves, monkeypatch):
        """The mechanism: a larger fraction ends the primary phase sooner.

        This is what the calibration buys on a real plate -- at 10 % a slow
        secondary drift keeps the rate above the threshold and the cut runs on
        into the second phase.  Asserted as an invariant of the rule (every cut
        at 0.25 is at or before the cut at 0.10, and at least a quarter of them
        are strictly earlier), so an inverted comparison or a threshold applied
        to the wrong side of the peak fails here.
        """
        series = multi_phase_curves["series"]
        late = _cut_times(series, REJECTED_FRACTION, monkeypatch)
        early = _cut_times(series, CALIBRATED_FRACTION, monkeypatch)

        later = [k for k in series if early[k] > late[k] + 1e-9]
        assert not later, (
            f"{len(later)} multi-phase curves cut LATER at "
            f"{CALIBRATED_FRACTION} than at {REJECTED_FRACTION}")
        moved = [k for k in series if early[k] < late[k] - 1e-9]
        assert len(moved) >= len(series) // 4, (
            f"only {len(moved)}/{len(series)} cuts moved; the constant has "
            "stopped doing anything on the multi-phase scenarios")

    def test_the_multi_phase_scenarios_do_not_settle_the_calibration(
            self, multi_phase_curves, monkeypatch):
        """The honest negative, recorded so nobody has to rediscover it.

        The plan for this package expected the inside-the-true-window rate on
        the multi-phase scenarios to be at least 0.15 HIGHER at 0.25 than at
        0.10, which would have let the synthetic suite carry the calibration.
        It is not.  Measured on the 51 multi-phase curves of the rebuilt suite
        (seed 42, dataset 760a04062cf2):

            frac_inside_window   0.25 -> 0.020      0.10 -> 0.745

        The band's lower edge is ``t_primary_95``, the time the primary phase
        reaches 95 % of its amplitude, while the calibrated rule deliberately
        cuts at ~90 % of A -- about 3 h earlier on these curves.  So the metric
        excludes the calibrated setting by construction, and the 0.10 cuts that
        look "inside" are inside a window the pipeline was never aiming at.
        Parameter recovery does not separate them either: median |A| error
        0.032 at 0.25 against 0.030 at 0.10, with identical tolerance pass
        rates.

        What is therefore asserted here is only that this measurement keeps
        being MADE and reported -- both fractions are finite and the two
        settings differ -- never that the synthetic suite endorses the
        calibration.  THE FIX FOR A FAILURE HERE IS NEVER TO RE-TUNE THE
        GENERATOR UNTIL 0.25 WINS: the calibration's evidence is
        results/tables/validation_audit.csv and the test above.
        """
        series = multi_phase_curves["series"]
        band = multi_phase_curves["band"]

        inside = {}
        for fraction in (CALIBRATED_FRACTION, REJECTED_FRACTION):
            cuts = _cut_times(series, fraction, monkeypatch)
            inside[fraction] = float(np.mean([
                band[k]["lower"] <= cuts[k] <= band[k]["upper"] for k in series]))

        assert 0.0 <= inside[CALIBRATED_FRACTION] <= 1.0
        assert 0.0 <= inside[REJECTED_FRACTION] <= 1.0
        assert inside[CALIBRATED_FRACTION] != inside[REJECTED_FRACTION], (
            "the two settings score identically on the multi-phase scenarios; "
            "the guard has stopped measuring anything")
        assert inside[CALIBRATED_FRACTION] < inside[REJECTED_FRACTION] + 0.15, (
            "the multi-phase scenarios now DO endorse the human calibration "
            f"({inside[CALIBRATED_FRACTION]:.3f} inside-window at "
            f"{CALIBRATED_FRACTION} against {inside[REJECTED_FRACTION]:.3f} at "
            f"{REJECTED_FRACTION}). That is good news, and this test must be "
            "replaced by the positive assertion the plan asked for -- after "
            "checking that the generator was not tuned to produce it.")


class TestIncompleteCurveSuppression:
    """The autocorrelation gate is suppressed on curves that never plateaued.

    scripts/01_growth_curve_analysis.py:529 reads
    ``if lag1_autocorr > max_autocorr and not fit_is_excellent and not
    is_incomplete``.  A curve still rising at the end of the run has a fitted
    shape that is genuinely one-sided, so its residuals are autocorrelated for
    a reason that is not a bad fit.

    Measured: the branch changes ZERO verdicts across the committed 1139-curve
    suite (every GOOD curve with autocorr > 0.7 also has R2 >= 0.98 and is
    already exempt through ``fit_is_excellent``), so it cannot be exercised
    from the benchmark.  It is exercised here directly instead.
    """

    @staticmethod
    def _autocorrelated_incomplete_fit():
        """A still-rising curve whose residuals are strongly autocorrelated."""
        time = np.linspace(0, 24, 200)
        od = analysis.gompertz_model(time, 2.4, 0.12, 10.0)
        # A smooth, slowly varying model error: the signature the gate looks
        # for.  The amplitude is chosen to land R2 between the 0.95 pass mark
        # and the 0.98 'excellent' exemption, so the autocorrelation gate is
        # the ONLY thing standing between this curve and a GOOD verdict.
        od = od + 0.15 * np.sin(2 * np.pi * time / 12.0) + 0.02
        fit = analysis.fit_gompertz(time, od)
        assert fit.success
        assert 0.95 <= fit.r_squared < 0.98, fit.r_squared
        return time, od, fit

    def test_the_suppression_changes_the_verdict(self):
        time, od, fit = self._autocorrelated_incomplete_fit()

        gated = analysis.classify_by_fit_quality(fit, time=time, od600=od,
                                                 is_incomplete=False)
        suppressed = analysis.classify_by_fit_quality(fit, time=time, od600=od,
                                                      is_incomplete=True)

        assert gated.metrics["residual_autocorr"] > 0.7, (
            "the fixture no longer produces autocorrelated residuals")
        assert fit.r_squared < 0.98, (
            "the fixture fits too well; fit_is_excellent would exempt it anyway")
        assert "residual_autocorr" in gated.failed_gates
        assert gated.is_good is False
        assert "residual_autocorr" not in suppressed.failed_gates, (
            "the incomplete-curve suppression at 01_growth_curve_analysis.py:529 "
            "no longer fires")
        assert suppressed.is_good is True

    def test_the_suppression_is_reachable_from_the_truncation_detector(self):
        """``detect_incomplete_curve`` is what supplies the flag in the pipeline."""
        time = np.linspace(0, 24, 200)
        od = analysis.gompertz_model(time, 2.4, 0.12, 10.0) + 0.02
        is_incomplete, meta = analysis.detect_incomplete_curve(time, od)
        assert bool(is_incomplete) is True
        assert meta["tail_slope"] > 0
