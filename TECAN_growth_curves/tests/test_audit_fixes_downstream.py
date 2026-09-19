"""
Regression tests for the downstream audit findings (scripts 05/06/07/08).

downstream-01  every strain is loaded from its OWN column, never from a
               substring "partial match" in whichever file glob returned first
trunc-05       the adaptive_r2 comparison uses the window 01 --adaptive fits,
               start index included
haldane-01     S0 is fixed, Ki is fitted rather than echoed back from the
               initial guess, an unidentifiable Ki is reported as NaN, and a
               failed ODE solve is reported as a failed fit
haldane-02     the Gompertz model is refitted on the window the Haldane fit is
               scored on before the AICc comparison
compare-01     a bad strain counts as rescued only when the re-truncated fit
               passes every gate 01 applies
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tecan_growth.legacy_data import find_curve, find_curve_ref  # noqa: E402

GROUP_DIRS = {
    'Group1': 'Group1/Group_1_DATA',
    'Group2': 'Group 2/Group_2_DATA',
    'Group3': 'Group 3/Group_3_DATA',
    'Group4': 'Group 4/Group4_DATA',
}
DATA_RAW = REPO_ROOT / "data" / "raw"
RESULTS_CSV = REPO_ROOT / "results" / "tables" / "all_groups_results.csv"


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h05():
    return _load_script("audit_haldane_analysis", "05_haldane_analysis.py")


@pytest.fixture(scope="module")
def adv06():
    return _load_script("audit_advanced_fitting", "06_advanced_fitting.py")


@pytest.fixture(scope="module")
def cmp07():
    return _load_script("audit_compare_truncation", "07_compare_truncation_methods.py")


@pytest.fixture(scope="module")
def gca():
    return _load_script("audit_growth_curve_analysis", "01_growth_curve_analysis.py")


def _group_dir(group):
    return DATA_RAW / GROUP_DIRS[group]


def _normalise(column):
    """The strain name 01 derives from a column name."""
    return str(column).replace('_blanked', '').replace('_', '-')


# =============================================================================
# downstream-01: the curve lookup
# =============================================================================

class TestCurveLookup:
    """load_raw_data() must return the strain's own column."""

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="pipeline results not generated")
    def test_every_strain_resolves_to_its_own_column(self):
        results = pd.read_csv(RESULTS_CSV)
        assert len(results) > 0
        wrong = []
        for _, row in results.iterrows():
            ref = find_curve_ref(_group_dir(row['group']), row['strain'])
            if ref is None or ref.name.upper() != str(row['strain']).upper():
                wrong.append((row['strain'], None if ref is None else ref.column))
        assert wrong == [], f"strains loaded from another column: {wrong}"

    @pytest.mark.parametrize("group,strain,expected_file,expected_column", [
        # Each of these used to answer with another medium's or another
        # strain's column, in glob order, through the substring fallback.
        ('Group2', 'LB-MAL1', 'LB_DATA.csv', 'LB_MAL1_blanked'),
        ('Group2', 'MALATHIONANDLB-MAL1', 'MALATHIONANDLB_DATA.csv',
         'MALATHIONANDLB_MAL1_blanked'),          # was ..._MAL10_blanked
        ('Group2', 'H2O-MAL1', 'H2O_DATA.csv', 'H2O_MAL1_blanked'),
        ('Group2', 'LAMBDACYHALOTHRIN-LCY1', 'LAMBDACYHALOTHRIN_DATA.csv',
         'LAMBDACYHALOTHRIN_LCY1_blanked'),       # was the ANDLB file
        ('Group1', 'LB-BIF2', 'LB_DATA.csv', 'LB_BIF2_blanked'),
        ('Group1', 'Bifenthrin-BIF2', 'Bifenthrin_DATA.csv', 'Bifenthrin_BIF2_blanked'),
        ('Group4', 'LB-CISPERM1', 'LB_DATA.csv', 'LB_CISPERM1'),
    ])
    def test_known_mis_mapped_strains(self, group, strain, expected_file, expected_column):
        ref = find_curve_ref(_group_dir(group), strain)
        assert ref is not None, f"{strain} not found in {group}"
        assert ref.file.name == expected_file
        assert ref.column == expected_column
        assert _normalise(ref.column).upper() == strain.upper()

    def test_missing_strain_is_reported_missing_not_guessed(self):
        # 'MAL' is a substring of many Group 2 columns; the old partial match
        # would have returned one of them.
        time, od, path, column = find_curve(_group_dir('Group2'), 'LB-MAL999')
        assert (time, od, path, column) == (None, None, None, None)

    def test_scripts_05_and_06_load_the_same_column_as_the_csv(self, h05, adv06):
        expected = pd.read_csv(_group_dir('Group2') / 'LB_DATA.csv')['LB_MAL1_blanked']
        for module in (h05, adv06):
            time, od = module.load_raw_data(DATA_RAW, 'LB-MAL1', 'Group2')
            assert time is not None
            np.testing.assert_allclose(od, expected.to_numpy(dtype=float))

    def test_unknown_group_returns_nothing(self, adv06):
        assert adv06.load_raw_data(DATA_RAW, 'LB-MAL1', 'GroupNope') == (None, None)


# =============================================================================
# trunc-05: the adaptive window
# =============================================================================

class TestAdaptiveTruncationWindow:

    def test_wrapper_matches_truncate_at_max_on_a_group1_curve(self, adv06, gca):
        time, od = adv06.load_raw_data(DATA_RAW, 'LB-BIF2', 'Group1')
        mask = np.isfinite(time) & np.isfinite(od)
        time, od = time[mask].astype(float), od[mask].astype(float)

        idx, confidence, meta = adv06.adaptive_r2_truncate_wrapper(time, od)
        expected = gca.truncate_at_max(time, od, use_adaptive=True)

        assert idx == int(expected.truncation_index)
        assert meta['start_idx'] == int(expected.start_index)
        assert 0.0 <= confidence <= 1.0

    def test_wrapper_reports_the_trimmed_start_on_a_noisy_baseline(self, adv06, gca):
        """The MCCV branch trims a start index; it must not be dropped."""
        time = np.linspace(0, 30, 80)
        rng = np.random.default_rng(3)
        od = gca.gompertz_model(time, 1.0, 0.25, 6.0) + rng.normal(0, 0.01, 80)
        od[:8] += rng.normal(0, 0.05, 8)            # noisy baseline
        od[-12:] -= np.linspace(0, 0.15, 12)        # post-peak decline

        idx, _confidence, meta = adv06.adaptive_r2_truncate_wrapper(time, od)
        expected = gca.truncate_at_max(time, od, use_adaptive=True)

        assert meta['start_idx'] == int(expected.start_index) > 0
        assert idx == int(expected.truncation_index)

    def test_fit_at_truncation_honours_the_start_index(self, cmp07, gca):
        """Fitting from 0 and from the window's start must differ."""
        time = np.linspace(0, 30, 80)
        rng = np.random.default_rng(5)
        od = gca.gompertz_model(time, 1.0, 0.25, 6.0) + rng.normal(0, 0.005, 80)
        od[:10] += 0.3                               # junk baseline

        from_zero = cmp07.fit_at_truncation(time, od, 60, 0)
        from_start = cmp07.fit_at_truncation(time, od, 60, 10)
        direct = gca.fit_gompertz(time[10:61], od[10:61])

        assert from_zero is not None and from_start is not None
        assert from_start.r_squared == pytest.approx(direct.r_squared)
        assert from_start.r_squared > from_zero.r_squared


# =============================================================================
# compare-01: what counts as a rescue
# =============================================================================

class TestBadStrainRescue:

    def test_flat_h2o_like_curve_is_not_rescued(self, cmp07, tmp_path):
        """
        A small, clean, low-amplitude sigmoid — an H2O control — fits well
        enough to pass R² and the parameter-error checks, which is how 12 of
        15 "rescued" strains got through with delta_od below the 0.15 gate.
        """
        time = np.linspace(0, 24, 100)
        rng = np.random.default_rng(7)
        od = 0.01 + cmp07.gompertz_model(time, 0.05, 0.01, 5.0) + rng.normal(0, 0.002, 100)

        data_dir = tmp_path / 'raw'
        group_dir = data_dir / 'Group1' / 'Group_1_DATA'
        group_dir.mkdir(parents=True)
        pd.DataFrame({'TIME[H]': time, 'H2O_FLAT_blanked': od}).to_csv(
            group_dir / 'H2O_DATA.csv', index=False)

        results_df = pd.DataFrame([{
            'strain': 'H2O-FLAT', 'group': 'Group1', 'is_good': False,
            'delta_od': float(od.max() - od[:10].mean()),
            'classification_reason': 'BAD: Insufficient growth',
        }])

        bad_df, rescued_df = cmp07.run_bad_strain_analysis(
            results_df, data_dir, cmp07.load_config(), tmp_path / 'out')

        # Precondition: the old rule (R² >= 0.95, errors < 20%) would have
        # called this a rescue.
        old_rule = bad_df[(bad_df['fit_success'] == True) &
                          (bad_df['r_squared'] >= 0.95) &
                          (bad_df['a_err_pct'] < 20) &
                          (bad_df['mu_err_pct'] < 20)]
        assert len(old_rule) > 0, "synthetic curve no longer reaches the old rescue rule"

        assert rescued_df.empty, "a flat H2O-like control was 'rescued'"
        assert (bad_df['rescue_verdict_good'] != True).all()
        reasons = ' '.join(bad_df['rescue_verdict_reason'].dropna().tolist())
        assert 'Insufficient growth' in reasons

    def test_methods_are_ranked_by_median_r2(self, cmp07):
        """A method with one catastrophic fit must not lose on its mean."""
        rows = []
        for method in cmp07.METHODS:
            r2s = {'first_peak': [0.99, 0.99, 0.99, 0.10],   # best median, bad mean
                   }.get(method, [0.96, 0.96, 0.96, 0.96])
            for r2 in r2s:
                rows.append({
                    'strain': f'S{r2}', 'group': 'Group1', 'method': method,
                    'fit_success': True, 'r_squared': r2, 'rmse': 0.01,
                    'a_err_pct': 1.0, 'mu_err_pct': 1.0,
                })
        summary = cmp07.compute_method_summary(pd.DataFrame(rows), {})
        assert summary.iloc[0]['method'] == 'first_peak'
        assert summary.iloc[0]['median_r2'] > summary.iloc[0]['mean_r2']
        assert 'n_failures' in summary.columns


# =============================================================================
# haldane-01: the Haldane fit
# =============================================================================

def _haldane_curve(h05, Ki, S0, q, mu_max=0.5, Ks=0.1, X_max=1.2, X0=0.01,
                   noise=1e-3, n=60, seed=0):
    """
    Synthetic data from 05's own ODE solver.

    ``q`` is chosen so the substrate really is consumed: mu(S) has to travel
    across the inhibited and uninhibited regimes during the run or the curve
    carries no information about Ki at all (see
    test_unidentifiable_ki_is_reported_as_nan for that case).
    """
    time = np.linspace(0, 50, n)
    X, _S = h05.solve_haldane(time, mu_max, Ks, Ki, X_max, q, X0, S0)
    od = X + np.random.default_rng(seed).normal(0, noise, len(time))
    return time, od


@pytest.fixture(scope="module")
def strong_inhibition_fit(h05):
    """Ki below S0: growth is strongly inhibited until the substrate drops."""
    time, od = _haldane_curve(h05, Ki=0.5, S0=1.0, q=1.0)
    return h05.fit_haldane(time, od, S0=1.0)


@pytest.fixture(scope="module")
def weak_inhibition_fit(h05):
    """Ki far above S0: inhibition is a small correction on the growth rate."""
    time, od = _haldane_curve(h05, Ki=20.0, S0=1.0, q=1.0)
    return h05.fit_haldane(time, od, S0=1.0)


class TestHaldaneIdentifiability:

    def test_s0_is_held_fixed(self, strong_inhibition_fit):
        assert strong_inhibition_fit['success'] is True
        assert strong_inhibition_fit['S0'] == 1.0

    def test_strong_inhibition_recovers_ki(self, strong_inhibition_fit):
        ki = strong_inhibition_fit['Ki_fitted']
        assert 0.25 <= ki <= 1.0, f"Ki={ki} is not within a factor of 2 of 0.5"

    def test_weak_inhibition_recovers_ki(self, weak_inhibition_fit):
        assert weak_inhibition_fit['success'] is True
        ki = weak_inhibition_fit['Ki_fitted']
        assert 10.0 <= ki <= 40.0, f"Ki={ki} is not within a factor of 2 of 20"

    def test_ki_is_not_the_initial_guess(self, strong_inhibition_fit, weak_inhibition_fit):
        """Every strain used to come back with Ki == 10.0, the initial guess."""
        strong = strong_inhibition_fit['Ki_fitted']
        weak = weak_inhibition_fit['Ki_fitted']
        assert not np.isclose(strong, 10.0, rtol=1e-3)
        assert not np.isclose(weak, 10.0, rtol=1e-3)
        assert strong < weak / 10, "Ki no longer separates strong from weak inhibition"

    def test_ki_reported_only_when_identified(self, strong_inhibition_fit,
                                              weak_inhibition_fit):
        for fit in (strong_inhibition_fit, weak_inhibition_fit):
            if fit['Ki_identified']:
                assert fit['Ki'] == fit['Ki_fitted']
            else:
                assert np.isnan(fit['Ki'])

    def test_unidentifiable_ki_is_reported_as_nan(self, h05):
        """
        With no substrate depletion mu(S) never varies, so the curve carries
        no information about Ki: it has to come back NaN, not as whatever
        value the optimiser stopped at.
        """
        time, od = _haldane_curve(h05, Ki=0.5, S0=1.0, q=1e-3, n=40)
        result = h05.fit_haldane(time, od, S0=1.0)
        assert result['success'] is True
        assert result['Ki_identified'] is False
        assert np.isnan(result['Ki'])
        assert np.isfinite(result['Ki_fitted'])

    def test_solver_failure_is_reported_as_a_failed_fit(self, h05, monkeypatch):
        """The flat fallback must never be dressed up as a successful fit."""
        def failing_solver(time, *args, return_status=False, **kwargs):
            flat = (np.full_like(time, 0.01, dtype=float),
                    np.full_like(time, 1.0, dtype=float))
            return flat + (False,) if return_status else flat

        monkeypatch.setattr(h05, 'solve_haldane', failing_solver)
        time = np.linspace(0, 50, 60)
        od = np.full_like(time, 0.01)
        result = h05.fit_haldane(time, od, S0=1.0, ki_starts=(10.0,))
        assert result['success'] is False
        assert result['Ki_identified'] is False

    def test_free_parameter_count_excludes_s0(self, strong_inhibition_fit):
        """AICc must count the six fitted parameters, not the fixed S0."""
        result = strong_inhibition_fit
        n = result['n_points']
        k = 6
        expected = n * np.log(result['sse'] / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        assert result['aic'] == pytest.approx(expected)


# =============================================================================
# haldane-02: the Gompertz side of the comparison
# =============================================================================

class TestGompertzComparisonWindow:

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="pipeline results not generated")
    def test_good_strain_gets_a_usable_gompertz_r2(self, h05):
        """
        LambdaCyhalothrinANDLB-LCY4 is GOOD in the pipeline, but scoring 01's
        truncated-window parameters on the full curve gave R² = 0.61 and handed
        the AICc comparison to Haldane by default.
        """
        strain, group = 'LambdaCyhalothrinANDLB-LCY4', 'Group1'
        results = pd.read_csv(RESULTS_CSV).set_index('strain')
        assert bool(results.loc[strain, 'is_good']) is True

        time, od = h05.load_raw_data(DATA_RAW, strain, group)
        mask = np.isfinite(time) & np.isfinite(od)
        time, od = time[mask].astype(float), od[mask].astype(float)

        pipeline = {
            'a': float(results.loc[strain, 'gompertz_a']),
            'mu': float(results.loc[strain, 'gompertz_mu']),
            'lambda': float(results.loc[strain, 'gompertz_lambda']),
        }
        params, source = h05.gompertz_on_window(time, od, pipeline)
        _aic, _bic, r2_comparison, _rmse = h05.gompertz_aic(
            time, od, params['a'], params['mu'], params['lambda'])
        _aic, _bic, r2_pipeline_params, _rmse = h05.gompertz_aic(
            time, od, pipeline['a'], pipeline['mu'], pipeline['lambda'])

        assert r2_comparison >= 0.9
        assert r2_comparison > r2_pipeline_params
        assert source == 'refit'

    def test_comparison_never_scores_worse_than_the_pipeline_parameters(self, h05, gca):
        """Whichever parameter set fits the window better is the one used."""
        time = np.linspace(0, 40, 120)
        rng = np.random.default_rng(11)
        od = gca.gompertz_model(time, 1.0, 0.3, 5.0) + rng.normal(0, 0.01, 120)
        od[80:] -= np.linspace(0, 0.4, 40)          # decline the model cannot fit

        pipeline = {'a': 1.0, 'mu': 0.3, 'lambda': 5.0}
        params, source = h05.gompertz_on_window(time, od, pipeline)
        assert source in ('refit', 'pipeline')
        assert (h05._gompertz_sse(time, od, params)
                <= h05._gompertz_sse(time, od, pipeline) + 1e-9)
