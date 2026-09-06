"""WP9 acceptance tests: gate ids and run provenance.

Two additive changes to the pipeline are covered here:

  * ``failed_gates`` -- a semicolon-joined list of canonical gate ids appended
    as the LAST column of ``processing_results.csv``. It is the machine
    readable twin of ``classification_reason``: today a report can only count
    scenarios, not say which gate produced a miss.
  * ``run_settings.json`` -- written beside every ``processing_results.csv``,
    recording the truncation mode, every threshold, the config and dataset
    hashes and the git revision. Nothing in the repository used to record which
    truncation mode produced a published number (``config.yaml`` sets adaptive
    while the committed headline is first-peak).

Neither changes a verdict: the non-regression half of this file re-runs the
committed synthetic plate and asserts the published confusion matrix
(319/100/35/26, accuracy 0.8729166666666667) unchanged.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / 'scripts'
SYNTH = REPO_ROOT / 'synthetic_data'
COMPREHENSIVE = SYNTH / 'output' / 'comprehensive_test'
SYNTH_DATA_DIR = COMPREHENSIVE / 'test_data' / 'DATA'
SYNTH_GROUND_TRUTH = COMPREHENSIVE / 'test_data' / 'ground_truth.csv'

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SYNTH))

#: The gate vocabulary WP9 fixes; frozen here so a rename has to be deliberate.
EXPECTED_CANONICAL = (
    'prefit_snr', 'prefit_rejected', 'too_few_points', 'fit_success',
    'r_squared', 'a_err_pct', 'mu_err_pct', 'snr', 'delta_od_ci', 'delta_od',
    'amplitude_vs_noise', 'monotone', 'residual_autocorr', 'ml_p_good',
)

EXPECTED_RUN_SETTINGS_KEYS = (
    'truncation_mode', 'use_adaptive', 'use_first_peak', 'smoothing_window',
    'buffer_points', 'min_points_for_fit', 'quality_gates', 'mccv',
    'ml_classify', 'config_path', 'config_sha256', 'git_sha', 'git_dirty',
    'data_dir', 'dataset_sha256', 'n_curves', 'wall_clock_s', 'timestamp_utc',
    'argv',
)


def _load_pipeline():
    """Import scripts/01_growth_curve_analysis.py (module name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        'growth_curve_analysis_wp9', str(SCRIPTS / '01_growth_curve_analysis.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def gca():
    return _load_pipeline()


@pytest.fixture(scope='module')
def synthetic_run(gca, tmp_path_factory):
    """The committed synthetic plate, first-peak truncation, no ML."""
    out = tmp_path_factory.mktemp('synth_first_peak')
    params = dict(gca.TRUNCATION_PARAMS)
    params.update(use_first_peak=True, use_adaptive=False)
    gca.process_growth_curves(
        data_dir=str(SYNTH_DATA_DIR), output_dir=str(out),
        truncation_params=params, verbose=False, no_plots=True,
        config_path=str(SCRIPTS / 'config.yaml'),
    )
    return out, pd.read_csv(out / 'processing_results.csv')


def _tiny_plate(directory):
    """One fittable curve and one flat well, 40 h at 30 min."""
    t = np.arange(0, 40, 0.5)
    a, mu, lam = 1.2, 0.2, 3.0
    good = a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1)) + 0.01
    flat = np.full_like(t, 0.02)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Time': t, 'GOODCURVE_blanked': good, 'FLAT_blanked': flat}).to_csv(
        directory / 'TINY_DATA.csv', index=False)
    return directory


def _run_cli(data_dir, out_dir, *flags):
    cmd = [sys.executable, str(SCRIPTS / '01_growth_curve_analysis.py'), str(data_dir),
           '-o', str(out_dir), '--no-plots', '-q', *flags]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads((Path(out_dir) / 'run_settings.json').read_text())


# ---------------------------------------------------------------------------
# The gate vocabulary
# ---------------------------------------------------------------------------

class TestGateVocabulary:

    def test_canonical_ids_are_the_contract(self, gca):
        assert gca.CANONICAL_GATE_IDS == EXPECTED_CANONICAL

    def test_extra_ids_are_disjoint_and_known(self, gca):
        assert not set(gca.EXTRA_GATE_IDS) & set(gca.CANONICAL_GATE_IDS)
        assert set(gca.KNOWN_GATE_IDS) == set(gca.CANONICAL_GATE_IDS) | set(gca.EXTRA_GATE_IDS)

    @pytest.mark.parametrize('reason,ids', [
        ('GOOD: Fit quality passed', []),
        ('GOOD: All criteria passed', []),
        ('BAD: Low R²: 0.812 < 0.95', ['r_squared']),
        ('BAD: High A error: inf% > 20.0%', ['a_err_pct']),
        ('BAD: High mu error: 31.2% > 20.0%', ['mu_err_pct']),
        ('BAD: Low SNR: 3.1 < 5.0', ['snr']),
        ('BAD: Delta OD not significant: CI lower=0.02 < 0.1', ['delta_od_ci']),
        ('BAD: Insufficient growth: delta_od=0.041 < 0.15', ['delta_od']),
        ('BAD: Growth amplitude below noise: A=0.010 < 3*noise=0.030',
         ['amplitude_vs_noise']),
        ('BAD: Non-monotonic signal: 0.48 < 0.55', ['monotone']),
        ('BAD: High residual autocorrelation: 0.83 > 0.7', ['residual_autocorr']),
        ('BAD: Pre-fit SNR too low (1.4 < 3.0)', ['prefit_snr']),
        ('BAD: Rejected by ML pre-fit gate', ['prefit_rejected']),
        ('BAD: Too few points after truncation: 12 < 20', ['too_few_points']),
        ('BAD: Insufficient data points (8 < 15)', ['too_few_points']),
        ('BAD: Maximum too early in experiment: 0.50 h < 1 h', ['peak_too_early']),
        ('BAD: Fit failed - Optimal parameters not found', ['fit_success']),
        ('BAD (ML): P(good)=0.123', ['ml_p_good']),
        ('BORDERLINE (ML): P(good)=0.512', ['ml_p_good']),
        ('BAD: Low R²: 0.5 < 0.95; High mu error: inf% > 20.0%; '
         'Insufficient growth: delta_od=0.1 < 0.15',
         ['r_squared', 'mu_err_pct', 'delta_od']),
    ])
    def test_reason_sentences_map_to_ids(self, gca, reason, ids):
        got, unmapped = gca.gate_ids_from_reason(reason)
        assert got == ids
        assert unmapped == []

    def test_unmapped_clause_is_reported_not_guessed(self, gca):
        ids, unmapped = gca.gate_ids_from_reason('BAD: Some future gate: 1 < 2')
        assert ids == []
        assert unmapped == ['Some future gate: 1 < 2']

    def test_none_and_empty_are_safe(self, gca):
        assert gca.gate_ids_from_reason(None) == ([], [])
        assert gca.gate_ids_from_reason('') == ([], [])


# ---------------------------------------------------------------------------
# The failed_gates column
# ---------------------------------------------------------------------------

class TestFailedGatesColumn:

    def test_column_is_last(self, synthetic_run):
        _, df = synthetic_run
        assert df.columns[-1] == 'failed_gates'

    def test_good_curves_have_no_failed_gate_and_bad_curves_do(self, synthetic_run):
        _, df = synthetic_run
        gates = df['failed_gates'].fillna('')
        assert (gates[df.is_good] == '').all()
        assert (gates[~df.is_good] != '').all()

    def test_every_id_is_canonical(self, gca, synthetic_run):
        _, df = synthetic_run
        seen = {g for row in df['failed_gates'].fillna('') for g in row.split(';') if g}
        assert seen <= set(gca.CANONICAL_GATE_IDS), seen - set(gca.CANONICAL_GATE_IDS)
        # the run must actually exercise several gates, not just one
        assert len(seen) >= 5

    def test_column_agrees_with_the_reason_sentence(self, gca, synthetic_run):
        _, df = synthetic_run
        for reason, gates in zip(df.classification_reason, df['failed_gates'].fillna('')):
            parsed, unmapped = gca.gate_ids_from_reason(reason)
            assert unmapped == [], reason
            assert parsed == [g for g in gates.split(';') if g], reason

    def test_verdicts_are_unchanged(self, synthetic_run):
        """The published synthetic numbers: 319/100/35/26, accuracy 0.8729166666666667."""
        _, df = synthetic_run
        gt = pd.read_csv(SYNTH_GROUND_TRUTH)
        curve = re.compile(r'CURVE\d+')
        df = df.assign(k=[curve.search(s).group(0) for s in df.strain])
        gt = gt.assign(k=[curve.search(s).group(0) for s in gt.strain_name])
        m = gt.merge(df, on='k', how='inner')
        assert len(m) == 480
        good = m.expected_class.str.upper() == 'GOOD'
        pred = m.is_good.astype(bool)
        tp = int((good & pred).sum()); tn = int((~good & ~pred).sum())
        fp = int((~good & pred).sum()); fn = int((good & ~pred).sum())
        assert (tp, tn, fp, fn) == (319, 100, 35, 26)
        assert (tp + tn) / len(m) == 0.8729166666666667


class TestCommittedResultsParse:
    """Every reason in a committed results table maps to a canonical gate id."""

    @pytest.mark.parametrize('csv_path', sorted(
        list((REPO_ROOT / 'results' / 'tables').glob('Group*_Results/processing_results.csv'))
        + list(COMPREHENSIVE.glob('*/processing_results.csv'))))
    def test_reasons_map_to_canonical_ids(self, gca, csv_path):
        df = pd.read_csv(csv_path)
        seen, unmapped = set(), []
        for reason in df.classification_reason.fillna(''):
            ids, un = gca.gate_ids_from_reason(reason)
            seen.update(ids)
            unmapped.extend(un)
        assert unmapped == [], f'{csv_path}: {unmapped[:3]}'
        assert seen <= set(gca.CANONICAL_GATE_IDS), seen - set(gca.CANONICAL_GATE_IDS)


# ---------------------------------------------------------------------------
# run_settings.json
# ---------------------------------------------------------------------------

class TestRunSettings:

    def test_written_beside_the_results_csv(self, synthetic_run):
        out, _ = synthetic_run
        assert (out / 'run_settings.json').exists()
        assert (out / 'processing_results.csv').exists()

    def test_keys_and_order(self, synthetic_run, gca):
        out, _ = synthetic_run
        settings = json.loads((out / 'run_settings.json').read_text())
        assert tuple(settings) == EXPECTED_RUN_SETTINGS_KEYS
        assert gca.RUN_SETTINGS_KEYS == EXPECTED_RUN_SETTINGS_KEYS

    def test_records_the_dataset_git_and_size(self, gca, synthetic_run):
        out, df = synthetic_run
        settings = json.loads((out / 'run_settings.json').read_text())
        assert settings['n_curves'] == len(df)
        assert settings['ml_classify'] is False
        assert settings['dataset_sha256'] == gca.dataset_sha256(SYNTH_DATA_DIR)
        assert len(settings['dataset_sha256']) == 64
        head = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
        assert settings['git_sha'] == head
        assert isinstance(settings['git_dirty'], bool)
        assert isinstance(settings['wall_clock_s'], float)

    def test_records_every_quality_gate_threshold(self, synthetic_run):
        out, _ = synthetic_run
        gates = json.loads((out / 'run_settings.json').read_text())['quality_gates']
        for key, value in [('min_r_squared', 0.95), ('max_param_error_pct', 20.0),
                           ('min_snr', 5.0), ('min_delta_od_ci', 0.1),
                           ('min_absolute_delta_od', 0.15), ('pre_fit_snr_threshold', 3.0),
                           ('pre_fit_delta_od_override', 0.5), ('excellent_r2_threshold', 0.98),
                           ('min_monotone_fraction', 0.55), ('max_residual_autocorr', 0.7),
                           ('noise_floor_od', 0.002), ('min_hours_to_peak', 1.0)]:
            assert gates[key] == value, key

    def test_dataset_hash_is_the_sorted_data_csv_contents(self, gca, tmp_path):
        import hashlib
        h = hashlib.sha256()
        for f in sorted(SYNTH_DATA_DIR.glob('*_DATA.csv'), key=lambda p: p.name):
            h.update(f.read_bytes())
        assert gca.dataset_sha256(SYNTH_DATA_DIR) == h.hexdigest()
        assert gca.dataset_sha256(tmp_path) is None

    def test_truncation_mode_follows_the_flag(self, tmp_path):
        data = _tiny_plate(tmp_path / 'data')
        first = _run_cli(data, tmp_path / 'fp', '--first-peak', '--no-adaptive')
        assert first['truncation_mode'] == 'first_peak'
        assert (first['use_adaptive'], first['use_first_peak']) == (False, True)
        assert first['mccv'] is None
        assert first['argv'][-2:] == ['--first-peak', '--no-adaptive']

        glob_max = _run_cli(data, tmp_path / 'gm', '--global-max', '--no-adaptive')
        assert glob_max['truncation_mode'] == 'global_max'
        assert (glob_max['use_adaptive'], glob_max['use_first_peak']) == (False, False)

        adaptive = _run_cli(data, tmp_path / 'ad', '--adaptive')
        assert adaptive['truncation_mode'] == 'adaptive'
        assert adaptive['use_adaptive'] is True
        assert adaptive['mccv']['n_cv_folds'] == 20
        assert adaptive['mccv']['n_coarse'] == 50
        assert adaptive['config_sha256'] is not None

    def test_mode_name_helper(self, gca):
        assert gca.truncation_mode_name({'use_adaptive': True, 'use_first_peak': True}) == 'adaptive'
        assert gca.truncation_mode_name({'use_adaptive': False, 'use_first_peak': True}) == 'first_peak'
        assert gca.truncation_mode_name({'use_adaptive': False, 'use_first_peak': False}) == 'global_max'


# ---------------------------------------------------------------------------
# The app path uses the same column
# ---------------------------------------------------------------------------

class TestTecanGrowthConsistency:

    def test_result_columns_end_with_failed_gates(self):
        from tecan_growth.analyze import RESULT_COLUMNS
        assert RESULT_COLUMNS[-1] == 'failed_gates'

    def test_short_curve_is_labelled_too_few_points(self):
        from tecan_growth.analyze import analyze_curve
        t = np.arange(0, 2.0, 0.5)
        cr = analyze_curve(t, np.full_like(t, 0.05), 'SHORT')
        assert cr.legacy.stage == 'too_few_points'
        assert cr.row['failed_gates'] == 'too_few_points'

    def test_flat_curve_gate_ids_match_the_reason(self):
        from tecan_growth.analyze import analyze_curve
        from tecan_growth.settings import AnalysisSettings
        t = np.arange(0, 40, 0.5)
        rng = np.random.default_rng(0)
        cr = analyze_curve(t, 0.05 + rng.normal(0, 0.002, t.size), 'FLAT',
                           AnalysisSettings.from_config(use_adaptive=False))
        assert not cr.is_good
        gca = _load_pipeline()
        parsed, unmapped = gca.gate_ids_from_reason(cr.row['classification_reason'])
        assert unmapped == []
        assert cr.row['failed_gates'].split(';') == parsed
