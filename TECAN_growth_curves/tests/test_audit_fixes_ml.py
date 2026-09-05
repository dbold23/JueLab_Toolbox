"""Regression tests for the ML / synthetic-data audit findings.

Findings covered:

  synth-01  ScenarioConfig had no 'instrument_noise' field, so importing
            curve_scenarios raised TypeError and the generator package was
            unusable.
  synth-02  Scenario categories were hard-coded (death_phase_severe filed as
            'Bad' although the scenario expects GOOD) instead of being derived
            from the scenario definitions.
  ml-01     The post-fit classifier was trained on a processing_results.csv
            produced WITH --ml-classify, in which every gate-rejected curve
            carries NaN for all fit features — so it learned
            "has fit features -> GOOD".
  ml-02     'CURVE0330' was parsed as concentration_numeric=330, i.e. the curve
            id, which encodes the scenario and therefore the label.
  ml-03     The audit merge read the CURRENT pipeline verdict instead of the
            audited one and NaN-ed out the gompertz features.
  ml-04     The ground-truth merge renamed delta_od/max_od/initial_od away.
  ml-05     Real audit rows were concatenated into the training frame before
            the train/test split, then "validated" on.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / 'scripts'
SYNTH = REPO_ROOT / 'synthetic_data'
COMPREHENSIVE = SYNTH / 'output' / 'comprehensive_test'
GROUND_TRUTH = COMPREHENSIVE / 'test_data' / 'ground_truth.csv'
DATA_DIR = COMPREHENSIVE / 'test_data' / 'DATA'
AUDIT_CSV = REPO_ROOT / 'results' / 'tables' / 'validation_audit.csv'
RESULTS_DIR = REPO_ROOT / 'results' / 'tables'

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SYNTH))


def _load_trainer():
    """Import 09_train_classifier.py (module name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        'train_classifier', str(SCRIPTS / '09_train_classifier.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def trainer():
    return _load_trainer()


# ---------------------------------------------------------------------------
# synth-01: curve_scenarios imports and every scenario is constructible
# ---------------------------------------------------------------------------

class TestScenarioDefinitions:

    def test_module_imports(self):
        import src.curve_scenarios as cs  # noqa: F401

    def test_every_scenario_instantiates(self):
        from dataclasses import fields
        from src.curve_scenarios import ALL_SCENARIOS, ScenarioConfig

        assert len(ALL_SCENARIOS) > 0
        names = {f.name for f in fields(ScenarioConfig)}
        for name, config in ALL_SCENARIOS.items():
            assert isinstance(config, ScenarioConfig), name
            assert config.expected_class in ('GOOD', 'BAD'), name
            # Rebuilding from its own fields must work (the TypeError case)
            clone = ScenarioConfig(**{f: getattr(config, f) for f in names})
            assert clone.name == config.name

    def test_instrument_noise_field_exists_and_is_sampled(self):
        from src.curve_scenarios import ALL_SCENARIOS, ScenarioSampler

        assert ALL_SCENARIOS['evaporation_drift'].instrument_noise is True
        assert ALL_SCENARIOS['bubble_spike'].instrument_noise is True
        assert ALL_SCENARIOS['standard'].instrument_noise is False

        for name, config in ALL_SCENARIOS.items():
            params = ScenarioSampler(config, seed=0).sample_parameters()
            assert params['instrument_noise'] == config.instrument_noise, name
            assert params['expected_class'] == config.expected_class, name


# ---------------------------------------------------------------------------
# synth-02: docs and report categories follow the scenario definitions
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def summary_module():
    spec = importlib.util.spec_from_file_location(
        'validation_summary', str(SYNTH / 'scripts' / 'validation_summary.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScenarioDocs:

    def test_categories_match_scenario_dicts(self, summary_module):
        from src.curve_scenarios import (
            BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS, GOOD_GROWTH_SCENARIOS,
        )
        cats = summary_module.SCENARIO_CATEGORIES
        assert set(cats['Good (standard growth)']) == set(GOOD_GROWTH_SCENARIOS)
        assert set(cats['Bad (no / failed growth)']) == set(BAD_CURVE_SCENARIOS)
        assert set(cats['Edge / borderline']) == set(EDGE_CASE_SCENARIOS)

    def test_good_edge_scenarios_not_filed_as_bad(self, summary_module):
        """death_phase_severe / sparse_data expect GOOD, so they are not 'Bad'."""
        from src.curve_scenarios import ALL_SCENARIOS

        bad_bucket = set(summary_module.SCENARIO_CATEGORIES['Bad (no / failed growth)'])
        for name in ('death_phase_severe', 'sparse_data'):
            assert ALL_SCENARIOS[name].expected_class == 'GOOD'
            assert name not in bad_bucket

    def test_every_scenario_is_categorised(self, summary_module):
        from src.curve_scenarios import ALL_SCENARIOS

        for name in ALL_SCENARIOS:
            assert summary_module.categorise_scenario(name) != 'Other', name

    def test_readme_tables_list_the_real_scenarios(self):
        from src.curve_scenarios import (
            BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS, GOOD_GROWTH_SCENARIOS,
        )
        readme = (SYNTH / 'README.md').read_text()
        coverage = readme.split('## Scenario Coverage', 1)[1].split('## Noise Models', 1)[0]

        good = coverage.split('### Bad Curve Scenarios', 1)[0]
        bad = coverage.split('### Bad Curve Scenarios', 1)[1].split('### Edge Case', 1)[0]
        edge = coverage.split('### Edge Case', 1)[1]

        for name in GOOD_GROWTH_SCENARIOS:
            assert f'| {name} |' in good, name
        for name in BAD_CURVE_SCENARIOS:
            assert f'| {name} |' in bad, name
        for name in EDGE_CASE_SCENARIOS:
            assert f'| {name} |' in edge, name

        # sparse_data expects GOOD; it must not be in the BAD table
        assert '| sparse_data |' not in bad

    def test_generate_comprehensive_docstring_counts(self):
        from src.curve_scenarios import (
            BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS, GOOD_GROWTH_SCENARIOS,
        )
        text = (SYNTH / 'scripts' / 'generate_comprehensive.py').read_text()
        head = text.split('def main', 1)[0]
        assert f'good growth scenarios ({len(GOOD_GROWTH_SCENARIOS)} types)' in head
        assert f'bad curve scenarios ({len(BAD_CURVE_SCENARIOS)} types)' in head
        assert f'edge case scenarios ({len(EDGE_CASE_SCENARIOS)} types)' in head


# ---------------------------------------------------------------------------
# ml-02: strain-name metadata must not leak the synthetic curve id
# ---------------------------------------------------------------------------

class TestMetadataFeatures:

    def test_synthetic_curve_id_is_not_a_concentration(self):
        from ml_classifier import extract_metadata_features

        meta = extract_metadata_features('CURVE0330')
        assert np.isnan(meta['concentration_numeric'])
        assert np.isnan(meta['is_control'])

    def test_pipeline_style_synthetic_name_is_nan(self):
        from ml_classifier import extract_metadata_features

        meta = extract_metadata_features('SYNTHETIC-80pts-19h-CURVE0330')
        assert np.isnan(meta['concentration_numeric'])
        assert np.isnan(meta['is_control'])

    def test_lab_convention_still_parsed(self):
        from ml_classifier import extract_metadata_features

        assert extract_metadata_features('H2O-MAL10') == {
            'is_control': 1.0, 'concentration_numeric': 10.0}
        assert extract_metadata_features('MALATHIONANDLB-MAL4') == {
            'is_control': 0.0, 'concentration_numeric': 4.0}

    def test_unknown_media_token_is_nan(self):
        from ml_classifier import extract_metadata_features

        meta = extract_metadata_features('NOTAMEDIUM-XYZ7')
        assert np.isnan(meta['is_control'])
        assert np.isnan(meta['concentration_numeric'])

    def test_more_than_three_digits_is_not_a_concentration(self):
        from ml_classifier import extract_metadata_features

        assert np.isnan(extract_metadata_features('H2O-MAL1234')['concentration_numeric'])

    def test_real_strain_names_parse(self):
        """Every audited real strain still yields usable metadata."""
        from ml_classifier import extract_metadata_features

        audit = pd.read_csv(AUDIT_CSV)
        for strain in audit['strain'].astype(str):
            meta = extract_metadata_features(strain)
            assert not np.isnan(meta['is_control']), strain
            assert not np.isnan(meta['concentration_numeric']), strain

    def test_prefit_features_carry_nan_metadata_for_synthetic(self):
        from ml_classifier import extract_prefit_features

        t = np.linspace(0, 24, 60)
        od = 0.05 + 1.0 / (1 + np.exp(-(t - 10) / 2))
        features = extract_prefit_features(t, od, strain_name='CURVE0330')
        assert np.isnan(features['concentration_numeric'])
        assert np.isnan(features['is_control'])


# ---------------------------------------------------------------------------
# ml-03: real audit labels come from the audited verdict
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def audit_df(trainer):
    if not AUDIT_CSV.exists():
        pytest.skip('audit CSV not present')
    return trainer.load_real_audit_data(str(AUDIT_CSV), str(RESULTS_DIR))


@pytest.mark.skipif(not AUDIT_CSV.exists(), reason='audit CSV not present')
class TestRealAuditLabels:

    def test_label_counts(self, audit_df):
        assert len(audit_df) == 85
        assert int(audit_df['label'].sum()) == 55
        assert int((audit_df['label'] == 0).sum()) == 30

    def test_labels_follow_the_audited_verdict(self, audit_df):
        audited = audit_df['audited_is_good'].astype(bool)
        correct = audit_df['audit_result'].str.strip().str.lower() == 'correct'
        expected = (audited == correct).astype(int)
        assert (audit_df['label'].values == expected.values).all()

    def test_unsure_rows_dropped(self, audit_df):
        assert not (audit_df['audit_result'].str.lower() == 'unsure').any()

    def test_feature_columns_survive_the_merge(self, audit_df, trainer):
        X, _ = trainer.prepare_postfit_features(audit_df)
        all_nan = [c for c in X.columns if X[c].isna().all()]
        assert all_nan == []
        for col in ('gompertz_a', 'gompertz_mu', 'gompertz_lambda', 'fit_r_squared'):
            assert X[col].notna().any(), col


# ---------------------------------------------------------------------------
# ml-04: ground-truth merge must not rename the pipeline's feature columns
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def merged(trainer):
    pr = COMPREHENSIVE / 'validation_latest' / 'processing_results.csv'
    if not (GROUND_TRUTH.exists() and pr.exists()):
        pytest.skip('synthetic ground truth / results not present')
    return trainer.load_training_data(str(GROUND_TRUTH), str(pr))


class TestSyntheticMergeKeepsFeatures:

    def test_overlapping_columns_keep_pipeline_names(self, merged):
        for col in ('delta_od', 'max_od', 'initial_od'):
            assert col in merged.columns
            assert f'{col}_pr' not in merged.columns
            assert f'gt_{col}' in merged.columns

    def test_no_feature_column_is_entirely_nan(self, merged, trainer):
        X, _ = trainer.prepare_postfit_features(merged)
        trainer.assert_features_present(X, 'synthetic')  # raises on all-NaN
        for col in ('delta_od', 'max_od'):
            assert X[col].notna().any()

    def test_assert_features_present_still_catches_a_dropped_column(self, trainer):
        X = pd.DataFrame({'delta_od': [np.nan, np.nan], 'max_od': [1.0, 2.0]})
        with pytest.raises(ValueError, match='delta_od'):
            trainer.assert_features_present(X, 'synthetic')


# ---------------------------------------------------------------------------
# ml-01 / ml-05: honest training features, trained on synthetic only
# ---------------------------------------------------------------------------

GOOD_SUBSET_SCENARIOS = ['standard', 'very_clean', 'high_A', 'fast_growth']
BAD_SUBSET_SCENARIOS = ['high_noise', 'borderline_noise', 'erratic',
                        'minimal_growth', 'contamination', 'flat_no_growth']


@pytest.fixture(scope='module')
def synthetic_subset(tmp_path_factory):
    """A small synthetic DATA directory (fast to fit) plus its ground truth."""
    source = DATA_DIR / 'SYNTHETIC_400pts_99h_DATA.csv'
    if not (source.exists() and GROUND_TRUTH.exists()):
        pytest.skip('synthetic test data not present')

    gt = pd.read_csv(GROUND_TRUTH)
    keep = []
    for scenario in GOOD_SUBSET_SCENARIOS + BAD_SUBSET_SCENARIOS:
        keep += list(gt.loc[gt['scenario'] == scenario, 'strain_name'].head(5))

    df = pd.read_csv(source)
    columns = [df.columns[0]] + [c for c in df.columns
                                 if any(k in c for k in keep)]
    assert len(columns) > 20, 'subset should contain a usable number of curves'

    root = tmp_path_factory.mktemp('synthetic_subset')
    data_dir = root / 'DATA'
    data_dir.mkdir()
    df[columns].to_csv(data_dir / 'SYNTHETIC_400pts_99h_DATA.csv', index=False)
    return {'root': root, 'data_dir': data_dir, 'n_curves': len(columns) - 1}


@pytest.fixture(scope='module')
def training_features(synthetic_subset, trainer):
    """Features built the way 09 now builds them: no ML gate involved."""
    out_dir = synthetic_subset['root'] / 'training_features'
    csv = trainer.build_training_features(
        str(synthetic_subset['data_dir']), str(out_dir),
        extra_args=['--no-adaptive'],      # keeps the test under a minute
    )
    return Path(csv)


@pytest.fixture(scope='module')
def trained_subset(training_features, trainer):
    df = trainer.load_training_data(str(GROUND_TRUTH), str(training_features))
    fit_df = df[df['fit_attempted']].reset_index(drop=True)
    X, y = trainer.prepare_postfit_features(fit_df)
    model, _ = trainer.train_model(X, y, 'hgb', n_folds=3)
    return {'model': model, 'X': trainer.clean_feature_matrix(X), 'y': y, 'df': fit_df}


class TestHonestTrainingFeatures:

    def test_features_are_built_without_the_ml_gate(self, training_features):
        pr = pd.read_csv(training_features)
        # No ML verdict columns filled in -> the run had no --ml-classify
        assert pr['p_good'].isna().all()
        assert pr['ml_classification'].isna().all()
        assert not pr['classification_reason'].astype(str).str.contains('ML').any()

    def test_bad_curves_reach_the_fit(self, trained_subset):
        """The whole point of ml-01: BAD curves must carry fit features."""
        y = trained_subset['y']
        assert (y == 0).sum() > 0, 'no BAD curves survived to the post-fit stage'
        assert (y == 1).sum() > 0
        X = trained_subset['X']
        assert X.loc[y == 0, 'fit_r_squared'].notna().any()

    def test_build_is_cached(self, synthetic_subset, training_features, trainer):
        """A second call reuses the CSV instead of re-running the pipeline."""
        before = training_features.stat().st_mtime_ns
        again = trainer.build_training_features(
            str(synthetic_subset['data_dir']),
            str(training_features.parent),
            extra_args=['--no-adaptive'],
        )
        assert Path(again) == training_features
        assert Path(again).stat().st_mtime_ns == before

    @staticmethod
    def _corrupt(row, **overrides):
        """Row with overridden fit features and derived features recomputed."""
        from ml_classifier import compute_derived_features

        corrupted = row.copy()
        for key, value in overrides.items():
            corrupted[key] = value
        for key, value in compute_derived_features(corrupted.to_dict()).items():
            if key in corrupted.index:
                corrupted[key] = value
        return pd.DataFrame([corrupted])

    def test_corrupted_fit_features_lower_p_good(self, trained_subset):
        """A GOOD row with a ruined fit must stop being GOOD."""
        model, X, y = trained_subset['model'], trained_subset['X'], trained_subset['y']
        probs = model.predict_proba(X)[:, 1]

        good_probs = np.where(y == 1, probs, -np.inf)
        idx = int(np.argmax(good_probs))
        assert probs[idx] >= 0.7, 'expected at least one confidently GOOD row'

        corrupted = self._corrupt(X.iloc[idx], fit_r_squared=0.0, a_err_pct=300.0)
        p_corrupt = float(model.predict_proba(corrupted)[0, 1])

        assert p_corrupt < 0.7, (
            f'p_good stayed at {p_corrupt:.3f} after ruining the fit features — '
            f'the classifier is not reading fit quality'
        )
        assert p_corrupt < probs[idx]

    def test_ruined_fit_is_classified_bad(self, trained_subset):
        """Ruining the whole fit-quality family drives every GOOD row to BAD."""
        model, X, y = trained_subset['model'], trained_subset['X'], trained_subset['y']

        worst = 0.0
        for idx in np.where(y == 1)[0]:
            corrupted = self._corrupt(
                X.iloc[idx], fit_r_squared=0.0, fit_rmse=0.5, fit_mae=0.4,
                a_err_pct=300.0, mu_err_pct=300.0,
            )
            worst = max(worst, float(model.predict_proba(corrupted)[0, 1]))
        assert worst < 0.3, f'a ruined fit still scored p_good={worst:.3f}'


class TestTrainerCommandLine:

    def test_training_refuses_gate_rejected_features(
        self, synthetic_subset, training_features, tmp_path
    ):
        """ml-01: a run where BAD curves were rejected before fitting is refused.

        Blanking the fit columns of every BAD row reproduces what
        --ml-classify does: the fitted rows are then all GOOD, and a classifier
        trained on them can only learn "has fit features -> GOOD".
        """
        pr = pd.read_csv(training_features)
        gt = pd.read_csv(GROUND_TRUTH)
        bad = set(gt.loc[gt['expected_class'] == 'BAD', 'strain_name'])
        is_bad = pr['strain'].apply(
            lambda s: any(name in str(s) for name in bad)
        )
        assert is_bad.any(), 'subset should contain BAD curves'

        fit_cols = ['fit_r_squared', 'fit_rmse', 'fit_mae',
                    'gompertz_a', 'gompertz_mu', 'gompertz_lambda']
        pr.loc[is_bad, fit_cols] = np.nan
        gated = tmp_path / 'gate_rejected_results.csv'
        pr.to_csv(gated, index=False)

        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / '09_train_classifier.py'),
             '--processing-results', str(gated),
             '--raw-data-dir', str(synthetic_subset['data_dir']),
             '--output', str(tmp_path / 'models'),
             '--n-folds', '3'],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, 'MPLBACKEND': 'Agg'},
        )
        assert proc.returncode != 0, 'training accepted single-class post-fit data'
        assert 'all one class' in proc.stdout

    def test_ml_gated_results_are_detected(self, trainer, training_features):
        validation = COMPREHENSIVE / 'validation_latest' / 'processing_results.csv'
        assert trainer.results_are_ml_gated(training_features) is False
        if validation.exists():
            gated = pd.read_csv(validation)
            if gated['p_good'].notna().any():
                assert trainer.results_are_ml_gated(validation) is True

    def test_default_does_not_train_on_the_ml_gated_validation_run(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / '09_train_classifier.py'), '--help'],
            capture_output=True, text=True, check=True,
        ).stdout
        assert 'validation_latest' not in out
        assert '--rebuild-features' in out
        assert 'training_features' in out

    def test_default_features_dir_is_untracked(self):
        trainer_mod = _load_trainer()
        assert trainer_mod.DEFAULT_TRAINING_FEATURES_DIR.endswith('training_features')
        gitignore = (REPO_ROOT / '.gitignore').read_text()
        assert 'training_features' in gitignore

    @pytest.mark.skipif(not AUDIT_CSV.exists(), reason='audit CSV not present')
    def test_real_audit_is_held_out(self, synthetic_subset, training_features, tmp_path):
        """ml-05: train on synthetic only, report real data as external test."""
        models_dir = tmp_path / 'models'
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / '09_train_classifier.py'),
             '--raw-data-dir', str(synthetic_subset['data_dir']),
             '--training-features-dir', str(training_features.parent),
             '--output', str(models_dir),
             '--validate-real',
             '--n-folds', '3'],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, 'MPLBACKEND': 'Agg'},
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        assert 'no data leakage' not in proc.stdout.lower()

        config = json.loads((models_dir / 'feature_config.json').read_text())
        assert config['trained_on'] == 'synthetic_only'
        assert config['real_audit_holdout']['used_for_training'] is False
        assert config['real_audit_holdout']['n_curves'] == 85

        # Training counts cover the synthetic subset only (no 85 real rows)
        n_curves = synthetic_subset['n_curves']
        assert config['n_training_samples'] == n_curves
        assert config['n_postfit_training_samples'] <= n_curves
        assert config['n_postfit_training_samples'] > 0
        assert config['postfit_class_distribution']['BAD'] > 0
        assert config['postfit_class_distribution']['GOOD'] > 0

        # The external test numbers are reported, not folded into training
        external = config['real_audit_holdout']['external_test_results']
        assert external is not None
        assert external['n_curves'] == 85
