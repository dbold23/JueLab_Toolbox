"""Integrity properties of the ML training path (WP11).

These pin the four ways the classifier could go back to being dishonest:

  * a shipped feature that is constant in training and variable at inference
    (the two metadata columns: NaN for all 480 synthetic rows, frozen to 0.0 by
    clean_feature_matrix, and provably inert in the shipped models);
  * training features cached under one truncation mode and used under another,
    with nothing in any output naming either mode;
  * curves entering the training set as label 0 because the pipeline cannot
    identify their asymptote, rather than because they are bad;
  * a scenario whose labels are unlearnable at the decision boundary being
    memorised instead of flagged.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO_ROOT / 'scripts'
SYNTH = REPO_ROOT / 'synthetic_data'
COMPREHENSIVE = SYNTH / 'output' / 'comprehensive_test'
GROUND_TRUTH = COMPREHENSIVE / 'test_data' / 'ground_truth.csv'
DATA_DIR = COMPREHENSIVE / 'test_data' / 'DATA'

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SYNTH))


def _load_trainer():
    """Import 09_train_classifier.py (module name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        'train_classifier_wp11', str(SCRIPTS / '09_train_classifier.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def trainer():
    return _load_trainer()


# ---------------------------------------------------------------------------
# A small two-plate subset: n_points / time_span genuinely vary across curves,
# which is what the real 480-curve set looks like and what makes
# "no constant shipped feature" a meaningful assertion.
# ---------------------------------------------------------------------------

#: Two plates on purpose: the 80-point / 19-hour plate makes n_points and
#: time_span vary, which is what the full 480-curve set looks like and what
#: makes "no constant shipped feature" a real assertion.
SUBSET_SCENARIOS = [
    ('standard', 6), ('very_clean', 6), ('high_A', 6), ('fast_growth', 6),
    ('short_experiment', 8),
    ('high_noise', 6), ('erratic', 6), ('minimal_growth', 6),
    ('contamination', 6), ('flat_no_growth', 6),
]


@pytest.fixture(scope='module')
def subset(tmp_path_factory):
    if not (GROUND_TRUTH.exists() and DATA_DIR.is_dir()):
        pytest.skip('synthetic test data not present')

    gt = pd.read_csv(GROUND_TRUTH)
    keep = set()
    for scenario, n in SUBSET_SCENARIOS:
        keep |= set(gt.loc[gt['scenario'] == scenario, 'strain_name'].head(n))
    assert keep, 'ground truth carries none of the subset scenarios'

    root = tmp_path_factory.mktemp('wp11_subset')
    data_dir = root / 'DATA'
    data_dir.mkdir()
    n_curves = 0
    n_plates = 0
    for source in sorted(DATA_DIR.glob('*_DATA.csv')):
        df = pd.read_csv(source)
        cols = [df.columns[0]] + [c for c in df.columns[1:]
                                  if any(k in c for k in keep)]
        if len(cols) < 4:
            continue
        df[cols].to_csv(data_dir / source.name, index=False)
        n_curves += len(cols) - 1
        n_plates += 1
    assert n_curves > 20, 'subset should contain a usable number of curves'
    assert n_plates >= 2, 'subset should span more than one plate geometry'
    return {'root': root, 'data_dir': data_dir, 'n_curves': n_curves}


@pytest.fixture(scope='module')
def trained(subset, trainer):
    """One real end-to-end training run, first-peak, into a scratch models dir."""
    models_dir = subset['root'] / 'models'
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / '09_train_classifier.py'),
         '--raw-data-dir', str(subset['data_dir']),
         '--training-features-dir', str(subset['root'] / 'training_features'),
         '--output', str(models_dir),
         '--truncation-mode', 'first_peak',
         '--n-folds', '3'],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, 'MPLBACKEND': 'Agg'},
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return {
        'models_dir': models_dir,
        'config': json.loads((models_dir / 'feature_config.json').read_text()),
        'features_dir': subset['root'] / 'training_features',
        'stdout': proc.stdout,
    }


# ---------------------------------------------------------------------------
# No constant shipped feature
# ---------------------------------------------------------------------------

def _n_bins(model):
    """n_bins_non_missing_ per column for a HistGradientBoosting model."""
    binner = getattr(model, '_bin_mapper', None)
    if binner is None:
        return None
    return list(binner.n_bins_non_missing_)


def test_no_constant_shipped_feature(trained):
    """Every shipped feature must be one the model can actually split on.

    Equivalently: n_bins_non_missing_ >= 2 for every column. The two metadata
    features used to sit here with n_bins_non_missing_ == 1.
    """
    import joblib

    config = trained['config']
    for kind, names_key in (('prefit_gate', 'prefit_feature_names'),
                            ('postfit_classifier', 'postfit_feature_names')):
        model = joblib.load(trained['models_dir'] / f'{kind}.joblib')
        names = config[names_key]
        bins = _n_bins(model)
        assert bins is not None and len(bins) == len(names), kind
        constant = [n for n, b in zip(names, bins) if b < 2]
        assert constant == [], f'{kind} ships constant feature(s): {constant}'

    assert config['constant_features']['postfit_deployment'] == []
    assert config['constant_features']['prefit_deployment'] == []


def test_metadata_features_are_not_model_features():
    from ml_classifier import (
        ALL_POSTFIT_FEATURES, METADATA_FEATURES, PREFIT_FEATURES,
        extract_metadata_features,
    )

    for feature in METADATA_FEATURES:
        assert feature not in PREFIT_FEATURES
        assert feature not in ALL_POSTFIT_FEATURES
    # The extractor itself stays: the KEY/plate path may supply media honestly.
    assert extract_metadata_features('H2O-MAL10')['is_control'] == 1.0


def test_metadata_removal_leaves_predictions_unchanged(trainer, trained):
    """Dropping the two columns is provably inert on the current features.

    Both are NaN for every synthetic row, so a model trained with them and one
    trained without them see the same information; any change in a published
    metric is therefore attributable to the dataset, not to this edit.
    """
    from ml_classifier import ALL_POSTFIT_FEATURES, METADATA_FEATURES

    results = trained['features_dir'] / 'processing_results.csv'
    df = trainer.load_training_data(str(GROUND_TRUTH), str(results))
    fit_df = df[df['fit_attempted']].reset_index(drop=True)

    X_without, y = trainer.prepare_postfit_features(fit_df)
    with_names = list(ALL_POSTFIT_FEATURES) + list(METADATA_FEATURES)
    X_with, _ = trainer.prepare_postfit_features(fit_df, feature_names=with_names)
    assert list(X_with.columns[-2:]) == list(METADATA_FEATURES)
    assert X_with[METADATA_FEATURES].isna().all().all(), (
        'synthetic curve names must not parse as media/concentration'
    )

    model_a, _ = trainer.train_model(X_without, y, 'hgb', n_folds=3)
    model_b, _ = trainer.train_model(X_with, y, 'hgb', n_folds=3)
    p_a = model_a.predict_proba(trainer.clean_feature_matrix(X_without))[:, 1]
    p_b = model_b.predict_proba(trainer.clean_feature_matrix(X_with))[:, 1]
    assert float(np.max(np.abs(p_a - p_b))) < 1e-9


def test_assert_features_present_has_no_metadata_exemption(trainer):
    X = pd.DataFrame({'is_control': [np.nan, np.nan], 'fit_r_squared': [0.9, 0.8]})
    with pytest.raises(ValueError, match='is_control'):
        trainer.assert_features_present(X, 'exempted?')


def test_constant_features_are_reported(trainer):
    X = pd.DataFrame({'flat': [1.0, 1.0, 1.0], 'varies': [1.0, 2.0, 3.0]})
    assert trainer.constant_features(X) == ['flat']


# ---------------------------------------------------------------------------
# Mode-pinned feature cache
# ---------------------------------------------------------------------------

def _fake_cache(tmp_path, mode=None, ml_gated=False, raw_dir=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'strain': ['SYNTHETIC_400pts_99h_CURVE0000_blanked'],
        'fit_r_squared': [0.99],
        'p_good': [0.9 if ml_gated else np.nan],
    }).to_csv(tmp_path / 'processing_results.csv', index=False)
    if mode is not None:
        (tmp_path / 'feature_build.json').write_text(json.dumps({
            'schema': 'feature_build/1',
            'truncation_mode': mode,
            'raw_data_dir': str(raw_dir) if raw_dir else None,
        }))
    return tmp_path


def test_cache_built_in_another_mode_is_rejected(trainer, tmp_path):
    """Acceptance: the refusal names both modes."""
    cache = _fake_cache(tmp_path / 'adaptive_cache', mode='adaptive')
    reusable, reason = trainer.check_feature_cache(
        cache, truncation_mode='first_peak', raw_data_dir=str(tmp_path))
    assert reusable is False
    assert 'adaptive' in reason and 'first_peak' in reason


def test_cache_in_the_same_mode_is_reused(trainer, tmp_path):
    cache = _fake_cache(tmp_path / 'fp_cache', mode='first_peak')
    reusable, reason = trainer.check_feature_cache(
        cache, truncation_mode='first_peak', raw_data_dir=str(tmp_path))
    assert reusable is True, reason


def test_cache_without_a_sidecar_is_unknown_mode(trainer, tmp_path):
    cache = _fake_cache(tmp_path / 'bare_cache', mode=None)
    reusable, reason = trainer.check_feature_cache(
        cache, truncation_mode='first_peak', raw_data_dir=str(tmp_path))
    assert reusable is False
    assert 'UNKNOWN' in reason


def test_ml_gated_cache_is_rejected(trainer, tmp_path):
    cache = _fake_cache(tmp_path / 'gated', mode='first_peak', ml_gated=True)
    reusable, reason = trainer.check_feature_cache(
        cache, truncation_mode='first_peak', raw_data_dir=str(tmp_path))
    assert reusable is False
    assert 'ML gate' in reason


def test_truncation_mode_flags_are_spelled_out(trainer):
    assert trainer.DEFAULT_TRUNCATION_MODE == 'first_peak'
    assert trainer.TRUNCATION_MODE_FLAGS['first_peak'] == ['--no-adaptive', '--first-peak']
    assert trainer.TRUNCATION_MODE_FLAGS['adaptive'] == ['--adaptive']
    assert trainer.resolve_truncation_mode(None, ['--no-adaptive']) == 'first_peak'
    with pytest.raises(ValueError, match='adaptive'):
        trainer.resolve_truncation_mode('first_peak', ['--adaptive'])


def test_feature_build_sidecar_records_the_mode(trained):
    sidecar = json.loads(
        (trained['features_dir'] / 'feature_build.json').read_text())
    assert sidecar['truncation_mode'] == 'first_peak'
    assert sidecar['ml_gated'] is False
    for key in ('extra_args', 'raw_data_dir', 'config_path',
                'truncation_block_sha256', 'analysis_script_git_sha'):
        assert key in sidecar
    # The pipeline's own sidecar (WP9) must agree with what we asked for.
    assert sidecar['pipeline_reported_truncation_mode'] in (None, 'first_peak')


def test_deployed_mode_mismatch_warns_at_load_time(tmp_path):
    import ml_classifier

    assert ml_classifier.check_truncation_mode('first_peak', 'first_peak') is None
    msg = ml_classifier.check_truncation_mode('first_peak', 'adaptive')
    assert msg and 'first_peak' in msg and 'adaptive' in msg

    cfg = tmp_path / 'feature_config.json'
    cfg.write_text(json.dumps({
        'postfit_feature_names': list(ml_classifier.ALL_POSTFIT_FEATURES),
        'truncation_mode': 'first_peak',
    }))
    clf = ml_classifier.PostFitClassifier(
        model_path=str(tmp_path / 'missing.joblib'),
        feature_config_path=str(cfg),
        truncation_mode='adaptive',
    )
    assert clf.trained_truncation_mode == 'first_peak'
    assert 'adaptive' in clf.truncation_mode_warning


def test_model_card_stamps_the_truncation_mode(trained):
    assert trained['config']['truncation_mode'] == 'first_peak'
    assert trained['config']['training_features']['truncation_mode'] == 'first_peak'


# ---------------------------------------------------------------------------
# Labels: scoring mode and the agreement guard
# ---------------------------------------------------------------------------

def test_a_not_identifiable_rows_never_enter_training(trainer):
    df = pd.DataFrame({
        'label': [1, 0, 0],
        'scoring_mode': ['standard', 'a_not_identifiable', 'standard'],
        'scenario': ['s1', 's2', 's3'],
        'fit_attempted': [True, True, True],
    })
    kept, n_dropped = trainer.drop_unscorable_rows(df)
    assert n_dropped == 1
    assert list(kept['scoring_mode']) == ['standard', 'standard']


def test_v1_ground_truth_has_nothing_to_drop(trainer):
    df = pd.DataFrame({'label': [1, 0], 'scenario': ['a', 'b']})
    kept, n_dropped = trainer.drop_unscorable_rows(df)
    assert n_dropped == 0 and len(kept) == 2


def test_label_agreement_baseline_is_per_scenario_not_a_flat_cut(trainer):
    """A BAD scenario in a GOOD-heavy set must not be flagged by prevalence alone.

    ``flat_bad`` below sits in its own region but has GOOD curves interleaved
    through half of it, so its 1-NN agreement is ~0.47 -- under the fixed 60 %
    cut the audit proposed it would be dropped. Its own baseline is ~0.10 (a
    randomly drawn neighbour is BAD only a tenth of the time), which it beats
    comfortably, so the guard leaves it alone.
    """
    rows = []
    for i in range(15):                       # the BAD scenario under test
        rows.append({'scenario': 'flat_bad', 'label': 0,
                     'fit_r_squared': 0.900 + 0.001 * i, 'delta_od': 0.05})
    for x in (0.4, 4.4, 8.4, 12.4):           # GOOD curves interleaved into it
        rows.append({'scenario': 'clean_good', 'label': 1,
                     'fit_r_squared': 0.900 + 0.001 * x, 'delta_od': 0.05})
    for i in range(40):                       # the GOOD bulk, far away
        rows.append({'scenario': 'clean_good', 'label': 1,
                     'fit_r_squared': 0.990 + 0.0001 * i, 'delta_od': 1.30})
    for i in range(5):                        # a second BAD scenario, far away
        rows.append({'scenario': 'other_bad', 'label': 0,
                     'fit_r_squared': 0.200 + 0.001 * i, 'delta_od': 0.01})
    df = pd.DataFrame(rows)
    df['fit_attempted'] = True

    report = trainer.label_agreement_report(df)
    by_name = {r['scenario']: r for r in report['scenarios']}
    flat = by_name['flat_bad']
    assert flat['baseline'] == pytest.approx(5 / 49, abs=0.01)
    assert flat['agreement'] < 0.6, 'the fixture must trip a flat 60% cut'
    assert flat['agreement'] > flat['baseline']
    assert report['flagged_scenarios'] == []


def test_label_agreement_flags_an_overlapping_boundary_scenario(trainer):
    """Two scenarios that occupy the same feature region are unlearnable."""
    rows = []
    for i in range(40):
        rows.append({'scenario': 'clean_good', 'label': 1,
                     'fit_r_squared': 0.995 + 0.0001 * i, 'delta_od': 1.30})
    for i in range(40):
        rows.append({'scenario': 'clean_bad', 'label': 0,
                     'fit_r_squared': 0.200 + 0.0010 * i, 'delta_od': 0.02})
    for i in range(15):                       # strictly interleaved pair
        rows.append({'scenario': 'borderline_r2_good', 'label': 1,
                     'fit_r_squared': 0.9800 + 0.0002 * i, 'delta_od': 0.60})
        rows.append({'scenario': 'borderline_r2_bad', 'label': 0,
                     'fit_r_squared': 0.9801 + 0.0002 * i, 'delta_od': 0.60})
    df = pd.DataFrame(rows)
    df['fit_attempted'] = True
    report = trainer.label_agreement_report(df)
    assert 'borderline_r2_good' in report['flagged_scenarios']
    assert 'borderline_r2_bad' in report['flagged_scenarios']
    assert 'clean_good' not in report['flagged_scenarios']

    kept, decision = trainer.apply_label_agreement_guard(df, report, enabled=True)
    assert set(decision['dropped_scenarios']) == {'borderline_r2_good',
                                                  'borderline_r2_bad'}
    assert decision['n_rows_dropped'] == 30
    assert 'borderline_r2_good' not in set(kept['scenario'])

    kept_off, decision_off = trainer.apply_label_agreement_guard(
        df, report, enabled=False)
    assert len(kept_off) == len(df)
    assert decision_off['dropped_scenarios'] == []


def test_guard_never_empties_the_training_set(trainer):
    rng = np.random.default_rng(2)
    rows = []
    for i in range(10):
        rows.append({'scenario': 'a', 'label': 1,
                     'fit_r_squared': 0.5 + rng.normal(0, 0.3),
                     'delta_od': 0.5 + rng.normal(0, 0.3)})
        rows.append({'scenario': 'b', 'label': 0,
                     'fit_r_squared': 0.5 + rng.normal(0, 0.3),
                     'delta_od': 0.5 + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    df['fit_attempted'] = True
    report = trainer.label_agreement_report(df)
    kept, decision = trainer.apply_label_agreement_guard(df, report, enabled=True)
    if report['flagged_scenarios']:
        assert decision['override'] is not None
        assert len(kept) == len(df)


def test_model_card_records_label_provenance(trained):
    config = trained['config']
    assert 'label_rule_version' in config
    assert 'truth_schema_version' in config
    assert isinstance(config['contributing_scenarios'], dict)
    assert config['contributing_scenarios']
    assert isinstance(config['dropped_scenarios'], list)
    assert 'report' in config['label_agreement']
    assert 'decision' in config['label_agreement']


# ---------------------------------------------------------------------------
# Lab weighting and the threshold sweep
# ---------------------------------------------------------------------------

def test_lab_weights_follow_the_lab_prior_bands(trainer):
    prior = json.loads((SYNTH / 'config' / 'lab_prior.json').read_text())
    fractions = {b['name']: b['fraction'] for b in prior['bands']}
    # Generated set: half full-culture, half dead. The lab is 44.6 % / 30.4 %.
    df = pd.DataFrame({'delta_od': [1.5] * 50 + [0.005] * 50, 'label': [1] * 50 + [0] * 50})
    weights, info = trainer.lab_sample_weights(
        df, SYNTH / 'config' / 'lab_prior.json')
    assert info['applied'] is True
    assert weights is not None and len(weights) == 100
    assert weights.mean() == pytest.approx(1.0)
    ratio = weights[0] / weights[-1]
    assert ratio == pytest.approx(fractions['full'] / fractions['dead'], rel=1e-6)


def test_lab_weights_are_optional(trainer, tmp_path):
    weights, info = trainer.lab_sample_weights(
        pd.DataFrame({'delta_od': [1.0]}), tmp_path / 'missing.json')
    assert weights is None and info['applied'] is False


def test_threshold_sweep_reports_specificity_and_npv(trainer):
    rng = np.random.default_rng(3)
    y = np.array([1] * 60 + [0] * 40)
    probs = np.clip(np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.15, 100), 0, 1)
    sweep = trainer.threshold_sweep(probs, y)
    keys = {'threshold', 'specificity', 'npv', 'recall', 'precision'}
    assert keys <= set(sweep['sweep'][0])
    assert len(sweep['sweep']) == 19
    assert 0.0 < sweep['bad_threshold'] < sweep['good_threshold'] < 1.0
    hit = [r for r in sweep['sweep'] if r['threshold'] == sweep['good_threshold']]
    assert hit and hit[0]['specificity'] >= trainer.TARGET_SPECIFICITY


def test_model_card_carries_the_sweep_and_recommendation(trained):
    config = trained['config']
    sweep = config['threshold_sweep']
    assert sweep['sweep'] and 'specificity' in sweep['sweep'][0]
    assert 'npv' in sweep['sweep'][0]
    rec = config['recommended_thresholds']
    assert rec['good_threshold'] is not None and rec['bad_threshold'] is not None
    assert config['lab_weighting']['train_split']['applied'] in (True, False)


def test_recommended_thresholds_are_reported_but_never_applied(tmp_path):
    """A retrain must not be able to move the deployed GOOD gate.

    The trainer's sweep can recommend e.g. 0.15/0.10; the classifier records
    that recommendation so an operator can see it, but the active gate stays
    the deployed 0.70/0.30 until config.yaml (or a caller) says otherwise.
    Adopting a recommendation is a policy decision, not a side effect of
    retraining.
    """
    import ml_classifier

    cfg = tmp_path / 'feature_config.json'
    cfg.write_text(json.dumps({
        'postfit_feature_names': list(ml_classifier.ALL_POSTFIT_FEATURES),
        'recommended_thresholds': {'good_threshold': 0.15, 'bad_threshold': 0.10},
    }))
    clf = ml_classifier.PostFitClassifier(
        model_path=str(tmp_path / 'missing.joblib'), feature_config_path=str(cfg))
    assert clf.recommended_thresholds == {
        'good_threshold': 0.15, 'bad_threshold': 0.10}
    assert clf.good_threshold == pytest.approx(
        ml_classifier.PostFitClassifier.DEFAULT_GOOD_THRESHOLD) == pytest.approx(0.7)
    assert clf.bad_threshold == pytest.approx(
        ml_classifier.PostFitClassifier.DEFAULT_BAD_THRESHOLD) == pytest.approx(0.3)

    # A caller that yields None (an unset config field, AnalysisSettings
    # forwarding a missing value) also lands on the deployed default, never on
    # the model's recommendation.
    unset = ml_classifier.PostFitClassifier(
        model_path=str(tmp_path / 'missing.joblib'), feature_config_path=str(cfg),
        good_threshold=None, bad_threshold=None)
    assert unset.good_threshold == pytest.approx(0.7)
    assert unset.bad_threshold == pytest.approx(0.3)

    # An explicit argument wins -- the pipeline's config keeps control.
    explicit = ml_classifier.PostFitClassifier(
        model_path=str(tmp_path / 'missing.joblib'), feature_config_path=str(cfg),
        good_threshold=0.8, bad_threshold=0.2)
    assert explicit.good_threshold == pytest.approx(0.8)
    assert explicit.bad_threshold == pytest.approx(0.2)


def test_truth_schema_version_comes_from_the_declared_schema(trainer, tmp_path):
    """v2 truth must be stamped '2', not 'unknown'.

    Contract B's ground truth carries ``label_rule_version`` as a column but no
    ``truth_schema_version`` -- the schema version is a property of the file,
    declared in synthetic_data/src/truth.py -- so the trainer resolves it from
    ``truth.TRUTH_SCHEMA_VERSION`` by matching the declared column set, and
    records how it decided.
    """
    truth = pytest.importorskip('src.truth')

    v2 = tmp_path / 'ground_truth_v2.csv'
    row = {c: '' for c in truth.GROUND_TRUTH_COLUMNS}
    row.update({'curve_id': 'CURVE0001', 'scenario': 'clean_exponential',
                'expected_class': 'GOOD', 'label_rule': 'scenario',
                'label_rule_version': truth.LABEL_RULE_VERSION,
                'scoring_mode': 'standard'})
    pd.DataFrame([row], columns=list(truth.GROUND_TRUTH_COLUMNS)).to_csv(v2, index=False)

    prov = trainer.truth_provenance(str(v2))
    assert prov['truth_schema_version'] == truth.TRUTH_SCHEMA_VERSION == '2'
    assert 'TRUTH_SCHEMA_VERSION' in prov['truth_schema_version_source']
    assert prov['label_rule_version'] == truth.LABEL_RULE_VERSION
    assert prov['label_rule_version_source'] == 'label_rule_version column'

    # v1 (the committed file) is schema 1 with scenario-level labels, and says
    # so without pretending either value was stated in the file.
    v1 = tmp_path / 'ground_truth_v1.csv'
    pd.DataFrame([{c: '' for c in trainer.V1_GROUND_TRUTH_COLUMNS}]).to_csv(v1, index=False)
    prov1 = trainer.truth_provenance(str(v1))
    assert prov1['truth_schema_version'] == '1'
    assert 'v1 column set' in prov1['truth_schema_version_source']
    assert prov1['label_rule_version'] == '0'
    assert 'inferred' in prov1['label_rule_version_source']

    # A frame matching neither schema is honestly unknown.
    other = tmp_path / 'other.csv'
    pd.DataFrame([{'curve_id': 'CURVE0001', 'whatever': 1}]).to_csv(other, index=False)
    assert trainer.truth_provenance(str(other))['truth_schema_version'] == 'unknown'


def test_model_card_records_how_each_version_was_determined(trained):
    config = trained['config']
    assert config['truth_schema_version'] not in (None, '')
    assert config['truth_schema_version_source']
    assert config['label_rule_version_source']
    # The sweep's recommendation is labelled advisory in the card, matching the
    # classifier, which does not apply it.
    assert 'ADVISORY' in config['recommended_thresholds']['status']


def test_prefit_gate_uses_the_models_own_feature_list(tmp_path):
    """A model shipped with an older feature list keeps working."""
    import ml_classifier

    cfg = tmp_path / 'feature_config.json'
    legacy = list(ml_classifier.PREFIT_FEATURES) + list(ml_classifier.METADATA_FEATURES)
    cfg.write_text(json.dumps({'prefit_feature_names': legacy}))
    gate = ml_classifier.PreFitGate(
        model_path=str(tmp_path / 'missing.joblib'), feature_config_path=str(cfg))
    assert gate.feature_names == legacy
