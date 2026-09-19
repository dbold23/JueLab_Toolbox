"""Integrity properties of the synthetic benchmark.

Why this file exists
--------------------
Every number the synthetic suite publishes used to rest on properties nothing
asserted.  No test imported ``PipelineValidator``; nothing checked that the
ground truth covered ``ALL_SCENARIOS``, the curve count, the confusion-matrix
arithmetic, the report schema, or that regenerating twice gave the same data.
That is how the catalog drifted to 37 scenarios while only 32 were generated --
and how ``test_readme_tables_list_the_real_scenarios`` was dutifully updated to
37 without anyone noticing the data had not moved.  It is also how the two
silent-failure modes survived: a benchmark that publishes ``accuracy 0.0``
because the join key was renamed, and one that publishes ``accuracy 0.900`` over
21 % of the suite, both exiting 0.

What this file may and may not pin
----------------------------------
It pins **schema, arithmetic, determinism, provenance and internal
consistency**:

* the committed report is reproduced from the committed results, to 6 dp,
  by re-running the scorer -- no accuracy is typed into this file;
* the confusion-matrix arithmetic is checked on a hand-built six-row fixture
  with a known TP/TN/FP/FN;
* the ground truth covers every defined scenario and matches its manifest.

It must **not** pin an accuracy, coverage, KS or recovery value that a future
change would be tempted to flatter.  The one measured comparison here (the
residual-autocorrelation KS statistic) is expressed against the frozen v1
baseline and a ceiling far above the measurement, in the direction of *more*
realism, and carries the re-baseline note the plan requires.

Data this file reads
--------------------
``synthetic_data/benchmark/v1/`` and ``synthetic_data/benchmark/v2/`` -- the two
frozen, fully tracked snapshots.  Nothing here reads a run directory, so the
whole file keeps working after ``make clean``.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH = REPO_ROOT / 'synthetic_data'
SCRIPTS = REPO_ROOT / 'scripts'
BENCH_V1 = SYNTH / 'benchmark' / 'v1'
BENCH_V2 = SYNTH / 'benchmark' / 'v2'
COMPREHENSIVE = SYNTH / 'output' / 'comprehensive_test'
GROUND_TRUTH = COMPREHENSIVE / 'test_data' / 'ground_truth.csv'
DATA_DIR = COMPREHENSIVE / 'test_data' / 'DATA'
MANIFEST = COMPREHENSIVE / 'MANIFEST.json'
REAL_RESULT_GLOB = 'results/tables/Group*_Results/processing_results.csv'

# Repo convention: scripts put synthetic_data/ on sys.path and import
# ``from src...`` / ``from validation...``.  There is no top-level package.
for path in (str(SYNTH), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Fixtures (session scoped: this file must stay a few seconds, not minutes)
# ---------------------------------------------------------------------------

def _require(path: Path):
    if not path.exists():
        pytest.skip(f'{path.relative_to(REPO_ROOT)} is not in the tree')
    return path


@pytest.fixture(scope='session')
def report_v2():
    return json.loads(_require(BENCH_V2 / 'report.json').read_text())


@pytest.fixture(scope='session')
def results_v2():
    return pd.read_csv(_require(BENCH_V2 / 'processing_results.csv'))


@pytest.fixture(scope='session')
def ground_truth():
    return pd.read_csv(_require(GROUND_TRUTH))


@pytest.fixture(scope='session')
def manifest():
    return json.loads(_require(MANIFEST).read_text())


@pytest.fixture(scope='session')
def rescored(tmp_path_factory):
    """The committed v2 results scored again, here and now.

    This is the reproduction test's engine and it is shared: scoring 1139
    curves costs ~2.5 s and several tests want the result.
    """
    from validation.benchmark import run_benchmark

    _require(BENCH_V2 / 'processing_results.csv')
    _require(GROUND_TRUTH)
    out = tmp_path_factory.mktemp('rescored')
    settings = BENCH_V2 / 'run_settings.json'
    return run_benchmark(
        BENCH_V2 / 'processing_results.csv',
        GROUND_TRUTH,
        out,
        truncation_mode='first_peak',
        run_settings_path=settings if settings.exists() else None,
        figures=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# The scenario catalog and the data must be the same suite
# ---------------------------------------------------------------------------

def test_ground_truth_covers_every_defined_scenario(ground_truth):
    """ALL_SCENARIOS and the committed ground truth name the same scenarios.

    This is the test whose absence let the catalog reach 37 scenarios while the
    data held 32.  It fails in BOTH directions on purpose: adding a scenario to
    ``ALL_SCENARIOS`` without regenerating breaks it, and so does a stale
    scenario left in the data after it was deleted from the catalog.
    """
    from src.curve_scenarios import ALL_SCENARIOS

    defined = set(ALL_SCENARIOS)
    generated = set(ground_truth['scenario'].unique())
    assert defined == generated, (
        'the scenario catalog and the committed ground truth disagree; '
        f'defined but never generated: {sorted(defined - generated)}; '
        f'generated but no longer defined: {sorted(generated - defined)}. '
        'Regenerate: cd synthetic_data && MPLBACKEND=Agg python '
        'scripts/generate_comprehensive.py -o output/comprehensive_test --seed 42'
    )


def test_a_label_may_only_leave_the_catalog_through_the_oracle(ground_truth):
    """expected_class stays binary, and only ``label_rule='oracle'`` may move it.

    The suite is allowed to disagree with its own catalog -- that is the whole
    point of the oracle rule, which labels the curve that was actually written
    rather than the curve that was requested.  What it is not allowed to do is
    disagree silently: every row whose label differs from its scenario's
    declared class must say ``label_rule='oracle'``, and every ``'scenario'``
    row must carry the declared label exactly.
    """
    from src.curve_scenarios import ALL_SCENARIOS

    assert set(ground_truth['expected_class']) <= {'GOOD', 'BAD'}
    declared = ground_truth['scenario'].map(
        lambda name: ALL_SCENARIOS[name].expected_class)
    differs = ground_truth['expected_class'] != declared
    rule = ground_truth['label_rule'].astype(str)

    assert not (differs & (rule == 'scenario')).any(), (
        'a scenario-labelled curve disagrees with the catalog: '
        f'{ground_truth.loc[differs & (rule == "scenario"), "curve_id"].tolist()[:10]}')
    assert set(rule[differs]) <= {'oracle'}


def test_manifest_matches_committed_data(manifest, ground_truth):
    """MANIFEST.json describes the files that are actually in the tree.

    Digest, curve count, scenario count and the GOOD/BAD split.  A regenerated
    dataset committed without its manifest -- or a manifest committed without
    its dataset -- fails here rather than in a report three steps later.
    """
    root = MANIFEST.parent
    for rel, digest in manifest['files_sha256'].items():
        path = root / rel
        assert path.exists(), f'MANIFEST.json lists a missing file: {rel}'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f'{rel} does not match its manifest digest'

    plates = sorted(p.name for p in DATA_DIR.glob('SYNTHETIC_*pts_*_DATA.csv'))
    listed = sorted(Path(rel).name for rel in manifest['files_sha256']
                    if rel.startswith('test_data/DATA/'))
    assert plates == listed, 'DATA/ holds plates the manifest does not list'

    assert manifest['n_curves_written'] == len(ground_truth)
    assert manifest['n_scenarios'] == ground_truth['scenario'].nunique()
    good = int((ground_truth['expected_class'] == 'GOOD').sum())
    assert manifest['expected_good'] == good
    assert manifest['expected_bad'] == len(ground_truth) - good
    assert manifest['seed'] == 42
    assert manifest['truncation_mode'] in ('first_peak', 'adaptive', 'global_max')

    # Requested vs written.  n_per_scenario counts REQUESTS; a replicate
    # scenario turns one request into several wells, so the two counts differ
    # and a reader who sums n_per_scenario under-reports the suite (step 9
    # announced '1057 synthetic curves' for 1139).  Both are recorded, and the
    # relationship between them is asserted here.
    assert sum(manifest['n_per_scenario'].values()) == manifest['n_curves_requested']
    assert manifest['n_curves_written'] >= manifest['n_curves_requested']
    assert set(manifest['n_per_scenario']) == set(ground_truth['scenario'].unique())


def test_every_plate_name_encodes_its_grid(ground_truth):
    """The plate file names carry points/duration/resolution, not just points.

    ``SYNTHETIC_400pts_99h_DATA.csv`` (the v1 name) silently bucketed curves of
    different time resolutions into one plate.  The v2 names carry the
    resolution too, and every name must match the curves inside it.
    """
    pattern = re.compile(r'^SYNTHETIC_(\d+)pts_(\d+)h_(\d+)m_DATA\.csv$')
    plates = sorted(DATA_DIR.glob('SYNTHETIC_*pts_*_DATA.csv'))
    assert plates, 'no synthetic plates in the tree'
    for plate in plates:
        hit = pattern.match(plate.name)
        assert hit, f'{plate.name} does not carry pts/hours/resolution'
        n_points = int(hit.group(1))
        resolution_h = int(hit.group(3)) / 60.0
        frame = pd.read_csv(plate, nrows=n_points + 5)
        assert len(frame) == n_points, f'{plate.name} holds {len(frame)} rows'
        times = frame.iloc[:, 0].to_numpy(dtype=float)
        step_min = float(np.median(np.diff(times))) * 60.0
        # The name carries the nominal reader interval; the written grid is
        # duration/(n-1), which rounds to it (100 h over 1205 points is 4.98 min).
        assert round(step_min) == round(resolution_h * 60), (
            f'{plate.name} says {hit.group(3)} min but steps {step_min:.3f} min'
        )
        span = float(times[-1] - times[0])
        assert abs(span - float(hit.group(2))) <= 1.0, (
            f'{plate.name} says {hit.group(2)} h but spans {span:.3f} h')


# ---------------------------------------------------------------------------
# Determinism and the forward-seeding guarantee
# ---------------------------------------------------------------------------

def _fingerprint(frame: pd.DataFrame) -> str:
    """A byte-level fingerprint of a generated suite: truth + OD vectors."""
    meta = frame.drop(columns=[c for c in ('time', 'od600') if c in frame.columns])
    digest = hashlib.sha256(meta.to_csv(index=False).encode('utf-8'))
    for column in ('time', 'od600'):
        if column in frame.columns:
            for value in frame[column]:
                digest.update(np.asarray(value, dtype=float).tobytes())
    return digest.hexdigest()


@pytest.fixture(scope='session')
def two_scenario_suite():
    from src.data_generator import SyntheticGrowthCurveGenerator

    config = {'standard': 3, 'flat_no_growth': 3}
    return SyntheticGrowthCurveGenerator().generate_comprehensive_test_set(
        config, seed=42, verbose=False)


def test_regeneration_is_deterministic(two_scenario_suite):
    """Two scenarios x three curves, seed 42, generated twice, identical.

    Byte-level: the ground-truth columns AND the OD vectors.  Before the
    seeding fix the second run differed, which is why ``--seed 0`` used to be
    non-reproducible.
    """
    from src.data_generator import SyntheticGrowthCurveGenerator

    again = SyntheticGrowthCurveGenerator().generate_comprehensive_test_set(
        {'standard': 3, 'flat_no_growth': 3}, seed=42, verbose=False)
    assert _fingerprint(again) == _fingerprint(two_scenario_suite)


def test_appending_a_scenario_does_not_move_existing_curves(two_scenario_suite):
    """The forward-seeding guarantee (interface: curve_seed).

    A per-curve seed derived from ``seed:scenario:index`` means appending a
    scenario, or changing ``--n-per-scenario``, cannot rewrite a curve that was
    already published.  Under the old ``seed + curve_id`` scheme every curve
    after the insertion point moved, so no benchmark number survived a catalog
    edit.
    """
    from src.data_generator import SyntheticGrowthCurveGenerator

    bigger = SyntheticGrowthCurveGenerator().generate_comprehensive_test_set(
        {'standard': 3, 'flat_no_growth': 3, 'high_noise': 3},
        seed=42, verbose=False)
    kept = bigger[bigger['scenario'].isin(['standard', 'flat_no_growth'])]
    kept = kept.reset_index(drop=True)

    for column in ('true_A', 'true_mu', 'true_lambda', 'expected_class'):
        pd.testing.assert_series_equal(
            kept[column], two_scenario_suite[column], check_names=False)
    for before, after in zip(two_scenario_suite['od600'], kept['od600']):
        assert np.array_equal(np.asarray(before, dtype=float),
                              np.asarray(after, dtype=float))


def test_a_different_scenario_count_does_not_move_the_kept_prefix():
    """Shrinking a scenario keeps the surviving prefix identical."""
    from src.data_generator import SyntheticGrowthCurveGenerator

    five = SyntheticGrowthCurveGenerator().generate_comprehensive_test_set(
        {'standard': 5}, seed=42, verbose=False)
    two = SyntheticGrowthCurveGenerator().generate_comprehensive_test_set(
        {'standard': 2}, seed=42, verbose=False)
    for before, after in zip(five['od600'][:2], two['od600']):
        assert np.array_equal(np.asarray(before, dtype=float),
                              np.asarray(after, dtype=float))


# ---------------------------------------------------------------------------
# The arithmetic, on a fixture whose answer is known by hand
# ---------------------------------------------------------------------------

def test_classification_metrics_on_a_hand_built_fixture():
    """Six rows with a hand-counted TP/TN/FP/FN = 2/2/1/1.

    Nothing about the real suite is asserted here; this is the scorer's
    arithmetic, checked against numbers a reader can verify in their head.
    Accuracy 4/6, precision 2/3, recall 2/3, specificity 2/3, balanced 2/3.
    """
    from validation.pipeline_validator import classification_metrics

    matched = pd.DataFrame({
        'curve_id': [f'CURVE{i:04d}' for i in range(1, 7)],
        'scenario': ['a', 'a', 'b', 'b', 'c', 'c'],
        'expected_class': ['GOOD', 'GOOD', 'GOOD', 'BAD', 'BAD', 'BAD'],
        'gt_is_good': [True, True, True, False, False, False],
        'pred_is_good': [True, True, False, True, False, False],
        'scoring_mode': ['standard'] * 6,
    })
    m = classification_metrics(matched)

    assert m['confusion_matrix'] == {
        'true_positives': 2, 'true_negatives': 2,
        'false_positives': 1, 'false_negatives': 1,
    }
    assert m['n'] == 6
    assert m['accuracy'] == pytest.approx(4 / 6)
    assert m['precision'] == pytest.approx(2 / 3)
    assert m['recall'] == pytest.approx(2 / 3)
    assert m['sensitivity'] == pytest.approx(2 / 3)
    assert m['specificity'] == pytest.approx(2 / 3)
    assert m['balanced_accuracy'] == pytest.approx(2 / 3)
    assert m['prevalence'] == pytest.approx(0.5)
    assert m['npv'] == pytest.approx(2 / 3)
    assert m['f1_score'] == pytest.approx(2 / 3)
    assert m['mcc'] == pytest.approx(1 / 3)


def test_a_perfect_and_an_inverted_fixture_score_as_they_must():
    """The two degenerate corners, so a sign error cannot hide in the middle."""
    from validation.pipeline_validator import classification_metrics

    base = pd.DataFrame({
        'curve_id': [f'CURVE{i:04d}' for i in range(1, 5)],
        'scenario': ['a'] * 4,
        'expected_class': ['GOOD', 'GOOD', 'BAD', 'BAD'],
        'gt_is_good': [True, True, False, False],
        'scoring_mode': ['standard'] * 4,
    })
    perfect = base.assign(pred_is_good=[True, True, False, False])
    inverted = base.assign(pred_is_good=[False, False, True, True])

    good = classification_metrics(perfect)
    assert good['accuracy'] == 1.0 and good['mcc'] == pytest.approx(1.0)
    assert good['specificity'] == 1.0 and good['sensitivity'] == 1.0

    bad = classification_metrics(inverted)
    assert bad['accuracy'] == 0.0 and bad['mcc'] == pytest.approx(-1.0)
    assert bad['confusion_matrix']['true_positives'] == 0


# ---------------------------------------------------------------------------
# The report: schema, provenance, and reproduction from the committed results
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL = ('schema_version', 'run', 'summary', 'classification',
                      'parameter_recovery', 'uncertainty', 'truncation',
                      'failure_analysis', 'headline')
REQUIRED_RUN_KEYS = ('truncation_mode', 'seed', 'git_sha', 'dataset_sha256',
                     'config_sha256', 'n_curves', 'n_scenarios', 'ml_classify',
                     'timestamp_utc', 'provenance')
REQUIRED_SUMMARY_KEYS = ('n_ground_truth', 'n_pipeline_results', 'n_matched',
                         'n_unmatched_ground_truth', 'n_unmatched_results',
                         'n_duplicate_ids', 'coverage', 'strict')
REQUIRED_CLASSIFICATION_KEYS = ('balanced_accuracy', 'sensitivity',
                                'specificity', 'accuracy', 'prevalence',
                                'precision', 'recall', 'npv', 'f1_score',
                                'mcc', 'n', 'confusion_matrix', 'per_scenario')


def test_report_schema_has_required_keys(report_v2):
    """The published report carries every block a reader is told to expect."""
    for key in REQUIRED_TOP_LEVEL:
        assert key in report_v2, f'report.json has no {key!r} block'
    for key in REQUIRED_RUN_KEYS:
        assert key in report_v2['run'], f'run block has no {key!r}'
    for key in REQUIRED_SUMMARY_KEYS:
        assert key in report_v2['summary'], f'summary block has no {key!r}'
    for key in REQUIRED_CLASSIFICATION_KEYS:
        assert key in report_v2['classification'], f'classification has no {key!r}'
    cm = report_v2['classification']['confusion_matrix']
    assert set(cm) == {'true_positives', 'true_negatives',
                       'false_positives', 'false_negatives'}


def test_report_names_its_truncation_mode_seed_dataset_and_commit(report_v2):
    """Every published number names the run that produced it.

    Truncation mode, seed, dataset digest, config digest and git SHA -- the
    five things without which a benchmark number cannot be reproduced or
    compared with another release.
    """
    run = report_v2['run']
    assert run['truncation_mode'] in ('first_peak', 'adaptive', 'global_max')
    assert isinstance(run['seed'], int)
    assert re.fullmatch(r'[0-9a-f]{40}', str(run['git_sha']))
    assert re.fullmatch(r'[0-9a-f]{64}', str(run['dataset_sha256']))
    assert re.fullmatch(r'[0-9a-f]{64}', str(run['config_sha256']))
    assert run['provenance'] in ('sidecar', 'declared')
    assert str(run['truncation_mode']) in report_v2['headline']


def test_report_dataset_digest_is_the_committed_suite(report_v2, manifest):
    """The report scored the dataset that is in the tree, not another one."""
    from validation.benchmark import dataset_sha256

    assert report_v2['run']['dataset_sha256'] == dataset_sha256(DATA_DIR)
    assert report_v2['run']['n_curves'] == manifest['n_curves_written']
    assert report_v2['run']['n_scenarios'] == manifest['n_scenarios']


def test_confusion_matrix_arithmetic_is_self_consistent(report_v2):
    """Accuracy, precision, recall and specificity agree with the matrix.

    A hand-edited headline cannot survive this, and neither can a metric
    computed over a different subset than the matrix it is printed beside.
    """
    for block in (report_v2['classification'],
                  report_v2['classification'].get('core32')):
        if not block:
            continue
        cm = block['confusion_matrix']
        tp, tn = cm['true_positives'], cm['true_negatives']
        fp, fn = cm['false_positives'], cm['false_negatives']
        assert tp + tn + fp + fn == block['n']
        assert block['accuracy'] == pytest.approx((tp + tn) / block['n'])
        assert block['precision'] == pytest.approx(tp / (tp + fp)) if tp + fp else True
        assert block['recall'] == pytest.approx(tp / (tp + fn))
        assert block['specificity'] == pytest.approx(tn / (tn + fp))
        assert block['balanced_accuracy'] == pytest.approx(
            0.5 * (tp / (tp + fn) + tn / (tn + fp)))


def test_committed_report_matches_committed_results(report_v2, rescored):
    """Re-scoring the committed results reproduces the committed report, 6 dp.

    No accuracy is written down here: the assertion is that the artefact in the
    tree is what today's scorer produces from the results in the tree.  It
    fails if the report was hand-edited, if it was written from a different
    results file, or if the scorer's arithmetic changes without the report
    being regenerated.
    """
    for block in ('classification', 'summary'):
        committed = report_v2[block]
        fresh = rescored[block]
        for key, value in committed.items():
            if key not in fresh or isinstance(value, (dict, list)):
                continue
            if isinstance(value, float):
                assert round(value, 6) == round(fresh[key], 6), f'{block}.{key}'
            elif isinstance(value, (int, bool, str)):
                assert value == fresh[key], f'{block}.{key}'

    assert (report_v2['classification']['confusion_matrix']
            == rescored['classification']['confusion_matrix'])
    assert (report_v2['classification']['core32']['confusion_matrix']
            == rescored['classification']['core32']['confusion_matrix'])
    for scenario, entry in report_v2['classification']['per_scenario'].items():
        fresh = rescored['classification']['per_scenario'][scenario]
        assert entry['n'] == fresh['n'], scenario
        assert entry['false_positives'] == fresh['false_positives'], scenario
        assert entry['false_negatives'] == fresh['false_negatives'], scenario


def test_headline_json_agrees_with_the_report(report_v2):
    """HEADLINE.json is a copy, so it must never disagree with report.json."""
    headline_path = BENCH_V2 / 'HEADLINE.json'
    headline = json.loads(_require(headline_path).read_text())
    published = headline['classification_headline']
    for key, value in published.items():
        if key in report_v2['classification'] and isinstance(value, (int, float)):
            assert value == pytest.approx(report_v2['classification'][key]), key
    assert headline['seed'] == report_v2['run']['seed']
    assert headline['truncation_mode'] == report_v2['run']['truncation_mode']
    assert headline['dataset_sha256'] == report_v2['run']['dataset_sha256']
    assert headline['git_sha'] == report_v2['run']['git_sha']


def test_the_report_states_a_complete_match(report_v2):
    """Coverage 1.0, nothing unmatched, nothing duplicated, strict on.

    The counters exist so that a partial run is visible in an old report file,
    not only as a run-time exception.  A published report must show a full
    match; a partial one is legal but must be labelled, which is what
    ``strict``/``coverage`` do.
    """
    summary = report_v2['summary']
    assert summary['coverage'] == 1.0
    assert summary['n_unmatched_ground_truth'] == 0
    assert summary['n_unmatched_results'] == 0
    assert summary['n_duplicate_ids'] == 0
    assert summary['strict'] is True
    assert summary['n_matched'] == summary['n_ground_truth']


# ---------------------------------------------------------------------------
# Matching: the two reproduced silent failures, and the honest cases
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def small_pair(ground_truth, results_v2):
    """A 40-curve slice of the committed pair, matched 1:1."""
    from validation.matching import extract_curve_id

    keep = set(ground_truth['strain_name'].map(extract_curve_id).head(40))
    gt = ground_truth[ground_truth['strain_name'].map(extract_curve_id).isin(keep)]
    pr = results_v2[results_v2['strain'].map(extract_curve_id).isin(keep)]
    assert len(gt) == len(pr) == 40
    return gt.reset_index(drop=True), pr.reset_index(drop=True)


def test_match_accepts_the_honest_case(small_pair):
    from validation.matching import match_results

    gt, pr = small_pair
    merged = match_results(gt, pr, strict=True)
    assert len(merged) == len(gt)
    assert merged.attrs['coverage'] == 1.0
    assert merged.attrs['n_unmatched_ground_truth'] == 0
    assert set(merged['gt_is_good'].unique()) <= {True, False}
    assert set(merged['pred_is_good'].unique()) <= {True, False}
    assert merged['curve_id'].str.fullmatch(r'CURVE\d+').all()


def test_match_raises_on_missing_and_duplicate_ids(small_pair):
    """The two failure modes that used to exit 0 with a published number.

    * renaming ``CURVE`` to ``WELL`` matched nothing and published
      ``accuracy 0.0``;
    * truncating the results to their first 100 rows published ``accuracy
      0.900`` over 21 % of the benchmark.

    Both are now :class:`BenchmarkMatchError`, and a duplicated id -- which
    silently doubles a curve's weight in the headline -- is one too.
    """
    from validation.matching import BenchmarkMatchError, match_results

    gt, pr = small_pair

    renamed = pr.copy()
    renamed['strain'] = renamed['strain'].str.replace('CURVE', 'WELL', regex=False)
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(gt, renamed, strict=True)
    assert excinfo.value.unmatched_ground_truth

    truncated = pr.head(10)
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(gt, truncated, strict=True)
    assert len(excinfo.value.unmatched_ground_truth) == 30

    duplicated = pd.concat([pr, pr.head(1)], ignore_index=True)
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(gt, duplicated, strict=True)
    assert excinfo.value.duplicate_ids


def test_a_partial_match_is_labelled_rather_than_hidden(small_pair):
    """``strict=False`` is allowed, but the frame carries the evidence."""
    from validation.matching import match_results

    gt, pr = small_pair
    merged = match_results(gt, pr.head(10), strict=False)
    assert len(merged) == 10
    assert merged.attrs['coverage'] == pytest.approx(0.25)
    assert merged.attrs['n_unmatched_ground_truth'] == 30
    assert len(merged.attrs['unmatched_ground_truth_ids']) == 30


def test_a_blank_verdict_is_undecided_not_bad(small_pair):
    """An empty ``is_good`` cell must not be scored as a confident BAD."""
    from validation.matching import BenchmarkMatchError, match_results

    gt, pr = small_pair
    blanked = pr.copy()
    blanked['is_good'] = blanked['is_good'].astype(object)
    blanked.loc[blanked.index[:3], 'is_good'] = np.nan

    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(gt, blanked, strict=True)
    assert len(excinfo.value.undecided_ids) == 3

    lenient = match_results(gt, blanked, strict=False)
    assert lenient.attrs['n_undecided'] == 3
    assert len(lenient.attrs['undecided_ids']) == 3


# ---------------------------------------------------------------------------
# Gate vocabulary: a reworded message must break CI, not empty a table
# ---------------------------------------------------------------------------

def test_every_reason_maps_to_a_known_gate(results_v2):
    """Every committed ``classification_reason`` parses into known gates.

    The attribution table is built by parsing these sentences on older CSVs.
    When the pipeline rewords a message and nothing notices, the table does not
    error -- it silently empties.  Parsing strictly here turns that into a
    failure with the offending sentence in the message.
    """
    from validation.pipeline_validator import parse_reason_gates

    unmapped = []
    for reason in results_v2['classification_reason'].dropna().unique():
        try:
            parse_reason_gates(reason, strict=True)
        except Exception as exc:                       # GateParseError
            unmapped.append(f'{reason!r}: {exc}')
    assert not unmapped, 'unmapped classification_reason(s):\n' + '\n'.join(unmapped[:5])


def test_real_plate_reasons_map_to_known_gates_too():
    """The same vocabulary must cover the four project plates."""
    from validation.pipeline_validator import parse_reason_gates

    frames = [pd.read_csv(p) for p in sorted(REPO_ROOT.glob(REAL_RESULT_GLOB))]
    if not frames:
        pytest.skip('committed real-plate results not present')
    for frame in frames:
        for reason in frame['classification_reason'].dropna().unique():
            parse_reason_gates(reason, strict=True)


def test_failed_gates_column_uses_the_canonical_vocabulary(results_v2):
    """WP9's ``failed_gates`` is a controlled vocabulary, and it is last."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location('analysis', SCRIPTS / '01_growth_curve_analysis.py')
    analysis = module_from_spec(spec)
    spec.loader.exec_module(analysis)

    assert results_v2.columns[-1] == 'failed_gates'
    known = set(analysis.KNOWN_GATE_IDS)
    seen = set()
    for cell in results_v2['failed_gates'].fillna(''):
        seen.update(g for g in str(cell).split(';') if g)
    assert seen <= known, f'unknown gate ids in failed_gates: {sorted(seen - known)}'

    good = results_v2[results_v2['is_good'].astype(bool)]
    assert (good['failed_gates'].fillna('') == '').all(), (
        'a GOOD curve names a failed gate')
    bad = results_v2[~results_v2['is_good'].astype(bool)]
    assert (bad['failed_gates'].fillna('').str.len() > 0).all(), (
        'a BAD curve names no failed gate')


def test_gate_attribution_covers_every_misclassified_curve(report_v2):
    """The attribution table accounts for the false negatives it reports."""
    failure = report_v2['failure_analysis']
    assert (failure['n_false_negatives_unfitted']
            + failure['n_false_negatives_gated']) == failure['n_false_negatives']
    source = failure['gate_source']
    counts = source if isinstance(source, dict) else {source: 1}
    assert set(counts) <= {'failed_gates', 'classification_reason'}, source
    assert 'failed_gates' in counts, (
        'the attribution fell back to parsing sentences even though the '
        'results carry WP9 failed_gates')


# ---------------------------------------------------------------------------
# Truth-side properties
# ---------------------------------------------------------------------------

def test_no_sentinel_r_squared(ground_truth):
    """0.0 and -1.0 are values, not 'not measured'.

    A curve with no defined fit quality must say so with ``metrics_defined``
    False and a blank number, so that no mean is ever taken over a sentinel.
    """
    r2 = pd.to_numeric(ground_truth['actual_r_squared'], errors='coerce')
    defined = ground_truth['metrics_defined'].astype(bool)
    sentinel = defined & r2.isin([0.0, -1.0])
    assert not sentinel.any(), (
        'rows with metrics_defined=True carry a sentinel actual_r_squared: '
        f'{ground_truth.loc[sentinel, "curve_id"].tolist()[:10]}')
    assert r2[~defined].isna().all(), (
        'metrics_defined=False rows must carry a blank actual_r_squared')


def test_borderline_scenarios_sit_near_their_gate(ground_truth):
    """A 'borderline' scenario must actually straddle the gate it names.

    Read against the pipeline's own thresholds, never against a hard-coded
    number here.  This is the test that catches the v1 defect: the shipped
    ``borderline_delta_od`` curves had delta-OD 0.32-0.58 against a 0.15 gate,
    i.e. up to four times the threshold they were supposed to sit beside, so
    the scenario measured nothing.
    """
    from tecan_growth.settings import AnalysisSettings

    gates = AnalysisSettings.from_config().to_dict()
    min_r2 = float(gates['min_r_squared'])
    min_delta = float(gates['min_absolute_delta_od'])

    delta = ground_truth.loc[ground_truth['scenario'] == 'borderline_delta_od',
                             'delta_od'].astype(float)
    assert len(delta) > 0
    assert (delta >= min_delta).all(), 'borderline_delta_od dips under the gate'
    assert (delta <= 2 * min_delta).all(), (
        f'borderline_delta_od reaches {delta.max():.3f}, more than twice the '
        f'{min_delta} gate it is named after')

    target = ground_truth.loc[ground_truth['scenario'] == 'borderline_r2_good',
                              'target_r_squared'].astype(float)
    assert (target >= min_r2).all() and (target <= min_r2 + 0.03).all()
    target = ground_truth.loc[ground_truth['scenario'] == 'borderline_r2_bad',
                              'target_r_squared'].astype(float)
    assert (target < min_r2).all() and (target >= min_r2 - 0.03).all()


def test_truth_schema_is_the_declared_one(ground_truth):
    """The committed ground truth is exactly truth.GROUND_TRUTH_COLUMNS."""
    from src import truth

    assert list(ground_truth.columns) == list(truth.GROUND_TRUTH_COLUMNS)
    assert truth.TRUTH_SCHEMA_VERSION == '2'
    assert set(ground_truth['label_rule_version'].astype(str)) == {
        str(truth.LABEL_RULE_VERSION)}
    assert set(ground_truth['label_rule']) <= {'scenario', 'oracle'}
    assert set(ground_truth['scoring_mode']) <= {
        'standard', 'a_not_identifiable', 'preprocessing_blind',
        'model_mismatch'}


# ---------------------------------------------------------------------------
# Uncertainty: presence and type only, never a value
# ---------------------------------------------------------------------------

def test_uncertainty_coverage_is_reported(report_v2):
    """The calibration block exists and is typed -- no coverage is pinned.

    Deliberately value-free.  A test asserting ``coverage_95 >= 0.9`` would be
    a ratchet: the cheapest way to pass it is to widen the reported error bars,
    which is exactly the wrong incentive.  What CI defends is that the numbers
    are reported at all, with the denominators beside them.
    """
    block = report_v2['uncertainty']
    assert block['nominal_coverage_68'] == pytest.approx(0.68)
    assert block['nominal_coverage_95'] == pytest.approx(0.95)
    assert isinstance(block['n'], int)

    for parameter, entry in block['parameters'].items():
        assert parameter in ('A', 'mu', 'lambda'), parameter
        for key in ('n', 'n_finite_se', 'n_non_finite_se'):
            assert isinstance(entry[key], int), f'{parameter}.{key}'
        for key in ('coverage_68', 'coverage_95', 'median_abs_z',
                    'median_signed_error', 'centered_coverage_95', 'se_ratio'):
            assert key in entry, f'{parameter} has no {key}'
            value = entry[key]
            assert value is None or isinstance(value, float), f'{parameter}.{key}'
        assert entry['n_finite_se'] + entry['n_non_finite_se'] == entry['n']


def test_parameter_recovery_reports_every_stratum_with_its_n(report_v2):
    """Four strata, each with its own n -- presence and typing only."""
    recovery = report_v2['parameter_recovery']
    for stratum in ('accepted_good', 'all_fitted_good', 'rejected_good',
                    'accepted_bad'):
        assert stratum in recovery, stratum
        assert isinstance(recovery[stratum]['n'], int), stratum
    assert 'tolerances' in recovery and 'true_source' in recovery


# ---------------------------------------------------------------------------
# Calibration against the real plates (a ceiling, in the honest direction)
# ---------------------------------------------------------------------------

def test_synthetic_residual_autocorr_is_closer_to_the_plates_than_v1():
    """The synthetic residuals must not drift back to white noise.

    Measured with a two-sample KS statistic between the pipeline's
    ``residual_autocorr`` on the synthetic suite and on the 92 real conditions:

        v1 (white noise, pre-rebuild)   KS = 0.5923   (n = 452 vs 73)
        v2 (coloured noise, this suite) KS = 0.3810   (n = 1036 vs 73)

    Two assertions, both against frozen committed files: v2 is closer to the
    plates than v1, and it is under a deliberately loose 0.50 ceiling.  The
    ceiling pushes toward *more* realism, never toward a flattering headline --
    a suite made easier by removing structure from its residuals fails it.

    RE-BASELINE POLICY: raising the ceiling requires a written justification in
    docs/BENCHMARK_LEDGER.md naming the measurement and the reason.  Lowering
    it after a genuine improvement is always allowed.
    """
    from scipy import stats

    KS_CEILING = 0.50

    real_paths = sorted(REPO_ROOT.glob(REAL_RESULT_GLOB))
    if not real_paths:
        pytest.skip('committed real-plate results not present')
    real = pd.concat([pd.read_csv(p) for p in real_paths], ignore_index=True)
    real_values = pd.to_numeric(real['residual_autocorr'], errors='coerce').dropna()

    def _ks(path: Path) -> float:
        frame = pd.read_csv(_require(path))
        values = pd.to_numeric(frame['residual_autocorr'], errors='coerce').dropna()
        return float(stats.ks_2samp(values.to_numpy(), real_values.to_numpy()).statistic)

    ks_v1 = _ks(BENCH_V1 / 'processing_results.csv')
    ks_v2 = _ks(BENCH_V2 / 'processing_results.csv')

    assert ks_v2 < ks_v1, (
        f'the rebuilt suite ({ks_v2:.4f}) is further from the real plates than '
        f'the pre-rebuild one ({ks_v1:.4f})')
    assert ks_v2 < KS_CEILING, (
        f'synthetic residual autocorrelation KS = {ks_v2:.4f} against the real '
        f'plates, above the {KS_CEILING} ceiling; see the re-baseline policy in '
        'this test docstring')


def test_the_suite_is_not_easier_than_the_plates_on_its_good_fraction(manifest):
    """The GOOD share of the suite stays inside the real plates' range.

    The real conditions run 49-60 % GOOD.  v1 shipped 71.9 %, which flatters
    every prevalence-sensitive metric.  The band is wide on purpose: this is a
    composition guard, not a target.
    """
    good_fraction = float(manifest['expected_good']) / float(manifest['n_curves_written'])
    assert 0.40 <= good_fraction <= 0.65, (
        f'the suite is {good_fraction:.1%} GOOD; the real plates are 49-60 %')


# ---------------------------------------------------------------------------
# The frozen v1 baseline must stay frozen
# ---------------------------------------------------------------------------

def test_the_v1_snapshot_is_intact():
    """v1 is the only evidence that the pipeline did not change under us.

    Three other test files run the pipeline over these plates and compare with
    the published 319/100/35/26.  If the snapshot moves, that comparison stops
    meaning anything, so its digests are checked here.
    """
    manifest_path = _require(BENCH_V1 / 'MANIFEST.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['dataset_version'] == 'v1'

    # The two prose files were edited after the manifest was hashed, so their
    # recorded digests are stale as committed.  They carry no measurement, so
    # the mismatch is recorded here rather than silently skipped: a THIRD
    # mismatching entry fails this test.  Fix: rewrite the two digests in
    # synthetic_data/benchmark/v1/MANIFEST.json and delete this set.
    KNOWN_STALE_DIGESTS = {'README.md', 'HEADLINE.json'}

    stale = []
    for rel, entry in manifest['files'].items():
        path = BENCH_V1 / rel
        assert path.exists(), f'the v1 snapshot lost {rel}'
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            stale.append(rel)
            continue
        assert path.stat().st_size == entry['bytes'], rel
    assert set(stale) <= KNOWN_STALE_DIGESTS, (
        f'the frozen v1 snapshot moved: {sorted(set(stale) - KNOWN_STALE_DIGESTS)}')


def test_the_v1_headline_is_labelled_as_history():
    """v1's numbers are true of v1 and must never read as current."""
    headline = json.loads(_require(BENCH_V1 / 'HEADLINE.json').read_text())
    assert headline['dataset_version'] == 'v1'
    readme = _require(BENCH_V1 / 'README.md').read_text().lower()
    assert 'v1' in readme
    assert any(word in readme for word in ('frozen', 'baseline', 'historical',
                                           'pre-rebuild'))
