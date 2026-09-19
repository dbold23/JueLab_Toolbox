"""WP8 acceptance tests: one generated summary, no contradicting numbers.

Findings covered
----------------
The renderer could not be run at all: its defaults pointed at
``output/comprehensive_test/pipeline_results/processing_results.csv`` and
``output/comprehensive_test/validation_report/validation_report.json``, neither
of which any step ever wrote, so ``python validation_summary.py`` died in
argparse validation.  It also read metrics from two JSON files with different
schemas and merged ground truth with pipeline results a third time, so the
markdown could disagree with both.

Three documents stated three different confusion matrices for the same 480
curves, and the one recommendation the suite made to the pipeline team --
"raise the R-squared / delta-OD acceptance thresholds" -- was derived from
mislabelled false positives whose parameters were demonstrably recoverable.

These tests pin the properties that let that happen: the renderer runs from a
run directory, every number it prints comes from that run's ``report.json``, it
writes its rendering only inside that directory, the only committed
``VALIDATION_SUMMARY.md`` is v1's labelled history (the current one is a build
artefact, rendered on demand), and no file in the tree states a confusion matrix
the report does not.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

RENDERER = SYNTH / 'scripts' / 'validation_summary.py'
BENCH_V2 = SYNTH / 'benchmark' / 'v2'
BENCH_V1 = SYNTH / 'benchmark' / 'v1'
SUMMARY_NAME = 'VALIDATION_SUMMARY.md'

ENV = {'MPLBACKEND': 'Agg', 'PATH': '/usr/bin:/bin:/usr/local/bin',
       'HOME': str(Path.home())}


def run_renderer(*args, cwd=REPO_ROOT):
    return subprocess.run([sys.executable, str(RENDERER), *args],
                          cwd=str(cwd), env=ENV, capture_output=True,
                          text=True, timeout=300)


@pytest.fixture(scope='module')
def report():
    path = BENCH_V2 / 'report.json'
    if not path.exists():
        pytest.skip('no published benchmark report in this checkout')
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    """The summary as this renderer produces it, into a scratch path."""
    if not (BENCH_V2 / 'report.json').exists():
        pytest.skip('no published benchmark report in this checkout')
    out = tmp_path_factory.mktemp('summary') / 'VALIDATION_SUMMARY.md'
    proc = run_renderer('--run-dir', str(BENCH_V2), '--output', str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# It runs
# ---------------------------------------------------------------------------

def test_it_runs_with_no_arguments():
    """The defaults used to name two directories nothing ever wrote."""
    proc = run_renderer()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith('# Synthetic Data Validation Summary')


def test_a_no_argument_run_leaves_no_file_behind():
    """The rendering is a build artefact; rendering it must not dirty the tree.

    WP8 first shipped this the other way round: rendering the published run
    wrote `output/comprehensive_test/VALIDATION_SUMMARY.md`, a path this package
    does not own and which the release deletes, so every render dropped an
    untracked file at a deleted path -- one that no commit would pick up and
    that made the release's own summary either missing or unstaged.
    """
    before = {p for p in REPO_ROOT.rglob(SUMMARY_NAME)}
    proc = run_renderer()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {p for p in REPO_ROOT.rglob(SUMMARY_NAME)}
    assert after == before, f'rendering created {sorted(after - before)}'


def test_it_never_writes_into_a_committed_snapshot(tmp_path):
    """Output goes beside the run's report, and never into a release snapshot.

    A snapshot under `synthetic_data/benchmark/` is fixed and its MANIFEST.json
    enumerates it, so a rendering of one goes to stdout instead.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location('vs_module', str(RENDERER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for run_dir in (tmp_path,
                    SYNTH / 'output' / 'comprehensive_test' / 'validation' / 'first_peak'):
        out = module.default_output_for(Path(run_dir))
        assert out is not None and out.name == SUMMARY_NAME
        assert out.parent.resolve() == Path(run_dir).resolve(), run_dir

    for snapshot in (BENCH_V1, BENCH_V2, SYNTH / 'benchmark' / 'v9'):
        assert module.default_output_for(snapshot) is None, snapshot


def test_output_still_writes_where_it_is_told(tmp_path):
    out = tmp_path / 'somewhere' / 'S.md'
    proc = run_renderer('--run-dir', str(BENCH_V2), '--output', str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists() and out.read_text(encoding='utf-8').startswith('# Synthetic')


def test_it_takes_a_run_dir():
    assert '--run-dir' in run_renderer('--help').stdout


def test_a_run_directory_without_a_report_is_named_not_guessed(tmp_path):
    proc = run_renderer('--run-dir', str(tmp_path))
    assert proc.returncode == 1
    assert 'No renderable run directory' in proc.stdout


def test_rendering_is_deterministic(tmp_path):
    """No wall-clock stamp: re-rendering one report reproduces the file."""
    if not (BENCH_V2 / 'report.json').exists():
        pytest.skip('no published benchmark report in this checkout')
    first, second = tmp_path / 'a.md', tmp_path / 'b.md'
    for out in (first, second):
        assert run_renderer('--run-dir', str(BENCH_V2),
                            '--output', str(out)).returncode == 0
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------

def test_the_first_table_names_the_truncation_mode(rendered, report):
    first_table = rendered.split('|', 1)[1]
    first_table = first_table[:first_table.index('\n\n')]
    mode = report['run']['truncation_mode']
    assert mode in first_table, 'the first table does not name its truncation mode'


def test_the_headline_leads_with_balanced_accuracy(rendered):
    head = rendered.split('## Headline', 1)[1].split('##', 1)[0]
    for lead in ('Balanced accuracy', 'Sensitivity', 'Specificity'):
        assert lead in head
    assert head.index('Balanced accuracy') < head.index('| Accuracy')


def test_every_headline_metric_matches_the_report(rendered, report):
    cls = report['classification']
    head = rendered.split('## Headline', 1)[1].split('### ', 1)[0]
    for key in ('balanced_accuracy', 'sensitivity', 'specificity', 'accuracy',
                'precision', 'npv', 'f1_score'):
        assert f'{cls[key] * 100:.1f}%' in head, key


def test_the_confusion_matrix_matches_the_report(rendered, report):
    cm = report['classification']['confusion_matrix']
    assert f"| {cm['true_positives']} (TP) | {cm['false_negatives']} (FN) |" in rendered
    assert f"| {cm['false_positives']} (FP) | {cm['true_negatives']} (TN) |" in rendered


def test_provenance_is_stamped(rendered, report):
    run = report['run']
    for value in (str(run['seed']), run['dataset_sha256'][:12],
                  run['git_sha'][:12], str(run['n_curves']),
                  str(run['n_scenarios'])):
        assert value in rendered, value


def test_the_in_sample_banner_appears_when_ml_is_in_the_verdict(tmp_path, report):
    """An ML-gated run is scored on the classifier's own training set."""
    run_dir = tmp_path / 'ml_run'
    run_dir.mkdir()
    doctored = json.loads(json.dumps(report))
    doctored['run']['ml_classify'] = True
    (run_dir / 'report.json').write_text(json.dumps(doctored))
    out = tmp_path / 'ml.md'
    assert run_renderer('--run-dir', str(run_dir),
                        '--output', str(out)).returncode == 0
    text = out.read_text(encoding='utf-8')
    assert 'IN-SAMPLE' in text
    assert 'training set' in text

    plain = tmp_path / 'plain.md'
    assert run_renderer('--run-dir', str(BENCH_V2),
                        '--output', str(plain)).returncode == 0
    assert 'IN-SAMPLE' not in plain.read_text(encoding='utf-8')


def test_the_held_out_blocks_are_reported_separately(rendered, report):
    excluded = report['classification'].get('excluded_from_headline', {})
    if not excluded.get('n'):
        pytest.skip('this run excludes nothing')
    block = rendered.split('## Blocks held out of the headline', 1)[1]
    for mode, n in excluded['by_scoring_mode'].items():
        assert mode in block
        assert str(n) in block


def test_the_four_recovery_strata_each_carry_an_n(rendered, report):
    recovery = report['parameter_recovery']
    block = rendered.split('## Parameter recovery', 1)[1].split('##', 1)[0]
    for stratum in ('accepted_good', 'all_fitted_good', 'rejected_good',
                    'accepted_bad'):
        if stratum in recovery:
            assert f"n = {recovery[stratum]['n']}" in block, stratum


def test_recovery_prose_is_generated_from_the_measured_pass_rate(rendered):
    """The renderer used to print 'Excellent recovery' whatever the data said."""
    assert 'Excellent recovery' not in rendered
    assert re.search(r'of the curves land within', rendered)


def test_the_selection_bias_and_false_positive_sentences_are_present(rendered, report):
    assert 'Selection bias' in rendered
    note = report['parameter_recovery']['selection_bias_delta']['note']
    assert note.split('.')[0] in rendered
    fp = report['classification']['confusion_matrix']['false_positives']
    assert f'The {fp} false positives' in rendered
    assert 'reported to the user' in rendered


def test_the_threshold_recommendation_is_withdrawn(rendered):
    """It was derived from mislabelled curves and aimed at the real pipeline."""
    lowered = rendered.lower()
    assert 'tighten borderline thresholds' not in lowered
    assert 'withdrawn' in lowered


def test_scenario_descriptions_come_from_the_catalog(rendered):
    from src.curve_scenarios import ALL_SCENARIOS

    for name in ('truncation_challenge', 'contamination'):
        assert ALL_SCENARIOS[name].description in rendered, name
    # The two descriptions the hand-kept table got wrong.
    assert 'Experiment ended before stationary phase' not in rendered
    assert 'Sudden OD spikes' not in rendered


def test_oracle_labelled_scenarios_are_not_called_flat(rendered):
    """borderline_noise and high_noise curves grow; only their fit is bad."""
    table = rendered.split('## Per-scenario accuracy', 1)[1].split('##', 1)[0]
    assert 'no / failed growth' not in table
    assert 'Label rule' in table
    for name in ('borderline_noise', 'high_noise'):
        row = [ln for ln in table.splitlines() if ln.startswith(f'| {name} |')]
        assert row, name
        assert 'oracle' in row[0], name


# ---------------------------------------------------------------------------
# One summary, and nothing contradicting it
# ---------------------------------------------------------------------------

def _tracked_markdown():
    proc = subprocess.run(['git', 'ls-files', '*.md'], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip('not a git checkout')
    return [REPO_ROOT / rel for rel in proc.stdout.split()]


def test_the_only_committed_summary_is_the_labelled_v1_history():
    """The current summary is a build artefact, so none of it is committed.

    v1's copy is history and says so. Anything else committed would be a fork of
    the run's own report.json, which is how the repository came to state three
    confusion matrices for one set of curves.
    """
    proc = subprocess.run(['git', 'ls-files', '*VALIDATION_SUMMARY.md'],
                          cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=120)
    if proc.returncode != 0:
        pytest.skip('not a git checkout')
    tracked = sorted(proc.stdout.split())
    assert tracked == ['synthetic_data/benchmark/v1/VALIDATION_SUMMARY.md'], tracked

    staged = subprocess.run(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR',
                             '--', '*VALIDATION_SUMMARY.md'],
                            cwd=str(REPO_ROOT), capture_output=True, text=True,
                            timeout=120)
    assert staged.returncode != 0 or sorted(staged.stdout.split()) in (
        [], ['synthetic_data/benchmark/v1/VALIDATION_SUMMARY.md']), staged.stdout


def test_any_summary_in_the_tree_is_the_rendering_of_the_run_beside_it(tmp_path):
    """A summary that is present is the rendering of its own report.json.

    Nothing is required to be present -- the file is a build artefact and a
    fresh checkout has none outside v1 -- but a file that is there and does not
    reproduce is a stale fork of the numbers.
    """
    for summary in sorted(REPO_ROOT.rglob(SUMMARY_NAME)):
        if BENCH_V1 in summary.parents:
            continue                     # frozen v1 history, rendered by v1's code
        report_path = summary.parent / 'report.json'
        assert report_path.exists(), (
            f'{summary.relative_to(REPO_ROOT)} sits beside no report.json, so no '
            f'run produced it')
        out = tmp_path / f'{summary.parent.name}.md'
        proc = run_renderer('--run-dir', str(summary.parent), '--output', str(out))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert summary.read_text(encoding='utf-8') == out.read_text(encoding='utf-8'), (
            f'{summary.relative_to(REPO_ROOT)} is not the rendering of its report')


def test_no_document_states_a_confusion_matrix_the_report_denies(report):
    """A count next to (TP)/(FP)/(FN)/(TN) must be this run's or v1's, labelled."""
    cm = report['classification']['confusion_matrix']
    core = (report['classification'].get('core32') or {}).get('confusion_matrix', {})
    allowed = {str(v) for v in cm.values()} | {str(v) for v in core.values()}
    pattern = re.compile(r'(\d+)\s*\((TP|TN|FP|FN)\)')
    for path in _tracked_markdown():
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[:3] == ('synthetic_data', 'benchmark', 'v1'):
            continue          # the frozen baseline, labelled as history
        if rel.parts[:2] == ('results', 'tables'):
            continue          # committed result tables, regenerated elsewhere
        for value, _label in pattern.findall(path.read_text(encoding='utf-8')):
            assert value in allowed, f'{rel} states {value} as a confusion cell'


def test_no_json_under_output_carries_the_legacy_confusion_keys():
    legacy = {'TP', 'TN', 'FP', 'FN'}
    for path in (SYNTH / 'output').rglob('*.json'):
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            continue

        def walk(node):
            if isinstance(node, dict):
                assert not legacy & set(node), f'{path} carries {legacy & set(node)}'
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(blob)


def test_the_root_pointer_does_not_restate_the_numbers():
    pointer = REPO_ROOT / 'validation_sum.md'
    if not pointer.exists():
        pytest.skip('no root pointer in this checkout')
    text = pointer.read_text(encoding='utf-8')
    assert len(text.splitlines()) < 60
    assert not re.search(r'\|\s*\*?\*?Accuracy', text)
    assert 'VALIDATION_SUMMARY.md' in text


def test_a_run_directory_is_not_described_as_the_published_snapshot(tmp_path):
    """The rendering says which kind of directory it came from, truthfully.

    The provenance sentence used to be keyed on "is this directory *not* under
    `output/comprehensive_test/validation/`", so every other run directory --
    a scratch copy, a run written anywhere else -- produced a document opening
    with "This is the rendering of the published `first_peak` run, taken from
    its tracked snapshot", which was false. It is now keyed on the same
    predicate that decides where the rendering is written.
    """
    if not (BENCH_V2 / 'report.json').exists():
        pytest.skip('no published benchmark report in this checkout')

    run_dir = tmp_path / 'first_peak'
    run_dir.mkdir()
    for name in ('report.json', 'per_scenario.csv', 'per_scenario_recovery.csv',
                 'gate_attribution.csv', 'per_curve.csv'):
        source = BENCH_V2 / name
        if source.exists():
            shutil.copy(source, run_dir / name)

    proc = run_renderer('--run-dir', str(run_dir))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (run_dir / SUMMARY_NAME).read_text(encoding='utf-8')
    assert 'tracked snapshot' not in text, text[:800]
    assert str(run_dir) in text

    snapshot = run_renderer('--run-dir', str(BENCH_V2))
    assert snapshot.returncode == 0, snapshot.stderr
    assert 'taken from its tracked snapshot' in snapshot.stdout
