"""WP6 acceptance tests: export, time-grid correctness and plate emission.

Findings covered
----------------
M28  Export bookkeeping.  ``export_for_analysis_pipeline`` bucketed curves on
     point count alone and took ``TIME[H]`` for the whole bucket from the first
     row, so a 50 h / 0.125 h scenario landing beside a 100 h / 0.25 h one
     (both 400 points) was written on the 100 h axis: every timestamp doubled,
     the pipeline read mu at half and lambda at twice their true values, and
     ground truth kept the original numbers.  The file label also used
     ``time[-1]``, which is one sample short, so a 100 h run shipped as
     ``SYNTHETIC_400pts_99h`` and a 150 h run as ``600pts_149h`` -- and those
     strings reach the pipeline as strain names.  Five ``eval()`` calls parsed
     CSV-round-tripped series.

M10  The benchmark bypassed the pipeline's whole preprocessing stage.  The
     plate path here is what lets blank selection, replicate averaging and the
     negative clip in ``tecan_growth/preprocess.py`` run on synthetic data --
     the stage that turns Group 4 PERMETHRINANDLB-CISPERM1's per-well maxima
     [0.016, 1.063, 0.010] into the smooth 0.345 OD sigmoid the pipeline calls
     GOOD.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import output_formatter as of                       # noqa: E402
from src.growth_models import GompertzModel                  # noqa: E402
from src.output_formatter import (                           # noqa: E402
    GroundTruthSchemaError,
    MixedTimeGridError,
    TECANFormatWriter,
    bucket_by_grid,
    grid_label,
    grid_of,
    ground_truth_columns,
    write_manifest,
)
from src.plate_writer import PlateFormatWriter                # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _curve(curve_id, duration_h, dt_h, *, A=1.2, mu=0.20, lam=4.0, seed=0, t0=0.0):
    """
    One generator-shaped row: a clean Gompertz on an explicit grid.

    The row carries the FULL v2 ground-truth schema (``truth.py``'s
    ``GROUND_TRUTH_COLUMNS``) minus ``strain_name``, because that is what the
    generator now hands the writer: ``create_ground_truth_csv`` selects, it
    does not invent a measurement, so a v1-shaped fixture would only be
    testing the schema error. ``test_fixture_row_is_schema_complete`` pins
    that coverage, and the ``true_*`` values are the realised ones -- this
    curve is a clean Gompertz, so it realises exactly what was requested.
    """
    n = int(round(duration_h / dt_h))
    t = t0 + np.arange(n) * dt_h
    clean = GompertzModel.compute(t, A, mu, lam)
    noisy = clean + np.random.default_rng(seed).normal(0, 0.002, n)
    noisy = np.maximum(noisy, 0.0)
    rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    ss_res = float(np.sum((noisy - clean) ** 2))
    ss_tot = float(np.sum((noisy - noisy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    # the primary phase is complete at 95% of the realised amplitude
    reached = np.nonzero(clean - clean[0] >= 0.95 * A)[0]
    t95 = float(t[reached[0]]) if len(reached) else float('nan')

    return {
        # identity and label
        'curve_id': curve_id,
        'scenario': f'scen_{int(round(duration_h))}h',
        'expected_class': 'GOOD',
        'label_rule': 'scenario',
        'label_rule_version': of.label_rule_version(),
        'scoring_mode': 'standard',
        'lab_band': '',
        # what was generated
        'model_type': 'gompertz', 'shape': 'gompertz',
        'requested_A': A, 'requested_mu': mu, 'requested_lambda': lam,
        # what was actually written
        'true_A': A, 'true_mu': mu, 'true_lambda': lam,
        'mu_realisation_ratio': 1.0, 'lambda_realisation_error': 0.0,
        'shape_solved': True,
        'initial_od': 0.0,
        # noise provenance
        'noise_level': 'low', 'noise_sources': 'preset',
        'target_r_squared': np.nan, 'actual_r_squared': r2,
        'artefact_free_r_squared': r2, 'artefact_free_rmse': rmse, 'rmse': rmse,
        # trace summary
        'max_od': float(noisy.max()), 'delta_od': float(noisy.max() - noisy[0]),
        'n_points': n,
        'time_resolution': dt_h,
        'duration_hours': n * dt_h,
        'seed': seed,
        # truncation / identifiability ground truth
        'true_window_end_h': 1.3 * t95, 't_primary_95': t95,
        't_second_start': np.nan, 't_death': np.nan,
        'oracle_A_relerr': 0.0, 'oracle_mu_relerr': 0.0, 'oracle_fit_ok': True,
        'true_frac_of_A_reached': 1.0,
        # replicates and artefacts
        'replicate_A_cv': 0.0, 'replicate_mu_cv': 0.0, 'outlier_well': False,
        'spike_indices': '', 'drift_od_per_h': 0.0, 'twin_curve_id': '',
        'metrics_defined': True,
        # series
        'time': t.tolist(),
        'od600': noisy.tolist(),
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M28 (1): mixed grids are separated
# ---------------------------------------------------------------------------

def test_mixed_grids_are_separated(tmp_path):
    """400 points at 0.25 h and 400 points at 0.125 h are two files, not one."""
    df = _frame([
        _curve(0, 100.0, 0.25, seed=1),
        _curve(1, 100.0, 0.25, seed=2),
        _curve(2, 50.0, 0.125, seed=3),
        _curve(3, 50.0, 0.125, seed=4),
    ])
    assert set(df['n_points']) == {400}, 'both grids must share the point count'

    writer = TECANFormatWriter(str(tmp_path))
    result = writer.export_for_analysis_pipeline(df)

    files = sorted(p.name for p in result['data_dir'].glob('*_DATA.csv'))
    assert files == ['SYNTHETIC_400pts_100h_15m_DATA.csv',
                     'SYNTHETIC_400pts_50h_8m_DATA.csv'], files

    last = {}
    for name in files:
        table = pd.read_csv(result['data_dir'] / name)
        assert len(table.columns) == 3, 'two curves plus TIME[H] per file'
        last[name] = float(table['TIME[H]'].iloc[-1])
    assert last['SYNTHETIC_400pts_100h_15m_DATA.csv'] == pytest.approx(99.75)
    assert last['SYNTHETIC_400pts_50h_8m_DATA.csv'] == pytest.approx(49.875)


def test_same_grid_key_different_time_vector_raises(tmp_path):
    """
    The allclose guard is what makes a key collision loud.

    Two curves can share (n_points, median dt) and still be on different axes;
    without the guard the second is silently rewritten onto the first one's
    timestamps.
    """
    ok = _curve(10, 100.0, 0.25, seed=5)
    shifted = _curve(11, 100.0, 0.25, seed=6, t0=13.0)   # same key, offset axis
    df = _frame([ok, shifted])
    assert of.grid_key(np.asarray(ok['time'])) == of.grid_key(np.asarray(shifted['time']))

    with pytest.raises(MixedTimeGridError) as exc:
        bucket_by_grid(df)
    assert '11' in str(exc.value)

    with pytest.raises(MixedTimeGridError):
        TECANFormatWriter(str(tmp_path)).export_for_analysis_pipeline(df)


def test_export_generated_curves_also_splits_grids(tmp_path):
    """The same fix applies to the group-by-scenario writer."""
    df = _frame([
        _curve(0, 100.0, 0.25, seed=1),
        _curve(1, 50.0, 0.125, seed=2),
    ])
    df['scenario'] = 'mixed'                       # one group, two grids
    files = TECANFormatWriter(str(tmp_path)).export_generated_curves(df)
    assert len(files) == 2, files
    assert sorted(files) == ['mixed|400pts_100h_15m', 'mixed|400pts_50h_8m']
    names = sorted(Path(p).name for p in files.values())
    assert names == ['SYNTHETIC_MIXED_400PTS_100H_15M_DATA.csv',
                     'SYNTHETIC_MIXED_400PTS_50H_8M_DATA.csv'], names
    for path in files.values():
        table = pd.read_csv(path)
        assert len(table.columns) == 2


def test_export_generated_curves_encodes_the_grid_for_single_grid_groups(tmp_path):
    """A single-grid group is keyed by its group name but still named by its grid."""
    df = _frame([_curve(0, 100.0, 0.25, seed=1), _curve(1, 100.0, 0.25, seed=2)])
    df['scenario'] = 'standard'
    files = TECANFormatWriter(str(tmp_path)).export_generated_curves(df)
    assert list(files) == ['standard']
    assert Path(files['standard']).name == 'SYNTHETIC_STANDARD_400PTS_100H_15M_DATA.csv'


# ---------------------------------------------------------------------------
# M28 (2): the name encodes the grid
# ---------------------------------------------------------------------------

def test_name_encodes_the_grid(tmp_path):
    """
    For every ground-truth row the DATA file name encodes the same
    (n_points, dt_minutes) pair, and duration_hours == n_points * dt.
    """
    grids = [(100.0, 0.25), (120.0, 0.25), (20.0, 0.25), (150.0, 0.25),
             (100.0, 1.0), (100.0, 100.0 / 1205)]
    rows = []
    for i, (duration, dt) in enumerate(grids):
        rows.append(_curve(i, duration, dt, seed=i))
    df = _frame(rows)

    writer = TECANFormatWriter(str(tmp_path))
    result = writer.export_for_analysis_pipeline(df)
    gt = pd.read_csv(result['ground_truth'])

    # every published grid label is dot-free and carries the resolution
    assert sorted(result['data_files']) == [
        'SYNTHETIC_100pts_100h_60m',
        'SYNTHETIC_1205pts_100h_5m',
        'SYNTHETIC_400pts_100h_15m',
        'SYNTHETIC_480pts_120h_15m',
        'SYNTHETIC_600pts_150h_15m',
        'SYNTHETIC_80pts_20h_15m',
    ], sorted(result['data_files'])
    for label in result['data_files']:
        assert '.' not in label

    # the row -> file mapping agrees with the row's own (n_points, dt)
    label_of_curve = {}
    for label, grid in result['grids'].items():
        for cid in grid['curve_ids']:
            label_of_curve[cid] = label

    pattern = re.compile(r'^SYNTHETIC_(\d+)pts_(\d+)h_(\d+)m$')
    for _, row in gt.iterrows():
        label = label_of_curve[int(row['curve_id'])]
        n_pts, hours, minutes = (int(g) for g in pattern.match(label).groups())
        source = df[df['curve_id'] == row['curve_id']].iloc[0]
        dt = float(source['time_resolution'])
        assert n_pts == int(row['n_points'])
        assert minutes == int(round(dt * 60))
        assert hours == int(round(int(row['n_points']) * dt))
        assert abs(float(row['duration_hours']) - int(row['n_points']) * dt) < 1e-6


def test_duration_uses_the_grid_not_the_last_sample():
    """``time[-1]`` is one sample short: that is where 99h-for-100h came from."""
    t = np.arange(400) * 0.25
    assert grid_of(t)['t_last'] == pytest.approx(99.75)
    assert grid_of(t)['duration_h'] == pytest.approx(100.0)
    assert grid_label(t) == 'SYNTHETIC_400pts_100h_15m'


# ---------------------------------------------------------------------------
# ground-truth schema
# ---------------------------------------------------------------------------

def test_fixture_row_is_schema_complete():
    """
    The fixture speaks the schema the generator writes.

    A v1-shaped fixture is why this file could pass while
    ``create_ground_truth_csv`` was never exercised on a real generator row:
    every case would stop at the schema error instead.
    """
    row = _curve(0, 100.0, 0.25)
    assert set(ground_truth_columns()) - set(row) == {'strain_name'}


def test_ground_truth_schema_complete(tmp_path):
    """The emitted column list equals the declared schema exactly, in order."""
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(3)])
    writer = TECANFormatWriter(str(tmp_path))
    path = writer.create_ground_truth_csv(df, verbose=False)
    assert list(pd.read_csv(path).columns) == ground_truth_columns()


def test_ground_truth_missing_column_is_named_not_dropped(tmp_path):
    """A measurement the generator did not compute is a loud error."""
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(2)])
    reconstructable = set(of.ground_truth_fill_defaults()) | {
        'strain_name', 'n_points', 'time_resolution', 'duration_hours',
        'requested_A', 'requested_mu', 'requested_lambda',
    }
    missing = [c for c in ground_truth_columns()
               if c not in reconstructable and c != 'curve_id'][0]
    df = df.drop(columns=[missing])

    with pytest.raises(GroundTruthSchemaError) as exc:
        TECANFormatWriter(str(tmp_path)).create_ground_truth_csv(df, verbose=False)
    assert missing in str(exc.value)

    # ... and the escape hatch still drops it, for callers that opt in
    path = TECANFormatWriter(str(tmp_path)).create_ground_truth_csv(
        df, strict=False, verbose=False)
    assert missing not in pd.read_csv(path).columns


def test_ground_truth_derives_what_is_a_pure_function_of_the_frame(tmp_path):
    """
    ``strain_name`` and the grid columns are formatting, not measurement.

    This is the shape a hand-built caller hands over -- the shape
    ``generate_synthetic.py``'s ``--scenario`` and manual branches had before
    they were routed through ``data_generator.curve_rows``: no grid columns,
    no ``target_r_squared``.  The file still comes out with the full schema,
    ``duration_hours`` computed as ``n_points * dt`` off the stored time
    vector (never ``time[-1]``), and ``target_r_squared`` -- a *request*, not
    a measurement -- left empty.
    """
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(2)])
    df = df.drop(columns=['strain_name'], errors='ignore').drop(
        columns=['duration_hours', 'time_resolution', 'target_r_squared'])

    writer = TECANFormatWriter(str(tmp_path))
    path = writer.create_ground_truth_csv(df, verbose=False)

    assert writer.last_ground_truth_fill['missing'] == []
    assert set(writer.last_ground_truth_fill['derived']) == {
        'strain_name', 'time_resolution', 'duration_hours'}
    assert writer.last_ground_truth_fill['filled'] == ['target_r_squared']

    gt = pd.read_csv(path)
    assert list(gt.columns) == ground_truth_columns()
    assert gt['duration_hours'].tolist() == [100.0, 100.0]     # not 99.75
    assert gt['strain_name'].tolist() == ['CURVE0000', 'CURVE0001']
    assert gt['target_r_squared'].isna().all()


def test_require_complete_still_forbids_any_reconstruction(tmp_path):
    """The generator can demand it computed every declared column itself."""
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(2)])
    df['strain_name'] = [f'CURVE{i:04d}' for i in df['curve_id']]
    complete = TECANFormatWriter(str(tmp_path))
    complete.create_ground_truth_csv(df, verbose=False, require_complete=True)

    partial = df.drop(columns=['duration_hours'])
    with pytest.raises(GroundTruthSchemaError) as exc:
        TECANFormatWriter(str(tmp_path)).create_ground_truth_csv(
            partial, verbose=False, require_complete=True)
    assert 'duration_hours' in str(exc.value)


@pytest.mark.parametrize('mode', [
    ['--scenario', 'standard', '--n-curves', '2'],
    ['--A', '1.2', '--mu', '0.2', '--lambda', '4.0', '--n-curves', '2'],
])
def test_generate_synthetic_script_still_runs(tmp_path, mode):
    """
    REGRESSION: ``generate_synthetic.py`` used to build its own row dicts in
    the ``--scenario`` and manual-parameter branches -- a v1 subset with no
    ``target_r_squared`` and no ``duration_hours`` -- so the v2 schema check
    turned a working script into an exit 1.  Both branches now go through
    ``data_generator.curve_rows``, the same full-schema row the comprehensive
    set writes, and the CLI has to keep running end to end.
    """
    out = tmp_path / mode[0].lstrip('-')
    proc = subprocess.run(
        [sys.executable,
         str(SYNTH / 'scripts' / 'generate_synthetic.py'), *mode, '-o', str(out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    gt = pd.read_csv(out / 'ground_truth.csv')
    assert list(gt.columns) == ground_truth_columns()
    assert len(gt) == 2
    assert (gt['duration_hours'] == 100.0).all()
    assert sorted(p.name for p in (out / 'DATA').glob('*_DATA.csv')) == [
        'SYNTHETIC_400pts_100h_15m_DATA.csv']


def test_ground_truth_schema_follows_truth_module_when_present(tmp_path, monkeypatch):
    """Once WP3's truth.py exists its GROUND_TRUTH_COLUMNS wins, in its order."""
    class _FakeTruth:
        GROUND_TRUTH_COLUMNS = ['curve_id', 'strain_name', 'scenario', 'expected_class',
                                'true_A', 'true_mu', 'true_lambda']
        TRUTH_SCHEMA_VERSION = '2'

    monkeypatch.setattr(of, '_load_truth_module', lambda: _FakeTruth)
    assert ground_truth_columns() == _FakeTruth.GROUND_TRUTH_COLUMNS
    assert of.ground_truth_schema_version() == '2'

    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(2)])
    path = TECANFormatWriter(str(tmp_path)).create_ground_truth_csv(df, verbose=False)
    assert list(pd.read_csv(path).columns) == _FakeTruth.GROUND_TRUTH_COLUMNS


def test_legacy_schema_matches_the_v1_benchmark_snapshot():
    """The v1 fallback is the schema of the archived v1 ground_truth.csv, in order.

    The release (WP12) moved the live ``comprehensive_test`` ground truth to truth
    schema v2 and snapshotted the pre-rebuild file under ``synthetic_data/benchmark/v1/``.
    That snapshot is what ``LEGACY_GROUND_TRUTH_COLUMNS`` has to keep describing, since
    it is the fallback WP7 reads when a v1 ground truth is handed to the validator.
    """
    committed = REPO_ROOT / 'synthetic_data' / 'benchmark' / 'v1' / 'ground_truth.csv'
    if not committed.exists():
        pytest.skip('v1 benchmark snapshot not present')
    header = pd.read_csv(committed, nrows=0).columns.tolist()
    assert header == list(of.LEGACY_GROUND_TRUTH_COLUMNS)


def test_live_ground_truth_follows_the_current_schema():
    """The regenerated ground truth carries the schema this writer emits today."""
    live = (REPO_ROOT / 'synthetic_data' / 'output' / 'comprehensive_test' /
            'test_data' / 'ground_truth.csv')
    if not live.exists():
        pytest.skip('regenerated ground truth not present')
    header = pd.read_csv(live, nrows=0).columns.tolist()
    assert header == list(ground_truth_columns())


# ---------------------------------------------------------------------------
# no eval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module_path', [
    SYNTH / 'src' / 'output_formatter.py',
    SYNTH / 'src' / 'plate_writer.py',
])
def test_no_eval(module_path):
    """Series that have been through a CSV round trip are parsed, not evaluated."""
    source = module_path.read_text()
    hits = [m.start() for m in re.finditer(r'(?<![_.\w])eval\s*\(', source)]
    assert hits == [], f"{module_path.name} still calls eval() at {hits}"


def test_parse_series_round_trips_lists_and_nan():
    """literal_eval handles the normal case; the fallback covers nan/inf tokens."""
    values = [0.0, 0.25, 0.5]
    assert of.parse_series(str(values)).tolist() == values
    assert of.parse_series(np.array(values)).tolist() == values
    parsed = of.parse_series(repr([1.0, float('nan'), float('inf')]))
    assert parsed[0] == 1.0 and np.isnan(parsed[1]) and np.isinf(parsed[2])
    with pytest.raises(ValueError):
        of.parse_series('[1.0, not-a-number]', curve_id=7)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_write_manifest_hashes_the_exported_files(tmp_path):
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(2)])
    writer = TECANFormatWriter(str(tmp_path))
    writer.export_for_analysis_pipeline(df)

    path = write_manifest(tmp_path, {'seed': 0, 'git_sha': 'deadbeef'})
    manifest = json.loads(path.read_text())

    assert manifest['seed'] == 0 and manifest['git_sha'] == 'deadbeef'
    assert set(manifest['file_sha256']) == {
        'DATA/SYNTHETIC_400pts_100h_15m_DATA.csv', 'ground_truth.csv'}
    assert all(len(v) == 64 for v in manifest['file_sha256'].values())
    assert manifest['file_sha256']['ground_truth.csv'] == of.sha256_file(
        tmp_path / 'ground_truth.csv')
    assert manifest['truth_schema_version'] == of.ground_truth_schema_version()


# ---------------------------------------------------------------------------
# M10: the plate path
# ---------------------------------------------------------------------------

def _write_plates(tmp_path, df, **kw):
    writer = PlateFormatWriter(str(tmp_path), **kw)
    return writer.write_plates(df)


def test_plate_files_match_the_group3_header_conventions(tmp_path):
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(4)])
    written = _write_plates(tmp_path, df)
    assert len(written['plates']) == 1
    plate = written['plates'][0]

    header = Path(plate['values_csv']).read_text().splitlines()[0].split(',')
    assert header[:3] == ['Cycle Nr.', 'Time [s]', 'Temp. [C]']
    # wells fill row-major (blanks A1-A3 first, as in GROUP3_KEY.csv) and the
    # value columns are emitted in TECAN column-major order restricted to the
    # wells actually used -- unassigned wells are omitted, not invented
    assert header[3:] == ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'], header
    from src.plate_writer import WELLS_COL_MAJOR
    order = [WELLS_COL_MAJOR.index(w) for w in header[3:]]
    assert order == sorted(order)

    values = pd.read_csv(plate['values_csv'])
    assert values['Cycle Nr.'].tolist() == list(range(1, 401))
    assert values['Time [s]'].iloc[1] == pytest.approx(900.0)   # 0.25 h in seconds

    key = pd.read_csv(plate['key_csv'])
    assert list(key.columns) == ['Cell', 'Media', 'Strain']
    assert (key['Strain'] == 'BLANK').sum() == 3
    assert set(key.loc[key['Strain'] != 'BLANK', 'Strain']) == {
        'CURVE0000', 'CURVE0001', 'CURVE0002', 'CURVE0003'}
    assert set(key['Media']) == {'SCEN01'}


def test_plate_roundtrip(tmp_path):
    """A generated plate loads through load_plate + preprocess_plate untouched."""
    from tecan_growth.io import load_plate
    from tecan_growth.preprocess import preprocess_plate

    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(4)])
    plate = _write_plates(tmp_path, df)['plates'][0]

    run = load_plate(plate['values_csv'], plate['key_csv'])
    assert run.n_cycles == 400
    assert run.time_h[-1] == pytest.approx(99.75)
    assert run.interval_h == pytest.approx(0.25)

    processed = preprocess_plate(run)
    assert sorted(processed.curve_ids) == [f'SCEN01-CURVE{i:04d}' for i in range(4)]
    assert set(processed.conditions.blank_source) == {'per_media'}
    assert set(processed.conditions.n_blank_wells) == {3}

    # the identity blank makes the plate path reproduce the per-curve path
    for i in range(4):
        _, od = processed.curve(f'SCEN01-CURVE{i:04d}')
        np.testing.assert_allclose(od, np.asarray(df.iloc[i]['od600']), atol=1e-12)


def test_plate_roundtrip_through_the_preprocess_script(tmp_path):
    """
    scripts/02_preprocess_raw_plate_data.py accepts the plate and its output
    columns still carry the CURVE\\d+ key that the validator
    (pipeline_validator.py:177) and the trainer (09_train_classifier.py:356)
    parse.
    """
    df = _frame([_curve(i, 100.0, 0.25, seed=i) for i in range(4)])
    plate = _write_plates(tmp_path, df)['plates'][0]

    out_dir = tmp_path / 'DATA'
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / '02_preprocess_raw_plate_data.py'),
         plate['values_csv'], plate['key_csv'], '-o', str(out_dir), '-q'],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr

    produced = sorted(out_dir.glob('*_DATA.csv'))
    assert [p.name for p in produced] == ['SCEN01_DATA.csv']

    table = pd.read_csv(produced[0])
    assert table.columns[0] == 'TIME[H]'
    curve_re = re.compile(r'(CURVE\d+)')
    ids = [curve_re.search(c).group(1) for c in table.columns[1:] if curve_re.search(c)]
    assert ids == [f'CURVE{i:04d}' for i in range(4)]
    assert list(table.columns[1:]) == [f'SCEN01_CURVE{i:04d}_blanked' for i in range(4)]
    assert float(table['TIME[H]'].iloc[-1]) == pytest.approx(99.75)


def test_plate_uses_supplied_replicate_and_blank_wells(tmp_path):
    """
    Per-well traces from the generator become real wells, so replicate
    averaging and the negative clip actually run.

    The condition below is the Group 4 PERMETHRINANDLB-CISPERM1 shape -- one
    well of three grew -- which the pipeline calls GOOD off the mean.
    """
    row = _curve(7, 100.0, 0.25, seed=11)
    n = int(row['n_points'])
    grower = np.asarray(row['od600'])
    flat_a = np.full(n, 0.016)
    flat_b = np.full(n, 0.010)
    row['well_od600'] = [grower.tolist(), flat_a.tolist(), flat_b.tolist()]
    row['blank_od600'] = [np.full(n, 0.12).tolist(),
                          np.full(n, 0.13).tolist(),
                          np.full(n, 0.11).tolist()]
    df = _frame([row])

    plate = _write_plates(tmp_path, df)['plates'][0]
    assert plate['blank_source'] == 'generator'
    assert plate['conditions'][0]['n_wells'] == 3

    from tecan_growth.io import load_plate
    from tecan_growth.preprocess import preprocess_plate
    processed = preprocess_plate(load_plate(plate['values_csv'], plate['key_csv']))

    cond = processed.conditions.iloc[0]
    assert cond.n_replicates == 3 and cond.n_blank_wells == 3
    _, od = processed.curve('SCEN01-CURVE0007')
    expected = np.maximum(np.mean([grower, flat_a, flat_b], axis=0) - 0.12, 0.0)
    np.testing.assert_allclose(od, expected, atol=1e-12)
    assert od[0] == 0.0, 'the blank over-subtraction must reach the negative clip'


def test_plates_split_by_scenario_grid_and_capacity(tmp_path):
    """One plate per (scenario, grid); a scenario wider than 96 wells splits."""
    rows = [_curve(i, 100.0, 0.25, seed=i) for i in range(2)]
    rows += [_curve(100 + i, 50.0, 0.125, seed=i) for i in range(2)]
    for r in rows[:2]:
        r['scenario'] = 'alpha'
    for r in rows[2:]:
        r['scenario'] = 'alpha'                      # same scenario, two grids
    big = [_curve(200 + i, 20.0, 0.25, seed=i) for i in range(120)]
    for r in big:
        r['scenario'] = 'beta'
    written = _write_plates(tmp_path, _frame(rows + big))

    plates = written['plates']
    scenarios = [p['scenario'] for p in plates]
    assert scenarios.count('alpha') == 2, 'two grids -> two plates'
    assert scenarios.count('beta') == 2, '120 conditions do not fit on one plate'
    assert len({p['media'] for p in plates}) == len(plates), 'media tokens are unique'
    assert all(p['n_wells'] <= 96 for p in plates)

    index = json.loads(Path(written['index']).read_text())
    assert index['n_plates'] == len(plates)
    seen = [c['curve_id'] for p in index['plates'] for c in p['conditions']]
    assert sorted(seen) == sorted(int(c) for c in _frame(rows + big)['curve_id'])
