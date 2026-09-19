"""
Output Formatter for TECAN-Compatible CSV Files

Formats synthetic growth curve data to match the format the TECAN analysis
pipeline reads:

* ``TIME[H]`` plus one ``{TREATMENT}_{STRAIN}_blanked`` column per curve
  (the *per-curve* path, still the default), and
* ``ground_truth.csv``, whose column list is owned by
  :mod:`synthetic_data.src.truth` (``GROUND_TRUTH_COLUMNS``) and is written
  here verbatim, in order, with no silent dropping.  A declared column the
  caller did not supply is either *derived* (a pure function of columns it did
  supply -- ``strain_name`` from ``curve_id``, ``duration_hours`` from the
  stored time vector) or *filled* from :data:`GROUND_TRUTH_FILL_DEFAULTS` when
  it is a pure annotation; a missing measurement is always
  :class:`GroundTruthSchemaError`, and whatever had to be reconstructed is
  reported rather than done quietly.

A plate-shaped export (``VALUES.csv`` + ``KEY.csv``, one column per well) lives
next door in :mod:`plate_writer`; it is what makes the benchmark walk through
``tecan_growth.preprocess`` instead of around it.

Three correctness rules this module enforces
--------------------------------------------
1. **One file per time grid, never per point count.**  Curves are bucketed on
   ``(len(t), round(median(diff(t)), 9))``.  Before the audit the bucket key
   was the point count alone and ``TIME[H]`` for the whole bucket was taken
   from the first row, so a 50 h / 0.125 h scenario landing beside a
   100 h / 0.25 h one (both 400 points) was written on the 100 h axis: every
   timestamp doubled, the pipeline read mu at half and lambda at twice their
   true values, and ground truth kept the original numbers.  Each row's time
   vector is now compared to the bucket reference with
   ``np.allclose(..., rtol=0, atol=1e-9)`` and a mismatch raises
   :class:`MixedTimeGridError` naming the offending ``curve_id``.

2. **The file name encodes the grid it actually holds.**
   ``SYNTHETIC_{n}pts_{duration}h_{dt_minutes}m`` with
   ``duration = n_points * dt`` -- ``SYNTHETIC_400pts_100h_15m``,
   ``SYNTHETIC_1205pts_100h_5m``.  The old label used ``time[-1]``, which is
   one sample short, so a 100 h run was published as ``SYNTHETIC_400pts_99h``
   and a 150 h run as ``600pts_149h``, and those strings reach the pipeline as
   strain names.  The resolution suffix is what separates the two 400-point
   grids, so it stays even though the duration is rounded.  The labels are
   dot-free because they become column names.

3. **No ``eval``.**  Time and OD series that have been through a CSV round trip
   are parsed with :func:`ast.literal_eval` (with a numeric fallback for the
   ``nan``/``inf`` tokens ``literal_eval`` rejects).
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# Errors
# =============================================================================

class GroundTruthSchemaError(ValueError):
    """
    Raised when ``curves_df`` is missing a declared ground-truth column that
    cannot be reconstructed from the columns it does carry.

    Columns that are a pure function of what is present (``strain_name`` from
    ``curve_id``, ``duration_hours`` from the time vector) are derived, and the
    annotation columns in :data:`GROUND_TRUTH_FILL_DEFAULTS` are filled with
    their documented placeholder.  Everything else -- the measurements, the
    oracle fit, the window times -- has to come from the generator, so its
    absence is this error and never a silent NaN.
    """

    def __init__(self, missing: Sequence[str], available: Sequence[str]):
        self.missing = list(missing)
        self.available = list(available)
        super().__init__(
            "ground truth is missing "
            f"{len(self.missing)} declared column(s): {', '.join(self.missing)}. "
            "These are computed by the generator (data_generator.py), not here: "
            "create_ground_truth_csv derives what is a pure function of the "
            "columns it was given and fills the declared annotation columns, "
            "but it cannot invent a measurement. "
            f"Columns present: {', '.join(self.available)}"
        )


class MixedTimeGridError(ValueError):
    """Raised when curves that share a grid key do not share a time vector."""


# =============================================================================
# Ground-truth schema (owned by truth.py once it exists)
# =============================================================================

#: The v1 schema actually committed in
#: ``synthetic_data/output/comprehensive_test/test_data/ground_truth.csv``.
#: Used only until ``truth.py`` (WP3) lands; ``truth.GROUND_TRUTH_COLUMNS``
#: wins whenever it can be imported.
LEGACY_GROUND_TRUTH_COLUMNS: Tuple[str, ...] = (
    'curve_id', 'strain_name', 'scenario', 'expected_class',
    'true_A', 'true_mu', 'true_lambda',
    'model_type', 'initial_od',
    'noise_level', 'target_r_squared', 'actual_r_squared',
    'rmse', 'max_od', 'delta_od', 'n_points', 'duration_hours',
)


def _load_truth_module():
    """
    Import WP3's ``truth.py`` lazily, however this module was imported.

    ``synthetic_data`` is not a package (scripts insert ``synthetic_data/`` on
    ``sys.path`` and import ``from src...``), so all three spellings are tried:
    package-relative, ``src.truth`` and bare ``truth``. Returns ``None`` until
    the file exists.
    """
    try:  # imported as synthetic_data.src.output_formatter / src.output_formatter
        from . import truth as _truth  # type: ignore  # noqa: WPS433
        return _truth
    except ImportError:
        pass
    for name in ('src.truth', 'truth'):
        try:
            return __import__(name, fromlist=['GROUND_TRUTH_COLUMNS'])
        except ImportError:
            continue
    return None


def ground_truth_columns() -> List[str]:
    """
    The ordered ground-truth column list.

    Imported lazily from :mod:`truth` so this module works both before and
    after WP3 lands that file; falls back to :data:`LEGACY_GROUND_TRUTH_COLUMNS`
    (the schema of the committed ``ground_truth.csv``) when it is absent.
    """
    module = _load_truth_module()
    columns = getattr(module, 'GROUND_TRUTH_COLUMNS', None) if module else None
    return list(columns) if columns else list(LEGACY_GROUND_TRUTH_COLUMNS)


def ground_truth_schema_version() -> str:
    """``truth.TRUTH_SCHEMA_VERSION`` when available, else ``'1'`` (the committed schema)."""
    module = _load_truth_module()
    version = getattr(module, 'TRUTH_SCHEMA_VERSION', None) if module else None
    return str(version) if version is not None else '1'


def label_rule_version() -> str:
    """``truth.LABEL_RULE_VERSION`` when available, else ``'1'``."""
    module = _load_truth_module()
    version = getattr(module, 'LABEL_RULE_VERSION', None) if module else None
    return str(version) if version is not None else '1'


# -----------------------------------------------------------------------------
# Reconstructing a declared column the caller did not hand us
# -----------------------------------------------------------------------------
#
# ``create_ground_truth_csv`` only *selects*: a measurement that is not in
# ``curves_df`` cannot be recovered here, and pretending otherwise is what the
# audit objected to.  Two narrow exceptions keep that rule honest without
# breaking every caller that predates the schema:
#
# * a **derivation** is a pure function of columns the caller did supply --
#   ``strain_name`` is a formatting of ``curve_id``, ``duration_hours`` is
#   ``n_points * dt`` off the time vector that is right there in the frame.
#   Nothing is invented, so these are always applied.
# * a **fill default** is the documented value of an *annotation* column: the
#   label rule in force, the scoring mode, the requested (not achieved) R^2.
#   These carry no measurement, so a caller that never had the concept gets
#   the neutral value rather than a crash.
#
# Every other declared column -- the measured ``true_*``, the realisation
# ratios, ``actual_r_squared``/``rmse``, the oracle fit, the window times, the
# seed -- is deliberately absent from both tables, so a generator that forgets
# to compute one still gets :class:`GroundTruthSchemaError`.  Whatever is
# derived or filled is reported (printed when ``verbose``, returned in the
# export result, and left on ``writer.last_ground_truth_fill``), and
# ``require_complete=True`` turns any fill at all back into an error.

#: Declared annotation columns and the value used when the caller has none.
#: ``truth.GROUND_TRUTH_FILL_DEFAULTS`` overrides this table once WP3 lands.
GROUND_TRUTH_FILL_DEFAULTS: Dict[str, Any] = {
    'target_r_squared': float('nan'),   # a *requested* R^2; nan == none requested
    'label_rule': 'scenario',
    'scoring_mode': 'standard',
    'lab_band': '',
    'shape': '',
    'noise_sources': '',
    'replicate_A_cv': 0.0,
    'replicate_mu_cv': 0.0,
    'outlier_well': False,
    'spike_indices': '',
    'drift_od_per_h': 0.0,
    'twin_curve_id': '',
}


def ground_truth_fill_defaults() -> Dict[str, Any]:
    """The fill table, with ``truth.GROUND_TRUTH_FILL_DEFAULTS`` winning if present."""
    defaults = dict(GROUND_TRUTH_FILL_DEFAULTS)
    defaults.setdefault('label_rule_version', label_rule_version())
    module = _load_truth_module()
    declared = getattr(module, 'GROUND_TRUTH_FILL_DEFAULTS', None) if module else None
    if declared:
        defaults.update(dict(declared))
    return defaults


def _row_grid(row: pd.Series) -> Optional[Dict[str, Any]]:
    """``grid_of`` for a row that carries a time vector, else ``None``."""
    if 'time' not in row.index:
        return None
    value = row['time']
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return None
    return grid_of(parse_series(value, what='time vector', curve_id=row.get('curve_id')))


def _derive_ground_truth_column(name: str, frame: pd.DataFrame) -> Optional[pd.Series]:
    """
    Rebuild one declared column from columns already in ``frame``.

    Returns ``None`` when the inputs the derivation needs are themselves
    missing, in which case the caller raises rather than guesses.
    """
    have = set(frame.columns)

    if name == 'strain_name' and 'curve_id' in have:
        return frame['curve_id'].apply(lambda x: f"CURVE{int(x):04d}")

    if name in ('n_points', 'time_resolution', 'duration_hours'):
        key = {'n_points': 'n_points', 'time_resolution': 'dt_h',
               'duration_hours': 'duration_h'}[name]
        if 'time' in have:
            values = [(_row_grid(row) or {}).get(key) for _, row in frame.iterrows()]
            if all(v is not None for v in values):
                return pd.Series(values, index=frame.index)
        # no time vector: n_points x dt still pins the third quantity
        if name == 'duration_hours' and {'n_points', 'time_resolution'} <= have:
            return frame['n_points'].astype(float) * frame['time_resolution'].astype(float)
        if name == 'time_resolution' and {'n_points', 'duration_hours'} <= have:
            return frame['duration_hours'].astype(float) / frame['n_points'].astype(float)
        return None

    # v1 frames carry the sampler's request under the ``true_*`` names; the
    # v1 -> v2 reading rule (interface contract B) is exactly this identity.
    if name in ('requested_A', 'requested_mu', 'requested_lambda'):
        source = name.replace('requested_', 'true_')
        if source in have:
            return frame[source]
        return None

    return None


# =============================================================================
# Series parsing and time-grid bookkeeping
# =============================================================================

def parse_series(value: Any, what: str = 'series', curve_id: Any = None) -> np.ndarray:
    """
    Turn a stored time/OD series into a float array.

    Accepts an ndarray, a list/tuple, or the string a CSV round trip leaves
    behind. Strings go through :func:`ast.literal_eval`; the plain-float
    fallback exists only because ``literal_eval`` rejects the ``nan``/``inf``
    tokens that ``list.__repr__`` happily writes. ``eval`` is never used.
    """
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        try:
            return np.asarray(ast.literal_eval(text), dtype=float)
        except (ValueError, SyntaxError, TypeError):
            body = text
            if body.startswith('[') and body.endswith(']'):
                body = body[1:-1]
            elif body.startswith('(') and body.endswith(')'):
                body = body[1:-1]
            parts = [p.strip() for p in body.split(',') if p.strip()]
            try:
                return np.asarray([float(p) for p in parts], dtype=float)
            except ValueError as exc:
                where = '' if curve_id is None else f" of curve_id={curve_id!r}"
                raise ValueError(
                    f"could not parse the {what}{where} from the stored string: {exc}"
                ) from exc
    where = '' if curve_id is None else f" of curve_id={curve_id!r}"
    raise TypeError(f"unsupported {what}{where} of type {type(value).__name__}")


def grid_of(time: Sequence[float]) -> Dict[str, Any]:
    """
    Describe a time vector: ``n_points``, ``dt_h``, ``dt_min`` and ``duration_h``.

    ``duration_h = n_points * dt`` -- the span the run covers, not ``time[-1]``,
    which is one sample short and is what put ``99h`` in the name of a 100 h run.
    """
    t = np.asarray(time, dtype=float)
    n = int(t.size)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    return {
        'n_points': n,
        'dt_h': dt,
        'dt_min': dt * 60.0,
        'duration_h': n * dt,
        't_last': float(t[-1]) if n else float('nan'),
    }


def grid_key(time: Sequence[float]) -> Tuple[int, float]:
    """The bucket key: ``(n_points, round(median(diff(t)), 9))``."""
    g = grid_of(time)
    return (g['n_points'], round(g['dt_h'], 9))


def grid_label(time: Sequence[float], prefix: str = 'SYNTHETIC') -> str:
    """
    ``SYNTHETIC_400pts_100h_15m`` -- point count, rounded duration, rounded
    sampling interval in minutes.

    Dot-free by construction: the label becomes part of a CSV column name and
    of the strain string the pipeline parses.
    """
    g = grid_of(time)
    return (
        f"{prefix}_{g['n_points']}pts"
        f"_{int(round(g['duration_h']))}h"
        f"_{int(round(g['dt_min']))}m"
    )


#: Tolerance for "these curves share a time vector" (absolute, in hours).
GRID_ATOL = 1e-9


def bucket_by_grid(
    curves_df: pd.DataFrame,
    time_column: str = 'time',
) -> List[Tuple[Tuple[int, float], np.ndarray, pd.DataFrame]]:
    """
    Split ``curves_df`` into one bucket per time grid.

    Returns ``[(grid_key, reference_time, sub_frame), ...]`` in first-appearance
    order. Every row in a bucket is checked against the bucket's reference time
    vector with ``atol=GRID_ATOL, rtol=0``; a mismatch raises
    :class:`MixedTimeGridError` naming the offending ``curve_id``. Two grids
    can collide on the key only if they share both the point count and the
    median step, and this guard is what turns that collision into a loud
    failure instead of a silently rescaled time axis.
    """
    order: List[Tuple[int, float]] = []
    reference: Dict[Tuple[int, float], np.ndarray] = {}
    members: Dict[Tuple[int, float], List[int]] = {}

    has_curve_id = 'curve_id' in curves_df.columns
    for pos, (idx, row) in enumerate(curves_df.iterrows()):
        cid = row['curve_id'] if has_curve_id else idx
        t = parse_series(row[time_column], what='time vector', curve_id=cid)
        key = grid_key(t)
        if key not in reference:
            order.append(key)
            reference[key] = t
            members[key] = []
        else:
            ref = reference[key]
            if t.shape != ref.shape or not np.allclose(t, ref, rtol=0, atol=GRID_ATOL):
                first = int(np.argmax(~np.isclose(t, ref, rtol=0, atol=GRID_ATOL))) if t.shape == ref.shape else 0
                raise MixedTimeGridError(
                    f"curve_id={cid!r} has grid key {key} but a different time vector than "
                    f"the other curves in that bucket (first difference at index {first}: "
                    f"{t[first] if first < t.size else float('nan')!r} vs {ref[first]!r}). "
                    "Curves are grouped by (n_points, median dt); two distinct grids sharing "
                    "that key must not be written under one TIME[H] column."
                )
        members[key].append(pos)

    return [(key, reference[key], curves_df.iloc[members[key]]) for key in order]


# =============================================================================
# Manifest / hashing helpers (used by generate_comprehensive.py)
# =============================================================================

def sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(paths: Iterable[Union[str, Path]], root: Union[str, Path]) -> Dict[str, str]:
    """``{path relative to root (posix): sha256}`` for every existing file in ``paths``."""
    root = Path(root)
    out: Dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        try:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = p.name
        out[rel] = sha256_file(p)
    return dict(sorted(out.items()))


def write_manifest(
    out_dir: Union[str, Path],
    payload: Mapping[str, Any],
    filename: str = 'MANIFEST.json',
    hash_paths: Optional[Iterable[Union[str, Path]]] = None,
) -> Path:
    """
    Write ``MANIFEST.json`` beside an export, owning the hashing.

    ``payload`` is copied verbatim (seed, scenario counts, git SHA, the exact
    analysis command, ...). ``file_sha256`` is added unless the caller already
    supplied it: by default every ``DATA/*.csv``, ``PLATES/**/*.csv`` and
    ``ground_truth.csv`` under ``out_dir``, hashed and keyed by their path
    relative to ``out_dir``. Also stamped: ``manifest_schema``,
    ``truth_schema_version`` and ``written_utc``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = dict(payload)
    if 'file_sha256' not in manifest:
        if hash_paths is None:
            candidates: List[Path] = []
            candidates += sorted((out_dir / 'DATA').glob('*.csv'))
            candidates += sorted((out_dir / 'PLATES').rglob('*.csv'))
            gt = out_dir / 'ground_truth.csv'
            if gt.is_file():
                candidates.append(gt)
            hash_paths = candidates
        manifest['file_sha256'] = hash_tree(hash_paths, out_dir)
    manifest.setdefault('manifest_schema', '1')
    manifest.setdefault('truth_schema_version', ground_truth_schema_version())
    manifest.setdefault('written_utc', datetime.now(timezone.utc).isoformat(timespec='seconds'))

    path = out_dir / filename
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    return path


# =============================================================================
# Per-curve TECAN export
# =============================================================================

@dataclass
class CurveExportConfig:
    """Configuration for exporting a curve."""
    curve_id: int
    treatment: str
    strain: str
    expected_class: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class TECANFormatWriter:
    """
    Format synthetic data to match real TECAN output format.
    Output format: TIME[H], {TREATMENT}_{STRAIN}_blanked columns
    """

    def __init__(self, output_dir: str):
        """
        Initialize writer with output directory.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        #: What the last :meth:`create_ground_truth_csv` had to reconstruct.
        self.last_ground_truth_fill: Dict[str, List[str]] = {
            'derived': [], 'filled': [], 'missing': [],
        }

    # ------------------------------------------------------------------ write
    def format_single_treatment(
        self,
        time: np.ndarray,
        curves: Dict[str, np.ndarray],
        treatment_name: str
    ) -> pd.DataFrame:
        """
        Format curves for a single treatment into TECAN-style DataFrame.

        Args:
            time: Time array in hours
            curves: Dict mapping strain_name to OD600 values
            treatment_name: Name of treatment (e.g., 'LB', 'PESTICIDE')

        Returns:
            DataFrame with TIME[H] and {treatment}_{strain}_blanked columns
        """
        # Built in one shot: a 390-column frame assembled column by column is
        # both slow and noisy (pandas PerformanceWarning on every insert).
        data: Dict[str, np.ndarray] = {'TIME[H]': np.asarray(time, dtype=float)}
        for strain_name, od_values in curves.items():
            data[f"{treatment_name}_{strain_name}_blanked"] = np.asarray(od_values, dtype=float)
        return pd.DataFrame(data)

    def write_treatment_csv(
        self,
        time: np.ndarray,
        curves: Dict[str, np.ndarray],
        treatment_name: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Write curves for a single treatment to CSV file.

        Args:
            time: Time array
            curves: Dict mapping strain to OD values
            treatment_name: Treatment name
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        df = self.format_single_treatment(time, curves, treatment_name)

        if filename is None:
            filename = f"{treatment_name}_DATA.csv"

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        return output_path

    # -------------------------------------------------------------- internals
    def _curves_of_bucket(self, group_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """``{strain: od}`` for one time-grid bucket, in row order."""
        curves: Dict[str, np.ndarray] = {}
        for _, row in group_df.iterrows():
            strain = f"CURVE{int(row['curve_id']):04d}"
            if strain in curves:
                raise ValueError(
                    f"curve_id={row['curve_id']!r} appears twice in one time-grid bucket; "
                    "a column name may hold only one curve"
                )
            curves[strain] = parse_series(row['od600'], what='od600 series',
                                          curve_id=row['curve_id'])
        return curves

    # ----------------------------------------------------------------- export
    def export_generated_curves(
        self,
        curves_df: pd.DataFrame,
        group_by: str = 'scenario',
        treatment_prefix: str = 'SYNTHETIC'
    ) -> Dict[str, Path]:
        """
        Export generated curves to TECAN-format CSV files.

        Curves are grouped by ``group_by`` and then, within each group, by time
        grid -- a CSV has exactly one ``TIME[H]`` column, so curves on
        different grids cannot share a file. (The old code took the whole
        group's time vector from its first row.)

        Args:
            curves_df: DataFrame from SyntheticGrowthCurveGenerator.generate_comprehensive_test_set()
            group_by: Column to group curves by ('scenario', 'expected_class', etc.)
            treatment_prefix: Prefix for treatment names

        The grid is encoded in every treatment name, single-grid groups
        included, so a file name can never claim a duration or resolution the
        file does not hold.

        Returns:
            Dict mapping group name to output file path. When a group spans
            several time grids the key is ``f"{group}|{grid_label}"``, one entry
            per file written.
        """
        output_files: Dict[str, Path] = {}

        for group_name, group_df in curves_df.groupby(group_by, sort=False):
            base = f"{treatment_prefix}_{group_name}".upper().replace(' ', '_')
            buckets = bucket_by_grid(group_df)

            for _, time, bucket_df in buckets:
                suffix = grid_label(time, prefix='').lstrip('_')
                treatment = f"{base}_{suffix}".upper()
                key = group_name if len(buckets) == 1 else f"{group_name}|{suffix}"

                output_files[key] = self.write_treatment_csv(
                    time, self._curves_of_bucket(bucket_df), treatment,
                    filename=f"{treatment}_DATA.csv",
                )

        return output_files

    def create_ground_truth_csv(
        self,
        curves_df: pd.DataFrame,
        output_filename: str = 'ground_truth.csv',
        strict: bool = True,
        verbose: bool = True,
        require_complete: bool = False,
    ) -> Path:
        """
        Write ``ground_truth.csv`` with exactly the declared schema, in order.

        The column list comes from :func:`ground_truth_columns` --
        ``truth.GROUND_TRUTH_COLUMNS`` once WP3's ``truth.py`` exists, the v1
        schema of the committed file until then. This function *only selects*:
        the oracle, window and realisation columns have to be computed in the
        generator, because a measurement that is not in ``curves_df`` cannot be
        recovered here.

        Two narrow exceptions keep older callers working without inventing a
        measurement (see :data:`GROUND_TRUTH_FILL_DEFAULTS`):

        * a **derivation** rebuilds a column that is a pure function of columns
          the caller did supply -- ``strain_name`` from ``curve_id``,
          ``duration_hours``/``n_points``/``time_resolution`` from the stored
          time vector, ``requested_*`` from the v1 ``true_*`` names;
        * a **fill default** supplies the neutral value of an *annotation*
          column (``scoring_mode='standard'``, ``target_r_squared=nan``, ...).

        Anything else missing is a :class:`GroundTruthSchemaError`. What was
        derived or filled is recorded on ``self.last_ground_truth_fill`` and
        printed when ``verbose``; nothing is ever dropped silently.

        Args:
            curves_df: DataFrame from generator
            output_filename: Output filename
            strict: raise :class:`GroundTruthSchemaError` on a missing declared
                column that cannot be derived or filled, instead of dropping it
                silently (the pre-audit behaviour, still available as
                ``strict=False``)
            verbose: print the output path and any reconstruction
            require_complete: also raise when a column *could* have been derived
                or filled -- the mode for the comprehensive generator, which is
                supposed to compute every declared column itself

        Returns:
            Path to ground truth CSV
        """
        columns = ground_truth_columns()
        gt_df = curves_df.copy()

        defaults = ground_truth_fill_defaults()
        derived: List[str] = []
        filled: List[str] = []
        unrecoverable: List[str] = []

        for column in columns:
            if column in gt_df.columns:
                continue
            series = _derive_ground_truth_column(column, gt_df)
            if series is not None:
                gt_df[column] = series
                derived.append(column)
            elif column in defaults:
                gt_df[column] = defaults[column]
                filled.append(column)
            else:
                unrecoverable.append(column)

        self.last_ground_truth_fill = {
            'derived': list(derived),
            'filled': list(filled),
            'missing': list(unrecoverable),
        }

        if unrecoverable and strict:
            raise GroundTruthSchemaError(unrecoverable, list(curves_df.columns))
        if require_complete and (derived or filled or unrecoverable):
            raise GroundTruthSchemaError(
                sorted(set(derived) | set(filled) | set(unrecoverable)),
                list(curves_df.columns),
            )
        if unrecoverable:                       # strict=False: the old behaviour
            columns = [c for c in columns if c in gt_df.columns]

        gt_df = gt_df[columns]

        output_path = self.output_dir / output_filename
        gt_df.to_csv(output_path, index=False)

        if verbose:
            print(f"Ground truth saved to: {output_path}")
            if derived:
                print(f"  derived from the columns supplied: {', '.join(derived)}")
            if filled:
                print(f"  filled with the declared default: {', '.join(filled)}")
            if unrecoverable:
                print(f"  DROPPED (strict=False): {', '.join(unrecoverable)}")
        return output_path

    def export_for_analysis_pipeline(
        self,
        curves_df: pd.DataFrame,
        data_subdir: str = 'DATA',
        create_ground_truth: bool = True,
        strict_ground_truth: bool = True,
    ) -> Dict[str, Any]:
        """
        Export curves in a format ready for the analysis pipeline.

        Creates:
        - DATA/ subdirectory with *_DATA.csv files, one per time grid
        - ground_truth.csv with expected results

        Curves with different time resolutions or durations are exported to
        separate CSV files since they cannot share the same TIME[H] column --
        the grid key is ``(n_points, median dt)``, and a bucket whose members
        disagree on the time vector is a :class:`MixedTimeGridError`, not a
        silent rescale.

        Args:
            curves_df: DataFrame from generator
            data_subdir: Subdirectory name for data files
            create_ground_truth: Whether to create ground truth file
            strict_ground_truth: pass through to :meth:`create_ground_truth_csv`

        Returns:
            Dict with paths to created files, a ``grids`` entry describing each
            written grid (n_points, dt_h, duration_h, n_curves) and, when a
            ground truth was written, a ``ground_truth_fill`` entry naming the
            columns that had to be derived or filled.
        """
        data_dir = self.output_dir / data_subdir
        data_dir.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Any] = {
            'data_dir': data_dir,
            'data_files': {},
            'grids': {},
            'ground_truth': None,
        }

        data_writer = TECANFormatWriter(str(data_dir))

        for _, time, bucket_df in bucket_by_grid(curves_df):
            group_name = grid_label(time)
            output_path = data_writer.write_treatment_csv(
                time, self._curves_of_bucket(bucket_df), group_name,
                filename=f"{group_name}_DATA.csv",
            )
            result['data_files'][group_name] = output_path
            grid = grid_of(time)
            grid['n_curves'] = int(len(bucket_df))
            grid['curve_ids'] = [int(c) for c in bucket_df['curve_id']]
            result['grids'][group_name] = grid

        if create_ground_truth:
            result['ground_truth'] = self.create_ground_truth_csv(
                curves_df, strict=strict_ground_truth
            )
            result['ground_truth_fill'] = dict(self.last_ground_truth_fill)

        return result


class BatchExporter:
    """
    Export large batches of synthetic data with proper organization.
    """

    def __init__(self, base_output_dir: str):
        """
        Initialize batch exporter.

        Args:
            base_output_dir: Base directory for all exports
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def export_by_scenario(
        self,
        curves_df: pd.DataFrame,
        scenarios_per_file: int = 1
    ) -> Dict[str, Path]:
        """
        Export curves organized by scenario.

        Creates separate directories for each scenario or group of scenarios.

        Args:
            curves_df: DataFrame from generator
            scenarios_per_file: Number of scenarios to combine per output file

        Returns:
            Dict mapping scenario to output directory
        """
        results = {}

        scenarios = curves_df['scenario'].unique()

        for scenario in scenarios:
            scenario_df = curves_df[curves_df['scenario'] == scenario]

            scenario_dir = self.base_dir / scenario
            writer = TECANFormatWriter(str(scenario_dir))

            writer.export_for_analysis_pipeline(
                scenario_df,
                data_subdir='.',
                create_ground_truth=True
            )

            results[scenario] = scenario_dir

        return results

    def export_comprehensive_test(
        self,
        curves_df: pd.DataFrame,
        test_name: str = 'comprehensive_test'
    ) -> Dict[str, Any]:
        """
        Export comprehensive test set with full organization.

        Creates:
        - {test_name}/
            - DATA/
                - SYNTHETIC_{n}pts_{duration}h_{dt}m_DATA.csv (one per time grid)
            - ground_truth.csv
            - config.json (test configuration)

        Args:
            curves_df: DataFrame from comprehensive generator
            test_name: Name for this test set

        Returns:
            Dict with all paths and metadata
        """
        test_dir = self.base_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        writer = TECANFormatWriter(str(test_dir))

        export_result = writer.export_for_analysis_pipeline(
            curves_df,
            data_subdir='DATA',
            create_ground_truth=True
        )

        config = {
            'test_name': test_name,
            'n_curves': len(curves_df),
            'n_good': len(curves_df[curves_df['expected_class'] == 'GOOD']),
            'n_bad': len(curves_df[curves_df['expected_class'] == 'BAD']),
            'scenarios': curves_df['scenario'].value_counts().to_dict(),
            'data_files': {k: str(v) for k, v in export_result['data_files'].items()},
            'grids': {k: {kk: vv for kk, vv in g.items() if kk != 'curve_ids'}
                      for k, g in export_result['grids'].items()},
            'ground_truth': str(export_result['ground_truth']),
        }

        config_path = test_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        export_result['config'] = config_path
        export_result['test_dir'] = test_dir

        print(f"\nExported comprehensive test to: {test_dir}")
        print(f"  Total curves: {config['n_curves']}")
        print(f"  Expected GOOD: {config['n_good']}")
        print(f"  Expected BAD: {config['n_bad']}")

        return export_result


# =============================================================================
# Convenience Functions
# =============================================================================

def export_quick_test(
    curves_df: pd.DataFrame,
    output_dir: str = 'synthetic_test_output'
) -> Dict[str, Any]:
    """
    Quick export for testing.

    Args:
        curves_df: DataFrame from generator
        output_dir: Output directory

    Returns:
        Export results
    """
    writer = TECANFormatWriter(output_dir)
    return writer.export_for_analysis_pipeline(curves_df)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from data_generator import generate_quick_test_set

    print("Testing output formatter...")

    print("\n1. Generating test data...")
    test_df = generate_quick_test_set(n_per_category=5, seed=42)
    print(f"   Generated {len(test_df)} curves")

    print("\n2. Exporting to TECAN format...")
    exporter = BatchExporter('test_output')
    result = exporter.export_comprehensive_test(test_df, test_name='demo_test')

    print("\n3. Created files:")
    for key, value in result.items():
        if isinstance(value, Path):
            print(f"   {key}: {value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"   {key}/{k}: {v}")

    print("\nDone!")
