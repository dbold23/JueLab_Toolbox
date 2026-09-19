"""
Plate-shaped export of synthetic curves (``VALUES.csv`` + ``KEY.csv``).

Why this exists
---------------
The default per-curve export writes one already-blanked column per curve
straight into ``test_data/DATA/``, and ``scripts/01_growth_curve_analysis.py``
consumes those columns directly. The whole preprocessing stage of the real
pipeline -- per-media blank selection and subtraction, replicate averaging,
negative clipping, ``od_sd``/``n_replicates`` (``tecan_growth/preprocess.py``)
-- is therefore never executed by the benchmark, while a single divergent well
is the commonest real artefact:

    Group 4 PERMETHRINANDLB-CISPERM1 per-well blanked maxima [0.016, 1.063, 0.010]
        -- one well of three grew; the averaged trace is a smooth 0.345 OD
        sigmoid the pipeline calls GOOD.
    Group 4 PERMETHRINANDLB-CYPERM2                          [0.036, 1.009, 1.240]
    Group 3 LAMBDACYHALOTHRINANDLB-LCY2                      [1.153, 1.409, 1.972]
    Group 3 H2O-IMID5                                        [0.053, 0.011, 0.003]

    40 % of Group 4 and 29 % of Group 3 conditions have relative well spread
    > 0.3 (8/20 and 7/24).

This module emits the same two files a lab plate arrives as, so a synthetic run
can be pushed through ``scripts/02_preprocess_raw_plate_data.py`` (or
``tecan_growth.load_plate`` + ``preprocess_plate``) before it reaches the
analysis script -- the same path the 92 real conditions take.

Format, copied from ``data/raw/Group 3``
----------------------------------------
``VALUES.csv``  ``Cycle Nr.,Time [s],Temp. [C],A1,B1,...,H12`` -- wells in
    TECAN column-major order, cycle numbers 1..n, **time in seconds**
    (``tecan_growth.io._time_to_hours`` reads the unit off the header, and
    ``01_growth_curve_analysis.py`` treats a >1000 first value as seconds).
``KEY.csv``     ``Cell,Media,Strain`` with ``BLANK`` in Strain for the
    media-only wells, exactly as ``GROUP3_KEY.csv``.

One plate per (scenario, time grid): a plate has a single time vector, and
scenarios differ in grid. A scenario with more conditions than fit on a
96-well plate is split over several plates. Every plate gets its own media
token (``SCEN01``, ``SCEN02``, ...) and every condition keeps its curve id as
the strain (``CURVE0007``), so ``ProcessedPlate.write_legacy_csvs`` produces
the column ``SCEN01_CURVE0007_blanked`` -- and both consumers of that name
(``synthetic_data/validation/pipeline_validator.py:177`` and
``scripts/09_train_classifier.py:356``) pull the id out with the same
``(CURVE\\d+)`` regex, so ``ground_truth.csv`` and the join key are unchanged.

What this module will and will not invent
-----------------------------------------
It writes the wells it is given and nothing else.

* ``well_od600`` (a sequence of per-well traces, or its string repr) is used
  when the generator supplies it -- that is the replicate structure WP3
  produces. ``blank_od600`` likewise supplies that media's blank wells.
* With neither present, a condition becomes **one** sample well carrying
  ``od600`` verbatim, and the blank wells are written as a constant
  ``blank_od`` (0.0 by default). Blank subtraction then runs as an exact
  identity, so the plate path reproduces the per-curve path curve for curve
  while still exercising blank selection, replicate averaging and the negative
  clip. Replicates are never faked by duplicating a trace: three identical
  wells would claim a replicate spread of exactly zero.
* Wells with no condition assigned are omitted from both files rather than
  filled with an invented reading; ``Temp. [C]`` is a constant placeholder
  (nothing downstream reads it).

The per-curve writer in :mod:`output_formatter` stays the default; this one is
opt-in (``generate_comprehensive.py --emit-plate``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # imported as synthetic_data.src.plate_writer / src.plate_writer
    from .output_formatter import bucket_by_grid, grid_of, parse_series
except ImportError:  # imported as a bare module (synthetic_data/src on sys.path)
    from output_formatter import bucket_by_grid, grid_of, parse_series  # type: ignore


# 96-well geometry ------------------------------------------------------------
PLATE_ROWS = "ABCDEFGH"
PLATE_COLS = tuple(range(1, 13))

#: Row-major (A1, A2, ... A12, B1, ...) -- how lab keys are written.
WELLS_ROW_MAJOR: Tuple[str, ...] = tuple(f"{r}{c}" for r in PLATE_ROWS for c in PLATE_COLS)
#: Column-major (A1, B1, ... H1, A2, ...) -- how TECAN exports the value columns.
WELLS_COL_MAJOR: Tuple[str, ...] = tuple(f"{r}{c}" for c in PLATE_COLS for r in PLATE_ROWS)

VALUES_META_COLUMNS: Tuple[str, str, str] = ("Cycle Nr.", "Time [s]", "Temp. [C]")
BLANK_STRAIN = "BLANK"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class PlateLayoutError(ValueError):
    """Raised when the requested plate layout cannot hold a single condition."""


def _as_well_traces(value: Any, n_points: int, what: str, curve_id: Any) -> Optional[List[np.ndarray]]:
    """Parse an optional per-well bundle into a list of length-``n_points`` traces."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        arr = parse_series(value, what=what, curve_id=curve_id)
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
    elif isinstance(value, np.ndarray):
        arr = np.atleast_2d(value.astype(float, copy=False))
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        arr = np.atleast_2d(np.asarray([np.asarray(v, dtype=float) for v in value], dtype=float))
    else:
        raise TypeError(f"unsupported {what} of type {type(value).__name__} (curve_id={curve_id!r})")

    if arr.ndim != 2:
        raise ValueError(f"{what} of curve_id={curve_id!r} is not a 2-D bundle of well traces")
    if arr.shape[1] != n_points and arr.shape[0] == n_points:
        arr = arr.T
    if arr.shape[1] != n_points:
        raise ValueError(
            f"{what} of curve_id={curve_id!r} has {arr.shape[1]} samples per well, "
            f"but the curve's time vector has {n_points}"
        )
    return [np.asarray(row, dtype=float) for row in arr]


class PlateFormatWriter:
    """
    Write synthetic curves as a set of TECAN plates.

    Parameters
    ----------
    output_dir
        The export root (the directory that also holds ``DATA/`` and
        ``ground_truth.csv``).
    plates_subdir
        Sub-directory for the plates; each plate is ``plate_NN/`` inside it.
    n_blank_wells
        Blank (media-only) wells per plate, 3 like ``GROUP3_KEY.csv``
        (3-6 is the real range). ``0`` writes a plate with no blanks, which
        ``preprocess_plate`` handles with a warning and no subtraction.
    blank_od
        The constant written into synthesised blank wells when the generator
        supplies none. ``0.0`` makes blank subtraction an exact identity, so
        the plate path reproduces the per-curve path; a non-zero value would be
        an uncalibrated invention.
    media_prefix
        Media token stem; plate *k* gets ``SCEN{k:02d}``.
    temperature_c
        Constant placeholder for the ``Temp. [C]`` column.
    float_format
        ``%.17g`` round-trips every float exactly, matching
        ``ProcessedPlate.write_legacy_csvs``.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        plates_subdir: str = 'PLATES',
        n_blank_wells: int = 3,
        blank_od: float = 0.0,
        media_prefix: str = 'SCEN',
        temperature_c: float = 37.0,
        float_format: str = '%.17g',
        well_order: Sequence[str] = WELLS_ROW_MAJOR,
    ):
        self.output_dir = Path(output_dir)
        self.plates_dir = self.output_dir / plates_subdir if plates_subdir else self.output_dir
        self.n_blank_wells = int(n_blank_wells)
        self.blank_od = float(blank_od)
        self.media_prefix = media_prefix
        self.temperature_c = float(temperature_c)
        self.float_format = float_format
        self.well_order = list(well_order)
        if self.n_blank_wells < 0 or self.n_blank_wells >= len(self.well_order):
            raise PlateLayoutError(
                f"n_blank_wells={self.n_blank_wells} does not leave room for a sample well "
                f"on a {len(self.well_order)}-well plate"
            )

    # ------------------------------------------------------------ well layout
    def _wells_per_condition(self, bucket_df: pd.DataFrame, n_points: int) -> Dict[Any, int]:
        """How many wells each curve in the bucket needs (1 unless it carries replicates)."""
        counts: Dict[Any, int] = {}
        for _, row in bucket_df.iterrows():
            cid = row['curve_id']
            if cid in counts:
                raise PlateLayoutError(
                    f"curve_id={cid!r} appears twice; a strain may occupy only one "
                    "set of wells on a plate"
                )
            counts[cid] = len(self._sample_traces(row, n_points))
        return counts

    @staticmethod
    def _sample_traces(row: pd.Series, n_points: int) -> List[np.ndarray]:
        """The per-well sample traces of one condition."""
        for column in ('well_od600', 'replicate_od600', 'well_od_raw'):
            if column in row.index:
                traces = _as_well_traces(row[column], n_points, column, row.get('curve_id'))
                if traces:
                    return traces
        return [parse_series(row['od600'], what='od600 series', curve_id=row.get('curve_id'))]

    def _blank_traces(self, bucket_df: pd.DataFrame, n_points: int) -> Tuple[List[np.ndarray], str]:
        """
        The plate's blank wells and where they came from.

        ``('generator', ...)`` when the frame carries a blank bundle,
        ``('constant', ...)`` for the synthesised identity blanks.
        """
        for column in ('blank_od600', 'blank_od_raw'):
            if column not in bucket_df.columns:
                continue
            for _, row in bucket_df.iterrows():
                traces = _as_well_traces(row[column], n_points, column, row.get('curve_id'))
                if traces:
                    return traces, 'generator'
        return [np.full(n_points, self.blank_od) for _ in range(self.n_blank_wells)], 'constant'

    # ---------------------------------------------------------------- writing
    def write_values_csv(
        self,
        path: Union[str, Path],
        time_h: np.ndarray,
        wells: Dict[str, np.ndarray],
    ) -> Path:
        """Write one ``VALUES.csv``: cycle number, time in seconds, temperature, one column per well."""
        time_h = np.asarray(time_h, dtype=float)
        n = time_h.size
        frame = pd.DataFrame({
            VALUES_META_COLUMNS[0]: np.arange(1, n + 1, dtype=int),
            VALUES_META_COLUMNS[1]: time_h * 3600.0,
            VALUES_META_COLUMNS[2]: np.full(n, self.temperature_c),
        })
        for well in [w for w in WELLS_COL_MAJOR if w in wells]:
            frame[well] = np.asarray(wells[well], dtype=float)
        path = Path(path)
        frame.to_csv(path, index=False, float_format=self.float_format)
        return path

    @staticmethod
    def write_key_csv(path: Union[str, Path], rows: Sequence[Tuple[str, str, str]]) -> Path:
        """Write one ``KEY.csv`` (``Cell,Media,Strain``), rows in plate order."""
        path = Path(path)
        pd.DataFrame(list(rows), columns=['Cell', 'Media', 'Strain']).to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------- main
    def write_plates(
        self,
        curves_df: pd.DataFrame,
        group_by: str = 'scenario',
        index_filename: str = 'plates_index.json',
    ) -> Dict[str, Any]:
        """
        Emit one plate per (``group_by`` value, time grid), split across plates
        when a group does not fit on 96 wells.

        Returns
        -------
        dict
            ``{'plates_dir', 'index', 'plates': [ ... ]}`` where each plate entry
            carries ``plate_id``, ``dir``, ``values_csv``, ``key_csv``,
            ``media``, ``scenario``, ``grid``, ``blank_wells`` and
            ``conditions`` (curve id, strain, wells).
        """
        self.plates_dir.mkdir(parents=True, exist_ok=True)

        plates: List[Dict[str, Any]] = []
        plate_no = 0

        groups: Iterable[Tuple[Any, pd.DataFrame]]
        if group_by and group_by in curves_df.columns:
            groups = curves_df.groupby(group_by, sort=False)
        else:
            groups = [(None, curves_df)]

        for group_name, group_df in groups:
            for _, time_h, bucket_df in bucket_by_grid(group_df):
                n_points = int(np.asarray(time_h).size)
                needed = self._wells_per_condition(bucket_df, n_points)
                capacity = len(self.well_order) - self.n_blank_wells
                widest = max(needed.values()) if needed else 1
                if widest > capacity:
                    raise PlateLayoutError(
                        f"a condition needs {widest} wells but only {capacity} are free after "
                        f"{self.n_blank_wells} blank well(s) on a {len(self.well_order)}-well plate"
                    )

                chunk: List[Any] = []
                used = 0
                for cid in needed:
                    if used + needed[cid] > capacity and chunk:
                        plate_no += 1
                        plates.append(self._write_one_plate(
                            plate_no, group_name, time_h,
                            bucket_df[bucket_df['curve_id'].isin(chunk)],
                        ))
                        chunk, used = [], 0
                    chunk.append(cid)
                    used += needed[cid]
                if chunk:
                    plate_no += 1
                    plates.append(self._write_one_plate(
                        plate_no, group_name, time_h,
                        bucket_df[bucket_df['curve_id'].isin(chunk)],
                    ))

        index_path = self.plates_dir / index_filename
        with open(index_path, 'w') as fh:
            json.dump({'n_plates': len(plates), 'plates': plates}, fh, indent=2, default=str)
            fh.write('\n')

        return {'plates_dir': self.plates_dir, 'index': index_path, 'plates': plates}

    # -------------------------------------------------------------- one plate
    def _write_one_plate(
        self,
        plate_no: int,
        scenario: Any,
        time_h: np.ndarray,
        bucket_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        n_points = int(np.asarray(time_h).size)
        media = f"{self.media_prefix}{plate_no:02d}"
        plate_dir = self.plates_dir / f"plate_{plate_no:02d}"
        plate_dir.mkdir(parents=True, exist_ok=True)

        wells: Dict[str, np.ndarray] = {}
        key_rows: List[Tuple[str, str, str]] = []
        cursor = 0

        blank_traces, blank_source = self._blank_traces(bucket_df, n_points)
        blank_traces = blank_traces[: self.n_blank_wells]
        blank_wells: List[str] = []
        for trace in blank_traces:
            well = self.well_order[cursor]
            cursor += 1
            wells[well] = trace
            key_rows.append((well, media, BLANK_STRAIN))
            blank_wells.append(well)

        conditions: List[Dict[str, Any]] = []
        for _, row in bucket_df.iterrows():
            cid = int(row['curve_id'])
            strain = f"CURVE{cid:04d}"
            traces = self._sample_traces(row, n_points)
            assigned: List[str] = []
            for trace in traces:
                well = self.well_order[cursor]
                cursor += 1
                wells[well] = trace
                key_rows.append((well, media, strain))
                assigned.append(well)
            conditions.append({'curve_id': cid, 'strain': strain, 'wells': assigned,
                               'n_wells': len(assigned)})

        values_csv = self.write_values_csv(plate_dir / 'VALUES.csv', time_h, wells)
        key_csv = self.write_key_csv(plate_dir / 'KEY.csv', key_rows)

        grid = grid_of(time_h)
        return {
            'plate_id': f"plate_{plate_no:02d}",
            'dir': str(plate_dir),
            'values_csv': str(values_csv),
            'key_csv': str(key_csv),
            'media': media,
            'scenario': None if scenario is None else str(scenario),
            'grid': grid,
            'n_wells': len(wells),
            'blank_wells': blank_wells,
            'blank_source': blank_source,
            'conditions': conditions,
            'legacy_data_csv': f"{_SAFE.sub('_', media)}_DATA.csv",
        }


def preprocess_written_plates(
    plates: Sequence[Dict[str, Any]],
    out_dir: Union[str, Path],
    verbose: bool = False,
) -> List[Path]:
    """
    Convenience: run every written plate through the real preprocessing stage
    (``tecan_growth.load_plate`` + ``preprocess_plate``) into one ``DATA``
    directory of ``{MEDIA}_DATA.csv`` files.

    Kept here so the plate path can be exercised without importing the CLI
    script; ``scripts/02_preprocess_raw_plate_data.py`` does exactly the same
    thing and is what the pipeline actually runs.
    """
    from tecan_growth.io import load_plate
    from tecan_growth.preprocess import preprocess_plate

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for plate in plates:
        run = load_plate(plate['values_csv'], plate['key_csv'])
        processed = preprocess_plate(run)
        if verbose:
            for msg in processed.qc:
                print(f"  QC [{plate['plate_id']}]: {msg}")
        written += processed.write_legacy_csvs(out_dir)
    return written
