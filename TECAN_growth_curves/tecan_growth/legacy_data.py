"""
Look up a legacy ``*_DATA.csv`` curve by the strain name the pipeline uses.

``scripts/01_growth_curve_analysis.py`` names every curve after the column it
came from::

    strain_name = column.replace('_blanked', '').replace('_', '-')

so ``LB_MAL1_blanked`` becomes ``LB-MAL1``. The downstream scripts (05 Haldane,
06 advanced fitting, and 07/08 which import from 06) have to walk *back* from
that name to the column, and they used to do it by scanning files in glob order
and accepting a substring "partial match" (``prefix in column and suffix in
column``). That is what made ``LB-MAL1`` answer with
``MALATHIONANDLB_MAL10_blanked`` and ``Bifenthrin-BIF2`` answer with
``BifenthrinANDLB_BIF2_blanked``: 33 of 92 strains were analysed from another
strain's — or another medium's — column.

The lookup here indexes *every* column of *every* ``*_DATA.csv`` in the group
first, then resolves the name against that index:

1. an exact (case-insensitive) hit on the normalised column name, and only if
   nothing anywhere matches exactly,
2. a token-anchored regular expression built from the query's own parts
   (``^LB[-_. ]+MAL1$``), which tolerates a different separator but can never
   match a longer name — and which is used only when it is unambiguous.

There is no substring fall-through, so a miss is reported as a miss instead of
silently returning the wrong curve.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "CurveRef",
    "normalise_curve_name",
    "build_curve_index",
    "find_curve_ref",
    "find_curve",
    "clear_cache",
]

# Same rule 01 uses to spot the time column (anchored, so a strain column that
# merely contains "time" cannot be mistaken for it).
_TIME_RE = re.compile(r"^\s*time\b", re.IGNORECASE)

# Separators a hand-edited export might use between the media and strain token.
_SEP_CLASS = r"[-_. ]+"


class CurveRef(NamedTuple):
    """Where one curve lives: which file, which column, and its pipeline name."""

    name: str          # normalised name, exactly as 01 would derive it
    file: Path
    column: str
    time_column: str


def normalise_curve_name(column: object) -> str:
    """The strain name 01 gives a column: ``_blanked`` stripped, ``_`` -> ``-``."""
    return str(column).replace("_blanked", "").replace("_", "-")


def _time_column(columns) -> Optional[str]:
    for col in columns:
        if _TIME_RE.match(str(col)):
            return col
    return None


# Index cache: {resolved group dir: (signature, index)}. The signature is the
# sorted (name, mtime, size) of the CSVs, so an edited or added file rebuilds.
_CACHE: Dict[str, Tuple[tuple, Dict[str, CurveRef]]] = {}


def clear_cache() -> None:
    """Forget every cached index (used by the tests)."""
    _CACHE.clear()


def _signature(files: List[Path]) -> tuple:
    sig = []
    for f in files:
        try:
            st = f.stat()
            sig.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((f.name, None, None))
    return tuple(sig)


def build_curve_index(group_dir: Union[str, Path]) -> Dict[str, CurveRef]:
    """
    Index every curve column under ``group_dir``, keyed by upper-cased name.

    Files are visited in sorted order so the index does not depend on the
    order the filesystem happens to return; if two files somehow expose the
    same normalised name the first one wins, deterministically.
    """
    group_path = Path(group_dir)
    if not group_path.exists():
        return {}

    files = sorted(group_path.glob("*_DATA.csv"))
    sig = _signature(files)
    cached = _CACHE.get(str(group_path.resolve()))
    if cached is not None and cached[0] == sig:
        return cached[1]

    index: Dict[str, CurveRef] = {}
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue
        time_col = _time_column(df.columns)
        if time_col is None:
            continue
        for col in df.columns:
            if col == time_col:
                continue
            name = normalise_curve_name(col)
            key = name.upper()
            if key in index:
                continue  # first file in sorted order wins
            index[key] = CurveRef(
                name=name, file=csv_path, column=col, time_column=time_col
            )

    _CACHE[str(group_path.resolve())] = (sig, index)
    return index


def find_curve_ref(
    group_dir: Union[str, Path], strain_name: str
) -> Optional[CurveRef]:
    """
    Resolve ``strain_name`` to the column it came from, or ``None``.

    Exact (case-insensitive) match first; a token-anchored regex only when no
    exact match exists anywhere in the group and exactly one column matches.
    """
    index = build_curve_index(group_dir)
    if not index:
        return None

    query = normalise_curve_name(strain_name)
    hit = index.get(query.upper())
    if hit is not None:
        return hit

    # Token-anchored fallback: the query's own parts, in order, anchored at both
    # ends. '^LB[-_. ]+MAL1$' matches 'LB_MAL1' but never 'LB_MAL10' or
    # 'MALATHIONANDLB_MAL1'.
    parts = [p for p in query.split("-") if p]
    if len(parts) < 2:
        return None
    pattern = re.compile(
        "^" + _SEP_CLASS.join(re.escape(p) for p in parts) + "$", re.IGNORECASE
    )
    matches = [
        ref
        for ref in index.values()
        if pattern.match(ref.name) or pattern.match(str(ref.column))
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def find_curve(
    group_dir: Union[str, Path], strain_name: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Path], Optional[str]]:
    """
    Load one curve by strain name.

    Returns ``(time, od, file, column)``; ``(None, None, None, None)`` when the
    strain has no column in this group. Values are coerced to float (unparsable
    cells become NaN) and are *not* NaN-filtered — callers mask them exactly as
    they did before. Time in seconds is converted to hours with the same rule
    01 applies, so downstream time axes match the main pipeline's.
    """
    ref = find_curve_ref(group_dir, strain_name)
    if ref is None:
        return None, None, None, None

    try:
        df = pd.read_csv(ref.file)
        time = pd.to_numeric(df[ref.time_column], errors="coerce").to_numpy(dtype=float)
        od = pd.to_numeric(df[ref.column], errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None, None, None, None

    if len(time) and np.nanmax(time) > 1000:
        time = time / 3600.0

    return time, od, ref.file, ref.column
