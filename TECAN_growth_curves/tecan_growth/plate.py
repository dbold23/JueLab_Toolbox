"""
Plate geometry helpers and the plate *key* (which well holds which media/strain).

A TECAN Spark 96-well export names wells ``A1`` … ``H12`` (row letter, column
number). Lab keys are small tables with the columns ``Cell`` (or ``Well``),
``Media`` and ``Strain``. Blank wells are marked by a strain label such as
``Blank``/``BLANK``; matching is case-insensitive here because real keys in
this project use both spellings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ROWS_96 = "ABCDEFGH"
COLS_96 = list(range(1, 13))
PLATE_96_WELLS: List[str] = [f"{r}{c}" for c in COLS_96 for r in ROWS_96]  # column-major, as TECAN exports them

_WELL_RE = re.compile(r"^\s*([A-Pa-p])\s*0*(\d{1,2})\s*$")

#: Strain labels (lower-cased, stripped) that mean "media-only blank well".
DEFAULT_BLANK_LABELS = ("blank", "blk", "bl", "media", "media only", "media-only", "blank well")

#: Accepted spellings of the key file columns (lower-cased).
KEY_COLUMN_ALIASES = {
    "well": ("cell", "well", "well id", "wellid", "position", "pos"),
    "media": ("media", "medium", "condition", "treatment"),
    "strain": ("strain", "sample", "isolate", "organism", "name"),
}


def normalize_well(label: object) -> Optional[str]:
    """Return a canonical well id like ``'A1'`` for ``'a01'``/``' A1 '``; ``None`` if not a well."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    m = _WELL_RE.match(str(label))
    if not m:
        return None
    row, col = m.group(1).upper(), int(m.group(2))
    if col < 1 or col > 24:
        return None
    return f"{row}{col}"


def well_sort_key(well: str):
    """Sort key giving row-major order (A1, A2, …, B1, …)."""
    w = normalize_well(well)
    if w is None:
        return (99, 99)
    return (ord(w[0]) - ord("A"), int(w[1:]))


def is_blank_label(value: object, blank_labels: Sequence[str] = DEFAULT_BLANK_LABELS) -> bool:
    """True if a strain label denotes a blank (media-only) well, case-insensitively."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {b.lower() for b in blank_labels}


def _find_column(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for a in aliases:
        if a in lowered:
            return lowered[a]
    return None


@dataclass
class KeyValidation:
    """Outcome of checking a key against the wells actually present in the data."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class PlateKey:
    """
    Mapping of wells to (media, strain).

    ``table`` always has the columns ``well, media, strain, is_blank, in_use``
    with one row per well that appears in the source key. Wells whose media
    and strain are both empty are kept with ``in_use=False`` so the plate map
    can show them as unused.
    """
    table: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    source: Optional[str] = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        blank_labels: Sequence[str] = DEFAULT_BLANK_LABELS,
        source: Optional[str] = None,
    ) -> "PlateKey":
        """Build a key from a raw table with (Cell|Well), Media, Strain columns."""
        warnings: List[str] = []
        if df is None or len(df.columns) == 0:
            raise ValueError("Key table is empty")
        well_col = _find_column(df.columns, KEY_COLUMN_ALIASES["well"])
        media_col = _find_column(df.columns, KEY_COLUMN_ALIASES["media"])
        strain_col = _find_column(df.columns, KEY_COLUMN_ALIASES["strain"])
        if well_col is None or media_col is None or strain_col is None:
            # Fall back to positional columns if there are exactly three
            if len(df.columns) == 3:
                well_col, media_col, strain_col = list(df.columns)
                warnings.append(
                    f"Key columns {list(df.columns)} were not recognised by name; "
                    "using them positionally as well, media, strain."
                )
            else:
                raise ValueError(
                    "Key table must have columns Cell/Well, Media and Strain "
                    f"(found: {list(df.columns)})"
                )

        rows = []
        seen: Dict[str, int] = {}
        for i, rec in df.iterrows():
            well = normalize_well(rec[well_col])
            if well is None:
                raw = rec[well_col]
                if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == "":
                    continue  # empty trailing row
                warnings.append(f"Key row {i + 2}: '{raw}' is not a well id (expected e.g. A1..H12); row ignored")
                continue
            media = _clean(rec[media_col])
            strain = _clean(rec[strain_col])
            in_use = bool(media or strain)
            if well in seen:
                warnings.append(f"Key lists well {well} more than once; the last entry is used")
                rows[seen[well]] = None
            seen[well] = len(rows)
            rows.append({
                "well": well,
                "media": media,
                "strain": strain,
                "is_blank": is_blank_label(strain, blank_labels) if in_use else False,
                "in_use": in_use,
            })
        rows = [r for r in rows if r is not None]
        if not rows:
            raise ValueError("Key table contains no usable well rows")
        table = pd.DataFrame(rows)
        # Unify names that differ only by case or inner whitespace ('Bifenthrin ' vs 'bifenthrin')
        for col in ("media", "strain"):
            canon: Dict[str, str] = {}
            merged = set()
            for name in table[col]:
                k = re.sub(r"\s+", " ", name).strip().lower()
                if k and k in canon and canon[k] != name:
                    merged.add(f"'{name}' -> '{canon[k]}'")
                canon.setdefault(k, name)
            table[col] = [canon[re.sub(r"\s+", " ", n).strip().lower()] if n else "" for n in table[col]]
            if merged:
                warnings.append(f"{col.capitalize()} names differing only by case/spacing were merged: " + ", ".join(sorted(merged)))
        # Rows in use must have both media and strain
        half = table[table.in_use & ((table.media == "") | (table.strain == ""))]
        for _, r in half.iterrows():
            warnings.append(
                f"Well {r.well} has media='{r.media}' strain='{r.strain}': both are needed; well treated as unused"
            )
        table.loc[half.index, "in_use"] = False
        table.loc[half.index, "is_blank"] = False
        table = table.sort_values("well", key=lambda s: s.map(well_sort_key)).reset_index(drop=True)
        return cls(table=table, warnings=warnings, source=source)

    @classmethod
    def from_csv(cls, path, **kwargs) -> "PlateKey":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
        return cls.from_dataframe(df, source=str(path), **kwargs)

    # ------------------------------------------------------------- properties
    @property
    def wells(self) -> List[str]:
        return list(self.table.well)

    @property
    def media_names(self) -> List[str]:
        return list(dict.fromkeys(self.table[self.table.in_use].media))

    def conditions(self) -> pd.DataFrame:
        """One row per (media, strain) with the wells it occupies."""
        used = self.table[self.table.in_use]
        grp = used.groupby(["media", "strain", "is_blank"], sort=False)
        out = grp["well"].agg(list).reset_index()
        out["n_wells"] = out["well"].map(len)
        return out.rename(columns={"well": "wells"})

    def wells_for(self, media: str, strain: str) -> List[str]:
        t = self.table
        return list(t[t.in_use & (t.media == media) & (t.strain == strain)].well)

    def blank_wells(self, media: Optional[str] = None) -> List[str]:
        t = self.table[self.table.in_use & self.table.is_blank]
        if media is not None:
            t = t[t.media == media]
        return list(t.well)

    # ------------------------------------------------------------- validation
    def validate(self, wells_in_data: Iterable[str]) -> KeyValidation:
        """Check the key against the well columns that exist in the measurement table."""
        v = KeyValidation(warnings=list(self.warnings))
        data_wells = {normalize_well(w) for w in wells_in_data} - {None}
        used = self.table[self.table.in_use]
        missing = [w for w in used.well if w not in data_wells]
        if missing:
            v.errors.append(
                f"{len(missing)} well(s) in the key have no data column: {', '.join(missing[:12])}"
                + (" …" if len(missing) > 12 else "")
            )
        unkeyed = sorted(data_wells - set(self.table.well), key=well_sort_key)
        if unkeyed:
            v.warnings.append(
                f"{len(unkeyed)} measured well(s) are not in the key and will be ignored: "
                + ", ".join(unkeyed[:12]) + (" …" if len(unkeyed) > 12 else "")
            )
        if used.empty:
            v.errors.append("No wells in the key have a media and strain assigned")
            return v
        for media in self.media_names:
            n_blank = len(self.blank_wells(media))
            n_samples = int((used.media == media).sum()) - n_blank
            if n_blank == 0:
                v.warnings.append(
                    f"Media '{media}' has no blank wells; blanks from the whole plate will be used if any exist, "
                    "otherwise no blank subtraction is applied for this media"
                )
            if n_samples == 0:
                v.warnings.append(f"Media '{media}' has only blank wells and no samples")
        cond = self.conditions()
        singles = cond[(~cond.is_blank) & (cond.n_wells == 1)]
        if len(singles):
            v.warnings.append(
                f"{len(singles)} condition(s) have a single well (no replicates): "
                + ", ".join(f"{m}/{s}" for m, s in zip(singles.media, singles.strain))
            )
        if len(self.blank_wells()) == 0:
            v.warnings.append("The key has no blank wells at all; curves will not be blank-corrected")
        return v

    # ----------------------------------------------------------------- output
    def to_dataframe(self) -> pd.DataFrame:
        """Legacy 3-column layout (Cell, Media, Strain) for saving a key file."""
        t = self.table
        return pd.DataFrame({"Cell": t.well, "Media": t.media, "Strain": t.strain})

    def plate_grid(self, value: str = "label") -> pd.DataFrame:
        """8x12 (or larger) grid DataFrame for display; ``value`` is 'label', 'media' or 'strain'."""
        wells = list(self.table.well)
        rows = sorted({w[0] for w in wells} | set(ROWS_96))
        cols = sorted({int(w[1:]) for w in wells} | set(COLS_96))
        grid = pd.DataFrame("", index=rows, columns=cols)
        for _, r in self.table.iterrows():
            if not r.in_use:
                continue
            if value == "media":
                txt = r.media
            elif value == "strain":
                txt = r.strain
            else:
                txt = f"{r.media} / {r.strain}"
            grid.loc[r.well[0], int(r.well[1:])] = txt
        return grid


#: Tokens (lower-cased) that mean "this well is not used".
UNUSED_TOKENS = ("", "nan", "none", "na", "n/a", "n.a.", "-", "--", "empty", "unused", "null")


def _clean(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in UNUSED_TOKENS else s
