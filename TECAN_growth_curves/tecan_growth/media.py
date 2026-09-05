"""
What is in the well: turning a media name into a treatment type.

Lab keys name the media by what was added to the well, for example ``LB``
(rich broth, the positive control), ``H2O`` (water, the negative control),
``MALATHION`` (pesticide in water) or ``MALATHIONANDLB`` (the same pesticide
with broth added). Comparing plates and groups needs those four *kinds* of
well rather than the raw text, so this module maps any spelling of a media
name onto one of :data:`TREATMENT_TYPES` plus the canonical pesticide name.

It also holds the colour assignment used by every figure: a strain (or a
media) keeps the same colour on every panel of the app, because colours are
handed out in a fixed order on first appearance and never recycled.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd

#: The four kinds of well the lab compares, in the order they are plotted.
TREATMENT_TYPES: List[str] = ["LB", "H2O", "Pesticide", "Pesticide+LB"]

#: Canonical spelling of every pesticide used in this project, keyed by the
#: media name with spaces, hyphens and case removed. Same list as
#: ``scripts/03_statistical_analysis.PESTICIDE_MAP``.
PESTICIDE_MAP: Dict[str, str] = {
    "BIFENTHRIN": "Bifenthrin",
    "FLUPYRADIFURONE": "Flupyradifurone",
    "LAMBDACYHALOTHRIN": "Lambda-Cyhalothrin",
    "MALATHION": "Malathion",
    "IMIDACLOPRID": "Imidacloprid",
    "PERMETHRIN": "Permethrin",
}

#: Media names that mean "rich broth only" (the positive control).
LB_NAMES = ("LB", "LURIABERTANI", "LBBROTH")

#: Media names that mean "water only" (the negative control).
WATER_NAMES = ("H2O", "H20", "WATER", "DIH2O", "DIWATER", "MILLIQ")

#: The eight categorical colours, in the order they are handed out.
CATEGORICAL_COLORS: List[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

#: Colour for anything past the eighth category ("Other").
OTHER_COLOR = "#898781"

# The trailing "+ LB" of a media name: 'MALATHIONANDLB', 'Malathion+LB',
# 'Malathion LB', 'malathion and lb' all mean pesticide plus broth.
_PLUS_LB_RE = re.compile(r"^(?P<base>.+?)[\s+_/&-]*(?:AND\s*)?LB$", re.IGNORECASE)
# The same thing written the other way round: 'LB+Malathion', 'LB and malathion'.
_LB_PLUS_RE = re.compile(r"^LB[\s+_/&-]+(?:AND\s+)?(?P<base>.+)$", re.IGNORECASE)


def _squash(name: str) -> str:
    """Upper-case a name and drop everything that is not a letter or digit."""
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def _tidy(name: str) -> str:
    """Title-case a raw name while keeping its hyphens ('lambda-cy' -> 'Lambda-Cy')."""
    cleaned = re.sub(r"[_\s]+", " ", str(name).strip())
    return re.sub(r"[A-Za-z0-9]+", lambda m: m.group(0).capitalize(), cleaned)


def canonical_pesticide(name: str) -> str:
    """
    Canonical spelling of a pesticide, e.g. ``'LAMBDACYHALOTHRIN'`` ->
    ``'Lambda-Cyhalothrin'``. Unknown names are tidied to title case.
    """
    return PESTICIDE_MAP.get(_squash(name), _tidy(name))


def classify_media(media: str) -> Dict[str, object]:
    """
    Describe what a well contained.

    Parameters
    ----------
    media : str
        The media name from the plate key, in any spelling: ``LB``, ``H2O``,
        ``MALATHION``, ``MALATHIONANDLB``, ``Malathion+LB``, ``malathion lb``.

    Returns
    -------
    dict
        ``treatment_type`` — one of :data:`TREATMENT_TYPES`;
        ``pesticide`` — canonical pesticide name, or ``None`` for the two controls;
        ``has_nutrients`` — ``True`` when the well contained LB broth.
    """
    raw = "" if media is None else str(media).strip()
    squashed = _squash(raw)
    if squashed in LB_NAMES:
        return {"treatment_type": "LB", "pesticide": None, "has_nutrients": True}
    if squashed in WATER_NAMES:
        return {"treatment_type": "H2O", "pesticide": None, "has_nutrients": False}

    match = _PLUS_LB_RE.match(raw) or _LB_PLUS_RE.match(raw)
    if match and _squash(match.group("base")):
        return {
            "treatment_type": "Pesticide+LB",
            "pesticide": canonical_pesticide(match.group("base")),
            "has_nutrients": True,
        }
    return {
        "treatment_type": "Pesticide",
        "pesticide": canonical_pesticide(raw),
        "has_nutrients": False,
    }


def annotate(table: pd.DataFrame, media_col: str = "media") -> pd.DataFrame:
    """
    Add ``treatment_type``, ``pesticide`` and ``has_nutrients`` columns to a
    results table. Returns a copy; the input is untouched.
    """
    out = table.copy()
    if media_col not in out.columns:
        raise KeyError(f"Column '{media_col}' is not in the table (found: {list(out.columns)})")
    parsed = [classify_media(m) for m in out[media_col]]
    out["treatment_type"] = [p["treatment_type"] for p in parsed]
    out["pesticide"] = [p["pesticide"] for p in parsed]
    out["has_nutrients"] = [p["has_nutrients"] for p in parsed]
    return out


def _slot_colors(names: Sequence[str]) -> Dict[str, str]:
    """Hand out the categorical colours in first-seen order; extras get grey."""
    out: Dict[str, str] = {}
    for name in names:
        label = "" if name is None else str(name)
        if label in out:
            continue
        out[label] = CATEGORICAL_COLORS[len(out)] if len(out) < len(CATEGORICAL_COLORS) else OTHER_COLOR
    return out


def strain_color_map(strains: Sequence[str]) -> Dict[str, str]:
    """
    Colour for every strain, so a strain looks the same on every figure.

    Colours are assigned in the order the strains are first seen (sort the
    input first if a stable order across runs matters). A ninth strain and
    beyond share the neutral grey rather than repeating a colour.
    """
    return _slot_colors(strains)


def media_color_map(medias: Sequence[str]) -> Dict[str, str]:
    """Colour for every media name, using the same fixed slots as strains."""
    return _slot_colors(medias)


def treatment_order(values: Sequence[str]) -> List[str]:
    """The treatment types present in ``values``, in :data:`TREATMENT_TYPES` order."""
    present = {str(v) for v in values}
    known = [t for t in TREATMENT_TYPES if t in present]
    return known + sorted(present - set(TREATMENT_TYPES))


def treatment_label(media: str) -> str:
    """
    Short human label for a media, e.g. ``'Malathion + LB'`` or ``'LB (control)'``.
    Used for axis ticks and legends.
    """
    info = classify_media(media)
    if info["treatment_type"] == "LB":
        return "LB (control)"
    if info["treatment_type"] == "H2O":
        return "H2O (control)"
    pesticide: Optional[str] = info["pesticide"]  # type: ignore[assignment]
    return f"{pesticide} + LB" if info["treatment_type"] == "Pesticide+LB" else str(pesticide)
