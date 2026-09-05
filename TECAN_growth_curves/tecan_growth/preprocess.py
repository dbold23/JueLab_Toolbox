"""
Blank subtraction and replicate averaging.

For every media on the plate the blank wells (media only, no cells) are
averaged at each time point and subtracted from every sample well of that
media. Replicate wells of the same (media, strain) are then averaged. This is
exactly what the lab's original notebook and ``scripts/02_preprocess_raw_plate_data.py``
do, with two deliberate additions:

* blank labels are matched case-insensitively (``Blank`` == ``BLANK``), and
* wells without a media/strain assignment are ignored instead of producing a
  ``nan`` media.

The default keeps the legacy behaviour of clipping negative blanked values to
zero (``clip_negative=True``) so results reproduce the committed data sets;
set it to ``False`` to keep the physically meaningful negative noise.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io import PlateRun
from .plate import PlateKey

LEGACY_TIME_COL = "TIME[H]"


def curve_id(media: str, strain: str) -> str:
    """Identifier used throughout the pipeline for a (media, strain) curve, e.g. ``'LB-MAL8'``."""
    return f"{media}-{strain}"


def legacy_column(media: str, strain: str) -> str:
    """Column name in the legacy ``{MEDIA}_DATA.csv`` files."""
    return f"{media}_{strain}_blanked"


@dataclass
class ProcessedPlate:
    """
    Blank-corrected, replicate-averaged curves plus everything needed to show
    the intermediate steps.
    """
    time_h: np.ndarray
    curves: pd.DataFrame        # tidy: curve_id, media, strain, time_h, od, od_raw_mean, od_sd, n_replicates, blank_mean
    replicates: pd.DataFrame    # tidy: well, media, strain, curve_id, time_h, od_raw, od_blanked
    blanks: pd.DataFrame        # tidy: media, time_h, blank_mean, blank_sd, n_blank_wells, blank_source
    conditions: pd.DataFrame    # one row per curve: curve_id, media, strain, wells, n_replicates, blank_source, n_blank_wells
    qc: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    run_summary: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ accessors
    @property
    def curve_ids(self) -> List[str]:
        return list(self.conditions.curve_id)

    def curve(self, cid: str) -> Tuple[np.ndarray, np.ndarray]:
        """(time_h, od_blanked) for one curve id."""
        c = self.curves[self.curves.curve_id == cid]
        if c.empty:
            raise KeyError(cid)
        return c.time_h.to_numpy(), c.od.to_numpy()

    def wide(self, media: Optional[str] = None) -> pd.DataFrame:
        """Legacy wide layout: ``TIME[H]`` plus one ``{MEDIA}_{STRAIN}_blanked`` column per curve."""
        cond = self.conditions if media is None else self.conditions[self.conditions.media == media]
        out = pd.DataFrame({LEGACY_TIME_COL: self.time_h})
        for _, r in cond.iterrows():
            _, od = self.curve(r.curve_id)
            out[legacy_column(r.media, r.strain)] = od
        return out

    def write_legacy_csvs(self, out_dir: os.PathLike) -> List[Path]:
        """Write one ``{MEDIA}_DATA.csv`` per media (the format the CLI scripts read)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for media in dict.fromkeys(self.conditions.media):
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", media)
            p = out_dir / f"{safe}_DATA.csv"
            self.wide(media).to_csv(p, index=False)
            paths.append(p)
        return paths


def preprocess_plate(
    run: PlateRun,
    key: Optional[PlateKey] = None,
    clip_negative: bool = True,
    blank_mode: str = "per_media",
    replicate_stat: str = "mean",
) -> ProcessedPlate:
    """
    Blank-subtract and average replicates.

    Parameters
    ----------
    run : PlateRun
        Raw kinetic data (from :func:`tecan_growth.io.load_plate`).
    key : PlateKey, optional
        Well assignments; defaults to ``run.key``.
    clip_negative : bool
        Clip blanked OD below zero to zero (legacy behaviour).
    blank_mode : {'per_media', 'global', 'none'}
        Which blank wells to subtract. ``per_media`` uses the blanks of the same
        media and falls back to all blanks on the plate when a media has none.
    replicate_stat : {'mean', 'median'}
        How replicate wells are combined.
    """
    key = key or run.key
    if key is None:
        raise ValueError("A plate key (well -> media/strain) is required; none was found in the export and none was given")
    if blank_mode not in ("per_media", "global", "none"):
        raise ValueError("blank_mode must be 'per_media', 'global' or 'none'")
    if replicate_stat not in ("mean", "median"):
        raise ValueError("replicate_stat must be 'mean' or 'median'")

    validation = key.validate(run.well_ids)
    qc: List[str] = list(validation.warnings)
    if validation.errors:
        raise ValueError("Plate key does not match the data: " + " | ".join(validation.errors))

    time_h = np.asarray(run.time_h, dtype=float)
    n = len(time_h)
    wells = run.wells
    used = key.table[key.table.in_use]
    combine = np.nanmean if replicate_stat == "mean" else np.nanmedian

    # ---- blanks ---------------------------------------------------------
    all_blank_wells = [w for w in key.blank_wells() if w in wells.columns]
    global_blank = combine(wells[all_blank_wells].to_numpy(), axis=1) if all_blank_wells else None
    blank_rows = []
    blank_for_media: Dict[str, Tuple[Optional[np.ndarray], str, int]] = {}
    for media in key.media_names:
        bw = [w for w in key.blank_wells(media) if w in wells.columns]
        if blank_mode == "none":
            vec, src, nb = None, "none", 0
        elif blank_mode == "per_media" and bw:
            vec, src, nb = combine(wells[bw].to_numpy(), axis=1), "per_media", len(bw)
        elif global_blank is not None:
            vec, src, nb = global_blank, "global", len(all_blank_wells)
            if blank_mode == "per_media":
                qc.append(f"Media '{media}': no blank wells; used the {len(all_blank_wells)} blank well(s) from the whole plate")
        else:
            vec, src, nb = None, "none", 0
            if blank_mode != "none":
                qc.append(f"Media '{media}': no blank wells anywhere on the plate; no blank subtraction applied")
        blank_for_media[media] = (vec, src, nb)
        sd = np.nanstd(wells[bw].to_numpy(), axis=1) if (src == "per_media" and len(bw) > 1) else (
            np.nanstd(wells[all_blank_wells].to_numpy(), axis=1) if (src == "global" and len(all_blank_wells) > 1) else np.full(n, np.nan))
        blank_rows.append(pd.DataFrame({
            "media": media, "time_h": time_h,
            "blank_mean": vec if vec is not None else np.zeros(n),
            "blank_sd": sd, "n_blank_wells": nb, "blank_source": src,
        }))
    blanks = pd.concat(blank_rows, ignore_index=True) if blank_rows else pd.DataFrame(
        columns=["media", "time_h", "blank_mean", "blank_sd", "n_blank_wells", "blank_source"])

    # ---- samples --------------------------------------------------------
    curve_frames, rep_frames, cond_rows = [], [], []
    samples = used[~used.is_blank]
    for (media, strain), grp in samples.groupby(["media", "strain"], sort=False):
        ws = [w for w in grp.well if w in wells.columns]
        if not ws:
            continue
        raw = wells[ws].to_numpy(dtype=float)
        vec, src, nb = blank_for_media.get(media, (None, "none", 0))
        blank_vec = vec if vec is not None else np.zeros(n)
        blanked_wells = raw - blank_vec[:, None]
        if clip_negative:
            blanked_wells_clip = np.maximum(blanked_wells, 0.0)
        else:
            blanked_wells_clip = blanked_wells
        raw_mean = combine(raw, axis=1)
        od = combine(raw, axis=1) - blank_vec
        if clip_negative:
            od = np.maximum(od, 0.0)
        od_sd = np.nanstd(raw, axis=1, ddof=1) if raw.shape[1] > 1 else np.full(n, np.nan)
        cid = curve_id(media, strain)
        n_nan = int(np.isnan(raw).sum())
        if n_nan:
            qc.append(f"{cid}: {n_nan} missing OD value(s) across wells {', '.join(ws)} were ignored in the average")
        curve_frames.append(pd.DataFrame({
            "curve_id": cid, "media": media, "strain": strain, "time_h": time_h,
            "od": od, "od_raw_mean": raw_mean, "od_sd": od_sd, "n_replicates": raw.shape[1], "blank_mean": blank_vec,
        }))
        for j, w in enumerate(ws):
            rep_frames.append(pd.DataFrame({
                "well": w, "media": media, "strain": strain, "curve_id": cid, "time_h": time_h,
                "od_raw": raw[:, j], "od_blanked": blanked_wells_clip[:, j],
            }))
        cond_rows.append({"curve_id": cid, "media": media, "strain": strain, "wells": ws,
                          "n_replicates": len(ws), "blank_source": src, "n_blank_wells": nb})

    if not cond_rows:
        raise ValueError("The key has no sample wells (every well is blank or unused)")

    curves = pd.concat(curve_frames, ignore_index=True)
    replicates = pd.concat(rep_frames, ignore_index=True)
    conditions = pd.DataFrame(cond_rows)

    # replicate disagreement QC: flag wells that deviate strongly from their siblings
    for cid, grp in replicates.groupby("curve_id"):
        piv = grp.pivot(index="time_h", columns="well", values="od_raw")
        if piv.shape[1] < 3:
            continue
        med = piv.median(axis=1)
        dev = (piv.sub(med, axis=0)).abs().mean()
        scale = max(float(np.nanmax(med) - np.nanmin(med)), 0.05)
        bad = dev[dev > 0.5 * scale]
        for w, d in bad.items():
            qc.append(f"{cid}: well {w} deviates from its replicates (mean |diff| {d:.3f} OD); check for contamination or a bubble")

    return ProcessedPlate(
        time_h=time_h, curves=curves, replicates=replicates, blanks=blanks, conditions=conditions, qc=qc,
        settings={"clip_negative": clip_negative, "blank_mode": blank_mode, "replicate_stat": replicate_stat},
        run_summary=run.summary(),
    )
