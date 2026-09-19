"""
Session state for the lab interface.

One dictionary-like object (``st.session_state``) holds the loaded plate, the
key, the processed curves, the analysis results, the settings used for that
analysis, and any manual overrides. Everything numeric is delegated to
:mod:`tecan_growth`; this module only decides *when* to recompute.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401  (sys.path)
from tecan_growth import PlateKey, load_plate, preprocess_plate, read_key
from tecan_growth.analyze import (PlateResults, analyze_curve, analyze_plate,
                                  reclassify_plate)
from tecan_growth.io import PlateRun
from tecan_growth.preprocess import ProcessedPlate, exclude_wells
from tecan_growth.settings import AnalysisSettings

#: Settings that only change the verdict, not the fit: a change here is
#: applied instantly with reclassify().
POSTFIT_KEYS = {
    "min_r_squared", "max_param_error_pct", "min_snr", "min_delta_od_ci",
    "min_absolute_delta_od", "excellent_r2_threshold", "min_monotone_fraction",
    "max_residual_autocorr", "postfit_good_threshold", "postfit_bad_threshold",
}
#: Settings that change how curves are prepared: a change here needs preprocessing again.
PREPROCESS_KEYS = {"clip_negative", "blank_mode"}

TRUNCATION_CHOICES = {
    "First peak (fast, default)": {"use_first_peak": True, "use_adaptive": False},
    "Global maximum": {"use_first_peak": False, "use_adaptive": False},
    "Adaptive (cross-validated, ~1-3 min per plate)": {"use_first_peak": True, "use_adaptive": True},
}


# --------------------------------------------------------------------------- basics

def S() -> Any:
    """Shortcut for ``st.session_state`` with defaults installed."""
    s = st.session_state
    # The interactive default is the fast window rule; config.yaml (batch mode) may ask for adaptive.
    s.setdefault("settings", AnalysisSettings.interactive_defaults())
    s.setdefault("overrides", {})
    s.setdefault("flips", [])
    s.setdefault("selected_curve", None)
    s.setdefault("run_name", "")
    return s


def override_cut(entry: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Manual window recorded by tecan_growth.export.set_truncation_override, if any."""
    t = (entry or {}).get("truncation")
    if not t:
        return None
    return int(t["start_index"]), int(t["end_index"])


def override_excluded(entry: Dict[str, Any]) -> List[str]:
    """Excluded wells recorded by tecan_growth.export.set_excluded_wells."""
    e = (entry or {}).get("excluded_wells")
    if not e:
        return []
    return list(e.get("wells", [])) if isinstance(e, dict) else list(e)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def settings_hash(settings: AnalysisSettings) -> str:
    return hashlib.md5(repr(sorted(settings.to_dict().items())).encode()).hexdigest()


# ---------------------------------------------------------------------------- loading

@st.cache_data(show_spinner=False, max_entries=8)
def _load_run_cached(name: str, data: bytes) -> PlateRun:
    buf = io.BytesIO(data)
    buf.name = name
    return load_plate(buf)


def load_run(name: str, data: bytes) -> PlateRun:
    """Parse an uploaded/example file (cached on its content hash)."""
    run = _load_run_cached(name, data)
    s = S()
    s["run"] = run
    s["input_name"] = name
    s["input_sha256"] = sha256_bytes(data)
    s["input_bytes"] = data
    s["key"] = run.key
    s["key_source"] = run.source.get("key_sheet") or ("none" if run.key is None else "sheet")
    s["key_table"] = run.key.to_dataframe() if run.key is not None else empty_key_table(run)
    s["processed"] = None
    s["results"] = None
    s["overrides"] = {}
    s["flips"] = []
    s["selected_curve"] = None
    stem = name.rsplit(".", 1)[0]
    # The sidebar text box owns the widget key; hand it the new default on the next rerun.
    s["run_name_pending"] = f"{run.metadata.get('date') or 'run'}_{stem}"[:60].replace(" ", "_")
    return run


def load_key_bytes(name: str, data: bytes) -> PlateKey:
    """Parse an uploaded key CSV/xlsx and make it the active key."""
    buf = io.BytesIO(data)
    buf.name = name
    key = read_key(buf)
    set_key(key, source=f"upload:{name}")
    return key


def empty_key_table(run: PlateRun) -> pd.DataFrame:
    """A blank key (one row per measured well) pre-marked with the instrument's BL/SM labels."""
    rows = []
    for w in run.well_ids:
        label = str(run.layout.get(w, "")).upper()
        strain = "BLANK" if label.startswith("BL") else ""
        rows.append({"Cell": w, "Media": "", "Strain": strain})
    return pd.DataFrame(rows)


def set_key(key: Optional[PlateKey], source: str = "edited") -> None:
    s = S()
    s["key"] = key
    s["key_source"] = source
    s["key_table"] = key.to_dataframe() if key is not None else s.get("key_table")
    s["processed"] = None
    s["results"] = None
    s["overrides"] = {}
    s["flips"] = []


def key_from_table(table: pd.DataFrame) -> Tuple[Optional[PlateKey], List[str]]:
    """Build a PlateKey from the editable table; returns (key or None, error messages)."""
    try:
        return PlateKey.from_dataframe(table, source="edited"), []
    except ValueError as e:
        return None, [str(e)]


def key_validation():
    """Validate the active key against the loaded plate."""
    s = S()
    run, key = s.get("run"), s.get("key")
    if run is None or key is None:
        return None
    return key.validate(run.well_ids)


def blocking_reason() -> Optional[str]:
    """Why the Run button is disabled, or None if it can run."""
    s = S()
    if s.get("run") is None:
        return "Load a plate export first"
    if s.get("key") is None:
        return "Assign media and strain to the wells first"
    v = key_validation()
    if v is not None and v.errors:
        return v.errors[0]
    key = s["key"]
    for media in key.media_names:
        if not key.blank_wells(media) and not key.blank_wells():
            return f"Add blank wells for {media} (or set blank mode to 'none' in Advanced)"
    return None


# ------------------------------------------------------------------------- processing

@st.cache_data(show_spinner=False, max_entries=8)
def _preprocess_cached(input_sha: str, key_records: Tuple[Tuple[str, str, str], ...],
                       clip_negative: bool, blank_mode: str, _run: PlateRun, _key: PlateKey) -> ProcessedPlate:
    return preprocess_plate(_run, _key, clip_negative=clip_negative, blank_mode=blank_mode)


def get_processed() -> ProcessedPlate:
    """Blank-corrected curves for the current plate/key/settings (cached)."""
    s = S()
    run, key, settings = s["run"], s["key"], s["settings"]
    records = tuple((r.Cell, r.Media, r.Strain) for r in key.to_dataframe().itertuples())
    processed = _preprocess_cached(s["input_sha256"], records, settings.clip_negative, settings.blank_mode, run, key)
    # Re-apply replicate exclusions recorded in the overrides
    for cid, entry in s["overrides"].items():
        wells = override_excluded(entry)
        if wells and cid in set(processed.conditions.curve_id):
            processed = exclude_wells(processed, cid, wells)
    s["processed"] = processed
    return processed


def run_analysis(progress=None) -> PlateResults:
    """Full analysis of every curve with the current settings, honouring manual cuts."""
    s = S()
    processed = get_processed()
    settings = s["settings"]
    results = analyze_plate(processed, settings, progress=progress)
    # Re-apply manual truncation windows
    for cid, entry in s["overrides"].items():
        cut = override_cut(entry)
        if cut and cid in results.curves:
            cr = results.curves[cid]
            new = analyze_curve(cr.time_h, cr.od, cid, settings, media=cr.media, strain=cr.strain,
                                wells=cr.wells, truncation_override=(int(cut[0]), int(cut[1])))
            results.curves[cid] = new
    _rebuild_table(results)
    old = s.get("results")
    s["flips"] = flips(old, results) if old is not None else []
    s["results"] = results
    s["run_settings_hash"] = settings_hash(settings)
    s["run_settings"] = AnalysisSettings(**settings.to_dict())
    return results


def _rebuild_table(results: PlateResults) -> None:
    from tecan_growth.analyze import RESULT_COLUMNS
    ids = list(results.table.curve_id)
    results.table = pd.DataFrame([results.curves[c].row for c in ids], columns=RESULT_COLUMNS)


def refit_curve(cid: str, cut: Optional[Tuple[int, int]] = None, adaptive: bool = False) -> None:
    """Refit one curve with a manual window (or the adaptive method) and store the override."""
    s = S()
    results: PlateResults = s["results"]
    cr = results.curves[cid]
    settings = s["settings"]
    if adaptive:
        settings = AnalysisSettings(**{**settings.to_dict(), "use_adaptive": True, "use_first_peak": True})
        new = analyze_curve(cr.time_h, cr.od, cid, settings, media=cr.media, strain=cr.strain, wells=cr.wells)
        if new.truncation is not None and new.legacy.stage == "classified":
            cut = (int(new.truncation.start_index), int(new.truncation.truncation_index))
    new = analyze_curve(cr.time_h, cr.od, cid, s["settings"], media=cr.media, strain=cr.strain,
                        wells=cr.wells, truncation_override=cut)
    old_call = "GOOD" if cr.is_good else "BAD"
    results.curves[cid] = new
    _rebuild_table(results)
    entry = s["overrides"].setdefault(cid, {})
    if cut is not None:
        from tecan_growth.export import set_truncation_override
        set_truncation_override(s["overrides"], cid, int(cut[0]), int(cut[1]),
                                note="adaptive method" if adaptive else "manual window")
    else:
        entry.pop("truncation", None)
    new_call = "GOOD" if new.is_good else "BAD"
    if new_call != old_call:
        s["flips"] = [(cid, old_call, new_call, "manual truncation")]


def set_excluded(cid: str, wells: Iterable[str]) -> None:
    """Exclude replicate wells for one curve, re-average and refit it."""
    s = S()
    from tecan_growth.export import set_excluded_wells
    wells = list(wells)
    entry = s["overrides"].setdefault(cid, {})
    if wells:
        set_excluded_wells(s["overrides"], cid, wells)
    else:
        entry.pop("excluded_wells", None)
    processed = get_processed()
    results: PlateResults = s["results"]
    t, od = processed.curve(cid)
    cond = processed.conditions.set_index("curve_id").loc[cid]
    new = analyze_curve(t, od, cid, s["settings"], media=str(cond.media), strain=str(cond.strain),
                        wells=list(cond.wells), truncation_override=override_cut(entry))
    results.curves[cid] = new
    _rebuild_table(results)


def apply_setting_changes() -> Optional[str]:
    """
    Called after a sidebar change. Returns 'reclassified', 'stale' or None.

    Post-fit-only changes are applied instantly; anything else marks the
    results stale until the user presses Re-run.
    """
    s = S()
    results = s.get("results")
    if results is None or "run_settings" not in s:
        return None
    changed = changed_keys(s["run_settings"], s["settings"])
    if not changed:
        return None
    if changed <= POSTFIT_KEYS and not s["settings"].use_ml:
        new = reclassify_plate(results, s["settings"])
        # keep manual cuts (reclassify keeps the fit, so nothing to redo)
        s["flips"] = flips(results, new)
        s["results"] = new
        s["run_settings"] = AnalysisSettings(**s["settings"].to_dict())
        s["run_settings_hash"] = settings_hash(s["settings"])
        return "reclassified"
    return "stale"


def changed_keys(a: AnalysisSettings, b: AnalysisSettings) -> set:
    da, db = a.to_dict(), b.to_dict()
    return {k for k in da if k != "extra" and da[k] != db.get(k)}


def is_stale() -> bool:
    s = S()
    return s.get("results") is not None and "run_settings" in s and bool(changed_keys(s["run_settings"], s["settings"]))


def flips(old: Optional[PlateResults], new: PlateResults) -> List[Tuple[str, str, str, str]]:
    """Curves whose verdict changed between two result sets."""
    if old is None:
        return []
    out = []
    for cid, cr in new.curves.items():
        if cid not in old.curves:
            continue
        a = "GOOD" if old.curves[cid].is_good else "BAD"
        b = "GOOD" if cr.is_good else "BAD"
        if a != b:
            gate = next((g.name for g in cr.failed_gates()), "") if not cr.is_good else next((g.name for g in old.curves[cid].failed_gates()), "")
            out.append((cid, a, b, gate))
    return out


def final_call(cid: str) -> str:
    """Verdict after any manual override."""
    s = S()
    entry = s["overrides"].get(cid, {})
    if entry.get("verdict"):
        return str(entry["verdict"])
    cr = s["results"].curves[cid]
    return "GOOD" if cr.is_good else "BAD"


def counts() -> Dict[str, int]:
    s = S()
    results = s.get("results")
    if results is None:
        return {"GOOD": 0, "BAD": 0, "not fitted": 0}
    good = bad = nf = 0
    for cid, cr in results.curves.items():
        call = final_call(cid)
        if cr.legacy.stage in ("prefit_rejected", "truncation_invalid", "too_few_points"):
            nf += 1
        elif call == "GOOD":
            good += 1
        else:
            bad += 1
    return {"GOOD": good, "BAD": bad, "not fitted": nf}


def update_setting(**kwargs) -> None:
    s = S()
    s["settings"] = AnalysisSettings(**{**s["settings"].to_dict(), **kwargs})


def truncation_choice_label(settings: AnalysisSettings) -> str:
    for label, cfg in TRUNCATION_CHOICES.items():
        if cfg["use_adaptive"] == settings.use_adaptive and cfg["use_first_peak"] == settings.use_first_peak:
            return label
    return list(TRUNCATION_CHOICES)[0]
