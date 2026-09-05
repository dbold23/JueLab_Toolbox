"""
Everything a run leaves behind: tables, a run folder, a workbook and a manifest.

The point of this module is that a finished analysis can be handed to somebody
else -- a lab mate, a collaborator, a reviewer -- and be understood and redone
without this software. A run folder therefore contains

* the numbers (``results_table.csv``, ``results_clean.csv``, ``curves_long.csv``),
* the inputs in the format the command-line pipeline reads
  (``processed/<MEDIA>_DATA.csv`` and ``key.csv``), so ``make step-2``
  reproduces ``processing_results.csv`` byte for byte,
* a methods paragraph you can paste into a report (``methods.md``),
* and ``run_manifest.json``: which file was analysed (with its SHA-256), which
  version of this code and which thresholds produced the numbers, and every
  manual change somebody made along the way.

Overrides
---------
Manual decisions travel through this module as one plain dictionary keyed by
curve id, so it can be stored in the app's session state and written to JSON
unchanged::

    {
      "IMIDACLOPRID-IMID2": {
        "verdict": "BAD",                  # hand-set call of record (GOOD/BAD)
        "note": "flat, the fit is noise",  # required for a verdict change
        "verdict_timestamp": "2025-03-16T09:12:03Z",
        "truncation": {"start_index": 0, "end_index": 180,
                       "previous": [0, 264], "note": "cut before the second rise",
                       "timestamp": "..."},
        "excluded_wells": {"wells": ["C6"], "note": "bubble", "timestamp": "..."},
        "review": {"audit_result": "wrong", "note": "", "timestamp": "..."}
      }
    }

Use :func:`set_verdict_override`, :func:`set_truncation_override`,
:func:`set_excluded_wells` and :func:`record_review` to build it; they stamp
the time for you.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import platform
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__, legacy
from .analyze import RESULT_COLUMNS, PlateResults, doubling_time_h, legacy_results_frame
from .io import PlateRun
from .plate import PlateKey, is_blank_label
from .preprocess import ProcessedPlate
from .settings import AnalysisSettings

#: Extra columns :func:`results_table` adds to ``RESULT_COLUMNS``.
EXPORT_COLUMNS: List[str] = [
    "treatment_type", "pesticide", "blank_source", "final_call", "override_note",
    "excluded_wells", "truncation_override",
]

#: Columns of :func:`clean_summary`, in order (report-ready).
CLEAN_COLUMNS: List[str] = [
    "Strain", "Treatment", "Strain_ID", "Treatment_Type", "Classification",
    "Max_OD600_A", "A_StdErr", "Growth_Rate_mu_OD_per_hr", "mu_StdErr",
    "Lag_Phase_lambda_hrs", "lambda_StdErr", "Doubling_Time_hrs", "R_squared", "RMSE",
]

#: Files a run folder always contains (relative to the folder).
RUN_FOLDER_FILES: List[str] = [
    "processing_results.csv", "results_table.csv", "results_clean.csv", "curves_long.csv",
    "key.csv", "run_manifest.json", "methods.md", "report.xlsx",
]

_ZWIETERING = ("Zwietering, M. H., Jongenburger, I., Rombouts, F. M. & van 't Riet, K. (1990). "
               "Modeling of the bacterial growth curve. Applied and Environmental Microbiology 56(6), 1875-1881.")


# =============================================================================
# small helpers
# =============================================================================

def _now() -> str:
    """Current UTC time as an ISO-8601 string (what goes into the manifest)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path) -> str:
    """SHA-256 of a file, so a manifest can prove which raw export was analysed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit of this repository, or ``None`` when git is unavailable."""
    root = Path(repo_root) if repo_root else legacy.REPO_ROOT
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(repo_root: Optional[Path] = None) -> Optional[bool]:
    root = Path(repo_root) if repo_root else legacy.REPO_ROOT
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas values into something ``json.dump`` accepts."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _classify_media_names(media_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Treatment type and pesticide for each media name (via :mod:`tecan_growth.media`)."""
    try:
        from .media import classify_media
    except ImportError as exc:  # pragma: no cover - media.py ships with the package
        raise ImportError(
            "tecan_growth.media is required to label treatments in exports "
            f"(classify_media): {exc}"
        ) from exc
    return {m: classify_media(m) for m in dict.fromkeys(media_names)}


# =============================================================================
# overrides
# =============================================================================

def _entry(overrides: Optional[Dict[str, Any]], curve_id: str) -> Dict[str, Any]:
    return dict((overrides or {}).get(curve_id, {}) or {})


def set_verdict_override(overrides: Dict[str, Any], curve_id: str, verdict: Optional[str],
                         note: str = "") -> Dict[str, Any]:
    """
    Hand-set the call of record for one curve (``'GOOD'``, ``'BAD'`` or ``None`` to clear).

    A note is required, because this overrules the software's own verdict and
    everybody reading the results later deserves to know why.
    """
    if verdict is not None:
        verdict = str(verdict).upper()
        if verdict not in ("GOOD", "BAD"):
            raise ValueError("verdict must be 'GOOD', 'BAD' or None")
        if not str(note).strip():
            raise ValueError("A note is required when you overrule the verdict")
    entry = overrides.setdefault(curve_id, {})
    if verdict is None:
        entry.pop("verdict", None)
        entry.pop("verdict_timestamp", None)
        entry.pop("note", None)
    else:
        entry["verdict"] = verdict
        entry["note"] = str(note).strip()
        entry["verdict_timestamp"] = _now()
    return overrides


def set_truncation_override(overrides: Dict[str, Any], curve_id: str,
                            start_index: int, end_index: int,
                            previous: Optional[Sequence[int]] = None, note: str = "") -> Dict[str, Any]:
    """Record a manual growth window (indices into the curve's cleaned arrays)."""
    overrides.setdefault(curve_id, {})["truncation"] = {
        "start_index": int(start_index), "end_index": int(end_index),
        "previous": [int(previous[0]), int(previous[1])] if previous is not None else None,
        "note": str(note), "timestamp": _now(),
    }
    return overrides


def set_excluded_wells(overrides: Dict[str, Any], curve_id: str, wells: Sequence[str],
                       note: str = "") -> Dict[str, Any]:
    """Record which replicate wells were left out of a curve's average."""
    overrides.setdefault(curve_id, {})["excluded_wells"] = {
        "wells": [str(w) for w in wells], "note": str(note), "timestamp": _now(),
    }
    return overrides


def record_review(overrides: Dict[str, Any], curve_id: str, audit_result: str,
                  note: str = "") -> Dict[str, Any]:
    """Record a human review of a curve (``correct`` / ``wrong`` / ``unsure`` / ``skip``)."""
    if audit_result not in ("correct", "wrong", "unsure", "skip"):
        raise ValueError("audit_result must be 'correct', 'wrong', 'unsure' or 'skip'")
    overrides.setdefault(curve_id, {})["review"] = {
        "audit_result": audit_result, "note": str(note), "timestamp": _now(),
    }
    return overrides


def overrides_frame(overrides: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the overrides dictionary into one tidy row per manual change."""
    rows: List[Dict[str, Any]] = []
    for cid, entry in (overrides or {}).items():
        entry = entry or {}
        if entry.get("verdict"):
            rows.append({"curve_id": cid, "change": "verdict", "value": entry["verdict"],
                         "previous": "", "note": entry.get("note", ""),
                         "timestamp": entry.get("verdict_timestamp", "")})
        tr = entry.get("truncation")
        if tr:
            prev = tr.get("previous")
            rows.append({"curve_id": cid, "change": "truncation",
                         "value": f"{tr.get('start_index')}-{tr.get('end_index')}",
                         "previous": f"{prev[0]}-{prev[1]}" if prev else "",
                         "note": tr.get("note", ""), "timestamp": tr.get("timestamp", "")})
        ex = entry.get("excluded_wells")
        if ex and ex.get("wells"):
            rows.append({"curve_id": cid, "change": "excluded_wells", "value": ", ".join(ex["wells"]),
                         "previous": "", "note": ex.get("note", ""), "timestamp": ex.get("timestamp", "")})
        rv = entry.get("review")
        if rv:
            rows.append({"curve_id": cid, "change": "review", "value": rv.get("audit_result", ""),
                         "previous": "", "note": rv.get("note", ""), "timestamp": rv.get("timestamp", "")})
    return pd.DataFrame(rows, columns=["curve_id", "change", "value", "previous", "note", "timestamp"])


# =============================================================================
# tables
# =============================================================================

def results_table(results: PlateResults, overrides: Optional[Dict[str, Any]] = None,
                  processed: Optional[ProcessedPlate] = None) -> pd.DataFrame:
    """
    The full per-curve results table: every analysis column plus the labels and
    manual decisions that belong to a run.

    Parameters
    ----------
    results : PlateResults
        Output of :func:`tecan_growth.analyze.analyze_plate`.
    overrides : dict, optional
        Manual decisions (see the module docstring). ``final_call`` follows
        them; the software's own verdict stays visible in ``classification``.
    processed : ProcessedPlate, optional
        Used to fill ``blank_source`` (which blanks were subtracted) and the
        wells excluded from each average. Without it, ``blank_source`` falls
        back to the plate-wide blank mode from the settings.

    Returns
    -------
    pd.DataFrame
        ``RESULT_COLUMNS`` followed by ``EXPORT_COLUMNS``.
    """
    table = results.table.copy()
    labels = _classify_media_names(list(table.media))
    table["treatment_type"] = [labels[m]["treatment_type"] for m in table.media]
    table["pesticide"] = [labels[m]["pesticide"] for m in table.media]

    blank_source = {}
    excluded_from_processed: Dict[str, List[str]] = {}
    if processed is not None:
        blank_source = dict(zip(processed.conditions.curve_id, processed.conditions.blank_source))
        excluded_from_processed = {k: list(v) for k, v in
                                   dict(processed.settings.get("excluded_wells", {})).items()}
    table["blank_source"] = [blank_source.get(cid, results.settings.blank_mode) for cid in table.curve_id]

    final_calls, notes, excluded, cuts = [], [], [], []
    for _, row in table.iterrows():
        entry = _entry(overrides, row.curve_id)
        call = str(row.classification) if row.classification else ("GOOD" if row.is_good else "BAD")
        note_parts = []
        if entry.get("verdict"):
            call = entry["verdict"]
            note_parts.append(f"verdict set by hand: {entry.get('note', '')}".strip())
        tr = entry.get("truncation")
        if tr and tr.get("note"):
            note_parts.append(f"cut: {tr['note']}")
        ex = entry.get("excluded_wells") or {}
        if ex.get("note"):
            note_parts.append(f"wells: {ex['note']}")
        final_calls.append(call)
        notes.append("; ".join(p for p in note_parts if p))
        wells_out = list(ex.get("wells", [])) or excluded_from_processed.get(row.curve_id, [])
        excluded.append(", ".join(wells_out))
        cr = results.curves.get(row.curve_id)
        cut = None
        if tr:
            cut = (tr.get("start_index"), tr.get("end_index"))
        elif cr is not None and cr.truncation_override is not None:
            cut = cr.truncation_override
        cuts.append("" if cut is None else f"{cut[0]}-{cut[1]}")

    table["final_call"] = final_calls
    table["override_note"] = notes
    table["excluded_wells"] = excluded
    table["truncation_override"] = cuts
    return table[RESULT_COLUMNS + EXPORT_COLUMNS]


def _numeric(column: pd.Series) -> np.ndarray:
    """Column as floats, with anything unparsable (or missing) as NaN."""
    return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)


def clean_summary(results: PlateResults, overrides: Optional[Dict[str, Any]] = None,
                  include_bad: bool = False) -> pd.DataFrame:
    """
    The report-ready summary: one row per curve with the growth parameters and
    their standard errors, in the column names the lab already shares
    (``growth_rate_summary_for_collaboration.csv``).

    Only curves whose final call is GOOD are included unless ``include_bad``
    is set; a BAD curve means "the model does not describe this data", so its
    parameters are not numbers to quote.
    """
    full = results_table(results, overrides)
    keep = full if include_bad else full[full.final_call.str.upper() == "GOOD"]
    strain_ids, treatments = [], []
    for cid in keep.curve_id:
        treatment, _, strain_id = str(cid).rpartition("-")
        strain_ids.append(strain_id or cid)
        treatments.append(treatment or cid)
    out = pd.DataFrame({
        "Strain": list(keep.curve_id),
        "Treatment": treatments,
        "Strain_ID": strain_ids,
        "Treatment_Type": list(keep.treatment_type),
        "Classification": [str(c).capitalize() for c in keep.final_call],
        "Max_OD600_A": _numeric(keep.gompertz_a),
        "A_StdErr": _numeric(keep.gompertz_a_err),
        "Growth_Rate_mu_OD_per_hr": _numeric(keep.gompertz_mu),
        "mu_StdErr": _numeric(keep.gompertz_mu_err),
        "Lag_Phase_lambda_hrs": _numeric(keep.gompertz_lambda),
        "lambda_StdErr": _numeric(keep.gompertz_lambda_err),
        "Doubling_Time_hrs": [doubling_time_h(a, m) for a, m in
                              zip(_numeric(keep.gompertz_a), _numeric(keep.gompertz_mu))],
        "R_squared": _numeric(keep.fit_r_squared),
        "RMSE": _numeric(keep.fit_rmse),
    })
    for col in CLEAN_COLUMNS[5:]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    out = out.sort_values(["Strain_ID", "Treatment_Type", "Treatment"]).reset_index(drop=True)
    return out[CLEAN_COLUMNS]


def curves_long(processed: ProcessedPlate, results: PlateResults) -> pd.DataFrame:
    """
    Every measured point of the run in one tidy table: the raw well readings,
    the blank that was subtracted, the blanked value, whether the point was
    inside the fitted window, and what the model says it should have been.

    Each curve contributes one row per replicate well and time point, plus a
    ``well = 'MEAN'`` row: that averaged series is what the Gompertz model was
    actually fitted to, so its ``residual`` is the residual of the fit.
    ``od_predicted`` is given over the whole time axis -- outside the fitted
    window it is an extrapolation, which is exactly what makes an early cut
    easy to see.
    """
    gca = legacy.growth_curve_analysis()
    frames: List[pd.DataFrame] = []
    for cid in processed.conditions.curve_id:
        c = processed.curves[processed.curves.curve_id == cid]
        t = c.time_h.to_numpy(dtype=float)
        media, strain = str(c.media.iloc[0]), str(c.strain.iloc[0])
        blank = c.blank_mean.to_numpy(dtype=float)
        predicted = np.full(len(t), np.nan)
        in_fit = np.zeros(len(t), dtype=bool)
        cr = results.curves.get(cid)
        if cr is not None and cr.fit_succeeded:
            f = cr.fit
            predicted = gca.gompertz_model(t, f.a_opt, f.mu_opt, f.lambda_opt)
            window = cr.truncation
            if window is not None and len(window.time_truncated):
                in_fit = (t >= float(window.time_truncated[0])) & (t <= float(window.time_truncated[-1]))
        reps = processed.replicates[processed.replicates.curve_id == cid]
        for well in dict.fromkeys(reps.well):
            r = reps[reps.well == well]
            frames.append(pd.DataFrame({
                "time_h": t, "media": media, "strain": strain, "curve_id": cid, "well": well,
                "od_raw": r.od_raw.to_numpy(dtype=float), "blank_mean": blank,
                "od_blanked": r.od_blanked.to_numpy(dtype=float),
                "in_fit": in_fit, "od_predicted": predicted,
                "residual": r.od_blanked.to_numpy(dtype=float) - predicted,
            }))
        frames.append(pd.DataFrame({
            "time_h": t, "media": media, "strain": strain, "curve_id": cid, "well": "MEAN",
            "od_raw": c.od_raw_mean.to_numpy(dtype=float), "blank_mean": blank,
            "od_blanked": c.od.to_numpy(dtype=float),
            "in_fit": in_fit, "od_predicted": predicted,
            "residual": c.od.to_numpy(dtype=float) - predicted,
        }))
    columns = ["time_h", "media", "strain", "curve_id", "well", "od_raw", "blank_mean",
               "od_blanked", "in_fit", "od_predicted", "residual"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# =============================================================================
# manifest and methods
# =============================================================================

def _model_hashes() -> List[Dict[str, str]]:
    models = legacy.REPO_ROOT / "models"
    out = []
    for name in ("prefit_gate.joblib", "postfit_classifier.joblib"):
        p = models / name
        if p.exists():
            out.append({"file": name, "sha256": file_sha256(p)})
    return out


def run_manifest(run: PlateRun, key: PlateKey, processed: ProcessedPlate, results: PlateResults,
                 overrides: Optional[Dict[str, Any]] = None, input_sha256: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Everything needed to say where a number came from, as a JSON-serialisable dict.

    It records the input file and its SHA-256, the commit of this code, the
    versions of Python and the numerical libraries, the instrument settings the
    reader wrote into the export, every threshold that was in force, whether
    the ML second opinion was used, and every manual change with its timestamp.
    """
    settings = results.settings
    table = results.table
    n_fitted = int(table.stage.eq("classified").sum())
    counts = {
        "n_curves": int(len(table)),
        "n_good": int(table.is_good.sum()),
        "n_bad": int(len(table) - int(table.is_good.sum())),
        "n_fitted": n_fitted,
        "n_not_fitted": int(len(table) - n_fitted),
        "n_wells_measured": len(run.well_ids),
        "n_blank_wells": len(key.blank_wells()) if key is not None else 0,
        "n_overridden_verdicts": int(sum(1 for e in (overrides or {}).values() if (e or {}).get("verdict"))),
    }
    final = results_table(results, overrides, processed)
    counts["n_final_good"] = int((final.final_call.str.upper() == "GOOD").sum())

    manifest: Dict[str, Any] = {
        "schema": "tecan_growth/run_manifest/1",
        "created_utc": _now(),
        "software": {
            "package": "tecan_growth",
            "version": __version__,
            "git_commit": git_commit(),
            "git_dirty": _git_dirty(),
            "repo_root": str(legacy.REPO_ROOT),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": _package_version("scipy"),
            "plotly": _package_version("plotly"),
            "streamlit": _package_version("streamlit"),
        },
        "input": {
            "file": Path(str(run.source.get("path", ""))).name or None,
            "path": run.source.get("path"),
            "format": run.source.get("format"),
            "sha256": input_sha256,
            "kinetic_sheet": run.source.get("kinetic_sheet"),
            "key_sheet": run.source.get("key_sheet"),
            "sheets_used": [s for s in (run.source.get("kinetic_sheet"), run.source.get("key_sheet")) if s],
            "loader_warnings": list(run.warnings),
        },
        "instrument": dict(run.metadata),
        "run_summary": run.summary(),
        "key": {
            "source": getattr(key, "source", None),
            "n_wells_assigned": int(key.table.in_use.sum()) if key is not None else 0,
            "media": list(key.media_names) if key is not None else [],
            "strains": sorted({s for s in key.table.strain if s and not is_blank_label(s)}) if key is not None else [],
            "warnings": list(key.warnings) if key is not None else [],
        },
        "preprocessing": {
            **{k: v for k, v in processed.settings.items()},
            "blank_sources": {str(r.curve_id): str(r.blank_source) for _, r in processed.conditions.iterrows()},
            "qc": list(processed.qc),
        },
        "settings": settings.to_dict(),
        "settings_changed_from_defaults": settings.describe_changes(),
        "classifier": {
            "mode": "rules + ML second opinion" if results.ml_used else "rules",
            "ml_used": bool(results.ml_used),
            "models": _model_hashes(),
        },
        "overrides": overrides or {},
        "counts": counts,
    }
    if extra:
        manifest["extra"] = extra
    return _jsonable(manifest)


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover - importlib.metadata is standard since 3.8
        return None


def methods_paragraph(settings: AnalysisSettings, results: Optional[PlateResults] = None) -> str:
    """
    A methods paragraph with the thresholds that were actually used, ready to
    paste into a lab report or a manuscript.
    """
    truncation = ("a Monte-Carlo cross-validated search for the window that predicts held-out points best"
                  if settings.use_adaptive else
                  ("the first local maximum of the smoothed curve" if settings.use_first_peak
                   else "the global maximum of the smoothed curve"))
    blanks = {"per_media": "the mean of the blank (media-only) wells of the same media",
              "global": "the mean of all blank wells on the plate",
              "none": "no blank"}.get(settings.blank_mode, settings.blank_mode)
    counts = ""
    if results is not None:
        counts = (f" Of {results.n_total} conditions on this plate, {results.n_good} passed every quality "
                  f"gate and are reported; the rest are marked BAD, which means the model could not be "
                  f"trusted to describe them, not that nothing grew.")
    ml = (" A trained classifier was consulted as a second opinion on the fitted curves."
          if (results is not None and results.ml_used) else
          " Classification was rule-based; no machine-learning model was used.")
    return (
        "Growth was followed as optical density at 600 nm in a TECAN Spark plate reader. "
        f"For every condition, {blanks} was subtracted from each replicate well at each time point"
        f"{' and negative values were set to zero' if settings.clip_negative else ''}, and the replicate "
        "wells were averaged. The growth window was found by smoothing the curve with a "
        f"{settings.smoothing_window}-point centred rolling mean and cutting at {truncation}, keeping "
        f"{settings.buffer_points} points beyond it; curves still rising at the end of the run were left "
        "whole. The modified Gompertz model of Zwietering et al. (1990), "
        "y(t) = A exp(-exp((mu e / A)(lambda - t) + 1)), was fitted to that window by non-linear least "
        "squares (scipy.optimize.curve_fit), giving the maximum optical density A, the maximum growth rate "
        "mu (OD/h), and the lag time lambda (h) with standard errors from the covariance matrix; the "
        "minimum doubling time was computed as ln(2) A / (mu e). A fit was accepted only when it converged "
        f"on at least {settings.min_points_after_truncation} points with R2 >= {settings.min_r_squared:g}, "
        f"standard errors of A and mu below {settings.max_param_error_pct:g} % of the estimate, and a total "
        f"change in optical density of at least {settings.min_absolute_delta_od:g}. Curves whose fit was not "
        f"excellent (R2 < {settings.excellent_r2_threshold:g}) additionally had to show a signal-to-noise "
        f"ratio of at least {settings.min_snr:g}, a growth increment at least "
        f"{settings.min_delta_od_ci:g} above twice the baseline noise, rising steps over at least "
        f"{settings.min_monotone_fraction:.0%} of the growth phase, and residual lag-1 autocorrelation below "
        f"{settings.max_residual_autocorr:g}. Curves whose signal-to-noise ratio was below "
        f"{settings.pre_fit_snr_threshold:g} and whose total growth was below "
        f"{settings.pre_fit_delta_od_override:g} OD were rejected before fitting."
        f"{ml}{counts}\n\nReference: {_ZWIETERING}"
    )


# =============================================================================
# run folder
# =============================================================================

def _settings_frame(results: PlateResults, processed: ProcessedPlate) -> pd.DataFrame:
    rows = [{"setting": "tecan_growth version", "value": __version__},
            {"setting": "git commit", "value": git_commit() or "unknown"},
            {"setting": "classifier", "value": "rules + ML second opinion" if results.ml_used else "rules"}]
    rows += [{"setting": k, "value": _jsonable(v)} for k, v in results.settings.to_dict().items() if k != "extra"]
    rows += [{"setting": f"preprocessing.{k}", "value": _jsonable(v)} for k, v in processed.settings.items()]
    return pd.DataFrame(rows)


def _write_processed_csvs(processed: ProcessedPlate, out_dir: Path) -> List[Path]:
    """Write the CLI-compatible ``<MEDIA>_DATA.csv`` files (one per media)."""
    return processed.write_legacy_csvs(out_dir)


def write_run_folder(out_dir, run: PlateRun, key: PlateKey, processed: ProcessedPlate,
                     results: PlateResults, overrides: Optional[Dict[str, Any]] = None,
                     input_sha256: Optional[str] = None,
                     figures: Optional[Dict[str, Any]] = None) -> List[Path]:
    """
    Write a complete, self-contained run folder and return the files written.

    The folder holds the results in three shapes (full table, report-ready
    summary, every point), the inputs in the command-line pipeline's own format
    so the run can be reproduced without this app, a workbook for Excel users,
    the methods paragraph and the manifest.

    Parameters
    ----------
    out_dir : path-like
        Folder to create (parents are created as needed).
    run, key, processed, results
        The four objects of one analysis.
    overrides : dict, optional
        Manual decisions; see the module docstring.
    input_sha256 : str, optional
        Checksum of the raw export, from :func:`file_sha256`.
    figures : dict of {name: plotly Figure}, optional
        Saved as ``figures/<name>.html`` with the plotting library embedded,
        so they open on a machine with no internet.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    full = results_table(results, overrides, processed)
    clean = clean_summary(results, overrides)
    long_table = curves_long(processed, results)
    over = overrides_frame(overrides)

    legacy_results_frame(results).to_csv(out / "processing_results.csv", index=False)
    written.append(out / "processing_results.csv")
    full.to_csv(out / "results_table.csv", index=False)
    written.append(out / "results_table.csv")
    clean.to_csv(out / "results_clean.csv", index=False)
    written.append(out / "results_clean.csv")
    long_table.to_csv(out / "curves_long.csv", index=False)
    written.append(out / "curves_long.csv")
    key.to_dataframe().to_csv(out / "key.csv", index=False)
    written.append(out / "key.csv")
    written.extend(_write_processed_csvs(processed, out / "processed"))

    manifest = run_manifest(run, key, processed, results, overrides, input_sha256)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    written.append(out / "run_manifest.json")

    (out / "methods.md").write_text(
        "# Methods\n\n" + methods_paragraph(results.settings, results) + "\n")
    written.append(out / "methods.md")

    # One workbook for people who live in Excel. The Curves sheet holds the
    # averaged curve of each condition (the series that was fitted); the
    # per-well data would be several hundred thousand cells and lives in
    # curves_long.csv instead.
    with pd.ExcelWriter(out / "report.xlsx", engine="openpyxl") as xl:
        full.to_excel(xl, sheet_name="Results", index=False)
        clean.to_excel(xl, sheet_name="Clean", index=False)
        long_table[long_table.well == "MEAN"].to_excel(xl, sheet_name="Curves", index=False)
        key.to_dataframe().to_excel(xl, sheet_name="Key", index=False)
        _settings_frame(results, processed).to_excel(xl, sheet_name="Settings", index=False)
        over.to_excel(xl, sheet_name="Overrides", index=False)
    written.append(out / "report.xlsx")

    for name, fig in (figures or {}).items():
        if not hasattr(fig, "write_html"):
            raise TypeError(f"figure '{name}' is not a plotly Figure (no write_html)")
        fig_dir = out / "figures"
        fig_dir.mkdir(exist_ok=True)
        path = fig_dir / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)
        written.append(path)

    return written


def bundle_zip(out_dir) -> bytes:
    """Zip a run folder into bytes, ready for a download button."""
    root = Path(out_dir)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def load_run_folder(run_dir) -> Dict[str, Any]:
    """
    Reopen a run folder: the results exactly as they were written, with no
    recomputation, so verdicts and manual changes come back unchanged.

    Returns
    -------
    dict
        ``results_table`` (DataFrame), ``manifest`` (dict) and ``overrides`` (dict).
    """
    root = Path(run_dir)
    table_path = root / "results_table.csv"
    manifest_path = root / "run_manifest.json"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} not found; this is not a run folder")
    table = pd.read_csv(table_path)
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    return {"results_table": table, "manifest": manifest, "overrides": manifest.get("overrides", {}) or {}}
