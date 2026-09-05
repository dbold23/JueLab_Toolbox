"""
Per-curve and per-plate analysis built on the validated core.

:func:`analyze_curve` wraps ``scripts/01_growth_curve_analysis.analyze_curve``
(pre-fit gate → truncation → Gompertz fit → classification) and adds a
transparent *gate table* that explains every GOOD/BAD decision.
:func:`analyze_plate` runs it over every curve of a :class:`ProcessedPlate`
and returns a tidy results table identical in content to the CLI's
``processing_results.csv`` plus the plate metadata.
"""
from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import legacy
from .preprocess import ProcessedPlate
from .settings import AnalysisSettings

#: Columns of the per-curve results table, in output order.
RESULT_COLUMNS = [
    "curve_id", "media", "strain", "wells", "n_replicates",
    "is_good", "classification", "classification_reason", "stage", "is_incomplete", "best_model",
    "gompertz_a", "gompertz_a_err", "gompertz_mu", "gompertz_mu_err", "gompertz_lambda", "gompertz_lambda_err",
    "doubling_time_h", "fit_r_squared", "fit_rmse", "fit_mae",
    "delta_od", "max_od", "snr", "prefit_snr", "a_err_pct", "mu_err_pct",
    "truncation_time", "points_used", "baseline_std", "monotone_fraction", "residual_autocorr", "delta_od_ci_lower",
    "p_good", "ml_classification", "rule_based_result", "rule_based_reason",
]


@dataclass
class Gate:
    """One quality check with the value it saw and the threshold it applied."""
    name: str
    description: str
    value: Optional[float]
    threshold: Optional[float]
    comparison: str          # '>=' or '<=' : value must satisfy `value comparison threshold`
    passed: Optional[bool]   # None when the gate did not apply
    applies: bool = True
    note: str = ""

    def __post_init__(self):
        # numpy bools are not `is False`; normalise everything to Python types
        self.passed = None if self.passed is None else bool(self.passed)
        self.applies = bool(self.applies)
        self.value = None if self.value is None else float(self.value)
        self.threshold = None if self.threshold is None else float(self.threshold)

    @property
    def status(self) -> str:
        if not self.applies or self.passed is None:
            return "not applied"
        return "pass" if self.passed else "FAIL"


@dataclass
class CurveResult:
    """Everything the interface needs about one analysed curve."""
    curve_id: str
    media: str
    strain: str
    wells: List[str]
    n_replicates: int
    time_h: np.ndarray
    od: np.ndarray
    legacy: Any                     # ProcessingResult from the core script
    gates: List[Gate]
    settings: AnalysisSettings
    row: Dict[str, Any] = field(default_factory=dict)
    #: manual (start, end) cut applied instead of the automatic one, if any
    truncation_override: Optional[Tuple[int, int]] = None

    # ---------------------------------------------------------- properties
    @property
    def is_good(self) -> bool:
        return bool(self.legacy.classification.is_good)

    @property
    def fit(self):
        return self.legacy.fit

    @property
    def truncation(self):
        return self.legacy.truncation

    @property
    def fit_succeeded(self) -> bool:
        return bool(self.legacy.fit is not None and self.legacy.fit.success)

    def fitted_curve(self, n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth model curve over the truncated window (empty arrays if no fit)."""
        if not self.fit_succeeded or self.truncation is None or len(self.truncation.time_truncated) == 0:
            return np.array([]), np.array([])
        gca = legacy.growth_curve_analysis()
        t = np.linspace(float(self.truncation.time_truncated[0]), float(self.truncation.time_truncated[-1]), n)
        return t, gca.gompertz_model(t, self.fit.a_opt, self.fit.mu_opt, self.fit.lambda_opt)

    def failed_gates(self) -> List[Gate]:
        return [g for g in self.gates if g.applies and g.passed is False]

    def explanation(self) -> str:
        """One paragraph in plain language."""
        L = self.legacy
        if L.stage == "prefit_rejected":
            return ("Rejected before fitting: the signal barely rises above the baseline noise "
                    f"(pre-fit signal-to-noise {L.prefit_snr:.1f}, growth {L.classification.metrics.get('delta_od', float('nan')):.3f} OD).")
        if L.stage == "truncation_invalid":
            return f"No usable growth window could be found: {L.classification.reason.replace('BAD: ', '')}."
        f = L.fit
        if f is None or not f.success:
            return f"The Gompertz model could not be fitted: {getattr(f, 'error_message', 'unknown error')}."
        head = (f"Gompertz fit: A = {f.a_opt:.3f} OD, mu = {f.mu_opt:.3f} OD/h, lambda = {f.lambda_opt:.2f} h, "
                f"R² = {f.r_squared:.3f} on {len(L.truncation.od_truncated)} points up to {L.truncation.truncation_time:.1f} h.")
        if L.is_incomplete:
            head += " The curve was still rising when the run ended, so the whole run was used."
        fails = self.failed_gates()
        if self.is_good:
            return head + " All quality gates passed."
        if fails:
            return head + " Failed: " + "; ".join(f"{g.description} ({_fmt(g.value)} vs {g.comparison} {_fmt(g.threshold)})" for g in fails) + "."
        return head + " " + L.classification.reason


@dataclass
class PlateResults:
    table: pd.DataFrame
    curves: Dict[str, CurveResult]
    settings: AnalysisSettings
    qc: List[str]
    run_summary: Dict[str, Any]
    ml_used: bool = False

    @property
    def n_good(self) -> int:
        return int(self.table.is_good.sum())

    @property
    def n_total(self) -> int:
        return int(len(self.table))

    def curve(self, cid: str) -> CurveResult:
        return self.curves[cid]


# =============================================================================
# per-curve
# =============================================================================

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.3g}"


def _make_ml(settings: AnalysisSettings):
    """Instantiate the ML gate/classifier if requested and available; else (None, None)."""
    if not settings.use_ml or not legacy.model_files_present():
        return None, None
    ml = legacy.ml_classifier()
    gate = ml.PreFitGate(reject_threshold=settings.prefit_reject_threshold)
    clf = ml.PostFitClassifier(good_threshold=settings.postfit_good_threshold, bad_threshold=settings.postfit_bad_threshold)
    return (gate if gate.model is not None else None), (clf if clf.model is not None else None)


def build_gates(result, settings: AnalysisSettings, od: np.ndarray) -> List[Gate]:
    """Reconstruct every quality gate the core applied (or would apply) to this curve."""
    th = settings.fit_quality_thresholds()
    m = result.classification.metrics
    gates: List[Gate] = []

    pre_snr = result.prefit_snr
    delta_pre = m.get("delta_od", float("nan"))
    prefit_fail = result.stage == "prefit_rejected"
    gates.append(Gate(
        "prefit_snr", "Pre-fit signal-to-noise", pre_snr, th["pre_fit_snr_threshold"], ">=",
        passed=not prefit_fail, applies=True,
        note=f"skipped fitting unless growth ≥ {th['pre_fit_delta_od_override']} OD",
    ))
    if result.stage != "classified":
        return gates

    f = result.fit
    fit_ok = bool(f is not None and f.success)
    r2 = f.r_squared if fit_ok else float("nan")
    excellent = fit_ok and r2 >= th["excellent_r2_threshold"]
    gates.append(Gate("fit_success", "Model fit converged", 1.0 if fit_ok else 0.0, 1.0, ">=", fit_ok))
    gates.append(Gate("r_squared", "Fit quality R²", r2, th["min_r_squared"], ">=", fit_ok and r2 >= th["min_r_squared"], applies=fit_ok))
    a_err = m.get("a_err_pct"); mu_err = m.get("mu_err_pct")
    gates.append(Gate("a_err_pct", "Uncertainty of A (%)", a_err, th["max_param_error_pct"], "<=",
                      (a_err is not None and a_err <= th["max_param_error_pct"]) if fit_ok else None, applies=fit_ok))
    gates.append(Gate("mu_err_pct", "Uncertainty of mu (%)", mu_err, th["max_param_error_pct"], "<=",
                      (mu_err is not None and mu_err <= th["max_param_error_pct"]) if fit_ok else None, applies=fit_ok))

    have_secondary = "snr" in m and len(od) > 10
    if have_secondary:
        snr = m.get("snr"); ci = m.get("delta_od_ci_lower"); delta = m.get("delta_od")
        bstd = m.get("baseline_std", float("nan")); mono = m.get("monotone_fraction"); ac = m.get("residual_autocorr")
        gates.append(Gate("snr", "Signal-to-noise", snr, th["min_snr"], ">=", snr >= th["min_snr"] if snr is not None else None,
                          applies=not excellent, note="bypassed for excellent fits" if excellent else ""))
        gates.append(Gate("delta_od_ci", "Growth minus 2×noise", ci, th["min_delta_od_ci"], ">=",
                          ci >= th["min_delta_od_ci"] if ci is not None else None,
                          applies=not excellent, note="bypassed for excellent fits" if excellent else ""))
        gates.append(Gate("delta_od", "Total growth (ΔOD)", delta, th["min_absolute_delta_od"], ">=",
                          delta >= th["min_absolute_delta_od"] if delta is not None else None))
        if fit_ok:
            gates.append(Gate("amplitude_vs_noise", "Fitted A vs 3×baseline noise", f.a_opt, 3 * bstd, ">=", f.a_opt >= 3 * bstd))
        gates.append(Gate("monotone", "Fraction of rising steps (first 60%)", mono, th["min_monotone_fraction"], ">=",
                          mono >= th["min_monotone_fraction"] if mono is not None else None,
                          applies=not excellent, note="bypassed for excellent fits" if excellent else ""))
        if ac is not None:
            applies_ac = (not excellent) and (not result.is_incomplete)
            gates.append(Gate("residual_autocorr", "Residual lag-1 autocorrelation", ac, th["max_residual_autocorr"], "<=",
                              ac <= th["max_residual_autocorr"], applies=applies_ac,
                              note="bypassed for excellent fits or incomplete curves" if not applies_ac else ""))
    if "p_good" in m and m.get("ml_classification") is not None:
        p = m.get("p_good")
        gates.append(Gate("ml_p_good", "ML probability of a good curve", p, settings.postfit_good_threshold, ">=",
                          p >= settings.postfit_good_threshold if p is not None else None,
                          note=f"BAD below {settings.postfit_bad_threshold}, BORDERLINE in between"))
    return gates


def analyze_curve(
    time_h: np.ndarray,
    od: np.ndarray,
    curve_id: str = "curve",
    settings: Optional[AnalysisSettings] = None,
    media: str = "",
    strain: str = "",
    wells: Optional[Sequence[str]] = None,
    _ml=None,
    truncation_override: Optional[Tuple[int, int]] = None,
) -> CurveResult:
    """
    Analyse one curve (time in hours, blanked OD) and explain the verdict.

    Parameters
    ----------
    time_h, od : np.ndarray
        Time in hours and blank-corrected OD600. Rows where either is NaN are
        dropped before anything else happens.
    curve_id, media, strain, wells
        Labels carried through to the results table.
    settings : AnalysisSettings, optional
        Thresholds; defaults to ``scripts/config.yaml``.
    truncation_override : tuple of (int, int), optional
        Manual growth window as ``(start_index, end_index)`` into the cleaned
        arrays (inclusive, i.e. the indices of ``cr.time_h``/``cr.od`` after
        NaN removal). Use it when a person moves the cut by hand; everything
        after truncation -- fit, gates, verdict -- is unchanged.
    """
    settings = settings or AnalysisSettings.from_config()
    gca = legacy.growth_curve_analysis()
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(od, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(t)
    t, y = t[mask], y[mask]
    gate, clf = _ml if _ml is not None else _make_ml(settings)

    if len(y) < settings.min_points_for_fit:
        cls = gca.ClassificationResult(is_good=False,
                                       reason=f"BAD: Insufficient data points ({len(y)} < {settings.min_points_for_fit})",
                                       metrics={"delta_od": float(np.max(y) - np.mean(y[:5])) if len(y) else float("nan"),
                                                "max_od": float(np.max(y)) if len(y) else float("nan")})
        res = gca.ProcessingResult(strain_name=curve_id, classification=cls, truncation=None, fit=None, stage="too_few_points")
    else:
        res = gca.analyze_curve(
            t, y, curve_id,
            fit_quality_thresholds=settings.fit_quality_thresholds(),
            truncation_params=settings.truncation_params(),
            prefit_gate=gate, postfit_clf=clf, verbose=False,
            truncation_override=None if truncation_override is None
            else (int(truncation_override[0]), int(truncation_override[1])),
        )
    gates = build_gates(res, settings, y) if res.stage != "too_few_points" else []
    cr = CurveResult(curve_id=curve_id, media=media, strain=strain, wells=list(wells or []),
                     n_replicates=len(wells or []), time_h=t, od=y, legacy=res, gates=gates, settings=settings,
                     truncation_override=None if truncation_override is None
                     else (int(truncation_override[0]), int(truncation_override[1])))
    cr.row = result_row(cr)
    return cr


def reclassify(cr: CurveResult, settings: AnalysisSettings) -> CurveResult:
    """
    Re-apply the quality gates to an already fitted curve, without refitting.

    Only the *post-fit* thresholds (R², parameter error, SNR, ΔOD, monotone
    fraction, autocorrelation) are re-evaluated, on the Gompertz fit that is
    already stored in ``cr``. That is exactly what happens when a slider in
    the interface changes a verdict threshold, and it is instant.

    Changing a threshold that acts *before* the fit (truncation window,
    smoothing, pre-fit signal-to-noise) does not belong here: rerun
    :func:`analyze_curve` for those.

    Parameters
    ----------
    cr : CurveResult
        A curve analysed earlier by :func:`analyze_curve`.
    settings : AnalysisSettings
        The new thresholds.

    Returns
    -------
    CurveResult
        A new result (the input is left untouched) whose verdict, gate table
        and results row reflect ``settings``.
    """
    _, clf = _make_ml(settings)
    if clf is not None:
        # The ML classifier is the call of record; it needs the full per-curve
        # path. Reuse the cut that was already chosen so nothing is refit twice.
        override = cr.truncation_override
        if override is None and cr.legacy.stage == "classified" and cr.truncation is not None:
            override = (int(cr.truncation.start_index), int(cr.truncation.truncation_index))
        return analyze_curve(cr.time_h, cr.od, cr.curve_id, settings, media=cr.media, strain=cr.strain,
                             wells=cr.wells, truncation_override=override)

    L = cr.legacy
    if L.stage == "classified":
        gca = legacy.growth_curve_analysis()
        cls = gca.classify_by_fit_quality(
            L.fit, settings.fit_quality_thresholds(),
            time=cr.time_h, od600=cr.od, is_incomplete=L.is_incomplete,
        )
        if "delta_od" not in cls.metrics:
            cls.metrics["delta_od"] = np.max(cr.od) - np.mean(cr.od[:5])
        cls.metrics["max_od"] = np.max(cr.od)
        cls.metrics["best_model"] = L.best_model
        new_legacy = replace(L, classification=cls)
    else:
        # Rejected before fitting, unusable cut, or too few points: no post-fit
        # gate applies, so only the gate table is rebuilt against the new numbers.
        new_legacy = replace(L, classification=copy.deepcopy(L.classification))

    gates = build_gates(new_legacy, settings, cr.od) if new_legacy.stage != "too_few_points" else []
    out = CurveResult(curve_id=cr.curve_id, media=cr.media, strain=cr.strain, wells=list(cr.wells),
                      n_replicates=cr.n_replicates, time_h=cr.time_h, od=cr.od, legacy=new_legacy,
                      gates=gates, settings=settings, truncation_override=cr.truncation_override)
    out.row = result_row(out)
    return out


def result_row(cr: CurveResult) -> Dict[str, Any]:
    """Flatten a CurveResult into the results-table row (same fields as processing_results.csv)."""
    L = cr.legacy
    m = L.classification.metrics
    f = L.fit
    fit_ok = bool(f is not None and f.success)
    row: Dict[str, Any] = {
        "curve_id": cr.curve_id, "media": cr.media, "strain": cr.strain,
        "wells": ",".join(cr.wells), "n_replicates": cr.n_replicates,
        "is_good": bool(L.classification.is_good),
        "classification": m.get("ml_classification") or ("GOOD" if L.classification.is_good else "BAD"),
        "classification_reason": L.classification.reason,
        "stage": L.stage, "is_incomplete": bool(L.is_incomplete), "best_model": m.get("best_model", L.best_model),
        "gompertz_a": f.a_opt if fit_ok else None, "gompertz_a_err": f.a_err if fit_ok else None,
        "gompertz_mu": f.mu_opt if fit_ok else None, "gompertz_mu_err": f.mu_err if fit_ok else None,
        "gompertz_lambda": f.lambda_opt if fit_ok else None, "gompertz_lambda_err": f.lambda_err if fit_ok else None,
        "doubling_time_h": doubling_time_h(f.a_opt, f.mu_opt) if fit_ok else None,
        "fit_r_squared": f.r_squared if fit_ok else None, "fit_rmse": f.rmse if fit_ok else None, "fit_mae": f.mae if fit_ok else None,
        "delta_od": m.get("delta_od"), "max_od": m.get("max_od"), "snr": m.get("snr"), "prefit_snr": L.prefit_snr,
        "a_err_pct": m.get("a_err_pct"), "mu_err_pct": m.get("mu_err_pct"),
        "truncation_time": L.truncation.truncation_time if L.truncation is not None else None,
        "points_used": len(L.truncation.od_truncated) if L.truncation is not None else None,
        "baseline_std": m.get("baseline_std"), "monotone_fraction": m.get("monotone_fraction"),
        "residual_autocorr": m.get("residual_autocorr"), "delta_od_ci_lower": m.get("delta_od_ci_lower"),
        "p_good": m.get("p_good"), "ml_classification": m.get("ml_classification"),
        "rule_based_result": m.get("rule_based_result"), "rule_based_reason": m.get("rule_based_reason"),
    }
    return row


def doubling_time_h(a: float, mu: float) -> Optional[float]:
    """
    Minimum doubling time (h) at the point of fastest growth.

    ``mu`` from the modified Gompertz model is an *absolute* rate (OD/h). At the
    inflection point the OD is A/e, so the specific rate there is mu·e/A and the
    doubling time is ln(2)·A/(mu·e).
    """
    if a is None or mu is None or not np.isfinite(a) or not np.isfinite(mu) or mu <= 0 or a <= 0:
        return None
    return float(math.log(2) * a / (mu * math.e))


# =============================================================================
# per-plate
# =============================================================================

def _worker(args):
    cid, t, y, settings, media, strain, wells = args
    cr = analyze_curve(t, y, cid, settings, media=media, strain=strain, wells=wells)
    return cr


def analyze_plate(
    processed: ProcessedPlate,
    settings: Optional[AnalysisSettings] = None,
    curve_ids: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> PlateResults:
    """
    Analyse every curve of a processed plate.

    Parameters
    ----------
    processed : ProcessedPlate
    settings : AnalysisSettings, optional
    curve_ids : sequence of str, optional
        Subset of curves to analyse (default: all).
    n_jobs : int
        Worker processes; 1 = sequential (safest, shows progress per curve).
    progress : callable(done, total, curve_id), optional
        Called after each curve.
    """
    settings = settings or AnalysisSettings.from_config()
    cond = processed.conditions
    ids = list(curve_ids) if curve_ids is not None else list(cond.curve_id)
    meta = cond.set_index("curve_id")
    jobs = []
    for cid in ids:
        t, y = processed.curve(cid)
        r = meta.loc[cid]
        jobs.append((cid, t, y, settings, str(r.media), str(r.strain), list(r.wells)))

    curves: Dict[str, CurveResult] = {}
    ml = _make_ml(settings)
    done = 0
    if n_jobs and n_jobs > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(n_jobs, os.cpu_count() or 1)) as ex:
                futs = {ex.submit(_worker, j): j[0] for j in jobs}
                for fut in as_completed(futs):
                    cr = fut.result()
                    curves[cr.curve_id] = cr
                    done += 1
                    if progress:
                        progress(done, len(jobs), cr.curve_id)
        except Exception:
            curves.clear(); done = 0  # fall back to the sequential path below
    if len(curves) < len(jobs):
        for cid, t, y, s, media, strain, wells in jobs:
            if cid in curves:
                continue
            curves[cid] = analyze_curve(t, y, cid, s, media=media, strain=strain, wells=wells, _ml=ml)
            done += 1
            if progress:
                progress(done, len(jobs), cid)

    rows = [curves[cid].row for cid in ids]
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return PlateResults(table=table, curves=curves, settings=settings, qc=list(processed.qc),
                        run_summary=dict(processed.run_summary), ml_used=ml[1] is not None)


def reclassify_plate(results: PlateResults, settings: AnalysisSettings) -> PlateResults:
    """
    Apply :func:`reclassify` to every curve of a plate and rebuild the table.

    Use it when only post-fit thresholds moved: no curve is refitted, so a
    96-well plate updates in milliseconds instead of seconds.
    """
    curves = {cid: reclassify(cr, settings) for cid, cr in results.curves.items()}
    ids = list(results.table.curve_id)
    table = pd.DataFrame([curves[cid].row for cid in ids], columns=RESULT_COLUMNS)
    return PlateResults(table=table, curves=curves, settings=settings, qc=list(results.qc),
                        run_summary=dict(results.run_summary), ml_used=results.ml_used)


def legacy_results_frame(results: PlateResults) -> pd.DataFrame:
    """The exact column set of the CLI's processing_results.csv (for comparisons/exports)."""
    t = results.table
    out = pd.DataFrame({
        "strain": t.curve_id, "is_good": t.is_good, "classification_reason": t.classification_reason,
        "delta_od": t.delta_od, "max_od": t.max_od, "initial_od": None, "snr": t.snr, "time_to_max": None,
        "r_squared": t.fit_r_squared, "a_err_pct": t.a_err_pct, "mu_err_pct": t.mu_err_pct, "best_model": t.best_model,
        "truncation_time": t.truncation_time, "points_used": t.points_used,
        "gompertz_a": t.gompertz_a, "gompertz_a_err": t.gompertz_a_err, "gompertz_mu": t.gompertz_mu, "gompertz_mu_err": t.gompertz_mu_err,
        "gompertz_lambda": t.gompertz_lambda, "gompertz_lambda_err": t.gompertz_lambda_err,
        "fit_r_squared": t.fit_r_squared, "fit_rmse": t.fit_rmse, "fit_mae": t.fit_mae,
        "baseline_std": t.baseline_std, "monotone_fraction": t.monotone_fraction, "residual_autocorr": t.residual_autocorr,
        "delta_od_ci_lower": t.delta_od_ci_lower, "p_good": t.p_good, "ml_classification": t.ml_classification,
    })
    return out
