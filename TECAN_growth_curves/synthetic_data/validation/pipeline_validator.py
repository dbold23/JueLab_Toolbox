"""
Pipeline Validator -- the metric library behind the synthetic benchmark.

Runs synthetic data through ``scripts/01_growth_curve_analysis.py`` and scores
the result against ground truth.  ``benchmark.py`` is the entry point that
assembles these functions into one run directory; this module holds the
measurements and keeps the historical ``PipelineValidator`` class and CLI
working for ``scripts/run_full_pipeline.py`` step 9.

What changed and why (audit findings)
-------------------------------------
M19  The join moved to :mod:`matching`, is computed once per validator
     instance, and refuses to publish a silently partial benchmark.
M14  ``run_pipeline`` takes ``truncation_mode`` with no default and appends the
     matching flag in *both* directions.  Before, ``use_adaptive=False``
     appended nothing and ``01_growth_curve_analysis.py`` resolved
     ``use_adaptive: true`` from ``scripts/config.yaml``, so the first-peak
     branch of this API was unreachable and every "first-peak" run through it
     was actually adaptive.  The 1800 s subprocess timeout (shorter than the
     adaptive run it defaulted to -- the committed 480-curve adaptive run was
     still going at 6685 s) is now a constructor argument defaulting to None.
     ``load_results`` and ``generate_validation_report`` create their output
     directory, so ``--skip-run`` with a fresh ``--output-dir`` no longer dies
     with FileNotFoundError.
M17  Gate attribution: which gate fired on each miss, parsed from
     ``failed_gates`` when the pipeline writes it and from
     ``classification_reason`` otherwise, with "never fitted" separated from
     "fitted then rejected".
M13  Parameter recovery is reported in four strata with their own n, so
     tightening a gate can no longer masquerade as better recovery.
M18  Uncertainty calibration: the pipeline's own error bars, measured against
     the truth they claim to cover.
M11  Specificity, balanced accuracy, prevalence and per-scenario/per-band
     breakdowns, none of which existed in either JSON.

Nothing here changes a gate, a threshold or a verdict.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # package import
    from .matching import BenchmarkMatchError, extract_curve_id, match_results
except ImportError:  # executed as a script (``python .../pipeline_validator.py``)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.matching import (  # type: ignore  # noqa: F401
        BenchmarkMatchError,
        extract_curve_id,
        match_results,
    )


# =============================================================================
# Constants
# =============================================================================

#: The 32 scenarios of the first published benchmark.  ``classification.core32``
#: is restricted to these so the headline stays comparable across releases even
#: after new (harder) scenarios are added.
CORE32_SCENARIOS: Tuple[str, ...] = (
    "baranyi_generated", "borderline_delta_od", "borderline_noise",
    "borderline_r2_bad", "borderline_r2_good", "contamination",
    "death_phase_moderate", "death_phase_severe", "dense_data",
    "drift_positive", "erratic", "fast_growth", "fit_failure",
    "flat_no_growth", "high_A", "high_noise", "logistic_generated",
    "long_experiment", "long_lag", "low_A", "minimal_growth",
    "outlier_contaminated", "pesticide_lb_typical", "pesticide_only",
    "richards_asymmetric", "short_experiment", "short_lag", "slow_growth",
    "sparse_data", "standard", "truncation_challenge", "very_clean",
)

#: Default parameter-recovery tolerances.  A and mu are relative, lambda is an
#: absolute number of hours (a relative tolerance is meaningless at lambda ~ 0).
DEFAULT_TOLERANCES: Dict[str, float] = {"A": 0.10, "mu": 0.15, "lambda": 1.0}

#: (name, ground-truth column, results column, standard-error column)
PARAM_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("A", "true_A", "gompertz_a", "gompertz_a_err"),
    ("mu", "true_mu", "gompertz_mu", "gompertz_mu_err"),
    ("lambda", "true_lambda", "gompertz_lambda", "gompertz_lambda_err"),
)

#: The nominal ("requested") counterparts, present only in ground truth v2.
REQUESTED_COLUMNS: Dict[str, str] = {
    "A": "requested_A", "mu": "requested_mu", "lambda": "requested_lambda",
}

#: delta-OD bands and the fraction of real conditions in each.  Measured in
#: M11 over the 92 conditions of ``data/raw/Group1..4`` (total rise per
#: condition): 31.5 / 12.0 / 8.7 / 1.1 / 2.2 / 44.6 %.  Overridden by
#: ``synthetic_data/config/lab_prior.json`` when WP5 has written it.
DEFAULT_LAB_PRIOR: Dict[str, Any] = {
    "source": "M11 real-plate scan of data/raw/Group1..4 (92 conditions)",
    "bands": [
        {"name": "<0.02",    "min": -math.inf, "max": 0.02,      "fraction": 0.315},
        {"name": "0.02-0.1", "min": 0.02,      "max": 0.10,      "fraction": 0.120},
        {"name": "0.1-0.3",  "min": 0.10,      "max": 0.30,      "fraction": 0.087},
        {"name": "0.3-0.6",  "min": 0.30,      "max": 0.60,      "fraction": 0.011},
        {"name": "0.6-1.2",  "min": 0.60,      "max": 1.20,      "fraction": 0.022},
        {"name": ">1.2",     "min": 1.20,      "max": math.inf,  "fraction": 0.446},
    ],
}

_LAB_PRIOR_PATH = Path(__file__).resolve().parents[1] / "config" / "lab_prior.json"

#: ``config/lab_prior.json`` names its edges ``delta_od_min``/``delta_od_max``
#: (``derive_lab_prior.py``).  Reading only ``min``/``max`` silently gave every
#: band (-inf, +inf), so every curve landed in the first band and the whole
#: stratified view collapsed - hence both spellings, and an explicit null.
_BAND_MIN_KEYS = ("delta_od_min", "min", "lower", "min_delta_od")
_BAND_MAX_KEYS = ("delta_od_max", "max", "upper", "max_delta_od")


def _band_edge(band: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    """First present, non-null edge among ``keys``; ``default`` (+/-inf) if none."""
    for key in keys:
        if key in band and band[key] is not None:
            return float(band[key])
    return default


# =============================================================================
# Gate attribution (M17)
# =============================================================================

class GateParseError(ValueError):
    """A ``classification_reason`` clause that maps to no known gate.

    Raised rather than silently dropped: a reworded pipeline message must break
    the test suite, not quietly empty the attribution table.
    """


_NUM = r"[-+]?(?:inf|nan|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

#: canonical gate id -> (clause regex, comparison direction)
#:
#: The ids are the ones ``tecan_growth/analyze.py:174-231`` already uses for the
#: same gates, plus ``too_few_points`` and ``prefit_rejected`` for the two
#: pre-fit exits that never reach ``build_gates``.  The regexes read the
#: human-readable message emitted by ``scripts/01_growth_curve_analysis.py``
#: (line numbers in comments) and are the documented fallback for CSVs written
#: before the pipeline gained a ``failed_gates`` column.
GATE_SPECS: Dict[str, Tuple[str, str]] = {
    # 01_growth_curve_analysis.py:2607
    "prefit_snr": (rf"Pre-fit SNR too low \(({_NUM}) < ({_NUM})\)", ">="),
    # 01_growth_curve_analysis.py:2593
    "prefit_rejected": (r"Rejected by ML pre-fit gate", ""),
    # 01_growth_curve_analysis.py:1674
    "too_few_points": (r"Too few points after truncation: (\d+) < (\d+)", ">="),
    # 01_growth_curve_analysis.py:275
    "fit_success": (r"Fit failed", ""),
    # 01_growth_curve_analysis.py:281
    "r_squared": (rf"Low R(?:²|2): ({_NUM}) < ({_NUM})", ">="),
    # 01_growth_curve_analysis.py:287
    "a_err_pct": (rf"High A error: ({_NUM})% > ({_NUM})%", "<="),
    # 01_growth_curve_analysis.py:289
    "mu_err_pct": (rf"High mu error: ({_NUM})% > ({_NUM})%", "<="),
    # 01_growth_curve_analysis.py:328
    "snr": (rf"Low SNR: ({_NUM}) < ({_NUM})", ">="),
    # 01_growth_curve_analysis.py:333
    "delta_od_ci": (rf"Delta OD not significant: CI lower=({_NUM}) < ({_NUM})", ">="),
    # 01_growth_curve_analysis.py:338
    "delta_od": (rf"Insufficient growth: delta_od=({_NUM}) < ({_NUM})", ">="),
    # 01_growth_curve_analysis.py:341
    "amplitude_vs_noise": (
        rf"Growth amplitude below noise: A=({_NUM}) < 3\*noise=({_NUM})", ">="),
    # 01_growth_curve_analysis.py:367
    "monotone": (rf"Non-monotonic signal: ({_NUM}) < ({_NUM})", ">="),
    # 01_growth_curve_analysis.py:398
    "residual_autocorr": (
        rf"High residual autocorrelation: ({_NUM}) > ({_NUM})", "<="),
    # ml_classifier.py:370
    "ml_p_good": (r"\(ML\): P\(good\)=(" + _NUM + r")", ">="),
}

#: Canonical order used for every gate column / matrix in the report.
GATE_IDS: Tuple[str, ...] = (
    "prefit_snr", "prefit_rejected", "too_few_points", "fit_success",
    "r_squared", "a_err_pct", "mu_err_pct", "snr", "delta_od_ci", "delta_od",
    "amplitude_vs_noise", "monotone", "residual_autocorr", "ml_p_good",
)

#: Gates that fire before or instead of a model fit; a curve that trips one of
#: these has no fitted parameters at all.
UNFITTED_GATES = frozenset({"prefit_snr", "prefit_rejected", "too_few_points", "fit_success"})

_GATE_RES = {gid: re.compile(pat) for gid, (pat, _) in GATE_SPECS.items()}


def _to_float(token: str) -> float:
    token = token.strip().lower()
    if token in ("inf", "+inf"):
        return math.inf
    if token == "-inf":
        return -math.inf
    if token == "nan":
        return math.nan
    try:
        return float(token)
    except ValueError:
        return math.nan


def parse_reason_gates(reason, strict: bool = True) -> List[Dict[str, Any]]:
    """Parse one ``classification_reason`` into the gates it names.

    Returns a list of ``{'gate', 'observed', 'threshold', 'margin'}`` dicts;
    ``margin`` is how far the gate was missed by (always >= 0 for a real
    failure), or NaN for the two gates that carry no numbers.

    ``strict`` makes an unrecognised clause a :class:`GateParseError` instead of
    a silent drop.
    """
    text = "" if reason is None else str(reason).strip()
    if not text or text.lower() in ("nan", "none"):
        return []

    verdict_good = text.upper().startswith("GOOD")
    if verdict_good:
        return []

    # ML post-fit classifier: "BAD (ML): P(good)=0.123"
    if "(ML): P(good)=" in text:
        hit = _GATE_RES["ml_p_good"].search(text)
        observed = _to_float(hit.group(1)) if hit else math.nan
        return [{"gate": "ml_p_good", "observed": observed,
                 "threshold": math.nan, "margin": math.nan}]

    body = re.sub(r"^(?:BAD|GOOD)\s*:\s*", "", text)

    # "BAD: Fit failed - <scipy message>" -- the message may itself contain ';'
    if body.startswith("Fit failed"):
        return [{"gate": "fit_success", "observed": 0.0,
                 "threshold": 1.0, "margin": 1.0}]

    out: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for clause in [c.strip() for c in body.split(";") if c.strip()]:
        matched = False
        for gid in GATE_IDS:
            rx = _GATE_RES[gid]
            hit = rx.search(clause)
            if not hit:
                continue
            direction = GATE_SPECS[gid][1]
            observed = threshold = margin = math.nan
            if hit.groups():
                observed = _to_float(hit.group(1))
                if len(hit.groups()) >= 2:
                    threshold = _to_float(hit.group(2))
            if direction == ">=":
                margin = threshold - observed
            elif direction == "<=":
                margin = observed - threshold
            out.append({"gate": gid, "observed": observed,
                        "threshold": threshold, "margin": margin})
            matched = True
            break
        if not matched:
            unknown.append(clause)

    if unknown and strict:
        raise GateParseError(
            "classification_reason clause(s) map to no known gate: "
            f"{unknown!r} (full reason: {text!r}). Update GATE_SPECS in "
            "synthetic_data/validation/pipeline_validator.py when the pipeline "
            "rewords a message."
        )
    return out


def _gates_from_column(value) -> Optional[List[str]]:
    """Decode WP9's ``failed_gates`` cell, or None when it carries no signal."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in ("nan", "none", "<na>"):
        return None
    if text == "":
        return []
    return [g.strip() for g in text.split(";") if g.strip()]


def gate_attribution(matched: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """One row per matched curve naming the gates that fired on it.

    Prefers the pipeline's machine-readable ``failed_gates`` column (WP9) and
    falls back to parsing ``classification_reason``; ``gate_source`` records
    which was used per row.  ``margin_<gate>`` is the numeric amount by which
    that gate was missed, and is only ever available from the reason text.
    """
    matched = matched.reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    have_column = "failed_gates" in matched.columns
    has_fit = "gompertz_a" in matched.columns
    a_err = matched["a_err_pct"] if "a_err_pct" in matched.columns else None

    for i, (_, row) in enumerate(matched.iterrows()):
        reason = row.get("classification_reason")
        parsed = parse_reason_gates(reason, strict=strict)
        parsed_ids = [p["gate"] for p in parsed]

        gt_good = bool(row.get("gt_is_good", False))
        pred_good = bool(row.get("pred_is_good", False))

        column_ids = _gates_from_column(row.get("failed_gates")) if have_column else None
        if column_ids is None and have_column and pred_good:
            # WP9 writes an empty string for a GOOD curve, which comes back from
            # the CSV as NaN; for an accepted curve that is genuinely "no gate
            # failed", not "no information".
            column_ids = []
        if column_ids is not None:
            gates = list(column_ids)
            source = "failed_gates"
        else:
            gates = parsed_ids
            source = "classification_reason"

        fitted = bool(pd.notna(row.get("gompertz_a"))) if has_fit else (
            not (set(gates) & UNFITTED_GATES))

        record: Dict[str, Any] = {
            "curve_id": row.get("curve_id"),
            "scenario": row.get("scenario"),
            "expected_class": row.get("expected_class"),
            "expected_is_good": gt_good,
            "predicted_is_good": pred_good,
            "correct": gt_good == pred_good,
            "error_type": (
                "" if gt_good == pred_good
                else ("false_negative" if gt_good else "false_positive")),
            "fitted": fitted,
            "gate_source": source,
            "failed_gates": ";".join(gates),
            "n_failed_gates": len(gates),
            "unfitted_gate": bool(set(gates) & UNFITTED_GATES),
            "degenerate_covariance": bool(
                a_err is not None and np.isinf(_safe_float(a_err.iloc[i]))),
            "classification_reason": reason,
        }
        margins = {p["gate"]: p for p in parsed}
        for gid in GATE_IDS:
            record[f"gate_{gid}"] = gid in gates
            m = margins.get(gid)
            record[f"margin_{gid}"] = m["margin"] if m else math.nan
            record[f"observed_{gid}"] = m["observed"] if m else math.nan
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.attrs["gate_ids"] = list(GATE_IDS)
    return frame


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def failure_analysis(matched: pd.DataFrame, attribution: Optional[pd.DataFrame] = None,
                     strict: bool = True) -> Dict[str, Any]:
    """The ``failure_analysis`` report block, including the scenario x gate matrix."""
    if attribution is None:
        attribution = gate_attribution(matched, strict=strict)
    attribution = attribution.reset_index(drop=True)

    gt_good = attribution["expected_is_good"].to_numpy(dtype=bool)
    pred_good = attribution["predicted_is_good"].to_numpy(dtype=bool)
    fn_mask = gt_good & ~pred_good
    fp_mask = ~gt_good & pred_good
    mis = attribution[~attribution["correct"]].copy()

    fn_rows = attribution.loc[fn_mask]
    n_fn_unfitted = int((~fn_rows["fitted"]).sum())
    n_fn_gated = int(fn_rows["fitted"].sum())

    by_gate: Dict[str, Dict[str, int]] = {}
    for scenario, sub in mis.groupby("scenario", dropna=False):
        counts = {gid: int(sub[f"gate_{gid}"].sum()) for gid in GATE_IDS}
        counts = {gid: n for gid, n in counts.items() if n}
        by_gate[str(scenario)] = counts

    by_gate_all_rejected: Dict[str, Dict[str, int]] = {}
    rejected = attribution[~attribution["predicted_is_good"]]
    for scenario, sub in rejected.groupby("scenario", dropna=False):
        counts = {gid: int(sub[f"gate_{gid}"].sum()) for gid in GATE_IDS}
        by_gate_all_rejected[str(scenario)] = {g: n for g, n in counts.items() if n}

    fp_rows = attribution.loc[fp_mask]
    degenerate = attribution[attribution["degenerate_covariance"]]

    return {
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        # M17: 'never fitted' and 'fitted then rejected' are different failures
        # and used to be merged into one count of 26.
        "n_false_negatives_unfitted": n_fn_unfitted,
        "n_false_negatives_gated": n_fn_gated,
        "n_false_positives_with_no_failed_gate": int((fp_rows["n_failed_gates"] == 0).sum()),
        "failure_scenarios": (
            mis["scenario"].value_counts().to_dict() if "scenario" in mis else {}),
        "false_negative_scenarios_unfitted": (
            fn_rows[~fn_rows["fitted"]]["scenario"].value_counts().to_dict()),
        "false_negative_scenarios_gated": (
            fn_rows[fn_rows["fitted"]]["scenario"].value_counts().to_dict()),
        "by_gate": by_gate,
        "by_gate_all_rejected": by_gate_all_rejected,
        "gate_totals_misclassified": {
            gid: int(mis[f"gate_{gid}"].sum()) for gid in GATE_IDS
            if int(mis[f"gate_{gid}"].sum())
        },
        # An at-bound singular covariance, not a near-miss: a_err_pct is inf,
        # so 'High A error' says the fit could not put a number on A at all.
        "degenerate_covariance": {
            "n": int(len(degenerate)),
            "definition": "a_err_pct is inf (singular covariance at a bound)",
            "by_scenario": degenerate["scenario"].value_counts().to_dict(),
            "n_misclassified": int((~degenerate["correct"]).sum()),
        },
        "gate_source": (
            attribution["gate_source"].value_counts().to_dict()
            if len(attribution) else {}),
    }


# =============================================================================
# Classification metrics (M11)
# =============================================================================

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: List[List[int]]


@dataclass
class ParameterRecovery:
    """Container for parameter recovery metrics."""
    parameter: str
    mean_error: float
    rmse: float
    r_squared: float
    bias: float
    n_samples: int


def confusion(gt_good: np.ndarray, pred_good: np.ndarray) -> Dict[str, int]:
    gt_good = np.asarray(gt_good, dtype=bool)
    pred_good = np.asarray(pred_good, dtype=bool)
    return {
        "true_positives": int(np.sum(gt_good & pred_good)),
        "true_negatives": int(np.sum(~gt_good & ~pred_good)),
        "false_positives": int(np.sum(~gt_good & pred_good)),
        "false_negatives": int(np.sum(gt_good & ~pred_good)),
    }


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def core_classification(gt_good: np.ndarray, pred_good: np.ndarray) -> Dict[str, Any]:
    """Every scalar the confusion matrix supports, ordered honestly.

    Balanced accuracy, sensitivity and specificity lead; raw accuracy follows
    with the prevalence that produced it, because on a 71.9 %-GOOD set the two
    differ by four points (0.873 vs 0.833) for reasons that have nothing to do
    with the pipeline.
    """
    cm = confusion(gt_good, pred_good)
    tp, tn = cm["true_positives"], cm["true_negatives"]
    fp, fn = cm["false_positives"], cm["false_negatives"]
    n = tp + tn + fp + fn

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    accuracy = _ratio(tp + tn, n)
    f1 = _ratio(2 * precision * sensitivity, precision + sensitivity)
    mcc_den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "prevalence": _ratio(tp + fn, n),
        "precision": precision,
        "recall": sensitivity,
        "npv": npv,
        "f1_score": f1,
        "mcc": mcc,
        "n": int(n),
        "confusion_matrix": cm,
    }


def load_lab_prior(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``synthetic_data/config/lab_prior.json``, or fall back to M11.

    The fallback fractions are the measured real-plate ones, not invented
    weights, and ``source`` says which of the two was used.
    """
    path = Path(path) if path is not None else _LAB_PRIOR_PATH
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return dict(DEFAULT_LAB_PRIOR, source=DEFAULT_LAB_PRIOR["source"] +
                        f" (unreadable {path.name}, using measured defaults)")
        bands = raw.get("bands") or raw.get("delta_od_bands")
        if isinstance(bands, list) and bands:
            normalised = []
            for band in bands:
                normalised.append({
                    "name": str(band.get("name", band.get("label", "?"))),
                    "min": _band_edge(band, _BAND_MIN_KEYS, -math.inf),
                    "max": _band_edge(band, _BAND_MAX_KEYS, math.inf),
                    "fraction": float(band.get("fraction", band.get("observed_fraction", 0.0))),
                    "description": band.get("description"),
                })
            # The derived file states the histogram edges it measured (the
            # lowest is 0.0, the highest open); open the two outer edges so the
            # bands partition the line and a curve with a slightly negative or
            # very large delta_od is banded rather than silently dropped.
            normalised.sort(key=lambda b: b["min"])
            normalised[0]["min"] = -math.inf
            normalised[-1]["max"] = math.inf
            return {"source": str(path), "bands": normalised, "raw": raw}
    return dict(DEFAULT_LAB_PRIOR)


def _band_of(value: float, bands: Sequence[Dict[str, Any]]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for band in bands:
        if band["min"] <= value < band["max"]:
            return band["name"]
    return None


def _truth_delta_od(matched: pd.DataFrame) -> Optional[pd.Series]:
    """The ground truth's own delta_od (never the pipeline's measurement)."""
    for column in ("delta_od_gt", "delta_od"):
        if column in matched.columns:
            # ``delta_od`` unsuffixed only exists when the results frame had no
            # column of that name, in which case it is the ground truth's.
            if column == "delta_od" and "delta_od_pred" in matched.columns:
                continue
            return pd.to_numeric(matched[column], errors="coerce")
    return None


def _declared_lab_band(matched: pd.DataFrame) -> Optional[pd.Series]:
    """The ground truth's own ``lab_band`` label, when the truth carries one."""
    for column in ("lab_band_gt", "lab_band"):
        if column in matched.columns:
            if column == "lab_band" and "lab_band_pred" in matched.columns:
                continue
            series = matched[column].astype("object")
            return series.where(series.notna() & (series.astype(str) != ""), None)
    return None


def _assign_bands(matched: pd.DataFrame,
                  bands: Sequence[Dict[str, Any]]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Band every row, preferring the measured ground-truth ``delta_od``.

    The declared ``lab_band`` of the ground truth is what the *sampler asked
    for*; the band a curve actually landed in is a property of the curve that
    was written, so the measured ``delta_od`` wins and the declared label is
    only a fallback (v1 ground truth has no ``delta_od``-free rows, but a
    truncated or hand-made truth file can).  The disagreement between the two
    is counted rather than hidden.
    """
    known = {band["name"] for band in bands}
    delta = _truth_delta_od(matched)
    declared = _declared_lab_band(matched)

    delta_values = ([None] * len(matched) if delta is None
                    else list(delta.to_numpy(dtype=float)))
    declared_values = ([None] * len(matched) if declared is None
                       else list(declared.to_numpy(dtype=object)))

    measured: List[Optional[str]] = [
        None if value is None else _band_of(_safe_float(value), bands)
        for value in delta_values
    ]

    assigned: List[Optional[str]] = []
    from_declared = 0
    n_disagrees = 0
    declared_counts: Dict[str, int] = {}
    for band_name, label in zip(measured, declared_values):
        if label is None or (isinstance(label, float) and math.isnan(label)):
            label = None
        else:
            label = str(label)
            if label not in known:
                label = None
        if label is not None:
            declared_counts[label] = declared_counts.get(label, 0) + 1
        if band_name is None and label is not None:
            assigned.append(label)
            from_declared += 1
            continue
        if band_name is not None and label is not None and band_name != label:
            n_disagrees += 1
        assigned.append(band_name)

    from_delta = int(sum(1 for value in measured if value is not None))
    unbanded = int(sum(1 for value in assigned if value is None))
    assigned = pd.Series(assigned, index=matched.index, dtype="object")
    return assigned, {
        "from_delta_od": from_delta,
        "from_declared": from_declared,
        "unbanded": unbanded,
        "n_disagrees": int(n_disagrees),
        "declared_counts": declared_counts,
    }


def per_scenario_table(matched: pd.DataFrame) -> pd.DataFrame:
    """n / accuracy / FP / FN per scenario."""
    if "scenario" not in matched.columns:
        return pd.DataFrame(columns=["scenario", "n", "accuracy",
                                     "false_positives", "false_negatives"])
    rows = []
    for scenario, sub in matched.groupby("scenario", dropna=False):
        gt = sub["gt_is_good"].to_numpy(dtype=bool)
        pred = sub["pred_is_good"].to_numpy(dtype=bool)
        cm = confusion(gt, pred)
        expected = sorted(set(sub["expected_class"].astype(str)))
        rows.append({
            "scenario": str(scenario),
            "n": int(len(sub)),
            "expected_class": "/".join(expected),
            "accuracy": _ratio(cm["true_positives"] + cm["true_negatives"], len(sub)),
            "true_positives": cm["true_positives"],
            "true_negatives": cm["true_negatives"],
            "false_positives": cm["false_positives"],
            "false_negatives": cm["false_negatives"],
        })
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def _scoring_mode(matched: pd.DataFrame) -> pd.Series:
    if "scoring_mode" in matched.columns:
        return matched["scoring_mode"].fillna("standard").astype(str)
    # Ground truth v1 knows only one policy.
    return pd.Series(["standard"] * len(matched), index=matched.index)


def classification_metrics(matched: pd.DataFrame,
                           lab_prior: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The full ``classification`` block.

    ``scoring_mode`` decides which curves are in the headline.  Curves marked
    ``a_not_identifiable`` are scored by the either/or rule in
    :func:`truncation_metrics` and excluded here; ``preprocessing_blind``
    curves are reported in their own block because ``write_legacy_csvs``
    (``tecan_growth/preprocess.py:81-93``) hands the pipeline the replicate
    mean only, so they are unpassable with the current gate set and a headline
    drop caused by them would be uninterpretable.
    """
    modes = _scoring_mode(matched)
    headline_mask = ~modes.isin(["a_not_identifiable", "preprocessing_blind"])
    headline = matched[headline_mask.to_numpy()]

    gt = headline["gt_is_good"].to_numpy(dtype=bool)
    pred = headline["pred_is_good"].to_numpy(dtype=bool)
    block = core_classification(gt, pred)

    scen = per_scenario_table(headline)
    block["macro_accuracy"] = float(scen["accuracy"].mean()) if len(scen) else 0.0
    block["per_scenario"] = {
        row["scenario"]: {
            "n": int(row["n"]),
            "expected_class": row["expected_class"],
            "accuracy": float(row["accuracy"]),
            "false_positives": int(row["false_positives"]),
            "false_negatives": int(row["false_negatives"]),
        }
        for _, row in scen.iterrows()
    }

    prior = lab_prior or load_lab_prior()
    bands = prior["bands"]
    per_band: Dict[str, Any] = {}
    weighted_num = 0.0
    weight_covered = 0.0
    band_names, assignment = _assign_bands(headline, bands)
    for band in bands:
        sel = (band_names == band["name"]).to_numpy()
        n = int(sel.sum())
        entry: Dict[str, Any] = {
            "n": n,
            "lab_fraction": band["fraction"],
            "min_delta_od": None if math.isinf(band["min"]) else band["min"],
            "max_delta_od": None if math.isinf(band["max"]) else band["max"],
            "n_declared_lab_band": assignment["declared_counts"].get(band["name"], 0),
        }
        if n:
            entry.update(core_classification(gt[sel], pred[sel]))
            weighted_num += band["fraction"] * entry["accuracy"]
            weight_covered += band["fraction"]
        per_band[band["name"]] = entry
    block["per_delta_od_band"] = per_band
    block["band_assignment"] = {
        "statistic": "ground-truth delta_od (falling back to the ground "
                     "truth's declared lab_band when delta_od is missing)",
        "outer_edges": "opened to -/+inf so the bands partition the line; "
                       "the derived file declares the lowest edge at its "
                       "measured minimum",
        "bands_source": prior.get("source"),
        "from_delta_od": assignment["from_delta_od"],
        "from_declared_lab_band": assignment["from_declared"],
        "unbanded": assignment["unbanded"],
        "n_declared_disagrees": assignment["n_disagrees"],
    }
    block["lab_prior_source"] = prior.get("source")
    block["lab_weighted_accuracy"] = (
        weighted_num / weight_covered if weight_covered > 0 else None)
    block["lab_weight_covered"] = weight_covered

    core_mask = headline["scenario"].isin(CORE32_SCENARIOS).to_numpy() \
        if "scenario" in headline.columns else np.ones(len(headline), dtype=bool)
    core_block = core_classification(gt[core_mask], pred[core_mask])
    core_block["scenarios"] = sorted(
        set(headline.loc[core_mask, "scenario"].astype(str))
        if "scenario" in headline.columns else [])
    block["core32"] = core_block

    excluded = matched[~headline_mask.to_numpy()]
    block["excluded_from_headline"] = {
        "n": int(len(excluded)),
        "by_scoring_mode": modes[~headline_mask].value_counts().to_dict(),
    }
    block["scoring_mode_counts"] = modes.value_counts().to_dict()
    return block


# =============================================================================
# Parameter recovery (M12, M13)
# =============================================================================

def _finite_pairs(sub: pd.DataFrame, gt_col: str, pred_col: str) -> pd.DataFrame:
    if gt_col not in sub.columns or pred_col not in sub.columns:
        return pd.DataFrame(columns=[gt_col, pred_col])
    valid = sub[[gt_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    return valid[np.isfinite(valid[gt_col]) & np.isfinite(valid[pred_col])]


def _param_stats(sub: pd.DataFrame, name: str, gt_col: str, pred_col: str,
                 tolerance: float) -> Dict[str, Any]:
    valid = _finite_pairs(sub, gt_col, pred_col)
    n = int(len(valid))
    stats: Dict[str, Any] = {
        "n": n, "n_samples": n, "parameter": name,
        "ground_truth_column": gt_col, "fitted_column": pred_col,
        "tolerance": tolerance,
        "tolerance_kind": "absolute_hours" if name == "lambda" else "relative",
    }
    if n == 0:
        stats.update({
            "mean_error": None, "bias": None, "rmse": None,
            "median_abs_error": None, "median_rel_error": None,
            "r_squared_across_scenarios": None,
            "tolerance_pass_rate": None, "recovery_status": "no_data",
        })
        return stats

    truth = valid[gt_col].to_numpy(dtype=float)
    fitted = valid[pred_col].to_numpy(dtype=float)
    errors = fitted - truth
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(errors) / np.where(truth != 0, np.abs(truth), np.nan)

    if name == "lambda":
        passed = np.abs(errors) <= tolerance
        median_error_for_status = float(np.nanmedian(np.abs(errors)))
    else:
        passed = rel <= tolerance
        median_error_for_status = float(np.nanmedian(rel))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    # Never publish an R2 for a handful of curves; and the pooled quantity is
    # dominated by the spread *between* scenarios, hence the name.
    r2 = (1 - ss_res / ss_tot) if (n >= 10 and ss_tot > 0) else None

    pass_rate = float(np.nanmean(passed.astype(float)))
    status = "ok"
    if pass_rate == 0.0 or median_error_for_status > 3 * tolerance:
        status = "degenerate"
    elif pass_rate < 0.5:
        status = "poor"

    stats.update({
        "mean_error": float(np.mean(errors)),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_error": float(np.median(errors)),
        "median_abs_error": float(np.median(np.abs(errors))),
        "median_rel_error": (None if np.all(np.isnan(rel))
                             else float(np.nanmedian(rel))),
        "r_squared_across_scenarios": r2,
        "tolerance_pass_rate": pass_rate,
        "recovery_status": status,
    })
    return stats


def _degenerate_gt_mask(matched: pd.DataFrame) -> np.ndarray:
    """Rows whose ground truth has no growth parameters to recover."""
    mask = np.zeros(len(matched), dtype=bool)
    for column in ("true_A", "true_lambda"):
        if column in matched.columns:
            values = pd.to_numeric(matched[column], errors="coerce").to_numpy(dtype=float)
            mask |= ~(values > 0)
    return mask


def _strata(matched: pd.DataFrame) -> Dict[str, np.ndarray]:
    gt = matched["gt_is_good"].to_numpy(dtype=bool)
    pred = matched["pred_is_good"].to_numpy(dtype=bool)
    if "gompertz_a" in matched.columns:
        fitted = pd.to_numeric(matched["gompertz_a"], errors="coerce").notna().to_numpy()
    else:
        fitted = np.zeros(len(matched), dtype=bool)
    return {
        # The historical definition, kept under its own name so a reader cannot
        # mistake it for unconditional recovery.
        "accepted_good": gt & pred,
        "all_fitted_good": gt & fitted,
        "rejected_good": gt & ~pred & fitted,
        "accepted_bad": ~gt & pred,
    }


def parameter_recovery(matched: pd.DataFrame,
                       tolerances: Optional[Dict[str, float]] = None,
                       ) -> Dict[str, Any]:
    """Parameter recovery in four strata, each with its own n (M13).

    ``accepted_good`` is the pipeline's own true positives -- the set the old
    report scored and the only one it scored.  ``all_fitted_good`` adds the
    fitted-but-rejected curves, and the difference between the two is the
    selection bias: on the committed run A-RMSE is 0.1418 over 319 accepted and
    0.3674 over all 333 fitted expected-GOOD curves, so tightening any gate
    mechanically "improves" the first number.
    """
    tolerances = dict(DEFAULT_TOLERANCES, **(tolerances or {}))
    degenerate = _degenerate_gt_mask(matched)
    scored = matched[~degenerate]

    has_requested = all(c in matched.columns for c in REQUESTED_COLUMNS.values())

    out: Dict[str, Any] = {}
    for stratum, mask in _strata(scored).items():
        sub = scored[mask]
        entry: Dict[str, Any] = {"n": int(len(sub))}
        for name, gt_col, pred_col, _se in PARAM_SPECS:
            entry[name] = _param_stats(sub, name, gt_col, pred_col, tolerances[name])
        entry["all_parameters_within_tolerance"] = _joint_pass_rate(sub, tolerances)
        out[stratum] = entry

    ag = out["accepted_good"]["A"]
    af = out["all_fitted_good"]["A"]
    out["selection_bias_delta"] = {
        "parameter": "A",
        "accepted_good_rmse": ag["rmse"],
        "all_fitted_good_rmse": af["rmse"],
        "ratio": (af["rmse"] / ag["rmse"]) if (ag["rmse"] and af["rmse"]) else None,
        "note": ("Recovery conditioned on the pipeline's own verdict. The gap is "
                 "selection, not accuracy: the rejected curves are the ones fitted "
                 "worst, so any tightening of a gate raises accepted_good_rmse's "
                 "flattery without changing the estimator."),
    }
    out["n_excluded_degenerate_gt"] = int(degenerate.sum())
    out["degenerate_gt_definition"] = "true_A <= 0 or true_lambda <= 0"
    out["tolerances"] = tolerances
    out["true_source"] = "measured_true" if "mu_realisation_ratio" in matched.columns \
        else "nominal_true(v1 ground truth)"

    if has_requested:
        legacy: Dict[str, Any] = {}
        for stratum, mask in _strata(scored).items():
            sub = scored[mask]
            legacy[stratum] = {"n": int(len(sub))}
            for name, _gt_col, pred_col, _se in PARAM_SPECS:
                legacy[stratum][name] = _param_stats(
                    sub, name, REQUESTED_COLUMNS[name], pred_col, tolerances[name])
        out["legacy_nominal"] = legacy
    else:
        out["legacy_nominal"] = {
            "available": False,
            "reason": ("ground truth v1: requested_* columns absent, so true_* IS "
                       "the nominal request and the two scorings coincide."),
        }
    return out


def _joint_pass_rate(sub: pd.DataFrame, tolerances: Dict[str, float]) -> Dict[str, Any]:
    """Fraction of curves where A, mu and lambda are all inside tolerance."""
    needed = [c for _n, c, p, _e in PARAM_SPECS for c in (c, p)]
    if any(c not in sub.columns for c in needed) or len(sub) == 0:
        return {"n": 0, "pass_rate": None}
    frame = sub[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) == 0:
        return {"n": 0, "pass_rate": None}
    ok = np.ones(len(frame), dtype=bool)
    for name, gt_col, pred_col, _se in PARAM_SPECS:
        truth = frame[gt_col].to_numpy(dtype=float)
        fitted = frame[pred_col].to_numpy(dtype=float)
        if name == "lambda":
            ok &= np.abs(fitted - truth) <= tolerances[name]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ok &= (np.abs(fitted - truth) / np.where(truth != 0, np.abs(truth), np.nan)
                       ) <= tolerances[name]
    return {"n": int(len(frame)), "pass_rate": float(np.mean(ok))}


def per_scenario_recovery(matched: pd.DataFrame,
                          tolerances: Optional[Dict[str, float]] = None,
                          ) -> pd.DataFrame:
    """Tolerance pass rates per scenario x model_type x parameter.

    This is the table that makes a broken generator visible: on the committed
    run ``baranyi_generated`` and ``logistic_generated`` both score 0.000, while
    the pooled number over all fitted expected-GOOD curves is 0.868.
    """
    tolerances = dict(DEFAULT_TOLERANCES, **(tolerances or {}))
    degenerate = _degenerate_gt_mask(matched)
    scored = matched[~degenerate]
    strata = _strata(scored)

    group_cols = [c for c in ("scenario", "model_type") if c in scored.columns]
    rows: List[Dict[str, Any]] = []
    for stratum in ("accepted_good", "all_fitted_good"):
        sub_all = scored[strata[stratum]]
        if not group_cols:
            groups: Iterable[Tuple[Any, pd.DataFrame]] = [(("all",), sub_all)]
        else:
            groups = sub_all.groupby(group_cols, dropna=False, observed=True)
        for key, sub in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_tuple = tuple(key_tuple)[:len(group_cols or ["scenario"])]
            for name, gt_col, pred_col, _se in PARAM_SPECS:
                stats = _param_stats(sub, name, gt_col, pred_col, tolerances[name])
                row = {"stratum": stratum}
                for col, value in zip(group_cols or ["scenario"], key_tuple):
                    row[col] = str(value)
                row.update({
                    "parameter": name,
                    "n": stats["n"],
                    "tolerance": stats["tolerance"],
                    "tolerance_kind": stats["tolerance_kind"],
                    "pass_rate": stats["tolerance_pass_rate"],
                    "median_rel_error": stats["median_rel_error"],
                    "median_abs_error": stats["median_abs_error"],
                    "rmse": stats["rmse"],
                    "bias": stats["bias"],
                    "r_squared_across_scenarios": stats["r_squared_across_scenarios"],
                    "recovery_status": stats["recovery_status"],
                })
                rows.append(row)
            # The joint "all three inside tolerance" rate is the number that
            # makes a broken converter unmissable (baranyi_generated and
            # logistic_generated are both 0.000 on the committed run).
            joint = _joint_pass_rate(sub, tolerances)
            row = {"stratum": stratum}
            for col, value in zip(group_cols or ["scenario"], key_tuple):
                row[col] = str(value)
            row.update({
                "parameter": "all", "n": joint["n"], "tolerance": None,
                "tolerance_kind": "A/mu relative, lambda absolute hours",
                "pass_rate": joint["pass_rate"],
                "median_rel_error": None, "median_abs_error": None,
                "rmse": None, "bias": None, "r_squared_across_scenarios": None,
                "recovery_status": (
                    "degenerate" if joint["pass_rate"] == 0.0 else "ok"),
            })
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        sort_cols = ["stratum"] + (group_cols or ["scenario"]) + ["parameter"]
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame


# =============================================================================
# Uncertainty calibration (M18)
# =============================================================================

def uncertainty_calibration(matched: pd.DataFrame,
                            stratum: str = "accepted_good") -> Dict[str, Any]:
    """Do the pipeline's own error bars cover the truth?

    Measured on the committed run over the 319 accepted curves: the nominal
    95 % interval covers A 46.7 % of the time, lambda 43.0 % and mu 75.9 %.
    Removing each parameter's median signed error raises A to 68.7 % and lambda
    to 63.0 %, so most of the A/lambda miss is a window-vs-whole-curve estimand
    mismatch (the fit describes the truncated growth window, ground truth
    describes the whole curve) and only roughly a factor of two is genuinely
    understated spread.  Both numbers are reported; quoting 46.7 % alone and
    blaming the covariance would be dishonest in the other direction.
    """
    degenerate = _degenerate_gt_mask(matched)
    scored = matched[~degenerate]
    mask = _strata(scored)[stratum]
    sub = scored[mask]

    out: Dict[str, Any] = {
        "stratum": stratum,
        "n": int(len(sub)),
        "nominal_coverage_68": 0.68,
        "nominal_coverage_95": 0.95,
        "note": uncertainty_calibration.__doc__.strip().splitlines()[0],
        "estimand_note": (
            "coverage_95 is the raw nominal-interval coverage; "
            "centered_coverage_95 removes each parameter's median signed error "
            "and therefore isolates the spread from the estimand mismatch."),
        "parameters": {},
        "per_scenario": {},
        "median_abs_z_by_autocorr_tercile": {},
    }

    autocorr = (pd.to_numeric(sub["residual_autocorr"], errors="coerce")
                if "residual_autocorr" in sub.columns else None)

    for name, gt_col, pred_col, se_col in PARAM_SPECS:
        out["parameters"][name] = _coverage_stats(sub, gt_col, pred_col, se_col)
        if autocorr is not None and len(sub) >= 3:
            out["median_abs_z_by_autocorr_tercile"][name] = _z_by_tercile(
                sub, gt_col, pred_col, se_col, autocorr)

    if "scenario" in sub.columns:
        for scenario, group in sub.groupby("scenario", dropna=False):
            out["per_scenario"][str(scenario)] = {
                name: _coverage_stats(group, gt_col, pred_col, se_col)
                for name, gt_col, pred_col, se_col in PARAM_SPECS
            }
    return out


def _coverage_stats(sub: pd.DataFrame, gt_col: str, pred_col: str,
                    se_col: str) -> Dict[str, Any]:
    n = int(len(sub))
    empty = {
        "n": n, "n_finite_se": 0, "n_non_finite_se": n,
        "coverage_68": None, "coverage_95": None, "median_abs_z": None,
        "median_signed_error": None, "centered_coverage_95": None,
        "se_ratio": None,
    }
    if any(c not in sub.columns for c in (gt_col, pred_col, se_col)) or n == 0:
        return empty

    truth = pd.to_numeric(sub[gt_col], errors="coerce").to_numpy(dtype=float)
    fitted = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(sub[se_col], errors="coerce").to_numpy(dtype=float)
    error = fitted - truth
    usable = np.isfinite(error) & np.isfinite(se) & (se > 0)
    n_finite = int(usable.sum())
    if n_finite == 0:
        return empty

    err = error[usable]
    sd = se[usable]
    z = np.abs(err) / sd
    median_signed = float(np.median(err))
    zc = np.abs(err - median_signed) / sd
    return {
        "n": n,
        "n_finite_se": n_finite,
        "n_non_finite_se": n - n_finite,
        "coverage_68": float(np.mean(z <= 1.0)),
        "coverage_95": float(np.mean(z <= 1.96)),
        "median_abs_z": float(np.median(z)),
        "median_signed_error": median_signed,
        "centered_coverage_95": float(np.mean(zc <= 1.96)),
        "se_ratio": float(np.median(np.abs(err)) / np.median(sd)) if np.median(sd) else None,
    }


def _z_by_tercile(sub: pd.DataFrame, gt_col: str, pred_col: str, se_col: str,
                  autocorr: pd.Series) -> Dict[str, Any]:
    values = autocorr.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return {}
    edges = np.nanquantile(values[finite], [1 / 3, 2 / 3])
    labels = np.full(len(values), "unknown", dtype=object)
    labels[finite & (values <= edges[0])] = "low"
    labels[finite & (values > edges[0]) & (values <= edges[1])] = "mid"
    labels[finite & (values > edges[1])] = "high"
    out: Dict[str, Any] = {"tercile_edges": [float(edges[0]), float(edges[1])]}
    for label in ("low", "mid", "high"):
        sel = labels == label
        stats = _coverage_stats(sub[sel], gt_col, pred_col, se_col)
        out[label] = {"n": stats["n"], "median_abs_z": stats["median_abs_z"],
                      "coverage_95": stats["coverage_95"]}
    return out


# =============================================================================
# Truncation metrics (M07, M08, M23)
# =============================================================================

def truncation_metrics(matched: pd.DataFrame) -> Dict[str, Any]:
    """Where the cut landed relative to the curve's own true growth window.

    Its own report block, never folded into accuracy: a truncation that lands
    30 h past the end of growth costs nothing in a verdict and everything in a
    parameter.  Requires the v2 ground-truth window columns
    (``t_primary_95``, ``t_second_start``, ``true_window_end_h``); on v1 ground
    truth the block reports itself unavailable rather than inventing a band.
    """
    have = [c for c in ("t_primary_95", "t_second_start", "true_window_end_h")
            if c in matched.columns]
    out: Dict[str, Any] = {
        "available": bool("t_primary_95" in matched.columns),
        "window_columns_present": have,
        "n_with_band": 0,
        "per_scenario": {},
    }
    if not out["available"]:
        out["reason"] = ("ground truth v1 carries no growth-window columns "
                         "(t_primary_95 / t_second_start); regenerate with "
                         "truth.py v2 to populate this block.")
        out.update(_scoring_mode_blocks(matched))
        return out

    if "truncation_time" not in matched.columns:
        out["available"] = False
        out["reason"] = "pipeline results carry no truncation_time column"
        out.update(_scoring_mode_blocks(matched))
        return out

    t_cut = pd.to_numeric(matched["truncation_time"], errors="coerce")
    t95 = pd.to_numeric(matched["t_primary_95"], errors="coerce")
    t2 = pd.to_numeric(matched.get("t_second_start"), errors="coerce") \
        if "t_second_start" in matched.columns else pd.Series(np.nan, index=matched.index)
    upper = t2.copy()
    if "true_window_end_h" in matched.columns:
        end = pd.to_numeric(matched["true_window_end_h"], errors="coerce")
        upper = upper.fillna(end)
    upper = upper.fillna(np.inf)

    band_known = t95.notna() & t_cut.notna()
    out["n_with_band"] = int(band_known.sum())
    inside = band_known & (t_cut >= t95) & (t_cut <= upper)
    distance = pd.Series(0.0, index=matched.index)
    distance[t_cut < t95] = (t_cut - t95)[t_cut < t95]
    over = t_cut > upper
    distance[over] = (t_cut - upper)[over]

    def _summary(sel: pd.Series) -> Dict[str, Any]:
        sel = sel & band_known
        n = int(sel.sum())
        if n == 0:
            return {"n": 0, "frac_inside_window": None,
                    "median_signed_distance_outside_h": None,
                    "p90_signed_distance_outside_h": None,
                    "median_abs_A_window_error": None}
        d = distance[sel].to_numpy(dtype=float)
        outside = d[d != 0.0]
        entry = {
            "n": n,
            "frac_inside_window": float(inside[sel].mean()),
            "median_signed_distance_outside_h": (
                float(np.median(outside)) if len(outside) else 0.0),
            "p90_signed_distance_outside_h": (
                float(np.percentile(np.abs(outside), 90)) if len(outside) else 0.0),
            "median_abs_A_window_error": None,
        }
        if "true_A_window" in matched.columns and "gompertz_a" in matched.columns:
            pair = _finite_pairs(matched[sel], "true_A_window", "gompertz_a")
            if len(pair):
                entry["median_abs_A_window_error"] = float(np.median(np.abs(
                    pair["gompertz_a"].to_numpy(float)
                    - pair["true_A_window"].to_numpy(float))))
        elif "true_A" in matched.columns and "gompertz_a" in matched.columns:
            pair = _finite_pairs(matched[sel], "true_A", "gompertz_a")
            if len(pair):
                entry["median_abs_A_window_error"] = float(np.median(np.abs(
                    pair["gompertz_a"].to_numpy(float)
                    - pair["true_A"].to_numpy(float))))
            entry["A_window_reference"] = "true_A (true_A_window absent)"
        return entry

    out["overall"] = _summary(pd.Series(True, index=matched.index))
    if "scenario" in matched.columns:
        for scenario in sorted(set(matched["scenario"].astype(str))):
            sel = matched["scenario"].astype(str) == scenario
            out["per_scenario"][scenario] = _summary(sel)
    out.update(_scoring_mode_blocks(matched))
    return out


def _observed_final_od(matched: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    for column, label in (("final_od", "results.final_od"),
                          ("max_od_pred", "results.max_od"),
                          ("max_od_gt", "ground_truth.max_od")):
        if column in matched.columns:
            return pd.to_numeric(matched[column], errors="coerce"), label
    if "max_od" in matched.columns:
        return pd.to_numeric(matched["max_od"], errors="coerce"), "max_od"
    return None, "unavailable"


def _scoring_mode_blocks(matched: pd.DataFrame) -> Dict[str, Any]:
    """The two non-standard scoring policies, each scored on its own terms."""
    modes = _scoring_mode(matched)
    out: Dict[str, Any] = {}

    sel = (modes == "a_not_identifiable").to_numpy()
    sub = matched[sel]
    final_od, source = _observed_final_od(sub)
    entry: Dict[str, Any] = {
        "n": int(len(sub)),
        "rule": ("correct if the pipeline rejected the curve, OR "
                 "|A_fit - observed final OD| / observed final OD <= 0.15 and "
                 "A_fit <= 1.20 x the last observed point"),
        "observed_final_od_source": source,
        "excluded_from_headline": True,
    }
    if len(sub) and final_od is not None and "gompertz_a" in sub.columns:
        a_fit = pd.to_numeric(sub["gompertz_a"], errors="coerce")
        pred_good = sub["pred_is_good"].to_numpy(dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (a_fit - final_od).abs() / final_od.abs()
        ok = (~pred_good) | ((rel <= 0.15) & (a_fit <= 1.20 * final_od)).fillna(False).to_numpy()
        entry["n_correct"] = int(np.sum(ok))
        entry["pass_rate"] = float(np.mean(ok))
    out["a_not_identifiable"] = entry

    sel = (modes == "preprocessing_blind").to_numpy()
    sub = matched[sel]
    block: Dict[str, Any] = {
        "n": int(len(sub)),
        "excluded_from_headline": True,
        "reason": ("write_legacy_csvs (tecan_growth/preprocess.py:81-93) emits the "
                   "replicate mean only, so these curves are unpassable with the "
                   "current gate set; a headline drop caused by them would be "
                   "uninterpretable."),
    }
    if len(sub):
        block.update(core_classification(
            sub["gt_is_good"].to_numpy(dtype=bool),
            sub["pred_is_good"].to_numpy(dtype=bool)))
    out["preprocessing_blind"] = block
    return out


# =============================================================================
# Run-length robustness
# =============================================================================

def run_length_robustness(matched: pd.DataFrame, data_dir=None,
                          cutoff_h: float = 24.0, min_duration_h: float = 48.0,
                          truncation_mode: str = "first_peak",
                          limit: Optional[int] = None) -> Dict[str, Any]:
    """Refit every long expected-GOOD curve at ``cutoff_h`` and compare mu.

    Reported as median AND p90 of ``|mu_24h / mu_full - 1|`` plus the worst five
    by name: on the real plates the median is 0.006 while the tail reaches 0.45
    (LB_PERM7 mu 0.546 -> 0.300), so a median-only statistic would hide exactly
    what this block exists to catch.

    Needs the raw curves, so it is skipped (and says so) when ``data_dir`` is
    not supplied.
    """
    out: Dict[str, Any] = {
        "available": False,
        "cutoff_h": cutoff_h,
        "min_duration_h": min_duration_h,
        "truncation_mode": truncation_mode,
    }
    if data_dir is None:
        out["reason"] = "data_dir not supplied; pass --data-dir to compute this block"
        return out

    data_dir = Path(data_dir)
    if not data_dir.exists():
        out["reason"] = f"data dir not found: {data_dir}"
        return out

    try:
        curves = _load_synthetic_curves(data_dir)
    except Exception as exc:  # pragma: no cover - defensive
        out["reason"] = f"could not read curves from {data_dir}: {exc}"
        return out

    try:
        analyze_curve, settings = _pipeline_analyzer(truncation_mode)
    except Exception as exc:  # pragma: no cover - defensive
        out["reason"] = f"pipeline analyzer unavailable: {exc}"
        return out

    duration = (pd.to_numeric(matched["duration_hours"], errors="coerce")
                if "duration_hours" in matched.columns else None)
    ratios: List[Tuple[str, float, float, float]] = []
    n_considered = 0
    for _, row in matched.iterrows():
        if not bool(row.get("gt_is_good", False)):
            continue
        if duration is not None:
            d = _safe_float(row.get("duration_hours"))
            if not (d >= min_duration_h):
                continue
        curve_id = str(row.get("curve_id"))
        record = curves.get(curve_id)
        if record is None:
            continue
        t, od = record
        if t.max() < min_duration_h or t.max() <= cutoff_h:
            continue
        n_considered += 1
        if limit is not None and n_considered > limit:
            n_considered -= 1
            break
        mu_full = _fit_mu(analyze_curve, settings, t, od, curve_id)
        keep = t <= cutoff_h
        mu_cut = _fit_mu(analyze_curve, settings, t[keep], od[keep], curve_id)
        if not (np.isfinite(mu_full) and np.isfinite(mu_cut)) or mu_full == 0:
            continue
        ratios.append((curve_id, mu_full, mu_cut, abs(mu_cut / mu_full - 1.0)))

    out["n_considered"] = n_considered
    out["n_scored"] = len(ratios)
    if not ratios:
        out["reason"] = "no curve satisfied duration >= %.1f h with a usable pair of fits" % min_duration_h
        return out

    values = np.array([r[3] for r in ratios], dtype=float)
    worst = sorted(ratios, key=lambda r: -r[3])[:5]
    n_unchanged = int(np.sum(values == 0.0))
    out.update({
        "available": True,
        "metric": "abs(mu_at_cutoff / mu_full - 1)",
        "n_identical_fits": n_unchanged,
        "note": (
            "A curve whose truncation already lands before the cutoff is "
            "refitted on the same window and contributes an exact 0. On the "
            "synthetic set that is most of them, because no scenario ends "
            "mid-growth (M07): every curve plateaus in the first sixth of its "
            "run, so this block currently measures the absence of the failure "
            "mode rather than the pipeline's robustness to it."
            if n_unchanged > len(values) / 2 else
            "Median and p90 are both reported because the real-plate "
            "distribution has a median of 0.006 and a tail reaching 0.45."),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "worst_5": [
            {"curve_id": c, "mu_full": mu_f, "mu_cut": mu_c, "rel_change": r}
            for c, mu_f, mu_c, r in worst
        ],
    })
    return out


def _load_synthetic_curves(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read the synthetic plate CSVs into ``{CURVE id: (time, od)}``."""
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(Path(data_dir).glob("*_DATA.csv")):
        frame = pd.read_csv(path)
        time_col = frame.columns[0]
        t = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
        for column in frame.columns[1:]:
            cid = extract_curve_id(column)
            od = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t) & np.isfinite(od)
            if ok.sum() > 5:
                curves[cid] = (t[ok], od[ok])
    return curves


def _pipeline_analyzer(truncation_mode: str):
    """Import the pipeline's own analyzer so the refit uses the real gates."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tecan_growth.analyze import analyze_curve  # type: ignore
    from tecan_growth.settings import AnalysisSettings  # type: ignore

    # The repo's own thresholds, so the refit is judged by the same gates; only
    # the truncation mode is overridden, and the ML models are left out because
    # they were trained on this very benchmark.
    settings = AnalysisSettings.from_config(
        use_adaptive=(truncation_mode == "adaptive"),
        use_first_peak=(truncation_mode != "adaptive"),
        use_ml=False,
    )
    return analyze_curve, settings


def _fit_mu(analyze_curve, settings, t: np.ndarray, od: np.ndarray,
            curve_id: str) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_curve(t, od, curve_id=curve_id, settings=settings)
    except Exception:
        return math.nan
    fit = getattr(result, "fit", None)
    if fit is None or not getattr(fit, "success", False):
        return math.nan
    return _safe_float(getattr(fit, "mu_opt", math.nan))


# =============================================================================
# Validator
# =============================================================================

TRUNCATION_MODES = ("first_peak", "adaptive")


class PipelineValidator:
    """Validate synthetic data against the analysis pipeline."""

    def __init__(
        self,
        analysis_script_path: str,
        synthetic_data_dir: str,
        ground_truth_csv: str,
        timeout: Optional[float] = None,
        strict: bool = True,
    ):
        """
        Args:
            analysis_script_path: Path to 01_growth_curve_analysis.py
            synthetic_data_dir: Directory with synthetic *_DATA.csv files
            ground_truth_csv: CSV with expected classifications and parameters
            timeout: Subprocess timeout in seconds. ``None`` (the default) means
                no timeout: the old hard-coded 1800 s was shorter than the
                adaptive run this class defaulted to, so the validator aborted
                itself and returned None (M14).
            strict: Fail on an incomplete ground-truth / results match (M19).
        """
        self.analysis_script = Path(analysis_script_path)
        self.data_dir = Path(synthetic_data_dir)
        self.ground_truth_path = Path(ground_truth_csv)
        self.timeout = timeout
        self.strict = strict

        self.ground_truth = pd.read_csv(ground_truth_csv)

        self.pipeline_results: Optional[pd.DataFrame] = None
        self.output_dir: Optional[Path] = None
        self.truncation_mode: Optional[str] = None
        self.wall_clock_s: Optional[float] = None
        self._matched: Optional[pd.DataFrame] = None

    # -- running ------------------------------------------------------------

    def run_pipeline(
        self,
        output_dir: str,
        truncation_mode: str,
        verbose: bool = True,
    ) -> Optional[Path]:
        """Execute the analysis pipeline on synthetic data.

        Args:
            output_dir: Directory for pipeline output.
            truncation_mode: ``'first_peak'`` or ``'adaptive'``. Required, with
                no default: the mode changes the answer (verdicts agree on only
                7/15 sparse_data curves between modes), so a caller must say
                which benchmark it is running. The matching flag is appended in
                both directions -- passing nothing let
                ``01_growth_curve_analysis.py`` resolve ``use_adaptive: true``
                from ``scripts/config.yaml``, which made the first-peak branch
                of this API unreachable (M14).
            verbose: Print progress.

        Returns:
            Path to processing_results.csv, or None if the run produced none.
        """
        import time as _time

        if truncation_mode not in TRUNCATION_MODES:
            raise ValueError(
                f"truncation_mode must be one of {TRUNCATION_MODES}, got {truncation_mode!r}")

        self.truncation_mode = truncation_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.analysis_script),
            str(self.data_dir),
            '-o', str(self.output_dir),
        ]
        cmd.append('--adaptive' if truncation_mode == 'adaptive' else '--no-adaptive')
        cmd.append('--no-plots')  # Skip per-curve plots for faster validation

        if verbose:
            print(f"Running pipeline ({truncation_mode}): {' '.join(cmd)}")

        started = _time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )
            if verbose:
                print("Pipeline stdout:")
                print(result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"Pipeline failed with code {result.returncode}")
                print("stderr:", result.stderr)
        except subprocess.TimeoutExpired:
            self.wall_clock_s = _time.time() - started
            print(f"Pipeline timed out after {self.timeout} s!")
            return None
        self.wall_clock_s = _time.time() - started

        results_path = self.output_dir / 'processing_results.csv'
        if results_path.exists():
            self.load_results(str(results_path))
            if verbose:
                print(f"Loaded {len(self.pipeline_results)} results from pipeline")
            return results_path
        print(f"Results file not found: {results_path}")
        return None

    def load_results(self, results_csv: str):
        """Load pipeline results from CSV and invalidate the cached match."""
        results_csv = Path(results_csv)
        # --skip-run with a fresh --output-dir used to die here with
        # FileNotFoundError because only run_pipeline created the directory.
        if self.output_dir is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.pipeline_results = pd.read_csv(results_csv)
        self._matched = None

    # -- matching -----------------------------------------------------------

    @property
    def matched(self) -> pd.DataFrame:
        """The ground truth / results join, computed once per instance.

        The old code recomputed it inside every metric method, so one report run
        printed 'Matched 480 curves' seven times.
        """
        if self._matched is None:
            if self.pipeline_results is None:
                raise ValueError("No pipeline results loaded")
            self._matched = match_results(
                self.ground_truth, self.pipeline_results, strict=self.strict)
        return self._matched

    def _match_results_to_ground_truth(self) -> pd.DataFrame:
        """Backwards-compatible alias for :attr:`matched`."""
        return self.matched

    def match_summary(self) -> Dict[str, Any]:
        attrs = self.matched.attrs
        return {
            "n_ground_truth": int(attrs.get("n_ground_truth", len(self.ground_truth))),
            "n_pipeline_results": int(attrs.get(
                "n_results", 0 if self.pipeline_results is None else len(self.pipeline_results))),
            "n_matched": int(attrs.get("n_matched", 0)),
            "n_unmatched_ground_truth": int(attrs.get("n_unmatched_ground_truth", 0)),
            "n_unmatched_results": int(attrs.get("n_unmatched_results", 0)),
            "n_duplicate_ids": int(attrs.get("n_duplicate_ids", 0)),
            "n_undecided_verdicts": int(attrs.get("n_undecided", 0)),
            "coverage": float(attrs.get("coverage", 0.0)),
            "strict": bool(attrs.get("strict", self.strict)),
        }

    # -- metrics ------------------------------------------------------------

    def compute_classification_metrics(self) -> ValidationMetrics:
        """Compare predicted vs expected classifications."""
        matched = self.matched
        if len(matched) == 0:
            print("Warning: No matched curves found!")
            return ValidationMetrics(
                accuracy=0, precision=0, recall=0, f1_score=0,
                true_positives=0, true_negatives=0, false_positives=0,
                false_negatives=0, confusion_matrix=[[0, 0], [0, 0]])

        block = classification_metrics(matched)
        cm = block["confusion_matrix"]
        return ValidationMetrics(
            accuracy=block["accuracy"],
            precision=block["precision"],
            recall=block["recall"],
            f1_score=block["f1_score"],
            true_positives=cm["true_positives"],
            true_negatives=cm["true_negatives"],
            false_positives=cm["false_positives"],
            false_negatives=cm["false_negatives"],
            # standard: [[TP,FN],[FP,TN]]
            confusion_matrix=[[cm["true_positives"], cm["false_negatives"]],
                              [cm["false_positives"], cm["true_negatives"]]],
        )

    def compute_classification_block(self) -> Dict[str, Any]:
        """The full classification block (specificity, strata, bands, core32)."""
        return classification_metrics(self.matched)

    def compute_parameter_recovery(self) -> Dict[str, ParameterRecovery]:
        """Legacy view: recovery over the pipeline's own true positives.

        Kept because ``validation_report.json`` has this shape and downstream
        readers parse it. Use :meth:`compute_parameter_recovery_block` for the
        four strata; the number here is ``accepted_good`` and nothing else.
        """
        block = parameter_recovery(self.matched)["accepted_good"]
        out: Dict[str, ParameterRecovery] = {}
        for name, _gt, _pred, _se in PARAM_SPECS:
            stats = block[name]
            if not stats["n"]:
                continue
            r2 = stats["r_squared_across_scenarios"]
            out[name] = ParameterRecovery(
                parameter=name,
                mean_error=stats["mean_error"],
                rmse=stats["rmse"],
                r_squared=0.0 if r2 is None else r2,
                bias=stats["bias"],
                n_samples=stats["n"],
            )
        return out

    def compute_parameter_recovery_block(
            self, tolerances: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return parameter_recovery(self.matched, tolerances)

    def compute_uncertainty(self) -> Dict[str, Any]:
        return uncertainty_calibration(self.matched)

    def compute_truncation_metrics(self) -> Dict[str, Any]:
        return truncation_metrics(self.matched)

    def compute_gate_attribution(self, strict: bool = True) -> pd.DataFrame:
        return gate_attribution(self.matched, strict=strict)

    def identify_failure_cases(self) -> pd.DataFrame:
        """Misclassified curves, with an ``error_type`` column."""
        matched = self.matched
        failures = matched[matched["gt_is_good"] != matched["pred_is_good"]].copy()
        failures["error_type"] = np.where(
            failures["gt_is_good"], "False Negative", "False Positive")
        return failures

    # -- reporting ----------------------------------------------------------

    def generate_validation_report(self, output_path: str) -> Dict[str, Any]:
        """Write the legacy ``validation_report.json``.

        The schema is deliberately unchanged (step 9 and
        ``validation_summary.py`` read it); the completeness counters are added
        under ``summary`` so a broken run is visible in an old report file and
        not only as a runtime exception. The richer report is ``report.json``,
        written by ``benchmark.py``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class_metrics = self.compute_classification_metrics()
        param_recovery = self.compute_parameter_recovery()
        failures = self.identify_failure_cases()
        attribution = gate_attribution(self.matched, strict=False)
        failures_block = failure_analysis(self.matched, attribution)

        summary = self.match_summary()
        summary["n_failures"] = int(len(failures))

        report = {
            'summary': summary,
            'classification': {
                'accuracy': class_metrics.accuracy,
                'precision': class_metrics.precision,
                'recall': class_metrics.recall,
                'f1_score': class_metrics.f1_score,
                'confusion_matrix': {
                    'true_positives': class_metrics.true_positives,
                    'true_negatives': class_metrics.true_negatives,
                    'false_positives': class_metrics.false_positives,
                    'false_negatives': class_metrics.false_negatives,
                },
            },
            'parameter_recovery': {
                name: {
                    'mean_error': pr.mean_error,
                    'rmse': pr.rmse,
                    'r_squared': pr.r_squared,
                    'bias': pr.bias,
                    'n_samples': pr.n_samples,
                }
                for name, pr in param_recovery.items()
            },
            'failure_analysis': {
                'n_false_positives': class_metrics.false_positives,
                'n_false_negatives': class_metrics.false_negatives,
                'n_false_negatives_unfitted': failures_block['n_false_negatives_unfitted'],
                'n_false_negatives_gated': failures_block['n_false_negatives_gated'],
                'failure_scenarios': (
                    failures['scenario'].value_counts().to_dict()
                    if 'scenario' in failures.columns else {}),
                'by_gate': failures_block['by_gate'],
            },
        }
        if self.truncation_mode:
            report['run'] = {
                'truncation_mode': self.truncation_mode,
                'wall_clock_s': self.wall_clock_s,
            }

        with open(output_path, 'w') as f:
            json.dump(json_safe(report), f, indent=2)

        print(f"Validation report saved to: {output_path}")
        return report

    def print_summary(self):
        """Print a summary of validation results."""
        block = self.compute_classification_block()
        cm = block["confusion_matrix"]
        param_recovery = self.compute_parameter_recovery_block()
        failures = self.identify_failure_cases()
        match = self.match_summary()

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        if self.truncation_mode:
            print(f"  truncation mode: {self.truncation_mode}")
        print("=" * 60)

        print("\nMATCHING:")
        print(f"  Ground truth: {match['n_ground_truth']}  results: "
              f"{match['n_pipeline_results']}  matched: {match['n_matched']} "
              f"(coverage {match['coverage']:.3f})")
        if match['n_unmatched_ground_truth'] or match['n_unmatched_results']:
            print(f"  UNMATCHED gt={match['n_unmatched_ground_truth']} "
                  f"results={match['n_unmatched_results']}")

        print("\nCLASSIFICATION PERFORMANCE:")
        print(f"  Balanced accuracy: {block['balanced_accuracy']:.3f}")
        print(f"  Sensitivity:       {block['sensitivity']:.3f}")
        print(f"  Specificity:       {block['specificity']:.3f}")
        print(f"  Accuracy:          {block['accuracy']:.3f} "
              f"(prevalence {block['prevalence']:.3f})")
        print(f"  Precision: {block['precision']:.3f}")
        print(f"  Recall:    {block['recall']:.3f}")
        print(f"  F1 Score:  {block['f1_score']:.3f}")

        print("\n  Confusion Matrix:")
        print("                    Predicted GOOD  Predicted BAD")
        print(f"    Actual GOOD:         {cm['true_positives']:4d}           "
              f"{cm['false_negatives']:4d}")
        print(f"    Actual BAD:          {cm['false_positives']:4d}           "
              f"{cm['true_negatives']:4d}")

        print("\nPARAMETER RECOVERY (by stratum, A):")
        for stratum in ("accepted_good", "all_fitted_good", "rejected_good", "accepted_bad"):
            entry = param_recovery[stratum]
            a = entry["A"]
            rmse = "n/a" if a["rmse"] is None else f"{a['rmse']:.4f}"
            print(f"  {stratum:<16} n={entry['n']:<4} A-RMSE {rmse}")

        print("\nFAILURE ANALYSIS:")
        print(f"  Total failures: {len(failures)}")
        if 'scenario' in failures.columns:
            print("  By scenario:")
            for scenario, count in failures['scenario'].value_counts().items():
                print(f"    {scenario}: {count}")

        print("\n" + "=" * 60)


def json_safe(obj):
    """Recursively make a report JSON-clean.

    NaN and +/-inf become null rather than the ``NaN``/``Infinity`` literals
    ``json.dump`` emits by default, which are not JSON and which every strict
    reader (including ``jq``) rejects. Numpy scalars become Python scalars and
    dict keys become strings.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


# =============================================================================
# CLI Entry Point
# =============================================================================
#
# Kept for scripts/run_full_pipeline.py step 9, which invokes this file
# directly with --skip-run --results-csv. benchmark.py is the entry point for
# everything else.

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate synthetic data against analysis pipeline')
    parser.add_argument('--pipeline', required=True,
                        help='Path to 01_growth_curve_analysis.py')
    parser.add_argument('--data-dir', required=True,
                        help='Directory with synthetic *_DATA.csv files')
    parser.add_argument('--ground-truth', required=True,
                        help='Path to ground_truth.csv')
    parser.add_argument('--output-dir', default='validation_output',
                        help='Directory for validation output')
    parser.add_argument('--truncation-mode', choices=list(TRUNCATION_MODES),
                        help='Truncation mode to run the pipeline in. Required '
                             'unless --skip-run is given.')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Subprocess timeout in seconds (default: none)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running pipeline, just load existing results')
    parser.add_argument('--results-csv',
                        help='Path to existing processing_results.csv (with --skip-run)')
    parser.add_argument('--no-strict', action='store_true',
                        help='Score a deliberately partial run instead of failing '
                             'on an incomplete ground-truth match')

    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = PipelineValidator(
        analysis_script_path=args.pipeline,
        synthetic_data_dir=args.data_dir,
        ground_truth_csv=args.ground_truth,
        timeout=args.timeout,
        strict=not args.no_strict,
    )
    validator.output_dir = output_dir

    if args.skip_run:
        if not args.results_csv:
            print("Error: --results-csv required with --skip-run")
            return 1
        validator.load_results(args.results_csv)
    else:
        if not args.truncation_mode:
            parser.error("--truncation-mode is required unless --skip-run is given "
                         "(the mode changes the answer; see M14)")
        if validator.run_pipeline(str(output_dir), args.truncation_mode) is None:
            return 1

    if args.truncation_mode:
        validator.truncation_mode = args.truncation_mode

    try:
        validator.print_summary()
        validator.generate_validation_report(str(output_dir / 'validation_report.json'))
    except BenchmarkMatchError as exc:
        # Exit non-zero so step 9 fails loudly instead of publishing accuracy 0.0.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
