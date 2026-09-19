"""
The synthetic benchmark's single entry point.

One run in, one directory out::

    python synthetic_data/validation/benchmark.py \
        --results-csv  <run>/processing_results.csv \
        --ground-truth <set>/ground_truth.csv \
        --out-dir      <run>/benchmark \
        [--truncation-mode first_peak|adaptive] [--run-settings PATH]

writes ``report.json``, ``per_curve.csv``, ``per_scenario.csv``,
``gate_attribution.csv``, ``per_scenario_recovery.csv`` and the comparison
figures.  The markdown summary is rendered separately from this directory (it
is not written here) so that exactly one generated summary exists.

Why one entry point (M25)
-------------------------
There used to be two, with incompatible JSON schemas: step 9 invoked
``pipeline_validator.py`` and got ``validation_report.json``, while
``validate_pipeline.py`` -- wired to no Makefile target and defaulting to a
``../ANALYSIS_SCRIPTS/`` path that does not exist -- wrote both
``validation_metrics.json`` (spelling the confusion matrix TP/FN) and
``validation_report.json`` (spelling it true_positives/false_negatives),
forcing the summary renderer to handle both.  ``report.json`` keeps the
``true_positives``/``false_negatives`` spelling so existing readers survive.

Provenance (M14)
----------------
The committed 87.3 % headline was produced with first-peak truncation while
``scripts/config.yaml`` sets ``use_adaptive: true``, so ``make step-9`` would
silently overwrite it with adaptive results, and no artefact in the repo
recorded which mode any number came from.  This module refuses to write a
report unless the mode is known: either from WP9's ``run_settings.json``
sidecar (``provenance: 'sidecar'``) or from an explicit ``--truncation-mode``
(``provenance: 'declared'``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # package import
    from .matching import BenchmarkMatchError, match_results
    from .pipeline_validator import (
        DEFAULT_TOLERANCES, PARAM_SPECS,
        TRUNCATION_MODES, classification_metrics, failure_analysis,
        gate_attribution, json_safe, load_lab_prior, parameter_recovery,
        per_scenario_recovery, per_scenario_table, run_length_robustness,
        truncation_metrics, uncertainty_calibration,
    )
    from .comparison_report import ComparisonReport
except ImportError:  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.matching import BenchmarkMatchError, match_results  # type: ignore
    from validation.pipeline_validator import (  # type: ignore
        DEFAULT_TOLERANCES, PARAM_SPECS,
        TRUNCATION_MODES, classification_metrics, failure_analysis,
        gate_attribution, json_safe, load_lab_prior, parameter_recovery,
        per_scenario_recovery, per_scenario_table, run_length_robustness,
        truncation_metrics, uncertainty_calibration,
    )
    from validation.comparison_report import ComparisonReport  # type: ignore


__all__ = [
    "BenchmarkProvenanceError",
    "REPORT_SCHEMA_VERSION",
    "run_benchmark",
    "main",
]

REPORT_SCHEMA_VERSION = "2"

_REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkProvenanceError(RuntimeError):
    """Raised when the run's truncation mode cannot be established.

    A report without provenance is not publishable: the same 480 curves score
    differently under the two truncation modes (verdicts agree on 7 of 15
    sparse_data curves), so a headline that does not name its mode is not a
    measurement.
    """


# =============================================================================
# Provenance helpers
# =============================================================================

def sha256_file(path) -> Optional[str]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(data_dir) -> Optional[str]:
    """Digest of the plate data a run was scored on, or ``None``.

    Deliberately byte-compatible with the pipeline's own
    ``dataset_sha256`` (``scripts/01_growth_curve_analysis.py``): SHA-256 over
    the concatenated contents of the ``*_DATA.csv`` files of ``data_dir``, in
    sorted filename order.  Computing it the same way is what lets a digest
    taken here be compared with the one WP9's ``run_settings.json`` recorded
    at run time instead of merely coexisting with it.

    Returns ``None`` when ``data_dir`` is missing, unreadable or holds no
    ``*_DATA.csv``.  There is no substitute digest: hashing something else
    (the ground-truth file, say) and publishing it under the name
    ``dataset_sha256`` would make the provenance header lie, which is the one
    thing it exists to prevent.
    """
    if data_dir is None:
        return None
    try:
        files = sorted(Path(data_dir).glob("*_DATA.csv"), key=lambda q: q.name)
    except Exception:
        return None
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        try:
            digest.update(path.read_bytes())
        except Exception:
            return None
    return digest.hexdigest()


def git_sha(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = repo_root or _REPO_ROOT
    out: Dict[str, Any] = {"git_sha": None, "git_dirty": None}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                             capture_output=True, text=True, timeout=30)
        if sha.returncode == 0:
            out["git_sha"] = sha.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root),
                                capture_output=True, text=True, timeout=30)
        if status.returncode == 0:
            out["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def find_run_settings(results_csv: Path,
                      explicit: Optional[Path] = None) -> Optional[Path]:
    """WP9 writes ``run_settings.json`` beside ``processing_results.csv``."""
    if explicit is not None:
        path = Path(explicit)
        return path if path.exists() else None
    candidate = Path(results_csv).parent / "run_settings.json"
    return candidate if candidate.exists() else None


def _read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mode_from_settings(settings: Dict[str, Any]) -> Optional[str]:
    mode = settings.get("truncation_mode")
    if mode in TRUNCATION_MODES or mode == "global_max":
        return mode
    if "use_adaptive" in settings:
        if settings.get("use_adaptive"):
            return "adaptive"
        if settings.get("use_first_peak", True):
            return "first_peak"
        return "global_max"
    return None


def _find_generation_summary(ground_truth_csv: Path) -> Optional[Path]:
    here = Path(ground_truth_csv).resolve().parent
    for parent in (here, here.parent, here.parent.parent):
        candidate = parent / "generation_summary.json"
        if candidate.exists():
            return candidate
    return None


def build_run_block(
    results_csv: Path,
    ground_truth_csv: Path,
    matched: pd.DataFrame,
    truncation_mode: Optional[str],
    run_settings: Optional[Dict[str, Any]],
    run_settings_path: Optional[Path],
    wall_clock_s: float,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """The provenance header every published number is stamped with."""
    settings = run_settings or {}
    sidecar_mode = _mode_from_settings(settings) if settings else None

    if sidecar_mode and truncation_mode and sidecar_mode != truncation_mode:
        raise BenchmarkProvenanceError(
            f"--truncation-mode says {truncation_mode!r} but the run_settings.json "
            f"beside the results says {sidecar_mode!r}. Refusing to publish a "
            "report whose header would contradict the run that produced it.")

    mode = sidecar_mode or truncation_mode
    provenance = "sidecar" if sidecar_mode else ("declared" if truncation_mode else None)
    if mode is None:
        raise BenchmarkProvenanceError(
            "cannot establish the truncation mode: no run_settings.json beside "
            f"{results_csv} and no --truncation-mode given. The two modes do not "
            "produce the same verdicts, so an unlabelled report is not "
            "publishable (M14).")

    seed = settings.get("seed")
    if seed is None and "seed" in matched.columns:
        seeds = pd.unique(pd.to_numeric(matched["seed"], errors="coerce").dropna())
        if len(seeds) == 1:
            seed = int(seeds[0])
    if seed is None:
        summary = _read_json(_find_generation_summary(ground_truth_csv))
        if summary:
            seed = summary.get("seed")

    scenarios = (sorted(set(matched["scenario"].astype(str)))
                 if "scenario" in matched.columns else [])

    # The dataset digest is the hash of the plate data, or nothing at all.
    # It used to fall back to hashing ground_truth.csv, which made
    # run['dataset_sha256'] equal to run['ground_truth_sha256'] while the
    # headline still announced it as "dataset <hash>" -- the same data
    # published two different "dataset" hashes depending on whether a sidecar
    # was present.  A missing digest is now reported as missing.
    sidecar_digest = settings.get("dataset_sha256") or None
    local_digest = dataset_sha256(data_dir) if data_dir else None
    if sidecar_digest and local_digest and sidecar_digest != local_digest:
        raise BenchmarkProvenanceError(
            f"--data-dir {data_dir} hashes to {local_digest[:12]} but the "
            f"run_settings.json beside the results was produced on dataset "
            f"{str(sidecar_digest)[:12]}. Refusing to publish a report that "
            "would name the wrong dataset.")
    dataset_digest = sidecar_digest or local_digest
    dataset_source = ("sidecar" if sidecar_digest
                      else ("data_dir" if local_digest else None))

    block: Dict[str, Any] = {
        "truncation_mode": mode,
        "provenance": provenance,
        "run_settings_path": str(run_settings_path) if run_settings_path else None,
        "ml_classify": settings.get("ml_classify"),
        "use_adaptive": settings.get("use_adaptive"),
        "use_first_peak": settings.get("use_first_peak"),
        "seed": seed,
        "n_curves": int(len(matched)),
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
        "results_csv": str(results_csv),
        "results_sha256": sha256_file(results_csv),
        "ground_truth_csv": str(ground_truth_csv),
        "ground_truth_sha256": sha256_file(ground_truth_csv),
        "dataset_sha256": dataset_digest,
        "dataset_sha256_source": dataset_source,
        "data_dir": str(data_dir) if data_dir else settings.get("data_dir"),
        "config_path": settings.get("config_path"),
        "config_sha256": settings.get("config_sha256"),
        "quality_gates": settings.get("quality_gates"),
        "wall_clock_s": float(wall_clock_s),
        "pipeline_wall_clock_s": settings.get("wall_clock_s"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    block.update(git_sha())
    if settings.get("git_sha"):
        block["pipeline_git_sha"] = settings["git_sha"]
        block["pipeline_git_dirty"] = settings.get("git_dirty")
    return block


def headline(report: Dict[str, Any]) -> str:
    """One line that can never be quoted without its provenance."""
    run = report["run"]
    cls = report["classification"]
    digest = run.get("dataset_sha256")
    # No digest is stated as no digest.  Printing some other file's hash under
    # the word "dataset" is worse than admitting the dataset was not supplied.
    dataset = (f"dataset {str(digest)[:12]}" if digest
               else "dataset unknown (no run_settings.json, no --data-dir)")
    return (
        f"accuracy {cls['accuracy']:.1%} "
        f"(balanced {cls['balanced_accuracy']:.1%}, "
        f"sensitivity {cls['sensitivity']:.1%}, specificity {cls['specificity']:.1%}) "
        f"-- {run['truncation_mode']} truncation, {run['n_curves']} curves, "
        f"seed {run['seed']}, {dataset}, "
        f"git {str(run['git_sha'])[:12]}"
    )


# =============================================================================
# Per-curve table
# =============================================================================

def build_per_curve(matched: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    """One row per matched curve: truth, prediction, gates and per-parameter error."""
    columns: List[str] = ["curve_id"]
    frame = pd.DataFrame({"curve_id": matched["curve_id"].values})

    def carry(name: str, *candidates: str):
        for candidate in candidates:
            if candidate in matched.columns:
                frame[name] = matched[candidate].values
                columns.append(name)
                return

    carry("scenario", "scenario")
    carry("model_type", "model_type")
    carry("scoring_mode", "scoring_mode")
    carry("label_rule", "label_rule")
    carry("expected_class", "expected_class")
    frame["gt_is_good"] = matched["gt_is_good"].values
    frame["pred_is_good"] = matched["pred_is_good"].values
    frame["correct"] = (matched["gt_is_good"].values == matched["pred_is_good"].values)
    frame["error_type"] = np.where(
        frame["correct"], "",
        np.where(frame["gt_is_good"], "false_negative", "false_positive"))
    columns += ["gt_is_good", "pred_is_good", "correct", "error_type"]

    frame["fitted"] = attribution["fitted"].values
    frame["failed_gates"] = attribution["failed_gates"].values
    frame["gate_source"] = attribution["gate_source"].values
    frame["degenerate_covariance"] = attribution["degenerate_covariance"].values
    columns += ["fitted", "failed_gates", "gate_source", "degenerate_covariance"]

    for name, gt_col, pred_col, se_col in PARAM_SPECS:
        if gt_col in matched.columns:
            frame[f"true_{name}"] = pd.to_numeric(matched[gt_col], errors="coerce").values
            columns.append(f"true_{name}")
        if pred_col in matched.columns:
            frame[f"fit_{name}"] = pd.to_numeric(matched[pred_col], errors="coerce").values
            columns.append(f"fit_{name}")
        if se_col in matched.columns:
            frame[f"se_{name}"] = pd.to_numeric(matched[se_col], errors="coerce").values
            columns.append(f"se_{name}")
        if gt_col in matched.columns and pred_col in matched.columns:
            error = frame[f"fit_{name}"] - frame[f"true_{name}"]
            frame[f"err_{name}"] = error
            columns.append(f"err_{name}")
            with np.errstate(divide="ignore", invalid="ignore"):
                frame[f"relerr_{name}"] = np.abs(error) / frame[f"true_{name}"].abs()
            columns.append(f"relerr_{name}")
            if se_col in matched.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    frame[f"z_{name}"] = np.abs(error) / frame[f"se_{name}"]
                columns.append(f"z_{name}")

    for extra in ("r_squared", "truncation_time", "points_used", "snr",
                  "residual_autocorr", "monotone_fraction", "p_good",
                  "classification_reason"):
        carry(extra, extra, f"{extra}_pred")
    carry("true_delta_od", "delta_od_gt", "delta_od")
    carry("observed_delta_od", "delta_od_pred")
    carry("t_primary_95", "t_primary_95")
    carry("t_second_start", "t_second_start")

    return frame[[c for c in columns if c in frame.columns]]


# =============================================================================
# The benchmark
# =============================================================================

def run_benchmark(
    results_csv,
    ground_truth_csv,
    out_dir,
    truncation_mode: Optional[str] = None,
    run_settings_path=None,
    tolerances: Optional[Dict[str, float]] = None,
    data_dir=None,
    strict: bool = True,
    figures: bool = True,
    run_length: bool = False,
    run_length_limit: Optional[int] = None,
    strict_gates: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Score one pipeline run against ground truth and write the run directory.

    Parameters
    ----------
    results_csv, ground_truth_csv
        The pipeline's ``processing_results.csv`` and the generator's
        ``ground_truth.csv`` (v1 or v2).
    out_dir
        Directory for ``report.json`` and friends; created if absent.
    truncation_mode
        ``'first_peak'`` or ``'adaptive'``.  Needed only when the run has no
        ``run_settings.json`` sidecar; when both exist they must agree.
    run_settings_path
        Explicit path to WP9's sidecar; by default it is looked for beside
        ``results_csv``.
    tolerances
        Parameter-recovery tolerances, defaulting to A 10 %, mu 15 %,
        lambda 1.0 h.
    data_dir
        The synthetic ``DATA`` directory.  Needed for the run-length
        robustness block, and for ``run.dataset_sha256`` when the run wrote no
        ``run_settings.json``: without either, the digest is reported as
        ``null`` rather than substituted with another file's hash.
    strict
        Fail on an incomplete ground-truth match (M19).
    strict_gates
        Fail when a ``classification_reason`` clause maps to no known gate, so
        a reworded pipeline message breaks the suite instead of silently
        emptying the attribution table.

    Returns
    -------
    dict
        The report, exactly as written to ``report.json``.
    """
    started = time.time()
    results_csv = Path(results_csv)
    ground_truth_csv = Path(ground_truth_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else None

    if truncation_mode is not None and truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"truncation_mode must be one of {TRUNCATION_MODES}, "
                         f"got {truncation_mode!r}")

    ground_truth = pd.read_csv(ground_truth_csv)
    results = pd.read_csv(results_csv)
    matched = match_results(ground_truth, results, strict=strict)

    settings_path = find_run_settings(results_csv, run_settings_path)
    settings = _read_json(settings_path)

    run_block = build_run_block(
        results_csv, ground_truth_csv, matched, truncation_mode,
        settings, settings_path, wall_clock_s=0.0, data_dir=data_dir)

    tolerances = dict(DEFAULT_TOLERANCES, **(tolerances or {}))

    attribution = gate_attribution(matched, strict=strict_gates)
    classification = classification_metrics(matched, lab_prior=load_lab_prior())
    recovery = parameter_recovery(matched, tolerances)
    uncertainty = uncertainty_calibration(matched)
    truncation = truncation_metrics(matched)
    failures = failure_analysis(matched, attribution)
    scen_table = per_scenario_table(matched)
    scen_recovery = per_scenario_recovery(matched, tolerances)
    per_curve = build_per_curve(matched, attribution)

    robustness = run_length_robustness(
        matched,
        data_dir=data_dir if run_length else None,
        truncation_mode=run_block["truncation_mode"],
        limit=run_length_limit,
    )

    attrs = matched.attrs
    summary = {
        "n_ground_truth": int(attrs.get("n_ground_truth", len(ground_truth))),
        "n_pipeline_results": int(attrs.get("n_results", len(results))),
        "n_matched": int(attrs.get("n_matched", len(matched))),
        "n_unmatched_ground_truth": int(attrs.get("n_unmatched_ground_truth", 0)),
        "n_unmatched_results": int(attrs.get("n_unmatched_results", 0)),
        "n_duplicate_ids": int(attrs.get("n_duplicate_ids", 0)),
        "n_undecided_verdicts": int(attrs.get("n_undecided", 0)),
        "coverage": float(attrs.get("coverage", 0.0)),
        "strict": bool(strict),
        "n_failures": int((~per_curve["correct"]).sum()),
        "unmatched_ground_truth_ids": list(attrs.get("unmatched_ground_truth_ids", []))[:50],
        "unmatched_results_ids": list(attrs.get("unmatched_results_ids", []))[:50],
        "duplicate_ids": list(attrs.get("duplicate_ids", []))[:50],
    }

    outputs: Dict[str, str] = {}
    per_curve.to_csv(out_dir / "per_curve.csv", index=False)
    outputs["per_curve"] = str(out_dir / "per_curve.csv")
    scen_table.to_csv(out_dir / "per_scenario.csv", index=False)
    outputs["per_scenario"] = str(out_dir / "per_scenario.csv")
    scen_recovery.to_csv(out_dir / "per_scenario_recovery.csv", index=False)
    outputs["per_scenario_recovery"] = str(out_dir / "per_scenario_recovery.csv")
    # One row per misclassified curve, with the numeric margin by which each
    # gate was missed; the full table is kept beside it for reference.
    mis = attribution[~attribution["correct"]]
    mis.to_csv(out_dir / "gate_attribution.csv", index=False)
    outputs["gate_attribution"] = str(out_dir / "gate_attribution.csv")
    attribution.to_csv(out_dir / "gate_attribution_all.csv", index=False)
    outputs["gate_attribution_all"] = str(out_dir / "gate_attribution_all.csv")

    if figures:
        try:
            report_figures = ComparisonReport(
                ground_truth, results, matched=matched).generate_full_report(
                    str(out_dir), prefix="validation")
            outputs.update({k: str(v) for k, v in report_figures.items()})
        except Exception as exc:  # pragma: no cover - plotting is not the metric
            outputs["figures_error"] = f"{type(exc).__name__}: {exc}"

    run_block["wall_clock_s"] = float(time.time() - started)

    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": run_block,
        "summary": summary,
        "classification": classification,
        "parameter_recovery": recovery,
        "uncertainty": uncertainty,
        "truncation": truncation,
        "run_length_robustness": robustness,
        "failure_analysis": failures,
        "outputs": outputs,
    }
    report["headline"] = headline(report)

    report_path = out_dir / "report.json"
    with open(report_path, "w") as handle:
        json.dump(json_safe(report), handle, indent=2)
    outputs["report"] = str(report_path)

    if verbose:
        print(report["headline"])
        print(f"Benchmark written to: {out_dir}")
    return report


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a pipeline run against synthetic ground truth.")
    parser.add_argument("--results-csv", required=True,
                        help="processing_results.csv from the pipeline run")
    parser.add_argument("--ground-truth", required=True,
                        help="ground_truth.csv from the generator")
    parser.add_argument("--out-dir", required=True,
                        help="Directory for report.json and the run's tables/figures")
    parser.add_argument("--truncation-mode", choices=list(TRUNCATION_MODES),
                        help="Required unless the run wrote a run_settings.json "
                             "sidecar beside its results")
    parser.add_argument("--run-settings",
                        help="Explicit path to the run_settings.json sidecar")
    parser.add_argument("--data-dir",
                        help="Synthetic DATA directory: supplies "
                             "run.dataset_sha256 when the run wrote no "
                             "run_settings.json (without either the digest is "
                             "null), and the run-length block")
    parser.add_argument("--tolerance-a", type=float, default=DEFAULT_TOLERANCES["A"])
    parser.add_argument("--tolerance-mu", type=float, default=DEFAULT_TOLERANCES["mu"])
    parser.add_argument("--tolerance-lambda", type=float,
                        default=DEFAULT_TOLERANCES["lambda"])
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the comparison figures")
    parser.add_argument("--run-length", action="store_true",
                        help="Refit every long expected-GOOD curve at 24 h "
                             "(needs --data-dir; slow)")
    parser.add_argument("--run-length-limit", type=int, default=None)
    parser.add_argument("--no-strict", action="store_true",
                        help="Score a deliberately partial run instead of failing "
                             "on an incomplete ground-truth match")
    parser.add_argument("--lenient-gates", action="store_true",
                        help="Do not fail on an unrecognised classification_reason")

    args = parser.parse_args(argv)

    try:
        run_benchmark(
            results_csv=args.results_csv,
            ground_truth_csv=args.ground_truth,
            out_dir=args.out_dir,
            truncation_mode=args.truncation_mode,
            run_settings_path=args.run_settings,
            tolerances={"A": args.tolerance_a, "mu": args.tolerance_mu,
                        "lambda": args.tolerance_lambda},
            data_dir=args.data_dir,
            strict=not args.no_strict,
            figures=not args.no_figures,
            run_length=args.run_length,
            run_length_limit=args.run_length_limit,
            strict_gates=not args.lenient_gates,
        )
    except (BenchmarkMatchError, BenchmarkProvenanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
