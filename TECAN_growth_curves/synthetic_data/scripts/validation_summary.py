#!/usr/bin/env python3
"""
validation_summary.py

Render VALIDATION_SUMMARY.md for one benchmark run.

Every number in the rendered markdown comes from that run's ``report.json``
(written by ``synthetic_data/validation/benchmark.py``).  Nothing is recomputed
here and nothing is hand-typed, so the markdown cannot disagree with the JSON.
The per-scenario, per-scenario-recovery and gate-attribution CSVs written beside
report.json are read only for the tables they carry.

Usage:
    python validation_summary.py                       # default run directory
    python validation_summary.py --run-dir synthetic_data/benchmark/v2
    python validation_summary.py --run-dir <dir> --output <path>

The default run directory is the first of these that holds a report.json:

    synthetic_data/output/comprehensive_test/validation/first_peak
    synthetic_data/output/comprehensive_test/validation/adaptive
    synthetic_data/benchmark/v2          (the tracked copy of the published run)

so a fresh checkout renders without arguments and a tree that has just run
`make step-9` renders the run it just produced.

The rendering goes beside the report it came from -- <run-dir>/VALIDATION_SUMMARY.md
unless --output says otherwise -- so the renderer never writes outside the
directory it was pointed at, and when that directory is a committed snapshot
under synthetic_data/benchmark/ it writes to stdout instead of dropping an extra
file into a snapshot its MANIFEST.json enumerates.  The rendered markdown is a
build artefact: no copy of it is committed, which is why no file in the
repository can hold a stale one.  The v1 summary, the only one that is
committed, is history and lives at
synthetic_data/benchmark/v1/VALIDATION_SUMMARY.md, labelled as such.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Scenario definitions are the single source of truth for which scenarios exist,
# what each one generates and which class it expects.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.curve_scenarios import (  # noqa: E402
    ALL_SCENARIOS,
    BAD_CURVE_SCENARIOS,
    EDGE_CASE_SCENARIOS,
    GOOD_GROWTH_SCENARIOS,
)

# ---------------------------------------------------------------------------
# Where a run lives
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # synthetic_data/
REPO_ROOT = BASE_DIR.parent
COMPREHENSIVE = BASE_DIR / "output" / "comprehensive_test"

REPORT_NAME = "report.json"
SUMMARY_NAME = "VALIDATION_SUMMARY.md"

# Searched in order; the first one holding a report.json wins.
DEFAULT_RUN_DIRS = (
    COMPREHENSIVE / "validation" / "first_peak",
    COMPREHENSIVE / "validation" / "adaptive",
    BASE_DIR / "benchmark" / "v2",
)

IN_SAMPLE_BANNER = (
    "**ML classifier active -- the synthetic curves are its training set; "
    "these numbers are IN-SAMPLE and are not a measure of generalisation.**"
)

# ---------------------------------------------------------------------------
# Scenario grouping
# ---------------------------------------------------------------------------
# Derived from the scenario definitions so the report cannot drift from what the
# generator produces.  The dictionary keys are load-bearing (tests pin them);
# the labels printed in the markdown are DISPLAY_CATEGORY, because "Bad (no /
# failed growth)" misdescribes the scenarios whose BAD label comes from the
# oracle rule -- borderline_noise and high_noise curves grow by 0.9-1.4 OD and
# are BAD because the fit is not recoverable, not because the well is flat.
SCENARIO_CATEGORIES = {
    "Good (standard growth)": sorted(GOOD_GROWTH_SCENARIOS),
    "Bad (no / failed growth)": sorted(BAD_CURVE_SCENARIOS),
    "Edge / borderline": sorted(EDGE_CASE_SCENARIOS),
}

DISPLAY_CATEGORY = {
    "Good (standard growth)": "Good growth",
    "Bad (no / failed growth)": "Bad curve",
    "Edge / borderline": "Edge / borderline",
}

CATEGORY_NOTE = {
    "Good (standard growth)":
        "Plain growth curves the pipeline is expected to accept.",
    "Bad (no / failed growth)":
        "Scenarios whose curves the pipeline is expected to reject. A BAD label "
        "here does not always mean a flat well: where `label rule = oracle` the "
        "curve grows, and the label says the fit is not recoverable from it.",
    "Edge / borderline":
        "Awkward but legitimate curves. Most of these expect GOOD; the expected "
        "class of each one is in the table.",
}


def describe_scenario(name: str) -> str:
    """Description and expected class, taken from the scenario definition.

    There is no second copy of the descriptions in this file: a hand-maintained
    table went stale (it described `truncation_challenge` as an experiment that
    ended before stationary phase, and `contamination` as OD spikes when it is a
    high-offset scenario). `src/curve_scenarios.py` is the only source.
    """
    config = ALL_SCENARIOS.get(name)
    if config is None:
        return "(scenario is not in the current catalog)"
    return f"{config.description} (expected {config.expected_class})"


def categorise_scenario(scenario_name: str) -> str:
    """Return the category key for a scenario."""
    for cat, members in SCENARIO_CATEGORIES.items():
        if scenario_name in members:
            return cat
    return "Other"


def display_category(scenario_name: str) -> str:
    return DISPLAY_CATEGORY.get(categorise_scenario(scenario_name), "Other")


def scenario_label_rule(name: str) -> str:
    config = ALL_SCENARIOS.get(name)
    return getattr(config, "label_rule", "") if config is not None else ""


def scenario_scoring_mode(name: str) -> str:
    config = ALL_SCENARIOS.get(name)
    return getattr(config, "scoring_mode", "") if config is not None else ""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def extract_curve_id(strain_name: str) -> str:
    """Extract the CURVE#### pattern from a pipeline strain name.

    Pipeline names look like 'SYNTHETIC-400pts-99h-CURVE0001'.
    """
    match = re.search(r"(CURVE\d+)", str(strain_name))
    return match.group(1) if match else str(strain_name)


def pct(value, decimals: int = 1) -> str:
    """Format a fraction as a percentage, or '--' when it is missing."""
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    return f"{value * 100:.{decimals}f}%"


def fmt(value, decimals: int = 4) -> str:
    """Format a float, or '--' when it is missing."""
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def num(value) -> str:
    """Format an integer-ish value, or '--' when it is missing."""
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    return f"{int(value)}"


def short_sha(value, n: int = 12) -> str:
    return str(value)[:n] if value else "--"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to_repo(path) -> str:
    """Path relative to the repository root when it is inside it."""
    if not path:
        return "--"
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(p)


def recovery_sentence(block: dict, label: str) -> str:
    """One sentence about a parameter, generated from the measured numbers.

    This replaces the hard-coded praise the previous renderer emitted ("Excellent
    recovery"), which survived a scenario with a 0 % tolerance pass rate.
    """
    if not block:
        return f"- **{label}**: not measured in this run."
    rate = block.get("tolerance_pass_rate")
    med = block.get("median_rel_error")
    tol = block.get("tolerance")
    kind = block.get("tolerance_kind", "")
    status = block.get("recovery_status", "")
    if rate is None:
        verdict = "not measured"
    elif rate >= 0.95:
        verdict = "recovered"
    elif rate >= 0.80:
        verdict = "mostly recovered"
    elif rate >= 0.60:
        verdict = "recovered on most curves"
    elif rate >= 0.40:
        verdict = "recovered on about half the curves"
    else:
        verdict = "not recovered"
    tol_txt = (f"{tol:g} h" if kind == "absolute_hours" else f"{tol:.0%}") if tol is not None else "the tolerance"
    parts = [
        f"- **{label}**: {verdict} -- {pct(rate)} of the curves land within {tol_txt}",
        f"median relative error {pct(med, 1)}",
        f"RMSE {fmt(block.get('rmse'))}",
        f"bias {fmt(block.get('bias'))}",
        f"n = {num(block.get('n'))}",
    ]
    line = ", ".join(parts) + "."
    if status and status != "ok":
        line += f" Recovery status flagged **{status}**."
    return line


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def snapshot_for_mode(mode: str):
    """A tracked benchmark snapshot that holds the run for `mode`, if there is one.

    `synthetic_data/benchmark/<version>/` holds the published copy of a run
    directory: the same report.json and the same CSVs, minus the plots. When the
    run directory itself has been cleaned away, that copy is the same run.
    """
    if not mode:
        return None
    for candidate in sorted((BASE_DIR / "benchmark").glob("*")):
        report_path = candidate / REPORT_NAME
        if not report_path.exists():
            continue
        try:
            with open(report_path, "r") as fh:
                run = json.load(fh).get("run", {})
        except (OSError, ValueError):
            continue
        if run.get("truncation_mode") == mode:
            return candidate
    return None


def resolve_run_dir(explicit) -> Path:
    """Return the run directory to render, or exit with a usable message."""
    if explicit:
        run_dir = Path(explicit)
        if (run_dir / REPORT_NAME).exists():
            return run_dir

        # A cleaned-away run directory still names its truncation mode. When the
        # tracked snapshot of that same run is in the tree, render that instead
        # of failing -- it is the identical report.json.
        fallback = snapshot_for_mode(run_dir.name)
        if fallback is not None:
            print(f"NOTE: {rel_to_repo(run_dir)} holds no {REPORT_NAME} "
                  f"(run `make step-9` to produce it).")
            print(f"      Rendering the tracked copy of that same "
                  f"{run_dir.name} run: {rel_to_repo(fallback)}")
            return fallback

        print("No renderable run directory.")
        print(f"  {rel_to_repo(run_dir)} holds no {REPORT_NAME}, and no tracked "
              f"snapshot under synthetic_data/benchmark/ carries a "
              f"'{run_dir.name}' run.")
        if (run_dir / "validation_report.json").exists():
            print("  It does hold a pre-rebuild validation_report.json, which this "
                  "renderer does not read; the summary rendered from it is frozen "
                  "beside it as VALIDATION_SUMMARY.md.")
        print("  Run `make step-9` first, or pass --run-dir.")
        sys.exit(1)

    for candidate in DEFAULT_RUN_DIRS:
        if (candidate / REPORT_NAME).exists():
            return candidate

    print("No renderable run directory.")
    for candidate in DEFAULT_RUN_DIRS:
        print(f"  looked in: {rel_to_repo(candidate)}")
    print("  Run `make step-9` first, or pass --run-dir.")
    sys.exit(1)


def is_snapshot_dir(run_dir: Path) -> bool:
    """True for a run directory that is a committed release snapshot.

    `synthetic_data/benchmark/v1`, `.../v2` and any later snapshot: a fixed,
    manifest-verified copy of a run, not a directory a run writes into.
    """
    try:
        run_dir.resolve().relative_to((BASE_DIR / "benchmark").resolve())
    except (ValueError, OSError):
        return False
    return True


def default_output_for(run_dir: Path):
    """Where a rendering of `run_dir` goes when --output is not given.

    Beside the report it was rendered from -- `run_dir/VALIDATION_SUMMARY.md` --
    for a real run directory: a summary is the rendering of one run and belongs
    to that run. The renderer never writes outside the directory it was pointed
    at.

    None for a committed snapshot under `synthetic_data/benchmark/`, meaning
    "write to stdout": a snapshot is fixed and its MANIFEST.json enumerates it,
    so rendering one must not drop an extra file into it. Pass --output to save
    the markdown.

    An earlier version wrote a snapshot's rendering to
    `output/comprehensive_test/VALIDATION_SUMMARY.md`, a path this renderer does
    not own and which the release deletes (the v1 file that lived there is kept
    as history at `benchmark/v1/VALIDATION_SUMMARY.md`). Every render therefore
    left an untracked build artefact at a deleted path, which no commit would
    have picked up. The summary is a build artefact: it is rendered on demand
    and no copy of it is committed, so the repository cannot hold a stale one.
    """
    if is_snapshot_dir(run_dir):
        return None
    return run_dir / SUMMARY_NAME


def flat_rule_block(per_curve, mode: str) -> dict | None:
    """Verdict-level counts for one `scoring_mode` block under the flat rule.

    "Flat rule" means the headline's own rule, applied unchanged: a curve is
    correct when the pipeline's accept/reject verdict matched the truth's
    GOOD/BAD.  It is *not* the same question as the block rule `report.json`
    records for ``a_not_identifiable`` (which also asks whether the reported A
    is supportable from a truncated run), and on that block the two rules give
    very different numbers -- which is precisely why both have to be printed.

    The counts come from ``per_curve.csv``, the same matched frame the report
    was computed from, not from a re-match of ground truth against results.
    Returns None when the file or its columns are absent.
    """
    if per_curve is None:
        return None
    needed = {"scoring_mode", "gt_is_good", "pred_is_good"}
    if not needed.issubset(set(per_curve.columns)):
        return None
    sub = per_curve[per_curve["scoring_mode"].astype(str) == mode]
    if not len(sub):
        return None
    gt = sub["gt_is_good"].astype(bool).to_numpy()
    pred = sub["pred_is_good"].astype(bool).to_numpy()
    n_correct = int((gt == pred).sum())
    return {"n": int(len(sub)), "n_correct": n_correct,
            "accuracy": n_correct / float(len(sub))}


def reported_flat_rule(block: dict) -> dict | None:
    """The block's flat-rule counts as `report.json` states them, if it does.

    `truncation.<block>.flat_rule` is written by the validator when it scores a
    held-out block under the headline's rule as well as under the block's own.
    When it is there the renderer prints it and nothing is recomputed.
    """
    flat = (block or {}).get("flat_rule") or {}
    n = flat.get("n", (block or {}).get("n"))
    n_correct = flat.get("n_correct")
    if n in (None, 0) or n_correct is None:
        return None
    accuracy = flat.get("accuracy")
    if accuracy is None:
        accuracy = n_correct / float(n)
    return {"n": int(n), "n_correct": int(n_correct), "accuracy": float(accuracy)}


def flat_rule_is_trustworthy(flat_control: dict | None,
                             reported_control: dict | None) -> bool:
    """Cross-check the recomputation against a block report.json already scores.

    ``preprocessing_blind`` is the control: `report.json` states its flat
    accuracy and confusion matrix, so if the recomputation disagrees with the
    JSON there, no recomputed number is printed anywhere in the document.  This
    keeps the guarantee that the markdown cannot contradict the report.
    """
    if not reported_control or reported_control.get("accuracy") is None:
        return True                      # nothing to check against; nothing to break
    if flat_control is None:
        return False
    return (flat_control["n"] == reported_control.get("n")
            and round(flat_control["accuracy"], 3)
            == round(float(reported_control["accuracy"]), 3))


def load_run(run_dir: Path) -> dict:
    """Load report.json plus the optional CSV tables written beside it."""
    report_path = run_dir / REPORT_NAME
    with open(report_path, "r") as fh:
        report = json.load(fh)

    if "classification" not in report:
        print(f"ERROR: {report_path} has no 'classification' block.", file=sys.stderr)
        print("  This renderer reads the report.json written by "
              "synthetic_data/validation/benchmark.py.", file=sys.stderr)
        print("  A pre-rebuild validation_report.json is not renderable; the v1 "
              "summary it produced is frozen at "
              "synthetic_data/benchmark/v1/VALIDATION_SUMMARY.md.", file=sys.stderr)
        sys.exit(1)

    def maybe_csv(name):
        path = run_dir / name
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:                      # pragma: no cover
                print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None

    # True only for a directory that really is a committed snapshot under
    # `synthetic_data/benchmark/`.  An earlier version inverted the test -- it
    # asked whether the directory sat under `output/comprehensive_test/
    # validation/` and called everything else a snapshot -- so rendering any
    # other run directory (a scratch copy, a run written elsewhere) produced a
    # document claiming to be "the published run, taken from its tracked
    # snapshot".  The same predicate that decides where the rendering is
    # written now decides what the rendering says about itself.
    from_snapshot = is_snapshot_dir(run_dir)

    return {
        "report": report,
        "from_snapshot": from_snapshot,
        "report_path": report_path,
        "report_sha256": sha256_of(report_path),
        "run_dir": run_dir,
        "per_scenario": maybe_csv("per_scenario.csv"),
        "per_scenario_recovery": maybe_csv("per_scenario_recovery.csv"),
        "gate_attribution": maybe_csv("gate_attribution.csv"),
        "per_curve": maybe_csv("per_curve.csv"),
    }


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------

def _metric_rows(block: dict) -> list:
    """The headline metrics, in the order this report leads with."""
    return [
        ("**Balanced accuracy**", pct(block.get("balanced_accuracy")),
         "mean of sensitivity and specificity; unaffected by class balance"),
        ("**Sensitivity (recall)**", pct(block.get("sensitivity", block.get("recall"))),
         "of the curves the truth calls GOOD, the fraction accepted"),
        ("**Specificity**", pct(block.get("specificity")),
         "of the curves the truth calls BAD, the fraction rejected"),
        ("Accuracy", pct(block.get("accuracy")),
         f"raw fraction correct at a GOOD prevalence of {pct(block.get('prevalence'))}"),
        ("Precision", pct(block.get("precision")),
         "of the curves accepted, the fraction the truth calls GOOD"),
        ("NPV", pct(block.get("npv")),
         "of the curves rejected, the fraction the truth calls BAD"),
        ("F1", pct(block.get("f1_score")), "harmonic mean of precision and recall"),
        ("MCC", fmt(block.get("mcc"), 4), "correlation of verdict and truth; 0 = chance"),
    ]


def render(data: dict) -> str:
    report = data["report"]
    run = report.get("run", {})
    summary = report.get("summary", {})
    cls = report.get("classification", {})

    lines = []

    def ln(text=""):
        lines.append(text)

    mode = run.get("truncation_mode", "unknown")

    # ------------------------------------------------------------------
    # Title and provenance
    # ------------------------------------------------------------------
    ln("# Synthetic Data Validation Summary")
    ln()
    ln(f"Generated by `synthetic_data/scripts/validation_summary.py` from "
       f"`{rel_to_repo(data['report_path'])}`, which was written at "
       f"{run.get('timestamp_utc', 'an unrecorded time')}. Every number below is "
       f"read from that file; none is recomputed here, and re-rendering the same "
       f"report reproduces this file byte for byte.")
    ln()

    if data.get("from_snapshot"):
        ln(f"This is the rendering of the published `{mode}` run, taken from its "
           f"tracked snapshot. Each truncation mode is a separate benchmark and "
           f"gets its own rendering of this file inside its own run directory; "
           f"`make step-9` writes those. This file is a build artefact and no "
           f"copy of it is committed: render it when you need it, and read the "
           f"numbers from `report.json` when you need to cite them.")
        ln()
    else:
        ln(f"This is the rendering of the `{mode}` run in "
           f"`{rel_to_repo(data['run_dir'])}`. Each truncation mode is a "
           f"separate benchmark and gets its own rendering of this file inside "
           f"its own run directory; `make step-9` writes those. This file is a "
           f"build artefact and no copy of it is committed: render it when you "
           f"need it, and read the numbers from `report.json` when you need to "
           f"cite them.")
        ln()

    if run.get("ml_classify"):
        ln(f"> {IN_SAMPLE_BANNER}")
        ln()

    ln("## Run provenance")
    ln()
    ln("| Field | Value |")
    ln("|---|---|")
    ln(f"| **Truncation mode** | `{mode}` "
       f"(use_adaptive={run.get('use_adaptive')}, use_first_peak={run.get('use_first_peak')}) |")
    ln(f"| ML classifier in the verdict | {run.get('ml_classify')} |")
    ln(f"| Seed | {run.get('seed', '--')} |")
    ln(f"| Curves generated | {num(run.get('n_curves'))} in "
       f"{num(run.get('n_scenarios'))} scenarios |")
    ln(f"| Curves matched to a pipeline verdict | {num(summary.get('n_matched'))} "
       f"(coverage {pct(summary.get('coverage'))}, strict={summary.get('strict')}) |")
    ln(f"| Dataset sha256 | `{short_sha(run.get('dataset_sha256'), 16)}` |")
    ln(f"| Ground truth sha256 | `{short_sha(run.get('ground_truth_sha256'), 16)}` |")
    ln(f"| Pipeline results sha256 | `{short_sha(run.get('results_sha256'), 16)}` |")
    ln(f"| Config | `{rel_to_repo(run.get('config_path'))}` "
       f"sha256 `{short_sha(run.get('config_sha256'), 16)}` |")
    ln(f"| Git SHA at run time | `{short_sha(run.get('git_sha'), 12)}` "
       f"(dirty={run.get('git_dirty')}) |")
    ln(f"| Run timestamp (UTC) | {run.get('timestamp_utc', '--')} |")
    ln(f"| Report schema | {report.get('schema_version', '--')} "
       f"(sha256 `{short_sha(data['report_sha256'], 16)}`) |")
    ln(f"| Provenance source | {run.get('provenance', '--')} |")
    ln()
    ln("Reproduce this rendering with:")
    ln()
    ln("```bash")
    ln(f"MPLBACKEND=Agg python synthetic_data/scripts/validation_summary.py "
       f"--run-dir {rel_to_repo(data['run_dir'])}")
    ln("```")
    ln()

    # ------------------------------------------------------------------
    # Headline
    # ------------------------------------------------------------------
    ln("## Headline")
    ln()
    n_head = cls.get("n")
    excluded = cls.get("excluded_from_headline", {}) or {}
    ln(f"Scored over **{num(n_head)} curves** in `{mode}` truncation mode. "
       f"Positive = GOOD (the pipeline accepted the curve and the truth agrees it "
       f"is usable); negative = BAD.")
    if excluded.get("n"):
        by_mode = ", ".join(f"{k} {v}" for k, v in sorted(
            (excluded.get("by_scoring_mode") or {}).items()))
        ln()
        ln(f"{num(excluded.get('n'))} further curves ({by_mode}) are held out of "
           f"the headline by their `scoring_mode` and are scored separately below, "
           f"each block under the rule its own section names -- the headline's flat "
           f"GOOD/BAD rule for every block, plus, for `a_not_identifiable`, the "
           f"stricter either/or rule `report.json` records for it, which gives a "
           f"different number on the same curves. Nothing is dropped: the flat "
           f"all-curve number is in that section.")
    ln()
    ln("| Metric | Value | Meaning |")
    ln("|---|---:|---|")
    for label, value, meaning in _metric_rows(cls):
        ln(f"| {label} | {value} | {meaning} |")
    ln()

    cm = cls.get("confusion_matrix", {}) or {}
    tp = cm.get("true_positives", 0)
    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)
    ln("| | Predicted GOOD | Predicted BAD | Total |")
    ln("|---|---:|---:|---:|")
    ln(f"| **Truth GOOD** | {tp} (TP) | {fn} (FN) | {tp + fn} |")
    ln(f"| **Truth BAD** | {fp} (FP) | {tn} (TN) | {fp + tn} |")
    ln(f"| **Total** | {tp + fp} | {fn + tn} | {tp + tn + fp + fn} |")
    ln()

    lab_acc = cls.get("lab_weighted_accuracy")
    if lab_acc is not None:
        ln(f"Two weighted views of the same verdicts: **lab-weighted accuracy "
           f"{pct(lab_acc)}** (curves reweighted to the delta-OD mix measured on "
           f"the real plates, source `{rel_to_repo(cls.get('lab_prior_source'))}`, "
           f"weight covered {fmt(cls.get('lab_weight_covered'), 3)}) and "
           f"**macro accuracy {pct(cls.get('macro_accuracy'))}** (every scenario "
           f"weighted equally).")
        ln()
        ln("Lab weighting is *not* the headline. About a third of real conditions "
           "are dead wells that the pre-fit SNR gate resolves trivially, so any "
           "lab-weighted figure runs higher than the uniform one for reasons that "
           "have nothing to do with how well hard curves are handled.")
        ln()

    core = cls.get("core32")
    if core:
        ln("### The 32 original scenarios, scored on their own")
        ln()
        ln(f"The suite the repository published before the rebuild covered "
           f"{len(core.get('scenarios', []))} scenarios. Scoring only those "
           f"scenarios in this run gives the number that is directly comparable "
           f"with the older published one. It is *not* the headline; the headline "
           f"is the whole suite above.")
        ln()
        ln("| Metric | Core-32 subset | Full headline |")
        ln("|---|---:|---:|")
        ln(f"| n | {num(core.get('n'))} | {num(n_head)} |")
        ln(f"| Balanced accuracy | {pct(core.get('balanced_accuracy'))} | "
           f"{pct(cls.get('balanced_accuracy'))} |")
        ln(f"| Sensitivity | {pct(core.get('sensitivity'))} | {pct(cls.get('sensitivity'))} |")
        ln(f"| Specificity | {pct(core.get('specificity'))} | {pct(cls.get('specificity'))} |")
        ln(f"| Accuracy | {pct(core.get('accuracy'))} | {pct(cls.get('accuracy'))} |")
        ln(f"| Precision | {pct(core.get('precision'))} | {pct(cls.get('precision'))} |")
        ln(f"| MCC | {fmt(core.get('mcc'), 4)} | {fmt(cls.get('mcc'), 4)} |")
        ccm = core.get("confusion_matrix", {}) or {}
        ln(f"| TP / TN / FP / FN | {ccm.get('true_positives')} / "
           f"{ccm.get('true_negatives')} / {ccm.get('false_positives')} / "
           f"{ccm.get('false_negatives')} | {tp} / {tn} / {fp} / {fn} |")
        ln()
        ln("The attribution of the difference between the two -- which part is the "
           "label rule, which the converters, which the new scenario families -- is "
           "in `docs/BENCHMARK_LEDGER.md`, at the repository root. "
           "It is not restated here.")
        ln()

    return "\n".join(lines) + "\n" + render_rest(data)


def render_rest(data: dict) -> str:
    """The second half of the document (kept separate to stay readable)."""
    report = data["report"]
    run = report.get("run", {})
    cls = report.get("classification", {})
    recovery = report.get("parameter_recovery", {})
    uncertainty = report.get("uncertainty", {})
    truncation = report.get("truncation", {})
    robustness = report.get("run_length_robustness", {})
    failures = report.get("failure_analysis", {})
    summary = report.get("summary", {})

    lines = []

    def ln(text=""):
        lines.append(text)

    # ------------------------------------------------------------------
    # Held out of the headline
    # ------------------------------------------------------------------
    ln("## Blocks held out of the headline, scored anyway")
    ln()
    a_ni = truncation.get("a_not_identifiable") or {}
    pb = truncation.get("preprocessing_blind") or {}
    per_curve_frame = data.get("per_curve")
    # report.json is preferred when it carries the block's flat-rule counts
    # itself; per_curve.csv is the fallback, and is cross-checked against the
    # one block whose flat accuracy the report already states.
    a_ni_flat = reported_flat_rule(a_ni)
    if a_ni_flat is None:
        a_ni_flat = flat_rule_block(per_curve_frame, "a_not_identifiable")
        pb_flat = flat_rule_block(per_curve_frame, "preprocessing_blind")
        if not flat_rule_is_trustworthy(pb_flat, pb):
            a_ni_flat = None
    if not a_ni and not pb:
        ln("This run excludes nothing from its headline.")
        ln()
    else:
        ln("A `scoring_mode` other than `standard` says the current gate set "
           "cannot decide a curve fairly, so those curves are scored here instead "
           "of silently inside the headline.")
        ln()
        if a_ni:
            if a_ni_flat:
                ln(f"**`a_not_identifiable`** -- {num(a_ni.get('n'))} curves, "
                   f"scored under two different rules that give two different "
                   f"numbers. Both are printed, because this block is the one "
                   f"place in the suite where they disagree.")
                ln()
                ln("| rule | correct | of | rate |")
                ln("|---|---:|---:|---:|")
                ln(f"| flat GOOD/BAD -- the headline's rule, applied unchanged | "
                   f"{num(a_ni_flat['n_correct'])} | {num(a_ni_flat['n'])} | "
                   f"{pct(a_ni_flat['accuracy'])} |")
                ln(f"| block rule -- the either/or rule below, which `report.json` "
                   f"records for this block | {num(a_ni.get('n_correct'))} | "
                   f"{num(a_ni.get('n'))} | {pct(a_ni.get('pass_rate'))} |")
                ln()
                ln(f"Block rule: {a_ni.get('rule', '--')}")
                ln()
                ln("The two rules ask different questions. The flat rule asks "
                   "only whether the accept/reject verdict matched the truth, and "
                   "on most of these curves it did -- the truth calls them GOOD "
                   "and the pipeline accepted them. The block rule additionally "
                   "asks whether the **reported A** is supportable from a run "
                   "that stopped before the plateau, and usually it is not: the "
                   "verdict is right while the amplitude the user would read off "
                   "is not identifiable. That is why the block is held out of the "
                   "headline. `docs/BENCHMARK_LEDGER.md` carries the same two "
                   "numbers side by side; neither of them is the block's score on "
                   "its own.")
                ln()
            else:
                ln(f"**`a_not_identifiable`** -- {num(a_ni.get('n'))} curves. "
                   f"{num(a_ni.get('n_correct'))} correct, pass rate "
                   f"{pct(a_ni.get('pass_rate'))}, under the block rule below and "
                   f"**not** under the headline's flat GOOD/BAD rule, which scores "
                   f"this block considerably higher (the held-out table in "
                   f"`docs/BENCHMARK_LEDGER.md` carries both; the flat count needs "
                   f"`per_curve.csv`, which this run directory does not carry). "
                   f"Block rule: {a_ni.get('rule', '--')}")
                ln()
        if pb:
            pcm = pb.get("confusion_matrix", {}) or {}
            ln(f"**`preprocessing_blind`** -- {num(pb.get('n'))} curves, scored "
               f"under the flat GOOD/BAD rule only (this block carries no second "
               f"rule): accuracy "
               f"{pct(pb.get('accuracy'))} (balanced {pct(pb.get('balanced_accuracy'))}, "
               f"TP/TN/FP/FN {pcm.get('true_positives')}/{pcm.get('true_negatives')}/"
               f"{pcm.get('false_positives')}/{pcm.get('false_negatives')}). "
               f"Reason: {pb.get('reason', '--')}")
            ln()
        counts = cls.get("scoring_mode_counts") or {}
        if counts:
            ln("Curves by scoring mode: " +
               ", ".join(f"`{k}` {v}" for k, v in sorted(counts.items())) + ".")
            ln()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    ln("## Matching")
    ln()
    ln("| Field | Value |")
    ln("|---|---:|")
    for label, key in [
        ("Ground-truth curves", "n_ground_truth"),
        ("Pipeline result rows", "n_pipeline_results"),
        ("Matched", "n_matched"),
        ("Unmatched ground truth", "n_unmatched_ground_truth"),
        ("Unmatched results", "n_unmatched_results"),
        ("Duplicate curve ids", "n_duplicate_ids"),
        ("Undecided verdicts", "n_undecided_verdicts"),
        ("Misclassified (all curves)", "n_failures"),
    ]:
        ln(f"| {label} | {num(summary.get(key))} |")
    ln()

    # ------------------------------------------------------------------
    # Parameter recovery, four strata
    # ------------------------------------------------------------------
    ln("## Parameter recovery")
    ln()
    tol = recovery.get("tolerances", {}) or {}
    ln(f"True values are the **measured** parameters of the curve that was "
       f"written (`{recovery.get('true_source', '--')}`), not the sampler's "
       f"request. Tolerances: A {tol.get('A')} relative, mu {tol.get('mu')} "
       f"relative, lambda {tol.get('lambda')} h absolute. "
       f"{num(recovery.get('n_excluded_degenerate_gt'))} curves are excluded from "
       f"every stratum as degenerate ground truth "
       f"({recovery.get('degenerate_gt_definition', '--')}).")
    ln()
    strata = [
        ("accepted_good", "Accepted, truth GOOD",
         "what a user actually sees: the curves the pipeline reported"),
        ("all_fitted_good", "All fitted, truth GOOD",
         "the same estimator without the pipeline's own filter"),
        ("rejected_good", "Rejected, truth GOOD",
         "the curves the gates threw away"),
        ("accepted_bad", "Accepted, truth BAD",
         "the false positives, measured against their own truth"),
    ]
    ln("| Stratum | n | A R² | A RMSE | A pass | mu R² | mu RMSE | mu pass | "
       "lambda R² | lambda RMSE | lambda pass |")
    ln("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, label, _why in strata:
        block = recovery.get(key) or {}
        if not block:
            continue
        cells = [label, num(block.get("n"))]
        for p in ("A", "mu", "lambda"):
            pb_ = block.get(p, {}) or {}
            cells += [fmt(pb_.get("r_squared_across_scenarios"), 4),
                      fmt(pb_.get("rmse"), 5),
                      pct(pb_.get("tolerance_pass_rate"))]
        ln("| " + " | ".join(cells) + " |")
    ln()
    for key, label, why in strata:
        if recovery.get(key):
            ln(f"- **{label}** ({why}), n = {num((recovery[key] or {}).get('n'))}.")
    ln()
    ln("R² here is `r_squared_across_scenarios`: 1 - SSE/SST of fitted against true "
       "over the pooled stratum. It goes negative when the fits are further from "
       "the truth than the stratum's own mean would be, which is what a rejected or "
       "unidentifiable set looks like; the pass rate and RMSE are the numbers to "
       "read on those rows.")
    ln()

    accepted = recovery.get("accepted_good") or {}
    if accepted:
        ln("On the curves a user is shown -- accepted, and GOOD in the truth:")
        ln()
        ln(recovery_sentence(accepted.get("A"), "A (maximum OD)"))
        ln(recovery_sentence(accepted.get("mu"), "mu (maximum growth rate)"))
        ln(recovery_sentence(accepted.get("lambda"), "lambda (lag time)"))
        allp = accepted.get("all_parameters_within_tolerance") or {}
        if allp:
            ln(f"- **All three within tolerance on the same curve**: "
               f"{pct(allp.get('pass_rate'))} of {num(allp.get('n'))} curves.")
        ln()

    bias = recovery.get("selection_bias_delta") or {}
    if bias:
        ln("### Selection bias")
        ln()
        ln(f"Accepted-only {bias.get('parameter')} RMSE "
           f"{fmt(bias.get('accepted_good_rmse'), 5)} against "
           f"{fmt(bias.get('all_fitted_good_rmse'), 5)} over every fitted GOOD "
           f"curve, a ratio of {fmt(bias.get('ratio'), 2)}. "
           f"{bias.get('note', '')}")
        ln()

    # The false positives, in the words the audit asked for.
    fp_n = (cls.get("confusion_matrix", {}) or {}).get("false_positives")
    per_curve = data.get("per_curve")
    fp_median = None
    if per_curve is not None and "error_type" in per_curve.columns:
        sub = per_curve[per_curve["error_type"] == "false_positive"]
        if "scoring_mode" in sub.columns:
            sub = sub[~sub["scoring_mode"].isin(["a_not_identifiable",
                                                 "preprocessing_blind"])]
        rel = pd.to_numeric(sub.get("relerr_A"), errors="coerce").dropna()
        if len(rel):
            fp_median = float(rel.abs().median())
    if fp_n:
        sentence = (f"The {fp_n} false positives in the headline are not silent "
                    f"failures: every one of them was fitted and reported to the "
                    f"user")
        if fp_median is not None:
            sentence += f", with median |A| error {pct(fp_median)}"
        sentence += (". A curve the pipeline accepts wrongly still carries "
                     "parameters, and those parameters are what a reader acts on.")
        ln("### The false positives")
        ln()
        ln(sentence)
        fp_all = failures.get("n_false_positives")
        no_gate = failures.get("n_false_positives_with_no_failed_gate")
        if no_gate is not None and fp_all:
            ln()
            share = "All" if no_gate == fp_all else f"{no_gate} of the"
            ln(f"{share} {fp_all} false positives in the run ({fp_n} of them inside "
               f"the headline, the rest in the held-out blocks) failed **no** gate "
               f"at all. There is no threshold to point at: they sit inside the "
               f"accepted region, not near its edge, and tightening a gate would "
               f"cost accepted GOOD curves before it caught them.")
        ln()

    # ------------------------------------------------------------------
    # Per-scenario recovery
    # ------------------------------------------------------------------
    psr = data.get("per_scenario_recovery")
    if psr is not None and len(psr):
        ln("## Per-scenario parameter recovery (accepted, truth GOOD)")
        ln()
        ln("Tolerance pass rate per scenario and parameter. `recovery_status` is "
           "the report's own flag: `degenerate` means the scenario's spread is too "
           "small (or its fits too far off) for an R² across curves to mean "
           "anything, `poor` means the pass rate is low.")
        ln()
        sub = psr[psr["stratum"] == "accepted_good"] if "stratum" in psr.columns else psr
        wide = {}
        for _, row in sub.iterrows():
            key = (row.get("scenario"), row.get("model_type"))
            entry = wide.setdefault(key, {"n": row.get("n")})
            entry[row.get("parameter")] = row
        ln("| Scenario | Model | n | A pass | mu pass | lambda pass | all three | Flags |")
        ln("|---|---|---:|---:|---:|---:|---:|---|")
        for (scenario, model), entry in sorted(wide.items(), key=lambda kv: str(kv[0][0])):
            flags = sorted({str(entry[p].get("recovery_status"))
                            for p in ("A", "mu", "lambda") if p in entry
                            and str(entry[p].get("recovery_status")) != "ok"})
            ln(f"| {scenario} | {model} | {num(entry.get('n'))} | "
               f"{pct(entry['A'].get('pass_rate')) if 'A' in entry else '--'} | "
               f"{pct(entry['mu'].get('pass_rate')) if 'mu' in entry else '--'} | "
               f"{pct(entry['lambda'].get('pass_rate')) if 'lambda' in entry else '--'} | "
               f"{pct(entry['all'].get('pass_rate')) if 'all' in entry else '--'} | "
               f"{', '.join(flags) if flags else ''} |")
        ln()

    # ------------------------------------------------------------------
    # Per-scenario accuracy
    # ------------------------------------------------------------------
    ln("## Per-scenario accuracy")
    ln()
    per_scenario = data.get("per_scenario")
    rows = []
    if per_scenario is not None and len(per_scenario):
        for _, r in per_scenario.iterrows():
            rows.append({
                "scenario": r.get("scenario"),
                "n": r.get("n"),
                "expected_class": r.get("expected_class"),
                "accuracy": r.get("accuracy"),
                "false_positives": r.get("false_positives"),
                "false_negatives": r.get("false_negatives"),
            })
    else:
        for name, r in (cls.get("per_scenario") or {}).items():
            rows.append({
                "scenario": name,
                "n": r.get("n"),
                "expected_class": r.get("expected_class"),
                "accuracy": r.get("accuracy"),
                "false_positives": r.get("false_positives"),
                "false_negatives": r.get("false_negatives"),
            })
    rows.sort(key=lambda r: (r["accuracy"] if r["accuracy"] is not None else 1.0,
                             str(r["scenario"])))

    ln("Sorted worst first. `label rule` says where the expected class came from: "
       "`scenario` means the catalog asserts it, `oracle` means it was measured on "
       "the curve that was actually written -- so an `oracle` scenario's curves can "
       "carry different labels, and the catalog bucket it is filed under is not a "
       "claim about any individual curve.")
    ln()
    ln("| Scenario | Catalog bucket | Expected | Label rule | Scoring mode | n | "
       "Accuracy | FP | FN |")
    ln("|---|---|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        s = r["scenario"]
        ln(f"| {s} | {display_category(s)} | {r['expected_class']} | "
           f"{scenario_label_rule(s) or '--'} | {scenario_scoring_mode(s) or '--'} | "
           f"{num(r['n'])} | {pct(r['accuracy'])} | {num(r['false_positives'])} | "
           f"{num(r['false_negatives'])} |")
    ln()
    n_perfect = sum(1 for r in rows if r["accuracy"] == 1.0)
    ln(f"{n_perfect} of {len(rows)} scenarios scored 100 %. The remaining "
       f"{len(rows) - n_perfect} each have at least one miss.")
    ln()

    # ------------------------------------------------------------------
    # Which gate fired
    # ------------------------------------------------------------------
    ln("## Which gate fired")
    ln()
    gate_source = failures.get("gate_source") or {}
    ln("Gate attribution source: " +
       (", ".join(f"`{k}` on {v} curves" for k, v in gate_source.items())
        if gate_source else "not recorded") +
       ". A curve can fail more than one gate, so the columns do not sum to the "
       "number of curves.")
    ln()
    totals = failures.get("gate_totals_misclassified") or {}
    if totals:
        ln("Gates fired on **misclassified** curves:")
        ln()
        ln("| Gate | Curves |")
        ln("|---|---:|")
        for gate, n in sorted(totals.items(), key=lambda kv: -kv[1]):
            ln(f"| `{gate}` | {n} |")
        ln()
    by_gate = failures.get("by_gate") or {}
    if by_gate:
        gates = sorted({g for row in by_gate.values() for g in row})
        firing = {k: v for k, v in by_gate.items() if v}
        silent = sorted(k for k, v in by_gate.items() if not v)
        ln("Per scenario, over its misclassified curves:")
        ln()
        ln("| Scenario | " + " | ".join(f"`{g}`" for g in gates) + " |")
        ln("|---" * (len(gates) + 1) + "|")
        for scenario in sorted(firing):
            row = firing[scenario]
            cells = [str(row.get(g, "")) for g in gates]
            ln(f"| {scenario} | " + " | ".join(cells) + " |")
        ln()
        if silent:
            ln("Scenarios whose misclassifications fired **no** gate at all -- the "
               "pipeline accepted them outright, so no threshold explains them: " +
               ", ".join(f"`{s}`" for s in silent) + ".")
            ln()
    ln(f"False negatives split {num(failures.get('n_false_negatives_unfitted'))} "
       f"never fitted / {num(failures.get('n_false_negatives_gated'))} fitted and "
       f"then gated. Full per-curve attribution, with each gate's margin, is in "
       f"`gate_attribution.csv` beside this file.")
    degen = failures.get("degenerate_covariance") or {}
    if degen:
        ln()
        ln(f"{num(degen.get('n'))} curves have a degenerate covariance "
           f"({degen.get('definition', '--')}), {num(degen.get('n_misclassified'))} "
           f"of them misclassified. On those curves the parameter-error gates "
           f"cannot discriminate: the error is infinite by construction.")
    ln()

    # ------------------------------------------------------------------
    # Error bars
    # ------------------------------------------------------------------
    ln("## Are the pipeline's error bars honest?")
    ln()
    params = (uncertainty or {}).get("parameters") or {}
    if not params:
        ln("Not measured in this run.")
        ln()
    else:
        ln(f"Measured on the `{uncertainty.get('stratum')}` stratum "
           f"(n = {num(uncertainty.get('n'))}). {uncertainty.get('note', '')} "
           f"{uncertainty.get('estimand_note', '')}")
        ln()
        ln("| Parameter | 68 % interval covers truth | 95 % | 95 % centred | "
           "median \\|z\\| | SE ratio |")
        ln("|---|---:|---:|---:|---:|---:|")
        for p in ("A", "mu", "lambda"):
            b = params.get(p) or {}
            ln(f"| {p} | {pct(b.get('coverage_68'))} | {pct(b.get('coverage_95'))} | "
               f"{pct(b.get('centered_coverage_95'))} | {fmt(b.get('median_abs_z'), 2)} | "
               f"{fmt(b.get('se_ratio'), 1)} |")
        ln()
        ln("A nominal 95 % interval should cover the truth about 95 % of the time. "
           "It does not. The reported standard errors come from the covariance of a "
           "least-squares fit that assumes independent residuals; the residuals are "
           "correlated in time, so the intervals are too narrow by roughly the SE "
           "ratio above. **Treat the pipeline's parameter standard errors as a "
           "relative precision indicator, not as a confidence interval.**")
        ln()
        terciles = (uncertainty or {}).get("median_abs_z_by_autocorr_tercile") or {}
        if terciles:
            ln("Split by residual autocorrelation (terciles of the same stratum), "
               "which is what drives it:")
            ln()
            ln("| Parameter | low tercile median \\|z\\| | mid | high | "
               "95 % coverage low / mid / high |")
            ln("|---|---:|---:|---:|---|")
            for p in ("A", "mu", "lambda"):
                b = terciles.get(p) or {}
                lo, mi, hi = b.get("low", {}), b.get("mid", {}), b.get("high", {})
                ln(f"| {p} | {fmt(lo.get('median_abs_z'), 2)} | "
                   f"{fmt(mi.get('median_abs_z'), 2)} | {fmt(hi.get('median_abs_z'), 2)} | "
                   f"{pct(lo.get('coverage_95'))} / {pct(mi.get('coverage_95'))} / "
                   f"{pct(hi.get('coverage_95'))} |")
            ln()

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------
    ln("## Truncation")
    ln()
    overall = (truncation or {}).get("overall") or {}
    if not truncation.get("available"):
        ln(f"Not available in this run: "
           f"{truncation.get('reason', 'the ground truth carries no window columns')}.")
        ln()
    else:
        ln(f"Does the fitted window land inside the window the truth says is the "
           f"primary growth phase? Measured on {num(overall.get('n'))} curves that "
           f"carry a window "
           f"(columns {', '.join(truncation.get('window_columns_present', []))}).")
        ln()
        ln("| Statistic | Value |")
        ln("|---|---:|")
        ln(f"| Fraction of cut points inside the true window | "
           f"{pct(overall.get('frac_inside_window'))} |")
        ln(f"| Median signed distance outside the window | "
           f"{fmt(overall.get('median_signed_distance_outside_h'), 2)} h |")
        ln(f"| 90th percentile of that distance | "
           f"{fmt(overall.get('p90_signed_distance_outside_h'), 2)} h |")
        ln(f"| Median \\|A\\| error against the window reference | "
           f"{fmt(overall.get('median_abs_A_window_error'), 4)} |")
        ln()
        ln(f"A negative median means the cut lands *before* the end of the true "
           f"window on a typical curve. Reference for the A comparison: "
           f"{overall.get('A_window_reference', '--')}. Per-scenario figures are in "
           f"`report.json` under `truncation.per_scenario`.")
        ln()

    # ------------------------------------------------------------------
    # Run-length robustness
    # ------------------------------------------------------------------
    ln("## Run-length robustness")
    ln()
    if robustness.get("available"):
        ln(f"Curves re-scored after truncating the trace at "
           f"{fmt(robustness.get('cutoff_h'), 1)} h, over curves at least "
           f"{fmt(robustness.get('min_duration_h'), 1)} h long, in "
           f"`{robustness.get('truncation_mode')}` mode.")
        ln()
        for key, value in robustness.items():
            if key in ("available", "reason"):
                continue
            ln(f"- {key}: {value}")
        ln()
    else:
        ln(f"Not computed in this run: {robustness.get('reason', '--')} "
           f"(cutoff {fmt(robustness.get('cutoff_h'), 1)} h over curves at least "
           f"{fmt(robustness.get('min_duration_h'), 1)} h long).")
        ln()

    # ------------------------------------------------------------------
    # Scenario descriptions
    # ------------------------------------------------------------------
    ln("## Scenario descriptions")
    ln()
    ln("Taken from `synthetic_data/src/curve_scenarios.py`, which is the only "
       "definition of what each scenario generates.")
    ln()
    scored = {r["scenario"] for r in rows}
    for cat, members in SCENARIO_CATEGORIES.items():
        ln(f"### {DISPLAY_CATEGORY.get(cat, cat)}")
        ln()
        ln(CATEGORY_NOTE.get(cat, ""))
        ln()
        for s in members:
            marker = "" if s in scored else " *(not scored in this run)*"
            rule = scenario_label_rule(s)
            rule_txt = f" [label rule: {rule}]" if rule and rule != "scenario" else ""
            ln(f"- **{s}**: {describe_scenario(s)}{rule_txt}{marker}")
        ln()

    # ------------------------------------------------------------------
    # What this run does not say
    # ------------------------------------------------------------------
    ln("## What this run does not say")
    ln()
    ln(f"- These numbers are for `{run.get('truncation_mode', 'unknown')}` "
       f"truncation only. A different truncation mode is a different benchmark and "
       f"gets its own run directory and its own rendering of this file.")
    ln("- Synthetic curves are not real plates. They are calibrated to measured "
       "quantities from the four included plates, but agreement here is a "
       "necessary condition for trusting the pipeline, not a sufficient one.")
    ln("- No threshold recommendation is derived from this file. Earlier versions "
       "recommended raising the R² and delta-OD acceptance thresholds on the "
       "strength of borderline false positives; those curves were mislabelled, "
       "their parameters were recoverable, and the recommendation would have "
       "tightened the real pipeline against curves it should accept. It has been "
       "withdrawn.")
    ln("- Parameter recovery on accepted curves is conditioned on the pipeline's "
       "own verdict. See the selection-bias figure above before reading it as "
       "estimator accuracy.")
    if run.get("ml_classify"):
        ln(f"- {IN_SAMPLE_BANNER}")
    ln()
    ln("---")
    ln()
    ln(f"Machine-readable source: `{rel_to_repo(data['report_path'])}` "
       f"(sha256 `{short_sha(data['report_sha256'], 16)}`). "
       f"Cross-run attribution: `docs/BENCHMARK_LEDGER.md`.")
    ln()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=("Render VALIDATION_SUMMARY.md for one benchmark run, from "
                     "that run's report.json."),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=("Benchmark run directory holding report.json (and the per_scenario / "
              "per_scenario_recovery / gate_attribution CSVs). Defaults to the "
              "first of validation/first_peak, validation/adaptive, benchmark/v2 "
              "that has one."),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Output path for the markdown (default: <run-dir>/VALIDATION_SUMMARY.md, "
              "or stdout when the run directory is a committed snapshot under "
              "synthetic_data/benchmark/).")
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help=("Deprecated. Path to a report.json; its parent directory is used as "
              "the run directory."),
    )
    for dead, what in [("--metrics-json", "validation_metrics.json"),
                       ("--ground-truth", "ground_truth.csv"),
                       ("--pipeline-results", "processing_results.csv")]:
        parser.add_argument(
            dead,
            type=str,
            default=None,
            help=(f"Deprecated and ignored. This renderer no longer reads "
                  f"{what}; every number comes from the run's report.json, so the "
                  f"markdown cannot disagree with it."),
        )
    return parser


def main():
    args = build_parser().parse_args()

    for dead, value in [("--metrics-json", args.metrics_json),
                        ("--ground-truth", args.ground_truth),
                        ("--pipeline-results", args.pipeline_results)]:
        if value:
            print(f"NOTE: {dead} is ignored; report.json is the only metric source.",
                  file=sys.stderr)

    run_dir = args.run_dir
    if run_dir is None and args.report_json:
        report_path = Path(args.report_json)
        if report_path.name != REPORT_NAME:
            print(f"NOTE: --report-json is deprecated; using {report_path.parent} "
                  f"as the run directory.", file=sys.stderr)
        run_dir = report_path.parent

    run_dir = resolve_run_dir(run_dir)
    data = load_run(run_dir)

    markdown = render(data)

    output_path = Path(args.output) if args.output else default_output_for(run_dir)
    report = data["report"]
    provenance = (f"  from   {rel_to_repo(data['report_path'])}\n"
                  f"  mode   {report.get('run', {}).get('truncation_mode')}  "
                  f"headline n={report.get('classification', {}).get('n')}  "
                  f"accuracy={report.get('classification', {}).get('accuracy')}")

    if output_path is None:
        # A committed snapshot: render to stdout rather than leaving an extra
        # untracked file inside a directory its MANIFEST.json enumerates.
        print(f"{rel_to_repo(run_dir)} is a committed snapshot, so the rendering "
              f"is not written into it.", file=sys.stderr)
        print(f"Markdown on stdout; pass --output PATH to save it.", file=sys.stderr)
        print(provenance, file=sys.stderr)
        sys.stdout.write(markdown)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Rendered {rel_to_repo(output_path)}")
    print(provenance)


if __name__ == "__main__":
    main()
