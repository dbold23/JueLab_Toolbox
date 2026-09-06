"""WP7 acceptance tests: matching, benchmark entry point, gate attribution.

Findings covered
----------------
M19  The ground-truth join was a silent inner join.  Renaming ``CURVE`` to
     ``WELL`` in the results published ``accuracy 0.0`` and exited 0; a results
     file truncated to 100 rows published ``accuracy 0.900`` over 21 % of the
     benchmark, also exit 0.  Both must now raise.
M14  ``run_pipeline(use_adaptive=False)`` appended no flag, so
     ``01_growth_curve_analysis.py`` resolved ``use_adaptive: true`` from
     ``scripts/config.yaml`` and the first-peak branch of the API was
     unreachable; ``--skip-run`` with a fresh ``--output-dir`` died with
     FileNotFoundError; the 1800 s timeout was shorter than the adaptive run
     the API defaulted to.
M17  No gate attribution, and "never fitted" merged with "fitted then
     rejected" into one count of 26.
M13  Parameter recovery conditioned on the pipeline's own verdict.
M18  No uncertainty-aware scoring.
M11  Neither JSON carried specificity, balanced accuracy or prevalence.

The binding acceptance test is
:func:`test_committed_report_is_reproduced`: the restructure must change no
classification number on the committed run.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.benchmark import (  # noqa: E402
    BenchmarkProvenanceError, run_benchmark,
)
from validation.comparison_report import ComparisonReport  # noqa: E402
from validation.matching import (  # noqa: E402
    BenchmarkMatchError, extract_curve_id, match_results,
)
from validation.pipeline_validator import (  # noqa: E402
    GATE_IDS, GateParseError, PipelineValidator, _gates_from_column,
    failure_analysis, gate_attribution, parse_reason_gates,
)

COMPREHENSIVE = SYNTH / 'output' / 'comprehensive_test'
GROUND_TRUTH = COMPREHENSIVE / 'test_data' / 'ground_truth.csv'
DATA_DIR = COMPREHENSIVE / 'test_data' / 'DATA'
RESULTS_CSV = COMPREHENSIVE / 'validation_latest' / 'processing_results.csv'
ANALYSIS_SCRIPT = REPO_ROOT / 'scripts' / '01_growth_curve_analysis.py'

# The committed benchmark, to full precision.  These are the numbers the
# restructure is required to leave untouched.
EXPECTED_ACCURACY = 0.8729166666666667
EXPECTED_PRECISION = 0.9011299435028248
EXPECTED_RECALL = 0.9246376811594202
EXPECTED_CONFUSION = {"tp": 319, "tn": 100, "fp": 35, "fn": 26}

pytestmark = pytest.mark.skipif(
    not (RESULTS_CSV.exists() and GROUND_TRUTH.exists()),
    reason="committed synthetic benchmark not present",
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def ground_truth():
    return pd.read_csv(GROUND_TRUTH)


@pytest.fixture(scope="module")
def results():
    return pd.read_csv(RESULTS_CSV)


@pytest.fixture(scope="module")
def matched(ground_truth, results):
    return match_results(ground_truth, results, strict=True)


@pytest.fixture(scope="module")
def validator():
    v = PipelineValidator(str(ANALYSIS_SCRIPT), str(DATA_DIR), str(GROUND_TRUTH))
    v.load_results(str(RESULTS_CSV))
    return v


@pytest.fixture(scope="module")
def committed_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("committed_benchmark")
    return run_benchmark(
        results_csv=RESULTS_CSV,
        ground_truth_csv=GROUND_TRUTH,
        out_dir=out,
        truncation_mode="first_peak",
        figures=False,
        verbose=False,
    )


@pytest.fixture(scope="module")
def two_curve_dir(tmp_path_factory):
    """A 2-curve DATA directory carved out of the committed benchmark.

    Decimated to every third sample (134 points over 99 h, a 45-minute reading
    interval the lab does use): the MCCV of the adaptive branch costs ~55 s on
    two 400-point curves and a fraction of that here, which keeps this file
    inside its time budget without weakening what it tests -- the point is that
    the two modes are both selectable and disagree.
    """
    root = tmp_path_factory.mktemp("two_curve")
    data = root / "DATA"
    data.mkdir()
    plate = pd.read_csv(DATA_DIR / "SYNTHETIC_400pts_99h_DATA.csv")
    columns = list(plate.columns[:3])
    plate[columns].iloc[::3].to_csv(
        data / "SYNTHETIC_134pts_99h_DATA.csv", index=False)

    ids = set()
    for column in columns[1:]:
        ids.add([p for p in column.split("_") if p.startswith("CURVE")][0])
    gt = pd.read_csv(GROUND_TRUTH)
    gt[gt["strain_name"].isin(ids)].to_csv(root / "ground_truth.csv", index=False)
    return root


# ------------------------------------------------------ acceptance test 1

def test_committed_report_is_reproduced(validator, committed_report):
    """The restructure must change no classification number (WP7 acceptance 1)."""
    metrics = validator.compute_classification_metrics()
    assert metrics.accuracy == EXPECTED_ACCURACY
    assert metrics.precision == EXPECTED_PRECISION
    assert metrics.recall == EXPECTED_RECALL
    assert metrics.true_positives == EXPECTED_CONFUSION["tp"]
    assert metrics.true_negatives == EXPECTED_CONFUSION["tn"]
    assert metrics.false_positives == EXPECTED_CONFUSION["fp"]
    assert metrics.false_negatives == EXPECTED_CONFUSION["fn"]

    block = committed_report["classification"]
    assert block["accuracy"] == EXPECTED_ACCURACY
    assert block["precision"] == EXPECTED_PRECISION
    assert block["recall"] == EXPECTED_RECALL
    assert block["confusion_matrix"] == {
        "true_positives": 319, "true_negatives": 100,
        "false_positives": 35, "false_negatives": 26,
    }
    # And the diagnostics the report never carried: specificity and balanced
    # accuracy sit four points below the published accuracy, purely prevalence.
    assert block["specificity"] == pytest.approx(100 / 135)
    assert block["balanced_accuracy"] == pytest.approx(0.8326892109500805)
    assert block["prevalence"] == pytest.approx(345 / 480)


def test_legacy_validation_report_keeps_its_schema(validator, tmp_path):
    """step 9 and validation_summary.py read this file; do not rename its keys."""
    path = tmp_path / "nested" / "validation_report.json"
    report = validator.generate_validation_report(str(path))
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["classification"]["accuracy"] == EXPECTED_ACCURACY
    cm = on_disk["classification"]["confusion_matrix"]
    assert set(cm) == {"true_positives", "true_negatives",
                       "false_positives", "false_negatives"}
    assert on_disk["parameter_recovery"]["A"]["n_samples"] == 319
    assert on_disk["summary"]["coverage"] == 1.0
    assert report["failure_analysis"]["failure_scenarios"]["sparse_data"] == 10


# ------------------------------------------------------ acceptance test 2

def test_match_raises_on_missing_and_duplicate_ids(ground_truth, results, matched):
    """A broken run must raise, not publish accuracy 0.0 and exit 0 (M19)."""
    # The committed pair matches completely.
    assert len(matched) == 480
    assert matched.attrs["coverage"] == 1.0
    assert matched.attrs["n_unmatched_ground_truth"] == 0
    assert matched.attrs["n_unmatched_results"] == 0

    # (a) CURVE renamed to WELL used to yield 'Matched 0 curves' and accuracy 0.
    renamed = results.copy()
    renamed["strain"] = renamed["strain"].str.replace("CURVE", "WELL", regex=False)
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(ground_truth, renamed)
    assert len(excinfo.value.unmatched_ground_truth) == 480

    # (b) truncation to 100 rows used to yield accuracy 0.900 over 21 % of the set.
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(ground_truth, results.head(100))
    assert len(excinfo.value.unmatched_ground_truth) == 380
    assert "380" in str(excinfo.value)

    # (c) a duplicated id.
    duplicated = pd.concat([results, results.head(1)], ignore_index=True)
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(ground_truth, duplicated)
    assert excinfo.value.duplicate_ids

    # (d) a blank verdict raises under strict and scores False under strict=False.
    blank = results.copy()
    blank["is_good"] = blank["is_good"].astype(object)
    blank.loc[0, "is_good"] = None
    blank_id = extract_curve_id(blank.loc[0, "strain"])
    with pytest.raises(BenchmarkMatchError) as excinfo:
        match_results(ground_truth, blank)
    assert excinfo.value.undecided_ids == [blank_id]

    lenient = match_results(ground_truth, blank, strict=False)
    assert lenient.attrs["undecided_ids"] == [blank_id]
    row = lenient[lenient["curve_id"] == blank_id]
    assert bool(row["pred_is_good"].iloc[0]) is False


def test_partial_run_is_labelled_in_the_report(ground_truth, results, tmp_path):
    """A --no-strict run still carries its own incompleteness in report.json."""
    partial = results.head(100)
    partial_csv = tmp_path / "processing_results.csv"
    partial.to_csv(partial_csv, index=False)
    report = run_benchmark(partial_csv, GROUND_TRUTH, tmp_path / "out",
                           truncation_mode="first_peak", strict=False,
                           figures=False, verbose=False)
    assert report["summary"]["n_unmatched_ground_truth"] == 380
    assert report["summary"]["coverage"] == pytest.approx(100 / 480)
    assert report["summary"]["strict"] is False


# ------------------------------------------------------ acceptance test 3

def test_first_peak_branch_is_reachable(two_curve_dir, tmp_path):
    """Both truncation modes are selectable and produce different cuts (M14)."""
    cuts = {}
    for mode in ("first_peak", "adaptive"):
        validator = PipelineValidator(
            str(ANALYSIS_SCRIPT), str(two_curve_dir / "DATA"),
            str(two_curve_dir / "ground_truth.csv"))
        path = validator.run_pipeline(str(tmp_path / mode), truncation_mode=mode,
                                      verbose=False)
        assert path is not None and path.exists()
        frame = pd.read_csv(path)
        assert len(frame) == 2
        cuts[mode] = sorted(frame["truncation_time"].tolist())

        report = run_benchmark(path, two_curve_dir / "ground_truth.csv",
                               tmp_path / f"bench_{mode}", truncation_mode=mode,
                               figures=False, verbose=False)
        # When the pipeline writes a run_settings.json sidecar (WP9) the mode
        # comes from it; build_run_block refuses a report whose header would
        # contradict it, so agreement here is itself the cross-check.
        assert report["run"]["truncation_mode"] == mode
        assert report["run"]["provenance"] in ("sidecar", "declared")

    # Adaptive keeps more of the curve than first-peak on these two.
    assert cuts["first_peak"] != cuts["adaptive"]
    assert min(cuts["adaptive"]) > min(cuts["first_peak"])


def test_run_pipeline_requires_a_truncation_mode():
    validator = PipelineValidator(str(ANALYSIS_SCRIPT), str(DATA_DIR), str(GROUND_TRUTH))
    with pytest.raises(TypeError):
        validator.run_pipeline("out")           # no mode at all
    with pytest.raises(ValueError):
        validator.run_pipeline("out", truncation_mode="whatever")


def test_timeout_is_a_constructor_argument():
    """1800 s was shorter than the adaptive run the API defaulted to (M14)."""
    default = PipelineValidator(str(ANALYSIS_SCRIPT), str(DATA_DIR), str(GROUND_TRUTH))
    assert default.timeout is None
    explicit = PipelineValidator(str(ANALYSIS_SCRIPT), str(DATA_DIR),
                                 str(GROUND_TRUTH), timeout=60)
    assert explicit.timeout == 60


# ------------------------------------------------------ acceptance test 4

def test_report_schema(committed_report):
    """Key presence and types only -- never pin coverage or accuracy here."""
    report = committed_report

    run = report["run"]
    for key in ("truncation_mode", "provenance", "seed", "git_sha",
                "dataset_sha256", "n_curves", "n_scenarios", "wall_clock_s",
                "timestamp_utc", "ml_classify", "config_sha256"):
        assert key in run, key
    assert run["truncation_mode"] in ("first_peak", "adaptive", "global_max")
    assert isinstance(run["n_curves"], int)
    assert isinstance(run["wall_clock_s"], float)

    cls = report["classification"]
    for key in ("balanced_accuracy", "sensitivity", "specificity", "accuracy",
                "prevalence", "precision", "recall", "npv", "mcc", "f1_score",
                "macro_accuracy", "lab_weighted_accuracy", "per_scenario",
                "per_delta_od_band", "core32", "confusion_matrix"):
        assert key in cls, key
    # balanced accuracy leads the block; accuracy sits beside its prevalence.
    order = list(cls)
    assert order[:5] == ["balanced_accuracy", "sensitivity", "specificity",
                         "accuracy", "prevalence"]
    assert isinstance(cls["per_scenario"], dict) and cls["per_scenario"]
    for entry in cls["per_scenario"].values():
        assert {"n", "accuracy", "false_positives", "false_negatives"} <= set(entry)
    assert len(cls["per_delta_od_band"]) == 6
    assert "accuracy" in cls["core32"]

    recovery = report["parameter_recovery"]
    for stratum in ("accepted_good", "all_fitted_good", "rejected_good", "accepted_bad"):
        assert stratum in recovery, stratum
        assert isinstance(recovery[stratum]["n"], int)
        for parameter in ("A", "mu", "lambda"):
            entry = recovery[stratum][parameter]
            assert {"n", "rmse", "r_squared_across_scenarios",
                    "tolerance_pass_rate", "recovery_status"} <= set(entry)
    assert "n_excluded_degenerate_gt" in recovery
    assert "selection_bias_delta" in recovery

    uncertainty = report["uncertainty"]
    assert set(uncertainty["parameters"]) == {"A", "mu", "lambda"}
    for entry in uncertainty["parameters"].values():
        assert {"n", "n_finite_se", "n_non_finite_se", "coverage_68",
                "coverage_95", "median_abs_z", "median_signed_error",
                "centered_coverage_95", "se_ratio"} <= set(entry)
    assert uncertainty["median_abs_z_by_autocorr_tercile"]

    assert "truncation" in report
    assert "a_not_identifiable" in report["truncation"]
    assert "preprocessing_blind" in report["truncation"]
    assert "by_gate" in report["failure_analysis"]
    assert "coverage" in report["summary"]
    assert report["schema_version"] == "2"

    outputs = report["outputs"]
    for name in ("per_curve", "per_scenario", "gate_attribution",
                 "per_scenario_recovery"):
        assert Path(outputs[name]).exists(), name
    # SUMMARY.md is WP8's to render, not this module's.
    assert not (Path(outputs["report"]).parent / "SUMMARY.md").exists()


def test_report_json_has_no_nan_literals(committed_report, tmp_path):
    """NaN/Infinity are not JSON; a strict reader must be able to load it."""
    path = Path(committed_report["outputs"]["report"])
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))


# ------------------------------------------------------ acceptance test 5

def test_gate_attribution_fixture(matched):
    """Which gate fired on each miss, on the committed run (M17)."""
    attribution = gate_attribution(matched, strict=True)
    analysis = failure_analysis(matched, attribution)

    # 12 expected-GOOD curves were never fitted; 14 were fitted and then gated.
    # The old report merged both into one count of 26.
    assert analysis["n_false_negatives_unfitted"] == 12
    assert analysis["n_false_negatives_gated"] == 14
    assert analysis["n_false_negatives"] == 26
    assert analysis["false_negative_scenarios_unfitted"] == {
        "sparse_data": 10, "fast_growth": 2}

    # All 35 false positives are simply 'GOOD: Fit quality passed'.
    assert analysis["n_false_positives"] == 35
    assert analysis["n_false_positives_with_no_failed_gate"] == 35

    logistic = analysis["by_gate"]["logistic_generated"]
    assert logistic["r_squared"] == 6
    assert logistic["residual_autocorr"] == 5
    assert logistic["mu_err_pct"] == 1
    # 9 curves trip the A-uncertainty gate, of which 8 are a singular
    # covariance (a_err_pct == inf) rather than a near-miss; the remaining one
    # is a genuine 23.5 % against the 20 % threshold.
    assert logistic["a_err_pct"] == 9
    logistic_rows = attribution[(attribution["scenario"] == "logistic_generated")
                                & (~attribution["correct"])]
    assert int(logistic_rows["degenerate_covariance"].sum()) == 8
    finite = logistic_rows[logistic_rows["gate_a_err_pct"]
                           & ~logistic_rows["degenerate_covariance"]]
    assert len(finite) == 1
    assert finite["observed_a_err_pct"].iloc[0] == pytest.approx(23.5, abs=0.1)

    # low_A loses exactly one curve, to R2 0.940 against the 0.95 gate.
    low_a = attribution[(attribution["scenario"] == "low_A") & (~attribution["correct"])]
    assert len(low_a) == 1
    assert low_a["failed_gates"].iloc[0] == "r_squared"
    assert low_a["observed_r_squared"].iloc[0] == pytest.approx(0.940)
    assert low_a["margin_r_squared"].iloc[0] == pytest.approx(0.010, abs=1e-6)

    # sparse_data is 10/10 'too few points after truncation'.
    sparse = attribution[(attribution["scenario"] == "sparse_data")
                         & (~attribution["correct"])]
    assert int(sparse["gate_too_few_points"].sum()) == 10


def test_gate_attribution_prefers_the_failed_gates_column(matched):
    """WP9's machine-readable column wins over the regex when present."""
    with_column = matched.copy()
    with_column["failed_gates"] = ""
    with_column.loc[with_column.index[0], "failed_gates"] = "r_squared;snr"
    attribution = gate_attribution(with_column, strict=True)
    assert set(attribution["gate_source"]) == {"failed_gates"}
    assert attribution["failed_gates"].iloc[0] == "r_squared;snr"
    assert bool(attribution["gate_r_squared"].iloc[0])
    assert bool(attribution["gate_snr"].iloc[0])
    # Everything else declared passing by the column is reported as passing.
    assert int(attribution["n_failed_gates"].sum()) == 2


def test_regex_parser_agrees_with_the_pipelines_failed_gates_column(tmp_path):
    """WP7's fallback parser and WP9's column must name the same gates.

    The regex exists for CSVs written before the pipeline gained the column;
    if the two ever disagree, one of the two attributions is wrong and the
    report silently changes meaning depending on which run it scored.
    """
    data = tmp_path / "DATA"
    data.mkdir()
    gt = pd.read_csv(GROUND_TRUTH)
    # A spread of verdicts: no-growth, degenerate fits, noise and clean growth.
    scenarios = ["erratic", "flat_no_growth", "fit_failure", "logistic_generated",
                 "high_noise", "standard"]
    ids = set(gt[gt["scenario"].isin(scenarios)]["strain_name"])
    plate = pd.read_csv(DATA_DIR / "SYNTHETIC_400pts_99h_DATA.csv")
    keep = [plate.columns[0]] + [c for c in plate.columns[1:]
                                 if extract_curve_id(c) in ids]
    plate[keep].to_csv(data / "SYNTHETIC_400pts_99h_DATA.csv", index=False)
    # sparse_data lives on its own plate and is the 'too few points' exit.
    shutil.copy(DATA_DIR / "SYNTHETIC_80pts_19h_DATA.csv", data)

    validator = PipelineValidator(str(ANALYSIS_SCRIPT), str(data),
                                  str(GROUND_TRUTH), strict=False)
    path = validator.run_pipeline(str(tmp_path / "out"),
                                  truncation_mode="first_peak", verbose=False)
    assert path is not None
    frame = pd.read_csv(path)
    if "failed_gates" not in frame.columns:
        pytest.skip("pipeline does not emit failed_gates yet (WP9)")

    seen = set()
    for _, row in frame.iterrows():
        from_column = _gates_from_column(row.get("failed_gates"))
        if from_column is None:
            from_column = [] if bool(row["is_good"]) else None
        from_regex = [g["gate"] for g in
                      parse_reason_gates(row["classification_reason"], strict=True)]
        assert from_column is not None, row["classification_reason"]
        assert set(from_column) == set(from_regex), row["classification_reason"]
        seen.update(from_regex)
    # The comparison is only worth anything if several gates actually fired.
    assert len(seen) >= 5
    assert seen <= set(GATE_IDS)


def test_every_committed_reason_maps_to_a_known_gate(results):
    """A reworded pipeline message must break the suite, not empty the table."""
    for reason in results["classification_reason"].dropna().unique():
        gates = parse_reason_gates(reason, strict=True)
        if str(reason).startswith("BAD"):
            assert gates, reason
        for entry in gates:
            assert entry["gate"] in GATE_IDS

    with pytest.raises(GateParseError):
        parse_reason_gates("BAD: Curve smells wrong: 3 > 2", strict=True)
    assert parse_reason_gates("BAD: Curve smells wrong", strict=False) == []


def test_gate_ids_match_the_pipeline():
    """The canonical ids are tecan_growth/analyze.py's, plus the two pre-fit exits."""
    from tecan_growth import analyze as analyze_module

    source = Path(analyze_module.__file__).read_text()
    for gate in ("prefit_snr", "fit_success", "r_squared", "a_err_pct",
                 "mu_err_pct", "snr", "delta_od_ci", "delta_od",
                 "amplitude_vs_noise", "monotone", "residual_autocorr",
                 "ml_p_good"):
        assert f'"{gate}"' in source, gate
        assert gate in GATE_IDS
    assert "too_few_points" in GATE_IDS
    assert "prefit_rejected" in GATE_IDS


# ------------------------------------------------------ acceptance test 6

def test_skip_run_creates_output_dir(tmp_path):
    """--skip-run with a nonexistent --output-dir used to raise FileNotFoundError."""
    out_dir = tmp_path / "does" / "not" / "exist"
    assert not out_dir.exists()
    proc = subprocess.run(
        [sys.executable, str(SYNTH / "validation" / "pipeline_validator.py"),
         "--pipeline", str(ANALYSIS_SCRIPT),
         "--data-dir", str(DATA_DIR),
         "--ground-truth", str(GROUND_TRUTH),
         "--output-dir", str(out_dir),
         "--skip-run", "--results-csv", str(RESULTS_CSV)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["classification"]["accuracy"] == EXPECTED_ACCURACY


def test_cli_exits_non_zero_on_a_broken_match(tmp_path):
    """Step 9 must fail loudly rather than publish accuracy 0.0 (M19)."""
    renamed = pd.read_csv(RESULTS_CSV)
    renamed["strain"] = renamed["strain"].str.replace("CURVE", "WELL", regex=False)
    broken = tmp_path / "processing_results.csv"
    renamed.to_csv(broken, index=False)

    proc = subprocess.run(
        [sys.executable, str(SYNTH / "validation" / "benchmark.py"),
         "--results-csv", str(broken),
         "--ground-truth", str(GROUND_TRUTH),
         "--out-dir", str(tmp_path / "out"),
         "--truncation-mode", "first_peak", "--no-figures"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert proc.returncode != 0
    assert not (tmp_path / "out" / "report.json").exists()


def test_benchmark_cli_writes_the_run_directory(tmp_path):
    out_dir = tmp_path / "bench"
    proc = subprocess.run(
        [sys.executable, str(SYNTH / "validation" / "benchmark.py"),
         "--results-csv", str(RESULTS_CSV),
         "--ground-truth", str(GROUND_TRUTH),
         "--out-dir", str(out_dir),
         "--truncation-mode", "first_peak", "--no-figures"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("report.json", "per_curve.csv", "per_scenario.csv",
                 "gate_attribution.csv", "per_scenario_recovery.csv"):
        assert (out_dir / name).exists(), name
    # The headline can never be quoted without its provenance.
    assert "first_peak truncation" in proc.stdout


# ------------------------------------------------------------- provenance

def test_report_is_refused_without_provenance(tmp_path):
    """No sidecar and no declared mode -> no report (M14)."""
    with pytest.raises(BenchmarkProvenanceError):
        run_benchmark(RESULTS_CSV, GROUND_TRUTH, tmp_path / "out",
                      figures=False, verbose=False)
    assert not (tmp_path / "out" / "report.json").exists()


def test_dataset_digest_is_never_the_ground_truth_file(committed_report):
    """No dataset, no digest -- and the headline says so (fix round 2).

    ``run['dataset_sha256']`` used to fall back to hashing ``ground_truth.csv``
    when there was neither a WP9 sidecar nor ``--data-dir``, so the same
    benchmark of the same data published ``dataset 596d8321eb4a`` (declared
    mode) or ``dataset 2ae090550a58`` (sidecar mode) for what the header calls
    one field.
    """
    from validation.benchmark import headline

    run = committed_report["run"]
    assert run["dataset_sha256"] is None
    assert run["dataset_sha256_source"] is None
    assert run["ground_truth_sha256"] is not None
    assert run["dataset_sha256"] != run["ground_truth_sha256"]

    line = headline(committed_report)
    assert "dataset unknown" in line
    assert run["ground_truth_sha256"][:12] not in line


def test_data_dir_digest_matches_the_pipelines_own(tmp_path):
    """--data-dir reproduces WP9's sidecar digest byte for byte."""
    import hashlib

    from validation.benchmark import dataset_sha256

    expected = hashlib.sha256()
    for path in sorted(DATA_DIR.glob("*_DATA.csv"), key=lambda q: q.name):
        expected.update(path.read_bytes())
    assert dataset_sha256(DATA_DIR) == expected.hexdigest()
    assert dataset_sha256(tmp_path) is None          # no *_DATA.csv, no digest

    report = run_benchmark(RESULTS_CSV, GROUND_TRUTH, tmp_path / "out",
                           truncation_mode="first_peak", data_dir=DATA_DIR,
                           figures=False, verbose=False)
    run = report["run"]
    assert run["dataset_sha256"] == expected.hexdigest()
    assert run["dataset_sha256_source"] == "data_dir"
    assert run["data_dir"] == str(DATA_DIR)


def test_a_contradicting_dataset_digest_is_refused(tmp_path):
    """A sidecar taken on other data than --data-dir cannot be published."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(RESULTS_CSV, run_dir / "processing_results.csv")
    (run_dir / "run_settings.json").write_text(json.dumps({
        "truncation_mode": "first_peak", "dataset_sha256": "deadbeef",
    }))
    with pytest.raises(BenchmarkProvenanceError):
        run_benchmark(run_dir / "processing_results.csv", GROUND_TRUTH,
                      tmp_path / "out", data_dir=DATA_DIR,
                      figures=False, verbose=False)


def test_run_settings_sidecar_is_read(tmp_path):
    """WP9's sidecar supplies the mode, and a contradicting flag is refused."""
    import shutil

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(RESULTS_CSV, run_dir / "processing_results.csv")
    (run_dir / "run_settings.json").write_text(json.dumps({
        "truncation_mode": "first_peak", "use_adaptive": False,
        "use_first_peak": True, "ml_classify": False,
        "config_sha256": "abc123", "dataset_sha256": "deadbeef",
        "seed": 42, "git_sha": "cafebabe",
    }))

    report = run_benchmark(run_dir / "processing_results.csv", GROUND_TRUTH,
                           tmp_path / "out", figures=False, verbose=False)
    assert report["run"]["provenance"] == "sidecar"
    assert report["run"]["truncation_mode"] == "first_peak"
    assert report["run"]["dataset_sha256"] == "deadbeef"
    assert report["run"]["config_sha256"] == "abc123"

    with pytest.raises(BenchmarkProvenanceError):
        run_benchmark(run_dir / "processing_results.csv", GROUND_TRUTH,
                      tmp_path / "conflict", truncation_mode="adaptive",
                      figures=False, verbose=False)


# ------------------------------------------------------------ diagnostics

def test_parameter_recovery_reports_the_selection_bias(committed_report):
    """A-RMSE 0.1418 over accepted GOOD vs 0.3674 over all fitted GOOD (M13)."""
    recovery = committed_report["parameter_recovery"]
    assert recovery["accepted_good"]["n"] == 319
    assert recovery["accepted_good"]["A"]["rmse"] == pytest.approx(0.14184640742757926)
    assert recovery["all_fitted_good"]["n"] == 333
    assert recovery["all_fitted_good"]["A"]["rmse"] == pytest.approx(0.3674327127314099)
    assert recovery["rejected_good"]["n"] == 14
    assert recovery["rejected_good"]["A"]["rmse"] == pytest.approx(1.659148990328731)
    assert recovery["accepted_bad"]["n"] == 35
    assert recovery["accepted_bad"]["A"]["median_rel_error"] == pytest.approx(
        0.08859266754534754)
    assert recovery["selection_bias_delta"]["ratio"] > 2.5

    # flat_no_growth and erratic have no parameters to recover.
    assert recovery["n_excluded_degenerate_gt"] == 30

    # Never an R2 for a stratum with n < 10 (there is none here, but the rule
    # must be in force).
    for stratum in recovery["accepted_good"], recovery["rejected_good"]:
        for parameter in ("A", "mu", "lambda"):
            entry = stratum[parameter]
            if entry["n"] < 10:
                assert entry["r_squared_across_scenarios"] is None

    # The pooled R2 is renamed: 0.825 for A coexists with a 2.3 % median error.
    a_stats = recovery["accepted_good"]["A"]
    assert "r_squared" not in a_stats
    assert a_stats["r_squared_across_scenarios"] == pytest.approx(0.8250662899246053)
    assert a_stats["median_rel_error"] == pytest.approx(0.0229, abs=1e-3)


def test_per_scenario_recovery_flags_the_broken_converters(committed_report):
    """baranyi_generated and logistic_generated recover 0.000 of their curves."""
    frame = pd.read_csv(committed_report["outputs"]["per_scenario_recovery"])
    joint = frame[(frame["stratum"] == "all_fitted_good") & (frame["parameter"] == "all")]
    rates = dict(zip(joint["scenario"], joint["pass_rate"]))
    assert rates["baranyi_generated"] == 0.0
    assert rates["logistic_generated"] == 0.0
    assert rates["standard"] > 0.5
    degenerate = frame[frame["recovery_status"] == "degenerate"]
    assert set(degenerate["scenario"]) >= {"baranyi_generated", "logistic_generated"}


def test_uncertainty_coverage_is_reported(committed_report):
    """Nominal 95 % intervals cover the truth 43-76 % of the time (M18)."""
    uncertainty = committed_report["uncertainty"]
    a = uncertainty["parameters"]["A"]
    mu = uncertainty["parameters"]["mu"]
    lam = uncertainty["parameters"]["lambda"]

    assert a["n_finite_se"] == 319
    assert lam["n_finite_se"] == 316
    assert lam["n_non_finite_se"] == 3
    assert a["coverage_95"] == pytest.approx(0.4671, abs=1e-3)
    assert mu["coverage_95"] == pytest.approx(0.7586, abs=1e-3)
    assert lam["coverage_95"] == pytest.approx(0.4304, abs=1e-3)

    # Most of the A/lambda miss is a window-vs-whole-curve estimand mismatch,
    # not understated spread: removing the median signed error recovers it.
    assert a["centered_coverage_95"] == pytest.approx(0.6865, abs=1e-3)
    assert lam["centered_coverage_95"] == pytest.approx(0.6297, abs=1e-3)
    assert a["centered_coverage_95"] > a["coverage_95"]
    assert a["se_ratio"] > 1.0

    terciles = uncertainty["median_abs_z_by_autocorr_tercile"]["A"]
    assert terciles["high"]["median_abs_z"] > terciles["low"]["median_abs_z"]


def test_truncation_block_declares_itself_unavailable_on_v1(committed_report):
    """v1 ground truth carries no growth window; say so, do not invent one."""
    truncation = committed_report["truncation"]
    assert truncation["available"] is False
    assert "t_primary_95" in truncation["reason"]
    assert truncation["a_not_identifiable"]["n"] == 0
    assert truncation["preprocessing_blind"]["n"] == 0


def test_v2_ground_truth_columns_are_tolerated(ground_truth, results, tmp_path):
    """The scorer must accept both v1 and v2 ground truth (interface B)."""
    v2 = ground_truth.copy()
    v2["scoring_mode"] = "standard"
    v2.loc[v2.index[:5], "scoring_mode"] = "preprocessing_blind"
    v2["requested_A"] = v2["true_A"] * 1.10
    v2["requested_mu"] = v2["true_mu"]
    v2["requested_lambda"] = v2["true_lambda"]
    v2["t_primary_95"] = 20.0
    v2["t_second_start"] = np.nan
    v2["true_window_end_h"] = 30.0
    v2_csv = tmp_path / "ground_truth_v2.csv"
    v2.to_csv(v2_csv, index=False)

    report = run_benchmark(RESULTS_CSV, v2_csv, tmp_path / "out",
                           truncation_mode="first_peak", figures=False,
                           verbose=False)
    # The five preprocessing_blind curves leave the headline and get their own
    # block; the rest of the benchmark is unchanged.
    assert report["classification"]["n"] == 475
    assert report["truncation"]["preprocessing_blind"]["n"] == 5
    assert report["classification"]["excluded_from_headline"]["n"] == 5
    assert report["truncation"]["available"] is True
    assert report["truncation"]["n_with_band"] == 480
    assert report["parameter_recovery"]["legacy_nominal"]["accepted_good"]["n"] == 319
    assert report["parameter_recovery"]["true_source"].startswith("nominal_true")


# ------------------------------------------------- comparison_report fixes

def test_comparison_report_pairs_are_row_aligned():
    """x and y used to be dropna'd independently, then sliced with iloc[:n]."""
    frame = pd.DataFrame({
        "true_A": [1.0, np.nan, 3.0, 4.0],
        "gompertz_a": [1.1, 2.2, np.nan, 4.4],
    })
    pairs = ComparisonReport._pairs(frame, "true_A", "gompertz_a")
    assert list(pairs.index) == [0, 3]
    assert list(pairs["true_A"]) == [1.0, 4.0]
    assert list(pairs["gompertz_a"]) == [1.1, 4.4]


def test_comparison_report_uses_the_shared_match(matched, ground_truth, results, tmp_path):
    report = ComparisonReport(ground_truth, results, matched=matched)
    assert report.matched is matched
    outputs = report.generate_full_report(str(tmp_path), prefix="v")
    assert set(outputs) == {"confusion_matrix", "parameter_scatter",
                            "error_distributions", "scenario_performance"}
    for path in outputs.values():
        assert Path(path).exists()


def test_match_is_computed_once(validator, monkeypatch):
    """One report run used to print 'Matched 480' seven times."""
    import validation.pipeline_validator as module

    calls = {"n": 0}
    original = module.match_results

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "match_results", counting)
    fresh = PipelineValidator(str(ANALYSIS_SCRIPT), str(DATA_DIR), str(GROUND_TRUTH))
    fresh.load_results(str(RESULTS_CSV))
    fresh.compute_classification_metrics()
    fresh.compute_parameter_recovery()
    fresh.identify_failure_cases()
    fresh.compute_uncertainty()
    assert calls["n"] == 1
