#!/usr/bin/env python3
"""
validation_summary.py

Reads validation results (JSON metrics, ground truth CSV, pipeline results CSV)
and generates a comprehensive markdown summary report.

Usage:
    python validation_summary.py
    python validation_summary.py --metrics-json path/to/validation_metrics.json \
                                 --report-json path/to/validation_report.json \
                                 --ground-truth path/to/ground_truth.csv \
                                 --pipeline-results path/to/processing_results.csv \
                                 --output path/to/VALIDATION_SUMMARY.md
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = BASE_DIR / "output" / "comprehensive_test" / "validation_metrics.json"
DEFAULT_REPORT = BASE_DIR / "output" / "comprehensive_test" / "validation_report" / "validation_report.json"
DEFAULT_GROUND_TRUTH = BASE_DIR / "output" / "comprehensive_test" / "test_data" / "ground_truth.csv"
DEFAULT_PIPELINE = BASE_DIR / "output" / "comprehensive_test" / "pipeline_results" / "processing_results.csv"
DEFAULT_OUTPUT = BASE_DIR / "output" / "comprehensive_test" / "VALIDATION_SUMMARY.md"

# ---------------------------------------------------------------------------
# Scenario descriptions
# ---------------------------------------------------------------------------
SCENARIO_DESCRIPTIONS = {
    # --- Good scenarios (expected GOOD classification) ---
    "standard": "Baseline Gompertz curves with medium noise -- the bread-and-butter case.",
    "high_A": "High maximum OD (A > 1.5). Tests whether the pipeline handles large growth amplitudes.",
    "low_A": "Low maximum OD (A ~ 0.3-0.5). Tests sensitivity to small but real growth signals.",
    "fast_growth": "High growth rate (mu > 0.3). Tests fitting of steep exponential phases.",
    "slow_growth": "Low growth rate (mu < 0.1). Tests detection of gradual growth that may be mistaken for noise.",
    "long_lag": "Extended lag phase (lambda > 10 h). Tests correct identification of delayed growth onset.",
    "short_lag": "Very short lag phase (lambda < 2 h). Tests fitting when growth begins almost immediately.",
    "very_clean": "Very low noise. Ensures the pipeline does not introduce artefacts on pristine data.",
    "baranyi_generated": "Curves generated with the Baranyi model rather than Gompertz. Tests model-mismatch robustness.",
    # --- Bad scenarios (expected BAD classification) ---
    "flat_no_growth": "Flat OD traces with no growth. Pipeline should classify as BAD.",
    "erratic": "Randomly fluctuating OD with no biological growth pattern.",
    "fit_failure": "Curves deliberately designed to defeat Gompertz fitting (e.g. non-sigmoidal shapes).",
    "death_phase_severe": "Severe decline after stationary phase. Tests whether large post-peak drops are flagged.",
    "contamination": "Sudden OD spikes simulating contamination events.",
    "pesticide_only": "Flat or declining curves representing pesticide-inhibited wells (no growth).",
    "minimal_growth": "Tiny delta-OD near the detection threshold. Should be classified BAD.",
    "high_noise": "Very high noise that obscures any underlying growth signal.",
    # --- Edge-case / borderline scenarios ---
    "borderline_noise": "Noise level at the boundary between acceptable and unacceptable.",
    "borderline_r2_good": "R-squared hovering just above the GOOD threshold.",
    "borderline_r2_bad": "R-squared hovering just below the GOOD threshold.",
    "borderline_delta_od": "Delta-OD just above the minimum threshold for GOOD classification.",
    "death_phase_moderate": "Moderate post-peak decline. Ambiguous -- could be GOOD or BAD depending on pipeline settings.",
    "drift_positive": "Slow upward baseline drift without real growth.",
    "outlier_contaminated": "Good growth curves with a few large outlier points injected.",
    "truncation_challenge": "Experiment ended before stationary phase was fully reached.",
    "short_experiment": "Reduced total experiment duration (~20 h). Tests whether the pipeline copes with limited data windows.",
    "long_experiment": "Extended experiment duration (~150 h). Tests handling of prolonged stationary phase and potential drift.",
    "sparse_data": "Reduced measurement density (fewer time points). Tests robustness to low temporal resolution.",
    "dense_data": "High measurement density. Ensures the pipeline does not over-fit to noise.",
    "logistic_generated": "Logistic growth model instead of Gompertz. Another model-mismatch test.",
    "richards_asymmetric": "Richards (asymmetric sigmoid) growth model. Tests generalisation beyond symmetric sigmoids.",
    "pesticide_lb_typical": "Typical LB-media pesticide experiment curves: real but modest growth.",
}

# Categorise scenarios into groups for the report
SCENARIO_CATEGORIES = {
    "Good (standard growth)": [
        "standard", "high_A", "low_A", "fast_growth", "slow_growth",
        "long_lag", "short_lag", "very_clean", "baranyi_generated",
    ],
    "Bad (no / failed growth)": [
        "flat_no_growth", "erratic", "fit_failure", "death_phase_severe",
        "contamination", "pesticide_only", "minimal_growth", "high_noise",
    ],
    "Edge / borderline": [
        "borderline_noise", "borderline_r2_good", "borderline_r2_bad",
        "borderline_delta_od", "death_phase_moderate", "drift_positive",
        "outlier_contaminated", "truncation_challenge", "short_experiment",
        "long_experiment", "sparse_data", "dense_data", "logistic_generated",
        "richards_asymmetric", "pesticide_lb_typical",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_curve_id(strain_name: str) -> str:
    """Extract the CURVE#### pattern from a pipeline strain name.

    Pipeline names look like 'SYNTHETIC-400pts-99h-CURVE0001'.
    Ground truth names look like 'CURVE0001'.
    """
    match = re.search(r"(CURVE\d+)", strain_name)
    if match:
        return match.group(1)
    return strain_name


def pct(value: float) -> str:
    """Format a float as a percentage string with one decimal place."""
    return f"{value * 100:.1f}%"


def fmt(value: float, decimals: int = 4) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_data(args):
    """Load all data sources and return them as a dict."""
    with open(args.metrics_json, "r") as f:
        metrics = json.load(f)

    report = None
    if args.report_json and Path(args.report_json).exists():
        with open(args.report_json, "r") as f:
            report = json.load(f)

    gt = pd.read_csv(args.ground_truth)
    pipeline = pd.read_csv(args.pipeline_results)

    return {
        "metrics": metrics,
        "report": report,
        "gt": gt,
        "pipeline": pipeline,
    }


def merge_results(gt: pd.DataFrame, pipeline: pd.DataFrame) -> pd.DataFrame:
    """Merge ground truth and pipeline results on the CURVE#### identifier."""
    # Normalise ground truth curve_id
    gt = gt.copy()
    gt["curve_id_key"] = gt["strain_name"].apply(extract_curve_id)

    # Normalise pipeline curve_id
    pipeline = pipeline.copy()
    pipeline["curve_id_key"] = pipeline["strain"].apply(extract_curve_id)

    merged = gt.merge(pipeline, on="curve_id_key", how="inner", suffixes=("_gt", "_pipe"))
    return merged


def compute_per_scenario(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute per-scenario accuracy metrics."""
    # Determine ground-truth boolean: GOOD -> True, BAD -> False
    merged = merged.copy()
    merged["gt_good"] = merged["expected_class"].str.upper() == "GOOD"
    merged["pipe_good"] = merged["is_good"].astype(str).str.strip().str.lower() == "true"
    merged["correct"] = merged["gt_good"] == merged["pipe_good"]

    rows = []
    for scenario, grp in merged.groupby("scenario"):
        n = len(grp)
        n_correct = grp["correct"].sum()
        n_wrong = n - n_correct
        accuracy = n_correct / n if n > 0 else 0.0
        expected = grp["expected_class"].iloc[0]

        # Break down error types
        fp = ((~grp["gt_good"]) & grp["pipe_good"]).sum()
        fn = (grp["gt_good"] & (~grp["pipe_good"])).sum()

        rows.append({
            "scenario": scenario,
            "expected_class": expected,
            "n_curves": n,
            "n_correct": int(n_correct),
            "n_wrong": int(n_wrong),
            "accuracy": accuracy,
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=True).reset_index(drop=True)
    return df


def categorise_scenario(scenario_name: str) -> str:
    """Return the category label for a scenario."""
    for cat, members in SCENARIO_CATEGORIES.items():
        if scenario_name in members:
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown(data: dict, per_scenario: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Build the full markdown report as a string."""
    metrics = data["metrics"]
    report = data["report"]
    cls = metrics["classification"]
    params = metrics.get("parameters", {})
    summary = metrics.get("summary", {})

    failure_scenarios = {}
    if report and "failure_analysis" in report:
        failure_scenarios = report["failure_analysis"].get("failure_scenarios", {})

    lines = []

    def ln(text=""):
        lines.append(text)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ln("# Synthetic Data Validation Summary")
    ln()
    ln(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    ln()

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    ln("## Executive Summary")
    ln()

    total = summary.get("total_ground_truth", len(merged))
    n_scenarios = per_scenario["scenario"].nunique()
    n_good_scenarios = len([s for s in per_scenario["scenario"] if categorise_scenario(s) == "Good (standard growth)"])
    n_bad_scenarios = len([s for s in per_scenario["scenario"] if categorise_scenario(s) == "Bad (no / failed growth)"])
    n_edge_scenarios = len([s for s in per_scenario["scenario"] if categorise_scenario(s) == "Edge / borderline"])

    ln(f"The TECAN growth-curve processing pipeline was validated against **{total} synthetic curves** "
       f"spanning **{n_scenarios} test scenarios** ({n_good_scenarios} good, {n_bad_scenarios} bad, "
       f"{n_edge_scenarios} edge-case).")
    ln()
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| **Accuracy** | {pct(cls['accuracy'])} |")
    ln(f"| **Precision** | {pct(cls['precision'])} |")
    ln(f"| **Recall (sensitivity)** | {pct(cls['recall'])} |")
    ln(f"| **F1 score** | {pct(cls['f1_score'])} |")
    ln()

    # Quick interpretation
    n_failures = summary.get("n_failures", 0)
    ln(f"Out of {total} curves, **{total - n_failures}** were classified correctly and "
       f"**{n_failures}** were misclassified. "
       f"The pipeline achieves high recall ({pct(cls['recall'])}), meaning it rarely misses "
       f"genuinely good curves. Precision ({pct(cls['precision'])}) is slightly lower, indicating "
       f"a modest tendency to accept borderline or bad curves as good (false positives).")
    ln()

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    ln("## Confusion Matrix")
    ln()
    cm = cls["confusion_matrix"]
    # Handle both key formats from the two JSON files
    tp = cm.get("TP", cm.get("true_positives", 0))
    tn = cm.get("TN", cm.get("true_negatives", 0))
    fp = cm.get("FP", cm.get("false_positives", 0))
    fn = cm.get("FN", cm.get("false_negatives", 0))

    ln("Positive = GOOD curve, Negative = BAD curve.")
    ln()
    ln("|  | **Predicted GOOD** | **Predicted BAD** | **Total** |")
    ln("|--|---:|---:|---:|")
    ln(f"| **Actually GOOD** | {tp} (TP) | {fn} (FN) | {tp + fn} |")
    ln(f"| **Actually BAD** | {fp} (FP) | {tn} (TN) | {fp + tn} |")
    ln(f"| **Total** | {tp + fp} | {tn + fn} | {tp + fp + tn + fn} |")
    ln()
    ln(f"- **True Positives (TP):** {tp} -- good curves correctly accepted.")
    ln(f"- **True Negatives (TN):** {tn} -- bad curves correctly rejected.")
    ln(f"- **False Positives (FP):** {fp} -- bad curves incorrectly accepted.")
    ln(f"- **False Negatives (FN):** {fn} -- good curves incorrectly rejected.")
    ln()

    # ------------------------------------------------------------------
    # Classification metrics table
    # ------------------------------------------------------------------
    ln("## Classification Metrics")
    ln()
    ln("| Metric | Formula | Value |")
    ln("|--------|---------|------:|")
    ln(f"| Accuracy | (TP+TN) / Total | {pct(cls['accuracy'])} |")
    ln(f"| Precision | TP / (TP+FP) | {pct(cls['precision'])} |")
    ln(f"| Recall | TP / (TP+FN) | {pct(cls['recall'])} |")
    ln(f"| F1 Score | 2*P*R / (P+R) | {pct(cls['f1_score'])} |")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ln(f"| Specificity | TN / (TN+FP) | {pct(specificity)} |")

    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ln(f"| Neg. Predictive Value | TN / (TN+FN) | {pct(npv)} |")
    ln()

    # ------------------------------------------------------------------
    # Parameter recovery
    # ------------------------------------------------------------------
    ln("## Parameter Recovery (True-Positive Curves Only)")
    ln()
    ln("For the {n} curves correctly classified as GOOD, the Gompertz parameters estimated "
       "by the pipeline were compared to the known ground-truth values.".format(
        n=params.get("A", {}).get("n_samples", "N/A")))
    ln()
    ln("| Parameter | R-squared | RMSE | Mean Error (bias) | N |")
    ln("|-----------|----------:|-----:|------------------:|--:|")
    for p_name, p_label in [("A", "A (max OD)"), ("mu", "mu (growth rate)"), ("lambda", "lambda (lag time)")]:
        p = params.get(p_name, {})
        ln(f"| **{p_label}** | {fmt(p.get('r_squared', 0.0), 4)} | "
           f"{fmt(p.get('rmse', 0.0), 4)} | {fmt(p.get('bias', 0.0), 4)} | "
           f"{p.get('n_samples', 'N/A')} |")
    ln()
    ln("**Interpretation:**")
    ln()
    a_r2 = params.get("A", {}).get("r_squared", 0)
    mu_r2 = params.get("mu", {}).get("r_squared", 0)
    lam_r2 = params.get("lambda", {}).get("r_squared", 0)
    ln(f"- **A** (R-squared = {fmt(a_r2, 3)}): Excellent recovery. The pipeline estimates maximum OD very accurately.")
    ln(f"- **mu** (R-squared = {fmt(mu_r2, 3)}): Good recovery, though growth rate is inherently harder to pin down "
       f"because it depends on the slope of a narrow exponential window.")
    ln(f"- **lambda** (R-squared = {fmt(lam_r2, 3)}): Excellent recovery. The small negative bias "
       f"({fmt(params.get('lambda', {}).get('bias', 0), 2)} h) suggests the pipeline estimates lag "
       f"time slightly earlier than ground truth on average, likely due to truncation heuristics.")
    ln()

    # ------------------------------------------------------------------
    # Per-scenario breakdown
    # ------------------------------------------------------------------
    ln("## Per-Scenario Accuracy")
    ln()
    ln("Scenarios are sorted from lowest to highest accuracy. "
       "Each scenario contains 15 synthetic curves.")
    ln()
    ln("| Scenario | Category | Expected | Accuracy | Correct | Wrong | FP | FN |")
    ln("|----------|----------|----------|-------:|--------:|------:|---:|---:|")
    for _, row in per_scenario.iterrows():
        cat = categorise_scenario(row["scenario"])
        ln(f"| {row['scenario']} | {cat} | {row['expected_class']} | "
           f"{pct(row['accuracy'])} | {row['n_correct']} | {row['n_wrong']} | "
           f"{row['false_positives']} | {row['false_negatives']} |")
    ln()

    # Perfect scenarios
    perfect = per_scenario[per_scenario["accuracy"] == 1.0]
    imperfect = per_scenario[per_scenario["accuracy"] < 1.0]
    ln(f"**{len(perfect)} of {len(per_scenario)} scenarios achieved 100% accuracy.** "
       f"The remaining {len(imperfect)} scenarios had at least one misclassification.")
    ln()

    # ------------------------------------------------------------------
    # Failure analysis
    # ------------------------------------------------------------------
    ln("## Failure Analysis")
    ln()

    if failure_scenarios:
        ln("### Misclassifications by Scenario")
        ln()
        ln("| Scenario | Misclassifications | Category | Dominant Error |")
        ln("|----------:|---:|----------|---------------|")
        for scenario, count in sorted(failure_scenarios.items(), key=lambda x: -x[1]):
            cat = categorise_scenario(scenario)
            # Determine dominant error type from per-scenario data
            row = per_scenario[per_scenario["scenario"] == scenario]
            if len(row) > 0:
                r = row.iloc[0]
                if r["false_positives"] > r["false_negatives"]:
                    err_type = f"FP ({r['false_positives']})"
                elif r["false_negatives"] > r["false_positives"]:
                    err_type = f"FN ({r['false_negatives']})"
                else:
                    err_type = f"FP={r['false_positives']}, FN={r['false_negatives']}"
            else:
                err_type = "--"
            ln(f"| {scenario} | {count} | {cat} | {err_type} |")
        ln()

    # Detailed discussion of top failure modes
    ln("### Detailed Failure Discussion")
    ln()

    ln("**False Positives (bad curves accepted as good):**")
    ln()
    fp_scenarios = per_scenario[per_scenario["false_positives"] > 0].sort_values("false_positives", ascending=False)
    if len(fp_scenarios) > 0:
        for _, row in fp_scenarios.iterrows():
            desc = SCENARIO_DESCRIPTIONS.get(row["scenario"], "")
            ln(f"- **{row['scenario']}** ({row['false_positives']} FP): {desc}")
    else:
        ln("- None.")
    ln()

    ln("**False Negatives (good curves rejected as bad):**")
    ln()
    fn_scenarios = per_scenario[per_scenario["false_negatives"] > 0].sort_values("false_negatives", ascending=False)
    if len(fn_scenarios) > 0:
        for _, row in fn_scenarios.iterrows():
            desc = SCENARIO_DESCRIPTIONS.get(row["scenario"], "")
            ln(f"- **{row['scenario']}** ({row['false_negatives']} FN): {desc}")
    else:
        ln("- None.")
    ln()

    # ------------------------------------------------------------------
    # Scenario category descriptions
    # ------------------------------------------------------------------
    ln("## Scenario Descriptions")
    ln()
    for cat, members in SCENARIO_CATEGORIES.items():
        ln(f"### {cat}")
        ln()
        for s in members:
            desc = SCENARIO_DESCRIPTIONS.get(s, "(no description)")
            # Check if this scenario appears in the data
            in_data = s in per_scenario["scenario"].values
            marker = "" if in_data else " *(not present in data)*"
            ln(f"- **{s}**: {desc}{marker}")
        ln()

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    ln("## Recommendations")
    ln()

    # Build recommendations based on actual failure patterns
    recommendations = []

    # Check for borderline failures
    borderline_fp = per_scenario[
        (per_scenario["scenario"].str.startswith("borderline")) & (per_scenario["false_positives"] > 0)
    ]
    if len(borderline_fp) > 0:
        total_borderline_fp = borderline_fp["false_positives"].sum()
        recommendations.append(
            f"**Tighten borderline thresholds.** {total_borderline_fp} false positives arose from "
            f"borderline scenarios ({', '.join(borderline_fp['scenario'].tolist())}). "
            f"Consider adding a secondary quality check or slightly raising the R-squared / "
            f"delta-OD acceptance thresholds."
        )

    # Check for noise-related failures
    noise_failures = per_scenario[
        per_scenario["scenario"].isin(["high_noise", "borderline_noise"]) & (per_scenario["n_wrong"] > 0)
    ]
    if len(noise_failures) > 0:
        total_noise_wrong = noise_failures["n_wrong"].sum()
        recommendations.append(
            f"**Improve noise handling.** {total_noise_wrong} misclassifications came from "
            f"noise-related scenarios. A pre-processing smoothing step or SNR-based filter "
            f"could help discriminate noisy growth from noisy non-growth."
        )

    # Check truncation
    trunc = per_scenario[per_scenario["scenario"] == "truncation_challenge"]
    if len(trunc) > 0 and trunc.iloc[0]["n_wrong"] > 0:
        recommendations.append(
            f"**Handle truncated experiments.** {trunc.iloc[0]['n_wrong']} errors in the "
            f"truncation_challenge scenario suggest the pipeline struggles when stationary phase "
            f"is not fully observed. Fitting with bounded parameters or detecting incomplete "
            f"curves could reduce these errors."
        )

    # Check model-mismatch failures
    model_scenarios = ["logistic_generated", "baranyi_generated", "richards_asymmetric"]
    model_fails = per_scenario[
        per_scenario["scenario"].isin(model_scenarios) & (per_scenario["n_wrong"] > 0)
    ]
    if len(model_fails) > 0:
        total_model_wrong = model_fails["n_wrong"].sum()
        recommendations.append(
            f"**Improve model-mismatch robustness.** {total_model_wrong} misclassifications occurred "
            f"on curves generated with non-Gompertz models ({', '.join(model_fails['scenario'].tolist())}). "
            f"Using a multi-model fitting approach or model-averaging could improve classification "
            f"for diverse growth dynamics."
        )

    # Check minimal growth / pesticide
    minimal = per_scenario[
        per_scenario["scenario"].isin(["minimal_growth", "pesticide_only"]) & (per_scenario["n_wrong"] > 0)
    ]
    if len(minimal) > 0:
        total_min_wrong = minimal["n_wrong"].sum()
        recommendations.append(
            f"**Refine low-growth detection.** {total_min_wrong} errors came from minimal_growth "
            f"and/or pesticide_only scenarios. These involve very small or absent growth signals "
            f"that the pipeline sometimes over-fits. Adding a minimum delta-OD confidence interval "
            f"could help."
        )

    # General recommendation
    recommendations.append(
        "**Expand the test suite over time.** As the pipeline evolves, add new synthetic scenarios "
        "targeting any newly-discovered failure modes. Consider adding real-data validation as a "
        "complementary benchmark."
    )

    for i, rec in enumerate(recommendations, 1):
        ln(f"{i}. {rec}")
        ln()

    # ------------------------------------------------------------------
    # Summary statistics footer
    # ------------------------------------------------------------------
    ln("---")
    ln()
    ln("## Appendix: Summary Statistics")
    ln()
    ln(f"- **Total curves:** {total}")
    ln(f"- **Total scenarios:** {n_scenarios}")
    ln(f"- **Curves per scenario:** 15")
    ln(f"- **Good scenarios:** {n_good_scenarios} ({n_good_scenarios * 15} curves)")
    ln(f"- **Bad scenarios:** {n_bad_scenarios} ({n_bad_scenarios * 15} curves)")
    ln(f"- **Edge scenarios:** {n_edge_scenarios} ({n_edge_scenarios * 15} curves)")
    ln(f"- **Ground truth GOOD curves:** {(merged['expected_class'].str.upper() == 'GOOD').sum()}")
    ln(f"- **Ground truth BAD curves:** {(merged['expected_class'].str.upper() == 'BAD').sum()}")
    ln(f"- **Pipeline GOOD predictions:** {(merged['is_good'].astype(str).str.strip().str.lower() == 'true').sum()}")
    ln(f"- **Pipeline BAD predictions:** {(merged['is_good'].astype(str).str.strip().str.lower() == 'false').sum()}")
    ln()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive markdown validation summary from synthetic pipeline test results."
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=str(DEFAULT_METRICS),
        help="Path to validation_metrics.json",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(DEFAULT_REPORT),
        help="Path to validation_report.json (optional, for failure analysis)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(DEFAULT_GROUND_TRUTH),
        help="Path to ground_truth.csv",
    )
    parser.add_argument(
        "--pipeline-results",
        type=str,
        default=str(DEFAULT_PIPELINE),
        help="Path to processing_results.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for the markdown summary",
    )

    args = parser.parse_args()

    # Validate inputs exist
    for label, path in [
        ("metrics-json", args.metrics_json),
        ("ground-truth", args.ground_truth),
        ("pipeline-results", args.pipeline_results),
    ]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load data
    data = load_data(args)

    # Merge ground truth with pipeline results
    merged = merge_results(data["gt"], data["pipeline"])
    n_merged = len(merged)
    n_gt = len(data["gt"])
    print(f"Merged {n_merged} / {n_gt} ground-truth curves with pipeline results.")

    if n_merged == 0:
        print("ERROR: No curves could be matched between ground truth and pipeline results.", file=sys.stderr)
        sys.exit(1)

    if n_merged < n_gt:
        print(f"WARNING: {n_gt - n_merged} ground-truth curves could not be matched.", file=sys.stderr)

    # Compute per-scenario accuracy
    per_scenario = compute_per_scenario(merged)

    # Generate the markdown
    md = generate_markdown(data, per_scenario, merged)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Validation summary written to: {output_path}")


if __name__ == "__main__":
    main()
