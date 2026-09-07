#!/usr/bin/env python3
"""
Export Clean Results for Collaboration and Generate Methodology Document

This script reads the combined pipeline results (all_groups_results.csv) and
produces two outputs:
  1. A clean, human-readable CSV of good fits for sharing with lab members.
  2. A methodology document (Markdown) describing the analysis pipeline.

Author: Generated for BIO380SP25 Research Project
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

EULER_E = math.e
LN2 = math.log(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH = REPO_ROOT / "synthetic_data"

# Where the synthetic benchmark's own artefacts live. The suite size and the
# headline are read from these at run time: this document used to hard-code
# "480 synthetic growth curves" and an accuracy that had already been
# superseded, and there was nothing to keep the two in step.
SYNTH_REPORTS = (
    SYNTH / "output" / "comprehensive_test" / "validation" / "first_peak" / "report.json",
    SYNTH / "output" / "comprehensive_test" / "validation" / "adaptive" / "report.json",
    SYNTH / "benchmark" / "v2" / "report.json",
)
SYNTH_MANIFEST = SYNTH / "output" / "comprehensive_test" / "MANIFEST.json"


def _load_json(path: Path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def synthetic_validation_section() -> str:
    """The 'Validation Results' section, built from the benchmark's own files.

    Reads the first report.json it finds (a step-9 run directory, else the
    tracked snapshot) and MANIFEST.json for the suite size. When neither is in
    the checkout the section says so instead of quoting a number no file in the
    tree supports.
    """
    report = None
    report_path = None
    for candidate in SYNTH_REPORTS:
        report = _load_json(candidate)
        if report and "classification" in report:
            report_path = candidate
            break
        report = None

    manifest = _load_json(SYNTH_MANIFEST)

    if report is None:
        source = "MANIFEST.json" if manifest else None
        n_curves = manifest.get("n_curves_written") if manifest else None
        n_scenarios = manifest.get("n_scenarios") if manifest else None
        if n_curves:
            return (
                f"The pipeline is validated against **{n_curves} synthetic growth "
                f"curves** across {n_scenarios} scenarios with known parameters "
                f"(`synthetic_data/`, suite size read from {source}). This "
                f"checkout holds no benchmark report, so no accuracy is quoted "
                f"here; run `make step-9` to produce one, and see "
                f"`docs/BENCHMARK_LEDGER.md`.\n"
            )
        return (
            "The pipeline is validated against a synthetic suite with known "
            "parameters (`synthetic_data/`). No benchmark report is present in "
            "this checkout, so no accuracy is quoted here; run `make step-9` to "
            "produce one, and see `docs/BENCHMARK_LEDGER.md`.\n"
        )

    run = report.get("run", {})
    cls = report.get("classification", {})
    recovery = (report.get("parameter_recovery", {}) or {}).get("accepted_good", {})
    n_curves = run.get("n_curves") or (manifest or {}).get("n_curves_written")
    n_scenarios = run.get("n_scenarios") or (manifest or {}).get("n_scenarios")

    def pc(value):
        return "--" if value is None else f"{value * 100:.1f}%"

    def r2(name):
        block = recovery.get(name) or {}
        value = block.get("r_squared_across_scenarios")
        return "--" if value is None else f"{value:.3f}"

    rel = report_path.relative_to(REPO_ROOT)
    lines = [
        f"The pipeline was validated on **{n_curves} synthetic growth curves** "
        f"across {n_scenarios} scenarios with known Gompertz, logistic, Richards, "
        f"Baranyi and Haldane parameters, generated with realistic noise, "
        f"artefacts and blanks calibrated to the plates in this study.",
        "",
        f"Numbers below are read from `{rel}` at export time "
        f"(`{run.get('truncation_mode')}` truncation, seed {run.get('seed')}, "
        f"dataset `{str(run.get('dataset_sha256'))[:12]}`, git "
        f"`{str(run.get('git_sha'))[:12]}`). They are not typed into this script.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Curves in the scored headline | {cls.get('n')} |",
        f"| Balanced accuracy | {pc(cls.get('balanced_accuracy'))} |",
        f"| Sensitivity | {pc(cls.get('sensitivity'))} |",
        f"| Specificity | {pc(cls.get('specificity'))} |",
        f"| Accuracy | {pc(cls.get('accuracy'))} |",
        f"| Parameter recovery R^2 (A) | {r2('A')} |",
        f"| Parameter recovery R^2 (mu) | {r2('mu')} |",
        f"| Parameter recovery R^2 (lambda) | {r2('lambda')} |",
        "",
        "Parameter recovery is pooled over the curves the pipeline accepted and "
        "the truth calls good; it is conditioned on the pipeline's own verdict, "
        "so it is not an unbiased measure of the estimator. The R^2 for A is the "
        "lowest of the three because the suite deliberately contains families "
        "whose asymptote is not identifiable from the observed window -- runs "
        "that stop before the plateau. The full account, "
        "including which scenarios are missed and why, is in "
        "`docs/BENCHMARK_LEDGER.md` and in the rendered "
        "`VALIDATION_SUMMARY.md`.",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# Strain Parsing
# =============================================================================

def parse_strain_column(strain_str: str):
    """
    Parse the compound strain identifier into Treatment and Strain_ID.

    The strain column uses the format TREATMENT-STRAINID, for example:
        "BifenthrinANDLB-BIF2"  -> Treatment="BifenthrinANDLB", Strain_ID="BIF2"
        "LB-MAL8"               -> Treatment="LB",              Strain_ID="MAL8"
        "H2O-PERM4"             -> Treatment="H2O",             Strain_ID="PERM4"

    We split on the LAST hyphen so that compound treatments with hyphens are
    handled correctly (e.g., "LambdaCyhalothrin-LCY4" splits into
    Treatment="LambdaCyhalothrin" and Strain_ID="LCY4").

    Parameters
    ----------
    strain_str : str
        Raw strain string from the CSV.

    Returns
    -------
    tuple of (str, str)
        (Treatment, Strain_ID)
    """
    if "-" in strain_str:
        last_hyphen = strain_str.rfind("-")
        treatment = strain_str[:last_hyphen]
        strain_id = strain_str[last_hyphen + 1:]
        return treatment, strain_id
    return strain_str, strain_str


def compute_doubling_time(a: float, mu: float) -> float:
    """
    Compute the doubling time from modified Gompertz parameters.

    The maximum specific growth rate in the modified Gompertz formulation is
    related to the instantaneous growth rate by:

        max_instantaneous_rate = mu * e / A

    The doubling time is therefore:

        t_d = ln(2) / (mu * e / A)

    Parameters
    ----------
    a : float
        Gompertz A parameter (maximum OD600 asymptote).
    mu : float
        Gompertz mu parameter (maximum specific growth rate, OD/hr).

    Returns
    -------
    float
        Doubling time in hours, or NaN if the calculation is invalid.
    """
    if pd.isna(a) or pd.isna(mu) or a <= 0 or mu <= 0:
        return float("nan")
    instantaneous_rate = mu * EULER_E / a
    if instantaneous_rate <= 0:
        return float("nan")
    return LN2 / instantaneous_rate


# =============================================================================
# CSV Export
# =============================================================================

def export_collaboration_csv(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Create a clean, human-readable CSV containing only good fits.

    Parameters
    ----------
    df : pd.DataFrame
        Full pipeline results (all_groups_results.csv).
    output_path : Path
        Destination file path.

    Returns
    -------
    pd.DataFrame
        The exported DataFrame (for downstream use / reporting).
    """
    # Keep only good fits
    good = df[df["is_good"] == True].copy()  # noqa: E712

    # Parse strain into Treatment and Strain_ID
    parsed = good["strain"].apply(parse_strain_column)
    good["Treatment"] = parsed.apply(lambda x: x[0])
    good["Strain_ID"] = parsed.apply(lambda x: x[1])

    # Compute doubling time
    good["Doubling_Time_hrs"] = good.apply(
        lambda row: compute_doubling_time(row["gompertz_a"], row["gompertz_mu"]),
        axis=1,
    )

    # Build the clean output DataFrame with renamed, reader-friendly columns
    clean = pd.DataFrame(
        {
            "Strain": good["strain"],
            "Group": good["group"],
            "Treatment": good["Treatment"],
            "Strain_ID": good["Strain_ID"],
            "Classification": "Good",
            "Max_OD600_A": good["gompertz_a"],
            "A_StdErr": good["gompertz_a_err"],
            "Growth_Rate_mu_OD_per_hr": good["gompertz_mu"],
            "mu_StdErr": good["gompertz_mu_err"],
            "Lag_Phase_lambda_hrs": good["gompertz_lambda"],
            "lambda_StdErr": good["gompertz_lambda_err"],
            "Doubling_Time_hrs": good["Doubling_Time_hrs"],
            "R_squared": good["fit_r_squared"],
            "RMSE": good["fit_rmse"],
        }
    )

    # Sort by Group then by Growth_Rate descending
    clean = clean.sort_values(
        by=["Group", "Growth_Rate_mu_OD_per_hr"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Round numeric columns for readability
    numeric_cols = [
        "Max_OD600_A", "A_StdErr",
        "Growth_Rate_mu_OD_per_hr", "mu_StdErr",
        "Lag_Phase_lambda_hrs", "lambda_StdErr",
        "Doubling_Time_hrs", "R_squared", "RMSE",
    ]
    for col in numeric_cols:
        clean[col] = clean[col].round(6)

    clean.to_csv(output_path, index=False)
    return clean


# =============================================================================
# Methodology Document
# =============================================================================

def generate_methodology(df: pd.DataFrame, clean: pd.DataFrame, output_path: Path):
    """
    Write a Markdown methodology document.

    Parameters
    ----------
    df : pd.DataFrame
        Full pipeline results (all groups).
    clean : pd.DataFrame
        The exported clean DataFrame (good fits only).
    output_path : Path
        Destination file path.
    """
    total = len(df)
    good = int(df["is_good"].sum())
    bad = total - good

    # Per-group summary
    group_summary_rows = []
    for group_name in sorted(df["group"].unique()):
        gdf = df[df["group"] == group_name]
        g_total = len(gdf)
        g_good = int(gdf["is_good"].sum())
        g_bad = g_total - g_good

        # Identify pesticides and strains from the group
        parsed = gdf["strain"].apply(parse_strain_column)
        treatments = sorted(set(parsed.apply(lambda x: x[0])))
        strain_ids = sorted(set(parsed.apply(lambda x: x[1])))

        # Identify the pesticide treatments (not LB or H2O)
        pesticides = [
            t for t in treatments
            if t.upper() not in ("LB", "H2O")
            and "ANDLB" not in t.upper()
        ]
        pesticide_str = ", ".join(pesticides) if pesticides else "None"
        strain_str = ", ".join(strain_ids)

        group_summary_rows.append(
            f"| {group_name} | {pesticide_str} | {strain_str} "
            f"| {g_total} | {g_good} | {g_bad} |"
        )

    group_table = "\n".join(group_summary_rows)

    # Build column definitions table
    column_defs = """\
| Column | Description |
|--------|-------------|
| Strain | Full strain identifier (TREATMENT-STRAINID) |
| Group | Experimental group (Group1 -- Group4) |
| Treatment | Treatment condition parsed from strain name (e.g., LB, BifenthrinANDLB, H2O) |
| Strain_ID | Short strain identifier parsed from strain name (e.g., BIF2, MAL8) |
| Classification | Good or Bad based on fit quality criteria |
| Max_OD600_A | Gompertz A parameter -- fitted maximum OD600 (asymptote) |
| A_StdErr | Standard error of the A parameter estimate |
| Growth_Rate_mu_OD_per_hr | Gompertz mu parameter -- maximum specific growth rate (OD600/hour) |
| mu_StdErr | Standard error of the mu parameter estimate |
| Lag_Phase_lambda_hrs | Gompertz lambda parameter -- lag phase duration (hours) |
| lambda_StdErr | Standard error of the lambda parameter estimate |
| Doubling_Time_hrs | Estimated doubling time: ln(2) / (mu * e / A). NaN for bad fits |
| R_squared | Coefficient of determination (R^2) of the Gompertz fit |
| RMSE | Root mean squared error of the Gompertz fit |"""

    # Read at run time from the benchmark's own artefacts (never hard-coded).
    synthetic_validation = synthetic_validation_section()

    md = f"""\
# TECAN Growth Curve Analysis Pipeline - Methodology

## Growth Model

All growth curves were fit to the **modified Gompertz equation**:

```
y(t) = A * exp( -exp( (mu * e / A) * (lambda - t) + 1 ) )
```

Where:

| Parameter | Symbol | Biological Meaning |
|-----------|--------|--------------------|
| A | A | Maximum population density (asymptotic OD600). Represents the carrying capacity of the culture under the given conditions. |
| mu | mu | Maximum specific growth rate (OD600 / hour). The steepest slope of the growth curve during exponential phase. |
| lambda | lambda | Lag phase duration (hours). The time before exponential growth begins, during which cells adapt to their environment. |

The modified Gompertz model is widely used for microbial growth because it
captures the three canonical phases of batch culture (lag, exponential,
stationary) in a single sigmoidal equation with biologically interpretable
parameters.

The **doubling time** is derived from the Gompertz parameters as:

```
t_d = ln(2) / (mu * e / A)
```

where `mu * e / A` gives the maximum instantaneous specific growth rate.

---

## Data Processing

### Preprocessing Steps

1. **Blank subtraction** -- For each media type, blank wells (no inoculum)
   were averaged across replicates at each time point. The blank average was
   subtracted from every strain well to correct for media absorbance.

2. **Triplicate averaging** -- Each strain/condition was measured in
   triplicate wells. The three replicates were averaged to produce a single
   representative growth curve per strain per condition.

3. **Time conversion** -- Raw plate reader times (in seconds) were converted
   to hours.

4. **Non-negativity enforcement** -- After blank subtraction, any negative
   OD600 values (arising from instrument noise) were set to zero.

---

## Classification Criteria

Growth curves were classified using a **fit-quality-based** approach, which
is more scientifically rigorous than raw OD thresholds because it evaluates
whether the curve actually follows Gompertz growth kinetics.

### Classification Thresholds

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| R-squared (R^2) | >= 0.95 | The Gompertz fit must explain at least 95% of variance in the truncated curve. |
| A parameter error | < 20% relative | The standard error of the fitted maximum OD must be less than 20% of the parameter value. |
| mu parameter error | < 20% relative | The standard error of the fitted growth rate must be less than 20% of the parameter value. |

A curve is classified as **Good** only if ALL three criteria are satisfied.
Otherwise it is classified as **Bad** with a specific reason recorded.

---

## Truncation Strategy

Before fitting, each growth curve was truncated to retain only the portion
that conforms to the Gompertz sigmoid shape (lag through early stationary
phase). Two complementary strategies were used:

### Adaptive Truncation (Primary)

1. **Biological estimate** -- The stationary phase onset was estimated by
   finding where the curve reached 90% of maximum OD and where the growth
   rate dropped below 5% of its maximum.

2. **R-squared optimization** -- A search window around the biological
   estimate was scanned. At each candidate truncation point, a quick
   Gompertz fit was performed, and the truncation point yielding the highest
   R^2 was selected. Ties in R^2 were broken by preferring the point closest
   to the biological estimate.

3. **Noisy-start trimming** -- The start of the curve was also evaluated.
   Data points before the onset of measurable growth (baseline noise) were
   trimmed to improve fit quality.

### First-Peak Truncation (Fallback)

For curves where adaptive truncation did not converge, the data was truncated
at the **first local maximum** after exponential growth. This prevents the
inclusion of diauxic shifts, death phase, or other non-Gompertz behavior.

---

## Validation Results

{synthetic_validation}

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib scipy
# or
pip install -r requirements.txt
```

### Quick Start

```bash
# Run the complete pipeline (all groups, preprocessing + analysis + combine)
cd ANALYSIS_SCRIPTS/
./run_all_groups.sh

# Run analysis on a single group
python 01_growth_curve_analysis.py DATA/<GROUP_FOLDER> -o OUTPUT/<GROUP>_Final --adaptive

# Preprocess raw Group 1 plate reader data
python 02_preprocess_raw_plate_data.py \\
    "DATA/Group1/GrowthRate_ Group1_Values.csv" \\
    "DATA/Group1/GROUP1_KEY_V2.csv" \\
    -o "DATA/Group1/Group_1_DATA"

# Generate collaboration exports (this script)
python 04_export_for_collaboration.py
# or with custom paths:
python 04_export_for_collaboration.py --input ../OUTPUT/all_groups_results.csv --output-dir ../OUTPUT
```

---

## CSV Column Definitions

The collaboration CSV (`growth_rate_summary_for_collaboration.csv`) contains
only **Good** fits and uses human-readable column names:

{column_defs}

---

## Experimental Groups

| Group | Pesticide(s) | Strains | Total Curves | Good | Bad |
|-------|-------------|---------|--------------|------|-----|
{group_table}
| **Total** | | | **{total}** | **{good}** | **{bad}** |

### Treatment Conditions per Strain

Each strain was measured under multiple conditions:

- **LB** -- LB broth only (positive control; bacteria should grow normally)
- **Pesticide + LB** -- Pesticide dissolved in LB broth (test condition)
- **Pesticide only** -- Pesticide in water without nutrients (negative control)
- **H2O** -- Water only (negative control; no growth expected)
"""

    output_path.write_text(md, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export clean collaboration CSV and methodology document "
            "from the growth curve analysis pipeline results."
        )
    )

    # Resolve default paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / ".." / "OUTPUT" / "all_groups_results.csv"
    default_output_dir = script_dir / ".." / "OUTPUT"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help=(
            "Path to the combined results CSV. "
            "Default: ../OUTPUT/all_groups_results.csv (relative to script location)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=(
            "Directory for output files. "
            "Default: ../OUTPUT (relative to script location)"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    # --- Validate input ---
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run the analysis pipeline first (./run_all_groups.sh) to generate results.")
        return

    # --- Ensure output directory exists ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading results from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total curves loaded: {len(df)}")
    print(f"  Good fits: {int(df['is_good'].sum())}")
    print(f"  Bad fits: {int((~df['is_good']).sum())}")

    # --- Export collaboration CSV ---
    csv_path = output_dir / "growth_rate_summary_for_collaboration.csv"
    print(f"\nExporting collaboration CSV to: {csv_path}")
    clean = export_collaboration_csv(df, csv_path)
    print(f"  Exported {len(clean)} good-fit rows")
    print(f"  Groups present: {sorted(clean['Group'].unique())}")

    # --- Generate methodology document ---
    md_path = output_dir / "pipeline_methodology.md"
    print(f"\nGenerating methodology document: {md_path}")
    generate_methodology(df, clean, md_path)
    print("  Methodology document written.")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  1. {csv_path}")
    print(f"     -> {len(clean)} good fits, sorted by Group then Growth Rate")
    print(f"  2. {md_path}")
    print(f"     -> Full methodology documentation")
    print()


if __name__ == "__main__":
    main()
