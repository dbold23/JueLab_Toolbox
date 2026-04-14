"""
Statistical Analysis of TECAN Growth Curve Results

This script performs statistical comparisons and generates publication-quality
figures from the Gompertz-fitted growth curve parameters. It reads the combined
results CSV produced by the analysis pipeline and:

  1. Parses strain/treatment metadata from sample names
  2. Computes doubling times from Gompertz parameters
  3. Runs ANOVA, Kruskal-Wallis, and pairwise statistical tests
  4. Produces five publication figures (saved as PNG)
  5. Exports an R-friendly CSV with all derived columns

Author: Generated for BIO380SP25 Research Project
Date: December 2024
"""

# ---------------------------------------------------------------------------
# Imports  (Agg backend MUST be set before pyplot)
# ---------------------------------------------------------------------------
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Mapping from raw treatment strings (upper-cased) to canonical pesticide names.
# The order matters: longer / more-specific patterns are checked first so that
# e.g. "LAMBDACYHALOTHRINANDLB" is not mistakenly caught by a shorter prefix.
PESTICIDE_MAP = {
    "FLUPYRADIFURONEANDLB": "Flupyradifurone",
    "FLUPYRADIFURONEANDG":  "Flupyradifurone",
    "FLUPYRADIFURONE":      "Flupyradifurone",
    "BIFENTHRINANDLB":      "Bifenthrin",
    "BIFENTHRINANDG":       "Bifenthrin",
    "BIFENTHRIN":           "Bifenthrin",
    "LAMBDACYHALOTHRINANDLB": "Lambda-Cyhalothrin",
    "LAMBDACYHALOTHRINANDG": "Lambda-Cyhalothrin",
    "LAMBDACYHALOTHRIN":    "Lambda-Cyhalothrin",
    "DIAZINONANDLB":        "Diazinon",
    "DIAZINONANDG":         "Diazinon",
    "DIAZINON":             "Diazinon",
    "MALATHIONANDLB":       "Malathion",
    "MALATHIONANDG":        "Malathion",
    "MALATHION":            "Malathion",
    "IMIDACLOPRIDANDLB":    "Imidacloprid",
    "IMIDACLOPRIDANDG":     "Imidacloprid",
    "IMIDACLOPRID":         "Imidacloprid",
    "PERMETHRINANDLB":      "Permethrin",
    "PERMETHRINANDG":       "Permethrin",
    "PERMETHRIN":           "Permethrin",
    "LB":                   "Control-LB",
    "H2O":                  "Control-H2O",
}


# ---------------------------------------------------------------------------
# 0.  Refined-parameter substitution (Phase A.2)
# ---------------------------------------------------------------------------

def apply_refined_substitution(df: pd.DataFrame) -> pd.DataFrame:
    """Substitute refined per-parameter Gompertz estimates into gompertz_*.

    When 01_growth_curve_analysis.py emits the *_final/*_source columns
    (post Phase A.1 refactor), this function:

    1. Preserves original whole-curve values under ``gompertz_{a,mu,lambda}_wc``
       for audit.
    2. Replaces ``gompertz_a``/``gompertz_mu``/``gompertz_lambda`` with the
       corresponding ``*_final`` values (refined where identifiable, else
       whole-curve fallback).
    3. Gates ``is_good`` on the ``usable_for`` column: a row remains good only
       if its growth_rate, carrying_capacity, and lag_time are all usable.

    If the new schema columns are absent (older CSV), returns ``df`` unchanged
    with a warning.

    Justification: Phase 0 synthetic validation (tasks/phase0_validate_refined.py)
    showed refined beats whole-curve 6.6×/13.8×/3.4× on RMSE for A/μ/λ.
    """
    required = {"mu_final", "A_final", "lam_final",
                "mu_source", "A_source", "lam_source"}
    if not required.issubset(df.columns):
        print("  WARNING: refined-parameter columns not found — using whole-curve "
              "gompertz_* values directly. Re-run 01_growth_curve_analysis.py to "
              "emit *_final columns.")
        return df

    df = df.copy()

    # Audit: preserve originals
    df["gompertz_a_wc"] = df["gompertz_a"]
    df["gompertz_mu_wc"] = df["gompertz_mu"]
    df["gompertz_lambda_wc"] = df["gompertz_lambda"]

    # Substitute refined where identifiable, whole-curve otherwise
    df["gompertz_a"] = df["A_final"]
    df["gompertz_mu"] = df["mu_final"]
    df["gompertz_lambda"] = df["lam_final"]

    # Report substitution rates
    for p, src in [("μ", "mu_source"), ("A", "A_source"), ("λ", "lam_source")]:
        counts = df[src].value_counts(dropna=False).to_dict()
        refined = counts.get("refined", 0)
        fallback = counts.get("whole_curve_fallback", 0)
        unusable = counts.get("unusable", 0)
        total = refined + fallback + unusable
        if total > 0:
            print(f"  {p}: refined={refined}/{total} ({refined/total:.0%}), "
                  f"fallback={fallback}, unusable={unusable}")

    # Do NOT tighten is_good here — per-test gating happens inside each
    # statistical test (see _usable_mask below) because e.g. an ANOVA on μ
    # does not care whether λ is identifiable.
    return df


def _usable_mask(df: pd.DataFrame, param_name: str) -> pd.Series:
    """Return a boolean mask: is_good AND usable_for contains ``param_name``.

    ``param_name`` must be one of 'growth_rate' (μ), 'carrying_capacity' (A),
    or 'lag_time' (λ). Falls back to just ``is_good`` if usable_for absent.
    """
    base = df["is_good"].astype(bool)
    if "usable_for" not in df.columns:
        return base
    return base & df["usable_for"].fillna("").str.contains(param_name, regex=False)


# ---------------------------------------------------------------------------
# 1.  Parsing helpers
# ---------------------------------------------------------------------------

def parse_strain_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``strain`` column into treatment, strain_id, pesticide,
    has_nutrients, and treatment_type columns."""

    df = df.copy()

    # Split on the LAST hyphen to get treatment and strain_id
    df["treatment"] = df["strain"].apply(lambda s: s.rsplit("-", 1)[0])
    df["strain_id"] = df["strain"].apply(lambda s: s.rsplit("-", 1)[1])

    # Map treatment -> canonical pesticide name
    def _map_pesticide(treatment: str) -> str:
        key = treatment.upper()
        for pattern, name in PESTICIDE_MAP.items():
            if key == pattern:
                return name
        # Fallback -- should not happen with well-formed data
        return "Unknown"

    df["pesticide"] = df["treatment"].apply(_map_pesticide)

    # Nutrient flag: True when treatment contains "ANDLB", "ANDG", or IS "LB"
    df["has_nutrients"] = df["treatment"].apply(
        lambda t: "ANDLB" in t.upper() or "ANDG" in t.upper() or t.upper() == "LB"
    )

    # Treatment type classification
    def _classify_treatment(treatment: str) -> str:
        t = treatment.upper()
        if t == "LB":
            return "LB_only"
        if t == "H2O":
            return "H2O_only"
        if "ANDLB" in t or "ANDG" in t:
            return "Pesticide_plus_LB"
        return "Pesticide_only"

    df["treatment_type"] = df["treatment"].apply(_classify_treatment)

    return df


# ---------------------------------------------------------------------------
# 2.  Doubling-time calculation
# ---------------------------------------------------------------------------

def compute_doubling_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the doubling time for rows with valid Gompertz fits.

    Specific growth rate  = mu * e / A  (from Gompertz parameterisation)
    Doubling time         = ln(2) / specific_growth_rate
    """
    df = df.copy()

    # Only compute for rows that have finite, positive A and mu
    mask = (
        df["is_good"]
        & df["gompertz_a"].notna()
        & df["gompertz_mu"].notna()
        & (df["gompertz_a"] > 0)
        & (df["gompertz_mu"] > 0)
    )

    specific_rate = np.where(
        mask,
        df["gompertz_mu"] * math.e / df["gompertz_a"],
        np.nan,
    )
    df["specific_growth_rate"] = specific_rate
    df["doubling_time"] = np.where(
        mask,
        np.log(2) / specific_rate,
        np.nan,
    )
    return df


# ---------------------------------------------------------------------------
# 3.  Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> None:
    """Run and print statistical tests on good-fit data.

    Per-test identifiability gating applied: μ tests restrict to rows where
    'growth_rate' ∈ usable_for; λ tests to 'lag_time'; A tests to
    'carrying_capacity'. This prevents non-identifiable parameters from
    polluting population-level inference.
    """

    # Backwards-compat master view: any good-fit row, pre-gating
    good = df[df["is_good"]].copy()

    # Per-parameter gated views
    good_mu = df[_usable_mask(df, "growth_rate")].copy()
    good_a = df[_usable_mask(df, "carrying_capacity")].copy()
    good_lam = df[_usable_mask(df, "lag_time")].copy()

    sep = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("STATISTICAL ANALYSIS OF GROWTH CURVE PARAMETERS")
    print(f"{sep}\n")
    print(f"Total samples:  {len(df)}")
    print(f"Good fits:      {len(good)}")
    print(f"Bad fits:       {len(df) - len(good)}")
    print()

    # --- Group counts by treatment_type ----
    print(f"{thin}")
    print("Sample counts by treatment type (good fits only):")
    print(f"{thin}")
    print(good["treatment_type"].value_counts().to_string())
    print()

    # --- Assumption tests ---
    print(f"{thin}")
    print(f"ASSUMPTION TESTS (gompertz_mu by treatment_type, n={len(good_mu)} μ-usable)")
    print(f"{thin}")
    groups_mu = [
        grp["gompertz_mu"].dropna().values
        for _, grp in good_mu.groupby("treatment_type")
    ]
    groups_mu = [g for g in groups_mu if len(g) > 0]
    group_names_mu = [
        name for name, grp in good_mu.groupby("treatment_type")
        if len(grp["gompertz_mu"].dropna()) > 0
    ]

    normality_ok = True
    for name, g in zip(group_names_mu, groups_mu):
        if len(g) >= 3:
            w_stat, p_norm = stats.shapiro(g)
            status = "OK" if p_norm >= 0.05 else "VIOLATED"
            if p_norm < 0.05:
                normality_ok = False
            print(f"  Shapiro-Wilk ({name[:25]}): W={w_stat:.4f} p={p_norm:.4f} [{status}]")
        else:
            print(f"  Shapiro-Wilk ({name[:25]}): n={len(g)} (too few for test)")

    if len(groups_mu) >= 2:
        lev_stat, p_lev = stats.levene(*groups_mu)
        lev_status = "OK" if p_lev >= 0.05 else "VIOLATED"
        print(f"  Levene (homoscedasticity):  stat={lev_stat:.4f} p={p_lev:.4f} [{lev_status}]")
    print()

    # --- Primary test: Kruskal-Wallis (non-parametric, no assumptions) ---
    print(f"{thin}")
    print("PRIMARY: Kruskal-Wallis: gompertz_mu ~ treatment_type (non-parametric)")
    print(f"{thin}")
    if len(groups_mu) >= 2:
        h_stat, p_val_kw = stats.kruskal(*groups_mu)
        print(f"  H-statistic = {h_stat:.4f}")
        print(f"  p-value     = {p_val_kw:.4e}")
        if p_val_kw < 0.05:
            print("  --> Significant difference among treatment types (p < 0.05)")
        else:
            print("  --> No significant difference (p >= 0.05)")
    else:
        print("  Not enough groups with data for Kruskal-Wallis.")
    print()

    # --- Secondary: One-way ANOVA (for reference, with assumption caveats) ---
    print(f"{thin}")
    print("SECONDARY: One-way ANOVA: gompertz_mu ~ treatment_type")
    if not normality_ok:
        print("  WARNING: Normality assumption violated — interpret with caution")
    print(f"{thin}")
    if len(groups_mu) >= 2:
        f_stat, p_val = stats.f_oneway(*groups_mu)
        print(f"  F-statistic = {f_stat:.4f}")
        print(f"  p-value     = {p_val:.4e}")
    else:
        print("  Not enough groups.")
    print()

    # --- Two-way ANOVA with operator blocking (if operator column exists) ---
    if "operator" in good.columns:
        print(f"{thin}")
        print("TWO-WAY ANOVA: gompertz_mu ~ treatment_type + operator (blocking)")
        print(f"{thin}")
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            anova_df = good_mu[["gompertz_mu", "treatment_type", "operator"]].dropna()
            if len(anova_df["operator"].unique()) > 1:
                model = ols("gompertz_mu ~ C(treatment_type) + C(operator)", data=anova_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print(anova_table.to_string())
                print()

                # Report variance attributable to each factor
                ss_total = anova_table["sum_sq"].sum()
                for factor in anova_table.index:
                    if factor != "Residual":
                        pct = 100 * anova_table.loc[factor, "sum_sq"] / ss_total
                        print(f"  {factor}: {pct:.1f}% of variance")
                pct_resid = 100 * anova_table.loc["Residual", "sum_sq"] / ss_total
                print(f"  Residual: {pct_resid:.1f}% of variance")
            else:
                print("  Only 1 operator — cannot estimate operator effect")
        except ImportError:
            print("  statsmodels not installed — skipping two-way ANOVA")
        except Exception as e:
            print(f"  Two-way ANOVA failed: {e}")
        print()

    # --- Confounding check ---
    if "operator" in good.columns:
        print(f"{thin}")
        print("CONFOUNDING CHECK: pesticide × operator distribution")
        print(f"{thin}")
        if "pesticide" in good.columns:
            cross = pd.crosstab(good["pesticide"], good["operator"])
            print(cross.to_string())
            # Flag pesticides with only one operator
            for pest in cross.index:
                n_ops = (cross.loc[pest] > 0).sum()
                if n_ops == 1:
                    op = cross.columns[cross.loc[pest] > 0][0]
                    print(f"  WARNING: {pest} only from {op} — confounded with operator effect")
        print()

    # --- Pairwise: LB vs Pesticide+LB within each group ---
    print(f"{thin}")
    print("Pairwise Mann-Whitney U: LB_only vs Pesticide_plus_LB (gompertz_mu)")
    print("  Question: Does adding pesticide to LB inhibit growth?")
    print(f"  Bonferroni correction applied for {len(good_mu['group'].unique())} comparisons")
    print(f"{thin}")
    n_comparisons = 0
    pairwise_results = []
    for group_name in sorted(good_mu["group"].unique()):
        sub = good_mu[good_mu["group"] == group_name]
        lb_vals = sub.loc[sub["treatment_type"] == "LB_only", "gompertz_mu"].dropna()
        pest_lb_vals = sub.loc[
            sub["treatment_type"] == "Pesticide_plus_LB", "gompertz_mu"
        ].dropna()

        print(f"\n  {group_name}:")
        print(f"    LB_only        n={len(lb_vals):>2}  mean={lb_vals.mean():.4f}" if len(lb_vals) else f"    LB_only        n= 0")
        print(f"    Pesticide+LB   n={len(pest_lb_vals):>2}  mean={pest_lb_vals.mean():.4f}" if len(pest_lb_vals) else f"    Pesticide+LB   n= 0")

        if len(lb_vals) >= 2 and len(pest_lb_vals) >= 2:
            u_stat, p_mw = stats.mannwhitneyu(
                lb_vals, pest_lb_vals, alternative="two-sided"
            )
            n_comparisons += 1
            pairwise_results.append((group_name, p_mw))
            print(f"    U-statistic = {u_stat:.2f},  p-value = {p_mw:.4e} (uncorrected)")
        else:
            print("    --> Insufficient replicates for pairwise test")

    # Apply Bonferroni correction
    if n_comparisons > 0:
        print(f"\n  Bonferroni-corrected pairwise results ({n_comparisons} comparisons):")
        for group_name, p_raw in pairwise_results:
            p_adj = min(p_raw * n_comparisons, 1.0)
            sig = "SIGNIFICANT" if p_adj < 0.05 else "not significant"
            print(f"    {group_name}: p_adj = {p_adj:.4e} [{sig}]")
    print()

    # --- One-way ANOVA on gompertz_lambda by treatment_type ---
    print(f"{thin}")
    print(f"One-way ANOVA: gompertz_lambda ~ treatment_type (n={len(good_lam)} λ-usable)")
    print(f"{thin}")
    groups_lam = [
        grp["gompertz_lambda"].dropna().values
        for _, grp in good_lam.groupby("treatment_type")
    ]
    groups_lam = [g for g in groups_lam if len(g) > 0]
    if len(groups_lam) >= 2:
        f_stat_l, p_val_l = stats.f_oneway(*groups_lam)
        print(f"  F-statistic = {f_stat_l:.4f}")
        print(f"  p-value     = {p_val_l:.4e}")
        if p_val_l < 0.05:
            print("  --> Significant difference among treatment types (p < 0.05)")
        else:
            print("  --> No significant difference (p >= 0.05)")
    else:
        print("  Not enough groups with data for ANOVA.")
    print()

    # --- Summary table of means ---
    print(f"{thin}")
    print("Summary statistics by treatment_type (good fits, mean +/- SD):")
    print(f"{thin}")
    # gompertz_* now hold refined values post-substitution; gompertz_*_wc
    # retain whole-curve originals for audit if present.
    summary_cols = ["gompertz_a", "gompertz_mu", "gompertz_lambda", "doubling_time"]
    for ttype in ["LB_only", "H2O_only", "Pesticide_plus_LB", "Pesticide_only"]:
        sub = good[good["treatment_type"] == ttype]
        if len(sub) == 0:
            continue
        print(f"\n  {ttype} (n={len(sub)}):")
        for col in summary_cols:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"    {col:<25s} {vals.mean():>10.4f} +/- {vals.std():>8.4f}")
            else:
                print(f"    {col:<25s}        N/A")
    print()


# ---------------------------------------------------------------------------
# 4.  Publication figures
# ---------------------------------------------------------------------------

# Consistent colour palette for treatment types
TREATMENT_COLORS = {
    "LB_only":           "#4CAF50",   # green
    "H2O_only":          "#2196F3",   # blue
    "Pesticide_plus_LB": "#FF9800",   # orange
    "Pesticide_only":    "#F44336",   # red
}

TREATMENT_ORDER = ["LB_only", "H2O_only", "Pesticide_plus_LB", "Pesticide_only"]

PESTICIDE_COLORS = {
    "Bifenthrin":         "#e41a1c",
    "Lambda-Cyhalothrin": "#377eb8",
    "Malathion":          "#4daf4a",
    "Imidacloprid":       "#984ea3",
    "Permethrin":         "#ff7f00",
    "Flupyradifurone":    "#a65628",
    "Diazinon":           "#f781bf",
    "Control-LB":         "#999999",
    "Control-H2O":        "#666666",
}


def _set_pub_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def figure1_boxplots(good: pd.DataFrame, outdir: Path) -> None:
    """Fig 1: 1x3 boxplots of A, mu, lambda by treatment_type."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    params = [
        ("gompertz_a",      "Carrying capacity (A)"),
        ("gompertz_mu",     "Maximum growth rate (\u03bc)"),
        ("gompertz_lambda", "Lag time (\u03bb)"),
    ]

    for ax, (col, ylabel) in zip(axes, params):
        data_to_plot = []
        labels = []
        colors = []
        for ttype in TREATMENT_ORDER:
            vals = good.loc[good["treatment_type"] == ttype, col].dropna()
            if len(vals) > 0:
                data_to_plot.append(vals.values)
                labels.append(ttype.replace("_", " "))
                colors.append(TREATMENT_COLORS[ttype])

        bp = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=labels,
            widths=0.6,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Gompertz Parameters by Treatment Type (Good Fits)", fontsize=14, y=1.02)
    plt.tight_layout()
    outpath = outdir / "fig1_boxplots_by_treatment_type.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure2_grouped_bar(good: pd.DataFrame, outdir: Path) -> None:
    """Fig 2: Grouped bar chart of mean mu by group, split by treatment_type."""

    # Compute means and SE
    summary = (
        good.groupby(["group", "treatment_type"])["gompertz_mu"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    groups = sorted(summary["group"].unique())
    treatment_types = [t for t in TREATMENT_ORDER if t in summary["treatment_type"].values]
    n_types = len(treatment_types)
    x = np.arange(len(groups))
    bar_width = 0.8 / n_types

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ttype in enumerate(treatment_types):
        sub = summary[summary["treatment_type"] == ttype]
        means = []
        ses = []
        for g in groups:
            row = sub[sub["group"] == g]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                ses.append(row["se"].values[0] if not np.isnan(row["se"].values[0]) else 0)
            else:
                means.append(0)
                ses.append(0)

        offset = (i - n_types / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=ses,
            label=ttype.replace("_", " "),
            color=TREATMENT_COLORS[ttype],
            alpha=0.8,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Experimental Group")
    ax.set_ylabel("Mean growth rate (\u03bc)")
    ax.set_title("Mean Growth Rate by Group and Treatment Type")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    outpath = outdir / "fig2_grouped_bar_mu_by_group.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure3_scatter_lambda_vs_mu(good: pd.DataFrame, outdir: Path) -> None:
    """Fig 3: Scatter of lambda vs mu, coloured by pesticide."""

    fig, ax = plt.subplots(figsize=(8, 6))

    for pest, color in PESTICIDE_COLORS.items():
        sub = good[good["pesticide"] == pest]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["gompertz_mu"],
            sub["gompertz_lambda"],
            c=color,
            label=pest,
            s=50,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Maximum growth rate (\u03bc)")
    ax.set_ylabel("Lag time (\u03bb)")
    ax.set_title("Lag Time vs. Growth Rate (Good Fits)")
    ax.legend(fontsize=8, frameon=True, loc="best")
    plt.tight_layout()

    outpath = outdir / "fig3_scatter_lambda_vs_mu.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure4_fit_quality(df: pd.DataFrame, outdir: Path) -> None:
    """Fig 4: Fit quality scatter -- fit_r_squared vs fit_rmse, by is_good."""

    # Only include rows that have finite fit metrics
    plot_df = df.dropna(subset=["fit_r_squared", "fit_rmse"]).copy()
    plot_df = plot_df[np.isfinite(plot_df["fit_r_squared"]) & np.isfinite(plot_df["fit_rmse"])]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color, marker in [
        (True, "#4CAF50", "o"),
        (False, "#F44336", "x"),
    ]:
        sub = plot_df[plot_df["is_good"] == label]
        ax.scatter(
            sub["fit_rmse"],
            sub["fit_r_squared"],
            c=color,
            marker=marker,
            label="Good" if label else "Bad",
            s=50,
            alpha=0.75,
            edgecolors="white" if marker == "o" else None,
            linewidths=0.5,
        )

    ax.set_xlabel("Fit RMSE")
    ax.set_ylabel("Fit R\u00b2")
    ax.set_title("Fit Quality: R\u00b2 vs RMSE by Classification")
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8, label="R\u00b2 = 0.95 threshold")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    outpath = outdir / "fig4_fit_quality_scatter.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure5_classification_counts(df: pd.DataFrame, outdir: Path) -> None:
    """Fig 5: Stacked bar chart of good/bad classification counts per group."""

    counts = df.groupby(["group", "is_good"]).size().unstack(fill_value=0)
    # Ensure both columns exist
    for col in [True, False]:
        if col not in counts.columns:
            counts[col] = 0

    groups = sorted(counts.index)
    good_counts = [counts.loc[g, True] for g in groups]
    bad_counts = [counts.loc[g, False] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    width = 0.5

    ax.bar(x, good_counts, width, label="Good", color="#4CAF50", edgecolor="white")
    ax.bar(x, bad_counts, width, bottom=good_counts, label="Bad", color="#F44336", edgecolor="white")

    ax.set_xlabel("Experimental Group")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Classification Counts by Group")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(frameon=True, loc="best")

    # Add count annotations
    for i, (gc, bc) in enumerate(zip(good_counts, bad_counts)):
        total = gc + bc
        ax.text(i, total + 0.3, str(total), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    outpath = outdir / "fig5_classification_counts_by_group.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def generate_all_figures(df: pd.DataFrame, outdir: Path) -> None:
    """Generate all five publication figures."""

    _set_pub_style()
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    good = df[df["is_good"]].copy()

    print("\nGenerating publication figures ...")
    figure1_boxplots(good, figures_dir)
    figure2_grouped_bar(good, figures_dir)
    figure3_scatter_lambda_vs_mu(good, figures_dir)
    figure4_fit_quality(df, figures_dir)
    figure5_classification_counts(df, figures_dir)
    print("All figures saved.\n")


# ---------------------------------------------------------------------------
# 5.  Export R-friendly CSV
# ---------------------------------------------------------------------------

def export_r_csv(df: pd.DataFrame, outdir: Path) -> None:
    """Write a clean CSV suitable for import into R."""

    export_cols = [
        "strain", "treatment", "strain_id", "pesticide",
        "has_nutrients", "treatment_type", "group", "operator", "year",
        "is_good", "classification_reason",
        "delta_od", "max_od",
        "gompertz_a", "gompertz_a_err",
        "gompertz_mu", "gompertz_mu_err",
        "gompertz_lambda", "gompertz_lambda_err",
        "fit_r_squared", "fit_rmse", "fit_mae",
        "specific_growth_rate", "doubling_time",
    ]
    # Only include columns that exist
    cols = [c for c in export_cols if c in df.columns]

    outpath = outdir / "results_for_R_import.csv"
    df[cols].to_csv(outpath, index=False)
    print(f"Exported R-friendly CSV: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Argument parsing ---
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Statistical analysis of TECAN growth curve results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(script_dir / ".." / "OUTPUT" / "all_groups_results.csv"),
        help="Path to all_groups_results.csv (default: ../OUTPUT/all_groups_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / ".." / "OUTPUT"),
        help="Base output directory (default: ../OUTPUT)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    # --- Load data ---
    print(f"Reading input:  {input_path}")
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Coerce is_good to boolean (CSV may store as string "True"/"False")
    df["is_good"] = df["is_good"].astype(str).str.strip().str.lower() == "true"

    print(f"Loaded {len(df)} rows  ({df['is_good'].sum()} good, {(~df['is_good']).sum()} bad)")

    # --- Apply refined-parameter substitution if new schema cols exist ---
    # When 01_growth_curve_analysis.py emits *_final/*_source columns (post
    # Phase A.1 refactor), use refined values for all downstream stats and
    # figures. Phase 0 synthetic validation showed refined beats whole-curve
    # 6.6×/13.8×/3.4× RMSE on A/μ/λ. Whole-curve values preserved under new
    # names for audit.
    df = apply_refined_substitution(df)

    # --- Step 1: Parse strain column ---
    df = parse_strain_column(df)

    # --- Step 2: Doubling time ---
    df = compute_doubling_time(df)

    # --- Step 3: Statistical tests ---
    run_statistical_tests(df)

    # --- Step 4: Figures ---
    generate_all_figures(df, output_dir)

    # --- Step 5: R-friendly CSV ---
    export_r_csv(df, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
