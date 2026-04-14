"""
Option 1 combined Gompertz × Haldane ranking.

For each pesticide+LB strain, compute:

    degradation_capacity = μ_final * A_final / (1 + S0 / Ki_median)

- μ_final, A_final: from all_groups_results.csv (refined where identifiable,
  else whole-curve fallback — see Phase A.1).
- Ki_median: posterior median from Bayesian Haldane
  (haldane_posterior_summary.csv).
- S0: nominal substrate concentration per pesticide (config.yaml, mg/L).

Writes tasks/degradation_capacity_ranking.csv and a poster figure.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo-relative paths: works from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent                # .../scripts/
TECAN_ROOT = SCRIPT_DIR.parent                              # .../TECAN_growth_curves/
REPO = TECAN_ROOT.parent
RESULTS = TECAN_ROOT / "results" / "tables"
PAPER_FIGURES = TECAN_ROOT / "paper" / "figures"
PAPER_SUPP = TECAN_ROOT / "paper" / "supplementary"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
PAPER_SUPP.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---- Load inputs ----
    all_res = pd.read_csv(RESULTS / "all_groups_results.csv")
    haldane = pd.read_csv(
        RESULTS / "Advanced_Analysis" / "bayesian_haldane" / "haldane_posterior_summary.csv"
    )
    with open(REPO / "TECAN_growth_curves" / "scripts" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    S0_map = {k.upper(): float(v) for k, v in cfg["pesticide_concentrations"].items()}

    # Both CSVs key on `strain`. Inner-join.
    df = haldane.merge(
        all_res[["strain", "mu_final", "A_final", "lam_final",
                 "mu_source", "A_source", "usable_for"]],
        on="strain", how="left"
    )
    # Only strains with refined μ (what the combined metric actually rewards)
    before = len(df)
    df = df[df["mu_final"].notna() & df["A_final"].notna() & df["Ki_median"].notna()]
    df = df[df["usable_for"].fillna("").str.contains("growth_rate")]
    print(f"  Kept {len(df)} / {before} strains after refined-μ gate")

    # Pull S0 per strain
    df["S0_mg_L"] = df["pesticide"].str.upper().map(S0_map).fillna(1.0)

    # ---- Compute combined metric ----
    # degradation_capacity = μ × A / (1 + S0/Ki)
    df["attenuation"] = 1.0 + (df["S0_mg_L"] / df["Ki_median"])
    df["degradation_capacity"] = (df["mu_final"] * df["A_final"]) / df["attenuation"]

    # Sensitivity: also compute HDI-based lower/upper bound using Ki HDI
    # Lower bound on capacity = use Ki_hdi_low (stronger inhibition → lower capacity)
    df["capacity_low"] = (df["mu_final"] * df["A_final"]) / (
        1.0 + df["S0_mg_L"] / df["Ki_hdi_low"]
    )
    df["capacity_high"] = (df["mu_final"] * df["A_final"]) / (
        1.0 + df["S0_mg_L"] / df["Ki_hdi_high"]
    )

    df = df.sort_values("degradation_capacity", ascending=False).reset_index(drop=True)

    # ---- Save ranking ----
    out_cols = [
        "strain", "pesticide",
        "mu_final", "A_final", "Ki_median",
        "Ki_hdi_low", "Ki_hdi_high",
        "S0_mg_L", "attenuation",
        "degradation_capacity", "capacity_low", "capacity_high",
        "mu_source",
    ]
    ranking_csv = PAPER_SUPP / "table_S4_degradation_capacity_ranking.csv"
    df[out_cols].to_csv(ranking_csv, index=False)
    print(f"  Ranking saved to {ranking_csv}")

    # ---- Print top / bottom ----
    print("\n=== TOP 10 BY degradation_capacity ===")
    print(df[["strain", "pesticide", "mu_final", "A_final",
              "Ki_median", "S0_mg_L", "degradation_capacity"]].head(10)
          .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print("\n=== BOTTOM 5 ===")
    print(df[["strain", "pesticide", "mu_final", "A_final",
              "Ki_median", "S0_mg_L", "degradation_capacity"]].tail(5)
          .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # ---- Poster figure ----
    # One horizontal bar per strain, colored by pesticide, with Ki-propagated error.
    PESTICIDE_COLORS = {
        "BIFENTHRIN": "#e41a1c", "PERMETHRIN": "#ff7f00",
        "LAMBDACYHALOTHRIN": "#377eb8",
        "IMIDACLOPRID": "#984ea3", "FLUPYRADIFURONE": "#a65628",
        "DIAZINON": "#f781bf", "MALATHION": "#4daf4a",
    }

    # Show top 20 (full list clutters) + highlight top 5
    top_n = min(20, len(df))
    sub = df.head(top_n).iloc[::-1].reset_index(drop=True)  # reverse so highest bar is at top
    y = np.arange(len(sub))
    colors = [PESTICIDE_COLORS.get(p, "#888") for p in sub["pesticide"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(y, sub["degradation_capacity"], color=colors,
                   edgecolor="black", linewidth=0.6)
    # Horizontal error bars from Ki HDI propagation
    xerr_lower = sub["degradation_capacity"] - sub["capacity_low"]
    xerr_upper = sub["capacity_high"] - sub["degradation_capacity"]
    ax.errorbar(
        sub["degradation_capacity"], y,
        xerr=[xerr_lower.clip(lower=0), xerr_upper.clip(lower=0)],
        fmt="none", ecolor="black", alpha=0.45, capsize=3, linewidth=1,
    )
    ax.set_yticks(y)
    # Compact labels: just the strain suffix (the part after last "-") + Ki
    def _short_label(strain: str, ki: float) -> str:
        short = strain.rsplit("-", 1)[-1]
        return f"{short}   (Ki={ki:.1f})"
    ax.set_yticklabels(
        [_short_label(s, k) for s, k in zip(sub["strain"], sub["Ki_median"])],
        fontsize=10,
    )
    # Annotate the top-ranked strain with its capacity value for poster punch
    for i, (cap, strain) in enumerate(zip(sub["degradation_capacity"], sub["strain"])):
        if i == len(sub) - 1:  # top (highest) bar is last after reversal
            ax.text(cap * 1.02, i, f"  {cap:.3f}",
                    va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel(
        r"Degradation capacity  $= \mu \cdot A / (1 + S_0/K_i)$   (arb. units)",
        fontsize=11,
    )
    ax.set_title(
        f"Top {top_n} Pesticide-Degrading Candidates\n"
        r"Combined growth kinetics ($\mu \times A$) attenuated by substrate inhibition ($K_i$)",
        fontsize=12, fontweight="bold",
    )
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=p.title())
               for p, c in PESTICIDE_COLORS.items() if p in sub["pesticide"].values]
    ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, sub["capacity_high"].clip(upper=sub["degradation_capacity"].max() * 2).max() * 1.05)
    plt.tight_layout()
    out = PAPER_FIGURES / "fig_degradation_capacity_ranking.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved paper figure: {out}")

    # Also drop a copy into ~/Desktop/poster_figures/ if that directory exists
    # (user-facing convenience; skipped silently when not present).
    desktop_poster = Path.home() / "Desktop" / "poster_figures"
    if desktop_poster.exists():
        import shutil
        shutil.copy(out, desktop_poster / "fig8_degradation_capacity_ranking.png")
        print(f"  Copied to {desktop_poster}/fig8_degradation_capacity_ranking.png")

    # ---- Top-5 callout (for poster prose) ----
    print("\n=== POSTER CALLOUT (top 5) ===")
    for i, row in df.head(5).iterrows():
        print(
            f"  {i+1}. {row['strain']}  ({row['pesticide'].title()})  "
            f"capacity={row['degradation_capacity']:.3f}  "
            f"(μ={row['mu_final']:.3f}, A={row['A_final']:.2f}, "
            f"Ki={row['Ki_median']:.1f}, S0={row['S0_mg_L']:.0f} mg/L)"
        )


if __name__ == "__main__":
    main()
