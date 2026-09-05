"""Compare: does the pesticide slow growth, and can the bacteria live on it?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from state import S, final_call, get_processed
from tecan_growth import media, plots

s = S()
st.title("Compare treatments")
results = s.get("results")
if results is None:
    st.info("Run an analysis on the Home page first.")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.stop()
processed = get_processed()

table = media.annotate(results.table)
table["final_call"] = [final_call(c) for c in table.curve_id]
include_bad = st.toggle("Include BAD curves (hollow markers)", value=False, key="cmp_bad")

PARAMS = {"μ — fastest growth rate": "gompertz_mu", "A — maximum OD": "gompertz_a",
          "λ — lag time": "gompertz_lambda", "doubling time at the inflection": "doubling_time_h"}
tabs = st.tabs(list(PARAMS))
for tab, (label, col) in zip(tabs, PARAMS.items()):
    with tab:
        st.plotly_chart(plots.parameter_dotplot(table, col, include_bad=include_bad), theme=None,
                        width="stretch", key=f"dot_{col}")

st.plotly_chart(plots.ratio_strip(table, include_bad=include_bad), theme=None, width="stretch")

strains = list(dict.fromkeys(table.strain))
strain = st.selectbox("Overlay one strain in every media", strains, key="cmp_strain")
st.plotly_chart(plots.strain_overlay(processed, results, strain), theme=None, width="stretch")

# ---------------------------------------------------------------- statistics
st.subheader("Is the difference more than noise?")
good = table[(table.final_call == "GOOD") & table.gompertz_mu.notna()]
groups = {k: g.gompertz_mu.to_numpy(dtype=float) for k, g in good.groupby("treatment_type")}
groups = {k: v for k, v in groups.items() if len(v) >= 3}
if len(groups) >= 2:
    from scipy import stats
    f_stat, p_anova = stats.f_oneway(*groups.values())
    h_stat, p_kw = stats.kruskal(*groups.values())
    st.dataframe(pd.DataFrame({
        "test": ["One-way ANOVA on μ", "Kruskal-Wallis on μ"],
        "groups": [", ".join(f"{k} (n={len(v)})" for k, v in groups.items())] * 2,
        "statistic": [f_stat, h_stat], "p-value": [p_anova, p_kw],
    }), hide_index=True, width="stretch",
        column_config={"statistic": st.column_config.NumberColumn(format="%.3f"),
                       "p-value": st.column_config.NumberColumn(format="%.4f")})
    st.caption("Only GOOD curves are used. With three replicate wells averaged into one curve per strain, "
               "n here is the number of strains per treatment; treat p-values as a guide, not a verdict.")
else:
    st.caption("Fewer than three GOOD curves in at least two treatment types, so no test is shown.")
