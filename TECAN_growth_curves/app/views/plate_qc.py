"""Plate QC: is the raw data sane before we trust any fit?"""
from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from state import S, get_processed
from tecan_growth import plots

s = S()
st.title("Plate QC")
run = s.get("run")
if run is None:
    st.info("Load a plate on the Home page first.")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.stop()

st.markdown("Raw-well checks that do not depend on any model: dead wells, edge effects, "
            "disagreeing replicates, blanks that grew, and whether the incubator held its temperature.")

st.plotly_chart(plots.delta_od_heatmap(run, s.get("key")), theme=None, width="stretch")

if s.get("key") is not None:
    try:
        processed = get_processed()
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()
    media = st.selectbox("Media", list(dict.fromkeys(processed.conditions.media)), key="qc_media")
    blanked = st.toggle("Show blank-subtracted values", value=True, key="qc_blanked")
    st.plotly_chart(plots.replicate_facets(processed, media, blanked=blanked), theme=None, width="stretch")

    blanks = processed.blanks[processed.blanks.media == media]
    if len(blanks):
        rise = float(blanks.blank_mean.max() - blanks.blank_mean.iloc[:5].mean())
        if rise > 0.1:
            st.warning(f"The {media} blank wells rose by {rise:.2f} OD during the run. "
                       "Media-only wells should stay flat; check for contamination.")
        else:
            st.caption(f"{media} blanks drifted by {rise:+.3f} OD over the run (flat is good).")
    if processed.qc:
        with st.expander(f"{len(processed.qc)} QC note(s)"):
            for q in processed.qc:
                st.write("• " + q)

st.plotly_chart(plots.temperature_trace(run), theme=None, width="stretch")
