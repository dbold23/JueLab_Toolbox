"""Curves: every fit, BAD first, with the reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from state import S, final_call, get_processed, override_cut, override_excluded, refit_curve, set_excluded
from tecan_growth import plots
from tecan_growth.export import record_review, set_verdict_override

s = S()
st.title("Curves")
results = s.get("results")
if results is None:
    st.info("Run an analysis on the Home page first.")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.stop()
processed = get_processed()

STATUS_COLOR = {"GOOD": "green", "BAD": "red", "not fitted": "grey"}


def _status(cid) -> str:
    cr = results.curves[cid]
    if cr.legacy.stage in ("prefit_rejected", "truncation_invalid", "too_few_points"):
        return "not fitted"
    return final_call(cid)


# ------------------------------------------------------------------ filters
table = results.table.copy()
table["status"] = [_status(c) for c in table.curve_id]
f1, f2, f3 = st.columns([2, 2, 3])
verdicts = f1.multiselect("Show", ["BAD", "GOOD", "not fitted"], default=["BAD", "GOOD", "not fitted"], key="cv_verdicts")
medias = f2.multiselect("Media", list(dict.fromkeys(table.media)), key="cv_media")
search = f3.text_input("Strain contains", key="cv_search")
view = table[table.status.isin(verdicts)]
if medias:
    view = view[view.media.isin(medias)]
if search.strip():
    view = view[view.strain.str.contains(search.strip(), case=False)]
order = {"BAD": 0, "not fitted": 1, "GOOD": 2}
view = view.assign(_o=view.status.map(order)).sort_values(["_o", "media", "strain"])

st.caption(f"{len(view)} of {len(table)} curves · BAD first. Click *Open* to see the full analysis of a curve.")

# ------------------------------------------------------------------ cards
cols = st.columns(3)
for i, row in enumerate(view.itertuples()):
    cr = results.curves[row.curve_id]
    with cols[i % 3].container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown(f"**{row.curve_id}**")
        h2.badge(row.status, color=STATUS_COLOR[row.status])
        st.plotly_chart(plots.curve_sparkline(cr), theme=None, width="stretch", config={"displayModeBar": False},
                        key=f"spark_{row.curve_id}")
        if cr.fit_succeeded:
            st.caption(f"A {cr.fit.a_opt:.2f} · μ {cr.fit.mu_opt:.3f} OD/h · λ {cr.fit.lambda_opt:.1f} h · R² {cr.fit.r_squared:.3f}")
        else:
            st.caption(cr.legacy.classification.reason.replace("BAD: ", "")[:70])
        if st.button("Open", key=f"open_{row.curve_id}", width="stretch"):
            s["selected_curve"] = row.curve_id
            st.rerun()

# ------------------------------------------------------------------ detail
cid = s.get("selected_curve")
if cid not in results.curves:
    st.stop()
cr = results.curves[cid]
st.divider()
d1, d2 = st.columns([4, 1])
d1.header(cid)
d2.badge(_status(cid), color=STATUS_COLOR[_status(cid)])
st.markdown(cr.explanation())
if cid in s["overrides"]:
    o = s["overrides"][cid]
    notes = []
    if override_cut(o):
        notes.append(f"manual window (points {override_cut(o)[0]}–{override_cut(o)[1]})")
    if override_excluded(o):
        notes.append(f"excluded wells {', '.join(override_excluded(o))}")
    if o.get("verdict"):
        notes.append(f"verdict set by hand to {o['verdict']}")
    if notes:
        st.info("Overrides on this curve: " + "; ".join(notes))

st.plotly_chart(plots.pipeline_strip(processed, cr), theme=None, width="stretch", key=f"strip_{cid}")

g1, g2 = st.columns([3, 2])
with g1:
    st.markdown("**Quality gates** — every check the verdict is built from")
    gt = plots.gate_table(cr)
    st.dataframe(gt, hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn(format="%.4g"),
                                "threshold": st.column_config.NumberColumn(format="%.4g")})
with g2:
    if cr.fit_succeeded:
        st.plotly_chart(plots.residual_plot(cr), theme=None, width="stretch", key=f"resid_{cid}")

with st.expander("Show the math"):
    if cr.fit_succeeded:
        f = cr.fit
        st.latex(r"y(t) = A\,\exp\!\left(-\exp\!\left(\frac{\mu e}{A}(\lambda - t) + 1\right)\right)")
        st.latex(rf"A = {f.a_opt:.4f} \pm {f.a_err:.4f}\ \text{{OD}},\quad "
                 rf"\mu = {f.mu_opt:.4f} \pm {f.mu_err:.4f}\ \text{{OD/h}},\quad "
                 rf"\lambda = {f.lambda_opt:.3f} \pm {f.lambda_err:.3f}\ \text{{h}}")
        ss_res = float(np.sum(f.residuals ** 2)); y = cr.truncation.od_truncated; ss_tot = float(np.sum((y - y.mean()) ** 2))
        st.latex(rf"R^2 = 1 - \frac{{\sum (y-\hat y)^2}}{{\sum (y-\bar y)^2}} = 1 - \frac{{{ss_res:.5f}}}{{{ss_tot:.5f}}} = {f.r_squared:.4f}")
        st.latex(rf"\text{{doubling time at the inflection}} = \frac{{\ln 2 \cdot A}}{{\mu e}} = {cr.row['doubling_time_h']:.3f}\ \text{{h}}")
        st.caption(f"Fitted on {len(y)} points between {cr.truncation.time_truncated[0]:.2f} h and "
                   f"{cr.truncation.truncation_time:.2f} h (indices {cr.truncation.start_index}–{cr.truncation.truncation_index}).")
    else:
        st.write("No model was fitted for this curve, so there is no equation to show.")

# ------------------------------------------------------------------ overrides
st.subheader("Disagree? Adjust and refit")
t = cr.time_h
o1, o2 = st.columns(2)
with o1:
    st.markdown("**Growth window**")
    if cr.truncation is not None:
        cur = (int(cr.truncation.start_index), int(cr.truncation.truncation_index))
    else:
        cur = (0, len(t) - 1)
    lo, hi = st.slider("Time window used for the fit (h)", float(t[0]), float(t[-1]),
                       (float(t[cur[0]]), float(t[cur[1]])), step=float(max(np.median(np.diff(t)), 0.01)),
                       key=f"win_{cid}", format="%.2f")
    i0 = int(np.searchsorted(t, lo)); i1 = int(np.searchsorted(t, hi, side="right") - 1)
    st.caption(f"{max(i1 - i0 + 1, 0)} points selected (at least {s['settings'].min_points_after_truncation} needed).")
    w1, w2, w3 = st.columns(3)
    if w1.button("Refit with this window", key=f"refit_{cid}", type="primary"):
        refit_curve(cid, (i0, i1))
        st.rerun()
    if w2.button("Try adaptive on this curve", key=f"adapt_{cid}", help="Cross-validated window search; a few seconds"):
        with st.spinner("Searching for the best window…"):
            refit_curve(cid, adaptive=True)
        st.rerun()
    if w3.button("Back to automatic", key=f"auto_{cid}"):
        refit_curve(cid, None)
        st.rerun()

with o2:
    st.markdown("**Replicate wells**")
    already = override_excluded(s["overrides"].get(cid, {}))
    all_wells = list(dict.fromkeys(list(processed.conditions.set_index("curve_id").loc[cid].wells) + already))
    excluded = st.multiselect("Exclude from the average", all_wells, default=already, key=f"excl_{cid}")
    if st.button("Apply exclusion", key=f"excl_btn_{cid}", disabled=len(excluded) >= len(all_wells)):
        set_excluded(cid, excluded)
        st.rerun()

    st.markdown("**Verdict**")
    current = s["overrides"].get(cid, {}).get("verdict")
    choice = st.radio("Call of record", ["Automatic", "GOOD", "BAD"],
                      index=0 if not current else (1 if current == "GOOD" else 2), horizontal=True, key=f"verd_{cid}")
    note = st.text_input("Why? (required for a manual verdict)", key=f"vnote_{cid}",
                         value=s["overrides"].get(cid, {}).get("note", "") or "")
    if st.button("Save verdict", key=f"vsave_{cid}", disabled=(choice != "Automatic" and not note.strip())):
        set_verdict_override(s["overrides"], cid, None if choice == "Automatic" else choice, note.strip())
        st.rerun()

st.markdown("**Review** (feeds the audit file used to check and retrain the classifier)")
r1, r2 = st.columns([2, 3])
rev = r1.radio("Is the automatic call right?", ["correct", "wrong", "unsure"], horizontal=True, key=f"rev_{cid}")
rnote = r2.text_input("Note", key=f"rnote_{cid}")
if st.button("Save review", key=f"rsave_{cid}"):
    record_review(s["overrides"], cid, rev, rnote.strip())
    st.toast("Review saved", icon=":material/check_circle:")
