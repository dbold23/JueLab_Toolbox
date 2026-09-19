"""
TECAN growth-curve analysis: the lab interface.

Run with:  streamlit run app/Home.py      (or: make app)
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from _bootstrap import EXAMPLE_PLATE, KEYS_DIR
from sidebar import render_sidebar
from state import (S, blocking_reason, key_from_table, key_validation, load_key_bytes,
                   load_run, run_analysis, set_key)
from tecan_growth import plots
from tecan_growth.plate import PlateKey, normalize_well

st.set_page_config(page_title="TECAN Growth Curves", page_icon=":material/biotech:", layout="wide",
                   initial_sidebar_state="expanded")


def _selected_wells(event) -> list:
    """Well ids from a Plotly selection event (click, box or lasso)."""
    wells = []
    try:
        pts = event.selection.points if hasattr(event, "selection") else event["selection"]["points"]
    except Exception:
        return wells
    for p in pts or []:
        cd = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        w = normalize_well(cd) if cd is not None else None
        if w:
            wells.append(w)
    return wells


def _receipt(run) -> str:
    m = run.metadata
    parts = [f"{run.n_cycles} cycles", f"every {run.interval_h * 60:.0f} min" if run.n_cycles > 1 else "",
             f"{run.duration_h:.1f} h", f"{len(run.well_ids)} wells"]
    if m.get("wavelength_nm"):
        parts.append(f"{m['wavelength_nm']} nm")
    if run.temperature_c is not None and len(run.temperature_c):
        import numpy as np
        t = np.asarray(run.temperature_c, dtype=float)
        if np.isfinite(t).any():
            parts.append(f"{np.nanmean(t):.1f} °C")
    if m.get("date"):
        parts.append(str(m["date"]))
    return " · ".join(p for p in parts if p)


def home() -> None:
    s = S()
    st.title("TECAN growth-curve analysis")
    st.markdown(
        "Drop a plate-reader export, tell the tool which wells hold what, press **Run**. "
        "Every curve is blank-corrected, averaged over its replicates, fitted with a Gompertz growth model "
        "and checked against quality gates so you can see *why* it was called GOOD or BAD."
    )

    # ------------------------------------------------------------- 1. load
    st.subheader("1 · Load a plate")
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        up = st.file_uploader("TECAN export (.xlsx) or values CSV", type=["xlsx", "xlsm", "csv"], key="upload_values")
    with c2:
        upk = st.file_uploader("Plate key (optional; Cell, Media, Strain)", type=["csv", "xlsx"], key="upload_key")
    with c3:
        st.markdown("&nbsp;")
        if st.button("Try the example plate", type="secondary", width="stretch", key="example_btn"):
            data = EXAMPLE_PLATE.read_bytes()
            load_run(EXAMPLE_PLATE.name, data)
            s["loaded_upload_id"] = None
            st.rerun()

    if up is not None and s.get("loaded_upload_id") != up.file_id:
        try:
            load_run(up.name, up.getvalue())
            s["loaded_upload_id"] = up.file_id
        except Exception as e:  # noqa: BLE001 - shown to the user
            st.error(f"Could not read {up.name}: {e}")
            return
    if upk is not None and s.get("loaded_key_id") != upk.file_id and s.get("run") is not None:
        try:
            load_key_bytes(upk.name, upk.getvalue())
            s["loaded_key_id"] = upk.file_id
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the key {upk.name}: {e}")

    run = s.get("run")
    if run is None:
        st.info("Load an export or try the example plate to begin.")
        with st.expander("What file do I need?"):
            st.markdown(
                "- The **.xlsx** that SparkControl / i-control saves at the end of a kinetic run. "
                "The tool finds the measurement table by its `Cycle Nr.` header on any sheet, so extra sheets are fine.\n"
                "- Or a **values CSV** with the columns `Cycle Nr., Time [s], Temp. [°C], A1 … H12`.\n"
                "- A **key** listing `Cell, Media, Strain` for each well. Mark media-only wells with the strain `BLANK`. "
                "If the workbook already has a key sheet it is picked up automatically."
            )
        return

    st.success(f"**{s['input_name']}** — {_receipt(run)}", icon=":material/check_circle:")
    for w in run.warnings:
        st.warning(w)

    # ------------------------------------------------------------- 2. key
    st.subheader("2 · Which wells hold what")
    if len(run.key_candidates) > 1:
        names = list(run.key_candidates)
        current = s.get("key_source")
        idx = names.index(current) if current in names else 0
        choice = st.selectbox("The workbook has several key sheets. Use:", names, index=idx, key="key_sheet_choice")
        if choice != s.get("key_source"):
            try:
                set_key(PlateKey.from_dataframe(run.key_candidates[choice], source=choice), source=choice)
            except ValueError as e:
                st.error(str(e))
    st.caption(f"Key source: **{s.get('key_source')}**. Edit the table or paint wells on the plate, then press *Apply key*.")

    kc1, kc2 = st.columns([2, 3])
    with kc1:
        edited = st.data_editor(s["key_table"], num_rows="fixed", hide_index=True, height=360, key="key_editor",
                                column_config={"Cell": st.column_config.TextColumn(disabled=True)})
        b1, b2 = st.columns(2)
        if b1.button("Apply key", type="primary", width="stretch", key="apply_key"):
            key, errs = key_from_table(edited)
            if key is None:
                st.error(errs[0])
            else:
                set_key(key, source="edited")
                st.rerun()
        b2.download_button("Download key CSV", data=edited.to_csv(index=False).encode(), file_name="plate_key.csv",
                           mime="text/csv", width="stretch", key="dl_key")

    with kc2:
        results = s.get("results")
        mode = "verdict" if results is not None else "layout"
        key_for_grid = s.get("key")
        if key_for_grid is None:
            key_for_grid, _ = key_from_table(edited)
        fig = plots.plate_grid(key_for_grid, results, run, mode=mode)
        event = st.plotly_chart(fig, key="plate_grid", on_select="rerun",
                                selection_mode=("points", "box", "lasso"), theme=None, width="stretch")
        wells = _selected_wells(event)
        if results is not None and len(wells) == 1 and key_for_grid is not None:
            row = key_for_grid.table[key_for_grid.table.well == wells[0]]
            if len(row) and row.iloc[0].in_use and not row.iloc[0].is_blank:
                cid = f"{row.iloc[0].media}-{row.iloc[0].strain}"
                if cid in results.curves:
                    s["selected_curve"] = cid
                    st.page_link("views/curves.py", label=f"Open {cid} in Curves", icon=":material/open_in_new:")
        with st.expander("Paint wells", expanded=results is None and s.get("key") is None):
            st.caption("Click, box-select or lasso wells on the plate, type the media and strain, then Assign.")
            p1, p2 = st.columns(2)
            media_in = p1.text_input("Media", key="paint_media", placeholder="LB, H2O, MALATHION, MALATHIONANDLB …")
            strain_in = p2.text_input("Strain", key="paint_strain", placeholder="MAL8 (or BLANK)")
            st.caption(f"Selected: {', '.join(wells) if wells else 'none'}")
            a1, a2, a3 = st.columns(3)
            table = s["key_table"].copy()
            sel = table.Cell.isin(wells)
            if a1.button("Assign", disabled=not wells or not media_in.strip(), key="paint_assign"):
                table.loc[sel, "Media"] = media_in.strip()
                table.loc[sel, "Strain"] = strain_in.strip() or "BLANK"
                s["key_table"] = table
                key, errs = key_from_table(table)
                if key is not None:
                    set_key(key, source="painted")
                    s["key_table"] = table
                st.rerun()
            if a2.button("Mark blank", disabled=not wells or not media_in.strip(), key="paint_blank"):
                table.loc[sel, "Media"] = media_in.strip()
                table.loc[sel, "Strain"] = "BLANK"
                s["key_table"] = table
                key, _ = key_from_table(table)
                if key is not None:
                    set_key(key, source="painted")
                    s["key_table"] = table
                st.rerun()
            if a3.button("Clear selected", disabled=not wells, key="paint_clear"):
                table.loc[sel, ["Media", "Strain"]] = ""
                s["key_table"] = table
                key, _ = key_from_table(table)
                set_key(key, source="painted")
                s["key_table"] = table
                st.rerun()

    v = key_validation()
    if v is not None:
        for e in v.errors:
            st.error(e)
        for w in v.warnings:
            st.warning(w)
        if s.get("key") is not None:
            cond = s["key"].conditions()
            n_blank = int(cond.is_blank.sum())
            st.caption(f"{len(cond) - n_blank} conditions across {len(s['key'].media_names)} media, "
                       f"{len(s['key'].blank_wells())} blank wells.")

    # ------------------------------------------------------------- 3. run
    st.subheader("3 · Run the analysis")
    reason = blocking_reason()
    label = "Run analysis" if reason is None else reason
    if st.button(label, type="primary", disabled=reason is not None, key="run_btn"):
        bar = st.progress(0.0, text="Preparing curves…")

        def _prog(done, total, cid):
            bar.progress(done / max(total, 1), text=f"Fitting {cid} ({done}/{total})")

        try:
            run_analysis(progress=_prog)
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(f"The analysis failed: {e}")
            return
        bar.empty()
        st.rerun()

    results = s.get("results")
    if results is not None:
        from state import counts
        c = counts()
        st.markdown(f"**{c['GOOD']} GOOD · {c['BAD']} BAD · {c['not fitted']} not fitted** — "
                    f"click a well above or open the Curves page to see every fit and why it was called.")
        st.page_link("views/curves.py", label="Go to Curves", icon=":material/arrow_forward:")
        for q in results.qc[:8]:
            st.caption(f"QC: {q}")


pages = [
    st.Page(home, title="Home", icon=":material/home:", default=True),
    st.Page("views/plate_qc.py", title="Plate QC", icon=":material/grid_view:"),
    st.Page("views/curves.py", title="Curves", icon=":material/show_chart:"),
    st.Page("views/compare.py", title="Compare", icon=":material/compare_arrows:"),
    st.Page("views/export.py", title="Export", icon=":material/download:"),
    st.Page("views/how_it_works.py", title="How it works", icon=":material/school:"),
]
pg = st.navigation(pages)
render_sidebar()
pg.run()
